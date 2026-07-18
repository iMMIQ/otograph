//! Silero VAD via ONE tilelang fused cooperative kernel launched through the
//! CUDA driver (`libcuda.so`) only — no ONNX Runtime / TensorRT at runtime.
//!
//! The whole forward (STFT -> 4 Conv1d -> LSTM cell -> FC) is one
//! cooperatively-launched kernel (`vad_staged`, grid=32, shared-mem weight
//! staging + grid-sync between stages) compiled from `vad_assets.rs`. Measured
//! on this Jetson Orin: ~47 us/window p50 (min ~42 us) — down from ~70 us for
//! the previous 7-kernel chain.
//!
//! Why fuse to one kernel: on this Jetson the per-launch dispatch floor is ~5 us
//! even when pipelined, and CUDA-graph replay does NOT reduce it (measured: a
//! 7-kernel graph == 7 individual launches). The 7-kernel chain was ~70 us Rust
//! (35 us dispatch + 10 us compute + memcpy). One cooperative launch collapses
//! the dispatches; shared-mem staging keeps the compute near the 7-kernel's; the
//! `xpad` input is zero-copy (mapped host memory) so there is no per-window
//! cuMemcpyHtoD. LSTM state (h, c) stays resident and is updated in place by the
//! kernel (it reads c, not h; Kz reads h before the gates write it), so there is
//! no ping-pong and the kernelParams are built once at load.
//!
//! The `speech_timestamps` state machine below is a line-by-line port of the
//! official `silero_vad` `get_speech_timestamps`; only the per-window `forward`
//! is the tilelang kernel.

use anyhow::{anyhow, Context, Result};
use libloading::Library;
use std::os::raw::{c_char, c_int, c_uint, c_void};

use crate::vad_assets::{KERNELS, WEIGHTS, KSpec, Slot};

const WINDOW: usize = 512; // 16 kHz -> 32 ms hop
const CONTEXT: usize = 64;
const INPUT_LEN: usize = WINDOW + CONTEXT; // 576
const XPAD_LEN: usize = 640; // INPUT_LEN + right reflect-pad 64

// ---------------- CUDA driver FFI ----------------

type Curesult = c_int;
type Cudevice = c_int;
type Cucontext = *mut c_void;
type Culibrary = *mut c_void;
type Cufunction = *mut c_void;
type Cudeviceptr = u64;
type Custream = *mut c_void;
const CUDA_SUCCESS: Curesult = 0;

fn cuerr(r: Curesult, what: &str) -> Result<()> {
    if r == CUDA_SUCCESS { Ok(()) } else { Err(anyhow!("CUDA driver error {r} in {what}")) }
}

struct Cuda {
    _lib: Library,
    init: unsafe extern "C" fn(c_uint) -> Curesult,
    device_get: unsafe extern "C" fn(*mut Cudevice, c_int) -> Curesult,
    primary_ctx_retain: unsafe extern "C" fn(*mut Cucontext, Cudevice) -> Curesult,
    ctx_set_current: unsafe extern "C" fn(Cucontext) -> Curesult,
    library_load_data: unsafe extern "C" fn(
        *mut Culibrary, *const c_void, *mut c_void, *mut *mut c_void, c_uint,
        *mut c_void, *mut *mut c_void, c_uint,
    ) -> Curesult,
    library_get_kernel: unsafe extern "C" fn(*mut Cufunction, Culibrary, *const c_char) -> Curesult,
    memset_d32: unsafe extern "C" fn(Cudeviceptr, c_uint, usize, Custream) -> Curesult,
    mem_alloc: unsafe extern "C" fn(*mut Cudeviceptr, usize) -> Curesult,
    mem_free: unsafe extern "C" fn(Cudeviceptr) -> Curesult,
    mem_host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> Curesult,
    mem_host_get_device_ptr: unsafe extern "C" fn(*mut Cudeviceptr, *mut c_void, c_uint) -> Curesult,
    mem_free_host: unsafe extern "C" fn(*mut c_void) -> Curesult,
    memcpy_h2d: unsafe extern "C" fn(Cudeviceptr, *const c_void, usize) -> Curesult,
    memcpy_d2h: unsafe extern "C" fn(*mut c_void, Cudeviceptr, usize) -> Curesult,
    launch_cooperative: unsafe extern "C" fn(
        Cufunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
        Custream, *mut *mut c_void, *mut *mut c_void,
    ) -> Curesult,
}

impl Cuda {
    fn open() -> Result<Self> {
        let lib = unsafe {
            Library::new("/usr/lib/aarch64-linux-gnu/libcuda.so.1")
                .or_else(|_| Library::new("libcuda.so.1"))
                .or_else(|_| Library::new("libcuda.so"))
        }
        .context("dlopen libcuda.so (need the CUDA driver on the loader path)")?;
        unsafe {
            macro_rules! sym {
                ($n:literal, $t:ty) => {{
                    *lib.get::<$t>($n.as_bytes()).with_context(|| format!("dlsym {}", $n))?
                }};
            }
            let init = sym!("cuInit", unsafe extern "C" fn(c_uint) -> Curesult);
            let device_get = sym!("cuDeviceGet", unsafe extern "C" fn(*mut Cudevice, c_int) -> Curesult);
            let primary_ctx_retain = sym!(
                "cuDevicePrimaryCtxRetain", unsafe extern "C" fn(*mut Cucontext, Cudevice) -> Curesult
            );
            let ctx_set_current = sym!("cuCtxSetCurrent", unsafe extern "C" fn(Cucontext) -> Curesult);
            let library_load_data = sym!(
                "cuLibraryLoadData",
                unsafe extern "C" fn(
                    *mut Culibrary, *const c_void, *mut c_void, *mut *mut c_void, c_uint,
                    *mut c_void, *mut *mut c_void, c_uint,
                ) -> Curesult
            );
            let library_get_kernel = sym!(
                "cuLibraryGetKernel",
                unsafe extern "C" fn(*mut Cufunction, Culibrary, *const c_char) -> Curesult
            );
            let memset_d32 = sym!(
                "cuMemsetD32Async",
                unsafe extern "C" fn(Cudeviceptr, c_uint, usize, Custream) -> Curesult
            );
            let mem_alloc = sym!("cuMemAlloc_v2", unsafe extern "C" fn(*mut Cudeviceptr, usize) -> Curesult);
            let mem_free = sym!("cuMemFree_v2", unsafe extern "C" fn(Cudeviceptr) -> Curesult);
            let mem_host_alloc = sym!(
                "cuMemHostAlloc",
                unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> Curesult
            );
            let mem_host_get_device_ptr = sym!(
                "cuMemHostGetDevicePointer_v2",
                unsafe extern "C" fn(*mut Cudeviceptr, *mut c_void, c_uint) -> Curesult
            );
            let mem_free_host = sym!("cuMemFreeHost", unsafe extern "C" fn(*mut c_void) -> Curesult);
            let memcpy_h2d = sym!("cuMemcpyHtoD_v2", unsafe extern "C" fn(Cudeviceptr, *const c_void, usize) -> Curesult);
            let memcpy_d2h = sym!("cuMemcpyDtoH_v2", unsafe extern "C" fn(*mut c_void, Cudeviceptr, usize) -> Curesult);
            let launch_cooperative = sym!(
                "cuLaunchCooperativeKernel",
                unsafe extern "C" fn(
                    Cufunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint,
                    Custream, *mut *mut c_void, *mut *mut c_void,
                ) -> Curesult
            );
            Ok(Cuda {
                _lib: lib, init, device_get, primary_ctx_retain, ctx_set_current,
                library_load_data, library_get_kernel, memset_d32, mem_alloc, mem_free,
                mem_host_alloc, mem_host_get_device_ptr, mem_free_host,
                memcpy_h2d, memcpy_d2h, launch_cooperative,
            })
        }
    }
}

// ---------------- public types (unchanged API) ----------------

#[derive(Clone)]
pub struct VadParams {
    pub threshold: f32,
    pub min_speech_duration_ms: u32,
    pub max_speech_duration_s: f32,
    pub min_silence_duration_ms: u32,
    pub speech_pad_ms: u32,
    pub min_silence_at_max_speech: u32,
    pub use_max_poss_sil_at_max_speech: bool,
}

impl Default for VadParams {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            max_speech_duration_s: 28.0,
            min_silence_duration_ms: 300,
            speech_pad_ms: 30,
            min_silence_at_max_speech: 98,
            use_max_poss_sil_at_max_speech: true,
        }
    }
}

pub struct Speech { pub start: usize, pub end: usize }

// ---------------- device-buffer slot sizes (f32 elems) ----------------

fn slot_nelems(s: Slot) -> usize {
    use Slot::*;
    match s {
        Xpad => XPAD_LEN,
        SpecG => 129 * 6,
        E0G => 128 * 6,
        E1G => 64 * 4,
        E2G => 64 * 3,
        FeatG => 128,
        ZG => 512,
        Prob => 1,
        H | C => 128,
        // weights: sized from their embedded blob at load
        Basis | Ew0 | Ew1 | Ew2 | Ew3 | Eb0 | Eb1 | Eb2 | Eb3
        | Wl | Rl | Lb | Fcw | Fcb => 0,
    }
}

fn slot_index(s: Slot) -> usize { s as u8 as usize }

// ---------------- the model ----------------

pub struct VadModel {
    cuda: Cuda,
    ctx: Cucontext,
    func: Cufunction,
    spec: &'static KSpec,
    ptr: Vec<Cudeviceptr>,           // device ptr per Slot
    // pre-built kernelParams (void**): state is in-place, so ptrs are stable and
    // this is built once at load — no per-forward allocation.
    slot_bytes: Vec<[u8; 8]>,
    kparams: Vec<*mut c_void>,
    xpad_host: *mut f32,             // mapped host buffer the GPU reads (zero-copy input)
    context: Vec<f32>,               // 64-sample rolling context (host)
    _lib: Culibrary,                 // keep the loaded cubin library alive
    bench: bool,
    bench_n: u64,

    bench_h2d: u64,
    bench_launch: u64,
    bench_d2h: u64,
}

// VadModel is moved across threads by spawn_blocking; the handles are plain
// integers/opaque and the primary context is set current on the worker thread
// before each `speech_timestamps`.
unsafe impl Send for VadModel {}

impl VadModel {
    /// Load the embedded fused VAD kernel. No model path or TRT cache — the
    /// cubin and weights are compiled in. Requires `libcuda.so` on the host.
    pub fn load() -> Result<Self> {
        let cuda = Cuda::open()?;
        let mut ctx: Cucontext = std::ptr::null_mut();
        unsafe {
            cuerr((cuda.init)(0), "cuInit")?;
            let mut dev: Cudevice = 0;
            cuerr((cuda.device_get)(&mut dev, 0), "cuDeviceGet")?;
            cuerr((cuda.primary_ctx_retain)(&mut ctx, dev), "cuDevicePrimaryCtxRetain")?;
            cuerr((cuda.ctx_set_current)(ctx), "cuCtxSetCurrent")?;
        }

        assert_eq!(KERNELS.len(), 1, "expected exactly one fused VAD kernel");
        let spec: &'static KSpec = &KERNELS[0];
        assert!(spec.cooperative, "VAD kernel must be cooperative (uses sync_grid)");
        let mut lib: Culibrary = std::ptr::null_mut();
        unsafe {
            cuerr(
                (cuda.library_load_data)(
                    &mut lib, spec.cubin.as_ptr() as *const c_void,
                    std::ptr::null_mut(), std::ptr::null_mut(), 0,
                    std::ptr::null_mut(), std::ptr::null_mut(), 0,
                ),
                "cuLibraryLoadData",
            )?;
        }
        let name = std::ffi::CString::new(spec.name)?;
        let mut func: Cufunction = std::ptr::null_mut();
        unsafe {
            cuerr((cuda.library_get_kernel)(&mut func, lib, name.as_ptr()), "cuLibraryGetKernel")?;
        }

        let n_slots = KERNELS.iter().flat_map(|k| k.slots.iter()).chain(WEIGHTS.iter().map(|(s, _)| s))
            .copied().map(slot_index).max().map(|m| m + 1).unwrap_or(0);
        let mut ptr = vec![0u64; n_slots];

        // resident weights
        for (slot, bytes) in WEIGHTS {
            let mut dp: Cudeviceptr = 0;
            unsafe {
                cuerr((cuda.mem_alloc)(&mut dp, bytes.len()), "cuMemAlloc weight")?;
                cuerr((cuda.memcpy_h2d)(dp, bytes.as_ptr() as *const c_void, bytes.len()), "cuMemcpyHtoD weight")?;
            }
            ptr[slot_index(*slot)] = dp;
        }
        // scratch + intermediates + state (zeroed). xpad is allocated as MAPPED
        // host memory (zero-copy): the GPU reads it via the device pointer with
        // no per-window cuMemcpyHtoD — saves ~4us/window on Tegra's unified mem.
        let xpad_host: *mut f32;
        {
            let mut raw: *mut c_void = std::ptr::null_mut();
            // CU_MEMHOSTALLOC_DEVICEMAP(2) | CU_MEMHOSTALLOC_WRITE_COMBINED(4)
            unsafe {
                cuerr((cuda.mem_host_alloc)(&mut raw, XPAD_LEN * 4, 6), "cuMemHostAlloc xpad")?;
            }
            xpad_host = raw as *mut f32;
            let mut xpad_dev: Cudeviceptr = 0;
            unsafe {
                cuerr((cuda.mem_host_get_device_ptr)(&mut xpad_dev, raw, 0), "cuMemHostGetDevicePointer")?;
                // zero it (memset via the device ptr — mapped, so write-combined host)
                cuerr((cuda.memset_d32)(xpad_dev, 0, XPAD_LEN, std::ptr::null_mut()), "cuMemsetD32 xpad")?;
            }
            ptr[slot_index(Slot::Xpad)] = xpad_dev;
        }
        for s in [Slot::SpecG, Slot::E0G, Slot::E1G, Slot::E2G,
                  Slot::FeatG, Slot::ZG, Slot::Prob, Slot::H, Slot::C]
        {
            let n = slot_nelems(s);
            let mut dp: Cudeviceptr = 0;
            unsafe {
                cuerr((cuda.mem_alloc)(&mut dp, n * 4), "cuMemAlloc scratch")?;
                cuerr((cuda.memset_d32)(dp, 0, n, std::ptr::null_mut()), "cuMemsetD32")?;
            }
            ptr[slot_index(s)] = dp;
        }

        // pre-build kernelParams (void**) from the KSpec slot order. State is
        // in-place so these ptrs never change — reuse every forward. Build the
        // slot_bytes first, then take pointers into the struct's own storage
        // (after the move) so they stay valid.
        let slot_bytes: Vec<[u8; 8]> = spec.slots.iter()
            .map(|&s| ptr[slot_index(s)].to_le_bytes()).collect();

        let mut m = VadModel {
            cuda, ctx, func, spec, ptr, slot_bytes,
            kparams: Vec::new(),
            xpad_host,
            context: vec![0.0; CONTEXT], _lib: lib,
            bench: false, bench_n: 0, bench_h2d: 0, bench_launch: 0, bench_d2h: 0,
        };
        m.kparams = m.slot_bytes.iter_mut().map(|b| b.as_mut_ptr() as *mut c_void).collect();
        Ok(m)
    }

    pub fn reset(&mut self) {
        unsafe {
            cuerr((self.cuda.ctx_set_current)(self.ctx), "cuCtxSetCurrent").ok();
            for &s in &[Slot::H, Slot::C] {
                let n = slot_nelems(s);
                cuerr((self.cuda.memset_d32)(self.ptr[slot_index(s)], 0, n, std::ptr::null_mut()), "cuMemsetD32").ok();
            }
        }
        self.context.iter_mut().for_each(|x| *x = 0.0);
    }

    /// Run the model on one 512-sample window (zero-padded if short). Mirrors
    /// the reference `OnnxWrapper.__call__`.
    fn forward(&mut self, window: &[f32]) -> Result<f32> {
        // input = context[64] ++ window[512] -> reflect-pad right 64 -> xpad[640]
        let mut xpad = [0f32; XPAD_LEN];
        xpad[..CONTEXT].copy_from_slice(&self.context);
        xpad[CONTEXT..INPUT_LEN].copy_from_slice(&window[..WINDOW]);
        for i in INPUT_LEN..XPAD_LEN {
            // np.pad right-reflect: xpad[i] = xfull[2*N - i - 2], N = INPUT_LEN
            xpad[i] = xpad[2 * INPUT_LEN - i - 2];
        }

        unsafe {
            let bench = self.bench;
            cuerr((self.cuda.ctx_set_current)(self.ctx), "cuCtxSetCurrent")?;
            // zero-copy input: write xpad straight into the mapped host buffer
            // the GPU reads (no cuMemcpyHtoD). On Tegra's unified memory this is
            // a coherent host write the cooperative kernel sees on launch.
            let t1 = std::time::Instant::now();
            std::ptr::copy_nonoverlapping(xpad.as_ptr(), self.xpad_host, XPAD_LEN);
            let t_h2d = t1.elapsed().as_nanos() as u64;
            let g = self.spec.grid;
            let b = self.spec.block;
            let t2 = std::time::Instant::now();
            cuerr(
                (self.cuda.launch_cooperative)(
                    self.func, g[0], g[1], g[2], b[0], b[1], b[2], self.spec.shmem,
                    std::ptr::null_mut(), self.kparams.as_mut_ptr(), std::ptr::null_mut(),
                ),
                "cuLaunchCooperativeKernel",
            )?;
            let t_launch = t2.elapsed().as_nanos() as u64;

            let mut prob = [0f32; 1];
            let t3 = std::time::Instant::now();
            cuerr(
                (self.cuda.memcpy_d2h)(
                    prob.as_mut_ptr() as *mut c_void,
                    self.ptr[slot_index(Slot::Prob)],
                    4,
                ),
                "cuMemcpyDtoH prob",
            )?;
            let t_d2h = t3.elapsed().as_nanos() as u64;
            // cuMemcpyDtoH_v2 is synchronous (blocks until the kernel finishes +
            // the 4-byte copy lands), so prob[0] is valid here.
            if bench {
                self.bench_h2d += t_h2d; self.bench_launch += t_launch; self.bench_d2h += t_d2h;
            }
            // context = last 64 samples of the window
            self.context.copy_from_slice(&window[WINDOW - CONTEXT..]);
            Ok(prob[0])
        }
    }

    /// Faithful port of `get_speech_timestamps`. Returns speech regions as
    /// sample-index half-open intervals [start, end).
    pub fn speech_timestamps(&mut self, audio: &[f32], p: &VadParams) -> Result<Vec<Speech>> {
        let sr = 16000f64;
        let ws = WINDOW;
        let min_speech_samples = sr * p.min_speech_duration_ms as f64 / 1000.0;
        let speech_pad_samples = sr * p.speech_pad_ms as f64 / 1000.0;
        let max_speech_samples = sr * p.max_speech_duration_s as f64 - ws as f64 - 2.0 * speech_pad_samples;
        let min_silence_samples = sr * p.min_silence_duration_ms as f64 / 1000.0;
        let min_silence_at_max = sr * p.min_silence_at_max_speech as f64 / 1000.0;
        let neg_threshold = (p.threshold - 0.15).max(0.01);
        let audio_len = audio.len();

        // --- per-window probabilities (zero-pad the final window) ---
        self.reset();
        let n_windows = audio_len.div_ceil(ws);
        let mut probs: Vec<f32> = Vec::with_capacity(n_windows);
        let mut buf = vec![0f32; WINDOW];
        let mut pos = 0usize;
        let bench = std::env::var("OTOGRAPH_VAD_BENCH").is_ok();
        self.bench = bench;
        self.bench_n = 0; self.bench_h2d = 0; self.bench_launch = 0; self.bench_d2h = 0;
        let mut times_ns: Vec<u64> = if bench { Vec::with_capacity(n_windows) } else { Vec::new() };
        while pos < audio_len {
            let take = (audio_len - pos).min(WINDOW);
            buf[..take].copy_from_slice(&audio[pos..pos + take]);
            for b in &mut buf[take..] {
                *b = 0.0;
            }
            let t0 = std::time::Instant::now();
            let pr = self.forward(&buf)?;
            if bench {
                times_ns.push(t0.elapsed().as_nanos() as u64);
                self.bench_n += 1;
            }
            probs.push(pr);
            pos += ws;
        }
        if bench && !times_ns.is_empty() {
            let mut sorted = times_ns.clone();
            sorted.sort_unstable();
            let n = sorted.len() as f64;
            let mean = sorted.iter().sum::<u64>() as f64 / n;
            let p50 = sorted[(n * 0.50) as usize];
            let p99 = sorted[(n * 0.99).min(n - 1.0) as usize];
            eprintln!(
                "[vad-bench] {n:.0} windows  mean={mean:.0}ns ({mean_us:.1}us)  p50={p50_us:.1}us  p99={p99_us:.1}us  min={min_us:.1}us",
                mean_us = mean / 1000.0, p50_us = p50 as f64 / 1000.0,
                p99_us = p99 as f64 / 1000.0, min_us = sorted[0] as f64 / 1000.0
            );
            let n = self.bench_n.max(1) as f64;
            eprintln!(
                "[vad-bench] phase us: h2d={:.1} launch={:.1} d2h={:.1}",
                self.bench_h2d as f64 / n / 1000.0,
                self.bench_launch as f64 / n / 1000.0,
                self.bench_d2h as f64 / n / 1000.0,
            );
        }

        // --- state machine (port of get_speech_timestamps) ---
        let mut speeches: Vec<Speech> = Vec::new();
        let mut cur_start: Option<usize> = None;
        let mut temp_end = 0usize;
        let mut prev_end = 0usize;
        let mut next_start = 0usize;
        let mut possible_ends: Vec<(usize, f64)> = Vec::new();

        for (i, prob) in probs.iter().enumerate() {
            let cur_sample = ws * i;
            let triggered = cur_start.is_some();

            if *prob >= p.threshold && temp_end != 0 {
                let sil = (cur_sample - temp_end) as f64;
                if sil > min_silence_at_max {
                    possible_ends.push((temp_end, sil));
                }
                temp_end = 0;
                if next_start < prev_end {
                    next_start = cur_sample;
                }
            }

            if *prob >= p.threshold && !triggered {
                cur_start = Some(cur_sample);
                continue;
            }

            if let Some(start) = cur_start {
                if (cur_sample - start) as f64 > max_speech_samples {
                    if p.use_max_poss_sil_at_max_speech && !possible_ends.is_empty() {
                        let (pe, dur) = *possible_ends
                            .iter()
                            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .unwrap();
                        prev_end = pe;
                        speeches.push(Speech { start, end: prev_end });
                        next_start = prev_end + dur as usize;
                        if (next_start as f64) < (prev_end as f64 + cur_sample as f64) {
                            cur_start = Some(next_start);
                        } else {
                            cur_start = None;
                        }
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        possible_ends.clear();
                    } else if prev_end != 0 {
                        speeches.push(Speech { start, end: prev_end });
                        if next_start < prev_end {
                            cur_start = None;
                        } else {
                            cur_start = Some(next_start);
                        }
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        possible_ends.clear();
                    } else {
                        speeches.push(Speech { start, end: cur_sample });
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        cur_start = None;
                        possible_ends.clear();
                        continue;
                    }
                }
            }

            let triggered = cur_start.is_some();
            if *prob < neg_threshold && triggered {
                if temp_end == 0 {
                    temp_end = cur_sample;
                }
                let sil_now = (cur_sample - temp_end) as f64;
                if !p.use_max_poss_sil_at_max_speech && sil_now > min_silence_at_max {
                    prev_end = temp_end;
                }
                if sil_now < min_silence_samples {
                    continue;
                }
                let start = cur_start.take().unwrap();
                let end = temp_end;
                if (end - start) as f64 > min_speech_samples {
                    speeches.push(Speech { start, end });
                }
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                cur_start = None;
                possible_ends.clear();
                continue;
            }
        }

        if let Some(start) = cur_start {
            if (audio_len - start) as f64 > min_speech_samples {
                speeches.push(Speech { start, end: audio_len });
            }
        }

        let sp_pad = speech_pad_samples as i64;
        for i in 0..speeches.len() {
            if i == 0 {
                speeches[i].start = (speeches[i].start as i64 - sp_pad).max(0) as usize;
            }
            if i != speeches.len() - 1 {
                let silence_dur = speeches[i + 1].start as i64 - speeches[i].end as i64;
                if (silence_dur as f64) < 2.0 * speech_pad_samples {
                    let half = ((silence_dur as f64) / 2.0).floor() as usize;
                    speeches[i].end += half;
                    speeches[i + 1].start = (speeches[i + 1].start as i64 - half as i64).max(0) as usize;
                } else {
                    speeches[i].end = (speeches[i].end + speech_pad_samples as usize).min(audio_len);
                    speeches[i + 1].start = (speeches[i + 1].start as i64 - sp_pad).max(0) as usize;
                }
            } else {
                speeches[i].end = (speeches[i].end + speech_pad_samples as usize).min(audio_len);
            }
        }

        Ok(speeches)
    }
}

impl Drop for VadModel {
    fn drop(&mut self) {
        unsafe {
            // free device allocations (skip Xpad — it's a mapped host pointer,
            // freed via cuMemFreeHost below, not cuMemFree).
            let xpad = self.ptr[slot_index(Slot::Xpad)];
            for &dp in &self.ptr {
                if dp != 0 && dp != xpad {
                    (self.cuda.mem_free)(dp);
                }
            }
            if !self.xpad_host.is_null() {
                (self.cuda.mem_free_host)(self.xpad_host as *mut c_void);
            }
        }
    }
}
