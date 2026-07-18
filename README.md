# otograph — SRT subtitles via Silero VAD + Qwen3-ASR

*otograph* — from Greek *oto-* (ear) + *-graph* (write): "ear-writing", turning
what is heard into text.

A small Rust CLI that walks a file or directory (recursively), extracts the audio
from every video/audio file, splits it into speech segments with **Silero VAD**
(running locally as compiled tilelang CUDA kernels via `libcuda.so` — no ONNX
Runtime / TensorRT), transcribes each segment with the **Qwen3-ASR-1.7B** vLLM
server, and writes `<media>.srt` next to each source.

The VAD segmentation logic is a faithful, line-by-line port of the official
`silero_vad` package's `get_speech_timestamps` (verified to match the reference
to 0.01 s — see [Validation](#validation)).

## Pipeline

```
media file ──ffmpeg──▶ 16 kHz mono PCM ──▶ Silero VAD (tilelang · GPU, libcuda)
                                              │
                            speech segments [start,end]
                                              │
                  each segment ──HTTP multipart──▶ Qwen3-ASR /v1/audio/transcriptions
                                              │
                              text + segment timestamps ──▶ <media>.srt
```

VAD runs as a single compiled-in tilelang CUDA kernel (one fused cooperative
cubin) launched through the CUDA driver (`libcuda.so`) — no ONNX Runtime or
TensorRT. The whole forward (STFT → 4 Conv1d → LSTM cell → FC) is one
cooperatively-launched kernel with shared-memory weight staging and grid-sync
between stages, plus zero-copy mapped input/output. It runs FP32-accurate
(max ~2.8e-6 vs the ONNX reference) at **~19.6 µs/window** end-to-end (p50),
down from ~70 µs for the earlier 7-kernel chain. Segments within a file are
transcribed **concurrently** (`--concurrency`).

## Prerequisites

- **ffmpeg** on PATH (any recent build).
- **libcuda.so** on the loader path (the CUDA driver — present on Jetson).
- The **Qwen3-ASR-1.7B** vLLM server running (default `http://localhost:8002`).
- Python 3 + pip + `tilelang` are **not** required to build or run — only to
  regenerate the VAD assets (see below).

## Build

```bash
cd otograph
cargo build --release
```

That's it. The fused VAD cubin and the model weights are embedded
(`assets/k_vad_staged.cubin`, `assets/w_*.bin`, generated into
`src/vad_assets.rs`), so
`cargo build` needs no ONNX, onnxruntime, or TensorRT. `ldd target/release/otograph`
shows no ML runtime — `libcuda.so` is reached via `dlopen` at startup.

## Regenerating the VAD assets (optional, developer-only)

If you change the tilelang kernel (`scripts/build_vad_kernels.py` imports it
from the sibling `../tilelang-poc/vad/kernels6.py: vad_staged`), rebuild the
embedded assets from the ONNX:

```bash
./scripts/download_model.sh     # -> model/silero_vad.onnx + model/silero_vad_16k.onnx (needs torch)
python3 scripts/build_vad_kernels.py   # -> assets/* + src/vad_assets.rs
cargo build --release
```

### VAD backend (fused cooperative tilelang kernel, Jetson Orin)

The VAD forward (STFT → 4 Conv1d → LSTM cell → FC) is hand-written in the
tilelang DSL as **one fused cooperative kernel**: 32 blocks, shared-memory
weight staging per stage, and `sync_grid` barriers between stages. It is
compiled to one standalone cubin and launched cooperatively via the CUDA driver
API (`cuLaunchCooperativeKernel`). State (`h`, `c`) stays resident on the GPU
and is updated in place by the kernel; input and probability output are mapped
host memory, so there are no per-window CUDA copies. The cooperative kernel is
launched once per file and persists across all stateful windows.

Why one fused kernel rather than the earlier 7-kernel chain: on this Jetson the
per-launch dispatch floor is ~5 µs even when pipelined, and CUDA-graph replay
does **not** reduce it (measured). The 7-kernel chain spent ~35 µs in dispatch
alone; fusing to one cooperative launch collapses that while shared-mem staging
keeps the compute fast.

Measured on this Jetson Orin (Rust, `OTOGRAPH_VAD_BENCH=1`):

| backend                  | per window (p50) | precision vs ONNX FP32 |
|--------------------------|-----------------:|------------------------|
| fused cooperative (this) | **~19.6 µs**     | max ~2.8e-6            |
| 7-kernel chain (prev)    | ~70 µs           | ~2.5e-7                |
| TensorRT FP16 (old)      | ~221 µs          | ~1e-3 (FP16)           |

The FP32 path matches the ONNX reference over 350× more closely than the old
FP16 path; the reference segment boundaries remain unchanged.

## Usage

```bash
# single file
otograph --language en video.mp4

# a whole directory (recursive); .srt written next to each source
otograph --language ja /path/to/clips

# language inferred per file from the name (talk.zh.mp4 -> zh)
otograph --lang-from-name /path/to/mixed

# dry run: segment only, no ASR (checks VAD without needing the server)
otograph --dry-run video.mp4
```

Key options (`otograph --help` for all):

| option | default | notes |
|---|---|---|
| `--server` | `http://localhost:8002` | Qwen3-ASR base URL |
| `--model` | `/models/Qwen3-ASR-1.7B` | served model name |
| `--language` | *(auto)* | `zh`/`ja`/`en`/… **Pass explicitly** — server auto-detect is unreliable. |
| `--lang-from-name` | off | parse language from each filename |
| `--concurrency` | `96` | max simultaneous ASR requests per file; tuned with the server defaults below |
| `--max-completion-tokens` | `200` | bounds ASR decode length |
| `--vad-threshold` | `0.5` | speech probability threshold |
| `--vad-min-silence-ms` | `300` | silence that splits two segments |
| `--vad-min-speech-ms` | `250` | drop speech shorter than this |
| `--vad-speech-pad-ms` | `30` | pad each segment each side |
| `--vad-max-segment-s` | `28` | cap a segment's length (< 30 = ASR native chunk) |
| `--force` | off | overwrite an existing `.srt` (otherwise skipped) |
| `--dry-run` | off | segment only, don't call the ASR server |

### Language handling

Qwen3-ASR needs an explicit `language`. For a mixed directory use
`--lang-from-name` (parses tokens like `foo.zh.mp4`, `bar_[ja].mkv`) or split by
language and run per-directory with `--language`.

## Validation

The VAD port was cross-checked against an independent numpy transcription of the
**official** `silero_vad` algorithm (`scripts/vadref.py`). On the OpenAI JFK
sample the two agree to the hundredth of a second, including the
`max_speech_duration_s` split branch:

```
Rust :   4 segments, 8.3s   #1 0.32→2.27  #2 3.27→4.45  #3 5.38→7.68  #4 8.16→11.00
Python:  4 segments, 8.3s   #1 0.32→2.27  #2 3.27→4.45  #3 5.38→7.68  #4 8.16→11.00
```

Reproduce with `scripts/vadref.py <model> <audio> [max_segment_s] [min_silence_ms]`.

## The Qwen3-ASR server (optimal config)

This tool calls `POST {server}/v1/audio/transcriptions` with
`model` + `language` + `max_completion_tokens=200` + the segment WAV. The
production server config (Jetson AGX Orin, bf16, eager):

```bash
docker run -d --runtime=nvidia --name qwen3-asr-1.7b \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v /path/to/Qwen3-ASR-1.7B:/models/Qwen3-ASR-1.7B:ro \
  -p 8002:8000 --shm-size 1g --entrypoint vllm qwen3-asr-vllm:audio \
  serve /models/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 512 --gpu-memory-utilization 0.03 \
  --max-num-seqs 128 --mm-processor-cache-gb 0 \
  --enable-prefix-caching --enforce-eager
```

- **`--enforce-eager`** is correct for this encoder-decoder model (FULL cuda
  graph capture is pathological: ~121 s/graph). PIECEWISE graphs give ~0 % gain
  because the workload is GPU-compute-bound.
- **`--concurrency=96`** is the latency/throughput knee on this Jetson. On a
  fixed 382-clip workload it sustained 34.19 requests/s with 2.38 s mean
  request latency. Concurrency 128 gained only 4.2% throughput (35.62
  requests/s) while increasing mean latency by 24% to 2.95 s.
- **The low-memory server defaults are deliberate.** A 0.03 GPU-memory budget
  provides a 1.48 GiB / 13,888-token KV cache for this short-context endpoint,
  down from 18.01 GiB at 0.30. Disabling the default 4 GiB multimodal processor
  cache avoids retaining one-off audio preprocessing results.
- **`max_completion_tokens`** must be bounded — without it the server runs away
  on non-speech (no stop tokens on the transcription path). 200 is plenty per
  segment.
- Each segment is ≤ 28 s, under the model's native 30 s chunk, so the encoder is
  used efficiently (the mel is padded to 30 s regardless of clip length).

## Notes (aarch64 / Jetson)

- The VAD talks to the GPU through the CUDA **driver** API (`libcuda.so`) via
  `libloading`, so there is nothing to compile-link against and no ML runtime to
  ship — `libcuda.so` is `dlopen`-ed at startup. (On this Tegra driver the cubin
  must be loaded with `cuLibraryLoadData`; the older `cuModuleLoadData` returns
  `CUDA_ERROR_INVALID_IMAGE`.)
- The fused kernel is launched with `cuLaunchCooperativeKernel` (it uses
  `cooperative_groups::this_grid().sync()` for the inter-stage barriers).
- VAD inference is sequential (stateful LSTM): ~19.6 µs/window (p50); ~0.007 s for
  an 11 s clip (about 344 windows at a 32 ms hop), scaling linearly. Set
  `OTOGRAPH_VAD_BENCH=1` to print per-window p50/p99/min.
- ffmpeg is invoked with raw `s16le` output (no WAV container) to avoid
  streaming-WAV header issues and parsing overhead.

## License

otograph is released under the [GNU Lesser General Public License
v3.0](LICENSE) (LGPL-3.0-or-later). The bundled third-party artifacts keep their
own licenses: Silero VAD weights (MIT; the embedded cubins are our compiled
derivative), libloading (ISC).
