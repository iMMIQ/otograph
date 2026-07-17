//! Silero VAD via ONNX Runtime — a faithful Rust port of the official
//! `silero_vad` package (v5/v6, `src/silero_vad/utils_vad.py`) `OnnxWrapper`
//! + `get_speech_timestamps`. Model I/O matches the reference exactly:
//!
//!   inputs:  input [1,576]  (= 64-sample context ++ 512-sample window, 16 kHz)
//!            state [2,1,128] (LSTM hidden/cell)
//!            sr    int64 scalar = 16000
//!   outputs: [0] prob scalar   [1] new state [2,1,128]
//!
//! The state machine below is a line-by-line port of `get_speech_timestamps`
//! (including its `max_speech_duration_s` smart-split), NOT a rewrite.

use anyhow::{Context, Result};
use ndarray::{arr0, Array2, Array3};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;

const WINDOW: usize = 512; // 16 kHz -> 32 ms hop
const CONTEXT: usize = 64;
const INPUT_LEN: usize = WINDOW + CONTEXT; // 576
const STATE: [usize; 3] = [2, 1, 128];

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
            max_speech_duration_s: 28.0, // cap each segment < 30 s native ASR chunk
            min_silence_duration_ms: 300, // subtitle-friendly split point
            speech_pad_ms: 30,
            min_silence_at_max_speech: 98,
            use_max_poss_sil_at_max_speech: true,
        }
    }
}

pub struct Speech {
    pub start: usize, // sample index
    pub end: usize,
}

pub struct VadModel {
    session: Session,
    state: Array3<f32>,
    context: Array2<f32>,
}

impl VadModel {
    pub fn load(path: &Path) -> Result<Self> {
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("ort session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("opt level: {e}"))?
            .with_intra_threads(1)
            .map_err(|e| anyhow::anyhow!("intra threads: {e}"))?
            .commit_from_file(path)
            .with_context(|| format!("loading VAD onnx {}", path.display()))?;
        Ok(Self {
            session,
            state: Array3::zeros(STATE),
            context: Array2::zeros((1, CONTEXT)),
        })
    }

    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.context.fill(0.0);
    }

    /// Run the model on one 512-sample window (already zero-padded if short),
    /// returning the speech probability. Mirrors `OnnxWrapper.__call__`.
    fn forward(&mut self, window: &[f32]) -> Result<f32> {
        // input = concat(context[64], window[512]) -> [1,576]
        let mut v = Vec::with_capacity(INPUT_LEN);
        v.extend_from_slice(self.context.as_slice().unwrap());
        v.extend_from_slice(window);
        let input = Array2::from_shape_vec((1, INPUT_LEN), v).unwrap();
        let sr = arr0(16000i64);

        let input_t = Tensor::from_array(input).map_err(|e| anyhow::anyhow!("input tensor: {e}"))?;
        let state_t = Tensor::from_array(self.state.clone()).map_err(|e| anyhow::anyhow!("state tensor: {e}"))?;
        let sr_t = Tensor::from_array(sr).map_err(|e| anyhow::anyhow!("sr tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs!(
                "input" => &input_t,
                "state" => &state_t,
                "sr" => &sr_t
            ))
            .map_err(|e| anyhow::anyhow!("ort run: {e}"))?;

        let prob = {
            let p = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| anyhow::anyhow!("extract prob: {e}"))?;
            p.iter().next().copied().unwrap_or(0.0)
        };
        let new_state = outputs[1]
            .try_extract_array::<f32>()
            .map_err(|e| anyhow::anyhow!("extract state: {e}"))?;
        for (dst, src) in self.state.iter_mut().zip(new_state.iter()) {
            *dst = *src;
        }
        // context = last 64 samples of the window
        for i in 0..CONTEXT {
            self.context[(0, i)] = window[WINDOW - CONTEXT + i];
        }
        Ok(prob)
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
        while pos < audio_len {
            let take = (audio_len - pos).min(WINDOW);
            buf[..take].copy_from_slice(&audio[pos..pos + take]);
            for b in &mut buf[take..] {
                *b = 0.0;
            }
            probs.push(self.forward(&buf)?);
            pos += ws;
        }

        // --- state machine (port of get_speech_timestamps) ---
        let mut speeches: Vec<Speech> = Vec::new();
        let mut cur_start: Option<usize> = None; // Some(start) <=> currently in speech
        let mut temp_end = 0usize;
        let mut prev_end = 0usize;
        let mut next_start = 0usize;
        let mut possible_ends: Vec<(usize, f64)> = Vec::new(); // (temp_end, silence_dur)

        for (i, prob) in probs.iter().enumerate() {
            let cur_sample = ws * i;
            let triggered = cur_start.is_some();

            // speech resumes after a temp_end -> record candidate silence split point
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

            // start of speech
            if *prob >= p.threshold && !triggered {
                cur_start = Some(cur_sample);
                continue;
            }

            // max speech length reached -> pick best cut point
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

            // silence detection while in speech
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

        // tail: still in speech at end of audio
        if let Some(start) = cur_start {
            if (audio_len - start) as f64 > min_speech_samples {
                speeches.push(Speech { start, end: audio_len });
            }
        }

        // --- speech padding pass (port of lines 428-440) ---
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
