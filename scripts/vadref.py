#!/usr/bin/env python3
"""Independent reference: faithful numpy transcription of the OFFICIAL silero-vad
OnnxWrapper + get_speech_timestamps (src/silero_vad/utils_vad.py). The ONNX forward
is byte-identical to the official code (which is itself numpy under the hood).
Used to cross-check the Rust port (src/vad.rs)."""
import os, sys, subprocess, numpy as np, onnxruntime

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(_ROOT, "model", "silero_vad.onnx")
AUDIO = sys.argv[2] if len(sys.argv) > 2 else None
if not AUDIO:
    sys.exit("usage: vadref.py [model.onnx] <audio> [max_segment_s] [min_silence_ms]")

# ---- load audio via ffmpeg -> raw s16le -> float32 (same bytes the Rust tool sees) ----
raw = subprocess.run(
    ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", AUDIO,
     "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-f", "s16le", "-"],
    check=True, capture_output=True).stdout
audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class OnnxWrapper:
    def __init__(self, path):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(path, sess_options=opts,
                                                    providers=["CPUExecutionProvider"])
        self.reset_states()
        self.sample_rates = [8000, 16000]

    def reset_states(self, batch_size=1):
        self._state = np.zeros((2, batch_size, 128), dtype=np.float32)
        self._context = np.zeros(0, dtype=np.float32)
        self._last_sr = 0
        self._last_batch_size = 0

    def __call__(self, x, sr):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        num_samples = 512 if sr == 16000 else 256
        assert x.shape[-1] == num_samples
        batch_size = x.shape[0]
        context_size = 64 if sr == 16000 else 32
        if not self._last_batch_size:
            self.reset_states(batch_size)
        if not len(self._context):
            self._context = np.zeros((batch_size, context_size), dtype=np.float32)
        x = np.concatenate([self._context, x], axis=1)
        ort_inputs = {"input": x, "state": self._state, "sr": np.array(sr, dtype=np.int64)}
        out, state = self.session.run(None, ort_inputs)
        self._state = state
        self._context = x[:, -context_size:]
        self._last_sr = sr
        self._last_batch_size = batch_size
        return float(np.asarray(out).reshape(-1)[0])


def get_speech_timestamps(audio, model, threshold=0.5, sampling_rate=16000,
                          min_speech_duration_ms=250, max_speech_duration_s=28.0,
                          min_silence_duration_ms=300, speech_pad_ms=30,
                          min_silence_at_max_speech=98, use_max_poss_sil_at_max_speech=True,
                          window_size_samples=512):
    audio = np.asarray(audio, dtype=np.float32)
    audio = audio.reshape(-1)
    window_size_samples = 512 if sampling_rate == 16000 else 256
    model.reset_states()
    sr = sampling_rate
    min_speech_samples = sr * min_speech_duration_ms / 1000
    speech_pad_samples = sr * speech_pad_ms / 1000
    max_speech_samples = sr * max_speech_duration_s - window_size_samples - 2 * speech_pad_samples
    min_silence_samples = sr * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sr * min_silence_at_max_speech / 1000
    audio_length_samples = len(audio)

    speech_probs = []
    for current_start in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start: current_start + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))))
        speech_probs.append(model(chunk, sampling_rate))

    neg_threshold = max(threshold - 0.15, 0.01)
    triggered = False
    speeches = []
    current_speech = {}
    temp_end = 0
    prev_end = next_start = 0
    possible_ends = []

    for i, speech_prob in enumerate(speech_probs):
        cur_sample = window_size_samples * i

        if (speech_prob >= threshold) and temp_end:
            sil_dur = cur_sample - temp_end
            if sil_dur > min_silence_samples_at_max_speech:
                possible_ends.append((temp_end, sil_dur))
            temp_end = 0
            if next_start < prev_end:
                next_start = cur_sample

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = cur_sample
            continue

        if triggered and (cur_sample - current_speech['start'] > max_speech_samples):
            if use_max_poss_sil_at_max_speech and possible_ends:
                prev_end, dur = max(possible_ends, key=lambda x: x[1])
                current_speech['end'] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                next_start = prev_end + dur
                if next_start < prev_end + cur_sample:
                    current_speech['start'] = next_start
                else:
                    triggered = False
                prev_end = next_start = temp_end = 0
                possible_ends = []
            else:
                if prev_end:
                    current_speech['end'] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech['start'] = next_start
                    prev_end = next_start = temp_end = 0
                    possible_ends = []
                else:
                    current_speech['end'] = cur_sample
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    possible_ends = []
                    continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = cur_sample
            sil_dur_now = cur_sample - temp_end
            if not use_max_poss_sil_at_max_speech and sil_dur_now > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if sil_dur_now < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                current_speech = {}
                prev_end = next_start = temp_end = 0
                triggered = False
                possible_ends = []
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += int(silence_duration // 2)
                speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - silence_duration // 2))
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i + 1]['start'] = int(max(0, speeches[i + 1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    return speeches


model = OnnxWrapper(MODEL)
kw = {}
if len(sys.argv) > 3: kw["max_speech_duration_s"] = float(sys.argv[3])
if len(sys.argv) > 4: kw["min_silence_duration_ms"] = int(sys.argv[4])
segs = get_speech_timestamps(audio, model, **kw)
print(f"audio: {len(audio)/16000:.2f}s, {len(segs)} segments, "
      f"{sum(s['end']-s['start'] for s in segs)/16000:.2f}s speech")
for i, s in enumerate(segs):
    print(f"  #{i+1}  {s['start']/16000:7.2f} -> {s['end']/16000:7.2f}  ({(s['end']-s['start'])/16000:5.1f}s)")
