# otograph — SRT subtitles via Silero VAD + Qwen3-ASR

*otograph* — from Greek *oto-* (ear) + *-graph* (write): "ear-writing", turning
what is heard into text.

A small Rust CLI that walks a file or directory (recursively), extracts the audio
from every video/audio file, splits it into speech segments with **Silero VAD**
(running locally via ONNX Runtime), transcribes each segment with the
**Qwen3-ASR-1.7B** vLLM server, and writes `<media>.srt` next to each source.

The VAD segmentation logic is a faithful, line-by-line port of the official
`silero_vad` package's `get_speech_timestamps` (verified to match the reference
to 0.01 s — see [Validation](#validation)).

## Pipeline

```
media file ──ffmpeg──▶ 16 kHz mono PCM ──▶ Silero VAD (ort, CPU)
                                              │
                            speech segments [start,end]
                                              │
                  each segment ──HTTP multipart──▶ Qwen3-ASR /v1/audio/transcriptions
                                              │
                              text + segment timestamps ──▶ <media>.srt
```

VAD runs on CPU (it's tiny, <3 ms/s of audio); only the ASR uses the GPU, so the
two don't contend. Segments within a file are transcribed **concurrently**
(`--concurrency`).

## Prerequisites

- **ffmpeg** on PATH (any recent build).
- **Python 3 + pip** — only used by the two download scripts (not at run time).
- The **Qwen3-ASR-1.7B** vLLM server running (default `http://localhost:8002`).

## One-time setup (downloads the VAD model + ONNX Runtime)

```bash
cd otograph
./scripts/download_model.sh        # -> model/silero_vad.onnx            (≈2 MB)
./scripts/download_onnxruntime.sh  # -> vendor/lib/libonnxruntime.so.1.x  (≈18 MB)
cargo build --release
```

Both scripts pull from the aliyun pypi mirror by default (set `PIP_INDEX=...` to
override). The onnxruntime is pinned to **1.23.2** to match the `api-23` ort
binding in `Cargo.toml`.

Then, before running otograph, tell ort's `load-dynamic` where the runtime is:

```bash
export ORT_DYLIB_PATH="$PWD/vendor/lib/libonnxruntime.so"
```

(Or `export LD_LIBRARY_PATH="$PWD/vendor/lib:$LD_LIBRARY_PATH"`.)

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
| `--concurrency` | `8` | max simultaneous ASR requests per file (server likes 64–128) |
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
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.30 \
  --max-num-seqs 256 --enable-prefix-caching --enforce-eager
```

- **`--enforce-eager`** is correct for this encoder-decoder model (FULL cuda
  graph capture is pathological: ~121 s/graph). PIECEWISE graphs give ~0 % gain
  because the workload is GPU-compute-bound.
- **Throughput** is GPU-compute-bound (~294 audio-s/s at concurrency 128);
  concurrency is the lever — raise `--concurrency` for batch directories.
- **`max_completion_tokens`** must be bounded — without it the server runs away
  on non-speech (no stop tokens on the transcription path). 200 is plenty per
  segment.
- Each segment is ≤ 28 s, under the model's native 30 s chunk, so the encoder is
  used efficiently (the mel is padded to 30 s regardless of clip length).

## Notes (aarch64 / Jetson)

- `ort` is built with `load-dynamic` + `api-23`, so it does **not** compile ONNX
  Runtime (the default build-from-source path fails to link on Jetson — missing
  `__cxa_call_terminate` / `__isoc23_strtoll`). The prebuilt Microsoft `.so` from
  the onnxruntime wheel is loaded at run time via `ORT_DYLIB_PATH`.
- VAD inference is sequential (stateful LSTM): ~0.3 s for an 11 s clip, scaling
  linearly. For very long files it's still cheap next to the ASR.
- ffmpeg is invoked with raw `s16le` output (no WAV container) to avoid
  streaming-WAV header issues and parsing overhead.

## License

otograph is released under the [GNU Lesser General Public License
v3.0](LICENSE) (LGPL-3.0-or-later). The bundled/downloaded third-party
artifacts keep their own licenses: ONNX Runtime (MIT), Silero VAD (MIT),
ort (MIT/Apache-2.0).
