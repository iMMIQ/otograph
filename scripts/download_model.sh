#!/usr/bin/env bash
# Fetch the official Silero VAD ONNX model by downloading the silero-vad PyPI
# wheel and extracting data/silero_vad.onnx. (GitHub raw/LFS was unreliable on
# this network; the wheel from the pypi mirror is reliable.)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$DIR/model/silero_vad.onnx"
MIRROR="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
mkdir -p "$DIR/model"

if [[ -s "$OUT" ]] && [[ $(stat -c%s "$OUT") -gt 1000000 ]]; then
  echo "model already present: $OUT ($(du -h "$OUT" | cut -f1))"; exit 0
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "downloading silero-vad wheel (mirror: $MIRROR)"
python3 -m pip download --no-deps --dest "$TMP" "silero-vad" -i "$MIRROR" >/dev/null
WHL="$(ls "$TMP"/silero_vad-*.whl | head -1)"
(cd "$TMP" && unzip -o -q "$WHL")
SRC="$(find "$TMP" -name silero_vad.onnx | head -1)"
cp "$SRC" "$OUT"
echo "installed: $OUT ($(du -h "$OUT" | cut -f1))"
