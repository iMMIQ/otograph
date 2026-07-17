#!/usr/bin/env bash
# Install Microsoft's prebuilt ONNX Runtime for this platform via a PyPI wheel
# (the aarch64 .so ships inside the onnxruntime wheel). Pinned to 1.23.2 to
# match the `api-23` binding in Cargo.toml.
#
# Result: vendor/lib/libonnxruntime.so (+ providers_shared). At runtime, point
# ort's load-dynamic at it via:
#   export ORT_DYLIB_PATH="$DIR/vendor/lib/libonnxruntime.so"
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="$DIR/vendor"
VER="${1:-1.23.2}"
MIRROR="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

mkdir -p "$VENDOR/lib"
SO="$VENDOR/lib/libonnxruntime.so"
if [[ -e "$SO" ]]; then echo "onnxruntime already present: $SO -> $(readlink -f "$SO")"; exit 0; fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "downloading onnxruntime==$VER wheel (mirror: $MIRROR)"
python3 -m pip download --no-deps --dest "$TMP" "onnxruntime==$VER" -i "$MIRROR" >/dev/null

WHL="$(ls "$TMP"/onnxruntime-*.whl | head -1)"
python3 -m wheel unpack "$WHL" -d "$TMP" >/dev/null 2>&1 || (cd "$TMP" && mkdir -p x && unzip -o -q "$WHL" -d x)
CAP="$(find "$TMP" -type d -name capi | head -1)"
cp -a "$CAP"/libonnxruntime.so.* "$VENDOR/lib/" 2>/dev/null || true
cp -a "$CAP"/libonnxruntime_providers_shared.so "$VENDOR/lib/" 2>/dev/null || true
REAL="$(ls "$VENDOR"/lib/libonnxruntime.so.* | grep -v providers | head -1)"
ln -sf "$(basename "$REAL")" "$VENDOR/lib/libonnxruntime.so"

echo "installed: $REAL"
echo "run with:  export ORT_DYLIB_PATH=\"$VENDOR/lib/libonnxruntime.so\""
