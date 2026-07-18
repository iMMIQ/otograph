#!/usr/bin/env python3
"""Export a fixed-16000-Hz Silero VAD ONNX from the silero-vad wheel's
torchscript model, so TensorRT can lower it.

The stock `silero_vad.onnx` carries a sample-rate-conditional `If` (the 16k vs
8k window-size branch) whose two branches have incompatible shapes — TensorRT
refuses it. The conditional lives only in the *outer* wrapper (context/sr
bookkeeping); the *inner* model `_model.forward(x, state) -> (prob, state)` is
pure compute with no control flow. We trace and export that inner model with sr
baked in (it never had sr — the outer model picked window size from sr). The
result is a drop-in for the stock model:

    inputs : input[1,576] (= context[64] ++ window[512]), state[2,1,128]
    outputs: prob[1,1], state_out[2,1,128]

`otograph` already assembles the [context, window] input and threads state, so
this needs no change there beyond not feeding the (now-absent) `sr` input.

Usage:  python3 scripts/export_vad_16k.py [silero_vad.jit] [out.onnx]
"""
import os, sys, importlib.resources, torch, onnx, onnx.shape_inference as si

def find_jit():
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        return sys.argv[1]
    try:  # from the installed silero_vad package
        import silero_vad  # noqa: F401
        with importlib.resources.path("silero_vad.data", "silero_vad.jit") as p:
            return str(p)
    except Exception:
        pass
    sys.exit("could not find silero_vad.jit; pass its path as the first arg "
             "(pip install silero-vad, or extract it from the wheel)")

def main():
    jit = find_jit()
    out = sys.argv[2] if len(sys.argv) > 2 else "model/silero_vad_16k.onnx"
    m = torch.jit.load(jit, map_location="cpu").eval()
    inner = m._model.eval()  # 16k inner model, stateless, no control flow

    x1 = torch.zeros(1, 576)         # context(64) ++ window(512)
    st = torch.zeros(2, 1, 128)      # (h, c)
    traced = torch.jit.trace(inner, (x1, st), check_trace=False, strict=False)
    torch.onnx.export(
        traced, (x1, st), out,
        opset_version=17, dynamo=False, do_constant_folding=True,
        input_names=["input", "state"],
        output_names=["prob", "state_out"],  # distinct name avoids input rename
    )
    g = si.infer_shapes(onnx.load(out))
    onnx.save(g, out)
    ifs = sum(1 for n in g.graph.node if n.op_type == "If")
    print(f"wrote {out}  (If nodes={ifs}, total={len(g.graph.node)}), "
          f"inputs={[i.name for i in g.graph.input]}, "
          f"outputs={[o.name for o in g.graph.output]}")

if __name__ == "__main__":
    main()
