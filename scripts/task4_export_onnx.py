"""Export the Task 4 generator to ONNX and verify it against PyTorch for all three styles.

    python -m scripts.task4_export_onnx [--ckpt ...] [--out outputs/task4/onnx/face2sketch_generator.onnx]
Inputs: photo (N,3,128,128) RGB float32 in [0,1], style (N,) int64 in {0,1,2}. Output: sketch in [0,1].
"""
import argparse
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from scripts._task4_common import load_config, load_data, save_json
from common.fs2k_data import STYLE_NAMES, to_unit
from common.gan_train import SketchExportWrapper
from common.sketch_gan import build_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task4.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    od = cfg["paths"]["out_dir"]
    ckpt = torch.load(args.ckpt or os.path.join(od, "checkpoints", "generator_best.pt"), map_location="cpu",
                      weights_only=False)
    out = args.out or os.path.join(od, "onnx", "face2sketch_generator.onnx")
    G, _ = build_models(ckpt["config"]); G.load_state_dict(ckpt["G"]); G.eval()
    wrapper = SketchExportWrapper(G).eval()
    size = cfg["image_size"]
    torch.onnx.export(wrapper, (torch.rand(1, 3, size, size), torch.zeros(1, dtype=torch.long)), out,
                      input_names=["photo", "style"], output_names=["sketch"], opset_version=17,
                      dynamic_axes={"photo": {0: "batch"}, "style": {0: "batch"}, "sketch": {0: "batch"}},
                      dynamo=False)
    onnx.checker.check_model(onnx.load(out))
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])

    _, _, test, _ = load_data(cfg)
    n = min(60, len(test[0]))
    xs = to_unit(test[0][:n])
    diffs = []
    for st in range(3):
        sty = torch.full((n,), st, dtype=torch.long)
        with torch.no_grad():
            pt = wrapper(xs, sty).numpy()
        diffs.append(float(np.abs(pt - sess.run(None, {"photo": xs.numpy(), "style": sty.numpy()})[0]).max()))
    assert max(diffs) < 1e-4, f"ONNX mismatch: {diffs}"
    one = {"photo": xs[:1].numpy(), "style": np.array([0], dtype=np.int64)}
    for _ in range(5):
        sess.run(None, one)
    lat = []
    for _ in range(50):
        t0 = time.perf_counter(); sess.run(None, one); lat.append((time.perf_counter() - t0) * 1000)
    save_json({"task": "face_to_sketch", "onnx_file": os.path.basename(out),
               "inputs": {"photo": {"shape": [None, 3, size, size], "range": [0, 1], "layout": "NCHW RGB float32"},
                          "style": {"shape": [None], "dtype": "int64",
                                    "values": {str(i): s for i, s in enumerate(STYLE_NAMES)}}},
               "outputs": {"sketch": {"shape": [None, 3, size, size], "range": [0, 1]}},
               "style_names": STYLE_NAMES, "config": ckpt["config"], "opset": 17,
               "onnx_vs_pytorch_max_abs_diff": max(diffs),
               "cpu_latency_ms_batch1": {"mean": float(np.mean(lat)), "p95": float(np.percentile(lat, 95))}},
              os.path.join(os.path.dirname(out), "task4_metadata.json"))
    print(f"Exported {out} | max diff per style {diffs} | CPU latency {np.mean(lat):.1f} ms")


if __name__ == "__main__":
    main()
