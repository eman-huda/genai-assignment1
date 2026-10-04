"""Finds, loads and describes the ONNX models. Missing models are reported, not fatal.

A task can consist of several ONNX files (Task 2: one classifier and three specialists);
it is "ready" only when every file loads.
"""
import json
import os
import time
from pathlib import Path

import onnxruntime as ort

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "/app/models"))

SPECS = {
    "task1": {"name": "Universal Restoration", "meta": "task1_metadata.json",
              "files": {"main": "task1_universal_dae.onnx"}},
    "task2": {"name": "Hard-Routed Restoration", "meta": "hard_routing_meta.json",
              "files": {"classifier": "corruption_classifier.onnx",
                        "salt_pepper": "specialist_salt_pepper.onnx",
                        "gaussian_blur": "specialist_gaussian_blur.onnx",
                        "occlusion": "specialist_occlusion.onnx"}},
    "task3": {"name": "Soft Mixture-of-Experts Restoration", "meta": "task3_metadata.json",
              "files": {"main": "soft_moe.onnx"}},
    "task4": {"name": "Face-to-Sketch Generator", "meta": "task4_metadata.json",
              "files": {"main": "face2sketch_generator.onnx"}},
}
ALL_FILES = [f for s in SPECS.values() if s["files"] for f in s["files"].values()] + \
            [s["meta"] for s in SPECS.values() if s["meta"]]


def find(filename):
    """Look for a file anywhere under MODELS_DIR, so the folder layout is forgiving."""
    if not filename or not MODELS_DIR.exists():
        return None
    hits = sorted(MODELS_DIR.rglob(filename))
    return hits[0] if hits else None


class Registry:
    def __init__(self):
        self.sessions, self.meta, self.status, self.files = {}, {}, {}, {}

    def load_all(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
        for key, spec in SPECS.items():
            if spec["files"] is None:
                self.status[key] = "planned"
                continue
            sessions, files, problem = {}, {}, None
            for part, fname in spec["files"].items():
                path = find(fname)
                if path is None:
                    problem = "missing"
                    break
                t0 = time.perf_counter()
                try:
                    sessions[part] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
                except Exception as exc:          # corrupt or incompatible file
                    problem = f"error: {exc.__class__.__name__}"
                    break
                files[part] = {"file": path.name, "size_mb": round(path.stat().st_size / 1e6, 1),
                               "load_ms": round((time.perf_counter() - t0) * 1000, 1)}
            if problem:
                self.status[key] = problem
                continue
            meta_path = find(spec["meta"])
            self.meta[key] = json.load(open(meta_path)) if meta_path else {}
            self.sessions[key], self.files[key], self.status[key] = sessions, files, "ready"

    def session(self, key, part="main"):
        return self.sessions[key][part]

    def describe(self, key):
        spec, meta = SPECS[key], self.meta.get(key, {})
        info = {"key": key, "name": spec["name"], "status": self.status.get(key, "missing")}
        if key not in self.sessions:
            return info
        parts = []
        for part, s in self.sessions[key].items():
            parts.append({"part": part, **self.files[key][part],
                          "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in s.get_inputs()],
                          "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in s.get_outputs()]})
        info["parts"] = parts
        # convenience fields for single-file models
        info.update({k: parts[0][k] for k in ("file", "size_mb", "load_ms", "inputs", "outputs")} if len(parts) == 1 else {})
        for k in ("config", "params", "latent_dim", "best_epoch", "style_names", "test_overall", "test",
                  "onnx_check", "onnx_vs_pytorch_max_abs_diff", "classifier_config", "specialist_config",
                  "routing_rule", "cpu_ms", "class_order", "temperature", "branch_labels"):
            if k in meta:
                info[k] = meta[k]
        return info


registry = Registry()
