"""FastAPI backend for the four Generative AI workspaces.

Task 1 (universal restoration) and Task 4 (face-to-sketch) are live.
Task 2 (hard routing) and Task 3 (soft mixture of experts) are planned: their endpoints
exist and return HTTP 501 until their ONNX models are added.
"""
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image

from . import imaging as im
from .corruptions import (BLUR_KERNELS, CONDITIONS, TEST_LEVELS, apply_corruption, sample_params,
                          sample_rectangles)
from .registry import MODELS_DIR, registry

app = FastAPI(title="GenAI Assignment 1 API", version="1.0", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SAMPLES_DIR = MODELS_DIR / "samples"
SAMPLE_EXT = (".png", ".jpg", ".jpeg", ".webp")
STARTED = time.time()


@app.on_event("startup")
def _load():
    registry.load_all()


def _require(key):
    status = registry.status.get(key)
    if status == "planned":
        raise HTTPException(501, f"{registry.describe(key)['name']} is not deployed yet.")
    if key not in registry.sessions:
        raise HTTPException(503, f"The {key} model is not available (status: {status}). "
                                 f"Check that its ONNX files are in the models folder.")
    return registry.sessions[key]


def _samples():
    if not SAMPLES_DIR.exists():
        return []
    return sorted(p.name for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in SAMPLE_EXT)


async def _read_image(file: Optional[UploadFile], sample: Optional[str]) -> Image.Image:
    if file is not None and file.filename:
        try:
            return im.decode_upload(await file.read())
        except im.ImageError as exc:
            raise HTTPException(400, str(exc))
    if sample:
        if not re.fullmatch(r"[\w\-. ]+", sample) or sample not in _samples():
            raise HTTPException(404, f"Sample '{sample}' was not found.")
        return Image.open(SAMPLES_DIR / sample).convert("RGB")
    raise HTTPException(400, "Provide an image: upload a file or choose a sample.")


def _ms(t0):
    return round((time.perf_counter() - t0) * 1000, 2)


# ----------------------------------------------------------------------------- system
@app.get("/api/health")
def health():
    return {"status": "ok", "uptime_s": round(time.time() - STARTED),
            "onnxruntime": ort.__version__, "providers": ort.get_available_providers(),
            "models": {k: registry.status.get(k) for k in ("task1", "task2", "task3", "task4")},
            "samples": len(_samples())}


@app.get("/api/models")
def models():
    return [registry.describe(k) for k in ("task1", "task2", "task3", "task4")]


@app.get("/api/samples")
def samples():
    return {"samples": _samples()}


@app.get("/api/samples/{name}")
def sample_file(name: str):
    if name not in _samples():
        raise HTTPException(404, "Sample not found.")
    return FileResponse(SAMPLES_DIR / name)


# ----------------------------------------------------------------------------- task 1
def _corruption_params(kind: str, severity: str, seed: int) -> dict:
    """Fixed test levels (low/medium/high) exactly as in the Task 1 test manifest,
    or 'random' to sample from the training ranges, as in the training data loader."""
    rng = np.random.default_rng(seed)
    if severity == "random":
        return sample_params(kind, rng, im.IMG_SIZE, im.IMG_SIZE)
    levels = dict(TEST_LEVELS[kind])
    if severity not in levels:
        raise HTTPException(400, "Severity must be low, medium, high or random.")
    p = {"type": kind, "severity": severity, "seed": seed, **levels[severity]}
    if kind == "occlusion":
        rects, cov = sample_rectangles(im.IMG_SIZE, im.IMG_SIZE, p["n_rects"], p["coverage"], rng)
        p.update({"target_coverage": p["coverage"], "coverage": cov, "rects": rects})
    return p


def _json_safe(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (list, tuple)):
            v = [[int(a) for a in r] if isinstance(r, (list, tuple)) else r for r in v]
        out[k] = v
    return out


async def _restoration_input(file, sample, corruption, severity, seed):
    """Shared by Tasks 1 and 2: load the image, resize as in training, optionally corrupt it.
    corruption: 'none' = use the image as it is (an already corrupted upload, no reference);
    'clean' = no corruption, the image is also the reference; or one of the corruption types."""
    allowed = ["none"] + CONDITIONS
    if corruption not in allowed:
        raise HTTPException(400, f"corruption must be one of {allowed}")
    img = await _read_image(file, sample)
    t0 = time.perf_counter()
    base = im.task1_preprocess(img)
    seed = int(seed) if seed is not None else int(np.random.default_rng().integers(0, 2**31 - 1))
    if corruption == "none":
        reference, x, params = None, base, {"type": "none", "note": "image used as uploaded"}
    elif corruption == "clean":
        reference, x, params = base, base, {"type": "clean"}
    else:
        reference = base
        params = _corruption_params(corruption, severity, seed)
        x = np.clip(apply_corruption(base, params), 0, 1).astype(np.float32)
    return x, reference, params, _ms(t0)


def _restoration_result(x, restored, reference, params, timing):
    result = {"input_image": im.png_data_url(x), "restored_image": im.png_data_url(restored),
              "corruption": _json_safe(params), "image_size": im.IMG_SIZE, "timing_ms": timing}
    if reference is not None:
        result["reference_image"] = im.png_data_url(reference)
        result["error_map"] = im.png_data_url(im.error_map(restored, reference))
        result["metrics"] = {"input": {"psnr": im.psnr(x, reference), "ssim": im.ssim(x, reference)},
                             "restored": {"psnr": im.psnr(restored, reference), "ssim": im.ssim(restored, reference)}}
    return result


@app.post("/api/universal/restore")
async def universal_restore(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                            corruption: str = Form("none"), severity: str = Form("medium"),
                            seed: Optional[int] = Form(None)):
    sess = _require("task1")["main"]
    t_all = time.perf_counter()
    x, reference, params, pre_ms = await _restoration_input(file, sample, corruption, severity, seed)
    t0 = time.perf_counter()
    restored = im.from_nchw(sess.run(None, {"input": im.to_nchw(x)})[0])
    timing = {"preprocess": pre_ms, "inference": _ms(t0)}
    result = _restoration_result(x, restored, reference, params, timing)
    timing["total"] = _ms(t_all)
    return result


# ----------------------------------------------------------------------------- task 2
EXPERT_LABELS = {"clean": "Identity bypass (no expert run)", "salt_pepper": "Salt-and-pepper specialist",
                 "gaussian_blur": "Gaussian blur specialist", "occlusion": "Occlusion specialist"}


@app.post("/api/hard-routing/restore")
async def hard_routing_restore(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                               corruption: str = Form("none"), severity: str = Form("medium"),
                               seed: Optional[int] = Form(None), routing: str = Form("predicted")):
    """routing='predicted': the classifier's argmax picks the branch (the operational system).
    routing='oracle': the known corruption label picks it (only when the app applied the corruption)."""
    sessions = _require("task2")
    t_all = time.perf_counter()
    if routing not in ("predicted", "oracle"):
        raise HTTPException(400, "routing must be 'predicted' or 'oracle'.")
    if routing == "oracle" and corruption == "none":
        raise HTTPException(400, "Oracle routing needs a known corruption. Choose a corruption in the app, "
                                 "or switch to classifier routing for an already corrupted upload.")
    x, reference, params, pre_ms = await _restoration_input(file, sample, corruption, severity, seed)
    xb = im.to_nchw(x)

    t0 = time.perf_counter()
    _, probs = sessions["classifier"].run(["logits", "probs"], {"input": xb})
    clf_ms = _ms(t0)
    probs = probs[0].astype(float)
    predicted = CONDITIONS[int(np.argmax(probs))]
    true_class = None if corruption == "none" else corruption
    routed = predicted if routing == "predicted" else true_class

    t0 = time.perf_counter()
    if routed == "clean":
        restored = x.copy()                                   # identity bypass: no expert is run
    else:
        restored = im.from_nchw(sessions[routed].run(None, {"input": xb})[0])
    expert_ms = _ms(t0)

    timing = {"preprocess": pre_ms, "classifier": clf_ms, "expert": expert_ms}
    result = _restoration_result(x, restored, reference, params, timing)
    result["classifier"] = {"probs": {c: float(p) for c, p in zip(CONDITIONS, probs)},
                            "predicted": predicted, "confidence": float(probs.max())}
    result["routing"] = {"mode": routing, "routed_class": routed, "expert": EXPERT_LABELS[routed],
                         "true_class": true_class,
                         "classifier_correct": None if true_class is None else predicted == true_class}
    timing["total"] = _ms(t_all)
    return result


# ----------------------------------------------------------------------------- task 3
@app.post("/api/soft-moe/restore")
async def soft_moe_restore(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                           corruption: str = Form("none"), severity: str = Form("medium"),
                           seed: Optional[int] = Form(None)):
    """One ONNX graph: gate -> softmax(logits / T) -> weighted sum of identity and three experts."""
    sess = _require("task3")["main"]
    t_all = time.perf_counter()
    x, reference, params, pre_ms = await _restoration_input(file, sample, corruption, severity, seed)
    t0 = time.perf_counter()
    restored, weights, logits = sess.run(["restored", "weights", "logits"], {"input": im.to_nchw(x)})
    infer_ms = _ms(t0)
    restored = im.from_nchw(restored)
    w = weights[0].astype(float)
    order = np.argsort(-w)
    timing = {"preprocess": pre_ms, "inference": infer_ms}
    result = _restoration_result(x, restored, reference, params, timing)
    result["routing"] = {
        "weights": {c: float(v) for c, v in zip(CONDITIONS, w)},
        "top_branch": CONDITIONS[int(order[0])],
        "contributors": [CONDITIONS[int(i)] for i in order if w[i] >= 0.10],     # branches with at least 10% weight
        "gate_class": CONDITIONS[int(np.argmax(logits[0]))],
        "entropy": float(-(np.clip(w, 1e-8, 1) * np.log(np.clip(w, 1e-8, 1))).sum() / np.log(len(w))),
        "temperature": registry.meta.get("task3", {}).get("temperature"),
        "true_class": None if corruption == "none" else corruption,
    }
    timing["total"] = _ms(t_all)
    return result


# ----------------------------------------------------------------------------- task 4
@app.post("/api/sketch")
async def sketch(file: Optional[UploadFile] = File(None), style: int = Form(1),
                 all_styles: bool = Form(False)):
    sess = _require("task4")["main"]
    t_all = time.perf_counter()
    if style not in (1, 2, 3):
        raise HTTPException(400, "style must be 1, 2 or 3.")
    img = await _read_image(file, None)
    t0 = time.perf_counter()
    x = im.task4_preprocess(img)
    pre_ms = _ms(t0)
    styles = [1, 2, 3] if all_styles else [style]
    batch = np.repeat(im.to_nchw(x), len(styles), axis=0)
    t0 = time.perf_counter()
    out = sess.run(None, {"photo": batch, "style": np.array([s - 1 for s in styles], dtype=np.int64)})[0]
    infer_ms = _ms(t0)
    sketches = []
    for i, s in enumerate(styles):
        sk = np.clip(out[i].transpose(1, 2, 0), 0, 1)
        sketches.append({"style": s, "image": im.png_data_url(sk), "image_512": im.png_data_url(sk, upscale=4)})
    return {"photo_input": im.png_data_url(x), "sketches": sketches, "image_size": im.IMG_SIZE,
            "timing_ms": {"preprocess": pre_ms, "inference": infer_ms, "total": _ms(t_all)}}
