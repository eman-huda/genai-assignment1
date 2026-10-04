"""Helpers shared by the Task 4 scripts: config, data loading, device, LPIPS."""
import json
import os
import random
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from common.fs2k_data import (PairedSketchDataset, make_or_load_fs2k_split, prepare_fs2k_cache,  # noqa: E402
                              find_fs2k_root, to_model)


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = v if os.path.isabs(v) else os.path.join(ROOT, v)
    for sub in ("checkpoints", "onnx", "results", "figures", "optuna", "samples"):
        os.makedirs(os.path.join(cfg["paths"]["out_dir"], sub), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["paths"]["cache"]), exist_ok=True)
    return cfg


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def tensors(photos, sketches, styles):
    return (torch.stack([to_model(p) for p in photos]), torch.stack([to_model(s) for s in sketches]),
            torch.as_tensor(styles, dtype=torch.long))


def load_data(cfg):
    """Returns the augmented train dataset, validation tensors, test tensors and the raw cache."""
    p = cfg["paths"]
    root = None if os.path.exists(p["cache"]) else find_fs2k_root(p["fs2k_dir"])
    data = prepare_fs2k_cache(root, p["cache"], cfg["image_size"])
    tr, va = make_or_load_fs2k_split(data["train_styles"], p["split"], cfg["seed"], cfg["val_fraction"])
    P, S, ST = data["train_photos"], data["train_sketches"], data["train_styles"]
    train_ds = PairedSketchDataset(P[tr], S[tr], ST[tr], augment=True)
    return (train_ds, tensors(P[va], S[va], ST[va]),
            tensors(data["test_photos"], data["test_sketches"], data["test_styles"]), data)


def make_lpips(dev):
    import lpips
    fn = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    for q in fn.parameters():
        q.requires_grad_(False)
    return fn


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
