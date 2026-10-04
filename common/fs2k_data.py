"""FS2K data preparation and the paired photo-sketch dataset (Task 4).

Layout of the official release (github.com/DengPingFan/FS2K):
    FS2K/photo/photo{1,2,3}/imageXXXX.(jpg|png)
    FS2K/sketch/sketch{1,2,3}/sketchXXXX.(jpg|png)
    FS2K/anno_train.json, FS2K/anno_test.json   (list of dicts with "image_name" and "style")
The photo -> sketch name mapping follows the authors' tools/split_train_test.py.
"""
import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]
FS2K_GDRIVE_ID = "1saIMhQ3dc5_ftkfGmBPbCluRn_zy7QQp"     # official link in the FS2K README
_EXTS = (".jpg", ".png", ".jpeg", ".JPG", ".PNG", ".JPEG")


def find_fs2k_root(base):
    for root, _, files in os.walk(base):
        if "anno_train.json" in files and "anno_test.json" in files:
            return root
    raise FileNotFoundError(f"No folder containing anno_train.json and anno_test.json under {base}")


def _with_ext(path_no_ext):
    for e in _EXTS:
        if os.path.exists(path_no_ext + e):
            return path_no_ext + e
    return None


def pair_paths(root, image_name):
    """'photo1/image0110' -> (photo path, sketch path). Either may be None if missing."""
    image_name = os.path.splitext(image_name.replace("\\", "/"))[0]
    folder, stem = image_name.split("/")[-2:]
    photo = _with_ext(os.path.join(root, "photo", folder, stem))
    sketch = _with_ext(os.path.join(root, "sketch", folder.replace("photo", "sketch"),
                                    stem.replace("image", "sketch")))
    return photo, sketch


def load_rgb(path, size):
    """Read any FS2K image as RGB uint8 (grayscale expanded, alpha composited on white), resized."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Could not read {path}")
    if img.dtype != np.uint8:
        img = (img / (img.max() or 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        a = img[..., 3:4].astype(np.float32) / 255.0
        img = (img[..., :3].astype(np.float32) * a + 255.0 * (1 - a)).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_hw = img.shape[:2]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA), orig_hw


def _load_split(root, anno_file, size):
    anno = json.load(open(os.path.join(root, anno_file)))
    photos, sketches, styles, names = [], [], [], []
    missing, size_mismatch = [], 0
    for a in anno:
        p, s = pair_paths(root, a["image_name"])
        if p is None or s is None:
            missing.append(a["image_name"])
            continue
        pi, phw = load_rgb(p, size)
        si, shw = load_rgb(s, size)
        size_mismatch += int(phw != shw)
        photos.append(pi); sketches.append(si); styles.append(int(a["style"])); names.append(a["image_name"])
    return (np.stack(photos), np.stack(sketches), np.array(styles, np.int64), np.array(names),
            missing, size_mismatch)


def prepare_fs2k_cache(root, cache_path, size=128):
    """Build (or load) a resized cache of the official train and test sets, with pairing checks."""
    if os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=False)
        return {k: d[k] for k in d.files}
    out, report = {}, {}
    for split, f in (("train", "anno_train.json"), ("test", "anno_test.json")):
        ph, sk, st, nm, missing, mism = _load_split(root, f, size)
        out.update({f"{split}_photos": ph, f"{split}_sketches": sk, f"{split}_styles": st, f"{split}_names": nm})
        report[split] = {"pairs": int(len(nm)), "missing": missing, "photo_sketch_size_mismatch": mism,
                         "style_counts": np.bincount(st, minlength=3).tolist()}
    np.savez(cache_path, **out)
    with open(os.path.splitext(cache_path)[0] + "_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    return out


def make_or_load_fs2k_split(styles, path, seed=42, val_frac=0.15):
    """Stratified (by style) train/validation split of the OFFICIAL training portion."""
    if os.path.exists(path):
        s = json.load(open(path))
        return np.array(s["train_idx"]), np.array(s["val_idx"])
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(styles))
    tr, va = train_test_split(idx, test_size=val_frac, random_state=seed, stratify=styles)
    tr, va = np.sort(tr), np.sort(va)
    json.dump({"seed": seed, "val_frac": val_frac, "train_idx": tr.tolist(), "val_idx": va.tolist()},
              open(path, "w"))
    return tr, va


def to_model(img_uint8):
    """HWC uint8 -> CHW float tensor in [-1, 1] (generator uses tanh)."""
    t = torch.from_numpy(np.ascontiguousarray(img_uint8.transpose(2, 0, 1))).float()
    return t / 127.5 - 1.0


def to_unit(t):
    return ((t + 1) / 2).clamp(0, 1)


def paired_augment(photo, sketch, rng, jitter_size=142):
    """Identical spatial transform for both images: resize to jitter_size, same random crop
    back to the original size, same horizontal flip. Keeps pixel correspondence."""
    h, w = photo.shape[:2]
    p = cv2.resize(photo, (jitter_size, jitter_size), interpolation=cv2.INTER_LINEAR)
    s = cv2.resize(sketch, (jitter_size, jitter_size), interpolation=cv2.INTER_LINEAR)
    i = int(rng.integers(0, jitter_size - h + 1)); j = int(rng.integers(0, jitter_size - w + 1))
    p, s = p[i:i + h, j:j + w], s[i:i + h, j:j + w]
    if rng.random() < 0.5:
        p, s = p[:, ::-1], s[:, ::-1]
    return p, s


class PairedSketchDataset(Dataset):
    """Returns (photo, sketch, style) with photo and sketch in [-1, 1]."""

    def __init__(self, photos, sketches, styles, augment=False):
        self.photos, self.sketches, self.styles, self.augment = photos, sketches, styles, augment

    def __len__(self):
        return len(self.photos)

    def __getitem__(self, i):
        p, s = self.photos[i], self.sketches[i]
        if self.augment:
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            p, s = paired_augment(p, s, rng)
        return to_model(p), to_model(s), int(self.styles[i])
