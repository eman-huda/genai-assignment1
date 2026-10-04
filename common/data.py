"""Dataset preparation and PyTorch datasets shared by Tasks 1, 2 and 3."""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .corruptions import CONDITIONS, COND2IDX, apply_corruption, sample_params


def _read_split_file(base, split):
    with open(os.path.join(base, "annotations", f"{split}.txt")) as f:
        return [ln.split()[0] for ln in f if ln.strip() and not ln.startswith("#")]


def prepare_pets_cache(data_root, cache_path, size=128):
    """Download Oxford-IIIT Pet once, convert to RGB, resize to size x size, cache as uint8."""
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}
    from torchvision.datasets import OxfordIIITPet
    for split in ("trainval", "test"):
        OxfordIIITPet(root=data_root, split=split, download=True)
    base = os.path.join(data_root, "oxford-iiit-pet")
    out = {}
    for split in ("trainval", "test"):
        names = _read_split_file(base, split)
        arr = np.zeros((len(names), size, size, 3), dtype=np.uint8)
        for i, n in enumerate(names):
            with Image.open(os.path.join(base, "images", n + ".jpg")) as im:
                arr[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BICUBIC))
        out[f"{split}_images"] = arr
        out[f"{split}_names"] = np.array(names)
    np.savez_compressed(cache_path, **out)
    return out


def make_or_load_split(names, split_path, seed=42, train_frac=0.8):
    """80/20 split of the official trainval set with seed 42, stored by index AND file name."""
    if os.path.exists(split_path):
        with open(split_path) as f:
            s = json.load(f)
        return np.array(s["train_idx"]), np.array(s["val_idx"])
    n = len(names)
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(round(train_frac * n))
    tr, va = sorted(perm[:n_train].tolist()), sorted(perm[n_train:].tolist())
    with open(split_path, "w") as f:
        json.dump({"seed": seed, "train_frac": train_frac, "train_idx": tr, "val_idx": va,
                   "train_names": [str(names[i]) for i in tr],
                   "val_names": [str(names[i]) for i in va]}, f)
    return np.array(tr), np.array(va)


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))


class RuntimeCorruptionDataset(Dataset):
    """Training dataset. A NEW condition and severity is sampled every time an image is loaded.

    Returns (corrupted, clean, condition_label). `conditions` restricts the sampled
    conditions (used later for the Task 2 specialists).
    """

    def __init__(self, images, conditions=CONDITIONS):
        self.images = images
        self.conditions = list(conditions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # torch's RNG is seeded differently in every DataLoader worker and every epoch,
        # so this gives fresh, non-duplicated corruptions while staying reproducible.
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        clean = self.images[i].astype(np.float32) / 255.0
        cond = self.conditions[int(rng.integers(len(self.conditions)))]
        params = sample_params(cond, rng, *clean.shape[:2])
        return to_tensor(apply_corruption(clean, params)), to_tensor(clean), COND2IDX[cond]


class ManifestDataset(Dataset):
    """Deterministic dataset for validation/test. Returns (corrupted, clean, label, entry_index)."""

    def __init__(self, images, entries):
        self.images = images
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, k):
        e = self.entries[k]
        clean = self.images[e["img_idx"]].astype(np.float32) / 255.0
        return to_tensor(apply_corruption(clean, e)), to_tensor(clean), COND2IDX[e["type"]], k
