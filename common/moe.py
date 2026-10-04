"""Task 3: jointly trained soft mixture-of-experts restoration.

Branch order matches CONDITIONS: [clean (identity), salt_pepper, gaussian_blur, occlusion],
so gate output k, routing weight w_k and corruption label k all refer to the same branch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from pytorch_msssim import ssim

from .corruptions import CONDITIONS, sample_params, apply_corruption

BRANCHES = list(CONDITIONS)
EXPERT_NAMES = BRANCHES[1:]          # salt_pepper, gaussian_blur, occlusion


class SoftMoERestorer(nn.Module):
    """x_hat = w0 * x + w1 * E_salt(x) + w2 * E_blur(x) + w3 * E_occ(x),  w = softmax(g(x) / T)."""

    def __init__(self, gate, experts, temperature=1.0):
        super().__init__()
        assert len(experts) == 3, "expected experts in order: salt_pepper, gaussian_blur, occlusion"
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def branch_outputs(self, x):
        """All four branch outputs stacked: (N, 4, 3, H, W). Branch 0 is the identity."""
        return torch.stack([x] + [e(x) for e in self.experts], dim=1)

    def forward(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits / self.temperature, dim=1)
        outs = self.branch_outputs(x)
        x_hat = (w[:, :, None, None, None] * outs).sum(dim=1)
        return x_hat, w, logits


def set_experts_trainable(model, trainable):
    for p in model.experts.parameters():
        p.requires_grad_(trainable)


def balance_loss(w):
    """sum_k (mean_batch(w_k) - 1/K)^2. On a class-balanced batch, perfect routing gives
    mean weights of exactly 1/K, so this term is zero at the ideal solution and only
    penalises collapse onto a subset of branches."""
    k = w.size(1)
    return ((w.mean(dim=0) - 1.0 / k) ** 2).sum()


def moe_loss(x_hat, y, logits, w, labels, lam):
    """L = l1*L1 + ssim*(1 - SSIM) + ce*CE + bal*L_balance.

    CE is computed on the raw gate logits (not logits / T), so the gate stays a calibrated
    corruption classifier and T only controls how sharp the routing is.
    """
    l1 = F.l1_loss(x_hat, y)
    s = ssim(x_hat, y, data_range=1.0, size_average=True)
    ce = F.cross_entropy(logits, labels)
    bal = balance_loss(w)
    total = lam["l1"] * l1 + lam["ssim"] * (1 - s) + lam["ce"] * ce + lam["bal"] * bal
    parts = {"l1": l1.detach(), "ssim": s.detach(), "ce": ce.detach(), "balance": bal.detach()}
    return total, parts


def _chw(img):
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()


class BalancedRuntimeDataset(Dataset):
    """Indexed by (image_index, condition_index). A fresh severity is sampled for the
    requested condition every time the item is loaded (runtime corruption)."""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        i, c = key
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        clean = self.images[i].astype(np.float32) / 255.0
        params = sample_params(CONDITIONS[c], rng, *clean.shape[:2])
        return _chw(apply_corruption(clean, params)), _chw(clean), c


class BalancedBatchSampler(Sampler):
    """Every batch has exactly batch_size / 4 images of each condition.
    Each image appears once per epoch; conditions are reshuffled every epoch."""

    def __init__(self, n_images, batch_size, n_classes=4, seed=42):
        assert batch_size % n_classes == 0, "batch size must be divisible by the number of conditions"
        self.n, self.bs, self.k, self.seed, self.epoch = n_images, batch_size, n_classes, seed, 0

    def __len__(self):
        return self.n // self.bs

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        order = rng.permutation(self.n)
        for b in range(len(self)):
            idx = order[b * self.bs:(b + 1) * self.bs]
            conds = np.repeat(np.arange(self.k), self.bs // self.k)
            rng.shuffle(conds)
            yield [(int(i), int(c)) for i, c in zip(idx, conds)]
