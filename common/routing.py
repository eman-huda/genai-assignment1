"""Task 2: balanced classifier batches, classifier training and hard-routed restoration.

Builds on the Task 1 modules (corruptions, data, losses_metrics) so the corruptions, split
and manifests are exactly the ones Task 1 used.
"""
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from .corruptions import CONDITIONS, apply_corruption, sample_params
from .data import to_tensor
from .losses_metrics import per_sample_metrics

EXPERT_NAMES = ["salt_pepper", "gaussian_blur", "occlusion"]
EXPERT_FOR_CLASS = {1: "salt_pepper", 2: "gaussian_blur", 3: "occlusion"}   # class 0 (clean) -> identity


class ForcedClassDataset(Dataset):
    """Index i encodes (image position, class) as i = pos * 4 + class. The sampler forces the
    condition; its severity is still sampled from the training ranges at load time."""

    def __init__(self, images_uint8):
        self.images = images_uint8

    def __len__(self):
        return len(self.images) * len(CONDITIONS)

    def __getitem__(self, i):
        pos, cls = divmod(int(i), len(CONDITIONS))
        clean = self.images[pos].astype(np.float32) / 255.0
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        params = sample_params(CONDITIONS[cls], rng, *clean.shape[:2])
        return to_tensor(apply_corruption(clean, params)), cls


class BalancedBatchSampler(Sampler):
    """Every batch holds exactly batch_size / 4 samples of each class. Each epoch every training
    image is used once, with its class assigned at random (25% of images per class)."""

    def __init__(self, n_images, batch_size, seed=42):
        assert batch_size % len(CONDITIONS) == 0, "batch size must be a multiple of 4"
        self.n, self.per_class = n_images, batch_size // len(CONDITIONS)
        self.rng = np.random.default_rng(seed)
        self.n_batches = (n_images // len(CONDITIONS)) // self.per_class

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        perm = self.rng.permutation(self.n)
        groups = [perm[c::len(CONDITIONS)] for c in range(len(CONDITIONS))]
        for b in range(self.n_batches):
            sl = slice(b * self.per_class, (b + 1) * self.per_class)
            batch = [int(p) * len(CONDITIONS) + c for c, g in enumerate(groups) for p in g[sl]]
            self.rng.shuffle(batch)
            yield batch


class ClassifierWithProbs(nn.Module):
    """ONNX wrapper: returns logits and softmax probabilities."""

    def __init__(self, clf):
        super().__init__()
        self.clf = clf

    def forward(self, x):
        logits = self.clf(x)
        return logits, torch.softmax(logits, dim=1)


@torch.no_grad()
def predict(clf, loader, device):
    """For ManifestDataset loaders (x, clean, label, index). Returns probs, labels, indices."""
    clf.eval()
    probs, labels, idx = [], [], []
    for batch in loader:
        probs.append(torch.softmax(clf(batch[0].to(device)).float(), 1).cpu())
        labels.append(batch[2])
        idx.append(batch[3])
    return torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(idx).numpy()


def classification_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    labels = list(range(len(CONDITIONS)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro",
                                                    zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s}, index=CONDITIONS)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(mp),
            "macro_recall": float(mr), "macro_f1": float(mf)}, per_class, cm


def _scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def fit_classifier(clf, config, train_loader, val_loader, *, epochs, lr, weight_decay, device,
                   best_path=None, last_path=None, patience=None, trial=None, log_fn=None, verbose=True):
    """AdamW + cosine decay, cross-entropy. Model selection on validation macro-F1.
    best_path is saved as {"state_dict", "config", "epoch", "val"} (the format Task 3 loads).
    last_path stores the full training state, so a disconnected session resumes."""
    use_amp = torch.cuda.is_available()
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    scaler = _scaler(use_amp)
    ce = nn.CrossEntropyLoss()
    history, best, start, bad, finished, extra = [], -math.inf, 0, 0, False, {}
    if last_path and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        clf.load_state_dict(ck["state_dict"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        history, best, bad, start, finished = ck["history"], ck["best"], ck["bad"], ck["epoch"] + 1, ck["finished"]
        extra = ck.get("extra", {})
        if verbose:
            print(f"Resumed from epoch {start} (best macro-F1 {best:.4f}, finished={finished})")
    if finished:
        return history, best
    if trial is not None:
        import optuna

    for ep in range(start, epochs):
        t0 = time.time()
        clf.train()
        tl, tc, n = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = clf(x)
            loss = ce(logits.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * x.size(0)
            tc += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        probs, labels, _ = predict(clf, val_loader, device)
        val_loss = float(F.nll_loss(torch.log(torch.from_numpy(probs).clamp_min(1e-8)), torch.from_numpy(labels)))
        m, _, _ = classification_metrics(labels, probs.argmax(1))
        lr_now = opt.param_groups[0]["lr"]
        sched.step()
        row = {"epoch": ep, "train_loss": tl / n, "train_acc": tc / n, "val_loss": val_loss,
               "val_acc": m["accuracy"], "val_macro_f1": m["macro_f1"], "lr": lr_now,
               "epoch_time_s": time.time() - t0}
        history.append(row)
        if log_fn:
            log_fn(row)
        if verbose:
            print(f"ep {ep + 1:3d}/{epochs} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} | "
                  f"val loss {val_loss:.4f} acc {m['accuracy']:.4f} F1 {m['macro_f1']:.4f} | {row['epoch_time_s']:.0f}s")
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            if best_path:
                torch.save({"state_dict": clf.state_dict(), "config": config, "epoch": ep, "val": m}, best_path)
        else:
            bad += 1
        if trial is not None:
            trial.report(m["macro_f1"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        stop = patience is not None and bad >= patience
        if last_path:
            torch.save({"state_dict": clf.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "scaler": scaler.state_dict(), "history": history, "best": best, "bad": bad,
                        "epoch": ep, "finished": stop or ep == epochs - 1, "extra": extra}, last_path)
        if stop:
            if verbose:
                print(f"Early stopping: no improvement for {patience} epochs")
            break
    return history, best


class HardRouter:
    """Classifier -> argmax -> specialist. Clean predictions use an identity bypass: the input is
    returned unchanged and no expert is run."""

    def __init__(self, classifier, experts):
        self.clf = classifier.eval()
        self.experts = {k: v.eval() for k, v in experts.items()}

    @torch.no_grad()
    def route(self, x, route_labels):
        out = x.clone()
        for cls, name in EXPERT_FOR_CLASS.items():
            sel = route_labels == cls
            if sel.any():
                out[sel] = self.experts[name](x[sel]).float().clamp(0, 1)
        return out

    @torch.no_grad()
    def __call__(self, x):
        probs = torch.softmax(self.clf(x).float(), 1)
        pred = probs.argmax(1)
        return self.route(x, pred), probs, pred


@torch.no_grad()
def evaluate_routing(router, loader, device):
    """Oracle routing (true manifest label) and predicted routing (classifier argmax) for every
    entry. Outputs identical to the clean target get NaN PSNR (it would be infinite)."""
    rows = []
    for x, y, lab, idx in loader:
        x, y, lab = x.to(device), y.to(device), lab.to(device)
        pred_out, probs, pred = router(x)
        res = {"in": x, "oracle": router.route(x, lab), "pred": pred_out}
        m = {k: per_sample_metrics(o, y) for k, o in res.items()}
        for j in range(x.size(0)):
            r = {"index": int(idx[j]), "true": int(lab[j]), "pred": int(pred[j]), "confidence": float(probs[j].max())}
            for c, name in enumerate(CONDITIONS):
                r[f"p_{name}"] = float(probs[j, c])
            for k in res:
                for met in ("psnr", "ssim", "l1"):
                    r[f"{k}_{met}"] = float(m[k][met][j])
            rows.append(r)
    df = pd.DataFrame(rows)
    df["correct"] = df.true == df.pred
    for k in ("in", "oracle", "pred"):
        df.loc[df[f"{k}_l1"] == 0, f"{k}_psnr"] = np.nan
    df["routing_loss_psnr"] = df.oracle_psnr - df.pred_psnr
    df["routing_loss_ssim"] = df.oracle_ssim - df.pred_ssim
    return df
