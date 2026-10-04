"""Training/evaluation loops shared by the restoration tasks."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from .corruptions import CONDITIONS
from .losses_metrics import per_sample_metrics, restoration_loss, selection_objective


def make_loader(ds, batch_size, shuffle, num_workers=2):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=shuffle,
                      persistent_workers=num_workers > 0)


def train_one_epoch(model, loader, optimizer, alpha, device):
    model.train()
    tot = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
    n = 0
    for x, y, *_ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss, l1, s = restoration_loss(model(x), y, alpha)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        b = x.size(0)
        tot["loss"] += loss.item() * b
        tot["l1"] += l1.item() * b
        tot["ssim"] += s.item() * b
        n += b
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def evaluate(model, loader, device, return_per_sample=False):
    """Evaluate on a ManifestDataset loader. Also computes the 'no restoration' input baseline."""
    model.eval()
    keys = ["l1", "psnr", "ssim", "in_l1", "in_psnr", "in_ssim", "index", "cond"]
    rows = {k: [] for k in keys}
    for x, y, c, k in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        m = per_sample_metrics(model(x), y)
        mi = per_sample_metrics(x, y)
        for name in ("l1", "psnr", "ssim"):
            rows[name].append(m[name].cpu().numpy())
            rows["in_" + name].append(mi[name].cpu().numpy())
        rows["index"].append(k.numpy())
        rows["cond"].append(c.numpy())
    per = {k: np.concatenate(v) for k, v in rows.items()}
    summary = {name: float(per[name].mean()) for name in ("l1", "psnr", "ssim")}
    summary["objective"] = selection_objective(summary["l1"], summary["ssim"])
    for ci, cname in enumerate(CONDITIONS):
        sel = per["cond"] == ci
        if sel.any():
            summary[f"ssim_{cname}"] = float(per["ssim"][sel].mean())
            summary[f"psnr_{cname}"] = float(per["psnr"][sel].mean())
    return (summary, per) if return_per_sample else (summary, None)
