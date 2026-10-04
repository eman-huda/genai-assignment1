"""Training and evaluation loops for the Task 4 conditional GAN (shared by the notebook and scripts)."""
import math
import time

import numpy as np
import torch
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from .fs2k_data import to_unit
from .sketch_gan import build_models, d_losses, g_losses


def selection_objective(lpips_value, ssim_value):
    """Model selection (lower is better). Fixed and independent of the tuned lambda_L1:
    LPIPS rewards perceptual and texture similarity, (1 - SSIM) rewards structure."""
    return float(lpips_value) + (1.0 - float(ssim_value))


def linear_decay(epochs):
    """pix2pix schedule: constant learning rate for the first half, then linear decay towards 0."""
    start = epochs // 2
    return lambda e: 1.0 if e < start else max(0.0, 1.0 - (e - start + 1) / (epochs - start + 1))


@torch.no_grad()
def generate(G, photos, styles, device, bs=64):
    """photos (N,3,H,W) in [-1,1], styles (N,) long -> generated sketches in [-1,1] on CPU."""
    G.eval()
    out = [G(photos[i:i + bs].to(device), styles[i:i + bs].to(device)).cpu() for i in range(0, len(photos), bs)]
    return torch.cat(out)


@torch.no_grad()
def per_sample_metrics(fake, real, lpips_fn=None, device="cpu", bs=64):
    """fake/real in [-1,1]. L1, PSNR and SSIM on [0,1]; LPIPS (AlexNet) on [-1,1]."""
    f01, r01 = to_unit(fake), to_unit(real)
    l1 = (f01 - r01).abs().flatten(1).mean(1)
    mse = ((f01 - r01) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    psnr = 10 * torch.log10(1.0 / mse)
    s = ssim(f01, r01, data_range=1.0, size_average=False)
    out = {"l1": l1, "psnr": psnr, "ssim": s}
    if lpips_fn is not None:
        lp = [lpips_fn(fake[i:i + bs].to(device), real[i:i + bs].to(device)).flatten().cpu()
              for i in range(0, len(fake), bs)]
        out["lpips"] = torch.cat(lp)
    return out


def evaluate_generator(G, photos, sketches, styles, device, lpips_fn):
    fake = generate(G, photos, styles, device)
    m = per_sample_metrics(fake, sketches, lpips_fn, device)
    summary = {k: float(v.mean()) for k, v in m.items()}
    summary["objective"] = selection_objective(summary["lpips"], summary["ssim"])
    for st in range(3):
        sel = styles == st
        if sel.any():
            summary[f"ssim_style{st + 1}"] = float(m["ssim"][sel].mean())
            summary[f"lpips_style{st + 1}"] = float(m["lpips"][sel].mean())
    return summary, m, fake


def fit_gan(cfg, train_ds, val, epochs, device, lpips_fn, log_fn=None, image_fn=None, trial=None,
            ckpt_path=None, eval_every=5, sample_every=10, num_workers=2, seed=42):
    """Train one configuration.

    val = (photos, sketches, styles) tensors for the validation split.
    Logs the four loss components separately every epoch, validation metrics every eval_every epochs,
    and fixed validation samples every sample_every epochs (through image_fn).
    Returns (best_objective, history, G_with_best_weights_loaded_if_ckpt).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    G, D = build_models(cfg)
    G, D = G.to(device), D.to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, linear_decay(epochs))
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, linear_decay(epochs))
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                        persistent_workers=num_workers > 0,
                        generator=torch.Generator().manual_seed(seed))
    lam = cfg["lambda_l1"]
    best, history = math.inf, []
    for epoch in range(1, epochs + 1):
        G.train(); D.train()
        t0 = time.time()
        sums = {k: 0.0 for k in ("d_real", "d_fake", "g_adv", "g_l1", "d_acc_real", "d_acc_fake")}
        n = 0
        for x, y, s in loader:
            x, y, s = x.to(device, non_blocking=True), y.to(device, non_blocking=True), s.to(device)
            fake = G(x, s)
            l_real, l_fake, acc_r, acc_f = d_losses(D, x, y, fake, s)
            opt_d.zero_grad(set_to_none=True)
            (0.5 * (l_real + l_fake)).backward()
            opt_d.step()
            adv, l1 = g_losses(D, x, y, fake, s)
            opt_g.zero_grad(set_to_none=True)
            (adv + lam * l1).backward()
            opt_g.step()
            b = len(x); n += b
            for k, v in (("d_real", l_real), ("d_fake", l_fake), ("g_adv", adv), ("g_l1", l1),
                         ("d_acc_real", acc_r), ("d_acc_fake", acc_f)):
                sums[k] += v.item() * b
        sch_g.step(); sch_d.step()
        rec = {"epoch": epoch, "time_s": time.time() - t0, "lr_g": opt_g.param_groups[0]["lr"],
               **{f"train/{k}": v / n for k, v in sums.items()}}
        if epoch % eval_every == 0 or epoch == epochs:
            summ, _, _ = evaluate_generator(G, *val, device, lpips_fn)
            rec.update({f"val/{k}": v for k, v in summ.items()})
            if summ["objective"] < best:
                best = summ["objective"]
                rec["val/best_objective"] = best
                if ckpt_path:
                    torch.save({"G": G.state_dict(), "D": D.state_dict(), "config": cfg,
                                "epoch": epoch, "val": summ}, ckpt_path)
            if trial is not None:
                import optuna
                trial.report(summ["objective"], epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            print(f"ep {epoch:03d} | D real {rec['train/d_real']:.3f} fake {rec['train/d_fake']:.3f} | "
                  f"G adv {rec['train/g_adv']:.3f} L1 {rec['train/g_l1']:.4f} | val SSIM {summ['ssim']:.4f} "
                  f"LPIPS {summ['lpips']:.4f} obj {summ['objective']:.4f}")
        history.append(rec)
        if log_fn:
            log_fn(rec)
        if image_fn and (epoch == 1 or epoch % sample_every == 0 or epoch == epochs):
            image_fn(G, epoch)
    if ckpt_path:
        G.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["G"])
    return best, history, G


class SketchExportWrapper(torch.nn.Module):
    """ONNX wrapper: photo in [0,1] RGB NCHW + style index (int64, 0..2) -> sketch in [0,1]."""

    def __init__(self, G):
        super().__init__()
        self.G = G

    def forward(self, photo, style):
        return (self.G(photo * 2 - 1, style) + 1) / 2
