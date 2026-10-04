"""Loss and metrics for the restoration tasks."""
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def restoration_loss(x_hat, x, alpha):
    """L = alpha * L1 + (1 - alpha) * (1 - SSIM)."""
    l1 = F.l1_loss(x_hat, x)
    s = ssim(x_hat, x, data_range=1.0, size_average=True)
    return alpha * l1 + (1 - alpha) * (1 - s), l1.detach(), s.detach()


@torch.no_grad()
def per_sample_metrics(x_hat, x):
    """Per-image L1, PSNR (dB, capped at 100 for identical images) and SSIM."""
    x_hat = x_hat.float().clamp(0, 1)
    l1 = (x_hat - x).abs().flatten(1).mean(1)
    mse = ((x_hat - x) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    psnr = 10 * torch.log10(1.0 / mse)
    s = ssim(x_hat, x, data_range=1.0, size_average=False)
    return {"l1": l1, "psnr": psnr, "ssim": s}


def selection_objective(l1, ssim_value):
    """Model-selection objective (lower is better). Fixed weights, so it does NOT depend on
    the tuned loss weight alpha; otherwise Optuna could 'win' by changing the metric itself."""
    return float(l1) + (1.0 - float(ssim_value))
