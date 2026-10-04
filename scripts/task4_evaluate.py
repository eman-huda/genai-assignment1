"""Evaluate the Task 4 generator on the official FS2K test set (true style for every photo).

    python -m scripts.task4_evaluate [--ckpt outputs/task4/checkpoints/generator_best.pt] [--no-fid]
Writes test_metrics.csv/.tex (L1, PSNR, SSIM, LPIPS, FID, KID overall and per style),
style_control.csv/.tex and test_per_sample.csv to outputs/task4/results.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch

from scripts._task4_common import device, load_config, load_data, make_lpips
from common.fs2k_data import STYLE_NAMES, to_unit
from common.gan_train import evaluate_generator, generate, per_sample_metrics
from common.sketch_gan import build_models


def fid_kid(real, fake, dev):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    n = len(real)
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    kid = KernelInceptionDistance(subset_size=min(50, n), normalize=True).to(dev)
    for i in range(0, n, 64):
        r, f = to_unit(real[i:i + 64]).to(dev), to_unit(fake[i:i + 64]).to(dev)
        fid.update(r, real=True); fid.update(f, real=False); kid.update(r, real=True); kid.update(f, real=False)
    km, ks = kid.compute()
    return float(fid.compute()), float(km), float(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task4.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--no-fid", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = os.path.join(cfg["paths"]["out_dir"], "results")
    dev = device()
    ckpt = torch.load(args.ckpt or os.path.join(cfg["paths"]["out_dir"], "checkpoints", "generator_best.pt"),
                      map_location="cpu", weights_only=False)
    G, _ = build_models(ckpt["config"]); G.load_state_dict(ckpt["G"]); G = G.to(dev).eval()
    _, _, test, data = load_data(cfg)
    lpips_fn = make_lpips(dev)

    _, per, fake = evaluate_generator(G, *test, dev, lpips_fn)
    res = pd.DataFrame({k: v.numpy() for k, v in per.items()})
    res["style"] = [STYLE_NAMES[s] for s in test[2].numpy()]
    res["name"] = data["test_names"]
    res.to_csv(os.path.join(out, "test_per_sample.csv"), index=False)

    rows = []
    for name, sel in [("Overall", np.ones(len(res), bool))] + [(s, (res["style"] == s).values) for s in STYLE_NAMES]:
        r = {"subset": name, "n": int(sel.sum()),
             **{k.upper(): res.loc[sel, k].mean() for k in ("l1", "psnr", "ssim", "lpips")}}
        if not args.no_fid:
            r["FID"], r["KID_mean"], r["KID_std"] = fid_kid(test[1][sel], fake[sel], dev)
        rows.append(r)
    tbl = pd.DataFrame(rows).set_index("subset")
    tbl.to_csv(os.path.join(out, "test_metrics.csv"))
    open(os.path.join(out, "test_metrics.tex"), "w").write(tbl.to_latex(float_format="%.4f"))
    print(tbl.round(4).to_string())

    outs = [generate(G, test[0], torch.full_like(test[2], st), dev) for st in range(3)]
    lp = torch.stack([per_sample_metrics(o, test[1], lpips_fn, dev)["lpips"] for o in outs], 1)
    ss = torch.stack([per_sample_metrics(o, test[1])["ssim"] for o in outs], 1)
    rows = []
    for st in range(3):
        sel = test[2] == st
        rows.append({"true style": STYLE_NAMES[st], "n": int(sel.sum()),
                     **{f"SSIM as {STYLE_NAMES[k]}": float(ss[sel, k].mean()) for k in range(3)},
                     **{f"LPIPS as {STYLE_NAMES[k]}": float(lp[sel, k].mean()) for k in range(3)},
                     "best LPIPS is true style (%)": float((lp[sel].argmin(1) == st).float().mean() * 100)})
    st_tbl = pd.DataFrame(rows).set_index("true style")
    st_tbl.to_csv(os.path.join(out, "style_control.csv"))
    open(os.path.join(out, "style_control.tex"), "w").write(st_tbl.to_latex(float_format="%.3f"))
    print(st_tbl.round(3).to_string())


if __name__ == "__main__":
    main()
