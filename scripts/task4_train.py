"""Train the Task 4 conditional GAN.

    python -m scripts.task4_train --stage optuna      # Optuna study (resumable, SQLite in outputs/task4/optuna)
    python -m scripts.task4_train --stage final       # retrain the best trial for the full schedule
    python -m scripts.task4_train --stage final --params '{"lr_g": 2e-4, ...}'   # explicit config
Add --no-wandb to run without Weights & Biases.
"""
import argparse
import json
import os

import optuna
import pandas as pd
import torch

from scripts._task4_common import device, load_config, load_data, make_lpips, save_json, seed_everything
from common.fs2k_data import to_unit
from common.gan_train import fit_gan, generate


def suggest(trial, space):
    cfg = {}
    for name, s in space.items():
        if s["type"] == "categorical":
            cfg[name] = trial.suggest_categorical(name, s["choices"])
        else:
            cfg[name] = trial.suggest_float(name, float(s["low"]), float(s["high"]), log=s.get("log", False))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task4.yaml")
    ap.add_argument("--stage", choices=["optuna", "final"], required=True)
    ap.add_argument("--params", default=None, help="JSON config for --stage final (default: best Optuna trial)")
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    import wandb
    from torchvision.utils import make_grid

    cfg = load_config(args.config)
    out = cfg["paths"]["out_dir"]
    dev = device()
    seed_everything(cfg["seed"])
    train_ds, val, _, _ = load_data(cfg)
    lpips_fn = make_lpips(dev)
    fixed = torch.cat([torch.where(val[2] == st)[0][:3] for st in range(3)])

    def image_fn_factory(save_dir):
        def image_fn(G, epoch):
            fake = generate(G, val[0][fixed], val[2][fixed], dev)
            grid = make_grid(to_unit(torch.cat([val[0][fixed], fake, val[1][fixed]])), nrow=len(fixed))
            wandb.log({"val_samples": wandb.Image(grid, caption="photo | generated | real"), "epoch": epoch})
            if save_dir:
                torch.save(fake, os.path.join(save_dir, f"fixed_val_epoch{epoch:03d}.pt"))
        return image_fn

    def run(params, epochs, name, group, trial=None, ckpt=None, eval_every=5, sample_every=10, save_dir=None):
        seed_everything(cfg["seed"])
        r = wandb.init(project=cfg["wandb_project"], name=name, group=group, job_type="task4",
                       config={**params, "epochs": epochs}, reinit=True)
        try:
            return fit_gan(params, train_ds, val, epochs, dev, lpips_fn, log_fn=wandb.log,
                           image_fn=image_fn_factory(save_dir), trial=trial, ckpt_path=ckpt,
                           eval_every=eval_every, sample_every=sample_every,
                           num_workers=cfg["num_workers"], seed=cfg["seed"])
        finally:
            r.finish()

    oc = cfg["optuna"]
    storage = f"sqlite:///{os.path.join(out, 'optuna', 'task4_study.db')}"
    study = optuna.create_study(study_name=oc["study_name"], storage=storage, load_if_exists=True,
                                direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg["seed"]),
                                pruner=optuna.pruners.MedianPruner(**oc["pruner"]))

    if args.stage == "optuna":
        if len(study.trials) == 0:
            study.enqueue_trial(oc["reference_trial"])

        def objective(trial):
            params = suggest(trial, oc["search_space"])
            best, _, _ = run(params, oc["epochs_per_trial"], f"t4-trial-{trial.number:03d}", "task4-optuna",
                             trial=trial, eval_every=oc["eval_every"], sample_every=oc["epochs_per_trial"])
            return best

        done = len([t for t in study.trials if t.state.name in ("COMPLETE", "PRUNED")])
        study.optimize(objective, n_trials=max(0, oc["trials"] - done), gc_after_trial=True)
        study.trials_dataframe().to_csv(os.path.join(out, "optuna", "trials.csv"), index=False)
        save_json({"search_space": oc["search_space"], "objective": oc["objective"],
                   "n_trials": len(study.trials),
                   "states": pd.Series([t.state.name for t in study.trials]).value_counts().to_dict(),
                   "best_trial": study.best_trial.number, "best_value": study.best_value,
                   "best_params": study.best_params}, os.path.join(out, "optuna", "study_summary.json"))
        print("Best:", study.best_params)
    else:
        params = json.loads(args.params) if args.params else dict(study.best_params)
        fc = cfg["final"]
        ckpt = os.path.join(out, "checkpoints", "generator_best.pt")
        best, hist, _ = run(params, fc["epochs"], "t4-final", "task4-final", ckpt=ckpt,
                            eval_every=fc["eval_every"], sample_every=fc["sample_every"],
                            save_dir=os.path.join(out, "samples"))
        pd.DataFrame(hist).to_csv(os.path.join(out, "results", "final_history.csv"), index=False)
        print(f"Best validation objective {best:.4f}; checkpoint {ckpt}")


if __name__ == "__main__":
    main()
