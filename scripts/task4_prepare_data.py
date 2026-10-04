"""Prepare FS2K for Task 4: pairing checks, 128x128 cache and the stratified 15% validation split.

Usage:
    python -m scripts.task4_prepare_data --config configs/task4.yaml [--download]
--download fetches the official archive from the Google Drive link in the FS2K README (needs gdown).
"""
import argparse
import os
import zipfile

import numpy as np

from scripts._task4_common import load_config, load_data
from common.fs2k_data import FS2K_GDRIVE_ID, STYLE_NAMES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task4.yaml")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fs2k_dir = cfg["paths"]["fs2k_dir"]
    if args.download and not os.path.exists(fs2k_dir):
        import gdown
        parent = os.path.dirname(fs2k_dir)
        os.makedirs(parent, exist_ok=True)
        zip_path = os.path.join(parent, "FS2K.zip")
        gdown.download(id=FS2K_GDRIVE_ID, output=zip_path, quiet=False)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(parent)
    train_ds, val, test, _ = load_data(cfg)
    print(f"train {len(train_ds)} | val {len(val[2])} | test {len(test[2])}")
    for name, st in (("train", train_ds.styles), ("val", val[2].numpy()), ("test", test[2].numpy())):
        print(f"  {name:5s}", dict(zip(STYLE_NAMES, np.bincount(st, minlength=3).tolist())))
    print("Pairing report:", cfg["paths"]["cache"].replace(".npz", "_report.json"))


if __name__ == "__main__":
    main()
