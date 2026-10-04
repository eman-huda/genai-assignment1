# Task 4: Style-conditioned face-to-sketch generation (conditional GAN)

A pix2pix-style conditional GAN that turns a face photograph into a sketch in one of the three FS2K styles.

- **Generator:** U-Net (7 down and 7 up stages, skip connection at every scale, 128 x 128).
- **Discriminator:** 70 x 70 PatchGAN. For a 128 x 128 input it outputs a 14 x 14 grid of real/fake logits.
- **Style condition:** a learned `nn.Embedding(3, E)` in each network. The style vector is expanded to a spatial map and concatenated to the input of both networks. In the generator it is concatenated again at the bottleneck.
- **Generator loss:** `L_G = BCE_adv + lambda_L1 * L1(y, G(x, s))`.
- **Discriminator loss:** `0.5 * (BCE_real + BCE_fake)`, both computed with logits.

## Files

| Path | Purpose |
|---|---|
| `notebooks/Task4_Face2Sketch_cGAN.ipynb` | Colab notebook: full pipeline plus every report figure and table |
| `common/fs2k_data.py` | FS2K pairing, 128 x 128 cache, stratified split, paired augmentation, dataset |
| `common/sketch_gan.py` | Generator, discriminator, style embedding, losses |
| `common/gan_train.py` | Training loop, metrics (L1, PSNR, SSIM, LPIPS), ONNX wrapper |
| `configs/task4.yaml` | Paths, Optuna search space, training schedule |
| `scripts/task4_prepare_data.py` | Download (optional), pairing checks, cache, split |
| `scripts/task4_train.py` | `--stage optuna` (study) and `--stage final` (full retraining) |
| `scripts/task4_evaluate.py` | Test metrics overall and per style, FID and KID, style-control check |
| `scripts/task4_export_onnx.py` | ONNX export, PyTorch comparison for all styles, latency, metadata |
| `requirements-task4.txt` | Python dependencies |

## Data

FS2K is not stored in this repository. You can get it in either of two ways:

- Run `python -m scripts.task4_prepare_data --download`, which fetches the official archive.
- Download it manually from https://github.com/DengPingFan/FS2K and extract it so that this folder exists: `data/raw/FS2K/{photo, sketch, anno_train.json, anno_test.json}`.

The official train/test split is used. The validation set is 15% of the official training portion, stratified by style, with seed 42. The official test set is used only by `task4_evaluate.py`.

Each pair is matched with the authors' naming rule, `photo1/image0110` -> `sketch1/sketch0110`. Missing pairs and size mismatches are written to `data/fs2k_128_report.json`.

## Commands (run from the repository root)

```bash
pip install -r requirements-task4.txt
python -m scripts.task4_prepare_data --download        # or place FS2K manually, then omit --download
python -m scripts.task4_train --stage optuna           # 16 trials x 20 epochs, resumable
python -m scripts.task4_train --stage final            # best trial, 200 epochs
python -m scripts.task4_evaluate
python -m scripts.task4_export_onnx
```

All outputs are written to `outputs/task4/`:

- `checkpoints/generator_best.pt`
- `optuna/` (SQLite study, trials.csv, summary)
- `results/` (CSV and LaTeX tables)
- `onnx/face2sketch_generator.onnx` and `onnx/task4_metadata.json`

Experiment tracking uses Weights & Biases, project `genai-assignment1`. Use `wandb login` first, or add `--no-wandb` to run without it.

## ONNX interface (used by the FastAPI backend)

| Name | Shape and type | Meaning |
|---|---|---|
| input `photo` | `(N, 3, 128, 128)` float32 | RGB in [0, 1] |
| input `style` | `(N,)` int64 | 0 = Style 1, 1 = Style 2, 2 = Style 3 |
| output `sketch` | `(N, 3, 128, 128)` | sketch in [0, 1] |

Preprocessing: convert to RGB, resize to 128 x 128 with `cv2.INTER_AREA`, then divide by 255. The discriminator is not exported.

## Trained model files

Model files are too large for the repository. They must be provided as either:

- a documented download link (for example the W&B artefact `task4-face2sketch` or a Google Drive link), or
- Git LFS: `git lfs track "*.onnx" "*.pt"`.
