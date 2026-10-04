# Generative AI, Assignment 1

**Eman Ul Huda (23i-2559), Department of Computer Science, FAST-NUCES**

Four generative systems, trained and evaluated in Google Colab and deployed as one browser application:

| Task | System | Data | Main result |
|---|---|---|---|
| 1 | Universal multi-corruption denoising autoencoder | Oxford-IIIT Pet, 128 × 128 | Test SSIM 0.704 |
| 2 | Corruption classifier + hard-routed specialist autoencoders | Oxford-IIIT Pet | Classifier accuracy 99.86 %, test SSIM 0.735 |
| 3 | Jointly trained soft mixture of experts | Oxford-IIIT Pet | Validation SSIM 0.839 (hard routing 0.783) |
| 4 | Style-conditioned face-to-sketch conditional GAN | FS2K | Test SSIM 0.463, LPIPS 0.211, FID 37.6 |

- **Report (IEEE format):** [`report/EmanUlHuda_23i-2559.pdf`](report/EmanUlHuda_23i-2559.pdf) (LaTeX source in `report/`)
- **Demonstration video:** https://youtu.be/B2mOOliPXoI
- **Interface design:** Google Stitch (prompt in `STITCH_PROMPT.md`, screenshots in the report)

## Run the application (one command)

You need Docker with Docker Compose v2. From the repository root:

```bash
docker compose up --build
```

Then open **http://localhost:8080**. On first start, the `models` service downloads the trained ONNX models from the shared Google Drive folder set in `.env` (`MODELS_FOLDER_URL`). You can also place the files in `./models` yourself; the list is in `README_APP.md`. To stop the app, press `Ctrl+C`, then run `docker compose down`.

In GitHub Codespaces, use `./codespaces-up.sh --build` instead. Some codespaces block Docker's internal network, and this script runs the services on the codespace's own network.

`README_APP.md` documents the architecture, the API endpoints and troubleshooting. The API documentation is also available at http://localhost:8080/api/docs while the app runs.

## Repository structure

| Path | Contents |
|---|---|
| `notebooks/` | Executed Colab notebooks for Tasks 1 to 4: data preparation, training, Optuna studies, evaluation, ONNX export and W&B logging |
| `common/` | Shared Python modules: corruptions and data (Tasks 1 to 3), autoencoder, classifier, routing, mixture of experts, FS2K data and the GAN (Task 4) |
| `scripts/`, `configs/` | Command-line scripts for Task 4 (`README_TASK4.md`) |
| `outputs/task1` to `outputs/task4` | Optuna studies (`.db`, trial CSVs, summaries), result tables (CSV and LaTeX), figures, ONNX metadata |
| `data/`, `manifests/` | Train/validation splits (seed 42) and the fixed validation and test corruption manifests |
| `backend/` | FastAPI service running the ONNX models (Dockerfile included) |
| `frontend/` | React + Tailwind CSS interface served by nginx (Dockerfile included) |
| `deploy/` | Model download at start-up, and Colab helpers that gather the models and outputs |
| `docker-compose.yml` | Starts the whole application |
| `report/` | Technical report (PDF and LaTeX source) |

## Reproduce the training

The notebooks run in Google Colab with a T4 GPU, and store everything under `MyDrive/genai-assignment1/`. Run them in this order:

1. `task1_universal_restoration.ipynb`: also builds the image cache, the seed-42 split and the corruption manifests that Tasks 2 and 3 reuse.
2. `task2_hard_routing.ipynb`
3. `task3_soft_moe.ipynb`: starts from the Task 2 checkpoints.
4. `task4_face2sketch_cgan.ipynb`: independent of Tasks 1 to 3. It can also be run with the scripts in `scripts/`.

Install the dependencies with `pip install -r requirements.txt`.

Experiments are tracked in Weights & Biases (project `genai-assignment1`). Each notebook needs the Colab secret `WANDB_API_KEY`.

Trained checkpoints and ONNX models are not stored in Git. They are shared through the Google Drive folder configured in `.env`.
