# Restore & Sketch: web application (Tasks 1 to 4)

One browser application for all four tasks:
- React and Tailwind CSS frontend, served by nginx.
- FastAPI backend that runs the ONNX models with ONNX Runtime (CPU).
- Docker Compose, which starts everything with one command.

| Workspace | Task | Status |
|---|---|---|
| Universal Restoration | 1 | Live |
| Hard-Routed Restoration | 2 | Live |
| Soft Mixture-of-Experts Restoration | 3 | Live |
| Face-to-Sketch Generator | 4 | Live |

## Start the application

You need Docker with Docker Compose v2. Run these commands from the repository root:

```bash
cp .env.example .env    # only if .env is not already in the repository
docker compose up --build
```

Then open **http://localhost:8080**.

At start-up the `models` service checks `./models`. If any model file is missing, it downloads the shared Google Drive folder given by `MODELS_FOLDER_URL` in `.env`. You can also place the files in `./models` yourself:

```
models/
├── task1_universal_dae.onnx
├── task1_metadata.json
├── corruption_classifier.onnx
├── specialist_salt_pepper.onnx
├── specialist_gaussian_blur.onnx
├── specialist_occlusion.onnx
├── hard_routing_meta.json
├── soft_moe.onnx
├── task3_metadata.json
├── face2sketch_generator.onnx
├── task4_metadata.json
└── samples/            (clean Oxford-IIIT Pet test images, used by the sample picker)
```

To stop the application, press `Ctrl+C`, then run `docker compose down`.

## Architecture

```
browser ──> frontend (nginx, port 8080) ──/api──> backend (FastAPI + ONNX Runtime, port 8000)
                                                       └── ./models (ONNX files, read-only)
```

The backend reuses the Task 1 corruption code (`backend/app/corruptions.py`) without changes. Corruptions applied in the app are therefore identical to those used in training and in the test manifest:
- **Low, Medium and High** are the fixed test levels.
- **Random** samples from the training ranges.

Preprocessing matches each task's training pipeline:
- **Tasks 1, 2 and 3:** PIL bicubic resize to 128 × 128.
- **Task 4:** OpenCV `INTER_AREA` resize to 128 × 128.

The PSNR and SSIM shown in the app use the same definitions as the training metrics (SSIM with an 11 × 11 Gaussian window and σ = 1.5).

## API

| Method and path | Purpose |
|---|---|
| `GET /api/health` | Server status, ONNX Runtime version, status of each model, number of samples |
| `GET /api/models` | ONNX inputs and outputs, file size, parameters, configuration and load time per model |
| `GET /api/samples`, `GET /api/samples/{name}` | Clean sample images |
| `POST /api/universal/restore` | Task 1. Form fields: `file` or `sample`, `corruption` (`none`, `clean`, `salt_pepper`, `gaussian_blur`, `occlusion`), `severity` (`low`, `medium`, `high`, `random`), optional `seed`. Returns input, restored image, clean target and error map (when a reference exists), PSNR and SSIM, corruption settings and timings. |
| `POST /api/hard-routing/restore` | Task 2. Same image and corruption fields as Task 1, plus `routing` (`predicted`: the classifier's argmax picks the branch; `oracle`: the known corruption picks it). Returns the four class probabilities, the predicted class, the branch used (the identity bypass for clean predictions), whether the classifier was correct, the restored image, quality scores and separate classifier and specialist timings. |
| `POST /api/soft-moe/restore` | Task 3. Same image and corruption fields as Task 1. Returns the four routing weights, the strongest branch, every branch with at least 10% weight, the gate's own class, the routing entropy, the restored image, quality scores and timings. |
| `POST /api/sketch` | Task 4. Form fields: `file`, `style` (1, 2 or 3), optional `all_styles=true`. Returns the 128 px sketch, a 512 px upscaled copy and timings. |

Interactive API documentation is at **http://localhost:8080/api/docs**.

## Development without Docker

```bash
cd backend && pip install -r requirements.txt && MODELS_DIR=../models uvicorn app.main:app --reload --port 8000
cd frontend && npm install && npm run dev      # http://localhost:5173, proxies /api to port 8000
```

## Troubleshooting

| Symptom | What to do |
|---|---|
| A workspace says its model is "missing" | The ONNX file is not in `./models`. Check the `docker compose` log of the `models` service, and check that the Drive folder is shared as "Anyone with the link". |
| The sidebar says the API is not responding | The backend container is still starting or has stopped. Run `docker compose ps` and `docker compose logs backend`. |
| The webcam does not open | The browser needs camera permission. The page must be opened on `localhost` or over `https`. |

## Running inside GitHub Codespaces

Some codespaces block Docker's internal network, so containers cannot reach each other. Start the app with `./codespaces-up.sh --build` instead. It runs every service on the codespace's own network. On a normal computer, use `docker compose up --build`.
