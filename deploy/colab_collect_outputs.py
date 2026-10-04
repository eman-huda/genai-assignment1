# Run as ONE cell in Google Colab (CPU is enough). It collects the small result files from
# MyDrive/genai-assignment1 into MyDrive/genai-assignment1/repo_outputs.zip for the GitHub repository:
# Optuna studies, result tables, figures, ONNX metadata, data splits and corruption manifests.
# Model weights, ONNX models, datasets and caches are skipped (they are shared through the models link).
from google.colab import drive
drive.mount("/content/drive")
import os, zipfile

P = "/content/drive/MyDrive/genai-assignment1"
SKIP_EXT = (".pt", ".pth", ".onnx", ".npz", ".zip", ".tar", ".gz")
SKIP_DIRS = {"checkpoints", "samples", "__pycache__", ".ipynb_checkpoints"}
MAX_MB = 50
out = f"{P}/repo_outputs.zip"
added, skipped = 0, []
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for task in ("task1", "task2", "task3", "task4"):
        for root, dirs, files in os.walk(f"{P}/{task}"):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for f in files:
                path = os.path.join(root, f)
                if f.endswith(SKIP_EXT):
                    continue
                if os.path.getsize(path) > MAX_MB * 1e6:
                    skipped.append(path); continue
                z.write(path, os.path.join("outputs", os.path.relpath(path, P))); added += 1
    for rel in ("data/split_seed42.json", "data/fs2k_split_seed42.json", "data/fs2k_128_report.json",
                "manifests/val_manifest.json", "manifests/test_manifest.json"):
        if os.path.exists(f"{P}/{rel}"):
            z.write(f"{P}/{rel}", rel); added += 1
print(f"{added} files -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
if skipped:
    print("Skipped because larger than 50 MB:", skipped)
