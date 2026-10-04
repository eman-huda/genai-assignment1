"""Download the trained ONNX models before the backend starts (run by Docker Compose).

If the files are already in ./models nothing is downloaded. Otherwise the shared Google Drive
folder in MODELS_FOLDER_URL (set in the .env file) is downloaded with gdown.
A failed download is reported but does not stop the app: missing models show as unavailable.
"""
import os
import subprocess
import sys
from pathlib import Path

MODELS = Path("/models")
REQUIRED = ["task1_universal_dae.onnx", "task1_metadata.json",
            "corruption_classifier.onnx", "specialist_salt_pepper.onnx", "specialist_gaussian_blur.onnx",
            "specialist_occlusion.onnx", "hard_routing_meta.json",
            "soft_moe.onnx", "task3_metadata.json",
            "face2sketch_generator.onnx", "task4_metadata.json"]


def missing():
    return [f for f in REQUIRED if not any(MODELS.rglob(f))]


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    todo = missing()
    if not todo:
        print("All model files are present. Nothing to download.")
        return
    url = os.environ.get("MODELS_FOLDER_URL", "").strip()
    if not url:
        print(f"Missing model files: {todo}. Set MODELS_FOLDER_URL in .env, or copy the files into ./models.")
        return
    print(f"Missing {todo}. Downloading the shared models folder from Google Drive...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", "gdown==5.2.0"], check=True)
    except subprocess.CalledProcessError:
        print("Could not install gdown (no internet inside Docker?). Copy the model files into ./models manually.")
        return
    import gdown
    try:
        gdown.download_folder(url, output=str(MODELS), quiet=False, use_cookies=False)
    except Exception as exc:
        print(f"Download failed: {exc}. Check that the folder is shared as 'Anyone with the link'.")
    still = missing()
    print("Done. All model files present." if not still else f"Still missing after download: {still}")


if __name__ == "__main__":
    main()
