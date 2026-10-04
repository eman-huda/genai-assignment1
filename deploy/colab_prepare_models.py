# Run this as ONE cell in Google Colab (any runtime type, no GPU needed).
# It gathers the trained models (Tasks 1 to 4) and 8 clean sample images into
# MyDrive/genai-assignment1/app_models, which the app downloads at start-up.
from google.colab import drive
drive.mount("/content/drive")

import os, shutil
import numpy as np
from PIL import Image

P = "/content/drive/MyDrive/genai-assignment1"
OUT = f"{P}/app_models"
os.makedirs(f"{OUT}/samples", exist_ok=True)
# Safe to re-run: existing files are overwritten, samples are kept.

for src in ["task1/onnx/task1_universal_dae.onnx", "task1/onnx/task1_metadata.json",
            "task2/onnx/corruption_classifier.onnx", "task2/onnx/specialist_salt_pepper.onnx",
            "task2/onnx/specialist_gaussian_blur.onnx", "task2/onnx/specialist_occlusion.onnx",
            "task2/onnx/hard_routing_meta.json",
            "task3/onnx/soft_moe.onnx", "task3/onnx/task3_metadata.json",
            "task4/onnx/face2sketch_generator.onnx", "task4/onnx/task4_metadata.json"]:
    shutil.copy2(f"{P}/{src}", OUT)
    print("copied", src)

# 8 clean images from the official Oxford-IIIT Pet TEST set (already RGB, 128 x 128),
# evenly spaced so that both cats and dogs appear.
d = np.load(f"{P}/data/pets_128.npz")
imgs, names = d["test_images"], d["test_names"]
for i in np.linspace(0, len(imgs) - 1, 8).astype(int):
    Image.fromarray(imgs[i]).save(f"{OUT}/samples/{names[i]}.png")

print("\napp_models now contains:")
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        p = os.path.join(root, f)
        print(f"  {os.path.relpath(p, OUT):45s} {os.path.getsize(p) / 1e6:6.1f} MB")
