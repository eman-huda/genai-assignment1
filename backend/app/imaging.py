"""Image decoding, preprocessing, encoding and quality metrics for the API."""
import base64
import io

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

IMG_SIZE = 128
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}


class ImageError(ValueError):
    """Raised for uploads that are not usable images; mapped to HTTP 400."""


def decode_upload(data: bytes) -> Image.Image:
    if not data:
        raise ImageError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ImageError("The image is larger than 10 MB. Upload a smaller file.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ImageError("The file is not a readable image. Use PNG, JPEG, WEBP or BMP.")
    if img.format not in ALLOWED_FORMATS:
        raise ImageError(f"{img.format} images are not supported. Use PNG, JPEG, WEBP or BMP.")
    img = ImageOps.exif_transpose(img)                     # respect phone/webcam orientation
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def task1_preprocess(img: Image.Image) -> np.ndarray:
    """Same as Task 1 training: RGB, 128x128 bicubic (PIL), float32 HWC in [0, 1]."""
    return np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC), dtype=np.float32) / 255.0


def task4_preprocess(img: Image.Image) -> np.ndarray:
    """Same as Task 4 training: RGB, 128x128 with cv2.INTER_AREA, float32 HWC in [0, 1]."""
    arr = np.asarray(img, dtype=np.uint8)
    return cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def to_nchw(hwc: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(hwc.transpose(2, 0, 1)[None], dtype=np.float32)


def from_nchw(x: np.ndarray) -> np.ndarray:
    return np.clip(x[0].transpose(1, 2, 0), 0.0, 1.0)


def to_uint8(hwc01: np.ndarray) -> np.ndarray:
    return (np.clip(hwc01, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def png_data_url(img: np.ndarray, upscale: int = 1) -> str:
    """HWC float [0,1] or uint8 RGB -> PNG data URL. upscale > 1 uses bicubic resizing."""
    arr = to_uint8(img) if img.dtype != np.uint8 else img
    if upscale > 1:
        arr = cv2.resize(arr, (arr.shape[1] * upscale, arr.shape[0] * upscale), interpolation=cv2.INTER_CUBIC)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 100.0 if mse < 1e-10 else float(10 * np.log10(1.0 / mse))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM with an 11x11 Gaussian window (sigma 1.5), data range 1, averaged over RGB.
    Same definition as pytorch_msssim used in training; borders are excluded ('valid')."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    vals = []
    for ch in range(3):
        x, y = a[..., ch].astype(np.float64), b[..., ch].astype(np.float64)
        blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)[5:-5, 5:-5]
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx ** 2, blur(y * y) - my ** 2, blur(x * y) - mx * my
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        vals.append(m.mean())
    return float(np.mean(vals))


def error_map(output: np.ndarray, reference: np.ndarray, vmax: float = 0.5) -> np.ndarray:
    """Mean absolute error over RGB, scaled to [0, vmax] and coloured (dark = small, bright = large)."""
    err = np.abs(output - reference).mean(axis=2)
    norm = to_uint8(np.clip(err / vmax, 0, 1))
    return cv2.cvtColor(cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA), cv2.COLOR_BGR2RGB)
