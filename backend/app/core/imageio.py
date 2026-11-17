"""Read-only image loading helpers shared by the stages. Source files are only ever opened for reading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Guard against decompression bombs while still allowing very large photographs.
Image.MAX_IMAGE_PIXELS = 300_000_000

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

BIT_DEPTH = {"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "CMYK": 8, "YCbCr": 8, "LAB": 8, "HSV": 8,
             "LA": 8, "PA": 8, "I": 32, "F": 32, "I;16": 16, "I;16L": 16, "I;16B": 16, "I;16N": 16}


def has_alpha_channel(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def open_oriented(path: str | Path) -> Image.Image:
    """Open, fully decode and apply EXIF orientation (in memory only)."""
    with Image.open(path) as im:
        im.load()
        out = ImageOps.exif_transpose(im)
        if out is im:
            out = im.copy()
    return out


def to_rgb(img: Image.Image, background=(255, 255, 255)) -> Image.Image:
    """RGB view for analysis. Transparency is composited on ``background`` (analysis only)."""
    if img.mode == "RGB":
        return img
    if has_alpha_channel(img):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, tuple(background))
        bg.paste(rgba, mask=rgba.getchannel("A"))
        return bg
    if img.mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        scale = 255.0 / (hi - lo) if hi > lo else 0.0
        return Image.fromarray(((arr - lo) * scale).astype(np.uint8), "L").convert("RGB")
    return img.convert("RGB")


def analysis_rgb(path: str | Path, max_side: int = 1024) -> Image.Image:
    """Oriented RGB image downscaled so metrics are comparable across resolutions."""
    img = to_rgb(open_oriented(path))
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        s = max_side / m
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return img


def gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)
