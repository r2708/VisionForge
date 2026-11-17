"""Process 12: perceptual hashes (pHash, dHash, aHash), 64 bits each, stored as 16 hex characters."""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.fft import dct

from app.core.imageio import open_oriented, to_rgb
from app.pipeline.base import ImageStage, ImgRef, StageContext


def _bits_to_hex(bits: np.ndarray) -> str:
    return f"{int(''.join('1' if b else '0' for b in bits.flatten()), 2):016x}"


def phash(img: Image.Image) -> str:
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    low = dct(dct(g, axis=0, norm="ortho"), axis=1, norm="ortho")[:8, :8]
    return _bits_to_hex(low > np.median(low))


def dhash(img: Image.Image) -> str:
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return _bits_to_hex(g[:, 1:] > g[:, :-1])


def ahash(img: Image.Image) -> str:
    g = np.asarray(img.convert("L").resize((8, 8), Image.LANCZOS), dtype=np.float64)
    return _bits_to_hex(g > g.mean())


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


class PerceptualHashStage(ImageStage):
    name = "phash"
    version = "1"
    title = "Perceptual hashes"
    processes = "12"
    label = "PHASH"

    def describe(self, d):
        return f"pHash {d['phash']}"

    def enabled(self, ctx):
        on = ctx.cfg.stages.perceptual_hash
        return (on is not False), "disabled in configuration"

    def process(self, ctx: StageContext, ref: ImgRef) -> dict:
        img = to_rgb(open_oriented(ref.abs_path))
        img.thumbnail((256, 256), Image.LANCZOS)
        return {"phash": phash(img), "dhash": dhash(img), "ahash": ahash(img)}
