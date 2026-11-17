"""Synthetic fixture dataset. Everything is generated procedurally (no third-party or private
images), deterministically from a seed."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from PIL.TiffImagePlugin import IFDRational


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def landscape(seed: int, size=(1024, 768)) -> Image.Image:
    """Sky gradient, sun, layered ridgelines and fine texture."""
    rng = np.random.default_rng(seed)
    w, h = size
    y = np.linspace(0, 1, h)[:, None]
    top, bottom = rng.integers(40, 120, 3), rng.integers(170, 250, 3)
    sky = (top[None, None, :] * (1 - y[..., None]) + bottom[None, None, :] * y[..., None]) * np.ones((1, w, 1))
    img = Image.fromarray(sky.astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    sx, sy, sr = int(rng.integers(w // 5, 4 * w // 5)), int(rng.integers(h // 8, h // 3)), int(rng.integers(30, 60))
    d.ellipse((sx - sr, sy - sr, sx + sr, sy + sr), fill=(255, 240, 200))
    for layer in range(4):
        base = h * (0.45 + 0.12 * layer)
        xs = np.linspace(0, w, 40)
        ys = base + np.cumsum(rng.normal(0, 14, 40)) - 20 * np.sin(xs / w * np.pi * (layer + 1))
        col = tuple(int(c) for c in rng.integers(20, 90, 3) + 25 * layer)
        d.polygon([(0, h)] + list(zip(xs.tolist(), ys.tolist())) + [(w, h)], fill=col)
    for _ in range(140):  # trees / rocks: gives corners for feature matching
        x, yy = int(rng.integers(0, w)), int(rng.integers(int(h * 0.55), h))
        s = int(rng.integers(4, 16))
        d.polygon([(x, yy - 2 * s), (x - s, yy), (x + s, yy)], fill=tuple(int(c) for c in rng.integers(0, 110, 3)))
    arr = np.asarray(img).astype(np.int16) + rng.normal(0, 3, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")


def shapes(seed: int, size=(900, 900)) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    img = Image.new("RGB", size, tuple(int(c) for c in rng.integers(200, 255, 3)))
    d = ImageDraw.Draw(img)
    for _ in range(45):
        x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(20, 110))
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            d.ellipse((x - r, y - r, x + r, y + r), fill=col, outline=(0, 0, 0), width=3)
        else:
            d.rectangle((x - r, y - r // 2, x + r, y + r // 2), fill=col, outline=(0, 0, 0), width=3)
    return img


def texture(seed: int, size=(1200, 800)) -> Image.Image:
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    f1, f2 = rng.uniform(0.01, 0.05, 2)
    chans = [127 + 110 * np.sin(xx * f1 * (i + 1) + rng.uniform(0, 6)) * np.cos(yy * f2 * (i + 1) + rng.uniform(0, 6)) for i in range(3)]
    tile = ((xx // 60 + yy // 60) % 2) * 30
    return Image.fromarray(np.clip(np.stack(chans, -1) + tile[..., None], 0, 255).astype(np.uint8), "RGB")


def build_fixture_dataset(root: Path, seed: int = 7) -> dict:
    """Creates ``root`` (must not exist or be empty) and returns a description of what is in it."""
    root = Path(root)
    (root / "nested" / "deeper").mkdir(parents=True, exist_ok=True)
    a, b, c = landscape(seed), shapes(seed + 1), texture(seed + 2)

    a.save(root / "scene_a.jpg", quality=92)
    shutil.copyfile(root / "scene_a.jpg", root / "nested" / "scene_a_copy.jpg")  # exact duplicate
    a.resize((800, 600), Image.LANCZOS).save(root / "nested" / "scene_a_resized.jpg", quality=80)  # perceptual duplicate
    crop = a.crop((90, 70, 960, 730)).copy()  # near duplicate: crop + overlay
    ImageDraw.Draw(crop).text((30, 30), "SAMPLE", fill=(255, 255, 255), font=_font(48))
    crop.save(root / "nested" / "deeper" / "scene_a_cropped.png")

    b.save(root / "scene_b.png")
    (root / "scene_b.txt").write_text("Overlapping colored circles and rectangles on a pale background.\n")
    c.save(root / "scene_c.webp", quality=92)
    landscape(seed + 10).save(root / "scene_d.jpg", quality=93)
    landscape(seed + 11, (768, 1024)).save(root / "nested" / "scene_e_portrait.jpg", quality=93)
    shapes(seed + 12, (1100, 800)).save(root / "nested" / "scene_f.png")
    texture(seed + 13, (1000, 1000)).save(root / "scene_g.jpg", quality=94)
    texture(seed + 14).save(root / "tex.bmp")
    shapes(seed + 15, (800, 640)).save(root / "nested" / "shapes.tif")

    # EXIF orientation 6 (needs a 90-degree clockwise rotation to display correctly)
    exif = Image.Exif()
    exif[0x0112] = 6
    exif[0x010F], exif[0x0110] = "FixtureCam", "Model X"
    landscape(seed + 20, (800, 600)).save(root / "rotated.jpg", quality=92, exif=exif)

    # EXIF with GPS data (must not reach the processed dataset)
    gps = Image.Exif()
    gps[0x010F] = "FixtureCam"
    r = IFDRational
    gps[0x8825] = {1: "N", 2: (r(10), r(20), r(30)), 3: "E", 4: (r(40), r(50), r(6))}
    landscape(seed + 21, (900, 700)).save(root / "with_gps.jpg", quality=92, exif=gps)

    # Transparency actually in use
    rgba = shapes(seed + 30, (700, 700)).convert("RGBA")
    alpha = Image.new("L", rgba.size, 0)
    ImageDraw.Draw(alpha).ellipse((60, 60, 640, 640), fill=255)
    rgba.putalpha(alpha)
    rgba.save(root / "alpha.png")

    landscape(seed + 40, (800, 800)).convert("CMYK").save(root / "cmyk.jpg", quality=92)
    g16 = (np.asarray(texture(seed + 41, (640, 640)).convert("L"), dtype=np.uint16) * 257)
    Image.fromarray(g16).save(root / "gray16.png")

    landscape(seed + 50, (200, 150)).save(root / "small.jpg", quality=90)  # below minimum size
    landscape(seed + 51).filter(ImageFilter.GaussianBlur(14)).save(root / "blurry.jpg", quality=90)
    dark = (np.asarray(landscape(seed + 52), dtype=np.float32) * 0.06).astype(np.uint8)
    Image.fromarray(dark).save(root / "dark.jpg", quality=90)

    txt = Image.new("RGB", (1000, 700), (250, 250, 245))
    dt = ImageDraw.Draw(txt)
    dt.text((80, 120), "HELLO WORLD", fill=(10, 10, 10), font=_font(110))
    dt.text((80, 330), "Dataset 2024", fill=(30, 30, 120), font=_font(90))
    txt.save(root / "text_image.png")

    # Broken and unsupported files
    (root / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0" + bytes(range(256)) * 20)
    whole = (root / "scene_b.png").read_bytes()
    (root / "truncated.png").write_bytes(whole[: len(whole) // 2])
    jpg = (root / "scene_d.jpg").read_bytes()
    (root / "nested" / "truncated.jpg").write_bytes(jpg[: len(jpg) // 3])
    (root / "empty.png").write_bytes(b"")
    (root / "notes.txt").write_text("not an image\n")
    (root / "archive.zip").write_bytes(b"PK\x03\x04fake")
    (root / ".hidden.jpg").write_bytes(b"hidden")

    (root / "metadata.csv").write_text(
        "relative_path,license,rights_status,creator,source_url,caption\n"
        "scene_d.jpg,CC0-1.0,verified,Fixture Generator,https://example.org/scene_d,A generated mountain landscape under a gradient sky.\n"
        "scene_c.webp,,restricted,,,\n")
    return {
        "supported_files": 25, "valid": 21, "invalid": 4, "unsupported": 4,  # .txt x2, .zip, .csv
        "invalid_names": {"corrupt.jpg", "truncated.png", "truncated.jpg", "empty.png"},
        "exact": {"scene_a.jpg", "nested/scene_a_copy.jpg"},
        "perceptual": {"scene_a.jpg", "nested/scene_a_copy.jpg", "nested/scene_a_resized.jpg"},
        "near": "nested/deeper/scene_a_cropped.png",
    }
