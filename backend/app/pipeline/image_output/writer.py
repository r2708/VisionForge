"""Stage 8: processed image creation and storage.

Colour-management policy (process 7): images tagged with a non-sRGB ICC profile are converted
to sRGB; untagged images are assumed to be sRGB and left as they are. Alpha policy (process 8):
transparency that is actually used is preserved (PNG) unless ``preserve_alpha`` is false, in
which case it is flattened onto ``alpha_background``. Orientation (process 6): EXIF orientation
is baked into the rendition. Sources are opened read-only; when no transformation is needed the
source bytes are copied as-is so JPEGs are not recompressed."""
from __future__ import annotations

import io
import os
import re
import shutil
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageCms
from sqlalchemy import func, select

from app.core.imageio import has_alpha_channel, open_oriented, to_rgb
from app.core.io import free_disk_gb, sha256_file
from app.core.logging import get_logger
from app.database.models import Image
from app.pipeline.base import ImageStage, ImgRef, StageContext

log = get_logger("output")
EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
OWN_RENDITION = re.compile(r"^img_[0-9a-f]{16,32}\.(jpg|png|webp)$")
OWN_TEMP = re.compile(r"^\.img_[0-9a-f]{16,32}\.\d+\.tmp$")
DIRECT_FORMATS = set(EXT)


def strip_jpeg_exif(data: bytes) -> bytes:
    """Remove APP1/Exif segments from a JPEG without touching the compressed image data."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG stream")
    out, i = bytearray(b"\xff\xd8"), 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            raise ValueError("malformed JPEG marker")
        marker = data[i + 1]
        if marker == 0xDA or 0xD0 <= marker <= 0xD9:  # start of scan / standalone markers: copy the rest
            break
        length = int.from_bytes(data[i + 2:i + 4], "big")
        segment = data[i:i + 2 + length]
        if not (marker == 0xE1 and segment[4:10] == b"Exif\x00\x00"):
            out += segment
        i += 2 + length
    out += data[i:]
    return bytes(out)


def plan_rendition(meta: dict, cfg) -> tuple[list[str], str]:
    """Decide which transformations are required and the output format."""
    t: list[str] = []
    fmt = meta["format"]
    if meta["exif_orientation"] != 1:
        t.append("exif_orientation_applied")
    icc = meta["icc"]
    if cfg.convert_to_srgb and icc["present"] and not icc["is_srgb"]:
        t.append("icc_converted_to_srgb")
    if meta["color_mode"] not in ("RGB", "RGBA", "L"):
        t.append(f"color_mode_{meta['color_mode']}_converted")
    if meta.get("bit_depth") and meta["bit_depth"] > 8:
        t.append("reduced_to_8_bit")
    keep_alpha = bool(meta["alpha_used"]) and cfg.preserve_alpha
    if meta["alpha_used"] and not cfg.preserve_alpha:
        t.append("alpha_flattened_on_background")
    if cfg.max_side and max(meta["width"], meta["height"]) > cfg.max_side:
        t.append(f"downscaled_to_max_side_{cfg.max_side}")
    if cfg.output_format == "auto":
        out_fmt = "PNG" if keep_alpha else (fmt if fmt in DIRECT_FORMATS else "PNG")
    else:
        out_fmt = cfg.output_format.upper()
        if keep_alpha and out_fmt == "JPEG":
            out_fmt = "PNG"  # JPEG cannot hold transparency; preserving alpha takes precedence
    if out_fmt != fmt:
        t.append(f"format_{fmt}_to_{out_fmt}")
    if meta.get("frames", 1) > 1:
        t.append("first_frame_only")
    return t, out_fmt


def render(path: str, meta: dict, cfg, out_fmt: str) -> PILImage.Image:
    img = open_oriented(path)
    icc_bytes = img.info.get("icc_profile")
    keep_alpha = bool(meta["alpha_used"]) and cfg.preserve_alpha
    if img.mode not in ("RGB", "RGBA", "L"):
        if has_alpha_channel(img):
            img = img.convert("RGBA")
        elif img.mode == "CMYK":
            img = _cmyk_to_rgb(img, icc_bytes)
            icc_bytes = None
        else:
            img = to_rgb(img)
    if cfg.convert_to_srgb and icc_bytes and not meta["icc"]["is_srgb"]:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
        if img.mode == "L" and src.profile.xcolor_space.strip() == "RGB":
            img = img.convert("RGB")  # a grayscale file tagged with an RGB profile: the profile only applies to RGB pixels
        # sRGB is an RGB space, so a grayscale image with a grayscale profile comes out as RGB.
        img = ImageCms.profileToProfile(img, src, ImageCms.createProfile("sRGB"), outputMode="RGB" if img.mode == "L" else img.mode)
    if img.mode == "RGBA" and not keep_alpha:
        img = to_rgb(img, cfg.alpha_background) if meta["alpha_used"] else img.convert("RGB")
    if out_fmt == "JPEG" and img.mode == "RGBA":
        img = to_rgb(img, cfg.alpha_background)
    if cfg.max_side and max(img.size) > cfg.max_side:
        s = cfg.max_side / max(img.size)
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), PILImage.LANCZOS)
    return img


def _cmyk_to_rgb(img, icc_bytes):
    if icc_bytes:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
            return ImageCms.profileToProfile(img, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
        except Exception:
            pass
    return img.convert("RGB")


class ImageOutputStage(ImageStage):
    name = "image_output"
    label = "OUTPUT"

    def describe(self, d):
        return f"{d['path']} {d['width']}x{d['height']} ({d['bytes'] / 1024:.0f} KB, {', '.join(d['transformations'])})"

    version = "1"
    title = "Processed image renditions"
    processes = "Stage 8"
    depends = ("validate",)
    soft_depends = ("quality",)

    def fingerprint(self, cfg):
        i = cfg.images
        return [i.output_format, i.jpeg_quality, i.webp_quality, i.preserve_alpha, list(i.alpha_background),
                i.convert_to_srgb, i.copy_unchanged_when_possible, i.strip_location_metadata, i.max_side]

    def eligible(self, ctx, img: Image) -> bool:
        return img.decision == "accepted" and bool(img.rights_eligible)

    def setup(self, ctx):
        ctx.ws.images_dir.mkdir(parents=True, exist_ok=True)
        preflight(ctx)

    def process(self, ctx: StageContext, ref: ImgRef) -> dict:
        cfg, meta = ctx.cfg.images, ref.deps["validate"]
        images_dir = ctx.ws.images_dir
        transformations, out_fmt = plan_rendition(meta, cfg)
        strip_gps = cfg.strip_location_metadata and meta.get("exif_has_gps")
        final = images_dir / f"{ref.image_id}.{EXT[out_fmt]}"
        tmp = images_dir / f".{ref.image_id}.{os.getpid()}.tmp"
        try:
            if not transformations and cfg.copy_unchanged_when_possible and not (strip_gps and out_fmt != "JPEG"):
                if strip_gps:
                    tmp.write_bytes(strip_jpeg_exif(Path(ref.abs_path).read_bytes()))
                    transformations = ["exif_removed_losslessly"]
                else:
                    shutil.copyfile(ref.abs_path, tmp)
                    transformations = ["copied_unchanged"]
            else:
                if strip_gps and not transformations:
                    transformations = ["reencoded_to_remove_location_metadata"]
                img = render(ref.abs_path, meta, cfg, out_fmt)
                if out_fmt == "JPEG":
                    img.save(tmp, "JPEG", quality=cfg.jpeg_quality, subsampling=0, optimize=True)
                elif out_fmt == "PNG":
                    img.save(tmp, "PNG", optimize=False)
                else:
                    img.save(tmp, "WEBP", quality=cfg.webp_quality, method=4)
            with PILImage.open(tmp) as check:  # every output must decode before it is committed
                check.load()
                w, h, fmt = check.width, check.height, check.format
            if fmt != out_fmt:
                raise ValueError(f"written file decoded as {fmt}, expected {out_fmt}")
            digest, size = sha256_file(tmp), tmp.stat().st_size
            for stale in images_dir.glob(f"{ref.image_id}.*"):
                if stale != final and stale.suffix != ".txt":
                    stale.unlink()
            os.replace(tmp, final)
        finally:
            if tmp.exists():
                tmp.unlink()
        return {"path": f"images/{final.name}", "sha256": digest, "bytes": size, "format": out_fmt, "width": w, "height": h,
                "aspect_ratio": round(w / h, 4), "is_upscaled": False, "transformations": transformations,
                "bucket": (ref.deps.get("quality") or {}).get("bucket"), "source_sha256": ref.sha256}

    def still_valid(self, ctx, img, data):
        p = ctx.ws.output_dir / data.get("path", "")
        return p.is_file() and p.stat().st_size == data.get("bytes")

    def apply(self, img, data):
        img.processed_path, img.processed_sha256 = data["path"], data["sha256"]


def preflight(ctx: StageContext) -> dict:
    """Disk-space check before any rendition is written."""
    st = ctx.cfg.storage
    with ctx.ws.db.session() as s:
        need = s.scalar(select(func.coalesce(func.sum(Image.file_size), 0)).where(
            Image.decision == "accepted", Image.rights_eligible.is_(True), Image.missing.is_(False),
            Image.processed_path.is_(None))) or 0
        have = s.scalar(select(func.coalesce(func.sum(Image.file_size), 0)).where(Image.processed_path.is_not(None))) or 0
    est_gb = need * 1.15 / 1024**3
    free = free_disk_gb(ctx.ws.output_dir)
    info = {"estimated_new_output_gb": round(est_gb, 3), "free_disk_gb": round(free, 2), "min_free_disk_gb": st.min_free_disk_gb}
    if free - est_gb < st.min_free_disk_gb:
        raise OSError(f"not enough disk space: ~{est_gb:.2f} GB needed, {free:.2f} GB free, "
                      f"{st.min_free_disk_gb} GB must remain free")
    if st.max_output_gb is not None and (have / 1024**3 + est_gb) > st.max_output_gb:
        raise OSError(f"processed image layer would exceed storage.max_output_gb ({st.max_output_gb} GB)")
    return info


def reconcile_outputs(ctx: StageContext) -> dict:
    """Keep ``images/`` equal to the set of export-eligible renditions. Renditions of images that
    are no longer accepted are moved to ``rejected/quarantine`` (derivatives only - sources are
    never touched), which prevents orphaned files and dangling manifest paths."""
    out, images_dir = ctx.ws.output_dir, ctx.ws.images_dir
    qdir = out / "rejected" / "quarantine"
    moved, missing = 0, 0
    expected: set[str] = set()
    last = 0
    while True:
        with ctx.ws.db.session() as s:
            imgs = list(s.scalars(select(Image).where(Image.id > last, Image.processed_path.is_not(None)).order_by(Image.id).limit(500)))
            if not imgs:
                break
            last = imgs[-1].id
            for im in imgs:
                p = out / im.processed_path
                ok = im.decision == "accepted" and bool(im.rights_eligible) and not im.missing
                if ok and p.is_file():
                    expected.add(p.name)
                    continue
                if ok:
                    missing += 1
                elif p.is_file():
                    qdir.mkdir(parents=True, exist_ok=True)
                    os.replace(p, qdir / p.name)
                    moved += 1
                im.processed_path = im.processed_sha256 = None
                im.split = None
    # Only files this pipeline created are ever touched; anything else in the folder is left alone.
    orphans, foreign = 0, 0
    if images_dir.is_dir():
        for f in images_dir.iterdir():
            if not f.is_file() or f.name in expected or f.suffix in (".txt", ".jsonl"):
                continue
            if OWN_TEMP.match(f.name):
                f.unlink()  # leftover temporary file from an interrupted write
            elif OWN_RENDITION.match(f.name):
                (qdir / "orphaned").mkdir(parents=True, exist_ok=True)
                os.replace(f, qdir / "orphaned" / f.name)
                orphans += 1
            else:
                foreign += 1
    return {"renditions_in_dataset": len(expected), "moved_to_quarantine": moved, "orphans_moved": orphans,
            "missing_renditions": missing, "unrecognised_files_left_untouched": foreign}
