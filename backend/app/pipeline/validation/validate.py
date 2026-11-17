"""Processes 2, 5, 6, 7, 8, 9: file validation, metadata, orientation, colour profile, alpha, geometry."""
from __future__ import annotations

import io
from fractions import Fraction

import numpy as np
from PIL import ExifTags, Image, ImageCms, UnidentifiedImageError

from app.core.imageio import BIT_DEPTH, has_alpha_channel
from app.pipeline.base import ImageStage, ImgRef, StageContext

_EXIF_KEEP = {"Make": "camera_make", "Model": "camera_model", "LensModel": "lens_model", "FocalLength": "focal_length_mm",
              "FNumber": "f_number", "ExposureTime": "exposure_time_s", "ISOSpeedRatings": "iso",
              "PhotographicSensitivity": "iso", "DateTimeOriginal": "datetime_original", "Software": "software"}
_ORIENT_TRANSFORMS = {1: [], 2: ["flip_horizontal"], 3: ["rotate_180"], 4: ["flip_vertical"], 5: ["transpose"],
                      6: ["rotate_270"], 7: ["transverse"], 8: ["rotate_90"]}


def _num(v):
    try:
        if isinstance(v, tuple) and len(v) == 2:
            return float(Fraction(int(v[0]), int(v[1])))
        return float(v)
    except Exception:
        return None


def read_exif(im: Image.Image) -> tuple[dict, int, bool]:
    """Selected capture facts straight from EXIF. GPS coordinates are deliberately not stored;
    only their presence is recorded."""
    out: dict = {}
    orientation, has_gps = 1, False
    try:
        exif = im.getexif()
    except Exception:
        return out, orientation, has_gps
    if not exif:
        return out, orientation, has_gps
    orientation = int(exif.get(0x0112, 1) or 1)
    if orientation not in _ORIENT_TRANSFORMS:
        orientation = 1
    try:
        has_gps = bool(exif.get_ifd(ExifTags.IFD.GPSInfo))
    except Exception:
        has_gps = False
    merged = dict(exif)
    try:
        merged.update(exif.get_ifd(ExifTags.IFD.Exif))
    except Exception:
        pass
    for tag, value in merged.items():
        name = ExifTags.TAGS.get(tag)
        key = _EXIF_KEEP.get(name)
        if not key or value in (None, "", b""):
            continue
        if key in ("focal_length_mm", "f_number", "exposure_time_s", "iso"):
            if isinstance(value, (tuple, list)) and key == "iso":
                value = value[0]
            value = _num(value)
            if value is None:
                continue
        else:
            value = str(value).strip().strip("\x00")[:120]
            if not value:
                continue
        out[key] = value
    return out, orientation, has_gps


def icc_description(im: Image.Image) -> tuple[bool, str | None]:
    icc = im.info.get("icc_profile")
    if not icc:
        return False, None
    try:
        prof = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        return True, (ImageCms.getProfileDescription(prof) or "").strip() or "unnamed profile"
    except Exception:
        return True, "unreadable profile"


def is_srgb_description(desc: str | None) -> bool:
    return bool(desc) and "srgb" in desc.lower()


class ValidateStage(ImageStage):
    name = "validate"
    version = "1"
    title = "File validation and metadata"
    processes = "2, 5, 6, 7, 8, 9"
    require_valid = False
    label = "VALIDATE"

    def describe(self, d):
        if d["status"] != "valid":
            return f"INVALID ({d['status']}): {d.get('reason') or ''}"
        return f"{d['width']}x{d['height']} {d['format']} {d['color_mode']}"

    def process(self, ctx: StageContext, ref: ImgRef) -> dict:
        def invalid(status: str, reason: str) -> dict:
            return {"status": status, "reason": reason[:300]}

        try:
            with Image.open(ref.abs_path) as probe:
                probe.verify()  # structural check; the handle is unusable afterwards
        except FileNotFoundError:
            raise
        except UnidentifiedImageError:
            return invalid("unreadable", "not a decodable image (unknown or damaged header)")
        except Image.DecompressionBombError:
            return invalid("too_large", "pixel count exceeds the decompression-bomb safety limit")
        except Exception as exc:
            if "truncat" in str(exc).lower():
                return invalid("truncated", "image data ends prematurely")
            return invalid("corrupt", f"structural verification failed: {type(exc).__name__}")

        try:
            with Image.open(ref.abs_path) as im:
                fmt, mode, raw_size = im.format, im.mode, im.size
                n_frames = getattr(im, "n_frames", 1)
                exif, orientation, has_gps = read_exif(im)
                has_icc, icc_desc = icc_description(im)
                alpha = has_alpha_channel(im)
                im.load()  # full decode: catches truncated data
                alpha_used = False
                if alpha:
                    a = np.asarray(im.convert("RGBA").getchannel("A"))
                    alpha_used = bool((a < 255).any())
        except Image.DecompressionBombError:
            return invalid("too_large", "pixel count exceeds the decompression-bomb safety limit")
        except OSError as exc:
            msg = str(exc).lower()
            if "truncated" in msg or "broken data stream" in msg or "unexpected end" in msg:
                return invalid("truncated", "image data ends prematurely")
            return invalid("corrupt", f"decode failed: {exc}")
        except Exception as exc:
            return invalid("corrupt", f"decode failed: {type(exc).__name__}")

        w, h = raw_size
        if orientation in (5, 6, 7, 8):
            w, h = h, w
        if w < 1 or h < 1:
            return invalid("corrupt", "zero-sized image")
        ar = w / h
        warnings = []
        ext_fmt = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".bmp": "BMP", ".tif": "TIFF", ".tiff": "TIFF"}
        if ext_fmt.get(ref.extension) and fmt and ext_fmt[ref.extension] != fmt:
            warnings.append(f"extension {ref.extension} does not match decoded format {fmt}")
        if n_frames and n_frames > 1:
            warnings.append(f"multi-frame file ({n_frames} frames); only the first frame is used")
        return {
            "status": "valid", "reason": None, "warnings": warnings,
            "format": fmt, "color_mode": mode, "bit_depth": BIT_DEPTH.get(mode),
            "raw_width": raw_size[0], "raw_height": raw_size[1], "width": w, "height": h,
            "megapixels": round(w * h / 1e6, 3), "aspect_ratio": round(ar, 4),
            "orientation": "square" if 0.97 <= ar <= 1.03 else ("landscape" if ar > 1 else "portrait"),
            "exif_orientation": orientation, "orientation_transforms": _ORIENT_TRANSFORMS[orientation],
            "exif": exif, "exif_has_gps": has_gps,
            "icc": {"present": has_icc, "description": icc_desc, "is_srgb": is_srgb_description(icc_desc) if has_icc else None},
            "has_alpha": alpha, "alpha_used": alpha_used, "frames": n_frames,
        }

    def apply(self, img, data: dict) -> None:
        img.valid = data["status"] == "valid"
        img.validation_status = data["status"]
        img.validation_reason = data.get("reason")
        if img.valid:
            img.width, img.height = data["width"], data["height"]
            img.format, img.color_mode = data["format"], data["color_mode"]
            img.aspect_ratio, img.orientation = data["aspect_ratio"], data["orientation"]
            img.has_alpha = bool(data["alpha_used"])
