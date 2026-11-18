"""Processes 16 and 28: resolution suitability and explainable quality aggregation.

Cheap by design: it only combines cached component metrics, so changing weights, minimum
dimensions or buckets re-runs this stage without touching pixels or models."""
from __future__ import annotations

import re

import math
from typing import Optional

from app.pipeline.base import ImageStage, ImgRef, StageContext


def aspect_bucket(width: int, height: int, bases: list[int], step: int) -> Optional[dict]:
    """Largest configured base resolution whose aspect-preserving bucket fits without upscaling."""
    ar = width / height
    best = None
    for base in sorted(bases):
        bw = max(step, int(round(math.sqrt(base * base * ar) / step)) * step)
        bh = max(step, int(round(math.sqrt(base * base / ar) / step)) * step)
        if width >= bw and height >= bh:
            best = {"base": base, "width": bw, "height": bh, "aspect_ratio": round(bw / bh, 4)}
    return best


class QualityAggregateStage(ImageStage):
    name = "quality"
    label = "QUALITY"

    def describe(self, d):
        flags = f", flags: {', '.join(d['flags'][:4])}" if d["flags"] else ""
        return f"overall {d['overall_score']}, resolution {d['resolution_status']}{flags}"

    version = "3"
    title = "Resolution suitability and overall quality"
    processes = "16, 28"
    depends = ("validate", "quality_technical")
    soft_depends = ("zero_shot", "ocr")

    def fingerprint(self, cfg):
        return [cfg.quality.weights.model_dump(), cfg.quality.min_width, cfg.quality.min_height,
                cfg.images.resolution_buckets, cfg.images.bucket_step]

    def process(self, ctx: StageContext, ref: ImgRef) -> dict:
        q, cfg = ref.deps["quality_technical"], ctx.cfg
        meta = ref.deps["validate"]
        w, h = meta["width"], meta["height"]
        zs, ocr = ref.deps.get("zero_shot"), ref.deps.get("ocr")

        bucket = aspect_bucket(w, h, cfg.images.resolution_buckets, cfg.images.bucket_step)
        meets_min = w >= cfg.quality.min_width and h >= cfg.quality.min_height
        res_status = "accepted" if meets_min else "below_minimum"
        top = max(cfg.images.resolution_buckets)
        res_score = 100 * min(1.0, math.sqrt(w * h) / top)

        components = {
            "sharpness": q["sharpness"]["score"], "noise": q["noise"]["score"],
            "compression": q["compression"]["score"], "exposure": q["exposure"]["exposure_score"],
            "contrast": q["exposure"]["contrast_score"], "resolution": round(res_score, 1),
            "composition": q["composition"]["score"],
            "aesthetic": zs["aesthetic"]["score"] if zs else None,
        }
        weights = cfg.quality.weights.model_dump()
        used = {k: weights[k] for k, v in components.items() if v is not None and weights[k] > 0}
        total_w = sum(used.values())
        overall = sum(components[k] * wt for k, wt in used.items()) / total_w if total_w else None

        flags = list(q["exposure"]["flags"]) + list(q["color"]["flags"])
        flags += q["composition"]["prominence_flags"] + q["composition"]["crop_flags"]
        if q["sharpness"]["score"] < 25:
            flags.append("blurry")
        if q["noise"]["score"] < 30:
            flags.append("noisy")
        if components["compression"] is not None and components["compression"] < 35:
            flags.append("compression_artifacts")
        jq = q["compression"].get("jpeg_quality_estimate")
        if jq is not None and jq < 60:
            flags.append("low_jpeg_quality")
        if not meets_min:
            flags.append("below_min_resolution")
        elif bucket is None:
            flags.append("below_smallest_bucket")
        if ocr:
            flags += ocr.get("flags", [])
            regions = ocr.get("regions", [])
            sparse = len(regions) <= SPARSE_TEXT_REGIONS
            if any(_looks_like_watermark(r, sparse) for r in regions):
                flags.append("watermark_text_detected")

        reasons = []
        if overall is not None:
            ranked = sorted(used, key=lambda k: (components[k] - overall) * used[k])
            for k in ranked[:3]:
                if components[k] < 60:
                    reasons.append(f"{k} score {components[k]:.0f}/100 (weight {used[k] / total_w:.0%})")
        return {
            "overall_score": None if overall is None else round(overall, 1),
            "components": components, "weights_used": {k: round(v / total_w, 4) for k, v in used.items()} if total_w else {},
            "missing_components": [k for k, v in components.items() if v is None],
            "resolution_status": res_status, "bucket": bucket, "flags": sorted(set(flags)), "reasons": reasons,
            "note": "weighted mean of available component scores; heuristic signals, not ground truth",
        }

    def apply(self, img, data):
        img.quality_score = data["overall_score"]


STOCK_MARKS = ("shutterstock", "gettyimages", "getty images", "istock", "alamy", "dreamstime", "adobe stock", "adobestock",
               "depositphotos", "123rf", "bigstock", "fotolia", "freepik", "vecteezy", "watermark")
_CREDIT = re.compile(r"©|\(c\)|\bcopyright\b|\ball rights reserved\b|\bphoto(graph)?s?\s*(by\b|credit|:)|\bimage\s*(by\b|credit|:)", re.I)
SPARSE_TEXT_REGIONS = 6  # a photo with a credit line has a few text regions; a screenshot or a document has many
_ADDRESS = re.compile(r"\bwww\.|\b[\w-]{3,}\.(com|net|org|co|io|tv|me|de|ru|uk|fr|in)\b|(^|\s)@[\w.]{3,}", re.I)


def _looks_like_watermark(region: dict, sparse_text: bool = True) -> bool:
    """Text that reads like a watermark or credit line. A stock-agency name or a copyright mark counts
    anywhere. A web address or @handle counts only as a small line near an edge of an image with little
    other text: on a sample of real images, addresses inside screenshots of web pages and apps were the
    false alarms, and those images are full of text."""
    text = region.get("text", "")
    if region.get("confidence", 1.0) < 0.6 or not text.strip():
        return False
    low = text.lower()
    if any(m in low for m in STOCK_MARKS) or _CREDIT.search(text):
        return True
    box = region.get("bbox") or [0, 0, 1, 1]
    x, y, w, h = box
    near_edge = min(x, y, 1 - (x + w), 1 - (y + h)) < 0.06
    return sparse_text and bool(_ADDRESS.search(text)) and near_edge and w * h < 0.02
