"""Assemble the public, validated training record for one image from its stage results."""
from __future__ import annotations

from typing import Optional

from app import PIPELINE_VERSION
from app.database.models import Image, StageResult
from app.pipeline.captioning.stage import effective_captions
from app.schemas.record import TrainingRecord


def _data(results: dict[str, StageResult], stage: str) -> Optional[dict]:
    r = results.get(stage)
    return r.data if r is not None and r.status == "done" else None


def _labels(zs: Optional[dict], name: str) -> list[str]:
    return [x["label"] for x in zs[name]["labels"]] if zs and name in zs else []


def camera_attributes(meta: dict, zs: Optional[dict]) -> dict:
    """Capture facts come only from EXIF; anything estimated from pixels is marked inferred."""
    exif = meta.get("exif", {})
    out = {}
    for key in ("camera_make", "camera_model", "lens_model", "focal_length_mm", "f_number", "exposure_time_s", "iso", "datetime_original"):
        out[key] = ({"value": exif[key], "status": "exif", "confidence": None} if key in exif
                    else {"value": None, "status": "unknown", "confidence": None})
    for name in ("viewpoint", "depth_of_field"):
        if zs and name in zs:
            top = zs[name]["labels"][0]
            out[name] = {"value": top["label"], "status": "inferred", "confidence": top["confidence"]}
        else:
            out[name] = {"value": None, "status": "unknown", "confidence": None}
    return out


def visual_annotations(img: Image, results: dict[str, StageResult]) -> dict:
    zs, q = _data(results, "zero_shot"), _data(results, "quality_technical")
    det, ocr, ppl = _data(results, "object_detection"), _data(results, "ocr"), _data(results, "people")
    seg, cap, meta = _data(results, "segmentation"), _data(results, "captioning"), _data(results, "validate") or {}
    qa = _data(results, "quality")
    comp = (q or {}).get("composition", {})
    edits = (img.user_edits or {}).get("tags")
    # Decided from text read in the image; None when no text was read (the CLIP guess alone is not reliable).
    watermark = "watermark_text_detected" in (qa or {}).get("flags", []) if ocr else None
    return {
        "categories": _labels(zs, "categories"),
        "objects": (det or {}).get("objects", []),
        "scene": (_labels(zs, "scene") or [None])[0],
        "relationships": (det or {}).get("relationships", []),
        "lighting": _labels(zs, "lighting"),
        "style": _labels(zs, "style"),
        "color_palette": (q or {}).get("color", {}).get("palette", []),
        "composition": {k: comp[k] for k in ("subject_placement", "saliency_centroid", "rule_of_thirds_distance",
                                             "horizontal_symmetry", "negative_space_fraction", "subject", "prominence_score",
                                             "crop_flags") if k in comp} | ({"status": "inferred"} if comp else {}),
        "camera_attributes": camera_attributes(meta, zs),
        "ocr": {"text": (ocr or {}).get("text", []), "regions": (ocr or {}).get("regions", [])},
        "watermarks_detected": watermark,
        "people": {k: ppl[k] for k in ("face_count", "status", "policy")} if ppl else {},
        "segmentation": ({"mask_path": f"annotations/masks/{img.image_id}.png", "foreground_fraction": seg["foreground_fraction"],
                          "bbox": seg["bbox"], "method": seg["method"]} if seg and seg.get("available") else None),
        "tags": edits if edits is not None else (cap or {}).get("tags", []),
        "annotation_status": "inferred" if (zs or det or ocr or ppl or q) else "not_computed",
    }


def build_record(img: Image, results: dict[str, StageResult], config_hash: str, split_seed: int) -> TrainingRecord:
    """Raises ``pydantic.ValidationError`` (or KeyError) when the record is not exportable."""
    out, cap = _data(results, "image_output"), _data(results, "captioning")
    qa, al = _data(results, "quality"), _data(results, "alignment")
    if out is None or not img.processed_path:
        raise KeyError("no processed rendition")
    caps = effective_captions(img, cap)
    edited = bool((img.user_edits or {}).get("captions"))
    comps = (qa or {}).get("components", {})
    src = img.source
    alignment = {"score": None, "method": None, "verification_status": "not_run", "warnings": []}
    if al:
        alignment = {"score": al.get("score"), "raw_cosine": al.get("raw_cosine"), "method": al.get("method"),
                     "verification_status": al.get("verification_status", "not_run"), "warnings": al.get("warnings", [])}
    if edited or img.review_status in ("approved", "edited"):
        alignment["verification_status"] = "manually_reviewed"
    record = {
        "image_id": img.image_id,
        "image_path": img.processed_path,
        "source": {"source_id": src.source_id, "relative_path": img.relative_path, "sha256": img.sha256,
                   "source_url": img.source_url or src.source_url, "creator": img.creator or src.creator,
                   "license": img.license or src.license or "unknown", "rights_status": img.effective_rights(),
                   "attribution": img.attribution or src.attribution},
        "processed_image": {k: out[k] for k in ("format", "width", "height", "aspect_ratio", "sha256", "is_upscaled",
                                                "transformations", "bucket")},
        "captions": {**caps, "source": "user_edited" if edited else (cap or {}).get("source"), "edited_by_user": edited},
        "visual_annotations": visual_annotations(img, results),
        "quality": {"overall_score": (qa or {}).get("overall_score"), "sharpness_score": comps.get("sharpness"),
                    "noise_score": comps.get("noise"), "exposure_score": comps.get("exposure"),
                    "contrast_score": comps.get("contrast"), "compression_score": comps.get("compression"),
                    "composition_score": comps.get("composition"), "aesthetic_score": comps.get("aesthetic"),
                    "resolution_status": (qa or {}).get("resolution_status"), "flags": (qa or {}).get("flags", [])},
        "alignment": alignment,
        "deduplication": {"exact_duplicate_group": img.exact_group, "perceptual_duplicate_group": img.perceptual_group,
                          "near_duplicate_group": img.near_group, "semantic_similarity_group": img.semantic_group,
                          "representative_status": img.representative_status},
        "processing": {"status": "completed",
                       "model_versions": {st: r.model_info for st, r in sorted(results.items()) if r.model_info and r.status == "done"},
                       "pipeline_version": PIPELINE_VERSION, "config_hash": config_hash,
                       "processed_at": max(r.updated_at for r in results.values()).isoformat()},
        "curation": {"decision": img.decision, "review_status": img.review_status,
                     "rejection_reasons": img.decision_reasons or [], "split": img.split, "split_seed": split_seed},
    }
    return TrainingRecord.model_validate(record)
