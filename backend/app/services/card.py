"""Text shared by the dataset cards: what the pipeline did to the images, step by step.

Everything is derived from the configuration and from the list of stages that are actually
enabled, so the card describes this run and not the tool in general."""
from __future__ import annotations

from typing import Optional

from app.pipeline.vision import labels as L

CAPTION_ADAPTERS = {
    "template": "template sentences assembled from the stored annotations (category, scene, lighting, colours, "
                "composition); no captioning model is involved",
    "sidecar": "captions supplied by the dataset builder (text file next to the image or a metadata file)",
    "smolvlm": "a local vision-language model",
    "none": "no captions",
    "auto": "the caption that came with the source image when there is one; a local vision-language model for the other images",
}


def _labels(name: str) -> str:
    return ", ".join(f"`{label}`" for label, _ in L.LABEL_SETS[name])


def _model_name(cfg, registry: Optional[dict], key: str) -> str:
    spec = (registry or {}).get(key)
    repo = getattr(spec, "repo", None)
    return f"`{repo}`" if repo else f"`{key}`"


def pipeline_steps(cfg, stages: list[dict], registry: Optional[dict] = None) -> list[str]:
    """Markdown lines: one numbered step per enabled stage, then the stages that did not run.
    ``stages`` is the output of ``runner.describe_pipeline``."""
    q, d, i, c, sp = cfg.quality, cfg.deduplication, cfg.images, cfg.captioning, cfg.splits
    on = {s["name"]: s["enabled"] for s in stages}
    clip = _model_name(cfg, registry, cfg.models.embedding)
    captioner = _model_name(cfg, registry, cfg.models.captioner) if c.adapter in ("smolvlm", "auto") else None
    detector = _model_name(cfg, registry, cfg.models.object_detector)
    lo, hi = q.manual_review_band
    dup_methods = [name for name, flag in (("exact (same SHA-256)", d.exact),
                                           (f"perceptual (pHash distance <= {d.perceptual_threshold}, cross-checked with dHash)", d.perceptual),
                                           ("near (candidate pairs verified geometrically with ORB + RANSAC)", d.near),
                                           (f"semantic (embedding cosine >= {d.semantic_threshold}, recorded for information)",
                                            d.semantic and on.get("embedding"))) if flag]
    weights = ", ".join(f"{k} {v:g}" for k, v in q.weights.model_dump().items())
    text = {
        "scan": ("Ingestion",
                 "Every file gets a stable `image_id` and a SHA-256 checksum. Files already seen are skipped."),
        "validate": ("Validation and metadata",
                     "Each file is verified and fully decoded; broken files are excluded. Width, height, format, colour "
                     "mode, bit depth, ICC profile, alpha use and selected EXIF fields are recorded."),
        "phash": ("Perceptual hashes", "pHash and dHash are computed for duplicate detection."),
        "quality_technical": ("Technical quality signals",
                              "Measured from pixels, no model: sharpness (Laplacian variance), noise, JPEG blockiness, "
                              "exposure and contrast (luma histogram), colour palette and colour cast, and composition from a "
                              "saliency map (subject placement, rule of thirds, symmetry, negative space, subject cut off at "
                              "the border)."),
        "people": ("Face presence", "Only the number of faces is stored. No identity, age, gender or any other personal "
                                    "trait is inferred."),
        "ocr": ("Visible text (OCR)", "Text and text regions are read with RapidOCR. Text that reads like a watermark or credit "
                                      "line (stock-agency names, copyright marks, a small web address or handle at an edge) sets "
                                      "the `watermark_text_detected` flag."),
        "segmentation": ("Subject segmentation", "Subject/background mask from GrabCut seeded by the saliency map."),
        "embedding": ("Image embeddings", f"CLIP image embeddings ({clip}), reused by the next steps."),
        "zero_shot": ("Zero-shot annotation",
                      f"CLIP ({clip}) scores each image against fixed label lists: category ({_labels('categories')}), "
                      f"style ({_labels('style')}), lighting ({_labels('lighting')}), scene ({_labels('scene')}), viewpoint and "
                      "depth of field. The same model gives an aesthetic ranking signal (prompt pairs, not a trained aesthetic "
                      "model)."),
        "object_detection": ("Object detection",
                             f"Open-vocabulary detection with {detector} over a fixed label list "
                             f"({', '.join(cfg.models.detection_labels or L.DETECTION_LABELS)}); boxes are stored as fractions of "
                             "the image, and spatial relationships are derived from box geometry only."),
        "quality": ("Overall quality score",
                    f"Weighted mean (0-100) of the available components ({weights}), plus a resolution check: at least "
                    f"{q.min_width}x{q.min_height} px."),
        "dedup": ("Duplicate grouping",
                  "Duplicates are grouped: " + "; ".join(dup_methods) + ". One representative per group is kept (best "
                  f"quality, then resolution); the others are {d.non_representative_action}d, never deleted."
                  + (" Every image is also compared with the images already exported in earlier shards (same checksum, or "
                     "pHash within the same distance confirmed by dHash) and excluded when it matches one." if d.cross_shard else "")
                  if dup_methods else "Duplicate detection is switched off."),
        "curation_pre": ("Preliminary filtering", "Images that already fail a hard rule are excluded before captioning."),
        "captioning": ("Captions and tags",
                       f"Caption source: {CAPTION_ADAPTERS.get(c.adapter, c.adapter)}"
                       + (f" ({captioner})" if captioner else "") + ". Each record has a short, a detailed and a comprehensive "
                       "caption, a generation prompt and a tag list. The "
                       f"`{c.training_caption_field}` caption is the `text` of the training manifests."),
        "alignment": ("Caption verification",
                      f"Image-text similarity with CLIP ({clip}) plus per-term checks for unsupported or missing details. "
                      "The result is a score, a verification status and warnings on each record; it is a flag for review, "
                      "not proof that a caption is correct."),
        "caption_diversity": ("Caption diversity", "Repeated openings, duplicate captions and low-information captions "
                                                   "are counted in `reports/alignment_report.json`."),
        "curation": ("Final filtering",
                     f"Rejected: below the minimum size, or overall quality under {q.min_overall_score:g} and outside the "
                     f"review band {lo:g}-{hi:g}"
                     + (f", or aesthetic score under {q.min_aesthetic_score:g}" if q.min_aesthetic_score is not None else "")
                     + f". Quarantined (kept out of the dataset, never deleted): quality under {q.min_overall_score:g} inside the "
                     "review band, duplicates that are not the representative, images with a failed stage or without a caption"
                     + (", images whose caption failed verification" if c.quarantine_failed_verification and on.get("alignment") else "")
                     + (", images with detected watermark text" if q.reject_watermarked and on.get("ocr") else "")
                     + (f", images with an aesthetic signal under {q.review_min_aesthetic_score:g} whatever their overall score"
                        if q.review_min_aesthetic_score is not None and on.get("zero_shot") else "")
                     + ". Rights status must be one of " + ", ".join(f"`{s}`" for s in cfg.rights.allowed_statuses)
                     + ". Every decision is stored with its reasons in `rejected/rejected_records.jsonl`."),
        "image_output": ("Image files",
                         "Accepted images are written to `images/`. EXIF orientation is applied"
                         + (", non-sRGB colour profiles are converted to sRGB" if i.convert_to_srgb else "")
                         + (", GPS location metadata is stripped" if i.strip_location_metadata else "")
                         + (f", the longest side is limited to {i.max_side} px" if i.max_side else "")
                         + ". Aspect ratio is kept and nothing is upscaled. A file that needs no change is a byte-for-byte "
                         "copy of the source (`copied_unchanged`); each record lists its own `transformations`."),
        "reconcile_outputs": ("Output consistency", "Written files are re-checked against their checksums; files that "
                                                    "no longer belong to the dataset are removed."),
        "splits": ("Train / validation / test split",
                   f"Deterministic split {sp.train:g} / {sp.validation:g} / {sp.test:g} (seed {sp.seed})"
                   + (". Images of one duplicate group always land in the same split" if sp.group_near_duplicates else "")
                   + (", and so do images of one semantic-similarity group (very similar subjects)" if sp.group_semantic else "")
                   + ". Leakage is re-checked independently before export."),
        "export": ("Export and validation",
                   "Every record is checked before it is written: image file exists and decodes, caption present, record "
                   "matches the schema, rights status allowed, no split leakage, checksum matches. Then manifests, metadata, "
                   "reports and `SHA256SUMS` are written."),
    }
    lines, skipped, n = [], [], 0
    for s in stages:
        if s["name"] not in text:
            continue
        title, body = text[s["name"]]
        if s["enabled"]:
            n += 1
            lines.append(f"{n}. **{title}** (`{s['name']}`): {body}")
        else:
            skipped.append(f"{title} (`{s['name']}`)")
    if skipped:
        lines += ["", "Not run for this dataset, so the matching fields are empty: " + ", ".join(skipped) + "."]
    return lines
