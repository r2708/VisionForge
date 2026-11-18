"""Section 8: export validation, manifests (JSONL/CSV/Parquet), minimal trainer manifests,
metadata layer, reports, dataset card and the package checksum manifest.

Only records that pass every check are written to training manifests. All exported paths are
relative to the dataset root and no private absolute path is ever written."""
from __future__ import annotations

import csv
import io
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image as PILImage
from sqlalchemy import func, select

from app import PIPELINE_VERSION, SCHEMA_VERSION
from app.core.io import atomic_write_bytes, atomic_write_json, atomic_write_jsonl, atomic_write_text, sha256_file
from app.core.logging import get_logger
from app.database.models import Image, Source, StageResult
from app.database.session import kv_get, kv_set
from app.pipeline.base import StageContext
from app.pipeline.captioning.stage import effective_captions
from app.pipeline.curation.splits import SPLITS, validate_splits
from app.pipeline.vision import labels as L

from .records import build_record, visual_annotations

log = get_logger("export")


def _iter_images(ctx: StageContext, *where) -> Iterator[tuple[Image, dict[str, StageResult]]]:
    last = 0
    while True:
        with ctx.ws.db.session() as s:
            imgs = list(s.scalars(select(Image).where(Image.id > last, Image.missing.is_(False), *where)
                                  .order_by(Image.id).limit(ctx.cfg.processing.db_chunk_size)))
            if not imgs:
                return
            last = imgs[-1].id
            res = defaultdict(dict)
            for r in s.scalars(select(StageResult).where(StageResult.image_pk.in_([i.id for i in imgs]))):
                res[r.image_pk][r.stage] = r
            for im in imgs:
                _ = im.source  # load while the session is open
                yield im, res[im.id]


def _flat(rec: dict) -> dict:
    va, q = rec["visual_annotations"], rec["quality"]
    return {
        "image_id": rec["image_id"], "image_path": rec["image_path"], "split": rec["curation"]["split"],
        "short_caption": rec["captions"]["short"], "detailed_caption": rec["captions"]["detailed"],
        "comprehensive_caption": rec["captions"]["comprehensive"], "generation_prompt": rec["captions"]["generation_prompt"],
        "caption_source": rec["captions"]["source"], "tags": "|".join(va["tags"]),
        "categories": "|".join(va["categories"]), "style": "|".join(va["style"]), "lighting": "|".join(va["lighting"]),
        "scene": va["scene"], "width": rec["processed_image"]["width"], "height": rec["processed_image"]["height"],
        "format": rec["processed_image"]["format"], "processed_sha256": rec["processed_image"]["sha256"],
        "quality_score": q["overall_score"], "aesthetic_score": q["aesthetic_score"],
        "alignment_score": rec["alignment"]["score"], "verification_status": rec["alignment"]["verification_status"],
        "review_status": rec["curation"]["review_status"], "source_id": rec["source"]["source_id"],
        "source_relative_path": rec["source"]["relative_path"], "source_sha256": rec["source"]["sha256"],
        "license": rec["source"]["license"], "rights_status": rec["source"]["rights_status"],
        "creator": rec["source"]["creator"], "attribution": rec["source"]["attribution"], "source_url": rec["source"]["source_url"],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    atomic_write_text(path, buf.getvalue())


def parquet_schema():
    """Explicit column types for the parquet manifests. Declaring them keeps every file of every
    shard on one schema: an empty list or an all-null column never gets a guessed type."""
    import pyarrow as pa

    text, strings = pa.string(), pa.list_(pa.string())
    return pa.schema([
        ("image_id", text), ("image_path", text), ("split", text), ("text", text),
        ("short_caption", text), ("detailed_caption", text), ("comprehensive_caption", text), ("generation_prompt", text),
        ("caption_source", text), ("tags", strings), ("categories", strings), ("style", strings), ("lighting", strings),
        ("scene", text), ("width", pa.int32()), ("height", pa.int32()), ("format", text), ("processed_sha256", text),
        ("quality_score", pa.float64()), ("aesthetic_score", pa.float64()), ("alignment_score", pa.float64()),
        ("verification_status", text), ("alignment_warnings", strings), ("quality_flags", strings),
        ("watermark_text_detected", pa.bool_()), ("face_count", pa.int32()), ("ocr_text", strings), ("object_labels", strings),
        ("review_status", text), ("source_id", text), ("source_relative_path", text), ("source_sha256", text),
        ("license", text), ("rights_status", text), ("creator", text), ("attribution", text), ("source_url", text),
        ("record_json", text),
    ])


def typed_row(rec: dict, caption_field: str) -> dict:
    """One parquet row: flat, typed columns for filtering plus the complete record as JSON text."""
    va, q, al = rec["visual_annotations"], rec["quality"], rec["alignment"]

    def strings(values) -> list[str]:
        return [str(v) for v in (values or []) if v is not None]

    return {
        "image_id": rec["image_id"], "image_path": rec["image_path"], "split": rec["curation"]["split"],
        "text": rec["captions"].get(caption_field) or "",
        "short_caption": rec["captions"]["short"], "detailed_caption": rec["captions"]["detailed"],
        "comprehensive_caption": rec["captions"]["comprehensive"], "generation_prompt": rec["captions"]["generation_prompt"],
        "caption_source": rec["captions"]["source"], "tags": strings(va["tags"]), "categories": strings(va["categories"]),
        "style": strings(va["style"]), "lighting": strings(va["lighting"]), "scene": va["scene"],
        "width": rec["processed_image"]["width"], "height": rec["processed_image"]["height"],
        "format": rec["processed_image"]["format"], "processed_sha256": rec["processed_image"]["sha256"],
        "quality_score": q["overall_score"], "aesthetic_score": q["aesthetic_score"], "alignment_score": al["score"],
        "verification_status": al["verification_status"], "alignment_warnings": strings(al["warnings"]),
        "quality_flags": strings(q["flags"]), "watermark_text_detected": va["watermarks_detected"],
        "face_count": (va["people"] or {}).get("face_count"), "ocr_text": strings((va["ocr"] or {}).get("text")),
        "object_labels": strings(o.get("label") for o in va["objects"]),
        "review_status": rec["curation"]["review_status"], "source_id": rec["source"]["source_id"],
        "source_relative_path": rec["source"]["relative_path"], "source_sha256": rec["source"]["sha256"],
        "license": rec["source"]["license"], "rights_status": rec["source"]["rights_status"],
        "creator": rec["source"]["creator"], "attribution": rec["source"]["attribution"],
        "source_url": rec["source"]["source_url"], "record_json": json.dumps(rec, ensure_ascii=False),
    }


def _write_parquet(path: Path, records: list[dict], caption_field: str) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist([typed_row(r, caption_field) for r in records], schema=parquet_schema())
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)


def _qa_pairs(rec: dict) -> list[dict]:
    """Simple question-answer pairs derived only from stored annotations."""
    va, qa = rec["visual_annotations"], []
    if rec["captions"]["short"]:
        qa.append({"question": "What does this image show?", "answer": rec["captions"]["short"], "basis": "caption"})
    names = list(dict.fromkeys(p["name"] for p in va["color_palette"] if p["fraction"] >= 0.15))
    if names:
        qa.append({"question": "What are the dominant colors in this image?", "answer": ", ".join(names[:3]), "basis": "color_palette"})
    if va["ocr"]["text"]:
        qa.append({"question": "What text is visible in this image?", "answer": "; ".join(va["ocr"]["text"][:5]), "basis": "ocr"})
    if va["style"]:
        qa.append({"question": "What visual style is this image?", "answer": va["style"][0].replace("_", " "), "basis": "inferred_style"})
    return qa


def run_export(ctx: StageContext, formats: Optional[list[str]] = None) -> dict:
    cfg, ws = ctx.cfg, ctx.ws
    out = ws.output_dir
    formats = formats or cfg.export.formats
    mode = cfg.project.mode
    config_hash = cfg.config_hash()
    field = cfg.captioning.training_caption_field
    need_caption = mode in ("text_to_image", "general_multimodal")
    allowed = set(cfg.rights.allowed_statuses)
    for d in ("manifests", "metadata", "rejected", "reports"):
        (out / d).mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    problems: list[dict] = []
    checks = Counter()
    pending_review = 0
    for im, res in _iter_images(ctx, Image.decision == "accepted", Image.rights_eligible.is_(True)):
        def bad(check: str, detail: str):
            problems.append({"image_id": im.image_id, "check": check, "detail": detail})
            checks[check] += 1

        if not im.processed_path:
            bad("processed_image_missing", "no processed rendition recorded")
            continue
        p = out / im.processed_path
        if not p.is_file():  # 1. path exists
            bad("path_missing", im.processed_path)
            continue
        try:  # 2. decodes
            with PILImage.open(p) as pim:
                pim.load()
        except Exception as exc:
            bad("not_decodable", type(exc).__name__)
            continue
        if sha256_file(p) != im.processed_sha256:  # 7. checksum
            bad("checksum_mismatch", im.processed_path)
            continue
        if cfg.rights.require_approved_status_for_export and im.effective_rights() not in allowed:  # 5. rights
            bad("rights_not_allowed", im.effective_rights())
            continue
        if im.split not in SPLITS:
            bad("no_split", "image has no split assignment")
            continue
        try:  # 4. schema
            rec = build_record(im, res, config_hash, cfg.splits.seed).model_dump(mode="json")
        except Exception as exc:
            bad("invalid_record", str(exc)[:300])
            continue
        if need_caption and not (rec["captions"].get(field) or "").strip():  # 3. caption
            bad("missing_caption", f"captions.{field} is empty")
            continue
        if mode == "vision_language" and not any(rec["captions"][k] for k in ("short", "detailed", "comprehensive")):
            bad("missing_caption", "no caption available")
            continue
        pending_review += im.review_status == "pending"
        records.append(rec)

    leakage = validate_splits(ctx)  # 6. leakage
    prev = {}
    with ws.db.session() as s:
        prev = kv_get(s, "report:split_leakage", {}) or {}
    leakage["assignment"] = prev.get("assignment")
    if leakage["leakage_detected"]:
        leaked = {i for leak in leakage["leaks"] for i in leak["image_ids"]}
        for rec in list(records):
            if rec["image_id"] in leaked:
                records.remove(rec)
                problems.append({"image_id": rec["image_id"], "check": "split_leakage", "detail": "duplicate group spans splits"})
                checks["split_leakage"] += 1

    records.sort(key=lambda r: r["image_id"])
    by_split = {k: [r for r in records if r["curation"]["split"] == k] for k in SPLITS}
    flat = [_flat(r) for r in records]
    written: list[str] = []

    def w_jsonl(rel: str, rows):
        atomic_write_jsonl(out / rel, rows)
        written.append(rel)

    # ---- full manifests
    if "jsonl" in formats:
        w_jsonl("manifests/all_records.jsonl", records)
        for k in SPLITS:
            w_jsonl(f"manifests/{k}.jsonl", by_split[k])
    if "csv" in formats:
        _write_csv(out / "manifests/all_records.csv", flat)
        written.append("manifests/all_records.csv")
        for k in SPLITS:
            _write_csv(out / f"manifests/{k}.csv", [f for f in flat if f["split"] == k])
            written.append(f"manifests/{k}.csv")
    if "parquet" in formats:
        _write_parquet(out / "manifests/all_records.parquet", records, field)
        written.append("manifests/all_records.parquet")
        for k in SPLITS:
            _write_parquet(out / f"manifests/{k}.parquet", by_split[k], field)
            written.append(f"manifests/{k}.parquet")

    # ---- minimal text-to-image manifests
    minimal = cfg.export.minimal_manifest
    images_dir = ws.images_dir
    images_dir.mkdir(parents=True, exist_ok=True)
    if mode in ("text_to_image", "general_multimodal") and minimal != "none":
        for k in SPLITS:
            w_jsonl(f"manifests/minimal_{k}.jsonl", ({"image": r["image_path"], "text": r["captions"][field]} for r in by_split[k]))
    hf_meta = images_dir / "metadata.jsonl"
    if mode != "vision_language" and minimal in ("hf_imagefolder", "both"):
        atomic_write_jsonl(hf_meta, ({"file_name": Path(r["image_path"]).name, "text": r["captions"][field],
                                      "split": r["curation"]["split"], "image_id": r["image_id"]} for r in records))
        written.append("images/metadata.jsonl")
    elif hf_meta.exists():
        hf_meta.unlink()
    keep_txt = set()
    if mode != "vision_language" and minimal in ("caption_txt", "both"):
        for r in records:
            name = Path(r["image_path"]).with_suffix(".txt").name
            atomic_write_text(images_dir / name, r["captions"][field].strip() + "\n")
            keep_txt.add(name)
    for stale in images_dir.glob("img_*.txt"):  # only caption files this pipeline wrote
        if stale.name not in keep_txt:
            stale.unlink()

    # ---- vision-language manifests
    if mode in ("vision_language", "general_multimodal"):
        for k in SPLITS:
            w_jsonl(f"manifests/vlm_{k}.jsonl", ({
                "image": r["image_path"], "image_id": r["image_id"], "short_caption": r["captions"]["short"],
                "detailed_caption": r["captions"]["detailed"], "comprehensive_caption": r["captions"]["comprehensive"],
                "attributes": {x: r["visual_annotations"][x] for x in ("categories", "scene", "lighting", "style", "tags")},
                "objects": r["visual_annotations"]["objects"], "relationships": r["visual_annotations"]["relationships"],
                "ocr_text": r["visual_annotations"]["ocr"]["text"],
                **({"qa": _qa_pairs(r)} if cfg.export.vlm_qa else {})} for r in by_split[k]))

    # ---- segmentation masks for exported records
    mask_dir = out / "annotations" / "masks"
    exported_ids = {r["image_id"] for r in records}
    for r in records:
        if r["visual_annotations"]["segmentation"]:
            sha = r["source"]["sha256"]
            src_mask = ws.cache_dir / "masks" / sha[:2] / f"{sha}.png"
            if src_mask.is_file():
                mask_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_mask, mask_dir / f"{r['image_id']}.png")
    if mask_dir.is_dir():
        for f in mask_dir.glob("img_*.png"):
            if f.stem not in exported_ids:
                f.unlink()

    # ---- metadata layer + rejected records (all known images, public fields only)
    meta_rows, cap_rows, q_rows, ann_rows, prov_rows, rejected = [], [], [], [], [], []
    dist = {k: Counter() for k in ("category", "style", "lighting", "scene", "orientation", "bucket", "source", "rights", "format")}
    q_scores, a_scores, al_scores = [], [], []
    verification = Counter()
    rights_all = Counter()
    rights_rows = []
    for im, res in _iter_images(ctx):
        d = {k: (v.data if v.status == "done" else None) for k, v in res.items()}
        rights = im.effective_rights()
        rights_all[rights] += 1
        prov_rows.append({"image_id": im.image_id, "source_id": im.source.source_id, "relative_path": im.relative_path,
                          "sha256": im.sha256, "file_size": im.file_size, "imported_at": im.imported_at.isoformat(),
                          "license": im.license or im.source.license, "rights_status": rights,
                          "creator": im.creator or im.source.creator, "attribution": im.attribution or im.source.attribution,
                          "source_url": im.source_url or im.source.source_url, "notes": im.notes})
        v = d.get("validate")
        meta_rows.append({"image_id": im.image_id, "valid": im.valid, "validation_status": im.validation_status,
                          "validation_reason": im.validation_reason,
                          **({k: v[k] for k in ("format", "color_mode", "bit_depth", "width", "height", "megapixels", "aspect_ratio",
                                                "orientation", "exif_orientation", "exif", "exif_has_gps", "icc", "has_alpha",
                                                "alpha_used", "warnings")} if v and v["status"] == "valid" else {})})
        if d.get("captioning"):
            c = d["captioning"]
            cap_rows.append({"image_id": im.image_id, **effective_captions(im, c), "tags": c.get("tags", []),
                             "source": c.get("source"), "adapter": c.get("adapter"), "warnings": c.get("warnings", []),
                             "model": res["captioning"].model_info, "template_version": c.get("template_version"),
                             "prompt_version": c.get("prompt_version"), "settings": c.get("settings"),
                             "generated_at": c.get("generated_at"), "edited_by_user": bool((im.user_edits or {}).get("captions"))})
        if d.get("quality_technical"):
            q_rows.append({"image_id": im.image_id, "aggregate": d.get("quality"), "technical": d["quality_technical"]})
        if im.valid:
            ann_rows.append({"image_id": im.image_id, **visual_annotations(im, res),
                             "zero_shot_detail": d.get("zero_shot"), "alignment": d.get("alignment")})
        exported = im.image_id in exported_ids
        if exported:
            zs = d.get("zero_shot") or {}
            dist["category"][im.category or "unknown"] += 1
            dist["style"][im.style or "unknown"] += 1
            dist["lighting"][(zs.get("lighting", {}).get("labels") or [{"label": "unknown"}])[0]["label"]] += 1
            dist["scene"][(zs.get("scene", {}).get("labels") or [{"label": "unknown"}])[0]["label"]] += 1
            dist["orientation"][im.orientation or "unknown"] += 1
            b = (d.get("quality") or {}).get("bucket")
            dist["bucket"][f"{b['width']}x{b['height']}" if b else "none"] += 1
            dist["source"][im.source.source_id] += 1
            dist["rights"][rights] += 1
            dist["format"][(d.get("image_output") or {}).get("format", "unknown")] += 1
            if im.quality_score is not None:
                q_scores.append(im.quality_score)
            if im.aesthetic_score is not None:
                a_scores.append(im.aesthetic_score)
            if im.alignment_score is not None:
                al_scores.append(im.alignment_score)
            verification[(d.get("alignment") or {}).get("verification_status", "not_run")] += 1
        else:
            reasons = list(im.decision_reasons or [])
            if im.decision == "accepted" and not im.rights_eligible:
                reasons.append(f"rights_status_not_allowed:{rights}")
            reasons += [f"export_check_failed:{p['check']}" for p in problems if p["image_id"] == im.image_id]
            rejected.append({"image_id": im.image_id, "source_id": im.source.source_id, "relative_path": im.relative_path,
                             "sha256": im.sha256, "decision": im.decision, "review_status": im.review_status,
                             "reasons": reasons, "rights_status": rights, "recoverable": True})
        if rights not in allowed and len(rights_rows) < 5000:
            rights_rows.append({"image_id": im.image_id, "rights_status": rights, "license": im.license or im.source.license})
    w_jsonl("metadata/image_metadata.jsonl", meta_rows)
    w_jsonl("metadata/captions.jsonl", cap_rows)
    w_jsonl("metadata/quality_scores.jsonl", q_rows)
    w_jsonl("metadata/annotations.jsonl", ann_rows)
    w_jsonl("metadata/provenance.jsonl", prov_rows)
    w_jsonl("rejected/rejected_records.jsonl", rejected)

    # ---- reports
    n = len(records)
    with ws.db.session() as s:
        reports = {k: kv_get(s, f"report:{k}", None) for k in ("duplicates", "caption_diversity", "curation")}
        scan = kv_get(s, "scan:last", None)
        total_images = s.scalar(select(func.count(Image.id)).where(Image.missing.is_(False)))
        invalid = s.scalar(select(func.count(Image.id)).where(Image.valid.is_(False), Image.missing.is_(False)))
        sources = [{"source_id": x.source_id, "label": x.label, "license": x.license, "rights_status": x.rights_status,
                    "creator": x.creator, "source_url": x.source_url} for x in s.scalars(select(Source))]

    def mean(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    coverage = {"basis": "exported records", "distributions": {k: dict(v.most_common()) for k, v in dist.items()},
                "overrepresented": [], "underrepresented": [], "absent_categories": []}
    if n:
        cats = dist["category"]
        uniform = n / max(1, len(cats))
        coverage["overrepresented"] = [{"category": c, "count": k, "share": round(k / n, 3)} for c, k in cats.items()
                                       if len(cats) > 1 and k > 2.5 * uniform]
        coverage["underrepresented"] = [{"category": c, "count": k, "share": round(k / n, 3)} for c, k in cats.items()
                                        if len(cats) > 1 and k < 0.25 * uniform]
        coverage["absent_categories"] = sorted({lbl for lbl, _ in L.LABEL_SETS["categories"]} - set(cats))
        coverage["note"] = ("Informational only. Balancing is optional (curation.max_per_category) and is "
                            "never applied silently; categories are inferred labels.")
    warnings = []
    if not n:
        warnings.append("no records were exported")
    if pending_review:
        warnings.append(f"{pending_review} exported record(s) are still awaiting manual review")
    blocked = sum(1 for r in rejected if any(x.startswith("rights_status_not_allowed") for x in r["reasons"]))
    if blocked:
        warnings.append(f"{blocked} accepted image(s) were excluded because their rights status is not in "
                        f"{sorted(allowed)}; update the rights status of sources you are entitled to use")
    if problems:
        warnings.append(f"{len(problems)} image(s) failed export validation and were left out")
    if leakage["leakage_detected"]:
        warnings.append("duplicate-group leakage across splits was detected; affected records were left out")
    for k in SPLITS:
        if n and not by_split[k] and getattr(cfg.splits, k) > 0:
            warnings.append(f"split '{k}' is empty")
    template_n = sum(1 for r in records if r["captions"]["source"] == "template_baseline")
    if template_n:
        warnings.append(f"{template_n} caption(s) are template baselines built from inferred annotations; "
                        "review them or switch to a captioning model before training")

    summary = {
        "schema_version": SCHEMA_VERSION, "pipeline_version": PIPELINE_VERSION, "config_hash": config_hash,
        "mode": mode, "generated_at": datetime.now(timezone.utc).isoformat(), "project": cfg.project.name,
        "counts": {"images_registered": total_images, "invalid_files": invalid, "exported": n,
                   "excluded": len(rejected), "splits": {k: len(v) for k, v in by_split.items()}},
        "averages": {"quality_score": mean(q_scores), "aesthetic_score": mean(a_scores), "alignment_score": mean(al_scores)},
        "caption_field_used_for_training": field, "caption_sources": dict(Counter(r["captions"]["source"] for r in records)),
        "configuration": cfg.public_dict(), "warnings": warnings,
    }
    validation = {"passed": not problems and not leakage["leakage_detected"],
                  "checks": {"1_paths_exist": checks["path_missing"] + checks["processed_image_missing"],
                             "2_images_decode": checks["not_decodable"], "3_caption_present": checks["missing_caption"],
                             "4_schema_valid": checks["invalid_record"], "5_rights_allowed": checks["rights_not_allowed"],
                             "6_no_split_leakage": leakage["leak_count"], "7_checksums_match": checks["checksum_mismatch"],
                             "no_split": checks["no_split"]},
                  "checks_note": "numbers are failure counts; 0 means the check passed for every record",
                  "problems": problems[:1000], "records_valid": n}
    rep = {
        "dataset_summary.json": summary,
        "duplicate_report.json": reports["duplicates"] or {"note": "deduplication has not run"},
        "quality_report.json": {"exported_records": n, "averages": summary["averages"],
                                "score_histogram": _hist(q_scores), "aesthetic_histogram": _hist(a_scores),
                                "flag_counts": dict(Counter(f for r in records for f in r["quality"]["flags"]).most_common()),
                                "curation": reports["curation"], "thresholds": cfg.quality.model_dump(mode="json"),
                                "note": "Scores are heuristic ranking signals; component metrics are in metadata/quality_scores.jsonl."},
        "alignment_report.json": {"exported_records": n, "mean_alignment_score": mean(al_scores),
                                  "score_histogram": _hist([x * 100 for x in al_scores]),
                                  "verification_status": dict(verification), "caption_diversity": reports["caption_diversity"],
                                  "note": "Embedding similarity is a flagging signal, not proof of caption correctness."},
        "rights_report.json": {"policy": cfg.rights.model_dump(mode="json"), "all_images_by_rights_status": dict(rights_all),
                               "exported_by_rights_status": dict(dist["rights"]), "sources": sources,
                               "images_without_allowed_rights": sum(v for k, v in rights_all.items() if k not in allowed),
                               "sample_images_without_allowed_rights": rights_rows[:500],
                               "note": "Rights statuses are supplied by the user; this tool does not verify licences."},
        "split_leakage_report.json": leakage,
        "coverage_report.json": coverage,
        "export_validation.json": validation,
    }
    for name, obj in rep.items():
        atomic_write_json(out / "reports" / name, obj)
        written.append(f"reports/{name}")
    from app.models.hub import load_registry
    from app.pipeline.runner import describe_pipeline
    from app.services.card import pipeline_steps

    steps = pipeline_steps(cfg, [st for st in describe_pipeline(ws, cfg) if st["name"] != "hf_upload"], load_registry(ws.root))
    card = dataset_card(summary, coverage, validation, leakage, sources, scan, cfg, steps)
    atomic_write_text(out / "dataset_card.md", card)
    # Hugging Face shows README.md as the dataset page; same card with the hub's metadata header.
    task = {"text_to_image": "text-to-image", "vision_language": "image-to-text", "general_multimodal": "text-to-image"}[mode]
    header = f"---\npretty_name: {json.dumps(cfg.project.name)}\ntask_categories:\n- {task}\ntags:\n- image-dataset-pipeline\n---\n\n"
    atomic_write_text(out / "README.md", header + card)
    written += ["dataset_card.md", "README.md"]

    # ---- 10. package checksum manifest
    lines = []
    for f in sorted(out.rglob("*")):
        rel = f.relative_to(out).as_posix()
        if f.is_file() and rel != "SHA256SUMS" and not rel.startswith("rejected/quarantine/") and not f.name.startswith("."):
            lines.append(f"{sha256_file(f)}  {rel}")
    atomic_write_bytes(out / "SHA256SUMS", ("\n".join(lines) + "\n").encode())
    result = {"exported": n, "splits": summary["counts"]["splits"], "validation_passed": validation["passed"],
              "problems": len(problems), "warnings": warnings, "formats": formats, "files_in_package": len(lines) + 1,
              "generated_at": summary["generated_at"]}
    with ws.db.session() as s:
        kv_set(s, "export:last", result)
    return result


def _hist(values: list[float]) -> dict:
    bins = Counter(min(9, int(v // 10)) for v in values)
    return {f"{i * 10}-{i * 10 + 10}": bins.get(i, 0) for i in range(10)}


def dataset_card(summary, coverage, validation, leakage, sources, scan, cfg, steps=None) -> str:
    c = summary["counts"]
    L_ = []
    a = L_.append
    a(f"# Dataset card: {summary['project']}\n")
    a(f"Generated {summary['generated_at']} by image-dataset-pipeline {summary['pipeline_version']} "
      f"(schema {summary['schema_version']}, config hash `{summary['config_hash']}`, mode `{summary['mode']}`).\n")
    a("## Contents\n")
    a(f"- Exported image-text records: **{c['exported']}** (train {c['splits']['train']}, validation "
      f"{c['splits']['validation']}, test {c['splits']['test']})")
    a(f"- Images registered from sources: {c['images_registered']} ({c['invalid_files']} invalid, {c['excluded']} not exported)")
    a("- `images/` processed image files; `manifests/` training manifests; `metadata/` annotations for all images; "
      "`rejected/` excluded records with reasons; `reports/` quality, duplicate, alignment, rights, coverage and leakage reports; "
      "`SHA256SUMS` package checksums.\n")
    if steps:
        a("## Processing steps\n")
        a("\n".join(steps) + "\n")
    a("## Sources and licences\n")
    if sources:
        a("| Source | Licence | Rights status | Creator | URL |\n|---|---|---|---|---|")
        for s in sources:
            a(f"| {s['label']} (`{s['source_id']}`) | {s['license'] or 'not provided'} | {s['rights_status']} | "
              f"{s['creator'] or 'not provided'} | {s['source_url'] or 'not provided'} |")
    a(f"\nRights policy: export requires status in {cfg.rights.allowed_statuses} "
      f"(enforced: {cfg.rights.require_approved_status_for_export}). Rights statuses were supplied by the dataset "
      "builder and were **not** verified by this tool. Per-image provenance is in `metadata/provenance.jsonl`.\n")
    a("## Filters applied\n")
    q, d = cfg.quality, cfg.deduplication
    a(f"- Minimum size {q.min_width}x{q.min_height} px; minimum overall quality {q.min_overall_score}/100; "
      f"manual review band {list(q.manual_review_band)}; minimum aesthetic score {q.min_aesthetic_score}.")
    a(f"- Duplicates: exact (SHA-256), perceptual (pHash distance <= {d.perceptual_threshold}), near (geometric verification), "
      f"semantic similarity (cosine >= {d.semantic_threshold}, informational). Non-representative duplicates were "
      f"{d.non_representative_action}d, never deleted.")
    a(f"- Split seed {cfg.splits.seed}; duplicate groups kept within one split: {cfg.splits.group_near_duplicates}. "
      f"Leakage detected: {leakage['leakage_detected']}.\n")
    a("## Transformations\n")
    a("Source files were never modified. Renditions apply EXIF orientation, convert non-sRGB ICC profiles to sRGB "
      f"({cfg.images.convert_to_srgb}), keep used transparency ({cfg.images.preserve_alpha}), keep the original aspect ratio and "
      "are never upscaled. Files needing no change are byte copies. Each record lists its own `transformations`.\n")
    a("## Model-generated content\n")
    a(f"- Captions used for training come from the `{summary['caption_field_used_for_training']}` field. "
      f"Caption sources: {summary['caption_sources'] or 'none'}.")
    a("- Categories, style, lighting, scene, viewpoint, aesthetic score, watermark likelihood and caption checks are "
      "**model inferences** (see `processing.model_versions` in each record) and may be wrong.")
    a("- Camera facts appear only when present in EXIF; anything estimated from pixels is marked `inferred`.")
    a("- People: only a face count is stored. No identity or personal trait is inferred.\n")
    a("## Quality summary\n")
    av = summary["averages"]
    a(f"Mean overall quality {av['quality_score']}, mean aesthetic signal {av['aesthetic_score']}, mean alignment score "
      f"{av['alignment_score']}. These are heuristic ranking signals, not measurements of ground truth.\n")
    if coverage["distributions"]["category"]:
        a("Category distribution (inferred): " + ", ".join(f"{k} {v}" for k, v in coverage["distributions"]["category"].items()) + ".\n")
    a("## Validation\n")
    a(f"Export validation passed: **{validation['passed']}**. Failure counts per check: {validation['checks']}.\n")
    a("## Limitations and warnings\n")
    for w in summary["warnings"] or ["none recorded"]:
        a(f"- {w}")
    a("- Quality, aesthetic and alignment thresholds are defaults that have not been tuned on this dataset.")
    a("- Automatic annotations have not been benchmarked against human labels for this dataset.\n")
    a("## Intended use\n")
    use = {"text_to_image": "Fine-tuning or training text-to-image models from image-caption pairs.",
           "vision_language": "Training or evaluating image captioning / vision-language models.",
           "general_multimodal": "Multimodal training with task-specific manifests."}[summary["mode"]]
    a(use + " Confirm that the licences above permit your intended use before training or redistribution.\n")
    a("## Loading\n")
    a("```python\nfrom datasets import load_dataset\n"
      "ds = load_dataset('json', data_files={s: f'manifests/minimal_{s}.jsonl' for s in ('train', 'validation', 'test')})\n"
      "# 'image' holds a path relative to the dataset root; 'text' holds the caption.\n```\n")
    return "\n".join(L_)
