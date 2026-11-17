"""Read-only views over real pipeline state for the CLI, API and dashboard."""
from __future__ import annotations

from datetime import timezone
from typing import Optional

from sqlalchemy import func, select

from app.core import hardware
from app.core.io import free_disk_gb
from app.core.workspace import Workspace
from app.database.models import Image, Job, JobStage, Source, StageResult, utcnow
from app.database.session import kv_get
from app.pipeline.captioning.stage import effective_captions
from app.pipeline.export.records import visual_annotations


def _count(s, *where) -> int:
    return s.scalar(select(func.count(Image.id)).where(Image.missing.is_(False), *where)) or 0


def job_view(job: Optional[Job]) -> Optional[dict]:
    if job is None:
        return None
    stages = []
    for js in job.stage_runs:
        info = js.info or {}
        eta = None
        rate = info.get("rate_per_s") if isinstance(info, dict) else None
        if js.status == "running" and rate:
            remaining = max(0, js.total - js.done - js.reused - js.failed - info.get("ineligible", 0) - info.get("blocked", 0))
            eta = round(remaining / rate, 1)
        stages.append({"stage": js.stage, "status": js.status, "reason": js.reason, "total": js.total, "done": js.done,
                       "reused": js.reused, "failed": js.failed, "rate_per_s": rate, "eta_seconds": eta,
                       "started_at": _iso(js.started_at), "finished_at": _iso(js.finished_at), "info": info})
    return {"id": job.id, "status": job.status, "mode": job.mode, "config_hash": job.config_hash,
            "current_stage": job.current_stage, "cancel_requested": job.cancel_requested, "error": job.error,
            "created_at": _iso(job.created_at), "started_at": _iso(job.started_at), "finished_at": _iso(job.finished_at),
            "stages": stages}


def _iso(dt):
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def dashboard(ws: Workspace) -> dict:
    with ws.db.session() as s:
        total = _count(s)
        valid = _count(s, Image.valid.is_(True))
        invalid = _count(s, Image.valid.is_(False))
        scan = kv_get(s, "scan:last", None) or {}
        failed_rows = s.execute(select(StageResult.stage, func.count(StageResult.id)).where(StageResult.status == "failed")
                                .group_by(StageResult.stage)).all()
        retryable = s.scalar(select(func.count(func.distinct(StageResult.image_pk))).where(
            StageResult.status == "failed", StageResult.stage != "validate")) or 0
        decisions = dict(s.execute(select(Image.decision, func.count(Image.id)).where(Image.missing.is_(False))
                                   .group_by(Image.decision)).all())
        avg_q, avg_a, avg_al = s.execute(select(func.avg(Image.quality_score), func.avg(Image.aesthetic_score),
                                                func.avg(Image.alignment_score)).where(
            Image.missing.is_(False), Image.valid.is_(True))).one()
        latest = s.scalar(select(Job).order_by(Job.created_at.desc()).limit(1))
        processed = _count(s, Image.processed_path.is_not(None))
        export = kv_get(s, "export:last", None)
        out = {
            "images": {
                "discovered": total, "valid": valid, "invalid": invalid, "not_yet_validated": total - valid - invalid,
                "unsupported_files": (scan.get("totals") or {}).get("unsupported", 0),
                "missing_from_source": s.scalar(select(func.count(Image.id)).where(Image.missing.is_(True))) or 0,
                "processed_renditions": processed,
                "pending_processing": _count(s, Image.decision == "accepted", Image.rights_eligible.is_(True), Image.processed_path.is_(None)),
            },
            "failures": {"by_stage": dict(failed_rows), "retryable_images": retryable},
            "duplicates": {
                "exact": _count(s, Image.exact_group.is_not(None)), "perceptual": _count(s, Image.perceptual_group.is_not(None)),
                "near": _count(s, Image.near_group.is_not(None)), "semantic_similar": _count(s, Image.semantic_group.is_not(None)),
                "non_representative": _count(s, Image.representative_status == "duplicate"),
            },
            "decisions": {k: decisions.get(k, 0) for k in ("accepted", "rejected", "quarantined", "pending")},
            "rights": {"accepted_but_blocked": _count(s, Image.decision == "accepted", Image.rights_eligible.is_(False)),
                       "export_eligible": _count(s, Image.decision == "accepted", Image.rights_eligible.is_(True))},
            "review": {"pending": _count(s, Image.review_status == "pending")},
            "averages": {"quality_score": _r(avg_q), "aesthetic_score": _r(avg_a), "alignment_score": _r(avg_al, 3)},
            "splits": dict(s.execute(select(Image.split, func.count(Image.id)).where(Image.split.is_not(None)).group_by(Image.split)).all()),
            "job": job_view(latest),
            "last_export": export,
            "sources": s.scalar(select(func.count(Source.id))) or 0,
            "mode": ws.config.project.mode,
            "usage": hardware.usage(),
            "generated_at": utcnow().isoformat(),
        }
    out["disk"] = {"output_free_gb": round(free_disk_gb(ws.output_dir), 1)}
    return out


def _r(v, nd=1):
    return None if v is None else round(float(v), nd)


def image_summary(im: Image) -> dict:
    """List-view payload. Never contains the absolute source path."""
    return {"image_id": im.image_id, "file_name": im.file_name, "relative_path": im.relative_path,
            "source_id": im.source.source_id, "valid": im.valid, "validation_status": im.validation_status,
            "width": im.width, "height": im.height, "format": im.format, "orientation": im.orientation,
            "category": im.category, "style": im.style, "quality_score": im.quality_score,
            "aesthetic_score": im.aesthetic_score, "alignment_score": im.alignment_score, "short_caption": im.short_caption,
            "decision": im.decision, "auto_decision": im.auto_decision, "manual_decision": im.manual_decision,
            "decision_reasons": im.decision_reasons or [], "review_status": im.review_status,
            "review_reasons": im.review_reasons or [], "rights_status": im.effective_rights(),
            "rights_eligible": im.rights_eligible, "split": im.split, "representative_status": im.representative_status,
            "has_processed": bool(im.processed_path),
            "duplicate_groups": {k: v for k, v in (("exact", im.exact_group), ("perceptual", im.perceptual_group),
                                                   ("near", im.near_group), ("semantic", im.semantic_group)) if v}}


def image_detail(ws: Workspace, image_id: str) -> dict:
    with ws.db.session() as s:
        im = s.scalar(select(Image).where(Image.image_id == image_id))
        if im is None:
            raise KeyError(image_id)
        res = {r.stage: r for r in im.results}
        d = {k: (r.data if r.status == "done" else None) for k, r in res.items()}
        cap = d.get("captioning")
        src = im.source
        return {
            **image_summary(im),
            "file_size": im.file_size, "sha256": im.sha256, "imported_at": _iso(im.imported_at),
            "validation_reason": im.validation_reason, "metadata": d.get("validate"),
            "captions": {**effective_captions(im, cap), "generated": {k: (cap or {}).get(k) for k in ("short", "detailed", "comprehensive", "generation_prompt")},
                         "source": (cap or {}).get("source"), "warnings": (cap or {}).get("warnings", []),
                         "edited_fields": sorted((im.user_edits or {}).get("captions", {}))},
            "annotations": visual_annotations(im, res) if im.valid else None,
            "zero_shot": d.get("zero_shot"), "quality": {"aggregate": d.get("quality"), "technical": d.get("quality_technical")},
            "alignment": d.get("alignment"), "hashes": d.get("phash"), "processed": d.get("image_output") if im.processed_path else None,
            "provenance": {"source_id": src.source_id, "source_label": src.label, "license": im.license or src.license,
                           "rights_status": im.effective_rights(), "creator": im.creator or src.creator,
                           "attribution": im.attribution or src.attribution, "source_url": im.source_url or src.source_url,
                           "notes": im.notes, "rights_set_on": "image" if im.rights_status else "source"},
            "stages": [{"stage": r.stage, "status": r.status, "error": r.error, "error_category": r.error_category,
                        "attempts": r.attempts, "model": r.model_info, "updated_at": _iso(r.updated_at)} for r in im.results],
            "dup_override": im.dup_override,
        }
