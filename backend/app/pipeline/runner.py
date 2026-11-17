"""Job orchestration: ordered stages, resumability, cancellation and telemetry.

Resuming a job simply runs the stage list again: every per-image stage skips work whose
idempotency key is unchanged, so completed images are never processed twice."""
from __future__ import annotations

import secrets
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy import select

from app.config.settings import PipelineConfig
from app.core.logging import get_logger
from app.core.runlock import RunLock
from app.core.workspace import Workspace
from app.database.models import Job, JobStage, utcnow
from app.models.hub import ModelHub
from app.pipeline.alignment.stage import AlignmentStage
from app.pipeline.base import Cancelled, ImageStage, StageContext, run_image_stage
from app.pipeline.captioning.stage import CaptioningStage, run_caption_diversity
from app.pipeline.curation.curate import run_curation
from app.pipeline.curation.splits import run_splits
from app.pipeline.deduplication.dedup import run_dedup
from app.pipeline.deduplication.hashing import PerceptualHashStage
from app.pipeline.export.exporter import run_export
from app.pipeline.image_output.writer import ImageOutputStage, reconcile_outputs
from app.pipeline.ingestion.scanner import scan_all
from app.pipeline.quality.aggregate import QualityAggregateStage
from app.pipeline.quality.technical import TechnicalQualityStage
from app.pipeline.validation.validate import ValidateStage
from app.pipeline.vision.stages import (EmbeddingStage, ObjectDetectionStage, OcrStage, PeopleStage,
                                        SegmentationStage, ZeroShotStage)

log = get_logger("runner")


class DatasetStage:
    """A stage that works on the whole collection (still cheap to repeat)."""

    def __init__(self, name: str, title: str, processes: str, fn: Callable[[StageContext], dict],
                 enabled: Callable[[StageContext], tuple[bool, str]] = lambda ctx: (True, "")):
        self.name, self.title, self.processes, self.fn, self.enabled = name, title, processes, fn, enabled


def _scan(ctx: StageContext) -> dict:
    out = scan_all(ctx.ws, cancel=ctx.cancel, progress=lambda st: ctx.progress("scan", {
        "source": st["label"], "files_seen": st["files_seen"], "new": st["new"], "unchanged": st["unchanged"],
        "unsupported": st["unsupported"]}))
    if ctx.cancel():
        raise Cancelled()
    return out["totals"]


def _write_outputs(ctx: StageContext) -> dict:
    return reconcile_outputs(ctx)


def _hf_upload(ctx: StageContext) -> dict:
    from app.services.hf_upload import maybe_upload

    return maybe_upload(ctx.ws, ctx.cfg, say=lambda m: ctx.progress("hf_upload", {"message": m}))


def _hf_on(ctx):
    hf = ctx.cfg.huggingface
    if not hf.enabled:
        return False, "Hugging Face upload is off (huggingface.enabled: false)"
    if not hf.repo_id:
        return False, "huggingface.repo_id is not set"
    return True, ""


def _captions_on(ctx):
    on = ctx.cfg.captioning.adapter != "none" and ctx.cfg.stages.captioning is not False
    return on, "captioning disabled"


def build_pipeline() -> list:
    return [
        DatasetStage("scan", "Folder ingestion, IDs and checksums", "1, 3, 4, 10", _scan),
        ValidateStage(), PerceptualHashStage(), TechnicalQualityStage(), PeopleStage(), OcrStage(), SegmentationStage(),
        EmbeddingStage(), ZeroShotStage(), ObjectDetectionStage(), QualityAggregateStage(),
        DatasetStage("dedup", "Duplicate and similarity grouping", "11, 13, 14, 15", run_dedup),
        DatasetStage("curation_pre", "Preliminary filtering", "52", lambda ctx: run_curation(ctx, final=False)),
        CaptioningStage(), AlignmentStage(),
        DatasetStage("caption_diversity", "Caption diversity analysis", "45", run_caption_diversity, _captions_on),
        DatasetStage("curation", "Filtering, redundancy control and rights eligibility", "52, 53, 54, 55, 59",
                     lambda ctx: run_curation(ctx, final=True)),
        ImageOutputStage(),
        DatasetStage("reconcile_outputs", "Processed image layer consistency", "Stage 8", _write_outputs),
        DatasetStage("splits", "Leakage-safe train/validation/test split", "56, 57, 58", run_splits),
        DatasetStage("export", "Validation, manifests, reports and dataset card", "Section 8", lambda ctx: run_export(ctx)),
        DatasetStage("hf_upload", "Upload to Hugging Face when the dataset reaches the size threshold", "optional",
                     _hf_upload, _hf_on),
    ]


STAGE_NAMES = [s.name for s in build_pipeline()]
# Stages that only re-derive decisions and files from cached results (no vision inference).
EXPORT_ONLY = ["curation", "image_output", "reconcile_outputs", "splits", "export", "hf_upload"]


def describe_pipeline(ws: Workspace, cfg: Optional[PipelineConfig] = None) -> list[dict]:
    cfg = cfg or ws.config
    ctx = StageContext(ws=ws, cfg=cfg, hub=ModelHub(cfg, ws.root))
    out = []
    for st in build_pipeline():
        on, reason = st.enabled(ctx)
        out.append({"name": st.name, "title": st.title, "processes": st.processes, "enabled": on,
                    "reason": "" if on else reason, "kind": "image" if isinstance(st, ImageStage) else "dataset"})
    return out


def new_job_id() -> str:
    return "job_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


class AlreadyRunning(RuntimeError):
    """Another live process is executing a job for this workspace."""


def mark_interrupted(ws: Workspace) -> int:
    """Recover after a crash or kill: when no live process holds the run lock, every job and
    stage still marked running is turned into ``interrupted`` so it can be resumed. While a run
    is alive (in this or another terminal) nothing is touched."""
    if RunLock(ws.checkpoints_dir).owner() is not None:
        return 0
    n = 0
    with ws.db.session() as s:
        for job in s.scalars(select(Job).where(Job.status.in_(("running", "queued")))):
            job.status, job.error = "interrupted", "process stopped before the job finished; resume to continue"
            for js in job.stage_runs:
                if js.status == "running":
                    js.status = "interrupted"
            n += 1
    return n


def active_job(ws: Workspace) -> Optional[str]:
    mark_interrupted(ws)
    with ws.db.session() as s:
        job = s.scalar(select(Job).where(Job.status.in_(("running", "queued"))))
        return job.id if job else None


def create_job(ws: Workspace, stages: Optional[list[str]] = None, cfg: Optional[PipelineConfig] = None) -> str:
    cfg = cfg or ws.config
    if stages:
        unknown = [s for s in stages if s not in STAGE_NAMES]
        if unknown:
            raise ValueError(f"unknown stage(s): {', '.join(unknown)}")
    running = active_job(ws)
    if running:
        raise AlreadyRunning(f"job {running} is already running (pid {RunLock(ws.checkpoints_dir).owner()}); "
                             "watch it with: image-pipeline status --watch 5")
    job_id = new_job_id()
    with ws.db.session() as s:
        s.add(Job(id=job_id, status="queued", mode=cfg.project.mode, config=cfg.model_dump(mode="json"),
                  config_hash=cfg.config_hash(), stages=stages or STAGE_NAMES, heartbeat_at=utcnow()))
    return job_id


def request_cancel(ws: Workspace, job_id: str) -> bool:
    with ws.db.session() as s:
        job = s.get(Job, job_id)
        if job is None or job.status not in ("running", "queued"):
            return False
        job.cancel_requested = True
        return True


def run_job(ws: Workspace, job_id: str, retry_failed: bool = False,
            on_stage: Optional[Callable[[str, str], None]] = None,
            on_progress: Optional[Callable[[str, dict], None]] = None,
            on_stage_end: Optional[Callable[[str, dict], None]] = None,
            on_item: Optional[Callable[[str, str, bool, str, float], None]] = None) -> dict:
    """Run (or resume) a job to completion in the calling thread."""
    lock = RunLock(ws.checkpoints_dir)
    if not lock.acquire():
        raise AlreadyRunning(f"another pipeline process (pid {lock.owner()}) is running for this workspace")
    try:
        return _run_job(ws, job_id, retry_failed, on_stage, on_progress, on_stage_end, on_item)
    finally:
        lock.release()


def _run_job(ws, job_id, retry_failed, on_stage, on_progress, on_stage_end, on_item) -> dict:
    with ws.db.session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise ValueError(f"job {job_id} not found")
        cfg = PipelineConfig.model_validate(job.config)
        wanted = list(job.stages or STAGE_NAMES)
        job.status, job.error, job.cancel_requested = "running", None, False
        job.started_at = job.started_at or utcnow()
        job.finished_at, job.heartbeat_at = None, utcnow()
        for js in list(job.stage_runs):
            s.delete(js)
    hub = ModelHub(cfg, ws.root)
    state = {"t": 0.0, "cancel": False}

    def cancel() -> bool:
        if state["cancel"]:
            return True
        if time.time() - state["t"] > 1.0:
            state["t"] = time.time()
            with ws.db.session() as s:
                j = s.get(Job, job_id)
                state["cancel"] = bool(j.cancel_requested)
                j.heartbeat_at = utcnow()
        return state["cancel"]

    def heartbeat() -> None:
        with ws.db.session() as s:
            s.get(Job, job_id).heartbeat_at = utcnow()

    ctx = StageContext(ws=ws, cfg=cfg, job_id=job_id, hub=hub, cancel=cancel, retry_failed=retry_failed, heartbeat=heartbeat)
    if on_progress:
        ctx.progress = on_progress
    if on_item:
        ctx.item = on_item

    def stage_end(name: str) -> None:
        if on_stage_end:
            with ws.db.session() as s:
                row = s.get(JobStage, ids[name])
                on_stage_end(name, {"status": row.status, "reason": row.reason, "total": row.total, "done": row.done,
                                    "reused": row.reused, "failed": row.failed, "info": row.info or {},
                                    "kind": "image" if name in image_stage_names else "dataset"})

    # Long dataset-level stages (dedup, export) may not reach a checkpoint for a while; a small
    # background beat keeps the job from being mistaken for a crashed one.
    stop_beat = threading.Event()

    def beat():
        while not stop_beat.wait(15):
            try:
                heartbeat()
            except Exception:
                pass

    beater = threading.Thread(target=beat, name=f"heartbeat-{job_id}", daemon=True)
    beater.start()
    pipeline = [st for st in build_pipeline() if st.name in wanted]
    image_stage_names = {st.name for st in pipeline if isinstance(st, ImageStage)}
    with ws.db.session() as s:
        ids = {}
        for st in pipeline:
            js = JobStage(job_id=job_id, stage=st.name, status="pending")
            s.add(js)
            s.flush()
            ids[st.name] = js.id
    final_status, error = "completed", None
    try:
        for st in pipeline:
            if cancel():
                raise Cancelled()
            on, reason = st.enabled(ctx)
            with ws.db.session() as s:
                js = s.get(JobStage, ids[st.name])
                s.get(Job, job_id).current_stage = st.name
                if not on:
                    js.status, js.reason = "skipped", reason
                else:
                    js.status, js.started_at = "running", utcnow()
            if not on:
                stage_end(st.name)
                continue
            if on_stage:
                on_stage(st.name, "running")
            log.info("job %s: stage %s started", job_id, st.name)
            t0 = time.time()
            try:
                if isinstance(st, ImageStage):
                    run_image_stage(ctx, st, js)
                    info = None
                else:
                    info = st.fn(ctx)
            except Cancelled:
                with ws.db.session() as s:
                    js = s.get(JobStage, ids[st.name])
                    js.status, js.finished_at = "cancelled", utcnow()
                stage_end(st.name)
                raise
            except Exception as exc:
                with ws.db.session() as s:
                    js = s.get(JobStage, ids[st.name])
                    js.status, js.finished_at, js.reason = "failed", utcnow(), f"{type(exc).__name__}: {exc}"[:500]
                stage_end(st.name)
                raise
            with ws.db.session() as s:
                js = s.get(JobStage, ids[st.name])
                js.status, js.finished_at = "completed", utcnow()
                if info is not None:
                    js.info = _compact(info)
                else:
                    js.info = {**(js.info or {}), "elapsed_s": round(time.time() - t0, 2)}
            stage_end(st.name)
            if on_stage:
                on_stage(st.name, "completed")
    except Cancelled:
        final_status = "cancelled"
    except KeyboardInterrupt:
        final_status, error = "interrupted", "stopped by the user (Ctrl+C); continue with: image-pipeline resume"
    except Exception as exc:
        final_status, error = "failed", f"{type(exc).__name__}: {exc}"[:1000]
        log.error("job %s failed: %s\n%s", job_id, error, traceback.format_exc())
    finally:
        stop_beat.set()
        beater.join(timeout=5)
        hub.unload()
    with ws.db.session() as s:
        job = s.get(Job, job_id)
        job.status, job.error, job.finished_at, job.current_stage = final_status, error, utcnow(), None
        cut = [js.stage for js in job.stage_runs if js.status == "running"]
        for js in job.stage_runs:
            if js.status == "running":
                js.status, js.finished_at = "interrupted", utcnow()
                js.reason = js.reason or "interrupted before it finished"
        job.summary = {js.stage: {"status": js.status, "total": js.total, "done": js.done, "reused": js.reused,
                                  "failed": js.failed, "reason": js.reason} for js in job.stage_runs}
        summary = {"job_id": job_id, "status": final_status, "error": error, "stages": job.summary}
    for name in cut:
        stage_end(name)
    log.info("job %s finished: %s", job_id, final_status)
    return summary


def _compact(info: dict, limit: int = 4000) -> dict:
    import json

    try:
        if len(json.dumps(info, default=str)) <= limit:
            return info
    except Exception:
        pass
    return {k: v for k, v in info.items() if isinstance(v, (int, float, str, bool)) or v is None}
