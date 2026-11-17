"""Stage framework: idempotency keys, per-image checkpoints, bounded workers, retries, cancellation."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from PIL import UnidentifiedImageError
from sqlalchemy import func, select

from app.config.settings import PipelineConfig, fingerprint
from app.core.hardware import is_oom
from app.core.logging import get_logger
from app.core.workspace import Workspace
from app.database.models import Image, JobStage, StageResult, utcnow

log = get_logger("stage")


class StageSkipped(Exception):
    """Raised by a stage for an image it cannot process for a non-error reason."""


class ModelUnavailable(RuntimeError):
    pass


class Cancelled(Exception):
    pass


def categorize_error(exc: BaseException) -> str:
    if is_oom(exc):
        return "out_of_memory"
    if isinstance(exc, ModelUnavailable):
        return "model_unavailable"
    if isinstance(exc, (UnidentifiedImageError,)) or "decompression bomb" in str(exc).lower():
        return "decode"
    if isinstance(exc, FileNotFoundError):
        return "source_missing"
    if isinstance(exc, OSError):
        return "io"
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError)):
        return "processing"
    return "internal"


RETRYABLE = {"io", "out_of_memory", "internal"}


@dataclass
class ImgRef:
    """Detached snapshot handed to worker threads (ORM objects never cross threads)."""

    pk: int
    image_id: str
    abs_path: str
    sha256: str
    extension: str
    file_name: str
    relative_path: str
    width: Optional[int]
    height: Optional[int]
    supplied_caption: Optional[str]
    deps: dict[str, dict] = field(default_factory=dict)  # stage -> data of upstream results


@dataclass
class StageContext:
    ws: Workspace
    cfg: PipelineConfig
    job_id: Optional[str] = None
    hub: Any = None  # app.models.hub.ModelHub
    cancel: Callable[[], bool] = lambda: False
    retry_failed: bool = False
    heartbeat: Callable[[], None] = lambda: None
    progress: Callable[[str, dict], None] = lambda name, info: None  # live progress for the terminal
    item: Callable[[str, str, bool, str, float], None] = lambda label, image_id, ok, message, seconds: None


class ImageStage:
    """A per-image stage. Subclasses implement ``process`` (or ``process_batch``)."""

    name: str = ""
    version: str = "1"
    title: str = ""
    processes: str = ""  # process numbers from the specification
    depends: tuple[str, ...] = ()
    soft_depends: tuple[str, ...] = ()
    batched: bool = False
    require_valid: bool = True

    label: str = ""  # short upper-case tag used in terminal lines, e.g. [VALIDATE]

    def describe(self, data: dict) -> str:
        """One short phrase about a finished result, for the terminal line."""
        return ""

    def enabled(self, ctx: StageContext) -> tuple[bool, str]:
        return True, ""

    def fingerprint(self, cfg: PipelineConfig) -> Any:
        return None

    def model_info(self, ctx: StageContext) -> Optional[dict]:
        return None

    def eligible(self, ctx: "StageContext", img: Image) -> bool:
        return True

    def still_valid(self, ctx: StageContext, img: Image, data: dict) -> bool:
        """Extra check before reusing a cached result (e.g. the output file still exists)."""
        return True

    def process(self, ctx: StageContext, ref: ImgRef) -> dict:
        raise NotImplementedError

    def process_batch(self, ctx: StageContext, refs: list[ImgRef]) -> list[Any]:
        return [self._safe(ctx, r) for r in refs]

    def apply(self, img: Image, data: dict) -> None:
        """Copy summary values onto the image row."""

    def batch_size(self, ctx: StageContext) -> int:
        return 8

    # ------------------------------------------------------------------
    def _safe(self, ctx: StageContext, ref: ImgRef):
        try:
            return self.process(ctx, ref)
        except Exception as exc:  # returned, not raised: one bad image never stops a stage
            return exc

    def setup(self, ctx: StageContext) -> None:
        """Called once before the first image is processed (e.g. load a model)."""

    def teardown(self, ctx: StageContext) -> None:
        pass


def idem_key(stage: ImageStage, cfg_fp: Any, model: Optional[dict], sha256: str, dep_keys: dict[str, Optional[str]]) -> str:
    return fingerprint([sha256, stage.name, stage.version, cfg_fp, model, dep_keys])[:32]


def run_image_stage(ctx: StageContext, stage: ImageStage, js: JobStage) -> None:
    """Run one stage over every eligible image, checkpointing after each chunk."""
    db = ctx.ws.db
    cfg = ctx.cfg
    cfg_fp = stage.fingerprint(cfg)
    model = stage.model_info(ctx)
    dep_names = tuple(stage.depends) + tuple(stage.soft_depends)
    chunk = cfg.processing.db_chunk_size
    max_attempts = 1 + cfg.processing.max_retries

    def base_filter(q):
        q = q.where(Image.missing.is_(False))
        return q.where(Image.valid.is_(True)) if stage.require_valid else q

    with db.session() as s:
        js_db = s.get(JobStage, js.id)
        js_db.total = s.scalar(base_filter(select(func.count(Image.id)))) or 0
        js_db.done = js_db.reused = js_db.failed = 0
    counts = {"done": 0, "reused": 0, "failed": 0, "ineligible": 0, "blocked": 0}
    started = time.time()
    total_images = 0
    with db.session() as s:
        total_images = s.get(JobStage, js.id).total
    live = {"processed": 0, "computed": 0}

    def report(n_computed: int = 0, n_other: int = 0) -> None:
        live["computed"] += n_computed
        live["processed"] += n_computed + n_other
        elapsed = time.time() - started
        rate = live["computed"] / elapsed if live["computed"] and elapsed > 0 else None
        ctx.progress(stage.name, {"processed": live["processed"], "total": total_images, "computed": live["computed"],
                                  "elapsed_s": elapsed, "rate_per_s": rate,
                                  "eta_s": (total_images - live["processed"]) / rate if rate else None})
    setup_done = False
    bs = stage.batch_size(ctx) if stage.batched else 1
    last_pk = 0
    pool = None if stage.batched else ThreadPoolExecutor(max_workers=cfg.processing.cpu_workers)
    try:
        while True:
            if ctx.cancel():
                raise Cancelled()
            with db.session() as s:
                imgs = list(s.scalars(base_filter(select(Image)).where(Image.id > last_pk).order_by(Image.id).limit(chunk)))
                if not imgs:
                    break
                last_pk = imgs[-1].id
                pks = [im.id for im in imgs]
                results: dict[tuple[int, str], StageResult] = {
                    (r.image_pk, r.stage): r for r in s.scalars(
                        select(StageResult).where(StageResult.image_pk.in_(pks), StageResult.stage.in_((stage.name,) + dep_names)))}
                work: list[tuple[Image, ImgRef, str, Optional[StageResult]]] = []
                for im in imgs:
                    if not stage.eligible(ctx, im):
                        counts["ineligible"] += 1
                        continue
                    deps, dep_keys, blocked = {}, {}, False
                    for d in dep_names:
                        r = results.get((im.id, d))
                        ok = r is not None and r.status == "done"
                        if d in stage.depends and not ok:
                            blocked = True
                            break
                        dep_keys[d] = r.idem_key if ok else None
                        if ok:
                            deps[d] = r.data or {}
                    if blocked:
                        counts["blocked"] += 1
                        continue
                    key = idem_key(stage, cfg_fp, model, im.sha256, dep_keys)
                    prev = results.get((im.id, stage.name))
                    if prev is not None and prev.idem_key == key:
                        if prev.status == "done" and stage.still_valid(ctx, im, prev.data or {}):
                            stage.apply(im, prev.data or {})
                            counts["reused"] += 1
                            continue
                        if prev.status == "failed" and not ctx.retry_failed and (
                                prev.attempts >= max_attempts or prev.error_category not in RETRYABLE):
                            counts["failed"] += 1
                            continue
                    ref = ImgRef(pk=im.id, image_id=im.image_id, abs_path=im.abs_path, sha256=im.sha256,
                                 extension=im.extension, file_name=im.file_name, relative_path=im.relative_path,
                                 width=im.width, height=im.height, supplied_caption=im.supplied_caption, deps=deps)
                    work.append((im, ref, key, prev))

                report(0, len(imgs) - len(work))
                if work:
                    if not setup_done:
                        stage.setup(ctx)
                        setup_done = True
                    refs = [w[1] for w in work]
                    t0 = time.time()
                    tag = stage.label or stage.name.upper()

                    def emit(ref, out, secs):
                        if isinstance(out, Exception):
                            ctx.item(tag, ref.image_id, False, f"{type(out).__name__}: {out}"[:160], secs)
                        else:
                            try:
                                msg = stage.describe(out)
                            except Exception:
                                msg = ""
                            ctx.item(tag, ref.image_id, True, msg, secs)

                    outputs, bs = _execute(ctx, stage, refs, pool, bs, max_attempts, report, emit)
                    per_ms = (time.time() - t0) * 1000 / max(1, len(refs))
                    for (im, ref, key, prev), (out, attempts) in zip(work, outputs):
                        row = prev or StageResult(image_pk=im.id, stage=stage.name)
                        if prev is None:
                            s.add(row)
                        row.idem_key, row.model_info, row.job_id, row.duration_ms = key, model, ctx.job_id, per_ms
                        row.updated_at = utcnow()
                        same_key_before = prev is not None and prev.idem_key == key
                        if isinstance(out, Exception):
                            row.status, row.data = "failed", None
                            row.error = f"{type(out).__name__}: {out}"[:500]
                            row.error_category = categorize_error(out)
                            row.attempts = (row.attempts or 0) + attempts if same_key_before else attempts
                            counts["failed"] += 1
                            log.warning("%s failed for %s [%s]: %s", stage.name, im.image_id, row.error_category, row.error)
                        else:
                            row.status, row.data, row.error, row.error_category = "done", out, None, None
                            row.attempts = attempts
                            stage.apply(im, out)
                            counts["done"] += 1
                js_db = s.get(JobStage, js.id)
                js_db.done, js_db.reused, js_db.failed = counts["done"], counts["reused"], counts["failed"]
                elapsed = time.time() - started
                js_db.info = {**counts, "elapsed_s": round(elapsed, 2),
                              "rate_per_s": round(counts["done"] / elapsed, 3) if counts["done"] and elapsed > 0 else None,
                              "batch_size": bs if stage.batched else None}
            ctx.heartbeat()
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if setup_done:
            stage.teardown(ctx)


def _execute(ctx, stage, refs, pool, bs, max_attempts, report=lambda n: None, emit=lambda ref, out, secs: None):
    """Returns ([(output_or_exception, attempts)], batch_size). Retries transient failures and
    halves the batch size after out-of-memory errors. ``emit`` is called once per image with
    its final outcome, as soon as it is known."""
    outputs: list[Any] = [None] * len(refs)
    attempts = [0] * len(refs)
    pending = list(range(len(refs)))

    def settle(j, r, secs, attempt):
        outputs[j] = r
        attempts[j] = attempt
        final = not isinstance(r, Exception) or attempt >= max_attempts or categorize_error(r) not in RETRYABLE
        if final:
            emit(refs[j], r, secs)

    def timed(k):
        t = time.time()
        r = stage._safe(ctx, refs[k])
        return r, time.time() - t

    for attempt in range(1, max_attempts + 1):
        if not pending:
            break
        if stage.batched:
            i = 0
            while i < len(pending):
                if ctx.cancel():
                    raise Cancelled()
                idxs = pending[i:i + bs]
                t0 = time.time()
                try:
                    res = stage.process_batch(ctx, [refs[j] for j in idxs])
                except Exception as exc:
                    if is_oom(exc) and bs > 1:
                        bs = max(1, bs // 2)
                        log.warning("%s: out of memory, batch size reduced to %d", stage.name, bs)
                        _free_memory()
                        continue  # retry the same slice with a smaller batch
                    res = [exc] * len(idxs)
                per = (time.time() - t0) / max(1, len(idxs))
                for j, r in zip(idxs, res):
                    settle(j, r, per, attempt)
                i += len(idxs)
                if attempt == 1:
                    report(len(idxs))
        else:
            for j, (r, secs) in zip(pending, pool.map(timed, pending)):
                settle(j, r, secs, attempt)
                if attempt == 1:
                    report(1)
        pending = [j for j in pending if isinstance(outputs[j], Exception) and categorize_error(outputs[j]) in RETRYABLE]
        if pending and attempt < max_attempts:
            time.sleep(min(0.2 * attempt, 1.0))
    return list(zip(outputs, attempts)), bs


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
