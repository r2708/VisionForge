"""Command-line interface: ``image-pipeline``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from sqlalchemy import select

from app import PIPELINE_VERSION
from app.config.settings import PipelineConfig, SourceCfg
from app.core import hardware
from app.core.workspace import Workspace, default_workspace_root

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Image dataset pipeline: run (process a folder / pick up new images), resume (continue after a stop), "
                       "status (checkpoint tables). Source folders are always read-only.")
models_app = typer.Typer(no_args_is_help=True, help="Model registry (nothing is downloaded without --yes).")
sources_app = typer.Typer(no_args_is_help=True, help="Source folders and their rights information.")
app.add_typer(models_app, name="models", hidden=True)
app.add_typer(sources_app, name="sources", hidden=True)

WS = typer.Option(None, "--workspace", "-w", help="Workspace directory (default: $IMAGE_PIPELINE_WORKSPACE or current directory).")
CFG = typer.Option(None, "--config", "-c", help="Pipeline YAML (default: <workspace>/configs/pipeline_config.yaml).")


def _ws(workspace: Optional[Path], config: Optional[Path] = None) -> Workspace:
    try:
        return Workspace(workspace or default_workspace_root(), config)
    except Exception as exc:
        typer.secho(f"Configuration error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)


def _print(obj) -> None:
    typer.echo(json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def _fail(msg: str, code: int = 1):
    typer.secho(msg, fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


@app.command("system-info", hidden=True)
def system_info(workspace: Optional[Path] = WS):
    """Inspect hardware, runtimes and model availability."""
    from app.models.hub import ModelHub

    ws = _ws(workspace)
    info = hardware.system_info(ws.output_dir)
    info["pipeline_version"] = PIPELINE_VERSION
    info["models"] = [{k: m[k] for k in ("key", "task", "installed", "usable", "download_size_gb", "ram_gb", "license", "reason")}
                      for m in ModelHub(ws.config, ws.root).describe()]
    try:
        dev = hardware.select_device(ws.config.processing.device, ws.config.processing.allow_cpu_fallback)
        info["selected_device"] = dev
        info["auto_batch_size"] = hardware.auto_batch_size(dev)
        if dev == "cpu":
            info["warning"] = "No GPU runtime selected: model stages will run on CPU and be slow."
    except Exception as exc:
        info["selected_device"] = None
        info["warning"] = str(exc)
    _print(info)


@app.command(hidden=True)
def init(workspace: Optional[Path] = WS, mode: str = typer.Option("text_to_image", help="text_to_image | vision_language | general_multimodal")):
    """Create a workspace with a default configuration file."""
    ws = _ws(workspace)
    if ws.config_path.exists():
        typer.echo(f"Configuration already exists: {ws.config_path}")
        return
    cfg = PipelineConfig()
    try:
        cfg.project.mode = mode  # validated on assignment
    except Exception as exc:
        _fail(f"Invalid mode: {exc}", 2)
    ws.save_config(cfg)
    typer.echo(f"Workspace ready. Configuration written to {ws.config_path}")


@app.command(hidden=True)
def scan(source: list[Path] = typer.Option(None, "--source", "-s", help="Image folder to add (repeatable)."),
         output: Optional[Path] = typer.Option(None, "--output", "-o", help="Workspace directory (same as --workspace)."),
         workspace: Optional[Path] = WS, config: Optional[Path] = CFG,
         label: Optional[str] = typer.Option(None, help="Stable label for the added source (single --source only)."),
         rights_status: Optional[str] = typer.Option(None, help="Rights status you attest for the added source(s)."),
         license: Optional[str] = typer.Option(None, help="Licence of the added source(s).")):
    """Register source folders (read-only) and scan them. Safe to repeat; resumes after interruption."""
    from app.pipeline.ingestion.scanner import scan_all

    ws = _ws(output or workspace, config)
    if source:
        cfg = ws.config.model_copy(deep=True)
        known = {str(Path(s.path).expanduser().resolve()) for s in cfg.paths.source_cfgs()}
        for p in source:
            try:
                resolved = ws.validate_source_dir(str(p))
                if str(resolved) not in known:
                    cfg.paths.sources = list(cfg.paths.sources) + [SourceCfg(
                        path=str(resolved), label=label if len(source) == 1 else None, rights_status=rights_status, license=license)]
            except Exception as exc:
                _fail(f"{p}: {exc}", 2)
        ws.save_config(PipelineConfig.model_validate(cfg.model_dump()))
    if not ws.config.paths.sources:
        _fail("No sources configured. Use --source PATH or add paths.sources to the configuration.", 2)
    try:
        out = scan_all(ws, progress=lambda st: typer.echo(f"\r  {st['label']}: {st['files_seen']} files seen, {st['new']} new", nl=False))
    except Exception as exc:
        _fail(f"\nScan failed: {exc}")
    typer.echo("")
    _print({"totals": out["totals"], "sources": [{k: s[k] for k in ("source_id", "label", "new", "unchanged", "changed", "unsupported",
                                                                     "unreadable", "missing", "unsupported_by_extension")}
                                                 for s in out["sources"]]})


DATASET_LABELS = {"scan": "SCAN", "dedup": "DEDUP", "curation_pre": "FILTER", "caption_diversity": "DIVERSITY",
                  "curation": "CURATE", "reconcile_outputs": "SYNC", "splits": "SPLIT", "export": "EXPORT",
                  "hf_upload": "UPLOAD"}


def _fmt_seconds(seconds) -> str:
    """07:03 / 1:07:03 style, '--:--' when unknown."""
    if seconds is None:
        return "--:--"
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _stage_meta() -> dict:
    from app.pipeline.runner import build_pipeline

    return {st.name: {"label": getattr(st, "label", "") or DATASET_LABELS.get(st.name, st.name.upper()),
                      "title": st.title, "image": hasattr(st, "process"), "obj": st} for st in build_pipeline()}


class _Console:
    """Terminal output for a run: one line per image and stage as it finishes
    (``[STAGE] image_id ✓ what was found (time)``), a throttled progress line with running
    totals and an ETA, one closing line per stage, and a summary table at the end."""

    def __init__(self, show_items: bool = True, every_seconds: float = 5.0):
        import threading
        import time

        from rich.console import Console

        from app.core.logging import get_logger

        self.time, self.out = time, Console(highlight=False, soft_wrap=True)
        self.meta = _stage_meta()
        for extra, (lab, title) in {"download": ("DOWNLOAD", "Download images from the URL lists"),
                                    "shards": ("SHARD", "Shards"), "upload": ("UPLOAD", "Upload shard")}.items():
            self.meta[extra] = {"label": lab, "title": title, "image": False, "obj": None}
        self.by_label = {m["label"]: n for n, m in self.meta.items()}
        self.show_items, self.every = show_items, every_seconds
        self.last = 0.0
        self.failed: dict[str, int] = {}
        self.started: dict[str, float] = {}
        self.rows: list[tuple] = []
        self.lock = threading.Lock()  # item lines arrive from worker threads
        self.log = get_logger("run")

    def _label(self, name: str) -> str:
        return self.meta.get(name, {}).get("label", name.upper())

    def _print(self, text: str, plain: str) -> None:
        with self.lock:
            self.out.print(text)
        self.log.info(plain)

    # ---- callbacks -------------------------------------------------------
    def stage(self, name: str, state: str) -> None:
        if state == "running":
            self.started[name] = self.time.time()
            self.last = self.time.time()
            self.failed[name] = 0
            self._print(f"[bold cyan]=== {self._label(name)}[/bold cyan]  {self.meta[name]['title']}",
                        f"=== {self._label(name)} {self.meta[name]['title']}")

    def item(self, label: str, image_id: str, ok: bool, message: str, seconds: float) -> None:
        if not ok:
            name = self.by_label.get(label, "")
            self.failed[name] = self.failed.get(name, 0) + 1
        plain = f"[{label}] {image_id} {'✓' if ok else '✗'} {message} ({seconds:.2f}s)"
        if self.show_items or not ok:
            mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
            if ok and message.startswith("INVALID"):
                mark = "[yellow]![/yellow]"  # the check worked; the file itself is unusable
            from rich.markup import escape

            with self.lock:
                self.out.print(f"\\[{label}] {image_id} {mark} {escape(message)} [dim]({seconds:.2f}s)[/dim]")
        self.log.info(plain)  # file log only; the terminal line is printed above

    def progress(self, name: str, info: dict) -> None:
        label = self._label(name)
        if "message" in info:
            self._print(f"\\[{label}] {info['message']}", f"[{label}] {info['message']}")
            return
        now = self.time.time()
        if name == "scan":
            if now - self.last >= 1.0:
                self.last = now
                text = f"{info['source']}: {info['files_seen']} files seen, {info['new']} new, {info['unchanged']} unchanged, {info['unsupported']} unsupported"
                self._print(f"\\[{label}] {text}", f"[{label}] {text}")
            return
        done, total = info["processed"], info["total"]
        if done == 0 or (now - self.last < self.every and done < total):
            return
        self.last = now
        rate = f"  {info['rate_per_s']:.1f} img/s" if info["rate_per_s"] else ""
        text = (f"[{name} {done}/{total}] failed={self.failed.get(name, 0)}  elapsed={_fmt_seconds(info['elapsed_s'])}  "
                f"eta={_fmt_seconds(info['eta_s']) if done < total else '00:00'}{rate}")
        self._print(f"[bold]\\{text}[/bold]", text)

    def end(self, name: str, st: dict) -> None:
        label = self._label(name)
        took = self.time.time() - self.started[name] if name in self.started else 0.0
        if st["status"] == "skipped":
            self._print(f"[dim]\\[{label}] - skipped: {st['reason']}[/dim]", f"[{label}] - skipped: {st['reason']}")
            self.rows.append((name, "[dim]skipped[/dim]", "-", "-", "-", "-", st["reason"] or ""))
            return
        if st["status"] != "completed":
            self._print(f"\\[{label}] [red]✗ {st['status']}[/red] {st['reason'] or ''}", f"[{label}] ✗ {st['status']} {st['reason'] or ''}")
            self.rows.append((name, f"[red]{st['status']}[/red]", str(st["done"]), str(st["reused"]), str(st["failed"]),
                              _fmt_seconds(took), (st["reason"] or "")[:60]))
            return
        if st["kind"] == "image":
            info = st["info"]
            na = info.get("ineligible", 0) + info.get("blocked", 0)
            text = (f"{st['total']} images: {st['done']} processed, {st['reused']} cached, {st['failed']} failed"
                    + (f", {na} not applicable" if na else ""))
            ok = st["failed"] == 0
            self.rows.append((name, "[green]ok[/green]" if ok else f"[red]{st['failed']} failed[/red]", str(st["done"]),
                              str(st["reused"]), str(st["failed"]), _fmt_seconds(took), ""))
        else:
            text, ok = self._summary(st["info"]), True
            self.rows.append((name, "[green]ok[/green]", "-", "-", "-", _fmt_seconds(took), text[:60]))
        mark = "[green]✓[/green]" if ok else "[red]✗[/red]"
        from rich.markup import escape

        with self.lock:
            self.out.print(f"\\[{label}] {mark} {escape(text)} [dim]({took:.1f}s)[/dim]")
        self.log.info(f"[{label}] {'✓' if ok else '✗'} {text} ({took:.1f}s)")

    def summary_table(self, elapsed: float) -> None:
        from rich.table import Table

        t = Table(title=f"Run summary ({_fmt_seconds(elapsed)})")
        for col, just in (("stage", "left"), ("result", "left"), ("processed", "right"), ("cached", "right"),
                          ("failed", "right"), ("time", "right"), ("details", "left")):
            t.add_column(col, justify=just, no_wrap=col != "details", overflow="ellipsis", max_width=60 if col == "details" else None)
        for row in self.rows:
            t.add_row(*row)
        self.out.print(t)

    @staticmethod
    def _summary(info: dict) -> str:
        parts = []
        for k, v in info.items():
            if isinstance(v, dict) and v and all(isinstance(x, (int, float)) for x in v.values()):
                parts.append(f"{k}=" + "/".join(f"{a}:{b}" for a, b in v.items()))
            elif isinstance(v, (int, float, str, bool)) and k not in ("generated_at", "final"):
                parts.append(f"{k}={v}")
            elif isinstance(v, list) and k == "warnings":
                parts.append(f"warnings={len(v)}")
        return " ".join(parts)[:260]


def _run(ws: Workspace, job_id: str, retry_failed: bool, quiet: bool = False) -> None:
    import signal
    import time

    from sqlalchemy import func

    from app.database.models import Image
    from app.pipeline.runner import AlreadyRunning, run_job

    def _stop(signum, frame):  # `kill <pid>` behaves like Ctrl+C: stop cleanly and stay resumable
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
    except (ValueError, OSError):
        pass
    t0 = time.time()
    con = _Console(show_items=not quiet)
    con.out.print(f"[bold]job {job_id}[/bold]  mode={ws.config.project.mode}  workers={ws.config.processing.cpu_workers}  "
                  f"log=logs/pipeline.log")
    try:
        summary = run_job(ws, job_id, retry_failed=retry_failed, on_stage=con.stage, on_progress=con.progress,
                          on_stage_end=con.end, on_item=con.item)
    except AlreadyRunning as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(3)
    con.summary_table(time.time() - t0)
    with ws.db.session() as s:
        total = s.scalar(select(func.count(Image.id)).where(Image.missing.is_(False))) or 0
        in_ds = s.scalar(select(func.count(Image.id)).where(Image.missing.is_(False), Image.processed_path.is_not(None))) or 0
    failed_images = sum(st["failed"] for st in summary["stages"].values())
    color = "green" if summary["status"] == "completed" else "red"
    con.out.print(f"[{color}]{summary['status']}[/{color}]: {in_ds}/{total} images in the dataset"
                  + (f", [red]{failed_images} stage failure(s)[/red]" if failed_images else "")
                  + (f"  ({summary['error']})" if summary["error"] else ""))
    if summary["status"] == "completed":
        from app.database.session import kv_get

        with ws.db.session() as s:
            exp = kv_get(s, "export:last", None)
        if exp:
            for w in exp["warnings"]:
                con.out.print(f"[yellow]warning:[/yellow] {w}")
            con.out.print(f"files: {ws.output_dir}")
        if failed_images:
            con.out.print("retry the failed ones with: image-pipeline resume --retry-failed")
            raise typer.Exit(2)
    else:
        con.out.print("[yellow]finished work is saved; continue with:[/yellow] image-pipeline resume")
        raise typer.Exit(1)


def _add_sources(ws: Workspace, source, label, rights_status, license) -> None:
    """Add new source folders to the configuration and/or record rights for the configured ones."""
    if not source and not rights_status and not license:
        return
    cfg = ws.config.model_copy(deep=True)
    current = cfg.paths.source_cfgs()
    known = {str(Path(s.path).expanduser().resolve()) for s in current}
    for p in source or []:
        try:
            resolved = ws.validate_source_dir(str(p))
        except Exception as exc:
            _fail(f"{p}: {exc}", 2)
        if str(resolved) not in known:
            current.append(SourceCfg(path=str(resolved), label=label if len(source) == 1 else None))
            known.add(str(resolved))
    for sc in current:  # rights given on the command line apply to sources that have none recorded yet
        if rights_status and sc.rights_status is None:
            sc.rights_status = rights_status
        if license and sc.license is None:
            sc.license = license
    cfg.paths.sources = current
    try:
        ws.save_config(PipelineConfig.model_validate(cfg.model_dump()))
    except Exception as exc:
        _fail(f"Invalid value: {exc}", 2)


def _with_workers(cfg: PipelineConfig, workers: Optional[int]) -> PipelineConfig:
    if not workers:
        return cfg
    data = cfg.model_dump(mode="json")
    data["processing"]["cpu_workers"] = workers
    try:
        return PipelineConfig.model_validate(data)
    except Exception as exc:
        _fail(f"--workers: {exc}", 2)


def _shard_state(ws: Workspace) -> Optional[dict]:
    from app.database.session import kv_get
    from app.services.shards import STATE_KEY

    with ws.db.session() as s:
        return kv_get(s, STATE_KEY, None)


def _run_shards(ws: Workspace, source: Optional[str], workers: Optional[int], rights_status, license,
                max_urls: Optional[int], retry_failed: bool, quiet: bool) -> None:
    """Shard mode: URL lists in, uploaded shards out."""
    import signal
    import time

    from app.pipeline.runner import AlreadyRunning
    from app.services.shards import ShardRunner

    def _stop(signum, frame):
        raise KeyboardInterrupt()

    try:
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
    except (ValueError, OSError):
        pass
    if workers:
        ws.save_config(_with_workers(ws.config, workers))
    con = _Console(show_items=not quiet)
    runner = ShardRunner(ws, con, quiet)
    try:
        state = runner.init_state(source or _shard_state(ws)["source"], rights_status, license)
    except Exception as exc:
        _fail(str(exc), 2)
    if max_urls:
        state["limit_index"] = state["index"] + max_urls
    elif source is not None and state.get("limit_index") is not None and state["index"] >= state["limit_index"]:
        state["limit_index"] = None  # `run FOLDER` without a limit continues through the whole list
    runner.save(state)
    hf = ws.config.huggingface
    goal = state["limit_index"] or state["total_urls"]
    con.out.print(f"[bold]shard mode[/bold]  source {state['source'] if state.get('kind') == 'hf' else Path(state['source']).name}  |  "
                  f"{state['index']}/{state['total_urls']} consumed, this run goes to {goal}  |  "
                  f"shard size {runner.target_mb():.0f} MB  |  next shard {runner.shard_name(state['shard'])}  |  "
                  + (f"upload to {hf.repo_id} ({'private' if hf.private else 'PUBLIC'})" if hf.enabled and hf.repo_id
                     else "Hugging Face upload OFF: shards stay in workspace/shards"))
    t0 = time.time()
    status, error, result = "completed", None, {}
    try:
        result = runner.run(state, retry_failed)
        status, error = result["status"], result["error"]
    except AlreadyRunning as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(3)
    except KeyboardInterrupt:
        status, error = "interrupted", "stopped by the user"
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    state = runner.load() or state
    from rich.table import Table

    t = Table(title=f"Shard run summary ({_fmt_seconds(time.time() - t0)})")
    t.add_column("item"), t.add_column("value", justify="right")
    for k, v in (("source", state["source"] if state.get("kind") == "hf" else Path(state["source"]).name),
                 ("images/URLs consumed", f"{state['index']}/{state['total_urls']}"), ("taken into shards", state["downloaded"]),
                 ("unsupported format, skipped", state.get("unsupported", 0)),
                 ("no longer available", state["gone"]), ("download failed", state["failed"]),
                 ("duplicates skipped", state["duplicates"]), ("too large, skipped", state["too_large"]),
                 ("shards uploaded", len(state["shards_done"])), ("records uploaded", state["records_uploaded"]),
                 ("shards waiting for upload", len(state["pending_upload"])),
                 ("open shard", f"{result.get('open_shard_mb', '?')} / {runner.target_mb():.0f} MB")):
        t.add_row(k, str(v))
    con.out.print(t)
    color = "green" if status == "completed" else "red"
    con.out.print(f"[{color}]{status}[/{color}]" + (f": {error}" if error else ""))
    if status == "completed":
        if result.get("open_shard_mb"):
            con.out.print(f"the open shard has {result['open_shard_mb']} MB; it is uploaded once it reaches {runner.target_mb():.0f} MB "
                          "(run again with more URLs)")
    else:
        con.out.print("[yellow]finished work is saved; continue with:[/yellow] image-pipeline resume")
        raise typer.Exit(1)


WORKERS = typer.Option(None, "--workers", "-j", help="Parallel workers for the CPU stages (default: processing.cpu_workers).")


@app.command()
def run(source: list[str] = typer.Argument(None, help="Image folder(s), a folder of URL lists, or a Hugging Face dataset (hf://org/name or its URL). "
                                                    "Needed the first time; afterwards just `run`."),
        workers: Optional[int] = WORKERS,
        rights_status: Optional[str] = typer.Option(None, "--rights-status", help="Rights you hold: verified | user_attested | unknown | restricted | not_for_training."),
        license: Optional[str] = typer.Option(None, help="Licence of the images, e.g. \"CC-BY-4.0\" or \"own work\"."),
        retry_failed: bool = typer.Option(False, "--retry-failed", help="Also retry images that failed in earlier runs."),
        quiet: bool = typer.Option(False, "--quiet", "-q", help="No per-image lines (progress and stage lines only)."),
        max_urls: Optional[int] = typer.Option(None, "--max-images", "--max-urls", min=1, help="Shard mode only: stop after this many more images/URLs."),
        workspace: Optional[Path] = WS, config: Optional[Path] = CFG):
    """Run the whole pipeline. Run it again whenever you add images: finished work is
    reused and only new or changed images are processed.

    If the folder holds URL lists (.tsv) instead of images, shard mode is used: download,
    process, upload each shard to Hugging Face at the size threshold, clean up, continue."""
    from app.pipeline.runner import AlreadyRunning, create_job
    from app.services.shards import is_url_list_dir, parse_hf_source

    ws = _ws(workspace, config)
    source = [str(p) for p in (source or [])]
    remote = [p for p in source if parse_hf_source(p)] or [p for p in source if is_url_list_dir(Path(p))]
    if remote or (not source and _shard_state(ws)):
        if len(source) > 1:
            _fail("shard mode takes exactly one source (a URL-list folder or one Hugging Face dataset)", 2)
        _run_shards(ws, remote[0] if remote else None, workers, rights_status, license, max_urls, retry_failed, quiet)
        return
    source = [Path(p) for p in source]
    _add_sources(ws, source, None, rights_status, license)
    if not ws.config.paths.sources:
        _fail("No image folder yet. First time:  image-pipeline run /path/to/images --rights-status user_attested", 2)
    if all(sc.rights_status in (None, "unknown") for sc in ws.config.paths.source_cfgs()) and ws.config.rights.require_approved_status_for_export:
        typer.secho("note: no rights status is recorded for your images, so they will be analysed but NOT exported.\n"
                    "      if you hold the rights:  image-pipeline run --rights-status user_attested", fg=typer.colors.YELLOW)
    try:
        job_id = create_job(ws, None, _with_workers(ws.config, workers))
    except AlreadyRunning as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(3)
    except Exception as exc:
        _fail(str(exc), 2)
    _run(ws, job_id, retry_failed, quiet)


def _resume(ws: Workspace, job_id: Optional[str], workers: Optional[int], retry_failed: bool, quiet: bool) -> None:
    from app.core.runlock import RunLock
    from app.database.models import Job
    from app.pipeline.runner import mark_interrupted

    owner = RunLock(ws.checkpoints_dir).owner()
    if owner is not None:
        typer.secho(f"a pipeline process (pid {owner}) is still running for this workspace; "
                    "watch it with: image-pipeline status --watch 5", fg=typer.colors.RED, err=True)
        raise typer.Exit(3)
    recovered = mark_interrupted(ws)  # stages left RUNNING by a killed process become interrupted
    with ws.db.session() as s:
        job = s.get(Job, job_id) if job_id else s.scalar(select(Job).order_by(Job.created_at.desc()).limit(1))
        if job is None:
            _fail("No job to resume. Start one with: image-pipeline run /path/to/images" if not job_id else f"Job {job_id} not found", 2)
        jid, st = job.id, job.status
        if workers:
            cfg = _with_workers(PipelineConfig.model_validate(job.config), workers)
            job.config = cfg.model_dump(mode="json")
        stages_before = {js.stage: js.status for js in job.stage_runs}
    if st == "completed" and not retry_failed:
        typer.echo(f"Nothing to resume: the last job ({jid}) completed. To process new images: image-pipeline run")
        return
    stopped_at = next((k for k, v in stages_before.items() if v in ("interrupted", "failed", "cancelled", "running")), None) \
        or next((k for k, v in stages_before.items() if v == "pending"), None)
    typer.echo(f"resuming job {jid} (was {st}" + (f", stopped at {stopped_at}" if stopped_at else "")
               + (", recovered after a crash" if recovered else "") + "): finished work is reused")
    _run(ws, jid, retry_failed, quiet)


@app.command()
def resume(workers: Optional[int] = WORKERS,
           retry_failed: bool = typer.Option(False, "--retry-failed", help="Also retry images that failed earlier."),
           quiet: bool = typer.Option(False, "--quiet", "-q", help="No per-image lines."),
           job_id: Optional[str] = typer.Option(None, "--job-id", help="Default: the last job."),
           workspace: Optional[Path] = WS):
    """Continue after a crash, shutdown, Ctrl+C or error: finished images and stages are
    skipped, the interrupted stage and everything after it run."""
    ws = _ws(workspace)
    if _shard_state(ws) and not job_id:
        from app.core.runlock import RunLock

        owner = RunLock(ws.checkpoints_dir).owner()
        if owner is not None:
            typer.secho(f"a pipeline process (pid {owner}) is still running; watch it with: image-pipeline status --watch 5",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(3)
        st = _shard_state(ws)
        typer.echo(f"resuming shard mode ({st['source']}) at image {st['index'] + 1}, {'shard_%04d' % st['shard']}: finished work is reused")
        _run_shards(ws, None, workers, None, None, None, retry_failed, quiet)
        return
    _resume(ws, job_id, workers, retry_failed, quiet)


@app.command("retry-failed", hidden=True)
def retry_failed_cmd(workers: Optional[int] = WORKERS, workspace: Optional[Path] = WS):
    """Re-run the images that failed (same as: resume --retry-failed)."""
    _resume(_ws(workspace), None, workers, True, False)


@app.command(hidden=True)
def start(source: list[Path] = typer.Option(None, "--source", "-s"), rights_status: Optional[str] = typer.Option(None, "--rights-status"),
          license: Optional[str] = typer.Option(None), workspace: Optional[Path] = WS, config: Optional[Path] = CFG,
          retry_failed: bool = typer.Option(False), quiet: bool = typer.Option(False, "--quiet", "-q")):
    """Old name for `run`."""
    run(source=[str(p) for p in (source or [])], max_urls=None, workers=None, rights_status=rights_status, license=license, retry_failed=retry_failed,
        quiet=quiet, workspace=workspace, config=config)


@app.command(hidden=True)
def cancel(job_id: Optional[str] = typer.Option(None, "--job-id"), workspace: Optional[Path] = WS):
    """Ask the running job to stop safely after the current batch."""
    from app.pipeline.runner import active_job, request_cancel

    ws = _ws(workspace)
    job_id = job_id or active_job(ws)
    ok = bool(job_id) and request_cancel(ws, job_id)
    typer.echo("Cancellation requested; resume later with: image-pipeline resume" if ok else "No job is running.")


# ------------------------------------------------------------------ status
MARKS = {"DONE": "[green]✓[/green]", "FAILED": "[red]✗[/red]", "RUNNING": "[yellow]…[/yellow]", "SKIPPED": "[cyan]-[/cyan]", "PENDING": "·"}


def _status_data(ws: Workspace, image_id: Optional[str], limit: int, only_failed: bool) -> dict:
    """Checkpoint rows: one per image with the status of every per-image stage."""
    from sqlalchemy import func

    from app.core.runlock import RunLock
    from app.database.models import Image, Job, StageResult
    from app.pipeline.runner import mark_interrupted
    from app.services.stats import dashboard

    mark_interrupted(ws)
    meta = _stage_meta()
    image_stages = [n for n, m in meta.items() if m["image"]]
    d = dashboard(ws)
    job = d["job"]
    js = {s["stage"]: s for s in (job["stages"] if job else [])}
    live = RunLock(ws.checkpoints_dir).owner()
    rows = []
    with ws.db.session() as s:
        failed_pks = select(StageResult.image_pk).where(StageResult.status == "failed")
        q = select(Image).where(Image.missing.is_(False))
        if image_id:
            q = q.where(Image.image_id == image_id)
        if only_failed:
            q = q.where(Image.id.in_(failed_pks))
        shown_of = s.scalar(select(func.count()).select_from(q.subquery())) or 0
        # failed images first, then the most recently added
        q = q.order_by(Image.id.in_(failed_pks).desc(), Image.id.desc()).limit(1 if image_id else limit)
        for im in s.scalars(q):
            res = {r.stage: r for r in im.results}
            stages, metrics, errors = {}, {}, {}
            for name in image_stages:
                r = res.get(name)
                run_state = (js.get(name) or {}).get("status")
                if r is not None and r.status == "done":
                    stages[name] = "DONE"
                    try:
                        metrics[name] = meta[name]["obj"].describe(r.data or {})
                    except Exception:
                        metrics[name] = ""
                elif r is not None:
                    stages[name], errors[name] = "FAILED", f"{r.error} (attempts: {r.attempts})"
                elif run_state == "skipped":
                    stages[name], metrics[name] = "SKIPPED", (js[name].get("reason") or "")
                elif run_state == "running" and live:
                    stages[name] = "RUNNING"
                elif im.valid is False and name != "validate":
                    stages[name], metrics[name] = "SKIPPED", "invalid file"
                else:
                    stages[name] = "PENDING"
            state = ("failed" if "FAILED" in stages.values() else "invalid" if im.valid is False
                     else "done" if im.processed_path else im.decision if im.decision in ("rejected", "quarantined")
                     else "processing" if "RUNNING" in stages.values() else "pending")
            rows.append({"image_id": im.image_id, "file": im.relative_path, "status": state, "decision": im.decision,
                         "reasons": im.decision_reasons or [], "split": im.split, "stages": stages, "metrics": metrics, "errors": errors})
        by_status = dict(s.execute(select(Image.decision, func.count(Image.id)).where(Image.missing.is_(False)).group_by(Image.decision)).all())
    return {"rows": rows, "matching": shown_of, "dashboard": d, "image_stages": image_stages, "meta": meta,
            "by_decision": by_status, "live_pid": live}


def _shard_tables(ws: Workspace) -> list:
    """Shard-mode overview (empty list when the workspace is a normal folder run)."""
    from rich.table import Table

    from app.services.hf_upload import upload_size

    st = _shard_state(ws)
    if not st:
        return []
    cfg = ws.config
    target = float(cfg.shards.target_mb or cfg.huggingface.min_size_mb)
    size_mb = upload_size(ws.output_dir)[0] / 1024**2 if ws.output_dir.is_dir() else 0.0
    goal = st["limit_index"] or st["total_urls"]
    t = Table(title="Shard mode")
    t.add_column("item"), t.add_column("value", justify="right"), t.add_column("notes")
    t.add_row("source", st["source"] if st.get("kind") == "hf" else Path(st["source"]).name,
              f"file {min(st['file_idx'] + 1, len(st['files']))} of {len(st['files'])}")
    if st.get("kind") == "hf":
        src_dir = ws.root / "downloads" / "_source"
        ready = sorted(f.name.split("__")[-1] for f in src_dir.glob("*.parquet")) if src_dir.is_dir() else []
        partial = [f.name.split("__")[-1].replace(".part", "") for f in src_dir.glob("*.part")] if src_dir.is_dir() else []
        t.add_row("source files on disk", f"{len(ready)} / {cfg.shards.prefetch_files} max",
                  ("ready: " + ", ".join(x.replace(".parquet", "") for x in ready) if ready else "none ready")
                  + (f"; downloading: {', '.join(x.replace('.parquet', '') for x in partial)}" if partial else ""))
    t.add_row("images consumed", f"{st['index']} / {st['total_urls']}", f"{100 * st['index'] / max(1, st['total_urls']):.3f}% of the source; current target {goal}")
    t.add_row("taken into shards", str(st["downloaded"]), f"unavailable {st['gone']}, failed {st['failed']}, duplicates {st['duplicates']}, "
                                                           f"too large {st['too_large']}, unsupported format {st.get('unsupported', 0)}")
    t.add_row("current shard", "shard_%04d" % st["shard"], f"{size_mb:.1f} / {target:.0f} MB ({100 * size_mb / target:.0f}%)")
    t.add_row("shards uploaded", str(len(st["shards_done"])), f"{st['records_uploaded']} records"
              + (f", last: {st['shards_done'][-1]['name']} at {st['shards_done'][-1].get('uploaded_at', '')[:19]}" if st["shards_done"] else ""))
    t.add_row("waiting for upload", str(len(st["pending_upload"])), ", ".join(e["name"] for e in st["pending_upload"]))
    return [t]


def _status_renderable(ws: Workspace, image_id: Optional[str], limit: int, only_failed: bool):
    from datetime import datetime, timezone

    from rich.console import Group
    from rich.markup import escape
    from rich.table import Table

    from app.services.hf_upload import upload_size

    data = _status_data(ws, image_id, limit, only_failed)
    rows, d, meta = data["rows"], data["dashboard"], data["meta"]
    if image_id:
        if not rows:
            return f"No image with id {image_id}."
        row = rows[0]
        t = Table(show_header=True, title=f"{row['image_id']}  {row['file']}  status={row['status']}  decision={row['decision']}"
                                           + (f"  split={row['split']}" if row["split"] else ""))
        t.add_column("stage"), t.add_column("status"), t.add_column("metrics / error")
        for name in data["image_stages"]:
            st = row["stages"][name]
            color = {"DONE": "green", "FAILED": "red", "RUNNING": "yellow", "SKIPPED": "cyan"}.get(st, "white")
            t.add_row(name.upper(), f"[{color}]{st}[/{color}]", escape((row["errors"].get(name) or row["metrics"].get(name) or "")[:110]))
        extra = f"reasons: {', '.join(row['reasons'])}" if row["reasons"] else ""
        return Group(t, extra) if extra else t
    if not d["images"]["discovered"]:
        shard = _shard_tables(ws)
        if shard:
            return Group(*shard, "No images in the current shard yet.")
        return "No images registered yet. Use `image-pipeline run /path/to/images`."

    # ---- run / stage table
    parts = []
    job = d["job"]
    if job:
        begun = datetime.fromisoformat(job["started_at"]) if job["started_at"] else None
        ended = datetime.fromisoformat(job["finished_at"]) if job["finished_at"] else None
        took = _fmt_seconds(((ended or datetime.now(timezone.utc)) - begun).total_seconds()) if begun else "--:--"
        color = {"completed": "green", "running": "yellow"}.get(job["status"], "red")
        hint = {"interrupted": "  -> image-pipeline resume", "cancelled": "  -> image-pipeline resume",
                "failed": "  -> image-pipeline resume"}.get(job["status"], "")
        t = Table(title=f"{job['id']}  status=[{color}]{job['status']}[/{color}]  elapsed={took}{hint}")
        for col, just in (("stage", "left"), ("status", "left"), ("progress", "right"), ("processed", "right"), ("cached", "right"),
                          ("failed", "right"), ("img/s", "right"), ("eta", "right"), ("metrics / error", "left")):
            t.add_column(col, justify=just, no_wrap=True, overflow="ellipsis", max_width=56 if col.startswith("metrics") else None)
        smap = {"completed": "DONE", "running": "RUNNING", "pending": "PENDING", "skipped": "SKIPPED", "failed": "FAILED",
                "cancelled": "FAILED", "interrupted": "FAILED"}
        colors = {"DONE": "green", "FAILED": "red", "RUNNING": "yellow", "SKIPPED": "cyan", "PENDING": "white"}
        for st in job["stages"]:
            info = st["info"] if isinstance(st["info"], dict) else {}
            state = smap.get(st["status"], st["status"].upper())
            shown = state if st["status"] not in ("cancelled", "interrupted") else st["status"].upper()
            handled = st["done"] + st["reused"] + st["failed"] + info.get("ineligible", 0) + info.get("blocked", 0)
            is_img = meta[st["stage"]]["image"] and st["total"]
            t.add_row(st["stage"].upper(), f"[{colors[state]}]{shown}[/{colors[state]}]",
                      f"{handled}/{st['total']} ({100 * handled / st['total']:.0f}%)" if is_img else "",
                      str(st["done"]) if is_img else "", str(st["reused"]) if is_img else "",
                      (f"[red]{st['failed']}[/red]" if st["failed"] else "0") if is_img else "",
                      f"{st['rate_per_s']:.1f}" if st["rate_per_s"] else "",
                      _fmt_seconds(st["eta_seconds"]) if st["eta_seconds"] is not None else "",
                      escape(st["reason"] or ("" if is_img else _Console._summary(info))))
        parts.append(t)
        if job["error"]:
            parts.append(f"[red]error:[/red] {escape(job['error'])}")

    # ---- checkpoint table: one row per image, one column per stage
    c = Table(title=f"{len(rows)} of {data['matching']} image(s)" + (" with failures" if only_failed else "")
                    + ("" if len(rows) == data["matching"] else f"  (more with --limit N)"))
    c.add_column("image_id", no_wrap=True), c.add_column("file", no_wrap=True, overflow="ellipsis", max_width=26), c.add_column("status", no_wrap=True)
    for name in data["image_stages"]:
        c.add_column(meta[name]["label"][:6], justify="center", no_wrap=True)
    for row in rows:
        color = {"done": "green", "failed": "red", "invalid": "red", "rejected": "red", "quarantined": "yellow", "processing": "yellow"}.get(row["status"], "white")
        c.add_row(row["image_id"], row["file"], f"[{color}]{row['status']}[/{color}]", *[MARKS[row["stages"][n]] for n in data["image_stages"]])
    parts.append(c)

    img, dec, dup = d["images"], d["decisions"], d["duplicates"]
    size_mb = upload_size(ws.output_dir)[0] / 1024**2
    hf = ws.config.huggingface
    running = sum(1 for s in (job["stages"] if job else []) if s["status"] == "running") if data["live_pid"] else 0
    parts.append(f"images by decision: {data['by_decision']}  |  {img['processed_renditions']}/{img['discovered']} in dataset, "
                 f"{running} stage(s) running" + (f" (pid {data['live_pid']})" if data["live_pid"] else ""))
    parts.append(f"valid {img['valid']}, invalid {img['invalid']}, new/unchecked {img['not_yet_validated']}  |  duplicates removed "
                 f"{dup['non_representative']}  |  awaiting review {d['review']['pending']}  |  failed (retryable) {d['failures']['retryable_images']}"
                 f"  |  splits {d['splits'] or '-'}")
    up = (f"Hugging Face: {hf.repo_id} ({'private' if hf.private else 'public'}), uploads at >= {hf.min_size_mb:.0f} MB"
          if hf.enabled and hf.repo_id else "Hugging Face upload: off")
    parts.append(f"output folder {size_mb:.1f} MB, free disk {d['disk']['output_free_gb']} GB  |  {up}")
    parts.append("[dim]✓ done   ✗ failed   … running   - skipped   · pending      details: image-pipeline status IMAGE_ID[/dim]")
    return Group(*( _shard_tables(ws) + parts))


@app.command()
def status(image_id: Optional[str] = typer.Argument(None, help="Show one image in detail."),
           as_json: bool = typer.Option(False, "--json", help="Machine readable output."),
           watch: Optional[float] = typer.Option(None, "--watch", "-w", help="Refresh every N seconds (monitor a running job from another terminal)."),
           limit: int = typer.Option(25, "--limit", "-n", min=1, max=5000, help="Rows in the per-image table."),
           failed: bool = typer.Option(False, "--failed", help="Only images with a failed stage."),
           workspace: Optional[Path] = typer.Option(None, "--workspace", help="Workspace directory.")):
    """Show the stage table of the last/current run and the per-image checkpoint table;
    --watch keeps it refreshing."""
    import time

    from rich.console import Console
    from rich.live import Live

    ws = _ws(workspace)
    console = Console()
    if as_json:
        data = _status_data(ws, image_id, limit, failed)
        console.print_json(json.dumps({"job": data["dashboard"]["job"], "counts": {k: data["dashboard"][k] for k in ("images", "decisions", "duplicates", "review", "failures", "splits")},
                                       "images": data["rows"], "matching": data["matching"], "running_pid": data["live_pid"]}, default=str))
        return
    if watch:
        interval = max(0.5, float(watch))
        with Live(_status_renderable(ws, image_id, limit, failed), console=console, refresh_per_second=4, screen=False) as live:
            try:
                while True:
                    time.sleep(interval)
                    live.update(_status_renderable(ws, image_id, limit, failed))
            except KeyboardInterrupt:
                pass
        return
    console.print(_status_renderable(ws, image_id, limit, failed))


@app.command(hidden=True)
def jobs(workspace: Optional[Path] = WS):
    """List jobs."""
    from app.database.models import Job

    ws = _ws(workspace)
    with ws.db.session() as s:
        for j in s.scalars(select(Job).order_by(Job.created_at.desc()).limit(30)):
            typer.echo(f"{j.id}  {j.status:<12} mode={j.mode}  config={j.config_hash}  created={j.created_at:%Y-%m-%d %H:%M}")


@app.command(hidden=True)
def review(job_id: Optional[str] = typer.Option(None, "--job-id", help="Accepted for symmetry; the queue is workspace-wide."),
           workspace: Optional[Path] = WS, limit: int = 30,
           show: str = typer.Option("pending", help="pending | failed | quarantined | rejected"),
           approve: list[str] = typer.Option(None, help="Image ID to approve (repeatable)."),
           reject: list[str] = typer.Option(None, help="Image ID to reject (repeatable)."),
           quarantine: list[str] = typer.Option(None, help="Image ID to quarantine (repeatable)."),
           restore: list[str] = typer.Option(None, help="Image ID to return to its automatic decision (repeatable).")):
    """List uncertain or failed samples and record review decisions. Nothing is deleted."""
    from app.database.models import Image, StageResult
    from app.services import review as rv

    ws = _ws(workspace)
    acted = False
    for action, ids in (("approve", approve), ("reject", reject), ("quarantine", quarantine), ("restore", restore)):
        for image_id in ids or []:
            try:
                out = rv.decide(ws, image_id, action)
                typer.echo(f"{image_id}: {out['decision']} - {out['consequence']}")
            except Exception as exc:
                typer.secho(f"{image_id}: {exc}", fg=typer.colors.RED)
            acted = True
    if acted:
        typer.echo("Run `image-pipeline export` to rebuild images and manifests.")
        return
    with ws.db.session() as s:
        q = select(Image).where(Image.missing.is_(False))
        if show == "pending":
            q = q.where(Image.review_status == "pending")
        elif show == "failed":
            q = q.where(Image.id.in_(select(StageResult.image_pk).where(StageResult.status == "failed")))
        elif show in ("quarantined", "rejected"):
            q = q.where(Image.decision == show)
        else:
            _fail("--show must be pending, failed, quarantined or rejected", 2)
        rows = list(s.scalars(q.order_by(Image.quality_score).limit(limit)))
        if not rows:
            typer.echo(f"No images in the '{show}' queue.")
        for im in rows:
            typer.echo(f"{im.image_id}  {im.decision:<11} q={im.quality_score}  {im.relative_path}")
            typer.echo(f"    reasons: {', '.join((im.decision_reasons or []) + (im.review_reasons or [])) or '-'}")
            if im.short_caption:
                typer.echo(f"    caption: {im.short_caption}")


@app.command(hidden=True)
def validate(job_id: Optional[str] = typer.Option(None, "--job-id"), workspace: Optional[Path] = WS):
    """Re-run export validation (paths, decoding, captions, schema, rights, leakage, checksums)."""
    ws = _ws(workspace)
    _export(ws, None)
    report = json.loads((ws.output_dir / "reports" / "export_validation.json").read_text())
    _print({k: report[k] for k in ("passed", "checks", "records_valid")})
    if not report["passed"]:
        raise typer.Exit(1)


def _export(ws: Workspace, formats: Optional[list[str]]):
    from app.pipeline.runner import EXPORT_ONLY, create_job, run_job

    cfg = ws.config
    if formats:
        cfg = cfg.model_copy(deep=True)
        cfg.export.formats = formats
        cfg = PipelineConfig.model_validate(cfg.model_dump())
    job_id = create_job(ws, EXPORT_ONLY, cfg)
    con = _Console(show_items=False)
    summary = run_job(ws, job_id, on_stage=con.stage, on_progress=con.progress, on_stage_end=con.end, on_item=con.item)
    if summary["status"] != "completed":
        _fail(f"Export failed: {summary['error']}")
    return job_id


@app.command(hidden=True)
def export(job_id: Optional[str] = typer.Option(None, "--job-id"), workspace: Optional[Path] = WS,
           format: list[str] = typer.Option(None, "--format", "-f", help="jsonl | csv | parquet (repeatable).")):
    """Rebuild decisions, processed images, splits and manifests from cached results (no vision inference)."""
    from app.database.session import kv_get

    ws = _ws(workspace)
    try:
        _export(ws, list(format) if format else None)
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(str(exc), 2)
    with ws.db.session() as s:
        _print(kv_get(s, "export:last", {}))
    typer.echo(f"Dataset written to {ws.output_dir}")


@app.command(hidden=True)
def report(job_id: Optional[str] = typer.Option(None, "--job-id"), workspace: Optional[Path] = WS,
           name: str = typer.Option("dataset_summary", help="dataset_summary | duplicate_report | quality_report | alignment_report | "
                                                            "rights_report | split_leakage_report | coverage_report | export_validation")):
    """Print a report produced by the last export."""
    ws = _ws(workspace)
    p = ws.output_dir / "reports" / f"{Path(name).name}.json"
    if not p.is_file():
        _fail("Report not found. Run `image-pipeline run` or `image-pipeline export` first.")
    typer.echo(p.read_text())


@app.command(hidden=True)
def stages(workspace: Optional[Path] = WS, config: Optional[Path] = CFG):
    """Show the stage plan for the current configuration and why stages are skipped."""
    from app.pipeline.runner import describe_pipeline

    for st in describe_pipeline(_ws(workspace, config)):
        mark = "on " if st["enabled"] else "off"
        typer.echo(f"[{mark}] {st['name']:<18} {st['title']} (processes {st['processes']})" + (f"\n        skipped: {st['reason']}" if st["reason"] else ""))


@app.command(hidden=True)
def upload(workspace: Optional[Path] = WS,
           force: bool = typer.Option(False, "--force", help="Upload even if the folder is below huggingface.min_size_mb.")):
    """Upload the exported dataset to the configured Hugging Face dataset repository now."""
    from app.services.hf_upload import maybe_upload

    ws = _ws(workspace)
    con = _Console()
    try:
        out = maybe_upload(ws, force=force, say=lambda m: con.progress("hf_upload", {"message": m}))
    except Exception as exc:
        _fail(f"Upload failed: {exc}")
    _print(out)
    if not out["uploaded"]:
        raise typer.Exit(1)


@app.command(hidden=True)
def benchmark(adapters: str = typer.Option("template", help="Comma-separated: template, sidecar or captioning model keys from the registry."),
              sample: int = typer.Option(20, min=1, max=500), seed: int = 42, workspace: Optional[Path] = WS):
    """Compare caption adapters on a sample of your images (latency, memory, failures) and write a human review sheet."""
    from app.services.benchmark import run_benchmark

    try:
        out = run_benchmark(_ws(workspace), [a.strip() for a in adapters.split(",") if a.strip()], sample, seed)
    except Exception as exc:
        _fail(str(exc))
    _print(out["report"])
    typer.echo(f"Results: {out['results']}\nReview sheet to fill in: {out['review_sheet']}")


@app.command("benchmark-summarize", hidden=True)
def benchmark_summarize(sheet: Path):
    """Turn a completed review sheet into accuracy, hallucination-rate and coverage figures."""
    from app.services.benchmark import summarize_sheet

    try:
        _print(summarize_sheet(sheet))
    except Exception as exc:
        _fail(str(exc))


@models_app.command("list")
def models_list(workspace: Optional[Path] = WS):
    """List registry models with size, licence, memory needs and install status."""
    from app.models.hub import ModelHub

    ws = _ws(workspace)
    _print(ModelHub(ws.config, ws.root).describe())


@models_app.command("download")
def models_download(key: str, yes: bool = typer.Option(False, "--yes", help="Confirm the download."), workspace: Optional[Path] = WS):
    """Show what would be downloaded; add --yes to download."""
    from app.models.hub import ModelHub, download_model

    ws = _ws(workspace)
    try:
        out = download_model(ModelHub(ws.config, ws.root), key, confirm=yes)
    except Exception as exc:
        _fail(str(exc))
    _print(out)
    if not yes and not out["already_installed"]:
        typer.echo("Nothing was downloaded. Re-run with --yes to download.")


@sources_app.command("list")
def sources_list(workspace: Optional[Path] = WS):
    from sqlalchemy import func

    from app.database.models import Image, Source

    ws = _ws(workspace)
    with ws.db.session() as s:
        for src in s.scalars(select(Source)):
            n = s.scalar(select(func.count(Image.id)).where(Image.source_pk == src.id, Image.missing.is_(False)))
            typer.echo(f"{src.source_id}  {src.label:<24} images={n}  rights={src.rights_status}  license={src.license or '-'}")


@sources_app.command("set-rights")
def sources_set_rights(source_id: str, rights_status: Optional[str] = typer.Option(None, "--status"),
                       license: Optional[str] = typer.Option(None), creator: Optional[str] = typer.Option(None),
                       attribution: Optional[str] = typer.Option(None), source_url: Optional[str] = typer.Option(None),
                       workspace: Optional[Path] = WS):
    """Record the rights information you attest for a source. The tool does not verify licences."""
    from app.services.review import set_source_rights

    try:
        _print(set_source_rights(_ws(workspace), source_id, rights_status=rights_status, license=license, creator=creator,
                                 attribution=attribution, source_url=source_url))
    except Exception as exc:
        _fail(str(exc), 2)


if __name__ == "__main__":
    app()
