"""Processes 1, 3, 4, 10: streaming folder ingestion, stable IDs, checksums, resumable scan state.

Source folders are only listed and read. Progress is committed in small batches, so an
interrupted scan resumes without re-hashing files that were already registered
(unchanged size + modification time)."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Optional

from sqlalchemy import select, update

from app.config.settings import RIGHTS_STATUSES, SourceCfg
from app.core.imageio import SUPPORTED_EXTENSIONS
from app.core.io import sha256_file
from app.core.logging import get_logger
from app.core.workspace import Workspace
from app.database.models import DecisionLog, Image, Source, StageResult, utcnow
from app.database.session import kv_set

log = get_logger("scan")
BATCH = 256


def make_source_id(label: Optional[str], root: Path) -> str:
    basis = label if label else str(root)
    return "src_" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:10]


def make_image_id(source_id: str, relative_path: str, sha256: str, length: int = 16) -> str:
    """Deterministic: the same source, relative path and content always give the same ID."""
    digest = hashlib.sha256(f"{source_id}\x00{relative_path}\x00{sha256}".encode("utf-8")).hexdigest()
    return "img_" + digest[:length]


def register_source(ws: Workspace, sc: SourceCfg) -> Source:
    root = ws.validate_source_dir(sc.path)
    sid = make_source_id(sc.label, root)
    with ws.db.session() as s:
        src = s.scalar(select(Source).where(Source.source_id == sid))
        if src is None:
            src = Source(source_id=sid, label=sc.label or root.name, root_path=str(root),
                         rights_status=sc.rights_status or ws.config.rights.default_status)
            s.add(src)
        src.root_path = str(root)
        for f in ("license", "creator", "attribution", "source_url", "notes", "metadata_file"):
            v = getattr(sc, f)
            if v is not None:
                setattr(src, f, v)
        if sc.rights_status is not None:
            src.rights_status = sc.rights_status
        s.flush()
        s.expunge(src)
        return src


def _walk(root: Path, recursive: bool, follow_symlinks: bool) -> Iterator[tuple[str, list[os.DirEntry]]]:
    """Yield (relative_dir, file entries) one directory at a time (bounded memory)."""
    stack = [root]
    while stack:
        d = stack.pop()
        files: list[os.DirEntry] = []
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.name.startswith("."):
                        continue
                    try:
                        if e.is_dir(follow_symlinks=follow_symlinks):
                            if recursive:
                                stack.append(Path(e.path))
                        elif e.is_file(follow_symlinks=follow_symlinks):
                            files.append(e)
                    except OSError:
                        continue
        except OSError as exc:
            log.warning("cannot list directory under source: %s", exc.strerror)
            continue
        files.sort(key=lambda e: e.name)
        yield (os.path.relpath(d, root) if d != root else ""), files


def scan_source(ws: Workspace, src: Source, progress: Optional[Callable[[dict], None]] = None,
                cancel: Optional[Callable[[], bool]] = None) -> dict:
    cfg = ws.config
    root = Path(src.root_path)
    token = uuid.uuid4().hex[:12]
    stats = {"source_id": src.source_id, "label": src.label, "files_seen": 0, "new": 0, "unchanged": 0,
             "changed": 0, "unsupported": 0, "unreadable": 0, "id_conflicts": 0, "missing": 0,
             "unsupported_by_extension": {}, "unsupported_samples": [], "completed": False}
    seen_pks: list[int] = []

    def flush_state():
        with ws.db.session() as s:
            kv_set(s, f"scan:{src.source_id}", {**stats, "updated_at": utcnow().isoformat()})

    with ThreadPoolExecutor(max_workers=cfg.processing.cpu_workers) as pool:
        for rel_dir, entries in _walk(root, cfg.processing.recursive_scan, cfg.processing.follow_symlinks):
            candidates = []
            for e in entries:
                stats["files_seen"] += 1
                ext = os.path.splitext(e.name)[1].lower()
                if ext not in SUPPORTED_EXTENSIONS:
                    stats["unsupported"] += 1
                    key = ext or "(none)"
                    stats["unsupported_by_extension"][key] = stats["unsupported_by_extension"].get(key, 0) + 1
                    if len(stats["unsupported_samples"]) < 50:
                        stats["unsupported_samples"].append(os.path.join(rel_dir, e.name))
                    continue
                candidates.append((e, ext))
            for i in range(0, len(candidates), BATCH):
                if cancel and cancel():
                    flush_state()
                    return stats
                _ingest_batch(ws, src, rel_dir, candidates[i:i + BATCH], pool, stats, seen_pks, token)
                flush_state()
                if progress:
                    progress(stats)

    # Files that disappeared since the previous scan are marked, never purged.
    with ws.db.session() as s:
        res = s.execute(update(Image).where(Image.source_pk == src.id, Image.missing.is_(False),
                                            Image.seen_token != token).values(missing=True))
        stats["missing"] = res.rowcount or 0
        dbsrc = s.get(Source, src.id)
        dbsrc.last_scan_at = utcnow()
    stats["completed"] = True
    flush_state()
    return stats


def _ingest_batch(ws, src, rel_dir, batch, pool, stats, seen_pks, token) -> None:
    rels = {os.path.join(rel_dir, e.name).replace(os.sep, "/"): (e, ext) for e, ext in batch}
    with ws.db.session() as s:
        existing = {im.relative_path: im for im in s.scalars(
            select(Image).where(Image.source_pk == src.id, Image.relative_path.in_(list(rels))))}
        todo = []
        for rel, (e, ext) in rels.items():
            try:
                st = e.stat()
            except OSError:
                stats["unreadable"] += 1
                continue
            im = existing.get(rel)
            if im is not None and im.file_size == st.st_size and im.mtime_ns == st.st_mtime_ns:
                im.seen_token = token
                im.missing = False
                stats["unchanged"] += 1
                continue
            todo.append((rel, e, ext, st, im))

        def _hash(item):
            try:
                return sha256_file(Path(item[1].path))
            except OSError as exc:
                return exc

        for item, digest in zip(todo, pool.map(_hash, todo)):
            rel, e, ext, st, im = item
            if isinstance(digest, Exception):
                stats["unreadable"] += 1
                log.warning("unreadable file skipped (%s): %s", src.source_id, rel)
                continue
            image_id = make_image_id(src.source_id, rel, digest)
            clash = s.scalar(select(Image).where(Image.image_id == image_id))
            if clash is not None and (im is None or clash.id != im.id):
                stats["id_conflicts"] += 1  # astronomically unlikely; fall back to a longer ID
                image_id = make_image_id(src.source_id, rel, digest, length=32)
            if im is None:
                s.add(Image(image_id=image_id, source_pk=src.id, abs_path=e.path, relative_path=rel,
                            file_name=e.name, extension=ext, file_size=st.st_size, mtime_ns=st.st_mtime_ns,
                            sha256=digest, seen_token=token, decision="pending", decision_reasons=[],
                            review_status="not_required", review_reasons=[], user_edits={}))
                stats["new"] += 1
            else:
                im.seen_token = token
                im.missing = False
                im.file_size, im.mtime_ns, im.abs_path = st.st_size, st.st_mtime_ns, e.path
                if im.sha256 != digest:
                    # Content changed: treat as a new image and invalidate everything derived from it.
                    s.add(DecisionLog(image_id=im.image_id, actor="auto", action="source_changed",
                                      detail={"new_image_id": image_id}))
                    s.query(StageResult).filter(StageResult.image_pk == im.id).delete()
                    im.image_id, im.sha256 = image_id, digest
                    im.valid = None
                    im.decision, im.auto_decision, im.manual_decision = "pending", None, None
                    im.decision_reasons, im.review_status, im.review_reasons = [], "not_required", []
                    im.processed_path = im.processed_sha256 = im.split = None
                    im.user_edits = {}
                    stats["changed"] += 1
                else:
                    stats["unchanged"] += 1


def apply_metadata_file(ws: Workspace, src: Source) -> dict:
    """Optional user-supplied CSV/JSON/JSONL keyed by ``relative_path`` or ``file_name``."""
    if not src.metadata_file:
        return {"applied": 0}
    path = Path(src.metadata_file).expanduser()
    if not path.is_absolute():
        path = Path(src.root_path) / path
    if not path.is_file():
        raise ValueError("metadata file not found")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
    elif suffix == ".jsonl":
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
    elif suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            rows = json.load(fh)
        if isinstance(rows, dict):
            rows = rows.get("images") or rows.get("records") or []
    else:
        raise ValueError("metadata file must be .csv, .json or .jsonl")
    applied, unmatched, invalid = 0, 0, 0
    fields = ("license", "creator", "attribution", "source_url", "notes")
    with ws.db.session() as s:
        for row in rows:
            if not isinstance(row, dict):
                invalid += 1
                continue
            key = (row.get("relative_path") or "").replace("\\", "/")
            q = select(Image).where(Image.source_pk == src.id)
            q = q.where(Image.relative_path == key) if key else q.where(Image.file_name == (row.get("file_name") or ""))
            matches = list(s.scalars(q.limit(2)))
            if len(matches) != 1:
                unmatched += 1
                continue
            im = matches[0]
            for f in fields:
                if row.get(f):
                    setattr(im, f, str(row[f]))
            rs = row.get("rights_status")
            if rs:
                if rs in RIGHTS_STATUSES:
                    im.rights_status = rs
                else:
                    invalid += 1
            cap = row.get("caption") or row.get("text")
            if cap:
                im.supplied_caption = str(cap).strip()
            applied += 1
    return {"applied": applied, "unmatched": unmatched, "invalid_values": invalid}


def scan_all(ws: Workspace, progress=None, cancel=None) -> dict:
    """Register every configured source and scan it. Safe to call repeatedly."""
    results = []
    for sc in ws.config.paths.source_cfgs():
        src = register_source(ws, sc)
        st = scan_source(ws, src, progress, cancel)
        if st["completed"] and src.metadata_file:
            st["metadata_file"] = apply_metadata_file(ws, src)
        results.append(st)
        log.info("scan %s: %s", src.source_id, {k: v for k, v in st.items() if isinstance(v, int)})
    out = {"sources": results, "finished_at": utcnow().isoformat(),
           "totals": {k: sum(r[k] for r in results) for k in ("files_seen", "new", "unchanged", "changed", "unsupported", "unreadable", "missing")}}
    with ws.db.session() as s:
        kv_set(s, "scan:last", out)
    return out
