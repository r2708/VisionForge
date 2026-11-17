"""Shard mode: the input is a folder of URL lists (``TsvHttpData-1.0`` files: url, size, md5)
instead of a folder of images.

Loop: download a batch -> run the normal pipeline on it -> when the output reaches the size
threshold, close it as a shard, upload the shard to Hugging Face and delete the local copies ->
continue with the next URLs. A cursor into the URL lists is stored after every download batch,
so stopping at any point and running again continues where it stopped.

Only files this tool downloaded or produced are ever deleted; the URL lists are read-only."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import ssl
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterator, Optional

from sqlalchemy import delete, func, select

from app.config.settings import PipelineConfig, SourceCfg
from app.core.io import atomic_write_text, free_disk_gb
from app.core.logging import get_logger
from app.core.runlock import RunLock
from app.core.workspace import Workspace
from app.database.models import Embedding, Image, KV, Source, UrlItem, utcnow
from app.database.session import kv_get, kv_set
from app.services import hf_upload

log = get_logger("shards")
TSV_HEADER = "TsvHttpData-1.0"
STATE_KEY = "shards:state"
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
GONE = {403, 404, 410, 451}


def is_url_list_dir(path: Path) -> bool:
    """True when the folder holds TsvHttpData URL lists and no image files."""
    path = Path(path)
    if not path.is_dir():
        return False
    tsvs = sorted(path.glob("*.tsv"))
    if not tsvs or any(f.suffix.lower() in IMAGE_EXT for f in path.iterdir() if f.is_file()):
        return False
    with open(tsvs[0], "r", encoding="utf-8", errors="replace") as fh:
        return fh.readline().strip() == TSV_HEADER


def list_files(source: Path) -> list[str]:
    def key(name: str):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]

    return sorted((f.name for f in Path(source).glob("*.tsv")), key=key)


def count_urls(source: Path, files: list[str]) -> int:
    total = 0
    for name in files:
        with open(Path(source) / name, "rb") as fh:
            total += sum(chunk.count(b"\n") for chunk in iter(lambda: fh.read(1 << 20), b""))
        total -= 1  # header line
    return max(0, total)


def read_urls(source: Path, state: dict) -> Iterator[tuple[int, str, int, str, int, int]]:
    """Yield (index, url, size, md5_base64, file_idx, next_offset) from the cursor onwards."""
    for fi in range(state["file_idx"], len(state["files"])):
        with open(Path(source) / state["files"][fi], "rb") as fh:
            fh.seek(state["offset"] if fi == state["file_idx"] else 0)
            while True:
                line = fh.readline()
                if not line:
                    break
                parts = line.decode("utf-8", errors="replace").rstrip("\r\n").split("\t")
                if len(parts) != 3 or not parts[0].startswith(("http://", "https://")) or not parts[1].isdigit():
                    continue  # header or malformed line
                yield state["index"], parts[0], int(parts[1]), parts[2], fi, fh.tell()
                state["index"] += 1
        state["file_idx"], state["offset"] = fi + 1, 0


HF_PREFIXES = ("hf://", "https://huggingface.co/datasets/", "http://huggingface.co/datasets/", "huggingface.co/datasets/")
MAGIC = ((b"\xff\xd8\xff", "jpg"), (b"\x89PNG\r\n\x1a\n", "png"), (b"GIF8", "gif"), (b"BM", "bmp"),
         (b"II*\x00", "tif"), (b"MM\x00*", "tif"))


def parse_hf_source(text: str) -> Optional[str]:
    """'hf://org/name' or a huggingface.co/datasets/org/name URL -> 'org/name' (None otherwise)."""
    text = str(text).strip()
    for prefix in HF_PREFIXES:
        if text.startswith(prefix):
            rest = text[len(prefix):].split("?")[0].strip("/")
            parts = rest.split("/")
            if len(parts) >= 2 and all(re.fullmatch(r"[A-Za-z0-9][\w.-]*", x) for x in parts[:2]):
                return "/".join(parts[:2])
    return None


def sniff_ext(data: bytes) -> Optional[str]:
    for magic, ext in MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def hf_dataset_info(repo: str, ctx) -> dict:
    """Parquet files and row count of a public Hugging Face dataset (read-only API call)."""
    req = urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}", headers={"User-Agent": "image-dataset-pipeline/0.1"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        info = json.loads(r.read().decode("utf-8"))
    if info.get("gated") or info.get("private"):
        raise ValueError(f"{repo} is private or gated; only public datasets are supported")
    files = sorted(x["rfilename"] for x in info.get("siblings", []) if x["rfilename"].endswith(".parquet"))
    if not files:
        raise ValueError(f"{repo} has no parquet files")
    card = info.get("cardData") or {}
    rows = 0
    di = card.get("dataset_info")
    for entry in (di if isinstance(di, list) else [di] if di else []):
        rows += sum(int(sp.get("num_examples") or 0) for sp in entry.get("splits", []))
    return {"files": files, "rows": rows or None, "license": card.get("license_name") or card.get("license")}


def download_file(url: str, dest: Path, ctx, timeout: float, say: Callable[[str], None]) -> int:
    """Resumable download of one large file (no credentials are sent: public data only)."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = time.time()
    for attempt in range(6):
        have = part.stat().st_size if part.exists() else 0
        headers = {"User-Agent": "image-dataset-pipeline/0.1"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout, context=ctx) as r:
                if have and r.status != 206:
                    have = 0  # server ignored the range: start again
                total = have + int(r.headers.get("Content-Length") or 0)
                with open(part, "ab" if have else "wb") as out:
                    while True:
                        block = r.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                        have += len(block)
                        if time.time() - last >= 5:
                            last = time.time()
                            say(f"{dest.name}: {have / 1024**2:.0f}/{total / 1024**2:.0f} MB")
            if total and have < total:
                raise OSError("connection closed early")
            os.replace(part, dest)
            return have
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and have:  # already complete
                os.replace(part, dest)
                return have
            if exc.code in (401, 403, 404):
                raise ValueError(f"cannot download {url.rsplit('/', 1)[-1]}: HTTP {exc.code}")
            err = f"HTTP {exc.code}"
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"[:160]
        say(f"{dest.name}: {err}; retrying ({attempt + 1}/6)")
        time.sleep(min(60, 5 * (attempt + 1)))
    raise OSError(f"download of {dest.name} failed after 6 attempts: {err}")


def _ssl_context():
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_one(url: str, size: int, md5_b64: str, dest: Path, cfg, ctx) -> tuple[str, str, int]:
    """Returns (status, detail, bytes). status: ok | exists | gone | too_large | failed."""
    if dest.is_file() and dest.stat().st_size == size:
        return "exists", "", size
    if size > cfg.max_file_mb * 1024 * 1024:
        return "too_large", f"{size / 1024**2:.1f} MB exceeds shards.max_file_mb", 0
    part = dest.with_name(dest.name + ".part")
    err = "unknown error"
    for attempt in range(cfg.retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "image-dataset-pipeline/0.1"})
            h = hashlib.md5()
            n = 0
            with urllib.request.urlopen(req, timeout=cfg.timeout_s, context=ctx) as r, open(part, "wb") as out:
                while True:
                    block = r.read(1 << 16)
                    if not block:
                        break
                    n += len(block)
                    if n > cfg.max_file_mb * 1024 * 1024:
                        raise ValueError("response larger than shards.max_file_mb")
                    h.update(block)
                    out.write(block)
            if md5_b64 and base64.b64encode(h.digest()).decode() != md5_b64:
                # The host no longer serves the listed file (Flickr replaces removed photos with a placeholder).
                part.unlink(missing_ok=True)
                return "gone", "content does not match the listed checksum (photo removed or replaced)", 0
            os.replace(part, dest)
            return "ok", "", n
        except urllib.error.HTTPError as exc:
            part.unlink(missing_ok=True)
            if exc.code in GONE:
                return "gone", f"HTTP {exc.code}", 0
            err = f"HTTP {exc.code}"
            if exc.code == 429:
                time.sleep(min(30, 5 * (attempt + 1)))
        except Exception as exc:  # timeouts, connection resets, DNS, ...
            part.unlink(missing_ok=True)
            err = f"{type(exc).__name__}: {exc}"[:160]
        time.sleep(min(8, 1.5 * (attempt + 1)))
    return "failed", err, 0


def _safe_name(index: int, url: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]", "_", url.rsplit("/", 1)[-1].split("?")[0])[:80] or "image.jpg"
    if not base.lower().endswith(IMAGE_EXT):
        base += ".jpg"
    return f"{index:09d}_{base}"


class Prefetcher(threading.Thread):
    """Downloads the next source files in the background while the current one is processed.

    At most ``window`` files are on disk at any time: the file being processed plus the next
    ``window - 1``. A further file is only fetched once the current one has been fully consumed
    and deleted (the window slides forward with ``state['file_idx']``)."""

    def __init__(self, state: dict, cache: Path, window: int, timeout: float, say: Callable[[str], None]):
        super().__init__(name="prefetch", daemon=True)
        self.state, self.cache, self.window, self.timeout, self.say = state, cache, max(1, window), timeout, say
        self.cv = threading.Condition()
        self.errors: dict[int, str] = {}
        self.active: Optional[int] = None
        self.stopping = False
        # Index of the source file whose images are being processed right now. It still counts
        # towards the window although its parquet file is already unpacked and deleted.
        self.hold: Optional[int] = None

    def local(self, idx: int) -> Path:
        return self.cache / self.state["files"][idx].replace("/", "__")

    def _next(self) -> Optional[int]:
        cur, n = self.state["file_idx"], len(self.state["files"])
        base = cur if self.hold is None else min(self.hold, cur)
        return next((i for i in range(cur, min(base + self.window, n)) if not self.local(i).is_file()), None)

    def release(self) -> None:
        """Processing of the held file finished: the window may slide forward."""
        with self.cv:
            self.hold = None
            self.cv.notify_all()

    def run(self) -> None:
        ctx = _ssl_context()
        last_note = [0.0]

        def quiet_say(msg: str) -> None:  # background progress is throttled so it does not drown the processing log
            if time.time() - last_note[0] >= 15 or "retrying" in msg:
                last_note[0] = time.time()
                self.say(f"(background) {msg}")

        while not self.stopping:
            with self.cv:
                idx = self._next()
                if idx is None:
                    self.cv.wait(timeout=1.0)
                    continue
                self.active = idx
            rel = self.state["files"][idx]
            try:
                self.say(f"fetching {rel} (file {idx + 1} of {len(self.state['files'])})"
                         + ("" if idx == self.state["file_idx"] else " in the background"))
                size = download_file(f"https://huggingface.co/datasets/{self.state['repo']}/resolve/main/{rel}",
                                     self.local(idx), ctx, self.timeout, quiet_say if idx != self.state["file_idx"] else self.say)
                self.say(f"{rel}: {size / 1024**2:.0f} MB ready")
                self.errors.pop(idx, None)
            except Exception as exc:
                self.errors[idx] = f"{type(exc).__name__}: {exc}"[:200]
                time.sleep(5)
            finally:
                with self.cv:
                    self.active = None
                    self.cv.notify_all()

    def wait_for(self, idx: int) -> Path:
        """Block until source file ``idx`` is on disk (raises if its download keeps failing)."""
        failures = 0
        with self.cv:
            while not self.local(idx).is_file():
                if idx in self.errors and self.active != idx:
                    failures += 1
                    err = self.errors.pop(idx)
                    if failures >= 3:
                        raise OSError(f"could not download {self.state['files'][idx]}: {err}")
                if not self.is_alive():
                    raise OSError("the background downloader stopped unexpectedly")
                self.cv.notify_all()
                self.cv.wait(timeout=1.0)
        return self.local(idx)

    def on_disk(self) -> int:
        return sum(1 for f in self.cache.glob("*.parquet")) if self.cache.is_dir() else 0

    def stop(self) -> None:
        self.stopping = True
        with self.cv:
            self.cv.notify_all()


class ShardRunner:
    def __init__(self, ws: Workspace, console, quiet: bool = False):
        self.ws, self.con, self.quiet = ws, console, quiet
        self.cfg = ws.config
        self.sh = self.cfg.shards

    # ---- state ----------------------------------------------------------
    def load(self) -> Optional[dict]:
        with self.ws.db.session() as s:
            return kv_get(s, STATE_KEY, None)

    def save(self, state: dict) -> None:
        with self.ws.db.session() as s:
            kv_set(s, STATE_KEY, {**state, "updated_at": utcnow().isoformat()})

    def init_state(self, source, rights_status: Optional[str], license: Optional[str]) -> dict:
        state = self.load()
        repo = parse_hf_source(source)
        info = None
        if repo:
            source = f"hf://{repo}"
            if state and state["source"] == source:
                files = state["files"]  # keep the stored file order; no network call needed to continue
            else:
                self.con.progress("shards", {"message": f"reading the file list of {repo} from Hugging Face..."})
                info = hf_dataset_info(repo, _ssl_context())
                files = info["files"]
        else:
            source = Path(source).expanduser().resolve()
            files = list_files(source)
        if state and state["source"] == str(source) and state["files"] == files:
            if rights_status:
                state["rights_status"] = rights_status
            if license:
                state["license"] = license
            return state
        if state and state.get("index"):
            raise ValueError("this workspace already tracks a different URL-list folder; use another --workspace for a new one")
        with self.ws.db.session() as s:
            if not state and s.scalar(select(func.count(Image.id))):
                raise ValueError("this workspace already holds images from a normal folder run; shard mode needs its own "
                                 "workspace (pass --workspace NEW_DIR or set IMAGE_PIPELINE_WORKSPACE)")
        if repo:
            total = info["rows"] or 0
            default_license = (f"{info['license'] or 'unspecified'} (as declared on the Hugging Face dataset card of {repo}; "
                               "each image keeps the copyright and licence of its original source; not verified)")
        else:
            self.con.progress("shards", {"message": f"counting URLs in {len(files)} list file(s)..."})
            total, default_license = count_urls(source, files), self.sh.license_note
        return {"kind": "hf" if repo else "urls", "repo": repo,
                "source": str(source), "files": files, "total_urls": total, "file_idx": 0, "offset": 0,
                "index": 0, "limit_index": None, "shard": 1, "shards_done": [], "pending_upload": [],
                "rights_status": rights_status, "license": license or default_license, "unsupported": 0,
                "downloaded": 0, "gone": 0, "failed": 0, "duplicates": 0, "too_large": 0, "records_uploaded": 0}

    def target_mb(self) -> float:
        return float(self.sh.target_mb or self.cfg.huggingface.min_size_mb)

    def shard_name(self, n: int) -> str:
        return f"shard_{n:04d}"

    def download_dir(self, n: int) -> Path:
        return self.ws.root / "downloads" / self.shard_name(n)

    # ---- download -------------------------------------------------------
    def download_round(self, state: dict, want_bytes: float) -> dict:
        """Consume URLs until about ``want_bytes`` of listed size is fetched. Returns round stats."""
        if state.get("kind") == "hf":
            return self.hf_round(state, want_bytes)
        dest = self.download_dir(state["shard"])
        dest.mkdir(parents=True, exist_ok=True)
        ctx = _ssl_context()
        limit = state["limit_index"]
        stats = {"ok": 0, "gone": 0, "failed": 0, "duplicate": 0, "too_large": 0, "bytes": 0, "urls": 0}
        est_total = max(1, int(want_bytes / (2.1 * 1024 * 1024)))
        t0 = time.time()
        meta_path = dest / "metadata.jsonl"
        reader = read_urls(Path(state["source"]), state)
        exhausted = False
        with ThreadPoolExecutor(max_workers=self.sh.download_workers) as pool:
            while stats["bytes"] < want_bytes:
                batch = []
                for row in reader:
                    batch.append(row)
                    if len(batch) >= self.sh.download_workers * 4 or (limit is not None and row[0] + 1 >= limit):
                        break
                else:
                    exhausted = True
                if not batch:
                    break
                with self.ws.db.session() as s:
                    seen = set(s.scalars(select(UrlItem.md5).where(UrlItem.md5.in_([b[3] for b in batch]), UrlItem.status == "ok")))
                jobs = []
                for idx, url, size, md5, fi, off in batch:
                    if md5 in seen:
                        jobs.append(None)
                    else:
                        seen.add(md5)
                        jobs.append(pool.submit(self._timed, url, size, md5, dest / _safe_name(idx, url), ctx))
                rows, meta_lines = [], []
                for (idx, url, size, md5, fi, off), fut in zip(batch, jobs):
                    if fut is None:
                        status, detail, n, secs = "duplicate", "same checksum as an earlier URL", 0, 0.0
                    else:
                        status, detail, n, secs = fut.result()
                    name = _safe_name(idx, url)
                    if status in ("ok", "exists"):
                        stats["ok"] += 1
                        stats["bytes"] += size
                        meta_lines.append(json.dumps({"relative_path": name, "source_url": url, "license": state["license"]}))
                        msg = f"{name} ({size / 1024**2:.2f} MB)" + (" already downloaded" if status == "exists" else "")
                        self.con.item("DOWNLOAD", f"url_{idx:09d}", True, msg, secs)
                        status = "ok"
                    else:
                        stats[status] += 1
                        self.con.item("DOWNLOAD", f"url_{idx:09d}", status == "duplicate", f"{status}: {detail}" if status != "duplicate" else "duplicate of an earlier URL, skipped", secs)
                    rows.append(dict(id=idx, url=url, md5=md5, size=size, status=status, shard=state["shard"], error=detail or None))
                    stats["urls"] += 1
                if meta_lines:
                    with open(meta_path, "a", encoding="utf-8") as fh:
                        fh.write("\n".join(meta_lines) + "\n")
                last = batch[-1]
                state["file_idx"], state["offset"], state["index"] = last[4], last[5], last[0] + 1
                if exhausted:
                    state["file_idx"], state["offset"] = len(state["files"]), 0
                for k, sk in (("ok", "downloaded"), ("gone", "gone"), ("failed", "failed"), ("duplicate", "duplicates"), ("too_large", "too_large")):
                    state[sk] += sum(1 for r in rows if r["status"] == k)
                with self.ws.db.session() as s:  # checkpoint: ledger rows and cursor together
                    for r in rows:
                        s.merge(UrlItem(**r))
                    kv_set(s, STATE_KEY, {**state, "updated_at": utcnow().isoformat()})
                elapsed = time.time() - t0
                rate = stats["urls"] / elapsed if elapsed > 0 else None
                self.con.failed["download"] = stats["gone"] + stats["failed"] + stats["too_large"]
                self.con.progress("download", {"processed": stats["urls"], "total": max(est_total, stats["urls"] + 1), "computed": stats["urls"],
                                               "elapsed_s": elapsed, "rate_per_s": rate,
                                               "eta_s": max(0.0, (want_bytes - stats["bytes"])) / (stats["bytes"] / elapsed) if stats["bytes"] else None})
                if limit is not None and state["index"] >= limit:
                    break
        stats["exhausted"] = exhausted or state["file_idx"] >= len(state["files"])
        return stats

    def hf_round(self, state: dict, want_bytes: float) -> dict:
        """Take images out of the dataset's parquet files until about ``want_bytes`` are extracted."""
        import pyarrow.parquet as pq

        dest = self.download_dir(state["shard"])
        dest.mkdir(parents=True, exist_ok=True)
        cache = self.ws.root / "downloads" / "_source"
        ctx = _ssl_context()
        limit = state["limit_index"]
        repo = state["repo"]
        stats = {"ok": 0, "gone": 0, "failed": 0, "duplicate": 0, "too_large": 0, "unsupported": 0, "bytes": 0, "urls": 0}
        t0 = time.time()
        meta_path = dest / "metadata.jsonl"
        say = lambda m: self.con.progress("download", {"message": m})  # noqa: E731
        while stats["bytes"] < want_bytes and state["file_idx"] < len(state["files"]) and not (limit is not None and state["index"] >= limit):
            rel = state["files"][state["file_idx"]]
            local = cache / rel.replace("/", "__")
            if not local.is_file():
                say(f"waiting for {rel} to finish downloading")
                local = self.prefetch.wait_for(state["file_idx"])
            pf = pq.ParquetFile(local)
            names = pf.schema_arrow.names
            img_col = "image" if "image" in names else next((n for n in names if str(pf.schema_arrow.field(n).type).startswith("struct") and "bytes" in str(pf.schema_arrow.field(n).type)), None)
            if img_col is None:
                raise ValueError(f"{rel} has no image column with embedded bytes")
            id_col = next((n for n in ("image_id", "id", "key") if n in names), None)
            cols = [c for c in (id_col, "ext" if "ext" in names else None, img_col) if c]
            stem = Path(rel).stem
            pos, stopped = 0, False
            for batch in pf.iter_batches(batch_size=32, columns=cols):
                n = batch.num_rows
                if pos + n <= state["offset"]:
                    pos += n
                    continue
                rows = batch.to_pylist()[max(0, state["offset"] - pos):]
                if limit is not None:
                    rows = rows[: max(0, limit - state["index"])]
                start = max(pos, state["offset"])
                ledger, meta_lines = [], []
                md5s = {}
                for k, row in enumerate(rows):
                    cell = row[img_col] or {}
                    data = cell.get("bytes") if isinstance(cell, dict) else None
                    md5s[k] = hashlib.md5(data).hexdigest() if data else None
                with self.ws.db.session() as s:
                    seen = set(s.scalars(select(UrlItem.md5).where(UrlItem.md5.in_([m for m in md5s.values() if m]), UrlItem.status == "ok")))
                for k, row in enumerate(rows):
                    idx = state["index"] + k
                    rid = str(row.get(id_col) or f"{stem}_{start + k:06d}") if id_col else f"{stem}_{start + k:06d}"
                    rid = re.sub(r"[^A-Za-z0-9._-]", "_", rid)[:80]
                    cell = row[img_col] or {}
                    data = cell.get("bytes") if isinstance(cell, dict) else None
                    ext = (sniff_ext(data) if data else None) or str(row.get("ext") or "").lower().lstrip(".")
                    ext = {"jpeg": "jpg", "tiff": "tif"}.get(ext, ext)
                    if not data:
                        status, detail = "failed", "row has no image bytes"
                    elif f".{ext}" not in IMAGE_EXT:
                        status, detail = "unsupported", f"format '{ext or 'unknown'}' is not supported"
                    elif len(data) > self.sh.max_file_mb * 1024 * 1024:
                        status, detail = "too_large", f"{len(data) / 1024**2:.1f} MB exceeds shards.max_file_mb"
                    elif md5s[k] in seen:
                        status, detail = "duplicate", "same bytes as an earlier image"
                    else:
                        seen.add(md5s[k])
                        name = f"{rid}.{ext}"
                        tmp = dest / (name + ".part")
                        tmp.write_bytes(data)
                        os.replace(tmp, dest / name)
                        status, detail = "ok", ""
                        stats["bytes"] += len(data)
                        meta_lines.append(json.dumps({"relative_path": name, "source_url": f"https://huggingface.co/datasets/{repo}",
                                                      "license": state["license"], "notes": f"{repo} {rel} image_id={rid}"}))
                    stats[status] += 1
                    stats["urls"] += 1
                    if status == "ok":
                        self.con.item("EXTRACT", f"row_{idx:09d}", True, f"{rid[:16]}….{ext} ({len(data) / 1024**2:.2f} MB)", 0.0)
                    else:
                        self.con.item("EXTRACT", f"row_{idx:09d}", status == "duplicate", f"{status}: {detail}", 0.0)
                    ledger.append(dict(id=idx, url=f"hf://{repo}/{rel}#{rid}", md5=md5s[k], size=len(data) if data else None,
                                       status=status, shard=state["shard"], error=detail or None))
                if meta_lines:
                    with open(meta_path, "a", encoding="utf-8") as fh:
                        fh.write("\n".join(meta_lines) + "\n")
                state["offset"] = start + len(rows)
                state["index"] += len(rows)
                for k2, sk in (("ok", "downloaded"), ("failed", "failed"), ("duplicate", "duplicates"), ("too_large", "too_large"), ("unsupported", "unsupported")):
                    state[sk] = state.get(sk, 0) + sum(1 for r in ledger if r["status"] == k2)
                with self.ws.db.session() as s:  # checkpoint: ledger rows and cursor together
                    for r in ledger:
                        s.merge(UrlItem(**r))
                    kv_set(s, STATE_KEY, {**state, "updated_at": utcnow().isoformat()})
                elapsed = time.time() - t0
                self.con.failed["download"] = stats["failed"] + stats["too_large"] + stats["unsupported"]
                self.con.progress("download", {"processed": stats["urls"], "total": max(stats["urls"] + 1, int(want_bytes / max(1, stats["bytes"] / max(1, stats["ok"])))) if stats["ok"] else stats["urls"] + 1,
                                               "computed": stats["urls"], "elapsed_s": elapsed, "rate_per_s": stats["urls"] / elapsed if elapsed > 0 else None,
                                               "eta_s": max(0.0, want_bytes - stats["bytes"]) / (stats["bytes"] / elapsed) if stats["bytes"] and elapsed > 0 else None})
                pos += n
                if stats["bytes"] >= want_bytes or (limit is not None and state["index"] >= limit):
                    stopped = state["offset"] < pf.metadata.num_rows
                    break
            if not stopped:  # this parquet file is fully consumed: move on and free the disk space
                state["file_idx"] += 1
                state["offset"] = 0
                del pf
                local.unlink(missing_ok=True)
                self.save(state)
                with self.prefetch.cv:
                    self.prefetch.hold = state["file_idx"] - 1  # its images are about to be processed
                    self.prefetch.cv.notify_all()
                break  # process this file's images right away instead of waiting for a whole shard's worth
        stats["exhausted"] = state["file_idx"] >= len(state["files"])
        return stats

    def _timed(self, url, size, md5, dest, ctx):
        t = time.time()
        status, detail, n = download_one(url, size, md5, dest, self.sh, ctx)
        return status, detail, n, time.time() - t

    # ---- pipeline on the current shard ---------------------------------
    def process(self, state: dict, retry_failed: bool) -> dict:
        from app.pipeline.runner import STAGE_NAMES, create_job, run_job

        folder = self.download_dir(state["shard"])
        cfg = self.cfg.model_copy(deep=True)
        cfg.paths.sources = [SourceCfg(path=str(folder), label=self.shard_name(state["shard"]),
                                       rights_status=state["rights_status"], license=state["license"],
                                       metadata_file="metadata.jsonl" if (folder / "metadata.jsonl").is_file() else None)]
        cfg = PipelineConfig.model_validate(cfg.model_dump())
        self.ws.save_config(cfg)
        self.cfg, self.sh = self.ws.config, self.ws.config.shards
        job_id = create_job(self.ws, [s for s in STAGE_NAMES if s != "hf_upload"])
        return run_job(self.ws, job_id, retry_failed=retry_failed, on_stage=self.con.stage, on_progress=self.con.progress,
                       on_stage_end=self.con.end, on_item=self.con.item)

    def output_mb(self) -> float:
        return hf_upload.upload_size(self.ws.output_dir)[0] / 1024**2

    def exported_records(self) -> int:
        with self.ws.db.session() as s:
            return (kv_get(s, "export:last", {}) or {}).get("exported", 0)

    # ---- close, upload, clean ------------------------------------------
    def finalize(self, state: dict) -> None:
        name = self.shard_name(state["shard"])
        size_mb, records = round(self.output_mb(), 1), self.exported_records()
        dest = self.ws.root / "shards" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(self.ws.output_dir), str(dest))
        self.con.progress("shards", {"message": f"{name} closed: {records} records, {size_mb} MB"})
        # Forget the per-image working state of this shard (the ledger of URLs is kept).
        with self.ws.db.session() as s:
            shas = list(s.scalars(select(Image.sha256)))
            for i in range(0, len(shas), 500):
                s.execute(delete(Embedding).where(Embedding.sha256.in_(shas[i:i + 500])))
            s.execute(delete(Image))
            s.execute(delete(Source))
            for key in ("export:last", "report:duplicates", "report:curation", "report:caption_diversity",
                        "report:split_leakage", "dedup:duplicate_of", "scan:last"):
                row = s.get(KV, key)
                if row is not None:
                    s.delete(row)
        shutil.rmtree(self.download_dir(state["shard"]), ignore_errors=True)  # our own downloaded copies
        shutil.rmtree(self.ws.cache_dir, ignore_errors=True)
        state["pending_upload"].append({"name": name, "records": records, "size_mb": size_mb})
        state["shard"] += 1
        self.save(state)
        self.upload_pending(state)

    def upload_pending(self, state: dict) -> None:
        hf = self.cfg.huggingface
        if not state["pending_upload"]:
            return
        if not (hf.enabled and hf.repo_id):
            self.con.progress("upload", {"message": f"{len(state['pending_upload'])} shard(s) kept locally in workspace/shards "
                                                    "(Hugging Face upload is off: set HF_TOKEN and HF_DATASET_REPO in .env)"})
            return
        for entry in list(state["pending_upload"]):
            folder = self.ws.root / "shards" / entry["name"]
            if not folder.is_dir():
                state["pending_upload"].remove(entry)
                continue
            done = state["shards_done"] + [entry]
            self.con.progress("upload", {"message": f"uploading {entry['name']} ({entry['records']} records, {entry['size_mb']} MB) "
                                                    f"to {hf.repo_id} ({'private' if hf.private else 'PUBLIC'})"})
            t0 = time.time()
            result = hf_upload._run_child({
                "repo_id": hf.repo_id, "private": hf.private, "folder": str(folder), "ignore": hf_upload.IGNORE, "mirrored": None,
                "path_in_repo": f"shards/{entry['name']}", "readme": self.root_readme(state, done),
                "message": f"Add {entry['name']}: {entry['records']} records, {entry['size_mb']} MB"})
            if not result.get("ok"):
                self.save(state)
                raise RuntimeError(f"upload of {entry['name']} failed: {result.get('error')} "
                                   "(the shard is kept in workspace/shards and is retried on the next run)")
            entry = {**entry, "uploaded_at": utcnow().isoformat(), "repo_url": result.get("repo_url")}
            state["pending_upload"] = [e for e in state["pending_upload"] if e["name"] != entry["name"]]
            state["shards_done"].append(entry)
            state["records_uploaded"] += entry["records"]
            if self.sh.delete_after_upload:
                shutil.rmtree(folder)
            self.save(state)
            self.con.progress("upload", {"message": f"{entry['name']} uploaded in {time.time() - t0:.0f}s: {result.get('repo_url')}"
                                                    + ("; local copy removed" if self.sh.delete_after_upload else "")})

    def root_readme(self, state: dict, done: list[dict]) -> str:
        total = sum(e["records"] for e in done)
        size = sum(e["size_mb"] for e in done)
        mode = self.cfg.project.mode
        task = {"text_to_image": "text-to-image", "vision_language": "image-to-text", "general_multimodal": "text-to-image"}[mode]
        lines = ["---", f"pretty_name: {json.dumps(self.cfg.project.name)}", "task_categories:", f"- {task}", "tags:",
                 "- image-dataset-pipeline", "---", "", f"# {self.cfg.project.name}", "",
                 f"Image-caption dataset built in shards by image-dataset-pipeline. **{total} records in {len(done)} shard(s), "
                 f"{size / 1024:.2f} GB.**", "",
                 "Each folder under `shards/` is a self-contained dataset: `images/`, `manifests/` (train / validation / test), "
                 "`metadata/`, `reports/` and its own `dataset_card.md` with filters, transformations and limitations.", "",
                 "## Source and licence", "",
                 f"Images were downloaded from the URLs in the builder's source lists. Licence as recorded by the builder: "
                 f"**{state['license']}**. Each record keeps its `source_url`. Licences were not verified by the tool, and per-image "
                 "author attribution is not included; check the original source before reuse.", "",
                 "## Caveats", "",
                 "- Captions, categories and quality scores are model-generated or heuristic; see each shard's dataset card.",
                 "- Duplicates are removed inside a shard and exact duplicates across shards; near-duplicates across shards and "
                 "across the train/validation/test splits of different shards are not checked.", "",
                 "## Shards", "", "| shard | records | size (MB) |", "|---|---:|---:|"]
        lines += [f"| {e['name']} | {e['records']} | {e['size_mb']} |" for e in done]
        return "\n".join(lines) + "\n"

    # ---- main loop ------------------------------------------------------
    def run(self, state: dict, retry_failed: bool = False) -> dict:
        lock = RunLock(self.ws.checkpoints_dir)
        if not lock.acquire():
            from app.pipeline.runner import AlreadyRunning

            raise AlreadyRunning(f"another pipeline process (pid {lock.owner()}) is running for this workspace")
        try:
            return self._loop(state, retry_failed)
        finally:
            if getattr(self, "prefetch", None) is not None:
                self.prefetch.stop()
            self.save(state)
            lock.release()

    def _loop(self, state: dict, retry_failed: bool) -> dict:
        target = self.target_mb()
        if state.get("kind") == "hf":
            # With a small --max-images limit there is no point downloading files ahead.
            window = 1 if state["limit_index"] is not None and state["limit_index"] - state["index"] < 1500 else self.sh.prefetch_files
            self.prefetch = Prefetcher(state, self.ws.root / "downloads" / "_source", window, max(60.0, self.sh.timeout_s),
                                       lambda m: self.con.progress("download", {"message": m}))
            self.prefetch.start()
            self.con.progress("download", {"message": f"background download on: up to {window} source file(s) kept on disk "
                                                      "(the one being processed + the next ones)"})
        self.upload_pending(state)  # shards closed earlier whose upload did not finish
        accept = 0.6  # share of downloaded bytes that ends up in the dataset; refined from this shard
        first = True
        allowed = self.cfg.rights.allowed_statuses
        if self.cfg.rights.require_approved_status_for_export and state["rights_status"] not in allowed:
            raise ValueError(f"shard mode needs a rights status that may be exported ({', '.join(allowed)}); "
                             "add --rights-status user_attested if you hold the rights to use these images")
        stalled = 0
        while True:
            limit = state["limit_index"]
            done_urls = state["file_idx"] >= len(state["files"]) or (limit is not None and state["index"] >= limit)
            have_downloads = self.download_dir(state["shard"]).is_dir() and any(self.download_dir(state["shard"]).glob("*.*"))
            size = self.output_mb() if self.ws.output_dir.is_dir() else 0.0
            if size >= target and self.exported_records():
                self.finalize(state)
                first = True
                continue
            if free_disk_gb(self.ws.root) < self.cfg.storage.min_free_disk_gb + 2 * target / 1024:
                raise OSError("not enough free disk space for another shard; free some space and run `image-pipeline resume`")
            round_stats = None
            if not done_urls:
                want = max(20 * 1024**2, (target - size) * 1024**2 * self.sh.oversample / accept)
                self.con.stage("download", "running")
                self.con.progress("download", {"message": f"{self.shard_name(state['shard'])}: output {size:.0f}/{target:.0f} MB, "
                                                          f"fetching about {want / 1024**2:.0f} MB of images "
                                                          f"(image {state['index'] + 1} of {limit or state['total_urls'] or '?'})"})
                round_stats = self.download_round(state, want)
                self.con.end("download", {"status": "completed", "kind": "dataset", "done": 0, "reused": 0, "failed": 0, "reason": None, "total": 0,
                                          "info": {k: round_stats[k] for k in ("urls", "ok", "gone", "failed", "duplicate", "too_large", "unsupported") if k in round_stats}
                                          | {"MB": round(round_stats["bytes"] / 1024**2, 1)}})
            elif not (first and have_downloads):
                break
            if round_stats is not None and round_stats["ok"] == 0 and not have_downloads:
                if round_stats["exhausted"] or (limit is not None and state["index"] >= limit):
                    break
                continue
            summary = self.process(state, retry_failed)
            if getattr(self, "prefetch", None) is not None:
                self.prefetch.release()  # this file is done: now the next one beyond the window may download
            first = False
            if summary["status"] != "completed":
                return {"status": summary["status"], "error": summary["error"], "state": state}
            got = self.output_mb()
            stalled = stalled + 1 if got <= size + 0.01 else 0
            if stalled >= 3:
                raise RuntimeError("three download rounds in a row added nothing to the dataset (everything was rejected, "
                                   "quarantined or unavailable); check `image-pipeline status` before continuing")
            with self.ws.db.session() as s:
                src_bytes = s.scalar(select(func.coalesce(func.sum(Image.file_size), 0)).where(Image.missing.is_(False))) or 0
            if src_bytes:
                accept = min(1.0, max(0.15, got * 1024**2 / src_bytes))
        leftover = self.output_mb() if self.ws.output_dir.is_dir() else 0.0
        return {"status": "completed", "error": None, "state": state, "open_shard_mb": round(leftover, 1), "target_mb": target}
