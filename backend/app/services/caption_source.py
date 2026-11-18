"""Captions that come with the source: a Hugging Face dataset of ``image_url`` + ``caption`` rows,
joined to the source images by ``sha256(image_url)`` (the image id used by PixMo image caches).

The caption dataset is downloaded once and turned into a local lookup table; after that a
lookup is a primary-key read. Nothing here needs a model."""
from __future__ import annotations

import hashlib
import sqlite3
import zlib
from pathlib import Path
from typing import Callable, Iterable

from app.core.logging import get_logger

log = get_logger("captions")


class CaptionIndex:
    def __init__(self, root: Path, dataset: str, say: Callable[[str], None] = lambda m: None):
        self.dataset, self.say = dataset, say
        self.dir = Path(root) / "sources" / dataset.replace("/", "__")
        self.path = self.dir / "captions.sqlite"
        self._con = None

    def _open(self) -> sqlite3.Connection:
        if self._con is None:
            self.dir.mkdir(parents=True, exist_ok=True)
            self._con = sqlite3.connect(self.path)
            self._con.execute("create table if not exists captions (id text primary key, url text, caption blob) without rowid")
            self._con.execute("create table if not exists files (name text primary key, n integer)")
        return self._con

    def ready(self) -> bool:
        return self.path.is_file() and self._open().execute("select 1 from files where name = '__complete__'").fetchone() is not None

    def count(self) -> int:
        return self._open().execute("select count(*) from captions").fetchone()[0]

    def ensure(self, ctx, timeout: float = 60.0) -> int:
        """Download and index the caption dataset if that has not been done yet. Safe to interrupt:
        files already indexed are skipped on the next call."""
        if self.ready():
            return self.count()
        import pyarrow.parquet as pq

        from app.services.shards import download_file, hf_dataset_info

        con = self._open()
        files = hf_dataset_info(self.dataset, ctx)["files"]
        done = {name for (name,) in con.execute("select name from files")}
        self.say(f"captions: preparing {self.dataset} ({len(files)} file(s)); this happens once")
        for rel in files:
            if rel in done:
                continue
            local = self.dir / rel.replace("/", "__")
            download_file(f"https://huggingface.co/datasets/{self.dataset}/resolve/main/{rel}", local, ctx, timeout,
                          lambda m: self.say("captions: " + m))
            pf = pq.ParquetFile(local)
            names = pf.schema_arrow.names
            if "image_url" not in names or "caption" not in names:
                raise ValueError(f"{self.dataset} needs image_url and caption columns (found: {', '.join(names)})")
            n = 0
            for batch in pf.iter_batches(batch_size=5000, columns=["image_url", "caption"]):
                rows = [(hashlib.sha256(u.encode("utf-8")).hexdigest(), u, zlib.compress(c.encode("utf-8")))
                        for u, c in zip(batch.column("image_url").to_pylist(), batch.column("caption").to_pylist()) if u and c and c.strip()]
                con.executemany("insert or ignore into captions values (?, ?, ?)", rows)
                n += len(rows)
            con.execute("insert or replace into files values (?, ?)", (rel, n))
            con.commit()
            del pf
            local.unlink(missing_ok=True)
            self.say(f"captions: indexed {rel} ({n} captions)")
        con.execute("insert or replace into files values ('__complete__', 0)")
        con.commit()
        total = self.count()
        self.say(f"captions: {total} captions from {self.dataset} are ready")
        return total

    def lookup(self, ids: Iterable[str]) -> dict[str, tuple[str, str]]:
        """{image id: (caption, original image url)} for the ids that have a caption."""
        ids = [i for i in ids if i]
        out: dict[str, tuple[str, str]] = {}
        con = self._open()
        for i in range(0, len(ids), 500):
            part = ids[i:i + 500]
            for rid, url, blob in con.execute(f"select id, url, caption from captions where id in ({','.join('?' * len(part))})", part):
                out[rid] = (zlib.decompress(blob).decode("utf-8"), url)
        return out

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
