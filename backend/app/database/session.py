"""Engine/session factory. One database per workspace."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import DB_SCHEMA_VERSION, KV, Base


class Database:
    def __init__(self, db_path: Path):
        url = os.environ.get("IMAGE_PIPELINE_DATABASE_URL")
        if url:
            self.engine: Engine = create_engine(url, future=True)
        else:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                f"sqlite:///{db_path}", future=True, connect_args={"check_same_thread": False, "timeout": 60}
            )

            @event.listens_for(self.engine, "connect")
            def _pragmas(dbapi_conn, _):
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA synchronous=NORMAL")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.execute("PRAGMA busy_timeout=60000")
                cur.close()

        self._factory = sessionmaker(self.engine, expire_on_commit=False, future=True)
        self.migrate()

    def migrate(self) -> None:
        """Create tables and record the schema version. Later versions add stepwise
        migrations here keyed on the stored version."""
        Base.metadata.create_all(self.engine)
        with self.session() as s:
            row = s.get(KV, "db_schema_version")
            if row is None:
                s.add(KV(key="db_schema_version", value=DB_SCHEMA_VERSION))
            elif row.value > DB_SCHEMA_VERSION:
                raise RuntimeError(
                    f"workspace database schema v{row.value} is newer than this application (v{DB_SCHEMA_VERSION})"
                )

    @contextmanager
    def session(self) -> Iterator[Session]:
        s = self._factory()
        try:
            yield s
            s.commit()
        except BaseException:
            s.rollback()
            raise
        finally:
            s.close()

    def dispose(self) -> None:
        self.engine.dispose()


def kv_get(s: Session, key: str, default=None):
    row = s.get(KV, key)
    return default if row is None else row.value


def kv_set(s: Session, key: str, value) -> None:
    row = s.get(KV, key)
    if row is None:
        s.add(KV(key=key, value=value))
    else:
        row.value = value
