"""SQLAlchemy models. SQLite is the default engine; nothing here is SQLite-specific,
so a PostgreSQL URL can be supplied later through ``IMAGE_PIPELINE_DATABASE_URL``."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import (JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, LargeBinary,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

DB_SCHEMA_VERSION = 1


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Source(Base):
    __tablename__ = "sources"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_id: Mapped[str] = mapped_column(String(40), unique=True)
    label: Mapped[str] = mapped_column(String(200))
    root_path: Mapped[str] = mapped_column(Text)  # private: never exported or sent to the frontend
    license: Mapped[Optional[str]] = mapped_column(String(200))
    rights_status: Mapped[str] = mapped_column(String(32), default="unknown")
    creator: Mapped[Optional[str]] = mapped_column(String(300))
    attribution: Mapped[Optional[str]] = mapped_column(Text)
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    metadata_file: Mapped[Optional[str]] = mapped_column(Text)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_scan_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    images: Mapped[list["Image"]] = relationship(back_populates="source")


class Image(Base):
    __tablename__ = "images"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[str] = mapped_column(String(40), unique=True)
    source_pk: Mapped[int] = mapped_column(ForeignKey("sources.id"), index=True)
    abs_path: Mapped[str] = mapped_column(Text)  # private
    relative_path: Mapped[str] = mapped_column(Text)
    file_name: Mapped[str] = mapped_column(Text)
    extension: Mapped[str] = mapped_column(String(16))
    file_size: Mapped[int] = mapped_column(Integer)
    mtime_ns: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    missing: Mapped[bool] = mapped_column(Boolean, default=False)
    seen_token: Mapped[Optional[str]] = mapped_column(String(16))

    # validation + metadata (stage: validate)
    valid: Mapped[Optional[bool]] = mapped_column(Boolean, index=True)
    validation_status: Mapped[Optional[str]] = mapped_column(String(32))
    validation_reason: Mapped[Optional[str]] = mapped_column(Text)
    width: Mapped[Optional[int]] = mapped_column(Integer)
    height: Mapped[Optional[int]] = mapped_column(Integer)
    format: Mapped[Optional[str]] = mapped_column(String(16))
    color_mode: Mapped[Optional[str]] = mapped_column(String(16))
    aspect_ratio: Mapped[Optional[float]] = mapped_column(Float)
    orientation: Mapped[Optional[str]] = mapped_column(String(16))
    has_alpha: Mapped[Optional[bool]] = mapped_column(Boolean)

    # rights / provenance (per-image values override the source defaults)
    license: Mapped[Optional[str]] = mapped_column(String(200))
    rights_status: Mapped[Optional[str]] = mapped_column(String(32))
    creator: Mapped[Optional[str]] = mapped_column(String(300))
    attribution: Mapped[Optional[str]] = mapped_column(Text)
    source_url: Mapped[Optional[str]] = mapped_column(Text)
    notes: Mapped[Optional[str]] = mapped_column(Text)
    supplied_caption: Mapped[Optional[str]] = mapped_column(Text)

    # denormalised summaries for filtering
    category: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    style: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    quality_score: Mapped[Optional[float]] = mapped_column(Float)
    aesthetic_score: Mapped[Optional[float]] = mapped_column(Float)
    alignment_score: Mapped[Optional[float]] = mapped_column(Float)
    short_caption: Mapped[Optional[str]] = mapped_column(Text)

    # deduplication
    exact_group: Mapped[Optional[str]] = mapped_column(String(40), index=True)
    perceptual_group: Mapped[Optional[str]] = mapped_column(String(40), index=True)
    near_group: Mapped[Optional[str]] = mapped_column(String(40), index=True)
    semantic_group: Mapped[Optional[str]] = mapped_column(String(40), index=True)
    representative_status: Mapped[Optional[str]] = mapped_column(String(32))
    dup_override: Mapped[Optional[str]] = mapped_column(String(32))  # user: keep | representative

    # curation
    decision: Mapped[str] = mapped_column(String(16), default="pending", index=True)
    auto_decision: Mapped[Optional[str]] = mapped_column(String(16))
    manual_decision: Mapped[Optional[str]] = mapped_column(String(16))
    decision_reasons: Mapped[list] = mapped_column(JSON, default=list)
    review_status: Mapped[str] = mapped_column(String(16), default="not_required", index=True)
    review_reasons: Mapped[list] = mapped_column(JSON, default=list)
    rights_eligible: Mapped[Optional[bool]] = mapped_column(Boolean)
    split: Mapped[Optional[str]] = mapped_column(String(16), index=True)
    split_group: Mapped[Optional[str]] = mapped_column(String(40))

    # processed rendition
    processed_path: Mapped[Optional[str]] = mapped_column(Text)  # relative to output_dir
    processed_sha256: Mapped[Optional[str]] = mapped_column(String(64))

    user_edits: Mapped[dict] = mapped_column(JSON, default=dict)  # caption/tag overrides made in review

    source: Mapped[Source] = relationship(back_populates="images")
    results: Mapped[list["StageResult"]] = relationship(back_populates="image", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("source_pk", "relative_path", name="uq_source_relpath"),
        Index("ix_images_quality", "quality_score"),
    )

    def effective_rights(self) -> str:
        return self.rights_status or (self.source.rights_status if self.source else None) or "unknown"


class StageResult(Base):
    """Per-image, per-stage checkpoint. ``idem_key`` encodes checksum, stage version, model
    version, relevant configuration and upstream keys; a matching key means the work is reusable."""

    __tablename__ = "stage_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_pk: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    stage: Mapped[str] = mapped_column(String(40))
    idem_key: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))  # done | failed
    data: Mapped[Optional[dict]] = mapped_column(JSON)
    error: Mapped[Optional[str]] = mapped_column(Text)
    error_category: Mapped[Optional[str]] = mapped_column(String(32))
    attempts: Mapped[int] = mapped_column(Integer, default=1)
    model_info: Mapped[Optional[dict]] = mapped_column(JSON)
    duration_ms: Mapped[Optional[float]] = mapped_column(Float)
    job_id: Mapped[Optional[str]] = mapped_column(String(40))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    image: Mapped[Image] = relationship(back_populates="results")
    __table_args__ = (
        UniqueConstraint("image_pk", "stage", name="uq_image_stage"),
        Index("ix_stage_status", "stage", "status"),
    )


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="queued")  # queued|running|completed|failed|cancelled|interrupted
    mode: Mapped[str] = mapped_column(String(32))
    config: Mapped[dict] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(16))
    stages: Mapped[list] = mapped_column(JSON, default=list)
    current_stage: Mapped[Optional[str]] = mapped_column(String(40))
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[Optional[str]] = mapped_column(Text)
    summary: Mapped[Optional[dict]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    stage_runs: Mapped[list["JobStage"]] = relationship(back_populates="job", cascade="all, delete-orphan", order_by="JobStage.id")


class JobStage(Base):
    __tablename__ = "job_stages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), index=True)
    stage: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(16), default="pending")  # pending|running|completed|skipped|failed|cancelled
    reason: Mapped[Optional[str]] = mapped_column(Text)
    total: Mapped[int] = mapped_column(Integer, default=0)
    done: Mapped[int] = mapped_column(Integer, default=0)
    reused: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    info: Mapped[Optional[dict]] = mapped_column(JSON)
    job: Mapped[Job] = relationship(back_populates="stage_runs")


class Embedding(Base):
    """Cache of image embeddings keyed by content checksum and model."""

    __tablename__ = "embeddings"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64))
    model_key: Mapped[str] = mapped_column(String(120))
    dim: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)  # float32, L2-normalised
    __table_args__ = (UniqueConstraint("sha256", "model_key", name="uq_embedding"),)


class SeenHash(Base):
    """Hashes of images exported in shards that are already closed. The images themselves are gone
    from the workspace by then; this ledger is what later shards are compared against."""

    __tablename__ = "seen_hashes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    phash: Mapped[Optional[str]] = mapped_column(String(16))
    dhash: Mapped[Optional[str]] = mapped_column(String(16))
    image_id: Mapped[str] = mapped_column(String(40))
    shard: Mapped[str] = mapped_column(String(40))


class DecisionLog(Base):
    """Audit trail of automatic and manual decisions."""

    __tablename__ = "decision_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    job_id: Mapped[Optional[str]] = mapped_column(String(40))
    image_id: Mapped[Optional[str]] = mapped_column(String(40), index=True)
    actor: Mapped[str] = mapped_column(String(16))  # auto | user
    action: Mapped[str] = mapped_column(String(48))
    detail: Mapped[Optional[dict]] = mapped_column(JSON)


class UrlItem(Base):
    """Ledger of URLs consumed in shard mode (kept after a shard's working data is cleaned up)."""

    __tablename__ = "url_items"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)  # position in the URL lists
    url: Mapped[str] = mapped_column(Text)
    md5: Mapped[Optional[str]] = mapped_column(String(32), index=True)
    size: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), index=True)  # ok | gone | failed | duplicate | too_large
    shard: Mapped[Optional[int]] = mapped_column(Integer)
    error: Mapped[Optional[str]] = mapped_column(Text)


class KV(Base):
    __tablename__ = "kv"
    key: Mapped[str] = mapped_column(String(120), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
