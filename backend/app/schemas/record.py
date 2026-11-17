"""Versioned training-record schema (specification section 4). Every exported record is
validated against these models; invalid records are rejected and reported."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app import SCHEMA_VERSION


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceInfo(_M):
    source_id: str
    relative_path: str
    sha256: str = Field(min_length=64, max_length=64)
    source_url: Optional[str] = None
    creator: Optional[str] = None
    license: Optional[str] = None
    rights_status: Literal["verified", "user_attested", "unknown", "restricted", "not_for_training"]
    attribution: Optional[str] = None

    @field_validator("relative_path")
    @classmethod
    def _relative(cls, v: str) -> str:
        if v.startswith(("/", "\\")) or (len(v) > 1 and v[1] == ":") or ".." in v.split("/"):
            raise ValueError("relative_path must be relative")
        return v


class ProcessedImage(_M):
    format: Literal["JPEG", "PNG", "WEBP"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    aspect_ratio: float = Field(gt=0)
    sha256: str = Field(min_length=64, max_length=64)
    is_upscaled: bool
    transformations: list[str]
    bucket: Optional[dict] = None


class Captions(_M):
    short: str = ""
    detailed: str = ""
    comprehensive: str = ""
    generation_prompt: str = ""
    alternatives: list[str] = Field(default_factory=list)
    source: Optional[str] = None
    edited_by_user: bool = False


class Inferred(_M):
    value: Optional[Any] = None
    status: Literal["exif", "inferred", "unknown", "user_supplied"]
    confidence: Optional[float] = None


class VisualAnnotations(_M):
    categories: list[str] = Field(default_factory=list)
    objects: list[dict] = Field(default_factory=list)
    scene: Optional[str] = None
    relationships: list[dict] = Field(default_factory=list)
    lighting: list[str] = Field(default_factory=list)
    style: list[str] = Field(default_factory=list)
    color_palette: list[dict] = Field(default_factory=list)
    composition: dict = Field(default_factory=dict)
    camera_attributes: dict[str, Inferred] = Field(default_factory=dict)
    ocr: dict = Field(default_factory=lambda: {"text": [], "regions": []})
    watermarks_detected: Optional[bool] = None
    people: dict = Field(default_factory=dict)
    segmentation: Optional[dict] = None
    tags: list[str] = Field(default_factory=list)
    annotation_status: Literal["inferred", "not_computed"] = "inferred"


class Quality(_M):
    overall_score: Optional[float] = None
    sharpness_score: Optional[float] = None
    noise_score: Optional[float] = None
    exposure_score: Optional[float] = None
    contrast_score: Optional[float] = None
    compression_score: Optional[float] = None
    composition_score: Optional[float] = None
    aesthetic_score: Optional[float] = None
    resolution_status: Optional[str] = None
    flags: list[str] = Field(default_factory=list)


class Alignment(_M):
    score: Optional[float] = None
    raw_cosine: Optional[float] = None
    method: Optional[str] = None
    verification_status: str = "not_run"
    warnings: list[str] = Field(default_factory=list)
    calibrated: bool = False


class Deduplication(_M):
    exact_duplicate_group: Optional[str] = None
    perceptual_duplicate_group: Optional[str] = None
    near_duplicate_group: Optional[str] = None
    semantic_similarity_group: Optional[str] = None
    representative_status: Optional[str] = None


class Processing(_M):
    status: str
    model_versions: dict
    pipeline_version: str
    config_hash: str
    processed_at: str


class Curation(_M):
    decision: Literal["accepted", "rejected", "quarantined", "pending"]
    review_status: str
    rejection_reasons: list[str] = Field(default_factory=list)
    split: Optional[Literal["train", "validation", "test"]] = None
    split_seed: Optional[int] = None


class TrainingRecord(_M):
    schema_version: str = SCHEMA_VERSION
    image_id: str
    image_path: str
    source: SourceInfo
    processed_image: ProcessedImage
    captions: Captions
    visual_annotations: VisualAnnotations
    quality: Quality
    alignment: Alignment
    deduplication: Deduplication
    processing: Processing
    curation: Curation

    @field_validator("image_path")
    @classmethod
    def _rel(cls, v: str) -> str:
        if v.startswith(("/", "\\")) or (len(v) > 1 and v[1] == ":") or ".." in v.split("/"):
            raise ValueError("image_path must be relative to the dataset root")
        return v
