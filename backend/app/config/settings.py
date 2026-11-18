"""Validated pipeline configuration.

Every section forbids unknown keys and uses real numeric types, so placeholder
strings such as ``configurable`` are rejected instead of being silently
interpreted. Units and ranges are documented on each field.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app import PIPELINE_VERSION

Mode = Literal["text_to_image", "vision_language", "general_multimodal"]
RightsStatus = Literal["verified", "user_attested", "unknown", "restricted", "not_for_training"]
RIGHTS_STATUSES = ("verified", "user_attested", "unknown", "restricted", "not_for_training")
Toggle = Union[bool, Literal["auto"]]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ProjectCfg(_Section):
    name: str = "image-dataset-project"
    mode: Mode = "text_to_image"
    pipeline_version: str = PIPELINE_VERSION


class SourceCfg(_Section):
    """One user-authorised, read-only input folder."""

    path: str
    label: Optional[str] = Field(None, description="Stable name used to derive the source ID. Defaults to a hash of the folder path.")
    license: Optional[str] = None
    rights_status: Optional[RightsStatus] = None
    creator: Optional[str] = None
    attribution: Optional[str] = None
    source_url: Optional[str] = None
    notes: Optional[str] = None
    metadata_file: Optional[str] = Field(None, description="Optional CSV/JSON/JSONL with per-image rights, provenance and captions.")


class PathsCfg(_Section):
    sources: list[Union[SourceCfg, str]] = Field(default_factory=list)
    output_dir: str = "./processed_dataset"
    checkpoints_dir: str = "./checkpoints"
    logs_dir: str = "./logs"
    cache_dir: str = "./cache"

    def source_cfgs(self) -> list[SourceCfg]:
        return [s if isinstance(s, SourceCfg) else SourceCfg(path=s) for s in self.sources]


class ProcessingCfg(_Section):
    recursive_scan: bool = True
    preserve_originals: bool = True
    resume_enabled: bool = True
    follow_symlinks: bool = False
    max_retries: int = Field(2, ge=0, le=10, description="Extra attempts for a failed image/stage.")
    cpu_workers: int = Field(4, ge=1, le=64)
    gpu_batch_size: Union[int, Literal["auto"]] = Field("auto", description="Images per model batch, or 'auto'.")
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    allow_cpu_fallback: bool = Field(True, description="Run model stages on CPU when no accelerator is available.")
    db_chunk_size: int = Field(200, ge=10, le=5000, description="Images loaded per checkpointed chunk.")

    @field_validator("preserve_originals")
    @classmethod
    def _originals_are_read_only(cls, v: bool) -> bool:
        if not v:
            raise ValueError("preserve_originals must be true: source images are always read-only")
        return v

    @field_validator("gpu_batch_size")
    @classmethod
    def _batch(cls, v):
        if isinstance(v, int) and not (1 <= v <= 512):
            raise ValueError("gpu_batch_size must be 'auto' or between 1 and 512")
        return v


class StagesCfg(_Section):
    """Stage switches. 'auto' enables a stage when it is relevant to the mode and its model is available."""

    perceptual_hash: Toggle = "auto"
    technical_quality: Toggle = "auto"
    people: Toggle = "auto"
    ocr: Toggle = False
    segmentation: Toggle = False
    embeddings: Toggle = "auto"
    zero_shot: Toggle = "auto"
    object_detection: Toggle = False
    captioning: Toggle = "auto"
    alignment: Toggle = "auto"


class ImagesCfg(_Section):
    output_format: Literal["auto", "jpeg", "png", "webp"] = "auto"
    jpeg_quality: int = Field(95, ge=50, le=100)
    webp_quality: int = Field(95, ge=50, le=100)
    preserve_aspect_ratio: bool = True
    auto_upscale: bool = False
    preserve_alpha: bool = True
    alpha_background: tuple[int, int, int] = Field((255, 255, 255), description="RGB used only when preserve_alpha is false.")
    convert_to_srgb: bool = Field(True, description="Convert images carrying a non-sRGB ICC profile to sRGB.")
    copy_unchanged_when_possible: bool = Field(True, description="Byte-copy the source when no transformation is needed (avoids recompression).")
    strip_location_metadata: bool = Field(True, description="Re-encoded renditions never carry EXIF; byte copies with GPS EXIF are re-encoded when this is true.")
    max_side: Optional[int] = Field(None, ge=64, description="Downscale so the longer side is at most this many pixels. Never upscales.")
    resolution_buckets: list[int] = Field(default_factory=lambda: [512, 768, 1024], description="Base training resolutions (pixels, square-equivalent area).")
    bucket_step: int = Field(64, ge=8, le=256, description="Bucket dimensions are multiples of this.")

    @field_validator("alpha_background")
    @classmethod
    def _rgb(cls, v):
        if any(not (0 <= c <= 255) for c in v):
            raise ValueError("alpha_background must be three 0-255 integers")
        return v

    @field_validator("resolution_buckets")
    @classmethod
    def _buckets(cls, v):
        if not v or any(b < 64 or b > 8192 for b in v):
            raise ValueError("resolution_buckets must contain values between 64 and 8192")
        return sorted(set(v))

    @model_validator(mode="after")
    def _aspect(self):
        if not self.preserve_aspect_ratio:
            raise ValueError("preserve_aspect_ratio=false (forced cropping/stretching) is not supported")
        if self.auto_upscale:
            raise ValueError("auto_upscale is not implemented; images are never upscaled")
        return self


class QualityWeights(_Section):
    """Relative weights of the component scores (each 0-100) in the overall score."""

    sharpness: float = Field(0.25, ge=0)
    noise: float = Field(0.10, ge=0)
    compression: float = Field(0.10, ge=0)
    exposure: float = Field(0.15, ge=0)
    contrast: float = Field(0.10, ge=0)
    resolution: float = Field(0.10, ge=0)
    composition: float = Field(0.05, ge=0)
    aesthetic: float = Field(0.15, ge=0)

    @model_validator(mode="after")
    def _nonzero(self):
        if sum(self.model_dump().values()) <= 0:
            raise ValueError("at least one quality weight must be positive")
        return self


class QualityCfg(_Section):
    min_width: int = Field(512, ge=1, description="Pixels, after EXIF orientation.")
    min_height: int = Field(512, ge=1)
    min_overall_score: float = Field(65, ge=0, le=100, description="Below this (and below the review band) an image is rejected.")
    min_aesthetic_score: Optional[float] = Field(None, ge=0, le=100)
    manual_review_band: tuple[float, float] = Field((55, 75), description="Overall scores inside [low, high] go to manual review.")
    weights: QualityWeights = Field(default_factory=QualityWeights)
    review_min_aesthetic_score: Optional[float] = Field(10, ge=0, le=100, description="Images below this aesthetic signal are quarantined for review even when the overall score passes.")
    reject_watermarked: bool = Field(True, description="Quarantine images with detected watermark text instead of only flagging them.")
    max_text_area_fraction: Optional[float] = Field(None, ge=0, le=1, description="Quarantine text-heavy images above this OCR area fraction.")

    @field_validator("manual_review_band")
    @classmethod
    def _band(cls, v):
        lo, hi = v
        if not (0 <= lo <= hi <= 100):
            raise ValueError("manual_review_band must satisfy 0 <= low <= high <= 100")
        return v


class DedupCfg(_Section):
    exact: bool = True
    perceptual: bool = True
    near: bool = True
    semantic: bool = True
    auto_delete: bool = False
    perceptual_threshold: int = Field(6, ge=0, le=32, description="Max pHash Hamming distance (0-64 bits) for a perceptual duplicate.")
    near_candidate_hamming: int = Field(24, ge=0, le=40, description="Max pHash distance for a near-duplicate candidate pair.")
    near_candidate_cosine: float = Field(0.80, ge=0, le=1, description="Min embedding cosine for a near-duplicate candidate pair.")
    near_max_candidates_per_image: int = Field(8, ge=1, le=50, description="Most similar images geometrically verified per image.")
    near_min_inliers: int = Field(25, ge=4, description="Min geometrically verified ORB matches to confirm a near duplicate.")
    near_min_inlier_ratio: float = Field(0.25, ge=0, le=1)
    semantic_threshold: float = Field(0.93, ge=0, le=1, description="Min embedding cosine for a semantic-similarity group.")
    non_representative_action: Literal["quarantine", "flag"] = Field("quarantine", description="Applied to exact/perceptual/near duplicates that are not the representative.")
    max_per_semantic_group: Optional[int] = Field(None, ge=1, description="Keep at most this many images per semantic group; others are quarantined.")
    cross_shard: bool = Field(True, description="Shard mode: also compare against the hashes of images exported in earlier shards.")

    @field_validator("auto_delete")
    @classmethod
    def _never_delete(cls, v):
        if v:
            raise ValueError("auto_delete is not supported: duplicates are quarantined or flagged, never deleted")
        return v


class CaptioningCfg(_Section):
    adapter: Literal["none", "template", "sidecar", "smolvlm", "auto"] = Field(
        "template",
        description="none | template (grounded baseline built from annotations) | sidecar (user-supplied captions) | smolvlm (local VLM) "
                    "| auto (the caption that came with the source when there is one, otherwise the local VLM).",
    )
    supplied_source: str = Field("user_supplied", description="Recorded as the caption source for captions that came with the source data.")
    quarantine_failed_verification: bool = Field(True, description="Quarantine images whose caption failed verification.")
    vlm_image_splitting: bool = Field(False, description="auto adapter: let the VLM look at the image as a grid of tiles. More detail, "
                                                         "but roughly ten times the input and far slower on small machines.")
    short_caption: bool = True
    detailed_caption: bool = True
    comprehensive_caption: bool = True
    generation_prompt: bool = True
    alternate_caption_count: int = Field(0, ge=0, le=5)
    verify_captions: bool = True
    strictness: Literal["lenient", "standard", "strict"] = "standard"
    max_new_tokens: int = Field(160, ge=16, le=1024)
    sidecar_extension: str = Field(".txt", description="Caption file next to each image for the sidecar adapter.")
    training_caption_field: Literal["short", "detailed", "comprehensive", "generation_prompt"] = "detailed"
    review_template_captions: bool = Field(True, description="Send template-baseline captions to manual review.")


class ModelsCfg(_Section):
    embedding: str = Field("clip-vit-base-patch32", description="Key from configs/model_registry.yaml, or 'none'.")
    captioner: str = Field("smolvlm2-256m", description="Registry key used by the smolvlm caption adapter.")
    ocr: Literal["none", "rapidocr"] = "rapidocr"
    segmentation: Literal["none", "grabcut_saliency"] = "grabcut_saliency"
    object_detector: str = Field("none", description="Registry key of a zero-shot detector, or 'none'.")
    detection_labels: list[str] = Field(default_factory=list, description="Open-vocabulary labels for the detector; defaults to a built-in list.")


class RightsCfg(_Section):
    require_approved_status_for_export: bool = True
    allowed_statuses: list[RightsStatus] = Field(default_factory=lambda: ["verified", "user_attested"])
    default_status: RightsStatus = "unknown"


class SplitsCfg(_Section):
    train: float = Field(0.90, ge=0, le=1)
    validation: float = Field(0.05, ge=0, le=1)
    test: float = Field(0.05, ge=0, le=1)
    seed: int = 42
    group_near_duplicates: bool = True
    group_semantic: bool = False
    group_by_source_directory: bool = False

    @model_validator(mode="after")
    def _sum(self):
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0 (got {total:.4f})")
        if self.train <= 0:
            raise ValueError("train ratio must be > 0")
        return self


class CurationCfg(_Section):
    max_per_category: Optional[int] = Field(None, ge=1, description="Optional cap per primary category (highest quality kept). Off by default.")
    exclude_categories: list[str] = Field(default_factory=list)


class ExportCfg(_Section):
    formats: list[Literal["jsonl", "csv", "parquet"]] = Field(default_factory=lambda: ["jsonl", "csv"])
    include_processed_images: bool = True
    remote_upload_enabled: bool = False
    minimal_manifest: Literal["hf_imagefolder", "caption_txt", "both", "none"] = "hf_imagefolder"
    vlm_qa: bool = Field(False, description="Add simple grounded question-answer pairs to the vision-language manifest.")

    @field_validator("remote_upload_enabled")
    @classmethod
    def _no_upload(cls, v):
        if v:
            raise ValueError("use the 'huggingface' section to enable uploading; this legacy switch must stay false")
        return v

    @model_validator(mode="after")
    def _images(self):
        if not self.include_processed_images:
            raise ValueError("include_processed_images must be true: a training dataset needs the image files")
        return self


class HuggingFaceCfg(_Section):
    """Optional upload of the finished dataset to a Hugging Face dataset repository. Off by default.
    The access token is never stored here: use `hf auth login` or the HF_TOKEN environment variable."""

    enabled: bool = False
    repo_id: Optional[str] = Field(None, description="username/dataset-name, or just dataset-name (your own account is used).")
    private: bool = Field(False, description="false = public repository (anyone can view and download).")
    min_size_mb: float = Field(1024, ge=0, description="Upload only once the output folder is at least this large (MB).")
    require_validation_passed: bool = True
    commit_message: str = "Update dataset"

    @field_validator("repo_id")
    @classmethod
    def _repo(cls, v):
        import re

        if v is not None and not re.fullmatch(r"([A-Za-z0-9][\w.-]*/)?[A-Za-z0-9][\w.-]*", v):
            raise ValueError("repo_id must be dataset-name or username/dataset-name (letters, digits, - _ . only)")
        return v


class ShardsCfg(_Section):
    """Shard mode (input is a folder of URL lists): download, process, close a shard at the size
    threshold, upload it, remove the local copies, continue."""

    target_mb: Optional[float] = Field(None, gt=0, description="Shard size in MB. Default: huggingface.min_size_mb.")
    download_workers: int = Field(8, ge=1, le=64)
    timeout_s: float = Field(30, gt=0)
    retries: int = Field(2, ge=0, le=10)
    max_file_mb: float = Field(40, gt=0, description="Listed files larger than this are skipped.")
    oversample: float = Field(1.15, ge=1, le=5, description="Safety factor when estimating how much to download for a shard.")
    delete_after_upload: bool = Field(True, description="Remove the local shard once Hugging Face has it.")
    prefetch_files: int = Field(3, ge=1, le=10, description="Hugging Face sources: most source files kept on disk at once "
                                                           "(the one being processed plus the ones downloaded ahead in the background).")
    license_note: str = "CC BY 2.0 as listed by the source (not verified)"
    caption_dataset: Optional[str] = Field(None, description="Hugging Face dataset with image_url and caption columns; its captions are "
                                                             "joined to the source images by sha256(image_url), e.g. allenai/pixmo-cap.")


class StorageCfg(_Section):
    min_free_disk_gb: float = Field(2.0, ge=0, description="Abort image writing when free space would drop below this.")
    max_output_gb: Optional[float] = Field(None, gt=0, description="Upper bound for the processed image layer.")
    thumbnail_size: int = Field(384, ge=64, le=1024)


class PipelineConfig(_Section):
    project: ProjectCfg = Field(default_factory=ProjectCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)
    processing: ProcessingCfg = Field(default_factory=ProcessingCfg)
    stages: StagesCfg = Field(default_factory=StagesCfg)
    images: ImagesCfg = Field(default_factory=ImagesCfg)
    quality: QualityCfg = Field(default_factory=QualityCfg)
    deduplication: DedupCfg = Field(default_factory=DedupCfg)
    captioning: CaptioningCfg = Field(default_factory=CaptioningCfg)
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    rights: RightsCfg = Field(default_factory=RightsCfg)
    splits: SplitsCfg = Field(default_factory=SplitsCfg)
    curation: CurationCfg = Field(default_factory=CurationCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    huggingface: HuggingFaceCfg = Field(default_factory=HuggingFaceCfg)
    shards: ShardsCfg = Field(default_factory=ShardsCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)

    # ---- helpers -----------------------------------------------------
    def public_dict(self) -> dict[str, Any]:
        """Configuration without machine-specific paths (safe for reports and hashing)."""
        d = self.model_dump(mode="json")
        d["paths"] = {"sources": len(d["paths"]["sources"])}
        return d

    def config_hash(self) -> str:
        d = self.public_dict()
        d.pop("huggingface", None)  # where the dataset is published does not change how it was built
        d.pop("shards", None)
        return fingerprint(d)



def fingerprint(obj: Any) -> str:
    """Stable short hash of any JSON-serialisable object."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: Optional[Path]) -> PipelineConfig:
    if path is None or not Path(path).exists():
        return PipelineConfig()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: configuration root must be a mapping")
    return PipelineConfig.model_validate(raw)


def save_config(cfg: PipelineConfig, path: Path) -> None:
    from app.core.io import atomic_write_text

    text = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    atomic_write_text(Path(path), text)
