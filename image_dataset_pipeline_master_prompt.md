# Production-Grade AI Image Dataset Curation & Preparation Pipeline

> **Master implementation prompt for Claude Code, Cursor, Codex, or another coding agent**  
> Primary objective: convert user-provided image folders into a carefully curated, quality-controlled, reproducible, training-ready image dataset for text-to-image generation models.  
> Secondary objective: retain structured visual annotations that can also support image captioning and vision-language model (VLM) training.

---

## 0. Instructions to the Coding Agent

Act as a senior machine-learning engineer, computer-vision researcher, data engineer, backend architect, and full-stack developer.

Build a real, modular, testable application—not a mock-up or a collection of placeholder functions. Start by inspecting the repository and the execution environment. Then propose a concise implementation plan, create the project structure, and implement the system in small verified phases.

### Non-negotiable engineering rules

1. **Never modify, overwrite, move, or delete source images.** Treat all user-provided input folders as read-only.
2. **Save actual image files in the processed dataset.** Metadata and captions alone are not a usable conventional text-to-image training dataset.
3. Keep source images, processed image renditions, annotations, reports, and training manifests logically separate.
4. Do not silently delete duplicates or low-quality images. Quarantine or mark them and provide review controls.
5. Do not invent visual facts. Store uncertain predictions as uncertain/inferred, not verified facts.
6. Do not claim camera/lens settings, location, identity, time, or provenance unless supported by EXIF or a reliable supplied source.
7. Use model adapters so models can be changed without rewriting pipeline logic.
8. Do not download large model weights automatically. Show model name, size, license, expected VRAM/RAM, and download action before obtaining them.
9. Make every stage resumable and idempotent. A crash must not force completed images to be processed again.
10. Preserve image-caption pairing and prevent duplicate/near-duplicate leakage across train, validation, and test splits.
11. Validate output schemas and image files before marking any record training-ready.
12. Log decisions, model versions, configurations, thresholds, errors, and timestamps so every output can be reproduced.
13. Prefer local inference when practical, while allowing optional external inference providers through explicit configuration.
14. Do not expose API keys, tokens, or private paths in logs, reports, or the frontend.
15. Do not implement any functionality that bypasses access controls or source-site restrictions. Input is a folder of images the user is authorized to process.

---

# 1. Product Goals and Operating Modes

Build an **AI Image Dataset Curation & Preparation System**. Its default profile is **Text-to-Image Dataset Mode**.

Support these configurable modes:

### Mode A — Text-to-Image (primary)

Create high-quality image-and-caption pairs. Prioritize image fidelity, accurate detailed captions, image-text alignment, visual diversity, composition, aesthetic quality, and resolution suitability. Export actual processed image files plus captions and manifests for a compatible training workflow.

### Mode B — Vision-Language / Image Understanding

Create image paths plus short captions, detailed captions, structured visual attributes, objects, scene descriptions, and optional question-answer annotations. Do not assume these annotations are automatically suitable for every VLM training recipe; make the export schema configurable.

### Mode C — General Multimodal Dataset

Retain reusable visual annotations and export separate task-specific manifests for different model families.

The user must be able to select a mode before processing. Avoid running every expensive model on every image when a stage is irrelevant to the selected mode.

## Primary success criteria

- Accurate, informative, non-hallucinated captions.
- Strong image-text alignment.
- High visual and technical image quality.
- A diverse dataset without excessive near-duplicates or category imbalance.
- Actual image files saved in the processed dataset.
- Reproducible train/validation/test splits.
- Recoverable processing with checkpoints.
- Clear rejected/quarantined records and reasons.
- Exportable datasets with validated paths and schemas.

Do not optimize for the highest raw image count. Optimize for useful, legally usable, diverse, well-described image samples.

---

# 2. Input and Source Management

Allow users to select one or more local image folders through the UI or CLI.

Support:

- Single folders and multiple folders.
- Nested directories and recursive scanning.
- Large collections and incremental additions.
- Common formats: JPEG, JPG, PNG, WEBP, BMP, TIFF, and formats supported by the chosen decoder.
- Optional user-provided CSV/JSON metadata and source manifests.

For every discovered image, register:

- Stable `image_id`.
- Original absolute path (private/internal only).
- Relative source path.
- File name and extension.
- File size.
- SHA-256 checksum.
- Source folder / source collection ID.
- Import timestamp.
- Optional source URL, creator, license, attribution, and acquisition notes supplied by the user.

Use deterministic IDs derived from a stable source identifier and file checksum, or another documented deterministic scheme. Do not rely only on file names because different folders can contain identical names.

### Source rights and provenance

- Track license and usage rights for each image or source collection.
- Support statuses such as `verified`, `user_attested`, `unknown`, `restricted`, and `not_for_training`.
- Never assume an image is public-domain or commercially usable because it is available locally or online.
- Allow the user to exclude unknown/restricted-license records from training exports.
- Preserve attribution and source URL fields where provided.
- Add a clear report of records with missing or unverified rights information.

---

# 3. Full Processing Workflow (45+ Processes)

Implement each process as an independently testable stage with its own status, logs, configuration, and output. Stages may be skipped when they are not relevant to the selected mode.

## Stage 1 — Ingestion, Validation, and Metadata

| # | Process | Required behavior | Output |
|---:|---|---|---|
| 1 | Folder ingestion | Recursively scan all selected folders without changing source files. | Source inventory |
| 2 | File validation | Detect unsupported, unreadable, truncated, and corrupted files. | Validation status and reason |
| 3 | Stable image ID | Generate deterministic unique IDs and detect ID conflicts. | `image_id` |
| 4 | Checksum extraction | Calculate SHA-256 and file size. | Hash and file metadata |
| 5 | Image metadata extraction | Read dimensions, format, color mode, bit depth where available, and EXIF. | Metadata record |
| 6 | Orientation handling | Read EXIF orientation and create a correctly oriented processed rendition without modifying source. | Orientation-corrected derivative |
| 7 | Color profile handling | Detect ICC profile and apply a documented consistent color-management policy. | Color metadata / normalized rendition |
| 8 | Alpha-channel handling | Detect transparency and preserve it when appropriate; do not flatten without a configured background policy. | Alpha metadata / derivative |
| 9 | Aspect-ratio analysis | Calculate width, height, aspect ratio, and image orientation. | Geometry attributes |
| 10 | Import checkpoint | Persist scan progress and safely resume after interruption. | Resumable scan state |

## Stage 2 — Duplicate and Similarity Detection

| # | Process | Required behavior | Output |
|---:|---|---|---|
| 11 | Exact duplicate detection | Use SHA-256 to detect byte-identical files. | Exact duplicate groups |
| 12 | Perceptual duplicate detection | Use pHash/dHash/aHash or equivalent to detect resized/recompressed variants. | Perceptual similarity groups |
| 13 | Near-duplicate detection | Detect minor edits, crops, watermarks, and small visual changes using suitable methods. | Near-duplicate candidates |
| 14 | Semantic similarity | Use image embeddings (e.g., CLIP/OpenCLIP or an equivalent encoder) and configurable cosine-similarity thresholds. | Semantic similarity candidates |
| 15 | Representative selection | Rank group members using quality, resolution, provenance, and user rules. Never delete automatically. | Recommended representative + review decision |

Important: semantic similarity is not the same as duplication. Two different images of the same subject or scene may both be valuable. Provide adjustable thresholds and manual review.

## Stage 3 — Technical Quality, Aesthetic Quality, and Suitability

| # | Process | Required behavior | Output |
|---:|---|---|---|
| 16 | Resolution suitability | Check minimum dimensions and suitability for configured training buckets. | Resolution eligibility |
| 17 | Blur and sharpness | Calculate blur/sharpness indicators and optionally use a learned quality model. | Blur/sharpness scores |
| 18 | Noise assessment | Estimate luminance/chroma noise where appropriate. | Noise score / flags |
| 19 | Compression artifact detection | Identify blocking, ringing, and severe compression damage where measurable. | Artifact flags |
| 20 | Exposure and contrast | Estimate underexposure, overexposure, dynamic range, and contrast. | Exposure/contrast metrics |
| 21 | Color analysis | Extract color distribution/palette and identify extreme color issues. | Color statistics |
| 22 | Aesthetic scoring | Use a suitable, documented aesthetic predictor as a ranking signal—not absolute truth. | Aesthetic score |
| 23 | Composition analysis | Estimate framing, balance, symmetry, negative space, and subject placement where supported. | Composition attributes |
| 24 | Subject prominence | Estimate whether the main subject is visible, sufficiently large, and distinguishable. | Prominence score / flags |
| 25 | Cropping analysis | Detect clipped subjects, awkward crops, and potentially important content near borders. | Crop-quality flags |
| 26 | Watermark/logo analysis | Detect likely watermarks and logos. Do not remove them automatically. | Watermark/logo flags |
| 27 | OCR/text analysis | Detect visible text and its regions; identify text-heavy images and possible malformed typography. | OCR text + regions + flags |
| 28 | Quality aggregation | Combine independent signals with explainable weights and configurable thresholds. | Overall quality score + reasons |

Do not treat one metric as a universal quality judgment. For example, a deliberately soft-focus image may be artistically valid. Keep component scores visible and allow dataset-specific thresholds.

## Stage 4 — Visual Understanding and Annotation

| # | Process | Required behavior | Output |
|---:|---|---|---|
| 29 | Broad image classification | Predict broad categories such as landscape, architecture, wildlife, vehicle, food, portrait, product, illustration, abstract, and others. | Multi-label categories |
| 30 | Object detection | Detect objects and return labels, confidence values, and bounding boxes when supported. | Object annotations |
| 31 | Segmentation | Optionally segment subjects/background or selected objects. | Masks / segmentation references |
| 32 | Scene understanding | Describe environment, setting, foreground, middle ground, and background. | Scene attributes |
| 33 | Object relationships | Extract grounded relationships such as “person beside vehicle” when visible. | Relationship annotations |
| 34 | Visual attribute extraction | Extract visible colors, textures, materials, shapes, and other grounded attributes. | Structured attributes |
| 35 | Lighting analysis | Classify visible lighting such as daylight, golden hour, backlighting, studio lighting, or low light. | Lighting attributes |
| 36 | Camera/composition inference | Infer viewpoint, framing, perspective, and depth of field only when visually supportable. | Inferred attributes with confidence |
| 37 | Style classification | Classify photorealistic, cinematic, illustration, anime, 3D render, painting, graphic design, etc. | Style labels |
| 38 | Human/face handling | Describe only task-relevant visible attributes; do not infer identity or sensitive personal traits. | Policy-compliant annotations |

Camera model, focal length, aperture, shutter speed, GPS, date, and other capture facts must come from EXIF or reliable supplied metadata. If estimated from pixels, label as `inferred` and include confidence. Never present estimates as confirmed camera facts.

## Stage 5 — Captioning and Prompt Construction

Generate separate, clearly distinguished text fields. Do not mix image description with imaginative prompt additions.

| # | Process | Required behavior | Output |
|---:|---|---|---|
| 39 | Short caption | Produce one concise, accurate sentence focused on the main visible content. | `short_caption` |
| 40 | Detailed caption | Describe subject, action, environment, composition, lighting, colors, and relevant background details. | `detailed_caption` |
| 41 | Comprehensive caption | Provide richer but fully image-grounded visual description for VLM/captioning tasks. | `comprehensive_caption` |
| 42 | Image-generation prompt | Convert visible characteristics into a useful generation prompt while avoiding unsupported details. | `generation_prompt` |
| 43 | Structured visual tags | Produce controlled, deduplicated tags and attributes. | Tags and attributes |
| 44 | Optional alternate captions | Generate multiple distinct, accurate caption variants when selected by the user. | Caption variants |
| 45 | Caption diversity analysis | Detect repetitive openings, boilerplate, low-information captions, and overused phrases. | Diversity score / warnings |

Caption rules:

- Describe only what is visible or supported by metadata.
- Put the most important subject and action first.
- Avoid empty phrases such as “a beautiful image of” unless meaningful.
- Avoid guessing a person’s identity, exact location, brand, species, or object model when uncertain.
- Do not invent emotions, stories, weather, time, camera settings, or context.
- Do not add style words such as “cinematic,” “professional,” or “8K” unless they describe the image or are explicitly marked as prompt styling rather than observed facts.
- Avoid copying the same generic caption template across many images.
- Keep captions grammatically clear and useful for training.
- Store model name, model version, prompt/template version, generation settings, and timestamp.

## Stage 6 — Caption Verification and Image-Text Alignment

| # | Process | Required behavior | Output |
|---:|---|---|---|
| 46 | Image-text alignment score | Use a suitable image-text embedding model as one signal for matching. | Alignment score |
| 47 | Caption-grounding verification | Use a second pass/model or targeted checks to identify unsupported objects and claims. | Verification result |
| 48 | Hallucination checks | Flag claims not supported by visual evidence or available metadata. | Hallucination flags |
| 49 | Missing-detail checks | Identify salient visible content omitted from the caption. | Missing-detail warnings |
| 50 | Confidence calibration | Record uncertainty and route low-confidence examples to review. Do not present raw model confidence as calibrated truth. | Confidence / review status |
| 51 | Manual review queue | Allow users to inspect image, captions, tags, quality metrics, and edit/approve/reject. | Human review decisions |

Important: CLIP-style similarity scores are not proof that a caption is factually correct. Use them as a ranking/flagging signal alongside visual verification and review.

## Stage 7 — Dataset Curation, Balance, and Split Safety

| # | Process | Required behavior | Output |
|---:|---|---|---|
| 52 | Multi-signal filtering | Apply configurable rules for corrupted, low-quality, irrelevant, duplicate, and rights-restricted records. | Accepted/rejected/quarantined status |
| 53 | Category balance analysis | Measure distribution of categories, styles, objects, and other labels. | Distribution reports |
| 54 | Diversity/coverage analysis | Find overrepresented and underrepresented visual concepts. | Coverage report |
| 55 | Redundancy control | Reduce repetitive examples without discarding meaningful visual variation. | Curated candidate set |
| 56 | Split grouping | Group exact and near-duplicate clusters before splitting. | Leakage-safe groups |
| 57 | Train/validation/test splitting | Split by duplicate group/source group where appropriate; support deterministic seed and configurable ratios. | Split manifests |
| 58 | Split validation | Check that duplicate groups and prohibited source overlaps do not leak across splits. | Leakage report |
| 59 | Rights eligibility | Exclude records without acceptable rights status according to the selected policy. | Rights-filtered manifests |

Do not force equal category counts if that would distort the real data distribution. Make balancing optional and report the effects of any sampling or weighting.

## Stage 8 — Processed Image Creation and Storage

This stage is mandatory for a normal image-generation training dataset.

### Required storage architecture

Keep three separate data layers:

1. **Original/source layer:** read-only user-provided files. Never overwrite or transform them.
2. **Processed image layer:** actual image files created for the curated dataset.
3. **Metadata/manifest layer:** JSONL/CSV/Parquet records that point to the processed files.

### Processed image requirements

- Save every accepted image that is included in a training manifest.
- Keep the source file untouched and record its checksum.
- Create a processed rendition only when required by orientation, color management, format compatibility, alpha policy, or configured training preparation.
- Preserve the original aspect ratio by default.
- Do not force every image into a square canvas or a single fixed resolution.
- Support configurable resolution buckets and aspect-ratio buckets for the target training recipe.
- Do not upscale automatically by default. If upscaling is enabled, clearly mark it as synthetic/upscaled and retain the original dimensions in metadata.
- Use high-quality compression. Avoid unnecessary JPEG recompression cycles.
- Prefer PNG or another suitable lossless format when transparency or lossless preservation is required; use a high-quality photographic format according to the training pipeline when appropriate.
- Store image derivatives with stable IDs and deterministic relative paths.
- Verify every output file can be opened and decoded after writing.
- Calculate a checksum for each processed output.
- Prevent orphaned files and dangling manifest paths.
- Use configurable storage policies for disk-space limits and cleanup of temporary files. Never clean originals.

### Example directory structure

```text
image-dataset-project/
├── input_sources/                  # Optional references only; originals remain user-owned
├── processed_dataset/
│   ├── images/
│   │   ├── img_000001.jpg
│   │   ├── img_000002.png
│   │   └── img_000003.webp
│   ├── manifests/
│   │   ├── all_records.jsonl
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   ├── metadata/
│   │   ├── image_metadata.jsonl
│   │   ├── captions.jsonl
│   │   ├── quality_scores.jsonl
│   │   ├── annotations.jsonl
│   │   └── provenance.jsonl
│   ├── rejected/
│   │   ├── rejected_records.jsonl
│   │   └── quarantine/              # Optional copies; do not move source files
│   ├── reports/
│   │   ├── dataset_summary.json
│   │   ├── duplicate_report.json
│   │   ├── quality_report.json
│   │   ├── alignment_report.json
│   │   ├── rights_report.json
│   │   └── split_leakage_report.json
│   └── dataset_card.md
├── checkpoints/
├── logs/
├── configs/
│   └── pipeline_config.yaml
└── README.md
```

The `input_sources/` directory is optional; do not copy all originals by default. Store secure references to the original paths and checksums. Allow an optional user-controlled source backup if explicitly configured.

---

# 4. Recommended Training Record Schema

Use a versioned schema and validate every record with Pydantic or an equivalent validator.

Example JSONL record:

```json
{
  "schema_version": "1.0",
  "image_id": "img_000001",
  "image_path": "images/img_000001.jpg",
  "source": {
    "source_id": "source_01",
    "relative_path": "landscapes/mountain_sunset.jpg",
    "sha256": "SOURCE_SHA256",
    "source_url": null,
    "creator": null,
    "license": "unknown",
    "rights_status": "unknown",
    "attribution": null
  },
  "processed_image": {
    "format": "JPEG",
    "width": 1920,
    "height": 1080,
    "aspect_ratio": 1.7778,
    "sha256": "PROCESSED_SHA256",
    "is_upscaled": false,
    "transformations": ["exif_orientation_applied"]
  },
  "captions": {
    "short": "A mountain range beneath a warm sunset sky.",
    "detailed": "A wide view of a mountain range beneath a warm sunset sky, with layered ridgelines and a glowing horizon.",
    "comprehensive": "A detailed description grounded in visible image content...",
    "generation_prompt": "A wide landscape composition showing...",
    "alternatives": []
  },
  "visual_annotations": {
    "categories": ["landscape", "nature"],
    "objects": [],
    "scene": "mountain landscape",
    "relationships": [],
    "lighting": ["sunset light"],
    "style": ["photorealistic"],
    "color_palette": [],
    "composition": {},
    "camera_attributes": {
      "viewpoint": {"value": "wide view", "status": "inferred", "confidence": 0.78},
      "focal_length": {"value": null, "status": "unknown", "confidence": null}
    },
    "ocr": {"text": [], "regions": []},
    "watermarks_detected": false
  },
  "quality": {
    "overall_score": 95,
    "sharpness_score": 92,
    "noise_score": 96,
    "exposure_score": 94,
    "aesthetic_score": 91,
    "resolution_status": "accepted",
    "flags": []
  },
  "alignment": {
    "score": 0.94,
    "method": "image_text_embedding_and_verification",
    "verification_status": "passed",
    "warnings": []
  },
  "deduplication": {
    "exact_duplicate_group": null,
    "perceptual_duplicate_group": null,
    "semantic_similarity_group": null,
    "representative_status": "unique"
  },
  "processing": {
    "status": "completed",
    "model_versions": {},
    "pipeline_version": "0.1.0",
    "config_hash": "CONFIG_HASH",
    "processed_at": "ISO-8601 timestamp"
  },
  "curation": {
    "decision": "accepted",
    "review_status": "not_required",
    "rejection_reasons": [],
    "split": "train",
    "split_seed": 42
  }
}
```

Use relative paths in exported manifests so datasets can be moved. Never place private absolute paths, credentials, or machine-specific information in public exports.

For the primary text-to-image export, also provide a minimal configurable schema expected by the chosen training framework, usually containing an image path and its associated text caption. Do not assume one universal schema works for every trainer.

---

# 5. Model and Technology Strategy

Use pluggable adapters. The names below are candidates to evaluate, not mandatory dependencies. Verify model availability, current versions, hardware needs, and licenses before adoption.

| Task | Candidate technologies/models | Notes |
|---|---|---|
| Image decoding and preprocessing | Pillow, OpenCV, imageio | Preserve source; avoid repeated lossy recompression. |
| Exact duplicates | SHA-256 | Fast and deterministic. |
| Perceptual duplicates | ImageHash (pHash/dHash/aHash) | Use thresholds and manual review. |
| Semantic similarity | OpenCLIP / CLIP / DINOv2 embeddings | Similarity is not proof of duplication. |
| Blur/noise/technical quality | OpenCV metrics + optional learned IQA models | Combine metrics; expose reasons. |
| Aesthetic scoring | A suitable pretrained aesthetic predictor | Treat as ranking signal, not objective truth. |
| OCR | PaddleOCR, RapidOCR, or another supported OCR engine | Optional; record regions and detected text. |
| Object detection | Grounding DINO or another supported detector | Optional and task-configurable. |
| Segmentation | SAM 2 or another supported segmentation model | Run only when useful. |
| Image captioning / VLM | Qwen vision-language family, Florence-2, BLIP-2, LLaVA, or other evaluated VLMs | Benchmark caption quality on the user's sample images. |
| Image-text alignment | CLIP/OpenCLIP or another image-text encoder | Use as one signal only. |
| Structured validation | Pydantic / JSON Schema | Reject invalid output records. |

### Model selection requirements

- Compare candidate models on a representative, user-approved sample.
- Evaluate factual accuracy, detail coverage, hallucination rate, caption diversity, latency, memory use, and license compatibility.
- Keep model selection configurable per stage.
- Cache reusable embeddings and inference outputs.
- Record model name, revision/version, precision, quantization, inference settings, and prompt template version.
- Do not run object detection, segmentation, OCR, and multiple VLMs on every image unless enabled or justified.
- Provide CPU fallback where feasible, but warn clearly about expected speed and memory constraints.
- Provide GPU acceleration and configurable batch size.

### Hardware-aware behavior

Inspect available hardware at runtime. For example, support a local GPU profile such as NVIDIA RTX 3060 12GB, but do not hard-code that assumption. Detect GPU model, available VRAM, system RAM, disk space, and supported runtimes.

Implement:

- Automatic conservative batch-size defaults.
- Batch-size adjustment after out-of-memory errors.
- Graceful CPU fallback only when configured.
- Model memory estimates before loading.
- Per-stage device selection.
- Option to use cloud/API inference with explicit user configuration and cost warnings.
- No automatic huge model downloads.

---

# 6. Storage, Performance, and Reliability

The pipeline must scale from a small folder to hundreds of thousands of images through incremental processing.

Implement:

- Streaming directory traversal; never load the whole dataset into RAM.
- Bounded worker pools and queues.
- Configurable CPU/GPU batch sizes.
- SQLite initially for local job state, with an abstraction that permits PostgreSQL later.
- Atomic writes for metadata/manifests.
- Per-image and per-stage checkpoints.
- Retry with bounded attempts and clear error categories.
- Idempotency keys based on image checksum, stage, model version, and configuration hash.
- Cache for embeddings, metadata, and model inference.
- Disk-space preflight checks and estimated output size.
- Temporary-file cleanup after successful commit only.
- Resume after application restart or system crash.
- Job cancellation that stops safely without corrupting completed outputs.
- Processing throughput and resource telemetry.

### Smart reprocessing rules

- If only the caption model changes, regenerate captions and re-run dependent caption validation; do not redo image decoding, metadata extraction, or duplicate detection unnecessarily.
- If a quality threshold changes, reuse cached component metrics and re-run filtering decisions.
- If export settings change, rebuild manifests/exports without re-running vision inference.
- If the source checksum changes, mark the source as a new/changed image and invalidate dependent stages.

---

# 7. User Interface and Controls

Build a clean, functional dashboard. The UI must display real backend state, not fabricated metrics.

### Main dashboard

Display:

- Total discovered images.
- Valid and invalid files.
- Processed and pending images.
- Failed and retryable images.
- Exact and near-duplicate counts.
- Accepted, rejected, and quarantined counts.
- Average technical quality, aesthetic, and image-text alignment scores.
- Images awaiting manual review.
- Current processing stage and throughput.
- Estimated remaining time only when based on observed processing rate; otherwise show unavailable.
- GPU/CPU/RAM/disk usage where supported.

### Dataset browser

Provide:

- Image thumbnails and full-resolution preview.
- Search and filters by category, style, quality, score, split, status, and rights status.
- Side-by-side original/processed preview.
- Short, detailed, and comprehensive captions.
- Generation prompt and tags.
- Quality and alignment component scores.
- Detection/OCR/segmentation overlays when present.
- Source provenance and license status.
- Manual edit, approve, reject, quarantine, and restore controls.

### Configuration controls

Allow users to configure:

- Processing mode.
- Enabled/disabled stages.
- Minimum dimensions and resolution buckets.
- Quality and aesthetic thresholds.
- Duplicate similarity thresholds.
- Caption style/length and number of alternatives.
- Caption verification strictness.
- Model selection and inference device.
- Batch size and worker count.
- Image output format and quality.
- Alpha-channel handling.
- Train/validation/test ratios and random seed.
- Rights policy.
- Output location and disk-space limits.

Every destructive-looking or exclusion-related action must show its consequences and preserve recoverability. The default must be non-destructive.

---

# 8. Export Formats and Dataset Packaging

Support:

- JSONL.
- CSV.
- Parquet.
- A minimal image-caption manifest for common text-to-image training workflows.
- Hugging Face `datasets`-compatible packaging where practical.
- A dataset card (`dataset_card.md`) documenting contents, source, licenses, filters, transformations, limitations, model-generated annotations, and intended uses.

### Export validation

Before export:

1. Confirm each accepted manifest path exists.
2. Confirm each image decodes successfully.
3. Confirm each record has a valid caption for the selected mode.
4. Confirm no invalid JSON/CSV/Parquet records.
5. Confirm no prohibited rights statuses are included.
6. Confirm no duplicate-group leakage across splits.
7. Confirm processed image checksums match the manifest.
8. Generate counts and reports for every split.
9. Generate an export summary with warnings.
10. Create a package checksum manifest.

Do not upload to any remote service by default. Remote upload must be an explicit, separately configured user action.

---

# 9. Proposed Project Structure

Use a modular structure similar to:

```text
image-dataset-pipeline/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── config/
│   │   ├── database/
│   │   ├── schemas/
│   │   ├── services/
│   │   ├── pipeline/
│   │   │   ├── ingestion/
│   │   │   ├── validation/
│   │   │   ├── metadata/
│   │   │   ├── deduplication/
│   │   │   ├── quality/
│   │   │   ├── vision/
│   │   │   ├── captioning/
│   │   │   ├── alignment/
│   │   │   ├── curation/
│   │   │   ├── image_output/
│   │   │   └── export/
│   │   ├── workers/
│   │   └── tests/
│   └── pyproject.toml
├── frontend/
│   ├── app/
│   ├── components/
│   ├── hooks/
│   └── services/
├── configs/
│   ├── pipeline_config.example.yaml
│   └── model_registry.yaml
├── scripts/
├── docs/
├── data/                         # Configurable; keep source/output paths separate
├── checkpoints/
├── logs/
├── .env.example
└── README.md
```

Use Python + FastAPI + Pydantic for the backend. A Next.js + React + TypeScript + Tailwind frontend is recommended. Use a maintainable local job queue/worker implementation first; add distributed workers only when justified. Avoid premature microservices.

---

# 10. Configuration Example

Create a documented YAML configuration with safe defaults. Example:

```yaml
project:
  name: image-dataset-project
  mode: text_to_image
  pipeline_version: "0.1.0"

paths:
  sources: []
  output_dir: ./processed_dataset
  checkpoints_dir: ./checkpoints
  logs_dir: ./logs

processing:
  recursive_scan: true
  preserve_originals: true
  resume_enabled: true
  max_retries: 2
  cpu_workers: 4
  gpu_batch_size: auto

images:
  output_format: auto
  jpeg_quality: 95
  preserve_aspect_ratio: true
  auto_upscale: false
  preserve_alpha: true
  resolution_buckets: [512, 768, 1024]

quality:
  min_width: 512
  min_height: 512
  min_overall_score: 65
  min_aesthetic_score: null
  manual_review_band: [55, 75]

deduplication:
  exact: true
  perceptual: true
  semantic: true
  auto_delete: false
  perceptual_threshold: configurable
  semantic_threshold: configurable

captioning:
  short_caption: true
  detailed_caption: true
  comprehensive_caption: true
  generation_prompt: true
  alternate_caption_count: 0
  verify_captions: true

rights:
  require_approved_status_for_export: true
  allowed_statuses: [verified, user_attested]

splits:
  train: 0.90
  validation: 0.05
  test: 0.05
  seed: 42
  group_near_duplicates: true

export:
  formats: [jsonl, csv]
  include_processed_images: true
  remote_upload_enabled: false
```

All thresholds above are examples. Validate configuration, document units/ranges, and do not silently treat placeholder strings such as `configurable` as numeric values in the actual implementation.

---

# 11. API and CLI Expectations

Provide a CLI for reliable batch workflows, for example:

```bash
# Inspect available hardware and dependencies
image-pipeline system-info

# Scan one or more folders
image-pipeline scan --source /path/to/images --output ./workspace

# Run or resume a configured pipeline
image-pipeline run --config ./configs/pipeline_config.yaml
image-pipeline resume --job-id JOB_ID

# Review failed or uncertain samples
image-pipeline review --job-id JOB_ID

# Validate and export a dataset
image-pipeline validate --job-id JOB_ID
image-pipeline export --job-id JOB_ID --format jsonl

# Generate a final report
image-pipeline report --job-id JOB_ID
```

Provide API endpoints for creating jobs, checking status, updating configuration, browsing records, approving/rejecting samples, retrying failures, validating, and exporting. Secure local APIs and validate all paths to prevent unintended filesystem access.

---

# 12. Testing and Acceptance Criteria

Create unit tests, integration tests, and a small end-to-end fixture dataset. Use synthetic or openly licensed test images; do not include private user images in tests.

### Functional acceptance criteria

- Recursive scan discovers supported images and reports unsupported/corrupt files.
- Source files remain unchanged; verify checksums before and after processing.
- Stable IDs are repeatable for the same source/checksum.
- Exact and near-duplicate results are recorded without automatic deletion.
- Quality metrics and thresholds are configurable.
- Caption records are linked to the correct image IDs.
- Low-confidence and failed records are reviewable.
- Processed image files are physically written to the output folder.
- Every exported manifest path resolves to an existing, decodable image.
- Train/validation/test splits are deterministic for a fixed seed.
- Duplicate groups do not cross splits when group leakage prevention is enabled.
- A stopped job resumes without duplicating completed work.
- Changing caption model settings does not unnecessarily rerun unrelated stages.
- Invalid JSON/schema records are rejected and reported.
- No unapproved rights-status records appear in exports when strict rights filtering is enabled.

### Quality acceptance criteria

Create a manually reviewed benchmark sample with diverse image types and compare candidate models. Report:

- Caption factual accuracy.
- Hallucination rate.
- Salient-detail coverage.
- Caption diversity.
- Image-text alignment behavior.
- Processing latency.
- GPU/CPU memory use.
- Failure rate.

Do not claim “best quality” based only on model names or similarity scores. Measure results on a representative sample and expose trade-offs.

---

# 13. Implementation Roadmap

Implement in the following order. Do not build all stages at once without tests.

### Phase 1 — Foundation and ingestion

- Inspect environment and hardware.
- Set up project, configuration, database, logging, and schemas.
- Implement folder scanning, validation, stable IDs, checksums, metadata, and checkpoints.
- Add tests.

### Phase 2 — Deduplication and basic quality

- Implement exact/perceptual duplicates.
- Add optional semantic embeddings.
- Implement technical quality signals and explainable scores.
- Add reviewable duplicate groups.

### Phase 3 — Image processing and storage

- Implement processed image writing, aspect-ratio preservation, alpha/color policies, resolution buckets, checksums, and file verification.
- Ensure original files are immutable.
- Add storage tests and disk-space preflight.

### Phase 4 — Vision analysis

- Implement model adapter interface and model registry.
- Add optional classification, object detection, OCR, segmentation, scene, lighting, and style analysis.
- Benchmark models on a sample before selecting defaults.

### Phase 5 — Caption generation and verification

- Add short, detailed, comprehensive captions and generation prompts.
- Add image-text alignment and hallucination review.
- Add manual approval/editing workflow.

### Phase 6 — Dataset curation and leakage-safe splits

- Add rights/provenance filtering, category coverage, balancing options, grouping, deterministic splits, and leakage reports.

### Phase 7 — Export and dataset card

- Generate JSONL/CSV/Parquet and target-framework manifests.
- Validate paths, image decodability, schemas, checksums, rights, and splits.
- Generate dataset card and summary reports.

### Phase 8 — Dashboard and optimization

- Complete dashboard, browsing, filters, review actions, telemetry, cancellation, retries, and smart reprocessing.
- Profile performance and optimize bottlenecks.
- Run complete end-to-end tests.

At the end of each phase, report what is implemented, what is tested, known limitations, and the exact command to run it.

---

# 14. Final Deliverables

Deliver:

1. A working application and CLI.
2. A modular backend with documented model adapters.
3. A dashboard connected to real processing state.
4. A configurable pipeline YAML.
5. Database schema and migrations where needed.
6. Unit and integration tests.
7. Example input/output using a small test fixture.
8. Validated processed image files physically saved in the dataset.
9. JSONL/CSV/Parquet exports as configured.
10. Dataset card, quality reports, duplicate reports, rights report, and split-leakage report.
11. Setup instructions for Windows, Linux, and macOS where dependencies permit.
12. Hardware/model setup guide including VRAM, RAM, disk, and model download requirements.
13. Known limitations and next-step recommendations.

## Final directive

Build a dependable system that converts raw image folders into a high-quality, well-documented, legally traceable, diverse, and training-ready image dataset. The processed dataset must contain the actual curated image files and their corresponding captions/metadata. Keep original images untouched. Prioritize correctness, reproducibility, image-text alignment, diversity, and quality over raw throughput or dataset size.

**Begin by inspecting the current repository and available hardware. Then implement Phase 1. Do not claim a feature is complete until its tests pass.**
