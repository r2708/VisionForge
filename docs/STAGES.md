# Stage map

How the 59 processes in the specification map to pipeline stages. Run `image-pipeline stages` to see
which are enabled for your configuration and why any are skipped.

Order of execution: `scan → validate → phash → quality_technical → people → ocr → segmentation →
embedding → zero_shot → object_detection → quality → dedup → curation_pre → captioning → alignment →
caption_diversity → curation → image_output → reconcile_outputs → splits → export`.

| # | Process | Stage | Module (`backend/app/pipeline/…`) | Method | Default |
|---:|---|---|---|---|---|
| 1 | Folder ingestion | `scan` | `ingestion/scanner.py` | streaming `os.scandir` walk | on |
| 2 | File validation | `validate` | `validation/validate.py` | structural verify + full decode | on |
| 3 | Stable image ID | `scan` | `ingestion/scanner.py` | SHA-256 of source ID + relative path + file checksum | on |
| 4 | Checksum | `scan` | `ingestion/scanner.py` | SHA-256, file size | on |
| 5 | Metadata | `validate` | `validation/validate.py` | dimensions, format, mode, bit depth, selected EXIF | on |
| 6 | Orientation | `validate` + `image_output` | `image_output/writer.py` | EXIF orientation baked into the rendition | on |
| 7 | Colour profile | `validate` + `image_output` | `image_output/writer.py` | non-sRGB ICC → sRGB; untagged assumed sRGB | on |
| 8 | Alpha | `validate` + `image_output` | `image_output/writer.py` | used alpha kept (PNG) or flattened by policy | on |
| 9 | Aspect ratio | `validate` | `validation/validate.py` | width/height/ratio/orientation class | on |
| 10 | Import checkpoint | `scan` | `ingestion/scanner.py` | batch commits; unchanged files skipped by size+mtime | on |
| 11 | Exact duplicates | `dedup` | `deduplication/dedup.py` | SHA-256 groups; in shard mode also against the hash ledger of earlier shards (`seen_hashes`) | on |
| 12 | Perceptual duplicates | `phash` + `dedup` | `deduplication/hashing.py` | pHash + dHash cross-check, banded Hamming search; in shard mode also against the hash ledger of earlier shards | on |
| 13 | Near duplicates | `dedup` | `deduplication/dedup.py` | candidates from hashes/embeddings, verified with ORB + RANSAC homography | on |
| 14 | Semantic similarity | `embedding` + `dedup` | `vision/stages.py` | CLIP embeddings, cosine threshold | on if model cached |
| 15 | Representative | `dedup` | `deduplication/dedup.py` | user override → quality band → resolution → rights → size | on |
| 16 | Resolution suitability | `quality` | `quality/aggregate.py` | minimum size + aspect bucket without upscaling | on |
| 17 | Blur / sharpness | `quality_technical` | `quality/technical.py` | Laplacian variance (90th-percentile tile) | on |
| 18 | Noise | `quality_technical` | `quality/technical.py` | Immerkaer estimate on smooth regions, luma + chroma | on |
| 19 | Compression artefacts | `quality_technical` | `quality/technical.py` | 8x8 blockiness ratio, JPEG quality estimate | on |
| 20 | Exposure / contrast | `quality_technical` | `quality/technical.py` | luma histogram statistics | on |
| 21 | Colour analysis | `quality_technical` | `quality/technical.py` | palette, colourfulness, saturation, cast | on |
| 22 | Aesthetic score | `zero_shot` | `vision/stages.py` | CLIP prompt-pair signal (not a trained aesthetic model) | on if model cached |
| 23 | Composition | `quality_technical` | `quality/technical.py` | spectral-residual saliency: balance, thirds, symmetry, negative space | on |
| 24 | Subject prominence | `quality_technical` | `quality/technical.py` | saliency peak and largest salient region | on |
| 25 | Cropping | `quality_technical` | `quality/technical.py` | salient region touching borders | on |
| 26 | Watermark / logo | `ocr` + `quality` | `quality/aggregate.py` | text read by OCR that looks like a watermark or credit line (agency names, copyright marks, a small address at an edge) sets `watermark_text_detected`; such images are quarantined (`quality.reject_watermarked`). The CLIP zero-shot probability is stored but decides nothing: on real images it fired on any visible text | needs OCR |
| 27 | OCR | `ocr` | `vision/stages.py` | RapidOCR text + regions | **off** (optional) |
| 28 | Quality aggregation | `quality` | `quality/aggregate.py` | weighted mean of available components, with reasons | on |
| 29 | Classification | `zero_shot` | `vision/stages.py`, `vision/labels.py` | CLIP zero-shot over 20 categories | on if model cached |
| 30 | Object detection | `object_detection` | `vision/stages.py` | zero-shot detector adapter | **off** (untested adapter) |
| 31 | Segmentation | `segmentation` | `vision/stages.py` | GrabCut seeded by saliency | **off** (optional) |
| 32 | Scene | `zero_shot` | `vision/stages.py` | CLIP zero-shot setting | on if model cached |
| 33 | Relationships | `object_detection` | `vision/stages.py` | geometry of detected boxes | off with detection |
| 34 | Visual attributes | `quality_technical` + `zero_shot` | — | colours from pixels; no texture/material model | partial |
| 35 | Lighting | `zero_shot` | `vision/stages.py` | CLIP zero-shot | on if model cached |
| 36 | Camera / composition inference | `zero_shot` | `vision/stages.py` | viewpoint, depth of field marked `inferred`; capture facts from EXIF only | on if model cached |
| 37 | Style | `zero_shot` | `vision/stages.py` | CLIP zero-shot over 11 styles | on if model cached |
| 38 | Human / face handling | `people` | `vision/stages.py` | face count only; no identity or traits | on |
| 39–42 | Short / detailed / comprehensive caption, generation prompt | `captioning` | `captioning/adapters.py` | `template`, `sidecar`, `smolvlm` or `auto` adapter. `auto` uses the caption that came with the source image and loads the VLM only for images without one | on (`template`) |
| 43 | Tags | `captioning` | `captioning/adapters.py` | controlled, de-duplicated, from annotations | on |
| 44 | Alternate captions | `captioning` | `captioning/adapters.py` | `alternate_caption_count` | off (0) |
| 45 | Caption diversity | `caption_diversity` | `captioning/stage.py` | repeated openings, duplicates, low-information captions | on |
| 46 | Alignment score | `alignment` | `alignment/stage.py` | CLIP image-text cosine, rescaled | on if model cached |
| 47–49 | Grounding, hallucination, missing detail | `alignment` | `alignment/stage.py` | per-term similarity checks against a fixed vocabulary, OCR/face/detection cross-checks | on if model cached |
| 50 | Confidence | `alignment` | `alignment/stage.py` | heuristic, labelled uncalibrated; routes to review | on if model cached |
| 51 | Manual review | — | `services/review.py`, `image-pipeline review` | approve / reject / quarantine / restore / edit | — |
| 52 | Filtering | `curation_pre`, `curation` | `curation/curate.py` | rules with recorded reasons, including quarantine for a failed caption check (`captioning.quarantine_failed_verification`), detected watermark text and a very low aesthetic signal (`quality.review_min_aesthetic_score`) | on |
| 53–54 | Balance / coverage | `export` | `export/exporter.py` | distributions, over/under-represented, absent categories | on |
| 55 | Redundancy control | `curation` | `curation/curate.py` | duplicate handling, optional semantic/category caps | caps off |
| 56 | Split grouping | `splits` | `curation/splits.py` | union of exact/perceptual/near groups (+ optional semantic, directory) | on |
| 57 | Splitting | `splits` | `curation/splits.py` | deterministic by seed, group-wise | on |
| 58 | Split validation | `splits`, `export` | `curation/splits.py` | independent leakage re-check | on |
| 59 | Rights eligibility | `curation`, `export` | `curation/curate.py` | status must be in `rights.allowed_statuses` | on (strict) |
| Stage 8 | Processed images | `image_output`, `reconcile_outputs` | `image_output/writer.py` | atomic write, decode check, checksum, orphan control | on |
| §8 | Export | `export` | `export/exporter.py` | 10 validation checks, manifests, reports, card, `SHA256SUMS` | on |

## Idempotency and smart reprocessing

Each per-image result stores a key built from: file checksum, stage name and version, the
configuration values that stage reads, the model name/revision, and the keys of the upstream results
it used. A stage is skipped for an image when the stored key equals the newly computed one.

| You change | What re-runs |
|---|---|
| Caption adapter, model or caption settings | `captioning`, `alignment`, curation, export |
| Quality weights, minimum size, buckets | `quality` (cheap aggregation), curation, export |
| Quality / aesthetic thresholds, review band, rights policy, dedup action | curation, image writing for newly accepted images, export |
| Dedup thresholds | `dedup`, curation, export |
| Output format / quality / alpha policy / max side | `image_output`, export |
| Split ratios or seed, export formats | `splits`, `export` |
| A source file's content | that image only, from the start (it gets a new ID) |

Failed images are recorded with an error category and attempt count. Transient categories are retried
up to `processing.max_retries` times; after that they are left alone until you pass `--retry-failed`.
