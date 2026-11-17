# Hardware and models

Hardware is detected at run time (`image-pipeline system-info`); nothing assumes a particular GPU.

- **Device:** `processing.device: auto` picks CUDA, then Apple MPS, then CPU. Set `allow_cpu_fallback: false`
  to make model stages fail fast instead of running slowly on CPU.
- **Batch size:** `gpu_batch_size: auto` sizes batches from currently free memory (at most about a quarter of it).
  After an out-of-memory error the batch size is halved and the same images are retried.
- **Disk:** before writing images the pipeline estimates the space needed and stops if free space would
  fall below `storage.min_free_disk_gb` (or the dataset would exceed `storage.max_output_gb`).

## Registry (`configs/model_registry.yaml`)

Sizes and memory figures are approximate. **Confirm each licence on its model card for your use.**

| Key | Used for | Download | RAM | VRAM | Licence (as listed) | Status in this project |
|---|---|---|---|---|---|---|
| `clip-vit-base-patch32` | embeddings, semantic similarity, zero-shot tags, aesthetic signal, caption verification | ~0.6 GB | ~1.5 GB | ~1 GB | MIT per OpenAI CLIP repo | used and tested |
| `clip-vit-large-patch14` | same, stronger and slower | ~1.7 GB | ~3 GB | ~2.5 GB | MIT per OpenAI CLIP repo | same adapter; not run here |
| `smolvlm2-256m` | captions | ~1 GB | ~2.5 GB | ~2 GB | Apache-2.0 | adapter tested on two images; caption quality not benchmarked |
| `smolvlm2-2.2b` | captions | ~4.5 GB | ~8 GB | ~6 GB | Apache-2.0 | same adapter; not run here |
| `owlv2-base` | object detection | ~0.7 GB | ~3 GB | ~3 GB | Apache-2.0 | adapter written, **never executed** (weights absent) |

No learned models are needed for: validation, hashing, technical quality, face counting (OpenCV Haar
cascade shipped with OpenCV), segmentation (GrabCut), OCR (RapidOCR's bundled ONNX models).

## What was measured

One run on an Apple M2 with 8 GB RAM, 410 synthetic 768x576 JPEGs, default configuration with the
`ml` extra (CLIP ViT-B/32 on MPS, template captions). These are single-run figures on synthetic
images, not a benchmark; your images and hardware will differ.

| Stage | Images/s (observed) |
|---|---|
| validation + metadata | ~1,180 |
| perceptual hashes | ~470 |
| technical quality (4 CPU workers) | ~40 |
| face counting | ~80 |
| CLIP embeddings (MPS) | ~36 |
| zero-shot annotation (from cached embeddings) | ~680 |
| caption verification | ~97 |
| image writing (mostly byte copies) | ~740 |

Whole job: 75 s wall-clock for 410 images. Deduplication took 38 s of that, almost all of it geometric
near-duplicate verification (3,770 candidate pairs, about 10 ms each). That step grows roughly linearly
with image count times `deduplication.near_max_candidates_per_image`; lower that value or raise
`near_candidate_cosine` for large collections. SmolVLM2-256M captioning was measured separately at
roughly 7 s for model load plus one short caption on this machine, i.e. several seconds per image
when all caption fields are generated.

`image-pipeline status` shows the real rates for your own run (each stage records `rate_per_s`).

## Choosing a captioning model

This was intentionally left open. The pipeline ships three adapters (`template`, `sidecar`, `smolvlm`)
and a benchmark command that measures latency, memory, failure rate and caption statistics on a sample
of *your* images and writes a sheet for human rating of accuracy, hallucinations and coverage:

```bash
image-pipeline benchmark --adapters template,smolvlm2-256m --sample 20
image-pipeline benchmark-summarize benchmarks/<run>/review_sheet.csv
```

On an 8 GB machine the 256M model is the only VLM in the registry that fits comfortably; expect short
captions that sometimes miss or invent detail. Larger local models need more memory. A hosted/API
captioner is not implemented; it would be a new adapter class in
`backend/app/pipeline/captioning/adapters.py` plus a registry entry, and would send your images to
that provider.

## Adding a model

1. Add an entry to `configs/model_registry.yaml` (repo, adapter, size, licence, memory).
2. If it needs a new loading path, add an adapter class in `backend/app/models/hub.py` and register it in `_ADAPTERS`.
3. Point `models.embedding`, `models.captioner` or `models.object_detector` at the key.

Stage results record the model name, revision, precision and prompt/template version, and changing a
model only re-runs the stages that depend on it.
