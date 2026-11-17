# Known limitations and next steps

Read this before using the output for training.

## What has and has not been verified

- **Tested:** 97 automated tests on procedurally generated images (25-file fixture with duplicates,
  corrupt/truncated files, EXIF rotation, GPS, alpha, CMYK, 16-bit, tiny/blurry/dark images), plus one
  410-image run. macOS on Apple silicon only.
- **Not tested on real photographs.** No real image collection has been processed. All thresholds
  (quality, review band, duplicate and similarity thresholds, caption-check thresholds) are untuned
  defaults; expect to adjust them after looking at results on your data.
- **Not tested at scale.** The design streams and checkpoints, but nothing beyond 410 images has been
  run. Deduplication holds hashes and embeddings in memory (about 2 KB per image for CLIP ViT-B/32) and
  near-duplicate verification costs roughly 10 ms per candidate pair.
- **Not run on Linux, Windows or NVIDIA GPUs.** The CUDA path and out-of-memory batch reduction are
  implemented but unexercised.
- **PostgreSQL is untested.** `IMAGE_PIPELINE_DATABASE_URL` is only a hook; SQLite is what works.
- **Object detection adapter has never been executed** (no detector weights locally). Treat it as
  experimental. Relationships depend on it and are purely geometric.

## Shard mode

- Tested end to end on 10 real Flickr URLs and on the first 80 images of `UWGZQ/pixmo_images` (upload
  off), and in automated tests with downloads and uploads faked. **No shard has been uploaded to the real Hub yet, and nothing near the full 9 million URLs
  has been run**: download speed, Flickr rate limits and how many listed photos are gone are unknown.
- Duplicates are removed inside a shard and exact duplicates across shards (by listed checksum).
  Near-duplicates across shards are not detected, and each shard has its own train/validation/test split.
- Per-image author attribution is not available from the URL lists; records keep the source URL and the
  licence text you configure (`shards.license_note`), unverified.
- Shard mode owns its workspace: after a shard is closed, the per-image working data for it is removed
  from the database (a ledger of consumed URLs is kept), so `status IMAGE_ID` only works for the open shard.

## Signals are heuristics

- Sharpness, noise, compression, exposure, composition, prominence and crop metrics are classical
  measurements. They can be wrong for intentional styles (soft focus, low key, flat illustration).
  Ringing artefacts are not measured.
- The **aesthetic score is a CLIP prompt-pair signal**, not a trained aesthetic predictor. It ranks
  loosely and scored the synthetic fixtures low across the board.
- **Categories, style, lighting, scene, viewpoint and watermark likelihood are CLIP zero-shot guesses**
  over fixed label lists (`vision/labels.py`), stored as `inferred` with uncalibrated confidence.
- Watermark/logo detection is a zero-shot probability plus OCR text patterns; there is no dedicated
  logo detector. Textures and materials are not extracted.
- Face counting uses an OpenCV Haar cascade, which misses faces and produces false positives.
- Segmentation is GrabCut seeded by saliency (subject/background only), not a learned model.
- Semantic groups are connected components above a cosine threshold, so chains of similar images can
  merge into one large group. They are informational unless `max_per_semantic_group` is set.

## Captions and verification

- The default **`template` captions are generic baselines** built from inferred annotations. They are
  flagged for review and should be replaced (own captions or a VLM) before serious training.
- SmolVLM2-256M works but is a very small model; its caption quality has **not** been benchmarked here.
  Use `image-pipeline benchmark` and the human review sheet on your own images.
- **Caption verification is not a second-model fact check.** It scores image-text similarity and tests
  caption terms from a fixed vocabulary of 156 words; claims outside that vocabulary are not
  checked, and when captions come from the template adapter the check is not independent. CLIP text
  input is limited to 77 tokens, so long captions are only partly scored.
- All confidences are uncalibrated and exist to route images to review.

## Not implemented

- Upscaling and hosted/API inference providers.
- **The Hugging Face upload has not been run against the real Hub.** Its conditions (size threshold, validation, exclusions) are tested with the network call replaced; the first real upload is the first real test. It sends the folder in one commit, which can be slow or fail part-way for very large datasets.
- A trained aesthetic predictor, learned IQA model, SAM-class segmentation, texture/material attributes.
- Balancing by weighting or resampling (only optional caps per category / semantic group).
- Stepwise database migrations (the schema is versioned and created on first use; there is one version).
- A dashboard or HTTP API (dropped by decision: the tool is run from the terminal).
- Animated or multi-frame files: only the first frame is used. Images above 8 bits per channel are
  reduced to 8 bits. Grey images with a non-sRGB ICC profile may fail conversion and are then quarantined.

## Behaviour worth knowing

- Rights statuses are **what you enter**; the tool records and enforces them but cannot verify a licence.
  With the default strict policy, images with unknown rights are analysed but never exported.
- Byte-copied renditions keep their original EXIF (camera make/model, timestamps) unless it contains
  GPS data, in which case EXIF is stripped losslessly. Re-encoded renditions carry no EXIF.
- Rejecting an already-exported image moves its rendition to `rejected/quarantine/` on the next export.
  Nothing in a source folder is touched.
- Cancellation takes effect between batches, so a long model batch finishes first.
- A stage interrupted mid-way restarts that stage's unfinished images; images finished before the last checkpoint (every `processing.db_chunk_size` images, 200 by default) are not repeated.
- Fixing a stage bug does not invalidate cached results unless the stage `version` is bumped; use
  `--retry-failed` for images that previously failed.

## Suggested next steps

1. Run on a few hundred of your real images, inspect the reports and `image-pipeline review`, and tune thresholds against what you see.
2. Decide the captioning approach with the benchmark and review sheet; add an API adapter if local
   models are not good enough on this hardware.
3. Enable OCR if text-heavy images matter; evaluate a detector before enabling object detection.
4. Replace the aesthetic signal with a trained predictor if aesthetic filtering is important.
5. Run on Linux/CUDA and at realistic scale before committing to a large job.
