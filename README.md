# Image Dataset Pipeline

Turns folders of images you are authorised to use into a curated, quality-checked, reproducible,
training-ready image-text dataset: real processed image files plus captions, annotations,
train/validation/test manifests and reports.

- **Source folders are read-only.** Nothing is ever moved, overwritten or deleted there.
- **Nothing is deleted anywhere.** Duplicates and low-quality images are rejected or quarantined with
  reasons, and every decision can be reviewed and reversed.
- **Resumable and idempotent.** Each image/stage result is checkpointed; a crash or a config change
  only repeats the work that is actually affected.
- **Local by default.** Models load from the local cache only; nothing is downloaded without an explicit
  command, and nothing is uploaded unless you enable the Hugging Face section in the config.

Built from `image_dataset_pipeline_master_prompt.md`. See `docs/` for the stage map, hardware/model
guide and known limitations — please read `docs/LIMITATIONS.md` before relying on the output.

## Layout

```
backend/     Python package (CLI, pipeline stages, tests)
configs/     pipeline_config.example.yaml, model_registry.yaml
docs/        setup, stages, hardware & models, limitations
scripts/     fixture generator and demo run
```

A **workspace** is one dataset project. It holds `configs/pipeline_config.yaml`, `checkpoints/pipeline.db`
(job state), `logs/`, `cache/` and `processed_dataset/`. The workspace defaults to the current
directory; pass `--workspace DIR` (or set `IMAGE_PIPELINE_WORKSPACE`) to use another. It must not be
inside a source folder.

## Install

Requires Python 3.10+ (developed on 3.12).

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e "backend[ml,ocr,dev]" # drop ml / ocr if you do not want model stages / OCR
```

Without the `ml` extra the pipeline still runs: embeddings, zero-shot tags, the aesthetic signal and
caption verification are skipped with a stated reason. Platform notes are in `docs/SETUP.md`.

## The three commands

```bash
source .venv/bin/activate

# 1. RUN - first time with the folder, and again every time you add new images
image-pipeline run /path/to/images --rights-status user_attested --workers 4   # first time
image-pipeline run                                                             # later: only new/changed images are processed

# 2. RESUME - after a crash, shutdown, Ctrl+C or error: continue from where it stopped
image-pipeline resume --workers 4

# 3. STATUS - stage table of the run + checkpoint table per image
image-pipeline status                  # everything
image-pipeline status img_ab12cd34...  # one image, stage by stage
image-pipeline status --watch 5        # refresh every 5 s from a second terminal
image-pipeline status --failed         # only images with a failed stage
image-pipeline status --json           # for scripts
```

What `run` prints:

```
=== VALIDATE  File validation and metadata
[VALIDATE] img_da60ebbcb3bbb5c2 ✓ 768x576 JPEG RGB (0.01s)
[VALIDATE] img_a86ce0557a7bef89 ! INVALID (unreadable): not a decodable image (0.10s)
[validate 120/410] failed=0  elapsed=00:12  eta=00:30  9.8 img/s
[VALIDATE] ✓ 410 images: 410 processed, 0 cached, 0 failed (41.2s)
[OCR] - skipped: disabled in configuration (optional stage)
...
Run summary (01:47)   <- table: stage, result, processed, cached, failed, time
completed: 282/410 images in the dataset
```

Run the same command again and finished work shows as `cached`. `--quiet` drops the per-image lines.
Everything is also written to `workspace/logs/pipeline.log`.

- State lives in `workspace/checkpoints/pipeline.db`; a pid lock (`run.lock`) lets only one pipeline
  process run per workspace. A second `run` while one is alive exits with code 3. If the process was
  killed, the lock is stale: the next `resume` (or `run`) marks the unfinished stage as interrupted and
  continues. `status` never disturbs a live run. More in `RESUME_COMMANDS.md`.
- The workspace is `./workspace` inside this project unless `IMAGE_PIPELINE_WORKSPACE` is set (also read
  from `.env`) or you pass `--workspace DIR`.
- `--rights-status` records the rights **you** hold for the images; the tool does not verify them. With
  the default strict policy, images without `verified` or `user_attested` are analysed but **not exported**.

### Shard mode from a Hugging Face dataset

```bash
image-pipeline run hf://UWGZQ/pixmo_images --rights-status user_attested --workers 4 --max-images 2000   # try a part first
image-pipeline run hf://UWGZQ/pixmo_images      # no limit: continue through the whole dataset
image-pipeline resume                           # after any stop
image-pipeline status --watch 5
```

The dataset's page URL works too. The dataset must be public and store images as bytes in parquet files.
One parquet file at a time is downloaded (resumable), its images are taken out row by row, processed, and
the output is uploaded in shards exactly as below; a parquet file is deleted once all its rows are consumed.
The licence text from the dataset card is recorded on every record, unverified. GIFs and other unsupported
formats are skipped and counted.

### Shard mode (folder of URL lists instead of images)

If the folder you pass contains `TsvHttpData-1.0` URL lists (`.tsv`: url, size, md5) and no images, `run`
switches to shard mode automatically:

```bash
image-pipeline run Images --rights-status user_attested --workers 4 --max-urls 1000   # try 1,000 URLs first
image-pipeline run Images            # no limit: continue through the lists
image-pipeline resume                # after any stop: continues at the stored URL position
image-pipeline status                # adds a "Shard mode" table (URLs consumed, current shard, uploads)
```

It downloads a batch (size and checksum verified), runs the normal pipeline on it, and when the output
reaches the shard size (`HF_MIN_SIZE_MB`, 1024 by default) it closes the shard, uploads it to
`shards/shard_0001/...` in the Hugging Face dataset, deletes the downloaded originals and the local shard,
and continues. If the upload fails, the shard stays in `workspace/shards/` and is retried on the next run.
With the upload off, shards simply accumulate in `workspace/shards/`. The URL lists are never modified.
A shard that has not reached the size threshold stays open until more URLs are processed. Settings are
under `shards:` in the config (download workers, timeouts, `max_file_mb`, `delete_after_upload`).

Advanced commands still exist but are hidden from `--help`: `retry-failed`, `review` (approve / reject /
quarantine / restore), `export`, `validate`, `report`, `upload`, `stages`, `cancel`, `benchmark`,
`sources`, `models`, `system-info`. Run any of them with `--help`.

Try it on generated test images:

```bash
bash scripts/demo.sh        # builds a synthetic fixture folder and runs the whole pipeline in ./demo_workspace
```

## Upload to Hugging Face (optional)

Off by default. When enabled, every successful `run` or `export` ends with an upload step that sends
`processed_dataset/` to a Hugging Face dataset repository **only if the folder is at least 1024 MB**;
below that it prints why it did not upload and moves on.

Fill in the two values in the `.env` file at the project root (git-ignored):

```
HF_TOKEN=            # write-access token from https://huggingface.co/settings/tokens
HF_DATASET_REPO=     # username/dataset-name; filling this in turns the upload on
```

`HF_PRIVATE` and `HF_MIN_SIZE_MB` in the same file control visibility and the size threshold. The token
is only handed to the Hugging Face client; it is never written to the config, database, logs or dataset.

The same settings can instead live in `<workspace>/configs/pipeline_config.yaml` (values in `.env` win):

```yaml
huggingface:
  enabled: true
  repo_id: your-username/your-dataset   # created if it does not exist
  private: false                        # false = PUBLIC: anyone can view and download
  min_size_mb: 1024
```

- Only export-eligible images are in the upload (rights status allowed, accepted). Quarantined
  renditions (`rejected/quarantine/`) are never sent.
- The upload is refused if the last export failed validation.
- Re-uploads send only new or changed files and remove remote files that left the dataset.
- `image-pipeline upload --workspace ./workspace` uploads on demand; add `--force` to ignore the size threshold.
- A public repository publishes your images and captions to everyone. Make sure you hold the rights to do that.

## Output

```
processed_dataset/
  images/            processed image files (+ metadata.jsonl for Hugging Face imagefolder)
  manifests/         all_records / train / validation / test in jsonl, csv, parquet; minimal_<split>.jsonl
  metadata/          image_metadata, captions, quality_scores, annotations, provenance (all images)
  rejected/          rejected_records.jsonl with reasons; quarantine/ holds renditions removed from the dataset
  reports/           dataset_summary, duplicate, quality, alignment, rights, split_leakage, coverage, export_validation
  dataset_card.md    (README.md is the same card with the Hugging Face header)
  SHA256SUMS         checksum of every file in the package
```

Manifests use paths relative to `processed_dataset/`, so the folder can be moved. Each record follows
the versioned schema in `backend/app/schemas/record.py`.

## Captions: what you get by default

The captioning model was deliberately left undecided, so the default adapter is **`template`**: a
grounded baseline sentence assembled from stored annotations. These captions are accurate but generic,
are always sent to the review queue, and the export warns about them. Before training, switch to one of:

- `captioning.adapter: sidecar` — your own captions (`image.txt` next to each image, or a `caption`
  column in a metadata file);
- `captioning.adapter: smolvlm` — a local vision-language model from `configs/model_registry.yaml`.

Compare options on your own images first:

```bash
image-pipeline benchmark --workspace ./workspace --adapters template,smolvlm2-256m --sample 20
# fill in the generated review_sheet.csv, then:
image-pipeline benchmark-summarize path/to/review_sheet.csv
```

### Captions from the source, a model only where needed (`adapter: auto`)

```yaml
captioning:
  adapter: auto                 # caption that came with the source image, else the local VLM
  supplied_source: pixmo-cap    # recorded as captions.source for source captions
shards:
  caption_dataset: allenai/pixmo-cap   # image_url + caption rows, joined by sha256(image_url)
```

The caption dataset is downloaded and indexed once (`workspace/sources/`). An image that has a caption
there never reaches the model; the VLM is loaded the first time an image without one turns up. Every
image prints its own line, so both kinds show in the same log:

```
[CAPTION] img_976bd35fde616648 ✓ pixmo-cap: "This black-and-white illustration, possibly from a comic..." (0.00s)
[CAPTION] an image has no caption from the source: loading smolvlm2-256m (used only for such images)
[CAPTION] img_e71635cc7a9ff1d1 ✓ smolvlm2-256m: "A rectangular piece of red granite with black speckles..." (1.21s)
[CAPTION] 120 caption(s) taken from the source, 44 generated by smolvlm2-256m (1.5s per image)
```

`captioning.vlm_image_splitting` stays `false` on small machines: with it on, SmolVLM2-256M took
23-220 s per image on an 8 GB Apple M2, against 1-5 s with it off.

### What keeps an image out of the dataset

| Rule | Setting | Result |
|---|---|---|
| Smaller than the minimum size, or low overall quality outside the review band | `quality.min_width/min_height`, `min_overall_score` | rejected |
| Caption too dissimilar to the image (CLIP) | `captioning.quarantine_failed_verification` | quarantined |
| Watermark or credit text read in the image (needs `stages.ocr: true`) | `quality.reject_watermarked` | quarantined |
| Very low aesthetic signal, whatever the overall score | `quality.review_min_aesthetic_score` | quarantined |
| Duplicate inside the shard, or of an image exported in an earlier shard | `deduplication.*`, `deduplication.cross_shard` | quarantined |

Quarantined images are never deleted; `image-pipeline review` can approve them.

## Tests

```bash
cd backend && python -m pytest          # 97 tests; model tests skip when weights are not cached
```

Tests use only procedurally generated images (`backend/app/tests/fixtures.py`).
