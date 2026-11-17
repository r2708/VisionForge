# Pipeline run / resume / status commands

All commands are run from the project folder with the virtual environment active:

```bash
cd /path/to/Image_Dataset && source .venv/bin/activate
```

## Run

```bash
image-pipeline run /path/to/images --rights-status user_attested --workers 4   # first time
image-pipeline run --workers 4                                                 # after adding new images
image-pipeline run --quiet                                                     # without per-image lines
```

Running again is always safe: finished images and stages are reused (`cached`), only new or changed
images are processed.

## If the pipeline shuts down or crashes

**What gets resumed:**
- Finished stages and images: skipped (already complete)
- The stage that was running: marked interrupted, its unfinished images run
- Stages that never started: run now
- Failed images: left alone unless you ask for a retry

```bash
image-pipeline resume --workers 4            # continue from where it stopped
image-pipeline resume --retry-failed         # also retry images that failed
image-pipeline retry-failed                  # same as the line above
```

`resume` continues the last job with the configuration it was started with. If a pipeline process is
still alive for this workspace, `run` and `resume` exit with code 3 instead of starting a second one.

## Check status

```bash
image-pipeline status                  # stage table of the run + one row per image
image-pipeline status img_XXXXXXXX     # one image, every stage with its result or error
image-pipeline status --watch 5        # auto-refresh every 5 seconds
image-pipeline status --failed         # only images with a failed stage
image-pipeline status --limit 200      # more rows in the per-image table
image-pipeline status --json           # machine readable
```

Marks in the per-image table: `✓` done, `✗` failed, `…` running, `-` skipped, `·` pending.

## Stop it yourself

Press Ctrl+C (or `kill <pid>`). The job is marked interrupted and everything finished so far is kept;
continue with `image-pipeline resume`.

## Rebuild the export or upload without re-analysing

```bash
image-pipeline export                  # decisions, processed images, splits, manifests
image-pipeline validate                # re-check the exported dataset
image-pipeline upload --force          # upload now, ignoring the 1 GB threshold
```
