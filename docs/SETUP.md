# Setup

Tested on macOS 15 (Apple silicon, Python 3.12). Linux and Windows are expected to work because the
code uses only cross-platform libraries, but **they have not been run** — see `LIMITATIONS.md`.

## All platforms

```bash
python3 -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -e "backend[ml,ocr,dev]"
image-pipeline system-info
```

Extras:

| Extra | Adds | Needed for |
|---|---|---|
| *(none)* | Pillow, NumPy, SciPy, OpenCV, SQLAlchemy, PyArrow | scan, validation, hashing, technical quality, dedup (exact/perceptual), image writing, splits, export |
| `ml` | PyTorch, transformers, huggingface_hub | embeddings, semantic similarity, zero-shot tags, aesthetic signal, caption verification, VLM captions |
| `ocr` | rapidocr-onnxruntime (bundles small ONNX models, about 15 MB) | visible-text detection |
| `dev` | pytest | running the tests |

### PyTorch per platform

- **macOS (Apple silicon):** the default `pip install torch` wheel includes MPS acceleration.
- **Linux / Windows with NVIDIA:** install the CUDA build that matches your driver *before* installing
  the package, following the selector at pytorch.org, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu124`.
- **CPU only:** works; model stages are slow. `image-pipeline system-info` prints a warning.

If you already have `opencv-python` installed, `opencv-python-headless` from the dependency list is
redundant; installing with `pip install --no-deps -e backend` avoids pulling a second copy.

### Windows notes

- Use a workspace path without very long nesting (classic 260-character path limit).
- Run commands from PowerShell or `cmd`; `scripts/demo.sh` needs Git Bash or WSL — the equivalent is:
  `python scripts\make_fixtures.py demo_images` then the three `image-pipeline` commands in the script.

## Models

Nothing is downloaded automatically, and the application runs the Hugging Face libraries in offline
mode. To add a model:

```bash
image-pipeline models list                         # size, licence, memory, installed?
image-pipeline models download clip-vit-base-patch32          # prints the plan, downloads nothing
image-pipeline models download clip-vit-base-patch32 --yes    # downloads
```

See `HARDWARE_AND_MODELS.md`.

## Environment variables

See `.env.example`. None are required for local use.
