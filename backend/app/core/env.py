"""Minimal dotenv loading and the environment overrides for the Hugging Face upload.

Values already present in the real environment always win. Secrets are only placed into the
process environment (for the Hugging Face client); they are never written to the config,
the database, logs or reports."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE_NAME = "." + "env"
_TRUE, _FALSE = {"1", "true", "yes", "on"}, {"0", "false", "no", "off"}


def parse_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key and value:  # empty placeholders are ignored
            out[key] = value
    return out


def load_env(extra_dirs: Iterable[Path] = ()) -> list[Path]:
    """Load the dotenv file from the workspace, the current directory and the project root (first wins)."""
    loaded = []
    seen = set()
    if os.environ.get("IMAGE_PIPELINE_SKIP_DOTENV") == "1":  # used by the test suite
        return loaded
    for d in [*extra_dirs, Path.cwd(), PROJECT_ROOT]:
        p = (Path(d) / ENV_FILE_NAME).resolve()
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        for k, v in parse_env_file(p).items():
            os.environ.setdefault(k, v)
        loaded.append(p)
    return loaded


def apply_hf_env(cfg):
    """HF_DATASET_REPO turns the upload on for that repository; HF_UPLOAD_ENABLED / HF_PRIVATE /
    HF_MIN_SIZE_MB adjust it. Returns a validated copy (or ``cfg`` itself when nothing is set)."""
    repo = os.environ.get("HF_DATASET_REPO", "").strip()
    enabled = os.environ.get("HF_UPLOAD_ENABLED", "").strip().lower()
    private = os.environ.get("HF_PRIVATE", "").strip().lower()
    min_mb = os.environ.get("HF_MIN_SIZE_MB", "").strip()
    if not (repo or enabled or private or min_mb):
        return cfg
    data = cfg.model_dump(mode="json")
    hf = data["huggingface"]
    if repo:
        hf["repo_id"], hf["enabled"] = repo, True
    if enabled in _TRUE | _FALSE:
        hf["enabled"] = enabled in _TRUE
    if private in _TRUE | _FALSE:
        hf["private"] = private in _TRUE
    if min_mb:
        hf["min_size_mb"] = float(min_mb)
    return type(cfg).model_validate(data)
