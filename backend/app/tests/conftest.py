from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from app.config.settings import PipelineConfig
from app.core.workspace import Workspace
from app.pipeline.runner import create_job, run_job
from app.tests.fixtures import build_fixture_dataset


def merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def base_config(source_dir: Path) -> dict:
    return {
        "paths": {"sources": [{"path": str(source_dir), "label": "fixture", "rights_status": "user_attested",
                               "license": "synthetic-test-data", "metadata_file": "metadata.csv"}]},
        "processing": {"cpu_workers": 2},
        # Model-backed stages are exercised separately in test_models.py.
        "stages": {"embeddings": False},
        "deduplication": {"near_candidate_hamming": 30},
    }


def make_ws(root: Path, source_dir: Path, **overrides) -> Workspace:
    ws = Workspace(root)
    ws.save_config(PipelineConfig.model_validate(merge(base_config(source_dir), overrides)))
    return ws


def reconfigure(ws: Workspace, **overrides) -> None:
    ws.save_config(PipelineConfig.model_validate(merge(ws.config.model_dump(mode="json"), overrides)))


def run(ws: Workspace, stages=None, retry_failed=False, on_stage=None) -> dict:
    return run_job(ws, create_job(ws, stages), retry_failed=retry_failed, on_stage=on_stage)


def tree_checksums(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


import os

# Tests never read the developer's dotenv file or Hugging Face settings, and never upload.
os.environ["IMAGE_PIPELINE_SKIP_DOTENV"] = "1"
for _k in ("HF_TOKEN", "HF_DATASET_REPO", "HF_PRIVATE", "HF_MIN_SIZE_MB", "HF_UPLOAD_ENABLED"):
    os.environ.pop(_k, None)


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("source") / "images"
    build_fixture_dataset(d)
    return d


@pytest.fixture(scope="session")
def fixture_info(tmp_path_factory) -> dict:
    return build_fixture_dataset(tmp_path_factory.mktemp("info") / "images")


@pytest.fixture()
def ws(tmp_path, fixture_dir) -> Workspace:
    w = make_ws(tmp_path / "ws", fixture_dir)
    yield w
    w.db.dispose()


@pytest.fixture(scope="session")
def done(tmp_path_factory, fixture_dir):
    """One complete pipeline run shared by read-only tests: (workspace, job summary, source checksums before)."""
    before = tree_checksums(fixture_dir)
    w = make_ws(tmp_path_factory.mktemp("done") / "ws", fixture_dir, export={"formats": ["jsonl", "csv", "parquet"]})
    summary = run(w)
    yield w, summary, before
    w.db.dispose()
