"""A workspace is one dataset project: configuration, job database, logs, cache and outputs."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from app.config.settings import PipelineConfig, load_config, save_config
from app.core.env import apply_hf_env, load_env
from app.core.io import is_within
from app.core.logging import setup_logging
from app.database.session import Database


class Workspace:
    def __init__(self, root: Path, config_path: Optional[Path] = None):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path).expanduser().resolve() if config_path else self.root / "configs" / "pipeline_config.yaml"
        load_env([self.root])
        self.config: PipelineConfig = apply_hf_env(load_config(self.config_path))
        self.db = Database(self.checkpoints_dir / "pipeline.db")
        self.log = setup_logging(self.logs_dir)

    # -- paths -----------------------------------------------------------
    def _resolve(self, p: str) -> Path:
        path = Path(p).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @property
    def output_dir(self) -> Path:
        return self._resolve(self.config.paths.output_dir)

    @property
    def checkpoints_dir(self) -> Path:
        return self._resolve(self.config.paths.checkpoints_dir)

    @property
    def logs_dir(self) -> Path:
        return self._resolve(self.config.paths.logs_dir)

    @property
    def cache_dir(self) -> Path:
        return self._resolve(self.config.paths.cache_dir)

    @property
    def images_dir(self) -> Path:
        return self.output_dir / "images"

    # -- config ----------------------------------------------------------
    def reload_config(self) -> PipelineConfig:
        self.config = apply_hf_env(load_config(self.config_path))
        return self.config

    def save_config(self, cfg: PipelineConfig) -> None:
        save_config(cfg, self.config_path)
        self.config = apply_hf_env(cfg)

    # -- safety ----------------------------------------------------------
    def validate_source_dir(self, path: str) -> Path:
        """A source must be an existing directory that neither contains nor lives inside
        any directory this application writes to."""
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p)
        p = p.resolve()
        if not p.is_dir():
            raise ValueError("source folder does not exist or is not a directory")
        for out in (self.output_dir, self.checkpoints_dir, self.logs_dir, self.cache_dir):
            if is_within(p, out):
                raise ValueError("source folder must not be inside a pipeline output directory")
            if is_within(out, p):
                raise ValueError(
                    "pipeline output directories must not be inside a source folder (sources are read-only); "
                    "choose a workspace outside the image folder"
                )
        if not os.access(p, os.R_OK):
            raise ValueError("source folder is not readable")
        return p


def default_workspace_root() -> Path:
    """$IMAGE_PIPELINE_WORKSPACE if set (also from the dotenv file), otherwise ``<project>/workspace``,
    so the three everyday commands work from any directory without extra options."""
    from app.core.env import PROJECT_ROOT

    load_env()
    env = os.environ.get("IMAGE_PIPELINE_WORKSPACE")
    return Path(env).expanduser().resolve() if env else (PROJECT_ROOT / "workspace").resolve()
