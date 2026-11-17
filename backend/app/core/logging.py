"""Logging with redaction of secrets and private absolute paths."""
from __future__ import annotations

import logging
import os
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_SECRET = re.compile(
    r"(?i)\b(api[_-]?key|token|secret|password|authorization)\b(\s*[=:]\s*|\s+)(bearer\s+)?[^\s,;'\"]+"
)
_HOME = str(Path.home())


def redact(text: str) -> str:
    text = _SECRET.sub(lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text)
    if _HOME and _HOME in text:
        text = text.replace(_HOME, "~")
    return text


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


class _StderrHandler(logging.StreamHandler):
    """Always writes to the current ``sys.stderr`` (which test runners and shells may swap)."""

    @property
    def stream(self):
        import sys

        return sys.stderr

    @stream.setter
    def stream(self, value):
        pass


_configured: set[str] = set()


def setup_logging(logs_dir: Path | None = None, level: str | None = None) -> logging.Logger:
    logger = logging.getLogger("pipeline")
    logger.setLevel(level or os.environ.get("IMAGE_PIPELINE_LOG_LEVEL", "INFO"))
    fmt = RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    if "console" not in _configured:
        h = _StderrHandler()
        h.setFormatter(fmt)
        h.setLevel(logging.WARNING)
        logger.addHandler(h)
        _configured.add("console")
    if logs_dir is not None:
        key = str(Path(logs_dir).resolve())
        if key not in _configured:
            Path(logs_dir).mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(Path(logs_dir) / "pipeline.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
            _configured.add(key)
    logger.propagate = False
    return logger


def get_logger(name: str = "") -> logging.Logger:
    return logging.getLogger("pipeline" + (f".{name}" if name else ""))
