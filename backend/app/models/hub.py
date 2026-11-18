"""Model registry and lazily loaded adapters.

Pipeline stages talk to adapters, never to a model library directly, so a model can be
swapped by editing the registry/configuration. Weights are loaded from the local cache
only; downloading requires an explicit user action (``download_model(confirm=True)``)."""
from __future__ import annotations

import importlib.util
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from app.core.hardware import auto_batch_size, select_device
from app.core.logging import get_logger
from app.pipeline.base import ModelUnavailable

log = get_logger("models")
_PKG_REGISTRY = Path(__file__).with_name("model_registry.yaml")


@dataclass
class ModelSpec:
    key: str
    task: list[str]
    adapter: str
    repo: str
    download_size_gb: Optional[float] = None
    ram_gb: Optional[float] = None
    vram_gb: Optional[float] = None
    license: Optional[str] = None
    notes: Optional[str] = None
    extra: dict = field(default_factory=dict)


def load_registry(workspace_root: Optional[Path] = None) -> dict[str, ModelSpec]:
    path = _PKG_REGISTRY
    if workspace_root is not None and (workspace_root / "configs" / "model_registry.yaml").exists():
        path = workspace_root / "configs" / "model_registry.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out = {}
    for key, v in (raw.get("models") or {}).items():
        known = {k: v.get(k) for k in ("task", "adapter", "repo", "download_size_gb", "ram_gb", "vram_gb", "license", "notes")}
        if not known["repo"] or not known["adapter"]:
            raise ValueError(f"model registry entry '{key}' needs 'repo' and 'adapter'")
        known["task"] = known["task"] or []
        out[key] = ModelSpec(key=key, **known)
    return out


def _hf_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers", "huggingface_hub"))


def cached_revision(repo: str) -> Optional[str]:
    """Snapshot hash of a locally cached Hugging Face model, or None when it is not present."""
    if importlib.util.find_spec("huggingface_hub") is None:
        return None
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo, "config.json")
    if not isinstance(p, str):
        return None
    snap = Path(p).parent
    has_weights = any(snap.glob("*.safetensors")) or any(snap.glob("*.bin"))
    return snap.name if has_weights else None


def _quiet():
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()


class ClipAdapter:
    """Image/text embedding adapter (CLIP family via transformers)."""

    def __init__(self, spec: ModelSpec, device: str, revision: str, local_only: bool):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        _quiet()
        self.spec, self.device, self.revision, self.torch = spec, device, revision, torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = CLIPModel.from_pretrained(spec.repo, local_files_only=local_only).to(device).eval()
            self.processor = CLIPProcessor.from_pretrained(spec.repo, local_files_only=local_only)
        self.logit_scale = float(self.model.logit_scale.exp().item())
        self._text_cache: dict[str, np.ndarray] = {}

    def info(self) -> dict:
        return {"name": self.spec.key, "repo": self.spec.repo, "revision": self.revision,
                "precision": "float32", "quantization": None, "library": "transformers"}

    def _features(self, out):
        t = out if isinstance(out, self.torch.Tensor) else out.pooler_output
        return self.torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy().astype(np.float32)

    def encode_images(self, images: list) -> np.ndarray:
        with self.torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            return self._features(self.model.get_image_features(**inputs))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._text_cache]
        for i in range(0, len(missing), 64):
            chunk = missing[i:i + 64]
            with self.torch.no_grad():
                inputs = self.processor(text=chunk, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
                feats = self._features(self.model.get_text_features(**inputs))
            for t, f in zip(chunk, feats):
                self._text_cache[t] = f
        return np.stack([self._text_cache[t] for t in texts])

    def token_count(self, text: str) -> int:
        return len(self.processor.tokenizer(text)["input_ids"])


class SmolVLMAdapter:
    """Local vision-language captioner (SmolVLM family via transformers)."""

    def __init__(self, spec: ModelSpec, device: str, revision: str, local_only: bool):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _quiet()
        self.spec, self.device, self.revision, self.torch = spec, device, revision, torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.processor = AutoProcessor.from_pretrained(spec.repo, local_files_only=local_only)
            self.model = AutoModelForImageTextToText.from_pretrained(
                spec.repo, local_files_only=local_only, torch_dtype=torch.float32).to(device).eval()

    def info(self) -> dict:
        return {"name": self.spec.key, "repo": self.spec.repo, "revision": self.revision, "precision": "float32",
                "quantization": None, "library": "transformers", "decoding": "greedy"}

    def ask(self, image, prompt: str, max_new_tokens: int = 160, split_image: Optional[bool] = None) -> str:
        """``split_image=False`` shows the model the image once instead of as a grid of tiles: about a
        tenth of the input tokens. On an 8 GB Apple M2 that was 3-5 s per image instead of 23-220 s,
        with captions of the same quality on the images compared."""
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        extra = {} if split_image is None else {"do_image_splitting": split_image}
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt", **extra).to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.1)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return " ".join(text.split()).strip()


class ZeroShotDetectorAdapter:
    """Open-vocabulary object detector (OWLv2 family through transformers)."""

    def __init__(self, spec: ModelSpec, device: str, revision: str, local_only: bool):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        _quiet()
        self.spec, self.device, self.revision, self.torch = spec, device, revision, torch
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.processor = AutoProcessor.from_pretrained(spec.repo, local_files_only=local_only)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(spec.repo, local_files_only=local_only).to(device).eval()

    def info(self) -> dict:
        return {"name": self.spec.key, "repo": self.spec.repo, "revision": self.revision, "precision": "float32", "library": "transformers"}

    def detect(self, image, labels: list[str], threshold: float = 0.25) -> list[dict]:
        w, h = image.size
        side = max(w, h)  # the processor pads the image to a square, and boxes come back in that square's coordinates
        inputs = self.processor(text=[[f"a photo of a {x}" for x in labels]], images=image, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        found = self.processor.post_process_grounded_object_detection(
            outputs, threshold=threshold, target_sizes=self.torch.tensor([[side, side]]))[0]
        out = []
        for box, score, idx in zip(found["boxes"].tolist(), found["scores"].tolist(), found["labels"].tolist()):
            x0, y0 = min(max(box[0] / w, 0.0), 1.0), min(max(box[1] / h, 0.0), 1.0)
            x1, y1 = min(max(box[2] / w, 0.0), 1.0), min(max(box[3] / h, 0.0), 1.0)
            if x1 <= x0 or y1 <= y0:
                continue
            out.append({"label": labels[int(idx)], "confidence": round(min(max(float(score), 0.0), 1.0), 4),
                        "bbox": [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]})
        return sorted(out, key=lambda o: -o["confidence"])


_ADAPTERS = {"clip": ClipAdapter, "smolvlm": SmolVLMAdapter, "hf_zero_shot_detector": ZeroShotDetectorAdapter}


class ModelHub:
    def __init__(self, cfg, workspace_root: Optional[Path] = None):
        self.cfg = cfg
        self.registry = load_registry(workspace_root)
        self._loaded: dict[str, Any] = {}

    # -- discovery --------------------------------------------------------
    def spec(self, key: str) -> ModelSpec:
        if key not in self.registry:
            raise ModelUnavailable(f"model '{key}' is not in the model registry")
        return self.registry[key]

    def status(self, key: str) -> dict:
        """Why a model can or cannot be used right now (never triggers a download)."""
        if key in (None, "", "none"):
            return {"usable": False, "reason": "no model configured"}
        if key not in self.registry:
            return {"usable": False, "reason": f"'{key}' is not in the model registry"}
        if not _hf_available():
            return {"usable": False, "reason": "torch/transformers are not installed (pip install '.[ml]')"}
        spec = self.registry[key]
        rev = cached_revision(spec.repo)
        if rev is None:
            return {"usable": False, "installed": False,
                    "reason": f"weights for '{key}' are not in the local cache; run: image-pipeline models download {key}"}
        try:
            device = select_device(self.cfg.processing.device, self.cfg.processing.allow_cpu_fallback)
        except Exception as exc:
            return {"usable": False, "installed": rev is not None, "reason": str(exc)}
        return {"usable": True, "installed": rev is not None, "revision": rev, "device": device, "reason": ""}

    def describe(self) -> list[dict]:
        rows = []
        for key, spec in self.registry.items():
            st = self.status(key)
            rows.append({"key": key, "task": spec.task, "adapter": spec.adapter, "repo": spec.repo,
                         "download_size_gb": spec.download_size_gb, "ram_gb": spec.ram_gb, "vram_gb": spec.vram_gb,
                         "license": spec.license, "notes": spec.notes, "installed": bool(st.get("installed")),
                         "usable": st["usable"], "reason": st.get("reason", ""), "revision": st.get("revision")})
        return rows

    def info(self, key: str) -> Optional[dict]:
        """Version information for idempotency keys, without loading the model."""
        st = self.status(key)
        if not st["usable"]:
            return None
        spec = self.registry[key]
        return {"name": key, "repo": spec.repo, "revision": st.get("revision"), "precision": "float32"}

    # -- loading ----------------------------------------------------------
    def get(self, key: str):
        if key in self._loaded:
            return self._loaded[key]
        st = self.status(key)
        if not st["usable"]:
            raise ModelUnavailable(st["reason"])
        spec = self.registry[key]
        cls = _ADAPTERS.get(spec.adapter)
        if cls is None:
            raise ModelUnavailable(f"no adapter named '{spec.adapter}'")
        local_only = True
        log.info("loading model %s on %s (est. RAM %s GB)", key, st["device"], spec.ram_gb)
        adapter = cls(spec, st["device"], st.get("revision") or "unknown", local_only)
        self._loaded[key] = adapter
        return adapter

    def unload(self, key: Optional[str] = None) -> None:
        from app.pipeline.base import _free_memory

        for k in ([key] if key else list(self._loaded)):
            self._loaded.pop(k, None)
        _free_memory()

    def batch_size(self, key: str, per_item_mb: float = 60.0) -> int:
        bs = self.cfg.processing.gpu_batch_size
        if isinstance(bs, int):
            return bs
        st = self.status(key)
        return auto_batch_size(st.get("device", "cpu"), per_item_mb)


def download_model(hub: ModelHub, key: str, confirm: bool = False) -> dict:
    """Explicit, user-initiated download. Returns the plan when ``confirm`` is false."""
    spec = hub.spec(key)
    plan = {"key": key, "repo": spec.repo, "download_size_gb": spec.download_size_gb, "license": spec.license,
            "ram_gb": spec.ram_gb, "vram_gb": spec.vram_gb, "already_installed": cached_revision(spec.repo) is not None}
    if not confirm or plan["already_installed"]:
        return {**plan, "downloaded": False}
    # Runs in a child process so the rest of the application stays in offline mode.
    import subprocess
    import sys

    code = ("import sys; from huggingface_hub import snapshot_download; "
            "snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.txt', '*.safetensors', '*.model'])")
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}
    proc = subprocess.run([sys.executable, "-c", code, spec.repo], env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError("download failed: " + (proc.stderr.strip().splitlines() or ["unknown error"])[-1][:300])
    return {**plan, "downloaded": True, "revision": cached_revision(spec.repo)}
