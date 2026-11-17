"""Runtime hardware inspection and device/batch-size selection. Nothing is hard-coded."""
from __future__ import annotations

import importlib.util
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

import psutil


def _torch():
    if importlib.util.find_spec("torch") is None:
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
    return torch


def gpu_info() -> list[dict[str, Any]]:
    torch = _torch()
    gpus: list[dict[str, Any]] = []
    if torch is None:
        return gpus
    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                gpus.append({
                    "runtime": "cuda", "index": i, "name": torch.cuda.get_device_name(i),
                    "vram_total_gb": round(total / 1024**3, 2), "vram_free_gb": round(free / 1024**3, 2),
                })
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            # Apple silicon shares system memory between CPU and GPU.
            vm = psutil.virtual_memory()
            gpus.append({
                "runtime": "mps", "index": 0, "name": _apple_chip() or "Apple GPU (MPS)",
                "vram_total_gb": round(vm.total / 1024**3, 2), "vram_free_gb": round(vm.available / 1024**3, 2),
                "shared_memory": True,
            })
    except Exception:  # hardware probing must never break the app
        pass
    return gpus


def _apple_chip() -> Optional[str]:
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=3)
        return out.stdout.strip() or None
    except Exception:
        return None


def system_info(disk_path: Optional[Path] = None) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    p = Path(disk_path or ".")
    while not p.exists() and p != p.parent:
        p = p.parent
    du = shutil.disk_usage(p)
    torch = _torch()
    return {
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu": {"model": _apple_chip() or platform.processor() or "unknown", "logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False)},
        "ram": {"total_gb": round(vm.total / 1024**3, 2), "available_gb": round(vm.available / 1024**3, 2)},
        "disk": {"total_gb": round(du.total / 1024**3, 1), "free_gb": round(du.free / 1024**3, 1)},
        "gpus": gpu_info(),
        "runtimes": {
            "torch": getattr(torch, "__version__", None) if torch else None,
            "transformers": _version("transformers"),
            "opencv": _version("cv2"),
            "pillow": _version("PIL"),
            "pyarrow": _version("pyarrow"),
            "rapidocr": _version("rapidocr_onnxruntime") or _version("rapidocr"),
        },
    }


def _version(mod: str) -> Optional[str]:
    if importlib.util.find_spec(mod) is None:
        return None
    try:
        import importlib

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = importlib.import_module(mod)
        return str(getattr(m, "__version__", "installed"))
    except Exception:
        return None


def usage() -> dict[str, Any]:
    """Live resource telemetry for the dashboard."""
    vm = psutil.virtual_memory()
    out: dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": vm.percent,
        "ram_used_gb": round((vm.total - vm.available) / 1024**3, 2),
        "ram_total_gb": round(vm.total / 1024**3, 2),
        "gpu": None,
    }
    torch = _torch()
    try:
        if torch is not None and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            out["gpu"] = {"runtime": "cuda", "memory_used_gb": round((total - free) / 1024**3, 2), "memory_total_gb": round(total / 1024**3, 2)}
        elif torch is not None and torch.backends.mps.is_available():
            out["gpu"] = {"runtime": "mps", "memory_used_gb": round(torch.mps.current_allocated_memory() / 1024**3, 3), "memory_total_gb": None}
    except Exception:
        pass
    return out


class DeviceUnavailable(RuntimeError):
    pass


def select_device(requested: str = "auto", allow_cpu_fallback: bool = True) -> str:
    torch = _torch()
    if torch is None:
        raise DeviceUnavailable("PyTorch is not installed; install the 'ml' extra to use model stages")
    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    if requested == "cuda":
        if has_cuda:
            return "cuda"
    elif requested == "mps":
        if has_mps:
            return "mps"
    elif requested == "cpu":
        return "cpu"
    else:  # auto
        if has_cuda:
            return "cuda"
        if has_mps:
            return "mps"
    if allow_cpu_fallback:
        return "cpu"
    raise DeviceUnavailable(f"device '{requested}' is unavailable and CPU fallback is disabled")


def auto_batch_size(device: str, per_item_mb: float = 60.0, cap: int = 32) -> int:
    """Conservative default: use at most ~25% of currently free accelerator/system memory."""
    try:
        if device == "cuda":
            torch = _torch()
            free, _ = torch.cuda.mem_get_info(0)
            free_mb = free / 1024**2
        else:
            free_mb = psutil.virtual_memory().available / 1024**2
        return int(max(1, min(cap, (free_mb * 0.25) // per_item_mb)))
    except Exception:
        return 4


def is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in msg or "mps backend out of memory" in msg
