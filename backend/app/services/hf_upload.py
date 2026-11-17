"""Upload the finished dataset folder to a Hugging Face dataset repository.

Opt-in only (``huggingface.enabled`` plus ``huggingface.repo_id``). It runs after a successful
export and only when the output folder has reached ``huggingface.min_size_mb``. The access
token is never read, stored or logged by this application: it comes from ``hf auth login``
or the ``HF_TOKEN`` environment variable and is used only by the Hugging Face client in a child
process (the rest of the application stays in offline mode)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Optional

from app.core.logging import get_logger
from app.core.workspace import Workspace
from app.database.models import utcnow
from app.database.session import kv_get, kv_set

log = get_logger("hf_upload")

# Never uploaded: renditions removed from the dataset, temp/hidden files.
IGNORE = ["rejected/quarantine/**", "rejected/quarantine/*", ".*", "**/.*", "*.tmp", "**/*.tmp"]
# Remote files under these folders that no longer exist locally are removed, so the repository mirrors the dataset.
MIRRORED = ["images/**", "manifests/**", "metadata/**", "reports/**", "annotations/**", "rejected/**"]

_CHILD = r"""
import json, sys
from huggingface_hub import HfApi
a = json.loads(sys.argv[1])
api = HfApi()
try:
    user = api.whoami()["name"]
except Exception as exc:
    print(json.dumps({"ok": False, "error": "not logged in to Hugging Face: run `hf auth login` or set HF_TOKEN"}))
    sys.exit(0)
if "/" not in a["repo_id"]:
    a["repo_id"] = user + "/" + a["repo_id"]   # bare dataset name: use the logged-in account
try:
    api.create_repo(a["repo_id"], repo_type="dataset", private=a["private"], exist_ok=True)
    info = api.upload_folder(repo_id=a["repo_id"], repo_type="dataset", folder_path=a["folder"],
                             path_in_repo=a.get("path_in_repo"), commit_message=a["message"],
                             ignore_patterns=a["ignore"], delete_patterns=a["mirrored"])
    if a.get("readme"):
        api.upload_file(path_or_fileobj=a["readme"].encode("utf-8"), path_in_repo="README.md", repo_id=a["repo_id"],
                        repo_type="dataset", commit_message="Update dataset card")
    print(json.dumps({"ok": True, "user": user, "commit_url": str(getattr(info, "commit_url", "")),
                      "repo_url": "https://huggingface.co/datasets/" + a["repo_id"]}))
except Exception as exc:
    print(json.dumps({"ok": False, "error": type(exc).__name__ + ": " + str(exc)[:400]}))
"""


def _ignored(rel: str) -> bool:
    return any(fnmatch(rel, pat) for pat in IGNORE) or any(part.startswith(".") for part in rel.split("/"))


def upload_size(folder: Path) -> tuple[int, int]:
    """(bytes, file count) of what would be uploaded."""
    total, n = 0, 0
    if folder.is_dir():
        for f in folder.rglob("*"):
            if f.is_file() and not _ignored(f.relative_to(folder).as_posix()):
                total += f.stat().st_size
                n += 1
    return total, n


def _run_child(args: dict) -> dict:
    env = {**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}
    proc = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(args)], env=env, capture_output=True, text=True)
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        return {"ok": False, "error": "upload process failed: " + (proc.stderr.strip().splitlines() or ["no output"])[-1][:300]}
    return json.loads(lines[-1])


def maybe_upload(ws: Workspace, cfg=None, force: bool = False, say: Callable[[str], None] = lambda m: None,
                 uploader: Optional[Callable[[dict], dict]] = None) -> dict:
    """Upload when every condition holds; otherwise return why nothing was uploaded.
    ``force`` skips only the size threshold (manual ``image-pipeline upload --force``)."""
    cfg = cfg or ws.config
    hf = cfg.huggingface
    out = ws.output_dir
    size, files = upload_size(out)
    size_mb = round(size / 1024**2, 1)
    base = {"uploaded": False, "repo_id": hf.repo_id, "size_mb": size_mb, "threshold_mb": hf.min_size_mb, "files": files}

    def skip(reason: str) -> dict:
        say(f"not uploaded: {reason}")
        return {**base, "reason": reason}

    if not hf.enabled:
        return skip("huggingface.enabled is false")
    if not hf.repo_id:
        return skip("huggingface.repo_id is not set (expected: username/dataset-name)")
    with ws.db.session() as s:
        export = kv_get(s, "export:last", None)
        last = kv_get(s, "hf:last", None)
    if not export or not export.get("exported"):
        return skip("there is no exported dataset yet")
    if hf.require_validation_passed and not export.get("validation_passed"):
        return skip("the last export did not pass validation")
    if not force and size_mb < hf.min_size_mb:
        return skip(f"output folder is {size_mb} MB, below the {hf.min_size_mb} MB threshold")
    if last and last.get("export_generated_at") == export.get("generated_at") and last.get("repo_id") == hf.repo_id:
        return skip("this export was already uploaded")

    say(f"uploading {files} files ({size_mb} MB) to {hf.repo_id} ({'private' if hf.private else 'PUBLIC'})")
    result = (uploader or _run_child)({
        "repo_id": hf.repo_id, "private": hf.private, "folder": str(out), "ignore": IGNORE, "mirrored": MIRRORED,
        "message": f"{hf.commit_message}: {export['exported']} records, {size_mb} MB (config {cfg.config_hash()})"})
    if not result.get("ok"):
        log.error("Hugging Face upload failed: %s", result.get("error"))
        raise RuntimeError(result.get("error") or "upload failed")
    record = {"repo_id": hf.repo_id, "private": hf.private, "size_mb": size_mb, "files": files,
              "records": export["exported"], "uploaded_at": utcnow().isoformat(),
              "export_generated_at": export.get("generated_at"), "commit_url": result.get("commit_url"),
              "repo_url": result.get("repo_url")}
    with ws.db.session() as s:
        kv_set(s, "hf:last", record)
    say(f"uploaded: {record['repo_url']}")
    return {**base, "uploaded": True, "repo_url": record["repo_url"], "commit_url": record["commit_url"], "records": export["exported"]}
