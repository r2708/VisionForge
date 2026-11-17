"""Hugging Face upload logic. The network call is replaced by a fake: nothing is uploaded by tests."""
import pytest
from pydantic import ValidationError

from app.config.settings import PipelineConfig
from app.services import hf_upload
from app.services.hf_upload import maybe_upload, upload_size
from app.tests.conftest import reconfigure, run


class Fake:
    def __init__(self, ok=True):
        self.calls, self.ok = [], ok

    def __call__(self, args):
        self.calls.append(args)
        if not self.ok:
            return {"ok": False, "error": "not logged in to Hugging Face: run `hf auth login` or set HF_TOKEN"}
        return {"ok": True, "commit_url": "https://huggingface.co/x/commit/1", "repo_url": "https://huggingface.co/datasets/" + args["repo_id"]}


def test_upload_is_off_by_default_and_repo_id_is_validated(done):
    ws, summary, _ = done
    assert PipelineConfig().huggingface.enabled is False and PipelineConfig().huggingface.min_size_mb == 1024
    assert summary["stages"]["hf_upload"]["status"] == "skipped"
    fake = Fake()
    assert maybe_upload(ws, uploader=fake)["reason"] == "huggingface.enabled is false" and not fake.calls
    assert PipelineConfig.model_validate({"huggingface": {"repo_id": "VisionForge"}}).huggingface.repo_id == "VisionForge"
    for bad in ("a/b/c", "has space/x", "/x", "name with space"):
        with pytest.raises(ValidationError):
            PipelineConfig.model_validate({"huggingface": {"repo_id": bad}})
    a = PipelineConfig.model_validate({"huggingface": {"enabled": True, "repo_id": "me/data"}})
    assert a.config_hash() == PipelineConfig().config_hash()  # publishing settings never invalidate cached work


def test_size_threshold_gates_the_upload(ws):
    run(ws)
    reconfigure(ws, huggingface={"enabled": True})
    fake = Fake()
    assert "repo_id is not set" in maybe_upload(ws, uploader=fake)["reason"]
    reconfigure(ws, huggingface={"repo_id": "someone/my-dataset"})
    out = maybe_upload(ws, uploader=fake)  # a few MB of fixtures: far below 1024 MB
    assert out["uploaded"] is False and "below the 1024.0 MB threshold" in out["reason"] and out["size_mb"] < 1024
    assert not fake.calls

    size, files = upload_size(ws.output_dir)
    reconfigure(ws, huggingface={"min_size_mb": size / 1024**2 - 0.01})  # threshold just under the real size
    out = maybe_upload(ws, uploader=fake)
    assert out["uploaded"] is True and out["repo_url"].endswith("someone/my-dataset") and out["files"] == files
    call = fake.calls[0]
    assert call["repo_id"] == "someone/my-dataset" and call["private"] is False and call["folder"] == str(ws.output_dir)
    assert "rejected/quarantine/**" in call["ignore"] and "images/**" in call["mirrored"]
    assert "token" not in str(call).lower()
    again = maybe_upload(ws, uploader=fake)  # same export: not sent twice
    assert again["uploaded"] is False and "already uploaded" in again["reason"] and len(fake.calls) == 1


def test_force_skips_only_the_size_check_and_failures_are_reported(ws):
    run(ws)
    reconfigure(ws, huggingface={"enabled": True, "repo_id": "someone/my-dataset", "private": True})
    fake = Fake()
    out = maybe_upload(ws, force=True, uploader=fake)
    assert out["uploaded"] and fake.calls[0]["private"] is True
    with ws.db.session() as s:
        from app.database.session import kv_get, kv_set
        exp = kv_get(s, "export:last")
        kv_set(s, "export:last", {**exp, "validation_passed": False, "generated_at": "later"})
    assert "did not pass validation" in maybe_upload(ws, force=True, uploader=fake)["reason"]
    with ws.db.session() as s:
        kv_set(s, "export:last", {**exp, "generated_at": "later-still"})
    with pytest.raises(RuntimeError, match="not logged in"):
        maybe_upload(ws, force=True, uploader=Fake(ok=False))


def test_quarantined_renditions_are_never_uploaded(ws):
    run(ws)
    before, _ = upload_size(ws.output_dir)
    q = ws.output_dir / "rejected" / "quarantine"
    q.mkdir(parents=True, exist_ok=True)
    (q / "img_0123456789abcdef.png").write_bytes(b"x" * 500_000)
    (ws.output_dir / ".DS_Store").write_bytes(b"x" * 1000)
    assert upload_size(ws.output_dir)[0] == before
    assert (ws.output_dir / "README.md").read_text().startswith("---\npretty_name:")


def test_pipeline_uploads_automatically_once_enabled(ws, monkeypatch):
    fake = Fake()
    monkeypatch.setattr(hf_upload, "_run_child", fake)
    reconfigure(ws, huggingface={"enabled": True, "repo_id": "someone/my-dataset", "min_size_mb": 1024})
    s = run(ws)
    assert s["status"] == "completed" and s["stages"]["hf_upload"]["status"] == "completed" and not fake.calls  # too small
    reconfigure(ws, huggingface={"min_size_mb": 0.5})
    s = run(ws)
    assert s["status"] == "completed" and len(fake.calls) == 1
    for st in ("validate", "quality_technical", "captioning", "image_output"):
        assert s["stages"][st]["done"] == 0  # changing upload settings re-ran nothing


def test_dotenv_sets_repo_and_enables_upload_without_leaking_the_token(tmp_path, fixture_dir, monkeypatch):
    import os

    from app.core.env import ENV_FILE_NAME, parse_env_file
    from app.tests.conftest import make_ws

    monkeypatch.delenv("IMAGE_PIPELINE_SKIP_DOTENV")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("app.core.env.PROJECT_ROOT", tmp_path)  # never the developer's real file
    root = tmp_path / "ws"
    root.mkdir()
    env_file = root / ENV_FILE_NAME
    env_file.write_text('# comment\nHF_TOKEN="hf_secretvalue123"\nHF_DATASET_REPO=someone/env-dataset\n'
                        "HF_PRIVATE=true\nHF_MIN_SIZE_MB=0.5   # small for the test\nEMPTY=\n")
    assert parse_env_file(env_file) == {"HF_TOKEN": "hf_secretvalue123", "HF_DATASET_REPO": "someone/env-dataset",
                                        "HF_PRIVATE": "true", "HF_MIN_SIZE_MB": "0.5"}
    ws = make_ws(root, fixture_dir)
    try:
        hf = ws.config.huggingface
        assert (hf.enabled, hf.repo_id, hf.private, hf.min_size_mb) == (True, "someone/env-dataset", True, 0.5)
        assert os.environ["HF_TOKEN"] == "hf_secretvalue123"  # available to the Hugging Face client only
        fake = Fake()
        monkeypatch.setattr(hf_upload, "_run_child", fake)
        s = run(ws)
        assert s["status"] == "completed" and fake.calls[0]["repo_id"] == "someone/env-dataset"
        leaked = [p for p in root.rglob("*") if p.is_file() and p != env_file and b"hf_secretvalue123" in p.read_bytes()]
        assert leaked == []  # not in the config, database, logs, reports or the dataset
        monkeypatch.setenv("HF_UPLOAD_ENABLED", "false")
        assert ws.reload_config().huggingface.enabled is False
    finally:
        for k in ("HF_TOKEN", "HF_DATASET_REPO", "HF_PRIVATE", "HF_MIN_SIZE_MB"):
            os.environ.pop(k, None)
        ws.db.dispose()
