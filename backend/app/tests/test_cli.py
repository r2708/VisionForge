"""The three everyday commands: run, resume, status."""
import json

from typer.testing import CliRunner

from app.cli import app
from app.tests.conftest import make_ws

runner = CliRunner()


def test_run_status_resume_commands(tmp_path, fixture_dir):
    root = tmp_path / "ws"
    make_ws(root, fixture_dir).db.dispose()
    w = ["--workspace", str(root)]

    first = runner.invoke(app, ["run", "--workers", "2", *w])
    assert first.exit_code == 0, first.output
    out = first.output
    assert "=== VALIDATE" in out and "[VALIDATE] img_" in out and "✓" in out          # one line per image and stage
    assert "INVALID (truncated)" in out                                                 # broken files are called out
    assert "[validate 25/25] failed=0" in out and "elapsed=" in out and "eta=" in out   # running totals
    assert "[OCR] - skipped" in out and "Run summary" in out and "images in the dataset" in out
    assert '[CAPTION] img_' in out and "[OUTPUT] img_" in out

    again = runner.invoke(app, ["run", "--quiet", *w])                                  # nothing new: everything cached
    assert again.exit_code == 0 and "[VALIDATE] ✓ 25 images: 0 processed, 25 cached, 0 failed" in again.output
    assert "[VALIDATE] img_" not in again.output

    data = json.loads(runner.invoke(app, ["status", "--json", "--limit", "100", "--workspace", str(root)]).output)
    assert data["job"]["status"] == "completed" and data["matching"] == 25 and len(data["images"]) == 25
    done = [r for r in data["images"] if r["status"] == "done"]
    assert done and all(v in ("DONE", "SKIPPED") for v in done[0]["stages"].values())
    bad = next(r for r in data["images"] if r["file"] == "corrupt.jpg")
    assert bad["status"] == "invalid" and bad["stages"]["validate"] == "DONE" and bad["stages"]["phash"] == "SKIPPED"

    table = runner.invoke(app, ["status", "--workspace", str(root)]).output
    assert "status=completed" in table and "image_id" in table and "✓" in table and "images by decision" in table
    one = runner.invoke(app, ["status", done[0]["image_id"], "--workspace", str(root)]).output
    assert "metrics / error" in one and "VALIDATE" in one and "IMAGE_OUTPUT" in one and "images/img_" in one
    assert "No image with id" in runner.invoke(app, ["status", "img_missing", "--workspace", str(root)]).output

    nothing = runner.invoke(app, ["resume", *w])
    assert nothing.exit_code == 0 and "Nothing to resume" in nothing.output


def test_resume_continues_an_interrupted_job(tmp_path, fixture_dir):
    from app.core.workspace import Workspace
    from app.pipeline.runner import create_job, request_cancel, run_job

    root = tmp_path / "ws"
    ws = make_ws(root, fixture_dir)
    job_id = create_job(ws)
    run_job(ws, job_id, on_stage=lambda name, state: request_cancel(ws, job_id) if (name, state) == ("quality_technical", "running") else None)
    ws.db.dispose()

    st = runner.invoke(app, ["status", "--workspace", str(root)]).output
    assert "status=cancelled" in st and "image-pipeline resume" in st
    res = runner.invoke(app, ["resume", "--workers", "2", "-w", str(root)])
    assert res.exit_code == 0, res.output
    assert f"resuming job {job_id} (was cancelled, stopped at " in res.output
    assert "[VALIDATE] ✓ 25 images: 0 processed, 25 cached, 0 failed" in res.output      # finished stages are not repeated
    assert "[CAPTION] img_" in res.output and "[EXPORT] ✓" in res.output                 # the remaining stages run
    check = Workspace(root)
    assert json.loads((check.output_dir / "reports/export_validation.json").read_text())["passed"]
    check.db.dispose()


def test_run_needs_a_folder_the_first_time(tmp_path):
    res = runner.invoke(app, ["run", "--workspace", str(tmp_path / "empty")])
    assert res.exit_code == 2 and "image-pipeline run /path/to/images" in res.output
