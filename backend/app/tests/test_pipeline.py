"""End-to-end behaviour: storage, exports, splits, resume, smart reprocessing, review, rights."""
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import select

from app.core.io import sha256_file
from app.database.models import DecisionLog, Image as Img, Job
from app.pipeline.curation.splits import validate_splits
from app.pipeline.base import StageContext
from app.pipeline.runner import EXPORT_ONLY, create_job, request_cancel, run_job
from app.schemas.record import TrainingRecord
from app.services import review as rv
from app.tests.conftest import make_ws, reconfigure, run, tree_checksums


def records(ws, name="all_records"):
    p = ws.output_dir / "manifests" / f"{name}.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def by_source(ws):
    return {r["source"]["relative_path"]: r for r in records(ws)}


def image_id(ws, rel):
    with ws.db.session() as s:
        return s.scalar(select(Img.image_id).where(Img.relative_path == rel))


def stage(summary, name):
    return summary["stages"][name]


# ---------------------------------------------------------------- storage
def test_job_completes_and_sources_are_untouched(done, fixture_dir):
    ws, summary, before = done
    assert summary["status"] == "completed", summary
    assert all(st["failed"] == 0 for st in summary["stages"].values())
    assert tree_checksums(fixture_dir) == before
    assert not any(p.suffix == ".tmp" or p.name.startswith(".img_") for p in fixture_dir.rglob("*"))


def test_processed_images_are_written_decodable_and_checksummed(done):
    ws, _, _ = done
    recs = records(ws)
    assert len(recs) >= 8
    on_disk = {p.name for p in ws.images_dir.iterdir() if p.suffix in (".jpg", ".png", ".webp")}
    assert on_disk == {Path(r["image_path"]).name for r in recs}  # no orphans, no dangling paths
    for r in recs:
        p = ws.output_dir / r["image_path"]
        assert p.is_file() and not Path(r["image_path"]).is_absolute()
        with Image.open(p) as im:
            im.load()
            assert (im.width, im.height) == (r["processed_image"]["width"], r["processed_image"]["height"])
            assert im.format == r["processed_image"]["format"]
        assert sha256_file(p) == r["processed_image"]["sha256"]
        assert r["processed_image"]["is_upscaled"] is False
        TrainingRecord.model_validate(r)  # schema


def test_renditions_follow_the_documented_policies(done, fixture_dir):
    ws, _, _ = done
    recs = by_source(ws)

    rot = recs["rotated.jpg"]["processed_image"]
    assert (rot["width"], rot["height"]) == (600, 800) and "exif_orientation_applied" in rot["transformations"]
    with Image.open(ws.output_dir / recs["rotated.jpg"]["image_path"]) as im:
        assert im.getexif().get(0x0112, 1) == 1  # orientation is baked in, not left as a tag

    same = recs["scene_b.png"]  # nothing to change: byte-identical copy, no recompression
    assert same["processed_image"]["transformations"] == ["copied_unchanged"]
    assert same["processed_image"]["sha256"] == same["source"]["sha256"]

    gps = recs["with_gps.jpg"]
    assert gps["processed_image"]["transformations"] == ["exif_removed_losslessly"]
    with Image.open(ws.output_dir / gps["image_path"]) as out, Image.open(fixture_dir / "with_gps.jpg") as src:
        assert not out.getexif().get_ifd(0x8825) and np.array_equal(np.asarray(out), np.asarray(src))

    with Image.open(ws.output_dir / recs["cmyk.jpg"]["image_path"]) as im:
        assert im.mode == "RGB"
    with Image.open(ws.output_dir / recs["alpha.png"]["image_path"]) as im:
        assert im.mode == "RGBA" and im.format == "PNG"  # used transparency is preserved
    assert recs["tex.bmp"]["processed_image"]["format"] == "PNG"
    assert recs["nested/shapes.tif"]["processed_image"]["format"] == "PNG"
    for r in recs.values():  # aspect ratio preserved, never upscaled
        with Image.open(fixture_dir / r["source"]["relative_path"]) as src:
            w, h = src.size
        pw, ph = r["processed_image"]["width"], r["processed_image"]["height"]
        assert {pw, ph} == {w, h}


def test_alpha_can_be_flattened_by_policy(tmp_path, fixture_dir):
    ws = make_ws(tmp_path / "ws", fixture_dir, images={"preserve_alpha": False, "alpha_background": [0, 255, 0]})
    run(ws)
    r = by_source(ws)["alpha.png"]
    assert "alpha_flattened_on_background" in r["processed_image"]["transformations"]
    with Image.open(ws.output_dir / r["image_path"]) as im:
        assert im.mode == "RGB" and im.getpixel((2, 2)) == (0, 255, 0)
    ws.db.dispose()


def test_max_side_downscales_without_changing_aspect(tmp_path, fixture_dir):
    ws = make_ws(tmp_path / "ws", fixture_dir, images={"max_side": 640}, quality={"min_width": 256, "min_height": 256})
    run(ws)
    for r in records(ws):
        p = r["processed_image"]
        assert max(p["width"], p["height"]) <= 640
    d = by_source(ws)["scene_d.jpg"]["processed_image"]
    assert (d["width"], d["height"]) == (640, 480) and "downscaled_to_max_side_640" in d["transformations"]
    ws.db.dispose()


# ---------------------------------------------------------------- export
def test_export_files_reports_and_validation(done):
    ws, _, _ = done
    out = ws.output_dir
    for rel in ("manifests/all_records.jsonl", "manifests/train.jsonl", "manifests/validation.jsonl", "manifests/test.jsonl",
                "manifests/all_records.csv", "manifests/all_records.parquet", "manifests/minimal_train.jsonl",
                "images/metadata.jsonl", "metadata/image_metadata.jsonl", "metadata/captions.jsonl",
                "metadata/quality_scores.jsonl", "metadata/annotations.jsonl", "metadata/provenance.jsonl",
                "rejected/rejected_records.jsonl", "reports/dataset_summary.json", "reports/duplicate_report.json",
                "reports/quality_report.json", "reports/alignment_report.json", "reports/rights_report.json",
                "reports/split_leakage_report.json", "reports/coverage_report.json", "reports/export_validation.json",
                "dataset_card.md", "SHA256SUMS"):
        assert (out / rel).is_file(), rel
    val = json.loads((out / "reports/export_validation.json").read_text())
    assert val["passed"] is True and set(val["checks"].values()) == {0}
    recs = records(ws)
    assert val["records_valid"] == len(recs)
    with open(out / "manifests/all_records.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["image_id"] for r in rows] == [r["image_id"] for r in recs]
    import pyarrow.parquet as pq

    table = pq.read_table(out / "manifests/all_records.parquet")
    assert table.num_rows == len(recs) and json.loads(table.column("record_json")[0].as_py())["image_id"] == recs[0]["image_id"]
    minimal = [json.loads(x) for x in (out / "manifests/minimal_train.jsonl").read_text().splitlines()]
    assert all(set(m) == {"image", "text"} and m["text"] and (out / m["image"]).is_file() for m in minimal)
    # Package checksum manifest covers the files it lists.
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        assert sha256_file(out / rel) == digest
    card = (out / "dataset_card.md").read_text()
    for needle in ("Sources and licences", "Transformations", "Model-generated content", "Limitations", "Intended use"):
        assert needle in card


def test_every_caption_stays_paired_with_its_image(done):
    ws, _, _ = done
    with ws.db.session() as s:
        ids = {im.image_id: im.relative_path for im in s.scalars(select(Img))}
    caps = {json.loads(x)["image_id"] for x in (ws.output_dir / "metadata/captions.jsonl").read_text().splitlines()}
    for r in records(ws):
        assert ids[r["image_id"]] == r["source"]["relative_path"]
        assert Path(r["image_path"]).stem == r["image_id"] and r["image_id"] in caps
        assert r["captions"]["detailed"].strip()
    hf = [json.loads(x) for x in (ws.images_dir / "metadata.jsonl").read_text().splitlines()]
    want = {Path(r["image_path"]).name: r["captions"]["detailed"] for r in records(ws)}
    assert {h["file_name"]: h["text"] for h in hf} == want


def test_exports_contain_no_private_paths(done, fixture_dir):
    ws, _, _ = done
    needles = [str(fixture_dir), str(ws.root), str(Path.home())]
    for p in ws.output_dir.rglob("*"):
        if p.is_file() and p.suffix in (".jsonl", ".json", ".csv", ".md", ".txt"):
            text = p.read_text(errors="ignore")
            assert not any(n in text for n in needles), p
    log = (ws.logs_dir / "pipeline.log").read_text()
    assert str(Path.home()) not in log


def test_rejected_records_are_listed_with_reasons(done, fixture_info):
    ws, _, _ = done
    rej = {r["relative_path"]: r for r in map(json.loads, (ws.output_dir / "rejected/rejected_records.jsonl").read_text().splitlines())}
    exported = set(by_source(ws))
    with ws.db.session() as s:
        everything = {im.relative_path for im in s.scalars(select(Img))}
    assert set(rej) | exported == everything and not set(rej) & exported
    assert rej["corrupt.jpg"]["reasons"] == ["invalid_file:unreadable"]
    assert rej["small.jpg"]["reasons"] == ["below_min_resolution"]
    # Low scores are rejected outright, or quarantined for review when they fall inside the review band.
    assert rej["blurry.jpg"]["decision"] in ("rejected", "quarantined")
    assert {"low_quality_score", "quality_below_threshold_pending_review"} & set(rej["blurry.jpg"]["reasons"])
    assert any(x.startswith("rights_status_not_allowed:restricted") for x in rej["scene_c.webp"]["reasons"])
    assert all(r["reasons"] and r["recoverable"] for r in rej.values())


# ---------------------------------------------------------------- rights
def test_strict_rights_policy_blocks_unapproved_statuses(done):
    ws, _, _ = done
    recs = by_source(ws)
    assert "scene_c.webp" not in recs  # marked restricted in the user's metadata file
    assert {r["source"]["rights_status"] for r in recs.values()} <= {"verified", "user_attested"}
    assert recs["scene_d.jpg"]["source"]["rights_status"] == "verified" and recs["scene_d.jpg"]["source"]["license"] == "CC0-1.0"
    assert recs["scene_d.jpg"]["source"]["source_url"] == "https://example.org/scene_d"
    rights = json.loads((ws.output_dir / "reports/rights_report.json").read_text())
    assert rights["all_images_by_rights_status"]["restricted"] == 1 and rights["images_without_allowed_rights"] == 1


def test_unknown_rights_are_never_assumed_usable(tmp_path, fixture_dir):
    ws = make_ws(tmp_path / "ws", fixture_dir, paths={"sources": [str(fixture_dir)]})  # no rights information at all
    summary = run(ws)
    assert summary["status"] == "completed" and records(ws) == []
    assert not any(ws.images_dir.glob("img_*"))
    s = json.loads((ws.output_dir / "reports/dataset_summary.json").read_text())
    assert any("rights status" in w for w in s["warnings"])
    # The user attests the rights -> the same cached results now export without re-analysis.
    with ws.db.session() as sess:
        from app.database.models import Source
        sid = sess.scalar(select(Source.source_id))
    rv.set_source_rights(ws, sid, rights_status="user_attested", license="own work")
    again = run(ws, EXPORT_ONLY)
    assert len(records(ws)) >= 8 and stage(again, "image_output")["done"] == len(records(ws))
    off = make_ws(tmp_path / "ws2", fixture_dir, paths={"sources": [str(fixture_dir)]},
                  rights={"require_approved_status_for_export": False})
    run(off)
    assert len(records(off)) >= 8 and {r["source"]["rights_status"] for r in records(off)} == {"unknown"}
    ws.db.dispose(), off.db.dispose()


# ---------------------------------------------------------------- splits
def test_splits_are_deterministic_for_a_seed(tmp_path, fixture_dir, done):
    ws, _, _ = done
    first = {r["image_id"]: r["curation"]["split"] for r in records(ws)}
    other = make_ws(tmp_path / "ws", fixture_dir)
    run(other)
    assert {r["image_id"]: r["curation"]["split"] for r in records(other)} == first
    assert all(r["curation"]["split_seed"] == 42 for r in records(other))
    reconfigure(other, splits={"seed": 7, "train": 0.6, "validation": 0.2, "test": 0.2})
    s = run(other, EXPORT_ONLY)
    changed = {r["image_id"]: r["curation"]["split"] for r in records(other)}
    assert changed != first and set(changed) == set(first)
    counts = json.loads((other.output_dir / "reports/split_leakage_report.json").read_text())["assignment"]["counts"]
    assert counts["validation"] >= 2 and counts["test"] >= 2 and sum(counts.values()) == len(first)
    assert stage(s, "image_output")["done"] == 0  # split changes never rewrite images
    other.db.dispose()


def test_duplicate_groups_never_cross_splits(tmp_path, fixture_dir):
    # Keep duplicates in the dataset (flag instead of quarantine) so grouping is what protects the splits.
    ws = make_ws(tmp_path / "ws", fixture_dir, deduplication={"non_representative_action": "flag"},
                 splits={"train": 0.5, "validation": 0.25, "test": 0.25, "seed": 3})
    run(ws)
    recs = by_source(ws)
    family = ["scene_a.jpg", "nested/scene_a_copy.jpg", "nested/scene_a_resized.jpg", "nested/deeper/scene_a_cropped.png"]
    assert all(f in recs for f in family)
    assert len({recs[f]["curation"]["split"] for f in family}) == 1
    report = json.loads((ws.output_dir / "reports/split_leakage_report.json").read_text())
    assert report["leakage_detected"] is False and report["assignment"]["multi_image_groups"] >= 1

    # The independent validator catches leakage if it is introduced.
    with ws.db.session() as s:
        im = s.scalar(select(Img).where(Img.relative_path == "nested/scene_a_copy.jpg"))
        im.split = "test" if im.split != "test" else "train"
    leak = validate_splits(StageContext(ws=ws, cfg=ws.config))
    assert leak["leakage_detected"] and leak["leak_count"] >= 1
    summary = run(ws, ["export"])  # export refuses to ship leaking records
    val = json.loads((ws.output_dir / "reports/export_validation.json").read_text())
    assert summary["status"] == "completed" and val["passed"] is False and val["checks"]["6_no_split_leakage"] >= 1
    assert not set(family) & set(by_source(ws))
    ws.db.dispose()


# ---------------------------------------------------------------- resume / idempotency
def test_second_run_reuses_everything(ws):
    first = run(ws)
    out_before = tree_checksums(ws.images_dir)
    second = run(ws)
    assert second["status"] == "completed"
    for name, st in second["stages"].items():
        assert st["done"] == 0, name
    assert stage(second, "validate")["reused"] == stage(first, "validate")["done"] == 25
    assert tree_checksums(ws.images_dir) == out_before


def test_cancelled_job_resumes_without_repeating_completed_work(ws):
    job_id = create_job(ws)

    def cancel_after_validate(name, state):
        if name == "phash" and state == "running":
            request_cancel(ws, job_id)

    first = run_job(ws, job_id, on_stage=cancel_after_validate)
    assert first["status"] == "cancelled"
    assert stage(first, "validate")["done"] == 25 and stage(first, "captioning")["status"] == "pending"
    assert not (ws.output_dir / "manifests").exists()

    resumed = run_job(ws, job_id)
    assert resumed["status"] == "completed"
    assert stage(resumed, "validate")["done"] == 0 and stage(resumed, "validate")["reused"] == 25
    assert stage(resumed, "captioning")["done"] > 0
    assert json.loads((ws.output_dir / "reports/export_validation.json").read_text())["passed"]


def test_crashed_job_is_marked_interrupted_and_resumable(ws):
    from datetime import timedelta

    from app.database.models import utcnow
    from app.pipeline.runner import mark_interrupted

    job_id = create_job(ws)
    with ws.db.session() as s:  # simulate a process that died mid-run
        j = s.get(Job, job_id)
        j.status, j.heartbeat_at = "running", utcnow() - timedelta(minutes=10)
    assert mark_interrupted(ws) == 1
    with ws.db.session() as s:
        assert s.get(Job, job_id).status == "interrupted"
    assert run_job(ws, job_id)["status"] == "completed"


def test_only_one_job_at_a_time(ws):
    import subprocess
    import sys

    from app.core.runlock import RunLock
    from app.pipeline.runner import AlreadyRunning

    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])  # stands in for a live pipeline process
    try:
        lock = RunLock(ws.checkpoints_dir)
        lock.path.parent.mkdir(parents=True, exist_ok=True)
        lock.path.write_text(str(other.pid))
        with ws.db.session() as s:  # a job that the other process is running
            from app.pipeline.runner import new_job_id
            jid = new_job_id()
            s.add(Job(id=jid, status="running", mode="text_to_image", config=ws.config.model_dump(mode="json"),
                      config_hash=ws.config.config_hash(), stages=[]))
        with pytest.raises(AlreadyRunning):
            create_job(ws)
        with pytest.raises(AlreadyRunning):
            run_job(ws, jid)
        from app.pipeline.runner import mark_interrupted
        assert mark_interrupted(ws) == 0  # a status check never disturbs a live run
    finally:
        other.kill()
        other.wait()
    # the owner is gone: the lock is stale, the job is recovered as interrupted and can be resumed
    from app.pipeline.runner import mark_interrupted
    assert mark_interrupted(ws) == 1
    with ws.db.session() as s:
        assert s.get(Job, jid).status == "interrupted"
    create_job(ws)


# ---------------------------------------------------------------- smart reprocessing
def test_caption_change_does_not_rerun_unrelated_stages(ws):
    run(ws)
    reconfigure(ws, captioning={"adapter": "sidecar"})
    s = run(ws)
    for name in ("validate", "phash", "quality_technical", "people", "quality"):
        assert stage(s, name)["done"] == 0 and stage(s, name)["reused"] > 0, name
    assert stage(s, "captioning")["done"] > 0
    recs = by_source(ws)
    assert recs["scene_b.png"]["captions"]["detailed"] == "Overlapping colored circles and rectangles on a pale background."
    assert recs["scene_b.png"]["captions"]["source"] == "user_supplied"
    assert recs["scene_d.jpg"]["captions"]["detailed"].startswith("A generated mountain landscape")  # from the metadata file
    # Images without a user caption are held back, not exported with an invented one.
    assert "scene_g.jpg" not in recs and "tex.bmp" not in recs
    rej = [json.loads(x) for x in (ws.output_dir / "rejected/rejected_records.jsonl").read_text().splitlines()]
    assert any(r["relative_path"] == "tex.bmp" and "missing_caption" in r["reasons"] for r in rej)


def test_threshold_change_reuses_cached_metrics(ws):
    run(ws)
    n_before = len(records(ws))
    reconfigure(ws, quality={"min_overall_score": 99, "manual_review_band": [98, 99]})
    s = run(ws)
    for name in ("validate", "phash", "quality_technical", "people", "quality", "captioning"):
        assert stage(s, name)["done"] == 0, name
    assert len(records(ws)) < n_before
    assert not any(ws.images_dir.glob("img_*")) or len(list(ws.images_dir.glob("img_*"))) == len(records(ws))
    # Renditions of newly rejected images leave the dataset folder but are kept in quarantine.
    assert len(list((ws.output_dir / "rejected" / "quarantine").glob("img_*"))) >= n_before - len(records(ws))

    reconfigure(ws, quality={"weights": {"sharpness": 1.0}})  # weights only re-run the cheap aggregation
    s = run(ws)
    assert stage(s, "quality_technical")["done"] == 0 and stage(s, "quality")["done"] == 21


def test_export_settings_change_rebuilds_only_manifests(ws):
    run(ws)
    reconfigure(ws, export={"formats": ["parquet"], "minimal_manifest": "caption_txt"})
    s = run(ws)
    assert all(st["done"] == 0 for st in s["stages"].values())
    recs = records(ws)  # jsonl from the previous export is still there
    txt = sorted(p.name for p in ws.images_dir.glob("*.txt"))
    assert len(txt) == len(recs) and not (ws.images_dir / "metadata.jsonl").exists()
    r = recs[0]
    assert (ws.output_dir / r["image_path"]).with_suffix(".txt").read_text().strip() == r["captions"]["detailed"]
    assert (ws.output_dir / "manifests/all_records.parquet").is_file()


# ---------------------------------------------------------------- review
def test_review_decisions_are_reversible_and_logged(ws, fixture_dir):
    run(ws)
    before = tree_checksums(fixture_dir)
    keep = image_id(ws, "scene_d.jpg")
    assert keep in {r["image_id"] for r in records(ws)}

    out = rv.decide(ws, keep, "reject", note="test")
    assert out["decision"] == "rejected" and "untouched" in out["consequence"]
    run(ws, EXPORT_ONLY)
    assert keep not in {r["image_id"] for r in records(ws)}
    assert not list(ws.images_dir.glob(f"{keep}.*")) and list((ws.output_dir / "rejected/quarantine").glob(f"{keep}.*"))

    rv.decide(ws, keep, "restore")
    run(ws, EXPORT_ONLY)
    assert keep in {r["image_id"] for r in records(ws)} and list(ws.images_dir.glob(f"{keep}.*"))

    low = image_id(ws, "blurry.jpg")  # automatically rejected; a reviewer may still approve it
    rv.decide(ws, low, "approve")
    run(ws, EXPORT_ONLY)
    rec = {r["image_id"]: r for r in records(ws)}[low]
    assert rec["curation"]["decision"] == "accepted" and rec["curation"]["review_status"] == "approved"
    assert "manual_override:accepted" in rec["curation"]["rejection_reasons"]
    with pytest.raises(ValueError):
        rv.decide(ws, image_id(ws, "corrupt.jpg"), "approve")  # an undecodable file can never be approved
    with ws.db.session() as s:
        actions = [d.action for d in s.scalars(select(DecisionLog).where(DecisionLog.actor == "user"))]
    assert actions == ["review_reject", "review_restore", "review_approve"]
    assert tree_checksums(fixture_dir) == before


def test_caption_edits_are_exported_and_marked(ws):
    run(ws)
    target = image_id(ws, "scene_d.jpg")
    rv.edit(ws, target, captions={"detailed": "  Hand   written caption. "}, tags=["Mountains", "mountains", " dusk "])
    run(ws, EXPORT_ONLY)
    rec = {r["image_id"]: r for r in records(ws)}[target]
    assert rec["captions"]["detailed"] == "Hand written caption." and rec["captions"]["edited_by_user"]
    assert rec["captions"]["source"] == "user_edited" and rec["visual_annotations"]["tags"] == ["mountains", "dusk"]
    assert rec["alignment"]["verification_status"] == "manually_reviewed" and rec["curation"]["review_status"] == "edited"
    rv.edit(ws, target, captions={"detailed": None})  # revert to the generated caption
    run(ws, EXPORT_ONLY)
    assert {r["image_id"]: r for r in records(ws)}[target]["captions"]["detailed"] != "Hand written caption."
    with pytest.raises(ValueError):
        rv.edit(ws, target, captions={"nonsense": "x"})


def test_duplicate_override_keeps_a_member(ws):
    run(ws)
    dup = image_id(ws, "nested/scene_a_resized.jpg")
    assert dup not in {r["image_id"] for r in records(ws)}
    rv.set_duplicate_override(ws, dup, "keep")
    run(ws)
    recs = {r["image_id"]: r for r in records(ws)}
    assert dup in recs and recs[dup]["deduplication"]["representative_status"] == "kept_by_user"
    rep = image_id(ws, "scene_a.jpg")
    assert recs[dup]["curation"]["split"] == recs[rep]["curation"]["split"] if rep in recs else True


# ---------------------------------------------------------------- damaged outputs are caught
def test_export_validation_catches_damaged_and_missing_outputs(ws):
    run(ws)
    recs = records(ws)
    broken, gone, tampered = recs[0], recs[1], recs[2]
    (ws.output_dir / broken["image_path"]).write_bytes(b"not an image")
    (ws.output_dir / gone["image_path"]).unlink()
    p = ws.output_dir / tampered["image_path"]
    with Image.open(p) as im:
        im2 = im.convert("RGB").transpose(Image.FLIP_LEFT_RIGHT)
    im2.save(p)
    summary = run(ws, ["export"])  # validation only: no repair stages
    val = json.loads((ws.output_dir / "reports/export_validation.json").read_text())
    assert summary["status"] == "completed" and val["passed"] is False
    assert val["checks"]["2_images_decode"] == 1 and val["checks"]["1_paths_exist"] == 1 and val["checks"]["7_checksums_match"] == 1
    bad = {broken["image_id"], gone["image_id"], tampered["image_id"]}
    assert {x["image_id"] for x in val["problems"]} == bad and not bad & {r["image_id"] for r in records(ws)}
    # A normal export repairs the processed layer from the untouched sources.
    run(ws, EXPORT_ONLY)
    assert json.loads((ws.output_dir / "reports/export_validation.json").read_text())["passed"] is True
    assert bad <= {r["image_id"] for r in records(ws)}


def test_failed_images_are_isolated_and_retryable(tmp_path, fixture_dir, monkeypatch):
    from app.pipeline.quality import technical

    src = tmp_path / "src"
    shutil.copytree(fixture_dir, src)
    ws = make_ws(tmp_path / "ws", src)
    real = technical.blockiness

    def flaky(path):
        if path.endswith("scene_g.jpg"):
            raise RuntimeError("simulated failure")
        return real(path)

    monkeypatch.setattr(technical, "blockiness", flaky)
    s = run(ws)
    assert s["status"] == "completed" and stage(s, "quality_technical")["failed"] == 1  # one bad image does not stop the job
    rej = {r["relative_path"]: r for r in map(json.loads, (ws.output_dir / "rejected/rejected_records.jsonl").read_text().splitlines())}
    assert "stage_failed:quality_technical" in rej["scene_g.jpg"]["reasons"]
    with ws.db.session() as sess:
        from app.database.models import StageResult
        row = sess.scalar(select(StageResult).where(StageResult.status == "failed"))
        assert row.attempts == 3 and row.error_category == "internal" and "simulated failure" in row.error
    monkeypatch.setattr(technical, "blockiness", real)
    again = run(ws)  # attempts exhausted: not retried automatically
    assert stage(again, "quality_technical")["failed"] == 1 and stage(again, "quality_technical")["done"] == 0
    fixed = run(ws, retry_failed=True)
    assert stage(fixed, "quality_technical") == {**stage(fixed, "quality_technical"), "done": 1, "failed": 0}
    ws.db.dispose()


def test_disk_space_preflight_blocks_writes(tmp_path, fixture_dir):
    ws = make_ws(tmp_path / "ws", fixture_dir, storage={"min_free_disk_gb": 10_000_000})
    s = run(ws)
    assert s["status"] == "failed" and "disk space" in s["error"]
    assert not any(ws.images_dir.glob("img_*")) if ws.images_dir.exists() else True
    ws.db.dispose()


def test_modes_select_manifests(tmp_path, fixture_dir):
    ws = make_ws(tmp_path / "ws", fixture_dir, project={"mode": "vision_language"}, export={"vlm_qa": True})
    run(ws)
    m = ws.output_dir / "manifests"
    assert (m / "vlm_train.jsonl").is_file() and not (m / "minimal_train.jsonl").exists()
    row = json.loads((m / "vlm_train.jsonl").read_text().splitlines()[0])
    assert {"image", "short_caption", "detailed_caption", "comprehensive_caption", "attributes", "objects", "qa"} <= set(row)
    assert all(q["basis"] for q in row["qa"])
    ws.db.dispose()


def test_cleanup_only_touches_files_the_pipeline_created(ws):
    run(ws)
    mine = ws.images_dir / "my_notes.txt"
    photo = ws.images_dir / "holiday.jpg"
    hidden = ws.images_dir / ".keep"
    stray = ws.images_dir / "img_0123456789abcdef.png"   # looks like an old rendition nobody references
    mine.write_text("keep me")
    hidden.write_text("")
    Image.new("RGB", (8, 8)).save(photo)
    Image.new("RGB", (8, 8)).save(stray)
    s = run(ws, EXPORT_ONLY)
    assert mine.read_text() == "keep me" and photo.is_file() and hidden.is_file()
    assert not stray.exists() and (ws.output_dir / "rejected/quarantine/orphaned" / stray.name).is_file()  # moved, not deleted
    assert s["stages"]["reconcile_outputs"]["status"] == "completed"
    assert json.loads((ws.output_dir / "reports/export_validation.json").read_text())["passed"]
