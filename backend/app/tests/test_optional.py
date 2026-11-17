"""Optional stages and integrations; each test skips cleanly when its dependency is absent."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms
from sqlalchemy import select

from app.database.models import Image as Img
from app.database.models import StageResult
from app.pipeline.vision.stages import ocr_available, spatial_relationships
from app.tests.conftest import make_ws, run
from app.tests.fixtures import landscape


def _data(ws, rel, stage):
    with ws.db.session() as s:
        im = s.scalar(select(Img).where(Img.relative_path == rel))
        r = s.scalar(select(StageResult).where(StageResult.image_pk == im.id, StageResult.stage == stage))
        return None if r is None else r.data


@pytest.mark.skipif(not ocr_available(), reason="rapidocr-onnxruntime is not installed")
def test_ocr_reads_visible_text_with_regions(tmp_path, fixture_dir):
    ws = make_ws(tmp_path / "ws", fixture_dir, stages={"ocr": True})
    s = run(ws)
    assert s["stages"]["ocr"]["status"] == "completed" and s["stages"]["ocr"]["failed"] == 0
    d = _data(ws, "text_image.png", "ocr")
    joined = " ".join(d["text"]).upper()
    assert "HELLO" in joined and "WORLD" in joined and d["text_area_fraction"] > 0.05
    assert all(len(r["bbox"]) == 4 and 0 <= r["confidence"] <= 1 for r in d["regions"])
    assert _data(ws, "scene_d.jpg", "ocr")["text"] == []
    recs = {r["source"]["relative_path"]: r for r in map(json.loads, (ws.output_dir / "manifests/all_records.jsonl").read_text().splitlines())}
    rec = recs["text_image.png"]
    assert rec["visual_annotations"]["ocr"]["text"] and "visible text" in rec["visual_annotations"]["tags"]
    assert "Visible text reads" in rec["captions"]["comprehensive"]
    ws.db.dispose()


def test_ocr_stage_is_skipped_by_default(done):
    _, summary, _ = done
    assert summary["stages"]["ocr"]["status"] == "skipped" and "optional" in summary["stages"]["ocr"]["reason"]
    assert summary["stages"]["segmentation"]["status"] == "skipped"
    assert summary["stages"]["object_detection"]["status"] == "skipped"  # expensive models never run unless enabled


def test_segmentation_masks_are_exported_for_accepted_images(tmp_path, fixture_dir):
    ws = make_ws(tmp_path / "ws", fixture_dir, stages={"segmentation": True})
    s = run(ws)
    assert s["stages"]["segmentation"]["failed"] == 0
    recs = [json.loads(x) for x in (ws.output_dir / "manifests/all_records.jsonl").read_text().splitlines()]
    with_mask = [r for r in recs if r["visual_annotations"]["segmentation"]]
    assert with_mask
    for r in with_mask:
        seg = r["visual_annotations"]["segmentation"]
        p = ws.output_dir / seg["mask_path"]
        with Image.open(p) as m:
            arr = np.asarray(m)
        assert m.mode == "L" and set(np.unique(arr)) <= {0, 255} and 0 < seg["foreground_fraction"] < 1
    exported = {r["image_id"] for r in with_mask}
    assert {p.stem for p in (ws.output_dir / "annotations/masks").glob("*.png")} == exported
    ws.db.dispose()


def test_relationships_are_derived_from_boxes_only():
    objs = [{"label": "person", "confidence": 0.9, "bbox": [0.1, 0.2, 0.2, 0.6]},
            {"label": "car", "confidence": 0.8, "bbox": [0.6, 0.3, 0.3, 0.4]},
            {"label": "hat", "confidence": 0.7, "bbox": [0.15, 0.2, 0.08, 0.1]}]
    rels = {(r["subject"], r["relation"], r["object"]) for r in spatial_relationships(objs)}
    assert ("person", "left of", "car") in rels and ("hat", "inside or in front of", "person") in rels
    assert all(r["status"] == "inferred" and r["basis"] == "bounding_box_geometry" for r in spatial_relationships(objs))


_PROFILES = [Path("/System/Library/ColorSync/Profiles/Display P3.icc"),
             Path("/System/Library/ColorSync/Profiles/AdobeRGB1998.icc"),
             Path("/usr/share/color/icc/colord/AdobeRGB1998.icc")]
_profile = next((p for p in _PROFILES if p.is_file()), None)


@pytest.mark.skipif(_profile is None, reason="no wide-gamut ICC profile available on this system")
def test_non_srgb_profiles_are_converted_and_srgb_is_left_alone(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    img = landscape(61, (800, 640))
    img.save(src / "wide_gamut.jpg", quality=95, icc_profile=_profile.read_bytes())
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    landscape(62, (800, 640)).save(src / "tagged_srgb.jpg", quality=95, icc_profile=srgb)  # a different picture
    ws = make_ws(tmp_path / "ws", src, paths={"sources": [{"path": str(src), "label": "icc", "rights_status": "user_attested"}]})
    run(ws)
    recs = {r["source"]["relative_path"]: r for r in map(json.loads, (ws.output_dir / "manifests/all_records.jsonl").read_text().splitlines())}
    wide, plain = recs["wide_gamut.jpg"], recs["tagged_srgb.jpg"]
    assert _data(ws, "wide_gamut.jpg", "validate")["icc"] == {"present": True, "description": _data(ws, "wide_gamut.jpg", "validate")["icc"]["description"], "is_srgb": False}
    assert "icc_converted_to_srgb" in wide["processed_image"]["transformations"]
    assert plain["processed_image"]["transformations"] == ["copied_unchanged"]
    with Image.open(ws.output_dir / wide["image_path"]) as out, Image.open(src / "wide_gamut.jpg") as original:
        assert out.size == original.size
        diff = np.abs(np.asarray(out, dtype=int) - np.asarray(original.convert("RGB"), dtype=int)).mean()
        assert 0.3 < diff < 40  # pixel values really were remapped, without wrecking the image
    ws.db.dispose()


def test_hugging_face_imagefolder_can_load_the_export(done):
    datasets = pytest.importorskip("datasets")
    ws, _, _ = done
    ds = datasets.load_dataset("imagefolder", data_dir=str(ws.images_dir), split="train",
                               cache_dir=str(ws.cache_dir / "hf_test"))
    recs = [json.loads(x) for x in (ws.output_dir / "manifests/all_records.jsonl").read_text().splitlines()]
    assert len(ds) == len(recs) and {"image", "text", "split"} <= set(ds.column_names)
    row = ds[0]
    assert row["image"].size[0] > 0 and row["text"] in {r["captions"]["detailed"] for r in recs}
    assert set(ds["split"]) <= {"train", "validation", "test"}
