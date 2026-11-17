"""Validation/metadata, hashing, technical quality and deduplication primitives."""
import io

import numpy as np
import pytest
from PIL import Image
from sqlalchemy import select

from app.database.models import Image as Img
from app.database.models import StageResult
from app.pipeline.deduplication.dedup import (cosine_topk_pairs, hamming_pairs, hamming_pairs_bruteforce,
                                              verify_near_duplicate)
from app.pipeline.deduplication.hashing import hamming, phash
from app.pipeline.image_output.writer import strip_jpeg_exif
from app.pipeline.quality.aggregate import aspect_bucket
from app.pipeline.quality.technical import exposure, sharpness
from app.tests.fixtures import landscape


def _row(ws, rel):
    with ws.db.session() as s:
        im = s.scalar(select(Img).where(Img.relative_path == rel))
        data = {r.stage: r.data for r in im.results}
        s.expunge(im)
        return im, data


def test_invalid_files_are_reported_with_reasons(done, fixture_info):
    ws, _, _ = done
    with ws.db.session() as s:
        bad = {im.file_name: (im.validation_status, im.validation_reason) for im in s.scalars(select(Img).where(Img.valid.is_(False)))}
        n_valid = len(list(s.scalars(select(Img).where(Img.valid.is_(True)))))
    assert set(bad) == fixture_info["invalid_names"] and n_valid == fixture_info["valid"]
    assert bad["truncated.jpg"][0] == "truncated" and bad["truncated.png"][0] == "truncated"
    assert bad["corrupt.jpg"][0] == "unreadable" and bad["empty.png"][0] == "unreadable"
    assert all(reason for _, reason in bad.values())


def test_metadata_orientation_alpha_and_exif(done):
    ws, _, _ = done
    im, d = _row(ws, "rotated.jpg")
    v = d["validate"]
    assert (v["raw_width"], v["raw_height"]) == (800, 600) and (im.width, im.height) == (600, 800)
    assert v["exif_orientation"] == 6 and im.orientation == "portrait"
    assert v["exif"]["camera_make"] == "FixtureCam" and v["exif"]["camera_model"] == "Model X"

    _, d = _row(ws, "alpha.png")
    assert d["validate"]["has_alpha"] and d["validate"]["alpha_used"]
    _, d = _row(ws, "scene_b.png")
    assert not d["validate"]["alpha_used"]
    _, d = _row(ws, "cmyk.jpg")
    assert d["validate"]["color_mode"] == "CMYK"
    _, d = _row(ws, "nested/scene_e_portrait.jpg")
    assert d["validate"]["aspect_ratio"] == 0.75 and d["validate"]["megapixels"] == pytest.approx(0.786, abs=0.001)


def test_gps_presence_is_recorded_but_coordinates_are_not(done):
    ws, _, _ = done
    _, d = _row(ws, "with_gps.jpg")
    assert d["validate"]["exif_has_gps"] is True
    assert "40.0" not in str(d["validate"]["exif"]) and "gps" not in str(d["validate"]["exif"]).lower()


def test_phash_is_stable_under_resize_and_recompression():
    a = landscape(3)
    buf = io.BytesIO()
    a.resize((640, 480)).save(buf, "JPEG", quality=70)
    b = Image.open(io.BytesIO(buf.getvalue()))
    assert hamming(phash(a), phash(b)) <= 6
    assert hamming(phash(a), phash(landscape(99))) > 12


def test_banded_hamming_search_matches_brute_force():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2**63, 300, dtype=np.uint64)
    noisy = base[:100] ^ np.array([1 << int(b) for b in rng.integers(0, 60, 100)], dtype=np.uint64)  # distance 1
    hashes = np.concatenate([base, noisy])
    for t in (0, 3, 6, 10):
        fast = {(i, j) for i, j, _ in hamming_pairs(hashes, t)}
        slow = {(i, j) for i, j, _ in hamming_pairs_bruteforce(hashes, t)}
        assert fast == slow and (t == 0 or len(fast) >= 100)


def test_cosine_topk_pairs():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(50, 16)).astype(np.float32)
    v[1] = v[0] + 0.01 * rng.normal(size=16)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    pairs = cosine_topk_pairs(v, 0.95, 3)
    assert [(i, j) for i, j, _ in pairs] == [(0, 1)]


def test_near_duplicate_geometric_verification(fixture_dir):
    a, crop = str(fixture_dir / "scene_a.jpg"), str(fixture_dir / "nested/deeper/scene_a_cropped.png")
    hit = verify_near_duplicate(a, crop, 25, 0.25)
    assert hit and hit["inliers"] >= 100
    assert verify_near_duplicate(a, str(fixture_dir / "scene_d.jpg"), 25, 0.25) is None  # different scene


def test_duplicate_groups_and_representative(done):
    ws, _, _ = done
    rows = {rel: _row(ws, rel)[0] for rel in ("scene_a.jpg", "nested/scene_a_copy.jpg", "nested/scene_a_resized.jpg",
                                               "nested/deeper/scene_a_cropped.png", "scene_d.jpg")}
    a, copy, small, crop, other = rows.values()
    assert a.exact_group and a.exact_group == copy.exact_group and small.exact_group is None
    assert a.perceptual_group == copy.perceptual_group == small.perceptual_group and a.perceptual_group
    assert crop.near_group and crop.near_group in (a.near_group, copy.near_group, small.near_group)
    assert other.exact_group is None and other.perceptual_group is None and other.near_group is None
    statuses = sorted(r.representative_status for r in (a, copy, small, crop))
    assert statuses == ["duplicate", "duplicate", "duplicate", "representative"]
    # Full-resolution original wins over the resized and the cropped variants.
    assert "representative" in (a.representative_status, copy.representative_status)
    # Nothing was deleted: every member is still registered and its source file still exists.
    from pathlib import Path
    assert all(Path(r.abs_path).is_file() for r in rows.values())
    for r in (a, copy, small, crop):
        if r.representative_status == "duplicate":
            assert r.decision == "quarantined" and r.decision_reasons[0].startswith("duplicate_of:img_")


def test_quality_signals_separate_good_from_bad(done):
    ws, _, _ = done
    q = {rel: _row(ws, rel)[1] for rel in ("scene_d.jpg", "blurry.jpg", "dark.jpg", "small.jpg")}
    good, blurry, dark, small = (q[k] for k in ("scene_d.jpg", "blurry.jpg", "dark.jpg", "small.jpg"))
    assert good["quality_technical"]["sharpness"]["score"] > 70 > 30 > blurry["quality_technical"]["sharpness"]["score"]
    assert "blurry" in blurry["quality"]["flags"]
    assert "underexposed" in dark["quality"]["flags"]
    assert good["quality"]["overall_score"] > blurry["quality"]["overall_score"]
    assert good["quality"]["overall_score"] > dark["quality"]["overall_score"]
    assert small["quality"]["resolution_status"] == "below_minimum" and "below_min_resolution" in small["quality"]["flags"]
    # Explainable: component scores, the weights actually used and reasons are all stored.
    agg = blurry["quality"]
    assert set(agg["components"]) >= {"sharpness", "noise", "compression", "exposure", "contrast", "resolution", "composition"}
    assert abs(sum(agg["weights_used"].values()) - 1) < 1e-3 and any("sharpness" in r for r in agg["reasons"])
    assert agg["components"]["aesthetic"] is None and "aesthetic" in agg["missing_components"]  # model stage was off


def test_sharpness_and_exposure_units():
    img = np.asarray(landscape(5).convert("L"), dtype=np.float32)
    assert sharpness(img)["score"] > sharpness(np.full_like(img, 128))["score"] == 0
    assert "underexposed" in exposure(img * 0.05)["flags"] and not exposure(img)["flags"]


def test_aspect_buckets_never_require_upscaling():
    b = aspect_bucket(1920, 1080, [512, 768, 1024], 64)
    assert b["base"] == 1024 and b["width"] <= 1920 and b["height"] <= 1080 and b["width"] % 64 == 0
    assert abs(b["aspect_ratio"] - 16 / 9) < 0.15
    assert aspect_bucket(600, 600, [512, 768, 1024], 64)["base"] == 512
    assert aspect_bucket(300, 300, [512, 768, 1024], 64) is None


def test_lossless_exif_strip_keeps_pixels(fixture_dir):
    raw = (fixture_dir / "with_gps.jpg").read_bytes()
    out = strip_jpeg_exif(raw)
    a, b = Image.open(io.BytesIO(raw)), Image.open(io.BytesIO(out))
    assert a.getexif().get_ifd(0x8825) and not dict(b.getexif())
    assert np.array_equal(np.asarray(a), np.asarray(b)) and len(out) < len(raw)
