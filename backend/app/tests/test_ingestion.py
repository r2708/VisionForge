import shutil

import pytest
from sqlalchemy import func, select

from app.database.models import Image, Source, StageResult
from app.pipeline.ingestion.scanner import make_image_id, scan_all
from app.tests.conftest import make_ws, run, tree_checksums


def _images(ws):
    with ws.db.session() as s:
        return {im.relative_path: im for im in s.scalars(select(Image))}


def test_recursive_scan_registers_supported_files(ws, fixture_info):
    out = scan_all(ws)
    assert out["totals"]["new"] == fixture_info["supported_files"]
    assert out["totals"]["unsupported"] == fixture_info["unsupported"]
    imgs = _images(ws)
    assert "nested/deeper/scene_a_cropped.png" in imgs  # nested folders
    assert not any(p.startswith(".") or "/." in p for p in imgs)  # hidden files are skipped
    rep = out["sources"][0]
    assert rep["unsupported_by_extension"] == {".txt": 2, ".zip": 1, ".csv": 1}
    for im in imgs.values():
        assert len(im.sha256) == 64 and im.file_size >= 0 and im.image_id.startswith("img_")


def test_non_recursive_scan(tmp_path, fixture_dir):
    w = make_ws(tmp_path / "ws", fixture_dir, processing={"recursive_scan": False})
    scan_all(w)
    assert all("/" not in p for p in _images(w))
    w.db.dispose()


def test_ids_are_deterministic_and_unique(tmp_path, fixture_dir):
    a, b = make_ws(tmp_path / "a", fixture_dir), make_ws(tmp_path / "b", fixture_dir)
    scan_all(a), scan_all(b)
    ia, ib = _images(a), _images(b)
    assert {p: i.image_id for p, i in ia.items()} == {p: i.image_id for p, i in ib.items()}
    assert len({i.image_id for i in ia.values()}) == len(ia)
    # Byte-identical files in different folders share a checksum but get different IDs.
    x, y = ia["scene_a.jpg"], ia["nested/scene_a_copy.jpg"]
    assert x.sha256 == y.sha256 and x.image_id != y.image_id
    with a.db.session() as s:
        sid = s.scalar(select(Source.source_id))
    assert x.image_id == make_image_id(sid, "scene_a.jpg", x.sha256)
    a.db.dispose(), b.db.dispose()


def test_rescan_is_idempotent(ws, fixture_info):
    scan_all(ws)
    again = scan_all(ws)
    assert again["totals"]["new"] == 0 and again["totals"]["unchanged"] == fixture_info["supported_files"]
    with ws.db.session() as s:
        assert s.scalar(select(func.count(Image.id))) == fixture_info["supported_files"]


def test_incremental_add_change_and_removal(tmp_path, fixture_dir):
    src = tmp_path / "src"
    shutil.copytree(fixture_dir, src)
    w = make_ws(tmp_path / "ws", src)
    run(w, ["scan", "validate", "phash"])
    before = _images(w)

    shutil.copyfile(src / "scene_b.png", src / "added.png")                      # new file
    (src / "scene_d.jpg").write_bytes((src / "scene_g.jpg").read_bytes())        # content changed
    (src / "blurry.jpg").unlink()                                                # removed
    out = scan_all(w)["totals"]
    assert (out["new"], out["changed"], out["missing"]) == (1, 1, 1)
    after = _images(w)
    assert after["scene_d.jpg"].image_id != before["scene_d.jpg"].image_id       # treated as a new image
    assert after["scene_d.jpg"].valid is None                                    # dependent state invalidated
    with w.db.session() as s:
        assert s.scalar(select(func.count(StageResult.id)).where(StageResult.image_pk == after["scene_d.jpg"].id)) == 0
        assert s.scalar(select(func.count(StageResult.id)).where(StageResult.image_pk == after["scene_a.jpg"].id)) == 2
    assert after["blurry.jpg"].missing is True                                   # marked, never purged
    w.db.dispose()


def test_scan_never_modifies_sources(ws, fixture_dir):
    before = tree_checksums(fixture_dir)
    scan_all(ws)
    assert tree_checksums(fixture_dir) == before


def test_source_and_output_must_not_overlap(tmp_path, fixture_dir):
    w = make_ws(tmp_path / "ws", fixture_dir)
    with pytest.raises(ValueError):
        w.validate_source_dir(str(w.output_dir))          # source inside output
    (fixture_dir.parent / "inner_ws").mkdir(exist_ok=True)
    inside = make_ws(fixture_dir.parent / "inner_ws", fixture_dir)
    with pytest.raises(ValueError):
        inside.validate_source_dir(str(fixture_dir.parent))  # output would live inside the source
    with pytest.raises(ValueError):
        w.validate_source_dir(str(tmp_path / "does-not-exist"))
    w.db.dispose(), inside.db.dispose()


def test_metadata_file_applies_rights_and_captions(ws):
    scan_all(ws)
    imgs = _images(ws)
    assert imgs["scene_d.jpg"].rights_status == "verified" and imgs["scene_d.jpg"].license == "CC0-1.0"
    assert imgs["scene_d.jpg"].supplied_caption.startswith("A generated mountain landscape")
    assert imgs["scene_c.webp"].rights_status == "restricted"
    assert imgs["scene_a.jpg"].rights_status is None
    with ws.db.session() as s:
        assert s.scalar(select(Image).where(Image.relative_path == "scene_a.jpg")).effective_rights() == "user_attested"
