"""Checks for the dataset-quality rules: typed export schema, captions that come with the source,
text-based watermark detection and duplicates of earlier shards."""
import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pydantic import ValidationError

from app.config.settings import CaptioningCfg, PipelineConfig
from app.pipeline.base import ImgRef
from app.pipeline.captioning.adapters import AutoCaptionAdapter, clean_supplied_caption, summary_of
from app.pipeline.deduplication.dedup import cross_hamming
from app.pipeline.export.exporter import _write_parquet, parquet_schema
from app.pipeline.quality.aggregate import _looks_like_watermark
from app.schemas.record import VisualAnnotations


def test_cross_hamming_finds_only_hashes_within_the_threshold():
    old = np.array([0x0F0F0F0F0F0F0F0F, 0xFFFF0000FFFF0000, 0x123456789ABCDEF0], dtype=np.uint64)
    new = np.array([0x0F0F0F0F0F0F0F0E,   # 1 bit from old[0]
                    0xFFFF0000FFFF00FF,   # 8 bits from old[1]: too far
                    0x123456789ABCDEF0],  # identical to old[2]
                   dtype=np.uint64)
    assert cross_hamming(new, old, 6) == {0: (0, 1), 2: (2, 0)}
    assert cross_hamming(new, np.array([], dtype=np.uint64), 6) == {}


@pytest.mark.parametrize("text, bbox, expected", [
    ("shutterstock", [0.3, 0.4, 0.4, 0.1], True),              # agency name anywhere
    ("© Jane Doe 2019", [0.4, 0.4, 0.2, 0.05], True),          # copyright mark anywhere
    ("www.example.com", [0.70, 0.95, 0.25, 0.03], True),       # small credit line at the bottom edge
    ("www.example.com", [0.30, 0.45, 0.40, 0.04], False),      # an address in the middle of a page is content
    ("Talk Like a Pirate Day", [0.2, 0.02, 0.6, 0.08], False),  # ordinary text
])
def test_watermark_text_patterns(text, bbox, expected):
    assert _looks_like_watermark({"text": text, "confidence": 0.9, "bbox": bbox}) is expected
    assert _looks_like_watermark({"text": text, "confidence": 0.3, "bbox": bbox}) is False  # unreadable text proves nothing


def test_an_address_in_a_text_heavy_image_is_not_a_watermark():
    line = {"text": "www.example.com", "confidence": 0.9, "bbox": [0.70, 0.95, 0.25, 0.03]}
    assert _looks_like_watermark(line, sparse_text=True) and not _looks_like_watermark(line, sparse_text=False)
    assert _looks_like_watermark({"text": "Adobe Stock | #574446882", "confidence": 0.9, "bbox": [0, 0.3, 0.05, 0.4]}, sparse_text=False)
    assert _looks_like_watermark({"text": "Photo by Jane Doe", "confidence": 0.9, "bbox": [0.4, 0.5, 0.2, 0.05]}, sparse_text=False)


def test_summary_keeps_whole_opening_sentences():
    text = "A red car is parked on a street. " + "It has four doors and a dented bumper. " * 30
    out = summary_of(text, max_words=20)
    assert out.startswith("A red car is parked on a street.") and out.endswith(".") and len(out.split()) <= 20
    assert summary_of("One long sentence " * 40 + "ends here.", max_words=10).endswith("ends here.")  # never cut mid-sentence


class _FakeHub:
    def __init__(self):
        self.loads, self.asked = 0, 0

    def get(self, key):
        self.loads += 1
        hub = self
        return SimpleNamespace(ask=lambda image, prompt, max_new_tokens, split_image=None: hub._ask())

    def _ask(self):
        self.asked += 1
        return "A dog runs across a field. The grass is green."


def _ref(tmp_path, caption):
    from PIL import Image

    path = tmp_path / "x.png"
    Image.new("RGB", (64, 64), "white").save(path)
    return ImgRef(pk=1, image_id="img_x", abs_path=str(path), sha256="ab" * 32, extension=".png", file_name="x.png",
                  relative_path="x.png", width=64, height=64, supplied_caption=caption)


def test_auto_captions_use_the_model_only_when_the_source_has_no_caption(tmp_path):
    hub, messages = _FakeHub(), []
    ctx = SimpleNamespace(cfg=PipelineConfig(), hub=hub, progress=lambda name, info: messages.append(info.get("message")))
    cfg = CaptioningCfg(adapter="auto", supplied_source="pixmo-cap")
    facts = {"lighting": None, "style": None}
    adapter = AutoCaptionAdapter(ctx)

    supplied = adapter.generate(ctx, _ref(tmp_path, "A tall lighthouse stands on a cliff. Waves break below it."), facts, cfg)
    assert supplied["source"] == "pixmo-cap" and supplied["short"] == "A tall lighthouse stands on a cliff."
    assert supplied["comprehensive"].endswith("Waves break below it.") and supplied["caption_model"] is None
    assert hub.loads == 0 and not messages  # the model was not even loaded

    generated = adapter.generate(ctx, _ref(tmp_path, None), facts, cfg)
    assert generated["source"] == "vlm" and generated["caption_model"] == "smolvlm2-256m"
    assert generated["short"] == "A dog runs across a field." and hub.loads == 1 and hub.asked == 1
    adapter.generate(ctx, _ref(tmp_path, None), facts, cfg)
    assert hub.loads == 1 and hub.asked == 2  # loaded once, one question per image
    assert any("loading smolvlm2-256m" in m for m in messages)
    assert adapter.summary().startswith("1 caption(s) taken from the source, 2 generated by smolvlm2-256m")


def _record(image_id, objects, ocr_text, creator):
    return {"image_id": image_id, "image_path": f"images/{image_id}.jpg",
            "captions": {"short": "s", "detailed": "d", "comprehensive": "c", "generation_prompt": "p", "source": "vlm"},
            "visual_annotations": {"tags": [], "categories": ["food"], "style": [], "lighting": [], "scene": None,
                                   "watermarks_detected": None, "people": {}, "ocr": {"text": ocr_text, "regions": []},
                                   "objects": objects},
            "processed_image": {"width": 640, "height": 480, "format": "JPEG", "sha256": "0" * 64},
            "quality": {"overall_score": 80.0, "aesthetic_score": None, "flags": []},
            "alignment": {"score": 0.5, "verification_status": "passed", "warnings": []},
            "curation": {"split": "train", "review_status": "not_required"},
            "source": {"source_id": "src", "relative_path": "a.jpg", "sha256": "1" * 64, "license": "x", "rights_status": "user_attested",
                       "creator": creator, "attribution": None, "source_url": None}}


def test_parquet_manifests_have_one_schema_whatever_the_values(tmp_path):
    """An empty list or an all-null column must not change a column type from file to file."""
    empty = _record("img_a", [], [], None)
    full = _record("img_b", [{"label": "cup", "confidence": 0.9, "bbox": [0, 0, 1, 1]}], ["OPEN"], "someone")
    _write_parquet(tmp_path / "a.parquet", [empty], "detailed")
    _write_parquet(tmp_path / "b.parquet", [full], "detailed")
    _write_parquet(tmp_path / "none.parquet", [], "detailed")
    a, b, none = (pq.read_table(tmp_path / f"{n}.parquet") for n in ("a", "b", "none"))
    assert a.schema.equals(parquet_schema()) and b.schema.equals(a.schema) and none.schema.equals(a.schema)
    assert pa.types.is_string(a.schema.field("object_labels").type.value_type) and pa.types.is_string(a.schema.field("creator").type)
    assert b.column("object_labels")[0].as_py() == ["cup"] and b.column("text")[0].as_py() == "d"
    assert json.loads(a.column("record_json")[0].as_py())["image_id"] == "img_a"


def test_nested_annotations_are_type_checked():
    ok = VisualAnnotations(objects=[{"label": "cup", "confidence": 0.8, "bbox": [0.1, 0.1, 0.2, 0.2]}],
                           ocr={"text": ["OPEN"], "regions": [{"text": "OPEN", "confidence": 0.9, "bbox": [0, 0, 0.5, 0.1]}]},
                           color_palette=[{"hex": "#ff0000", "name": "red", "fraction": 0.5}], people={"face_count": 0})
    assert ok.objects[0].label == "cup"
    for bad in ({"objects": [{"label": "cup", "confidence": 1.4, "bbox": [0, 0, 1, 1]}]},     # confidence out of range
                {"objects": [{"label": "cup", "confidence": 0.5, "bbox": [0, 0, 1]}]},        # box needs four numbers
                {"objects": [{"label": "cup", "confidence": 0.5}]},                           # box missing
                {"color_palette": [{"hex": "red", "name": "red", "fraction": 0.5}]},          # not a hex colour
                {"ocr": {"text": "OPEN", "regions": []}},                                     # text must be a list
                {"people": {"face_count": -1}}):
        with pytest.raises(ValidationError):
            VisualAnnotations(**bad)


def test_an_image_already_exported_in_an_earlier_shard_is_kept_out(tmp_path, fixture_dir):
    from sqlalchemy import select

    from app.database.models import Image, SeenHash, StageResult
    from app.tests.conftest import make_ws, run

    ws = make_ws(tmp_path / "ws", fixture_dir)
    run(ws)
    with ws.db.session() as s:
        img = s.scalars(select(Image).where(Image.decision == "accepted", Image.representative_status == "unique")).first()
        hashes = s.scalar(select(StageResult.data).where(StageResult.image_pk == img.id, StageResult.stage == "phash"))
        image_id, sha, others = img.image_id, img.sha256, s.query(Image).filter(Image.decision == "accepted").count()
        # one earlier image with the same bytes, one that is only perceptually the same (pHash one bit off)
        near = f"{int(hashes['phash'], 16) ^ 1:016x}"
        s.add_all([SeenHash(sha256=sha, phash=hashes["phash"], dhash=hashes["dhash"], image_id="img_old_exact", shard="shard_0001"),
                   SeenHash(sha256="f" * 64, phash=near, dhash=hashes["dhash"], image_id="img_old_near", shard="shard_0002")])
    out = run(ws)
    assert out["status"] == "completed"
    with ws.db.session() as s:
        img = s.scalar(select(Image).where(Image.image_id == image_id))
        assert img.representative_status == "duplicate" and img.decision == "quarantined"
        assert img.decision_reasons == ["duplicate_of:shard_0001/img_old_exact"]
        assert s.query(Image).filter(Image.decision == "accepted").count() == others - 1  # nothing else was touched
    ws.db.dispose()


@pytest.mark.parametrize("raw, cleaned", [
    ("**Detailed Caption:**\n\nThis image shows a white page.", "This image shows a white page."),
    ("Here is a cleaned-up and detailed caption for the image:\n\n---\n\nA screenshot of a forum post.", "A screenshot of a forum post."),
    ("### Detailed Caption for the Image\n\nA dog on a beach.", "A dog on a beach."),
    ("**Description:**\n\nA two-panel scene.\n\n**Top Panel:**\nJim stands.", "A two-panel scene. Top Panel: Jim stands."),
    ("Here are two photographs of a lake.", "Here are two photographs of a lake."),          # a real description, not a lead-in
    ("Image of a red car on a road.", "Image of a red car on a road."),
    ("A detailed drawing of a caption contest: two dogs.", "A detailed drawing of a caption contest: two dogs."),
])
def test_wrapping_around_a_source_caption_is_removed(raw, cleaned):
    assert clean_supplied_caption(raw) == cleaned
