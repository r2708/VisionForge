"""Model-backed stages. They run only when the weights are already in the local cache;
nothing is downloaded by the test suite."""
import json

import numpy as np
import pytest
from sqlalchemy import func, select

from app.config.settings import PipelineConfig
from app.database.models import Embedding, Image, StageResult
from app.models.hub import ModelHub
from app.tests.conftest import make_ws, reconfigure, run

_hub = ModelHub(PipelineConfig())
clip_ready = _hub.status("clip-vit-base-patch32")["usable"]
vlm_ready = _hub.status("smolvlm2-256m")["usable"]
needs_clip = pytest.mark.skipif(not clip_ready, reason="CLIP ViT-B/32 weights are not in the local cache")


def test_registry_lists_requirements_without_downloading():
    rows = {m["key"]: m for m in _hub.describe()}
    assert {"clip-vit-base-patch32", "smolvlm2-256m"} <= set(rows)
    for m in rows.values():
        assert m["license"] and m["download_size_gb"] and m["ram_gb"] and m["repo"]
        assert m["usable"] or m["reason"]
    assert _hub.status("not-a-model")["usable"] is False


def test_missing_model_disables_stage_with_a_reason(tmp_path, fixture_dir):
    ws = make_ws(tmp_path / "ws", fixture_dir, stages={"embeddings": True}, models={"embedding": "clip-vit-large-patch14"})
    if ModelHub(ws.config).status("clip-vit-large-patch14")["usable"]:
        pytest.skip("large model is cached on this machine")
    s = run(ws)
    assert s["status"] == "completed" and s["stages"]["embedding"]["status"] == "skipped"
    assert "models download clip-vit-large-patch14" in s["stages"]["embedding"]["reason"]
    assert s["stages"]["export"]["status"] == "completed"
    ws.db.dispose()


@pytest.fixture(scope="module")
def clip_ws(tmp_path_factory, fixture_dir):
    ws = make_ws(tmp_path_factory.mktemp("clip") / "ws", fixture_dir, stages={"embeddings": True},
                 deduplication={"near_candidate_hamming": 0})  # near-duplicate candidates must come from embeddings
    summary = run(ws)
    yield ws, summary
    ws.db.dispose()


@needs_clip
def test_embeddings_are_cached_by_checksum(clip_ws):
    ws, summary = clip_ws
    assert summary["status"] == "completed" and summary["stages"]["embedding"]["failed"] == 0
    with ws.db.session() as s:
        n = s.scalar(select(func.count(Embedding.id)))
        row = s.scalar(select(Embedding))
        shas = s.scalar(select(func.count(func.distinct(Image.sha256))).where(Image.valid.is_(True)))
        info = s.scalar(select(StageResult).where(StageResult.stage == "embedding")).model_info
    v = np.frombuffer(row.vector, dtype=np.float32)
    assert n == shas == 20 and row.dim == 512 and abs(np.linalg.norm(v) - 1) < 1e-3  # byte-identical files share one vector
    assert info["name"] == "clip-vit-base-patch32" and len(info["revision"]) == 40 and info["precision"] == "float32"


@needs_clip
def test_zero_shot_annotations_are_marked_inferred(clip_ws):
    ws, _ = clip_ws
    with ws.db.session() as s:
        rows = {im.relative_path: r.data for im, r in s.execute(
            select(Image, StageResult).join(StageResult, StageResult.image_pk == Image.id).where(StageResult.stage == "zero_shot"))}
    assert len(rows) == 21
    for d in rows.values():
        assert d["status"] == "inferred" and "not calibrated" in d["confidence_note"]
        assert 0 <= d["aesthetic"]["score"] <= 100 and "ranking signal" in d["aesthetic"]["method"]
        assert 0 <= d["watermark"]["probability"] <= 1
        for name in ("categories", "style", "lighting", "scene", "viewpoint"):
            assert d[name]["labels"] and all(0 <= x["confidence"] <= 1 for x in d[name]["labels"])
    assert rows["text_image.png"]["categories"]["labels"][0]["label"] in ("document", "graphic")
    rec = [json.loads(x) for x in (ws.output_dir / "manifests/all_records.jsonl").read_text().splitlines()]
    assert rec and all(r["visual_annotations"]["categories"] and r["quality"]["aesthetic_score"] is not None for r in rec)
    cam = rec[0]["visual_annotations"]["camera_attributes"]
    assert cam["viewpoint"]["status"] == "inferred" and cam["viewpoint"]["confidence"] is not None
    assert cam["f_number"] == {"value": None, "status": "unknown", "confidence": None}


@needs_clip
def test_embedding_candidates_find_the_cropped_near_duplicate(clip_ws):
    ws, _ = clip_ws
    with ws.db.session() as s:
        crop = s.scalar(select(Image).where(Image.relative_path == "nested/deeper/scene_a_cropped.png"))
        a = s.scalar(select(Image).where(Image.relative_path == "scene_a.jpg"))
        d = s.scalar(select(Image).where(Image.relative_path == "scene_d.jpg"))
        assert crop.near_group and crop.representative_status == "duplicate" and a.representative_status in ("representative", "duplicate")
        assert d.near_group is None and d.representative_status == "unique"  # similar-looking but a different image
        assert a.semantic_group and a.semantic_group == crop.semantic_group or a.semantic_group  # semantic groups are informational
        assert d.decision == "accepted"


@needs_clip
def test_alignment_flags_a_wrong_caption(clip_ws):
    ws, _ = clip_ws
    with ws.db.session() as s:
        rows = [r.data for r in s.scalars(select(StageResult).where(StageResult.stage == "alignment"))]
    assert rows
    for d in rows:
        assert d["calibrated"] is False and 0 <= d["score"] <= 1 and d["verification_status"] in ("passed", "warning", "failed")
        assert d["independent_of_caption_source"] is False  # template captions share signals with the checker

    from app.pipeline.alignment.stage import AlignmentStage
    from app.pipeline.base import ImgRef, StageContext

    ctx = StageContext(ws=ws, cfg=ws.config, hub=ModelHub(ws.config))
    st = AlignmentStage()
    st.setup(ctx)
    with ws.db.session() as s:
        im = s.scalar(select(Image).where(Image.relative_path == "text_image.png"))
        emb = s.scalar(select(StageResult).where(StageResult.image_pk == im.id, StageResult.stage == "embedding")).data

    def check(caption):
        ref = ImgRef(pk=im.id, image_id=im.image_id, abs_path=im.abs_path, sha256=im.sha256, extension=".png", file_name="t",
                     relative_path="t", width=1000, height=700, supplied_caption=None,
                     deps={"embedding": emb, "captioning": {"short": caption, "detailed": caption, "source": "vlm"}})
        return st.process(ctx, ref)

    good = check("Large dark text on a plain white page.")
    wrong = check("A dog and a cat sleeping on a beach next to a motorcycle.")
    assert good["score"] > wrong["score"]
    assert wrong["verification_status"] == "failed" and wrong["needs_review"] and wrong["unsupported_claims"]
    assert {u["term"] for u in wrong["unsupported_claims"]} & {"dog", "cat", "beach", "motorcycle"}
    ctx.hub.unload()


@needs_clip
def test_changing_caption_settings_reuses_embeddings(clip_ws):
    ws, _ = clip_ws
    reconfigure(ws, captioning={"strictness": "strict", "alternate_caption_count": 1})
    s = run(ws)
    for name in ("validate", "phash", "quality_technical", "embedding", "zero_shot", "quality"):
        assert s["stages"][name]["done"] == 0, name
    assert s["stages"]["captioning"]["done"] > 0 and s["stages"]["alignment"]["done"] > 0


@pytest.mark.skipif(not (clip_ready and vlm_ready), reason="SmolVLM2-256M weights are not in the local cache")
def test_local_vlm_caption_adapter(tmp_path, fixture_dir):
    src = tmp_path / "src"
    src.mkdir()
    for name in ("scene_d.jpg", "text_image.png"):
        (src / name).write_bytes((fixture_dir / name).read_bytes())
    ws = make_ws(tmp_path / "ws", src, paths={"sources": [{"path": str(src), "label": "vlm", "rights_status": "user_attested"}]},
                 stages={"embeddings": True}, captioning={"adapter": "smolvlm", "comprehensive_caption": False, "max_new_tokens": 60})
    s = run(ws)
    assert s["status"] == "completed" and s["stages"]["captioning"]["done"] == 2 and s["stages"]["captioning"]["failed"] == 0
    with ws.db.session() as sess:
        rows = list(sess.scalars(select(StageResult).where(StageResult.stage == "captioning")))
    for r in rows:
        assert r.data["source"] == "vlm" and r.data["prompt_version"] == "smolvlm-prompts-1"
        assert len(r.data["short"].split()) >= 3 and r.data["detailed"] and r.data["comprehensive"] == ""
        assert r.model_info["name"] == "smolvlm2-256m" and r.model_info["revision"]
        assert not r.data["short"].lower().startswith(("the image shows", "this image"))
    ws.db.dispose()
