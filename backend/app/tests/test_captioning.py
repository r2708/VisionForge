import json

from sqlalchemy import select

from app.database.models import Image, StageResult
from app.pipeline.base import ImgRef, StageContext
from app.pipeline.captioning.adapters import TemplateAdapter, build_tags, clean_caption, grounded_facts
from app.pipeline.captioning.stage import run_caption_diversity
from app.tests.conftest import run


def test_clean_caption_removes_boilerplate_and_unfinished_text():
    assert clean_caption("The image shows a red car parked on a street.") == "A red car parked on a street."
    assert clean_caption("In this image, we can see two dogs running.") == "Two dogs running."
    assert clean_caption("This is a photo of a lake at dawn") == "A lake at dawn."
    long = "A wide valley with a river running through green fields below. The mountains in the dista"
    assert clean_caption(long) == "A wide valley with a river running through green fields below."
    assert clean_caption("   ") == ""


def _ref(deps):
    return ImgRef(pk=1, image_id="img_x", abs_path="/nonexistent", sha256="ab" * 32, extension=".jpg", file_name="x.jpg",
                  relative_path="x.jpg", width=800, height=600, supplied_caption=None, deps=deps)


def _zs(category, conf, style="photorealistic"):
    def one(label, c):
        return {"labels": [{"label": label, "confidence": c}], "top": [], "uncertain": c < 0.35}
    return {"categories": one(category, conf), "style": one(style, 0.9), "lighting": one("golden_hour", 0.8),
            "scene": one("outdoor_natural", 0.9), "viewpoint": one("wide_shot", 0.7), "depth_of_field": one("deep", 0.6)}


def test_template_captions_use_only_grounded_facts():
    from app.config.settings import CaptioningCfg

    deps = {"validate": {"orientation": "landscape"},
            "quality_technical": {"color": {"dominant_color_names": ["blue", "orange"], "is_grayscale": False},
                                  "composition": {"subject_placement": "centered"}},
            "zero_shot": _zs("landscape", 0.8),
            "ocr": {"text": ["OPEN"]}, "people": {"face_count": 2}}
    ref = _ref(deps)
    facts = grounded_facts(ref)
    out = TemplateAdapter().generate(None, ref, facts, CaptioningCfg(alternate_caption_count=2))
    text = out["comprehensive"]
    assert "landscape" in out["short"] and "blue and orange" in text and '"OPEN"' in text
    assert "face" not in text  # faces are only mentioned for people/portrait categories
    for banned in ("beautiful", "stunning", "8k", "professional", "cinematic", "masterpiece"):
        assert banned not in (text + out["generation_prompt"]).lower()
    assert out["prompt_styling_additions"] == [] and 1 <= len(out["alternatives"]) <= 2
    assert "golden hour" in build_tags(facts) and "visible text" in build_tags(facts)

    unsure = _ref({**deps, "zero_shot": _zs("wildlife", 0.2)})  # low confidence: the subject is not asserted
    low = TemplateAdapter().generate(None, unsure, grounded_facts(unsure), CaptioningCfg())
    assert "animal" not in low["comprehensive"] and low["short"].startswith("An image in")


def test_caption_records_carry_versions_and_warnings(done):
    ws, _, _ = done
    with ws.db.session() as s:
        rows = list(s.scalars(select(StageResult).where(StageResult.stage == "captioning")))
    assert rows
    for r in rows:
        assert r.data["template_version"] == "template-1" and r.data["generated_at"] and r.model_info["name"] == "template"
        assert r.data["source"] == "template_baseline"
    with ws.db.session() as s:
        pending = list(s.scalars(select(Image).where(Image.decision == "accepted")))
    # Baseline captions are never silently trusted: every one is routed to review.
    assert pending and all(im.review_status == "pending" and "baseline_template_caption" in im.review_reasons for im in pending)


def test_caption_diversity_flags_repetition(ws):
    run(ws)
    with ws.db.session() as s:
        for r in s.scalars(select(StageResult).where(StageResult.stage == "captioning")):
            r.data = {**r.data, "detailed": "A photo of a thing."}
    out = run_caption_diversity(StageContext(ws=ws, cfg=ws.config))
    with ws.db.session() as s:
        from app.database.session import kv_get
        rep = kv_get(s, "report:caption_diversity")
    assert rep["unique_caption_ratio"] < 0.2 and rep["top_openings"][0]["share"] == 1.0 and out["warnings"] >= 2
    assert rep["repeated_captions"][0]["count"] == rep["captions_analyzed"] and rep["diversity_score"] < 30


def test_benchmark_measures_and_leaves_human_metrics_blank(ws, tmp_path):
    import csv

    from app.services.benchmark import run_benchmark, summarize_sheet

    run(ws)
    out = run_benchmark(ws, ["template", "no-such-model"], sample=6)
    rep = out["report"]["adapters"]
    t = rep["template"]
    assert t["images"] == 6 and t["failures"] == 0 and t["latency_s"]["mean"] >= 0 and t["mean_words"] > 3
    assert t["human_rated"]["factual_accuracy"] is None and "not measured" in t["human_rated"]["status"]  # never invented
    assert rep["no-such-model"]["skipped"] and rep["no-such-model"]["reason"]
    with open(out["review_sheet"], newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6 and all(r["factual_accuracy_1to5"] == "" for r in rows)
    assert summarize_sheet(out["review_sheet"])["adapters"]["template"] == {
        "captions": 6, "rated": 0, "unrated": 6, "mean_factual_accuracy_1to5": None, "hallucination_rate": None,
        "mean_hallucinated_claims": None, "mean_salient_details_missing": None}
    for i, r in enumerate(rows[:4]):  # a reviewer rates four of the six
        r["factual_accuracy_1to5"], r["hallucinated_claims_count"], r["salient_details_missing_count"] = "4", str(i % 2), "1"
    filled = tmp_path / "filled.csv"
    with open(filled, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = summarize_sheet(filled)["adapters"]["template"]
    assert (s["rated"], s["unrated"], s["mean_factual_accuracy_1to5"], s["hallucination_rate"]) == (4, 2, 4.0, 0.5)
