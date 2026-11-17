"""Processes 46-50: image-text alignment score, targeted grounding checks, hallucination
and missing-detail flags, and a review-routing confidence.

Embedding similarity is a ranking/flagging signal, never proof that a caption is correct.
Every value here is uncalibrated and exists to decide what a human should look at."""
from __future__ import annotations

import re

import numpy as np

from app.config.settings import fingerprint
from app.pipeline.base import ImageStage, ImgRef, StageContext
from app.pipeline.vision import labels as L
from app.pipeline.vision.stages import load_embedding

# Cosine range that typical CLIP image-caption pairs occupy; used only to rescale to 0-1.
COS_LO, COS_HI = 0.15, 0.33
STRICTNESS = {
    "lenient": {"unsupported_z": -1.25, "missing_z": 2.6, "min_score": 0.15, "fail_fraction": 0.5},
    "standard": {"unsupported_z": -0.75, "missing_z": 2.1, "min_score": 0.25, "fail_fraction": 0.34},
    "strict": {"unsupported_z": -0.25, "missing_z": 1.7, "min_score": 0.35, "fail_fraction": 0.0},
}
PERSON_TERMS = {"person", "man", "woman", "people", "girl", "boy", "child", "face", "portrait", "men", "women", "children"}


def _vocabulary() -> tuple[list[str], dict[str, str]]:
    term_cat: dict[str, str] = {}
    for cat, terms in L.CATEGORY_TERMS.items():
        for t in terms:
            term_cat.setdefault(t, cat)
    for t in L.DETECTION_LABELS:
        term_cat.setdefault(t, "object")
    return sorted(term_cat), term_cat


class AlignmentStage(ImageStage):
    name = "alignment"
    label = "VERIFY"

    def describe(self, d):
        if d.get("score") is None:
            return d["verification_status"]
        return f"match {d['score']:.2f}, {d['verification_status']}" + (f", {len(d['warnings'])} warning(s)" if d["warnings"] else "")

    version = "1"
    title = "Caption verification and image-text alignment"
    processes = "46, 47, 48, 49, 50"
    depends = ("captioning", "embedding")
    soft_depends = ("zero_shot", "object_detection", "ocr", "people")
    batched = True

    def enabled(self, ctx):
        if ctx.cfg.stages.alignment is False:
            return False, "disabled in configuration"
        if ctx.cfg.stages.embeddings is False:
            return False, "requires the embedding stage, which is disabled"
        if ctx.cfg.captioning.adapter == "none" or ctx.cfg.stages.captioning is False:
            return False, "captioning is disabled"
        st = ctx.hub.status(ctx.cfg.models.embedding)
        return st["usable"], st.get("reason", "") or "requires the embedding model"

    def fingerprint(self, cfg):
        c = cfg.captioning
        return [c.verify_captions, c.strictness, c.training_caption_field, fingerprint([L.CATEGORY_TERMS, L.DETECTION_LABELS])]

    def model_info(self, ctx):
        return ctx.hub.info(ctx.cfg.models.embedding)

    def batch_size(self, ctx):
        return 32

    def setup(self, ctx):
        self.clip = ctx.hub.get(ctx.cfg.models.embedding)
        self.terms, self.term_cat = _vocabulary()
        self.term_vecs = self.clip.encode_texts([f"a photo of {t}" for t in self.terms])
        self.patterns = {t: re.compile(rf"\b{re.escape(t)}\b", re.I) for t in self.terms}

    def process(self, ctx: StageContext, ref: ImgRef) -> dict:
        cap, cfg = ref.deps["captioning"], ctx.cfg.captioning
        v = load_embedding(ctx, ref.sha256, ref.deps["embedding"]["model_key"])
        if v is None:
            raise ValueError("embedding missing from cache")
        fields = {k: cap.get(k) for k in ("short", "detailed", "comprehensive") if cap.get(k)}
        if not fields:
            return {"score": None, "raw_cosine": None, "verification_status": "no_caption", "warnings": ["no caption to verify"],
                    "unsupported_claims": [], "possibly_missing": [], "confidence": 0.0, "needs_review": True,
                    "method": "image_text_embedding_and_targeted_checks", "calibrated": False}
        vecs = self.clip.encode_texts(list(fields.values()))
        cos = {k: float(vecs[i] @ v) for i, k in enumerate(fields)}
        truncated = [k for k, t in fields.items() if self.clip.token_count(t) > 77]
        main_field = cfg.training_caption_field if cfg.training_caption_field in cos else next(iter(cos))
        raw = cos[main_field]
        score = float(np.clip((raw - COS_LO) / (COS_HI - COS_LO), 0, 1))
        out = {"score": round(score, 4), "raw_cosine": round(raw, 4), "scored_field": main_field,
               "per_field_cosine": {k: round(c, 4) for k, c in cos.items()}, "truncated_fields": truncated,
               "method": "image_text_embedding_and_targeted_checks", "calibrated": False,
               "score_note": f"cosine rescaled from [{COS_LO}, {COS_HI}] to [0, 1]; a flagging signal only"}
        warnings = [f"{k} caption exceeds the 77-token text-encoder limit; only its beginning was scored" for k in truncated]
        if not cfg.verify_captions:
            return {**out, "verification_status": "not_verified", "warnings": warnings, "unsupported_claims": [],
                    "possibly_missing": [], "confidence": round(score, 3), "needs_review": False}

        th = STRICTNESS[cfg.strictness]
        text = " ".join(fields.values())
        sims = self.term_vecs @ v
        z = (sims - sims.mean()) / (sims.std() + 1e-8)
        claimed = [i for i, t in enumerate(self.terms) if self.patterns[t].search(text)]
        detected = {o["label"] for o in (ref.deps.get("object_detection") or {}).get("objects", []) if o["confidence"] >= 0.3}
        has_detector = "object_detection" in ref.deps
        unsupported = []
        for i in claimed:
            t = self.terms[i]
            weak = z[i] < th["unsupported_z"]
            contradicted = has_detector and t in L.DETECTION_LABELS and t not in detected and z[i] < 0.5
            if weak or contradicted:
                unsupported.append({"term": t, "z": round(float(z[i]), 2),
                                    "evidence": "not found by the object detector" if contradicted and not weak
                                    else "low image-text similarity relative to other concepts"})
        claimed_cats = {self.term_cat[self.terms[i]] for i in claimed}
        missing = []
        for i in np.argsort(-z)[:8]:
            t = self.terms[i]
            if z[i] >= th["missing_z"] and self.term_cat[t] not in claimed_cats and not self.patterns[t].search(text):
                if self.term_cat[t] not in {m["category"] for m in missing}:
                    missing.append({"term": t, "category": self.term_cat[t], "z": round(float(z[i]), 2)})
        missing = missing[:3]
        ocr_text = (ref.deps.get("ocr") or {}).get("text", [])
        if ocr_text and not re.search(r"\b(text|sign|word|letter|caption|logo|label|reads|written|title)\w*\b", text, re.I):
            warnings.append("visible text was detected but the caption does not mention it")
        faces = (ref.deps.get("people") or {}).get("face_count", 0)
        if faces and not any(re.search(rf"\b{p}\b", text, re.I) for p in PERSON_TERMS):
            warnings.append("a face was detected (inferred) but the caption mentions no person")
        if unsupported:
            warnings.append("possibly unsupported: " + ", ".join(u["term"] for u in unsupported))
        if missing:
            warnings.append("possibly missing: " + ", ".join(m["term"] for m in missing))
        if score < th["min_score"]:
            warnings.append(f"low image-text alignment score ({score:.2f})")

        frac = len(unsupported) / len(claimed) if claimed else 0.0
        failed = score < th["min_score"] or (unsupported and frac > th["fail_fraction"])
        status = "failed" if failed else ("warning" if (unsupported or missing or len(warnings) > len(truncated)) else "passed")
        confidence = float(np.clip(0.5 * score + 0.5 * (1 - frac) - 0.08 * len(missing), 0, 1))
        independent = cap.get("source") != "template_baseline"
        if not independent:
            warnings.append("caption was built from the same zero-shot signals used here, so this check is not independent")
        return {**out, "verification_status": status, "warnings": warnings, "unsupported_claims": unsupported,
                "possibly_missing": missing, "claims_checked": len(claimed), "confidence": round(confidence, 3),
                "confidence_note": "heuristic and uncalibrated; used only to route images to review",
                "independent_of_caption_source": independent, "strictness": cfg.strictness,
                "needs_review": status == "failed" or (status == "warning" and confidence < 0.6)}

    def apply(self, img, data):
        img.alignment_score = data.get("score")
