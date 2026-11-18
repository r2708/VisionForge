"""Captioning stage (39-44) and dataset-level caption diversity analysis (45)."""
from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import select

from app.database.models import Image, StageResult
from app.database.session import kv_set
from app.pipeline.base import ImageStage, ImgRef, StageContext

from .adapters import (AutoCaptionAdapter, SidecarAdapter, SmolVLMCaptionAdapter, TemplateAdapter, build_tags, grounded_facts)

EMPTY_PHRASES = ("a beautiful image of", "a stunning image of", "an image of an image", "a picture of a picture",
                 "high quality", "8k", "4k", "masterpiece", "best quality")


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'-]+", text or ""))


class CaptioningStage(ImageStage):
    name = "captioning"
    label = "CAPTION"

    def describe(self, d):
        text = d.get("short") or d.get("detailed") or ""
        origin = (d.get("caption_model") or d.get("source") or "") if d.get("adapter") == "auto" else ""
        return (f"{origin}: " if origin else "") + (f'"{text[:90]}"' if text else "no caption")

    version = "1"
    title = "Captions, generation prompt and tags"
    processes = "39, 40, 41, 42, 43, 44"
    depends = ("validate",)
    soft_depends = ("quality_technical", "zero_shot", "people", "ocr", "object_detection")
    batched = True

    def enabled(self, ctx):
        c = ctx.cfg
        if c.stages.captioning is False or c.captioning.adapter == "none":
            return False, "captioning disabled (adapter: none)"
        if c.captioning.adapter in ("smolvlm", "auto"):
            st = ctx.hub.status(c.models.captioner)
            return st["usable"], st.get("reason", "")
        return True, ""

    def fingerprint(self, cfg):
        c = cfg.captioning
        adapter_version = {"template": TemplateAdapter().version(), "sidecar": SidecarAdapter().version(),
                           "smolvlm": {"prompt_version": "smolvlm-prompts-1"},
                           "auto": {"prompt_version": "smolvlm-prompts-1", "template_version": "auto-2", "source": c.supplied_source}}.get(c.adapter)
        return [c.adapter, adapter_version, c.short_caption, c.detailed_caption, c.comprehensive_caption,
                c.generation_prompt, c.alternate_caption_count, c.max_new_tokens, c.sidecar_extension,
                c.vlm_image_splitting if c.adapter == "auto" else None]

    def model_info(self, ctx):
        if ctx.cfg.captioning.adapter in ("smolvlm", "auto"):
            return ctx.hub.info(ctx.cfg.models.captioner)
        return {"name": ctx.cfg.captioning.adapter, "revision": None}

    def batch_size(self, ctx):
        return 1 if ctx.cfg.captioning.adapter in ("smolvlm", "auto") else 64  # one at a time: each terminal line shows its own time

    def chunk_size(self, ctx):
        # Model captions are slow: save them every few images so a stop loses little work.
        return 20 if ctx.cfg.captioning.adapter in ("smolvlm", "auto") else None

    def eligible(self, ctx, img: Image) -> bool:
        # Do not spend caption inference on images that are already excluded for hard reasons.
        if img.manual_decision == "accepted":
            return True
        dropped_duplicate = (img.representative_status == "duplicate"
                             and ctx.cfg.deduplication.non_representative_action == "quarantine")
        return img.auto_decision != "rejected" and not dropped_duplicate

    def setup(self, ctx):
        a = ctx.cfg.captioning.adapter
        if a == "template":
            self.adapter = TemplateAdapter()
        elif a == "sidecar":
            self.adapter = SidecarAdapter()
        elif a == "auto":
            self.adapter = AutoCaptionAdapter(ctx)
        else:
            self.adapter = SmolVLMCaptionAdapter(ctx.hub.get(ctx.cfg.models.captioner))

    def teardown(self, ctx):
        used_model = ctx.cfg.captioning.adapter == "smolvlm" or getattr(self.adapter, "vlm", None) is not None
        if isinstance(self.adapter, AutoCaptionAdapter):
            ctx.progress("captioning", {"message": self.adapter.summary()})
        self.adapter = None  # drop the stage's own reference too, or the weights stay in memory
        if used_model:
            ctx.hub.unload(ctx.cfg.models.captioner)

    def process(self, ctx: StageContext, ref: ImgRef) -> dict:
        cfg = ctx.cfg.captioning
        facts = grounded_facts(ref)
        out = self.adapter.generate(ctx, ref, facts, cfg)
        for field, on in (("short", cfg.short_caption), ("detailed", cfg.detailed_caption),
                          ("comprehensive", cfg.comprehensive_caption), ("generation_prompt", cfg.generation_prompt)):
            if not on:
                out[field] = ""
        warnings = []
        main = out.get(cfg.training_caption_field) or out.get("detailed") or out.get("short") or ""
        if out.get("missing"):
            warnings.append("no_user_caption_found")
        elif word_count(main) < 4:
            warnings.append("low_information_caption")
        if any(p in main.lower() for p in EMPTY_PHRASES):
            warnings.append("empty_or_styling_phrase")
        by_model = self.adapter.name == "smolvlm" or bool(out.get("caption_model"))
        return {**out, "adapter": self.adapter.name, "source": out.get("source") or self.adapter.source, **self.adapter.version(),
                "tags": build_tags(facts), "facts_used": facts, "warnings": warnings,
                "settings": {"decoding": "greedy", "max_new_tokens": cfg.max_new_tokens} if by_model else {},
                "generated_at": datetime.now(timezone.utc).isoformat()}

    def still_valid(self, ctx, img, data):
        if data.get("adapter") not in ("sidecar", "auto"):
            return True
        ref = ImgRef(pk=img.id, image_id=img.image_id, abs_path=img.abs_path, sha256=img.sha256, extension=img.extension,
                     file_name=img.file_name, relative_path=img.relative_path, width=img.width, height=img.height,
                     supplied_caption=img.supplied_caption)
        return SidecarAdapter().read(ref, ctx.cfg.captioning)[1] == data.get("sidecar_sha256")

    def apply(self, img, data):
        img.short_caption = (img.user_edits or {}).get("captions", {}).get("short") or data.get("short") or data.get("detailed") or None


def effective_captions(img: Image, data: dict | None) -> dict:
    """Generated captions with any manual edits applied on top."""
    base = {k: (data or {}).get(k, "") for k in ("short", "detailed", "comprehensive", "generation_prompt")}
    base["alternatives"] = list((data or {}).get("alternatives", []))
    edits = (img.user_edits or {}).get("captions", {})
    for k, v in edits.items():
        if k in base and v is not None:
            base[k] = v
    return base


def run_caption_diversity(ctx: StageContext) -> dict:
    """Process 45: repetitive openings, boilerplate, duplicated and low-information captions."""
    field = ctx.cfg.captioning.training_caption_field
    openings, texts, lengths = Counter(), Counter(), []
    vocab: Counter = Counter()
    low_info, by_text = [], {}
    with ctx.ws.db.session() as s:
        q = (select(Image, StageResult.data).join(StageResult, StageResult.image_pk == Image.id)
             .where(StageResult.stage == "captioning", StageResult.status == "done", Image.missing.is_(False))
             .execution_options(yield_per=500))
        for img, data in s.execute(q):
            text = effective_captions(img, data).get(field) or ""
            if not text:
                continue
            words = re.findall(r"[a-z0-9'-]+", text.lower())
            lengths.append(len(words))
            vocab.update(words)
            openings[" ".join(words[:3])] += 1
            norm = " ".join(words)
            texts[norm] += 1
            by_text.setdefault(norm, []).append(img.image_id)
            if len(words) < 4 and len(low_info) < 200:
                low_info.append(img.image_id)
    n = len(lengths)
    if n == 0:
        report = {"captions_analyzed": 0, "field": field, "warnings": ["no captions available"]}
    else:
        top_open = openings.most_common(10)
        dup = [(t, c) for t, c in texts.most_common(20) if c > 1]
        unique_ratio = len(texts) / n
        ttr = len(vocab) / max(1, sum(vocab.values()))
        opening_share = top_open[0][1] / n
        # 0-100: unique captions and varied openings matter most.
        score = round(100 * (0.5 * unique_ratio + 0.3 * (1 - opening_share if n > 4 else 1.0) + 0.2 * min(1.0, ttr * 4)), 1)
        warnings = []
        if n > 4 and opening_share > 0.4:
            warnings.append(f"{opening_share:.0%} of captions start with '{top_open[0][0]}'")
        if unique_ratio < 0.9:
            warnings.append(f"{1 - unique_ratio:.0%} of captions are exact repeats of another caption")
        if low_info:
            warnings.append(f"{len(low_info)} caption(s) have fewer than four words")
        report = {
            "captions_analyzed": n, "field": field, "diversity_score": score,
            "mean_words": round(sum(lengths) / n, 1), "min_words": min(lengths), "max_words": max(lengths),
            "unique_caption_ratio": round(unique_ratio, 4), "type_token_ratio": round(ttr, 4),
            "top_openings": [{"opening": o, "count": c, "share": round(c / n, 4)} for o, c in top_open],
            "repeated_captions": [{"caption": t, "count": c, "image_ids": by_text[t][:20]} for t, c in dup],
            "overused_words": [{"word": w, "count": c} for w, c in vocab.most_common(40) if len(w) > 3][:15],
            "low_information_image_ids": low_info, "warnings": warnings,
        }
    with ctx.ws.db.session() as s:
        kv_set(s, "report:caption_diversity", report)
    return {"diversity_score": report.get("diversity_score"), "warnings": len(report.get("warnings", []))}
