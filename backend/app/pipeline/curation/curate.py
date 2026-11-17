"""Processes 52-55 and 59: multi-signal filtering, redundancy control, optional balancing and
rights eligibility. Decisions are recorded with reasons; a manual decision always wins and
nothing is deleted, so every outcome can be reviewed and reversed."""
from __future__ import annotations

from collections import Counter, defaultdict

from sqlalchemy import select

from app.database.models import DecisionLog, Image, StageResult
from app.database.session import kv_get, kv_set
from app.pipeline.base import StageContext

STAGES_USED = ("quality", "zero_shot", "ocr", "captioning", "alignment")
MANUAL_REVIEW_STATES = ("approved", "rejected", "edited")


def run_curation(ctx: StageContext, final: bool = True) -> dict:
    """``final=False`` is the pre-pass used to avoid captioning images that are already
    excluded for hard reasons; caption-related rules are applied only in the final pass."""
    cfg, db = ctx.cfg, ctx.ws.db
    q, d, r = cfg.quality, cfg.deduplication, cfg.rights
    lo, hi = q.manual_review_band
    captions_expected = final and cfg.captioning.adapter != "none" and cfg.stages.captioning is not False
    counts: Counter = Counter()
    reason_counts: Counter = Counter()

    with db.session() as s:
        dup_of = kv_get(s, "dedup:duplicate_of", {}) or {}

    # ---- optional caps need a global view: rank members by quality first
    capped: dict[int, str] = {}
    if final and (d.max_per_semantic_group or cfg.curation.max_per_category):
        with db.session() as s:
            rows = s.execute(select(Image.id, Image.semantic_group, Image.category, Image.quality_score, Image.image_id)
                             .where(Image.valid.is_(True), Image.missing.is_(False))).all()
        if d.max_per_semantic_group:
            groups = defaultdict(list)
            for row in rows:
                if row.semantic_group:
                    groups[row.semantic_group].append(row)
            for members in groups.values():
                members.sort(key=lambda x: (-(x.quality_score or 0), x.image_id))
                for m in members[d.max_per_semantic_group:]:
                    capped[m.id] = "redundant_in_semantic_group"
        if cfg.curation.max_per_category:
            cats = defaultdict(list)
            for row in rows:
                if row.category and row.id not in capped:
                    cats[row.category].append(row)
            for members in cats.values():
                members.sort(key=lambda x: (-(x.quality_score or 0), x.image_id))
                for m in members[cfg.curation.max_per_category:]:
                    capped[m.id] = "category_cap"

    last = 0
    while True:
        with db.session() as s:
            imgs = list(s.scalars(select(Image).where(Image.id > last, Image.missing.is_(False)).order_by(Image.id)
                                  .limit(cfg.processing.db_chunk_size)))
            if not imgs:
                break
            last = imgs[-1].id
            res: dict[tuple[int, str], StageResult] = {}
            for row in s.scalars(select(StageResult).where(StageResult.image_pk.in_([i.id for i in imgs]))):
                res[(row.image_pk, row.stage)] = row
            for im in imgs:
                reject, quarantine, review, flags = [], [], [], []

                def data(stage):
                    row = res.get((im.id, stage))
                    return row.data if row is not None and row.status == "done" else None

                failed_stages = sorted(st for (pk, st), row in res.items() if pk == im.id and row.status == "failed")
                if im.valid is None:
                    quarantine.append("not_validated")
                elif im.valid is False:
                    reject.append(f"invalid_file:{im.validation_status}")
                else:
                    qd = data("quality")
                    if qd:
                        flags = qd.get("flags", [])
                        if qd["resolution_status"] != "accepted":
                            reject.append("below_min_resolution")
                        score = qd.get("overall_score")
                        if score is not None:
                            in_band = lo <= score <= hi
                            if score < q.min_overall_score:
                                if in_band:
                                    quarantine.append("quality_below_threshold_pending_review")
                                    review.append("quality_in_review_band")
                                else:
                                    reject.append("low_quality_score")
                            elif in_band:
                                review.append("quality_in_review_band")
                        aes = qd["components"].get("aesthetic")
                        if q.min_aesthetic_score is not None and aes is not None and aes < q.min_aesthetic_score:
                            reject.append("low_aesthetic_score")
                        if "watermark_likely" in flags or "watermark_text_detected" in flags:
                            (quarantine if q.reject_watermarked else review).append("possible_watermark")
                        od = data("ocr")
                        if q.max_text_area_fraction is not None and od and od["text_area_fraction"] > q.max_text_area_fraction:
                            quarantine.append("text_heavy")
                    elif final:
                        quarantine.append("quality_not_computed")
                    if im.category and im.category in cfg.curation.exclude_categories:
                        reject.append(f"excluded_category:{im.category}")
                    if im.representative_status == "duplicate":
                        if d.non_representative_action == "quarantine":
                            quarantine.append("duplicate_of:" + dup_of.get(im.image_id, "group_representative"))
                        else:
                            flags = flags + ["duplicate_flagged"]
                    if im.id in capped:
                        quarantine.append(capped[im.id])
                    for st in failed_stages:
                        if st != "validate":
                            quarantine.append(f"stage_failed:{st}")
                    if captions_expected and not reject and not quarantine:
                        cd = data("captioning")
                        main = ((im.user_edits or {}).get("captions", {}).get(cfg.captioning.training_caption_field)
                                or (cd or {}).get(cfg.captioning.training_caption_field) or "")
                        if not main.strip():
                            quarantine.append("missing_caption")
                        else:
                            if cd and cd.get("source") == "template_baseline" and cfg.captioning.review_template_captions:
                                review.append("baseline_template_caption")
                            if cd and "low_information_caption" in cd.get("warnings", []):
                                review.append("low_information_caption")
                            ad = data("alignment")
                            if ad and ad.get("needs_review"):
                                review.append(f"caption_verification_{ad['verification_status']}")
                                if ad["verification_status"] == "failed" and cfg.captioning.strictness == "strict":
                                    quarantine.append("caption_verification_failed")

                auto = "rejected" if reject else ("quarantined" if quarantine else "accepted")
                reasons = reject + quarantine
                rights = im.effective_rights()
                im.rights_eligible = (not r.require_approved_status_for_export) or rights in r.allowed_statuses
                old = im.decision
                im.auto_decision = auto
                im.decision = im.manual_decision or auto
                im.decision_reasons = reasons if not im.manual_decision else reasons + [f"manual_override:{im.manual_decision}"]
                if im.review_status not in MANUAL_REVIEW_STATES:
                    im.review_status = "pending" if review and auto != "rejected" else "not_required"
                im.review_reasons = review
                if old != im.decision:
                    s.add(DecisionLog(job_id=ctx.job_id, image_id=im.image_id, actor="auto", action="decision_changed",
                                      detail={"from": old, "to": im.decision, "reasons": reasons}))
                counts[im.decision] += 1
                counts["review_pending"] += im.review_status == "pending"
                counts["rights_blocked"] += (im.decision == "accepted" and not im.rights_eligible)
                for reason in reasons:
                    reason_counts[reason.split(":")[0]] += 1
        ctx.heartbeat()

    summary = {"final": final, "decisions": {k: counts[k] for k in ("accepted", "rejected", "quarantined")},
               "review_pending": counts["review_pending"], "accepted_but_rights_blocked": counts["rights_blocked"],
               "reasons": dict(reason_counts.most_common()),
               "balancing": {"max_per_category": cfg.curation.max_per_category,
                             "max_per_semantic_group": d.max_per_semantic_group, "images_capped": len(capped)}}
    with db.session() as s:
        kv_set(s, "report:curation", summary)
    return summary
