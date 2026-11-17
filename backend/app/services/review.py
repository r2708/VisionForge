"""Manual review actions (process 51). Every action is logged and reversible; nothing is deleted."""
from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from app.config.settings import RIGHTS_STATUSES
from app.core.workspace import Workspace
from app.database.models import DecisionLog, Image, Source

ACTIONS = {"approve": "accepted", "reject": "rejected", "quarantine": "quarantined"}
CAPTION_FIELDS = ("short", "detailed", "comprehensive", "generation_prompt")


def _get(s, image_id: str) -> Image:
    im = s.scalar(select(Image).where(Image.image_id == image_id))
    if im is None:
        raise KeyError(f"image {image_id} not found")
    return im


def decide(ws: Workspace, image_id: str, action: str, note: Optional[str] = None) -> dict:
    """approve | reject | quarantine | restore (restore returns the image to the automatic decision)."""
    if action not in ACTIONS and action != "restore":
        raise ValueError("action must be approve, reject, quarantine or restore")
    with ws.db.session() as s:
        im = _get(s, image_id)
        before = {"decision": im.decision, "review_status": im.review_status}
        if action == "restore":
            im.manual_decision = None
            im.decision = im.auto_decision or "pending"
            im.review_status = "pending" if im.review_reasons else "not_required"
        else:
            if action == "approve" and im.valid is not True:
                raise ValueError("an invalid or unvalidated file cannot be approved")
            im.manual_decision = ACTIONS[action]
            im.decision = ACTIONS[action]
            im.review_status = "approved" if action == "approve" else "rejected"
        s.add(DecisionLog(image_id=image_id, actor="user", action=f"review_{action}",
                          detail={"before": before, "after": {"decision": im.decision, "review_status": im.review_status}, "note": note}))
        consequence = {
            "approve": "The image will be included in the next export if its rights status is allowed.",
            "reject": "The image is excluded from exports. Its source file is untouched; use restore to undo.",
            "quarantine": "The image is held out of exports for later review. Its source file is untouched; use restore to undo.",
            "restore": "The manual decision was removed; the automatic decision applies again.",
        }[action]
        return {"image_id": image_id, "decision": im.decision, "review_status": im.review_status,
                "rights_eligible": im.rights_eligible, "consequence": consequence,
                "next_step": "Run an export (or the pipeline) to rebuild processed images and manifests."}


def edit(ws: Workspace, image_id: str, captions: Optional[dict] = None, tags: Optional[list[str]] = None,
         rights: Optional[dict] = None) -> dict:
    with ws.db.session() as s:
        im = _get(s, image_id)
        edits = dict(im.user_edits or {})
        changed = []
        if captions:
            cur = dict(edits.get("captions", {}))
            for k, v in captions.items():
                if k not in CAPTION_FIELDS:
                    raise ValueError(f"unknown caption field '{k}'")
                if v is None:
                    cur.pop(k, None)  # None reverts to the generated caption
                else:
                    cur[k] = " ".join(str(v).split())
                changed.append(f"captions.{k}")
            edits["captions"] = cur
            if "short" in cur:
                im.short_caption = cur["short"]
        if tags is not None:
            clean = list(dict.fromkeys(t.strip().lower() for t in tags if t and t.strip()))
            edits["tags"] = clean
            changed.append("tags")
        if rights:
            for k, v in rights.items():
                if k == "rights_status":
                    if v not in RIGHTS_STATUSES:
                        raise ValueError(f"rights_status must be one of {RIGHTS_STATUSES}")
                    im.rights_status = v
                elif k in ("license", "creator", "attribution", "source_url", "notes"):
                    setattr(im, k, v)
                else:
                    raise ValueError(f"unknown rights field '{k}'")
                changed.append(k)
            allowed = ws.config.rights.allowed_statuses
            im.rights_eligible = (not ws.config.rights.require_approved_status_for_export) or im.effective_rights() in allowed
        im.user_edits = edits
        if (captions or tags is not None) and im.review_status in ("pending", "not_required"):
            im.review_status = "edited"
        s.add(DecisionLog(image_id=image_id, actor="user", action="edit", detail={"fields": changed}))
        return {"image_id": image_id, "changed": changed, "review_status": im.review_status}


def set_duplicate_override(ws: Workspace, image_id: str, override: Optional[str]) -> dict:
    """keep = not a redundant duplicate; representative = prefer this member; None clears."""
    if override not in (None, "keep", "representative"):
        raise ValueError("override must be keep, representative or null")
    with ws.db.session() as s:
        im = _get(s, image_id)
        im.dup_override = override
        s.add(DecisionLog(image_id=image_id, actor="user", action="duplicate_override", detail={"override": override}))
    return {"image_id": image_id, "dup_override": override,
            "next_step": "Re-run the dedup and curation stages to apply this choice."}


def set_source_rights(ws: Workspace, source_id: str, **fields) -> dict:
    with ws.db.session() as s:
        src = s.scalar(select(Source).where(Source.source_id == source_id))
        if src is None:
            raise KeyError(f"source {source_id} not found")
        for k, v in fields.items():
            if v is None:
                continue
            if k == "rights_status" and v not in RIGHTS_STATUSES:
                raise ValueError(f"rights_status must be one of {RIGHTS_STATUSES}")
            if k not in ("rights_status", "license", "creator", "attribution", "source_url", "notes"):
                raise ValueError(f"unknown field '{k}'")
            setattr(src, k, v)
        s.add(DecisionLog(actor="user", action="source_rights_updated", detail={"source_id": source_id, "fields": sorted(fields)}))
        return {"source_id": source_id, "rights_status": src.rights_status, "license": src.license,
                "next_step": "Run the pipeline or an export so eligibility and manifests are recalculated."}
