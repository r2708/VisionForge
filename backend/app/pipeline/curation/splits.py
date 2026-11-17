"""Processes 56-58: leakage-safe grouping, deterministic splitting and split validation."""
from __future__ import annotations

import hashlib
import os
from collections import Counter, defaultdict

from sqlalchemy import select, update

from app.database.models import Image
from app.database.session import kv_set
from app.pipeline.base import StageContext
from app.pipeline.deduplication.dedup import UnionFind

SPLITS = ("train", "validation", "test")


def _candidates(s):
    return s.execute(select(Image.id, Image.image_id, Image.exact_group, Image.perceptual_group, Image.near_group,
                            Image.semantic_group, Image.source_pk, Image.relative_path, Image.sha256)
                     .where(Image.decision == "accepted", Image.rights_eligible.is_(True), Image.missing.is_(False),
                            Image.processed_path.is_not(None)).order_by(Image.image_id)).all()


def group_keys(row, cfg) -> list[str]:
    """Every key that must not be shared by images in different splits."""
    keys = [f"sha:{row.sha256}"]  # byte-identical files always stay together
    if cfg.group_near_duplicates:
        keys += [k for k in (row.exact_group, row.perceptual_group, row.near_group) if k]
    if cfg.group_semantic and row.semantic_group:
        keys.append(row.semantic_group)
    if cfg.group_by_source_directory:
        keys.append(f"dir:{row.source_pk}:{os.path.dirname(row.relative_path)}")
    return keys


def run_splits(ctx: StageContext) -> dict:
    cfg = ctx.cfg.splits
    with ctx.ws.db.session() as s:
        rows = _candidates(s)
    uf = UnionFind()
    key_owner: dict[str, int] = {}
    for r in rows:
        uf.find(r.id)
        for k in group_keys(r, cfg):
            if k in key_owner:
                uf.union(key_owner[k], r.id)
            else:
                key_owner[k] = r.id
    by_root = defaultdict(list)
    for r in rows:
        by_root[uf.find(r.id)].append(r)
    groups = []
    for members in by_root.values():
        ids = sorted(m.image_id for m in members)
        gid = "grp_" + hashlib.sha256("|".join(ids).encode()).hexdigest()[:12]
        order = hashlib.sha256(f"{cfg.seed}:{gid}".encode()).hexdigest()
        groups.append((order, gid, members))
    groups.sort()  # deterministic pseudo-random order for a fixed seed

    ratios = {"train": cfg.train, "validation": cfg.validation, "test": cfg.test}
    total = len(rows)
    target = {k: v * total for k, v in ratios.items()}
    count = Counter()
    assign: dict[int, tuple[str, str]] = {}
    for _, gid, members in groups:
        # Largest remaining deficit wins; ties resolve in the fixed order train, validation, test.
        split = max((k for k in SPLITS if ratios[k] > 0), key=lambda k: (target[k] - count[k], -SPLITS.index(k)))
        count[split] += len(members)
        for m in members:
            assign[m.id] = (split, gid)

    with ctx.ws.db.session() as s:
        s.execute(update(Image).values(split=None, split_group=None))
        for pk, (split, gid) in assign.items():
            s.execute(update(Image).where(Image.id == pk).values(split=split, split_group=gid))
    report = validate_splits(ctx)
    report["assignment"] = {"seed": cfg.seed, "requested_ratios": ratios, "groups": len(groups),
                            "multi_image_groups": sum(1 for g in groups if len(g[2]) > 1),
                            "counts": {k: count[k] for k in SPLITS},
                            "actual_ratios": {k: round(count[k] / total, 4) if total else 0 for k in SPLITS}}
    if total and any(ratios[k] > 0 and count[k] == 0 for k in SPLITS):
        report["warnings"].append("at least one split is empty: the dataset is too small for the requested ratios")
    with ctx.ws.db.session() as s:
        kv_set(s, "report:split_leakage", report)
    return report


def validate_splits(ctx: StageContext) -> dict:
    """Independent re-check (process 58): no grouping key may appear in more than one split."""
    cfg = ctx.cfg.splits
    with ctx.ws.db.session() as s:
        rows = s.execute(select(Image.id, Image.image_id, Image.exact_group, Image.perceptual_group, Image.near_group,
                                Image.semantic_group, Image.source_pk, Image.relative_path, Image.split, Image.sha256)
                         .where(Image.split.is_not(None))).all()
    seen: dict[str, set] = defaultdict(set)
    members: dict[str, list] = defaultdict(list)
    for r in rows:
        for k in group_keys(r, cfg):
            seen[k].add(r.split)
            members[k].append(r.image_id)
    leaks = [{"key": k, "splits": sorted(v), "image_ids": members[k][:20]} for k, v in seen.items() if len(v) > 1]
    return {"leakage_detected": bool(leaks), "leaks": leaks[:200], "leak_count": len(leaks), "images_checked": len(rows),
            "grouping": {"near_duplicates": cfg.group_near_duplicates, "semantic": cfg.group_semantic,
                         "source_directory": cfg.group_by_source_directory},
            "warnings": [] if cfg.group_near_duplicates else
            ["group_near_duplicates is disabled: only byte-identical files are checked for leakage"]}
