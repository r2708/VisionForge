"""Processes 11, 13, 14, 15: exact, perceptual, near-duplicate and semantic grouping plus
representative recommendation. Nothing is deleted; results are recorded for review."""
from __future__ import annotations

import hashlib
from collections import defaultdict
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np
from sqlalchemy import select, update

from app.core.imageio import analysis_rgb
from app.core.logging import get_logger
from app.database.models import Embedding, Image, SeenHash, Source, StageResult
from app.database.session import kv_set
from app.pipeline.base import Cancelled, StageContext

log = get_logger("dedup")
RIGHTS_RANK = {"verified": 0, "user_attested": 1, "unknown": 2, "restricted": 3, "not_for_training": 4}
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


class UnionFind:
    def __init__(self):
        self.p: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def groups(self) -> list[list[int]]:
        out = defaultdict(list)
        for x in self.p:
            out[self.find(x)].append(x)
        return [sorted(g) for g in out.values() if len(g) > 1]


def popcount64(x: np.ndarray) -> np.ndarray:
    return _POP[x.view(np.uint8).reshape(-1, 8)].sum(axis=1)


def hamming_pairs(hashes: np.ndarray, threshold: int) -> list[tuple[int, int, int]]:
    """All index pairs with Hamming distance <= threshold. Uses the pigeonhole principle:
    two 64-bit hashes within distance t agree exactly on at least one of t+1 bit bands."""
    n = len(hashes)
    if n < 2:
        return []
    found: dict[tuple[int, int], int] = {}
    bands = threshold + 1
    edges = np.linspace(0, 64, bands + 1).astype(int)
    for b in range(bands):
        lo, width = int(edges[b]), int(edges[b + 1] - edges[b])
        if width == 0:
            continue
        key = (hashes >> np.uint64(lo)) & np.uint64((1 << width) - 1)
        order = np.argsort(key, kind="stable")
        sk = key[order]
        starts = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1], True])
        for s, e in zip(starts[:-1], starts[1:]):
            if e - s < 2:
                continue
            idx = order[s:e]
            for c in range(0, len(idx), 2048):  # bounded memory inside very large buckets
                part = idx[c:c + 2048]
                x = hashes[part][:, None] ^ hashes[idx][None, :]
                d = popcount64(x.reshape(-1)).reshape(x.shape)
                ii, jj = np.nonzero(d <= threshold)
                for a, bb in zip(ii, jj):
                    i, j = int(part[a]), int(idx[bb])
                    if i < j:
                        found[(i, j)] = int(d[a, bb])
    return [(i, j, d) for (i, j), d in found.items()]


def cross_hamming(new: np.ndarray, old: np.ndarray, threshold: int) -> dict[int, tuple[int, int]]:
    """For each hash in ``new``, the closest hash in ``old`` within ``threshold`` bits:
    {new index: (old index, distance)}. Same banding idea as ``hamming_pairs``, but only pairs
    across the two sets are examined, so a large ``old`` set stays cheap."""
    best: dict[int, tuple[int, int]] = {}
    if not len(new) or not len(old):
        return best
    edges = np.linspace(0, 64, threshold + 2).astype(int)
    for b in range(threshold + 1):
        lo, width = int(edges[b]), int(edges[b + 1] - edges[b])
        if width == 0:
            continue
        mask = np.uint64((1 << width) - 1)
        old_key = (old >> np.uint64(lo)) & mask
        order = np.argsort(old_key, kind="stable")
        sorted_key = old_key[order]
        new_key = (new >> np.uint64(lo)) & mask
        starts, ends = np.searchsorted(sorted_key, new_key, "left"), np.searchsorted(sorted_key, new_key, "right")
        for i in np.flatnonzero(ends > starts):
            cand = order[starts[i]:ends[i]]
            d = popcount64(old[cand] ^ new[i])
            k = int(np.argmin(d))
            if d[k] <= threshold and (int(i) not in best or d[k] < best[int(i)][1]):
                best[int(i)] = (int(cand[k]), int(d[k]))
    return best


HASH_BRUTE_FORCE_LIMIT = 6000


def hamming_pairs_bruteforce(hashes: np.ndarray, threshold: int, chunk: int = 512) -> list[tuple[int, int, int]]:
    """Exhaustive variant for loose thresholds on small collections."""
    out = []
    for s in range(0, len(hashes), chunk):
        x = hashes[s:s + chunk][:, None] ^ hashes[None, :]
        d = popcount64(x.reshape(-1)).reshape(x.shape)
        ii, jj = np.nonzero(d <= threshold)
        out += [(s + int(a), int(j), int(d[a, j])) for a, j in zip(ii, jj) if s + a < j]
    return out


def cosine_pairs(vecs: np.ndarray, threshold: float, chunk: int = 1024, cap: int = 2_000_000):
    """Index pairs (i<j) whose cosine similarity is >= threshold, computed in chunks."""
    out = []
    n = len(vecs)
    for s in range(0, n, chunk):
        sims = vecs[s:s + chunk] @ vecs.T
        ii, jj = np.nonzero(sims >= threshold)
        for a, j in zip(ii, jj):
            i = s + int(a)
            if i < j:
                out.append((i, int(j), float(sims[a, j])))
        if len(out) > cap:
            log.warning("similarity pair cap reached; raise the threshold for this dataset")
            break
    return out


def cosine_topk_pairs(vecs: np.ndarray, threshold: float, k: int, chunk: int = 1024):
    """For every image, its ``k`` most similar other images above ``threshold`` (pairs as i<j)."""
    out: dict[tuple[int, int], float] = {}
    n = len(vecs)
    k = min(k, n - 1)
    if k < 1:
        return []
    for s in range(0, n, chunk):
        sims = vecs[s:s + chunk] @ vecs.T
        rows = np.arange(sims.shape[0])
        sims[rows, s + rows] = -1.0  # ignore self
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        for a in rows:
            for j in top[a]:
                v = float(sims[a, j])
                if v >= threshold:
                    i = s + int(a)
                    out[(min(i, int(j)), max(i, int(j)))] = v
    return [(i, j, v) for (i, j), v in out.items()]


@lru_cache(maxsize=512)
def _orb(path: str):
    gray = np.asarray(analysis_rgb(path, 512).convert("L"))
    return cv2.ORB_create(nfeatures=800).detectAndCompute(gray, None)


def verify_near_duplicate(path_a: str, path_b: str, min_inliers: int, min_ratio: float) -> Optional[dict]:
    """Geometric verification: ORB matches consistent with one homography (handles crops,
    resizes, small edits and overlays)."""
    try:
        (ka, da), (kb, db) = _orb(path_a), _orb(path_b)
    except Exception:
        return None
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        return None
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
    if len(good) < max(8, min_inliers // 2):
        return None
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None or mask is None:
        return None
    inliers = int(mask.sum())
    ratio = inliers / max(1, min(len(ka), len(kb)))
    if inliers >= min_inliers and ratio >= min_ratio:
        return {"inliers": inliers, "inlier_ratio": round(ratio, 3)}
    return None


def _gid(prefix: str, image_ids: list[str]) -> str:
    return f"{prefix}_" + hashlib.sha256("|".join(sorted(image_ids)).encode()).hexdigest()[:12]


def run_dedup(ctx: StageContext) -> dict:
    cfg, db = ctx.cfg.deduplication, ctx.ws.db
    with db.session() as s:
        rows = s.execute(select(Image.id, Image.image_id, Image.sha256, Image.abs_path, Image.quality_score,
                                Image.width, Image.height, Image.file_size, Image.rights_status, Image.source_pk,
                                Image.dup_override)
                         .where(Image.valid.is_(True), Image.missing.is_(False)).order_by(Image.id)).all()
        ph = {pk: d for pk, d in s.execute(select(StageResult.image_pk, StageResult.data)
                                           .where(StageResult.stage == "phash", StageResult.status == "done"))}
        src_rights = dict(s.execute(select(Source.id, Source.rights_status)).all())
    info = {r.id: r for r in rows}
    pks = [r.id for r in rows]
    pos = {pk: i for i, pk in enumerate(pks)}
    report: dict = {"thresholds": cfg.model_dump(), "images_considered": len(pks), "notes": []}

    say = lambda msg: ctx.progress("dedup", {"message": msg})  # noqa: E731
    say(f"comparing {len(pks)} images: exact and perceptual duplicates")
    # ---- exact (11)
    exact_uf = UnionFind()
    by_sha = defaultdict(list)
    for r in rows:
        by_sha[r.sha256].append(r.id)
    if cfg.exact:
        for members in by_sha.values():
            for m in members[1:]:
                exact_uf.union(members[0], m)

    # ---- perceptual (12 -> groups)
    perc_uf = UnionFind()
    hashed = [pk for pk in pks if pk in ph]
    p_arr = np.array([int(ph[pk]["phash"], 16) for pk in hashed], dtype=np.uint64)
    d_arr = np.array([int(ph[pk]["dhash"], 16) for pk in hashed], dtype=np.uint64)
    perc_pairs = set()
    if cfg.perceptual and len(hashed) > 1:
        for i, j, d in hamming_pairs(p_arr, cfg.perceptual_threshold):
            # A second, independent hash guards against chance pHash collisions on flat images.
            if int(popcount64(np.array([d_arr[i] ^ d_arr[j]], dtype=np.uint64))[0]) <= cfg.perceptual_threshold + 6:
                perc_uf.union(hashed[i], hashed[j])
                perc_pairs.add((hashed[i], hashed[j]))
    elif cfg.perceptual and not hashed:
        report["notes"].append("perceptual hashing results unavailable; perceptual grouping skipped")
    if ctx.cancel():
        raise Cancelled()

    # ---- embeddings for near-duplicate candidates and semantic similarity
    vecs, emb_pks = None, []
    if cfg.near or cfg.semantic:
        with db.session() as s:
            model_keys = {pk: (d or {}).get("model_key") for pk, d in s.execute(
                select(StageResult.image_pk, StageResult.data).where(StageResult.stage == "embedding", StageResult.status == "done"))}
            keys = {k for k in model_keys.values() if k}
            if len(keys) == 1:
                mk = keys.pop()
                table = {sha: np.frombuffer(v, dtype=np.float32) for sha, v in
                         s.execute(select(Embedding.sha256, Embedding.vector).where(Embedding.model_key == mk))}
                emb_pks = [pk for pk in pks if pk in model_keys and info[pk].sha256 in table]
                if emb_pks:
                    vecs = np.stack([table[info[pk].sha256] for pk in emb_pks])
            elif len(keys) > 1:
                report["notes"].append("embeddings from more than one model present; re-run the embedding stage")
        if vecs is None:
            report["notes"].append("no embeddings available: semantic grouping skipped; near-duplicate search limited to hash candidates")

    # ---- near duplicates (13)
    near_uf = UnionFind()
    near_evidence = {}
    if cfg.near:
        cand: dict[tuple[int, int], dict] = {}
        if 1 < len(hashed) <= HASH_BRUTE_FORCE_LIMIT:
            per_image: dict[int, int] = defaultdict(int)
            for i, j, d in sorted(hamming_pairs_bruteforce(p_arr, cfg.near_candidate_hamming), key=lambda t: t[2]):
                if per_image[i] < cfg.near_max_candidates_per_image and per_image[j] < cfg.near_max_candidates_per_image:
                    per_image[i] += 1
                    per_image[j] += 1
                    cand[(hashed[i], hashed[j])] = {"phash_distance": d}
        elif len(hashed) > HASH_BRUTE_FORCE_LIMIT and vecs is None:
            report["notes"].append(f"more than {HASH_BRUTE_FORCE_LIMIT} images and no embeddings: near-duplicate search skipped")
        if vecs is not None:
            for i, j, sim in cosine_topk_pairs(vecs, cfg.near_candidate_cosine, cfg.near_max_candidates_per_image):
                cand.setdefault((emb_pks[i], emb_pks[j]), {})["cosine"] = round(sim, 4)
        checked = 0
        say(f"near duplicates: verifying up to {len(cand)} candidate pairs geometrically")
        for n_seen, ((a, b), ev) in enumerate(sorted(cand.items()), 1):
            if n_seen % 250 == 0:
                say(f"near duplicates: {n_seen}/{len(cand)} candidate pairs checked, {len(near_evidence)} confirmed")
            if info[a].sha256 == info[b].sha256 or (a, b) in perc_pairs:
                continue
            if perc_uf.find(a) == perc_uf.find(b) or near_uf.find(a) == near_uf.find(b):
                continue
            if checked % 50 == 0 and ctx.cancel():
                raise Cancelled()
            checked += 1
            v = verify_near_duplicate(info[a].abs_path, info[b].abs_path, cfg.near_min_inliers, cfg.near_min_inlier_ratio)
            if v:
                near_uf.union(a, b)
                near_evidence[(a, b)] = {**ev, **v}
        report["near_candidates_checked"] = checked
        _orb.cache_clear()

    # ---- semantic similarity (14)
    sem_uf = UnionFind()
    if cfg.semantic and vecs is not None:
        say("semantic similarity groups")
        for i, j, sim in cosine_pairs(vecs, cfg.semantic_threshold):
            sem_uf.union(emb_pks[i], emb_pks[j])

    # ---- representative recommendation (15) over the duplicate cluster (exact + perceptual + near)
    cluster = UnionFind()
    for uf in (exact_uf, perc_uf, near_uf):
        for g in uf.groups():
            for m in g[1:]:
                cluster.union(g[0], m)

    def rank(pk: int):
        r = info[pk]
        rights = r.rights_status or src_rights.get(r.source_pk) or "unknown"
        # Quality is compared in 5-point bands so tiny score differences do not outrank resolution.
        return (0 if r.dup_override == "representative" else 1, -round((r.quality_score or 0) / 5),
                -((r.width or 0) * (r.height or 0)), RIGHTS_RANK.get(rights, 2), -r.file_size, r.image_id)

    status = {pk: "unique" for pk in pks}
    duplicate_of: dict[int, int] = {}
    for g in cluster.groups():
        best = min(g, key=rank)
        for m in g:
            if m == best:
                status[m] = "representative"
            elif info[m].dup_override == "keep":
                status[m] = "kept_by_user"
            else:
                status[m] = "duplicate"
                duplicate_of[m] = best

    # ---- earlier shards: exact copies and perceptual duplicates of images that were already exported
    earlier: dict[int, str] = {}
    if cfg.cross_shard:
        with db.session() as s:
            seen = s.execute(select(SeenHash.sha256, SeenHash.phash, SeenHash.dhash, SeenHash.image_id, SeenHash.shard)).all()
        if seen:
            say(f"comparing with {len(seen)} images exported in earlier shards")
            by_seen_sha = {r.sha256: r for r in seen}
            for pk in pks:
                hit = by_seen_sha.get(info[pk].sha256)
                if hit is not None:
                    earlier[pk] = f"{hit.shard}/{hit.image_id}"
            with_hash = [r for r in seen if r.phash and r.dhash]
            if cfg.perceptual and with_hash and hashed:
                old_p = np.array([int(r.phash, 16) for r in with_hash], dtype=np.uint64)
                old_d = np.array([int(r.dhash, 16) for r in with_hash], dtype=np.uint64)
                for i, (j, _) in cross_hamming(p_arr, old_p, cfg.perceptual_threshold).items():
                    if hashed[i] not in earlier and int(popcount64(np.array([d_arr[i] ^ old_d[j]], dtype=np.uint64))[0]) <= cfg.perceptual_threshold + 6:
                        earlier[hashed[i]] = f"{with_hash[j].shard}/{with_hash[j].image_id}"
            # The rest of a duplicate group duplicates the same earlier image.
            for g in cluster.groups():
                ref = next((earlier[m] for m in g if m in earlier), None)
                if ref:
                    for m in g:
                        earlier.setdefault(m, ref)
            for pk in list(earlier):
                if info[pk].dup_override == "keep":
                    del earlier[pk]
                else:
                    status[pk] = "duplicate"
                    duplicate_of.pop(pk, None)
        report["earlier_shards"] = {"images_compared_against": len(seen), "duplicates_of_earlier_shards": len(earlier),
                                    "method": "exact checksum, then pHash within the perceptual threshold confirmed by dHash",
                                    "examples": [{"image_id": info[pk].image_id, "duplicate_of": ref} for pk, ref in list(earlier.items())[:200]]}

    def label(uf: UnionFind, prefix: str) -> tuple[dict[int, str], list[dict]]:
        mapping, groups = {}, []
        for g in uf.groups():
            gid = _gid(prefix, [info[m].image_id for m in g])
            for m in g:
                mapping[m] = gid
            groups.append({"group_id": gid, "size": len(g), "recommended_representative": info[min(g, key=rank)].image_id,
                           "members": [{"image_id": info[m].image_id, "quality_score": info[m].quality_score,
                                        "width": info[m].width, "height": info[m].height,
                                        "representative_status": status[m]} for m in g]})
        groups.sort(key=lambda x: (-x["size"], x["group_id"]))
        return mapping, groups

    maps = {}
    for field, uf, prefix in (("exact_group", exact_uf, "ex"), ("perceptual_group", perc_uf, "pd"),
                              ("near_group", near_uf, "nd"), ("semantic_group", sem_uf, "sg")):
        maps[field], groups = label(uf, prefix)
        kind = field.replace("_group", "")
        report[kind] = {"group_count": len(groups), "image_count": sum(g["size"] for g in groups),
                        "groups": groups[:500], "truncated": len(groups) > 500}
    report["near"]["evidence"] = [{"a": info[a].image_id, "b": info[b].image_id, **ev} for (a, b), ev in list(near_evidence.items())[:500]]
    report["duplicate_images"] = sum(1 for v in status.values() if v == "duplicate")
    report["policy"] = ("Duplicates are never deleted. Non-representative members are "
                        f"{'quarantined' if cfg.non_representative_action == 'quarantine' else 'flagged'}; "
                        "semantic groups are informational unless max_per_semantic_group is set.")

    with db.session() as s:
        for i in range(0, len(pks), 500):
            for pk in pks[i:i + 500]:
                s.execute(update(Image).where(Image.id == pk).values(
                    exact_group=maps["exact_group"].get(pk), perceptual_group=maps["perceptual_group"].get(pk),
                    near_group=maps["near_group"].get(pk), semantic_group=maps["semantic_group"].get(pk),
                    representative_status=status[pk]))
        kv_set(s, "report:duplicates", report)
        kv_set(s, "dedup:duplicate_of", {info[k].image_id: info[v].image_id for k, v in duplicate_of.items()}
               | {info[k].image_id: ref for k, ref in earlier.items()})
    return {k: report[k]["group_count"] for k in ("exact", "perceptual", "near", "semantic")} | {
        "duplicate_images": report["duplicate_images"], "duplicates_of_earlier_shards": len(earlier)}
