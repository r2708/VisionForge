"""Caption-model benchmark on a sample of the user's own images.

Measured automatically: latency, memory, failure rate, caption length/diversity and the
image-text alignment signal. Factual accuracy, hallucination rate and salient-detail coverage
cannot be measured without a person looking at the images, so the run also writes a review
sheet for a human to fill in; ``summarize_sheet`` turns a completed sheet into those numbers.
No quality claim is made from model names or similarity scores alone."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil
from sqlalchemy import select

from app.core.io import atomic_write_json
from app.core.workspace import Workspace
from app.database.models import Image
from app.models.hub import ModelHub
from app.pipeline.base import ImgRef, StageContext
from app.pipeline.captioning.adapters import (SidecarAdapter, SmolVLMCaptionAdapter, TemplateAdapter, grounded_facts)
from app.pipeline.captioning.stage import word_count
from app.pipeline.vision.stages import load_embedding

SHEET_COLUMNS = ["image_id", "relative_path", "adapter", "short", "detailed",
                 "factual_accuracy_1to5", "hallucinated_claims_count", "salient_details_missing_count", "notes"]


def _peak_rss_mb() -> float:
    try:
        import resource  # not available on Windows

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    except ImportError:
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1024**2


def _sample(ws: Workspace, n: int, seed: int) -> list[ImgRef]:
    with ws.db.session() as s:
        imgs = list(s.scalars(select(Image).where(Image.valid.is_(True), Image.missing.is_(False),
                                                  Image.representative_status != "duplicate")))
        imgs.sort(key=lambda im: hashlib.sha256(f"{seed}:{im.sha256}".encode()).hexdigest())
        refs = []
        for im in imgs[:n]:
            deps = {r.stage: r.data for r in im.results if r.status == "done"}
            refs.append(ImgRef(pk=im.id, image_id=im.image_id, abs_path=im.abs_path, sha256=im.sha256, extension=im.extension,
                               file_name=im.file_name, relative_path=im.relative_path, width=im.width, height=im.height,
                               supplied_caption=im.supplied_caption, deps=deps))
        return refs


def run_benchmark(ws: Workspace, adapters: list[str], sample: int = 20, seed: int = 42) -> dict:
    cfg = ws.config
    hub = ModelHub(cfg, ws.root)
    ctx = StageContext(ws=ws, cfg=cfg, hub=hub)
    refs = _sample(ws, sample, seed)
    if not refs:
        raise ValueError("no valid images to benchmark; run the pipeline (at least scan and validate) first")
    out_dir = ws.root / "benchmarks" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    clip = hub.get(cfg.models.embedding) if hub.status(cfg.models.embedding)["usable"] else None
    results, rows = {}, []
    for name in adapters:
        entry: dict = {"adapter": name}
        try:
            if name == "template":
                adapter, entry["model"] = TemplateAdapter(), None
            elif name == "sidecar":
                adapter, entry["model"] = SidecarAdapter(), None
            elif name in hub.registry and hub.registry[name].adapter == "smolvlm":
                st = hub.status(name)
                if not st["usable"]:
                    raise RuntimeError(st["reason"])
                rss0 = psutil.Process().memory_info().rss
                adapter = SmolVLMCaptionAdapter(hub.get(name))
                entry["model"] = hub.info(name)
                entry["load_rss_delta_mb"] = round((psutil.Process().memory_info().rss - rss0) / 1024**2, 1)
            else:
                raise RuntimeError(f"'{name}' is not a caption adapter or a captioning model in the registry")
        except Exception as exc:
            results[name] = {**entry, "skipped": True, "reason": str(exc)}
            continue
        lat, failures, caps, aligns = [], 0, [], []
        for ref in refs:
            t0 = time.perf_counter()
            try:
                cap = adapter.generate(ctx, ref, grounded_facts(ref), cfg.captioning)
            except Exception as exc:
                failures += 1
                rows.append({"image_id": ref.image_id, "relative_path": ref.relative_path, "adapter": name,
                             "short": "", "detailed": "", "notes": f"FAILED: {type(exc).__name__}"})
                continue
            lat.append(time.perf_counter() - t0)
            text = cap.get("detailed") or cap.get("short") or ""
            caps.append(text)
            if clip is not None and text and "embedding" in ref.deps:
                v = load_embedding(ctx, ref.sha256, ref.deps["embedding"]["model_key"])
                if v is not None:
                    aligns.append(float(clip.encode_texts([text])[0] @ v))
            rows.append({"image_id": ref.image_id, "relative_path": ref.relative_path, "adapter": name,
                         "short": cap.get("short", ""), "detailed": cap.get("detailed", "")})
        nonempty = [c for c in caps if c.strip()]
        openings = Counter(" ".join(c.lower().split()[:3]) for c in nonempty)
        lat_sorted = sorted(lat)
        entry.update({
            "images": len(refs), "failures": failures, "failure_rate": round(failures / len(refs), 4),
            "empty_captions": len(caps) - len(nonempty),
            "latency_s": {"mean": round(sum(lat) / len(lat), 3), "median": round(lat_sorted[len(lat) // 2], 3),
                          "max": round(lat_sorted[-1], 3)} if lat else None,
            "process_peak_rss_mb": round(_peak_rss_mb(), 1),
            "mean_words": round(sum(word_count(c) for c in nonempty) / len(nonempty), 1) if nonempty else None,
            "unique_caption_ratio": round(len(set(nonempty)) / len(nonempty), 3) if nonempty else None,
            "most_common_opening_share": round(openings.most_common(1)[0][1] / len(nonempty), 3) if nonempty else None,
            "mean_clip_cosine": round(sum(aligns) / len(aligns), 4) if aligns else None,
            "human_rated": {"factual_accuracy": None, "hallucination_rate": None, "salient_detail_coverage": None,
                            "status": "not measured - fill in review_sheet.csv and run `image-pipeline benchmark-summarize`"},
        })
        results[name] = entry
        if entry.get("model"):
            hub.unload(name)
    hub.unload()
    with open(out_dir / "review_sheet.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=SHEET_COLUMNS, restval="")
        w.writeheader()
        w.writerows(rows)
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "sample_size": len(refs), "seed": seed,
              "adapters": results,
              "notes": ["process_peak_rss_mb is the peak for the whole process so far, so it is cumulative across adapters "
                        "run in one command; benchmark one adapter per run for a clean memory figure.",
                        "mean_clip_cosine is a similarity signal, not a measure of caption correctness.",
                        "Accuracy, hallucination rate and detail coverage require the human review sheet."]}
    atomic_write_json(out_dir / "results.json", report)
    return {"results": str(out_dir / "results.json"), "review_sheet": str(out_dir / "review_sheet.csv"), "report": report}


def summarize_sheet(path: Path) -> dict:
    """Aggregate a human-completed review sheet. Unrated rows are counted, never scored."""
    per: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            a = per.setdefault(row["adapter"], {"rows": 0, "rated": 0, "acc": [], "hall": [], "miss": []})
            a["rows"] += 1

            def num(key) -> Optional[float]:
                v = (row.get(key) or "").strip()
                try:
                    return float(v) if v else None
                except ValueError:
                    return None

            acc, hall, miss = num("factual_accuracy_1to5"), num("hallucinated_claims_count"), num("salient_details_missing_count")
            if acc is None and hall is None and miss is None:
                continue
            a["rated"] += 1
            if acc is not None:
                if not 1 <= acc <= 5:
                    raise ValueError(f"factual_accuracy_1to5 must be 1-5 (image {row['image_id']})")
                a["acc"].append(acc)
            if hall is not None:
                a["hall"].append(hall)
            if miss is not None:
                a["miss"].append(miss)
    out = {}
    for name, a in per.items():
        out[name] = {
            "captions": a["rows"], "rated": a["rated"], "unrated": a["rows"] - a["rated"],
            "mean_factual_accuracy_1to5": round(sum(a["acc"]) / len(a["acc"]), 2) if a["acc"] else None,
            "hallucination_rate": round(sum(1 for h in a["hall"] if h > 0) / len(a["hall"]), 3) if a["hall"] else None,
            "mean_hallucinated_claims": round(sum(a["hall"]) / len(a["hall"]), 2) if a["hall"] else None,
            "mean_salient_details_missing": round(sum(a["miss"]) / len(a["miss"]), 2) if a["miss"] else None,
        }
    return {"adapters": out, "note": "Computed only from rows a reviewer filled in."}
