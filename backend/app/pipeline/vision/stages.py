"""Vision stages: embeddings (14), zero-shot annotation (22, 26, 29, 32, 35, 36, 37),
people/face presence (38), OCR (27), segmentation (31), object detection + relationships (30, 33).

Everything predicted here is stored with ``status: "inferred"`` and an uncalibrated confidence."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from sqlalchemy import select

from app.config.settings import fingerprint
from app.core.imageio import analysis_rgb
from app.database.models import Embedding
from app.pipeline.base import ImageStage, ImgRef, ModelUnavailable, StageContext
from app.pipeline.quality.technical import spectral_saliency

from . import labels as L


def _toggle(value, auto_default: bool) -> bool:
    return auto_default if value == "auto" else bool(value)


def load_embedding(ctx: StageContext, sha256: str, model_key: str):
    with ctx.ws.db.session() as s:
        row = s.scalar(select(Embedding).where(Embedding.sha256 == sha256, Embedding.model_key == model_key))
        return None if row is None else np.frombuffer(row.vector, dtype=np.float32).copy()


def release_model(ctx: StageContext, stage, attribute: str, key) -> None:
    """Free a stage's model when the stage is done. Both references have to go (the stage's own and
    the hub's) or the weights stay in memory: on an 8 GB machine three models held at once pushed the
    system into swap and made the whole computer sluggish. The next stage that needs the model loads
    it again, which takes a few seconds."""
    from app.pipeline.base import _free_memory

    setattr(stage, attribute, None)
    if key:
        ctx.hub.unload(key)
    else:
        _free_memory()


def worth_annotating(ctx: StageContext, img) -> bool:
    """False for images that are already certain to be rejected for their size: the slow optional
    annotators (text reading, object detection) are not spent on them."""
    q = ctx.cfg.quality
    return img.manual_decision == "accepted" or not (img.width and img.height and (img.width < q.min_width or img.height < q.min_height))


# ------------------------------------------------------------------ embeddings
class EmbeddingStage(ImageStage):
    name = "embedding"
    label = "EMBED"

    def describe(self, d):
        return f"embedding {'reused from cache' if d.get('cached') else str(d.get('dim', '')) + '-d'}"

    version = "1"
    title = "Image embeddings"
    processes = "14"
    depends = ("validate",)
    batched = True

    def enabled(self, ctx):
        key = ctx.cfg.models.embedding
        if not _toggle(ctx.cfg.stages.embeddings, True):
            return False, "disabled in configuration"
        st = ctx.hub.status(key)
        return st["usable"], st.get("reason", "")

    def model_info(self, ctx):
        return ctx.hub.info(ctx.cfg.models.embedding)

    def batch_size(self, ctx):
        return ctx.hub.batch_size(ctx.cfg.models.embedding, per_item_mb=40)

    def setup(self, ctx):
        self.clip = ctx.hub.get(ctx.cfg.models.embedding)
        self.model_key = f"{self.clip.spec.key}@{self.clip.revision}"

    def teardown(self, ctx):
        release_model(ctx, self, "clip", ctx.cfg.models.embedding)

    def process_batch(self, ctx, refs):
        out: list = [None] * len(refs)
        with ctx.ws.db.session() as s:
            have = set(s.scalars(select(Embedding.sha256).where(
                Embedding.model_key == self.model_key, Embedding.sha256.in_([r.sha256 for r in refs]))))
        images, idx = [], []
        for i, r in enumerate(refs):
            if r.sha256 in have:
                out[i] = {"model_key": self.model_key, "cached": True}
                continue
            try:
                images.append(analysis_rgb(r.abs_path, 448))
                idx.append(i)
            except Exception as exc:
                out[i] = exc
        if images:
            vecs = self.clip.encode_images(images)  # may raise OOM -> handled by the runner
            with ctx.ws.db.session() as s:
                seen = set()
                for i, v in zip(idx, vecs):
                    sha = refs[i].sha256
                    if sha not in seen:  # byte-identical files share one embedding
                        s.add(Embedding(sha256=sha, model_key=self.model_key, dim=int(v.shape[0]), vector=v.tobytes()))
                        seen.add(sha)
                    out[i] = {"model_key": self.model_key, "dim": int(v.shape[0]), "cached": False}
        return out

    def still_valid(self, ctx, img, data):
        return load_embedding(ctx, img.sha256, data.get("model_key", "")) is not None


# ------------------------------------------------------------------ zero-shot annotation
def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


class ZeroShotStage(ImageStage):
    name = "zero_shot"
    label = "TAGS"

    def describe(self, d):
        return (f"{d['categories']['labels'][0]['label']} / {d['style']['labels'][0]['label']} / "
                f"{d['lighting']['labels'][0]['label']}, aesthetic {d['aesthetic']['score']:.0f}")

    version = "1"
    title = "Zero-shot categories, style, lighting, scene, aesthetic and watermark signals"
    processes = "22, 26, 29, 32, 35, 36, 37"
    depends = ("embedding",)
    batched = True  # runs on the main thread: it shares the text encoder

    def enabled(self, ctx):
        if not _toggle(ctx.cfg.stages.zero_shot, True):
            return False, "disabled in configuration"
        if not _toggle(ctx.cfg.stages.embeddings, True):
            return False, "requires the embedding stage, which is disabled"
        st = ctx.hub.status(ctx.cfg.models.embedding)
        return st["usable"], st.get("reason", "") or "requires the embedding model"

    def fingerprint(self, cfg):
        return fingerprint([L.LABEL_SETS, L.AESTHETIC_PAIRS, L.WATERMARK_PROMPTS])

    def model_info(self, ctx):
        return ctx.hub.info(ctx.cfg.models.embedding)

    def batch_size(self, ctx):
        return 64

    def setup(self, ctx):
        self.clip = ctx.hub.get(ctx.cfg.models.embedding)
        self.text = {name: self.clip.encode_texts([p for _, p in items]) for name, items in L.LABEL_SETS.items()}
        self.aes = [(self.clip.encode_texts([a])[0], self.clip.encode_texts([b])[0]) for a, b in L.AESTHETIC_PAIRS]
        self.wm = self.clip.encode_texts(L.WATERMARK_PROMPTS)

    def teardown(self, ctx):
        release_model(ctx, self, "clip", ctx.cfg.models.embedding)

    def process(self, ctx, ref: ImgRef) -> dict:
        v = load_embedding(ctx, ref.sha256, ref.deps["embedding"]["model_key"])
        if v is None:
            raise ValueError("embedding missing from cache")
        scale = self.clip.logit_scale
        out: dict = {"status": "inferred", "confidence_note": "softmax over a fixed prompt list; not calibrated"}
        for name, items in L.LABEL_SETS.items():
            probs = _softmax(scale * (self.text[name] @ v))
            order = np.argsort(-probs)
            ranked = [{"label": items[i][0], "confidence": round(float(probs[i]), 4)} for i in order[:5]]
            keep = [r for r in ranked if r["confidence"] >= L.MULTI_LABEL_MIN.get(name, 0.2)][: L.MAX_LABELS.get(name, 1)]
            out[name] = {"labels": keep or ranked[:1], "top": ranked, "uncertain": ranked[0]["confidence"] < 0.35}
        pair_scores = [float(_softmax(scale * np.array([float(a @ v), float(b @ v)]))[0]) for a, b in self.aes]
        out["aesthetic"] = {"score": round(100 * float(np.mean(pair_scores)), 1),
                            "method": "clip_prompt_pairs_v1 (ranking signal, not an objective measure)"}
        wm = _softmax(scale * (self.wm @ v))
        p_wm = float(wm[0] + wm[1])
        out["watermark"] = {"probability": round(p_wm, 4), "likely": p_wm >= 0.65,
                            "method": "clip_zero_shot_prompts (uncalibrated)"}
        return out

    def apply(self, img, data):
        img.category = data["categories"]["labels"][0]["label"]
        img.style = data["style"]["labels"][0]["label"]
        img.aesthetic_score = data["aesthetic"]["score"]


# ------------------------------------------------------------------ people / faces
class PeopleStage(ImageStage):
    """Counts frontal faces only. No identity, age, gender, ethnicity or emotion is inferred."""

    name = "people"
    label = "FACES"

    def describe(self, d):
        return f"{d['face_count']} face(s)"

    version = "1"
    title = "Face presence (count only)"
    processes = "38"
    depends = ("validate",)

    def enabled(self, ctx):
        return _toggle(ctx.cfg.stages.people, True), "disabled in configuration"

    def setup(self, ctx):
        self.cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")

    def process(self, ctx, ref):
        img = analysis_rgb(ref.abs_path, 640)
        gray = cv2.equalizeHist(np.asarray(img.convert("L")))
        cascade = cv2.CascadeClassifier(self.cascade_path)  # per call: the classifier is not thread-safe
        m = max(24, min(gray.shape) // 12)
        boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(m, m))
        h, w = gray.shape
        faces = [{"bbox": [round(x / w, 4), round(y / h, 4), round(bw / w, 4), round(bh / h, 4)]} for x, y, bw, bh in boxes]
        return {"face_count": len(faces), "faces": faces, "status": "inferred",
                "method": "opencv_haar_frontalface (can miss faces and produce false positives)",
                "policy": "presence and count only; identity and sensitive traits are never inferred"}


# ------------------------------------------------------------------ OCR
def ocr_available() -> bool:
    return importlib.util.find_spec("rapidocr_onnxruntime") is not None


class OcrStage(ImageStage):
    name = "ocr"
    label = "OCR"

    def describe(self, d):
        return f"{len(d['regions'])} text region(s)" + (f": {' | '.join(d['text'])[:60]}" if d["text"] else "")

    version = "1"
    title = "Visible text (OCR)"
    processes = "27"
    depends = ("validate",)
    batched = True  # the ONNX session is used from one thread

    def enabled(self, ctx):
        if not _toggle(ctx.cfg.stages.ocr, ctx.cfg.project.mode != "text_to_image"):
            return False, "disabled in configuration (optional stage)"
        if ctx.cfg.models.ocr == "none":
            return False, "no OCR engine configured"
        return ocr_available(), "rapidocr-onnxruntime is not installed (pip install '.[ocr]')"

    def model_info(self, ctx):
        import importlib.metadata as md

        return {"name": "rapidocr-onnxruntime", "version": md.version("rapidocr-onnxruntime")}

    def eligible(self, ctx, img):
        return worth_annotating(ctx, img)

    def setup(self, ctx):
        from rapidocr_onnxruntime import RapidOCR

        self.engine = RapidOCR()

    def teardown(self, ctx):
        release_model(ctx, self, "engine", None)

    def process(self, ctx, ref):
        img = analysis_rgb(ref.abs_path, 1280)
        w, h = img.size
        result, _ = self.engine(np.asarray(img)[:, :, ::-1])
        regions, area = [], 0.0
        for box, text, conf in result or []:
            xs, ys = [p[0] for p in box], [p[1] for p in box]
            x0, y0 = min(max(min(xs) / w, 0.0), 1.0), min(max(min(ys) / h, 0.0), 1.0)
            x1, y1 = min(max(max(xs) / w, 0.0), 1.0), min(max(max(ys) / h, 0.0), 1.0)
            area += max(0.0, x1 - x0) * max(0.0, y1 - y0)
            regions.append({"text": text, "confidence": round(min(max(float(conf), 0.0), 1.0), 4),
                            "bbox": [round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)]})
        confident = [r for r in regions if r["confidence"] >= 0.6]
        low_conf = len(regions) - len(confident)
        flags = []
        frac = min(1.0, area)
        if frac > 0.25:
            flags.append("text_heavy")
        if regions and low_conf / len(regions) > 0.5:
            flags.append("possibly_malformed_text")  # many low-confidence regions: garbled or stylised lettering
        return {"text": [r["text"] for r in confident], "regions": regions, "text_area_fraction": round(frac, 4),
                "flags": flags, "status": "inferred"}


# ------------------------------------------------------------------ segmentation
class SegmentationStage(ImageStage):
    """Classical subject/background split: GrabCut seeded by spectral saliency. Masks are
    written to the cache and exported with accepted images."""

    name = "segmentation"
    label = "SEGMENT"

    def describe(self, d):
        return f"mask, {d['foreground_fraction']:.0%} foreground" if d.get("available") else f"no mask ({d.get('reason')})"

    version = "2"
    title = "Subject/background segmentation"
    processes = "31"
    depends = ("validate",)

    def enabled(self, ctx):
        if not _toggle(ctx.cfg.stages.segmentation, ctx.cfg.project.mode != "text_to_image"):
            return False, "disabled in configuration (optional stage)"
        return ctx.cfg.models.segmentation != "none", "no segmentation method configured"

    def mask_path(self, ctx, sha256: str) -> Path:
        return ctx.ws.cache_dir / "masks" / sha256[:2] / f"{sha256}.png"

    def process(self, ctx, ref):
        img = analysis_rgb(ref.abs_path, 384)
        rgb = np.asarray(img)
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        sal = cv2.resize(spectral_saliency(gray), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = np.full(sal.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        mask[sal > sal.mean() + 0.5 * sal.std()] = cv2.GC_PR_FGD
        mask[sal > np.percentile(sal, 97)] = cv2.GC_FGD
        mask[sal <= np.percentile(sal, 20)] = cv2.GC_BGD  # definite background seeds
        if not (mask == cv2.GC_FGD).any() or (mask == cv2.GC_PR_FGD).mean() > 0.95:
            return {"available": False, "reason": "no distinct salient region", "status": "inferred"}
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
        fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
        ys, xs = np.nonzero(fg)
        fraction = float((fg > 0).mean())
        if len(xs) == 0 or fraction < 0.005 or fraction > 0.95:
            return {"available": False, "reason": "no separable subject (degenerate mask)", "status": "inferred"}
        h, w = fg.shape
        out = self.mask_path(ctx, ref.sha256)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.png")
        PILImage.fromarray(fg, "L").save(tmp)
        tmp.replace(out)
        return {"available": True, "status": "inferred", "method": "grabcut_seeded_by_spectral_saliency",
                "mask_size": [w, h], "foreground_fraction": round(float((fg > 0).mean()), 4),
                "bbox": [round(xs.min() / w, 4), round(ys.min() / h, 4),
                         round((xs.max() - xs.min() + 1) / w, 4), round((ys.max() - ys.min() + 1) / h, 4)]}

    def still_valid(self, ctx, img, data):
        return not data.get("available") or self.mask_path(ctx, img.sha256).exists()


# ------------------------------------------------------------------ object detection
def spatial_relationships(objects: list[dict], limit: int = 12) -> list[dict]:
    """Purely geometric relations between detected boxes (grounded in the detections)."""
    rels = []
    objs = sorted(objects, key=lambda o: -o["confidence"])[:8]
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            ax, ay, aw, ah = a["bbox"]
            bx, by, bw, bh = b["bbox"]
            acx, acy, bcx, bcy = ax + aw / 2, ay + ah / 2, bx + bw / 2, by + bh / 2
            ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
            iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
            inter = ix * iy
            if inter > 0.9 * min(aw * ah, bw * bh):
                small, big = (a, b) if aw * ah < bw * bh else (b, a)
                rel = (small["label"], "inside or in front of", big["label"])
            elif inter > 0:
                rel = (a["label"], "overlapping", b["label"])
            elif abs(acx - bcx) >= abs(acy - bcy):
                rel = (a["label"], "left of" if acx < bcx else "right of", b["label"])
            else:
                rel = (a["label"], "above" if acy < bcy else "below", b["label"])
            rels.append({"subject": rel[0], "relation": rel[1], "object": rel[2], "status": "inferred",
                         "basis": "bounding_box_geometry",
                         "confidence": round(min(a["confidence"], b["confidence"]), 4)})
    return rels[:limit]


class ObjectDetectionStage(ImageStage):
    name = "object_detection"
    label = "OBJECTS"

    def describe(self, d):
        return f"{len(d['objects'])} object(s), {len(d['relationships'])} relation(s)"

    version = "1"
    title = "Object detection and spatial relationships"
    processes = "30, 33"
    depends = ("validate",)
    batched = True

    def enabled(self, ctx):
        if not _toggle(ctx.cfg.stages.object_detection, ctx.cfg.project.mode != "text_to_image"):
            return False, "disabled in configuration (optional stage)"
        st = ctx.hub.status(ctx.cfg.models.object_detector)
        return st["usable"], st.get("reason", "")

    def fingerprint(self, cfg):
        return cfg.models.detection_labels or L.DETECTION_LABELS

    def eligible(self, ctx, img):
        return worth_annotating(ctx, img)

    def model_info(self, ctx):
        return ctx.hub.info(ctx.cfg.models.object_detector)

    def batch_size(self, ctx):
        return 1

    def chunk_size(self, ctx):
        return 40  # about a minute of detection per checkpoint

    def setup(self, ctx):
        self.det = ctx.hub.get(ctx.cfg.models.object_detector)
        self.labels = ctx.cfg.models.detection_labels or L.DETECTION_LABELS

    def process(self, ctx, ref):
        if not hasattr(self.det, "detect"):
            raise ModelUnavailable("configured object_detector is not a detection model")
        objects = self.det.detect(analysis_rgb(ref.abs_path, 960), self.labels)
        return {"objects": objects, "relationships": spatial_relationships(objects), "status": "inferred"}

    def teardown(self, ctx):
        release_model(ctx, self, "det", ctx.cfg.models.object_detector)
