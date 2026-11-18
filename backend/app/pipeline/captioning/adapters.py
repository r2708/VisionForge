"""Caption adapters (processes 39-44). Each adapter returns the same fields, kept strictly
separate: short / detailed / comprehensive descriptions and a generation prompt.

* ``template``  - grounded baseline assembled only from stored annotations (no model).
* ``sidecar``   - captions supplied by the user (text file next to the image or metadata file).
* ``smolvlm``   - local vision-language model through the model hub.
"""
from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Optional

from app.core.imageio import analysis_rgb
from app.pipeline.base import ImgRef, StageContext

TEMPLATE_VERSION = "template-1"
PROMPT_VERSION = "smolvlm-prompts-1"

MEDIUM = {
    "photorealistic": "A photograph of", "cinematic": "A photograph of", "black_and_white_photo": "A black-and-white photograph of",
    "digital_illustration": "A digital illustration of", "anime": "An anime-style drawing of", "3d_render": "A 3D render of",
    "oil_painting": "An oil painting of", "watercolor": "A watercolor painting of", "sketch": "A sketch of",
    "graphic_design": "A flat graphic of", "pixel_art": "Pixel art of",
}
STYLE_WORD = {
    "photorealistic": "photograph", "cinematic": "photograph", "black_and_white_photo": "black-and-white photograph",
    "digital_illustration": "digital illustration", "anime": "anime style", "3d_render": "3D render",
    "oil_painting": "oil painting", "watercolor": "watercolor painting", "sketch": "pencil sketch",
    "graphic_design": "flat graphic design", "pixel_art": "pixel art",
}
SUBJECT = {
    "landscape": "a natural landscape", "architecture": "a building", "cityscape": "a city scene", "interior": "an indoor room",
    "wildlife": "a wild animal", "pet": "a pet animal", "vehicle": "a vehicle", "food": "food", "portrait": "a person",
    "people": "a group of people", "product": "an object on a plain background", "plant": "a plant", "sports": "a sports scene",
    "illustration": "an illustrated scene", "abstract": "an abstract pattern", "document": "a document or screenshot",
    "graphic": "a graphic design", "aerial": "an aerial view", "night_sky": "a night sky", "underwater": "an underwater scene",
}
LIGHTING = {
    "daylight": "in bright daylight", "golden_hour": "in warm, low sunlight", "backlit": "backlit, with the subject in silhouette",
    "studio_lighting": "under even studio lighting", "low_light": "in low light", "overcast": "under soft, diffuse light",
    "indoor_artificial": "under indoor artificial light", "neon": "lit by colored neon light",
}
SCENE = {"indoor": "indoors", "outdoor_urban": "in an urban outdoor setting", "outdoor_natural": "outdoors in a natural setting",
         "studio_backdrop": "against a plain backdrop"}
VIEW = {"eye_level": "seen at eye level", "low_angle": "seen from a low angle", "high_angle": "seen from a high angle",
        "top_down": "seen from directly above", "close_up": "in a close-up view", "wide_shot": "in a wide shot"}

_BOILERPLATE = re.compile(
    r"^\s*(the|this)\s+(image|picture|photo|photograph)\s+(shows|depicts|features|displays|captures|is of|presents)\s+"
    r"|^\s*in\s+this\s+(image|picture|photo),?\s+(we\s+(can\s+)?see\s+|there\s+(is|are)\s+)?"
    r"|^\s*this\s+is\s+(an?\s+)?(image|picture|photo|photograph)\s+of\s+", re.I)


def clean_caption(text: str) -> str:
    """Normalise whitespace, drop empty openers and a trailing unfinished sentence."""
    text = " ".join((text or "").split()).strip().strip('"')
    text = _BOILERPLATE.sub("", text).strip()
    if not text:
        return ""
    if text[-1] not in ".!?":
        cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
        text = text[: cut + 1] if cut > 40 else text.rstrip(",;:") + "."
    return text[0].upper() + text[1:]


def _join(words: list[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def _top(zs: Optional[dict], name: str, min_conf: float = 0.35) -> Optional[str]:
    if not zs or name not in zs:
        return None
    best = zs[name]["labels"][0]
    return best["label"] if best["confidence"] >= min_conf else None


def grounded_facts(ref: ImgRef) -> dict:
    """Facts available for captions and tags, each traceable to a stored annotation."""
    zs, q = ref.deps.get("zero_shot"), ref.deps.get("quality_technical") or {}
    meta = ref.deps.get("validate") or {}
    det, ocr, ppl = ref.deps.get("object_detection"), ref.deps.get("ocr"), ref.deps.get("people")
    colors = (q.get("color") or {}).get("dominant_color_names", [])[:3]
    objects: list[str] = []
    for o in sorted((det or {}).get("objects", []), key=lambda o: -o["confidence"]):
        if o["confidence"] >= 0.35 and o["label"] not in objects:
            objects.append(o["label"])
    return {
        "category": _top(zs, "categories"), "style": _top(zs, "style"), "lighting": _top(zs, "lighting"),
        "scene": _top(zs, "scene"), "viewpoint": _top(zs, "viewpoint", 0.45), "dof": _top(zs, "depth_of_field", 0.75),
        "colors": colors, "grayscale": bool((q.get("color") or {}).get("is_grayscale")),
        "orientation": meta.get("orientation"), "placement": (q.get("composition") or {}).get("subject_placement"),
        "objects": objects[:6], "text": [t for t in (ocr or {}).get("text", []) if len(t.strip()) > 1][:3],
        "faces": (ppl or {}).get("face_count", 0),
    }


def build_tags(f: dict) -> list[str]:
    tags = []
    for key in ("category", "style", "lighting", "scene", "viewpoint"):
        if f.get(key) and f[key] not in ("flat_no_lighting", "no_setting"):
            tags.append(f[key].replace("_", " "))
    if f.get("dof"):
        tags.append(f"{f['dof']} depth of field")
    tags += [f"{c} tones" for c in f["colors"]]
    if f["grayscale"]:
        tags.append("monochrome")
    if f.get("orientation"):
        tags.append(f"{f['orientation']} orientation")
    tags += f["objects"]
    if f["text"]:
        tags.append("visible text")
    if f["faces"]:
        tags.append("face visible")
    seen, out = set(), []
    for t in tags:
        t = t.lower().strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


class TemplateAdapter:
    name = "template"
    source = "template_baseline"

    def version(self) -> dict:
        return {"template_version": TEMPLATE_VERSION}

    def generate(self, ctx: StageContext, ref: ImgRef, facts: dict, cfg) -> dict:
        f = facts
        variant = int(hashlib.sha256(ref.sha256.encode()).hexdigest()[:6], 16)
        subject = SUBJECT.get(f["category"])
        medium = MEDIUM.get(f["style"], "An image of") if subject else None
        color_phrase = f"mostly {_join(f['colors'])} tones" if f["colors"] and not f["grayscale"] else None
        light, scene, view = LIGHTING.get(f["lighting"]), SCENE.get(f["scene"]), VIEW.get(f["viewpoint"])

        if subject:
            shorts = [f"{medium} {subject}.", f"{medium} {subject}{' ' + scene if scene else ''}.",
                      f"{subject[0].upper() + subject[1:]}{' ' + light if light else ''}, shown as "
                      f"{_article(STYLE_WORD.get(f['style'], 'image'))}."]
            short = shorts[variant % len(shorts)]
        elif color_phrase:
            short = f"An image in {color_phrase}."
        else:
            short = "A monochrome image." if f["grayscale"] else "An image."

        parts = [short]
        s2 = [p for p in (scene if scene and scene not in short else None, light if light and light not in short else None) if p]
        if s2:
            parts.append(("The scene is " if variant % 2 else "It is shown ") + _join(s2) + ".")
        if color_phrase and color_phrase not in short:
            parts.append(("Colors are " if variant % 3 else "The palette is ") + color_phrase + ".")
        elif f["grayscale"] and "black-and-white" not in short:
            parts.append("The image is monochrome.")
        detailed = " ".join(parts)

        more = list(parts)
        if f["placement"] and f["orientation"]:
            more.append(f"The most visually prominent area is {_placement(f['placement'])} of a {f['orientation']}-format frame.")
        if view:
            more.append(f"The view is {view.replace('seen ', '').replace('in a ', 'a ')}.")
        if f["dof"] == "shallow":
            more.append("The background appears out of focus.")
        if f["objects"]:
            more.append(f"Detected objects include {_join(f['objects'])}.")
        if f["faces"] and f["category"] in ("portrait", "people"):
            more.append("One face is visible." if f["faces"] == 1 else f"{f['faces']} faces are visible.")
        if f["text"]:
            more.append("Visible text reads " + "; ".join(f'"{t}"' for t in f["text"]) + ".")
        comprehensive = " ".join(more)

        prompt_bits = [subject, scene, light, f"{_join(f['colors'])} color palette" if color_phrase else None,
                       "monochrome" if f["grayscale"] else None, STYLE_WORD.get(f["style"]),
                       view.replace("seen ", "") if view else None,
                       f"{f['orientation']} orientation" if f["orientation"] else None]
        generation_prompt = ", ".join(b for b in prompt_bits if b)

        alternatives = []
        if subject and cfg.alternate_caption_count:
            pool = [f"{medium} {subject}{', ' + light if light else ''}.",
                    f"{subject[0].upper() + subject[1:]}{' ' + scene if scene else ''}"
                    f"{' in ' + color_phrase if color_phrase else ''}.",
                    f"{medium} {subject}{' ' + view if view else ''}."]
            alternatives = [a for a in dict.fromkeys(pool) if a != short][: cfg.alternate_caption_count]
        return {"short": short, "detailed": detailed, "comprehensive": comprehensive,
                "generation_prompt": generation_prompt, "alternatives": alternatives,
                "prompt_styling_additions": []}


def _article(noun: str) -> str:
    return ("an " if noun[0].lower() in "aeiou" else "a ") + noun


def _placement(p: str) -> str:
    return "the center" if p == "centered" else f"the {p} part"


class SidecarAdapter:
    name = "sidecar"
    source = "user_supplied"

    def version(self) -> dict:
        return {"template_version": "sidecar-1"}

    @staticmethod
    def sidecar_path(ref: ImgRef, ext: str) -> Path:
        return Path(ref.abs_path).with_suffix(ext)

    def read(self, ref: ImgRef, cfg) -> tuple[Optional[str], Optional[str]]:
        p = self.sidecar_path(ref, cfg.sidecar_extension)
        if p.is_file():
            raw = p.read_bytes()[:20000]
            return raw.decode("utf-8", errors="replace").strip(), hashlib.sha256(raw).hexdigest()
        if ref.supplied_caption:
            return ref.supplied_caption.strip(), hashlib.sha256(ref.supplied_caption.encode()).hexdigest()
        return None, None

    def generate(self, ctx, ref, facts, cfg) -> dict:
        text, digest = self.read(ref, cfg)
        if not text:
            return {"short": "", "detailed": "", "comprehensive": "", "generation_prompt": "", "alternatives": [],
                    "missing": True, "sidecar_sha256": None, "prompt_styling_additions": []}
        text = " ".join(text.split())
        first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
        return {"short": first, "detailed": text, "comprehensive": text, "generation_prompt": text,
                "alternatives": [], "sidecar_sha256": digest, "prompt_styling_additions": []}


class SmolVLMCaptionAdapter:
    name = "smolvlm"
    source = "vlm"
    PROMPTS = {
        "short": "Describe the main subject of this image in one short sentence. Mention only what is clearly visible.",
        "detailed": ("Describe this image in two or three sentences: the main subject and what it is doing, the setting, "
                     "the lighting and the main colors. Mention only what is clearly visible."),
        "comprehensive": ("Write a thorough description of everything visible in this image: the subjects and where they are, "
                          "the background, colors, lighting and any visible text. Do not guess names, places, brands or "
                          "anything that cannot be seen."),
        "alternate": "Describe this image in one sentence using different wording from this description: \"{short}\"",
    }

    def __init__(self, vlm):
        self.vlm = vlm

    def version(self) -> dict:
        return {"prompt_version": PROMPT_VERSION}

    def generate(self, ctx, ref, facts, cfg) -> dict:
        image = analysis_rgb(ref.abs_path, 768)
        out = {"short": "", "detailed": "", "comprehensive": "", "alternatives": []}
        if cfg.short_caption:
            out["short"] = clean_caption(self.vlm.ask(image, self.PROMPTS["short"], 48))
        if cfg.detailed_caption:
            out["detailed"] = clean_caption(self.vlm.ask(image, self.PROMPTS["detailed"], min(cfg.max_new_tokens, 120)))
        if cfg.comprehensive_caption:
            out["comprehensive"] = clean_caption(self.vlm.ask(image, self.PROMPTS["comprehensive"], cfg.max_new_tokens))
        for _ in range(cfg.alternate_caption_count if out["short"] else 0):
            alt = clean_caption(self.vlm.ask(image, self.PROMPTS["alternate"].format(short=out["short"]), 48))
            if alt and alt not in out["alternatives"] and alt != out["short"]:
                out["alternatives"].append(alt)
        # The prompt reuses the model's own description plus annotation-derived terms; nothing is invented.
        base = (out["detailed"] or out["short"]).rstrip(".")
        extras = [x for x in (LIGHTING.get(facts["lighting"]), STYLE_WORD.get(facts["style"])) if x and x.lower() not in base.lower()]
        out["generation_prompt"] = ", ".join([base] + extras) if base else ""
        out["prompt_styling_additions"] = []
        return out


def _sentences(text: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if x.strip()]


def summary_of(text: str, max_words: int = 75) -> str:
    """The opening sentences of a long description, up to about ``max_words`` words; never cut mid-sentence."""
    out, n = [], 0
    for sentence in _sentences(text):
        words = len(sentence.split())
        if out and n + words > max_words:
            break
        out.append(sentence)
        n += words
    return " ".join(out)


_HEADER_WORD = r"(?:detailed|descriptive|description|caption|image|screenshot|cleaned-up|cleaned|refined|final|for|the|of|and|a)"
_LEAD_IN = re.compile(
    r"^\s*(?:(?:sure|certainly|okay)[,!.]?\s*)?"
    r"(?:here(?:'s|\u2019s| is| are)\b[^\n:]{0,120}:"                         # "Here is a cleaned-up and detailed caption for the image:"
    rf"|(?:#+\s*|\*\*)?{_HEADER_WORD}(?:[ -]{_HEADER_WORD}){{0,6}}\s*:?\s*(?:\*\*)?\s*:?(?=\s))"   # "**Detailed Caption:**", "### Caption for the image"
    r"\s*", re.I)


def clean_supplied_caption(text: str) -> str:
    """Remove what a language model wrapped around a caption: an announcing first line, markdown
    emphasis, headings and rules. The description itself is left as written."""
    text = (text or "").strip()
    for _ in range(3):
        text = re.sub(r"^\s*-{3,}\s*", "", text)
        m = _LEAD_IN.match(text)
        if not m or not m.group(0).strip() or ":" not in m.group(0) and "#" not in m.group(0) and "*" not in m.group(0):
            break
        text = text[m.end():]
    text = re.sub(r"(?m)^\s*(?:-{3,}|#+)\s*", "", text)
    text = text.replace("**", "").replace("__", "")
    return " ".join(text.split())


class AutoCaptionAdapter:
    """The caption that came with the source when there is one; the local vision-language model
    only for images without one. The model is loaded the first time it is actually needed, so a
    batch where every image already has a caption never touches it."""

    name = "auto"
    source = "vlm"
    VLM_PROMPT = SmolVLMCaptionAdapter.PROMPTS["detailed"]

    def __init__(self, ctx: StageContext):
        self.supplied = SidecarAdapter()
        self.vlm = None
        self.model_key = ctx.cfg.models.captioner
        self.from_source = self.generated = 0
        self.vlm_seconds = 0.0

    def version(self) -> dict:
        return {"prompt_version": PROMPT_VERSION, "template_version": "auto-2"}

    def _model(self, ctx: StageContext):
        if self.vlm is None:
            ctx.progress("captioning", {"message": f"an image has no caption from the source: loading {self.model_key} "
                                                   "(used only for such images)"})
            started = time.time()
            self.vlm = ctx.hub.get(self.model_key)
            ctx.progress("captioning", {"message": f"{self.model_key} loaded in {time.time() - started:.1f}s"})
        return self.vlm

    def generate(self, ctx: StageContext, ref: ImgRef, facts: dict, cfg) -> dict:
        text, digest = self.supplied.read(ref, cfg)
        if text:
            self.from_source += 1
            text = clean_supplied_caption(text)
            sentences = _sentences(text)
            detailed = summary_of(text)
            return {"short": sentences[0] if sentences else text, "detailed": detailed, "comprehensive": text,
                    "generation_prompt": detailed, "alternatives": [], "sidecar_sha256": digest, "prompt_styling_additions": [],
                    "source": cfg.supplied_source, "caption_model": None}
        vlm = self._model(ctx)
        started = time.time()
        image = analysis_rgb(ref.abs_path, 768 if cfg.vlm_image_splitting else 512)
        detailed = clean_caption(vlm.ask(image, self.VLM_PROMPT, min(cfg.max_new_tokens, 120), split_image=cfg.vlm_image_splitting))
        self.generated += 1
        self.vlm_seconds += time.time() - started
        sentences = _sentences(detailed)
        extras = [x for x in (LIGHTING.get(facts["lighting"]), STYLE_WORD.get(facts["style"])) if x and x.lower() not in detailed.lower()]
        return {"short": sentences[0] if sentences else detailed, "detailed": detailed, "comprehensive": detailed,
                "generation_prompt": ", ".join([detailed.rstrip(".")] + extras) if detailed else "", "alternatives": [],
                "sidecar_sha256": None, "prompt_styling_additions": [], "source": "vlm", "caption_model": self.model_key}

    def summary(self) -> str:
        text = f"{self.from_source} caption(s) taken from the source, {self.generated} generated by {self.model_key}"
        if self.generated:
            text += f" ({self.vlm_seconds / self.generated:.1f}s per image)"
        return text
