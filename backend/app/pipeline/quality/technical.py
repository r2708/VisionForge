"""Processes 17-21 and 23-25: technical quality signals computed from pixels.

All metrics are classical, explainable heuristics measured on an oriented RGB copy whose
longer side is at most 1024 px (so values are comparable across resolutions). Every raw
measurement is stored next to its 0-100 score; none of them is a universal judgment.
The mapping constants below are documented defaults, exposed for inspection in the output."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np
from PIL import Image

from app.core.imageio import analysis_rgb, open_oriented, to_rgb
from app.pipeline.base import ImageStage, ImgRef, StageContext

NAMED_COLORS = {
    "black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128), "red": (220, 30, 30),
    "orange": (240, 140, 20), "yellow": (240, 220, 40), "green": (40, 160, 60), "teal": (20, 150, 150),
    "blue": (40, 90, 210), "purple": (130, 60, 180), "pink": (240, 130, 180), "brown": (120, 75, 40),
    "beige": (220, 200, 165),
}


def color_name(rgb) -> str:
    r, g, b = [c / 255 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if v < 0.16:
        return "black"
    if s < 0.12:
        return "white" if v > 0.85 else "gray"
    return min(NAMED_COLORS, key=lambda n: sum((a - b_) ** 2 for a, b_ in zip(NAMED_COLORS[n], rgb)))


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _lin(x: float, lo: float, hi: float) -> float:
    """0 at ``lo``, 1 at ``hi``."""
    return _clip01((x - lo) / (hi - lo))


# ---------------------------------------------------------------- sharpness
def sharpness(gray: np.ndarray) -> dict:
    gray = gray.astype(np.float64)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx, gy = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(gx ** 2 + gy ** 2))
    # Best 10% of 32-px tiles: an image with a sharp subject and blurred background stays "sharp".
    h, w = gray.shape
    tiles = []
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    for y in range(0, h - 31, 32):
        for x in range(0, w - 31, 32):
            tiles.append(lap[y:y + 32, x:x + 32].var())
    local_peak = float(np.percentile(tiles, 90)) if tiles else lap_var
    score = 100 * _lin(np.log10(max(local_peak, 1e-3)), np.log10(15.0), np.log10(600.0))
    return {"laplacian_variance": round(lap_var, 2), "tenengrad": round(tenengrad, 2),
            "local_peak_variance": round(local_peak, 2), "score": round(score, 1),
            "method": "laplacian_variance_p90_tiles_log_scaled[15..600]"}


# ---------------------------------------------------------------- noise
_NOISE_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def _sigma(channel: np.ndarray) -> float:
    """Immerkaer's fast noise estimate restricted to the smoothest half of the image."""
    channel = channel.astype(np.float64)
    resp = np.abs(cv2.filter2D(channel, -1, _NOISE_K))[1:-1, 1:-1]
    gx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)[1:-1, 1:-1]
    gy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)[1:-1, 1:-1]
    grad = np.abs(gx) + np.abs(gy)
    mask = grad <= np.median(grad)
    if mask.sum() < 64:
        mask = np.ones_like(grad, dtype=bool)
    return float(np.sqrt(np.pi / 2) * resp[mask].mean() / 6.0)


def noise(rgb: np.ndarray) -> dict:
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    luma, chroma = _sigma(ycc[..., 0]), (_sigma(ycc[..., 1]) + _sigma(ycc[..., 2])) / 2
    score = 100 * (1 - _lin(luma, 1.5, 14.0))
    return {"luma_sigma": round(luma, 3), "chroma_sigma": round(chroma, 3), "score": round(score, 1),
            "method": "immerkaer_smooth_regions[1.5..14]"}


# ---------------------------------------------------------------- compression
_JPEG_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                       14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                       49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


def jpeg_quality_estimate(path: str):
    """Approximate libjpeg quality from the luminance quantisation table (None for non-JPEG)."""
    try:
        with Image.open(path) as im:
            if im.format != "JPEG" or not getattr(im, "quantization", None):
                return None
            table = np.array(sorted(im.quantization.items())[0][1], dtype=np.float64)
    except Exception:
        return None
    if table.size != 64:
        return None
    # Zig-zag order does not matter for the mean ratio.
    scale = float(np.mean(np.sort(table) * 100.0 / np.sort(_JPEG_LUMA)))
    q = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(round(min(100, max(1, q))))


def blockiness(path: str) -> dict:
    """8x8 block-boundary discontinuity measured at native resolution on a centre crop."""
    img = to_rgb(open_oriented(path)).convert("L")
    w, h = img.size
    cw, ch = min(w, 1024) // 8 * 8, min(h, 1024) // 8 * 8
    if cw < 32 or ch < 32:
        return {"blockiness_ratio": None, "score": None}
    x0, y0 = ((w - cw) // 2) // 8 * 8, ((h - ch) // 2) // 8 * 8
    g = np.asarray(img.crop((x0, y0, x0 + cw, y0 + ch)), dtype=np.float64)
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    cols, rows = np.arange(dh.shape[1]), np.arange(dv.shape[0])
    edge_h, edge_v = dh[:, cols % 8 == 7], dv[rows % 8 == 7, :]
    inner_h, inner_v = dh[:, cols % 8 != 7], dv[rows % 8 != 7, :]
    inner = (inner_h.mean() + inner_v.mean()) / 2
    ratio = float(((edge_h.mean() + edge_v.mean()) / 2 + 0.25) / (inner + 0.25))
    return {"blockiness_ratio": round(ratio, 3), "score": round(100 * (1 - _lin(ratio, 1.10, 1.90)), 1)}


# ---------------------------------------------------------------- exposure / contrast
def exposure(gray: np.ndarray) -> dict:
    p1, p50, p99 = [float(v) for v in np.percentile(gray, [1, 50, 99])]
    mean, std = float(gray.mean()), float(gray.std())
    low, high = float((gray <= 4).mean()), float((gray >= 251).mean())
    flags = []
    if mean < 45 and p99 < 150:
        flags.append("underexposed")
    if mean > 215 and p1 > 110:
        flags.append("overexposed")
    if low > 0.30:
        flags.append("crushed_shadows")
    if high > 0.30:
        flags.append("clipped_highlights")
    penalty = 60 * _lin(abs(mean - 122) / 122, 0.55, 1.0) + 40 * _lin(max(low, high), 0.10, 0.60)
    exp_score = 100 - penalty
    dyn = p99 - p1
    con_score = 100 * (0.5 * _lin(std, 8, 50) + 0.5 * _lin(dyn, 40, 200))
    if dyn < 60:
        flags.append("low_contrast")
    return {"mean_luma": round(mean, 2), "median_luma": round(p50, 2), "std_luma": round(std, 2),
            "p1": round(p1, 1), "p99": round(p99, 1), "dynamic_range": round(dyn, 1),
            "shadow_clip_fraction": round(low, 4), "highlight_clip_fraction": round(high, 4),
            "exposure_score": round(exp_score, 1), "contrast_score": round(con_score, 1), "flags": flags}


# ---------------------------------------------------------------- colour
def color_stats(img: Image.Image) -> dict:
    rgb = np.asarray(img, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean() / 255)
    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
    a_mean, b_mean = float(lab[..., 1].mean() - 128), float(lab[..., 2].mean() - 128)
    cast = float(np.hypot(a_mean, b_mean))
    small = img.copy()
    small.thumbnail((160, 160))
    q = small.quantize(colors=6, method=Image.MEDIANCUT)
    pal = q.getpalette()[: 6 * 3]
    counts = sorted(q.getcolors(), reverse=True)
    total = sum(c for c, _ in counts)
    palette = []
    for c, idx in counts:
        col = tuple(pal[idx * 3: idx * 3 + 3])
        palette.append({"hex": "#%02x%02x%02x" % col, "name": color_name(col), "fraction": round(c / total, 3)})
    flags = []
    is_gray = colorfulness < 4.0
    if is_gray:
        flags.append("grayscale")
    elif cast > 28 and colorfulness < 45:
        flags.append("strong_color_cast")
    if sat > 0.85:
        flags.append("oversaturated")
    names: list[str] = []
    for p in palette:
        if p["fraction"] >= 0.10 and p["name"] not in names:
            names.append(p["name"])
    return {"colorfulness": round(colorfulness, 2), "mean_saturation": round(sat, 3), "is_grayscale": is_gray,
            "cast_magnitude": round(cast, 2), "palette": palette, "dominant_color_names": names[:4], "flags": flags}


# ---------------------------------------------------------------- saliency / composition
def spectral_saliency(gray: np.ndarray, size: int = 64) -> np.ndarray:
    """Spectral-residual saliency (Hou & Zhang, 2007); returns a [0,1] map of shape (size, size)."""
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-8)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(f)))) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), 2.5)
    sal -= sal.min()
    return sal / sal.max() if sal.max() > 0 else sal


def composition(gray: np.ndarray) -> dict:
    sal = spectral_saliency(gray)
    n = sal.shape[0]
    total = sal.sum() + 1e-9
    ys, xs = np.mgrid[0:n, 0:n]
    cx, cy = float((sal * xs).sum() / total / (n - 1)), float((sal * ys).sum() / total / (n - 1))
    left, top = float(sal[:, : n // 2].sum() / total), float(sal[: n // 2, :].sum() / total)
    thirds = [(x, y) for x in (1 / 3, 2 / 3) for y in (1 / 3, 2 / 3)]
    thirds_dist = float(min(np.hypot(cx - x, cy - y) for x, y in thirds))
    center_dist = float(np.hypot(cx - 0.5, cy - 0.5))
    g = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float64)
    g = (g - g.mean()) / (g.std() + 1e-6)
    symmetry = float((g * g[:, ::-1]).mean())
    edges = cv2.Canny(cv2.resize(gray, (256, 256)).astype(np.uint8), 60, 160)
    cells = edges.reshape(16, 16, 16, 16).mean(axis=(1, 3)) / 255
    negative_space = float((cells < 0.01).mean())

    # Subject estimate: largest connected high-saliency region.
    mask = (sal > max(0.35, sal.mean() + sal.std())).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    subject, flags, crop_flags = None, [], []
    if num > 1:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, w, h, area = [int(v) for v in stats[k]]
        mass = float(sal[labels == k].sum() / total)
        subject = {"bbox": [round(x / n, 3), round(y / n, 3), round(w / n, 3), round(h / n, 3)],
                   "area_fraction": round(area / (n * n), 4), "saliency_mass": round(mass, 3)}
        sides = [s for s, hit in (("left", x <= 0), ("top", y <= 0), ("right", x + w >= n), ("bottom", y + h >= n)) if hit]
        if sides and area / (n * n) > 0.03:
            crop_flags.append("subject_touches_border:" + ",".join(sides))
        if area / (n * n) < 0.015:
            flags.append("small_subject")
    else:
        flags.append("no_distinct_subject")
    peak_ratio = float(sal.max() / (sal.mean() + 1e-9))
    border = np.ones_like(sal, dtype=bool)
    b = max(1, n // 16)
    border[b:-b, b:-b] = False
    border_mass = float(sal[border].sum() / total)
    if border_mass > 0.45:
        crop_flags.append("salient_content_near_borders")
    balance = 1 - min(1.0, abs(left - 0.5) * 2 + abs(top - 0.5) * 2)
    placement = 1 - _lin(min(thirds_dist, center_dist), 0.10, 0.45)
    prominence = 100 * _clip01(0.6 * _lin(peak_ratio, 2.0, 7.0) + 0.4 * _lin((subject or {}).get("saliency_mass", 0.0), 0.10, 0.60))
    score = 100 * (0.5 * balance + 0.5 * placement)
    return {
        "saliency_centroid": [round(cx, 3), round(cy, 3)], "left_mass": round(left, 3), "top_mass": round(top, 3),
        "rule_of_thirds_distance": round(thirds_dist, 3), "center_distance": round(center_dist, 3),
        "horizontal_symmetry": round(symmetry, 3), "negative_space_fraction": round(negative_space, 3),
        "subject_placement": _placement_name(cx, cy), "score": round(score, 1),
        "subject": subject, "prominence_score": round(prominence, 1), "prominence_flags": flags,
        "border_saliency_mass": round(border_mass, 3), "crop_flags": crop_flags,
        "method": "spectral_residual_saliency (heuristic; status=inferred)",
    }


def _placement_name(cx: float, cy: float) -> str:
    col = "left" if cx < 0.40 else ("right" if cx > 0.60 else "center")
    row = "upper" if cy < 0.40 else ("lower" if cy > 0.60 else "middle")
    return "centered" if (col, row) == ("center", "middle") else f"{row} {col}"


class TechnicalQualityStage(ImageStage):
    name = "quality_technical"
    label = "TECH"

    def describe(self, d):
        return (f"sharpness {d['sharpness']['score']:.0f}, noise {d['noise']['score']:.0f}, "
                f"exposure {d['exposure']['exposure_score']:.0f}, contrast {d['exposure']['contrast_score']:.0f}")

    version = "1"
    title = "Technical quality signals"
    processes = "17, 18, 19, 20, 21, 23, 24, 25"
    depends = ("validate",)

    def enabled(self, ctx):
        return (ctx.cfg.stages.technical_quality is not False), "disabled in configuration"

    def process(self, ctx: StageContext, ref: ImgRef) -> dict:
        img = analysis_rgb(ref.abs_path, 1024)
        rgb = np.asarray(img)
        gray = np.asarray(img.convert("L"), dtype=np.float32)
        comp = blockiness(ref.abs_path)
        comp["jpeg_quality_estimate"] = jpeg_quality_estimate(ref.abs_path)
        return {"sharpness": sharpness(gray), "noise": noise(rgb), "compression": comp,
                "exposure": exposure(gray), "color": color_stats(img), "composition": composition(gray),
                "analysis_size": list(img.size)}
