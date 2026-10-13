"""Measured sharpen / denoise policy for the LF day master.

The sampled colour-analysis pass measures a detail score and a noise score per
frame (``measure_detail`` / ``measure_noise``); ``compute_correction`` turns
them into an unsharp amount+radius and an hqdn3d strength here. No caller holds
its own literal: every endpoint below is the single home.

The sharpen ramp is the photo develop ramp (``photos/develop.py``,
``_sharpen_amount``) re-implemented rather than imported so the two modules
stay independent of each other. Keep the numbers in step.

The rendition table (``RENDITIONS``) and the honest-resolution rule
(``HONEST_RESOLUTION_RULE``) live here too: the one place a delivered frame's
width, height, aspect, codec and bitrate floor are written down.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from gomoveshift.pipeline.encoding_constants import RENDITIONS, Rendition

#: Replaces the old "never scale up" wording. Declared upscaling is allowed
#: (a render may enlarge a source on purpose); what is forbidden is a
#: *label* -- a 4K title, a 4K metadata line, a 4K gate pass -- on pixels the
#: source never had. Comments and help text point here rather than restating it.
HONEST_RESOLUTION_RULE = "never claim a native resolution the source lacked"


def aspect_of(width: int, height: int) -> tuple[int, int]:
    """Reduced width:height of a measured frame (0x0 -> (0, 0))."""
    g = math.gcd(width, height)
    return (width // g, height // g) if g else (0, 0)


__all__ = [
    "HONEST_RESOLUTION_RULE",
    "RENDITIONS",
    "Rendition",
    "UpscalePlan",
    "UpscaleRefusedError",
    "aspect_of",
    "claim_native_resolution_allowed",
    "rendition_for",
    "upscale_plan",
]

#: The one resampling scaler every declared upscale (long-form and shorts) uses.
UPSCALE_SCALER = "zscale"
UPSCALE_KERNEL = "spline36"


class UpscaleRefusedError(ValueError):
    """The profile forbids upscaling and the source is below the rendition."""


@dataclass(frozen=True)
class UpscalePlan:
    """A declared resampling upscale: one scaler, the source size, the target size.

    ``crop`` is the centre-crop applied after scaling when the source is not the
    rendition's aspect (cover-fit), else ``None``.
    """

    scaler: str
    filter: str
    source: tuple[int, int]
    target: tuple[int, int]
    crop: tuple[int, int] | None = None
    note: str = ""


def _even_up(n: float) -> int:
    n = math.ceil(n)
    return n + (n % 2)


def upscale_plan(
    source_w: int, source_h: int, rendition: Rendition, policy: str
) -> UpscalePlan | None:
    """The upscale a source needs to reach ``rendition``, or ``None`` if it already does.

    ``resample`` scales with ``zscale`` (spline36). ``ml`` resolves to resample in
    this unit (no ML scaler is wired). ``off`` raises ``UpscaleRefusedError``.
    A non-rendition-aspect source is cover-fit (scaled until both sides cover the
    target) and centre-cropped; the crop is recorded on the plan.
    """
    tw, th = rendition.width, rendition.height
    if not tw or not th or (source_w >= tw and source_h >= th):
        return None
    if policy == "off":
        raise UpscaleRefusedError(
            f"source {source_w}x{source_h} is below the {tw}x{th} rendition and the media "
            f"profile upscale policy is 'off' ({HONEST_RESOLUTION_RULE})"
        )
    if policy not in ("resample", "ml"):
        raise UpscaleRefusedError(f"unknown upscale policy {policy!r}")
    note = "ml resolves to resample in this release" if policy == "ml" else ""
    kernel = f"filter={UPSCALE_KERNEL}"
    if source_w * th == source_h * tw:
        return UpscalePlan(
            UPSCALE_SCALER,
            f"{UPSCALE_SCALER}=w={tw}:h={th}:{kernel}",
            (source_w, source_h),
            (tw, th),
            None,
            note,
        )
    factor = max(tw / source_w, th / source_h)
    sw, sh = _even_up(source_w * factor), _even_up(source_h * factor)
    flt = f"{UPSCALE_SCALER}=w={sw}:h={sh}:{kernel},crop={tw}:{th}"
    return UpscalePlan(UPSCALE_SCALER, flt, (source_w, source_h), (tw, th), (tw, th), note)


def claim_native_resolution_allowed(profile, plan: UpscalePlan | None) -> bool:
    """May the delivery label itself with the native resolution?

    Always for a native source; for an upscaled one only when the profile does
    not restrict the claim to native pixels (``claim_native_resolution_only``).
    """
    return plan is None or not getattr(profile, "claim_native_resolution_only", True)


def rendition_for(name: str) -> Rendition:
    """The named rendition; an unknown name raises ``KeyError`` listing the known ones."""
    try:
        return RENDITIONS[name]
    except KeyError:
        raise KeyError(f"unknown rendition {name!r}; known: {sorted(RENDITIONS)}") from None


# Sharpen ramp, on log10(variance of Laplacian, 8-bit luma levels, bounded view).
# At/below SHARPEN_DETAIL_FLOOR (log10 0.5 -> ~3.2) the frame is featureless or
# unusable: sharpening would only amplify noise, so amount is 0. At/above
# SHARPEN_DETAIL_CRISP (log10 3.0 -> 1000) it is already crisp: amount is 0.
# Between, the amount falls linearly from SHARPEN_MAX_AMOUNT (softest) to 0.
SHARPEN_MAX_AMOUNT = 0.8
SHARPEN_DETAIL_FLOOR = 0.5
SHARPEN_DETAIL_CRISP = 3.0
SHARPEN_RADIUS = 1.2
SHARPEN_REFERENCE_WIDTH = 1920.0
SHARPEN_DETAIL_MAX_DIM = 1536.0

# Denoise ramp, on the noise sigma in 8-bit luma levels (Immerkaer estimate).
# Below DENOISE_NOISE_FLOOR the source is clean: strength 0. At/above
# DENOISE_NOISE_FULL the strength is DENOISE_MAX_STRENGTH; linear between.
DENOISE_NOISE_FLOOR = 1.5
DENOISE_NOISE_FULL = 6.0
DENOISE_MAX_STRENGTH = 4.0

_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)


def _luma(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:  # already a luma/gray plane
        return rgb.astype(np.float64)
    return rgb.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])


def _bounded_luma(rgb: np.ndarray, max_dim: float) -> np.ndarray:
    luma = _luma(rgb)
    longest = max(luma.shape)
    if longest <= max_dim:
        return luma
    scale = max_dim / longest
    size = (max(1, round(luma.shape[1] * scale)), max(1, round(luma.shape[0] * scale)))
    return cv2.resize(luma, size, interpolation=cv2.INTER_AREA)


def sharpness_score(gray_or_rgb: np.ndarray) -> float:
    """THE sharpness metric: variance of the Laplacian on a bounded luma view.

    Accepts an RGB frame or a 2-D gray plane. The thumbnail probe, the photo
    cull, the LF grade and the phase A source gate all call this -- no other
    module may carry its own Laplacian.
    """
    return float(
        cv2.Laplacian(_bounded_luma(gray_or_rgb, SHARPEN_DETAIL_MAX_DIM), cv2.CV_64F).var()
    )


measure_detail = sharpness_score

# Phase A refuses a source whose median sharpness (over SOURCE_SHARPNESS_SAMPLES
# frames) falls below this. Derivation: the thumbnail probe calls a frame
# fully "well-focused" at Laplacian variance 500 (thumbnail_generator
# SHARPNESS_SATURATION); the floor is 10% of that, 50 (log10 1.7), where the
# sharpen ramp above is already ~half spent on a frame and a smeared lens or
# stuck-focus clip is what remains. Scale factor 1.0: the probe's 4-neighbour
# kernel is cv2.Laplacian's ksize-1 kernel, and its 960 px bound versus this
# module's 1536 px bound only matters above 960 px.
# PROVISIONAL until measured on real footage.
SOURCE_SHARPNESS_FLOOR = 50.0
SOURCE_SHARPNESS_SAMPLES = 5


def measure_noise(rgb: np.ndarray) -> float:
    """Noise sigma in luma levels: median |Immerkaer high-pass| (grain is dense, edges sparse)."""
    luma = _luma(rgb)
    if min(luma.shape) < 3:
        return 0.0
    response = cv2.filter2D(luma, cv2.CV_64F, _NOISE_KERNEL, borderType=cv2.BORDER_REPLICATE)
    return float(np.median(np.abs(response[1:-1, 1:-1])) / (0.6745 * 6.0))


def sharpen_amount_for(detail_score: float, width: int, *, disabled: bool) -> tuple[float, float]:
    """Return ``(amount, radius)``; ``(0.0, 0.0)`` when disabled, crisp, or unusable."""
    if disabled:
        return 0.0, 0.0
    log_detail = math.log10(max(detail_score, 1e-6))
    if log_detail <= SHARPEN_DETAIL_FLOOR or log_detail >= SHARPEN_DETAIL_CRISP:
        return 0.0, 0.0
    t = (log_detail - SHARPEN_DETAIL_FLOOR) / (SHARPEN_DETAIL_CRISP - SHARPEN_DETAIL_FLOOR)
    amount = round(SHARPEN_MAX_AMOUNT * (1.0 - t), 3)
    radius = SHARPEN_RADIUS * max(1.0, width / SHARPEN_REFERENCE_WIDTH)
    return amount, round(radius, 3)


def denoise_strength_for(noise_score: float, *, disabled: bool) -> float:
    """hqdn3d strength from the noise sigma; 0.0 for clean sources or when disabled."""
    if disabled or noise_score <= DENOISE_NOISE_FLOOR:
        return 0.0
    t = min(1.0, (noise_score - DENOISE_NOISE_FLOOR) / (DENOISE_NOISE_FULL - DENOISE_NOISE_FLOOR))
    return round(DENOISE_MAX_STRENGTH * t, 2)


# ── Post-rendition picture gate ───────────────────────────────────────────
# Ringing: near strong edges, how far do pixels poke past the local envelope of
# a slightly blurred copy of the frame? A clean step edge never exceeds its own
# plateau (0.0); a Gibbs halo (over/undershoot ripple either side of the edge)
# does. Measurable on the delivered frame alone.
RINGING_EDGE_ZONE = 9  # px dilation of the strong-edge mask: halos sit a few px off the edge
RINGING_ENVELOPE = 5  # local morphological window
RINGING_BLUR_SIGMA = 0.8
RINGING_GRADIENT_FLOOR = 1.0  # flat / noise-only frames have no edges to ring on

# PROVISIONAL until measured on real footage. Derived from the synthetic
# contract in tests/test_post_rendition_picture_gate.py: a 60->190 step scores
# ~0.00005 clean and ~0.0217 with a +/-24-level Gibbs ripple; 0.01 sits between
# (~half the haloed score, 200x the clean one) so a ripple of roughly a dozen levels trips it while
# an unsharpened frame never does. Re-derive from real FINALs.
RINGING_MAX = 0.01

# Fraction of source sharpness a delivered frame may lose before the gate
# fails. PROVISIONAL: H.265 re-encode at the delivery bitrate and the scale to
# the rendition legitimately cost some Laplacian variance, so parity would
# false-fail; 25% leaves room for that while a rendition step that smeared the
# picture (denoise too hard, bad scale) loses far more. Tune on real footage.
DELIVERED_SHARPNESS_TOLERANCE = 0.25


def ringing_score(gray_or_rgb: np.ndarray) -> float:
    """Mean halo/overshoot near strong edges, normalised to luma range (0..1)."""
    luma = _bounded_luma(gray_or_rgb, SHARPEN_DETAIL_MAX_DIM).astype(np.float32)
    if min(luma.shape) < 3:
        return 0.0
    gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(gx, gy)
    threshold = max(float(grad.mean() + 2.0 * grad.std()), RINGING_GRADIENT_FLOOR)
    edges = (grad > threshold).astype(np.uint8)
    if not edges.any():
        return 0.0
    zone = cv2.dilate(edges, np.ones((RINGING_EDGE_ZONE, RINGING_EDGE_ZONE), np.uint8)).astype(bool)
    blurred = cv2.GaussianBlur(luma, (0, 0), RINGING_BLUR_SIGMA)
    window = np.ones((RINGING_ENVELOPE, RINGING_ENVELOPE), np.uint8)
    over = np.maximum(luma - cv2.dilate(blurred, window), 0.0)
    under = np.maximum(cv2.erode(blurred, window) - luma, 0.0)
    return float(((over + under)[zone]).mean() / 255.0)


def picture_changed(params) -> bool:
    """True when a rendition step altered the picture (CorrectionParams or mapping).

    Reads with defaults so a future step only has to add a key: sharpen amount,
    denoise strength, or a truthy stabilise / upscale marker.
    """

    def read(key: str):
        if isinstance(params, Mapping):
            return params.get(key)
        return getattr(params, key, None)

    for key in ("sharpen_amount", "denoise_strength"):
        value = read(key)
        if isinstance(value, (int, float)) and value > 0:
            return True
    return any(bool(read(key)) for key in ("stabilise", "stabilize", "upscale"))


def sample_delivered_picture(path, duration: float, n: int = SOURCE_SHARPNESS_SAMPLES):
    """``(sharpness_values, ringing_values)`` of ``n`` evenly spaced frames of ``path``.

    One reduced-size ffmpeg extraction per sample; undecodable frames are dropped.
    """
    sharp: list[float] = []
    ring: list[float] = []
    for i in range(n):
        ts = max(duration, 0.0) * (i + 0.5) / n
        cmd = [
            "ffmpeg", "-loglevel", "error", "-ss", f"{ts:.3f}", "-i", str(path),
            "-frames:v", "1", "-vf", f"scale={int(SHARPEN_DETAIL_MAX_DIM)}:-2",
            "-f", "image2pipe", "-c:v", "png", "-",
        ]  # fmt: skip
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode != 0 or not r.stdout:
            continue
        gray = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            sharp.append(sharpness_score(gray))
            ring.append(ringing_score(gray))
    return sharp, ring


def sample_delivered_sharpness(path, duration: float, n: int = SOURCE_SHARPNESS_SAMPLES) -> list:
    """Sharpness of ``n`` evenly spaced frames of a delivered file."""
    return sample_delivered_picture(path, duration, n)[0]


def picture_changed_for_day(cache_path) -> bool:
    """True when the day's colour cache records any sharpen/denoise/marker step.

    Reads the ``corrections`` list of ``day_<date>_color_cache.json`` (each entry
    a ``CorrectionParams`` dict). A missing or unreadable file is False, with a
    warning -- the gate then records "skipped", never a guess.
    """
    path = Path(cache_path)
    try:
        corrections = json.loads(path.read_text()).get("corrections") or []
    except (OSError, ValueError, AttributeError):
        logging.getLogger(__name__).warning(
            "picture gate: colour cache %s unreadable -- treating picture as unchanged", path
        )
        return False
    return any(picture_changed(c) for c in corrections)


def picture_inputs_from_state(state) -> tuple[float | None, dict[int, bool]]:
    """``(source_sharpness, {lf_index: picture_changed})`` for the post-rendition gate.

    ``state.source_sharpness`` is per source file with no LF->source mapping, so
    the sharpness is the median over every measured source (``None`` when none
    was measured). TODO(later unit): map each LF to its day's sources.
    ``picture_changed`` is per day (``DayArtifacts.regular_lf_indices`` -> that
    day's colour cache next to the day master), fanned out to the day's LFs.
    """
    measured = [v for v in (getattr(state, "source_sharpness", None) or []) if v is not None]
    sharpness = float(statistics.median(measured)) if measured else None
    changed: dict[int, bool] = {}
    for art in getattr(state, "day_artifacts", None) or []:
        master = getattr(art, "master_path", None)
        record = getattr(art, "master_record", None) or {}
        day_changed = bool(record.get("upscaled_from")) or (
            picture_changed_for_day(Path(master).parent / f"day_{art.date}_color_cache.json")
            if master
            else False
        )
        for idx in getattr(art, "regular_lf_indices", None) or []:
            changed[idx] = day_changed
    return sharpness, changed


# --- Thumbnail finish (image quality) -----------------------------------------
# Contrast +15 % min, saturation +15-20 %, mild unsharp, applied to the grab before
# the text styles. Contrast shifts luma only, then saturation blends away from luma.
THUMB_CONTRAST = 1.17
THUMB_SATURATION = 1.175
THUMB_UNSHARP_AMOUNT = 0.5
THUMB_UNSHARP_RADIUS = 1.0
# Bands the gate accepts: contrast at least +15 % with 0.25 as the cap, saturation +15-20 %.
THUMB_CONTRAST_BAND = (0.15, 0.25)
THUMB_SATURATION_BAND = (0.15, 0.20)
THUMB_GAIN_BOOSTS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
THUMB_MAX_BYTES = 2 * 1024 * 1024
THUMB_JPEG_QUALITIES = (95, 90, 85)


class ThumbnailTooLargeError(RuntimeError):
    """The encoded thumbnail exceeds THUMB_MAX_BYTES at every quality step."""


THUMB_PROXY_LONG_EDGE = 400
THUMB_FULL_RES_TRIES = 3


def _finish_sigma(width: int) -> float:
    return THUMB_UNSHARP_RADIUS * max(width / SHARPEN_REFERENCE_WIDTH, 0.5)


def _finish_pass(
    rgb: np.ndarray, contrast: float, saturation: float, sigma: float | None = None
) -> np.ndarray:
    f = rgb.astype(np.float32)
    w = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    luma = f @ w
    mean = float(luma.mean())
    shift = (contrast - 1.0) * (luma - mean)
    f = f + shift[..., None]
    luma = luma + shift
    f = luma[..., None] + saturation * (f - luma[..., None])
    f = np.clip(f, 0.0, 255.0)
    blurred = cv2.GaussianBlur(f, (0, 0), _finish_sigma(rgb.shape[1]) if sigma is None else sigma)
    f = f + THUMB_UNSHARP_AMOUNT * (f - blurred)
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


def thumbnail_finish(rgb: np.ndarray) -> np.ndarray:
    """Finished frame only; see thumbnail_finish_measured."""
    return thumbnail_finish_measured(rgb)[0]


def thumbnail_finish_measured(rgb: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Finish ``rgb``; return (frame, measured deltas of that frame).

    Contrast and saturation gains are stepped independently (THUMB_GAIN_BOOSTS).
    All pairs are measured on a downscaled proxy; the pair with the most room to
    its nearest band edge is applied at full size and measured. The proxy drifts
    (noise averages away), so a miss corrects the proxy deltas by the measured
    drift and tries the next pair, at most THUMB_FULL_RES_TRIES full-size
    apply + measure rounds. With no in-band proxy pair, boosts 1.0 / 1.0
    (contrast x1.17, saturation x1.175) are applied once. A result still out of
    band is returned as is; the caller ships the frame unfinished.
    """
    h, w = rgb.shape[:2]
    scale = min(1.0, THUMB_PROXY_LONG_EDGE / max(h, w))
    if scale < 1.0:
        proxy = cv2.resize(
            rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
        )
    else:
        proxy = rgb
    sigma = _finish_sigma(w) * (proxy.shape[1] / w)
    cands = []
    for c_boost in THUMB_GAIN_BOOSTS:
        contrast = 1.0 + (THUMB_CONTRAST - 1.0) * c_boost
        for s_boost in THUMB_GAIN_BOOSTS:
            saturation = 1.0 + (THUMB_SATURATION - 1.0) * s_boost
            out = _finish_pass(proxy, contrast, saturation, sigma)
            cands.append((contrast, saturation, measure_finish_deltas(proxy, out)))
    drift = {"contrast": 0.0, "saturation": 0.0}
    result = None
    for _ in range(THUMB_FULL_RES_TRIES):
        scored = [
            (_band_room(proxy, {k: pd[k] + drift[k] for k in drift}), -i, c, s, pd)
            for i, (c, s, pd) in enumerate(cands)
        ]
        room, _, contrast, saturation, pd = max(scored, key=lambda t: t[:2])
        if room <= 0.0:
            break
        out = _finish_pass(rgb, contrast, saturation)
        d = measure_finish_deltas(rgb, out)
        result = (out, d)
        if finish_deltas_in_band(rgb, d):
            return result
        drift = {k: d[k] - pd[k] for k in drift}
        cands = [t for t in cands if t[:2] != (contrast, saturation)]
    if result is not None:
        return result
    out = _finish_pass(rgb, 1.0 + (THUMB_CONTRAST - 1.0), 1.0 + (THUMB_SATURATION - 1.0))
    return out, measure_finish_deltas(rgb, out)


def _chroma_mean(rgb: np.ndarray) -> float:
    f = rgb.astype(np.float64)
    return float((f.max(axis=-1) - f.min(axis=-1)).mean())


def measure_finish_deltas(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    """Fractional change in luma contrast (std) and mean chroma, after vs before."""
    c0, c1 = float(_luma(before).std()), float(_luma(after).std())
    s0, s1 = _chroma_mean(before), _chroma_mean(after)
    return {
        "contrast": c1 / c0 - 1.0 if c0 > 0 else 0.0,
        "saturation": s1 / s0 - 1.0 if s0 > 0 else 0.0,
    }


def _band_room(before: np.ndarray, d: dict[str, float]) -> float:
    """Smallest distance from a measurable delta to its band edge (<= 0: out of band)."""
    room = float("inf")
    if _luma(before).std() >= 1.0:
        lo, hi = THUMB_CONTRAST_BAND
        room = min(room, d["contrast"] - lo, hi - d["contrast"])
    if _chroma_mean(before) >= 1.0:
        lo, hi = THUMB_SATURATION_BAND
        room = min(room, d["saturation"] - lo, hi - d["saturation"])
    return room


def finish_deltas_in_band(before: np.ndarray, d: dict[str, float]) -> bool:
    """True when measured deltas ``d`` sit inside the accepted bands.

    An axis the source cannot express (a flat frame has no contrast to raise; a
    grey one has no chroma) is not measurable and passes.
    """
    return _band_room(before, d) >= 0.0


def thumbnail_finish_in_band(before: np.ndarray, after: np.ndarray) -> bool:
    """True when ``after`` measures inside the accepted bands against ``before``."""
    return finish_deltas_in_band(before, measure_finish_deltas(before, after))


def encode_thumbnail_jpeg(img, output_path: str) -> int:
    """Save ``img`` (PIL) as JPEG under THUMB_MAX_BYTES, or refuse before writing.

    Steps quality down THUMB_JPEG_QUALITIES; raises ThumbnailTooLargeError when
    even the last step is over the cap. Returns the bytes written.
    """
    import io
    from pathlib import Path

    size = 0
    for q in THUMB_JPEG_QUALITIES:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q)
        size = buf.tell()
        if size <= THUMB_MAX_BYTES:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(buf.getvalue())
            return size
    raise ThumbnailTooLargeError(
        f"thumbnail is {size} bytes at JPEG q{THUMB_JPEG_QUALITIES[-1]}; "
        f"cap is {THUMB_MAX_BYTES} (YouTube rejects >2 MB)"
    )
