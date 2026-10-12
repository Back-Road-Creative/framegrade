"""Named soft masks — a registry of PURE functions of a graded frame.

Every colour knob the develop chain itself takes is whole-frame, and the two spatial
masks that chain owns (``_sky_mask``, ``_radial_vignette_mask``) are welded there to
exactly one effect each — a darkened sky, a vignette — so nothing in the chain lets a
colour recipe choose its own region. This registry is where one names it.
``mask_for(name, rgb)`` returns an ``(H, W)`` float32 mask in [0, 1] for a graded
``HxWx3`` frame and ``MASKS`` enumerates the legal names, so a caller can refuse an
unknown one by listing what exists.

``develop._layered`` is the render-side consumer — one ``mask_for`` call per ``layers:``
entry (a look's, or ``--layers``' own), lerping that layer's grade through the mask it
names — and the five ``_MASKED`` built-in looks are written out of nothing else. A layer
carries colour AND tone knobs, may name a PAIR of entries plus a ``mask_op`` to reach a
region no single one expresses, and may pass a named mask's own shape params. ``config``
mirrors this registry rather than importing it (``LAYER_MASK_PARAMS``, ``LAYER_MASK_OPS``,
kept honest by a sync test), so the gate at config LOAD refuses a bad layer without
dragging the pipeline in — a mask added here needs that literal updated too.

Four properties, each pinned by a test: **deterministic** (a pure function of the frame
— no RNG, no clock, no global state); **feathered** (every transition a smoothstep,
never a 0/1 cliff); **resolution-anchored** (nothing here depends on the pixel COUNT —
positions are normalized coordinates and tone is metered on a bounded view — so one name
means one look at 1200 px and at 5600 px, ``GRAIN_REFERENCE_DIM``'s lesson pre-applied);
**range-closed** (finite, in [0, 1], for black, white, flat and noise frames alike).

The develop helpers reused here (``_luma``, ``_peak``, ``_bounded_view``, ``_sky_mask``,
``_radial_vignette_mask``) are imported INSIDE the functions on purpose: ``develop``
imports this module back — in ``_layered``, and again while its own module body computes
``DEVELOP_ALGO_VERSION`` — so a module-level import here would be a circular import.
Do not "tidy" them to the top.
"""

import inspect

import cv2
import numpy as np

# Luma windows. The two boundaries and the transition width are fractions of the frame's
# OWN measured tonal span (``LUMA_SPAN_PERCENTILES``, metered on a ``MASK_MEASURE_MAX_DIM``
# view — a 20 MP percentile is a full sort, and sensor speckle must not set the span).
# Frame-relative is not a refinement but the difference between working and inert: this
# grade chain tops out near 0.66 luma and never clips, so an ABSOLUTE highlight window
# sits above every pixel of every photograph — the lesson ``_halation_seed``, the dehaze
# haze floor and ``_sky_mask``'s brightness ramp each learned. A frame whose span falls
# under ``LUMA_MIN_SPAN`` holds no structure to partition (a flat wall, a whiteout), so
# it falls back to absolute tone rather than partitioning its own noise.
LUMA_EDGES = (1.0 / 3.0, 2.0 / 3.0)
LUMA_WINDOW_WIDTH = 0.25
LUMA_SPAN_PERCENTILES = (1.0, 99.0)
LUMA_MIN_SPAN = 0.05
MASK_MEASURE_MAX_DIM = 512.0
# ``radial``: an oval of weight, ``cx``/``cy`` of the way across and down the frame
# (0.5, 0.5 = the middle), falling off to zero at ``size`` multiples of the shipped
# vignette's own reach, stretched across (``aspect`` > 1) or down (< 1). Every value is a
# FRACTION or a ratio, never a pixel, so one recipe names one oval at 1200 px and at
# 5600 px; a subject that does not stand dead centre is what the knobs are for. At the
# defaults this delegates to the vignette gain untouched, so the falloff every shipped
# recipe rides is the SAME function rather than arithmetic that agrees with it.
RADIAL_CENTRE = (0.5, 0.5)
RADIAL_SIZE = 1.0
RADIAL_ASPECT = 1.0
# ``linear``: a gradient whose ramp runs along ``angle`` degrees (0 = toward the right
# edge, 90 = toward the bottom, y growing downward) over ``span`` of the frame, topping
# out ``offset`` of the frame short of the near edge — 0 tops out ON that edge, which is
# where every ramp had to end before. A bottom-of-frame ground glow at the defaults; a
# tilt-shift band is this mask plus its ``invert=True`` twin at the mirrored angle.
LINEAR_ANGLE = 90.0
LINEAR_SPAN = 0.5
LINEAR_OFFSET = 0.0
# ``glints``: the highlight window kept only where its bright region is SMALL. Every other
# tonal mask is a per-pixel test, so on a landscape whose bright region IS the sky a
# signage tint through ``highlights`` grades a third of the picture (measured).
# This one groups the bright pixels into connected regions first and fades out the ones
# covering more than ``max_area`` of the frame, reaching zero at ``GLINT_AREA_ROLLOFF``
# multiples of it — a size ramp, not a cliff at one area. The region areas and the softening
# radius are FRACTIONS, metered on the ``MASK_MEASURE_MAX_DIM`` view, so "small" means the
# same share of the picture at 1200 px and at 5600 px.
GLINT_MAX_AREA = 0.02
GLINT_AREA_ROLLOFF = 2.0
GLINT_EDGE_FRACTION = 0.01

# ``texture``: where the picture holds DETAIL — the local spread of luma, which is the one
# question tone and position cannot ask (a busy hedge and the flat sky above it can sit at
# the identical brightness in the identical part of the frame). The neighbourhood is a
# FRACTION of the metering view's long edge, and the meter runs on the same bounded view the
# tone masks use, so the answer is the scene's detail rather than the export's pixel count —
# and that view's INTER_AREA downscale is what stops sensor grain reading as detail.
# Normalized against the frame's OWN spread (its high percentile, not its max — one hot
# pixel must not set the scale), so "textured" means the same on a busy frame and a quiet
# one; a frame whose spread falls under ``TEXTURE_MIN_SPREAD`` holds no detail to find (a
# wall, a whiteout) and selects nothing rather than magnifying its own floor.
TEXTURE_RADIUS = 0.03
TEXTURE_REFERENCE_PERCENTILE = 98.0
TEXTURE_EDGES = (0.12, 0.55)
TEXTURE_MIN_SPREAD = 0.004

# ``hue``: WHICH COLOUR a pixel is — the question tone, position and detail cannot ask,
# and the reason "the foliage", "the actual blue of the sky rather than every bright
# pixel" and "the red car" were not expressible as regions (``_apply_hsl`` is the only
# hue-selective control in the chain and is whole-frame by construction). The band is
# ``width`` degrees of the colour wheel centred on ``center``, and hue is ANGULAR, so the
# distance is measured the short way round: a red band centred on 0 covers 350 and 10
# alike instead of splitting at the seam. ``HUE_EDGE_WIDTH`` degrees of smoothstep
# straddle each edge, so a gradient crossing out of the band shows no boundary. A
# near-grey pixel HAS no colour to name (the angle derived at zero chroma is arbitrary),
# so a chroma gate fades it out below ``min_chroma`` and a grey frame selects exactly
# nothing — ``texture``'s precedent, not magnifying its own floor. Chroma is metered as
# the pixel's own saturation (its chroma over its peak channel), a RATIO, so the gate
# means the same on a dark pixel and a bright one and needs no frame-relative span the
# way the luma windows do. Per-pixel throughout — no statistic, no neighbourhood — hence
# resolution-anchored by construction. The RGB→hue derivation MATCHES
# ``color_correction._apply_hsl``'s vectorised block exactly rather than importing it:
# that block is welded inside the applier's own HSV round trip over an (N, 3) buffer with
# no seam to call. Hue and saturation are ratios, so it reads a frame at either scale.
HUE_CENTER = 120.0
HUE_WIDTH = 90.0
HUE_EDGE_WIDTH = 40.0
HUE_MIN_CHROMA = 0.10
HUE_CHROMA_WIDTH = 0.10


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """The shared soft edge: clamped cubic, flat-tangented at both ends."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _tone(rgb: np.ndarray) -> np.ndarray:
    """This frame's luma mapped onto its own measured tonal span, in [0, 1]."""
    from .develop import _bounded_view, _luma, _peak

    peak = _peak(rgb)
    tone = _luma(rgb) / peak
    view = _luma(_bounded_view(rgb, MASK_MEASURE_MAX_DIM)) / peak
    low, high = np.percentile(view, LUMA_SPAN_PERCENTILES)
    if high - low < LUMA_MIN_SPAN:  # no range to partition: absolute tone, not noise
        return np.clip(tone, 0.0, 1.0).astype(np.float32)
    return np.clip((tone - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _above(rgb: np.ndarray, edge: float) -> np.ndarray:
    """Smoothstepped weight of the tone sitting ABOVE ``edge`` of the frame's span."""
    return _smoothstep((_tone(rgb) - edge) / LUMA_WINDOW_WIDTH + 0.5)


def _sky(rgb: np.ndarray, **_: object) -> np.ndarray:
    from .develop import _sky_mask

    return _sky_mask(rgb)


def _normalized_grid(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The frame as coordinates rather than pixels: -1 at the top/left edge, 1 at the
    bottom/right one, whatever its pixel count — where the resolution anchoring above
    comes from, and why every geometry knob here is a fraction of the frame."""
    height, width = rgb.shape[:2]
    ny = np.linspace(-1.0, 1.0, height, dtype=np.float32) if height > 1 else np.zeros(1, np.float32)
    nx = np.linspace(-1.0, 1.0, width, dtype=np.float32) if width > 1 else np.zeros(1, np.float32)
    return np.meshgrid(ny, nx, indexing="ij")


def _radial(
    rgb: np.ndarray,
    *,
    cx: float = RADIAL_CENTRE[0],
    cy: float = RADIAL_CENTRE[1],
    size: float = RADIAL_SIZE,
    aspect: float = RADIAL_ASPECT,
    **_: object,
) -> np.ndarray:
    """The shipped vignette falloff read as a mask, and MOVEABLE: 1 on the named spot,
    0 at the reach ``size`` sets, so "lift the subject" is expressible wherever the
    subject stands rather than only when it happens to be dead centre."""
    from .develop import VIGNETTE_FEATHER, _radial_vignette_mask

    if (cx, cy, size, aspect) == (*RADIAL_CENTRE, RADIAL_SIZE, RADIAL_ASPECT):
        return _radial_vignette_mask(rgb.shape[0], rgb.shape[1], 1.0)
    if not size > 0.0:
        raise ValueError(f"radial mask size must be > 0, got {size}")
    if not aspect > 0.0:
        raise ValueError(f"radial mask aspect must be > 0, got {aspect}")
    gy, gx = _normalized_grid(rgb)
    dx = (gx - (2.0 * cx - 1.0)) / aspect
    dy = (gy - (2.0 * cy - 1.0)) * aspect
    radius = np.sqrt((dx * dx + dy * dy) * 0.5) / size  # 0 on the spot, 1 at its reach
    return (1.0 - np.clip(radius, 0.0, 1.0) ** VIGNETTE_FEATHER).astype(np.float32)


def _linear(
    rgb: np.ndarray,
    *,
    angle: float = LINEAR_ANGLE,
    span: float = LINEAR_SPAN,
    offset: float = LINEAR_OFFSET,
    **_,
):
    if not span > 0.0:
        raise ValueError(f"linear mask span must be > 0, got {span}")
    gy, gx = _normalized_grid(rgb)
    theta = np.radians(angle)
    reach = gx * np.cos(theta) + gy * np.sin(theta)  # -1 at the far edge, 1 at the near one
    crest = 1.0 - 2.0 * offset  # where the ramp TOPS OUT: the near edge at offset 0
    return _smoothstep((reach - (crest - 2.0 * span)) / (2.0 * span))


def _glints(rgb: np.ndarray, *, max_area: float = GLINT_MAX_AREA, **_) -> np.ndarray:
    """Lit signage, not the lit sky: the highlight weight with its LARGE regions faded."""
    from .develop import _bounded_view

    if not max_area > 0.0:
        raise ValueError(f"glints mask max_area must be > 0, got {max_area}")
    view = _bounded_view(rgb, MASK_MEASURE_MAX_DIM)
    seeds = (_above(view, LUMA_EDGES[1]) >= 0.5).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(seeds, connectivity=8)
    share = stats[:, cv2.CC_STAT_AREA] / float(seeds.size)
    keep = 1.0 - _smoothstep((share / max_area - 1.0) / (GLINT_AREA_ROLLOFF - 1.0))
    keep[0] = 0.0  # label 0 is the unlit background, never a region
    radius = max(GLINT_EDGE_FRACTION * min(seeds.shape), 1.0)
    small = cv2.GaussianBlur(keep[labels].astype(np.float32), (0, 0), radius)
    if small.shape != rgb.shape[:2]:
        small = cv2.resize(small, rgb.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    return _smoothstep(small) * _above(rgb, LUMA_EDGES[1])


def _texture(rgb: np.ndarray, **_: object) -> np.ndarray:
    """Local standard deviation of luma, as a fraction of this frame's own spread."""
    from .develop import _bounded_view, _luma, _peak

    view = _luma(_bounded_view(rgb, MASK_MEASURE_MAX_DIM)) / _peak(rgb)
    side = 2 * max(1, round(TEXTURE_RADIUS * max(view.shape))) + 1  # odd, and never zero-wide
    box = (side, side)
    mean = cv2.blur(view, box, borderType=cv2.BORDER_REFLECT)
    square = cv2.blur(view * view, box, borderType=cv2.BORDER_REFLECT)
    spread = np.sqrt(np.maximum(square - mean * mean, 0.0))  # negative only by round-off
    reference = float(np.percentile(spread, TEXTURE_REFERENCE_PERCENTILE))
    if reference < TEXTURE_MIN_SPREAD:
        return np.zeros(rgb.shape[:2], np.float32)
    low, high = TEXTURE_EDGES
    mask = _smoothstep((spread / reference - low) / (high - low)).astype(np.float32)
    return cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)


def _hue(
    rgb: np.ndarray,
    *,
    center: float = HUE_CENTER,
    width: float = HUE_WIDTH,
    min_chroma: float = HUE_MIN_CHROMA,
    **_,
) -> np.ndarray:
    """The foliage, not every green-ish pixel: a feathered band of the colour wheel."""
    if not width > 0.0:
        raise ValueError(f"hue mask width must be > 0, got {width}")
    if not 0.0 <= min_chroma < 1.0:
        raise ValueError(f"hue mask min_chroma must be in [0, 1), got {min_chroma}")
    frame = rgb.astype(np.float64, copy=False)
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    mx = np.max(frame, axis=2)
    chroma = mx - np.min(frame, axis=2)
    nz = chroma > 1e-12
    hue = np.zeros_like(mx)
    is_r = nz & (mx == r)
    is_g = nz & (mx == g) & ~is_r
    is_b = nz & ~is_r & ~is_g
    hue[is_r] = ((g - b)[is_r] / chroma[is_r]) % 6.0
    hue[is_g] = ((b - r)[is_g] / chroma[is_g]) + 2.0
    hue[is_b] = ((r - g)[is_b] / chroma[is_b]) + 4.0
    hue *= 60.0
    saturation = np.where(mx > 1e-12, chroma / np.where(nz, mx, 1.0), 0.0)
    apart = np.abs((hue - center + 180.0) % 360.0 - 180.0)  # the short way round the wheel
    band = 1.0 - _smoothstep((apart - 0.5 * width) / HUE_EDGE_WIDTH + 0.5)
    return (band * _smoothstep((saturation - min_chroma) / HUE_CHROMA_WIDTH)).astype(np.float32)


#: Every legal mask name, enumerated so a caller can refuse an unknown one by listing
#: what exists rather than failing on a bare ``KeyError``. The three luma windows are one
#: ramp differenced, which is what makes them sum to exactly 1 across the tonal range (a
#: layer set covers the frame with no seam) and ``midtones`` non-negative by construction.
MASKS = {
    "sky": _sky,
    "ground": lambda rgb, **_: 1.0 - _sky(rgb),
    "shadows": lambda rgb, **_: 1.0 - _above(rgb, LUMA_EDGES[0]),
    "midtones": lambda rgb, **_: _above(rgb, LUMA_EDGES[0]) - _above(rgb, LUMA_EDGES[1]),
    "highlights": lambda rgb, **_: _above(rgb, LUMA_EDGES[1]),
    "radial": _radial,
    "linear": _linear,
    "glints": _glints,
    "texture": _texture,
    "hue": _hue,
}

#: The shape params each mask accepts, DERIVED from its signature (the keyword-only
#: arguments) rather than hand-kept, so a mask grows a knob by declaring one and no list
#: goes stale. ``mask_for`` refuses every other name, which is what stops ``angle`` on
#: ``sky`` — real for ``linear``, silently inert anywhere else — from being written at all.
MASK_PARAMS = {
    name: frozenset(
        p.name for p in inspect.signature(fn).parameters.values() if p.kind is p.KEYWORD_ONLY
    )
    for name, fn in MASKS.items()
}


#: The two-operand combinators a layer may name (``mask_op``), enumerated for the same
#: reason ``MASKS`` is. A mask is a weight in [0, 1], so each is one line of array
#: algebra AND range-closed by construction: a product and a maximum of two members of
#: [0, 1] stay inside it, and ``subtract`` is the product with the complement. What they
#: buy is the region no single entry expresses — ``highlights`` is every pixel above 2/3
#: luma, which on a landscape IS a third of the sky, while "the highlights inside the
#: centre" is the signage. Two operands and one operator, deliberately: three would want
#: precedence rules, and this is a config file, not an expression language.
MASK_OPS = {
    "intersect": lambda a, b: a * b,
    "union": np.maximum,
    "subtract": lambda a, b: a * (1.0 - b),
}


def _feathered(mask: np.ndarray, feather: float) -> np.ndarray:
    """Widen (``feather`` > 1) or tighten (< 1) a mask's transition band with 0 and 1
    PINNED — a feather that lifted a zero region would leak its layer frame-wide.
    ``feather == 1.0`` is each mask's calibrated shape, returned untouched."""
    if feather == 1.0:
        return mask
    signed = 2.0 * mask - 1.0
    return 0.5 + 0.5 * np.sign(signed) * np.abs(signed) ** feather


def mask_for(
    name,
    rgb: np.ndarray,
    *,
    op: str | None = None,
    feather: float = 1.0,
    invert: bool = False,
    **params: float,
) -> np.ndarray:
    """The soft mask named ``name`` for graded frame ``rgb``: (H, W) float32 in [0, 1].

    ``name`` is one registry name, or — with ``op`` naming a ``MASK_OPS`` combinator —
    a PAIR of them, which is how a recipe reaches a region narrower (or wider) than any
    single entry. ``feather`` scales how soft the transitions are, ``invert`` returns the
    complement (what makes a tilt-shift band expressible without a second entry), and
    ``params`` carries a mask's own shape arguments (``linear``'s ``angle``/``span``).
    An unknown name or op — or a param no named mask accepts — raises ``ValueError``
    naming every legal one, so a knob that would ride along inert is refused instead.
    Params are checked against the UNION of the named masks' own params: on one name that
    is exactly that mask's set, and on a pair either operand's shape knob is legal.

    ORDER IS PINNED, and a test holds it: the two RAW masks combine FIRST, then
    ``feather``, then ``invert``. ``_feathered`` reshapes a transition band, so
    feathering each operand first would bend both bands before they ever meet and the
    same pair would land a different region — with the pinned order there is exactly one
    band to soften (the combination's own) and ``feather``/``invert`` mean on a pair what
    they already mean on a single mask."""
    names = (name,) if isinstance(name, str) else tuple(name)
    for one in names:
        if one not in MASKS:
            raise ValueError(f"unknown mask {one!r}; expected one of: {', '.join(sorted(MASKS))}")
    if op is not None and op not in MASK_OPS:
        raise ValueError(f"unknown mask op {op!r}; expected one of: {', '.join(sorted(MASK_OPS))}")
    if len(names) != (2 if op else 1):
        raise ValueError(f"mask op {op!r} takes {2 if op else 1} mask name(s), got {list(names)}")
    if not feather > 0.0:
        raise ValueError(f"mask feather must be > 0, got {feather}")
    accepted = frozenset().union(*(MASK_PARAMS[one] for one in names))
    if unknown := set(params) - accepted:
        legal = ", ".join(sorted(accepted)) or "none"
        raise ValueError(f"mask {name!r} takes no {', '.join(sorted(unknown))} (accepts: {legal})")
    raw = [np.asarray(MASKS[one](rgb, **params), dtype=np.float32) for one in names]
    mask = _feathered(raw[0] if op is None else MASK_OPS[op](*raw), feather)
    if invert:
        mask = 1.0 - mask
    return np.clip(mask, 0.0, 1.0).astype(np.float32)
