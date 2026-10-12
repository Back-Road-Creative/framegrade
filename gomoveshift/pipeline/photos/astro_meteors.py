"""Meteor re-insertion onto the stacked master (after the
stack has been developed).

CONTRACT, read before calling ``reinsert``: this module never runs the develop chain
and never imports another ``astro_*`` module. Every ``layer`` handed to ``reinsert``
MUST already be the SAME shot pushed through calibrate -> register -> gradient removal
-> the identical black-point/m stretch as ``base`` (``astro_develop.develop_stack``,
applied to base and each meteor frame with the SAME params) -- composing that chain is
the orchestrator's job. A layer independently stretched with its own black
point or m produces a smear: the lighten blend below only
reads as a seamless composite when base and layer already share calibration, geometry,
and tone.

WHAT THIS MODULE IS: a base-referenced max composite (``out = maximum(out, base +
mask * gate(layer - base))``) gated by two per-pixel thresholds -- a per-layer
measured noise floor and a star-core exclusion -- not a local-background (own-level)
model. See ``_layer_gate`` and ``reinsert`` for what that distinction costs.
"""

import cv2
import numpy as np

# Line thickness (px) drawn for a streak segment before feathering -- wide enough to
# cover a real meteor's optical width plus registration slop AND leave the core fully
# saturated once ``STREAK_FEATHER_PX`` of it is spent on the edge transition.
# PROVISIONAL pending a reference night card, same discipline as
# astro_triage.py's constants.
STREAK_LINE_WIDTH_PX = 10
# Distance (px) either side of a streak's true edge over which the mask ramps 0..1 --
# a Gaussian blur of a THIN stroke drags its own centre down along with the edge, so
# the ramp is built from a signed distance transform instead (``horizon_mask``'s
# ``_smoothstep``, applied to distance-from-edge rather than distance-from-a-row).
# PROVISIONAL, same caveat.
STREAK_FEATHER_PX = 4.0

# Multiple of a layer's own measured background noise (MAD-based sigma, sampled
# OUTSIDE the mask) an excess must clear before any of it counts as signal -- tracks
# each layer's actual noise instead of one sigma value guessed at authoring time.
# Ramped by _layer_gate, not a hard cliff, so a tapering streak crossing the
# floor does not step.
NOISE_FLOOR_SIGMA_MULT = 6.0
# LOWER BOUND on the floor (display-space ADU/DN), not just a fallback: the floor is
# max(this, NOISE_FLOOR_SIGMA_MULT x measured sigma), so it governs whenever a layer's
# background is quieter than ~4 ADU sigma -- every star-wing fixture in the suite -- and
# is the sole floor when a mask leaves under 100 unmasked pixels. PROVISIONAL.
EXCESS_NOISE_FLOOR = 24.0

# Kernel size (px) for estimating a frame's own local sky level around a candidate
# star -- wide enough that one point source is a small minority of the box average.
STAR_LOCAL_BG_PX = 15
# ADU a ``base`` pixel must sit above ITS OWN box-blurred neighbourhood before it is
# treated as an already-resolved star and excluded from ever receiving meteor light,
# at ANY wobble amplitude: a star's own shot noise (~sqrt(ADU), ~200 ADU at 40000 ADU)
# routinely exceeds a fixed excess floor by many multiples, so gating on wobble
# amplitude alone cannot suppress it. PROVISIONAL, same caveat.
STAR_PROMINENCE_ADU = 500.0


class MeteorInputError(ValueError):
    """A meteor layer, mask, or streak list handed to this module did not fit ``base``."""


def _smoothstep(x: np.ndarray) -> np.ndarray:
    """Clamped cubic ease, flat-tangented at both ends -- astro_stack.py's shared shape,
    duplicated rather than imported (this module imports no other ``astro_*`` module)."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def streak_mask(shape_hw, streaks, feather_px=STREAK_FEATHER_PX):
    """``(H, W)`` float32 mask in [0, 1]: each ``(x1, y1, x2, y2)`` in ``streaks`` drawn
    as a thick line, then feathered by a signed distance transform (1.0 deep inside the
    stroke, 0.0 far outside, a smoothstep ramp across ``feather_px`` centred on the true
    edge) so the composite has no hard seam yet the stroke's own core stays saturated.
    An empty ``streaks`` returns an all-zero mask -- nothing is eligible for meteor light."""
    height, width = shape_hw
    binary = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in streaks:
        start = (int(round(x1)), int(round(y1)))
        end = (int(round(x2)), int(round(y2)))
        cv2.line(binary, start, end, 1, thickness=STREAK_LINE_WIDTH_PX, lineType=cv2.LINE_8)
    if feather_px <= 0 or not binary.any():
        return binary.astype(np.float32)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    return _smoothstep((inside - outside) / feather_px + 0.5)


def _resolve_mask(shape_hw, entry, feather_px):
    """``entry`` is either a precomputed ``(H, W)`` ndarray mask or an iterable of streak
    tuples -- accepting both is what lets an orchestrator that already built its masks
    skip redrawing them. Disambiguated by TYPE (``ndarray`` vs. not), never by shape --
    a one-streak list is itself a ``(1, 4)`` array once coerced, indistinguishable from a
    mask by shape alone."""
    if isinstance(entry, np.ndarray):
        if entry.ndim != 2 or entry.shape != tuple(shape_hw):
            raise MeteorInputError(f"mask shape {entry.shape} != base shape {tuple(shape_hw)}")
        return entry.astype(np.float32)
    return streak_mask(shape_hw, entry, feather_px)


def _star_prominence_mask(base_f):
    """``(H, W)`` bool: True where ``base`` already carries a persistent point source
    (a star), box-blurred at ``STAR_LOCAL_BG_PX`` so one point source is a small
    minority of its own local average. Excluded from reinsertion outright, regardless
    of a candidate layer's wobble there -- a fixed noise floor cannot suppress a star
    (its own shot noise routinely exceeds one), but persistence in the frozen ``base``
    can, at any wobble amplitude. Dilated by ``STAR_LOCAL_BG_PX`` (the same box
    already sized "wide enough that one point source is a small minority of the box
    average") so a real PSF's WINGS -- individually too dim to clear the prominence
    threshold on their own -- are covered too, ONLY up to FWHM 11px (measured): the
    dilation radius is FIXED, not the star's own light, so wider PSFs leak past it
    (see ``test_reinsert_star_wings_leak_grows_past_the_dilation_radius``)."""
    bright = base_f.max(axis=-1)
    local_bg = cv2.blur(bright, (STAR_LOCAL_BG_PX, STAR_LOCAL_BG_PX))
    core = ((bright - local_bg) > STAR_PROMINENCE_ADU).astype(np.uint8)
    kernel = np.ones((STAR_LOCAL_BG_PX, STAR_LOCAL_BG_PX), dtype=np.uint8)
    return cv2.dilate(core, kernel) > 0


def _layer_gate(diff, mask):
    """Soft (smoothstep) ramp from 0 at/below a layer-measured noise floor to the raw
    excess at/above double that floor. floor = max(``EXCESS_NOISE_FLOOR``,
    ``NOISE_FLOOR_SIGMA_MULT`` x the layer's own MAD sigma where ``mask`` is ~0), so on a
    quiet layer the constant binds. A THRESHOLD GATE on ``layer - base``, NOT a local-
    background subtraction -- it never estimates or removes a layer's own per-pixel
    sky level. A local-background model (large-kernel morphological opening, or an
    annulus median per pixel) is deliberately not used: a layer uniformly elevated
    across the WHOLE fixture, not just under its mask, is indistinguishable from
    genuine ambient drift by any background statistic, so subtracting one zeroes
    the other in this module's own regression fixtures.

    Disclosed cost: ramping is not guaranteed SMALLER than the un-gated step it
    replaces -- measured on one taper, the un-gated single-pixel step is 2.00 ADU, the
    ramped one 6.00 ADU (``test_reinsert_gate_ramps_instead_of_stepping_across_the_noise_floor``).
    Softening the ramp width would only relocate that crossing, so it is disclosed
    rather than chased further. Sub-floor signal is discarded, not recovered."""
    background = diff[mask < 0.01]
    if background.size >= 100:
        sigma = np.median(np.abs(background - np.median(background))) * 1.4826
        floor = max(EXCESS_NOISE_FLOOR, NOISE_FLOOR_SIGMA_MULT * sigma)
    else:
        floor = EXCESS_NOISE_FLOOR
    raw = np.clip(diff, 0.0, None, out=diff)  # reuse diff's buffer -- caller is done with it
    gate = _smoothstep((raw - floor) / floor)
    return np.multiply(raw, gate, out=raw)


def reinsert(base, layers, streaks_or_masks=None, feather_px=STREAK_FEATHER_PX):
    """Composite each already-processed layer onto ``base`` via a BASE-REFERENCED MAX
    COMPOSITE, gated (NOT a local-background/excess-signal model -- see
    ``_layer_gate`` for that distinction and its cost). Per layer: raw excess is
    ``layer - base`` (against the frozen ``base``, never the mutating accumulator),
    ramped by ``_layer_gate``, zeroed outright by ``_star_prominence_mask`` (core plus
    dilated wings, BOUNDED to FWHM 11px -- see its docstring) -- folded in via
    ``out = maximum(out, base + mask * gate(...))``.
    Every layer is gated against the SAME fixed ``base`` rather than the running
    accumulator, which makes the fold commutative regardless of layer order.
    Zero layers is a no-op. See the module docstring for the develop-chain contract.
    No per-pixel local-background estimate is made (see ``_layer_gate`` for why).

    Two byte-identical calling forms: ``reinsert(base, layers, streaks_or_masks)`` or
    ``reinsert(base, pairs)`` (a lazily-consumed iterable of ``(layer,
    streak-or-mask)``). The accumulator and the per-layer excess (reused in place
    through clip/gate/mask/add, one buffer) keep peak RSS from climbing with layer
    count -- measured plateauing after the second streamed layer on a 2000x3000x3
    uint16 fixture (``ru_maxrss``, one process per point), rather than holding two
    full-res layers at once."""
    base = np.asarray(base)
    if streaks_or_masks is None:
        pairs = layers
    else:
        if len(layers) != len(streaks_or_masks):
            raise MeteorInputError(
                f"reinsert needs equal-length layers/streaks_or_masks, "
                f"got {len(layers)} and {len(streaks_or_masks)}"
            )
        pairs = zip(layers, streaks_or_masks)
    base_f = base.astype(np.float32)
    out = base_f.copy()
    shape_hw = base.shape[:2]
    star_mask = _star_prominence_mask(base_f)
    for layer, entry in pairs:
        layer = np.asarray(layer)
        if layer.shape != base.shape:
            raise MeteorInputError(f"reinsert: layer shape {layer.shape} != base {base.shape}")
        if layer.dtype != base.dtype:
            raise MeteorInputError(f"reinsert: layer dtype {layer.dtype} != base {base.dtype}")
        mask = _resolve_mask(shape_hw, entry, feather_px)
        diff = layer.astype(np.float32)
        diff -= base_f  # excess computed in place -- one scratch buffer, not two
        lifted = _layer_gate(diff, mask)  # aliases diff's buffer; gated in place too
        lifted[star_mask] = 0.0
        np.multiply(lifted, mask[..., None], out=lifted)
        np.add(lifted, base_f, out=lifted)
        np.maximum(out, lifted, out=out)  # accumulate in place -- no new out[] each layer
    info = np.iinfo(base.dtype) if np.issubdtype(base.dtype, np.integer) else None
    if info is not None:
        out = np.clip(np.round(out), info.min, info.max)
    return out.astype(base.dtype)
