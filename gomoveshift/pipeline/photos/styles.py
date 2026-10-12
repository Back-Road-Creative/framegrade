"""Style library + variant selector — phase 2 of the photo style variants.

Each style is a PURE transform ``(base, scene) -> CorrectionParams`` over the
measured photo-mode correction, clamped to STYLE-LAYER rails wider than the
corrective clamps — no new filter machinery; every variant renders through the
same ``_grade`` chain. Every non-natural style is a real look identity that
ADAPTS to the SceneProfile's color readings (chroma headroom, measured warmth,
dynamic range) instead of applying fixed deltas — fixed deltas rendered every
variant as the same theme on real photos.

The matte fade presets are the one group whose move is a SHAPED tone curve
rather than scalar fields: they set the three per-channel curve fields to one
identical knot list, which the existing applier renders as a master curve.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from sun_track import is_golden_elevation

from ..color_correction import (
    GAMMA_MAX,
    CorrectionParams,
)
from .scene import SceneProfile
from .subject import CONFIDENCE_MODERATE, SubjectCategory, SubjectReading

StyleTransform = Callable[[CorrectionParams, SceneProfile], CorrectionParams]

# Style-layer rails. Saturation/WB/contrast deliberately exceed
# compute_correction's corrective clamps (1.25 / ±0.05 / 0.10): those are
# measurement-layer taste, not renderer limits — the grade chain itself is
# unbounded (saturation is pure luma-preserving chroma scaling). A muted scene's
# base correction already sits near the corrective caps, so railing styles to
# them collapsed every variant into the base grade. Shadow/highlight keep the
# documented 0-20 band and gamma its 0.9-GAMMA_MAX rail.
SHADOW_LIFT_MAX = 20.0
HIGHLIGHT_PULL_MAX = 20.0
STYLE_SATURATION_CAP = 1.55  # bounds the doubled vivid push from a corrective-cap base
STYLE_WB_CAP = 0.10
STYLE_CONTRAST_CAP = 0.20
GAMMA_MIN = 0.9
# Black-point rails: crush matches the renderer's full negative authority; fade
# now spans the full BLACK_POINT_MAX — the sweep-chosen muted moody fade uses
# all of it (the old +16 "washed" ceiling predates the look-scale bake).
STYLE_BLACK_POINT_CRUSH_MAX = -10.0
STYLE_BLACK_POINT_FADE_MAX = 20.0
# B&W: chroma scaled by exactly zero — the one deliberate value under the desat floor.
MONO_SATURATION = 0.0
# Style-layer desat floor. The corrective SATURATION_PULLDOWN_MIN (0.80) sat
# ABOVE every look-scale desat target on typical bases and flattened moody's
# muted/rich lean into one value, so styles floor here instead; only mono goes lower.
STYLE_DESAT_FLOOR = 0.25

# Style moves — hand-calibrated at look scale via a strength sweep on a
# real sample (each style rendered at x1/x2/x3 delta amplification: x1 read
# nearly identical, x2 chosen, x3 too loud; these constants bake the x2
# column). Every result is bounded by the rails above at apply time.
# Scene-dependent scaling per style lives with its transform.
VIVID_SATURATION_GAIN = 1.24  # full-strength gain, toward STYLE_SATURATION_CAP
VIVID_SATURATION_FLOOR = 0.24  # additive floor so a near-cap base still moves
VIVID_HEADROOM_GAIN_FLOOR = 0.25  # push strength kept on a rich (headroom~0) frame
# Rich-frame rolloff. Below this headroom the frame is already near-saturated
# (a red close-up, not a muted vista); vivid's full intensity — chroma push,
# low-headroom contrast compensation, and black crush — cartoons and crushes it,
# so all three attenuate linearly to ~0 as headroom → 0. The knee sits below
# every calibration/bake reference (all headroom ≥ 0.85), so it retunes ONLY
# genuinely rich frames and leaves the calibration sweep untouched.
VIVID_RICH_KNEE = 0.4
VIVID_CONTRAST_BOOST = 0.10
VIVID_LOW_HEADROOM_CONTRAST = 0.08  # extra S-curve when saturation has no room
VIVID_SHADOW_SCALE = 0.0  # remove the lift entirely (x2 of the old halving)
VIVID_CRUSH = -10.0  # x2 target -12 pinned at the black-point authority floor
MOODY_FADE = 15.0  # matte output-floor lift (x2 target 20 scaled with the extra below)
MOODY_FADE_MUTED_EXTRA = 5.0  # muted lean: fade+extra land ON the +20 authority
MOODY_SATURATION_SCALE = 0.50
MOODY_SATURATION_SCALE_MUTED = 0.70  # ...and less extra desat
MOODY_SATURATION_SCALE_RICH = 0.30  # rich frames lean desat instead
MOODY_MUTED_HEADROOM_MIN = 0.9  # p95 sat <= 0.1 reads low-chroma
MOODY_COOL_SHIFT = 0.06  # blue-ward shadow/mid shift (teal sits in the shadows)
MOODY_TEAL_RED_CUT = 0.03  # small red cut completes the teal
MOODY_HIGH_NEUTRALIZE = 0.024  # highlights counter-shifted back toward neutral
MOODY_CONTRAST_BOOST = 0.07  # stronger mid contrast
MOODY_CONTRAST_MUTED_EXTRA = 0.03
# Warm-scene teal restraint. Moody is a fog/low-key LANDSCAPE look; its teal
# cross-cast lands on a warm close-up (a warm kitchen shot) as muddy brown, not
# a look — the cool shift fights the scene's warmth. When the scene reads warm
# (moody is here only as filler, never its favored scene), the WB moves scale
# down so it reads as a clean desaturated-cool grade instead of mud. Gated on
# warmth alone, so the neutral hazy landscape moody is bit-identical.
MOODY_WARM_SHIFT_SCALE = 0.35
GOLDEN_HIGH_WARM_SHIFT = 0.09  # warmth concentrated in the highlight range
GOLDEN_HIGH_BLUE_CUT = 0.06
GOLDEN_MID_WARM_SHIFT = 0.04  # gentler mid shift keeps shadows honest
GOLDEN_MID_BLUE_CUT = 0.024
GOLDEN_WARM_SCENE_SCALE = 0.4  # an already-golden shot must not cook (MID moves)
# Highlight moves keep more strength on a warm scene than the mid moves do: the
# golden LOOK is a warm-highlight glow, and throttling it to the mid scale (0.4)
# rendered golden < the distinctness floor from natural on warm frames, so it
# was dropped and a muddy filler backfilled — exactly where golden fits best.
# The mid throttle still guards against an all-over orange cast.
GOLDEN_WARM_HIGH_SCALE = 0.7
GOLDEN_WARM_FRACTION_HIGH = 0.5  # warm-hue chroma share that reads already-warm
GOLDEN_HIGHLIGHT_SOFTEN = 6.0  # extra pull softens the top end
# DR-tuned mono: a flat scene needs the S-curve and crush most, a wide one
# little. UNCHANGED by the look-scale bake — the sweep read mono as already
# structurally distinct at x1.
MONO_CONTRAST_BOOST_FLAT = 0.06
MONO_CONTRAST_BOOST_NORMAL = 0.04
MONO_CONTRAST_BOOST_WIDE = 0.02
MONO_CRUSH_FLAT = -8.0
MONO_CRUSH_NORMAL = -5.0
MONO_CRUSH_WIDE = -3.0
# Soft is hand-set past a blind double (a pure double still read weak).
SOFT_FADE = 12.0  # pastel matte floor
SOFT_GAMMA_LIFT = 0.14  # pastel: brighten mids
SOFT_CONTRAST_SCALE = 0.0  # x2 of the old halving flattens the S-curve away
SOFT_SATURATION_SCALE = 0.70
# White-balance bracket (daylight / cloudy / shade) — the camera preset ladder
# expressed as WB deltas over the MEASURED base, so the trio brackets it rather
# than replacing it: daylight is the reference-5500K look, a small cool step off
# a base metered on a scene that usually reads warmer; cloudy adds roughly a
# preset step of warmth and shade twice that, matching how the presets space on
# a real camera. Blue counter-moves at WB_BLUE_RATIO of red — slightly under a
# mirror, so the shift reads as temperature and not a magenta/green tint. Every
# step sits well inside STYLE_WB_CAP so the rails never clip the ladder's order.
WB_DAYLIGHT_SHIFT = -0.02
WB_CLOUDY_SHIFT = 0.03
WB_SHADE_SHIFT = 0.06
WB_BLUE_RATIO = -0.8
WB_HIGH_SCALE = 0.75  # highlights follow the mids, restrained so speculars stay clean
# Matte fade presets — a SHAPED tail move where black_point is a single LINEAR
# output-floor lift of the whole frame. Each preset sets curve_r == curve_g ==
# curve_b to ONE knot list, so the existing per-channel applier renders it
# through the same Fritsch-Carlson monotone interpolant the master curve uses:
# a master-shaped tone curve for free, no new applier — and one curve on all
# three channels means no colour cast, by construction. Knots are (x, y) pairs,
# x strictly ascending in [0, 1], the whole set monotone (a non-monotone set
# inverts the interpolant and destroys tone rather than shaping it).
# BOTH knot sets are shaped for WHERE THE CURVE ACTUALLY RUNS: _apply_channel_curves
# sits early in _grade — right after the master _apply_curves, BEFORE colorbalance,
# saturation and gamma — and at that point a real graded frame's tone sits in
# roughly [0, 0.6] (measured p99 ~0.51-0.58 on the reference NEFs). A knot set that
# only bites in the top decile is an identity on a real photo: the first cut of the
# knee rolled off past x=0.6 and rendered ~1.3 levels from the shoulder, one preset
# shipped twice. Both moves therefore land inside that occupied range.
# fmt: off
# Shoulder: a rolled TOE across the SHADOWS AND LOW MIDS. The floor lifts 0.11
# (≈28 8-bit levels) at black, is still ≈17 levels at x=0.25 and decays to ~0 by
# x=0.5, so the shadows ride a soft ramp instead of the whole frame washing out;
# white pins at 1.0, highlights untouched.
MATTE_SHOULDER_CURVE = (
    0.0, 0.11,  0.1, 0.19,  0.25, 0.315,  0.5, 0.525,  1.0, 1.0,
)
# Knee: a compressed SHOULDER over the UPPER MIDS — the opposite tonal idea, and
# in the band a real frame occupies. Identity through x=0.15 (black stays true),
# then the slope falls 1.0 -> 0.61 so the top of the histogram rolls off early:
# ≈9 levels down at x=0.45, ≈22 at x=0.6, into a printed white rather than paper.
MATTE_KNEE_CURVE = (
    0.0, 0.0,  0.15, 0.15,  0.3, 0.29,  0.45, 0.415,  0.6, 0.515,  1.0, 0.76,
)
# fmt: on
MATTE_SHOULDER_DESAT = 0.90  # a faded print reads slightly less saturated
MATTE_KNEE_DESAT = 0.90  # the curve, not the scalars, carries each identity
MATTE_KNEE_GAMMA_LIFT = 0.10  # the rolled top darkens the frame; hold the mids

_WB_FIELDS = ("white_balance_r", "white_balance_b", "white_balance_r_high", "white_balance_b_high")

# Strength lever. Every style is a DELTA over the measured base, so one factor
# amplifies the whole look — and the style-layer rails scale with it (railing an
# amplified look back to the x1 caps is what collapsed the variants pre-bake).
# x1 is the shipped bake, returned untouched. The hard ceilings below are
# RENDERER authority, not taste: they bind at every strength.
DEFAULT_STRENGTH = 1.0
SATURATION_HARD_CAP = 2.0  # past this chroma clips to the primaries, not "more colour"
WB_HARD_CAP = 0.20
CONTRAST_MONOTONE_MARGIN = 0.01
# Look fields the lever scales. denoise/sharpen are metering budgets measured
# off the frame, never look moves, and ride through from the base untouched.
_LOOK_FIELDS = (
    "shadow_lift",
    "highlight_pull",
    "saturation_mult",
    "contrast_curve",
    *_WB_FIELDS,
    "gamma",
    "black_point",
)


def _rail(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _styled(base: CorrectionParams, **changes: float) -> CorrectionParams:
    """Apply a style's field changes, bounded to the style-layer rails."""
    params = replace(base, **changes)
    wb = {f: _rail(getattr(params, f), -STYLE_WB_CAP, STYLE_WB_CAP) for f in _WB_FIELDS}
    return replace(
        params,
        shadow_lift=_rail(params.shadow_lift, 0.0, SHADOW_LIFT_MAX),
        highlight_pull=_rail(params.highlight_pull, 0.0, HIGHLIGHT_PULL_MAX),
        saturation_mult=_rail(params.saturation_mult, MONO_SATURATION, STYLE_SATURATION_CAP),
        contrast_curve=_rail(params.contrast_curve, 0.0, STYLE_CONTRAST_CAP),
        gamma=_rail(params.gamma, GAMMA_MIN, GAMMA_MAX),
        black_point=_rail(
            params.black_point, STYLE_BLACK_POINT_CRUSH_MAX, STYLE_BLACK_POINT_FADE_MAX
        ),
        **wb,
    )


def _contrast_ceiling(shadow_lift: float, highlight_pull: float) -> float:
    """Largest S-curve offset that keeps the curve applier's control points increasing.

    Mirrors the four points ``_apply_curves`` builds: the two midtone anchors
    move by ±contrast toward the shadow floor and the highlight ceiling, so the
    end with less room bounds the offset. Past it the monotone interpolant
    inverts and the grade destroys tone instead of shaping it. The shipped x1
    cap already sits inside this limit; it only binds once the lever amplifies.
    """
    wh_out = 1.0 - highlight_pull / 255.0
    wh_top = wh_out * (0.95 + highlight_pull / 500.0) if highlight_pull >= 5.0 else wh_out
    shadow_room = 0.25 - (shadow_lift / 255.0) * 0.5
    highlight_room = wh_top - 0.75 + (1.0 - wh_out) * 0.5
    return max(0.0, min(shadow_room, highlight_room) - CONTRAST_MONOTONE_MARGIN)


def _amplified(
    base: CorrectionParams, styled: CorrectionParams, strength: float
) -> CorrectionParams:
    """Scale a style's delta over ``base`` by ``strength``, bounded to rails that
    scale with it up to renderer authority.

    Saturation floors at the style desat floor so an amplified moody can never
    desaturate into mono — only mono's exact-zero chroma sits lower. black_point
    and gamma keep their x1 rails: both already pin at the renderer clamp. The
    styled CURVE fields are carried through unscaled — a knot list has no
    meaningful linear amplification, and rebuilding from ``base`` alone would
    silently DROP a matte preset's whole shape at any strength but x1.
    """
    changes = {
        f: getattr(base, f) + strength * (getattr(styled, f) - getattr(base, f))
        for f in _LOOK_FIELDS
    }
    params = replace(
        base, **changes, curve_r=styled.curve_r, curve_g=styled.curve_g, curve_b=styled.curve_b
    )
    wb_cap = min(WB_HARD_CAP, STYLE_WB_CAP * strength)
    wb = {f: _rail(getattr(params, f), -wb_cap, wb_cap) for f in _WB_FIELDS}
    sat_floor = MONO_SATURATION if styled.saturation_mult <= MONO_SATURATION else STYLE_DESAT_FLOOR
    sat_cap = min(SATURATION_HARD_CAP, 1.0 + (STYLE_SATURATION_CAP - 1.0) * strength)
    shadow_lift = _rail(params.shadow_lift, 0.0, SHADOW_LIFT_MAX)
    highlight_pull = _rail(params.highlight_pull, 0.0, HIGHLIGHT_PULL_MAX)
    return replace(
        params,
        shadow_lift=shadow_lift,
        highlight_pull=highlight_pull,
        saturation_mult=_rail(params.saturation_mult, sat_floor, max(sat_floor, sat_cap)),
        contrast_curve=_rail(
            params.contrast_curve,
            0.0,
            min(STYLE_CONTRAST_CAP * strength, _contrast_ceiling(shadow_lift, highlight_pull)),
        ),
        gamma=_rail(params.gamma, GAMMA_MIN, GAMMA_MAX),
        black_point=_rail(
            params.black_point, STYLE_BLACK_POINT_CRUSH_MAX, STYLE_BLACK_POINT_FADE_MAX
        ),
        **wb,
    )


def _natural(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    """Identity over the measured photo-mode correction — the anchor variant."""
    return replace(base)


def _vivid(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # Punchy identity: black crush + saturation push scaled by how much chroma
    # the frame can absorb (headroom~1 muted -> full strength, ~0 rich -> the
    # gain floor); what the frame can't take in chroma it takes in contrast.
    # The additive floor (headroom-scaled) keeps a near-cap base moving;
    # _styled's rail supplies the min() against STYLE_SATURATION_CAP.
    # Below the rich knee the frame is already near-saturated; the same rolloff
    # attenuates the whole vivid intensity — chroma push, the low-headroom
    # contrast compensation, AND the black crush — so a red close-up gets a
    # clean pop instead of the crushed-granite look the contrast+crush produced
    # at extreme strength. rich_att is 1.0 at/above the knee (bake untouched).
    headroom = _rail(scene.chroma_headroom, 0.0, 1.0)
    rich_att = min(1.0, headroom / VIVID_RICH_KNEE)
    strength = (VIVID_HEADROOM_GAIN_FLOOR + (1.0 - VIVID_HEADROOM_GAIN_FLOOR) * headroom) * rich_att
    return _styled(
        base,
        saturation_mult=max(
            base.saturation_mult * (1.0 + (VIVID_SATURATION_GAIN - 1.0) * strength),
            base.saturation_mult + VIVID_SATURATION_FLOOR * strength,
        ),
        contrast_curve=base.contrast_curve
        + VIVID_CONTRAST_BOOST
        + VIVID_LOW_HEADROOM_CONTRAST * (1.0 - headroom) * rich_att,
        shadow_lift=base.shadow_lift * VIVID_SHADOW_SCALE,
        black_point=base.black_point + VIVID_CRUSH * rich_att,
    )


def _moody(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # Matte/teal identity: faded floor, cool shadows/mids with highlights
    # counter-shifted toward neutral, desaturated, stronger mids. Muted or
    # low-chroma scenes lean tone (deeper fade, more contrast, less extra
    # desat); rich frames lean desat. Desaturation floors at the style-layer
    # desat floor; only mono sits below.
    muted = scene.sat_profile == "muted" or scene.chroma_headroom >= MOODY_MUTED_HEADROOM_MIN
    fade = MOODY_FADE + (MOODY_FADE_MUTED_EXTRA if muted else 0.0)
    contrast = MOODY_CONTRAST_BOOST + (MOODY_CONTRAST_MUTED_EXTRA if muted else 0.0)
    if muted:
        sat_scale = MOODY_SATURATION_SCALE_MUTED
    elif scene.sat_profile == "rich":
        sat_scale = MOODY_SATURATION_SCALE_RICH
    else:
        sat_scale = MOODY_SATURATION_SCALE
    # On a warm scene the teal cross-cast reads as mud, not a look — restrain the
    # WB moves (never the fade/desat/contrast that carry the matte identity).
    shift = MOODY_WARM_SHIFT_SCALE if scene.warmth == "warm" else 1.0
    return _styled(
        base,
        black_point=base.black_point + fade,
        saturation_mult=max(STYLE_DESAT_FLOOR, base.saturation_mult * sat_scale),
        white_balance_b=base.white_balance_b + MOODY_COOL_SHIFT * shift,
        white_balance_r=base.white_balance_r - MOODY_TEAL_RED_CUT * shift,
        white_balance_b_high=base.white_balance_b_high - MOODY_HIGH_NEUTRALIZE * shift,
        white_balance_r_high=base.white_balance_r_high + MOODY_HIGH_NEUTRALIZE * shift,
        contrast_curve=base.contrast_curve + contrast,
    )


def _golden(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # Warm-highlight identity: the push concentrates in the highlight range
    # with a gentler mid shift, scaled down on a measured-warm scene — an
    # already-golden shot must not cook.
    warm = scene.warmth == "warm" or scene.warm_fraction >= GOLDEN_WARM_FRACTION_HIGH
    high_scale = GOLDEN_WARM_HIGH_SCALE if warm else 1.0
    mid_scale = GOLDEN_WARM_SCENE_SCALE if warm else 1.0
    return _styled(
        base,
        white_balance_r_high=base.white_balance_r_high + GOLDEN_HIGH_WARM_SHIFT * high_scale,
        white_balance_b_high=base.white_balance_b_high - GOLDEN_HIGH_BLUE_CUT * high_scale,
        white_balance_r=base.white_balance_r + GOLDEN_MID_WARM_SHIFT * mid_scale,
        white_balance_b=base.white_balance_b - GOLDEN_MID_BLUE_CUT * mid_scale,
        highlight_pull=base.highlight_pull + GOLDEN_HIGHLIGHT_SOFTEN,
    )


def _mono(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # DR-tuned: a flat frame gets the strongest S-curve AND crush, a wide one mild.
    if scene.dynamic_range == "flat":
        boost, crush = MONO_CONTRAST_BOOST_FLAT, MONO_CRUSH_FLAT
    elif scene.dynamic_range == "wide":
        boost, crush = MONO_CONTRAST_BOOST_WIDE, MONO_CRUSH_WIDE
    else:
        boost, crush = MONO_CONTRAST_BOOST_NORMAL, MONO_CRUSH_NORMAL
    return _styled(
        base,
        saturation_mult=MONO_SATURATION,
        contrast_curve=base.contrast_curve + boost,
        black_point=base.black_point + crush,
    )


def _soft(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # Pastel identity: matte fade + gamma lift over a flattened curve and desat.
    return _styled(
        base,
        black_point=base.black_point + SOFT_FADE,
        gamma=base.gamma + SOFT_GAMMA_LIFT,
        contrast_curve=base.contrast_curve * SOFT_CONTRAST_SCALE,
        saturation_mult=max(STYLE_DESAT_FLOOR, base.saturation_mult * SOFT_SATURATION_SCALE),
    )


def _white_balanced(base: CorrectionParams, shift: float) -> CorrectionParams:
    """Temperature-only move: mids by ``shift``, highlights along with them."""
    return _styled(
        base,
        white_balance_r=base.white_balance_r + shift,
        white_balance_b=base.white_balance_b + shift * WB_BLUE_RATIO,
        white_balance_r_high=base.white_balance_r_high + shift * WB_HIGH_SCALE,
        white_balance_b_high=base.white_balance_b_high + shift * WB_BLUE_RATIO * WB_HIGH_SCALE,
    )


def _daylight(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # Cool end of the bracket — the reference-daylight rendering.
    return _white_balanced(base, WB_DAYLIGHT_SHIFT)


def _cloudy(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # A preset step warmer, the way a camera compensates overcast light.
    return _white_balanced(base, WB_CLOUDY_SHIFT)


def _shade(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # Warm end — open shade is the bluest light the presets correct for.
    return _white_balanced(base, WB_SHADE_SHIFT)


def _matte_curved(base: CorrectionParams, curve: tuple[float, ...], **changes: float):
    """A style whose fade is a SHAPED curve: the scalar moves take the style-layer
    rails through ``_styled``, then the same knot list lands on all three channel
    curve fields (the curve itself is bounded by its own knots, in [0, 1]).

    The strength lever does NOT scale the shape: ``_LOOK_FIELDS`` holds scalars
    only, so an amplified preset pushes its desat/gamma move and keeps the baked
    curve as-is (an explicit operator ``--curve-r/g/b`` replaces it outright).
    """
    return replace(_styled(base, **changes), curve_r=curve, curve_g=curve, curve_b=curve)


def _matte_shoulder(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # Rolled-toe fade: shadows lift onto a soft ramp, highlights held at white.
    return _matte_curved(
        base,
        MATTE_SHOULDER_CURVE,
        saturation_mult=max(STYLE_DESAT_FLOOR, base.saturation_mult * MATTE_SHOULDER_DESAT),
    )


def _matte_knee(base: CorrectionParams, scene: SceneProfile) -> CorrectionParams:
    # Printed fade: the top end rolls off early, mids held by a small gamma lift.
    return _matte_curved(
        base,
        MATTE_KNEE_CURVE,
        saturation_mult=max(STYLE_DESAT_FLOOR, base.saturation_mult * MATTE_KNEE_DESAT),
        gamma=base.gamma + MATTE_KNEE_GAMMA_LIFT,
    )


# Declaration order IS the selector's tie-break order. The matte presets and the
# WB trio append after the six ranked styles and are deliberately absent from
# _FAVORED, so they are never PREFERRED and auto-selection is bit-identical;
# both groups are operator-opt-in, and tail the ladder a backfill walks.
STYLES: dict[str, StyleTransform] = {
    "natural": _natural,
    "vivid": _vivid,
    "moody": _moody,
    "golden": _golden,
    "mono": _mono,
    "soft": _soft,
    "matte_shoulder": _matte_shoulder,
    "matte_knee": _matte_knee,
    "daylight": _daylight,
    "cloudy": _cloudy,
    "shade": _shade,
}

# Natural correction vs artistic restyling. Natural colour correction is the
# DEFAULT for real travel/nature photography and always anchors `select_styles`;
# the white-balance trio is correction too — three declarative temperature steps
# around the MEASURED base, the bracket a photographer would set in camera. Every
# other registered style is a look identity: a deliberate artistic restyling of
# the frame, shipped as an ADDITIONAL variant beside the natural one, never in
# place of it (`plan_variants` renders each style to its own file, so the
# naturally-corrected original is always preserved).
NATURAL_STYLES = frozenset({"natural", "daylight", "cloudy", "shade"})


def is_artistic_style(name: str) -> bool:
    """True when ``name`` is an artistic restyling rather than natural correction.

    An unregistered name raises rather than guessing — a caller labelling an
    output it cannot identify would be exactly the mislabelling this answers.
    """
    if name not in STYLES:
        raise ValueError(f"unknown style {name!r}")
    return name not in NATURAL_STYLES


# Suitability weights. Suppression sinks a style below every unfavored one.
SCORE_BASE = 1.0
SCORE_FAVORED = 2.0
SCORE_LEANING = 1.0
SCORE_SUPPRESSED = -10.0
# Vivid needs chroma room to differentiate: favored only when the scene is
# bright AND clear AND can absorb the push; a muted scene failing that test
# takes the penalty — below every unpenalized style, strictly above the
# foggy/near-mono suppression floor.
VIVID_HEADROOM_MIN = 0.5
VIVID_MUTED_PENALTY = 2.0

# Golden hour by CAPTURE TIME, the half of golden the pixels can't show: low
# sun still lights a shaded or backlit subject that measures neutral. Honest
# limitation — EXIF ``DateTimeOriginal`` is camera-LOCAL wall time carrying no
# timezone, so this is a fixed approximation of the low-sun hours, not a solar
# calculation: it does not move with latitude, season or DST. Start hour
# inclusive, end hour exclusive (17 <= h < 21 is 17:00 through 20:59).
# When a GPX track supplied a real ``solar_elevation_deg``, the golden
# predicate below prefers ``is_golden_elevation`` and this window never runs —
# the wall clock is the NO-TRACK fallback only, unchanged.
GOLDEN_HOUR_MORNING_START = 5
GOLDEN_HOUR_MORNING_END = 9
GOLDEN_HOUR_EVENING_START = 17
GOLDEN_HOUR_EVENING_END = 21


def _is_golden_hour(hour: int | None) -> bool:
    """``None`` — every source without capture EXIF — is never golden hour, so
    selection stays bit-identical to before this rule on those paths."""
    if hour is None:
        return False
    return (
        GOLDEN_HOUR_MORNING_START <= hour < GOLDEN_HOUR_MORNING_END
        or GOLDEN_HOUR_EVENING_START <= hour < GOLDEN_HOUR_EVENING_END
    )


# Favored-scene predicate per candidate; vivid's suppression/leaning live in _score.
_FAVORED: dict[str, Callable[[SceneProfile], bool]] = {
    "vivid": lambda s: (
        s.exposure_class == "bright"
        and s.haze == "clear"
        and s.chroma_headroom >= VIVID_HEADROOM_MIN
    ),
    "moody": lambda s: s.haze in ("hazy", "foggy") or s.exposure_class == "low_key",
    "golden": lambda s: (
        s.warmth == "warm"
        or (
            is_golden_elevation(s.solar_elevation_deg)
            if s.solar_elevation_deg is not None
            else _is_golden_hour(s.capture_hour)
        )
    ),
    "mono": lambda s: s.is_near_monochrome or s.haze == "foggy",
    "soft": lambda s: s.exposure_class == "bright" and s.sat_profile in ("muted", "normal"),
}

# Leaning predicates — one SCORE_LEANING nudge toward structurally-different
# styles on scenes whose chroma can't differentiate (muted). Applied only when
# the style isn't already favored, so a leaning never reorders a favored trio
# (foggy keeps moody before mono). Mono carries NO leaning: its muted+flat
# nudge fired on corpus-typical light — judged wrong on all nine offers of the
# labeled corpus, never best — so mono ranks by _FAVORED alone. Soft also
# leans on a warm scene: warm is not moody's scene (moody is a fog/low-key
# landscape look), so when golden — the warm favorite — is later
# distinctness-dropped for being too close to natural, soft is the airy
# alternative that backfills instead of a muddy moody. It never reorders the
# warm favored pick: golden still outscores it, and on a warm RICH scene vivid
# ties soft and wins the declaration tie-break.
_LEANING: dict[str, Callable[[SceneProfile], bool]] = {
    "soft": lambda s: s.sat_profile == "muted" or s.warmth == "warm",
}


def _score(name: str, scene: SceneProfile) -> float:
    if name == "vivid" and (scene.haze == "foggy" or scene.is_near_monochrome):
        return SCORE_SUPPRESSED
    favored = _FAVORED[name](scene)
    score = SCORE_BASE + (SCORE_FAVORED if favored else 0.0)
    if name == "vivid":
        if scene.sat_profile == "rich":
            score += SCORE_LEANING
        elif scene.sat_profile == "muted" and not favored:
            score -= VIVID_MUTED_PENALTY
    elif not favored and name in _LEANING and _LEANING[name](scene):
        score += SCORE_LEANING
    return score


# Floor below which a subject reading is too weak to route by. Named
# and pinned to subject.py's own calibration rather than a bare number:
# CONFIDENCE_MODERATE is "two agreeing EXIF facts" -- it clears every operator
# correction, every grouped/tracked-night/sequenced inference, and the plain
# two-fact PORTRAIT and action-shutter WILDLIFE readings, but excludes every
# single-fact WEAK guess (LANDSCAPE always; WILDLIFE and LONG_EXPOSURE on
# their weakest path) and UNKNOWN (confidence 0.0) -- a weak reading must
# never outrank the measured SceneProfile ranking below.
SUBJECT_ROUTING_CONFIDENCE_MIN = CONFIDENCE_MODERATE

# Category -> the one style its EXIF evidence supports promoting toward
# ``natural``, each with the fact that earns it. A category absent here is
# left unrouted ON PURPOSE: its EXIF proves optics or action, never light
# quality or dynamic range, so no style preference is evidenced --
# WILDLIFE (long lens / fast shutter says nothing about colour), PORTRAIT
# (subject-separating aperture says nothing about colour), LONG_EXPOSURE
# (a bright tripod frame says nothing about colour), and LANDSCAPE (also
# never clears the floor: it is subject.py's own weakest reading).
SUBJECT_STYLE_ROUTING: dict[SubjectCategory, str] = {
    # A tracked/dark high-ISO sky IS a low-key frame -- moody's own scene
    # predicate (``exposure_class == "low_key"``) already favors exactly this;
    # the subject reading just says WHY the frame is low-key.
    SubjectCategory.ASTRO: "moody",
    # The bracket detector grouped these frames because one exposure could not
    # hold the scene's range -- matte_shoulder's highlight roll-off is built
    # for the dynamic range that grouping just proved.
    SubjectCategory.EXPOSURE_BRACKET: "matte_shoulder",
}


def select_styles(
    scene: SceneProfile, count: int = 3, subject: SubjectReading | None = None
) -> list[str]:
    """Pick ``count`` style names — ``natural`` always anchors; the rest rank by
    suitability score, ties broken by ``STYLES`` declaration order, so one
    profile yields one list, always.

    Past the ranked styles the list continues into the OPT-IN TAIL — every
    registered style no predicate ranks, in declaration order. Asking for the
    whole ladder (``count=len(STYLES)``, what the distinctness backfill does)
    must hand back the whole ladder, or the backfill exhausts against styles it
    was never allowed to consider and ships near-identical variants instead.
    The tail is APPENDED, never scored into the ranking, so every count up to
    the ranked names is bit-identical: reachable, never preferred.

    ``subject`` is optional and additive: when its category is in
    ``SUBJECT_STYLE_ROUTING`` and its confidence clears
    ``SUBJECT_ROUTING_CONFIDENCE_MIN``, the routed style is moved to right
    after ``natural`` — it may reorder what comes after ``natural``, never
    displace it. ``None``, ``UNKNOWN``, an unrouted category, or a reading
    below the floor leave the ladder bit-identical to no-subject selection.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    names = list(STYLES)
    ranked = sorted(_FAVORED, key=lambda name: (-_score(name, scene), names.index(name)))
    tail = [n for n in names if n != "natural" and n not in _FAVORED]
    ladder = ["natural", *ranked, *tail]
    routed = SUBJECT_STYLE_ROUTING.get(subject.category) if subject is not None else None
    if routed is not None and subject.confidence >= SUBJECT_ROUTING_CONFIDENCE_MIN:
        rest = [n for n in ladder if n != routed]
        ladder = [rest[0], routed, *rest[1:]]
    return ladder[:count]


def apply_style(
    name: str,
    base: CorrectionParams,
    scene: SceneProfile,
    strength: float = DEFAULT_STRENGTH,
) -> CorrectionParams:
    """Run one named style transform at ``strength``; an unknown name raises
    ``ValueError``, as does a non-positive strength."""
    if strength <= 0.0:
        raise ValueError(f"strength must be > 0, got {strength}")
    try:
        transform = STYLES[name]
    except KeyError:
        raise ValueError(f"Unknown style {name!r}; available: {', '.join(STYLES)}") from None
    styled = transform(base, scene)
    if strength == DEFAULT_STRENGTH:
        return styled
    return _amplified(base, styled, strength)
