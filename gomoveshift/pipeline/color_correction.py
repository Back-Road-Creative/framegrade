"""Color correction — parameter computation, LUT generation.

Split from color.py.
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .color_analysis import (
    FrameStats,
    SegmentStats,
    aggregate_segment_stats,
)
from .quality import denoise_strength_for, sharpen_amount_for

logger = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────────


@dataclass
class CorrectionParams:
    """Color correction parameters derived from histogram analysis."""

    shadow_lift: float = 0.0  # Levels to lift shadows (0-20)
    highlight_pull: float = 0.0  # Levels to pull highlights (0-20)
    saturation_mult: float = 1.0  # Saturation multiplier (0.80-1.25)
    vibrance: float = 0.0  # Low-sat-weighted chroma boost (photo develop only; 0 = OFF)
    contrast_curve: float = 0.0  # S-curve midtone contrast (0.0-0.15)
    white_balance_r: float = 0.0  # Red channel shift (-0.05 to +0.05)
    white_balance_b: float = 0.0  # Blue channel shift (-0.05 to +0.05)
    # The green-magenta axis, the other half of what white balance means: a
    # fluorescent or forest-canopy cast lives here and on no other control.
    # Photo develop only — no video producer sets it, so 0.0 keeps every video
    # output and baked LUT bit-identical (see _apply_colorbalance).
    white_balance_g: float = 0.0  # Green channel shift (+ green / - magenta)
    white_balance_r_high: float = 0.0  # Red highlight-range shift (-0.05 to +0.05)
    white_balance_b_high: float = 0.0  # Blue highlight-range shift (-0.05 to +0.05)
    white_balance_g_high: float = 0.0  # Green highlight-range shift (photo only)
    gamma: float = 1.0  # Gamma correction (0.9-1.30)
    denoise_strength: float = 0.0  # hqdn3d strength (0-6, 0=off)
    sharpen_amount: float = 0.0  # unsharp strength (0-1.5, 0=off)
    sharpen_radius: float = 0.0  # unsharp radius in px (0 = filter default)
    # User opt-outs (--no-sharpen / --no-denoise), recorded so state shows
    # the zero was a choice, not a measurement.
    sharpen_disabled: bool = False
    denoise_disabled: bool = False
    black_point: float = 0.0  # Signed black-point move in 8-bit levels (photo styles only)
    # Per-hue-band HSL controls (photo develop only) over fixed anchor hues — red
    # 0°, yellow 60°, green 120°, cyan 180°, blue 240°, magenta 300°, or the same
    # with ORANGE 30° second. Each is empty (OFF), length 6 or length 7; the six
    # form keeps the pre-orange anchors, so shipped recipes stay bit-identical.
    # All-neutral ⇒ _apply_hsl is a bit-identical no-op; the video path and baked
    # LUTs never set them and stay untouched.
    hue_shift: tuple[float, ...] = ()  # per-band hue rotation, degrees (0 = neutral)
    hue_sat: tuple[float, ...] = ()  # per-band saturation multiplier (1.0 = neutral)
    hue_lum: tuple[float, ...] = ()  # per-band value/luminance multiplier (1.0 = neutral)
    # Split-toning (photo develop only): three independent signed (r,g,b) tints
    # added by BT.709 luma weight — one per tonal range, the shadows, the midtones
    # (where a photograph's tone actually sits) and the highlights. Each empty (OFF)
    # or length 3; all-empty ⇒ _apply_splittone is a bit-identical no-op (excluded
    # from params_to_lut3d_cube).
    splittone_shadow: tuple[float, ...] = ()
    splittone_midtone: tuple[float, ...] = ()
    splittone_highlight: tuple[float, ...] = ()
    # ── CURVES ── Operator-drawn tone curves (photo develop only). Each is empty
    # (OFF) or a FLAT list of alternating x,y control points (even length >= 4,
    # i.e. >= 2 points; x strictly ascending in [0, 1], y in [0, 1]).
    # All-empty ⇒ _apply_luma_curve / _apply_channel_curves are bit-identical
    # no-ops; the video path and baked LUTs never set them and stay untouched.
    # curve_l is the MASTER tone curve — one shape run on R, G and B alike, so it
    # shapes tone without a colour cast by construction; the three channel curves
    # are the cross-process move that rides on top of it.
    curve_l: tuple[float, ...] = ()  # master luma tone curve, applied to all three
    curve_r: tuple[float, ...] = ()  # additional red-channel tone curve
    curve_g: tuple[float, ...] = ()  # additional green-channel tone curve
    curve_b: tuple[float, ...] = ()  # additional blue-channel tone curve
    # Monochrome conversion (photo develop only). mono_mix is empty (OFF) or the
    # length-3 custom R/G/B luma weights of a channel mixer — a red-weighted mix
    # is the classic red filter that darkens a blue sky. duotone is empty (OFF) or
    # the length-6 (shadow_r,g,b, high_r,g,b) ends of a luma ramp in [0, 1]
    # (sepia/selenium/cyanotype). Both empty ⇒ _apply_monomix is a bit-identical
    # no-op; neither reaches params_to_lut3d_cube, so video output is untouched.
    # duotone_strength in [0, 1] dials that ramp back toward the plain monochrome
    # (1.0 = the full ramp, the shipped default and bit-identical; 0.0 = no tone).
    # duotone_mid is empty (OFF, a straight two-ender) or a third (r,g,b) anchor
    # the ramp passes through at mid luma — a tritone.
    mono_mix: tuple[float, ...] = ()
    duotone: tuple[float, ...] = ()
    duotone_strength: float = 1.0
    duotone_mid: tuple[float, ...] = ()
    # Reference-image colour match (photo develop only): the six numbers of a
    # Reinhard Lab transfer — a chosen hero frame's (L, a, b) MEANS followed by
    # its (L, a, b) STANDARD DEVIATIONS, measured once by ``lab_mean_std``.
    # Empty (OFF) ⇒ _apply_reference_match is a bit-identical no-op; it never
    # reaches params_to_lut3d_cube, so video output is untouched.
    reference_lab: tuple[float, ...] = ()
    # Operator exposure/recovery (photo develop only). ``exposure_ev`` is a stops
    # gain applied FIRST in ``develop._grade``, ahead of the tone curve, so it
    # moves the whole frame the way a camera's exposure dial does (0 = OFF).
    # ``shadow_recovery``/``highlight_recovery`` ADD onto the measured
    # ``shadow_lift``/``highlight_pull`` inside ``_apply_curves`` rather than
    # replacing them, exactly as the white-balance fields add onto the measured
    # cast — an operator lifting shadows further must not discard the metering
    # that already lifted them some. Each 0.0 (OFF) keeps both this and the
    # video path (which never sets them) bit-identical; excluded from
    # ``params_to_lut3d_cube`` is unnecessary since 0.0 is already a no-op there.
    exposure_ev: float = 0.0
    shadow_recovery: float = 0.0
    highlight_recovery: float = 0.0

    def __post_init__(self) -> None:
        for name in ("hue_shift", "hue_sat", "hue_lum"):
            band = getattr(self, name)
            if band and len(band) not in (6, 7):
                raise ValueError(
                    f"{name} must be empty, length 6 (red/yellow/green/cyan/blue/magenta) "
                    f"or length 7 (orange 30° second), got length {len(band)}"
                )
        for name in ("splittone_shadow", "splittone_midtone", "splittone_highlight"):
            tint = getattr(self, name)
            if tint and len(tint) != 3:
                raise ValueError(
                    f"{name} must be empty or length 3 (signed r,g,b tint), got length {len(tint)}"
                )
        for name in ("curve_l", "curve_r", "curve_g", "curve_b"):
            pts = getattr(self, name)
            if not pts:
                continue
            if len(pts) % 2 or len(pts) < 4:
                raise ValueError(
                    f"{name} must be an even-length flat list of >= 2 (x, y) control "
                    f"points, got length {len(pts)}"
                )
            xs = pts[0::2]
            if any(b <= a for a, b in zip(xs, xs[1:])):
                raise ValueError(f"{name} x control points must be strictly ascending, got {xs}")
        mix, duo = self.mono_mix, self.duotone
        if mix and len(mix) != 3:
            raise ValueError(f"mono_mix must be empty or length 3 (r,g,b weights), got {len(mix)}")
        # Deliberately NOT normalized — a sum away from 1 is a legitimate exposure
        # move — but a non-positive sum inverts or blacks out the frame.
        if mix and sum(mix) <= 0.0:
            raise ValueError(f"mono_mix weights must sum to more than 0, got {mix}")
        if duo and len(duo) != 6:
            raise ValueError(
                f"duotone must be empty or length 6 (shadow, high rgb), got {len(duo)}"
            )
        if any(not 0.0 <= v <= 1.0 for v in duo):
            raise ValueError(f"duotone values must each be in [0, 1], got {duo}")
        if not 0.0 <= self.duotone_strength <= 1.0:
            raise ValueError(f"duotone_strength must be in [0, 1], got {self.duotone_strength}")
        mid = self.duotone_mid
        if mid and (len(mid) != 3 or any(not 0.0 <= v <= 1.0 for v in mid)):
            raise ValueError(f"duotone_mid must be empty or length 3 r,g,b in [0, 1], got {mid}")
        # A mid anchor with no ramp to anchor would render as nothing at all.
        if mid and not duo:
            raise ValueError("duotone_mid needs duotone (the ramp's two ends)")
        ref = self.reference_lab
        if ref and len(ref) != 6:
            raise ValueError(
                f"reference_lab must be empty or length 6 (Lab means then standard "
                f"deviations), got length {len(ref)}"
            )
        if any(s < 0.0 for s in ref[3:]):
            raise ValueError(f"reference_lab standard deviations must be >= 0, got {ref[3:]}")


# Black-point tone move (photo styles only) — signed 8-bit levels, applied as
# the final step of _apply_curves. POSITIVE lifts the OUTPUT floor (matte
# fade); NEGATIVE cuts the INPUT floor (crushed blacks). The video ramps never
# set the field, and under MIN_USEFUL the step is skipped entirely so video
# output — including LUTs baked through the same appliers — stays bit-identical.
BLACK_POINT_MIN = -10.0
BLACK_POINT_MAX = 20.0
BLACK_POINT_MIN_USEFUL = 0.5


# ── Correction Parameter Generation ──────────────────────────────────


# Deep-underexposure headroom. The fixed gamma (1.15) and shadow-lift (15)
# ceilings both saturate below mid_tone 35, so a nearly-black dusk frame used to
# receive exactly the same correction as a merely dim one — no headroom at the
# one place it is most needed. Below the knee the correction now scales with the
# measured deficit; at and above it the math is bit-identical to before, so
# ordinary footage regrades to the same pixels.
DEEP_UNDEREXPOSURE_MID = 35.0
DEEP_SHADOW_LIFT_BONUS = 5.0  # keeps the ceiling at the documented 20 levels
DEEP_GAMMA_BONUS = 0.07  # continues the base slope (35/500) past the knee
GAMMA_MAX = 1.30  # safety rail; unreachable for mid_tone >= 0

# Denoise ramp. black_point is a brightness proxy for sensor noise: the darker
# the scene the more gain was applied. The old binary step (<10 -> 5.0) left
# visibly noisy dusk/overcast footage at black_point 10-25 with exactly zero
# denoise. The zero-point sits at 20, safely below typical daylight black
# points (~27), so correctly exposed daylight sources skip the denoise pass and
# its encode-time cost. Below MIN_USEFUL the hqdn3d pass costs encode time for
# an invisible effect, so it snaps off.
DENOISE_MAX_STRENGTH = 5.0
DENOISE_BLACK_POINT_FULL = 8.0
DENOISE_BLACK_POINT_OFF = 20.0
DENOISE_MIN_USEFUL = 0.5

# Contrast S-curve. The old trigger was |mid_tone - 128|, which assumes a
# mid-gray median means "this image needs contrast" — false for a wide-histogram
# source that already has plenty.
# A global `contrast_curve = 0.0` would keep already-contrasty footage from
# getting a curve, but it would also deny the curve to the flat sources that
# are the entire reason the dimension exists.
# The trigger is the measured white-black SPREAD — the signal that actually
# separates flat footage from contrasty footage. The ramp reaches zero at 130,
# well below the spread of a wide-histogram daylight source, so such footage
# regrades bit-identically; peak strength stays low enough that a curve is never
# visibly overdone. Below MIN_USEFUL the curve is imperceptible and snaps off
# rather than adding a no-op filter.
CONTRAST_SPREAD_OFF = 130.0
CONTRAST_SPREAD_FULL = 90.0
CONTRAST_CURVE_MAX = 0.06
CONTRAST_MIN_USEFUL = 0.005

# Dehaze. Haze deposits a milky veil that lifts the black point AND compresses
# the spread. The S-curve above already fires on the compression, so a naive
# "add contrast for flat footage" term would DOUBLE-apply and risk crushing the
# shadows. Instead the dehaze increment keys on the ADDITIONAL signal haze
# carries that plain flatness does not — the lift of the black point above a
# clear-footage baseline — and rides the SAME contrast_curve. There is one
# curve and one combined cap, so contrast is applied exactly once. The veil
# factor ramps 0->1 across [FLOOR, FULL]; below the floor (typical clear-day
# black point ~27) it is exactly 0, so clear
# and merely-underexposed footage — whose blacks sit low, not lifted — regrade
# unchanged. The increment is bounded to DEHAZE_CONTRAST_MAX, chosen so the
# combined curve keeps the 0.25 shadow knot well above black (proven in
# test_dehaze_composes_without_crushing_blacks).
DEHAZE_VEIL_BLACK_FLOOR = 30.0
DEHAZE_VEIL_BLACK_FULL = 70.0
DEHAZE_CONTRAST_MAX = 0.04

# Overcast light. Diffuse cloud light flattens contrast the same way haze does,
# but — unlike haze — it need not lift the blacks, so the veil factor above can
# miss it. This is the one place Phase-A's weather signal (a frame-brightness
# band; see gomoveshift.pipeline.weather) reaches the grade: rather than plumb
# the day-segment classification through drive_scores and LF assembly, the band
# is reconstructed locally from the median brightness the stats already carry.
# It is a BAND, not a threshold — too dark reads as dusk/tunnel (served by the
# lift/gamma ramps) and too bright as harsh sun (adding contrast would clip), so
# the factor peaks at the band midpoint and tapers to 0 at both edges. Those
# edges are the weather module's rainy/sunny thresholds rescaled from its 0-1000
# brightness axis to the 0-255 luminance the grader measures (400 -> 102,
# 580 -> 148; the peak at 125 mirrors weather._classify's confidence midpoint).
# The increment it gates rides the SAME contrast_curve as dehaze under the SAME
# combined cap (see the contrast block), so it is never a second knob.
OVERCAST_LUMA_MIN = 102.0
OVERCAST_LUMA_MAX = 148.0

# Ceiling on the white-balance shift, and therefore on the cast the curve is
# allowed to stretch — see _cast_is_correctable.
CAST_CORRECTABLE_MAX = 0.05

# Grade modes. Every ramp below was tuned for GoPro video that is squeezed
# full->limited range (16-235) for YouTube; photos export full-range sRGB
# JPEG, so photo mode switches off the two video-only behaviors: the
# unconditional 2-level shadow-lift/highlight-pull floors (they exist only
# to protect the squeeze — on a still they lift blacks and dull whites for
# no reason) and full-strength warm-cast neutralization (golden-hour light
# is the subject of a still, not an error). Video stays the default, so
# every existing caller grades byte-identically.
GRADE_MODE_VIDEO = "video"
GRADE_MODE_PHOTO = "photo"

# The slice of a WARM cast (red above / blue below the green reference)
# that photo mode leaves in the image: only the excess beyond it is
# neutralized, so a golden-hour cast (~0.02-0.04 in mean units) mostly
# survives, a stronger one still loses its excess, and past the clamp's
# reach the two modes correct identically. A COOL cast is corrected with
# full authority in both modes — only warmth is scenic.
PHOTO_WARM_CAST_ALLOWANCE = 0.03

# Photo-mode scene guards. A video frame is one sample of a stream graded for
# YouTube, so the ramps err toward correction; a still IS the scene, so photo
# mode protects deliberate looks the ramps read as defects. All four guards
# below are photo-only — video grades byte-identically to before.
#
# Low-key stills (silhouettes, night shots). The lift/gamma ramps read "dark"
# as "underexposed" and pull a deliberate dark scene toward gray. Underexposure
# drags the WHOLE histogram down, highlights included; a low-key composition
# pins its blacks at (or near) true black and holds its midtones under the
# deep-underexposure knee while KEEPING bright highlights. That intact top end
# is the signature separating intent from accident, so only a still showing it
# gets the damped (never disabled) response — a dark frame without it keeps the
# full correction.
LOW_KEY_BLACK_POINT_MAX = 8.0
LOW_KEY_WHITE_POINT_MIN = 200.0
LOW_KEY_RESPONSE_SCALE = 0.25

# Mist/fog stills. The base S-curve plus the dehaze/overcast increment can
# stack to the combined cap on a deliberate mist scene — dehazing the weather.
# The low-spread/low-sat class (the same saturation bound the profile detector
# uses, so the two cannot drift apart) caps the stacked response at the base
# ceiling: the flatness curve still runs, the atmospheric stack does not.
PHOTO_MIST_CONTRAST_MAX = CONTRAST_CURVE_MAX

# Sunset stills. Mean saturation above the pulldown onset TOGETHER WITH a warm
# cast reads as golden-hour light filling the frame — the scene, not an
# oversaturation artifact — so the still keeps its saturation whole. The
# threshold sits at the bottom of the golden-hour cast range that
# PHOTO_WARM_CAST_ALLOWANCE documents (~0.02-0.04 in mean units); a
# neutral/cool oversaturated still keeps the pulldown, that IS the artifact.
PHOTO_SUNSET_WARM_CAST_MIN = 0.02

# Source-profile detection. Every slope below is tuned for GoPro "Natural"; a
# Flat / Protune-Log source ships deliberately low-contrast AND desaturated, so
# the Natural ramp leaves it washed out at the 1.25 cap. Underexposure also
# measures low-spread, so the midtone window is what separates "log-encoded"
# (whole histogram compressed toward the middle) from "shot in the dark"
# (histogram pinned at the bottom) — the latter is already served by the
# shadow-lift and gamma ramps and must not also be handed an S-curve.
PROFILE_MID_TONE_MIN = 60.0
PROFILE_MID_TONE_MAX = 180.0
FLAT_PROFILE_SATURATION = 0.22
FLAT_SATURATION_GAIN = 1.15
FLAT_SATURATION_CAP = 1.45
SOURCE_PROFILE_NATURAL = "natural"
SOURCE_PROFILE_FLAT = "flat"

# Saturation pull-down. The ramp below used to be floored at 1.0, so the grader
# could only ever ADD saturation: harsh midday sun or an already-punchy camera
# profile got no correction at all and could only be pushed further.
# The pull-down MIRRORS the boost ramp rather than inventing a second tuning —
# same band width (0.35) and same authority (0.20 of multiplier) — anchored at
# the metric's ceiling instead of its target, so the band is [0.65, 1.00] and
# the floor lands at 0.80, the exact mirror of the 1.20 the boost ramp reaches.
# Nothing between the boost ramp's zero point (0.45) and the onset is touched,
# which is what keeps every previously-graded, human-approved source at 1.0.
SATURATION_EXCESS_ONSET = 0.65
SATURATION_EXCESS_BAND = 0.35
SATURATION_PULLDOWN_MIN = 0.80


def _cast_is_correctable(r_cast: float, b_cast: float) -> bool:
    """True when colorbalance still has the authority to undo the source cast.

    ffmpeg's ``curves`` master channel steepens R, G and B independently, so an
    S-curve AMPLIFIES whatever cast the source already carries — and it runs
    *before* colorbalance, whose shift is clamped to ±CAST_CORRECTABLE_MAX.
    On a strongly-cast flat frame, c=0.03 already cancels the entire
    white-balance correction and c=0.06 ships a worse cast than no grade at all.
    Beyond the clamp's authority the curve therefore stays off: contrast is
    never bought at the price of a cast nothing downstream can claw back.
    """
    return max(abs(r_cast), abs(b_cast)) <= CAST_CORRECTABLE_MAX


def _saturation_is_measurable(stats: SegmentStats) -> bool:
    """True when the measured saturation is evidence rather than an artifact.

    ``analyze_frame`` measures saturation as the mean of (max-min)/max. That
    ratio INFLATES on near-black pixels, where a couple of codes of sensor noise
    read as full chroma, and COLLAPSES as channels clip together at the top. So
    a crushed or blown-out frame can report a high number while carrying no
    correctable excess at all, and pulling saturation out of it would grade on
    noise. The sane-exposure window is the one the module already defines for
    source-profile detection, reused here so the two cannot drift apart.
    """
    return PROFILE_MID_TONE_MIN <= stats.mid_tone <= PROFILE_MID_TONE_MAX


def _is_low_contrast(stats: SegmentStats) -> bool:
    """True when the histogram is compressed *and* the exposure is sane."""
    if not (PROFILE_MID_TONE_MIN <= stats.mid_tone <= PROFILE_MID_TONE_MAX):
        return False
    return (stats.white_point - stats.black_point) < CONTRAST_SPREAD_OFF


def _is_low_key(stats: SegmentStats) -> bool:
    """True when the darkness reads as deliberate low-key, not underexposure.

    Blacks at (or near) true black, midtones under the deep-underexposure
    knee, highlights intact — see the LOW_KEY_* constants for why the intact
    top end is what separates a silhouette from an underexposed frame.
    """
    return (
        stats.black_point <= LOW_KEY_BLACK_POINT_MAX
        and stats.mid_tone < DEEP_UNDEREXPOSURE_MID
        and stats.white_point >= LOW_KEY_WHITE_POINT_MIN
    )


def _haze_veil_factor(stats: SegmentStats) -> float:
    """0..1 estimate of the atmospheric veil lifting the blacks.

    Haze lifts the black point above where a clear scene would sit; measured as
    that rise above the clear-footage floor, normalized and clamped. It is the
    ONLY signal separating haze from an ordinary flat histogram, so the dehaze
    increment it gates is self-limiting: a scene whose blacks sit at (or below)
    the clear-day baseline scores 0 and earns no dehaze at all, no matter how
    compressed the rest of the histogram is. Underexposure — blacks pinned LOW —
    therefore also scores 0 and keeps its lift/gamma treatment untouched.
    """
    veil = (stats.black_point - DEHAZE_VEIL_BLACK_FLOOR) / (
        DEHAZE_VEIL_BLACK_FULL - DEHAZE_VEIL_BLACK_FLOOR
    )
    return min(1.0, max(0.0, veil))


def _overcast_light_factor(stats: SegmentStats) -> float:
    """0..1 estimate that the scene sits in the overcast (flat diffuse) light band.

    Triangular membership in the moderate-brightness window the weather detector
    labels OVERCAST, keyed on the median luminance (the closest analogue the grade
    stats carry to the mean-brightness signal Phase-A measures). It peaks at the
    band midpoint and falls to 0 at both edges, so a dark (dusk/tunnel) or a bright
    (harsh-sun / washed-out) scene scores 0 and earns no overcast contrast at all —
    the increment it gates is self-limiting, and byte-identical to today wherever
    the scene is not measurably overcast.
    """
    midpoint = (OVERCAST_LUMA_MIN + OVERCAST_LUMA_MAX) / 2.0
    half_width = (OVERCAST_LUMA_MAX - OVERCAST_LUMA_MIN) / 2.0
    factor = 1.0 - abs(stats.mid_tone - midpoint) / half_width
    return min(1.0, max(0.0, factor))


def detect_source_profile(stats: SegmentStats) -> str:
    """Classify the source tone profile from the measured stats.

    Returns ``SOURCE_PROFILE_FLAT`` for a low-contrast, low-saturation source
    (GoPro Flat, Protune-Log and friends) and ``SOURCE_PROFILE_NATURAL`` for
    everything else — including haze, which compresses the histogram without
    desaturating, and underexposure, which is dark rather than flat.
    """
    if _is_low_contrast(stats) and stats.saturation <= FLAT_PROFILE_SATURATION:
        return SOURCE_PROFILE_FLAT
    return SOURCE_PROFILE_NATURAL


def _sky_excluded(stats: SegmentStats) -> SegmentStats:
    """Re-meter exposure and white balance with the bright sky down-weighted.

    ``analyze_frame`` measures the foreground (non-sky) white point, mid tone and
    per-channel means alongside the sky's area share. Here the whole-frame meter
    is blended toward that foreground meter by the measured ``sky_fraction``: the
    more sky the frame holds, the more the sky is excluded from the grade. The
    substitution is self-limiting by construction — with little sky the weight is
    tiny AND the foreground meter barely differs from the whole frame, so the
    change vanishes; with none it is EXACTLY the whole-frame meter (early return),
    so no previously-graded footage regrades. The black point (a dark-end
    statistic a bright sky does not touch) and saturation are left whole-frame.
    Downstream this feeds the SAME white-balance ±CAST_CORRECTABLE_MAX clamp and
    the same S-curve cast gate — it changes WHICH cast is metered, not the budget.
    """
    sf = stats.sky_fraction
    if sf <= 0.0:
        return stats
    w = 1.0 - sf
    return replace(
        stats,
        white_point=stats.white_point * w + stats.fg_white_point * sf,
        mid_tone=stats.mid_tone * w + stats.fg_mid_tone * sf,
        r_mean=stats.r_mean * w + stats.fg_r_mean * sf,
        g_mean=stats.g_mean * w + stats.fg_g_mean * sf,
        b_mean=stats.b_mean * w + stats.fg_b_mean * sf,
        sky_fraction=0.0,
    )


def compute_correction(
    stats: SegmentStats,
    *,
    mode: str = GRADE_MODE_VIDEO,
    no_sharpen: bool = False,
    no_denoise: bool = False,
    frame_width: int = 1920,
) -> CorrectionParams:
    """
    Compute color correction parameters from segment statistics.

    Tuned for GoPro "Natural" profile with full-range (color_range=pc)
    preservation. GoPro Natural is slightly flat and slightly warm;
    the correction adds gentle contrast and corrects the warm cast.

    In video mode (the default) output targets limited range (16-235) for
    YouTube SDR: shadow lift and highlight pull scale with measured stats
    to compensate for the full->limited range compression across any
    footage type. Photo mode grades a full-range still: no range floors,
    and warm casts keep PHOTO_WARM_CAST_ALLOWANCE of themselves.
    """
    if mode not in (GRADE_MODE_VIDEO, GRADE_MODE_PHOTO):
        raise ValueError(f"Unknown grade mode: {mode!r}")

    # Meter the foreground: a bright sky is down-weighted by its measured area
    # share so it no longer drags the exposure/white-balance grade (no-op when
    # the frame holds no sky). Every ramp and gate below reads the re-metered
    # stats, so the exclusion is applied coherently in one place.
    stats = _sky_excluded(stats)

    # The 2-level lift/pull floors protect the full->limited squeeze; a
    # full-range still earns lift/pull only from its measured ramps (never
    # negative — a negative ramp is a crush, not a correction).
    range_floor = 2.0 if mode == GRADE_MODE_VIDEO else 0.0

    # Deliberate-low-key stills damp (never disable) the lift/gamma response
    # so a silhouette stays a silhouette — see _is_low_key.
    low_key_still = mode == GRADE_MODE_PHOTO and _is_low_key(stats)

    # Shadow lift: adaptive to measured black point. Lower black_point means
    # shadows sit closer to the limited-range floor (16) after conversion,
    # so they need more lift to preserve detail.
    #   black_point=0  -> lift=15  (deep blacks, maximum lift)
    #   black_point=27 -> lift~=11 (typical GoPro daylight)
    #   black_point=87 -> lift=2   (bright overcast, minimal lift)
    # Slope tightened from 0.2 to 0.15 because dark
    # scenes were under-lifted; flatter slope keeps mid-range black_points
    # higher so deeper blacks preserve more shadow detail post limited-range conversion.
    #
    # deep_deficit is how far under DEEP_UNDEREXPOSURE_MID the scene actually
    # sits, normalized to 0-1. It is exactly 0.0 for anything at or above the
    # knee, which is what keeps normally-exposed footage grading unchanged.
    deep_deficit = max(0.0, DEEP_UNDEREXPOSURE_MID - stats.mid_tone) / DEEP_UNDEREXPOSURE_MID
    shadow_lift = 15.0 - stats.black_point * 0.15 + deep_deficit * DEEP_SHADOW_LIFT_BONUS
    shadow_lift = max(range_floor, min(15.0 + DEEP_SHADOW_LIFT_BONUS, shadow_lift))
    if low_key_still:
        shadow_lift *= LOW_KEY_RESPONSE_SCALE

    # Highlight pull: adaptive to measured white point. Higher white_point
    # means highlights sit closer to the limited-range ceiling (235) after
    # conversion, so they need more pull to prevent clipping.
    #   white_point=170 -> pull=2   (no bright highlights)
    #   white_point=195 -> pull~=5   (typical GoPro daylight)
    #   white_point=245 -> pull=15  (bright sky/clipping)
    highlight_pull = max(range_floor, min(15.0, (stats.white_point - 170) * 0.2))

    # Saturation: GoPro Natural is intentionally low-sat for grading flexibility.
    # For direct-to-YouTube driving content, boost more eagerly.
    # Scale: 1.0 at sat>=0.45, up to 1.20 at sat=0.05, capped at 1.25
    # Above the onset the same ramp runs in reverse, so an oversaturated source
    # is pulled DOWN by its measured excess (see SATURATION_EXCESS_ONSET). The
    # statistic is self-limiting: measured saturation is a mean over pixels, so
    # a frame where only part of the image is over the top moves the aggregate
    # only by that region's share and earns only that share of the pull.
    if stats.saturation >= 0.45:
        excess = stats.saturation - SATURATION_EXCESS_ONSET
        if excess > 0 and _saturation_is_measurable(stats):
            if mode == GRADE_MODE_PHOTO and (
                stats.r_mean - stats.g_mean >= PHOTO_SUNSET_WARM_CAST_MIN
            ):
                # Saturated AND warm reads as golden-hour light filling the
                # frame — the scene itself, so the still keeps it whole.
                saturation_mult = 1.0
            else:
                t = min(1.0, excess / SATURATION_EXCESS_BAND)
                saturation_mult = 1.0 - t * 0.20
        else:
            saturation_mult = 1.0
    elif stats.saturation >= 0.1:
        t = (0.45 - stats.saturation) / 0.35
        saturation_mult = 1.0 + t * 0.20
    else:
        saturation_mult = 1.0 + (0.45 - max(0.0, stats.saturation)) / 0.35 * 0.20
    saturation_mult = min(1.25, max(SATURATION_PULLDOWN_MIN, saturation_mult))

    # A Flat/Log source is desaturated by design; the Natural ramp above tops
    # out at 1.25 and ships it washed out. Detection is measured, so a Natural
    # source can never reach this branch and regrades unchanged. The inference
    # is video-only: a still is never Protune footage, and fog/mist stills
    # measure exactly like a log profile (compressed spread, muted sat) — the
    # gain would saturate the weather.
    profile = detect_source_profile(stats) if mode == GRADE_MODE_VIDEO else SOURCE_PROFILE_NATURAL
    if profile == SOURCE_PROFILE_FLAT:
        saturation_mult = min(FLAT_SATURATION_CAP, saturation_mult * FLAT_SATURATION_GAIN)
        logger.debug(
            "Flat/Log source profile detected (spread=%.1f, sat=%.3f) — saturation to %.3f",
            stats.white_point - stats.black_point,
            stats.saturation,
            saturation_mult,
        )

    # Per-channel cast vs the green reference. Measured here rather than at the
    # white-balance block below because the S-curve must not stretch a cast that
    # colorbalance has no authority to claw back.
    r_cast = stats.r_mean - stats.g_mean
    b_cast = stats.b_mean - stats.g_mean

    # S-curve, gated on the measured spread (see CONTRAST_SPREAD_* above).
    # Contrasty footage takes the else branch and gets exactly 0.0.
    if _is_low_contrast(stats) and _cast_is_correctable(r_cast, b_cast):
        flatness = min(
            1.0,
            max(
                0.0,
                (CONTRAST_SPREAD_OFF - (stats.white_point - stats.black_point))
                / (CONTRAST_SPREAD_OFF - CONTRAST_SPREAD_FULL),
            ),
        )
        # Base S-curve on the measured flatness PLUS one atmospheric increment
        # for flat light: haze (lifted blacks, _haze_veil_factor) OR overcast
        # (the weather brightness band, _overcast_light_factor). The two signals
        # share ONE increment headroom via max(), so contrast is applied exactly
        # once and the combined curve stays bounded by CONTRAST_CURVE_MAX +
        # DEHAZE_CONTRAST_MAX — neither can stack past that cap or crush the
        # blacks. Clear, non-overcast footage scores 0 on both and reduces to the
        # pre-dehaze/pre-overcast grade bit-for-bit.
        atmospheric = max(_haze_veil_factor(stats), _overcast_light_factor(stats))
        contrast_curve = CONTRAST_CURVE_MAX * flatness
        contrast_curve += DEHAZE_CONTRAST_MAX * flatness * atmospheric
        contrast_curve = min(contrast_curve, CONTRAST_CURVE_MAX + DEHAZE_CONTRAST_MAX)
        if mode == GRADE_MODE_PHOTO and stats.saturation <= FLAT_PROFILE_SATURATION:
            # Low-spread + low-sat is the mist/fog scene class; a still keeps
            # its veil — the stacked response caps at the base ceiling.
            contrast_curve = min(contrast_curve, PHOTO_MIST_CONTRAST_MAX)
        contrast_curve = round(contrast_curve, 4)
        if contrast_curve < CONTRAST_MIN_USEFUL:
            contrast_curve = 0.0
    else:
        contrast_curve = 0.0

    # White balance: correct the cast measured above, clamped to the authority
    # the filter is allowed (CAST_CORRECTABLE_MAX). Photo mode spends that
    # authority only on the warm excess beyond PHOTO_WARM_CAST_ALLOWANCE so
    # golden-hour light survives a still; a cool cast is corrected with full
    # authority in both modes.
    r_correct, b_correct = r_cast, b_cast
    if mode == GRADE_MODE_PHOTO:
        r_correct = max(0.0, r_cast - PHOTO_WARM_CAST_ALLOWANCE) if r_cast > 0.0 else r_cast
        b_correct = min(0.0, b_cast + PHOTO_WARM_CAST_ALLOWANCE) if b_cast < 0.0 else b_cast
    white_balance_r = -max(-CAST_CORRECTABLE_MAX, min(CAST_CORRECTABLE_MAX, r_correct))
    white_balance_b = -max(-CAST_CORRECTABLE_MAX, min(CAST_CORRECTABLE_MAX, b_correct))

    # Highlight-only white balance. A cast concentrated in the bright end barely
    # moves the whole-frame means the shift above corrects, so it is measured
    # directly over the highlight region (r/g/b_mean_high) and applied on the
    # colorbalance highlight range. The cast is weighted by the highlight area
    # share, so a frame with few highlights earns little correction — the
    # estimator is self-limiting when evidence is weak. It shares the same
    # ±CAST_CORRECTABLE_MAX authority as the shadow/midtone shift and never
    # stacks with it: the highlight range begins (0.75) where the midtone weight
    # has already fallen to zero.
    r_cast_high = (stats.r_mean_high - stats.g_mean_high) * stats.highlight_frac
    b_cast_high = (stats.b_mean_high - stats.g_mean_high) * stats.highlight_frac
    white_balance_r_high = -max(-CAST_CORRECTABLE_MAX, min(CAST_CORRECTABLE_MAX, r_cast_high))
    white_balance_b_high = -max(-CAST_CORRECTABLE_MAX, min(CAST_CORRECTABLE_MAX, b_cast_high))

    # Gamma correction: lift dark midtones, pull bright midtones.
    # Stronger correction (divisor 500 vs 1000) for limited-range output.
    # Below DEEP_UNDEREXPOSURE_MID the deficit term continues the base slope
    # instead of pinning at the old fixed 1.15 ceiling.
    if stats.mid_tone < 110:
        gamma = 1.0 + (110 - stats.mid_tone) / 500.0 + deep_deficit * DEEP_GAMMA_BONUS
    elif stats.mid_tone > 145:
        gamma = 1.0 - (stats.mid_tone - 145) / 600.0
    else:
        gamma = 1.0
    gamma = max(0.9, min(GAMMA_MAX, gamma))
    if low_key_still and gamma > 1.0:
        gamma = 1.0 + (gamma - 1.0) * LOW_KEY_RESPONSE_SCALE

    # Denoise: ramped with the measured black point, not a binary step.
    # Dark driving footage looks worse without temporal smoothing, so the
    # very-dark case gets full strength. Denoise on daylight footage
    # (black_point around 26) makes no visible difference, so daylight stays at
    # 0.0, which is why the ramp reaches zero at 20.
    # The former `denoise_enabled` kwarg is gone: no production call site ever
    # passed it, so it was a dead knob, and `--skip-color` is the real escape.
    ramp = (DENOISE_BLACK_POINT_OFF - stats.black_point) / (
        DENOISE_BLACK_POINT_OFF - DENOISE_BLACK_POINT_FULL
    )
    denoise_strength = round(DENOISE_MAX_STRENGTH * min(1.0, max(0.0, ramp)), 2)
    if denoise_strength < DENOISE_MIN_USEFUL:
        denoise_strength = 0.0
    # Measured grain can only raise the black-point ramp's strength; the
    # operator opt-out beats both.
    denoise_strength = max(
        denoise_strength, denoise_strength_for(stats.noise_score, disabled=no_denoise)
    )
    if no_denoise:
        denoise_strength = 0.0

    # Sharpen is measured per day master (quality.sharpen_amount_for): zero when
    # the sampled detail is already crisp or unusable. Video only: a photo
    # develop applies its own detail-ramped unsharp, so a second one here
    # would double-sharpen.
    sharpen_amount, sharpen_radius = sharpen_amount_for(
        stats.detail_score, frame_width, disabled=no_sharpen or mode != GRADE_MODE_VIDEO
    )
    return CorrectionParams(
        shadow_lift=shadow_lift,
        highlight_pull=highlight_pull,
        saturation_mult=round(saturation_mult, 3),
        contrast_curve=contrast_curve,
        white_balance_r=round(white_balance_r, 4),
        white_balance_b=round(white_balance_b, 4),
        white_balance_r_high=round(white_balance_r_high, 4),
        white_balance_b_high=round(white_balance_b_high, 4),
        gamma=round(gamma, 4),
        denoise_strength=denoise_strength,
        sharpen_amount=sharpen_amount,
        sharpen_radius=sharpen_radius,
        sharpen_disabled=no_sharpen,
        denoise_disabled=no_denoise,
    )


def _average_params(params_list: list[CorrectionParams]) -> CorrectionParams:
    """Average all correction params into a single set."""
    return CorrectionParams(
        shadow_lift=float(np.mean([p.shadow_lift for p in params_list])),
        highlight_pull=float(np.mean([p.highlight_pull for p in params_list])),
        saturation_mult=float(np.mean([p.saturation_mult for p in params_list])),
        contrast_curve=float(np.mean([p.contrast_curve for p in params_list])),
        white_balance_r=float(np.mean([p.white_balance_r for p in params_list])),
        white_balance_b=float(np.mean([p.white_balance_b for p in params_list])),
        white_balance_r_high=float(np.mean([p.white_balance_r_high for p in params_list])),
        white_balance_b_high=float(np.mean([p.white_balance_b_high for p in params_list])),
        gamma=float(np.mean([p.gamma for p in params_list])),
        denoise_strength=float(np.mean([p.denoise_strength for p in params_list])),
        sharpen_amount=float(np.mean([p.sharpen_amount for p in params_list])),
    )


def compute_cluster_correction(
    cluster_stats: list[SegmentStats],
    *,
    mode: str = GRADE_MODE_VIDEO,
) -> CorrectionParams:
    """Compute a single correction for an entire cluster of files.

    Uses median stats across all files in the cluster, then applies the
    standard compute_correction logic in the given grade mode.

    The per-channel means are carried through with every other statistic:
    omitting them left them at the neutral FrameStats defaults, which made
    the downstream white-balance casts evaluate to zero and shipped every
    cluster-graded batch with no white balance at all.

    Args:
        cluster_stats: List of SegmentStats from files in one cluster.

    Returns:
        CorrectionParams for the cluster. Neutral if empty.
    """
    if not cluster_stats:
        return CorrectionParams()

    aggregated = aggregate_segment_stats(
        [
            FrameStats(
                black_point=s.black_point,
                white_point=s.white_point,
                mid_tone=s.mid_tone,
                saturation=s.saturation,
                r_mean=s.r_mean,
                g_mean=s.g_mean,
                b_mean=s.b_mean,
                r_mean_high=s.r_mean_high,
                g_mean_high=s.g_mean_high,
                b_mean_high=s.b_mean_high,
                highlight_frac=s.highlight_frac,
                sky_fraction=s.sky_fraction,
                fg_white_point=s.fg_white_point,
                fg_mid_tone=s.fg_mid_tone,
                fg_r_mean=s.fg_r_mean,
                fg_g_mean=s.fg_g_mean,
                fg_b_mean=s.fg_b_mean,
                detail_score=s.detail_score,
                noise_score=s.noise_score,
            )
            for s in cluster_stats
        ]
    )
    return compute_correction(aggregated, mode=mode)


# ── LUT Math Helpers ────────────────────────────────────────────────


def _monotone_cubic_interp(xs: np.ndarray, ys: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Monotone cubic Hermite interpolation (Fritsch-Carlson).

    Matches ffmpeg's curves filter spline behaviour: smooth, non-overshooting.
    All arrays are 1-D float64. ``x_eval`` values outside ``xs`` range are
    clamped to the boundary values.

    Args:
        xs: Strictly increasing knot x-coordinates (N,).
        ys: Knot y-coordinates (N,).
        x_eval: Points at which to evaluate the spline (M,).

    Returns:
        Interpolated y-values (M,).
    """
    n = len(xs)
    if n < 2:
        return np.full_like(x_eval, ys[0] if n == 1 else 0.0)

    # Compute secants
    dx = np.diff(xs)
    dy = np.diff(ys)
    secants = dy / dx

    # Initial tangent estimates: average of neighboring secants
    tangents = np.zeros(n, dtype=np.float64)
    tangents[0] = secants[0]
    tangents[-1] = secants[-1]
    for i in range(1, n - 1):
        tangents[i] = (secants[i - 1] + secants[i]) / 2.0

    # Fritsch-Carlson monotonicity fix
    for i in range(n - 1):
        if abs(secants[i]) < 1e-12:
            tangents[i] = 0.0
            tangents[i + 1] = 0.0
        else:
            alpha = tangents[i] / secants[i]
            beta = tangents[i + 1] / secants[i]
            # Restrict to circle of radius 3
            mag = alpha * alpha + beta * beta
            if mag > 9.0:
                tau = 3.0 / np.sqrt(mag)
                tangents[i] = tau * alpha * secants[i]
                tangents[i + 1] = tau * beta * secants[i]

    # Evaluate using cubic Hermite basis
    # Clamp x_eval to domain
    x_clamped = np.clip(x_eval, xs[0], xs[-1])
    result = np.empty_like(x_clamped)

    # Find intervals via searchsorted
    indices = np.searchsorted(xs, x_clamped, side="right") - 1
    indices = np.clip(indices, 0, n - 2)

    h = dx[indices]
    t = (x_clamped - xs[indices]) / h

    # Hermite basis functions
    h00 = (1.0 + 2.0 * t) * (1.0 - t) ** 2
    h10 = t * (1.0 - t) ** 2
    h01 = t**2 * (3.0 - 2.0 * t)
    h11 = t**2 * (t - 1.0)

    result = (
        h00 * ys[indices]
        + h10 * h * tangents[indices]
        + h01 * ys[indices + 1]
        + h11 * h * tangents[indices + 1]
    )

    return result


def _apply_curves(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Apply curves filter math (shadow lift, highlight pull, S-curve contrast).

    Replicates the control-point logic from ``_build_single_filter``, then
    applies the photo-only black-point move as a final step (exact no-op under
    ``BLACK_POINT_MIN_USEFUL`` so the video chain and baked LUTs are untouched).

    Args:
        rgb: (N, 3) float64 array in [0, 1].
        params: Correction parameters.

    Returns:
        (N, 3) float64 array with curves applied.
    """
    # The operator recovery knobs ADD onto the measured amount rather than
    # replacing it — see CorrectionParams.shadow_recovery/highlight_recovery.
    shadow_total = params.shadow_lift + params.shadow_recovery
    highlight_total = params.highlight_pull + params.highlight_recovery
    has_shadow = shadow_total >= 1
    has_highlight = highlight_total >= 1
    has_contrast = params.contrast_curve > 0.001
    has_black_point = abs(params.black_point) >= BLACK_POINT_MIN_USEFUL
    if not (has_shadow or has_highlight or has_contrast or has_black_point):
        return rgb.copy()

    if has_shadow or has_highlight or has_contrast:
        bl_out = shadow_total / 255.0
        wh_out = 1.0 - (highlight_total / 255.0)
        c = params.contrast_curve

        # Highlight knee: the ceiling scales ONLY the top
        # (1.0) control point; the 0.75 knee anchor keeps the un-squeezed wh_out.
        # Mirror of color_filters._build_single_filter — replaces the old
        # whole-range ``colorlevels`` multiply that darkened midtones.
        wh_top = wh_out
        if highlight_total >= 5.0:
            wh_top *= 0.95 + (highlight_total / 500.0)

        # Control points — same as _build_single_filter
        xs = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float64)
        ys = np.array(
            [
                bl_out,
                0.25 + bl_out * 0.5 - c,
                0.75 - (1.0 - wh_out) * 0.5 + c,
                wh_top,
            ],
            dtype=np.float64,
        )

        # ffmpeg curves master channel applies same curve to R, G, B independently
        out = np.empty_like(rgb)
        for ch in range(3):
            out[:, ch] = _monotone_cubic_interp(xs, ys, rgb[:, ch])
    else:
        out = rgb.copy()

    if has_black_point:
        # Python-float arithmetic on purpose: a np.float64 scalar would upcast
        # a float32 frame (NEP 50) and break _grade's dtype contract.
        bp = min(max(params.black_point, BLACK_POINT_MIN), BLACK_POINT_MAX)
        if bp > 0:
            floor = bp / 255.0  # matte fade: lift the output floor
            out = floor + (1.0 - floor) * out
        else:
            cut = -bp / 255.0  # crush: cut the input floor
            out = np.clip((out - cut) / (1.0 - cut), 0.0, 1.0)

    return out


def _apply_luma_curve(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """The MASTER tone curve (photo develop only).

    ``params.curve_l`` is empty (OFF) or a FLAT list of alternating x,y control
    points (>= 2 points, x strictly ascending in [0, 1]) — the same form the
    channel curves take, evaluated through the same Fritsch-Carlson monotone
    interpolant ``_apply_curves`` uses. Where ``_apply_curves`` builds a FIXED
    four-knot parametric skeleton out of shadow_lift/highlight_pull/contrast,
    this is a shape the operator draws: any number of knots, anywhere.

    ONE curve on all three channels, so it shapes tone and can never introduce a
    colour cast — by construction, not by convention. It runs BEFORE
    ``_apply_channel_curves`` so the cross-process per-channel curves keep riding
    on top of the master tone shape, exactly as they ride on ``_apply_curves``.
    Both sit early in ``_grade`` (before colorbalance/saturation/gamma), where a
    real graded frame occupies roughly [0, 0.6] — a knot set that only bites in
    the top decile is an identity on a real photograph.

    Empty is a bit-identical no-op by construction (early ``rgb.copy()``).
    Deliberately EXCLUDED from ``params_to_lut3d_cube``, like ``_apply_hsl`` and
    ``_apply_channel_curves``: it rides inside the photo ``_grade`` only, so
    every video output and baked LUT stays bit-identical.

    Args:
        rgb: (N, 3) float array in [0, 1].
        params: source of the ``curve_l`` control points.

    Returns:
        (N, 3) array (same dtype), tone-curved — or a copy of ``rgb`` if OFF.
    """
    pts = params.curve_l
    if not pts:
        return rgb.copy()

    arr = np.asarray(pts, dtype=np.float64)
    xs, ys = arr[0::2], arr[1::2]
    out = rgb.copy()
    for ch in range(3):
        out[:, ch] = _monotone_cubic_interp(xs, ys, out[:, ch].astype(np.float64))
    return np.clip(out, 0.0, 1.0).astype(rgb.dtype, copy=False)


def _apply_channel_curves(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Cross-process per-channel tone curves (photo develop only).

    Each of ``params.curve_r`` / ``curve_g`` / ``curve_b`` is empty (OFF) or a
    FLAT list of alternating x,y control points (>= 2 points, x strictly
    ascending in [0, 1]). For every channel that carries a curve, an ADDITIONAL
    monotone cubic curve — the same Fritsch-Carlson interpolant the master
    ``_apply_curves`` uses — rides on top of the R/G/B master curve,
    independently per channel: the film "cross-process" look. Channels without a
    curve pass through untouched.

    An all-empty params is a bit-identical no-op by construction (early
    ``rgb.copy()``). Deliberately EXCLUDED from ``params_to_lut3d_cube``: like
    ``_apply_hsl`` this rides inside the photo ``_grade`` only, so every video
    output and baked LUT stays bit-identical.

    Args:
        rgb: (N, 3) float array in [0, 1].
        params: source of the ``curve_r`` / ``curve_g`` / ``curve_b`` controls.

    Returns:
        (N, 3) array (same dtype), channel-curved — or a copy of ``rgb`` if OFF.
    """
    curves = (params.curve_r, params.curve_g, params.curve_b)
    if not any(curves):
        return rgb.copy()

    out = rgb.copy()
    for ch, pts in enumerate(curves):
        if not pts:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        out[:, ch] = _monotone_cubic_interp(arr[0::2], arr[1::2], out[:, ch].astype(np.float64))
    return np.clip(out, 0.0, 1.0).astype(rgb.dtype, copy=False)


def _apply_colorbalance(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Apply colorbalance filter math (white balance R/G/B shift).

    ffmpeg colorbalance applies per-range (shadow/midtone/highlight) additive
    shifts weighted by luminance position.  The shadow weight peaks at 0 and
    fades by ~0.333; the midtone weight peaks at 0.5; highlight peaks at 1.0.

    We replicate the shadow (rs, bs), midtone (rm, bm) and highlight (rh, bh)
    shifts used in ``_build_single_filter`` where rs==rm and bs==bm, with the
    highlight shifts carried independently (white_balance_*_high).

    The GREEN channel is the third axis and is photo-develop only: the
    green-magenta tint no other control can express (split-tone is luma-keyed,
    HSL is hue-keyed). ``compute_correction`` never sets ``white_balance_g`` /
    ``white_balance_g_high`` in either mode, so on every video params they hold
    their 0.0 default, the neutrality guard below fires exactly as before and
    both the encoder's ffmpeg mirror (which pins gs/gm/gh at 0) and
    ``params_to_lut3d_cube`` stay bit-identical.

    Args:
        rgb: (N, 3) float64 array in [0, 1].
        params: Correction parameters.

    Returns:
        (N, 3) float64 array with color balance applied.
    """
    if (
        abs(params.white_balance_r) <= 0.001
        and abs(params.white_balance_b) <= 0.001
        and abs(params.white_balance_g) <= 0.001
        and abs(params.white_balance_r_high) <= 0.001
        and abs(params.white_balance_b_high) <= 0.001
        and abs(params.white_balance_g_high) <= 0.001
    ):
        return rgb.copy()

    out = rgb.copy()

    # ffmpeg colorbalance shadow/midtone/highlight weights (simplified).
    # Shadow peaks at val=0 and fades to 0 by 0.25; midtone is a triangle
    # peaking at 0.5 (also 0 by 0.75); highlight is the mirror of shadow,
    # rising from 0 at 0.75 to 1 at 1.0. Because midtone is already 0 at 0.75
    # the highlight shift never stacks on top of the shadow/midtone shift.
    # For each channel, shift = shift * (shadow_w + mid_w) + shift_h * high_w.
    for ch, shift, shift_h in [
        (0, params.white_balance_r, params.white_balance_r_high),
        (1, params.white_balance_g, params.white_balance_g_high),
        (2, params.white_balance_b, params.white_balance_b_high),
    ]:
        if abs(shift) <= 0.001 and abs(shift_h) <= 0.001:
            continue
        vals = out[:, ch]
        shadow_w = np.clip(1.0 - vals * 4.0, 0.0, 1.0)
        mid_w = np.clip(1.0 - np.abs(vals - 0.5) * 4.0, 0.0, 1.0)
        high_w = np.clip((vals - 0.75) * 4.0, 0.0, 1.0)
        total_shift = shift * (shadow_w + mid_w) + shift_h * high_w
        out[:, ch] = vals + total_shift

    return out


def _apply_splittone(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Split-tone: three independent signed (r,g,b) tints keyed to luma (photo only).

    Real split-toning the scalar white-balance fields cannot express: one tint per
    tonal range — shadows, midtones, highlights — each a DIFFERENT colour.
    ``params.splittone_shadow`` / ``splittone_midtone`` / ``splittone_highlight``
    are each an empty tuple (OFF) or a length-3 signed (r,g,b) push. Three BT.709-luma
    weights PARTITION the range (they sum to 1 at every luma, so no seam and no
    double-count): the shadow ramp is full at black and gone by mid-luma, the
    highlight ramp its mirror, and the midtone tent fills exactly the gap between
    them — nothing at either end, full at mid-luma, which is where this chain's
    tone actually sits. ``rgb`` is (N, 3) float in [0, 1]; returns the same dtype
    (a copy if neutral).

    All-empty / all-neutral is a bit-identical no-op by construction. Deliberately
    EXCLUDED from ``params_to_lut3d_cube`` — split-toning rides inside the photo
    ``_grade`` only, so every video output and baked LUT stays bit-identical,
    exactly as ``_apply_hsl`` is excluded.
    """
    sh, mid, hi = params.splittone_shadow, params.splittone_midtone, params.splittone_highlight
    sh_on = bool(sh) and any(abs(v) > 1e-9 for v in sh)
    mid_on = bool(mid) and any(abs(v) > 1e-9 for v in mid)
    hi_on = bool(hi) and any(abs(v) > 1e-9 for v in hi)
    if not sh_on and not mid_on and not hi_on:
        return rgb.copy()

    # BT.709 luma, then three luma weights that partition the range (inline
    # literals so the develop cache hash sees any retune): the shadow weight is 1
    # at black and fades to 0 by mid-luma (0.5); the highlight weight is its
    # mirror, 0 at 0.5 rising to 1 at white; the midtone weight is the tent that
    # fills the gap — 0 at black, 1 at 0.5, 0 at white. The three sum to 1 at
    # every luma, so ranges hand off without a seam and never double-count.
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    out = rgb.copy()
    if sh_on:
        w = np.clip(1.0 - 2.0 * luma, 0.0, 1.0)[:, np.newaxis]
        out = out + np.asarray(sh, dtype=rgb.dtype) * w
    if mid_on:
        w = np.clip(1.0 - np.abs(2.0 * luma - 1.0), 0.0, 1.0)[:, np.newaxis]
        out = out + np.asarray(mid, dtype=rgb.dtype) * w
    if hi_on:
        w = np.clip(2.0 * luma - 1.0, 0.0, 1.0)[:, np.newaxis]
        out = out + np.asarray(hi, dtype=rgb.dtype) * w
    return np.clip(out, 0.0, 1.0).astype(rgb.dtype, copy=False)


def _apply_saturation(rgb: np.ndarray, sat_mult: float) -> np.ndarray:
    """Apply saturation adjustment (eq filter saturation parameter).

    ffmpeg eq filter adjusts saturation in a luma-preserving way: it scales
    the chroma (distance from luma) by the multiplier.

    Args:
        rgb: (N, 3) float64 array in [0, 1].
        sat_mult: Saturation multiplier (1.0 = no change).

    Returns:
        (N, 3) float64 array with saturation adjusted.
    """
    if abs(sat_mult - 1.0) < 0.01:
        return rgb.copy()

    # BT.709 luma
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    luma = luma[:, np.newaxis]  # (N, 1) for broadcasting

    # Scale chroma around luma
    out = luma + sat_mult * (rgb - luma)
    return out


def _apply_vibrance(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Low-saturation-weighted chroma boost (photo develop only).

    A "smart" saturation: where ``_apply_saturation`` scales every pixel's chroma
    uniformly, ``_apply_vibrance`` weights the boost per pixel by ``(1 - s)`` —
    ``s`` being that pixel's saturation ``(max-min)/max`` (guarded at max=0). Muted
    pixels take the full ``params.vibrance`` push while an already-rich pixel
    (a saturated sky, deep foliage) barely moves, so overall colour comes up
    without those areas cartooning. It is finer than vivid's scene-level headroom
    gate — the protection is per-pixel, not per-frame. The boost itself is the same
    luma-preserving chroma scale around BT.709 luma that ``_apply_saturation`` uses.

    ``params.vibrance`` is 0.0 (OFF, early ``rgb.copy()`` ⇒ bit-identical) or a
    positive push, clamped here to a sane [0, 2] (inline literals so the develop
    cache hash sees any retune). ``rgb`` is (N, 3) float in [0, 1]; returns the
    same dtype. Deliberately EXCLUDED from ``params_to_lut3d_cube`` — vibrance
    rides the photo ``_grade`` only, so every video output and baked LUT stays
    bit-identical, exactly as ``_apply_hsl`` is excluded.

    Args:
        rgb: (N, 3) float array in [0, 1].
        params: source of the ``vibrance`` control.

    Returns:
        (N, 3) array (same dtype), chroma-boosted — or a copy of ``rgb`` if OFF.
    """
    if params.vibrance == 0.0:
        return rgb.copy()

    vib = min(2.0, max(0.0, params.vibrance))
    mx = np.max(rgb, axis=1)
    mn = np.min(rgb, axis=1)
    # Per-pixel saturation (max-min)/max, guarding a black pixel's zero max.
    sat = np.where(mx > 1e-12, (mx - mn) / np.where(mx > 1e-12, mx, 1.0), 0.0)
    # Full push on muted pixels (s→0), ~none on already-rich ones (s→1).
    mult = (1.0 + vib * (1.0 - sat))[:, np.newaxis]

    luma = (0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2])[:, np.newaxis]
    out = luma + mult * (rgb - luma)
    return np.clip(out, 0.0, 1.0).astype(rgb.dtype, copy=False)


def _apply_monomix(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """B&W channel mixer + duotone toning (photo develop only).

    The two monochrome controls a fixed ``saturation_mult = 0`` cannot express.
    ``params.mono_mix`` is empty (OFF) or three custom R/G/B luma weights: the
    darkroom channel mixer, where a red-weighted mix is the red filter that
    renders a blue sky dark and red rock bright. Weights are deliberately NOT
    normalized — a sum away from 1 is a legitimate exposure move — and the
    resulting luma is clipped to [0, 1] before it is broadcast to all three
    channels. ``params.duotone`` is empty (OFF) or the two ends of a luma ramp
    ``(shadow_r, shadow_g, shadow_b, high_r, high_g, high_b)`` in [0, 1], applied
    as ``shadow + luma * (high - shadow)`` — sepia, selenium, cyanotype. The ramp
    keys on the ``mono_mix`` luma when one is set, else on BT.709 luma, so
    duotone alone still tones the frame's own luma. ``rgb`` is (N, 3) float in
    [0, 1]; returns the same dtype (a copy if OFF).

    That ramp REPLACES every pixel's colour, which on a real photograph resolved
    a whole landscape to one tone — the reason ``looks.synthwave`` is built out of
    split-tones instead. ``params.duotone_strength`` is the lever it lacked: it
    lerps the toned result back toward the plain monochrome luma, 1.0 being the
    full ramp (the shipped default, taken by an early branch so it stays the exact
    pixels of before) and 0.0 no tone at all. ``params.duotone_mid`` is empty (a
    straight two-ender) or a third (r,g,b) anchor at mid luma — a tritone, the
    ramp bent through a midtone colour of its own instead of whatever the line
    between the ends passes through. It is refused without a ``duotone``, where it
    would render as nothing.

    This stage runs BEFORE ``_apply_splittone`` in ``photos.develop._grade``: it
    writes every pixel, so a split-tone applied ahead of it would be wiped and
    split-toned monochrome — selenium shadows over a red-filter B&W — could not be
    expressed at all. Both mono controls are empty on every shipped look and
    style, so the position is invisible to them.

    Both empty is a bit-identical no-op by construction (early ``rgb.copy()``).
    Deliberately EXCLUDED from ``params_to_lut3d_cube`` — this rides inside the
    photo ``_grade`` only, so every video output and baked LUT stays
    bit-identical, exactly as ``_apply_hsl`` is excluded.
    """
    mix, duo = params.mono_mix, params.duotone
    if not mix and not duo:
        return rgb.copy()

    if mix:
        luma = mix[0] * rgb[:, 0] + mix[1] * rgb[:, 1] + mix[2] * rgb[:, 2]
    else:
        # BT.709 luma (inline literals, like _apply_saturation, so the develop
        # cache hash sees any retune).
        luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    luma = np.clip(luma, 0.0, 1.0)[:, np.newaxis]

    if duo:
        shadow = np.asarray(duo[:3], dtype=np.float64)
        high = np.asarray(duo[3:], dtype=np.float64)
        if params.duotone_mid:
            # Tritone: two half-ramps meeting AT the mid anchor, so the join is
            # continuous (both branches equal ``mid`` at luma 0.5) and the ends
            # are exactly where the two-ender put them.
            mid = np.asarray(params.duotone_mid, dtype=np.float64)
            t = luma * 2.0
            out = np.where(luma <= 0.5, shadow + t * (mid - shadow), mid + (t - 1.0) * (high - mid))
        else:
            out = shadow + luma * (high - shadow)
        if params.duotone_strength < 1.0:
            grey = np.repeat(luma, 3, axis=1)
            out = grey + params.duotone_strength * (out - grey)
    else:
        out = np.repeat(luma, 3, axis=1)
    return np.clip(out, 0.0, 1.0).astype(rgb.dtype, copy=False)


def _apply_hsl(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Per-hue-band HSL adjustment over fixed anchor hues (photo develop only).

    The look control the scalar ``CorrectionParams`` fields cannot express:
    independent hue-rotation / saturation / value gains keyed to where a pixel
    sits on the colour wheel. ``params.hue_shift`` / ``hue_sat`` / ``hue_lum``
    are each an empty tuple (OFF), a length-6 control over the anchors red 0°,
    yellow 60°, green 120°, cyan 180°, blue 240°, magenta 300°, or a length-7 one
    that adds ORANGE at 30° in second place — skin and the golden-hour range,
    which six anchors can only reach by dragging the whole warm end. Each length
    carries its OWN anchor array (never a neutral orange spliced into a six-list),
    so a six-slot control interpolates exactly as it did before orange existed and
    every shipped recipe stays bit-identical; the three controls are independent,
    so one may be six slots while another is seven. Each per-band gain is
    interpolated SMOOTHLY around the wheel (``np.interp`` against that anchor array
    made wrap-around by appending a 360° == 0° anchor) rather than at hard band
    edges, so no seam shows between bands. Shift adds (mod 360); sat/lum multiply.

    An all-empty / all-neutral params is a bit-identical no-op by construction
    (early ``rgb.copy()``). Deliberately EXCLUDED from ``params_to_lut3d_cube``:
    HSL rides inside the photo ``_grade`` only, so every video output and baked
    LUT stays bit-identical — exactly as denoise/sharpen are excluded.

    Args:
        rgb: (N, 3) float array in [0, 1].
        params: source of the ``hue_shift`` / ``hue_sat`` / ``hue_lum`` controls.

    Returns:
        (N, 3) array (same dtype), hue-adjusted — or a copy of ``rgb`` if neutral.
    """
    shift, sat_ctrl, lum_ctrl = params.hue_shift, params.hue_sat, params.hue_lum
    neutral = (
        (not shift or all(abs(v) < 1e-9 for v in shift))
        and (not sat_ctrl or all(abs(v - 1.0) < 1e-9 for v in sat_ctrl))
        and (not lum_ctrl or all(abs(v - 1.0) < 1e-9 for v in lum_ctrl))
    )
    if neutral:
        return rgb.copy()

    # RGB → HSV (vectorized, hue in degrees).
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.max(rgb, axis=1).astype(np.float64)
    chroma = mx - np.min(rgb, axis=1)
    nz = chroma > 1e-12
    hue = np.zeros_like(mx)
    is_r = nz & (mx == r)
    is_g = nz & (mx == g) & ~is_r
    is_b = nz & ~is_r & ~is_g
    hue[is_r] = ((g - b)[is_r] / chroma[is_r]) % 6.0
    hue[is_g] = ((b - r)[is_g] / chroma[is_g]) + 2.0
    hue[is_b] = ((r - g)[is_b] / chroma[is_b]) + 4.0
    hue *= 60.0
    sat = np.where(mx > 1e-12, chroma / np.where(nz, mx, 1.0), 0.0)
    val = mx

    # Per-band gains sampled at the ORIGINAL hue, then applied. Each control picks
    # the anchor array its own LENGTH names — six slots keep the pre-orange geometry
    # exactly, seven put orange at 30° — plus a wrap-around 360° == 0° so np.interp
    # blends the magenta→red arc instead of clamping. An OFF control is the neutral
    # scalar: no anchors to choose, and `+ 0.0` / `* 1.0` is exact. Band geometry
    # stays inline (like _apply_saturation's BT.709 constants) so the develop cache
    # hash sees any retune.
    def _gain(ctrl, base):
        if not ctrl:
            return base
        arr = np.asarray(ctrl, dtype=np.float64)
        anchors = (
            [0.0, 30.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0]
            if len(arr) == 7
            else [0.0, 60.0, 120.0, 180.0, 240.0, 300.0, 360.0]
        )
        return np.interp(hue, np.array(anchors), np.append(arr, arr[0]))

    new_hue = np.mod(hue + _gain(shift, 0.0), 360.0)
    sat = np.clip(sat * _gain(sat_ctrl, 1.0), 0.0, 1.0)
    val = np.clip(val * _gain(lum_ctrl, 1.0), 0.0, 1.0)

    # HSV → RGB (vectorized).
    h6 = np.mod(new_hue / 60.0, 6.0)
    i = np.floor(h6).astype(np.int64)
    f = h6 - i
    p = val * (1.0 - sat)
    q = val * (1.0 - sat * f)
    t = val * (1.0 - sat * (1.0 - f))
    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    out_r = np.select(conds, [val, q, p, p, t, val])
    out_g = np.select(conds, [t, val, val, q, p, p])
    out_b = np.select(conds, [p, p, t, val, val, q])
    out = np.clip(np.stack([out_r, out_g, out_b], axis=1), 0.0, 1.0)
    return out.astype(rgb.dtype, copy=False)


def _apply_exposure(rgb: np.ndarray, exposure_ev: float) -> np.ndarray:
    """Multiply by ``2**exposure_ev`` stops (photo develop only, first of the
    chain — ahead of the tone curve, the way a camera's exposure dial precedes
    shadow/highlight recovery). 0.0 is an exact no-op; the caller clips."""
    if exposure_ev == 0.0:
        return rgb.copy()
    return rgb * (2.0**exposure_ev)


def _apply_gamma(rgb: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction (eq filter gamma parameter).

    ffmpeg eq applies gamma as ``v^(1/gamma)`` -- gamma > 1 brightens.

    Args:
        rgb: (N, 3) float64 array in [0, 1].
        gamma: Gamma value (1.0 = no change).

    Returns:
        (N, 3) float64 array with gamma applied.
    """
    if abs(gamma - 1.0) <= 0.001:
        return rgb.copy()

    # Clamp to avoid negative values in power function
    safe = np.clip(rgb, 0.0, 1.0)
    return np.power(safe, 1.0 / gamma)


def lab_mean_std(rgb: np.ndarray) -> tuple[float, ...]:
    """Measure one frame's CIE Lab statistics — the six numbers ``reference_lab`` IS.

    Returns ``(L_mean, a_mean, b_mean, L_std, a_std, b_std)`` in OpenCV's float
    Lab convention (L in [0, 100], a/b in [-127, 127]). Accepts uint8, uint16 or
    float [0, 1] pixels in any shape whose last axis is RGB. The ONE place a
    reference's numbers are produced, so it and ``_apply_reference_match``'s own
    measurement cannot drift apart.
    """
    import cv2

    peak = 65535.0 if rgb.dtype == np.uint16 else 255.0 if rgb.dtype == np.uint8 else 1.0
    flat = np.ascontiguousarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / peak
    lab = cv2.cvtColor(np.clip(flat, 0.0, 1.0), cv2.COLOR_RGB2Lab).reshape(-1, 3)
    return (*(float(v) for v in lab.mean(axis=0)), *(float(v) for v in lab.std(axis=0)))


def _apply_reference_match(rgb: np.ndarray, params: CorrectionParams) -> np.ndarray:
    """Reinhard Lab mean/std transfer toward a reference frame (photo develop only).

    The move a saved look cannot make: instead of recalling PARAMETERS, this
    lands the frame on another frame's measured colour. Both are read in CIE Lab
    — perceptually uniform, and lightness separated from chroma — and each of the
    three channels is shifted and scaled so this frame's mean and standard
    deviation equal ``params.reference_lab``'s. It is deliberately not a
    histogram transfer: two moments per channel carry the tone and the cast
    without importing the reference's tonal quirks onto a different subject.

    It rides ``_grade`` LAST, after ``_apply_gamma``, because any stage after it
    would pull the frame straight back off the reference. Distinct from
    lighting-cluster sharing, which NEUTRALIZES a shoot toward its own shared
    correction: this EMULATES a chosen frame.

    ``rgb`` is (N, 3) float in [0, 1]; returns the same dtype. An empty
    ``params.reference_lab`` (OFF) is a bit-identical no-op by construction
    (early ``rgb.copy()``). A channel with no spread is shifted only — an
    infinite ratio has nothing to scale — and the per-channel scale is railed
    (inline literals, like ``_apply_saturation``'s, so the develop cache hash
    sees any retune) so a near-uniform frame cannot be blown apart. Deliberately
    EXCLUDED from ``params_to_lut3d_cube``: this rides the photo ``_grade`` only,
    so every video output and baked LUT stays bit-identical, exactly as
    ``_apply_hsl`` is excluded.
    """
    ref = params.reference_lab
    if not ref:
        return rgb.copy()

    import cv2

    src = np.clip(np.ascontiguousarray(rgb, dtype=np.float32).reshape(-1, 1, 3), 0.0, 1.0)
    lab = cv2.cvtColor(src, cv2.COLOR_RGB2Lab).reshape(-1, 3)
    mean, std = lab.mean(axis=0), lab.std(axis=0)
    spread = std > 1e-4
    scale = np.clip(
        np.where(spread, np.asarray(ref[3:]) / np.where(spread, std, 1.0), 1.0), 0.0, 4.0
    )
    out = (lab - mean) * scale + np.asarray(ref[:3])
    out[:, 0] = np.clip(out[:, 0], 0.0, 100.0)
    out[:, 1:] = np.clip(out[:, 1:], -127.0, 127.0)
    matched = np.ascontiguousarray(out, dtype=np.float32).reshape(-1, 1, 3)
    out = cv2.cvtColor(matched, cv2.COLOR_Lab2RGB).reshape(rgb.shape)
    return np.clip(out, 0.0, 1.0).astype(rgb.dtype, copy=False)


def params_to_lut3d_cube(
    params: CorrectionParams,
    cube_size: int = 64,
    *,
    output_path: Path,
) -> Path:
    """Bake color correction parameters into a .cube 3D LUT file.

    Pure numpy math -- no ffmpeg dependency. Deterministic output.
    Replicates the CPU filter chain: curves (with the highlight knee folded
    into the top control point) -> colorbalance -> eq(sat+gamma).

    Denoise and sharpen are spatial/temporal filters and cannot be represented
    in a 3D LUT -- they are intentionally excluded.

    Args:
        params: Color correction parameters to bake.
        cube_size: LUT resolution per axis (default 64, yielding 262144 entries).
        output_path: Where to write the .cube file.

    Returns:
        Path to the written .cube file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build 3D grid of RGB sample points in [0, 1]
    # .cube format: R varies fastest, then G, then B (outer loop)
    axis = np.linspace(0.0, 1.0, cube_size, dtype=np.float64)

    # Build flat array in .cube order: R fastest, G middle, B slowest
    rgb_flat = np.empty((cube_size**3, 3), dtype=np.float64)
    idx = 0
    for bi in range(cube_size):
        for gi in range(cube_size):
            for ri in range(cube_size):
                rgb_flat[idx, 0] = axis[ri]
                rgb_flat[idx, 1] = axis[gi]
                rgb_flat[idx, 2] = axis[bi]
                idx += 1

    # Apply filter chain in the SAME order as the CPU filter string
    # (color_filters._build_single_filter): curves (highlight knee folded into
    # the top control point, gated on highlight_pull>=5) -> colorbalance ->
    # saturation -> gamma. The knee replaced the old whole-range colorlevels
    # multiply, which darkened midtones on bright-sky footage; the CPU chain and
    # this mirror fold it identically so the baked LUT stays in parity.
    result = _apply_curves(rgb_flat, params)
    result = _apply_colorbalance(result, params)
    result = _apply_saturation(result, params.saturation_mult)
    result = _apply_gamma(result, params.gamma)

    # Clamp to [0, 1]
    result = np.clip(result, 0.0, 1.0)

    # Write .cube file
    lines = [
        'TITLE "GMS Color Correction"',
        f"LUT_3D_SIZE {cube_size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
        "",
    ]
    # Format data lines
    for i in range(result.shape[0]):
        lines.append(f"{result[i, 0]:.6f} {result[i, 1]:.6f} {result[i, 2]:.6f}")

    output_path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %d-point 3D LUT to %s", cube_size**3, output_path)
    return output_path


def _grade_algo_version() -> str:
    """Short fingerprint of the grade-PRODUCING code so the color cache auto-
    invalidates when the algorithm changes.

    A ``contrast_curve = 0`` dark-grade fix once shipped WITHOUT bumping
    ``COLOR_CACHE_SCHEMA_VERSION``, so a stable color cache written in the buggy
    window replayed the old S-curve on a same-source re-run. Hashing the source
    of the correction producers removes that manual-bump failure mode — change
    the grade math and the cache key changes automatically (robust by construction).
    The param->ffmpeg/LUT translation is rebuilt fresh from the cached params
    every run, so it does NOT affect cache validity and is intentionally not
    hashed. Incidental edits (comments) cause a harmless re-analysis; a frozen
    install with no readable source falls back to a static tag
    (the schema version still bounds staleness).
    """
    import hashlib
    import inspect

    try:
        src = inspect.getsource(compute_correction) + inspect.getsource(compute_cluster_correction)
    except (OSError, TypeError):
        return "src-unavailable"
    return hashlib.sha256(src.encode("utf-8")).hexdigest()[:12]


# Computed once at import: the grade-algorithm fingerprint baked into the color
# cache key (see _grade_algo_version and day_master_encoder color-cache helpers).
GRADE_ALGO_VERSION = _grade_algo_version()
