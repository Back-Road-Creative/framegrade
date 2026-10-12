"""Rendered-frame QA: void, unloaded-tile and flat-black pixel detectors.

Every detector here is pure numpy. The one function that touches disk is
``save_refusal_evidence``, named so the exception stays the only thing that
carries a refused frame out of the render loop. The render loop (``proof.py``'s ``render_frames`` and
``scene_renderer.py``) calls ``check_frame``/``assert_frame_ok`` per rendered
frame.

The scene style (``scene/scene.js: buildStyle``) paints both the base
``background`` layer and the fog beyond bundle coverage ``#101418`` — that is
also the colour a raster tile shows through wherever it hasn't loaded, since a
missing/erroring tile just leaves the background painted underneath. The
scene page (``scene/index.html``) sets no ``body`` background, so wherever the
canvas itself does not paint, the browser's own default white shows through —
the "page-white"/page-background void a wide pose can expose before the
canvas has painted.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter

logger = logging.getLogger(__name__)

# Two decimals, because the comparisons here are strict. A frame fails on
# ``measured > limit``, so printing both at one decimal turned a
# small overshoot into "ends 4.0px from the marker (max 4.0px)" -- a sentence that
# reads as a PASS and costs real diagnosis time. The measured
# value must never render identically to the ceiling it broke.
_PX_DECIMALS = 2


def format_px(value: float) -> str:
    """One pixel measurement, printed so a failing value cannot read as the limit."""
    return f"{value:.{_PX_DECIMALS}f}"


PAGE_VOID_RGB: tuple[int, int, int] = (255, 255, 255)
UNLOADED_RGB: tuple[int, int, int] = (16, 20, 24)  # #101418, scene.js background/fog colour
FLAT_BLACK_MAX = 8  # all channels <= this counts as flat-black terrain
TRAVELLED_LINE_RGB: tuple[int, int, int] = (0xFF, 0xCF, 0x3F)  # scene.js ROUTE_BRIGHT_COLOR


@dataclass(frozen=True)
class FrameQaThresholds:
    max_void_fraction: float = 0.005
    max_unloaded_fraction: float = 0.005
    max_black_fraction: float = 0.005
    tolerance: int = 6
    # Floor for travelled_line_min_width's measurement near the marker; only
    # applied when check_frame is given a marker position (see check_frame).
    min_travelled_line_width_px: float = 2.0
    # Ceiling for line_end_to_marker_px: the gap, in px, between the edge of
    # the marker's exclusion disc and the nearest travelled-line pixel. 0
    # means the line reaches right up to the disc edge (marker sitting on
    # the line, as intended); a growing gap is the "red dot lags behind the
    # bright line" defect that is easy to see by eye. Only applied when
    # check_frame is given a marker position (see check_frame).
    max_line_end_to_marker_px: float = 4.0

    # --- Opening-shot viewer checks (check_opening_frame) -- tuned against
    # openings rendered at fractions 0/0.25/0.5/0.75/1 of the imagery ladder
    # (blue_marble/gibs/naip_coarse/naip bands). min_opening_mean_luma and
    # min_opening_colourfulness sit strictly between the worst values of an
    # ungraded, trailhead-locked camera (a dark, flat opening) and the worst
    # values of the graded camera, so the ungraded opening fails on luma and
    # the graded one clears.
    min_opening_mean_luma: float = 45.0  # 0-255 grayscale mean
    min_opening_colourfulness: float = 12.5  # Hasler-Suesstrunk metric

    # A ladder rung this far below the ladder's OWN median sharpness is a
    # band handover gone soft (see check_opening_ladder). On a ladder with soft
    # handovers the bad rungs sit near a quarter of the median, the healthy ones
    # at about 0.46 and up (the lowest healthy rung is the deliberately wide
    # establishing shot). 0.35 sits between them -- it refuses the soft
    # handovers and clears the healthy rungs. Unlike min_opening_sharpness this is a
    # RATIO, so it carries over to a bundle whose absolute sharpness differs.
    min_opening_ladder_sharpness_ratio: float = 0.35

    # min_opening_sharpness is a pixel-content floor, calibrated against the
    # worst sharpness a still-defective (single-mosaic-level, narrow-bbox)
    # opening camera produces across a dense fraction walk
    # (0/0.4/0.6/0.8/0.85/0.9/1.0): its worst sample sits at a low ratio,
    # upstream of check_opening_sharpness's floor, far below the lowest sample
    # just above that neighbourhood (still below the 0.5 floor, i.e. still a
    # raster the fixed camera would never select). 50.0 sits strictly between
    # the two: it fails the still-broken camera with margin and clears the
    # lowest healthier-but-still-under-floor sample with margin.
    #
    # A much lower floor is a mistake. Such a value comes from measuring the
    # travel+grade camera over a bundle that predates the band ladder, so its
    # worst samples are drawn from rasters at low ratios -- rasters below
    # check_opening_sharpness's own 0.5 floor, which the FIXED camera never
    # selects at all. Calibrating a pixel-content floor to frames the fixed
    # pipeline cannot produce is how a gate ends up certifying the defect it
    # was built to catch.
    # 50.0 is a judgement from the ratio arithmetic, not a measurement
    # against a bundle carrying the full band ladder: if a render trips it,
    # that is the gate reporting, not a false alarm, and the honest fix is to
    # re-measure on a current bundle -- never to lower it until the render
    # passes.
    # check_opening_sharpness (the resolution-based upstream gate, not this
    # pixel-content late gate) is the one invariant proven against the real
    # fix by geometry, not measurement -- see its own tests.
    min_opening_sharpness: float = 50.0  # variance of Laplacian on luma


@dataclass(frozen=True)
class FrameQaResult:
    frame: int
    void_fraction: float
    unloaded_fraction: float
    black_fraction: float
    problems: tuple[str, ...] = ()
    # None when check_frame was not given a marker position, or no travelled
    # line pixel was visible in range -- never a guessed 0.
    travelled_line_width_px: float | None = None
    # Gap in px between the marker's exclusion disc and the nearest
    # travelled-line pixel outside it. None under the same conditions as
    # travelled_line_width_px -- appended last so existing positional/keyword
    # callers are unaffected.
    line_end_to_marker_px: float | None = None

    @property
    def ok(self) -> bool:
        return not self.problems


def marker_xy_from_labels(labels: Iterable[dict]) -> tuple[float, float] | None:
    """The traveller marker's on-screen position from a capture's label list.

    Shared by every ``check_frame`` caller that gets ``labels`` back from
    ``capture.render`` (``scene_renderer.py``, ``proof.py``) so "how do I find
    the marker" has one answer -- the same entry ``place_labels`` (proof.py)
    already keeps label pills clear of.
    """
    for entry in labels:
        if entry.get("kind") == "marker" and entry.get("visible"):
            return (float(entry["x"]), float(entry["y"]))
    return None


def _within(rgb: np.ndarray, target: tuple[int, int, int], tolerance: int) -> np.ndarray:
    """uint8 HxW mask (255 = match) of pixels within ``tolerance`` of ``target``
    on every channel. ``cv2.inRange`` on the uint8 frame is the same inclusive
    test as ``|rgb - target| <= tolerance`` without three int16 frame copies --
    a uint8 channel cannot sit outside [0, 255], so clipping the bounds there
    changes nothing."""
    low = np.array([max(0, c - tolerance) for c in target], dtype=np.uint8)
    high = np.array([min(255, c + tolerance) for c in target], dtype=np.uint8)
    return cv2.inRange(rgb, low, high)


def _color_fraction(
    rgb: np.ndarray,
    target: tuple[int, int, int],
    tolerance: int,
    valid: np.ndarray | None,
    total: int,
) -> float:
    matches = _within(rgb, target, tolerance)
    if valid is not None:
        matches = cv2.bitwise_and(matches, valid)
    return float(cv2.countNonZero(matches)) / total if total else 0.0


def check_frame(
    frame_index: int,
    rgb: np.ndarray,
    thresholds: FrameQaThresholds = FrameQaThresholds(),
    *,
    hud_mask: np.ndarray | None = None,
    marker_xy: tuple[float, float] | None = None,
    marker_geometry_only: bool = False,
    field_pixels: int | None = None,
) -> FrameQaResult:
    """Check one rendered frame for void/unloaded/flat-black pixel fractions.

    ``hud_mask`` (bool, HxW, True = ignore) excludes HUD/letterbox regions
    from every fraction's denominator and numerator.

    ``field_pixels``, when given, is every fraction's denominator instead of
    the unmasked pixel count. A caller whose mask hides an overlay that sits
    INSIDE the judged field (the dashboard rail on a portrait frame, or on a
    small landscape frame where the rail's 72 px floor reaches into the map)
    passes the field's own area: the overlay stops counting as void, and the
    caps keep the calibration they had when the field was judged whole. A
    mask that only shrinks the denominator tips a frame that was under the
    cap over it (0.48% -> 0.52% on a 640x360 export).

    ``marker_xy``, when given, also measures ``travelled_line_min_width`` and
    ``line_end_to_marker_px`` near the marker and fails the frame below
    ``thresholds.min_travelled_line_width_px`` / above
    ``thresholds.max_line_end_to_marker_px``. With no ``marker_xy`` (the
    caller has no marker screen position to hand over) both checks are
    skipped entirely -- never guessed -- and both fields are ``None``.

    ``marker_geometry_only``, when true, skips the void/unloaded/flat-black
    pixel-fraction detectors (their fractions are reported as 0.0, with no
    problems from them) and runs only the marker-geometry checks above. A
    renderer that already QA'd its own raw capture (``checks_raw_frames``)
    has judged pixel fractions on the *raw* frame; the caller re-checks only
    the marker geometry on its composed frame, where dashboard/label pixels
    would otherwise trip the pixel-fraction detectors spuriously.
    """
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
    ):
        raise ValueError(
            f"frame {frame_index}: expected an HxWx3 uint8 array, got "
            f"shape={getattr(rgb, 'shape', None)} dtype={getattr(rgb, 'dtype', None)}"
        )

    height, width = rgb.shape[0], rgb.shape[1]
    problems = []
    void_fraction = unloaded_fraction = black_fraction = 0.0

    rgb = np.ascontiguousarray(rgb)  # cv2 needs it; a no-op for a capture
    if not marker_geometry_only:
        valid: np.ndarray | None = None
        total = height * width
        if hud_mask is not None:
            if hud_mask.shape != (height, width):
                raise ValueError(
                    f"frame {frame_index}: hud_mask shape {hud_mask.shape} "
                    f"does not match frame shape {(height, width)}"
                )
            valid = np.where(hud_mask.astype(bool), 0, 255).astype(np.uint8)
            total = cv2.countNonZero(valid)
        if field_pixels is not None:
            if field_pixels <= 0:
                raise ValueError(
                    f"frame {frame_index}: field_pixels must be positive, got {field_pixels}"
                )
            total = field_pixels

        void_fraction = _color_fraction(rgb, PAGE_VOID_RGB, thresholds.tolerance, valid, total)
        unloaded_fraction = _color_fraction(rgb, UNLOADED_RGB, thresholds.tolerance, valid, total)
        black_fraction = _color_fraction(rgb, (0, 0, 0), FLAT_BLACK_MAX, valid, total)

        for fraction, max_fraction, label in (
            (void_fraction, thresholds.max_void_fraction, "page-white"),
            (unloaded_fraction, thresholds.max_unloaded_fraction, "unloaded-tile"),
            (black_fraction, thresholds.max_black_fraction, "flat-black"),
        ):
            if fraction > max_fraction:
                problems.append(
                    f"frame {frame_index}: {fraction * 100:.2f}% {label} "
                    f"(max {max_fraction * 100:.2f}%)"
                )

    line_width_px = None
    line_end_px = None
    if marker_xy is not None:
        # One window, one mask, both line checks: they ask about the same pixels.
        window = _travelled_line_colour_mask(
            rgb, marker_xy, TRAVELLED_LINE_RGB, _LINE_TOLERANCE, _LINE_RADIUS_PX
        )
        line_width_px = _min_width_in(window, _LINE_RADIUS_PX, _MARKER_EXCLUDE_PX, _END_MARGIN_PX)
        if line_width_px is not None and line_width_px < thresholds.min_travelled_line_width_px:
            problems.append(
                f"frame {frame_index}: travelled line {format_px(line_width_px)}px wide near the "
                f"marker (min {format_px(thresholds.min_travelled_line_width_px)}px)"
            )

        line_end_px = _line_end_in(window, _LINE_RADIUS_PX, _MARKER_EXCLUDE_PX)
        if line_end_px is not None and line_end_px > thresholds.max_line_end_to_marker_px:
            problems.append(
                f"frame {frame_index}: travelled line ends {format_px(line_end_px)}px from the "
                f"marker (max {format_px(thresholds.max_line_end_to_marker_px)}px)"
            )

    return FrameQaResult(
        frame=frame_index,
        void_fraction=void_fraction,
        unloaded_fraction=unloaded_fraction,
        black_fraction=black_fraction,
        problems=tuple(problems),
        travelled_line_width_px=line_width_px,
        line_end_to_marker_px=line_end_px,
    )


# The travelled-line checks' settings, shared by ``check_frame`` and the public
# functions' defaults so the two can never drift apart.
_LINE_RADIUS_PX = 200.0
_LINE_TOLERANCE = 40
_MARKER_EXCLUDE_PX = 12.0
_END_MARGIN_PX = 6.0

# Both line checks ask only about pixels within ``radius_px`` of the marker, so
# they run on a window around it rather than the whole frame -- the full-frame
# version cost 450 ms of a ~1 s film frame. The window reaches this far past
# the radius so the width check's distance transform sees what it would on the
# full frame: cv2 treats beyond-the-edge as line, never as background, so a
# crop can change a pixel's distance only where the true distance is already
# larger than this margin -- a solid line-coloured blob ~120 px across, never a
# stroke. Equality with the full-frame version is pinned by
# tests/test_route_story_frame_qa_speed.py.
_LINE_WINDOW_MARGIN_PX = 64


@dataclass(frozen=True)
class _LineWindow:
    """The line-coloured pixels around the marker and each one's distance from it."""

    colour_mask: np.ndarray  # bool, window-shaped
    dist_from_marker: np.ndarray  # float64, window-shaped


def _travelled_line_colour_mask(
    rgb: np.ndarray,
    marker_xy: tuple[float, float],
    colour: tuple[int, int, int],
    tolerance: int,
    radius_px: float,
) -> _LineWindow:
    """Shared by ``travelled_line_min_width`` and ``line_end_to_marker_px``:
    the pixels matching ``colour`` within ``tolerance``, and each pixel's
    distance from ``marker_xy``, over the window of the frame within
    ``radius_px`` + ``_LINE_WINDOW_MARGIN_PX`` of the marker.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected an HxWx3 array, got shape={rgb.shape}")

    height, width = rgb.shape[0], rgb.shape[1]
    marker_x, marker_y = marker_xy
    reach = radius_px + _LINE_WINDOW_MARGIN_PX
    x0 = min(width, max(0, math.floor(marker_x - reach)))
    x1 = max(x0, min(width, math.ceil(marker_x + reach) + 1))
    y0 = min(height, max(0, math.floor(marker_y - reach)))
    y1 = max(y0, min(height, math.ceil(marker_y + reach) + 1))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist_from_marker = np.sqrt((xx - marker_x) ** 2 + (yy - marker_y) ** 2)
    if x1 == x0 or y1 == y0:
        return _LineWindow(np.zeros(dist_from_marker.shape, dtype=bool), dist_from_marker)
    window = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    return _LineWindow(_within(window, colour, tolerance) > 0, dist_from_marker)


def _line_end_in(window: _LineWindow, radius_px: float, marker_exclude_px: float) -> float | None:
    dist = window.dist_from_marker
    mask = window.colour_mask & (dist >= marker_exclude_px) & (dist <= radius_px)
    if not mask.any():
        return None
    return float(dist[mask].min()) - marker_exclude_px


def line_end_to_marker_px(
    rgb: np.ndarray,
    marker_xy: tuple[float, float],
    *,
    radius_px: float = _LINE_RADIUS_PX,
    colour: tuple[int, int, int] = TRAVELLED_LINE_RGB,
    tolerance: int = _LINE_TOLERANCE,
    marker_exclude_px: float = _MARKER_EXCLUDE_PX,
) -> float | None:
    """Gap, in px, between the edge of the marker's exclusion disc and the
    NEAREST travelled-line-coloured pixel outside that disc, within
    ``radius_px`` of the marker.

    Catches the "red dot lags behind the bright travelled line" defect: the
    marker is drawn correctly, but the line drawn up to it stops short (a
    stale/late line-progress update), leaving a visible gap a viewer's eye
    catches instantly even though no pixel-fraction detector above sees
    anything wrong. 0.0 means the line reaches right up to the disc edge.
    Returns ``None`` when no travelled-line pixel is visible within
    ``radius_px`` at all (e.g. the very first frame, before any line has
    been drawn) -- never a guessed 0.
    """
    window = _travelled_line_colour_mask(
        np.ascontiguousarray(rgb), marker_xy, colour, tolerance, radius_px
    )
    return _line_end_in(window, radius_px, marker_exclude_px)


# Shortest medial axis, in px, that counts as a line to measure a width along.
# Two cases motivate it: a stroke that sits almost wholly under the marker disc
# leaves a tiny island of a few px, and a sampling ring can cut a healthy stroke
# into short pieces. A genuine 1 px hairline measures a ridge as long as it is
# (tens of px even across a small proof frame), so the floor keeps the
# detector's teeth.
_MIN_MEASURABLE_RIDGE_PX = 8


def travelled_line_min_width(
    rgb: np.ndarray,
    marker_xy: tuple[float, float],
    *,
    radius_px: float = _LINE_RADIUS_PX,
    colour: tuple[int, int, int] = TRAVELLED_LINE_RGB,
    tolerance: int = _LINE_TOLERANCE,
    marker_exclude_px: float = _MARKER_EXCLUDE_PX,
    end_margin_px: float = _END_MARGIN_PX,
) -> float | None:
    """Minimum on-screen width, in px, of the travelled-line colour within
    ``radius_px`` of the marker.

    MapLibre drapes ``travelled-line`` onto 3D terrain, so a steep slope
    viewed at a grazing angle can foreshorten it to a sub-pixel hairline even
    though the layer's own ``line-width`` paint property never changes --
    this measures what actually lands on the rendered pixels, not the paint
    property. Returns ``None`` when too little line of the target colour is
    visible in range to measure (the very first frames, the marker disc
    covering it, or a sliver of it), rather than reporting a spurious width.

    Method: a boolean colour mask is distance-transformed (px to nearest
    non-line pixel); the local ridge of that distance field -- pixels equal
    to the local max in a small neighbourhood -- approximates the line's
    medial axis, and twice the ridge value there approximates the line's
    true width at that point. The reported width is the 5th percentile
    (not the strict minimum) over the ridge, excluding points too close to
    the marker disc or the sampling radius's own edge (both of which can
    look artificially thin): a real switchback puts several sharp bends
    within the sampling radius, and a sharp bend's own join pixel reads a
    couple of px narrower than the stroke either side of it purely from
    corner geometry, not a rendering defect -- a strict min chases that
    corner-of-one-pixel noise. A genuine draped-line hairline is a stretch
    of many consecutive thin ridge pixels, which the 5th percentile still
    catches.
    """
    window = _travelled_line_colour_mask(
        np.ascontiguousarray(rgb), marker_xy, colour, tolerance, radius_px
    )
    return _min_width_in(window, radius_px, marker_exclude_px, end_margin_px)


def _min_width_in(
    window: _LineWindow, radius_px: float, marker_exclude_px: float, end_margin_px: float
) -> float | None:
    colour_mask, dist_from_marker = window.colour_mask, window.dist_from_marker
    in_range = (dist_from_marker >= marker_exclude_px) & (
        dist_from_marker <= radius_px - end_margin_px
    )
    in_ring = colour_mask & in_range
    if not in_ring.any():
        return None

    # The transform runs on the WHOLE colour mask of the window, never the
    # ringed subset: cropping to the ring makes the ring's own cut read as
    # background, so a stroke crossing the boundary measures a pixel across at
    # the cut while measuring its true width everywhere else. The window
    # reaches ``_LINE_WINDOW_MARGIN_PX`` past the ring for the same reason.
    # Only the ridge is then sampled inside the ring, which keeps the question
    # local to the marker without letting the crop invent thinness.
    dist_transform = cv2.distanceTransform(colour_mask.astype(np.uint8), cv2.DIST_L2, 5)
    local_max = maximum_filter(dist_transform, size=5)
    ridge = (dist_transform == local_max) & in_ring & (dist_transform > 0)
    # A width is measured ALONG a length, and the ridge IS that length. A few
    # pixels of line colour poking out from under the marker disc -- the walker
    # still beside the start, the stroke drawn correctly but almost wholly
    # covered -- are the absence of a line, not a hairline, and get the same
    # ``None`` as a fully hidden one.
    if int(ridge.sum()) < _MIN_MEASURABLE_RIDGE_PX:
        return None

    # cv2.distanceTransform measures pixel-CENTRE to pixel-centre, so a solid
    # strip n pixels wide has a ridge value of n/2 rounded up to the nearest
    # integer step, i.e. 2*ridge overstates the painted width by exactly 1px
    # (verified against known-width synthetic strips) -- correct for it here
    # rather than at every call site.
    return float(np.percentile(dist_transform[ridge], 5) * 2.0 - 1.0)


def check_opening_frame(
    rgb: np.ndarray, thresholds: FrameQaThresholds = FrameQaThresholds()
) -> tuple[str, ...]:
    """Viewer-legible checks for an opening/establishing shot.

    The pixel-fraction detectors in ``check_frame`` catch a frame that
    failed to render; they say nothing about a frame that rendered but reads
    as dark, flat and blurred to a human eye -- a defect a viewer
    would reject a film for. Three floors, each a plain-English "below floor"
    problem string when tripped:

    - mean luma (0-255 grayscale mean): catches a dark opener.
    - colourfulness (Hasler-Suesstrunk metric on RG/YB opponent channels):
      catches a flat, desaturated opener.
    - sharpness (variance of the Laplacian of luma): catches a blurred/
      out-of-focus opener.
    """
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
    ):
        raise ValueError(
            f"expected an HxWx3 uint8 array, got "
            f"shape={getattr(rgb, 'shape', None)} dtype={getattr(rgb, 'dtype', None)}"
        )

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_luma = float(gray.mean())

    red = rgb[:, :, 0].astype(np.float64)
    green = rgb[:, :, 1].astype(np.float64)
    blue = rgb[:, :, 2].astype(np.float64)
    red_green = red - green
    yellow_blue = 0.5 * (red + green) - blue
    colourfulness = float(
        np.sqrt(red_green.std() ** 2 + yellow_blue.std() ** 2)
        + 0.3 * np.sqrt(red_green.mean() ** 2 + yellow_blue.mean() ** 2)
    )

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    problems = []
    for value, floor, label in (
        (mean_luma, thresholds.min_opening_mean_luma, "mean luma"),
        (colourfulness, thresholds.min_opening_colourfulness, "colourfulness"),
        (sharpness, thresholds.min_opening_sharpness, "sharpness"),
    ):
        if value < floor:
            problems.append(f"opening frame {label} {value:.1f} below floor {floor:.1f}")
    return tuple(problems)


def _opening_sharpness(rgb: np.ndarray) -> float:
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or rgb.dtype != np.uint8
    ):
        raise ValueError(
            f"expected an HxWx3 uint8 array, got "
            f"shape={getattr(rgb, 'shape', None)} dtype={getattr(rgb, 'dtype', None)}"
        )
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def check_opening_ladder(
    frames: Sequence[np.ndarray], thresholds: FrameQaThresholds = FrameQaThresholds()
) -> tuple[str, ...]:
    """Refuse an opening zoom that goes soft partway down the imagery ladder.

    ``check_opening_frame``'s floors are absolute, and a film can look
    dark and fuzzy while zooming in even though every sampled frame clears
    them: sharpness across the ladder can show two sharp dips where the ladder
    hands over to a coarser raster, yet even the worst dip clears the blur
    floor by a wide factor. An absolute floor set high enough to catch it would refuse
    legitimately soft imagery elsewhere.

    The giveaway is the ladder's own SHAPE, so this compares each rung to the
    ladder's median rather than to a constant: a rung below
    ``min_opening_ladder_sharpness_ratio`` of that median is reported by
    index. Self-calibrating, so it needs no per-bundle number and stays
    honest when a whole ladder is legitimately softer or sharper than
    another's.

    A ladder where EVERY rung is equally soft has no shape to fault and is
    left to ``check_opening_frame``'s absolute floor -- the two checks answer
    different questions and neither replaces the other. Fewer than three
    rungs is not a shape either, and is not judged.
    """
    sharpness = [_opening_sharpness(frame) for frame in frames]
    if len(sharpness) < 3:
        return ()
    median = float(np.median(sharpness))
    if median <= 0.0:
        return ()
    floor = median * thresholds.min_opening_ladder_sharpness_ratio
    return tuple(
        f"opening ladder rung {index} sharpness {value:.1f} is "
        f"{value / median:.2f} of the ladder median {median:.1f} "
        f"(floor {thresholds.min_opening_ladder_sharpness_ratio:.2f})"
        for index, value in enumerate(sharpness)
        if value < floor
    )


class FrameQaError(RuntimeError):
    """A frame failed QA; carries the result, the pose and the frame that failed."""

    def __init__(
        self,
        result: FrameQaResult,
        pose: dict | None = None,
        *,
        rgb: np.ndarray | None = None,
    ) -> None:
        problems = "; ".join(result.problems) or "frame QA failed"
        message = f"frame {result.frame} failed frame QA: {problems}"
        if pose is not None:
            message += f" (pose: {pose})"
        super().__init__(message)
        self.result = result
        self.pose = pose
        self.rgb = rgb


@dataclass
class FrameQaFault:
    """One frame that failed QA, kept for the end-of-render report.

    ``rgb`` is the refused pixels, and only the FIRST fault of a render carries
    them: a long leg is thousands of multi-megabyte frames and holding them all
    would turn a reporting feature into an out-of-memory crash.
    """

    result: FrameQaResult
    pose: dict | None = None
    rgb: np.ndarray | None = field(default=None, repr=False)


@dataclass
class FrameQaLedger:
    """Collects QA faults during a render instead of aborting it.

    A route-story film can capture for hours and then raise on a single frame
    whose travelled line ends a few pixels from the marker -- entirely
    cosmetic. The staging root is a
    ``TemporaryDirectory``, so the refusal took every frame of the run with it:
    no mp4, no receipt, nothing to look at. That is the late-gate anti-pattern,
    where the cost of the gate firing is the whole artifact.

    The gate itself is correct and stays -- it is the detector for the marker
    lagging the travelled line, a defect visible by eye. What
    changes is the price of it firing. A render given a ledger finishes, writes
    its film, reports every faulted frame, and THEN fails. A reviewer gets both
    the footage and the exact list; nobody re-renders for hours to learn
    which frame it was.

    Written from capture WORKER THREADS (``_capture_leg_parallel`` fans the
    capture step across workers), so every mutation takes the lock.
    """

    faults: list[FrameQaFault] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(
        self,
        result: FrameQaResult,
        pose: dict | None = None,
        *,
        rgb: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            keep = rgb if not self.faults else None
            self.faults.append(FrameQaFault(result=result, pose=pose, rgb=keep))

    @property
    def fault_count(self) -> int:
        with self._lock:
            return len(self.faults)

    def summary(self) -> dict:
        """Receipt/report numbers: how many frames faulted and which sentences."""
        with self._lock:
            faults = list(self.faults)
        problems: dict[str, int] = {}
        for fault in faults:
            for problem in fault.result.problems:
                # Strip the per-frame prefix so the same defect on 900 frames
                # reads as one row with a count, not 900 near-identical lines.
                _, _, sentence = problem.partition(": ")
                problems[sentence or problem] = problems.get(sentence or problem, 0) + 1
        return {
            "frames_faulted": len(faults),
            "frames": [fault.result.frame for fault in faults],
            "problems": problems,
        }


def save_refusal_evidence(error: FrameQaError, output: Path | str) -> list[Path]:
    """Write a refused frame and its pose beside ``output``; return what landed.

    A film renders every frame inside a ``TemporaryDirectory`` that is deleted
    as the refusal unwinds, so the pixels the run judged exist only on the
    exception. ``output`` is the operator's own film path, which outlives the
    staging root -- the evidence goes next to it, named after the film so two
    renders sharing a directory cannot overwrite each other.

    The record carries the measured numbers as well as the sentence: arguing
    about a threshold needs the value that lost, and re-deriving it means
    re-rendering the pose against the same bundle.

    Writing evidence never raises. The refusal is the failure worth reporting
    and an unwritable directory must not replace it with an unrelated
    ``OSError``; whatever did land is returned either way. It is never SILENT
    either: a failure swallowed bare would leave no way to tell whether the
    evidence had been lost or never attempted, so losing it is logged.
    """
    output = Path(output)
    stem = f"{output.stem}.qa-refused"
    written: list[Path] = []
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        record: dict = {
            "frame": error.result.frame,
            "problems": list(error.result.problems),
            "pose": error.pose,
            "void_fraction": error.result.void_fraction,
            "unloaded_fraction": error.result.unloaded_fraction,
            "black_fraction": error.result.black_fraction,
            "travelled_line_width_px": error.result.travelled_line_width_px,
            "line_end_to_marker_px": error.result.line_end_to_marker_px,
        }
        if error.rgb is not None:
            image_path = output.parent / f"{stem}.png"
            Image.fromarray(np.asarray(error.rgb, dtype=np.uint8), mode="RGB").save(image_path)
            written.append(image_path)
            record["frame_image"] = image_path.name
        sidecar = output.parent / f"{stem}.json"
        sidecar.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(sidecar)
    except (OSError, ValueError, TypeError) as exc:
        # TypeError joins the list because a pose carrying a numpy scalar is
        # not JSON-serialisable, and that must not be the one exception that
        # escapes and replaces the refusal with an unrelated traceback.
        logger.warning(
            "route-story frame QA could not write its refusal evidence beside %s: %s", output, exc
        )
        return written
    return written


def assert_frame_ok(result: FrameQaResult, pose: dict | None = None) -> None:
    if not result.ok:
        raise FrameQaError(result, pose)


def summarize(results: Iterable[FrameQaResult]) -> dict:
    """Receipt numbers: frames checked/failed and the worst fraction per metric."""
    results = list(results)
    summary: dict = {
        "frames_checked": len(results),
        "frames_failed": sum(1 for r in results if not r.ok),
    }
    for key, attr in (
        ("void", "void_fraction"),
        ("unloaded", "unloaded_fraction"),
        ("black", "black_fraction"),
    ):
        if results:
            worst = max(results, key=lambda r: getattr(r, attr))
            summary[key] = {"worst_fraction": getattr(worst, attr), "frame": worst.frame}
        else:
            summary[key] = {"worst_fraction": 0.0, "frame": None}

    measured = [r for r in results if r.travelled_line_width_px is not None]
    if measured:
        thinnest = min(measured, key=lambda r: r.travelled_line_width_px)
        summary["travelled_line_width"] = {
            "worst_px": thinnest.travelled_line_width_px,
            "frame": thinnest.frame,
        }
    else:
        summary["travelled_line_width"] = {"worst_px": None, "frame": None}
    return summary
