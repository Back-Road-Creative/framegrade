"""Render a Route Story terrain-proof clip from a prepared offline map bundle.

Builds a 20-30 s :class:`~gomoveshift.pipeline.route_story.planner.ShotPlan`
(descent onto the route, reveal of the DEM's most relief-rich section, an
along-route ``chase`` glued to the real recorded track, a settling arrival,
and a pullback to the whole route), renders every frame through a
``SceneCapture``-shaped object, pipes frames to ffmpeg, and returns a receipt
dict recording sources, zooms, timing, and a determinism check. Browser
capture and the ffmpeg pipe are both injectable so the shot-planning and
frame-loop logic is testable without a real Chromium or ffmpeg process.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bundle import decode_terrain_rgb, lonlat_to_tile
from .frame_qa import assert_frame_ok, check_frame, marker_xy_from_labels, summarize
from .gpx import select_gpx_day
from .hud import _BOLD_FONT
from .metrics import RoutePoint, distance_m
from .opening import CameraPose, _smootherstep
from .planner import (
    ENLARGEMENT_MAX,
    Shot,
    ShotPlan,
    _initial_bearing_deg,
    _mercator_lonlat,
    _mercator_xy,
    _shortest_angle_delta,
    arrival,
    assert_smooth,
    descent,
    fit_zoom,
    max_shot_zoom,
    pullback,
    reveal,
)
from .scene.capture import SceneCapture, SceneRenderError, determinism_check
from .sun import SceneLight, light_at, light_for_pose, sun_sweep

logger = logging.getLogger(__name__)

FINAL_ZOOM = 15.0
# planner.DEFAULT_MOTION_LIMITS caps pitch_speed_deg_s at 15 and a
# smootherstep ease's peak rate is amplitude * 1.875 / duration, so a pitch
# swing this size needs >= 35*1.875/15 = 4.375s to stay under the cap with
# margin -- this bounds REVEAL_FINAL_PITCH against PULLBACK_SECONDS below.
REVEAL_FINAL_PITCH = 35.0
# A "cinematic flight pace", not the vehicle's real speed -- but bounded by
# real geometry, not chosen freely: the chase's per-segment heading transition
# rate scales linearly with this speed (peak_rate ~= delta_bearing * 1.875 *
# CHASE_SPEED_MPS / segment_length_m, since heading is paced proportional to
# each segment's own real arc length -- see chase()._bearing_at_time_fraction).
# Chosen to keep peak bearing_speed_deg_s under planner.DEFAULT_MOTION_LIMITS's
# 25 deg/s cap with a comfortable margin.
CHASE_SPEED_MPS = 30.0
DESCENT_SECONDS = 14.0  # planner.descent()'s own floor -- cannot go lower
REVEAL_SECONDS = 6.0  # planner.reveal()'s own floor -- cannot go lower
CHASE_SECONDS = 3.5
ARRIVAL_SECONDS = 1.5
PULLBACK_SECONDS = 5.0  # keeps peak pitch_speed_deg_s under the 15 deg/s cap
RELIEF_WINDOW_POINTS = 30
# A consumer GPS fix is good to a few metres, so the bearing between two
# fixes closer than this is noise, not the road. chase() gives each segment a
# time share proportional to its length, so a sub-metre hop got under one
# frame for a 16 degree heading change: hundreds of deg/s of pitch.
CHASE_MIN_POINT_SPACING_M = 3.0


# ---------------------------------------------------------------------------
# Relief section selection (from the bundle's own DEM tiles)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliefSection:
    """The route index with the greatest nearby elevation range, from the DEM."""

    index: int
    relief_range_m: float
    window_points: int


def _forward_fill(values: Sequence[float | None]) -> list[float]:
    filled: list[float | None] = list(values)
    last: float | None = None
    for i, value in enumerate(filled):
        if value is None:
            filled[i] = last
        else:
            last = value
    next_value: float | None = None
    for i in range(len(filled) - 1, -1, -1):
        if filled[i] is None:
            filled[i] = next_value
        else:
            next_value = filled[i]
    if any(value is None for value in filled):
        raise ValueError("terrain samples are entirely missing along the route")
    return [float(value) for value in filled]  # type: ignore[arg-type]


def choose_relief_section(
    points: Sequence[RoutePoint],
    sample_height: Callable[[float, float], float | None],
    *,
    window_points: int = RELIEF_WINDOW_POINTS,
) -> ReliefSection:
    """Pick the route index whose local elevation window has the widest range."""
    heights = [sample_height(point.longitude, point.latitude) for point in points]
    if sum(1 for h in heights if h is not None) < 2:
        raise ValueError("not enough terrain samples along the route to choose a relief section")
    filled = _forward_fill(heights)
    half = max(1, window_points // 2)
    best_index, best_range = 0, -1.0
    for i in range(len(filled)):
        lo, hi = max(0, i - half), min(len(filled), i + half + 1)
        window = filled[lo:hi]
        span = max(window) - min(window)
        if span > best_range:
            best_range = span
            best_index = i
    return ReliefSection(index=best_index, relief_range_m=best_range, window_points=window_points)


def terrain_height_sampler(bundle_root: Path, zoom: int) -> Callable[[float, float], float | None]:
    """Build a height sampler backed by this bundle's own written terrain tiles."""

    def sample(lon: float, lat: float) -> float | None:
        x_frac, y_frac = lonlat_to_tile(lon, lat, zoom)
        x, y = int(x_frac), int(y_frac)
        tile_path = bundle_root / "terrain" / str(zoom) / str(x) / f"{y}.png"
        if not tile_path.is_file():
            return None
        with Image.open(tile_path) as image:
            array = np.asarray(image.convert("RGB"))
        px = min(int((x_frac - x) * array.shape[1]), array.shape[1] - 1)
        py = min(int((y_frac - y) * array.shape[0]), array.shape[0] - 1)
        return float(decode_terrain_rgb(array[py : py + 1, px : px + 1])[0, 0])

    return sample


# ---------------------------------------------------------------------------
# Chase sub-route selection
# ---------------------------------------------------------------------------


def select_chase_points(
    points: Sequence[RoutePoint], start_index: int, target_distance_m: float
) -> list[RoutePoint]:
    """Walk forward (falling back to backward) from ``start_index`` for ~target_distance_m.

    A point closer than ``CHASE_MIN_POINT_SPACING_M`` to the last kept point
    is skipped (its distance still counts toward the target): its bearing is
    GPS noise, and chase() would cross it in under a frame.
    """
    if not 0 <= start_index < len(points):
        raise ValueError("chase start index out of range")

    def _walk(step: int) -> list[RoutePoint]:
        selected = [points[start_index]]
        covered = 0.0
        i = start_index
        while covered < target_distance_m and 0 <= i + step < len(points):
            covered += distance_m(points[i], points[i + step])
            if distance_m(selected[-1], points[i + step]) >= CHASE_MIN_POINT_SPACING_M:
                selected.append(points[i + step])
            i += step
        if step < 0:
            selected.reverse()
        return selected

    forward = _walk(1)
    if len(forward) >= 2:
        return forward
    backward = _walk(-1)
    if len(backward) >= 2:
        return backward
    raise ValueError("not enough distinct route points to build a chase")


# ---------------------------------------------------------------------------
# The "chase" shot: camera glued to the real recorded route, smoothed heading
# ---------------------------------------------------------------------------


def _smoothed_bearings(points: Sequence[RoutePoint], window: int = 9) -> list[float]:
    """Circular-mean-smoothed vertex-to-next-vertex bearings (len(points)-1 of them).

    Raw point-to-point bearings on a real GPX track are noisy (GPS jitter and
    real sharp turns); averaging over a small neighboring window gives a
    heading signal with no frame-to-frame jitter without moving the camera
    off the real recorded path.
    """
    raw = [
        _initial_bearing_deg(a.longitude, a.latitude, b.longitude, b.latitude)
        for a, b in zip(points, points[1:])
    ]
    if len(raw) <= 1:
        return raw
    half = window // 2
    smoothed = []
    for i in range(len(raw)):
        lo, hi = max(0, i - half), min(len(raw), i + half + 1)
        window_vals = raw[lo:hi]
        sin_sum = sum(math.sin(math.radians(b)) for b in window_vals)
        cos_sum = sum(math.cos(math.radians(b)) for b in window_vals)
        smoothed.append(math.degrees(math.atan2(sin_sum, cos_sum)) % 360.0)
    return smoothed


def _arc_length_progress(
    points: Sequence[RoutePoint],
) -> tuple[list[tuple[float, float]], list[float], list[float], float]:
    """Per-segment Mercator length bookkeeping shared by position and time lookups."""
    xy = [_mercator_xy(point.longitude, point.latitude) for point in points]
    seg_lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(xy, xy[1:])]
    total_length = sum(seg_lengths)
    cumulative = [0.0]
    for length in seg_lengths:
        cumulative.append(cumulative[-1] + length)
    return xy, seg_lengths, cumulative, total_length


def _segment_at_fraction(
    fraction: float, seg_lengths: Sequence[float], cumulative: Sequence[float], total_length: float
) -> tuple[int, float]:
    """(segment index, local 0..1) for an arc-length ``fraction`` of the whole polyline."""
    target = fraction * total_length
    seg = 0
    while seg < len(seg_lengths) - 1 and cumulative[seg + 1] < target:
        seg += 1
    seg_len = seg_lengths[seg] or 1e-9
    local = min(max((target - cumulative[seg]) / seg_len, 0.0), 1.0)
    return seg, local


def chase_route_time_s(
    points: Sequence[RoutePoint], frame: int, start_frame: int, frame_count: int
) -> float:
    """The route's own recorded ``time_s`` at ``frame``'s position within a chase shot.

    Uses the exact same arc-length fraction ``chase()``'s ``pose_fn`` uses for
    position (``_smootherstep`` on ``frame``/``frame_count``, then the same
    arc-length-weighted segment lookup) so the light's clock always matches
    where the camera actually is along the real recorded track, instead of a
    second, independently-invented frame-to-time mapping.
    """
    _, seg_lengths, cumulative, total_length = _arc_length_progress(points)
    fraction = _smootherstep((frame - start_frame) / frame_count)
    seg, local = _segment_at_fraction(fraction, seg_lengths, cumulative, total_length)
    return points[seg].time_s + (points[seg + 1].time_s - points[seg].time_s) * local


def chase(
    points: Sequence[RoutePoint],
    zoom: float,
    pitch: float,
    fps: float,
    seconds: float,
    *,
    start_frame: int = 0,
) -> Shot:
    """Camera position is glued to the real recorded route; only heading is smoothed.

    Zoom and pitch are held constant -- only position (arc-length along the
    real polyline) and bearing move, eased in/out over the whole shot so its
    boundary derivatives match the zero-velocity convention every other
    builder in ``planner`` uses.
    """
    if len(points) < 2:
        raise ValueError("chase needs at least two route points")
    if seconds <= 0:
        raise ValueError("chase duration must be positive")
    frame_count = round(seconds * fps)
    end_frame = start_frame + frame_count

    xy, seg_lengths, cumulative, total_length = _arc_length_progress(points)
    if total_length <= 0:
        raise ValueError("chase route points are coincident")
    bearings = _smoothed_bearings(points)
    num_segments = len(seg_lengths)

    def _position_at_fraction(fraction: float) -> tuple[float, float]:
        seg, local = _segment_at_fraction(fraction, seg_lengths, cumulative, total_length)
        ax, ay = xy[seg]
        bx, by = xy[seg + 1]
        return _mercator_lonlat(ax + (bx - ax) * local, ay + (by - ay) * local)

    def _bearing_at_time_fraction(time_fraction: float) -> float:
        # Position is paced by arc length (eased overall, so apparent speed
        # is roughly constant along the real route). Heading uses its OWN,
        # separate LINEAR (un-eased) arc-length-to-time schedule instead of
        # reusing that eased pacing: the arc-length ease moves fastest at
        # shot mid-time, so a vertex that happens to fall near the middle
        # would get squeezed into far fewer real-time frames than one near
        # either end, and even a fully eased per-segment blend spikes
        # bearing_accel_deg_s2 once that crossing is fast enough.
        #
        # An earlier version gave every vertex an EQUAL share of the shot's
        # duration regardless of its real-world segment length. That is
        # wrong in the opposite direction: a route with many closely-spaced
        # GPS points (any real track has this) then divides the shot's
        # duration into many more, much narrower per-vertex time windows
        # than a route with few points -- inflating the instantaneous
        # heading rate by a factor of num_segments for no route-geometry
        # reason. Allocating each segment a time share PROPORTIONAL TO ITS
        # OWN REAL ARC LENGTH (exactly mirroring _position_at_fraction's own
        # lookup, just against linear time instead of the eased fraction)
        # removes that segment-count amplification: the per-segment
        # transition rate this produces is (within the smootherstep peak
        # factor of 1.875) the route's own real yaw rate at CHASE_SPEED_MPS,
        # which is the quantity CHASE_SPEED_MPS was tuned against.
        target = min(max(time_fraction, 0.0), 1.0) * total_length
        seg = 0
        while seg < num_segments - 1 and cumulative[seg + 1] < target:
            seg += 1
        seg_len = seg_lengths[seg] or 1e-9
        local = min(max((target - cumulative[seg]) / seg_len, 0.0), 1.0)
        bearing_a = bearings[seg]
        bearing_b = bearings[min(seg + 1, num_segments - 1)]
        # smootherstep (not linear) in `local`: rate and acceleration are both
        # zero at local=0 and local=1, so heading rate doesn't jump at a
        # vertex crossing -- the same zero-derivative-at-the-seam convention
        # every builder in `planner` relies on for cross-shot smoothness. It
        # also gives the whole shot zero bearing rate at frame=start_frame and
        # frame=end_frame for free, since those land exactly on local=0 of the
        # first segment and local=1 of the last.
        eased_local = _smootherstep(local)
        return (bearing_a + _shortest_angle_delta(bearing_a, bearing_b) * eased_local) % 360.0

    def pose_fn(frame: int) -> CameraPose:
        arc_eased = _smootherstep((frame - start_frame) / frame_count)
        longitude, latitude = _position_at_fraction(arc_eased)
        bearing = _bearing_at_time_fraction((frame - start_frame) / frame_count)
        return CameraPose(longitude, latitude, zoom, pitch, bearing)

    return Shot("chase", start_frame, end_frame, pose_fn)


# ---------------------------------------------------------------------------
# Full shot plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProofPlan:
    plan: ShotPlan
    relief: ReliefSection
    chase_points: list[RoutePoint]
    chase_distance_m: float
    route_bbox: tuple[float, float, float, float]


def build_shot_plan(
    points: Sequence[RoutePoint],
    sample_height: Callable[[float, float], float | None],
    fps: float,
    width_px: float,
    height_px: float,
    *,
    min_zoom: float = 0.0,
    max_zoom: float = 20.0,
) -> ProofPlan:
    """Assemble and validate the descent -> reveal -> chase -> arrival -> pullback plan.

    ``min_zoom`` is the bundle's own imagery floor: the wide overview and the
    pullback are clamped to it, because a zoom the bundle wrote no tiles for
    renders nothing at best and leaves the scene unsettled at worst.
    ``max_zoom`` is the render's zoom cap (``planner.max_shot_zoom``): only
    the arrival's cosmetic push-in is clamped to it. A cap below the chase's
    own zoom is left for ``assert_within_zoom`` to reject, since the bundle
    then cannot hold the proof at all.
    """
    if len(points) < 2:
        raise ValueError("Route Story proof needs at least two route points")
    lons = [point.longitude for point in points]
    lats = [point.latitude for point in points]
    route_bbox = (min(lons), min(lats), max(lons), max(lats))

    relief = choose_relief_section(points, sample_height)
    chase_target_distance_m = CHASE_SPEED_MPS * CHASE_SECONDS
    chase_points = select_chase_points(points, relief.index, chase_target_distance_m)
    chase_distance_m = sum(distance_m(a, b) for a, b in zip(chase_points, chase_points[1:]))

    heading = _smoothed_bearings(chase_points)[0]
    center_lon = (route_bbox[0] + route_bbox[2]) / 2.0
    center_lat = (route_bbox[1] + route_bbox[3]) / 2.0
    overview_zoom = max(fit_zoom(route_bbox, width_px, height_px, padding_px=80.0), min_zoom)
    overview = CameraPose(center_lon, center_lat, overview_zoom, 0.0, heading)

    relief_point = chase_points[0]
    descent_shot = descent(
        overview, relief_point, fps, DESCENT_SECONDS, final_zoom=FINAL_ZOOM, start_frame=0
    )
    descent_end = descent_shot._boundary_pose(descent_shot.end_frame)

    reveal_shot = reveal(
        descent_end,
        fps,
        REVEAL_SECONDS,
        final_pitch=REVEAL_FINAL_PITCH,
        start_frame=descent_shot.end_frame,
    )
    reveal_end = reveal_shot._boundary_pose(reveal_shot.end_frame)

    chase_shot = chase(
        chase_points,
        reveal_end.zoom,
        reveal_end.pitch,
        fps,
        CHASE_SECONDS,
        start_frame=reveal_shot.end_frame,
    )
    chase_end = chase_shot._boundary_pose(chase_shot.end_frame)

    arrival_target = CameraPose(
        chase_end.longitude,
        chase_end.latitude,
        max(min(chase_end.zoom + 0.6, max_zoom), chase_end.zoom),
        chase_end.pitch,
        chase_end.bearing,
    )
    arrival_shot = arrival(
        chase_end, arrival_target, fps, ARRIVAL_SECONDS, start_frame=chase_shot.end_frame
    )
    arrival_end = arrival_shot._boundary_pose(arrival_shot.end_frame)

    pullback_shot = pullback(
        arrival_end,
        route_bbox,
        fps,
        PULLBACK_SECONDS,
        width_px=width_px,
        height_px=height_px,
        padding_px=80.0,
        start_frame=arrival_shot.end_frame,
        min_zoom=min_zoom,
    )

    plan = ShotPlan((descent_shot, reveal_shot, chase_shot, arrival_shot, pullback_shot))
    plan.validate()
    assert_smooth(plan, fps)

    return ProofPlan(
        plan=plan,
        relief=relief,
        chase_points=chase_points,
        chase_distance_m=chase_distance_m,
        route_bbox=route_bbox,
    )


def _shot_at(plan: ShotPlan, frame: int) -> Shot:
    """The :class:`Shot` covering ``frame``, by the same ``[start, end)`` scan ``ShotPlan.pose_at`` uses."""
    for shot in plan.shots:
        if shot.start_frame <= frame < shot.end_frame:
            return shot
    raise ValueError(f"frame {frame} is outside the shot plan")


def marker_position(
    plan: ShotPlan, frame: int, relief_point: RoutePoint, chase_end_point: RoutePoint
) -> tuple[float, float]:
    """Where the on-screen position marker sits for a given frame."""
    shot = _shot_at(plan, frame)
    if shot.kind == "chase":
        pose = shot.pose_at(frame)
        return pose.longitude, pose.latitude
    if shot.kind in ("descent", "reveal"):
        return relief_point.longitude, relief_point.latitude
    return chase_end_point.longitude, chase_end_point.latitude


def route_light_fn(
    plan: ShotPlan,
    points: Sequence[RoutePoint],
    chase_points: Sequence[RoutePoint],
    trip_epoch: datetime,
    center_lat: float,
    center_lon: float,
) -> Callable[[int], SceneLight]:
    """Build the per-frame :class:`SceneLight` lookup for ``render_frames``.

    Chase frames use the sun's real position at *that frame's own camera
    position and the route's own recorded clock time* (see
    ``chase_route_time_s``, which mirrors ``chase()``'s own arc-length
    mapping). The closing (pullback) shot sweeps the sun from the chase's own
    end time to the trip's end time via ``sun_sweep``. Every other shot
    (descent, reveal, arrival -- none has its own route time) falls back to
    the trip's start time, at the route's centre, matching the honest
    single-light baseline this replaces.
    """
    chase_shot = next(s for s in plan.shots if s.kind == "chase")
    closing_shot = plan.shots[-1]
    trip_start_time = trip_epoch + timedelta(seconds=min(p.time_s for p in points))
    trip_end_time = trip_epoch + timedelta(seconds=max(p.time_s for p in points))
    chase_end_time = trip_epoch + timedelta(seconds=chase_points[-1].time_s)

    closing_frame_count = closing_shot.end_frame - closing_shot.start_frame
    closing_sweep: list[SceneLight] | None = None
    if closing_shot is not chase_shot and closing_frame_count >= 2:
        closing_sweep = sun_sweep(
            center_lat, center_lon, chase_end_time, trip_end_time, closing_frame_count
        )

    def light_fn(frame: int) -> SceneLight:
        shot = _shot_at(plan, frame)
        if shot.kind == "chase":
            frame_count = shot.end_frame - shot.start_frame
            time_s = chase_route_time_s(chase_points, frame, shot.start_frame, frame_count)
            pose = shot.pose_at(frame)
            pose_time = trip_epoch + timedelta(seconds=time_s)
            return light_for_pose(
                pose_time, pose.latitude, pose.longitude, fallback=trip_start_time
            )
        if shot is closing_shot:
            if closing_sweep is not None:
                return closing_sweep[frame - closing_shot.start_frame]
            # A one-frame closing shot: sun_sweep needs >= 2 frames, so fall
            # back to the trip's own end light rather than a degenerate sweep.
            return light_at(center_lat, center_lon, trip_end_time)
        return light_for_pose(None, center_lat, center_lon, fallback=trip_start_time)

    return light_fn


# ---------------------------------------------------------------------------
# Frame rendering: labels, sinks, the render loop
# ---------------------------------------------------------------------------


# Label text is sized off the frame height so a 1080p film does not inherit the
# ~10 px bitmap font that was only ever legible on a small proof clip.
LABEL_FONT_FRACTION = 0.021
LABEL_FONT_MIN_PX = 12
_LABEL_FILL = (10, 20, 30)


def label_font_px(height: int) -> int:
    """Label text height for a frame this tall (about 23 px at 1080p)."""
    return max(LABEL_FONT_MIN_PX, round(height * LABEL_FONT_FRACTION))


def _label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(_BOLD_FONT, size)
    except OSError:
        # Pillow's bundled face scales too, so a box with no DejaVu still gets
        # readable labels instead of the fixed-size bitmap font.
        return ImageFont.load_default(size=size)


@dataclass(frozen=True)
class PlacedLabel:
    """One label pill that won its place: the text, where it goes, and its box."""

    text: str
    text_xy: tuple[float, float]
    box: tuple[float, float, float, float]


def _boxes_overlap(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float], gap: float
) -> bool:
    return a[0] - gap < b[2] and b[0] - gap < a[2] and a[1] - gap < b[3] and b[1] - gap < a[3]


# The marker is a 7 px circle with a 2 px stroke in a 1080-line scene
# (``scene.js``); this keeps a pill clear of it and of its heading glyph.
MARKER_CLEAR_FRACTION = 0.02


def marker_clear_box(x: float, y: float, height: int) -> tuple[float, float, float, float]:
    """The square around the traveller's marker that no label pill may touch."""
    reach = max(12.0, height * MARKER_CLEAR_FRACTION)
    return (x - reach, y - reach, x + reach, y + reach)


def place_labels(
    labels: Sequence[dict],
    width: int,
    height: int,
    extra_keep_clear: Sequence[tuple[float, float, float, float]] = (),
) -> list[PlacedLabel]:
    """Choose which labels a frame shows so that no two pills overlap.

    Labels are taken in the order given (the label file's own priority order)
    and a label whose pill would touch one already placed is left out -- a
    clean subset reads better than a stack of pills hiding each other.

    An entry of ``kind`` ``"marker"`` is not a label: it is where the scene
    drew the traveller, and no pill may cover it. A pill that would is moved
    just above the marker, else just below; with neither free it is left out.
    The name of the place the traveller is standing on is the one worth
    keeping, and the traveller is the one thing a frame may never hide.

    ``extra_keep_clear`` is zero or more additional ``(left, top, right,
    bottom)`` pixel boxes a caller wants no pill drawn over -- the dashboard
    rail and elevation-profile boxes ``scene_renderer`` resolves once per
    frame size via ``layout.hud_layout()``, or any other exogenous region a
    caller knows about. They are folded into the same ``keep_clear`` list a
    marker box uses, so a colliding pill gets the identical above/below
    recovery a marker collision already gets, never a special case.

    A pill is anchored at its label's own ``x``, so one near a side edge used
    to draw past the frame and the name came out cut in half. It is slid back
    inside instead of dropped -- the label still belongs to a feature the
    viewer can see, and a whole name a few pixels off its anchor reads better
    than half a name on it. Sliding happens before the collision checks, so a
    pill that slides onto an earlier one is still left out. A name too wide
    for the frame at any position has no honest placement and is dropped.
    """
    size = label_font_px(height)
    font = _label_font(size)
    pad = max(3, round(size * 0.3))
    gap = max(2, round(size * 0.2))
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    keep_clear = [
        marker_clear_box(float(entry["x"]), float(entry["y"]), height)
        for entry in labels
        if entry.get("kind") == "marker" and entry.get("visible")
    ]
    keep_clear.extend(extra_keep_clear)
    placed: list[PlacedLabel] = []
    for label in labels:
        if not label.get("visible") or label.get("kind") == "marker":
            continue
        text = str(label.get("name") or "")
        if not text:
            continue
        x, y = float(label["x"]), float(label["y"])
        bbox = probe.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        pill_width = w + 2 * pad
        pill_height = h + 2 * pad
        if pill_width > width:
            continue
        left = min(max(x - pad, 0.0), float(width - pill_width))
        tops = [y - h - pad]
        for clear in keep_clear:
            if _boxes_overlap((left, tops[0], left + pill_width, tops[0] + pill_height), clear, 0):
                tops = [clear[1] - gap - pill_height, clear[3] + gap]
                break
        for top in tops:
            box = (left, top, left + pill_width, top + pill_height)
            if top < 0 or box[3] > height:
                continue
            if any(_boxes_overlap(box, clear, 0) for clear in keep_clear):
                continue
            if any(_boxes_overlap(box, other.box, gap) for other in placed):
                continue
            placed.append(PlacedLabel(text, (left + pad - bbox[0], top + pad - bbox[1]), box))
            break
    return placed


def _draw_labels(
    frame: np.ndarray,
    labels: Sequence[dict],
    extra_keep_clear: Sequence[tuple[float, float, float, float]] = (),
) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = _label_font(label_font_px(image.height))
    for pill in place_labels(labels, image.width, image.height, extra_keep_clear):
        draw.rectangle(list(pill.box), fill=_LABEL_FILL)
        draw.text(pill.text_xy, pill.text, fill=(255, 255, 255), font=font)
    return np.asarray(image, dtype=np.uint8)


class ListFrameSink:
    """In-memory sink for tests -- collects frames instead of piping to ffmpeg."""

    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def close(self) -> bool:
        return True


def _ffmpeg_argv(output_path: Path, width: int, height: int, fps: float) -> list[str]:
    """The ffmpeg invocation itself, factored out so a test can substitute it."""
    return [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]


class FfmpegFrameSink:
    """Pipes raw RGB24 frames into ffmpeg: libx264, crf 18, preset slow, yuv420p.

    ffmpeg's own stderr is redirected to a log file next to the output,
    NEVER left on an undrained ``subprocess.PIPE``: once ffmpeg writes more
    than the pipe's own buffer (64 KB on Linux) with nothing reading it, it
    blocks inside its own stderr write and ``close()``'s ``proc.wait()``
    never returns -- every long render used to hang at 100% frames written.
    """

    def __init__(self, output_path: Path, width: int, height: int, fps: float) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path = output_path
        self._log_path = output_path.with_name(output_path.name + ".ffmpeg.log")
        self._log_file = open(self._log_path, "wb")
        self._proc = subprocess.Popen(
            _ffmpeg_argv(output_path, width, height, fps),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log_file,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg frame sink has no stdin pipe (already closed?)")
        self._proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def _log_tail(self, max_lines: int = 20) -> str:
        try:
            text = self._log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(ffmpeg log unavailable)"
        return "\n".join(text.splitlines()[-max_lines:])

    def close(self) -> bool:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()
        self._log_file.close()
        if self._proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {self._proc.returncode} encoding "
                f"{self._output_path}: {self._log_tail()}"
            )
        return True


@dataclass(frozen=True)
class RenderStats:
    frame_count: int
    render_seconds_p50: float
    render_seconds_p95: float
    wall_seconds: float
    determinism_ok: bool
    light_summary: dict[str, Any]
    frame_qa: dict[str, Any]


@dataclass(frozen=True)
class _FrameResult:
    """One rendered, labeled frame plus the bookkeeping the render loop needs."""

    frame: int
    array: np.ndarray
    duration: float
    azimuth_deg: float
    elevation_deg: float
    night: bool
    qa_result: Any
    determinism_ok: bool | None  # None unless this frame is one of the checked ones


def _pose_dict(
    plan: ShotPlan, marker_fn: Callable[[int], tuple[float, float]], frame: int
) -> dict[str, Any]:
    pose = plan.pose_at(frame)
    marker_lon, marker_lat = marker_fn(frame)
    return {
        "longitude": pose.longitude,
        "latitude": pose.latitude,
        "zoom": pose.zoom,
        "pitch": pose.pitch,
        "bearing": pose.bearing,
        "marker": [marker_lon, marker_lat],
    }


def _render_one_frame(
    plan: ShotPlan,
    capture: Any,
    marker_fn: Callable[[int], tuple[float, float]],
    light_fn: Callable[[int], SceneLight],
    frame: int,
    determinism_frames: set[int],
) -> _FrameResult:
    """Render, QA-check and label one frame -- the unit both the serial and
    parallel render loops share, so their per-frame behaviour cannot drift.
    """
    pose_dict = _pose_dict(plan, marker_fn, frame)
    frame_light = light_fn(frame)

    t0 = time.perf_counter()
    array, labels = capture.render(pose_dict, light=frame_light)
    duration = time.perf_counter() - t0

    qa_result = check_frame(frame, array, marker_xy=marker_xy_from_labels(labels))
    assert_frame_ok(qa_result, pose_dict)

    determinism_ok: bool | None = None
    if frame in determinism_frames:
        lit_pose = {**pose_dict, "light": frame_light.to_scene(pose_dict["zoom"])}
        determinism_ok = determinism_check(capture, lit_pose)

    return _FrameResult(
        frame=frame,
        array=_draw_labels(array, labels),
        duration=duration,
        azimuth_deg=frame_light.azimuth_deg,
        elevation_deg=frame_light.elevation_deg,
        night=frame_light.night,
        qa_result=qa_result,
        determinism_ok=determinism_ok,
    )


def _stats_from_results(
    frame_count: int,
    durations: list[float],
    azimuths: list[float],
    elevations: list[float],
    night_frames: int,
    qa_results: list[Any],
    determinism_ok: bool,
    wall_seconds: float,
) -> RenderStats:
    return RenderStats(
        frame_count=frame_count,
        render_seconds_p50=float(np.percentile(durations, 50)),
        render_seconds_p95=float(np.percentile(durations, 95)),
        wall_seconds=wall_seconds,
        determinism_ok=determinism_ok,
        light_summary={
            "azimuth_deg_min": min(azimuths),
            "azimuth_deg_max": max(azimuths),
            "elevation_deg_min": min(elevations),
            "elevation_deg_max": max(elevations),
            "night_frames": night_frames,
        },
        frame_qa=summarize(qa_results),
    )


def _render_frames_serial(
    plan: ShotPlan,
    capture: Any,
    sink: Any,
    marker_fn: Callable[[int], tuple[float, float]],
    *,
    light_fn: Callable[[int], SceneLight],
) -> RenderStats:
    """The original single-``capture`` render loop -- unchanged behaviour.

    Every frame is QA-checked (``frame_qa.check_frame``) on its raw array,
    before label drawing; a bad frame aborts the render via
    ``frame_qa.assert_frame_ok``, but the sink is still closed first so a
    failed render never leaves an orphaned ffmpeg process.
    """
    frame_count = plan.frame_count
    determinism_frames = {0, frame_count // 2}
    durations: list[float] = []
    azimuths: list[float] = []
    elevations: list[float] = []
    night_frames = 0
    qa_results: list[Any] = []
    determinism_ok = True
    start_wall = time.perf_counter()
    try:
        for frame in range(frame_count):
            result = _render_one_frame(
                plan, capture, marker_fn, light_fn, frame, determinism_frames
            )
            durations.append(result.duration)
            azimuths.append(result.azimuth_deg)
            elevations.append(result.elevation_deg)
            if result.night:
                night_frames += 1
            qa_results.append(result.qa_result)
            if result.determinism_ok is not None and not result.determinism_ok:
                determinism_ok = False
            sink.write(result.array)
    except BaseException:
        sink.close()
        raise
    ok = sink.close()
    if not ok:
        raise RuntimeError("frame sink failed to close cleanly (ffmpeg encode failed)")
    wall_seconds = time.perf_counter() - start_wall
    return _stats_from_results(
        frame_count,
        durations,
        azimuths,
        elevations,
        night_frames,
        qa_results,
        determinism_ok,
        wall_seconds,
    )


def _strided_frame_ranges(frame_count: int, workers: int) -> list[range]:
    """Round-robin frame ownership: worker ``i`` renders frames ``i, i+N, i+2N, ...``.

    Striped rather than contiguous: the consumer (see ``_render_frames_parallel``)
    drains queues strictly in frame order, so a CONTIGUOUS range would leave
    every worker but whichever one the consumer is currently reading blocked
    on its own bounded queue almost immediately -- the render is then no
    faster than serial. Striding means every worker's next frame is always
    within one queue-depth of being consumed, so all of them stay busy.

    Never returns more ranges than ``frame_count`` -- an idle worker with no
    frames to own would still need its own (pointless) capture.
    """
    if frame_count <= 0:
        return []
    workers = max(1, min(workers, frame_count))
    return [range(i, frame_count, workers) for i in range(workers)]


_QUEUE_POLL_S = 0.05
# The longest a healthy worker can owe the consumer a frame: three fresh
# browsers' readiness waits plus a pose's three doubling settle budgets is
# under six minutes. Past this the worker is stuck, not slow.
WORKER_STALL_TIMEOUT_S = 900.0
# How long cancelled workers get to finish a pose and close their browsers.
WORKER_JOIN_DEADLINE_S = 180.0


def _render_worker(
    capture_factory: Callable[[], Any],
    plan: ShotPlan,
    marker_fn: Callable[[int], tuple[float, float]],
    light_fn: Callable[[int], SceneLight],
    frame_range: range,
    determinism_frames: set[int],
    result_queue: "queue.Queue[Any]",
    cancel_event: threading.Event,
    failure: list[BaseException],
) -> None:
    """Render ``frame_range`` end to end inside this thread, on its own capture.

    Playwright's sync API is not thread-safe across threads sharing one
    browser instance, so this worker's capture is created, driven and closed
    without ever crossing a thread boundary. A render/QA failure records the
    ORIGINAL exception in ``failure`` and sets ``cancel_event`` instead of
    raising across the thread boundary or queuing it; every other worker
    notices the event between frames and stops. The consumer polls with a
    timeout (see ``_get_or_raise``) so it can never block forever on a worker
    that stopped -- cancelled or failed -- without producing the frame it's
    waiting for.
    """
    try:
        with capture_factory() as capture:
            for frame in frame_range:
                if cancel_event.is_set():
                    return
                try:
                    result = _render_one_frame(
                        plan, capture, marker_fn, light_fn, frame, determinism_frames
                    )
                except BaseException as exc:
                    failure.append(exc)
                    cancel_event.set()
                    return
                result_queue.put(result)
    except BaseException as exc:
        failure.append(exc)
        cancel_event.set()


def _get_or_raise(
    result_queue: "queue.Queue[Any]",
    cancel_event: threading.Event,
    failure: list[BaseException],
    stall_timeout_s: float = WORKER_STALL_TIMEOUT_S,
) -> Any:
    """Block for the next result, but never forever: once some worker has
    failed (``cancel_event`` set), a queue that keeps timing out means the
    frame it owed us is never coming, so raise the ORIGINAL failure instead
    of hanging. A worker that neither fails nor delivers -- blocked inside a
    browser call that has no timeout of its own -- is caught by the stall
    budget instead: no frame for ``stall_timeout_s`` aborts the render.
    """
    deadline = time.monotonic() + stall_timeout_s
    while True:
        try:
            return result_queue.get(timeout=_QUEUE_POLL_S)
        except queue.Empty:
            if cancel_event.is_set() and failure:
                raise failure[0] from None
            if time.monotonic() >= deadline:
                raise SceneRenderError(
                    f"no frame from a capture worker in {stall_timeout_s:g} s; "
                    "it is stuck in a browser call, not rendering"
                ) from None


def _drain_and_join(
    result_queues: list["queue.Queue[Any]"],
    threads: list[threading.Thread],
    deadline_s: float = WORKER_JOIN_DEADLINE_S,
) -> None:
    """After cancellation, unstick any worker blocked writing to a queue the
    consumer abandoned, then wait for every worker (and so every capture's
    ``__exit__``) to finish -- up to ``deadline_s``. A worker stuck inside a
    browser call never returns; it is a daemon thread, so it is reported and
    left behind rather than allowed to hang the render it already failed.
    """
    deadline = time.monotonic() + deadline_s
    while any(t.is_alive() for t in threads) and time.monotonic() < deadline:
        for q in result_queues:
            try:
                q.get(timeout=_QUEUE_POLL_S)
            except queue.Empty:
                pass
    stuck = [t.name for t in threads if t.is_alive()]
    if stuck:
        logger.error(
            "%d capture worker(s) still running %g s after cancellation: %s",
            len(stuck),
            deadline_s,
            ", ".join(stuck),
        )


def _render_frames_parallel(
    plan: ShotPlan,
    capture_factory: Callable[[], Any],
    sink: Any,
    marker_fn: Callable[[int], tuple[float, float]],
    *,
    light_fn: Callable[[int], SceneLight],
    workers: int,
) -> RenderStats:
    """One independent capture per worker, frames written to ``sink`` in strict order.

    Frames are assigned round-robin across workers (worker ``i`` owns frames
    ``i, i+N, i+2N, ...`` -- see ``_strided_frame_ranges``), not as
    contiguous blocks: the consumer below always requests frame ``f`` from
    queue ``f % N``, cycling through every worker's queue in turn, so no
    worker is ever left idle behind the others while the consumer works
    through an earlier worker's whole block. A bounded queue per worker
    (backpressure) still means the whole clip is never held in memory at
    once, and ``sink.write`` still always sees frames as
    ``range(frame_count)`` regardless of which worker actually finishes
    first.
    """
    frame_count = plan.frame_count
    determinism_frames = {0, frame_count // 2}
    ranges = _strided_frame_ranges(frame_count, workers)
    num_workers = len(ranges)

    result_queues: list["queue.Queue[Any]"] = [queue.Queue(maxsize=4) for _ in ranges]
    cancel_event = threading.Event()
    failure: list[BaseException] = []
    threads = [
        threading.Thread(
            target=_render_worker,
            args=(
                capture_factory,
                plan,
                marker_fn,
                light_fn,
                frame_range,
                determinism_frames,
                result_queues[i],
                cancel_event,
                failure,
            ),
            daemon=True,
        )
        for i, frame_range in enumerate(ranges)
    ]
    for thread in threads:
        thread.start()

    durations: list[float] = []
    azimuths: list[float] = []
    elevations: list[float] = []
    night_frames = 0
    qa_results: list[Any] = []
    determinism_ok = True
    start_wall = time.perf_counter()

    try:
        for frame in range(frame_count):
            item = _get_or_raise(result_queues[frame % num_workers], cancel_event, failure)
            durations.append(item.duration)
            azimuths.append(item.azimuth_deg)
            elevations.append(item.elevation_deg)
            if item.night:
                night_frames += 1
            qa_results.append(item.qa_result)
            if item.determinism_ok is not None and not item.determinism_ok:
                determinism_ok = False
            sink.write(item.array)
    except BaseException:
        cancel_event.set()
        _drain_and_join(result_queues, threads)
        sink.close()
        raise

    _drain_and_join(result_queues, threads)
    ok = sink.close()
    if not ok:
        raise RuntimeError("frame sink failed to close cleanly (ffmpeg encode failed)")
    wall_seconds = time.perf_counter() - start_wall
    return _stats_from_results(
        frame_count,
        durations,
        azimuths,
        elevations,
        night_frames,
        qa_results,
        determinism_ok,
        wall_seconds,
    )


DEFAULT_MAX_RENDER_WORKERS = 6


def default_render_workers(cpu_count: int | None = None) -> int:
    """Half the box's cores, capped at 6 and floored at 1.

    Deliberately conservative: a Route Story render shares the machine with
    other work, so it should never try to claim every core.
    """
    if cpu_count is None:
        cpu_count = os.cpu_count() or 1
    return min(DEFAULT_MAX_RENDER_WORKERS, max(1, cpu_count // 2))


def _resolve_workers(workers: int | None) -> int:
    """The worker count to actually render with, clamped to what can WORK.

    Every scene render path resolves its worker count here, so this is the one
    place that can stop an operator's ``--workers 6`` from producing a film
    that cannot start: on the export host a second concurrent scene browser
    stops both from ever painting (the measurements are in
    ``scene.capture.max_concurrent_captures``). A render that asks for six
    would otherwise die that way only after a long prepare and preflight.
    """
    from .scene.capture import MAX_CONCURRENT_CAPTURES_ENV, max_concurrent_captures

    requested = _requested_workers(workers)
    ceiling = max_concurrent_captures()
    if requested > ceiling:
        logger.warning(
            "route-story: %d capture workers requested, rendering with %d -- "
            "concurrent scene browsers do not reach ready on this host; "
            "raise %s if this one is different",
            requested,
            ceiling,
            MAX_CONCURRENT_CAPTURES_ENV,
        )
        return ceiling
    return requested


def _requested_workers(workers: int | None) -> int:
    if workers is not None:
        return max(1, int(workers))
    env_value = os.environ.get("GMS_ROUTE_STORY_RENDER_WORKERS")
    if env_value:
        try:
            return max(1, int(env_value))
        except ValueError:
            pass
    return default_render_workers()


def render_frames(
    plan: ShotPlan,
    capture: Any,
    sink: Any,
    marker_fn: Callable[[int], tuple[float, float]],
    *,
    light_fn: Callable[[int], SceneLight],
    workers: int = 1,
    capture_factory: Callable[[], Any] | None = None,
) -> RenderStats:
    """Render every frame of ``plan`` and hand it to ``sink``, in frame order.

    ``workers`` <= 1 (the default) renders serially through the single
    already-entered ``capture``, exactly as before parallel capture existed.
    ``workers`` > 1 ignores ``capture`` and instead builds one independent
    capture per worker from ``capture_factory`` (required in that case, a
    zero-argument callable returning a context-managed capture); every
    behaviour of the serial loop -- per-frame QA, the determinism check on
    frames ``{0, frame_count // 2}``, the light summary, and closing the sink
    on any failure -- is preserved. See ``docs/pipeline-internals.md`` for the
    ordering and cancellation contract.
    """
    if workers <= 1:
        return _render_frames_serial(plan, capture, sink, marker_fn, light_fn=light_fn)
    if capture_factory is None:
        raise ValueError("render_frames needs a capture_factory when workers > 1")
    return _render_frames_parallel(
        plan, capture_factory, sink, marker_fn, light_fn=light_fn, workers=workers
    )


def _git_head() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def render_proof_with_receipt(
    gpx_path: Path,
    bundle_root: Path,
    output_path: Path,
    *,
    recorded_on: date,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    workers: int | None = None,
) -> Path:
    """Render a proof clip and write its receipt beside it; returns the receipt path.

    The one entry the ``route-story-proof`` handler and the in-drive trail film
    share, so both write ``<clip>.receipt.json`` the same way.
    """
    receipt = render_proof_clip(
        gpx_path,
        bundle_root,
        output_path,
        recorded_on=recorded_on,
        fps=fps,
        width=width,
        height=height,
        workers=workers,
    )
    receipt_path = Path(str(output_path) + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    return receipt_path


def render_proof_clip(
    gpx_path: Path,
    bundle_root: Path,
    output_path: Path,
    *,
    recorded_on: date,
    fps: float = 30.0,
    width: int = 1920,
    height: int = 1080,
    capture_factory: Callable[[Path, int, int], Any] | None = None,
    sink_factory: Callable[[Path, int, int, float], Any] | None = None,
    workers: int | None = None,
    readback: str = "jpeg",
) -> dict[str, Any]:
    """Build the shot plan, render it, and return the receipt payload.

    ``recorded_on`` selects the single calendar day the plan is built from
    (``select_gpx_day`` -- the same day-selection ``prepare_bundle`` uses),
    for both the route points and ``trip_epoch``: a GPX carrying other-date
    or duplicated tracks never widens the route bbox or shifts the sun's
    clock away from the day actually being filmed. ``workers`` defaults to
    ``GMS_ROUTE_STORY_RENDER_WORKERS`` if set, else ``default_render_workers()``.
    ``readback`` selects the default capture's screenshot format (``"jpeg"``
    or ``"png"``) and is ignored when the caller supplies its own
    ``capture_factory``.
    """
    resolved_workers = _resolve_workers(workers)
    gpx_path = Path(gpx_path)
    selection = select_gpx_day(gpx_path, recorded_on)
    points = [point for segment in selection.segments for point in segment]
    bundle_root = Path(bundle_root)
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    sample_height = terrain_height_sampler(bundle_root, manifest["terrain"]["maxzoom"])

    imagery_min_zoom = float(manifest["imagery"].get("minzoom", 0.0))
    # Enlargement-cap zoom check: fail before any browser/ffmpeg process starts rather
    # than mid-render or (worse) after an over-enlarged clip is already cut.
    imagery_maxzoom = float(manifest["imagery"]["maxzoom"])
    max_zoom = max_shot_zoom(imagery_maxzoom)
    proof_plan = build_shot_plan(
        points,
        sample_height,
        fps,
        width,
        height,
        min_zoom=imagery_min_zoom,
        max_zoom=max_zoom,
    )
    plan = proof_plan.plan
    plan.assert_within_zoom(max_zoom)

    relief_point = proof_plan.chase_points[0]
    chase_end_point = proof_plan.chase_points[-1]

    def marker_fn(frame: int) -> tuple[float, float]:
        return marker_position(plan, frame, relief_point, chase_end_point)

    # Per-frame light: the real sun at the chase's own route position and
    # clock time, swept across the closing pullback, and a fixed trip-start
    # light for the opening shots that have no route time of their own.
    # Replaces a fixed LIGHT_AZIMUTH constant, which put the highlight on the
    # wrong side of the terrain for any route not shot facing northwest at
    # dusk, and then a single whole-clip sun position (route-centre, mid-trip
    # time) -- honest but static as the camera actually moves.
    center_lon = (proof_plan.route_bbox[0] + proof_plan.route_bbox[2]) / 2.0
    center_lat = (proof_plan.route_bbox[1] + proof_plan.route_bbox[3]) / 2.0
    trip_epoch = selection.source_started_at
    light_fn = route_light_fn(
        plan, points, proof_plan.chase_points, trip_epoch, center_lat, center_lon
    )

    capture_factory = capture_factory or (
        lambda root, w, h: SceneCapture(root, w, h, readback=readback)
    )
    sink_factory = sink_factory or (lambda path, w, h, f: FfmpegFrameSink(path, w, h, f))

    sink = sink_factory(Path(output_path), width, height, fps)
    if resolved_workers <= 1:
        with capture_factory(bundle_root, width, height) as capture:
            stats = render_frames(plan, capture, sink, marker_fn, light_fn=light_fn)
    else:
        stats = render_frames(
            plan,
            None,
            sink,
            marker_fn,
            light_fn=light_fn,
            workers=resolved_workers,
            capture_factory=lambda: capture_factory(bundle_root, width, height),
        )

    duration_seconds = stats.frame_count / fps
    render_fps = stats.frame_count / stats.wall_seconds if stats.wall_seconds > 0 else 0.0
    x_realtime = stats.wall_seconds / duration_seconds if duration_seconds > 0 else 0.0

    return {
        "schema": "route-story-proof-receipt/2",
        "output": str(output_path),
        "gpx": str(gpx_path),
        "bundle": str(bundle_root),
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": stats.frame_count,
        "duration_seconds": duration_seconds,
        "render_seconds_p50": stats.render_seconds_p50,
        "render_seconds_p95": stats.render_seconds_p95,
        "wall_seconds": stats.wall_seconds,
        "render_fps": render_fps,
        "x_realtime": x_realtime,
        "workers": resolved_workers,
        "readback": readback,
        "determinism_ok": stats.determinism_ok,
        "relief_section": {
            "index": proof_plan.relief.index,
            "relief_range_m": proof_plan.relief.relief_range_m,
            "window_points": proof_plan.relief.window_points,
        },
        "chase_distance_m": proof_plan.chase_distance_m,
        "chase_point_count": len(proof_plan.chase_points),
        "shots": [
            {"kind": shot.kind, "start_frame": shot.start_frame, "end_frame": shot.end_frame}
            for shot in plan.shots
        ],
        "sources": {
            "imagery": manifest["imagery"].get("sources", []),
            "terrain": manifest["terrain"].get("sources", []),
            "labels": manifest.get("labels", {}).get("sources", []),
        },
        "terrain_zoom_used": manifest["terrain"]["maxzoom"],
        "light": {
            **stats.light_summary,
            "source": "sun_track",
        },
        "zoom_cap": {
            "imagery_maxzoom": imagery_maxzoom,
            "enlargement": ENLARGEMENT_MAX,
            "max_zoom": max_zoom,
        },
        "frame_qa": stats.frame_qa,
        "git_head": _git_head(),
    }
