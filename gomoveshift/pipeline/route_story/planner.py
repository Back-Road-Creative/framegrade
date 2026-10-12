"""Renderer-independent camera shot planning for the Route Story drone move.

Produces one :class:`~gomoveshift.pipeline.route_story.opening.CameraPose` per frame from a
sequence of named :class:`Shot` objects. Nothing here touches MapLibre or any renderer; a
:class:`ShotPlan` is pure math that a preview or final export consumes identically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Literal, Sequence

from gomoveshift.pipeline.route_story.metrics import EARTH_RADIUS_M, RoutePoint
from gomoveshift.pipeline.route_story.opening import CameraPose, _smootherstep

ShotKind = Literal["descent", "reveal", "follow", "arc", "arrival", "return", "pullback"]

PoseFn = Callable[[int], CameraPose]

WORLD_TILE_PX = 512
EARTH_CIRCUMFERENCE_M = 40_075_016.686
MAX_FIT_ZOOM = 22.0

# Enlargement cap: no shot may exceed the imagery maxzoom by more than +0.32
# (1.25x); arriving at z15.6 on z15 imagery would be 1.52x.
ENLARGEMENT_MAX = 1.25

DEFAULT_MOTION_LIMITS: dict[str, float] = {
    "bearing_speed_deg_s": 25.0,
    "bearing_accel_deg_s2": 40.0,
    "pitch_speed_deg_s": 15.0,
    "zoom_velocity_per_s": 1.2,
}


# --- Web Mercator (MapLibre 512px-tile convention) ---------------------------------------


def _mercator_xy(longitude: float, latitude: float) -> tuple[float, float]:
    """Project lon/lat to normalized (0..1) Web Mercator world coordinates."""
    x = (longitude + 180.0) / 360.0
    lat_rad = math.radians(latitude)
    y = 0.5 - math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0)) / (2.0 * math.pi)
    return x, y


def _mercator_lonlat(x: float, y: float) -> tuple[float, float]:
    """Invert :func:`_mercator_xy`."""
    longitude = x * 360.0 - 180.0
    lat_rad = 2.0 * math.atan(math.exp((0.5 - y) * 2.0 * math.pi)) - math.pi / 2.0
    return longitude, math.degrees(lat_rad)


def fit_zoom(
    bbox: tuple[float, float, float, float],
    width_px: float,
    height_px: float,
    padding_px: float = 0.0,
) -> float:
    """MapLibre-convention zoom that fits ``bbox`` inside a ``width_px``x``height_px`` viewport."""
    min_lon, min_lat, max_lon, max_lat = bbox
    x0, y0 = _mercator_xy(min_lon, max_lat)
    x1, y1 = _mercator_xy(max_lon, min_lat)
    span_x = abs(x1 - x0)
    span_y = abs(y1 - y0)
    avail_w = max(width_px - 2.0 * padding_px, 1.0)
    avail_h = max(height_px - 2.0 * padding_px, 1.0)
    zoom_x = math.log2(avail_w / (span_x * WORLD_TILE_PX)) if span_x > 1e-12 else MAX_FIT_ZOOM
    zoom_y = math.log2(avail_h / (span_y * WORLD_TILE_PX)) if span_y > 1e-12 else MAX_FIT_ZOOM
    return max(0.0, min(zoom_x, zoom_y, MAX_FIT_ZOOM))


def max_shot_zoom(imagery_maxzoom: float, enlargement: float = ENLARGEMENT_MAX) -> float:
    """Return the highest zoom a shot may reach on imagery whose maxzoom is ``imagery_maxzoom``.

    ``enlargement`` is a linear scale factor (1.25 == 1.25x, the enlargement cap);
    each doubling of zoom halves the ground sample distance, so the zoom
    offset for a given enlargement is ``log2(enlargement)``.
    """
    if enlargement < 1:
        raise ValueError("enlargement must be >= 1")
    return imagery_maxzoom + math.log2(enlargement)


# --- Spherical geometry, for the orbiting arc shot ---------------------------------------


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def _initial_bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return math.degrees(math.atan2(x, y)) % 360.0


def _destination_point(
    lon: float, lat: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    delta = distance_m / EARTH_RADIUS_M
    theta = math.radians(bearing_deg)
    phi1 = math.radians(lat)
    lambda1 = math.radians(lon)
    phi2 = math.asin(
        math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    longitude = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return longitude, math.degrees(phi2)


def _shortest_angle_delta(start: float, end: float) -> float:
    delta = (end - start + 180.0) % 360.0 - 180.0
    return 180.0 if delta == -180.0 else delta


# --- Shot / ShotPlan ----------------------------------------------------------------------


@dataclass(frozen=True)
class Shot:
    """One named camera move, addressable by absolute frame number."""

    kind: ShotKind
    start_frame: int
    end_frame: int
    _pose_fn: PoseFn

    def __post_init__(self) -> None:
        if self.end_frame <= self.start_frame:
            raise ValueError("shot end_frame must be greater than start_frame")

    def pose_at(self, frame: int) -> CameraPose:
        """Resolve a pose for a frame inside this shot's ``[start_frame, end_frame)`` range."""
        if not self.start_frame <= frame < self.end_frame:
            raise ValueError(f"frame {frame} is outside shot {self.kind!r}")
        return self._pose_fn(frame)

    def _boundary_pose(self, frame: int) -> CameraPose:
        """Evaluate the underlying pose function, including the exclusive end frame.

        Used only to check continuity against the next shot's start pose.
        """
        if not self.start_frame <= frame <= self.end_frame:
            raise ValueError(f"frame {frame} is outside shot {self.kind!r}")
        return self._pose_fn(frame)


def _assert_pose_close(
    a: CameraPose, b: CameraPose, tolerance: float, from_kind: str, to_kind: str
) -> None:
    for field in ("longitude", "latitude", "zoom", "pitch", "bearing"):
        left, right = getattr(a, field), getattr(b, field)
        if abs(left - right) > tolerance:
            raise ValueError(
                f"pose discontinuity between shot {from_kind!r} and {to_kind!r} "
                f"on {field}: {left!r} != {right!r}"
            )


@dataclass(frozen=True)
class ShotPlan:
    """An ordered, frame-addressable sequence of shots covering the whole film."""

    shots: tuple[Shot, ...]

    def __post_init__(self) -> None:
        if not self.shots:
            raise ValueError("a shot plan needs at least one shot")

    @property
    def frame_count(self) -> int:
        return self.shots[-1].end_frame

    def pose_at(self, frame: int) -> CameraPose:
        for shot in self.shots:
            if shot.start_frame <= frame < shot.end_frame:
                return shot.pose_at(frame)
        raise ValueError(f"frame {frame} is outside the shot plan")

    def validate(self, *, tolerance: float = 1e-9) -> None:
        """Raise on gaps, overlaps, or a discontinuous pose at any shot join."""
        previous = self.shots[0]
        if previous.start_frame != 0:
            raise ValueError("a shot plan must start at frame 0")
        for shot in self.shots[1:]:
            if shot.start_frame != previous.end_frame:
                relation = "gap" if shot.start_frame > previous.end_frame else "overlap"
                raise ValueError(f"{relation} between shot {previous.kind!r} and {shot.kind!r}")
            join_a = previous._boundary_pose(previous.end_frame)
            join_b = shot.pose_at(shot.start_frame)
            _assert_pose_close(join_a, join_b, tolerance, previous.kind, shot.kind)
            previous = shot

    def assert_within_zoom(self, max_zoom: float, *, samples_per_shot: int = 16) -> None:
        """Raise ``ValueError`` naming the worst pose exceeding the enlargement cap.

        Samples every shot's boundary frames (first and last valid frame)
        plus evenly spaced frames in between, rather than every frame —
        cheap enough to run per plan while still catching an arrival or
        descent that overshoots (the proof's z15.6 on z15 imagery). Within a
        violating shot, the sample with the highest zoom is reported, since
        that is the one that actually names the overshoot operators need to
        see rather than an arbitrary earlier threshold-crossing sample.
        """
        for shot_index, shot in enumerate(self.shots):
            frame_count = shot.end_frame - shot.start_frame
            n_samples = min(samples_per_shot, frame_count)
            if n_samples <= 1:
                frames: Iterable[int] = (shot.start_frame,)
            else:
                frames = sorted(
                    {
                        shot.start_frame + round(i * (frame_count - 1) / (n_samples - 1))
                        for i in range(n_samples)
                    }
                )
            worst_frame: int | None = None
            worst_zoom = max_zoom
            for frame in frames:
                zoom = shot.pose_at(frame).zoom
                if zoom > worst_zoom:
                    worst_frame, worst_zoom = frame, zoom
            if worst_frame is not None:
                raise ValueError(
                    f"shot {shot_index} ({shot.kind!r}) frame {worst_frame} "
                    f"zoom {worst_zoom:.4f} exceeds max_shot_zoom {max_zoom:.4f}"
                )


# --- Builders -------------------------------------------------------------------------------


def descent(
    start: CameraPose,
    target: RoutePoint,
    fps: float,
    seconds: float = 16.0,
    *,
    final_zoom: float = 16.0,
    start_frame: int = 0,
) -> Shot:
    """Zoom in on ``target``, target-locked in Web Mercator so it stays near frame centre."""
    if not 14.0 <= seconds <= 18.0:
        raise ValueError("descent must be 14-18 seconds")
    frame_count = round(seconds * fps)
    end_frame = start_frame + frame_count
    sx, sy = _mercator_xy(start.longitude, start.latitude)
    tx, ty = _mercator_xy(target.longitude, target.latitude)
    bearing = start.bearing

    def pose_fn(frame: int) -> CameraPose:
        eased = _smootherstep((frame - start_frame) / frame_count)
        longitude, latitude = _mercator_lonlat(sx + (tx - sx) * eased, sy + (ty - sy) * eased)
        zoom = start.zoom + (final_zoom - start.zoom) * eased
        return CameraPose(longitude, latitude, zoom, 0.0, bearing)

    return Shot("descent", start_frame, end_frame, pose_fn)


def reveal(
    at: CameraPose,
    fps: float,
    seconds: float = 7.0,
    *,
    final_pitch: float = 45.0,
    start_frame: int = 0,
) -> Shot:
    """Tilt the camera down from ``at.pitch`` to ``final_pitch``, position held fixed."""
    if not 6.0 <= seconds <= 8.0:
        raise ValueError("reveal must be 6-8 seconds")
    if not 35.0 <= final_pitch <= 50.0:
        raise ValueError("reveal final_pitch must be 35-50 degrees")
    frame_count = round(seconds * fps)
    end_frame = start_frame + frame_count

    def pose_fn(frame: int) -> CameraPose:
        eased = _smootherstep((frame - start_frame) / frame_count)
        pitch = at.pitch + (final_pitch - at.pitch) * eased
        return CameraPose(at.longitude, at.latitude, at.zoom, pitch, at.bearing)

    return Shot("reveal", start_frame, end_frame, pose_fn)


def arc(
    center: RoutePoint,
    base: CameraPose,
    fps: float,
    seconds: float = 7.5,
    *,
    degrees: float = 30.0,
    start_frame: int = 0,
) -> Shot:
    """Orbit ``base`` around ``center`` by ``degrees``, radius held constant, looking inward."""
    if not 6.0 <= seconds <= 9.0:
        raise ValueError("arc must be 6-9 seconds")
    if not 20.0 <= abs(degrees) <= 40.0:
        raise ValueError("arc sweep must be 20-40 degrees")
    frame_count = round(seconds * fps)
    end_frame = start_frame + frame_count
    radius_m = _haversine_m(base.longitude, base.latitude, center.longitude, center.latitude)
    azimuth0 = _initial_bearing_deg(
        center.longitude, center.latitude, base.longitude, base.latitude
    )

    def pose_fn(frame: int) -> CameraPose:
        eased = _smootherstep((frame - start_frame) / frame_count)
        azimuth = azimuth0 + degrees * eased
        longitude, latitude = _destination_point(
            center.longitude, center.latitude, azimuth, radius_m
        )
        bearing = (azimuth + 180.0) % 360.0
        return CameraPose(longitude, latitude, base.zoom, base.pitch, bearing)

    return Shot("arc", start_frame, end_frame, pose_fn)


def arrival(
    base: CameraPose,
    target: CameraPose,
    fps: float,
    seconds: float = 2.0,
    *,
    start_frame: int = 0,
) -> Shot:
    """A short settle/push-in transition from ``base`` to ``target``."""
    if seconds <= 0:
        raise ValueError("arrival duration must be positive")
    frame_count = round(seconds * fps)
    end_frame = start_frame + frame_count
    sx, sy = _mercator_xy(base.longitude, base.latitude)
    tx, ty = _mercator_xy(target.longitude, target.latitude)
    bearing_delta = _shortest_angle_delta(base.bearing, target.bearing)

    def pose_fn(frame: int) -> CameraPose:
        eased = _smootherstep((frame - start_frame) / frame_count)
        longitude, latitude = _mercator_lonlat(sx + (tx - sx) * eased, sy + (ty - sy) * eased)
        zoom = base.zoom + (target.zoom - base.zoom) * eased
        pitch = base.pitch + (target.pitch - base.pitch) * eased
        bearing = (base.bearing + bearing_delta * eased) % 360.0
        return CameraPose(longitude, latitude, zoom, pitch, bearing)

    return Shot("arrival", start_frame, end_frame, pose_fn)


def return_to(
    base_after: CameraPose,
    current: CameraPose,
    fps: float,
    seconds: float = 0.7,
    *,
    start_frame: int = 0,
) -> Shot:
    """Return from ``current`` to exactly the pose that was active before a deviation."""
    if seconds <= 0:
        raise ValueError("return duration must be positive")
    frame_count = round(seconds * fps)
    end_frame = start_frame + frame_count
    sx, sy = _mercator_xy(current.longitude, current.latitude)
    tx, ty = _mercator_xy(base_after.longitude, base_after.latitude)
    bearing_delta = _shortest_angle_delta(current.bearing, base_after.bearing)

    def pose_fn(frame: int) -> CameraPose:
        if frame >= end_frame:
            return base_after
        eased = _smootherstep((frame - start_frame) / frame_count)
        longitude, latitude = _mercator_lonlat(sx + (tx - sx) * eased, sy + (ty - sy) * eased)
        zoom = current.zoom + (base_after.zoom - current.zoom) * eased
        pitch = current.pitch + (base_after.pitch - current.pitch) * eased
        bearing = (current.bearing + bearing_delta * eased) % 360.0
        return CameraPose(longitude, latitude, zoom, pitch, bearing)

    return Shot("return", start_frame, end_frame, pose_fn)


def pullback(
    end: CameraPose,
    route_bbox: tuple[float, float, float, float],
    fps: float,
    seconds: float = 6.0,
    *,
    width_px: float = 1920.0,
    height_px: float = 1080.0,
    padding_px: float = 80.0,
    start_frame: int = 0,
    min_zoom: float = 0.0,
) -> Shot:
    """Rise from ``end`` to a zoom that fits the whole route bbox for the output aspect.

    ``min_zoom`` floors the target: a bundle only holds tiles for the zooms it
    actually wrote, and a wider shot than that would render nothing.
    """
    if seconds <= 0:
        raise ValueError("pullback duration must be positive")
    frame_count = round(seconds * fps)
    end_frame = start_frame + frame_count
    min_lon, min_lat, max_lon, max_lat = route_bbox
    center_lon = (min_lon + max_lon) / 2.0
    center_lat = (min_lat + max_lat) / 2.0
    target_zoom = max(fit_zoom(route_bbox, width_px, height_px, padding_px), min_zoom)
    sx, sy = _mercator_xy(end.longitude, end.latitude)
    tx, ty = _mercator_xy(center_lon, center_lat)

    def pose_fn(frame: int) -> CameraPose:
        if frame >= end_frame:
            return CameraPose(center_lon, center_lat, target_zoom, 0.0, end.bearing)
        eased = _smootherstep((frame - start_frame) / frame_count)
        longitude, latitude = _mercator_lonlat(sx + (tx - sx) * eased, sy + (ty - sy) * eased)
        zoom = end.zoom + (target_zoom - end.zoom) * eased
        pitch = end.pitch * (1.0 - eased)
        return CameraPose(longitude, latitude, zoom, pitch, end.bearing)

    return Shot("pullback", start_frame, end_frame, pose_fn)


# --- Motion bounds ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MotionReport:
    """Per-frame rates of change of bearing, pitch and zoom, plus their derivatives."""

    bearing_speed_deg_s: tuple[float, ...]
    bearing_accel_deg_s2: tuple[float, ...]
    bearing_jerk_deg_s3: tuple[float, ...]
    pitch_speed_deg_s: tuple[float, ...]
    zoom_velocity_per_s: tuple[float, ...]


def _pose_bearing(pose: object) -> float:
    return pose.bearing if hasattr(pose, "bearing") else pose.bearing_deg


def _pose_pitch(pose: object) -> float:
    return pose.pitch if hasattr(pose, "pitch") else pose.pitch_deg


def motion_report_from_poses(poses: Sequence[object], fps: float) -> MotionReport:
    """Build a :class:`MotionReport` from any per-frame pose sequence at ``fps``.

    Each pose needs a ``bearing``/``pitch`` pair OR a ``bearing_deg``/``pitch_deg``
    pair (:class:`~gomoveshift.pipeline.route_story.continuity.CameraState`'s
    names), plus a ``zoom`` attribute either way -- so this accepts both
    :class:`~gomoveshift.pipeline.route_story.opening.CameraPose` (a
    :class:`ShotPlan`'s planned poses) and ``CameraState`` (a scene renderer's
    actually-rendered ones) without either module importing the other.
    """
    bearing_speed = [
        _shortest_angle_delta(_pose_bearing(a), _pose_bearing(b)) * fps
        for a, b in zip(poses, poses[1:])
    ]
    pitch_speed = [(_pose_pitch(b) - _pose_pitch(a)) * fps for a, b in zip(poses, poses[1:])]
    zoom_velocity = [(b.zoom - a.zoom) * fps for a, b in zip(poses, poses[1:])]
    bearing_accel = [(b - a) * fps for a, b in zip(bearing_speed, bearing_speed[1:])]
    bearing_jerk = [(b - a) * fps for a, b in zip(bearing_accel, bearing_accel[1:])]
    return MotionReport(
        tuple(bearing_speed),
        tuple(bearing_accel),
        tuple(bearing_jerk),
        tuple(pitch_speed),
        tuple(zoom_velocity),
    )


def motion_report(plan: ShotPlan, fps: float) -> MotionReport:
    poses = [plan.pose_at(frame) for frame in range(plan.frame_count)]
    return motion_report_from_poses(poses, fps)


def hectic_windows(
    report: MotionReport,
    fps: float,
    *,
    count: int = 3,
    window_s: float = 4.0,
    min_gap_s: float = 6.0,
) -> list[tuple[float, float]]:
    """The ``count`` most hectic camera moments as ``[start_s, end_s]`` windows.

    Ranks frames by ``|bearing_speed_deg_s|`` and greedily takes the
    highest-magnitude peaks at least ``min_gap_s`` apart -- so one violent
    turn doesn't claim every slot -- then expands each survivor to a
    ``window_s``-wide window centred on it, clipped to the film's own
    duration, and returns them in playback order. Empty when
    ``report`` has fewer than two frames of bearing speed: a film that short
    has nothing to build a window around.
    """
    series = report.bearing_speed_deg_s
    if len(series) < 2:
        return []
    duration_s = (len(series) + 1) / fps
    ranked = sorted(range(len(series)), key=lambda index: abs(series[index]), reverse=True)
    chosen_times: list[float] = []
    for index in ranked:
        if len(chosen_times) >= count:
            break
        peak_s = (index + 0.5) / fps
        if all(abs(peak_s - other) >= min_gap_s for other in chosen_times):
            chosen_times.append(peak_s)
    half = window_s / 2.0
    return [
        (max(0.0, peak_s - half), min(duration_s, peak_s + half)) for peak_s in sorted(chosen_times)
    ]


def _nearest_rank_percentile(sorted_abs_values: list[float], percentile: float) -> float:
    """Nearest-rank percentile of an already-sorted ascending list of magnitudes."""
    count = len(sorted_abs_values)
    if count == 0:
        return 0.0
    rank = max(1, math.ceil(percentile / 100.0 * count))
    return sorted_abs_values[min(rank, count) - 1]


@dataclass(frozen=True)
class MotionStat:
    """Summary statistics for one motion series' absolute values."""

    mean_abs: float
    p95_abs: float
    max_abs: float
    frames_over_limit: int


def _motion_stat(series: tuple[float, ...], limit: float | None) -> MotionStat:
    if not series:
        return MotionStat(0.0, 0.0, 0.0, 0)
    abs_values = sorted(abs(value) for value in series)
    mean_abs = sum(abs_values) / len(abs_values)
    frames_over_limit = 0 if limit is None else sum(1 for value in abs_values if value > limit)
    return MotionStat(
        mean_abs=mean_abs,
        p95_abs=_nearest_rank_percentile(abs_values, 95.0),
        max_abs=abs_values[-1],
        frames_over_limit=frames_over_limit,
    )


@dataclass(frozen=True)
class MotionSummary:
    """A perceptual-motion "before number" for a :class:`ShotPlan`.

    Measures the camera path only, never the rendered pixels — see
    ``docs/route-story-planner.md`` for what to compare between two plans.
    """

    bearing_speed: MotionStat
    bearing_accel: MotionStat
    pitch_speed_max_abs: float
    zoom_velocity_max_abs: float
    frame_count: int
    fps: float
    limits: dict[str, float]

    def as_receipt(self) -> dict:
        """Plain, JSON-serialisable representation under the ``"motion"`` key."""

        def stat_dict(stat: MotionStat) -> dict:
            return {
                "mean_abs": round(stat.mean_abs, 3),
                "p95_abs": round(stat.p95_abs, 3),
                "max_abs": round(stat.max_abs, 3),
                "frames_over_limit": stat.frames_over_limit,
            }

        return {
            "motion": {
                "bearing_speed": stat_dict(self.bearing_speed),
                "bearing_accel": stat_dict(self.bearing_accel),
                "pitch_speed_max_abs": round(self.pitch_speed_max_abs, 3),
                "zoom_velocity_max_abs": round(self.zoom_velocity_max_abs, 3),
                "frame_count": self.frame_count,
                "fps": round(self.fps, 3),
                "limits": {key: round(value, 3) for key, value in self.limits.items()},
            }
        }


def motion_summary(
    report: MotionReport,
    fps: float,
    limits: dict[str, float] | None = None,
) -> MotionSummary:
    """Reduce a :class:`MotionReport` to the numbers that answer "is this calmer?"."""
    limits = DEFAULT_MOTION_LIMITS if limits is None else limits
    pitch_abs = [abs(value) for value in report.pitch_speed_deg_s]
    zoom_abs = [abs(value) for value in report.zoom_velocity_per_s]
    return MotionSummary(
        bearing_speed=_motion_stat(report.bearing_speed_deg_s, limits.get("bearing_speed_deg_s")),
        bearing_accel=_motion_stat(report.bearing_accel_deg_s2, limits.get("bearing_accel_deg_s2")),
        pitch_speed_max_abs=max(pitch_abs, default=0.0),
        zoom_velocity_max_abs=max(zoom_abs, default=0.0),
        frame_count=len(report.bearing_speed_deg_s) + 1,
        fps=fps,
        limits=dict(limits),
    )


def summarize_plan_motion(
    plan: ShotPlan,
    fps: float,
    limits: dict[str, float] | None = None,
) -> MotionSummary:
    """Compose :func:`motion_report` and :func:`motion_summary` for a whole plan."""
    return motion_summary(motion_report(plan, fps), fps, limits)


def assert_smooth(
    plan: ShotPlan,
    fps: float,
    limits: dict[str, float] | None = None,
) -> None:
    """Raise ``ValueError`` naming the worst frame if any motion bound is exceeded."""
    limits = DEFAULT_MOTION_LIMITS if limits is None else limits
    report = motion_report(plan, fps)
    checks = (
        ("bearing_speed_deg_s", report.bearing_speed_deg_s),
        ("bearing_accel_deg_s2", report.bearing_accel_deg_s2),
        ("pitch_speed_deg_s", report.pitch_speed_deg_s),
        ("zoom_velocity_per_s", report.zoom_velocity_per_s),
    )
    for name, series in checks:
        limit = limits.get(name)
        if limit is None:
            continue
        for index, value in enumerate(series):
            if abs(value) > limit:
                raise ValueError(f"{name} exceeds {limit} at frame {index}: {value:.3f}")
