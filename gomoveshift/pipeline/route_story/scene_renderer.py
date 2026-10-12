"""A ``MapRenderer`` that draws film map-leg frames through the offline
MapLibre terrain scene instead of the flat Pillow satellite composite.

This module owns nothing about the scene itself (``scene/capture.py``), the
sun (``sun.py``), or the dashboard/labels (``hud.py``) -- it composes them
into one ``MapRenderer.render``/``render_leg`` call so ``film.py`` can swap
renderers without knowing which one produced a frame.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any, Callable, ClassVar, Iterator, Protocol, Sequence

from PIL import Image

from .continuity import CameraContinuityError, CameraState, check_camera_continuity
from .dashboard import dashboard_state_at
from .frame_qa import (
    FrameQaError,
    FrameQaLedger,
    FrameQaThresholds,
    check_frame,
    marker_xy_from_labels,
)
from .hud import _draw_dashboard, dashboard_fields_for, draw_profile, set_active_dashboard_fields
from .layers import (
    LegLabelPlanner,
    PhotoPinBeat,
    _draw_gap_badge,
    _point_at_time,
    route_endpoints,
    visible_pins,
)
from .layout import hud_layout
from .metrics import RoutePoint, distance_m
from .planner import DEFAULT_MOTION_LIMITS, _initial_bearing_deg, _shortest_angle_delta
from .profile_geometry import ProfileGeometry, build_profile
from .proof import (
    _QUEUE_POLL_S,
    WORKER_JOIN_DEADLINE_S,
    WORKER_STALL_TIMEOUT_S,
    _draw_labels,
    _get_or_raise,
    _smoothed_bearings,
    _strided_frame_ranges,
)
from .sun import light_for_pose
from .timeline import Scene, StoryTimeline

# Minimum route-time width of the bearing-smoothing window used by a single,
# leg-unaware ``render()`` call. GPS jitter on a real track needs at least
# this much averaging to cancel.
_MIN_BEARING_WINDOW_S = 5.0

# A map leg's own chord-bearing window is a fraction of the leg's route
# duration (see ``_chord_bearing``), floored at this many route seconds so a
# very short leg still averages a real neighbourhood of GPS points.
#
# This fraction is deliberately close to the WHOLE leg, not a small slice of
# it. A real map leg can compress hundreds to thousands of route-seconds into
# one ~14 s displayed leg; if that route doubles back on itself (a real
# out-and-back, a driveway turnaround) the net rotation the camera would have
# to sweep to keep pointing exactly along every local switchback is many
# times what `bearing_speed_deg_s`/`bearing_accel_deg_s2` (planner.py) can
# physically cover in the frame budget -- no window size fixes that, because
# a bang-bang move at those caps has a fixed minimum time for a given total
# turn. Widening the chord toward the leg's own endpoints instead cancels the
# switchback in the window itself: the target becomes the leg's NET travel
# direction, which is what a follow camera can actually keep up with, and
# happens to be within it well before the window reaches a full 1.0 (fraction
# 0.15 pinned the camera at the speed cap for ~47% of a real, heavily
# switchback-laden trip's frames; 0.998 brings that under 5%, on both the
# real trip and a synthetic worst-case out-and-back).
_MIN_CHORD_WINDOW_S = 60.0
_CHORD_WINDOW_FRACTION = 0.998

# Frames one capture worker may run ahead of the in-order consumer (the same
# depth ``proof._render_frames_parallel`` uses): backpressure, not a tuning knob.
_WORKER_QUEUE_DEPTH = 4

# A map leg can compress hundreds of real route-seconds into a handful of
# displayed seconds, so every raw sampled instant changes on every output
# frame -- the dashboard digits flicker even though nothing on the ground
# actually moved that fast. Holding the sampled instant steady across this
# many seconds of OUTPUT time (via ``dashboard_state_at``'s own
# ``cadence_frames``) is what keeps "the dash stats
# go way too fast" from being true.
_DASHBOARD_CADENCE_S = 0.5

logger = logging.getLogger(__name__)

# Below this chord length, two points this close together carry no reliable
# direction of travel (an out-and-back turnaround, a stop) -- chasing the
# noise there is exactly what produced a camera that mostly looks sideways.
_MIN_CHORD_DISTANCE_M = 15.0

_MAP_HEIGHT_FRACTION = 0.855


@dataclass
class _CaptureJob:
    """One leg's slice of work for one pooled worker."""

    frame_range: range
    segments: Sequence[Sequence[RoutePoint]]
    times: Sequence[float]
    bearings: Sequence[float]
    width: int
    height: int
    results: "queue.Queue[Any]"
    cancel_event: threading.Event
    failure: list[BaseException]
    done: threading.Event


@dataclass
class _CapturedFrame:
    """One frame's CAPTURE-step output: the raw (label-undrawn) pixels plus
    everything ``_finish_frame`` needs to draw labels/HUD and compose the
    final image.

    This is what a parallel worker now hands back on its result queue,
    instead of a finished ``Image`` -- so the never-evict label rule can be
    applied SEQUENTIALLY, in real frame order, by the single-threaded
    consumer (see ``_capture_leg_parallel``) after every worker's capture,
    the genuinely expensive part, has already run in parallel. No frame is
    ever captured twice: only the cheap pixel-drawing step moves.
    """

    array: Any
    labels: list[dict[str, Any]]
    layout: Any
    dashboard_box: tuple[float, float, float, float]
    profile_box: tuple[float, float, float, float]
    map_height: int
    in_gap: bool
    at_time: float
    segments: Sequence[Sequence[RoutePoint]]
    width: int
    height: int
    frame_index: int | None


@dataclass
class _CaptureWorker:
    """A thread that owns ONE browser for the life of the film.

    Playwright's sync API is not thread-safe across threads, so a capture must
    be opened and driven from the same thread -- which is why the browser
    cannot simply be a shared object handed to a new thread each leg. The
    thread itself is what persists; legs arrive on ``jobs``.
    """

    jobs: "queue.Queue[_CaptureJob | None]"
    thread: threading.Thread | None = None


class _SceneRendererCapture(Protocol):
    def render(
        self, pose: dict[str, Any], light: Any = None
    ) -> tuple[Any, list[dict[str, Any]]]: ...


def scene_map_height(height: int) -> int:
    """The map field height for a full film frame of ``height`` -- shared
    with ``layers._render_map`` so a scene capture is sized identically."""
    return int(height * _MAP_HEIGHT_FRACTION)


def _target_bearing(
    segments: Sequence[Sequence[RoutePoint]], at_time: float, window_s: float
) -> float:
    """The route's own smoothed heading around ``at_time`` -- NOT rate-limited.

    Used only by the single-frame, leg-unaware ``render()`` convenience --
    a map leg's own frames go through ``render_leg``'s ``_chord_bearing`` /
    ``_plan_bearings`` instead, which is what keeps what actually gets
    displayed within the planner's own bearing motion caps.
    """
    points = sorted(
        (
            point
            for segment in segments
            for point in segment
            if abs(point.time_s - at_time) <= window_s
        ),
        key=lambda point: point.time_s,
    )
    if len(points) < 2:
        return 0.0
    bearings = _smoothed_bearings(points)
    closest = min(range(len(bearings)), key=lambda i: abs(points[i].time_s - at_time))
    return bearings[closest]


def _leg_dashboard_timeline(times: Sequence[float]) -> StoryTimeline:
    """A genuine per-leg ``StoryTimeline`` mapping each OUTPUT frame index to
    this leg's own (non-uniformly spaced) recording time, so
    ``dashboard_state_at``'s ``cadence_frames`` has real output-frame
    structure to hold steady across -- the single-instant ``Scene`` built
    per frame previously made every frame its own frame 0, which made
    cadence quantization a no-op.

    One Scene per consecutive frame pair, using each ``times[i]``/
    ``times[i + 1]`` verbatim (never resampled): frame ``i`` always resolves
    to exactly ``times[i]``, matching the render's own pose.
    """
    if len(times) < 2:
        at_time = times[0] if times else 0.0
        return StoryTimeline(fps=1, scenes=[Scene("overview", 0, 1, at_time, at_time)])
    scenes = [
        Scene("route_follow", index, index + 1, times[index], times[index + 1])
        for index in range(len(times) - 1)
    ]
    return StoryTimeline(fps=1, scenes=scenes)


def _travelled_coordinates(
    segments: Sequence[Sequence[RoutePoint]],
    at_time: float,
    marker: RoutePoint,
    in_gap: bool,
) -> list[list[list[float]]]:
    """The travelled portion of ``segments`` at ``at_time``, as a
    MultiLineString-shaped list of coordinate lists -- ONE per display
    segment that has already started -- so scene.js draws exactly the same
    track the marker and camera already agree on (never a separate progress
    fraction re-applied to the bundle's own, differently-segmented
    ``route.geojson``, which is what let the travelled line drift ahead of
    or behind the marker).

    Cut at exactly the same gaps the display segments are already split on:
    each part holds only points from ONE segment, never a straight edge
    bridging a GPS gap between two of them. The last vertex of the returned
    geometry is always ``marker``'s own coordinate -- the interpolated
    position at ``at_time`` on a normal frame; when ``at_time`` falls in a
    gap and the marker is held on a point whose segment has not started yet,
    that point becomes its own single-vertex part instead of stretching the
    previous segment's line across the gap to reach it.
    """
    marker_point = [marker.longitude, marker.latitude]
    parts: list[list[list[float]]] = []
    for segment in segments:
        if not segment or segment[0].time_s > at_time:
            continue
        points = [[point.longitude, point.latitude] for point in segment if point.time_s <= at_time]
        if not in_gap and segment[0].time_s <= at_time <= segment[-1].time_s:
            points.append(marker_point)
        parts.append(points)
    if in_gap and not any(part and part[-1] == marker_point for part in parts):
        parts.append([marker_point, marker_point])
    if not parts:
        parts.append([marker_point, marker_point])
    return parts


def _marker_at(segments: Sequence[Sequence[RoutePoint]], at_time: float) -> tuple[RoutePoint, bool]:
    """The route point shown at ``at_time``, and whether it fell in a GPS gap."""
    marker = _point_at_time(segments, at_time)
    if marker is not None:
        return marker, False
    nearest = min(
        (point for segment in segments for point in segment),
        key=lambda point: abs(point.time_s - at_time),
    )
    return nearest, True


def _chord_bearing(
    segments: Sequence[Sequence[RoutePoint]],
    at_time: float,
    window_s: float,
    leg_start: float,
    leg_end: float,
    previous: float,
) -> float:
    """The route's travel direction AT THE LEG'S OWN SCALE: the bearing of the
    chord from ``at_time - window_s`` to ``at_time + window_s`` (clamped to the
    leg's own bounds), not a short-range vertex-to-vertex heading.

    A film map leg can put ~30 real route-seconds between two adjacent
    *displayed* frames; a short window still just resamples GPS-jitter and
    switchback noise at that scale, which is what previously made the raw
    per-frame target swing wildly and saturate any downstream rate limiter.
    Below ``_MIN_CHORD_DISTANCE_M`` the two chord endpoints are too close to
    carry a reliable direction (an out-and-back turnaround, a stop), so this
    holds ``previous`` rather than chase that noise.
    """
    start_t = max(leg_start, at_time - window_s)
    end_t = min(leg_end, at_time + window_s)
    start_point, _ = _marker_at(segments, start_t)
    end_point, _ = _marker_at(segments, end_t)
    if distance_m(start_point, end_point) < _MIN_CHORD_DISTANCE_M:
        return previous
    return _initial_bearing_deg(
        start_point.longitude, start_point.latitude, end_point.longitude, end_point.latitude
    )


def _plan_bearings(
    targets: Sequence[float],
    start_bearing: float | None,
    start_velocity: float,
    fps: float,
    max_speed_deg_s: float,
    max_accel_deg_s2: float,
) -> tuple[list[float], list[float]]:
    """Chase each raw target bearing under BOTH a speed and an acceleration cap.

    A pure speed limiter (clamp the per-frame step) still lets the camera
    accelerate instantly and then hold max speed for an entire leg -- which
    is legal by the speed cap alone but violates
    ``planner.DEFAULT_MOTION_LIMITS["bearing_accel_deg_s2"]`` and, in
    practice, spins the camera through the whole shot. This is a
    textbook trapezoidal motion profile: each frame's desired velocity is
    capped both by ``max_speed_deg_s`` and by the velocity that lets it
    decelerate to exactly zero right as it reaches the target
    (``sqrt(2 * max_accel_deg_s2 * |delta|)``), so it never overshoots; the
    velocity itself can then only change by ``max_accel_deg_s2 / fps`` per
    frame. Returns the displayed bearing AND angular velocity per frame, so a
    caller can carry both across a leg boundary.
    """
    if not targets:
        return [], []
    # A hair under the nominal caps: `planner.assert_smooth` compares with a
    # strict `>`, and a value planned to land exactly ON a cap can overshoot
    # it by a few floating-point ulps and fail that comparison for no real
    # motion reason.
    safety = 1.0 - 1e-9
    max_speed = max_speed_deg_s * safety
    max_velocity_step = (max_accel_deg_s2 * safety) / fps
    bearing = targets[0] if start_bearing is None else start_bearing
    velocity = start_velocity
    bearings: list[float] = []
    velocities: list[float] = []
    for target in targets:
        delta = _shortest_angle_delta(bearing, target)
        if delta == 0.0:
            desired_velocity = 0.0
        else:
            magnitude = min(max_speed, math.sqrt(2.0 * max_accel_deg_s2 * safety * abs(delta)))
            desired_velocity = math.copysign(magnitude, delta)
        velocity += max(-max_velocity_step, min(max_velocity_step, desired_velocity - velocity))
        velocity = max(-max_speed, min(max_speed, velocity))
        # No separate "don't overshoot the target" clip here: `desired_velocity`
        # is already the exact braking profile for THIS frame's target, so
        # clipping the resulting step to `delta` would silently change the
        # displayed velocity by more than `max_velocity_step` the moment the
        # target itself jumps between frames -- breaking the acceleration cap
        # this whole function exists to hold. A route-direction target that
        # jumps is followed by a still-capped correction on the next frame
        # instead, which is what actually keeps every frame within both caps.
        step = velocity / fps
        bearing = (bearing + step) % 360.0
        bearings.append(bearing)
        velocities.append(velocity)
    return bearings, velocities


@dataclass
class SceneMapRenderer:
    """Renders film map-leg frames as an oblique follow shot over real terrain.

    ``capture`` is any object with a ``render(pose, light=...) -> (array,
    labels)`` method -- normally an already-entered ``scene.capture.SceneCapture``,
    or a fake in tests. It must have been constructed for a frame of size
    ``(width, scene_map_height(height))`` for whatever ``width``/``height`` this
    renderer is called with; a mismatched capture raises.

    Deliberately not a frozen dataclass: the follow camera has real inertia
    -- ``render_leg`` carries the last frame's displayed bearing AND angular
    velocity from one map leg to the next, across however many times
    ``render_leg`` is called on this same instance over the life of one film
    export, so the camera never snaps back to a raw target bearing (or a
    dead stop) at a leg boundary.
    """

    capture: _SceneRendererCapture
    trip_epoch: datetime
    max_zoom: float
    follow_zoom: float = 15.5
    pitch: float = 60.0
    credit: str = ""
    fps: float = 30.0
    bearing_speed_deg_s: float = DEFAULT_MOTION_LIMITS["bearing_speed_deg_s"]
    bearing_accel_deg_s2: float = DEFAULT_MOTION_LIMITS["bearing_accel_deg_s2"]
    name: str = "maplibre-scene"
    # (time_s, longitude, latitude) per placed photo/video beat -- see
    # film._beat_pins. Filtered to "already played" and collapsed by
    # ``layers.visible_pins`` per frame, the same recipe the Pillow renderer
    # and the recap map use. A ``dataclasses.replace`` clone (the per-worker
    # capture path) carries this like every other field -- no special casing.
    pins: Sequence[PhotoPinBeat] = ()
    workers: int = 1
    capture_factory: Callable[[], Any] | None = field(default=None, repr=False)
    # No frame from a worker for this long aborts the leg (see ``_get_or_raise``).
    stall_timeout_s: float = WORKER_STALL_TIMEOUT_S
    join_deadline_s: float = WORKER_JOIN_DEADLINE_S
    frame_qa: FrameQaThresholds | None = field(default_factory=FrameQaThresholds)
    # Given a ledger, a QA fault is RECORDED and the render carries on; the
    # caller fails the run from the ledger once the film exists. Left None, a
    # fault raises immediately -- which is what the film preflight wants, since
    # it judges a handful of frames precisely so the run can refuse before
    # spending hours. See FrameQaLedger for why the two modes exist.
    qa_ledger: FrameQaLedger | None = field(default=None, repr=False)
    # The recorded points the dashboard measures when the drawn line is a
    # display-smoothed copy of them; None measures what is drawn.
    measured_segments: Sequence[Sequence[RoutePoint]] | None = field(default=None, repr=False)
    # Tells a caller (the film preflight) every frame is already QA'd raw here.
    checks_raw_frames: ClassVar[bool] = True
    # Built ONCE per film (see ``_profile_geometry_for``), never per frame --
    # the distance/elevation profile is fixed for the whole film, so nothing
    # about a single frame's render should recompute it.
    _profile_geometry: ProfileGeometry | None = field(default=None, init=False, repr=False)
    _profile_geometry_key: int | None = field(default=None, init=False, repr=False)
    _last_bearing: float | None = field(default=None, init=False, repr=False)
    _last_bearing_velocity: float = field(default=0.0, init=False, repr=False)
    # Set for the length of one ``render_leg`` SERIAL call only (see there):
    # gives an in-view label memory across that leg's frames so a collision
    # or rank flip never pops it out for a single frame. None outside a leg
    # (a lone ``render()`` call has no "previous frame" to carry) --
    # ``_finish_frame`` skips the reorder entirely then. The PARALLEL branch
    # (``_capture_leg_parallel``) gets the identical persistence from its own
    # leg-scoped ``LegLabelPlanner``, applied in the single-threaded consumer
    # instead of here -- see ``_CapturedFrame``/``_finish_frame``.
    _leg_label_planner: LegLabelPlanner | None = field(default=None, init=False, repr=False)
    # The most recently rendered frame's marker position, in this capture's
    # own pixel space -- set every _render_frame call so frame_qa's
    # travelled-line-width floor runs against the real marker, never a
    # guessed centre. checks_raw_frames=True means preflight's own loop never
    # re-derives this; it is read by callers (proof.py) that want it too.
    last_marker_xy: tuple[float, float] | None = field(default=None, init=False, repr=False)
    # One record per ``render_leg`` call that passed its camera continuity
    # check -- {"checked": frame count, "cuts": planned-cut count, "issues":
    # ()} -- appended here so ``film_receipt`` can carry it; a failed check
    # raises ``CameraContinuityError`` instead of ever appending. init=False
    # for the same reason as ``_pool``: a per-worker ``dataclasses.replace``
    # clone must not inherit or duplicate the parent's ledger.
    camera_continuity: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)
    # Every ``render_leg`` call's planned per-frame ``CameraState``s, in render
    # order, across the whole film -- the actually-rendered camera path
    # ``film_receipt`` reduces to a motion summary. Appended alongside
    # ``camera_continuity`` (same list, same guard: only once a leg's states
    # pass ``check_camera_continuity``), never rebuilt from it, so the two
    # can't drift apart. init=False for the same per-worker-clone reason as
    # ``camera_continuity``.
    rendered_poses: list[CameraState] = field(default_factory=list, init=False, repr=False)
    # Capture workers, opened on the first parallel leg and kept for the whole
    # film -- see ``_ensure_pool``. init=False so a ``dataclasses.replace``
    # clone (what each worker renders through) never inherits, or shuts down,
    # the pool it is a member of.
    _pool: list[_CaptureWorker] | None = field(default=None, init=False, repr=False)

    def measuring(self, segments: Sequence[Sequence[RoutePoint]]) -> "SceneMapRenderer":
        # Same instance on purpose: the follow camera's carried bearing state
        # and the caller's handle on this renderer both survive.
        self.measured_segments = segments
        # The profile is built here, once per film, from the same display
        # segments the dashboard measures -- never rebuilt per frame.
        self._profile_geometry = build_profile(segments)
        self._profile_geometry_key = id(segments)
        return self

    def _profile_geometry_for(self, segments: Sequence[Sequence[RoutePoint]]) -> ProfileGeometry:
        """The film's profile geometry, built once and cached by identity.

        ``measuring()`` is the normal film-level call site. A lone
        ``render()``/``render_leg()`` call that never went through
        ``measuring()`` (single-frame legacy path) still gets a profile,
        built from whatever segments it was actually called with.
        """
        source = self.measured_segments or segments
        key = id(source)
        if self._profile_geometry is None or self._profile_geometry_key != key:
            self._profile_geometry = build_profile(source)
            self._profile_geometry_key = key
        return self._profile_geometry

    def render(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        at_time: float,
        width: int,
        height: int,
    ) -> Image.Image:
        """Render one frame at its raw (unlimited) target bearing.

        There is no adjacent frame to rate-limit against in a single call --
        a map leg's frames must go through ``render_leg`` instead, which is
        what keeps consecutive displayed frames within the planner's
        bearing-speed cap on a real, heavily time-compressed route.
        """
        bearing = _target_bearing(segments, at_time, _MIN_BEARING_WINDOW_S)
        return self._render_frame(segments, at_time, bearing, width, height)

    def render_leg(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        times: Sequence[float],
        width: int,
        height: int,
    ) -> Iterator[Image.Image]:
        if not times:
            return
        leg_start, leg_end = times[0], times[-1]
        window_s = max(_MIN_CHORD_WINDOW_S, _CHORD_WINDOW_FRACTION * (leg_end - leg_start))
        fallback = (
            self._last_bearing
            if self._last_bearing is not None
            else _target_bearing(segments, leg_start, _MIN_BEARING_WINDOW_S)
        )
        targets = []
        for at_time in times:
            fallback = _chord_bearing(segments, at_time, window_s, leg_start, leg_end, fallback)
            targets.append(fallback)
        bearings, velocities = _plan_bearings(
            targets,
            self._last_bearing,
            self._last_bearing_velocity,
            self.fps,
            self.bearing_speed_deg_s,
            self.bearing_accel_deg_s2,
        )
        self._last_bearing = bearings[-1]
        self._last_bearing_velocity = velocities[-1]
        # Judge the PLANNED camera states for this leg before any capture
        # worker starts -- a leg is one continuous follow shot with no
        # internal planned cut, so every frame-to-frame jump here is real
        # (a bearing jump is already excluded by ``_plan_bearings``'s own
        # speed/accel caps; this also catches a marker-position snap, e.g.
        # across a GPS gap, that bearing planning alone can't see).
        zoom = min(self.follow_zoom, self.max_zoom)
        camera_states = [
            CameraState(
                center_lat=_marker_at(segments, at_time)[0].latitude,
                center_lon=_marker_at(segments, at_time)[0].longitude,
                bearing_deg=bearing,
                zoom=zoom,
                pitch_deg=self.pitch,
            )
            for at_time, bearing in zip(times, bearings)
        ]
        issues = check_camera_continuity(camera_states)
        if issues:
            raise CameraContinuityError(issues)
        self.camera_continuity.append({"checked": len(camera_states), "cuts": 0, "issues": ()})
        self.rendered_poses.extend(camera_states)
        # Built ONCE per leg (not per frame): a genuine per-leg timeline gives
        # ``dashboard_state_at``'s cadence quantization real output-frame
        # structure to hold steady across, and every frame of the leg -- serial
        # or parallel worker path -- shares it.
        dashboard_timeline = _leg_dashboard_timeline(times)
        dashboard_cadence_frames = max(1, round(_DASHBOARD_CADENCE_S * self.fps))
        # Frozen once for this leg from what its own segments actually carry
        # -- never re-decided per frame, so a mid-leg sensor dropout can't
        # remove or reflow a dashboard column. Set as module state (mirroring
        # the clock anchor / motion-cell mode above ``_draw_dashboard``)
        # rather than threaded onto ``_CaptureJob``/``_render_frame``, and
        # cleared once the leg finishes so it never leaks into an unrelated
        # single-frame ``render()`` call.
        set_active_dashboard_fields(dashboard_fields_for(self.measured_segments or segments))
        try:
            if self.workers <= 1 or self.capture_factory is None:
                self._leg_label_planner = LegLabelPlanner()
                try:
                    for index, (at_time, bearing) in enumerate(zip(times, bearings)):
                        yield self._render_frame(
                            segments,
                            at_time,
                            bearing,
                            width,
                            height,
                            index,
                            dashboard_timeline=dashboard_timeline,
                            dashboard_cadence_frames=dashboard_cadence_frames,
                        )
                finally:
                    self._leg_label_planner = None
                return
            yield from self._capture_leg_parallel(
                segments,
                times,
                bearings,
                width,
                height,
                dashboard_timeline,
                dashboard_cadence_frames,
            )
        finally:
            set_active_dashboard_fields(None)

    def _capture_leg_parallel(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        times: Sequence[float],
        bearings: Sequence[float],
        width: int,
        height: int,
        dashboard_timeline: StoryTimeline,
        dashboard_cadence_frames: int,
    ) -> Iterator[Image.Image]:
        """Fan the CAPTURE step (only) across ``self.workers`` threads, streamed.

        Every frame's bearing is already planned serially by ``render_leg``
        above -- the follow camera's inertia state must never be touched
        from more than one thread. Each worker drives its OWN capture
        (Playwright's sync API is not thread-safe across threads sharing one
        browser) over a disjoint, striped slice of frame indices. Same
        consumer as ``proof._render_frames_parallel``: one BOUNDED queue per
        worker, drained strictly in frame order, so the yielded order and
        every pose match a serial render while a worker can only run a few
        frames ahead -- a leg is thousands of multi-megabyte frames and is
        never held whole. A worker failure raises the ORIGINAL exception, so
        the film run aborts rather than encoding a partial leg.

        Those captures belong to a pool that OUTLIVES this leg (see
        ``_ensure_pool``): a browser is launched once per worker per film,
        not once per worker per leg.
        """
        if self.capture_factory is None:
            raise ValueError("parallel capture needs a capture_factory")
        ranges = _strided_frame_ranges(len(times), self.workers)
        workers = self._ensure_pool(len(ranges))
        result_queues: list[queue.Queue[Any]] = [
            queue.Queue(maxsize=_WORKER_QUEUE_DEPTH) for _ in ranges
        ]
        cancel_event = threading.Event()
        failure: list[BaseException] = []
        done = [threading.Event() for _ in ranges]
        for index, frame_range in enumerate(ranges):
            workers[index].jobs.put(
                _CaptureJob(
                    frame_range=frame_range,
                    segments=segments,
                    times=times,
                    bearings=bearings,
                    width=width,
                    height=height,
                    results=result_queues[index],
                    cancel_event=cancel_event,
                    failure=failure,
                    done=done[index],
                )
            )
        # Leg-scoped, single-threaded incumbent memory -- same rule, same
        # class, as the serial branch's ``self._leg_label_planner``, but kept
        # local here rather than on ``self``: this loop, not any worker, is
        # the one thread that ever sees frames in true order, and each
        # worker's own ``renderer`` clone must still carry ``None`` so a
        # worker's ``_capture_frame`` call never tries to reorder labels
        # itself (see ``_worker_loop``).
        planner = LegLabelPlanner()
        try:
            for index in range(len(times)):
                captured = _get_or_raise(
                    result_queues[index % len(ranges)],
                    cancel_event,
                    failure,
                    self.stall_timeout_s,
                )
                yield self._finish_frame(
                    captured, dashboard_timeline, dashboard_cadence_frames, planner
                )
        except BaseException:
            # A leg that failed leaves its browsers in an unknown state --
            # one may be mid-crash, or stuck inside a call that never
            # returns. Reusing those on the next leg would spread this
            # leg's failure into a healthy one, so the pool is torn down
            # and the next leg opens fresh browsers.
            cancel_event.set()
            self._drain_until_done(result_queues, done)
            self.close()
            raise
        finally:
            # Also runs when the consumer stops early (generator closed).
            cancel_event.set()
            self._drain_until_done(result_queues, done)

    def _drain_until_done(
        self, result_queues: list["queue.Queue[Any]"], done: list[threading.Event]
    ) -> None:
        """Unstick any worker blocked writing to a queue the consumer
        abandoned, then wait for each to finish ITS JOB -- never for the
        thread to exit, which in a pool only happens at ``close``.
        """
        deadline = time.monotonic() + self.join_deadline_s
        while not all(event.is_set() for event in done) and time.monotonic() < deadline:
            for result_queue in result_queues:
                try:
                    result_queue.get(timeout=_QUEUE_POLL_S)
                except queue.Empty:
                    pass
        stuck = sum(1 for event in done if not event.is_set())
        if stuck:
            logger.error(
                "%d capture worker(s) still running %g s after cancellation",
                stuck,
                self.join_deadline_s,
            )

    def _ensure_pool(self, count: int) -> list[_CaptureWorker]:
        """The film's capture workers, started on first use and reused after.

        Starting a fresh browser per worker per leg costs six Chromium
        launches at every leg boundary, and a launch that misses its 60 s
        readiness timeout has to be relaunched, which idles the healthy workers
        because frames drain in strict order. The thread is what has to persist, not just the browser
        handle -- Playwright's sync API pins a browser to the thread that
        opened it.
        """
        pool = self._pool
        if pool is not None and len(pool) == count:
            return pool
        self.close()
        pool = []
        for index in range(count):
            worker = _CaptureWorker(jobs=queue.Queue())
            worker.thread = threading.Thread(
                target=self._worker_loop,
                args=(worker,),
                daemon=True,
                name=f"route-story-capture-{index}",
            )
            worker.thread.start()
            pool.append(worker)
        self._pool = pool
        return pool

    def _worker_loop(self, worker: _CaptureWorker) -> None:
        """Open one capture lazily, then render whatever legs arrive on ``jobs``.

        The capture opens on the FIRST job rather than at thread start so a
        launch failure is reported through that leg's own failure list --
        the consumer's error path -- instead of dying unobserved in a thread
        nobody is waiting on yet.
        """
        factory = self.capture_factory
        capture: Any = None
        rendering_capture: Any = None
        entered = False
        try:
            while True:
                job = worker.jobs.get()
                if job is None:
                    return
                try:
                    if not entered:
                        if factory is None:  # pragma: no cover -- guarded by caller
                            raise ValueError("parallel capture needs a capture_factory")
                        capture = factory()
                        # Render through what __enter__ RETURNS, not the context
                        # manager itself. They are the same object for
                        # SceneCapture, which returns self -- but a factory is
                        # free to hand back a wrapper, and using the wrapper as
                        # the capture is how this pool lost every photo pin in
                        # the multi-worker path (the per-leg threads it replaced
                        # used ``with factory() as worker_capture``). ``capture``
                        # stays bound to the manager so __exit__ still closes it.
                        rendering_capture = capture.__enter__()
                        entered = True
                    renderer = replace(self, capture=rendering_capture)
                    for index in job.frame_range:
                        if job.cancel_event.is_set():
                            break
                        # CAPTURE only -- never labels/HUD (see ``_CapturedFrame``).
                        # The consumer finishes each frame sequentially so the
                        # never-evict label rule sees true frame order.
                        job.results.put(
                            renderer._capture_frame(
                                job.segments,
                                job.times[index],
                                job.bearings[index],
                                job.width,
                                job.height,
                                index,
                            )
                        )
                except BaseException as exc:  # noqa: BLE001 -- surfaced by the consumer
                    job.failure.append(exc)
                    job.cancel_event.set()
                finally:
                    job.done.set()
        finally:
            if entered:
                capture.__exit__(None, None, None)

    def close(self) -> None:
        """Shut every pooled browser. Idempotent; safe before any leg ran.

        ``film.py`` calls this at the end of an export. The worker threads are
        daemons, so without it a finished film would leave browsers alive
        until the process itself exits.
        """
        pool, self._pool = self._pool, None
        if not pool:
            return
        for worker in pool:
            worker.jobs.put(None)
        deadline = time.monotonic() + self.join_deadline_s
        for worker in pool:
            if worker.thread is None:
                continue
            remaining = deadline - time.monotonic()
            if remaining > 0:
                worker.thread.join(timeout=remaining)
        stuck = [w for w in pool if w.thread is not None and w.thread.is_alive()]
        if stuck:
            logger.error(
                "%d capture worker(s) did not shut down within %g s",
                len(stuck),
                self.join_deadline_s,
            )

    def _render_frame(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        at_time: float,
        bearing: float,
        width: int,
        height: int,
        frame_index: int | None = None,
        dashboard_timeline: StoryTimeline | None = None,
        dashboard_cadence_frames: int = 1,
    ) -> Image.Image:
        """Capture then finish one frame, in that order, on this thread.

        The single-frame ``render()`` call and ``render_leg``'s SERIAL branch
        both still go through this -- only ``_capture_leg_parallel`` calls
        ``_capture_frame``/``_finish_frame`` separately, across the
        worker/consumer boundary. ``self._leg_label_planner`` is whatever
        ``render_leg`` set for this call (a fresh ``LegLabelPlanner`` for a
        serial leg, ``None`` outside one).
        """
        captured = self._capture_frame(segments, at_time, bearing, width, height, frame_index)
        return self._finish_frame(
            captured, dashboard_timeline, dashboard_cadence_frames, self._leg_label_planner
        )

    def _capture_frame(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        at_time: float,
        bearing: float,
        width: int,
        height: int,
        frame_index: int | None = None,
    ) -> "_CapturedFrame":
        """The CAPTURE step only: the slow, terrain-aware browser render plus
        frame QA -- no label reordering or pixel drawing. This is the piece
        ``_capture_leg_parallel`` fans across workers; everything after it
        (``_finish_frame``) is cheap enough to run sequentially in the
        consumer instead, which is what lets the never-evict label rule see
        a real frame-to-frame order without capturing any frame twice.
        """
        marker, in_gap = _marker_at(segments, at_time)

        map_height = scene_map_height(height)
        zoom = min(self.follow_zoom, self.max_zoom)
        pose: dict[str, Any] = {
            "longitude": marker.longitude,
            "latitude": marker.latitude,
            "zoom": zoom,
            "pitch": self.pitch,
            "bearing": bearing,
            "marker": [marker.longitude, marker.latitude],
            "marker_heading": _target_bearing(segments, at_time, _MIN_BEARING_WINDOW_S),
            "travelled": _travelled_coordinates(segments, at_time, marker, in_gap),
            # Sent on EVERY pose, not cached/sent-once here: a pose dict is
            # what a frame cache keys on (see scene/frame_cache.py), so it
            # must fully determine the pixels on its own -- a page that
            # never received a "first" pose (a cache hit skipped it, or a
            # fresh page opened mid-leg) must still be able to draw this
            # frame's route correctly from this pose alone. scene.js is what
            # dedupes the repeated, usually-identical geometry against
            # re-tiling the source on every frame.
            "route": [
                [[point.longitude, point.latitude] for point in segment]
                for segment in segments
                if segment
            ],
        }
        # Only once a beat has played: a pin-less pose stays the exact dict it
        # was before pins existed, so a frame cache keyed on it keeps its frames.
        pins = visible_pins(self.pins, at_time)
        if pins:
            pose["pins"] = [list(pin) for pin in pins]
        # The day's start and finish, fixed for every frame -- unlike the
        # photo pins above these never wait for a beat, so they are keyed off
        # the track alone. Omitted entirely for an empty track, keeping a
        # pose that predates endpoints byte-identical for the frame cache.
        endpoints = route_endpoints(segments)
        if endpoints is not None:
            pose["endpoints"] = [list(endpoint) for endpoint in endpoints]
        pose_time = self.trip_epoch + timedelta(seconds=at_time)
        light = light_for_pose(
            pose_time, marker.latitude, marker.longitude, fallback=self.trip_epoch
        )

        array, labels = self.capture.render(pose, light=light)
        if array.shape[0] != map_height or array.shape[1] != width:
            raise ValueError(
                "scene capture frame size "
                f"{(array.shape[1], array.shape[0])} does not match the requested "
                f"film frame {(width, map_height)}"
            )
        self.last_marker_xy = marker_xy_from_labels(labels)
        # Judge the RAW capture: the label pills and dashboard drawn over it
        # are dark fills within the unloaded-tile detector's tolerance.
        if self.frame_qa is not None:
            # The index reported must be a FRAME index. It used to be
            # ``int(at_time)`` -- trip SECONDS -- so a refusal read "frame
            # 11503" on a film of 4353 frames and sent two diagnoses down the
            # wrong path before anyone read this line. Callers that have no
            # frame index (render_frame, one pose on its own) still fall back
            # to the clock, and the pose in the message pins it either way.
            index = int(at_time) if frame_index is None else frame_index
            qa = check_frame(index, array, self.frame_qa, marker_xy=self.last_marker_xy)
            if not qa.ok:
                if self.qa_ledger is None:
                    raise FrameQaError(qa, pose, rgb=array)
                self.qa_ledger.record(qa, pose, rgb=array)
        # Resolved before labels draw, not after: the dashboard rail and
        # elevation-profile boxes below are HUD keep-clear zones a label pill
        # must not land on, so the layout has to exist before _draw_labels
        # runs. Reusing this one instance below (rather than a second
        # hud_layout() call) keeps it the single source of truth per frame.
        layout = hud_layout(width, height)
        dashboard_box = (
            layout.dashboard_left,
            layout.dashboard_panel_top,
            layout.dashboard_right,
            layout.dashboard_panel_top + layout.dashboard_panel_height,
        )
        profile_box = (
            layout.profile_left,
            layout.profile_top,
            layout.profile_right,
            layout.profile_top + layout.profile_height,
        )
        return _CapturedFrame(
            array=array,
            labels=labels,
            layout=layout,
            dashboard_box=dashboard_box,
            profile_box=profile_box,
            map_height=map_height,
            in_gap=in_gap,
            at_time=at_time,
            segments=segments,
            width=width,
            height=height,
            frame_index=frame_index,
        )

    def _finish_frame(
        self,
        captured: "_CapturedFrame",
        dashboard_timeline: StoryTimeline | None,
        dashboard_cadence_frames: int,
        planner: LegLabelPlanner | None,
    ) -> Image.Image:
        """Everything after capture: the never-evict label reorder, drawing
        the pills, and the HUD -- cheap CPU work, always run on ONE thread so
        ``planner`` (whichever leg-scoped ``LegLabelPlanner`` the caller
        passed) sees frames in true, strictly increasing order regardless of
        which worker captured any of them.
        """
        labels = captured.labels
        if planner is not None:
            labels = planner.step(labels)
        array = _draw_labels(
            captured.array, labels, extra_keep_clear=(captured.dashboard_box, captured.profile_box)
        )

        canvas = Image.new("RGB", (captured.width, captured.height), "#071219")
        canvas.paste(Image.fromarray(array, mode="RGB"), (0, 0))

        if dashboard_timeline is not None and captured.frame_index is not None:
            timeline, dashboard_frame, cadence_frames = (
                dashboard_timeline,
                captured.frame_index,
                dashboard_cadence_frames,
            )
        else:
            timeline = StoryTimeline(
                fps=1, scenes=[Scene("overview", 0, 1, captured.at_time, captured.at_time)]
            )
            dashboard_frame, cadence_frames = 0, 1
        # The line drawn may be display-smoothed; the numbers never are.
        dashboard = dashboard_state_at(
            timeline,
            dashboard_frame,
            self.measured_segments or captured.segments,
            moving_speed_mps=0.5,
            cadence_frames=cadence_frames,
        )
        draw_profile(
            canvas, self._profile_geometry_for(captured.segments), captured.at_time, captured.layout
        )
        _draw_dashboard(canvas, dashboard, self.credit)
        if captured.in_gap:
            _draw_gap_badge(canvas, captured.map_height)
        return canvas
