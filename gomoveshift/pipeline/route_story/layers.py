"""Route Story film map layer: geographic projection and on-map drawing.

This module owns turning credited satellite rasters and terrain into a single
camera view for one frame — label selection, the opening continent-to-trail
camera move, the route line, the marker, orientation cues, and public-name
labels. It does not know about photo/video overlays, typography sizing, or
unit formatting (``hud``) or how frames become encoded clips (``assembly``).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import numpy as np
from PIL import Image, ImageDraw

from . import grade, graded_cache
from .dashboard import dashboard_state_at
from .frame_state import frame_state_at
from .hud import _draw_dashboard, _font, _interpolate, _smoothstep, dashboard_fields_for
from .metrics import RoutePoint, distance_m
from .satellite import BACKDROP_BAND, SatelliteRaster
from .terrain import TerrainGrid
from .timeline import Scene, StoryTimeline

# One (time_s, longitude, latitude) row per placed photo/video beat -- see
# film._beat_pins, the only builder. A pure filter/collapse step so both
# renderers (and the recap map) draw identically from one recipe.
PhotoPinBeat = tuple[float, float, float]

# Two beats within this many metres of one another draw as one pin, not two
# almost-overlapping ones (a photo and the video shot at the same overlook).
_PIN_COLLAPSE_DISTANCE_M = 15.0


def visible_pins(beats: Sequence[PhotoPinBeat], at_time: float) -> list[tuple[float, float]]:
    """(longitude, latitude) pins for every beat that has already played.

    A beat becomes visible the instant its own route-clock time is reached
    (``<=``, not ``<``): the map leg that resumes right after a photo/video
    still starts exactly at that beat's time, so the pin must already be
    there on the very first frame after the still, never one frame late --
    and never before the still has played. Beats within
    ``_PIN_COLLAPSE_DISTANCE_M`` metres of an already-kept pin are dropped
    rather than drawn twice at (almost) the same spot -- kept in ``beats``'
    own order, so the first beat at a shared location wins.
    """
    kept: list[RoutePoint] = []
    pins: list[tuple[float, float]] = []
    for time_s, longitude, latitude in beats:
        if time_s > at_time:
            continue
        point = RoutePoint(time_s, latitude, longitude)
        if any(distance_m(point, other) < _PIN_COLLAPSE_DISTANCE_M for other in kept):
            continue
        kept.append(point)
        pins.append((longitude, latitude))
    return pins


_LABEL_PRIORITY = {
    "town": 100,
    "city": 98,
    "village": 96,
    "park": 86,
    "forest": 84,
    "peak": 80,
    "trailhead": 76,
    "viewpoint": 74,
    "waterfall": 72,
    "cave": 70,
    "cliff": 68,
    "water": 64,
    "road": 54,
    "trail": 50,
    "poi": 45,
}
_LABEL_SPAN_DEFAULTS = {
    "town": 25.0,
    "city": 25.0,
    "village": 18.0,
    "park": 4.0,
    "forest": 4.0,
    "peak": 2.0,
    "trailhead": 1.0,
    "viewpoint": 1.0,
    "waterfall": 1.0,
    "cave": 1.0,
    "cliff": 1.5,
    "water": 0.75,
    "road": 0.8,
    "trail": 0.5,
    "poi": 1.0,
}


@functools.lru_cache(maxsize=8)
def _raster_image(source: Path) -> Image.Image:
    """Keep approved raster pixels resident during a frame sequence.

    A story can have thousands of frames but only a handful of immutable,
    reviewed imagery sources.  Reopening multi-megapixel JPEGs per frame makes
    the renderer unnecessarily slow and does not improve output quality.
    """
    with Image.open(source) as opened:
        return opened.convert("RGB").copy()


@functools.lru_cache(maxsize=None)
def _raster_width(source: Path) -> int:
    """A raster's pixel width, read from the file header -- never a decode.

    Every question the opening asks about a raster's RESOLUTION goes through
    here. It is unbounded on purpose: an entry is one integer, and a bounded
    cache is what made this slow (see ``docs/pipeline-internals.md``).
    """
    with Image.open(source) as opened:
        return int(opened.size[0])


@functools.lru_cache(maxsize=8)
def _graded_raster_image(source: Path) -> Image.Image:
    """The opening's still-photo grade (haze/exposure/vibrance/punch), applied
    once per raster and cached -- grading a full multi-megapixel mosaic is
    too slow to repeat for every one of an opening's frames, and every frame
    drawn from the same raster should look identical anyway.

    Cached on disk as well as in memory: the grade is a pure function of the
    raster's bytes and ``grade.py``, and grading a typical opening set (14
    bundle rasters, 298 MP, 2.7 s/MP) costs 13.5 minutes at the start of EVERY
    render -- then again whenever the eight-entry memory cache evicted one.
    """
    cached_path = _graded_cache_path(source)
    graded = graded_cache.load(cached_path)
    if graded is None:
        graded = grade.grade_still(_raster_image(source))
        graded_cache.store(cached_path, graded)
    return graded


def _grade_code_identity() -> str:
    return graded_cache.code_identity(grade)


def _graded_cache_path(source: Path) -> Path | None:
    return graded_cache.cache_path(source, _grade_code_identity())


@dataclass(frozen=True)
class MapLabel:
    """A public geographic name eligible for the map overlay."""

    name: str
    latitude: float
    longitude: float
    kind: str = "poi"
    max_span_lon: float | None = None
    min_span_lon: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("map labels need a name")
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ValueError("map label coordinates must be valid geographic values")
        if self.max_span_lon is not None and self.max_span_lon <= 0:
            raise ValueError("map label visibility span must be positive")
        if self.min_span_lon is not None and self.min_span_lon <= 0:
            raise ValueError("map label minimum visibility span must be positive")
        if (
            self.min_span_lon is not None
            and self.max_span_lon is not None
            and self.min_span_lon > self.max_span_lon
        ):
            raise ValueError("map label visibility band is inverted")

    @property
    def visibility_span(self) -> float:
        return self.max_span_lon or _LABEL_SPAN_DEFAULTS.get(self.kind, 1.0)

    @property
    def minimum_visibility_span(self) -> float:
        return self.min_span_lon or 0.0


def load_map_labels(source: Path) -> tuple[MapLabel, ...]:
    """Load a small, reviewed public-label manifest without network access."""
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("map label manifest must be a JSON list")
    labels: list[MapLabel] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each map label must be an object")
        labels.append(
            MapLabel(
                str(item["name"]),
                float(item["latitude"]),
                float(item["longitude"]),
                str(item.get("kind", "poi")).lower(),
                float(item["max_span_lon"]) if item.get("max_span_lon") is not None else None,
                float(item["min_span_lon"]) if item.get("min_span_lon") is not None else None,
            )
        )
    return tuple(labels)


def select_visible_labels(
    labels: Sequence[MapLabel],
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    span_lon: float,
    max_labels: int = 12,
) -> tuple[MapLabel, ...]:
    """Choose deterministic, scale-aware labels for one map viewport."""
    if span_lon <= 0 or max_labels <= 0:
        return ()
    center_lat, center_lon = (south + north) / 2, (west + east) / 2
    candidates: list[MapLabel] = []
    seen: set[str] = set()
    for label in labels:
        key = label.name.casefold()
        if key in seen or not (
            west <= label.longitude <= east and south <= label.latitude <= north
        ):
            continue
        if span_lon > label.visibility_span or span_lon < label.minimum_visibility_span:
            continue
        seen.add(key)
        candidates.append(label)
    candidates.sort(
        key=lambda label: (
            -_LABEL_PRIORITY.get(label.kind, _LABEL_PRIORITY["poi"]),
            abs(label.latitude - center_lat) + abs(label.longitude - center_lon),
            label.name.casefold(),
        )
    )
    return tuple(candidates[:max_labels])


# The opening camera's pose is a pure function of ``fraction`` (see
# ``_opening_camera``), so its whole path is already determined before a
# single frame renders. The label plan below walks that whole path once, at
# this fixed grid, instead of re-ranking labels from scratch every frame --
# a per-frame, stateless re-rank is what let the visible set flicker between
# four different label combinations in under two seconds (see
# docs/pipeline-internals.md).
_OPENING_LABEL_SAMPLES = 241

# A label whose qualifying on-screen run is shorter than this fraction of the
# whole move is dropped rather than flashed for a frame or two -- roughly
# 1.5s of a nominal 12s intro (``film.DEFAULT_INTRO_SECONDS``). layers.py has
# no access to the run's real fps/duration here, only ``fraction``, so this
# is a fixed fraction-space floor rather than a seconds one.
_OPENING_LABEL_DWELL_FRACTION = 0.09

# Each label's appearance and disappearance fades over this much of the move
# instead of popping in or out on a single frame -- roughly 12 frames of a
# nominal 360-frame (12s at 30fps) intro.
_OPENING_LABEL_RAMP_FRACTION = 12 / 360


def _advance_active_labels(
    active: dict[str, int],
    windows: list[tuple[str, int, int]],
    index: int,
    ranked_candidate_keys: Sequence[str],
    max_labels: int,
) -> None:
    """One step of the never-evict-an-incumbent windowing rule.

    ``ranked_candidate_keys`` is this step's candidates, best first. A key
    already in ``active`` keeps its slot regardless of its rank this step --
    only a key that is no longer a candidate at all is closed out (its
    window appended) -- and a newcomer takes a slot only once one is free.
    Shared by ``_opening_label_windows`` (which walks a whole analytic
    camera path before any frame renders) and ``LegLabelPlanner`` (which
    walks a 3D leg's real per-frame projected labels one frame at a time, as
    they arrive from the capture): both apply the identical rule, only the
    source of a step's ranked candidates differs.
    """
    candidate_set = set(ranked_candidate_keys)
    for key in [key for key in active if key not in candidate_set]:
        windows.append((key, active.pop(key), index - 1))
    free_slots = max_labels - len(active)
    for key in ranked_candidate_keys:
        if key in active or free_slots <= 0:
            continue
        active[key] = index
        free_slots -= 1


class LegLabelPlanner:
    """Per-3D-leg label memory: an incumbent is never bumped by a newcomer
    while it is still a candidate.

    A 3D leg's per-frame labels are only known as ``scene.js`` projects each
    pose in turn during capture (terrain-aware screen coordinates, computed
    in-browser) -- unlike the opening's analytic camera, there is no whole
    leg's sample list available before a single frame renders, so this plan
    is built online, one frame at a time, instead of walked once up front
    like ``_opening_label_windows``. It still applies the exact same
    never-evict rule (``_advance_active_labels``): a label already shown
    keeps its pill's priority in ``place_labels``'s collision greedy even
    when a same-frame newcomer would otherwise have out-ranked or collided
    with it -- the fix for a label popping out for one frame while it is
    still in view.

    Only meaningful for a leg's SERIAL capture (one frame after another, in
    order); the parallel capture path stripes frames across workers, so no
    worker ever sees two consecutive frames and "incumbent" would be
    meaningless there.
    """

    def __init__(self, max_labels: int = 12) -> None:
        self.max_labels = max_labels
        self._active: dict[str, int] = {}
        self._windows: list[tuple[str, int, int]] = []
        self._index = -1

    def step(self, labels: Sequence[dict]) -> list[dict]:
        """Advance one frame; return ``labels`` reordered incumbents-first.

        Reordering (not filtering) is what actually fixes the pop: the
        entries themselves are still whatever ``scene.js`` projected this
        frame, but an incumbent now wins ``place_labels``'s collision greedy
        over a newcomer at the same spot, rather than losing to whichever of
        the two happened to sort first in the raw list.
        """
        self._index += 1
        candidates = [
            label
            for label in labels
            if label.get("visible") and label.get("kind") != "marker" and label.get("name")
        ]
        ranked_keys: list[str] = []
        seen: set[str] = set()
        for label in candidates:
            key = str(label["name"]).casefold()
            if key not in seen:
                seen.add(key)
                ranked_keys.append(key)
        was_active = set(self._active)
        _advance_active_labels(
            self._active, self._windows, self._index, ranked_keys, self.max_labels
        )
        allowed = set(self._active)
        incumbents = [
            label for label in candidates if str(label["name"]).casefold() in was_active & allowed
        ]
        newcomers = [
            label for label in candidates if str(label["name"]).casefold() in allowed - was_active
        ]
        passthrough = [label for label in labels if label not in candidates]
        return [*incumbents, *newcomers, *passthrough]


def _opening_label_candidacy(
    label: MapLabel, west: float, south: float, east: float, north: float, span_lon: float
) -> bool:
    return (
        west <= label.longitude <= east
        and south <= label.latitude <= north
        and label.minimum_visibility_span <= span_lon <= label.visibility_span
    )


@functools.lru_cache(maxsize=4)
def _opening_label_windows(
    rasters: tuple[SatelliteRaster, ...],
    target: RoutePoint,
    width: int,
    height: int,
    labels: tuple[MapLabel, ...],
    max_labels: int = 12,
) -> tuple[tuple[MapLabel, float, float], ...]:
    """Plan the opening's whole label timeline once: ``(label, on, off)`` fractions.

    A label already on screen is never evicted to make room for a
    higher-ranked newcomer -- a newcomer waits for a free slot -- and a
    label whose qualifying run is shorter than ``_OPENING_LABEL_DWELL_FRACTION``
    never appears at all, rather than for one flashed frame. Both rules need
    the whole move's candidacy walked, in order, once; a per-frame re-rank
    cannot see whether a label it would newly select is about to fall below
    the dwell floor a few frames later.
    """
    unique_labels: dict[str, MapLabel] = {}
    for label in labels:
        unique_labels.setdefault(label.name.casefold(), label)

    poses = []
    for index in range(_OPENING_LABEL_SAMPLES):
        fraction = index / (_OPENING_LABEL_SAMPLES - 1)
        center_lat, center_lon, span_lon = _opening_camera(rasters, target, fraction, width)
        span_lat = span_lon * height / width
        poses.append(
            (
                center_lon - span_lon / 2,
                center_lat - span_lat / 2,
                center_lon + span_lon / 2,
                center_lat + span_lat / 2,
                span_lon,
            )
        )

    active: dict[str, int] = {}
    windows: list[tuple[str, int, int]] = []
    for index, (west, south, east, north, span_lon) in enumerate(poses):
        candidates = sorted(
            (
                label
                for label in unique_labels.values()
                if _opening_label_candidacy(label, west, south, east, north, span_lon)
            ),
            key=lambda label: (
                -_LABEL_PRIORITY.get(label.kind, _LABEL_PRIORITY["poi"]),
                abs(label.latitude - (south + north) / 2)
                + abs(label.longitude - (west + east) / 2),
                label.name.casefold(),
            ),
        )
        _advance_active_labels(
            active, windows, index, [label.name.casefold() for label in candidates], max_labels
        )
    last_index = len(poses) - 1
    for key, start in active.items():
        windows.append((key, start, last_index))

    step = 1.0 / (_OPENING_LABEL_SAMPLES - 1)
    dwell_steps = max(1, round(_OPENING_LABEL_DWELL_FRACTION / step))
    plan = []
    for key, start_idx, end_idx in windows:
        if end_idx - start_idx + 1 < dwell_steps:
            continue
        plan.append((unique_labels[key], start_idx * step, end_idx * step))
    return tuple(plan)


def _opening_visible_labels(
    rasters: Sequence[SatelliteRaster],
    target: RoutePoint,
    fraction: float,
    width: int,
    height: int,
    labels: Sequence[MapLabel],
) -> tuple[tuple[MapLabel, float], ...]:
    """This frame's slice of the opening's whole-move label plan: ``(label, alpha)``."""
    if not labels:
        return ()
    windows = _opening_label_windows(tuple(rasters), target, width, height, tuple(labels))
    visible = []
    for label, start, end in windows:
        if not (start <= fraction <= end):
            continue
        ramp = min(_OPENING_LABEL_RAMP_FRACTION, (end - start) / 2)
        alpha = 1.0 if ramp <= 0 else min(1.0, (fraction - start) / ramp, (end - fraction) / ramp)
        visible.append((label, max(0.0, min(1.0, alpha))))
    return tuple(visible)


def _draw_opening_map_labels(
    image: Image.Image,
    visible: Sequence[tuple[MapLabel, float]],
    west: float,
    north: float,
    span_lon: float,
    span_lat: float,
    map_height: int,
) -> None:
    """Draw the opening's planned label set for one frame, each at its own fade alpha.

    Layout (collision avoidance, screen-edge margins) mirrors
    ``_draw_map_labels``; the difference is which labels are on screen
    (``_opening_visible_labels``'s whole-move plan, never a per-frame
    re-rank) and that each draws at a fade alpha instead of always opaque.
    """
    if not visible:
        return
    width, _ = image.size
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(max(13, width // 105), bold=True)
    occupied: list[tuple[float, float, float, float]] = []
    for label, alpha in visible:
        if alpha <= 0:
            continue
        x = (label.longitude - west) / span_lon * width
        y = (north - label.latitude) / span_lat * map_height
        if y < 18 or y > map_height - 64:
            continue
        text_width = draw.textlength(label.name, font=font)
        box = (x + 12, y - font.size - 7, x + text_width + 22, y + 7)
        if any(
            box[0] < other[2] and box[2] > other[0] and box[1] < other[3] and box[3] > other[1]
            for other in occupied
        ):
            continue
        occupied.append(box)
        radius = 5 if label.kind in {"town", "city", "village"} else 4
        marker_alpha = round(255 * alpha)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(0xF5, 0xB7, 0x3B, marker_alpha),
            outline=(0x07, 0x12, 0x19, marker_alpha),
            width=2,
        )
        draw.text(
            (x + 12, y - font.size - 3),
            label.name,
            font=font,
            fill=(255, 255, 255, marker_alpha),
            stroke_width=3,
            stroke_fill=(4, 14, 18, round(210 * alpha)),
        )


def _render_opening_map(
    rasters: Sequence[SatelliteRaster],
    target: RoutePoint,
    fraction: float,
    width: int,
    height: int,
    labels: Sequence[MapLabel] = (),
    terrain: TerrainGrid | None = None,
) -> Image.Image:
    """One geographic camera move from the recognisable country to the trailhead."""
    center_lat, center_lon, span_lon = _opening_camera(rasters, target, fraction, width)
    image = _render_opening_rasters(
        rasters, target, fraction, center_lat, center_lon, span_lon, width, height
    )
    span_lat = span_lon * height / width
    visible = _opening_visible_labels(rasters, target, fraction, width, height, labels)
    _draw_opening_map_labels(
        image,
        visible,
        center_lon - span_lon / 2,
        center_lat + span_lat / 2,
        span_lon,
        span_lat,
        height,
    )
    return image


# The floor a raster's resolution must clear to be trusted to resolve a
# frame without a visible upscale -- 1.0 = no magnification at all. Applied
# to the whole move. Measured against a real rejected film's bundle: even a
# well-formed pyramid's adjacent zoom tiers hand off with SOME slack (that
# bundle's own gibs-to-naip_coarse and naip_coarse-to-naip handoffs measure as low as
# ~0.48 and ~0.20 source px per output px at their worst sampled fraction).
# 0.5 catches a genuinely unresolvable raster set (a single global raster
# covering a trailhead-zoom span measures ~0.000008 here) while giving a
# real multi-tier pyramid's handoffs some room; a raster set that still
# trips this floor is telling the truth -- its zoom tiers are too sparse to
# open on, and the fix is a finer intermediate tier in the bundle, not a
# looser check here.
_OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX = 0.5

# The floor used only to pick the opening's OWN establishing framing
# (``_country_framing``): a one-off, static choice, so held to a stricter
# bar -- this is what tells a whole-world backdrop band apart from a band
# actually captured to show a useful, recognisable slice of the country.
_OPENING_COUNTRY_FRAMING_MIN_RATIO = 1.25

# The widest longitude span that can still read as one country in an
# establishing shot. The contiguous United States is ~59 degrees and Australia
# ~40, so this leaves generous room for a legitimately wide country while
# refusing anything hemispheric: a 180 degree span shows an ocean and two
# continents, and 360 shows the planet. Only ever used to REJECT candidates --
# the framing itself is still whichever surviving raster is widest, never a
# hardcoded box.
_OPENING_MAX_COUNTRY_SPAN_LON = 120.0

# The camera's final, most zoomed-in span -- unchanged from before this
# camera learned to travel: the opening still ends locked exactly here.
_OPENING_TARGET_SPAN_LON = 0.085

# The centre finishes travelling to the target by this fraction of the move,
# well before the last third's steep final descent in scale -- so nothing
# drifts sideways while the camera is diving in fastest.
_OPENING_CENTER_LOCK_FRACTION = 0.62

# The span (zoom) itself starts easing later than the centre, over the whole
# remaining move -- this is what keeps the country-scale establishing shot
# on screen long enough to read before the camera starts diving in.
_OPENING_SPAN_EASE_START = 0.24


def _country_framing(rasters: Sequence[SatelliteRaster], width: int) -> tuple[float, float, float]:
    """Centre and longitude span of the widest raster that still reads sharp.

    A raster's own bounding box, drawn at its own resolution, needs exactly
    ``image.width / width`` source pixels per output pixel -- so "can this
    raster's own bbox fill the frame sharply" reduces to comparing its own
    pixel width against the frame's. A coarse whole-world band (few pixels
    per degree) is wide but fails that test outright; a band captured to
    show a useful slice of the country at real resolution passes it with
    room to spare. Among the rasters that pass, the widest one is the most
    honest "recognisable country" establishing shot this imagery can
    actually support -- never a hardcoded continental box. Falls back to the
    widest raster available if none clears the floor, rather than raising
    here (``check_opening_sharpness`` is the place that raises).
    """
    # Whole-world rasters are excluded before anything is measured: they
    # clear the ratio below easily (the z4 backdrop mosaic is ~4096 px wide)
    # and then win "widest", which is the film that opened on a map with
    # Africa and Australia in frame and only reached the United States
    # three-quarters of the way down.
    #
    # The test is the SPAN, not the band name. Naming the backdrop band was
    # the first attempt and it left the real hole open: the render command
    # also passes --national-raster, which film.py builds with hardcoded
    # -180/180 bounds and no band at all, so `band != BACKDROP_BAND` was
    # true of it and it won anyway (measured: the film and its own review
    # pack both framed (0.0, 0.0, 360)). A name only excludes sources that
    # label themselves; how much of the planet a raster covers is visible
    # whatever it came from, and holds for every imagery source added later.
    #
    # If nothing clears the cap, fall back to the NARROWEST rather than the
    # widest -- picking the widest there would reinstate exactly the framing
    # this just refused.
    framing_candidates = [
        raster
        for raster in rasters
        if raster.band != BACKDROP_BAND
        and raster.east - raster.west <= _OPENING_MAX_COUNTRY_SPAN_LON
    ]
    if not framing_candidates:
        narrowest = min(rasters, key=lambda raster: raster.east - raster.west)
        return (
            (narrowest.south + narrowest.north) / 2,
            (narrowest.west + narrowest.east) / 2,
            narrowest.east - narrowest.west,
        )
    sharp = [
        raster
        for raster in framing_candidates
        if _raster_width(raster.source) / width >= _OPENING_COUNTRY_FRAMING_MIN_RATIO
    ]
    widest = max(sharp or framing_candidates, key=lambda raster: raster.east - raster.west)
    return (
        (widest.south + widest.north) / 2,
        (widest.west + widest.east) / 2,
        widest.east - widest.west,
    )


def _opening_camera(
    rasters: Sequence[SatelliteRaster], target: RoutePoint, fraction: float, width: int
) -> tuple[float, float, float]:
    """Country-to-trailhead camera pose: the centre holds the country framing
    still, then travels, then the camera descends in scale onto the target.

    The centre only starts moving once the span also starts easing in
    (``_OPENING_SPAN_EASE_START``) -- while the span is still exactly the
    country framing's own, the centre must stay exactly the country framing's
    own too, or the viewport immediately drifts outside the very raster
    ``_country_framing`` picked to fill it sharply (that raster's bbox has no
    margin: it IS the country framing). It then eases to the target by
    ``_OPENING_CENTER_LOCK_FRACTION`` and holds there, locked, for the rest
    -- so the last third of the opening (the steepest part of the zoom) has
    no lateral drift at all, only scale.
    """
    country_lat, country_lon, country_span = _country_framing(rasters, width)
    center_progress = _smoothstep(
        (min(fraction, _OPENING_CENTER_LOCK_FRACTION) - _OPENING_SPAN_EASE_START)
        / (_OPENING_CENTER_LOCK_FRACTION - _OPENING_SPAN_EASE_START)
    )
    center_lat = country_lat + (target.latitude - country_lat) * center_progress
    center_lon = country_lon + (target.longitude - country_lon) * center_progress
    span_progress = _smoothstep(
        (fraction - _OPENING_SPAN_EASE_START) / (1.0 - _OPENING_SPAN_EASE_START)
    )
    span_lon = country_span * ((_OPENING_TARGET_SPAN_LON / country_span) ** span_progress)
    return center_lat, center_lon, span_lon


_OPENING_SHARPNESS_CHECK_FRACTIONS: tuple[float, ...] = tuple(round(i * 0.02, 2) for i in range(51))


def check_opening_sharpness(
    rasters: Sequence[SatelliteRaster],
    target: RoutePoint,
    width: int,
    height: int,
    fractions: Sequence[float] = _OPENING_SHARPNESS_CHECK_FRACTIONS,
) -> None:
    """Raise before the long render if the opening move would ever blur.

    Walks the same camera pose and raster selection ``_render_opening_map``
    uses at a representative sample of the move, and raises the moment a
    covering raster would have to be upscaled past
    ``_OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX`` to fill the frame -- once,
    upstream of rendering (and encoding) the whole opening from it, instead
    of a silent per-frame upscale reaching the delivered film. The floor is a sharpness
    cap on this move, not the resolution-label rule (pipeline/quality.py
    HONEST_RESOLUTION_RULE: never claim a native resolution the source lacked).

    The default fraction step is 0.02 (51 samples), not the previous 0.15 --
    a real walk of the whole move at 0.01 across the pre-fix band layout
    found a below-floor window only 0.05 of the move wide (f=0.53-0.58), one
    a 0.15 step could step clean over. The cost is one header read per raster,
    once (``_raster_width``) -- pure arithmetic beyond that, so the density is
    free.
    """
    for fraction in fractions:
        center_lat, center_lon, span_lon = _opening_camera(rasters, target, fraction, width)
        span_lat = span_lon * height / width
        west, east = center_lon - span_lon / 2, center_lon + span_lon / 2
        south, north = center_lat - span_lat / 2, center_lat + span_lat / 2
        raster = _viewport_raster(rasters, west, east, south, north)
        source_px_per_deg = _raster_width(raster.source) / (raster.east - raster.west)
        output_px_per_deg = width / span_lon
        ratio = source_px_per_deg / output_px_per_deg
        if ratio < _OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX:
            raise ValueError(
                f"opening camera at fraction {fraction:.2f} needs a span of "
                f"{span_lon:.4f}° wide, but the covering raster "
                f"{raster.credit!r} ({raster.source.name}) only resolves "
                f"{ratio:.2f} source px per output px "
                f"(floor {_OPENING_MIN_SOURCE_PX_PER_OUTPUT_PX}) -- it would "
                "have to be upscaled to fill the frame"
            )


def _viewport_raster(
    rasters: Sequence[SatelliteRaster], west: float, east: float, south: float, north: float
) -> SatelliteRaster:
    # Select one source that fully covers the viewport.  Partial rectangles
    # are never composited: that was the cause of the visible seams and noisy
    # handoffs at band transitions.  The next camera frame supplies a natural temporal blend.
    covering = [
        raster
        for raster in rasters
        if raster.west <= west
        and raster.east >= east
        and raster.south <= south
        and raster.north >= north
    ]
    return covering[-1] if covering else rasters[0]


# The opening's own graded raster tone-matched against its nearest coarser
# neighbour, computed once per raster (cacheable, deterministic) rather than
# per frame -- matching every other frame drawn from the same raster looking
# identical (like the grade it wraps, ``_graded_raster_image``). On a real
# bundle the ``naip_coarse``
# mosaic itself used to be a patchwork of separately toned NAIP source tiles,
# not one uniform raster -- that tonal seam is now equalised at bundle build
# time (``satellite._mosaic`` solves and feathers a per-channel offset across
# each tile-to-tile seam, before ``_even_latitude`` respaces the rows), so a
# raster handed to this pass is already tone-consistent within itself. What
# this raster-to-raster tone match fixes is the separate, still-visible step
# at a BAND hand-off (vivid green GIBS to grey-blue NAIP): the
# finer raster's mean colour is shifted onto its coarser neighbour's,
# measured over the geographic patch they both cover, so the hand-off reads
# as one map continuing rather than a different layer starting. Its spread
# is left alone: the finer raster's contrast IS its detail.
_OPENING_TONE_MATCH_SAMPLE_PX = 64


def _overlap_bounds(
    a: SatelliteRaster, b: SatelliteRaster
) -> tuple[float, float, float, float] | None:
    west, east = max(a.west, b.west), min(a.east, b.east)
    south, north = max(a.south, b.south), min(a.north, b.north)
    if west >= east or south >= north:
        return None
    return west, south, east, north


def _raster_patch_mean(
    image: Image.Image, raster: SatelliteRaster, bounds: tuple[float, float, float, float]
) -> np.ndarray:
    west, south, east, north = bounds
    x0 = (west - raster.west) / (raster.east - raster.west) * image.width
    x1 = (east - raster.west) / (raster.east - raster.west) * image.width
    y0 = (raster.north - north) / (raster.north - raster.south) * image.height
    y1 = (raster.north - south) / (raster.north - raster.south) * image.height
    patch = image.resize(
        (_OPENING_TONE_MATCH_SAMPLE_PX, _OPENING_TONE_MATCH_SAMPLE_PX),
        resample=Image.Resampling.BOX,
        box=(x0, y0, x1, y1),
    )
    pixels = np.asarray(patch, dtype=np.float64).reshape(-1, 3)
    return pixels.mean(axis=0)


@functools.lru_cache(maxsize=8)
def _tone_matched_opening_image(
    rasters: tuple[SatelliteRaster, ...], raster: SatelliteRaster
) -> Image.Image:
    """``raster``'s graded opening image, colour-matched to its nearest coarser
    neighbour over the patch they both cover -- see the module comment above
    ``_OPENING_TONE_MATCH_SAMPLE_PX``. Falls back to the plain graded image
    when ``raster`` has no coarser neighbour, or none that overlaps it."""
    graded = _graded_raster_image(raster.source)
    own_span = raster.east - raster.west
    coarser = min(
        (other for other in rasters if other is not raster and other.east - other.west > own_span),
        key=lambda other: other.east - other.west,
        default=None,
    )
    if coarser is None:
        return graded
    bounds = _overlap_bounds(raster, coarser)
    if bounds is None:
        return graded
    fine_mean = _raster_patch_mean(graded, raster, bounds)
    coarse_mean = _raster_patch_mean(_graded_raster_image(coarser.source), coarser, bounds)
    pixels = np.asarray(graded, dtype=np.float64)
    # Colour only: scaling contrast to the blurrier neighbour's spread would
    # flatten the very detail the finer raster is on screen for (one real
    # opening ladder dropped to 0.29 of its median at 0.67 of the move).
    matched = pixels - fine_mean + coarse_mean
    return Image.fromarray(np.clip(matched, 0, 255).astype(np.uint8), "RGB")


# The opening camera's pose is a pure function of ``fraction`` (see
# ``_opening_camera``), so exactly where the covering raster hands off is
# already determined before a single frame renders. Walked once per move, at
# this fixed grid (same idea and density as ``_opening_label_windows``).
_OPENING_RASTER_TRANSITION_SAMPLES = 241

# How much of the move, right after a hand-off, still blends the outgoing
# (coarser) raster in with the incoming one instead of cutting to it on one
# frame -- roughly 5-6 frames of a nominal 360-frame (12s at 30fps) intro.
_OPENING_RASTER_CROSSFADE_FRACTION = 0.015


@functools.lru_cache(maxsize=4)
def _opening_raster_transitions(
    rasters: tuple[SatelliteRaster, ...], target: RoutePoint, width: int, height: int
) -> tuple[float, ...]:
    """Fractions, once per move, where the opening's covering raster hands off."""
    transitions = []
    previous_source = None
    for index in range(_OPENING_RASTER_TRANSITION_SAMPLES):
        fraction = index / (_OPENING_RASTER_TRANSITION_SAMPLES - 1)
        center_lat, center_lon, span_lon = _opening_camera(rasters, target, fraction, width)
        span_lat = span_lon * height / width
        west, east = center_lon - span_lon / 2, center_lon + span_lon / 2
        south, north = center_lat - span_lat / 2, center_lat + span_lat / 2
        primary = _viewport_raster(rasters, west, east, south, north)
        if previous_source is not None and primary.source != previous_source:
            transitions.append(fraction)
        previous_source = primary.source
    return tuple(transitions)


def _opening_raster_crop(
    image: Image.Image,
    raster: SatelliteRaster,
    west: float,
    east: float,
    south: float,
    north: float,
    width: int,
    height: int,
) -> Image.Image:
    x0 = (west - raster.west) / (raster.east - raster.west) * image.width
    x1 = (east - raster.west) / (raster.east - raster.west) * image.width
    y0 = (raster.north - north) / (raster.north - raster.south) * image.height
    y1 = (raster.north - south) / (raster.north - raster.south) * image.height
    return image.transform(
        (width, height),
        Image.Transform.QUAD,
        (x0, y0, x0, y1, x1, y1, x1, y0),
        resample=Image.Resampling.BICUBIC,
    )


def finest_last(rasters: Sequence[SatelliteRaster]) -> tuple[SatelliteRaster, ...]:
    """Order rasters coarse to fine, so the covering one chosen is the sharpest."""

    def degrees_per_pixel(raster: SatelliteRaster) -> float:
        return (raster.east - raster.west) / _raster_width(raster.source)

    return tuple(sorted(rasters, key=degrees_per_pixel, reverse=True))


def opening_credit(
    rasters: Sequence[SatelliteRaster],
    target: RoutePoint,
    fraction: float,
    width: int,
    height: int,
) -> str:
    """The credit of the raster an opening frame is drawn from."""
    center_lat, center_lon, span_lon = _opening_camera(rasters, target, fraction, width)
    span_lat = span_lon * height / width
    return _viewport_raster(
        rasters,
        center_lon - span_lon / 2,
        center_lon + span_lon / 2,
        center_lat - span_lat / 2,
        center_lat + span_lat / 2,
    ).credit


def _render_opening_rasters(
    rasters: Sequence[SatelliteRaster],
    target: RoutePoint,
    fraction: float,
    center_lat: float,
    center_lon: float,
    span_lon: float,
    width: int,
    height: int,
) -> Image.Image:
    """Composite one GRADED, tone-matched raster crop for the opening camera.

    A copy of ``_render_geographic_rasters``'s projection, scoped to the
    opening so its once-per-raster grade (``_graded_raster_image``) never
    touches the shared following-map renderer every other leg uses. Unlike
    that shared renderer, a hand-off between two covering rasters here
    cross-fades over ``_OPENING_RASTER_CROSSFADE_FRACTION`` of the move
    instead of cutting on a single frame (see
    ``_opening_raster_transitions``/``_tone_matched_opening_image``); a
    partial rectangle is still never composited -- only two FULLY covering
    rasters ever blend, and only right around the moment the finer one
    starts winning the pick.
    """
    span_lat = span_lon * height / width
    west, east = center_lon - span_lon / 2, center_lon + span_lon / 2
    south, north = center_lat - span_lat / 2, center_lat + span_lat / 2
    ordered = tuple(rasters)
    primary = _viewport_raster(ordered, west, east, south, north)
    primary_image = _opening_raster_crop(
        _tone_matched_opening_image(ordered, primary),
        primary,
        west,
        east,
        south,
        north,
        width,
        height,
    )
    covering = [
        raster
        for raster in ordered
        if raster.west <= west
        and raster.east >= east
        and raster.south <= south
        and raster.north >= north
    ]
    if len(covering) < 2:
        return primary_image
    transitions = _opening_raster_transitions(ordered, target, width, height)
    onset = max((t for t in transitions if t <= fraction), default=None)
    if onset is None:
        return primary_image
    distance = fraction - onset
    if distance >= _OPENING_RASTER_CROSSFADE_FRACTION:
        return primary_image
    weight = _smoothstep(distance / _OPENING_RASTER_CROSSFADE_FRACTION)
    runner_up = covering[-2]
    runner_up_image = _opening_raster_crop(
        _tone_matched_opening_image(ordered, runner_up),
        runner_up,
        west,
        east,
        south,
        north,
        width,
        height,
    )
    return Image.blend(runner_up_image, primary_image, weight)


def _render_geographic_rasters(
    rasters: Sequence[SatelliteRaster],
    center_lat: float,
    center_lon: float,
    span_lon: float,
    width: int,
    height: int,
) -> Image.Image:
    """Composite registered, offline rasters into a single continuous camera view."""
    span_lat = span_lon * height / width
    west, east = center_lon - span_lon / 2, center_lon + span_lon / 2
    south, north = center_lat - span_lat / 2, center_lat + span_lat / 2
    raster = _viewport_raster(rasters, west, east, south, north)
    source = _raster_image(raster.source)
    x0 = (west - raster.west) / (raster.east - raster.west) * source.width
    x1 = (east - raster.west) / (raster.east - raster.west) * source.width
    y0 = (raster.north - north) / (raster.north - raster.south) * source.height
    y1 = (raster.north - south) / (raster.north - raster.south) * source.height
    # Pillow's QUAD takes the source corners as NW, SW, SE, NE. Listing them
    # clockwise (NW, NE, SE, SW) transposes the picture: east lands at the
    # bottom and south on the right, under labels that are still north-up.
    return source.transform(
        (width, height),
        Image.Transform.QUAD,
        (x0, y0, x0, y1, x1, y1, x1, y0),
        resample=Image.Resampling.BICUBIC,
    )


OVERVIEW_MARGIN = 0.12
OVERVIEW_MIN_MARGIN = 0.03


def overview_viewport(
    segments: Sequence[Sequence[RoutePoint]],
    rasters: Sequence[SatelliteRaster],
    width: int,
    map_height: int,
) -> tuple[float, float, float]:
    """Centre and longitude span of a view that holds the whole route.

    A following map leg centres on the marker, so at the end of the day it
    shows the last corner of the route -- and, near the edge of the finest
    raster, a viewport that raster no longer fully covers, which falls back to
    a coarser level and draws the recap from blocky upscaled pixels. The recap
    instead fits the route's bounding box, and gives up margin (down to
    ``OVERVIEW_MIN_MARGIN``) before it gives up the finest raster: when the
    route fits inside it, the view is slid to lie wholly inside it too.
    """
    points = [point for segment in segments for point in segment]
    if not points:
        raise ValueError("an overview needs at least one route point")
    west = min(point.longitude for point in points)
    east = max(point.longitude for point in points)
    south = min(point.latitude for point in points)
    north = max(point.latitude for point in points)
    aspect = map_height / width

    def span_for(margin: float) -> float:
        grow = 1.0 + 2.0 * margin
        return max(0.01, (east - west) * grow, (north - south) * grow / aspect)

    center_lat, center_lon = (south + north) / 2, (west + east) / 2
    span_lon = span_for(OVERVIEW_MARGIN)
    fine = rasters[-1]
    fits = min(fine.east - fine.west, (fine.north - fine.south) / aspect)
    if span_for(OVERVIEW_MIN_MARGIN) <= fits:
        # Shave a hair off so float noise on an exact fit cannot lose coverage.
        span_lon = min(span_lon, fits * 0.999)
        half_lon, half_lat = span_lon / 2, span_lon * aspect / 2
        center_lon = min(max(center_lon, fine.west + half_lon), fine.east - half_lon)
        center_lat = min(max(center_lat, fine.south + half_lat), fine.north - half_lat)
    return center_lat, center_lon, span_lon


def _draw_map_pins(
    draw: ImageDraw.ImageDraw,
    pins: Sequence[tuple[float, float]],
    west: float,
    north: float,
    span_lon: float,
    span_lat: float,
    width: int,
    map_height: int,
    height: int,
) -> None:
    """White, dark-outlined pins marking where a played photo/video beat is.

    Deliberately far in colour from both the travelled-line yellow
    (``frame_qa.TRAVELLED_LINE_RGB``) and the marker's own fill, so a pin
    beside the marker never reads as either to frame QA's colour detectors.
    """
    if not pins:
        return
    radius = max(3, round(height * 6 / 1080))
    for longitude, latitude in pins:
        x = (longitude - west) / span_lon * width
        y = (north - latitude) / span_lat * map_height
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="#ffffff",
            outline="#071219",
            width=2,
        )


# The day's start and finish, pinned for the whole film. Both sit far in
# colour from the travelled line (``frame_qa.TRAVELLED_LINE_RGB``), from the
# marker's amber and from the white photo pin, so none of frame QA's colour
# detectors can count one as another -- a pin lying on the route in a
# near-enough yellow would be measured as line width, and the gate built to
# catch a hairline route would pass on the pin instead.
ROUTE_START_RGB: tuple[int, int, int] = (0x35, 0xD0, 0x7A)
ROUTE_FINISH_RGB: tuple[int, int, int] = (0xE2, 0x4C, 0x6F)


# Every route stroke is a fraction of the frame's width, so a larger master
# gets a proportionally heavier line instead of the 1080p one stretched thin.
# The divisors are named because ``scene.js`` -- the production renderer --
# derives its own widths from these same numbers, and
# ``tests/test_route_story_line_width_parity.py`` reads them out of this module
# to build the JavaScript it expects to find. Changing one here without the
# other turns that test red rather than shipping two different-looking films.
# The values are the ones 1080p already shipped (8 / 3 / 6 at 1920), so this
# scaling does not restyle any film already approved at that size.
ROUTE_HALO_WIDTH_DIVISOR = 220
ROUTE_LINE_WIDTH_DIVISOR = 500
TRAVELLED_LINE_WIDTH_DIVISOR = 280


def route_halo_width_px(width: int) -> int:
    """The dark backing stroke, always wider than the line it sits behind."""
    return max(5, width // ROUTE_HALO_WIDTH_DIVISOR)


def route_line_width_px(width: int) -> int:
    """The pale not-yet-travelled route."""
    return max(2, width // ROUTE_LINE_WIDTH_DIVISOR)


def travelled_line_width_px(width: int) -> int:
    """The bright travelled route.

    Its floor is the highest of the three: frame QA measures this stroke and
    refuses a film whose travelled line goes hairline, so a small frame must
    not scale it below what the gate will accept.
    """
    return max(4, width // TRAVELLED_LINE_WIDTH_DIVISOR)


def route_endpoints(
    segments: Sequence[Sequence[RoutePoint]],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """``((lon, lat) start, (lon, lat) finish)`` from the track, or ``None``.

    The finish is the last fix of the LAST segment, not of the first: a
    recorder pause splits the day into several segments and the day does not
    end at the first pause. A one-point track is degenerate but not an error
    -- it started and ended in the same place, and the map says so.
    """
    recorded = [segment for segment in segments if segment]
    if not recorded:
        return None
    first, last = recorded[0][0], recorded[-1][-1]
    return (first.longitude, first.latitude), (last.longitude, last.latitude)


def _draw_endpoint_pins(
    draw: ImageDraw.ImageDraw,
    endpoints: tuple[tuple[float, float], tuple[float, float]] | None,
    west: float,
    north: float,
    span_lon: float,
    span_lat: float,
    width: int,
    map_height: int,
    height: int,
) -> None:
    """Fixed start and finish pins, present on every map frame.

    Unlike photo pins these never accumulate and never wait for a beat: the
    untravelled route line already runs to the finish on frame one, so
    withholding its pin would only make the map disagree with its own line.
    """
    if endpoints is None:
        return
    radius = max(4, round(height * 8 / 1080))
    for (longitude, latitude), colour in zip(endpoints, (ROUTE_START_RGB, ROUTE_FINISH_RGB)):
        x = (longitude - west) / span_lon * width
        y = (north - latitude) / span_lat * map_height
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=colour,
            outline="#071219",
            width=2,
        )


def _render_map(
    rasters: Sequence[SatelliteRaster],
    segments: Sequence[Sequence[RoutePoint]],
    at_time: float,
    width: int,
    height: int,
    *,
    show_dashboard: bool = True,
    show_gap_badge: bool = True,
    labels: Sequence[MapLabel] = (),
    terrain: TerrainGrid | None = None,
    overview: bool = False,
    measured_segments: Sequence[Sequence[RoutePoint]] | None = None,
    pins: Sequence[PhotoPinBeat] = (),
) -> Image.Image:
    timeline = StoryTimeline(fps=1, scenes=[Scene("overview", 0, 1, at_time, at_time)])
    # One frame_state_at call resolves both the drawn marker and the
    # dashboard beneath it (dashboard_state_at below makes the same call on
    # the same inputs), so the two can never independently interpolate to
    # two different positions for the same instant.
    frame_state = frame_state_at(timeline, 0, measured_segments or segments)
    in_gap = not frame_state.data_available
    if frame_state.data_available:
        marker = frame_state
    else:
        marker = min(
            (point for segment in segments for point in segment),
            key=lambda point: abs(point.time_s - at_time),
        )
    map_height = int(height * 0.855)
    # Keep the marker in the available map field, away from the lower stats rail.
    center_lat = marker.latitude - 0.0025
    center_lon = marker.longitude
    span_lon = max(0.075, (rasters[-1].east - rasters[-1].west) * 0.42)
    if overview:
        center_lat, center_lon, span_lon = overview_viewport(segments, rasters, width, map_height)
    span_lat = span_lon * map_height / width
    west, north = center_lon - span_lon / 2, center_lat + span_lat / 2
    south = north - span_lat
    image = _render_geographic_rasters(rasters, center_lat, center_lon, span_lon, width, map_height)
    if terrain is not None:
        image = Image.alpha_composite(
            image.convert("RGBA"),
            terrain.overlay(west, south, center_lon + span_lon / 2, north, width, map_height),
        ).convert("RGB")
    canvas = Image.new("RGB", (width, height), "#071219")
    canvas.paste(image, (0, 0))
    overlay = Image.new("RGBA", (width, map_height), (3, 12, 17, 58))
    canvas.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)

    def xy(point: RoutePoint) -> tuple[float, float]:
        return (
            (point.longitude - west) / span_lon * width,
            (north - point.latitude) / span_lat * map_height,
        )

    for segment in segments:
        coords = [xy(point) for point in segment]
        if len(coords) > 1:
            draw.line(coords, fill="#071219", width=route_halo_width_px(width), joint="curve")
            draw.line(coords, fill="#a2b7be", width=route_line_width_px(width), joint="curve")
            completed = [point for point in segment if point.time_s <= at_time]
            if len(completed) > 1:
                draw.line(
                    [xy(point) for point in completed],
                    fill="#f5b73b",
                    width=travelled_line_width_px(width),
                    joint="curve",
                )

    _draw_map_pins(
        draw,
        visible_pins(pins, at_time),
        west,
        north,
        span_lon,
        span_lat,
        width,
        map_height,
        height,
    )
    # After the photo pins and before the marker: the endpoints outrank a
    # coincident photo pin, and the marker outranks everything.
    _draw_endpoint_pins(
        draw,
        route_endpoints(measured_segments or segments),
        west,
        north,
        span_lon,
        span_lat,
        width,
        map_height,
        height,
    )

    # The line drawn may be display-smoothed; the numbers never are.
    dashboard = dashboard_state_at(timeline, 0, measured_segments or segments, moving_speed_mps=0.5)
    # Frozen from what these segments actually carry -- never re-decided per
    # frame -- so a mid-film sensor dropout can't remove or reflow a column.
    dashboard_fields = dashboard_fields_for(measured_segments or segments)
    x, y = xy(marker)
    radius = max(7, width // 125)
    draw.ellipse(
        (x - radius - 3, y - radius - 3, x + radius + 3, y + radius + 3),
        fill="#182329",
        outline="#fff",
        width=2,
    )
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="#f5b73b")
    if show_dashboard:
        _draw_dashboard(canvas, dashboard, rasters[-1].credit, fields=dashboard_fields)
    if in_gap and show_gap_badge:
        _draw_gap_badge(canvas, map_height)
    _draw_map_labels(canvas, labels, west, north, span_lon, span_lat, map_height)
    _draw_map_orientation(canvas, center_lat, span_lon, map_height)
    return canvas


class MapRenderer(Protocol):
    """One delivery frame of the map leg at a route clock time.

    A ``MapRenderer`` owns the entire map view — geography, route line,
    marker, dashboard, labels, and terrain — for a single call.  Swapping the
    implementation swaps what a map-leg frame looks like without film.py or
    assembly.py knowing which renderer produced it.
    """

    name: str

    def render(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        at_time: float,
        width: int,
        height: int,
    ) -> Image.Image: ...

    def render_leg(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        times: Sequence[float],
        width: int,
        height: int,
    ) -> Iterator[Image.Image]:
        """Render a whole map leg's frames, in order.

        This exists (rather than ``assembly._append_map_leg`` calling
        ``render`` once per frame) because a renderer whose camera has real
        inertia — a bearing that cannot snap between frames — needs every
        frame's route-clock time up front to plan a leg-wide motion within
        its own bounds; a stateless per-frame ``render`` call cannot see far
        enough ahead to do that. ``PillowMapRenderer``'s camera has no
        inertia, so its implementation is exactly a per-frame ``render``
        loop with byte-identical output.
        """
        ...

    def measuring(self, segments: Sequence[Sequence[RoutePoint]]) -> "MapRenderer":
        """This renderer, measuring its dashboard over ``segments``.

        The line it draws may be a display-smoothed copy; the numbers on the
        rail must come from the recorded points, and the rail is translucent,
        so only the renderer itself can get them right.
        """
        ...


@dataclass(frozen=True)
class PillowMapRenderer:
    """The existing Pillow-composited 2D satellite map renderer."""

    rasters: Sequence[SatelliteRaster]
    labels: Sequence[MapLabel] = ()
    terrain: TerrainGrid | None = None
    name: str = "pillow-2d"
    # The recorded points the dashboard measures when the drawn line is a
    # display-smoothed copy of them; None measures what is drawn.
    measured_segments: Sequence[Sequence[RoutePoint]] | None = None
    pins: Sequence[PhotoPinBeat] = ()

    def measuring(self, segments: Sequence[Sequence[RoutePoint]]) -> "PillowMapRenderer":
        return replace(self, measured_segments=segments)

    def render(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        at_time: float,
        width: int,
        height: int,
    ) -> Image.Image:
        return _render_map(
            self.rasters,
            segments,
            at_time,
            width,
            height,
            labels=self.labels,
            terrain=self.terrain,
            measured_segments=self.measured_segments,
            pins=self.pins,
        )

    def render_leg(
        self,
        segments: Sequence[Sequence[RoutePoint]],
        times: Sequence[float],
        width: int,
        height: int,
    ) -> Iterator[Image.Image]:
        for at_time in times:
            yield self.render(segments, at_time, width, height)


def _point_at_time(segments: Sequence[Sequence[RoutePoint]], at_time: float) -> RoutePoint | None:
    for segment in segments:
        if not segment or at_time < segment[0].time_s or at_time > segment[-1].time_s:
            continue
        for first, second in zip(segment, segment[1:]):
            if first.time_s <= at_time <= second.time_s:
                fraction = (at_time - first.time_s) / (second.time_s - first.time_s)
                return RoutePoint(
                    at_time,
                    first.latitude + (second.latitude - first.latitude) * fraction,
                    first.longitude + (second.longitude - first.longitude) * fraction,
                    _interpolate(first.elevation_m, second.elevation_m, fraction),
                )
    return None


def _draw_gap_badge(image: Image.Image, map_height: int) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(max(12, image.width // 90), bold=True)
    text = "GPS GAP • POSITION HELD"
    box = (
        image.width * 0.40,
        map_height * 0.08,
        image.width * 0.60,
        map_height * 0.08 + font.size + 18,
    )
    draw.rounded_rectangle(
        box, radius=9, fill=(4, 14, 18, 215), outline=(245, 183, 59, 230), width=2
    )
    draw.text((image.width * 0.50, box[1] + 9), text, font=font, fill="#f5d48b", anchor="ma")


def _draw_map_orientation(
    image: Image.Image, center_lat: float, span_lon: float, map_height: int
) -> None:
    """Draw scale and north cues in reserved map space."""
    draw = ImageDraw.Draw(image, "RGBA")
    width, _ = image.size
    miles_wide = max(0.01, span_lon * 69.172 * np.cos(np.deg2rad(center_lat)))
    choices = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0)
    miles = max(choice for choice in choices if choice <= miles_wide * 0.22)
    bar_width = max(50, round(width * miles / miles_wide))
    x0, y0 = int(width * 0.045), map_height - 42
    draw.rounded_rectangle(
        (x0 - 10, y0 - 22, x0 + bar_width + 16, y0 + 14),
        radius=8,
        fill=(4, 14, 18, 185),
    )
    draw.line((x0, y0, x0 + bar_width, y0), fill="white", width=4)
    draw.line((x0, y0 - 5, x0, y0 + 5), fill="white", width=2)
    draw.line((x0 + bar_width, y0 - 5, x0 + bar_width, y0 + 5), fill="white", width=2)
    font = _font(max(11, width // 120), bold=True)
    draw.text((x0, y0 - 20), f"{miles:g} mi", font=font, fill="white")
    arrow_x, arrow_y = width - int(width * 0.06), 34
    draw.line((arrow_x, arrow_y + 24, arrow_x, arrow_y), fill="white", width=3)
    draw.polygon(
        ((arrow_x, arrow_y - 5), (arrow_x - 7, arrow_y + 8), (arrow_x + 7, arrow_y + 8)),
        fill="white",
    )
    draw.text((arrow_x, arrow_y + 36), "N", font=font, fill="white", anchor="mm")


def _draw_map_labels(
    image: Image.Image,
    labels: Sequence[MapLabel],
    west: float,
    north: float,
    span_lon: float,
    span_lat: float,
    map_height: int,
) -> None:
    """Draw readable public names with a small collision-avoiding layout."""
    if not labels:
        return
    width, _ = image.size
    east, south = west + span_lon, north - span_lat
    visible = select_visible_labels(
        labels,
        west=west,
        south=south,
        east=east,
        north=north,
        span_lon=span_lon,
    )
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(max(13, width // 105), bold=True)
    occupied: list[tuple[float, float, float, float]] = []
    for label in visible:
        x = (label.longitude - west) / span_lon * width
        y = (north - label.latitude) / span_lat * map_height
        if y < 18 or y > map_height - 64:
            continue
        text_width = draw.textlength(label.name, font=font)
        box = (x + 12, y - font.size - 7, x + text_width + 22, y + 7)
        if any(
            box[0] < other[2] and box[2] > other[0] and box[1] < other[3] and box[3] > other[1]
            for other in occupied
        ):
            continue
        occupied.append(box)
        radius = 5 if label.kind in {"town", "city", "village"} else 4
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="#f5b73b",
            outline="#071219",
            width=2,
        )
        draw.text(
            (x + 12, y - font.size - 3),
            label.name,
            font=font,
            fill="white",
            stroke_width=3,
            stroke_fill=(4, 14, 18, 210),
        )
