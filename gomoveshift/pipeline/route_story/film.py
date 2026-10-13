"""A local, evidence-led Route Story film renderer.

The renderer is deliberately tile-free: it works from credited raster assets
and the original media files already admitted to a private Route Story.  It
does not infer a position for an unanchored item or synthesize visual detail.

This module owns the film-level contract — what a moment is, the export
receipt, the display-safety filter, the top-level render loop that sequences
scenes, and the CLI entry point. Map projection lives in ``layers``, overlay
drawing and unit formatting live in ``hud``, and frame-to-file assembly lives
in ``assembly``; this module imports from all three rather than owning their
mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image

from ...paths import gms_cache_home
from ..drive_phases.thumbnail_generator import generate_lf_thumbnail
from ..encoding_constants import RENDITIONS
from ..photos.exif_gps import (
    extract_exif_capture_metadata,
    extract_exif_capture_timestamp,
    extract_exif_gps,
)
from ..validation.ffprobe_client import FFprobeClient
from .activity import MAPPING_VERSION, resolve_activity, resolve_units, track_centroid
from .anchors import MediaAnchor, MediaEvidence, anchor_media_to_segments, captured_at_route_seconds
from .assembly import (
    MAP_SETTLE_SECONDS,
    VIDEO_ARRIVAL_HOLD_SECONDS,
    FilmTimeline,
    _append_hold,
    _append_map_leg,
    _append_photo,
    _concat,
    _encode_frames,
    _encode_source_video,
    _interlude_card,  # noqa: F401 -- re-exported for tests/test_route_story_interlude.py
    _save_frame,
    _video_evidence,
    map_leg_seconds,
    plan_map_legs,
)
from .beats import beats_receipt, extract_beats
from .bundle import _sha256_file
from .continuity import analyze_rendered_continuity
from .delivery import HEVC_PIXEL_FORMAT, check_delivery
from .display_track import SmoothedTrack, smooth_display_segments
from .encode import encode_final
from .frame_qa import FrameQaError, FrameQaLedger, save_refusal_evidence
from .gpx import select_gpx_day
from .hud import (
    DEFAULT_WATERMARK_TEXT,
    STILL_SOURCES,
    DisplayCaps,
    _draw_media_arrival,
    _draw_opening,
    _draw_recap,
    _offset_from_capture,
    crop_black_border,
    dashboard_receipt,
    draw_opening_title,
    high_point_callout,
    motion_cell_mode,
    parse_utc_offset,
    resolve_dashboard_utc_offset,
    resolve_opening_title,
    set_active_dashboard_clock,
    set_active_display_caps,
    set_active_display_units,
    set_active_motion_cell_mode,
)
from .layers import (
    MapLabel,
    MapRenderer,
    PhotoPinBeat,
    PillowMapRenderer,
    _render_map,
    _render_opening_map,
    check_opening_sharpness,
    finest_last,
    load_map_labels,
    opening_credit,
    select_visible_labels,
)
from .metrics import RoutePoint, distance_m, summarize_track
from .planner import hectic_windows, max_shot_zoom, motion_report_from_poses, motion_summary
from .preflight import run_preflight
from .presets import PRESETS, FilmTooLongError, apply_preset, capped_still_count, check_ceiling
from .privacy import apply_privacy_gate
from .public_receipt import public_receipt
from .review_pack import (
    MARKER_SHEET_NAME,
    Runner,
    gather_review_inputs,
    write_motion_clips,
    write_review_pack,
)
from .route_privacy import (
    PrivacyZone,
    RouteTrim,
    RouteTrimError,
    mask_zones,
    receipt_fragment,
    trim_route,
)
from .satellite import SatelliteRaster, bundle_opening_rasters, opening_pyramid_receipt
from .select import SelectionOptions
from .still_selection import (
    PHOTO_SECONDS_FLOOR,
    STILLS_SHARE_CAP,
    max_stills_for_cap,
    resolve_photo_seconds,
    select_kept_photos,
    solve_map_seconds,
)
from .terrain import TerrainGrid

logger = logging.getLogger(__name__)

#: Valid --pacing-mode values. "recorded" is the only implemented pacing
#: table today -- a real, disclosed default that establishes the
#: versioned-in-the-receipt contract before a second value is added.
PACING_MODE_CHOICES = ["recorded"]

#: Bump whenever the presentation-timing table behind --pacing-mode changes
#: (independent of .activity.MAPPING_VERSION, which versions the unrelated
#: activity/region/units tables) -- so a re-render under a different pacing
#: table is distinguishable on the receipt from a re-render under a
#: different activity/units table.
PRESENTATION_MAPPING_VERSION = "2026-09-28.1"

_MAPPED_STATUSES = {"embedded_coordinates", "timestamp_interpolated"}

# render_route_story_film's own intro/recap scene lengths -- named here so
# main()'s pacing cap uses the SAME values the render call gets, rather than
# a second copy of the numbers that could silently drift from them.
DEFAULT_INTRO_SECONDS = 12.0
DEFAULT_RECAP_SECONDS = 6.0

# Where along the opening zoom the preflight samples the imagery ladder.
# Five evenly spaced fractions can straddle a soft band completely: on a
# rejected film the two soft rungs sat near f=0.50 and f=0.67, and a
# 0/0.25/0.5/0.75/1.0 walk sampled neither handover cleanly. Denser toward
# the end, where the ladder hands over between rasters most often and where
# the "fuzzy when zooming in" complaint lives. These are Pillow frames, so
# the extra samples cost a preflight second, not render time.
OPENING_LADDER_FRACTIONS = (0.0, 0.2, 0.35, 0.5, 0.6, 0.67, 0.75, 0.82, 0.88, 0.94, 1.0)

# The display_track.smooth_display_segments() knobs the drawn line renders
# with -- named here (not re-typed at each call site) so the receipt's
# display_smoothing block and the actual smoothing call can never disagree.
DISPLAY_SMOOTHING_WINDOW_M = 25.0
DISPLAY_SMOOTHING_MAX_SHIFT_M = 12.0

# A same-day photo shot within this many seconds of the recording -- before
# it starts or after it ends -- is a "bookend": it plays with no claimed
# route position instead of being refused or silently dropped. Further out
# than this is not the same outing and stays ordinary unplaced media.
BOOKEND_MAX_OUTSIDE_S = 6 * 3600.0

# A bookend's still card never uses UNPLACED_PHOTO_CAPTION ("...not placed on
# the route map") -- on a deliberate bookend that reads as an error to a
# viewer. No location words, no pin, just what it is.
BOOKEND_BEFORE_CAPTION = "Before the trail"
BOOKEND_AFTER_CAPTION = "After the trail"


def _staging_base() -> Path:
    """The pure path resolution behind ``_staging_root`` -- no filesystem write.

    Reuses the same ``GMS_INTERMEDIATE_DIR`` override and non-tmp default the
    rest of the service resolves staging through (``paths.gms_intermediate_dir``);
    a film's working frames have no ``output_base``/``project_name`` of their
    own, so this stays a plain override-or-default rather than a second copy
    of that resolver's drvfs logic.
    """
    override = os.environ.get("GMS_INTERMEDIATE_DIR", "")
    base = Path(override) if override else gms_cache_home()
    return base / "route-story-film"


def _staging_root() -> Path:
    """Where Route Story film scratch data lives -- never ``/tmp`` (service contract)."""
    root = _staging_base()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _kept_frames_root(output: Path) -> Path:
    """Where one output's composited frames live while a render runs, and
    where they stay if that render is QA-refused.

    Unlike the deleted ``TemporaryDirectory`` this used to be, a QA refusal
    (mid-capture or the end-of-render ledger, see ``FrameQaLedger``) leaves
    this directory on disk instead of taking every rendered frame down with
    it -- so a reviewer, or a future ``--repair`` run, has something to look
    at besides the refusal evidence. Bounded like the frame cache: one
    retained directory per output, keyed on its resolved path, so repeated
    refusals for the same film never grow the state dir past one render's
    worth of frames. A fresh render for the same output clears whatever a
    prior refusal left here before it starts, and a render that finishes
    without a QA fault clears it again on the way out.
    """
    output = Path(output)
    digest = hashlib.sha256(str(output.resolve()).encode("utf-8")).hexdigest()[:16]
    return _staging_root() / "kept-frames" / f"{output.stem}-{digest}"


__all__ = [
    "BOOKEND_AFTER_CAPTION",
    "BOOKEND_BEFORE_CAPTION",
    "BOOKEND_MAX_OUTSIDE_S",
    "MapLabel",
    "PhotoMoment",
    "classify_bookend",
    "clean_display_segments",
    "crop_black_border",
    "drawn_segments",
    "film_receipt",
    "label_visibility_intervals",
    "load_map_labels",
    "main",
    "map_leg_seconds",
    "render_route_story_film",
    "repair_route_story_film",
    "route_end_time",
    "select_visible_labels",
]


@dataclass(frozen=True)
class _AnchoredMediaView:
    """Adapts ``MediaAnchor``'s field names to ``beats.AnchoredMediaLike``."""

    route_time_s: float | None
    lat: float | None
    lon: float | None


@dataclass(frozen=True)
class PhotoMoment:
    """One original still and its audited location decision."""

    source: Path
    anchor: MediaAnchor
    # The EXIF-derived route-clock capture time (``captured_at_route_seconds``
    # applied to the shutter's own timestamp) -- distinct from
    # ``anchor.track_time_s``, which is where the nearest GPX track point is,
    # not necessarily when the shutter fired. Optional so every existing
    # constructor call keeps working; best-pick selection requires it.
    captured_at_exif_s: float | None = None

    def __post_init__(self) -> None:
        source = Path(self.source)
        if source.is_symlink() or not source.is_file():
            raise ValueError("Route Story photo must be a regular source file")

    @property
    def is_mapped(self) -> bool:
        return self.anchor.status in _MAPPED_STATUSES and self.anchor.track_time_s is not None


def classify_bookend(photo: PhotoMoment, track_start_s: float, track_end_s: float) -> str | None:
    """ "before_recording" / "after_recording" when ``photo`` is a bookend, else ``None``.

    A bookend is a same-day photo the anchor left unanchored for being
    outside the observed track window (``unanchored_outside_track``) -- never
    a photo unanchored for some OTHER reason (a gap inside the recording, a
    coordinate conflict) -- with a known EXIF capture time no more than
    ``BOOKEND_MAX_OUTSIDE_S`` before ``track_start_s`` or after ``track_end_s``.
    Pure: the single rule every bookend site (refusal, sequencing, selection,
    receipt) shares, so it can never drift into a second copy.
    """
    if photo.anchor.status != "unanchored_outside_track":
        return None
    captured_at = photo.captured_at_exif_s
    if captured_at is None:
        return None
    if captured_at < track_start_s:
        return "before_recording" if track_start_s - captured_at <= BOOKEND_MAX_OUTSIDE_S else None
    if captured_at > track_end_s:
        return "after_recording" if captured_at - track_end_s <= BOOKEND_MAX_OUTSIDE_S else None
    return None


@dataclass(frozen=True)
class _MediaBeat:
    """One placed, on-route media event -- a photo hold or the located video."""

    kind: str  # "photo" | "video"
    time_s: float
    photo: PhotoMoment | None = None


def _video_is_placed(source_video: Path | None, source_video_anchor: MediaAnchor | None) -> bool:
    """A video is a located route beat only with a resolved, undisputed clock time.

    A GPS/clock conflict is not a route position either -- it is unplaced
    media, same as a photo with no defensible coordinates.
    """
    return (
        source_video is not None
        and source_video_anchor is not None
        and source_video_anchor.track_time_s is not None
        and source_video_anchor.status != "coordinate_time_conflict"
    )


def _ordered_media_beats(
    mapped: Sequence[PhotoMoment],
    source_video: Path | None,
    source_video_anchor: MediaAnchor | None,
) -> list[_MediaBeat]:
    """Every placed photo and, when placed, the video -- in true route-clock order.

    ``mapped`` already carries only route-anchored photos; a placed video is
    inserted at its own moment rather than always trailing every photo, so a
    video shot between two photos plays between them, not after both.
    """
    beats = [_MediaBeat("photo", photo.anchor.track_time_s or 0.0, photo=photo) for photo in mapped]
    if _video_is_placed(source_video, source_video_anchor) and source_video_anchor is not None:
        track_time_s = source_video_anchor.track_time_s
        if track_time_s is not None:
            beats.append(_MediaBeat("video", track_time_s))
    return sorted(beats, key=lambda beat: beat.time_s)


def _beat_pins(
    beats: Sequence[_MediaBeat], source_video_anchor: MediaAnchor | None
) -> list[PhotoPinBeat]:
    """One ``(time_s, longitude, latitude)`` row per beat with resolved coordinates.

    ``layers.visible_pins`` is what turns this into a per-frame pin list
    (time filter + near-duplicate collapse) -- this just resolves each
    beat's own anchor coordinates, a photo's or the placed video's, without
    inventing one for an anchor that never got coordinates.
    """
    pins: list[PhotoPinBeat] = []
    for beat in beats:
        if beat.kind == "photo" and beat.photo is not None:
            longitude, latitude = beat.photo.anchor.longitude, beat.photo.anchor.latitude
        elif beat.kind == "video" and source_video_anchor is not None:
            longitude, latitude = source_video_anchor.longitude, source_video_anchor.latitude
        else:
            continue
        if longitude is None or latitude is None:
            continue
        pins.append((beat.time_s, longitude, latitude))
    return pins


def _map_leg_bounds(
    start_time: float,
    end_time: float,
    mapped: Sequence[PhotoMoment],
    source_video: Path | None,
    source_video_anchor: MediaAnchor | None,
) -> list[tuple[float, float]]:
    """The (start, end) route-clock span of every map leg this film will render.

    Order matches the render loop: one leg per placed media beat (photo or
    video) in true route-clock order, then the final leg to ``end_time``.
    """
    bounds: list[tuple[float, float]] = []
    cursor = start_time
    for beat in _ordered_media_beats(mapped, source_video, source_video_anchor):
        target = min(end_time, max(start_time, beat.time_s))
        bounds.append((cursor, target))
        cursor = target
    bounds.append((cursor, end_time))
    return bounds


def label_visibility_intervals(
    segments: Sequence[Sequence[RoutePoint]],
    labels: Sequence[MapLabel],
    local: SatelliteRaster,
) -> dict[str, list[dict[str, float]]]:
    """Approximate, from route evidence alone, when each label was on screen.

    Mirrors the map-leg viewport span ``layers._render_map`` uses without
    opening a frame -- a receipt-time approximation, not a frame-exact replay.
    A label's visibility closes the moment the route leaves its viewport, or
    at a segment break (a signal gap is never bridged into one interval).
    """
    if not labels:
        return {}
    span_lon = max(0.075, (local.east - local.west) * 0.42)
    closed: dict[str, list[tuple[float, float]]] = {}
    for segment in segments:
        open_intervals: dict[str, list[float]] = {}
        for point in segment:
            west = point.longitude - span_lon / 2
            east = point.longitude + span_lon / 2
            south = point.latitude - span_lon / 2
            north = point.latitude + span_lon / 2
            visible_now = {
                label.name
                for label in select_visible_labels(
                    labels, west=west, south=south, east=east, north=north, span_lon=span_lon
                )
            }
            for name in visible_now:
                window = open_intervals.setdefault(name, [point.time_s, point.time_s])
                window[1] = point.time_s
            for name in [name for name in open_intervals if name not in visible_now]:
                start, end = open_intervals.pop(name)
                closed.setdefault(name, []).append((start, end))
        for name, (start, end) in open_intervals.items():
            closed.setdefault(name, []).append((start, end))
    return {
        name: [{"start_s": start, "end_s": end} for start, end in windows]
        for name, windows in closed.items()
    }


def _display_filter_receipt(
    max_speed_mps: float,
    activity: str | None,
    activity_source: str | None,
    units: str | None,
    units_source: str | None,
    hud_speed_cap_mps: float | None,
    grade_cap_percent: float | None,
) -> dict[str, object]:
    """The receipt's ``display_filter`` block, plus the HUD's own caps when resolved.

    ``hud_speed_cap_mps``/``grade_cap_percent`` are only ever both ``None`` or
    both set (``film.main`` always resolves an ``ActivityProfile`` before
    calling ``film_receipt``); a caller that never resolved one -- the direct
    unit-test callers of this function -- gets the block without a
    ``"hud"`` key, unchanged.
    """
    block: dict[str, object] = {
        "max_speed_mps": max_speed_mps,
        "max_gap_s": 90.0,
        "activity": activity,
        "activity_source": activity_source,
        "units": units,
        "units_source": units_source,
    }
    if hud_speed_cap_mps is not None or grade_cap_percent is not None:
        block["hud"] = {
            "speed_cap_mps": hud_speed_cap_mps,
            "grade_cap_percent": grade_cap_percent,
        }
    return block


def film_receipt(
    *,
    segments: Sequence[Sequence[RoutePoint]],
    national: SatelliteRaster,
    local: SatelliteRaster,
    regional: Sequence[SatelliteRaster] = (),
    photos: Sequence[PhotoMoment],
    output: Path | None = None,
    source_video: Path | None = None,
    source_video_anchor: MediaAnchor | None = None,
    labels: Sequence[MapLabel] = (),
    terrain: TerrainGrid | None = None,
    map_renderer: MapRenderer | None = None,
    map_seconds: float | None = None,
    bookend_info: dict[str, tuple[str, float]] | None = None,
    bookend_play_order: Sequence[str] = (),
    max_speed_mps: float = 8.0,
    activity: str | None = None,
    activity_source: str | None = None,
    units: str | None = None,
    units_source: str | None = None,
    hud_speed_cap_mps: float | None = None,
    grade_cap_percent: float | None = None,
    mapping_version: str | None = None,
    pacing_mode: str = "recorded",
    presentation_mapping_version: str = PRESENTATION_MAPPING_VERSION,
    motion: dict | None = None,
    route_trim: dict | None = None,
) -> dict[str, object]:
    """Return privacy-safe evidence for a completed private film export.

    ``bookend_info`` maps a kept bookend photo's ``media_id`` to
    ``(placement, seconds_outside_track)`` -- overriding the ordinary
    mapped/unmapped placement for that one entry. ``bookend_play_order`` is
    every kept bookend's ``media_id`` in the order it actually plays.

    ``motion`` (``planner.MotionSummary.as_receipt()["motion"]``) and
    ``route_trim`` (``route_privacy.receipt_fragment(result)["route_trim"]``)
    are recorded verbatim under those keys when supplied, and omitted
    entirely when ``None`` -- the receipt shape is defined here, in one
    place, rather than by a caller mutating the dict afterward. When
    ``motion`` is left ``None`` and ``map_renderer`` exposes at least two
    entries on ``rendered_poses`` (the scene renderer's actual per-frame
    camera path, not the plan) plus an ``fps``, it is computed from those
    poses instead -- the real motion of the rendered camera, never invented
    for a renderer that didn't record one.

    ``max_speed_mps`` is the display speed filter's cap -- ordinarily the
    resolved :class:`route_story.activity.ActivityProfile`'s own
    ``max_speed_mps`` (``activity``/``activity_source`` record which one and
    how it was resolved, for the receipt only; they do not affect
    filtering). ``units``/``units_source``/``mapping_version`` similarly
    record the resolved display units and the table version that produced
    them, for the receipt only. ``hud_speed_cap_mps``/``grade_cap_percent``
    record what the same profile set the HUD's own speed/grade caps to, for
    the receipt only -- they do not affect this function's own filtering.
    ``pacing_mode`` (default ``"recorded"``, the only implemented value)
    and ``presentation_mapping_version`` (default
    :data:`PRESENTATION_MAPPING_VERSION`) disclose which presentation-timing
    table produced this render, versioned independently of ``mapping_version``
    (:data:`route_story.activity.MAPPING_VERSION`), which versions the
    unrelated activity/region/units tables.
    """
    bookend_info = bookend_info or {}
    qualified_segments = clean_display_segments(
        segments, max_speed_mps=max_speed_mps, max_gap_s=90.0
    )
    points = [point for segment in segments for point in segment]
    if points and qualified_segments:
        mapped_photos = sorted(
            (photo for photo in photos if photo.is_mapped),
            key=lambda photo: photo.anchor.track_time_s or 0,
        )
        leg_bounds = _map_leg_bounds(
            min(point.time_s for point in points),
            qualified_segments[-1][-1].time_s,
            mapped_photos,
            source_video,
            source_video_anchor,
        )
        time_compression, _ = plan_map_legs(leg_bounds, fps=1, map_seconds=map_seconds)
        photo_pins = len(
            _beat_pins(
                _ordered_media_beats(mapped_photos, source_video, source_video_anchor),
                source_video_anchor,
            )
        )
    else:
        time_compression = 1.0
        photo_pins = 0
    media_views = [
        _AnchoredMediaView(photo.anchor.track_time_s, photo.anchor.latitude, photo.anchor.longitude)
        for photo in photos
    ]
    if source_video_anchor is not None:
        media_views.append(
            _AnchoredMediaView(
                source_video_anchor.track_time_s,
                source_video_anchor.latitude,
                source_video_anchor.longitude,
            )
        )
    beats = extract_beats(qualified_segments, media=media_views) if qualified_segments else []
    receipt: dict[str, object] = {
        "outputs": (
            [{"path": Path(output).name, "sha256": _sha256_file(Path(output))}]
            if output is not None
            else []
        ),
        "beats": beats_receipt(beats),
        "label_visibility": label_visibility_intervals(qualified_segments, labels, local),
        "map_renderer": map_renderer.name if map_renderer is not None else "pillow-2d",
        "camera_continuity": getattr(map_renderer, "camera_continuity", None),
        "national_imagery": national.receipt(),
        "regional_imagery": [raster.receipt() for raster in regional],
        "local_imagery": local.receipt(),
        "route_summary": asdict(summarize_track(qualified_segments, moving_speed_mps=0.5)),
        "route_summary_raw": asdict(summarize_track(segments, moving_speed_mps=0.5)),
        "display_filter": _display_filter_receipt(
            max_speed_mps,
            activity,
            activity_source,
            units,
            units_source,
            hud_speed_cap_mps,
            grade_cap_percent,
        ),
        "mapping_version": mapping_version,
        "pacing_mode": pacing_mode,
        "presentation_mapping_version": presentation_mapping_version,
        "time_compression": time_compression,
        "photo_pins": photo_pins,
        "map_seconds": map_seconds,
        "map_labels": {
            "count": len(labels),
            "names": [label.name for label in labels],
            "kinds": sorted({label.kind for label in labels}),
        },
        "photos": [
            {
                "media_id": photo.anchor.media_id,
                "anchor_status": photo.anchor.status,
                "placement": (
                    bookend_info[photo.anchor.media_id][0]
                    if photo.anchor.media_id in bookend_info
                    else "route_moment"
                    if photo.is_mapped
                    else "unlocated_memory"
                ),
                "interpolation_method": photo.anchor.interpolation_method,
                "rejected_gps": _rejected_gps(photo.anchor),
                "required_speed_mps": photo.anchor.required_speed_mps,
                **(
                    {"seconds_outside_track": bookend_info[photo.anchor.media_id][1]}
                    if photo.anchor.media_id in bookend_info
                    else {}
                ),
            }
            for photo in photos
        ],
        "bookend_photos": list(bookend_play_order),
        "source_video": (
            {
                "media_id": Path(source_video).name,
                "anchor_status": source_video_anchor.status if source_video_anchor else None,
                "placement": (
                    "review_required"
                    if source_video_anchor
                    and source_video_anchor.status == "coordinate_time_conflict"
                    else "route_moment"
                    if source_video_anchor and source_video_anchor.track_time_s is not None
                    else "unlocated_memory"
                ),
                "interpolation_method": (
                    source_video_anchor.interpolation_method if source_video_anchor else None
                ),
                "rejected_gps": (
                    _rejected_gps(source_video_anchor) if source_video_anchor else None
                ),
                "required_speed_mps": (
                    source_video_anchor.required_speed_mps if source_video_anchor else None
                ),
            }
            if source_video is not None
            else None
        ),
    }
    if motion is None and map_renderer is not None:
        rendered_poses = getattr(map_renderer, "rendered_poses", None)
        renderer_fps = getattr(map_renderer, "fps", None)
        if rendered_poses is not None and len(rendered_poses) >= 2 and renderer_fps is not None:
            motion = motion_summary(
                motion_report_from_poses(rendered_poses, renderer_fps), renderer_fps
            ).as_receipt()["motion"]
    if motion is not None:
        receipt["motion"] = motion
    if route_trim is not None:
        receipt["route_trim"] = route_trim
    return receipt


def _rejected_gps(anchor: MediaAnchor) -> dict[str, float] | None:
    """The physically-impossible embedded GPS an anchor discarded, if any."""
    if anchor.rejected_latitude is None and anchor.rejected_longitude is None:
        return None
    return {"latitude": anchor.rejected_latitude, "longitude": anchor.rejected_longitude}


def clean_display_segments(
    segments: Sequence[Sequence[RoutePoint]], *, max_speed_mps: float, max_gap_s: float
) -> list[list[RoutePoint]]:
    """Keep only continuous, plausible GPS edges for displayed telemetry.

    The untouched GPX remains the source evidence.  This derived route simply
    prevents an endpoint jump or a long recording gap from becoming a false
    on-screen speed, grade, distance, or animated line.
    """
    if max_speed_mps <= 0 or max_gap_s <= 0:
        raise ValueError("display GPS limits must be positive")
    accepted: list[list[RoutePoint]] = []
    for source in segments:
        current: list[RoutePoint] = []
        for first, second in zip(source, source[1:]):
            duration = second.time_s - first.time_s
            speed = distance_m(first, second) / duration if duration > 0 else float("inf")
            if 0 < duration <= max_gap_s and speed <= max_speed_mps:
                if not current:
                    current.append(first)
                current.append(second)
            elif len(current) > 1:
                accepted.append(current)
                current = []
            else:
                current = []
        if len(current) > 1:
            accepted.append(current)
    return accepted


def route_end_time(
    segments: Sequence[Sequence[RoutePoint]], *, max_speed_mps: float = 8.0, max_gap_s: float = 90.0
) -> float:
    """Return the last accepted GPS edge — the true end of a displayable story.

    A raw recording can run past the last plausible edge (an endpoint jump, a
    dangling trailing fix). The story must end there, not on an implausible
    tail the display already refuses to draw.
    """
    qualified = clean_display_segments(segments, max_speed_mps=max_speed_mps, max_gap_s=max_gap_s)
    if not qualified:
        raise ValueError("Route Story has no plausible continuous GPS edges to display")
    return qualified[-1][-1].time_s


def drawn_segments(
    segments: Sequence[Sequence[RoutePoint]], *, smooth: bool, max_speed_mps: float = 8.0
) -> tuple[list[list[RoutePoint]], SmoothedTrack | None]:
    """The cleaned display segments, plus a display-only smoothed track when ``smooth``.

    ``render_route_story_film`` and ``main``'s preflight sampling both call
    this single pure helper instead of each deriving their own line, so a
    preflight frame can never sample a different line than the film actually
    draws. The first element is always the un-smoothed cleaned segments --
    every MEASUREMENT (``summarize_track``, ``end_time``, leg bounds, the
    dashboard state) stays keyed to it regardless of ``smooth``; only what a
    caller actually DRAWS should read ``SmoothedTrack.segments`` off the
    second element. Never raises -- an empty result (no plausible continuous
    GPS edge) is a caller decision, same as ``clean_display_segments``.
    """
    cleaned = clean_display_segments(segments, max_speed_mps=max_speed_mps, max_gap_s=90.0)
    if not smooth or not cleaned:
        return cleaned, None
    track = smooth_display_segments(
        cleaned, window_m=DISPLAY_SMOOTHING_WINDOW_M, max_shift_m=DISPLAY_SMOOTHING_MAX_SHIFT_M
    )
    return cleaned, track


def _display_smoothing_receipt(
    smoothed_track: SmoothedTrack | None, *, enabled: bool
) -> dict[str, object]:
    """The receipt's evidence that the drawn line was smoothed -- or wasn't.

    Shared by the full receipt and the ``--preflight-only`` partial receipt so
    neither can silently disagree with what the other reports.
    """
    return {
        "enabled": enabled,
        "recipe": smoothed_track.recipe if smoothed_track is not None else None,
        "window_m": DISPLAY_SMOOTHING_WINDOW_M if enabled else None,
        "max_shift_m": DISPLAY_SMOOTHING_MAX_SHIFT_M if enabled else None,
        "max_shift_m_observed": (
            smoothed_track.max_shift_m_observed if smoothed_track is not None else None
        ),
        "mean_shift_m": smoothed_track.mean_shift_m if smoothed_track is not None else None,
        "points_clamped": smoothed_track.points_clamped if smoothed_track is not None else None,
    }


def _render_recap_map(
    rasters: Sequence[SatelliteRaster],
    segments: Sequence[Sequence[RoutePoint]],
    end_time: float,
    width: int,
    height: int,
    *,
    labels: Sequence[MapLabel] = (),
    terrain: TerrainGrid | None = None,
    pins: Sequence[PhotoPinBeat] = (),
) -> Image.Image:
    """The recap's map: the whole day in one view, with no rail and no gap badge."""
    return _render_map(
        rasters,
        segments,
        end_time,
        width,
        height,
        show_dashboard=False,
        show_gap_badge=False,
        labels=labels,
        terrain=terrain,
        overview=True,
        pins=pins,
    )


def _last_saved_frame(frames: Path, next_frame: int) -> Image.Image | None:
    """The frame the viewer saw last, or None when nothing was written there
    (a stand-in scene that writes no pixels) -- then there is no dissolve."""
    path = frames / f"{next_frame - 1:06d}.jpg"
    if next_frame < 1 or not path.is_file():
        return None
    with Image.open(path) as saved:
        return saved.convert("RGB")


def render_route_story_film(
    *,
    segments: Sequence[Sequence[RoutePoint]],
    national: SatelliteRaster,
    local: SatelliteRaster,
    regional: Sequence[SatelliteRaster] = (),
    photos: Sequence[PhotoMoment],
    output: Path,
    source_video: Path | None = None,
    source_video_anchor: MediaAnchor | None = None,
    labels: Sequence[MapLabel] = (),
    terrain: TerrainGrid | None = None,
    map_renderer: MapRenderer | None = None,
    opening_rasters: Sequence[SatelliteRaster] = (),
    fps: int = 30,
    width: int = 1920,
    height: int = 1080,
    intro_seconds: float = DEFAULT_INTRO_SECONDS,
    map_seconds: float | None = None,
    photo_seconds: float = 3.1,
    recap_seconds: float = DEFAULT_RECAP_SECONDS,
    still_source: str = "developed",
    recording_start_utc: datetime | None = None,
    utc_offset: str | None = None,
    before_bookends: Sequence[PhotoMoment] = (),
    after_bookends: Sequence[PhotoMoment] = (),
    display_smoothing: bool = True,
    title: str | None = None,
    recorded_on: date | None = None,
    watermark: str | None = DEFAULT_WATERMARK_TEXT,
    max_speed_mps: float = 8.0,
) -> FilmTimeline:
    """Render an H.264 Route Story film from local assets and original media.

    Mapped photos and a placed source video arrive on the route in true
    route-clock order.  A placed video is not an interlude: the map flies to
    its route moment, a "VIDEO ARRIVAL" badge shows, the clip itself plays
    right there -- its own audio, near-silent and attenuate-only like every
    other surface -- and the map then resumes from that same moment to the
    end.  A photo with no defensible coordinates, or a video whose anchor
    never resolved to an undisputed route moment, is unplaced media: it is
    left out of the film entirely, never drawn and never appended after the
    last map leg.  ``film.main`` is where unplaced media is named and gated
    (``--allow-unplaced``); this function simply never draws it.

    ``before_bookends``/``after_bookends`` are same-day photos shot just
    outside the recording (see ``classify_bookend``); they play, in capture
    order, right after the opening and right before the recap respectively,
    using the same unplaced-style still card (no pin, no route position
    claim) as any other unmapped photo -- but they never trigger the
    unplaced-media refusal and are never left out for being unplaced.

    ``map_seconds``, when supplied, overrides the total on-screen seconds
    given to every map leg combined (``plan_map_legs``'s honest, globally
    proportional time-compression factor); left unset, the total matches
    what the old per-leg ``map_leg_seconds`` clock would have produced.

    ``display_smoothing`` (default on) draws a display-only smoothed line
    (``display_track.smooth_display_segments``) for every map leg, the
    video-arrival hold and the recap map. It never changes a measurement --
    ``summarize_track``, leg bounds, pacing and the dashboard rail all stay
    keyed to the recorded, un-smoothed points either way.

    Returns the ``FilmTimeline`` this render planned -- every hold and cut in
    absolute output-frame indices, once every clip's length was known -- for
    a caller to hand to ``continuity.analyze_rendered_continuity`` against
    the delivered file. It is record-only: nothing here reads it back.
    """
    if fps <= 0 or width < 320 or height < 180:
        raise ValueError("Route Story film needs a positive fps and a 320x180 frame or larger")
    if min(intro_seconds, photo_seconds, recap_seconds) <= 0:
        raise ValueError("Route Story scene durations must be positive")
    if map_seconds is not None and map_seconds <= 0:
        raise ValueError("Route Story scene durations must be positive")
    points = [point for segment in segments for point in segment]
    if not points:
        raise ValueError("Route Story film needs at least one route point")
    if source_video is not None:
        source_video = Path(source_video)
        if source_video.is_symlink() or not source_video.is_file():
            raise ValueError("Route Story video must be a regular source file")

    mapped = sorted(
        (photo for photo in photos if photo.is_mapped),
        key=lambda photo: photo.anchor.track_time_s or 0,
    )
    start_time = min(point.time_s for point in points)
    display_segments, smoothed_track = drawn_segments(
        segments, smooth=display_smoothing, max_speed_mps=max_speed_mps
    )
    if not display_segments:
        raise ValueError("Route Story has no plausible continuous GPS edges to display")
    draw_segments = smoothed_track.segments if smoothed_track is not None else display_segments
    end_time = display_segments[-1][-1].time_s
    summary = summarize_track(display_segments, moving_speed_mps=0.5)
    map_rasters = (national, *regional, local)
    # With a bundle pyramid the opening picks the sharpest raster covering each
    # frame; without one it keeps the caller's coarse-to-fine order.
    opening_set = finest_last((*map_rasters, *opening_rasters)) if opening_rasters else map_rasters
    renderer = map_renderer or PillowMapRenderer(
        rasters=map_rasters, labels=labels, terrain=terrain
    )
    # The line drawn may be display-smoothed; the dashboard rail each renderer
    # bakes into its frames must not be. The rail fill is translucent, so the
    # right numbers can never be re-stamped over the wrong ones afterwards --
    # the renderer itself measures the recorded points, or the film refuses.
    active_renderer: MapRenderer = renderer
    if smoothed_track is not None:
        measuring = getattr(renderer, "measuring", None)
        if measuring is None:
            raise ValueError(
                f"map renderer {getattr(renderer, 'name', renderer)!r} cannot measure the "
                "recorded points while drawing a smoothed line; give it measuring() or "
                "pass display_smoothing=False"
            )
        active_renderer = measuring(display_segments)

    ordered_beats = _ordered_media_beats(mapped, source_video, source_video_anchor)
    pin_beats = _beat_pins(ordered_beats, source_video_anchor)
    # Only a renderer that declares its own ``pins`` field (both renderers
    # this module ships) is asked to carry them; a caller-supplied
    # ``MapRenderer`` with no such field renders exactly as it did before.
    if hasattr(renderer, "pins"):
        renderer = replace(renderer, pins=pin_beats)
    # A per-frame QA fault must cost one frame, not the whole render. Preflight
    # has already judged its sample frames with no ledger, so a film that is
    # broken from the first pose still refuses before spending hours; from here
    # on faults are collected and the run fails from the ledger below, once the
    # film exists to look at. Same optional-field pattern as ``pins`` above: a
    # caller-supplied MapRenderer without the field renders as it always did.
    qa_ledger = FrameQaLedger()
    qa_failure: FrameQaError | None = None
    timeline = FilmTimeline()
    if hasattr(active_renderer, "qa_ledger"):
        active_renderer = replace(active_renderer, qa_ledger=qa_ledger)
    leg_bounds = _map_leg_bounds(start_time, end_time, mapped, source_video, source_video_anchor)
    time_compression, leg_frame_counts = plan_map_legs(leg_bounds, fps, map_seconds=map_seconds)

    # The dashboard's TIME cell reads route-second 0 as this clock; its motion
    # cell (PACE vs SPEED) is fixed for the whole film by the trip's overall
    # moving-average speed, never re-decided per frame. Scoped here, for the
    # whole render, the same way the badge these replaced was scoped per leg.
    clock_anchor = None
    if recording_start_utc is not None and utc_offset is not None:
        offset_tz = parse_utc_offset(utc_offset)
        clock_anchor = recording_start_utc.astimezone(offset_tz)
    set_active_dashboard_clock(clock_anchor)
    set_active_motion_cell_mode(motion_cell_mode(summary.moving_average_speed_mps))
    # The opening card and the summit badge are both derived here, once, from
    # the track and the labels file -- never re-derived per frame, and never
    # invented: an operator title wins outright, no nearby place means no
    # place line, and a track with no elevation gets no badge at all.
    opening_title, opening_subtitle = resolve_opening_title(
        title,
        labels,
        start=(points[0].latitude, points[0].longitude),
        end=(points[-1].latitude, points[-1].longitude),
        recorded_on=recorded_on,
    )
    high_point = high_point_callout(points)
    kept_root = _kept_frames_root(output)
    if kept_root.exists():
        shutil.rmtree(kept_root, ignore_errors=True)
    kept_root.mkdir(parents=True, exist_ok=True)
    try:
        root = kept_root
        recap_frames = root / "recap-frames"
        recap_frames.mkdir()

        # One frame sequence per part -- a placed video splits the sequence
        # in two (frames up to its arrival badge, then frames from its
        # departure onward), each numbered from 0. No video, or an unplaced
        # one, means exactly one part: today's whole-film sequence.
        parts: list[Path] = [root / "main-1-frames"]
        parts[0].mkdir()
        frames = parts[0]
        frame = 0
        main_segment = timeline.new_segment()
        video_segment: int | None = None

        opening_count = max(1, round(intro_seconds * fps))
        for opening_frame in range(opening_count):
            fraction = opening_frame / max(1, opening_count - 1)
            image = _opening_frame(
                opening_set,
                points[0],
                fraction,
                width,
                height,
                labels,
                title=opening_title,
                subtitle=opening_subtitle,
            )
            _save_frame(image, frames, frame, watermark=watermark)
            frame += 1

        # Bookends shot before the recording started play right after the
        # opening and before the first map leg -- capture order, no pin, no
        # route position claim (the same unplaced-style still card as any
        # other unmapped photo).
        for bookend in before_bookends:
            timeline.record_cut(main_segment, frame)
            frame = _append_photo(
                frames,
                frame,
                bookend,
                fps,
                photo_seconds,
                width,
                height,
                mapped=False,
                still_source=still_source,
                caption=BOOKEND_BEFORE_CAPTION,
                watermark=watermark,
            )
            timeline.record_cut(main_segment, frame - 1)

        # The still that just played dissolves into the next map leg. Read
        # back from the saved frame: it is exactly what the viewer saw last.
        after_still: Image.Image | None = None
        if before_bookends:
            after_still = _last_saved_frame(frames, frame)
        previous_time = start_time
        for leg_index, beat in enumerate(ordered_beats):
            target = leg_bounds[leg_index][1]
            frame, leg_last_image = _append_map_leg(
                frames,
                frame,
                draw_segments,
                active_renderer,
                previous_time,
                target,
                frame_count=leg_frame_counts[leg_index],
                width=width,
                height=height,
                callout=high_point,
                watermark=watermark,
                dissolve_from=after_still,
                fps=fps,
            )
            timeline.mark_opening_end(frame)
            after_still = None
            beat_photo = beat.photo
            # The map stops on the beat's own route moment before its
            # media plays. Without it the camera is still moving when the
            # picture arrives, so the still reads as an interruption
            # rather than as the thing the ride just arrived at. The
            # settled frame IS the leg's own last rendered frame -- never
            # a second, un-rate-limited render() call at the same time,
            # which would snap the bearing the viewer just watched settle.
            if leg_last_image is None:
                raise RuntimeError("a map leg with frame_count >= 1 rendered no frames")
            settled = leg_last_image.copy()
            if beat.kind == "photo" and beat_photo is not None:
                frame = _append_hold(
                    frames,
                    frame,
                    settled,
                    fps,
                    MAP_SETTLE_SECONDS,
                    timeline=timeline,
                    timeline_segment=main_segment,
                )
                timeline.record_cut(main_segment, frame)
                frame = _append_photo(
                    frames,
                    frame,
                    beat_photo,
                    fps,
                    photo_seconds,
                    width,
                    height,
                    mapped=True,
                    still_source=still_source,
                    watermark=watermark,
                    dissolve_from=settled,
                )
                timeline.record_cut(main_segment, frame - 1)
                after_still = _last_saved_frame(frames, frame)
            else:
                # The hold must come from the SAME renderer every map leg
                # uses -- a scene renderer's held frame otherwise pops to the
                # unrelated flat Pillow map right before the clip plays.
                # No extra render call: the badge is drawn once, in place,
                # on the leg's own last frame, and every hold frame reuses it.
                _draw_media_arrival(settled, "VIDEO ARRIVAL", width, height)
                frame = _append_hold(
                    frames,
                    frame,
                    settled,
                    fps,
                    VIDEO_ARRIVAL_HOLD_SECONDS,
                    timeline=timeline,
                    timeline_segment=main_segment,
                )
                # The part just finished has all its frames now -- close it
                # before opening the (as yet unencoded, unmeasured) placed
                # video's own segment and the part that resumes after it.
                timeline.close_segment(main_segment, frame)
                video_segment = timeline.new_segment()
                frames = root / f"main-{len(parts) + 1}-frames"
                frames.mkdir()
                frame = 0
                parts.append(frames)
                main_segment = timeline.new_segment()
            previous_time = target

        frame, _ = _append_map_leg(
            frames,
            frame,
            draw_segments,
            active_renderer,
            previous_time,
            end_time,
            frame_count=leg_frame_counts[-1],
            width=width,
            height=height,
            callout=high_point,
            watermark=watermark,
            dissolve_from=after_still,
            fps=fps,
        )
        timeline.mark_opening_end(frame)

        # Bookends shot after the recording ended play right after the last
        # map leg and before the recap -- same rule as before_bookends.
        for bookend in after_bookends:
            timeline.record_cut(main_segment, frame)
            frame = _append_photo(
                frames,
                frame,
                bookend,
                fps,
                photo_seconds,
                width,
                height,
                mapped=False,
                still_source=still_source,
                caption=BOOKEND_AFTER_CAPTION,
                watermark=watermark,
            )
            timeline.record_cut(main_segment, frame - 1)

        # The last main part now has every frame it will ever get.
        timeline.close_segment(main_segment, frame)
        recap_segment = timeline.new_segment()
        recap_count = max(1, round(recap_seconds * fps))
        # The recap map is the same picture every frame; only the numbers'
        # fade changes. Render it once and draw each fade on a fresh copy.
        recap_map = _render_recap_map(
            map_rasters,
            draw_segments,
            end_time,
            width,
            height,
            labels=labels,
            terrain=terrain,
            pins=pin_beats,
        )
        for recap_frame in range(recap_count):
            image = recap_map.copy()
            _draw_recap(image, summary, recap_frame / max(1, recap_count - 1))
            _save_frame(image, recap_frames, recap_frame, watermark=watermark)
        timeline.close_segment(recap_segment, recap_count)

        part_clips = [root / f"main-{index + 1}.mp4" for index in range(len(parts))]
        for part_frames, part_clip in zip(parts, part_clips):
            _encode_frames(part_frames, part_clip, fps)
        recap = root / "recap.mp4"
        _encode_frames(recap_frames, recap, fps)

        clips = [part_clips[0]]
        if len(part_clips) > 1 and source_video is not None:
            video_clip = root / "video.mp4"
            _encode_source_video(source_video, video_clip, fps, width, height, watermark=watermark)
            clips.append(video_clip)
            clips.extend(part_clips[1:])
            # The placed clip's own length was never known while frames
            # were being planned -- only once it is actually encoded.
            if video_segment is not None:
                video_seconds = FFprobeClient(subprocess).probe_duration(video_clip)
                timeline.close_segment(video_segment, max(1, round(video_seconds * fps)))
        clips.append(recap)

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        _concat(clips, output)
        qa_failure = _report_qa_faults(
            qa_ledger, output, fps=fps, width=width, height=height, legs=len(leg_bounds)
        )
    except FrameQaError as error:
        # kept_root (unlike the TemporaryDirectory this used to be) survives
        # this refusal -- see _kept_frames_root -- but the refused pixels
        # themselves are still written out beside the operator's own output
        # path, which is the one place a reviewer already knows to look.
        for path in save_refusal_evidence(error, output):
            logger.error("route-story frame QA refused a frame; evidence: %s", path)
        raise
    finally:
        set_active_dashboard_clock(None)
        set_active_motion_cell_mode("speed")
    if qa_failure is not None:
        raise qa_failure
    # No FrameQaError escaped and no ledger fault was pending: this render is
    # shippable, so whatever a prior refusal for this same output left behind
    # is no longer needed.
    shutil.rmtree(kept_root, ignore_errors=True)
    return timeline


# The delivered product is 1080p30. A smaller or slower film is a preview, and a
# preview is something a run asks for by name (--width/--height/--fps), never
# what it gets by saying nothing.
DELIVERY_WIDTH = 1920
DELIVERY_HEIGHT = 1080
DELIVERY_FPS = 30


_CODEC_FAMILY = {"hevc": "hevc", "h264": "h264"}


def build_rendition_check(preset: Any, delivered: Mapping[str, Any]) -> dict[str, bool] | None:
    """Record-only: does the delivered file match ``preset``'s rendition of record?

    ``None`` when the preset names no rendition. ``codec_ok`` is recorded only
    for a preset whose delivery encode is HEVC (``hevc_delivery``): social
    names an hevc row but ships libx264, so a codec comparison there would
    record a mismatch that is not one.
    """
    if preset.rendition is None:
        return None
    rendition_of_record = RENDITIONS[preset.rendition]
    video_info = delivered.get("video") or {}
    duration = delivered.get("duration_s") or 0.0
    check = {
        "audio_stream": delivered.get("audio") is not None,
        "size_ok": (video_info.get("width"), video_info.get("height"))
        == (rendition_of_record.width, rendition_of_record.height),
        "within_ceiling": rendition_of_record.max_seconds is None
        or duration <= rendition_of_record.max_seconds,
    }
    if preset.hevc_delivery:
        check["codec_ok"] = video_info.get("codec") == _CODEC_FAMILY.get(
            rendition_of_record.codec, rendition_of_record.codec
        )
    return check


def _report_qa_faults(
    ledger: FrameQaLedger,
    output: Path,
    *,
    fps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    legs: int | None = None,
) -> FrameQaError | None:
    """Write the QA fault report beside the film; return the failure to raise.

    Called only once ``output`` exists. A faulted render is still a FAILED
    render and the returned error is raised by the caller -- nothing downstream
    treats the film as shippable -- but the operator now has the footage AND
    the list, instead of the four hours of nothing a mid-capture refusal used
    to leave behind.

    The error is returned rather than raised here so it escapes *outside* the
    caller's ``except FrameQaError`` block: that handler exists to rescue
    pixels from the staging ``TemporaryDirectory`` before it is deleted, and
    letting it re-handle this one would overwrite the evidence just written.

    ``fps``/``width``/``height``/``legs`` (all optional -- omitted only by a
    caller with nothing to say, never by this render) become a
    ``render_identity`` block, and each fault's own recorded pose becomes a
    ``poses`` map keyed by its frame number -- both additive, so an old
    reader that only knows ``frames_faulted``/``frames``/``problems`` still
    loads this file. ``--repair`` is what reads them back: the identity lets
    it refuse a report written for a different render instead of splicing a
    frame into the wrong one, and a fault's pose is what lets it recapture
    that exact frame without a full re-render.
    """
    if not ledger.faults:
        return None
    summary = ledger.summary()
    if fps is not None or width is not None or height is not None or legs is not None:
        summary["render_identity"] = {
            "output": output.name,
            "fps": fps,
            "width": width,
            "height": height,
            "legs": legs,
        }
    summary["poses"] = {
        str(fault.result.frame): fault.pose for fault in ledger.faults if fault.pose is not None
    }
    report = output.parent / f"{output.stem}.qa-faults.json"
    try:
        report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.error(
            "route-story frame QA faulted %d frame(s); report: %s",
            summary["frames_faulted"],
            report,
        )
    except OSError as exc:
        logger.warning("route-story frame QA could not write %s: %s", report, exc)
    first = ledger.faults[0]
    error = FrameQaError(first.result, first.pose, rgb=first.rgb)
    for path in save_refusal_evidence(error, output):
        logger.error("route-story frame QA evidence: %s", path)
    worst = ", ".join(f"{sentence} (x{count})" for sentence, count in summary["problems"].items())
    return FrameQaError(
        replace(
            first.result,
            problems=(
                f"{summary['frames_faulted']} frame(s) failed QA in a film that WAS written to "
                f"{output}: {worst}",
            ),
        ),
        first.pose,
    )


def _repair_scene_frame(
    scene_capture, pose: dict, *, frame_index: int, width: int, height: int
) -> Image.Image:
    """Recapture one pose, cache-bypassed, and refuse it if it is still bad.

    Frame QA runs on the RAW capture (before labels and the canvas), the same
    point ``scene_renderer`` checks a normal frame, so a pose that recaptures
    into the same void/unloaded tile raises ``FrameQaError`` instead of being
    pasted back into the film.
    """
    from .frame_qa import FrameQaThresholds, check_frame, marker_xy_from_labels
    from .proof import _draw_labels

    array, labels = scene_capture.render(pose, bypass_cache=True)
    result = check_frame(
        frame_index, array, FrameQaThresholds(), marker_xy=marker_xy_from_labels(labels)
    )
    if not result.ok:
        raise FrameQaError(result, pose, rgb=array)
    array = _draw_labels(array, labels)
    canvas = Image.new("RGB", (width, height), "#071219")
    canvas.paste(Image.fromarray(array, mode="RGB"), (0, 0))
    return canvas


def repair_route_story_film(
    qa_faults_path: Path,
    *,
    output: Path,
    fps: int,
    width: int,
    height: int,
    recapture: Callable[[dict, int], Image.Image],
    watermark: str | None = None,
) -> None:
    """Re-render ONLY the frames a prior QA-refused render faulted.

    Reads ``qa_faults_path`` (``_report_qa_faults``'s output, extended with
    ``render_identity`` and ``poses``) and the kept frames that render left
    under ``_kept_frames_root(output)``, recaptures each faulted frame's own
    recorded pose through ``recapture`` (a caller-supplied, cache-bypassing
    single-pose renderer called as ``recapture(pose, frame_index)`` -- see
    ``_repair_scene_frame``, which re-runs frame QA on the recaptured pixels),
    overwrites that one frame file, and re-encodes.

    Every check below is a REFUSAL, never a fallback: this must never guess
    its way into overwriting a frame it cannot prove belongs to this render.
    A report with more than one map leg is refused outright -- a fault's
    frame number resets at every leg boundary, so without a per-leg offset
    this repair cannot tell which leg's frame it would be overwriting; a
    single-leg render (the common case: a route with no placed media beat
    splitting it into more than one leg) has no such ambiguity.
    """
    output = Path(output)
    report = json.loads(Path(qa_faults_path).read_text(encoding="utf-8"))
    identity = report.get("render_identity")
    expected = {"output": output.name, "fps": fps, "width": width, "height": height}
    if identity is None or {k: identity.get(k) for k in expected} != expected:
        raise ValueError(
            f"--repair identity mismatch: {qa_faults_path} was written for "
            f"{identity!r}, this run asked to repair {expected!r} -- refusing to "
            "repair a different render"
        )
    if identity.get("legs") != 1:
        raise ValueError(
            f"--repair only supports a single-map-leg render; {qa_faults_path} "
            f"recorded {identity.get('legs')!r} leg(s), so a frame number is not "
            "unique across legs and repair cannot tell which one to overwrite"
        )
    kept_root = _kept_frames_root(output)
    if not kept_root.is_dir():
        raise ValueError(
            f"no kept frames at {kept_root} for {output}; --repair needs the "
            "frames a QA-refused render left behind, not a fresh render"
        )
    frames_dir = kept_root / "main-1-frames"
    if not frames_dir.is_dir():
        raise ValueError(f"{frames_dir} is missing; kept frames are incomplete, cannot repair")
    frame_indices = report.get("frames") or []
    if not frame_indices:
        raise ValueError(f"{qa_faults_path} names no faulted frames; nothing to repair")
    poses = report.get("poses") or {}
    for frame_index in frame_indices:
        if poses.get(str(frame_index)) is None:
            raise ValueError(
                f"frame {frame_index} has no recorded pose in {qa_faults_path}; "
                "cannot repair it without one -- refusing to fall back to a full re-render"
            )
    for frame_index in frame_indices:
        image = recapture(poses[str(frame_index)], frame_index)
        _save_frame(image, frames_dir, frame_index, watermark=watermark)
    main_clip = kept_root / "main-1.mp4"
    _encode_frames(frames_dir, main_clip, fps)
    recap_clip = kept_root / "recap.mp4"
    _encode_frames(kept_root / "recap-frames", recap_clip, fps)
    _concat([main_clip, recap_clip], output)


def configure_progress_logging() -> None:
    """Send the film's progress lines to stderr, where a run's log file collects them."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")


def _opening_frame(
    rasters: Sequence[SatelliteRaster],
    start: RoutePoint,
    fraction: float,
    width: int,
    height: int,
    labels: Sequence[MapLabel],
    title: str = "",
    subtitle: str = "",
) -> Image.Image:
    """One opening frame, credited to the imagery it actually shows."""
    image = _render_opening_map(rasters, start, fraction, width, height, labels)
    _draw_opening(image, fraction, opening_credit(rasters, start, fraction, width, height))
    draw_opening_title(image, fraction, title, subtitle)
    return image


def _utc_offset_arg(text: str) -> str:
    """argparse ``type=`` for ``--utc-offset``: the validated raw text, not a ``timezone``."""
    try:
        parse_utc_offset(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return text


def _privacy_zone_arg(text: str) -> PrivacyZone:
    """argparse ``type=`` for repeatable ``--privacy-zone LAT,LON,RADIUS_M``."""
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--privacy-zone must be LAT,LON,RADIUS_M, got {text!r}")
    try:
        center_lat, center_lon, radius_m = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--privacy-zone values must be numbers: {text!r}"
        ) from exc
    try:
        return PrivacyZone(center_lat=center_lat, center_lon=center_lon, radius_m=radius_m)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--privacy-zone {text!r}: {exc}") from exc


def _delivery_continuity(
    output: Path, fps: int, timeline: FilmTimeline, delivered: dict | None
) -> dict | None:
    """Record the rendered pass of ``continuity`` under ``delivered["continuity"]``.

    Record-only, upstream of nothing: an analysis failure -- a bad decode, an
    unresolved timeline -- is recorded as ``{"error": ...}`` and never raises,
    never changes ``delivered``'s own pass/fail status. ``delivered`` that is
    not a dict, or a ``timeline`` that is not a ``FilmTimeline`` (a
    caller-supplied stand-in, or a future shape change), is returned untouched
    rather than guessed at.
    """
    if not isinstance(delivered, dict) or not isinstance(timeline, FilmTimeline):
        return delivered
    delivered = dict(delivered)
    try:
        cut_indices, hold_spans = timeline.resolve()
        delivered["continuity"] = analyze_rendered_continuity(
            output, fps=fps, cut_indices=cut_indices, hold_spans=hold_spans
        )
    except Exception as error:  # noqa: BLE001 -- record-only, never raise
        delivered["continuity"] = {"error": str(error)}
    return delivered


def _review_motion_clips(
    pack_dir: Path | None,
    review_pack_kwargs: dict | None,
    rendered_poses: Sequence[object] | None,
    fps: float,
    video: Path,
    *,
    runner: Runner = subprocess.run,
) -> None:
    """Cut continuous clips of the delivered film's most hectic camera motion
    into its review pack, so a reviewer watches the motion instead of
    inferring it from stills.

    No-op when the pack was never requested (``pack_dir``/``review_pack_kwargs``
    are ``None`` -- there is no operator to see the clips) or the renderer
    exposed fewer than two rendered poses to measure motion from. A cutting
    failure is recorded into the pack as a single ``motion_clips_error``
    entry, never raised -- a reviewer without clips can still review; a
    reviewer without any pack at all cannot.
    """
    if pack_dir is None or review_pack_kwargs is None:
        return
    if rendered_poses is None or len(rendered_poses) < 2:
        return
    try:
        report = motion_report_from_poses(rendered_poses, fps)
        windows = hectic_windows(report, fps)
        clips = write_motion_clips(video, pack_dir, windows, runner=runner)
    except Exception as error:  # noqa: BLE001 -- record-only, never raise
        clips = [{"motion_clips_error": str(error)}]
    write_review_pack(pack_dir, motion_clips=clips, **review_pack_kwargs)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a private Route Story export from explicitly supplied local assets."""
    parser = argparse.ArgumentParser(description="Render an evidence-led Route Story film")
    parser.add_argument("--gpx", type=Path, required=True)
    parser.add_argument(
        "--recorded-on", required=True, help="Selected GPX calendar date (YYYY-MM-DD)"
    )
    parser.add_argument("--photos-dir", type=Path, required=True)
    parser.add_argument("--national-raster", type=Path, required=True)
    parser.add_argument("--national-credit", required=True)
    parser.add_argument("--local-raster", type=Path, required=True)
    parser.add_argument("--local-credit", required=True)
    parser.add_argument(
        "--local-bounds",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        required=True,
    )
    parser.add_argument("--regional-raster", action="append", type=Path, default=[])
    parser.add_argument("--regional-credit", action="append", default=[])
    parser.add_argument(
        "--regional-bounds",
        action="append",
        nargs=4,
        type=float,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        default=[],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument(
        "--public-receipt",
        type=Path,
        help="Also write a distribution-safe copy of the receipt here -- imagery "
        "reduced to credit only, map labels reduced to a count, photos reduced "
        "to a count. Requires --receipt and a nonzero --trim-start-m/"
        "--trim-end-m: a route with no explicit privacy trim has nothing safe "
        "to publish",
    )
    parser.add_argument("--source-video", type=Path)
    parser.add_argument(
        "--still-source",
        choices=STILL_SOURCES,
        default="developed",
        help="RAW photo holds: the service's developed grade (default) or the "
        "camera's own embedded full-size rendering",
    )
    parser.add_argument("--labels-json", type=Path)
    parser.add_argument("--dem-json", type=Path)
    parser.add_argument("--fps", type=int, default=DELIVERY_FPS)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default="full",
        help="Editorial defaults for delivery size and pacing: 'full' (today's "
        "16:9 defaults, unchanged) or 'social' (a 9:16 short-form cut capped "
        f"at {PRESETS['social'].max_seconds:.0f}s). Explicit --width/--height/"
        "--photo-seconds/--map-seconds always override the preset.",
    )
    # Sentinel default of None (never the literal it used to hard-code): the
    # preset fills an unset value below (apply_preset), and an explicit flag
    # always wins -- both need to tell "not passed" apart from "passed the
    # same number the old default happened to be".
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--photo-seconds", type=float, default=None)
    parser.add_argument(
        "--no-best-pick",
        action="store_true",
        help="Keep every photo instead of showing only the best of each "
        "look-alike group (default: on -- see --keep to rescue a specific loser)",
    )
    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="NAME",
        help="Keep this photo even if its look-alike group has a different "
        "winner (repeatable; a file name in --photos-dir, case-insensitive)",
    )
    parser.add_argument(
        "--map-seconds",
        type=float,
        default=None,
        help="Total on-screen seconds across every map leg combined, under one "
        "honest, globally proportional time-compression factor (default: the "
        "old per-leg reading-time clock's sum for the same legs)",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=None,
        help="Deliver a film of this length (e.g. 90): intro, recap and stills "
        "keep their seconds and the map legs share the rest, under the same "
        "proportional compression as --map-seconds (a placed source video's "
        "clip is added on top). Mutually exclusive with --map-seconds",
    )
    parser.add_argument("--max-interpolation-gap", type=float, default=60.0)
    parser.add_argument(
        "--utc-offset",
        type=_utc_offset_arg,
        default=None,
        help="Fallback UTC offset (e.g. -04:00 or +05:30) for the dashboard TIME "
        "cell, used only when the photos' zoned EXIF offsets disagree or none "
        "exist; with neither, the TIME cell is omitted rather than guessed",
    )
    parser.add_argument(
        "--renderer",
        choices=("pillow", "scene"),
        default="pillow",
        help="Map-leg renderer: the flat Pillow satellite composite, or the "
        "offline MapLibre terrain scene (requires --bundle)",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="Offline scene bundle root from 'route-story prepare' (required with --renderer scene)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel scene-capture workers for --renderer scene (default: "
        "GMS_ROUTE_STORY_RENDER_WORKERS or half the box's cores, capped at 6)",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="The trip's display name, burned over the opening camera. Omitted, "
        "the card names the closest places in --labels-json to the track's start "
        "and end plus the recording date, and stays blank when neither is known",
    )
    parser.add_argument(
        "--preflight-frames",
        type=int,
        default=12,
        help="Frames sampled across the whole story and QA-checked before the "
        "long render starts (0 disables; diagnostic only, not a film substitute)",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Render the preflight contact sheet, run its QA, and exit without "
        "the long film render",
    )
    parser.add_argument(
        "--review-pack",
        action="store_true",
        help="Write the operator review pack (sheets, JSON, and the delivered "
        "film's hectic-motion clips) on a full render too, not just "
        "--preflight-only; requires --preflight-frames > 0 or --preflight-only",
    )
    parser.add_argument(
        "--allow-unplaced",
        action="store_true",
        help="Proceed when a photo or the source video has no defensible route "
        "position (default: refuse before any render starts, naming every "
        "unplaced file); unplaced media is never drawn either way",
    )
    parser.add_argument(
        "--no-display-smoothing",
        action="store_true",
        help="Draw the raw cleaned GPS line instead of the display-only "
        "smoothed one (default: smoothing on). Never changes a measurement "
        "-- distance, pace and every other dashboard number always come "
        "from the recorded points either way.",
    )
    parser.add_argument(
        "--watermark",
        default=DEFAULT_WATERMARK_TEXT,
        help="Maker's mark burned bottom-right on every delivered frame "
        f"(default: {DEFAULT_WATERMARK_TEXT!r}); see --no-watermark to omit it",
    )
    parser.add_argument(
        "--no-watermark",
        action="store_true",
        help="Omit the maker's mark entirely (default: on)",
    )
    parser.add_argument(
        "--skip-blur",
        action="store_true",
        default=False,
        dest="skip_blur",
        help="DIAGNOSTIC ESCAPE: composite stills without blurring detected "
        "faces/plates/occupants. Blur is ON by default; this ships identifiable "
        "people — never use for a public film.",
    )
    parser.add_argument(
        "--no-bookends",
        action="store_true",
        help="Treat a same-day photo shot just outside the GPX recording as "
        "ordinary unplaced media -- refused or dropped like any other -- "
        "instead of showing it as a bookend (default: after-recording "
        f"bookends on, within {BOOKEND_MAX_OUTSIDE_S / 3600:.0f}h of the "
        "recording; before-recording bookends are ordinary unplaced media "
        "either way unless --pre-trip-photos is also passed)",
    )
    parser.add_argument(
        "--pre-trip-photos",
        action="store_true",
        help="Show a same-day photo shot just before the recording started "
        "as a bookend (default: excluded -- a driving film shouldn't open "
        "with pre-trip photos; excluded names are recorded in the receipt). "
        "No effect together with --no-bookends, which excludes it either way",
    )
    parser.add_argument(
        "--repair",
        type=Path,
        default=None,
        metavar="QA_FAULTS_JSON",
        help="Re-render only the frames named in a prior QA-refused render's "
        "<output-stem>.qa-faults.json, from the kept frames it left behind "
        "(never a full re-render). Requires --bundle (repair recaptures "
        "through the scene renderer) and refuses loudly on a missing kept "
        "directory, an identity mismatch against this run's --output/--fps/"
        "--width/--height, or a multi-map-leg render, where a frame number "
        "is not unique. Every other required flag (--gpx, --photos-dir, etc.) "
        "still has to parse but is otherwise unused -- a repair run never "
        "touches the GPX, photos, or rasters again",
    )
    parser.add_argument(
        "--thumbnail",
        type=Path,
        help="Write a YouTube thumbnail candidate to this path (full preset "
        "only -- a social 9:16 cut has no long-form upload to sit beside). "
        "A generator that can't produce one is recorded on the receipt "
        "rather than failing the export.",
    )
    parser.add_argument(
        "--credit",
        action="append",
        default=[],
        metavar="TEXT",
        help="Extra imagery/attribution line for the --metadata description "
        "(repeatable, in addition to the national/regional/local raster "
        "credits already on the receipt)",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Write built upload metadata (title/description/tags) as JSON "
        "to this path, checked against the published-titles ledger before "
        "writing so a duplicate title fails the run rather than publishing",
    )
    parser.add_argument(
        "--activity",
        choices=["hiking", "cycling", "driving", "auto"],
        default="auto",
        help="The trip's activity -- drives the display speed filter's max "
        "plausible speed and (absent --title) the derived metadata title/"
        "tags (default: auto, guessed from the route's moving-average "
        "speed like the metadata builder has always done)",
    )
    parser.add_argument(
        "--units",
        choices=["metric", "imperial", "auto"],
        default="auto",
        help="Display units carried in the receipt (default: auto, guessed "
        "from the route's region -- imperial in the US/UK, metric "
        "elsewhere)",
    )
    parser.add_argument(
        "--pacing-mode",
        choices=PACING_MODE_CHOICES,
        default="recorded",
        help="The presentation-timing table this render used (default: "
        "recorded, the only implemented value today). Disclosed on the "
        "receipt as pacing_mode alongside presentation_mapping_version, so "
        "a re-render under a different pacing table is distinguishable "
        "from a re-render under a different --activity/--units table",
    )
    parser.add_argument(
        "--trim-start-m",
        type=float,
        default=0.0,
        help="drop route points within N metres of the recorded start/end "
        "before anything is drawn or summarized",
    )
    parser.add_argument(
        "--trim-end-m",
        type=float,
        default=0.0,
        help="drop route points within N metres of the recorded start/end "
        "before anything is drawn or summarized",
    )
    parser.add_argument(
        "--privacy-zone",
        dest="privacy_zone",
        type=_privacy_zone_arg,
        action="append",
        default=[],
        metavar="LAT,LON,RADIUS_M",
        help="mask an interior privacy zone: drop every route point within "
        "RADIUS_M metres of LAT,LON before anything is drawn or summarized "
        "(repeatable, applied right after --trim-start-m/--trim-end-m)",
    )
    args = parser.parse_args(argv)
    if args.public_receipt is not None:
        # Refuse before anything else runs -- including the trim itself,
        # which can be expensive on a long GPX -- so an operator can never
        # end up with a half-written public artifact for an untrimmed route.
        if args.receipt is None:
            parser.error("--public-receipt requires --receipt")
        if args.trim_start_m == 0.0 and args.trim_end_m == 0.0:
            parser.error(
                "--public-receipt requires a nonzero --trim-start-m/--trim-end-m "
                "(an untrimmed route has nothing safe to publish)"
            )
    configure_progress_logging()
    if args.repair is not None and args.bundle is None:
        parser.error("--repair requires --bundle (repair recaptures via the scene renderer)")
    if args.renderer == "scene" and args.bundle is None:
        parser.error("--bundle is required when --renderer scene")
    if args.map_seconds is not None and args.map_seconds <= 0:
        parser.error("--map-seconds must be positive")
    if args.target_seconds is not None:
        if args.target_seconds <= 0:
            parser.error("--target-seconds must be positive")
        if args.map_seconds is not None:
            parser.error("--target-seconds and --map-seconds are mutually exclusive")
    if args.review_pack and args.preflight_frames <= 0 and not args.preflight_only:
        parser.error(
            "--review-pack requires --preflight-frames > 0 or --preflight-only "
            "(there is nothing rendered yet to build a pack from otherwise)"
        )
    display_smoothing = not args.no_display_smoothing
    # Resolved AFTER the --map-seconds/--target-seconds exclusivity checks
    # above: those need to tell "the operator passed --map-seconds" apart
    # from "unset", and a preset fill would make every social run look like
    # the operator passed it.
    preset = PRESETS[args.preset]
    args.width, args.height, args.photo_seconds, args.map_seconds = apply_preset(
        preset,
        width=args.width,
        height=args.height,
        photo_seconds=args.photo_seconds,
        map_seconds=args.map_seconds,
    )
    intro_seconds = preset.intro_seconds
    recap_seconds = preset.recap_seconds

    if args.repair is not None:
        # A repair never touches the GPX/photos/rasters pipeline below --
        # it recaptures named poses straight through the scene renderer and
        # re-encodes the kept frames a QA-refused render already left
        # behind. NOTE (named limitation, not a bug): the recaptured frame
        # carries the map + labels only, never the dashboard overlay --
        # that overlay depends on timeline/cadence state a repair-only
        # invocation has no way to reconstruct without replaying the whole
        # render. A faulted frame is virtually always a map-tile artifact
        # the dashboard drawn on top of it does not explain, so this is an
        # acceptable, disclosed gap rather than a silent visual regression.
        from .scene.capture import SceneCapture
        from .scene_renderer import scene_map_height

        map_height = scene_map_height(args.height)
        with SceneCapture(args.bundle, args.width, map_height) as scene_capture:

            def recapture(pose: dict, frame_index: int) -> Image.Image:
                return _repair_scene_frame(
                    scene_capture,
                    pose,
                    frame_index=frame_index,
                    width=args.width,
                    height=args.height,
                )

            try:
                repair_route_story_film(
                    args.repair,
                    output=args.output,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    watermark=(None if args.no_watermark else args.watermark),
                    recapture=recapture,
                )
            except FrameQaError as error:
                for path in save_refusal_evidence(error, args.output):
                    logger.error("repair refused; evidence at %s", path)
                raise
        return 0

    selection = select_gpx_day(args.gpx, date.fromisoformat(args.recorded_on))
    try:
        trim_result = trim_route(selection.segments, RouteTrim(args.trim_start_m, args.trim_end_m))
    except RouteTrimError as error:
        parser.error(str(error))
    # The trimmed route is the ONLY route the film ever sees from here on --
    # every downstream surface (activity/units, pacing, render, receipt)
    # reads ``selection.segments``, never the untrimmed source.
    selection = replace(selection, segments=trim_result.segments)
    zone_result = None
    if args.privacy_zone:
        try:
            zone_result = mask_zones(selection.segments, args.privacy_zone)
        except RouteTrimError as error:
            parser.error(str(error))
        # Same rule as the start/end trim above: every downstream surface
        # reads the zone-masked segments, never the pre-mask route.
        selection = replace(selection, segments=zone_result.segments)
    sources = [
        path
        for path in sorted(args.photos_dir.iterdir())
        if path.suffix.lower()
        in {".nef", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif"}
    ]
    available_names = {source.name.lower(): source.name for source in sources}
    keep_names: set[str] = set()
    for raw_name in args.keep:
        matched = available_names.get(raw_name.lower())
        if matched is None:
            parser.error(
                f"--keep {raw_name!r} is not a photo in {args.photos_dir}; "
                f"available: {', '.join(sorted(available_names.values())) or '(none)'}"
            )
        keep_names.add(matched)
    photos = []
    photo_utc_offsets: list[str | None] = []
    for source in sources:
        metadata = extract_exif_capture_metadata(source)
        if metadata.capture_utc is None:
            raise ValueError(f"{source.name} has no explicitly zoned capture time")
        wall_local = extract_exif_capture_timestamp(source)
        photo_utc_offsets.append(
            _offset_from_capture(wall_local, metadata.capture_utc)
            if wall_local is not None
            else None
        )
        gps = extract_exif_gps(source)
        exif_route_seconds = captured_at_route_seconds(
            metadata.capture_utc, selection.source_started_at
        )
        evidence = MediaEvidence(
            source.name,
            captured_at_s=exif_route_seconds,
            latitude=gps[0] if gps else None,
            longitude=gps[1] if gps else None,
        )
        photos.append(
            PhotoMoment(
                source,
                anchor_media_to_segments(
                    evidence,
                    selection.segments,
                    max_interpolation_gap_s=args.max_interpolation_gap,
                ),
                captured_at_exif_s=exif_route_seconds,
            )
        )
    # Blur + EXIF-strip every still into a working copy BEFORE any is
    # composited or exported -- the film reads only the gated copies from
    # here on, never an original photo path (privacy.apply_privacy_gate).
    privacy_result = apply_privacy_gate(
        photos, blur=(False if args.skip_blur else None), work_dir=_staging_root()
    )
    photos = list(privacy_result.photos)
    west, south, east, north = args.local_bounds
    national = SatelliteRaster(args.national_raster, -180, -90, 180, 90, args.national_credit)
    local = SatelliteRaster(args.local_raster, west, south, east, north, args.local_credit)
    if not (len(args.regional_raster) == len(args.regional_credit) == len(args.regional_bounds)):
        raise ValueError("every regional raster needs matching bounds and a credit")
    regional = tuple(
        SatelliteRaster(path, *bounds, credit)
        for path, credit, bounds in zip(
            args.regional_raster, args.regional_credit, args.regional_bounds, strict=True
        )
    )
    labels = load_map_labels(args.labels_json) if args.labels_json is not None else ()
    terrain = TerrainGrid.from_json(args.dem_json) if args.dem_json is not None else None
    video_anchor = None
    if args.source_video is not None:
        video_anchor = anchor_media_to_segments(
            _video_evidence(args.source_video, selection.source_started_at),
            selection.segments,
            max_interpolation_gap_s=args.max_interpolation_gap,
        )

    # Same-day photos shot just outside the recording are bookends, not
    # unplaced media -- classify_bookend is the single rule; --no-bookends
    # skips this and every non-mapped photo goes back to ordinary unplaced
    # handling below.
    track_points = [point for segment in selection.segments for point in segment]
    before_bookends: list[PhotoMoment] = []
    after_bookends: list[PhotoMoment] = []
    bookend_class: dict[str, tuple[str, float]] = {}
    excluded_pretrip_photos: list[str] = []
    if not args.no_bookends and track_points:
        track_start_s = min(point.time_s for point in track_points)
        track_end_s = max(point.time_s for point in track_points)
        for photo in photos:
            classification = classify_bookend(photo, track_start_s, track_end_s)
            if classification is None:
                continue
            if classification == "before_recording" and not args.pre_trip_photos:
                # Excluded by default: a deliberate exclusion, not unplaced
                # media, so the default run still renders without
                # --allow-unplaced. Named on the receipt so the operator sees
                # what was left out; --pre-trip-photos brings them back.
                excluded_pretrip_photos.append(photo.source.name)
                continue
            captured_at = photo.captured_at_exif_s or 0.0
            outside_s = (
                track_start_s - captured_at
                if classification == "before_recording"
                else captured_at - track_end_s
            )
            bookend_class[photo.source.name] = (classification, outside_s)
            target = before_bookends if classification == "before_recording" else after_bookends
            target.append(photo)
        before_bookends.sort(key=lambda photo: photo.captured_at_exif_s or 0.0)
        after_bookends.sort(key=lambda photo: photo.captured_at_exif_s or 0.0)
    excluded_pretrip_photos.sort()
    bookend_names = set(bookend_class)

    unplaced_media: list[dict[str, object]] = [
        {
            "media_id": photo.anchor.media_id,
            "anchor_status": photo.anchor.status,
            "reason": "no defensible route position",
        }
        for photo in photos
        if not photo.is_mapped
        and photo.source.name not in bookend_names
        and photo.source.name not in excluded_pretrip_photos
    ]
    if args.source_video is not None and not _video_is_placed(args.source_video, video_anchor):
        unplaced_media.append(
            {
                "media_id": args.source_video.name,
                "anchor_status": video_anchor.status if video_anchor else None,
                "reason": "no defensible route position",
            }
        )
    if unplaced_media:
        names = ", ".join(
            f"{item['media_id']} ({item['anchor_status']})" for item in unplaced_media
        )
        if not args.allow_unplaced:
            parser.error(
                f"unplaced media would be left out of the film: {names}; "
                "pass --allow-unplaced to render without it"
            )
        logger.warning("unplaced media left out of the film: %s", names)

    # Best-of-look-alike-group selection runs over already-placed photos plus
    # bookends -- both are selection candidates (bookends with no placement,
    # so a look-alike pair of bookends still collapses to the best one). A
    # truly unplaced photo is refused or logged above and never reaches here.
    mapped_photos = [photo for photo in photos if photo.is_mapped]
    not_considered = [
        {"name": photo.source.name, "anchor_status": photo.anchor.status}
        for photo in photos
        if not photo.is_mapped and photo.source.name not in bookend_names
    ]
    selection_options = SelectionOptions(enabled=not args.no_best_pick, keep=frozenset(keep_names))
    kept_all, selection_result = select_kept_photos(
        mapped_photos + before_bookends + after_bookends,
        still_source=args.still_source,
        options=selection_options,
    )
    kept_names_set = {photo.source.name for photo in kept_all}
    kept_mapped = [photo for photo in mapped_photos if photo.source.name in kept_names_set]
    kept_before_bookends = [
        photo for photo in before_bookends if photo.source.name in kept_names_set
    ]
    kept_after_bookends = [photo for photo in after_bookends if photo.source.name in kept_names_set]
    render_photos = [
        photo
        for photo in photos
        if (not photo.is_mapped and photo.source.name not in bookend_names)
        or photo.source.name in kept_names_set
    ]
    bookend_receipt_info = {
        name: info for name, info in bookend_class.items() if name in kept_names_set
    }
    bookend_play_order = [photo.source.name for photo in kept_before_bookends] + [
        photo.source.name for photo in kept_after_bookends
    ]
    if args.target_seconds is not None:
        # Solved once, here, from the final kept photos; every later reader of
        # args.map_seconds (preflight, pacing, the film, the receipt) then sees
        # one number, exactly as if the operator had passed --map-seconds.
        video_seconds = 0.0
        if _video_is_placed(args.source_video, video_anchor):
            # The clip plays whole, plus its arrival hold; both count toward
            # the delivered length the operator asked for.
            clip_seconds = FFprobeClient(subprocess).probe_duration(args.source_video)
            if clip_seconds <= 0:
                parser.error(f"--target-seconds cannot read {args.source_video}'s duration")
            video_seconds = clip_seconds + VIDEO_ARRIVAL_HOLD_SECONDS
        try:
            args.map_seconds, _ = solve_map_seconds(
                target_seconds=args.target_seconds,
                kept_count=len(kept_mapped) + len(kept_before_bookends) + len(kept_after_bookends),
                mapped_count=len(kept_mapped),
                requested_photo_seconds=args.photo_seconds,
                intro_seconds=intro_seconds,
                recap_seconds=recap_seconds,
                extra_seconds=video_seconds,
            )
        except ValueError as error:
            parser.error(str(error))
        logger.info(
            "target %.1fs: map legs get %.1fs combined", args.target_seconds, args.map_seconds
        )

    def photo_selection_block() -> dict[str, object]:
        return {
            **selection_result.to_dict(),
            "enabled": not args.no_best_pick,
            "kept_override": sorted(keep_names),
            "capture_time_source": "exif",
            "not_considered": not_considered,
        }

    # The pacing cap needs the map seconds this run will actually use --
    # mirrors render_route_story_film's own leg-bounds/plan_map_legs call so
    # the estimate matches what gets rendered, using the final kept photos.
    # The located video's own duration is not cheaply knowable before an
    # encode, so it is left out of the "other scenes" denominator.
    pacing_points = [point for segment in selection.segments for point in segment]
    pacing_display_segments = clean_display_segments(
        selection.segments, max_speed_mps=8.0, max_gap_s=90.0
    )
    observed_max_speed_mps = (
        summarize_track(pacing_display_segments, moving_speed_mps=0.5).moving_average_speed_mps
        if pacing_display_segments
        else None
    )
    activity_profile = resolve_activity(args.activity, observed_max_speed_mps)
    activity_source = "explicit" if args.activity != "auto" else "auto"
    units, units_source = resolve_units(args.units, track_centroid(selection.segments))
    logger.info("route-story units: %s (%s)", units, units_source)
    if args.metadata is not None:
        # The metadata module is imported lazily and guarded: an unavailable
        # module just skips this early check -- the post-render
        # check_title_against_ledger below still gates the write either way.
        try:
            from .metadata import DuplicateTitleError, LedgerUnreadableError, preflight_title
        except ImportError:
            pass
        else:
            try:
                # Same names/activity the post-render metadata build will
                # use (map_labels.names is just this labels list, unchanged
                # by the render) -- decide a duplicate/unverifiable title
                # BEFORE the long render, not after.
                preflight_title(
                    recorded_on=args.recorded_on,
                    title=args.title,
                    activity=(activity_profile.name if activity_source == "explicit" else None),
                    names=[label.name for label in labels],
                )
            except (DuplicateTitleError, LedgerUnreadableError) as error:
                parser.error(str(error))
    # Scoped for the whole render, same module-state shape as the dashboard
    # clock/motion-cell settings above -- reset in the ``finally`` below so a
    # lone ``render()``/single-frame call outside this CLI path still draws
    # with the legacy 8.0 m/s / 30.0% caps and imperial units.
    set_active_display_caps(
        DisplayCaps(activity_profile.hud_speed_cap_mps, activity_profile.grade_cap_percent)
    )
    set_active_display_units(units)
    if pacing_points and pacing_display_segments:
        pacing_start = min(point.time_s for point in pacing_points)
        pacing_end = pacing_display_segments[-1][-1].time_s
        pacing_leg_bounds = _map_leg_bounds(
            pacing_start, pacing_end, kept_mapped, args.source_video, video_anchor
        )
        _, pacing_frame_counts = plan_map_legs(
            pacing_leg_bounds, args.fps, map_seconds=args.map_seconds
        )
        used_map_seconds = sum(pacing_frame_counts) / args.fps
    else:
        used_map_seconds = args.map_seconds or 0.0
    dashboard_utc_offset, dashboard_time_cell = resolve_dashboard_utc_offset(
        photo_utc_offsets, args.utc_offset
    )
    dashboard_motion_mode = motion_cell_mode(observed_max_speed_mps)
    # Trim kept stills DOWN TO what the 35% cap allows even at the floor --
    # robust by construction, so floor_hit below is an unreachable guard in a
    # normal run rather than a warning the film ships anyway. Bookends carry
    # no route position of their own, unlike a mapped still, so a bookend is
    # dropped before any mapped still; within each tier the lowest-sharpness
    # (worst-ranked) survivors go first, using the scores select_kept_photos
    # already computed for every candidate.
    dropped_for_cap: list[str] = []
    cap_limit = capped_still_count(
        preset,
        cap_limit=max_stills_for_cap(
            other_seconds=intro_seconds + recap_seconds + used_map_seconds
        ),
    )
    all_kept_stills = kept_mapped + kept_before_bookends + kept_after_bookends
    if len(all_kept_stills) > cap_limit:
        candidate_scores = {
            name: score for group in selection_result.groups for name, score in group.scores.items()
        }
        bookend_pool = sorted(
            kept_before_bookends + kept_after_bookends,
            key=lambda photo: candidate_scores[photo.source.name].sharpness,
        )
        mapped_pool = sorted(
            kept_mapped, key=lambda photo: candidate_scores[photo.source.name].sharpness
        )
        drop_count = len(all_kept_stills) - cap_limit
        dropped_names = {photo.source.name for photo in (bookend_pool + mapped_pool)[:drop_count]}
        dropped_for_cap = sorted(dropped_names)
        kept_mapped = [photo for photo in kept_mapped if photo.source.name not in dropped_names]
        kept_before_bookends = [
            photo for photo in kept_before_bookends if photo.source.name not in dropped_names
        ]
        kept_after_bookends = [
            photo for photo in kept_after_bookends if photo.source.name not in dropped_names
        ]
        render_photos = [photo for photo in render_photos if photo.source.name not in dropped_names]
        bookend_receipt_info = {
            name: info for name, info in bookend_receipt_info.items() if name not in dropped_names
        }
        bookend_play_order = [name for name in bookend_play_order if name not in dropped_names]
    # intro_seconds/recap_seconds are the SAME values the render_route_story_film
    # call below actually uses (passed explicitly there now) -- both come
    # from the one resolved preset, so this reading and that render can
    # never disagree.
    pacing = resolve_photo_seconds(
        requested_seconds=args.photo_seconds,
        kept_count=len(kept_mapped) + len(kept_before_bookends) + len(kept_after_bookends),
        intro_seconds=intro_seconds,
        map_seconds=used_map_seconds,
        recap_seconds=recap_seconds,
    )
    pacing["dropped_for_cap"] = dropped_for_cap
    if pacing["floor_hit"]:
        logger.warning(
            "stills would take %.0f%% of runtime even at the %.1fs floor (cap %.0f%%); "
            "keeping the floor",
            pacing["stills_share"] * 100,
            PHOTO_SECONDS_FLOOR,
            STILLS_SHARE_CAP * 100,
        )
    photo_seconds = pacing["used_seconds"]

    # Robust by construction: a preset with a length ceiling (social) is
    # refused BEFORE any render starts, not after an irreversible encode --
    # the same duration probe the --target-seconds path above already uses
    # for a placed video's own contribution.
    ceiling_video_seconds = 0.0
    if _video_is_placed(args.source_video, video_anchor):
        ceiling_clip_seconds = FFprobeClient(subprocess).probe_duration(args.source_video)
        if ceiling_clip_seconds > 0:
            ceiling_video_seconds = ceiling_clip_seconds + VIDEO_ARRIVAL_HOLD_SECONDS
    planned_seconds = (
        intro_seconds
        + recap_seconds
        + used_map_seconds
        + photo_seconds * (len(kept_mapped) + len(kept_before_bookends) + len(kept_after_bookends))
        + ceiling_video_seconds
    )
    try:
        check_ceiling(preset, planned_seconds=planned_seconds)
    except FilmTooLongError as error:
        parser.error(str(error))

    map_renderer = None
    scene_capture = None
    if args.renderer == "scene":
        from .proof import _resolve_workers
        from .scene.capture import SceneCapture
        from .scene_renderer import SceneMapRenderer, scene_map_height

        manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
        max_zoom = max_shot_zoom(float(manifest["imagery"]["maxzoom"]))
        scene_capture = SceneCapture(
            args.bundle, args.width, scene_map_height(args.height)
        ).__enter__()
        resolved_workers = _resolve_workers(args.workers)
        map_renderer = SceneMapRenderer(
            capture=scene_capture,
            trip_epoch=selection.source_started_at,
            max_zoom=max_zoom,
            credit=args.local_credit,
            fps=args.fps,
            workers=resolved_workers,
            capture_factory=(
                (lambda: SceneCapture(args.bundle, args.width, scene_map_height(args.height)))
                if resolved_workers > 1
                else None
            ),
        )
    if map_renderer is None:
        map_renderer = PillowMapRenderer(
            rasters=(national, *regional, local), labels=labels, terrain=terrain
        )

    # Computed once, before preflight and again for the receipt below, so a
    # preflight frame can never sample a different line than the film draws
    # (or than the receipt reports it drew).
    display_cleaned, display_smoothed_track = drawn_segments(
        selection.segments, smooth=display_smoothing, max_speed_mps=activity_profile.max_speed_mps
    )
    display_smoothing_block = _display_smoothing_receipt(
        display_smoothed_track, enabled=display_smoothing
    )
    preflight_draw_segments = (
        display_smoothed_track.segments
        if display_smoothed_track is not None
        else display_cleaned or selection.segments
    )
    # The preflight sheet's rail reads the recorded points too, like the film's.
    preflight_renderer = (
        map_renderer.measuring(display_cleaned)
        if display_smoothed_track is not None
        else map_renderer
    )

    preflight_result = None
    review_pack_dir: Path | None = None
    review_pack_kwargs: dict[str, object] | None = None
    opening_dir = tempfile.TemporaryDirectory(prefix="route-story-opening-", dir=_staging_root())
    opening_pyramid: list[dict[str, object]] = []
    try:
        # Computed up front (rather than beside the render below) so the
        # preflight gate can check the SAME opening imagery -- and the
        # camera it actually drives -- before the long render starts.
        opening_rasters = (
            bundle_opening_rasters(args.bundle, Path(opening_dir.name))
            if args.bundle is not None
            else ()
        )
        opening_pyramid = opening_pyramid_receipt(opening_rasters)
        if args.preflight_frames > 0 or args.preflight_only:
            points = [point for segment in selection.segments for point in segment]
            if points:
                preflight_sheet = args.output.with_name(args.output.stem + "-preflight.jpg")
                # Only with the pack: the marker close-ups are a thing an
                # operator reads, and an operator opts in with --review-pack
                # on a full render (there is none by default otherwise).
                marker_sheet_path: Path | None = None
                if args.preflight_only or args.review_pack:
                    # Before the frame checks, not after: a run they refuse is
                    # the one the operator most needs the pack for.
                    pack_dir = args.output.with_name(args.output.stem + "-review-pack")
                    pack_inputs = gather_review_inputs(
                        pack_dir,
                        # The same raster set the film's own opening renders
                        # from (below), never a subset: fed only the flat CLI
                        # rasters, this sheet showed an opening the film would
                        # not produce -- the one gate built to catch a bad
                        # establishing shot, blind to it by construction.
                        rasters=finest_last((national, *regional, local, *opening_rasters)),
                        first_point=points[0],
                        labels=labels,
                        points=points,
                        photos=photos,
                        kept_mapped=kept_mapped,
                        source_video=args.source_video,
                        video_placed=_video_is_placed(args.source_video, video_anchor),
                        video_clock_s=video_anchor.track_time_s if video_anchor else None,
                        width=args.width,
                        height=args.height,
                        still_source=args.still_source,
                        before_bookends=kept_before_bookends,
                        after_bookends=kept_after_bookends,
                        dropped_bookend_names=bookend_names - set(bookend_play_order),
                    )
                    marker_sheet_path = pack_dir / MARKER_SHEET_NAME
                    review_pack_kwargs = {
                        "selection": photo_selection_block(),
                        "preflight_sheet": preflight_sheet,
                        "marker_sheet": marker_sheet_path,
                        **pack_inputs,
                    }
                    write_review_pack(pack_dir, **review_pack_kwargs)
                    review_pack_dir = pack_dir
                    print(pack_dir)
                # After the review pack is written, never before: this gate
                # raises on unsharp opening imagery, and a run it refuses is
                # exactly the one the operator most needs the pack for.
                # Only checked/sampled with a bundle's own imagery pyramid:
                # the flat --national-raster/--local-raster fallback is a
                # lower-fidelity legacy path this sharpness floor was never
                # tuned against, and the film that prompted this gate used
                # --bundle (see docs/pipeline-internals.md).
                opening_frames: list[Image.Image] = []
                if opening_rasters:
                    opening_set = finest_last((national, *regional, local, *opening_rasters))
                    check_opening_sharpness(opening_set, points[0], args.width, args.height)
                    # The card the film will burn, on the frames the operator
                    # reviews -- a pack showing an untitled opening would
                    # approve something the render does not produce.
                    preview_title, preview_subtitle = resolve_opening_title(
                        args.title,
                        labels,
                        start=(points[0].latitude, points[0].longitude),
                        end=(points[-1].latitude, points[-1].longitude),
                        recorded_on=date.fromisoformat(args.recorded_on),
                    )
                    opening_frames = [
                        _opening_frame(
                            opening_set,
                            points[0],
                            fraction,
                            args.width,
                            args.height,
                            labels,
                            title=preview_title,
                            subtitle=preview_subtitle,
                        )
                        for fraction in OPENING_LADDER_FRACTIONS
                    ]
                preflight_result = run_preflight(
                    preflight_renderer,
                    preflight_draw_segments,
                    start_time=min(point.time_s for point in points),
                    end_time=max(point.time_s for point in points),
                    width=args.width,
                    height=args.height,
                    count=args.preflight_frames,
                    output=preflight_sheet,
                    opening_frames=opening_frames,
                    marker_sheet=marker_sheet_path,
                )
                print(preflight_result.sheet)
        if args.preflight_only:
            if args.receipt is not None:
                partial_receipt: dict[str, object] = {
                    "photo_selection": photo_selection_block(),
                    "stills_pacing": pacing,
                    "review_pack": str(args.output.with_name(args.output.stem + "-review-pack")),
                    "bookend_photos": bookend_play_order,
                    "excluded_pretrip_photos": excluded_pretrip_photos,
                    "display_smoothing": display_smoothing_block,
                }
                if preflight_result is not None:
                    partial_receipt["preflight"] = {
                        "frames": preflight_result.frames,
                        "passed": preflight_result.passed,
                        "sheet": str(preflight_result.sheet) if preflight_result.sheet else None,
                    }
                args.receipt.parent.mkdir(parents=True, exist_ok=True)
                args.receipt.write_text(
                    json.dumps(partial_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            return 0
        # Assemble to a scratch file beside the output; the delivery encode
        # below writes the named output, so a half-finished render never
        # sits at the path a caller was promised a finished film at.
        assembled = args.output.with_name(args.output.stem + ".assembled" + args.output.suffix)
        timeline = render_route_story_film(
            segments=selection.segments,
            national=national,
            local=local,
            regional=regional,
            photos=render_photos,
            output=assembled,
            source_video=args.source_video,
            source_video_anchor=video_anchor,
            labels=labels,
            terrain=terrain,
            map_renderer=map_renderer,
            opening_rasters=opening_rasters,
            fps=args.fps,
            width=args.width,
            height=args.height,
            intro_seconds=intro_seconds,
            recap_seconds=recap_seconds,
            photo_seconds=photo_seconds,
            map_seconds=args.map_seconds,
            still_source=args.still_source,
            recording_start_utc=selection.source_started_at,
            utc_offset=dashboard_utc_offset,
            before_bookends=kept_before_bookends,
            after_bookends=kept_after_bookends,
            display_smoothing=display_smoothing,
            title=args.title,
            recorded_on=date.fromisoformat(args.recorded_on),
            watermark=None if args.no_watermark else args.watermark,
            max_speed_mps=activity_profile.max_speed_mps,
        )
        try:
            if preset.hevc_delivery:
                encode_final(assembled, args.output, rendition=preset.rendition)
            else:
                encode_final(assembled, args.output)
        finally:
            assembled.unlink(missing_ok=True)
    finally:
        set_active_display_caps(None)
        set_active_display_units(None)
        opening_dir.cleanup()
        # The scene renderer's capture workers are daemon threads holding a
        # browser each, opened once for the whole film -- nothing but this
        # shuts them down at export end.
        if map_renderer is not None and hasattr(map_renderer, "close"):
            map_renderer.close()
        if scene_capture is not None:
            scene_capture.__exit__(None, None, None)
    delivery_spec: dict[str, str] = {}
    if preset.hevc_delivery:
        delivery_spec = {
            "expected_codec": RENDITIONS[preset.rendition].codec,
            "expected_pix_fmt": HEVC_PIXEL_FORMAT,
        }
    delivered = check_delivery(
        args.output, width=args.width, height=args.height, fps=args.fps, **delivery_spec
    )
    delivered = _delivery_continuity(args.output, args.fps, timeline, delivered)
    _review_motion_clips(
        review_pack_dir,
        review_pack_kwargs,
        getattr(map_renderer, "rendered_poses", None),
        args.fps,
        args.output,
    )

    # Record-only check that the delivered file matches the preset's
    # rendition of record -- no second encode; encode_final() above is the
    # only delivery encode. When the preset names no rendition (full), this
    # stays None and the receipt carries no rendition_check key.
    rendition_check = build_rendition_check(preset, delivered)

    # A social (9:16) cut has no long-form upload to sit beside -- only the
    # full preset gets a thumbnail, even when --thumbnail was passed.
    thumbnail_block: dict[str, object] | None = None
    if args.thumbnail is not None and preset.name == "full":
        thumb_title, thumb_subtitle = resolve_opening_title(
            args.title,
            labels,
            start=(track_points[0].latitude, track_points[0].longitude),
            end=(track_points[-1].latitude, track_points[-1].longitude),
            recorded_on=date.fromisoformat(args.recorded_on),
        )
        generated = generate_lf_thumbnail(
            args.output,
            thumb_title,
            thumb_subtitle,
            output_dir=args.thumbnail.parent,
            duration=delivered.get("duration_s") or 0.0,
        )
        if generated is None:
            # The film is still good -- a thumbnail miss is recorded, never
            # an exception that would throw away a finished export.
            thumbnail_block = {"status": "failed"}
        else:
            args.thumbnail.parent.mkdir(parents=True, exist_ok=True)
            Path(generated).replace(args.thumbnail)
            thumbnail_block = {
                "path": args.thumbnail.name,
                "sha256": _sha256_file(args.thumbnail),
            }

    if args.receipt is not None:
        receipt = film_receipt(
            segments=selection.segments,
            national=national,
            local=local,
            regional=regional,
            photos=render_photos,
            output=args.output,
            source_video=args.source_video,
            source_video_anchor=video_anchor,
            labels=labels,
            terrain=terrain,
            map_renderer=map_renderer,
            map_seconds=args.map_seconds,
            bookend_info=bookend_receipt_info,
            bookend_play_order=bookend_play_order,
            max_speed_mps=activity_profile.max_speed_mps,
            activity=activity_profile.name,
            activity_source=activity_source,
            units=units,
            units_source=units_source,
            hud_speed_cap_mps=activity_profile.hud_speed_cap_mps,
            grade_cap_percent=activity_profile.grade_cap_percent,
            mapping_version=MAPPING_VERSION,
            pacing_mode=args.pacing_mode,
            presentation_mapping_version=PRESENTATION_MAPPING_VERSION,
            route_trim=receipt_fragment(trim_result, zone_result)["route_trim"],
        )
        receipt["delivery_check"] = delivered
        # Where the opening card + first map leg end: an LF borrows it as its opening.
        receipt["opening"] = {
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            # a stand-in renderer (tests, a repair-only run) returns no FilmTimeline: no span
            "end_frame": (
                timeline.opening_end_frame if isinstance(timeline, FilmTimeline) else None
            ),
        }
        receipt["photo_selection"] = photo_selection_block()
        receipt["stills_pacing"] = pacing
        receipt["dashboard"] = dashboard_receipt(
            dashboard_motion_mode, dashboard_time_cell, dashboard_utc_offset
        )
        receipt["display_smoothing"] = display_smoothing_block
        receipt["gpx_day_selection"] = {
            "recorded_on": args.recorded_on,
            "selected_segments": len(selection.segments),
            "source_segments": selection.source_segment_count,
            "excluded_segments": selection.excluded_segment_count,
            "excluded_other_date_segments": selection.excluded_other_date_count,
            "excluded_duplicate_segments": selection.excluded_duplicate_count,
        }
        if preflight_result is not None:
            receipt["preflight"] = {
                "frames": preflight_result.frames,
                "passed": preflight_result.passed,
                "sheet": str(preflight_result.sheet) if preflight_result.sheet else None,
            }
        receipt["opening_pyramid"] = opening_pyramid
        receipt["unplaced_media"] = unplaced_media
        receipt["excluded_pretrip_photos"] = excluded_pretrip_photos
        receipt["privacy"] = {
            **privacy_result.attestation,
            "withheld": list(privacy_result.withheld),
        }
        receipt["export"] = {
            "file": args.output.name,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "still_source": args.still_source,
        }
        # No second encode: encode_final() above is the rendition of record.
        # rendition_check (computed from the SAME check_delivery measurement
        # as delivery_check, above) is a record-only confirmation that the
        # delivered file actually matches that rendition -- present only
        # when the preset names one.
        receipt["preset"] = {
            "name": preset.name,
            "size": [args.width, args.height],
            "rendition": preset.rendition,
        }
        if rendition_check is not None:
            receipt["preset"]["rendition_check"] = rendition_check
        if thumbnail_block is not None:
            receipt["thumbnail"] = thumbnail_block
        if args.metadata is not None:
            # The metadata module is imported lazily and guarded: an
            # unavailable module degrades to a receipt note instead of
            # crashing this export.
            try:
                from .metadata import (
                    DuplicateTitleError,
                    LedgerUnreadableError,
                    build_route_story_metadata,
                    check_title_against_ledger,
                )
            except ImportError:
                receipt["metadata"] = {"status": "builder-unavailable"}
            else:
                meta = build_route_story_metadata(
                    receipt,
                    recorded_on=args.recorded_on,
                    title=args.title,
                    credits=args.credit,
                    activity=(activity_profile.name if activity_source == "explicit" else None),
                )
                try:
                    check_title_against_ledger(meta)
                except (DuplicateTitleError, LedgerUnreadableError) as error:
                    # A duplicate/unverifiable title must not silently
                    # publish -- fail the run before the metadata (or
                    # anything downstream of it) is written anywhere.
                    parser.error(str(error))
                args.metadata.parent.mkdir(parents=True, exist_ok=True)
                args.metadata.write_text(
                    json.dumps(asdict(meta), indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                receipt["metadata"] = {
                    "path": args.metadata.name,
                    "title": meta.title,
                    "format": meta.format,
                }
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.public_receipt is not None:
            sanitized = public_receipt(receipt, trim_result)
            args.public_receipt.parent.mkdir(parents=True, exist_ok=True)
            args.public_receipt.write_text(
                json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    print(args.output)
    if delivered["issues"]:
        for issue in delivered["issues"]:
            print(f"delivery check: {issue}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
