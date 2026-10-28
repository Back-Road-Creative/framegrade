"""Route-only map inset: the recorded route and a position marker over bundle imagery.

An opt-in corner overlay for long driving videos, drawn from an offline Route
Story bundle only: "imagery with route" (no network). Road vectors appear only in
the separate opt-in ``roads`` mode, which draws the bundle's ``roads.geojson``
under the route. Privacy comes first: ``trim_route`` and ``mask_zones`` run before
any framing, label or pixel exists. The base image is rendered once; only the
marker changes per frame, following the *source* clock through the edit
timeline (the caller aligns the GPS clock to it).
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw

from .edit_timeline import EditTimeline
from .route_story.bundle import verify_bundle
from .route_story.hud import _font
from .route_story.layers import (
    MapLabel,
    _draw_map_labels,
    _raster_image,
    route_halo_width_px,
    route_line_width_px,
)
from .route_story.metrics import RoutePoint, distance_m
from .route_story.route_privacy import (
    PrivacyZone,
    RouteTrim,
    RouteTrimError,
    mask_zones,
    receipt_fragment,
    trim_route,
)
from .route_story.satellite import SatelliteRaster, bundle_opening_rasters

RECEIPT_SCHEMA = "map-inset-receipt/1"
CORNERS = ("bottom-left", "bottom-right", "top-left", "top-right")
HIDDEN_REASONS = ("insert", "crossfade", "gap")

# Framing margins tried in order; the widest that keeps trimmed-away points out wins.
_MARGINS = (0.12, 0.06, 0.02, 0.0)
_MIN_SPAN_LON = 0.004
_INSET_ASPECT = 0.75  # height / width
_LABEL_CLEARANCE_M = 250.0

_HALO_RGB = "#071219"
_LINE_RGB = "#a2b7be"
_MARKER_RGB = "#f5b73b"
_MARKER_RING_RGB = "#182329"

# Roads mode: muted lines under the route, wider for major classes.
_ROAD_MAJOR_RGB = "#8a9a9f"
_ROAD_MINOR_RGB = "#5f6f74"
ROAD_MAJOR_CLASSES = frozenset(
    {"motorway", "trunk", "primary", "motorway_link", "trunk_link", "primary_link"}
)
ROAD_MAJOR_WIDTH_DIVISOR = 130
ROAD_MINOR_WIDTH_DIVISOR = 300
ROADS_CREDIT = "Roads: © OpenStreetMap contributors"
MODE_ROUTE, MODE_ROADS = "route", "roads"


class MapInsetRefused(ValueError):  # noqa: N818 -- a refusal, named for what the caller sees
    """Raised before anything is rendered or written; the message lists every reason."""

    def __init__(self, reasons: Sequence[str]):
        self.reasons = tuple(reasons)
        super().__init__("map inset refused: " + "; ".join(self.reasons))


@dataclass(frozen=True)
class MarkerState:
    """Where the marker is for one output time, or why it is hidden."""

    visible: bool
    hidden_reason: str | None = None
    source_time_s: float | None = None
    latitude: float | None = None
    longitude: float | None = None


class _RouteIndex:
    """Position at a source time, never interpolated across a segment boundary."""

    def __init__(self, segments: Sequence[Sequence[RoutePoint]]):
        ordered = [sorted(s, key=lambda p: p.time_s) for s in segments if s]
        ordered.sort(key=lambda s: s[0].time_s)
        self._segments = ordered
        self._starts = [s[0].time_s for s in ordered]
        self._times = [[p.time_s for p in s] for s in ordered]

    def locate(self, time_s: float) -> tuple[float, float] | None:
        k = bisect.bisect_right(self._starts, time_s) - 1
        if k < 0:
            return None
        points, times = self._segments[k], self._times[k]
        if time_s > times[-1]:
            return None
        j = bisect.bisect_right(times, time_s) - 1
        a = points[j]
        if j == len(points) - 1 or times[j] == time_s:
            return a.latitude, a.longitude
        b = points[j + 1]
        f = (time_s - a.time_s) / (b.time_s - a.time_s)
        lat = a.latitude + (b.latitude - a.latitude) * f
        return lat, a.longitude + (b.longitude - a.longitude) * f


@dataclass(frozen=True)
class InsetPlan:
    """Everything fixed for one clip: framing, cached base image, timeline mapping."""

    size: tuple[int, int]
    rect: tuple[int, int, int, int]
    frame_size: tuple[int, int]
    fps: float
    frame_count: int
    extent: tuple[float, float, float, float]  # west, south, east, north
    segments: tuple[tuple[RoutePoint, ...], ...]
    base: Image.Image
    credits: tuple[str, ...]
    hidden_counts: dict[str, int]
    privacy: dict
    manifest_sha256: str
    timeline: EditTimeline = field(repr=False)
    index: _RouteIndex = field(repr=False)
    mode: str = MODE_ROUTE

    def project(self, latitude: float, longitude: float) -> tuple[float, float]:
        """Inset pixel of a position (north-up, linear in degrees over the extent)."""
        west, south, east, north = self.extent
        return (
            (longitude - west) / (east - west) * self.size[0],
            (north - latitude) / (north - south) * self.size[1],
        )


def marker_state(plan: InsetPlan, output_time_s: float) -> MarkerState:
    """The marker for one output second: pure, cheap, no pixels."""
    return _marker_state(plan.timeline, plan.index, output_time_s)


def _marker_state(timeline: EditTimeline, index: _RouteIndex, t: float) -> MarkerState:
    positions = timeline.output_to_source(t)
    if not positions:
        return MarkerState(False, "insert")
    if len(positions) == 2:
        return MarkerState(False, "crossfade")
    source_time = positions[0].time
    where = index.locate(source_time)
    if where is None:
        return MarkerState(False, "gap", source_time)
    return MarkerState(True, None, source_time, where[0], where[1])


def render_inset_frame(plan: InsetPlan, output_time_s: float) -> Image.Image:
    """One RGBA inset frame: the cached base, plus the marker when it is visible."""
    frame = plan.base.copy()
    state = marker_state(plan, output_time_s)
    if state.visible:
        x, y = plan.project(state.latitude, state.longitude)
        radius = max(4, plan.size[0] // 50)
        draw = ImageDraw.Draw(frame)
        draw.ellipse(
            (x - radius - 2, y - radius - 2, x + radius + 2, y + radius + 2),
            fill=_MARKER_RING_RGB,
            outline="#fff",
            width=2,
        )
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=_MARKER_RGB)
    return frame


@dataclass(frozen=True)
class InsetReceipt:
    """What a clip holds, with no route coordinates and no local paths."""

    bundle_manifest_sha256: str
    privacy: dict
    frame_size: tuple[int, int]
    inset_rect: tuple[int, int, int, int]
    fps: float
    frame_count: int
    hidden_frames: dict[str, int]
    credits: tuple[str, ...]
    wall_time_s: float
    schema: str = RECEIPT_SCHEMA
    mode: str = MODE_ROUTE  # recorded only when it is not the default route mode

    def to_dict(self) -> dict:
        data = {
            "schema": self.schema,
            "bundle_manifest_sha256": self.bundle_manifest_sha256,
            "privacy": self.privacy,
            "frame_size": list(self.frame_size),
            "inset_rect": list(self.inset_rect),
            "fps": self.fps,
            "frame_count": self.frame_count,
            "hidden_frames": dict(self.hidden_frames),
            "credits": list(self.credits),
            "wall_time_s": self.wall_time_s,
        }
        if self.mode != MODE_ROUTE:
            data["mode"] = self.mode
        return data

    @classmethod
    def load(cls, path: Path) -> InsetReceipt:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema") != RECEIPT_SCHEMA:
            raise ValueError(f"not a {RECEIPT_SCHEMA} receipt")
        try:
            hidden = {k: int(data["hidden_frames"][k]) for k in HIDDEN_REASONS}
            return cls(
                bundle_manifest_sha256=str(data["bundle_manifest_sha256"]),
                privacy=dict(data["privacy"]),
                frame_size=tuple(int(v) for v in data["frame_size"]),  # type: ignore[arg-type]
                inset_rect=tuple(int(v) for v in data["inset_rect"]),  # type: ignore[arg-type]
                fps=float(data["fps"]),
                frame_count=int(data["frame_count"]),
                hidden_frames=hidden,
                credits=tuple(str(c) for c in data["credits"]),
                wall_time_s=float(data["wall_time_s"]),
                mode=str(data.get("mode", MODE_ROUTE)),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"malformed {RECEIPT_SCHEMA} receipt: {error!r}") from error


def _frame_extent(
    points: Sequence[RoutePoint], margin: float, size: tuple[int, int]
) -> tuple[float, float, float, float]:
    west = min(p.longitude for p in points)
    east = max(p.longitude for p in points)
    south = min(p.latitude for p in points)
    north = max(p.latitude for p in points)
    width, height = size
    centre_lat = (south + north) / 2
    shrink = math.cos(
        math.radians(centre_lat)
    )  # a degree of longitude is shorter than one of latitude
    grow = 1.0 + 2.0 * margin
    span_lon = max(
        (east - west) * grow,
        (north - south) * grow * width / (height * shrink),
        _MIN_SPAN_LON,
    )
    span_lat = span_lon * shrink * height / width
    centre_lon = (west + east) / 2
    return (
        centre_lon - span_lon / 2,
        centre_lat - span_lat / 2,
        centre_lon + span_lon / 2,
        centre_lat + span_lat / 2,
    )


def _inside(extent: tuple[float, float, float, float], point: RoutePoint) -> bool:
    west, south, east, north = extent
    return south <= point.latitude <= north and west <= point.longitude <= east


def _choose_extent(
    kept: Sequence[RoutePoint], trimmed_away: Sequence[RoutePoint], size: tuple[int, int]
) -> tuple[float, float, float, float] | None:
    for margin in _MARGINS:
        extent = _frame_extent(kept, margin, size)
        if not any(_inside(extent, p) for p in trimmed_away):
            return extent
    return None


def _load_labels(bundle: Path) -> list[MapLabel]:
    path = bundle / "labels.geojson"
    if not path.is_file():
        return []
    labels: list[MapLabel] = []
    for feature in json.loads(path.read_text(encoding="utf-8")).get("features", ()):
        try:
            lon, lat = feature["geometry"]["coordinates"][:2]
            props = feature["properties"]
            labels.append(
                MapLabel(
                    str(props["name"]),
                    float(lat),
                    float(lon),
                    str(props.get("kind", "poi")).lower(),
                    float(props["max_span_lon"]) if props.get("max_span_lon") else None,
                    float(props["min_span_lon"]) if props.get("min_span_lon") else None,
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return labels


def _safe_labels(
    labels: Sequence[MapLabel],
    zones: Sequence[PrivacyZone],
    trimmed_away: Sequence[RoutePoint],
) -> list[MapLabel]:
    """Drop a name that sits in a privacy zone or beside a trimmed-away point."""
    safe = []
    for label in labels:
        spot = RoutePoint(0.0, label.latitude, label.longitude)
        if any(
            distance_m(spot, RoutePoint(0.0, z.center_lat, z.center_lon)) <= z.radius_m
            for z in zones
        ):
            continue
        if any(distance_m(spot, p) <= _LABEL_CLEARANCE_M for p in trimmed_away):
            continue
        safe.append(label)
    return safe


def credit_text(credit: str, *, roads: bool = False) -> str:
    """The credit strip text; the OSM line is added in roads mode only."""
    text = f"Imagery: {credit}"
    return f"{text} · {ROADS_CREDIT}" if roads else text


def road_width_px(highway: str, width: int) -> int:
    """Line width of a road class, scaled with the inset width."""
    if highway in ROAD_MAJOR_CLASSES:
        return max(2, width // ROAD_MAJOR_WIDTH_DIVISOR)
    return max(1, width // ROAD_MINOR_WIDTH_DIVISOR)


def _load_roads(bundle: Path) -> list[tuple[str, list[tuple[float, float]]]]:
    """``(highway, [(lon, lat), ...])`` per LineString of ``roads.geojson``."""
    path = bundle / "roads.geojson"
    if not path.is_file():
        return []
    roads = []
    try:
        features = json.loads(path.read_text(encoding="utf-8")).get("features", ())
    except (OSError, ValueError, AttributeError):
        return []
    for feature in features:
        try:
            geometry = feature["geometry"]
            if geometry["type"] != "LineString":
                continue
            coords = [(float(c[0]), float(c[1])) for c in geometry["coordinates"]]
            highway = str(feature["properties"].get("highway", ""))
        except (KeyError, TypeError, ValueError, IndexError, AttributeError):
            continue
        if len(coords) >= 2:
            roads.append((highway, coords))
    return roads


def _roads_in_view(
    roads: Sequence[tuple[str, list[tuple[float, float]]]],
    extent: tuple[float, float, float, float],
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Roads whose bounding box meets the extent (a cheap per-feature reject)."""
    west, south, east, north = extent
    kept = []
    for highway, coords in roads:
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        if max(lons) < west or min(lons) > east or max(lats) < south or min(lats) > north:
            continue
        kept.append((highway, coords))
    return kept


def _draw_credit(image: Image.Image, credit: str, *, roads: bool = False) -> None:
    width, height = image.size
    size = max(8, width // 34)
    font = _font(size)
    draw = ImageDraw.Draw(image)
    lines: list[str] = []
    current = ""
    for word in credit_text(credit, roads=roads).split():
        trial = f"{current} {word}".strip()
        if current and draw.textlength(trial, font=font) > width - 8:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    strip_height = len(lines) * (size + 3) + 4
    strip = Image.new("RGBA", (width, strip_height), (4, 14, 18, 190))
    image.alpha_composite(strip, (0, height - strip_height))
    draw = ImageDraw.Draw(image)
    for n, line in enumerate(lines):
        draw.text((4, height - strip_height + 3 + n * (size + 3)), line, font=font, fill="white")


def _render_base(
    raster: SatelliteRaster,
    extent: tuple[float, float, float, float],
    segments: Sequence[Sequence[RoutePoint]],
    labels: Sequence[MapLabel],
    size: tuple[int, int],
    roads: Sequence[tuple[str, list[tuple[float, float]]]] | None = None,
) -> Image.Image:
    width, height = size
    west, south, east, north = extent
    source = _raster_image(raster.source)
    x0 = (west - raster.west) / (raster.east - raster.west) * source.width
    x1 = (east - raster.west) / (raster.east - raster.west) * source.width
    y0 = (raster.north - north) / (raster.north - raster.south) * source.height
    y1 = (raster.north - south) / (raster.north - raster.south) * source.height
    image = source.transform(
        size,
        Image.Transform.QUAD,
        (x0, y0, x0, y1, x1, y1, x1, y0),
        resample=Image.Resampling.BICUBIC,
    )
    span_lon, span_lat = east - west, north - south
    draw = ImageDraw.Draw(image)
    # Minor roads first so a major road stays on top where they meet.
    for major in (False, True):
        for highway, line in roads or ():
            if (highway in ROAD_MAJOR_CLASSES) != major:
                continue
            pixels = [
                ((lon - west) / span_lon * width, (north - lat) / span_lat * height)
                for lon, lat in line
            ]
            draw.line(
                pixels,
                fill=_ROAD_MAJOR_RGB if major else _ROAD_MINOR_RGB,
                width=road_width_px(highway, width),
                joint="curve",
            )
    for segment in segments:
        coords = [
            ((p.longitude - west) / span_lon * width, (north - p.latitude) / span_lat * height)
            for p in segment
        ]
        if len(coords) > 1:
            draw.line(coords, fill=_HALO_RGB, width=route_halo_width_px(width), joint="curve")
            draw.line(coords, fill=_LINE_RGB, width=route_line_width_px(width), joint="curve")
    _draw_map_labels(image, labels, west, north, span_lon, span_lat, height)
    base = image.convert("RGBA")
    _draw_credit(base, raster.credit, roads=roads is not None)
    return base


def _inset_rect(
    frame_size: tuple[int, int], size: tuple[int, int], corner: str, margin_fraction: float
) -> tuple[int, int, int, int]:
    margin = round(frame_size[0] * margin_fraction)
    x = margin if corner.endswith("left") else frame_size[0] - margin - size[0]
    y = margin if corner.startswith("top") else frame_size[1] - margin - size[1]
    return x, y, size[0], size[1]


def plan_route_inset(
    bundle: Path,
    segments: Sequence[Sequence[RoutePoint]],
    timeline: EditTimeline,
    frame_size: tuple[int, int],
    *,
    inset_fraction: float = 0.20,
    corner: str = "bottom-left",
    margin_fraction: float = 0.02,
    fps: float = 10.0,
    trim: RouteTrim | None = None,
    zones: Sequence[PrivacyZone] = (),
    min_coverage: float = 0.5,
    workdir: Path | None = None,
    roads: bool = False,
) -> InsetPlan:
    """Check every refusal reason, then render the cached base image.

    ``roads=True`` also draws the bundle's ``roads.geojson`` under the route and
    refuses when the layer is missing or no road lies in the framed extent.

    Raises :class:`MapInsetRefused` (listing all reasons) before rendering anything.
    """
    if corner not in CORNERS:
        raise ValueError(f"corner must be one of {CORNERS}, got {corner!r}")
    if not 0 < inset_fraction <= 0.5 or fps <= 0 or not 0 <= min_coverage <= 1:
        raise ValueError("inset_fraction must be in (0, 0.5], fps > 0, min_coverage in [0, 1]")
    bundle = Path(bundle)
    inset_width = max(16, round(frame_size[0] * inset_fraction))
    size = (inset_width, round(inset_width * _INSET_ASPECT))
    reasons: list[str] = []

    bundle_ok = False
    if not (bundle / "manifest.json").is_file():
        reasons.append("bundle is missing or has no manifest.json")
    else:
        try:
            problems = verify_bundle(bundle)
        except (OSError, ValueError) as error:
            problems = [f"manifest unreadable: {error}"]
        reasons.extend(f"bundle problem: {p}" for p in problems)
        bundle_ok = not problems

    input_points = [p for s in segments for p in s]
    kept_segments: list[list[RoutePoint]] = []
    trim_result = zone_result = None
    try:
        trim_result = trim_route(segments, trim or RouteTrim())
        zone_result = mask_zones(trim_result.segments, zones)
        kept_segments = [list(s) for s in zone_result.segments]
    except RouteTrimError as error:
        reasons.append(f"route is empty after privacy trim and zones: {error}")
    if kept_segments:
        after_trim = {id(p) for s in trim_result.segments for p in s}  # type: ignore[union-attr]
        trimmed_away = [p for p in input_points if id(p) not in after_trim]
        index = _RouteIndex(kept_segments)
    else:
        trimmed_away = []
        index = _RouteIndex([])

    duration = timeline.duration
    frame_count = max(0, math.ceil(duration * fps - 1e-9))
    hidden = dict.fromkeys(HIDDEN_REASONS, 0)
    visible = 0
    for i in range(frame_count):
        state = _marker_state(timeline, index, i / fps)
        if state.visible:
            visible += 1
        else:
            hidden[state.hidden_reason] += 1  # type: ignore[index]
    if frame_count == 0:
        reasons.append("the edit timeline has no duration")
    elif kept_segments and visible / frame_count < min_coverage:
        reasons.append(
            f"marker coverage {visible / frame_count:.2f} is below the minimum {min_coverage:.2f}"
        )

    extent = None
    if kept_segments:
        kept_points = [p for s in kept_segments for p in s]
        extent = _choose_extent(kept_points, trimmed_away, size)
        if extent is None:
            reasons.append(
                "privacy: the framing around the kept route cannot exclude points the trim removed"
            )
    base = None
    credit = ""
    road_lines = None
    if roads and bundle_ok:
        if not (bundle / "roads.geojson").is_file():
            reasons.append("roads were requested but the bundle has no roads.geojson")
        elif extent is not None:
            road_lines = _roads_in_view(_load_roads(bundle), extent)
            if not road_lines:
                reasons.append("roads were requested but there are no roads in view of the route")
    if bundle_ok and extent is not None:
        # Imagery is only checked once the bundle and the framing are usable.
        with tempfile.TemporaryDirectory() as scratch:
            work = Path(workdir) if workdir is not None else Path(scratch)
            raster, imagery_reasons = _covering_raster(bundle, extent, work)
            reasons.extend(imagery_reasons)
            if not reasons and raster is not None:
                labels = _safe_labels(_load_labels(bundle), zones, trimmed_away)
                base = _render_base(raster, extent, kept_segments, labels, size, road_lines)
                credit = raster.credit
    if reasons or base is None or extent is None or trim_result is None or zone_result is None:
        raise MapInsetRefused(reasons)
    return InsetPlan(
        size=size,
        rect=_inset_rect(frame_size, size, corner, margin_fraction),
        frame_size=(int(frame_size[0]), int(frame_size[1])),
        fps=float(fps),
        frame_count=frame_count,
        extent=extent,
        segments=tuple(tuple(s) for s in kept_segments),
        base=base,
        credits=(credit, ROADS_CREDIT) if roads else (credit,),
        hidden_counts=hidden,
        privacy=receipt_fragment(trim_result, zone_result if zones else None),
        manifest_sha256=hashlib.sha256((bundle / "manifest.json").read_bytes()).hexdigest(),
        timeline=timeline,
        index=index,
        mode=MODE_ROADS if roads else MODE_ROUTE,
    )


def _covering_raster(
    bundle: Path, extent: tuple[float, float, float, float], work: Path
) -> tuple[SatelliteRaster | None, list[str]]:
    """The finest bundle raster that fully covers the framed extent, or why none does."""
    try:
        rasters = bundle_opening_rasters(bundle, work)
    except (OSError, ValueError, KeyError) as error:
        return None, [f"imagery could not be read from the bundle: {error!r}"]
    west, south, east, north = extent
    covering = [
        r
        for r in rasters
        if r.west <= west and r.east >= east and r.south <= south and r.north >= north
    ]
    if not covering:
        return None, ["imagery in the bundle does not cover the framed route extent"]
    return covering[-1], []


def render_route_inset(
    bundle: Path,
    segments: Sequence[Sequence[RoutePoint]],
    timeline: EditTimeline,
    output: Path,
    frame_size: tuple[int, int],
    *,
    workdir: Path | None = None,
    **options,
) -> InsetReceipt:
    """Write the RGBA overlay clip at ``output`` and its receipt beside it.

    ``options`` are those of :func:`plan_route_inset`. Both files are written to
    a temporary name and renamed on success; a refusal writes nothing.
    """
    started = time.perf_counter()
    plan = plan_route_inset(bundle, segments, timeline, frame_size, workdir=workdir, **options)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write the inset clip")
    output = Path(output)
    receipt_path = output.with_name(output.name + ".receipt.json")
    partial_clip = output.with_name(output.name + ".part")
    partial_receipt = receipt_path.with_name(receipt_path.name + ".part")
    command = [
        ffmpeg, "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgba",
        "-s", f"{plan.size[0]}x{plan.size[1]}", "-framerate", repr(plan.fps), "-i", "-",
        "-an", "-c:v", "qtrle", "-pix_fmt", "argb", "-f", "mov", str(partial_clip),
    ]  # fmt: skip
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            if process.stdin is None:
                raise RuntimeError("ffmpeg stdin pipe was not opened")
            try:
                for i in range(plan.frame_count):
                    process.stdin.write(render_inset_frame(plan, i / plan.fps).tobytes())
            finally:
                process.stdin.close()
                code = process.wait()
            if code != 0:
                errors.seek(0)
                tail = errors.read().decode("utf-8", "replace")[-400:]
                raise RuntimeError(f"ffmpeg failed ({code}): {tail}")
        receipt = InsetReceipt(
            bundle_manifest_sha256=plan.manifest_sha256,
            privacy=plan.privacy,
            frame_size=plan.frame_size,
            inset_rect=plan.rect,
            fps=plan.fps,
            frame_count=plan.frame_count,
            hidden_frames=dict(plan.hidden_counts),
            credits=plan.credits,
            wall_time_s=round(time.perf_counter() - started, 3),
            mode=plan.mode,
        )
        partial_receipt.write_text(
            json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        partial_clip.replace(output)
        partial_receipt.replace(receipt_path)
    finally:
        partial_clip.unlink(missing_ok=True)
        partial_receipt.unlink(missing_ok=True)
    return receipt
