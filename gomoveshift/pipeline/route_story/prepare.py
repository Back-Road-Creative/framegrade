"""Prepare an offline Route Story map bundle from a GPX trip.

Orchestrates the sibling modules (``gpx``, ``bundle``, ``sources``,
``scene.vendor``) into one content-addressed bundle: a DEM + orthoimagery +
OSM labels + tile pyramid + vendored MapLibre build, verified before being
handed back. All network access happens through the caller-injected
``session`` (public map source fetchers) and ``fetch_maplibre`` (vendor
tarball) callables — this module performs no I/O of its own beyond writing
bundle files under the returned bundle directory. Exactly one Overpass query
is issued per call, never one per frame or per tile.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any, Callable, Sequence

from ...paths import gms_route_story_cache_dir
from .. import fs_safety
from .backdrop import BACKDROP_RECIPE_VERSION, grade_backdrop
from .bands import (
    BAND_PLAN_VERSION,
    DEFAULT_REGIONAL_MARGIN_KM,
    GIBS_MAX_REQUEST_PX,
    ZoomBand,
    band_footprint,
    band_pixel_size,
    bands_receipt,
    plan_imagery_bands,
    plan_terrain_bands,
)
from .borders import BORDERS_RECIPE_VERSION, border_coverage, overlay_borders
from .bundle import (
    Footprint,
    TilePyramidReport,
    bundle_dir_for,
    bundle_file_digest,
    footprint_dimensions_km,
    footprint_for_route,
    rewrite_manifest,
    tiles_covering,
    verify_bundle,
    write_manifest,
    write_tile_pyramid,
    written_tiles_footprint,
)
from .gnis import GNIS_HOST, GnisFetchError, gnis_features_for_bbox, to_label_geojson
from .gpx import GpxDaySelection, select_gpx_day
from .metrics import RoutePoint
from .scene.vendor import MAPLIBRE_VERSION, ensure_maplibre
from .sources import (
    ARCGIS_EXPORT_GRID_VERSION,
    COPERNICUS_PROVENANCE_NAME,
    DEM_3DEP_PROVENANCE_NAME,
    NAIP_PROVENANCE_NAME,
    GeoRaster,
    SourceCoverageError,
    SourceFetchError,
    SourceProvenance,
    fetch_3dep_dem,
    fetch_copernicus_dem,
    fetch_gibs_blue_marble,
    fetch_gibs_reference_features,
    fetch_naip_imagery,
    fetch_osm_labels,
    gibs_scene_for,
)

# Bumped from "osm-1" now that a bundle's labels are a GNIS+OSM merge, not OSM
# alone — the two shapes must never share a content-addressed cache entry.
LABELS_SOURCE_VERSION = "gnis-osm-1"

GNIS_PROVENANCE_NAME = "GNIS Domestic Names"

DEFAULT_MARGIN_M = 1500.0

# The focal band (bands.py's "naip") is the extent every imagery-inside-terrain check anchors to
# (`_kind_extent_fields`) — an `imagery_zooms` that excludes all of it leaves
# nothing written for the focal band, which falls back to the raw (unsnapped)
# footprint and can then reach past the terrain extent. Checked up front, so
# the failure is a clear ValueError before any network call, not a confusing
# verify_bundle problem after a wasted fetch. Narrowed from 13-16 to 15-16
# when `naip` shrank to its own z15-only band (a `naip_mid` band, z12-14, now
# sits between it and `naip_coarse` — bands.py's module docstring).
FOCAL_IMAGERY_ZOOMS = range(15, 16)

# The shared z2-4 Blue Marble tile cache is dateless and keyed only by this
# version, never by footprint or recorded_on — bumping it invalidates every
# bundle's cached tiles at once if the fetch/encode ever changes shape.
# Bumped to -2 when backdrop grading (`backdrop.grade_backdrop`) started
# being applied before the shared tiles are written, so a pre-grading cache
# on disk is never reused as though it were graded. Bumped to -3 when the
# coastline/border overlay (`borders.overlay_borders`) started being drawn
# on top of the graded tiles — the shared tiles' pixels changed shape again.
BLUE_MARBLE_CACHE_VERSION = "blue-marble-3"

# A real measurement over a CONUS bbox (-128,22,-64,52) at 1920x900 gave
# `border_coverage` == 0.0277 for the GIBS reference-features overlay. The
# shared z2-4 band's bbox is the whole globe, which always has coastline, so
# anything anywhere near 0 there means a failed/near-empty fetch, not a
# legitimately borderless footprint — this floor sits well below that real
# measurement so normal variance never trips it.
MIN_SHARED_BORDER_COVERAGE = 0.002

# Cloud-free by default: the z5-8 ``gibs`` band renders the dateless Blue
# Marble backdrop (graded, like the shared z2-4 band) instead of the
# trip-date true-colour scene, which can carry cloud. The trip-date scene
# (with its existing cloud-fraction fallback, untouched) is an explicit
# opt-in — see ``_gibs_band_raster``.
DEFAULT_TRIP_DATE_IMAGERY = False

# Terrain zoom range is chosen against a typical route footprint of a few km to
# a few tens of km and USGS 3DEP's ~10 m/px native resolution, not picked
# arbitrarily: Web-Mercator tile resolution at zoom z and a mid-latitude (~35 deg)
# route is
# ~127826/2**z m/px (256 px tiles), so z=13 (~15.6 m/px) is the finest zoom
# that does not fabricate detail the DEM doesn't have; z=9 (~250 m/px)
# comfortably covers the wide pullback.
#
# Imagery spans the full z2-15 range across six per-band sources planned by
# `bands.plan_imagery_bands` (shared blue_marble z2-4, national gibs z5-8,
# regional naip_coarse z9-10, naip_mid z11-12, naip_fine z13-14, focal naip
# z15) — a single focal-sized raster can never fully cover a wide, coarse-zoom
# tile, so before the bands existed the bundle floor was silently z13 (see
# `bands.py`'s module docstring).
DEFAULT_IMAGERY_ZOOMS: tuple[int, ...] = tuple(range(2, 16))
DEFAULT_TERRAIN_ZOOMS: tuple[int, ...] = (9, 10, 11, 12, 13)
IMAGERY_METRES_PER_PX = 2.0
# None tries the service's own Overpass mirror list (see sources.fetch_osm_labels),
# because the canonical host can refuse a client's IPv4 on 443.
DEFAULT_OVERPASS_ENDPOINT: str | None = None
DEFAULT_LABEL_SOURCES: tuple[str, ...] = ("gnis", "osm")

# Real terrain relief, not an invented exaggeration; ``EXAGGERATION_MAX`` is
# the ceiling an editor may dial in, never the default.
DEFAULT_EXAGGERATION = 1.0
EXAGGERATION_MIN = 1.0
EXAGGERATION_MAX = 1.25


logger = logging.getLogger(__name__)


def _clamp_exaggeration(value: float) -> float:
    return max(EXAGGERATION_MIN, min(EXAGGERATION_MAX, value))


@dataclass(frozen=True)
class PrepareReport:
    """What a ``prepare_bundle`` call actually fetched and wrote, for the receipt."""

    bundle_dir: Path
    footprint: Footprint
    imagery_source: str
    terrain_source: str
    terrain_fallback_used: bool
    imagery_zooms: tuple[int, ...]
    terrain_zooms: tuple[int, ...]
    imagery_tiles_written: int
    imagery_tiles_skipped: int
    terrain_tiles_written: int
    terrain_tiles_skipped: int
    label_count: int
    excluded_other_date_count: int
    excluded_duplicate_count: int
    verify_problems: list[str]
    labels_degraded: bool
    label_problems: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["bundle_dir"] = str(self.bundle_dir)
        return payload


def _route_points(segments: Sequence[Sequence[RoutePoint]]) -> list[RoutePoint]:
    points = [point for segment in segments for point in segment]
    if not points:
        raise ValueError("Route Story GPX has no usable track points")
    return points


def _bbox(footprint: Footprint) -> tuple[float, float, float, float]:
    return (footprint.west, footprint.south, footprint.east, footprint.north)


def _raster_footprint(raster: GeoRaster) -> Footprint:
    return Footprint(west=raster.west, south=raster.south, east=raster.east, north=raster.north)


def _route_geojson(selection: GpxDaySelection) -> dict[str, Any]:
    features = []
    for segment in selection.segments:
        if len(segment) < 2:
            continue
        coordinates = [[point.longitude, point.latitude] for point in segment]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coordinates},
                "properties": {},
            }
        )
    return {
        "type": "FeatureCollection",
        "features": features,
        "properties": {
            "excluded_other_date_count": selection.excluded_other_date_count,
            "excluded_duplicate_count": selection.excluded_duplicate_count,
        },
    }


def _fetch_terrain(
    session: Any, bbox: tuple[float, float, float, float], *, metres_per_px: float = 10.0
) -> tuple[GeoRaster, bool]:
    """Fetch USGS 3DEP, falling back to Copernicus only on a coverage error.

    Used independently for the focal and regional terrain bands — each has
    its own DEM, and either can fall back to Copernicus without the other
    doing so (3DEP might cover a small focal box but not a much wider
    regional one).
    """
    try:
        return fetch_3dep_dem(session, bbox, metres_per_px=metres_per_px), False
    except SourceCoverageError:
        return fetch_copernicus_dem(session, bbox), True


def _gnis_provenance() -> SourceProvenance:
    return SourceProvenance(
        name=GNIS_PROVENANCE_NAME,
        url=f"https://{GNIS_HOST}/StagedProducts/GeographicNames/DomesticNames",
        licence="Public domain (USGS)",
        attribution="USGS Geographic Names Information System",
        acquired="unknown",
    )


def _label_key(feature: dict[str, Any]) -> tuple[str, str]:
    properties = feature["properties"]
    return (str(properties.get("name", "")).strip().lower(), str(properties.get("kind", "")))


# Kinds kept as regional context outside the route footprint — the ladder's
# coarsest, most orienting entries (a nearby town, a named summit, a gap the
# route threads toward). Every kind is kept inside the footprint; this list
# only trims the wider naip_coarse-band ring so it doesn't flood with minor
# features (streams, trails, etc.) just because they happen to fall inside
# the regional context area.
_CONTEXT_LABEL_KINDS = frozenset({"place", "summit", "gap"})


def _filter_context_labels(
    features: list[dict[str, Any]], footprint_bbox: tuple[float, float, float, float]
) -> list[dict[str, Any]]:
    """Drop minor-kind features that fall outside the route footprint.

    A feature inside ``footprint_bbox`` is always kept, whatever its kind. A
    feature outside it is kept only when its kind is one of
    ``_CONTEXT_LABEL_KINDS`` — the regional naip_coarse-band bbox labels are
    fetched over is much larger than the route itself, and without this most
    features it turns up would be minor context noise.
    """
    west, south, east, north = footprint_bbox
    kept = []
    for feature in features:
        lon, lat = feature["geometry"]["coordinates"]
        if west <= lon <= east and south <= lat <= north:
            kept.append(feature)
        elif feature["properties"].get("kind") in _CONTEXT_LABEL_KINDS:
            kept.append(feature)
    return kept


def _fetch_labels(
    session: Any,
    bbox: tuple[float, float, float, float],
    *,
    gnis_cache_dir: Path,
    overpass_endpoint: str | None,
    label_sources: Sequence[str],
) -> tuple[list[dict[str, Any]], list[SourceProvenance], bool, tuple[str, ...]]:
    """Fetch every configured label source independently and merge them.

    Each source is fetched on its own: a source that raises or comes back
    empty is recorded as a problem, never fatal to the bundle — the merged
    set ships whatever the other source(s) produced. On a (normalised name,
    kind) collision GNIS wins, since it is the more precise, curated source.
    Returns ``(features, provenances, degraded, problems)``; ``degraded`` is
    True whenever any configured source failed/came back empty OR the merged
    set is empty.
    """
    by_source: dict[str, list[dict[str, Any]]] = {}
    provenances: list[SourceProvenance] = []
    problems: list[str] = []

    if "gnis" in label_sources:
        try:
            gnis_features = gnis_features_for_bbox(session, bbox, gnis_cache_dir)
            gnis_geojson = to_label_geojson(gnis_features)["features"]
            by_source["gnis"] = gnis_geojson
            provenances.append(_gnis_provenance())
            if not gnis_geojson:
                problems.append("gnis: no features in bbox")
        except (GnisFetchError, OSError) as exc:
            problems.append(f"gnis: {exc}")

    if "osm" in label_sources:
        try:
            osm_result = fetch_osm_labels(session, bbox, endpoint=overpass_endpoint)
            osm_features = osm_result["features"]
            by_source["osm"] = osm_features
            provenances.append(osm_result["provenance"])
            if not osm_features:
                problems.append("osm: no features in bbox")
        except (SourceFetchError, OSError) as exc:
            problems.append(f"osm: {exc}")

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for source_name in ("osm", "gnis"):  # gnis last so it wins a collision
        for feature in by_source.get(source_name, []):
            merged[_label_key(feature)] = feature
    features = sorted(
        merged.values(), key=lambda f: (f["properties"]["kind"], f["properties"]["name"])
    )

    degraded = bool(problems) or not features
    return features, provenances, degraded, tuple(problems)


def _write_labels(bundle_root: Path, features: Sequence[dict[str, Any]]) -> None:
    (bundle_root / "labels.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": list(features)}),
        encoding="utf-8",
    )


def _repair_degraded_labels(
    session: Any,
    report: PrepareReport,
    *,
    route_bbox: tuple[float, float, float, float],
    gnis_cache_dir: Path,
    overpass_endpoint: str | None,
    label_sources: Sequence[str],
) -> PrepareReport:
    """Retry the label fetch alone on a cache hit whose labels came back degraded.

    A bundle's cache address is built from its imagery and terrain sources
    (``_band_source_versions``); the label sources are not in it, and
    ``verify_bundle`` is perfectly happy with a bundle whose Overpass half
    failed. So a prepare that lost Overpass wrote a GNIS-only bundle, and that
    bundle then answered every later request for the same footprint — the
    degradation was permanent until someone deleted the directory by hand.

    Refusing the whole cache hit would re-fetch thousands of terrain and
    imagery tiles to repair one small JSON file, so the retry is scoped to
    what actually failed. The bbox comes from the manifest the bundle already
    carries rather than being re-derived here: a repair that labelled a
    different area than the bundle was built for would be a new defect.

    The bundle is upgraded only when the retry is strictly better — undegraded
    AND no smaller than what is on disk. An undegraded fetch only means every
    source answered, not that it answered as fully as last time, and silently
    trading a complete label set for a thin one is worse than staying
    degraded. Anything else leaves every byte where it was.

    Returns the report to use: an upgraded copy, or ``report`` unchanged.
    """
    if not report.labels_degraded:
        return report
    manifest_path = report.bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    label_bbox = manifest.get("labels", {}).get("bbox")
    if not label_bbox:
        return report

    features, provenances, degraded, problems = _fetch_labels(
        session,
        (label_bbox[0], label_bbox[1], label_bbox[2], label_bbox[3]),
        gnis_cache_dir=gnis_cache_dir,
        overpass_endpoint=overpass_endpoint,
        label_sources=label_sources,
    )
    features = _filter_context_labels(features, route_bbox)
    if degraded or len(features) < report.label_count:
        return report

    _write_labels(report.bundle_dir, features)
    manifest["labels"] = {
        **manifest["labels"],
        "sources": [asdict(p) for p in provenances],
        "degraded": False,
        "problems": list(problems),
        "count": len(features),
    }
    manifest["files"]["labels.geojson"] = bundle_file_digest(report.bundle_dir / "labels.geojson")
    rewrite_manifest(report.bundle_dir, manifest)
    logger.info(
        "route-story bundle %s: labels repaired in place (%d -> %d features)",
        report.bundle_dir.name,
        report.label_count,
        len(features),
    )
    return replace(
        report,
        label_count=len(features),
        labels_degraded=False,
        label_problems=problems,
    )


def _report_from_existing(
    bundle_root: Path,
    footprint: Footprint,
    terrain_source: str,
    terrain_fallback_used: bool,
    imagery_zooms: Sequence[int],
    terrain_zooms: Sequence[int],
) -> PrepareReport:
    manifest = json.loads((bundle_root / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((bundle_root / "labels.geojson").read_text(encoding="utf-8"))
    route = json.loads((bundle_root / "route.geojson").read_text(encoding="utf-8"))
    route_properties = route.get("properties", {})
    labels_manifest = manifest.get("labels", {})
    return PrepareReport(
        bundle_dir=bundle_root,
        footprint=footprint,
        imagery_source=NAIP_PROVENANCE_NAME,
        terrain_source=terrain_source,
        terrain_fallback_used=terrain_fallback_used,
        imagery_zooms=tuple(sorted(set(imagery_zooms))),
        terrain_zooms=tuple(sorted(set(terrain_zooms))),
        imagery_tiles_written=sum(
            1 for relpath in manifest["files"] if relpath.startswith("imagery/")
        ),
        imagery_tiles_skipped=0,
        terrain_tiles_written=sum(
            1 for relpath in manifest["files"] if relpath.startswith("terrain/")
        ),
        terrain_tiles_skipped=0,
        label_count=len(labels["features"]),
        excluded_other_date_count=route_properties.get("excluded_other_date_count", 0),
        excluded_duplicate_count=route_properties.get("excluded_duplicate_count", 0),
        verify_problems=[],
        labels_degraded=labels_manifest.get("degraded", False),
        label_problems=tuple(labels_manifest.get("problems", ())),
    )


def _band_source_versions(
    *,
    gibs_chosen: str,
    recorded_on: date,
    terrain_focal_name: str,
    terrain_regional_name: str,
    regional_margin_km: float,
) -> dict[str, str]:
    """The per-source cache-key fields, given what a fetch has (or might) resolve to.

    Used both by the real fetch path (with the gibs scene and terrain
    provenances it actually chose) and by ``_find_cached_bundle`` (trying
    every combination a fetch could resolve to) — the two must build this
    identically or a cache hit is never found. The focal and regional
    terrain bands are keyed separately (``"terrain"`` kept for the focal
    band, matching every cache entry written before the regional band
    existed) since either can fall back to Copernicus independently of the
    other. ``regional_margin_km`` is folded in here too: it changes the
    regional bands' bbox (and so their tile content) without changing the
    overall route footprint, so two calls differing only in it must not
    collide on the same cache directory.
    """
    return {
        "imagery_blue_marble": BLUE_MARBLE_CACHE_VERSION,
        "imagery_gibs": f"gibs-{gibs_chosen}-{recorded_on.isoformat()}",
        "imagery_naip_coarse": NAIP_PROVENANCE_NAME,
        "imagery_naip_mid": NAIP_PROVENANCE_NAME,
        "imagery_naip_fine": NAIP_PROVENANCE_NAME,
        "imagery_naip": NAIP_PROVENANCE_NAME,
        "terrain": terrain_focal_name,
        "terrain_regional": terrain_regional_name,
        "regional_margin_km": repr(regional_margin_km),
        "arcgis_export": ARCGIS_EXPORT_GRID_VERSION,
        "labels": LABELS_SOURCE_VERSION,
        "maplibre": MAPLIBRE_VERSION,
        # Any Blue Marble raster (shared z2-4 or per-bundle z5-8) is graded
        # by `backdrop.grade_backdrop` before tiling — bumping the recipe
        # must change every affected bundle's id, not just the shared cache.
        "backdrop_recipe": BACKDROP_RECIPE_VERSION,
        # Same reasoning for the coastline/border overlay drawn on top of
        # every Blue Marble raster — an old, borderless bundle must never
        # look like a cache hit for the new (bordered) recipe.
        "borders_recipe": BORDERS_RECIPE_VERSION,
        # The shape of the ladder itself, not just which sources it drew on:
        # `bands.plan_imagery_bands` can change which source covers which
        # zoom, and how wide each band reaches, without moving any field
        # above or the zoom ranges folded in by `bundle_dir_for`.
        "bands": BAND_PLAN_VERSION,
    }


def _select_and_footprint(
    gpx_path: Path, recorded_on: date, margin_m: float
) -> tuple[GpxDaySelection, Footprint]:
    """The exact day selection + footprint ``prepare_bundle`` keys its bundle on.

    Shared with ``find_prepared_bundle`` so a caller resolving a bundle can
    never derive a footprint that drifts from what a real ``prepare_bundle``
    call for the same GPX/day/margin would compute.
    """
    selection = select_gpx_day(Path(gpx_path), recorded_on)
    points = _route_points(selection.segments)
    footprint = footprint_for_route(points, margin_m=margin_m)
    return selection, footprint


def find_prepared_bundle(
    gpx_path: Path,
    *,
    recorded_on: date,
    margin_m: float = DEFAULT_MARGIN_M,
    terrain_exaggeration: float = DEFAULT_EXAGGERATION,
    regional_margin_km: float = DEFAULT_REGIONAL_MARGIN_KM,
    imagery_zooms: Sequence[int] = DEFAULT_IMAGERY_ZOOMS,
    terrain_zooms: Sequence[int] = DEFAULT_TERRAIN_ZOOMS,
    trip_date_imagery: bool = DEFAULT_TRIP_DATE_IMAGERY,
) -> Path | None:
    """Return the bundle dir a prior ``prepare_bundle`` call wrote for these args, or ``None``.

    Performs exactly ``prepare_bundle``'s own day selection, footprint, and
    exaggeration clamping before delegating to ``_find_cached_bundle`` — a
    caller (the route-story-proof CLI) that supplies the same args always
    resolves the identical bundle a matching ``prepare_bundle`` call would
    use or reuse, with no second, independently-derived cache key that could
    drift from prepare's own. Performs no network I/O.
    """
    _selection, footprint = _select_and_footprint(gpx_path, recorded_on, margin_m)
    clamped_exaggeration = _clamp_exaggeration(terrain_exaggeration)
    cached = _find_cached_bundle(
        footprint,
        imagery_zooms,
        terrain_zooms,
        recorded_on,
        clamped_exaggeration,
        regional_margin_km,
        trip_date_imagery,
    )
    return cached.bundle_dir if cached is not None else None


def _find_cached_bundle(
    footprint: Footprint,
    imagery_zooms: Sequence[int],
    terrain_zooms: Sequence[int],
    recorded_on: date,
    terrain_exaggeration: float,
    regional_margin_km: float,
    trip_date_imagery: bool,
) -> PrepareReport | None:
    """Return a report for an already-verified bundle, without touching the network.

    Every source-version combination a fetch could resolve to is
    content-addressed and checked; if any already verifies clean, the whole
    fetch is skipped. The focal and regional terrain bands each have two
    outcomes (primary 3DEP, or the Copernicus fallback) independently of one
    another — cross-producted here since neither is knowable before the
    fetch actually happens. The gibs band has two possible outcomes only
    when ``trip_date_imagery`` is set (the trip-date scene, or its own
    cloudy-day Blue Marble fallback); with the default off, the band always
    resolves to Blue Marble, so only that one choice is probed.
    ``regional_margin_km`` is not cross-producted — it is a caller-supplied
    plan input, not a fetch outcome, so it is passed straight through into
    the cache key.
    """
    for terrain_focal_name, focal_fallback in (
        (DEM_3DEP_PROVENANCE_NAME, False),
        (COPERNICUS_PROVENANCE_NAME, True),
    ):
        for terrain_regional_name, regional_fallback in (
            (DEM_3DEP_PROVENANCE_NAME, False),
            (COPERNICUS_PROVENANCE_NAME, True),
        ):
            gibs_choices = ("daily", "blue_marble") if trip_date_imagery else ("blue_marble",)
            for gibs_chosen in gibs_choices:
                source_versions = _band_source_versions(
                    gibs_chosen=gibs_chosen,
                    recorded_on=recorded_on,
                    terrain_focal_name=terrain_focal_name,
                    terrain_regional_name=terrain_regional_name,
                    regional_margin_km=regional_margin_km,
                )
                candidate = bundle_dir_for(
                    footprint,
                    source_versions,
                    terrain_exaggeration=terrain_exaggeration,
                    imagery_zooms=imagery_zooms,
                    terrain_zooms=terrain_zooms,
                )
                if candidate.exists() and not verify_bundle(candidate):
                    return _report_from_existing(
                        candidate,
                        footprint,
                        terrain_focal_name,
                        focal_fallback or regional_fallback,
                        imagery_zooms,
                        terrain_zooms,
                    )
    return None


def _written_zoom_range(kind: str, report: TilePyramidReport) -> dict[str, int]:
    """The zoom range the manifest may declare: the tiles actually on disk.

    ``write_tile_pyramid`` skips any tile the source raster only partly
    covers, so a requested wide zoom can end up with no tiles at all. A
    manifest that still advertised it made the scene request tiles that 404,
    which left MapLibre permanently unsettled — so the declared range is
    derived from the write report, never from the requested list.
    """
    zooms = sorted({z for z, _x, _y in report.written})
    if not zooms:
        raise ValueError(
            f"{kind}: no tile was written for any requested zoom — "
            "the source raster does not fully cover a single tile"
        )
    return {"minzoom": zooms[0], "maxzoom": zooms[-1]}


def _kind_extent_fields(report: TilePyramidReport, footprint: Footprint) -> dict[str, Any]:
    """The manifest's ``extent``/``footprint_km`` for one written tile pyramid.

    Falls back to the raw source footprint only when nothing was written
    (``_written_zoom_range`` already raises for that case before this is
    reached in the normal path, so this only guards direct callers).
    """
    zooms = sorted({z for z, _x, _y in report.written})
    extent = written_tiles_footprint(report, zooms[-1]) if zooms else footprint
    return {
        "extent": {
            "west": extent.west,
            "south": extent.south,
            "east": extent.east,
            "north": extent.north,
        },
        "footprint_km": footprint_dimensions_km(extent),
    }


def _gibs_request_size(band: ZoomBand) -> tuple[int, int]:
    """Pixel size for one GIBS WMS request, for either GIBS-backed band.

    ``band_raster_size`` mosaics a band into several sub-requests once its
    raw pixel size passes a source's request-size ceiling — worthwhile for
    NAIP's real ground detail, but the two GIBS bands (``blue_marble``,
    ``gibs``) only ever back the coarsest zooms (2-8: a national opening
    shot), so the simpler choice made here is to clamp to one request no
    larger than ``GIBS_MAX_REQUEST_PX`` on a side rather than mosaic
    sub-requests for resolution those zooms never render.
    """
    width_px, height_px = band_pixel_size(band)
    return min(width_px, GIBS_MAX_REQUEST_PX), min(height_px, GIBS_MAX_REQUEST_PX)


def _regrade(raster: GeoRaster) -> GeoRaster:
    """Return ``raster`` with its array run through ``grade_backdrop``."""
    return GeoRaster(
        array=grade_backdrop(raster.array),
        west=raster.west,
        south=raster.south,
        east=raster.east,
        north=raster.north,
        provenance=raster.provenance,
    )


def _bordered(
    session: Any,
    raster: GeoRaster,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[GeoRaster, dict[str, Any], SourceProvenance]:
    """Overlay coastline/state/national borders onto an already-graded raster.

    Fetches the reference-features overlay for the SAME bbox and request
    size the raster itself was fetched at. Never raises on low/zero
    coverage — a small inland footprint can legitimately contain no line at
    all; the caller decides whether that is acceptable for its band.
    """
    overlay = fetch_gibs_reference_features(session, bbox, width=width, height=height)
    coverage = border_coverage(overlay.array)
    bordered_array = overlay_borders(raster.array, overlay.array)
    bordered = GeoRaster(
        array=bordered_array,
        west=raster.west,
        south=raster.south,
        east=raster.east,
        north=raster.north,
        provenance=raster.provenance,
    )
    decision = {"layer": "Reference_Features_15m", "coverage": coverage}
    return bordered, decision, overlay.provenance


def _gibs_band_raster(
    session: Any, band: ZoomBand, recorded_on: date, *, trip_date_imagery: bool
) -> tuple[GeoRaster, dict[str, Any], SourceProvenance | None]:
    """Fetch and grade the z5-8 ``gibs`` band raster.

    Cloud-free by default (``trip_date_imagery=False``): fetches the
    dateless Blue Marble backdrop directly, graded like the shared z2-4
    band, with no daily-scene or cloud-fraction call at all. Unlike that
    shared band, this one's bbox is footprint-dependent at z5-8 — many more,
    much smaller tiles than the handful of huge z2-4 tiles that cover the
    whole globe, so two different trips rarely land in the same tiles — so
    it is fetched and written per bundle rather than through a shared cache.

    ``trip_date_imagery=True`` opts into the trip-date true-colour scene via
    ``gibs_scene_for``'s existing cloud-fraction fallback, untouched here;
    Blue Marble is graded whichever way that decision lands, the daily
    scene never is (only Blue Marble measured dark). Whichever raster is
    Blue Marble also gets the coastline/border overlay, graded first then
    bordered; a chosen daily scene never does — its own imagery already
    shows real coastline. Returns ``(raster, decision, borders_provenance)``;
    ``borders_provenance`` is ``None`` whenever no borders fetch was made.
    """
    width, height = _gibs_request_size(band)
    if not trip_date_imagery:
        raster = fetch_gibs_blue_marble(session, band.bbox, width=width, height=height)
        graded = _regrade(raster)
        bordered, borders_decision, borders_provenance = _bordered(
            session, graded, band.bbox, width, height
        )
        decision = {
            "chosen": "blue_marble",
            "cloud_fraction": None,
            "threshold": None,
            "trip_date_imagery": False,
            "borders": borders_decision,
        }
        return bordered, decision, borders_provenance

    raster, decision = gibs_scene_for(session, band.bbox, recorded_on, width=width, height=height)
    decision = {**decision, "trip_date_imagery": True}
    if decision["chosen"] == "blue_marble":
        graded = _regrade(raster)
        raster, borders_decision, borders_provenance = _bordered(
            session, graded, band.bbox, width, height
        )
        decision["borders"] = borders_decision
    else:
        decision["borders"] = None
        borders_provenance = None
    return raster, decision, borders_provenance


def _empty_tile_report() -> TilePyramidReport:
    return TilePyramidReport(written=[], skipped_partial=[])


def _blue_marble_shared_dir() -> Path:
    return gms_route_story_cache_dir() / "shared" / "blue-marble" / BLUE_MARBLE_CACHE_VERSION


def _blue_marble_provenance_path(shared_dir: Path) -> Path:
    return shared_dir / "provenance.json"


def _blue_marble_borders_provenance_path(shared_dir: Path) -> Path:
    return shared_dir / "borders-provenance.json"


def _imagery_tile_path(root: Path, z: int, x: int, y: int) -> Path:
    return root / "imagery" / str(z) / str(x) / f"{y}.jpg"


def _band_needed_tiles(
    band: ZoomBand, requested_zooms: Sequence[int]
) -> list[tuple[int, int, int]]:
    """Every ``(z, x, y)`` ``band`` would write for the zooms ``requested_zooms`` asks for.

    Every zoom in ``band.zooms`` is fully covered by ``band.bbox`` by
    construction (see ``bands.py``'s module docstring), so the tile list is
    knowable from geometry alone, before any raster is fetched — this is
    what lets the shared Blue Marble cache decide whether a fetch is needed
    at all without touching the network first.
    """
    zooms = sorted(set(band.zooms) & set(requested_zooms))
    footprint = band_footprint(band)
    return [(z, x, y) for z in zooms for x, y in tiles_covering(footprint, z)]


# A fallback identical to what `sources.fetch_gibs_blue_marble` itself would
# record, used only when nothing has ever been fetched into the shared cache
# (an `imagery_zooms` that excludes 2-4 entirely) — the band still needs a
# provenance entry in the manifest even though it wrote no tiles.
_BLUE_MARBLE_PROVENANCE_FALLBACK = SourceProvenance(
    name="NASA GIBS BlueMarble_ShadedRelief_Bathymetry",
    url="https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
    licence="NASA open data (attribution requested)",
    attribution="NASA EOSDIS GIBS / Worldview",
    acquired="unknown",
)

# Same reasoning, for the shared band's own borders overlay fetch.
_BLUE_MARBLE_BORDERS_PROVENANCE_FALLBACK = SourceProvenance(
    name="NASA GIBS Reference_Features_15m",
    url="https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi",
    licence="unverified",
    attribution="NASA EOSDIS GIBS / Worldview",
    acquired="unknown",
)


def _ensure_blue_marble_tiles(
    session: Any, band: ZoomBand, requested_zooms: Sequence[int]
) -> tuple[SourceProvenance, SourceProvenance, list[tuple[int, int, int]], int]:
    """Fetch and write the shared z2-4 Blue Marble tiles only when the cache lacks any.

    The tiles live under a cache dir keyed only by ``BLUE_MARBLE_CACHE_VERSION``
    — never by footprint or ``recorded_on`` — so a second bundle whose
    ``blue_marble`` band lands in the same already-cached tiles makes zero
    Blue Marble HTTP calls. Returns ``(provenance, borders_provenance,
    needed_tiles, fetched_tile_count)``; ``needed_tiles`` is every tile this
    band contributes at ``requested_zooms``, whether just fetched or already
    sitting in the cache. ``borders_provenance`` records the source of the
    overlay baked into these shared tiles — persisted alongside the base
    provenance so a cache-hit bundle can still name it in its own manifest.
    """
    shared_dir = _blue_marble_shared_dir()
    needed = _band_needed_tiles(band, requested_zooms)
    provenance_path = _blue_marble_provenance_path(shared_dir)
    borders_provenance_path = _blue_marble_borders_provenance_path(shared_dir)
    missing = [t for t in needed if not _imagery_tile_path(shared_dir, *t).is_file()]
    fetched = 0
    if needed and (
        missing or not provenance_path.is_file() or not borders_provenance_path.is_file()
    ):
        band_zooms = sorted(set(band.zooms) & set(requested_zooms))
        width, height = _gibs_request_size(band)
        raster = fetch_gibs_blue_marble(session, band.bbox, width=width, height=height)
        overlay = fetch_gibs_reference_features(session, band.bbox, width=width, height=height)
        coverage = border_coverage(overlay.array)
        if coverage < MIN_SHARED_BORDER_COVERAGE:
            # Hard fail BEFORE any directory is created or tile written: a
            # near-empty overlay here means a failed/near-empty fetch over a
            # whole-globe bbox that always has coastline, never a
            # legitimately borderless one — shipping this ungraded/
            # borderless would silently poison every bundle sharing this
            # cache.
            raise SourceFetchError(
                f"Blue Marble reference-features coverage too low ({coverage:.4f}) for the "
                f"shared z2-4 band bbox {band.bbox}; refusing to write borderless tiles to "
                "the shared cache"
            )
        graded = grade_backdrop(raster.array)
        bordered = overlay_borders(graded, overlay.array)
        shared_dir.mkdir(parents=True, exist_ok=True)
        report = write_tile_pyramid(
            shared_dir,
            "imagery",
            bordered,
            band_footprint(band),
            band_zooms,
            resample="lanczos",
        )
        fetched = len(report.written)
        provenance_path.write_text(json.dumps(asdict(raster.provenance)), encoding="utf-8")
        borders_provenance_path.write_text(json.dumps(asdict(overlay.provenance)), encoding="utf-8")
    if provenance_path.is_file():
        provenance = SourceProvenance(**json.loads(provenance_path.read_text(encoding="utf-8")))
    else:
        provenance = _BLUE_MARBLE_PROVENANCE_FALLBACK
    if borders_provenance_path.is_file():
        borders_provenance = SourceProvenance(
            **json.loads(borders_provenance_path.read_text(encoding="utf-8"))
        )
    else:
        borders_provenance = _BLUE_MARBLE_BORDERS_PROVENANCE_FALLBACK
    return provenance, borders_provenance, needed, fetched


def _install_blue_marble_tiles(
    bundle_root: Path, shared_dir: Path, needed: Sequence[tuple[int, int, int]]
) -> tuple[TilePyramidReport, int, int]:
    """Hard-link (copy on ``OSError``, e.g. across filesystems) each needed tile
    from the shared cache into the bundle's own imagery tree, at the same
    z/x/y path — the bundle stays self-contained since MapLibre only ever
    serves the bundle dir, never the shared cache. An existing destination
    (a re-prepare into a bundle dir that failed verification) is replaced,
    never copied onto: it may already be a hard link to ``source``, and
    ``copy2`` refuses to copy a file onto itself.
    """
    linked = 0
    copied = 0
    written: list[tuple[int, int, int]] = []
    for tile in needed:
        source = _imagery_tile_path(shared_dir, *tile)
        if not source.is_file():
            continue  # nothing fetched for this tile (band_zooms excluded it)
        destination = _imagery_tile_path(bundle_root, *tile)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.unlink(missing_ok=True)
        try:
            os.link(source, destination)
            linked += 1
        except OSError:
            fs_safety.safe_copy(source, destination)
            copied += 1
        written.append(tile)
    return TilePyramidReport(written=sorted(written), skipped_partial=[]), linked, copied


def _write_band(
    bundle_root: Path,
    raster: GeoRaster,
    band: ZoomBand,
    imagery_zooms: Sequence[int],
    *,
    extent: Footprint | None,
) -> TilePyramidReport:
    """Write one imagery band's tiles, restricted to the requested zoom range.

    A band writes only the intersection of its own zoom range with the
    caller's ``imagery_zooms`` — a narrower request never makes a wider band
    write zooms nobody asked for, and a band entirely outside the request
    writes nothing at all.
    """
    band_zooms = sorted(set(band.zooms) & set(imagery_zooms))
    if not band_zooms:
        return _empty_tile_report()
    return write_tile_pyramid(
        bundle_root,
        "imagery",
        raster.array,
        _raster_footprint(raster),
        band_zooms,
        resample="lanczos",
        extent=extent,
    )


def _written_zoom_range_or_none(report: TilePyramidReport) -> dict[str, int] | None:
    zooms = sorted({z for z, _x, _y in report.written})
    return {"minzoom": zooms[0], "maxzoom": zooms[-1]} if zooms else None


def _write_terrain_band(
    bundle_root: Path, raster: GeoRaster, band: ZoomBand, terrain_zooms: Sequence[int]
) -> TilePyramidReport:
    """Write one terrain band's tiles, restricted to the requested zoom range.

    Mirrors ``_write_band`` for imagery: a band writes only the intersection
    of its own zoom range with the caller's ``terrain_zooms``, and a band
    entirely outside the request writes nothing at all.
    """
    band_zooms = sorted(set(band.zooms) & set(terrain_zooms))
    if not band_zooms:
        return _empty_tile_report()
    return write_tile_pyramid(
        bundle_root, "terrain", raster.array, _raster_footprint(raster), band_zooms
    )


def _dedup_provenances(provenances: Sequence[SourceProvenance]) -> list[dict[str, Any]]:
    """``manifest["terrain"]["sources"]``: both bands' provenances, de-duplicated by name
    (the focal and regional DEM fetches often resolve to the same source)."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for provenance in provenances:
        if provenance.name not in seen:
            seen.add(provenance.name)
            result.append(asdict(provenance))
    return result


def _terrain_bands_manifest(
    bands: Sequence[ZoomBand],
    reports: dict[str, TilePyramidReport],
    provenances: dict[str, SourceProvenance],
) -> list[dict[str, Any]]:
    """Per-band manifest detail: the plan (``bands_receipt``) plus what actually
    got written for each terrain band."""
    entries = []
    for receipt, band in zip(bands_receipt(bands), bands):
        report = reports[band.source]
        entries.append(
            {
                **receipt,
                "tiles_written": len(report.written),
                "tiles_skipped": len(report.skipped_partial),
                "written_zoom_range": _written_zoom_range_or_none(report),
                "provenance": asdict(provenances[band.source]),
            }
        )
    return entries


def _imagery_bands_manifest(
    bands: Sequence[ZoomBand],
    reports: dict[str, TilePyramidReport],
    provenances: dict[str, SourceProvenance],
    gibs_decision: dict[str, Any],
    blue_marble_stats: dict[str, int],
) -> list[dict[str, Any]]:
    """Per-band manifest detail: the plan (``bands_receipt``) plus what actually
    got fetched and written for each band — the cloud call for gibs, and the
    shared-cache linked/copied/fetched tile counts for blue_marble."""
    entries = []
    for receipt, band in zip(bands_receipt(bands), bands):
        report = reports[band.source]
        entry = {
            **receipt,
            "tiles_written": len(report.written),
            "tiles_skipped": len(report.skipped_partial),
            "written_zoom_range": _written_zoom_range_or_none(report),
            "provenance": asdict(provenances[band.source]),
        }
        if band.source == "gibs":
            entry["cloud_decision"] = gibs_decision
        if band.source == "blue_marble":
            entry["shared_cache"] = True
            entry.update(blue_marble_stats)
        entries.append(entry)
    return entries


def prepare_bundle(
    gpx_path: Path,
    session: Any,
    fetch_maplibre: Callable[[str], bytes],
    *,
    recorded_on: date,
    margin_m: float = DEFAULT_MARGIN_M,
    imagery_zooms: Sequence[int] = DEFAULT_IMAGERY_ZOOMS,
    terrain_zooms: Sequence[int] = DEFAULT_TERRAIN_ZOOMS,
    imagery_metres_per_px: float = IMAGERY_METRES_PER_PX,
    regional_margin_km: float = DEFAULT_REGIONAL_MARGIN_KM,
    overpass_endpoint: str | None = DEFAULT_OVERPASS_ENDPOINT,
    terrain_exaggeration: float = DEFAULT_EXAGGERATION,
    gnis_cache_dir: Path | None = None,
    label_sources: Sequence[str] = DEFAULT_LABEL_SOURCES,
    trip_date_imagery: bool = DEFAULT_TRIP_DATE_IMAGERY,
) -> PrepareReport:
    """Build, write, and verify a Route Story bundle for one GPX trip.

    ``recorded_on`` selects the single calendar day the bundle is built
    from (``select_gpx_day``) — a GPX file carrying other-date or
    duplicated tracks (a common real-world export) never widens the
    footprint or double-draws the route. Returns the same content-addressed
    directory on a repeat call for the same footprint + source-version
    combination (``bundle_dir_for``); a bundle whose manifest already
    verifies clean is not re-fetched.

    ``regional_margin_km`` is passed identically to both `plan_imagery_bands`
    and `plan_terrain_bands`, so the regional imagery (``naip_coarse``) and
    regional terrain (``terrain_regional``) bands always share the exact same
    bbox regardless of its value — a caller (a test, mainly) that shrinks it
    gets a proportionally smaller z9-12 tile envelope at the default's ~30
    m/px terrain resolution and NAIP resolution unchanged.

    ``trip_date_imagery`` (default ``False``) governs only the z5-8 ``gibs``
    band: off, it renders the dateless, cloud-free Blue Marble backdrop; on,
    it opts into the trip-date true-colour scene (which can carry cloud) via
    the existing cloud-fraction fallback. See ``_gibs_band_raster``.
    """
    if not (set(imagery_zooms) & set(FOCAL_IMAGERY_ZOOMS)):
        raise ValueError(
            "imagery_zooms must include at least one of the focal zooms "
            f"{FOCAL_IMAGERY_ZOOMS.start}-{FOCAL_IMAGERY_ZOOMS.stop - 1}: got {tuple(imagery_zooms)}"
        )
    selection, footprint = _select_and_footprint(gpx_path, recorded_on, margin_m)
    bbox = _bbox(footprint)
    terrain_exaggeration = _clamp_exaggeration(terrain_exaggeration)
    effective_gnis_cache_dir = (
        gnis_cache_dir if gnis_cache_dir is not None else gms_route_story_cache_dir() / "gnis"
    )

    cached = _find_cached_bundle(
        footprint,
        imagery_zooms,
        terrain_zooms,
        recorded_on,
        terrain_exaggeration,
        regional_margin_km,
        trip_date_imagery,
    )
    if cached is not None:
        return _repair_degraded_labels(
            session,
            cached,
            route_bbox=bbox,
            gnis_cache_dir=effective_gnis_cache_dir,
            overpass_endpoint=overpass_endpoint,
            label_sources=label_sources,
        )

    imagery_bands = plan_imagery_bands(
        bbox, focal_margin_m=margin_m, regional_margin_km=regional_margin_km
    )
    blue_marble_band = next(b for b in imagery_bands if b.source == "blue_marble")
    gibs_band = next(b for b in imagery_bands if b.source == "gibs")
    naip_coarse_band = next(b for b in imagery_bands if b.source == "naip_coarse")
    naip_mid_band = next(b for b in imagery_bands if b.source == "naip_mid")
    naip_fine_band = next(b for b in imagery_bands if b.source == "naip_fine")
    naip_band = next(b for b in imagery_bands if b.source == "naip")

    terrain_bands = plan_terrain_bands(
        bbox, focal_margin_m=margin_m, regional_margin_km=regional_margin_km
    )
    terrain_regional_band = next(b for b in terrain_bands if b.source == "terrain_regional")
    terrain_focal_band = next(b for b in terrain_bands if b.source == "terrain_focal")

    terrain_focal_raster, terrain_focal_fallback = _fetch_terrain(session, terrain_focal_band.bbox)
    terrain_regional_raster, terrain_regional_fallback = _fetch_terrain(
        session, terrain_regional_band.bbox, metres_per_px=terrain_regional_band.target_m_per_px
    )
    terrain_fallback_used = terrain_focal_fallback or terrain_regional_fallback

    (
        blue_marble_provenance,
        blue_marble_borders_provenance,
        blue_marble_needed,
        blue_marble_fetched,
    ) = _ensure_blue_marble_tiles(session, blue_marble_band, imagery_zooms)

    gibs_raster, gibs_decision, gibs_borders_provenance = _gibs_band_raster(
        session, gibs_band, recorded_on, trip_date_imagery=trip_date_imagery
    )
    naip_coarse_raster = fetch_naip_imagery(
        session, naip_coarse_band.bbox, metres_per_px=naip_coarse_band.target_m_per_px
    )
    naip_mid_raster = fetch_naip_imagery(
        session, naip_mid_band.bbox, metres_per_px=naip_mid_band.target_m_per_px
    )
    naip_fine_raster = fetch_naip_imagery(
        session, naip_fine_band.bbox, metres_per_px=naip_fine_band.target_m_per_px
    )
    imagery_raster = fetch_naip_imagery(
        session, naip_band.bbox, metres_per_px=imagery_metres_per_px
    )
    label_features, label_provenances, labels_degraded, label_problems = _fetch_labels(
        session,
        naip_coarse_band.bbox,
        gnis_cache_dir=effective_gnis_cache_dir,
        overpass_endpoint=overpass_endpoint,
        label_sources=label_sources,
    )
    label_features = _filter_context_labels(label_features, bbox)

    source_versions = {
        **_band_source_versions(
            gibs_chosen=gibs_decision["chosen"],
            recorded_on=recorded_on,
            terrain_focal_name=terrain_focal_raster.provenance.name,
            terrain_regional_name=terrain_regional_raster.provenance.name,
            regional_margin_km=regional_margin_km,
        ),
    }
    bundle_root = bundle_dir_for(
        footprint,
        source_versions,
        terrain_exaggeration=terrain_exaggeration,
        imagery_zooms=imagery_zooms,
        terrain_zooms=terrain_zooms,
    )
    bundle_root.mkdir(parents=True, exist_ok=True)

    blue_marble_report, blue_marble_linked, blue_marble_copied = _install_blue_marble_tiles(
        bundle_root, _blue_marble_shared_dir(), blue_marble_needed
    )

    # Terrain writes first: each band's own tile grid (coarser tiles truncate
    # more against the raster edge) becomes the extent its matching imagery
    # band shares, so no imagery band is ever written wider than the mesh
    # actually under it (the imagery-inside-terrain rule, applied per band). Blue Marble and gibs stay
    # national/wide-margin and are meant to reach past any terrain mesh, so
    # they still write their own band footprint.
    terrain_regional_report = _write_terrain_band(
        bundle_root, terrain_regional_raster, terrain_regional_band, terrain_zooms
    )
    terrain_focal_report = _write_terrain_band(
        bundle_root, terrain_focal_raster, terrain_focal_band, terrain_zooms
    )
    combined_terrain_report = TilePyramidReport(
        written=sorted(terrain_regional_report.written + terrain_focal_report.written),
        skipped_partial=sorted(
            terrain_regional_report.skipped_partial + terrain_focal_report.skipped_partial
        ),
    )

    focal_terrain_zooms_written = sorted({z for z, _x, _y in terrain_focal_report.written})
    shared_extent = (
        written_tiles_footprint(terrain_focal_report, focal_terrain_zooms_written[-1])
        if focal_terrain_zooms_written
        else None
    )
    regional_terrain_zooms_written = sorted({z for z, _x, _y in terrain_regional_report.written})
    regional_terrain_extent = (
        written_tiles_footprint(terrain_regional_report, regional_terrain_zooms_written[-1])
        if regional_terrain_zooms_written
        else band_footprint(naip_coarse_band)
    )

    gibs_report = _write_band(
        bundle_root, gibs_raster, gibs_band, imagery_zooms, extent=band_footprint(gibs_band)
    )
    naip_coarse_report = _write_band(
        bundle_root,
        naip_coarse_raster,
        naip_coarse_band,
        imagery_zooms,
        extent=regional_terrain_extent,
    )
    # Each imagery band is bound by the extent of the terrain band whose zooms
    # actually cover it: never reach past the mesh on disk. `naip_mid` is
    # z11-12, inside `terrain_regional`'s z9-12, so it shares the REGIONAL
    # extent -- it moved off `shared_extent` when it gave up z13-14.
    naip_mid_report = _write_band(
        bundle_root,
        naip_mid_raster,
        naip_mid_band,
        imagery_zooms,
        extent=regional_terrain_extent,
    )
    # `naip_fine` (z13-14) is what now shares `terrain_focal`'s bbox and
    # written tile footprint (bands.py's `_naip_fine_bbox`) -- `terrain_focal`
    # is z13 only, which sits inside `naip_fine`'s range. `naip` still has no
    # terrain zoom of its own past z13, so it stays bound by the same
    # `shared_extent` it always was.
    naip_fine_report = _write_band(
        bundle_root, naip_fine_raster, naip_fine_band, imagery_zooms, extent=shared_extent
    )
    imagery_report = _write_band(
        bundle_root, imagery_raster, naip_band, imagery_zooms, extent=shared_extent
    )
    combined_imagery_report = TilePyramidReport(
        written=sorted(
            blue_marble_report.written
            + gibs_report.written
            + naip_coarse_report.written
            + naip_mid_report.written
            + naip_fine_report.written
            + imagery_report.written
        ),
        skipped_partial=sorted(
            gibs_report.skipped_partial
            + naip_coarse_report.skipped_partial
            + naip_mid_report.skipped_partial
            + naip_fine_report.skipped_partial
            + imagery_report.skipped_partial
        ),
    )

    (bundle_root / "route.geojson").write_text(
        json.dumps(_route_geojson(selection)), encoding="utf-8"
    )
    _write_labels(bundle_root, label_features)
    ensure_maplibre(bundle_root, fetch_maplibre)

    manifest_fields = {
        "footprint": {
            "west": footprint.west,
            "south": footprint.south,
            "east": footprint.east,
            "north": footprint.north,
        },
        "imagery": {
            **_written_zoom_range("imagery", combined_imagery_report),
            # The top-level extent stays the focal band's own — the wide
            # gibs/naip_coarse bands are meant to reach past the terrain
            # mesh, so only the focal extent is meaningful to the imagery-inside-terrain check.
            **_kind_extent_fields(imagery_report, _raster_footprint(imagery_raster)),
            "tile_size": 256,
            "format": "jpg",
            # Deduplicated by name: the shared z2-4 band and the per-bundle
            # z5-8 gibs band both draw borders from the same GIBS layer in
            # the default (cloud-free) mode, so their provenances collide —
            # a chosen trip-date daily scene drops the gibs band's own
            # (`gibs_borders_provenance` is None then), but the shared
            # band's is still named here.
            "sources": _dedup_provenances(
                [
                    blue_marble_provenance,
                    gibs_raster.provenance,
                    naip_coarse_raster.provenance,
                    naip_mid_raster.provenance,
                    naip_fine_raster.provenance,
                    imagery_raster.provenance,
                    blue_marble_borders_provenance,
                    *([gibs_borders_provenance] if gibs_borders_provenance is not None else []),
                ]
            ),
            "bands": _imagery_bands_manifest(
                imagery_bands,
                {
                    "blue_marble": blue_marble_report,
                    "gibs": gibs_report,
                    "naip_coarse": naip_coarse_report,
                    "naip_mid": naip_mid_report,
                    "naip_fine": naip_fine_report,
                    "naip": imagery_report,
                },
                {
                    "blue_marble": blue_marble_provenance,
                    "gibs": gibs_raster.provenance,
                    "naip_coarse": naip_coarse_raster.provenance,
                    "naip_mid": naip_mid_raster.provenance,
                    "naip_fine": naip_fine_raster.provenance,
                    "naip": imagery_raster.provenance,
                },
                gibs_decision,
                {
                    "tiles_linked": blue_marble_linked,
                    "tiles_copied": blue_marble_copied,
                    "tiles_fetched": blue_marble_fetched,
                },
            ),
        },
        "terrain": {
            **_written_zoom_range("terrain", combined_terrain_report),
            # The top-level extent stays the focal band's own, exactly as
            # before the regional band existed — the imagery-inside-
            # terrain check compares against this focal extent only.
            **_kind_extent_fields(terrain_focal_report, _raster_footprint(terrain_focal_raster)),
            "tile_size": 256,
            "encoding": "mapbox",
            "exaggeration": terrain_exaggeration,
            "sources": _dedup_provenances(
                [terrain_focal_raster.provenance, terrain_regional_raster.provenance]
            ),
            "vertical_datum": "unknown",
            "bands": _terrain_bands_manifest(
                terrain_bands,
                {
                    "terrain_regional": terrain_regional_report,
                    "terrain_focal": terrain_focal_report,
                },
                {
                    "terrain_regional": terrain_regional_raster.provenance,
                    "terrain_focal": terrain_focal_raster.provenance,
                },
            ),
        },
        "labels": {
            "sources": [asdict(p) for p in label_provenances],
            "degraded": labels_degraded,
            "problems": list(label_problems),
            "count": len(label_features),
            "bbox": list(naip_coarse_band.bbox),
        },
    }
    write_manifest(bundle_root, manifest_fields)
    problems = verify_bundle(bundle_root)

    return PrepareReport(
        bundle_dir=bundle_root,
        footprint=footprint,
        imagery_source=imagery_raster.provenance.name,
        terrain_source=terrain_focal_raster.provenance.name,
        terrain_fallback_used=terrain_fallback_used,
        imagery_zooms=tuple(sorted(set(imagery_zooms))),
        terrain_zooms=tuple(sorted(set(terrain_zooms))),
        imagery_tiles_written=len(combined_imagery_report.written),
        imagery_tiles_skipped=len(combined_imagery_report.skipped_partial),
        terrain_tiles_written=len(combined_terrain_report.written),
        terrain_tiles_skipped=len(combined_terrain_report.skipped_partial),
        label_count=len(label_features),
        excluded_other_date_count=selection.excluded_other_date_count,
        excluded_duplicate_count=selection.excluded_duplicate_count,
        verify_problems=problems,
        labels_degraded=labels_degraded,
        label_problems=label_problems,
    )
