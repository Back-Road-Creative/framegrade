"""Offline Route Story map asset bundle: footprint, Web-Mercator tile math,
Terrain-RGB codec, tile pyramid writer, and content-addressed manifest.

Bundle layout (shared verbatim with the fetch and scene code)::

    <bundle>/manifest.json
    <bundle>/imagery/{z}/{x}/{y}.jpg        256px XYZ, Web Mercator
    <bundle>/terrain/{z}/{x}/{y}.png        256px Terrain-RGB (Mapbox encoding)
    <bundle>/labels.geojson
    <bundle>/roads.geojson                  OSM highway LineStrings (schema /2 onward)
    <bundle>/route.geojson
    <bundle>/vendor/maplibre-gl-6.10.0/

Nothing in this module makes a network call. It only knows how to shape,
encode, and checksum a bundle whose raw imagery/terrain arrays already live
in memory — fetching those arrays is ``sources.py``'s job.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from ...paths import gms_route_story_cache_dir
from .metrics import RoutePoint

MANIFEST_SCHEMA = "route-story-bundle/2"
# /1 is the legacy layout without ``roads.geojson``; both stay verifiable.
MANIFEST_SCHEMAS = ("route-story-bundle/1", MANIFEST_SCHEMA)
TILE_SIZE = 256

_EARTH_RADIUS_M = 6_371_008.8
_WEB_MERCATOR_LAT_LIMIT = 85.05112877980659

# Mapbox Terrain-RGB encoding: height_m = -10000 + (R*65536+G*256+B) * 0.1
_TERRAIN_BASE_M = -10000.0
_TERRAIN_STEP_M = 0.1
_TERRAIN_MAX_UNIT = 256**3 - 1


# ---------------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Footprint:
    """A validated geographic bounding box."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not (-180.0 <= self.west <= 180.0 and -180.0 <= self.east <= 180.0):
            raise ValueError("footprint longitude must be in [-180, 180]")
        if not (-90.0 <= self.south <= 90.0 and -90.0 <= self.north <= 90.0):
            raise ValueError("footprint latitude must be in [-90, 90]")
        if not self.west < self.east:
            raise ValueError("footprint west must be less than east")
        if not self.south < self.north:
            raise ValueError("footprint south must be less than north")


def footprint_for_route(points: Sequence[RoutePoint], margin_m: float) -> Footprint:
    """Expand a route's bounding box by a metric margin.

    Longitude margin is widened by ``1/cos(lat)`` at the route's mid
    latitude so the margin is a genuine metric distance, not a fixed
    number of degrees that shrinks toward the poles.
    """
    if not points:
        raise ValueError("footprint_for_route needs at least one point")
    if margin_m < 0:
        raise ValueError("margin_m must be non-negative")
    lats = [p.latitude for p in points]
    lons = [p.longitude for p in points]
    south, north = min(lats), max(lats)
    west, east = min(lons), max(lons)
    lat_margin_deg = (margin_m / _EARTH_RADIUS_M) * (180.0 / math.pi)
    cos_lat = max(math.cos(math.radians((south + north) / 2.0)), 1e-9)
    lon_margin_deg = lat_margin_deg / cos_lat
    return Footprint(
        west=max(west - lon_margin_deg, -180.0),
        south=max(south - lat_margin_deg, -90.0),
        east=min(east + lon_margin_deg, 180.0),
        north=min(north + lat_margin_deg, 90.0),
    )


# ---------------------------------------------------------------------------
# Web-Mercator tile math
# ---------------------------------------------------------------------------


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    """Return the fractional (x, y) XYZ tile coordinate of a lon/lat at zoom z."""
    if not -180.0 <= lon <= 180.0:
        raise ValueError("lon must be in [-180, 180]")
    if not -_WEB_MERCATOR_LAT_LIMIT < lat < _WEB_MERCATOR_LAT_LIMIT:
        raise ValueError("lat outside the Web Mercator range")
    if z < 0:
        raise ValueError("z must be non-negative")
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def _mercator_y_to_lat(y: float, z: int) -> float:
    n = 2**z
    m = math.pi * (1.0 - 2.0 * y / n)
    return math.degrees(math.atan(math.sinh(m)))


def tile_bounds(z: int, x: int, y: int) -> Footprint:
    """Return the geographic bounds of one XYZ tile."""
    n = 2**z
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = _mercator_y_to_lat(y, z)
    south = _mercator_y_to_lat(y + 1, z)
    return Footprint(west=west, south=south, east=east, north=north)


def tiles_covering(footprint: Footprint, z: int) -> list[tuple[int, int]]:
    """Return the sorted list of tile (x, y) covering a footprint at zoom z."""
    n = 2**z
    north = min(footprint.north, _WEB_MERCATOR_LAT_LIMIT)
    south = max(footprint.south, -_WEB_MERCATOR_LAT_LIMIT)
    x_west, y_north = lonlat_to_tile(footprint.west, north, z)
    x_east, y_south = lonlat_to_tile(footprint.east, south, z)
    x_min = max(0, int(math.floor(x_west + 1e-9)))
    x_max = min(n - 1, int(math.floor(x_east - 1e-9)))
    y_min = max(0, int(math.floor(y_north + 1e-9)))
    y_max = min(n - 1, int(math.floor(y_south - 1e-9)))
    if x_max < x_min or y_max < y_min:
        return []
    return sorted((x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1))


# ---------------------------------------------------------------------------
# Terrain-RGB codec
# ---------------------------------------------------------------------------


def encode_terrain_rgb(heights: np.ndarray) -> np.ndarray:
    """Encode float32 metre heights as Mapbox Terrain-RGB uint8 (H, W, 3)."""
    heights = np.asarray(heights, dtype=np.float64)
    if np.isnan(heights).any():
        raise ValueError("terrain heights must not contain NaN")
    unit = (heights - _TERRAIN_BASE_M) / _TERRAIN_STEP_M
    if unit.min() < 0 or unit.max() > _TERRAIN_MAX_UNIT:
        raise ValueError("terrain heights are outside the encodable Terrain-RGB range")
    unit = np.round(unit).astype(np.uint32)
    # Shifts into a preallocated uint8 array: the uint8 cast keeps the low byte,
    # the same bytes as ``(unit // 256**k) % 256`` at ~1/4 the cost per tile
    # (every terrain tile of a prepare goes through here).
    out = np.empty(unit.shape + (3,), dtype=np.uint8)
    out[..., 0] = unit >> 16
    out[..., 1] = unit >> 8
    out[..., 2] = unit
    return out


def decode_terrain_rgb(rgb: np.ndarray) -> np.ndarray:
    """Decode Mapbox Terrain-RGB uint8 (H, W, 3) back to float64 metre heights."""
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError("terrain-rgb array must have shape (H, W, 3)")
    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    return _TERRAIN_BASE_M + (r * 65536.0 + g * 256.0 + b) * _TERRAIN_STEP_M


# ---------------------------------------------------------------------------
# Tile pyramid writer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TilePyramidReport:
    """Which tiles a ``write_tile_pyramid`` call actually wrote vs skipped."""

    written: list[tuple[int, int, int]]
    skipped_partial: list[tuple[int, int, int]]


def _fully_covered(tile: Footprint, source: Footprint, eps: float = 1e-9) -> bool:
    return (
        tile.west >= source.west - eps
        and tile.east <= source.east + eps
        and tile.south >= source.south - eps
        and tile.north <= source.north + eps
    )


def _tile_lonlat_grid(
    z: int, x: int, y: int, size: int = TILE_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon, lat) pixel-center grids, shape (size, size), for tile z/x/y."""
    n = 2**z
    col = np.arange(size) + 0.5
    row = np.arange(size) + 0.5
    lon = (x + col / size) / n * 360.0 - 180.0
    y_frac = y + row / size
    m = math.pi * (1.0 - 2.0 * y_frac / n)
    lat = np.degrees(np.arctan(np.sinh(m)))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    return lon_grid, lat_grid


def _to_source_rowcol(
    lon: np.ndarray, lat: np.ndarray, source_footprint: Footprint, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    col = (lon - source_footprint.west) / (source_footprint.east - source_footprint.west) * (w - 1)
    row = (
        (source_footprint.north - lat) / (source_footprint.north - source_footprint.south) * (h - 1)
    )
    return row, col


def _bilinear(source: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``source`` at fractional ``(row, col)``, in float64.

    Only the gathered corner pixels are converted to float64 -- never the
    whole raster. Callers sample one 256x256 tile at a time, so converting
    the source up front cost a full-raster copy per tile (minutes per band
    for a regional Copernicus DEM at 3600 px/degree).
    """
    h, w = source.shape[:2]
    row0 = np.clip(np.floor(row).astype(int), 0, h - 1)
    col0 = np.clip(np.floor(col).astype(int), 0, w - 1)
    row1 = np.clip(row0 + 1, 0, h - 1)
    col1 = np.clip(col0 + 1, 0, w - 1)
    fr = np.clip(row - row0, 0.0, 1.0)
    fc = np.clip(col - col0, 0.0, 1.0)
    if source.ndim == 3:
        fr = fr[..., None]
        fc = fc[..., None]

    def corner(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return source[r, c].astype(np.float64)

    top = corner(row0, col0) * (1 - fc) + corner(row0, col1) * fc
    bottom = corner(row1, col0) * (1 - fc) + corner(row1, col1) * fc
    return top * (1 - fr) + bottom * fr


def _sample_tile_bilinear(
    source: np.ndarray, source_footprint: Footprint, z: int, x: int, y: int
) -> np.ndarray:
    """``_bilinear`` over tile z/x/y's pixel centres, gathered on 1-D axes.

    A tile's lon depends only on its column and its lat only on its row, so
    the corner gathers index with ``np.ix_`` over one row axis and one column
    axis instead of two full ``TILE_SIZE`` x ``TILE_SIZE`` index grids. Same
    operations per pixel, so the result is bit-identical to sampling the
    meshgrid -- it only skips building and indexing with the redundant grids.
    """
    n = 2**z
    centres = np.arange(TILE_SIZE) + 0.5
    lon = (x + centres / TILE_SIZE) / n * 360.0 - 180.0
    m = math.pi * (1.0 - 2.0 * (y + centres / TILE_SIZE) / n)
    lat = np.degrees(np.arctan(np.sinh(m)))
    row, col = _to_source_rowcol(lon, lat, source_footprint, source.shape[:2])
    h, w = source.shape[:2]
    row0 = np.clip(np.floor(row).astype(int), 0, h - 1)
    col0 = np.clip(np.floor(col).astype(int), 0, w - 1)
    row1 = np.clip(row0 + 1, 0, h - 1)
    col1 = np.clip(col0 + 1, 0, w - 1)
    fr = np.clip(row - row0, 0.0, 1.0)[:, None]
    fc = np.clip(col - col0, 0.0, 1.0)[None, :]
    if source.ndim == 3:
        fr = fr[..., None]
        fc = fc[..., None]

    def corner(r: np.ndarray, c: np.ndarray) -> np.ndarray:
        return source[np.ix_(r, c)].astype(np.float64)

    top = corner(row0, col0) * (1 - fc) + corner(row0, col1) * fc
    bottom = corner(row1, col0) * (1 - fc) + corner(row1, col1) * fc
    return top * (1 - fr) + bottom * fr


def _sample_imagery(
    source_array: np.ndarray,
    source_footprint: Footprint,
    lon: np.ndarray,
    lat: np.ndarray,
    resample: str,
) -> np.ndarray:
    row, col = _to_source_rowcol(lon, lat, source_footprint, source_array.shape[:2])
    if resample == "bilinear":
        return _bilinear(source_array, row, col)
    # lanczos: crop the covering source window and let Pillow resample it.
    # Valid because callers only reach here for tiles fully covered by the
    # source raster (write_tile_pyramid skips partial tiles beforehand).
    h, w = source_array.shape[:2]
    row_min = int(np.clip(math.floor(row.min()), 0, h - 1))
    row_max = int(np.clip(math.ceil(row.max()), 0, h - 1))
    col_min = int(np.clip(math.floor(col.min()), 0, w - 1))
    col_max = int(np.clip(math.ceil(col.max()), 0, w - 1))
    crop = source_array[row_min : row_max + 1, col_min : col_max + 1]
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return _bilinear(source_array, row, col)
    resized = Image.fromarray(crop.astype(np.uint8), mode="RGB").resize(
        (TILE_SIZE, TILE_SIZE), Image.Resampling.LANCZOS
    )
    return np.asarray(resized, dtype=np.float64)


def written_tiles_footprint(report: TilePyramidReport, zoom: int) -> Footprint | None:
    """Return the union geographic bounds of the tiles ``report`` wrote at ``zoom``.

    ``None`` when nothing was written at that zoom. This is the *actual*
    tile-grid extent, which a coarser zoom's larger tiles can truncate more
    than a finer zoom's — the mismatch behind two map layers rendering with
    different footprints even when both were fetched for the same bbox.
    """
    tiles = [(x, y) for z_, x, y in report.written if z_ == zoom]
    if not tiles:
        return None
    bounds = [tile_bounds(zoom, x, y) for x, y in tiles]
    return Footprint(
        west=min(b.west for b in bounds),
        south=min(b.south for b in bounds),
        east=max(b.east for b in bounds),
        north=max(b.north for b in bounds),
    )


def footprint_dimensions_km(footprint: Footprint) -> dict[str, float]:
    """Return ``{"width_km", "height_km"}`` for a footprint, for the receipt."""
    deg_to_m = (math.pi / 180.0) * _EARTH_RADIUS_M
    height_km = (footprint.north - footprint.south) * deg_to_m / 1000.0
    cos_lat = max(math.cos(math.radians((footprint.south + footprint.north) / 2.0)), 1e-9)
    width_km = (footprint.east - footprint.west) * deg_to_m * cos_lat / 1000.0
    return {"width_km": round(width_km, 3), "height_km": round(height_km, 3)}


def write_tile_pyramid(
    root: Path,
    kind: str,
    source_array: np.ndarray,
    source_footprint: Footprint,
    zooms: Iterable[int],
    *,
    resample: str = "bilinear",
    extent: Footprint | None = None,
) -> TilePyramidReport:
    """Write every tile of ``kind`` covering ``source_footprint`` at each zoom.

    A tile only partially covered by the source raster is skipped and
    recorded rather than filled with invented data — see
    ``TilePyramidReport.skipped_partial``. Terrain heights are always
    sampled bilinearly before Terrain-RGB encoding; ``resample`` only
    selects the imagery resampling method. ``extent``, when given, is the
    geographic area tiles are written for (candidate tiles come from
    ``extent`` instead of ``source_footprint``); a tile is still skipped
    unless the source raster fully covers it — this only ever narrows what
    gets written, never invents coverage the raster doesn't have. Use it to
    make two kinds fetched over the same bbox share one written tile grid
    instead of each kind's own tile size deciding its own footprint.
    """
    if kind not in ("imagery", "terrain"):
        raise ValueError("kind must be 'imagery' or 'terrain'")
    if resample not in ("bilinear", "lanczos"):
        raise ValueError("resample must be 'bilinear' or 'lanczos'")
    root = Path(root)
    written: list[tuple[int, int, int]] = []
    skipped: list[tuple[int, int, int]] = []
    target_footprint = extent if extent is not None else source_footprint
    for z in sorted(set(zooms)):
        for x, y in tiles_covering(target_footprint, z):
            bounds = tile_bounds(z, x, y)
            if not _fully_covered(bounds, source_footprint):
                skipped.append((z, x, y))
                continue
            tile_dir = root / kind / str(z) / str(x)
            tile_dir.mkdir(parents=True, exist_ok=True)
            if kind == "terrain":
                sampled = _sample_tile_bilinear(source_array, source_footprint, z, x, y)
                encoded = encode_terrain_rgb(sampled)
                Image.fromarray(encoded, mode="RGB").save(tile_dir / f"{y}.png", optimize=True)
            elif resample == "bilinear":
                sampled = _sample_tile_bilinear(source_array, source_footprint, z, x, y)
            else:
                lon, lat = _tile_lonlat_grid(z, x, y)
                sampled = _sample_imagery(source_array, source_footprint, lon, lat, resample)
            if kind == "imagery":
                arr = np.clip(sampled, 0, 255).astype(np.uint8)
                Image.fromarray(arr, mode="RGB").save(
                    tile_dir / f"{y}.jpg", quality=92, optimize=True
                )
            written.append((z, x, y))
    return TilePyramidReport(written=sorted(written), skipped_partial=sorted(skipped))


# ---------------------------------------------------------------------------
# Manifest / verification
# ---------------------------------------------------------------------------


def _iter_bundle_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json")


def _sha256_file(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def write_manifest(root: Path, manifest_fields: dict) -> Path:
    """Write ``manifest.json`` with a sha256 of every other file under root.

    The schema is the current one (``/2``) when ``manifest_fields`` carries the
    ``roads`` block, else the legacy ``/1`` layout that has no road layer.
    """
    root = Path(root)
    files = {str(p.relative_to(root).as_posix()): _sha256_file(p) for p in _iter_bundle_files(root)}
    schema = MANIFEST_SCHEMA if "roads" in manifest_fields else MANIFEST_SCHEMAS[0]
    return rewrite_manifest(root, {**manifest_fields, "schema": schema, "files": files})


def rewrite_manifest(root: Path, manifest: dict) -> Path:
    """Write an already-complete manifest dict back, with no re-hashing.

    ``write_manifest`` builds the ``files`` map by hashing every file under
    the bundle; a caller that has changed exactly one file (and knows its new
    digest) would otherwise pay a full re-hash of several thousand terrain and
    imagery tiles to record it. Both paths serialise through here so a
    rewritten manifest is byte-for-byte what a freshly written one would be.
    """
    destination = Path(root) / "manifest.json"
    with destination.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return destination


def bundle_file_digest(path: Path) -> str:
    """The sha256 a manifest's ``files`` map records for one bundle file."""
    return _sha256_file(Path(path))


def verify_bundle(root: Path) -> list[str]:
    """Return a list of problems: missing manifest/file, checksum mismatch, extra file."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    if manifest.get("schema") not in MANIFEST_SCHEMAS:
        problems.append(f"unexpected schema: {manifest.get('schema')!r}")
    files: dict = manifest.get("files", {})
    on_disk = {p.relative_to(root).as_posix() for p in _iter_bundle_files(root)}
    for relpath, expected_sha in sorted(files.items()):
        path = root / relpath
        if not path.is_file():
            problems.append(f"missing file: {relpath}")
            continue
        if _sha256_file(path) != expected_sha:
            problems.append(f"checksum mismatch: {relpath}")
    for relpath in sorted(on_disk - set(files)):
        problems.append(f"extra file not in manifest: {relpath}")
    if manifest.get("schema") == MANIFEST_SCHEMA:
        problems.extend(_roads_layer_problems(root, manifest))
    problems.extend(_declared_zoom_problems(root, manifest))
    problems.extend(_imagery_extent_problems(manifest))
    problems.extend(_duplicate_route_feature_problems(root))
    return problems


def _roads_layer_problems(root: Path, manifest: dict) -> list[str]:
    """Flag a /2 bundle that lacks its road layer or the manifest block describing it."""
    problems: list[str] = []
    if not (root / "roads.geojson").is_file():
        problems.append("missing file: roads.geojson (required by schema /2)")
    if not isinstance(manifest.get("roads"), dict):
        problems.append("manifest has no roads block (required by schema /2)")
    return problems


def _imagery_extent_problems(manifest: dict) -> list[str]:
    """Flag a manifest whose imagery extent reaches past its terrain extent."""
    imagery = manifest.get("imagery")
    terrain = manifest.get("terrain")
    if not isinstance(imagery, dict) or not isinstance(terrain, dict):
        return []
    imagery_extent, terrain_extent = imagery.get("extent"), terrain.get("extent")
    if not isinstance(imagery_extent, dict) or not isinstance(terrain_extent, dict):
        return []
    eps = 1e-6
    if (
        imagery_extent["west"] < terrain_extent["west"] - eps
        or imagery_extent["east"] > terrain_extent["east"] + eps
        or imagery_extent["south"] < terrain_extent["south"] - eps
        or imagery_extent["north"] > terrain_extent["north"] + eps
    ):
        return ["imagery extent is not a subset of the terrain extent"]
    return []


def _duplicate_route_feature_problems(root: Path) -> list[str]:
    """Flag a route.geojson holding two byte-identical feature geometries."""
    route_path = root / "route.geojson"
    if not route_path.is_file():
        return []
    try:
        route = json.loads(route_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    geometries = [
        json.dumps(feature.get("geometry"), sort_keys=True) for feature in route.get("features", [])
    ]
    if len(geometries) != len(set(geometries)):
        return ["route.geojson has duplicate feature geometries"]
    return []


def _declared_zoom_problems(root: Path, manifest: dict) -> list[str]:
    """Flag a manifest that promises a tile zoom the bundle does not hold.

    ``write_tile_pyramid`` skips tiles the source raster only partly covers,
    so a requested zoom can end up with no tiles at all. A manifest that
    still advertises it makes the scene request tiles that 404.
    """
    problems: list[str] = []
    for kind in ("imagery", "terrain"):
        declared = manifest.get(kind)
        if not isinstance(declared, dict):
            continue
        minzoom, maxzoom = declared.get("minzoom"), declared.get("maxzoom")
        if not isinstance(minzoom, int) or not isinstance(maxzoom, int):
            continue
        for zoom in range(minzoom, maxzoom + 1):
            directory = root / kind / str(zoom)
            if not any(directory.glob("*/*")):
                problems.append(f"{kind}: manifest declares zoom {zoom} but holds no such tile")
        problems.extend(_band_zoom_agreement_problems(kind, declared))
    return problems


def _band_zoom_agreement_problems(kind: str, declared: dict) -> list[str]:
    """Flag a band whose ``zooms`` and ``written_zoom_range`` disagree.

    A band carries the zooms it was PLANNED for and the zooms that were
    actually written, and ``satellite`` reads the written range while the
    scene style reads the planned one. Nothing cross-checked them, so a
    hand-patched bundle could carry ``zooms: [9, 11]`` beside
    ``written_zoom_range: 10-11`` and still verify clean -- two surfaces
    quietly disagreeing about the same band.
    """
    problems: list[str] = []
    for band in declared.get("bands") or []:
        if not isinstance(band, dict):
            continue
        written = band.get("written_zoom_range")
        zooms = band.get("zooms")
        if not isinstance(written, dict) or not isinstance(zooms, list) or len(zooms) != 2:
            continue
        low, high = written.get("minzoom"), written.get("maxzoom")
        if not isinstance(low, int) or not isinstance(high, int):
            continue
        if low < zooms[0] or high > zooms[1]:
            problems.append(
                f"{kind}: band {band.get('source')!r} declares zooms {zooms} "
                f"but records writing {low}-{high}"
            )
    return problems


# ---------------------------------------------------------------------------
# Content-addressed bundle directory
# ---------------------------------------------------------------------------


def bundle_dir_for(
    footprint: Footprint,
    source_versions: dict[str, str],
    *,
    terrain_exaggeration: float = 1.0,
    imagery_zooms: Sequence[int] | None = None,
    terrain_zooms: Sequence[int] | None = None,
) -> Path:
    """Return the cache directory for a footprint + source-version combination.

    ``terrain_exaggeration`` is folded into the key so two bundles differing
    only in the exaggeration baked into their terrain tiles don't collide in
    the cache; the default (1.0, "no exaggeration") reproduces every existing
    caller's directory unchanged.

    ``imagery_zooms``/``terrain_zooms`` fold the bundle's SHAPE in beside its
    sources. Without them a bundle built for z13-15 answered a request for
    z2-15 -- ``prepare`` short-circuits its whole fetch on any existing
    directory that verifies clean, and the report it builds from that
    directory states the zooms that were ASKED for, so the receipt claimed a
    wide bundle and the render found nothing at the wide end. Order and
    duplicates are canonicalised, so a caller that lists the same zooms
    differently gets the same bundle rather than a second copy of it.

    Both default to ``None``, which reproduces the pre-shape key exactly:
    bundles already on disk keep their address and are superseded by a
    shape-keyed one rather than orphaned.
    """
    canonical: dict[str, object] = {
        "footprint": {
            "west": round(footprint.west, 6),
            "south": round(footprint.south, 6),
            "east": round(footprint.east, 6),
            "north": round(footprint.north, 6),
        },
        "source_versions": dict(sorted(source_versions.items())),
        "terrain_exaggeration": round(terrain_exaggeration, 3),
    }
    if imagery_zooms is not None:
        canonical["imagery_zooms"] = sorted(set(imagery_zooms))
    if terrain_zooms is not None:
        canonical["terrain_zooms"] = sorted(set(terrain_zooms))
    payload = json.dumps(canonical, sort_keys=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return gms_route_story_cache_dir() / digest
