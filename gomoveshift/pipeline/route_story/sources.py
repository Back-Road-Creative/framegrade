"""Fetchers for free public map source data, returning in-memory georeferenced arrays.

Each fetcher takes an injected HTTP session (anything shaped like ``requests.Session``)
and a ``bbox`` of ``(west, south, east, north)`` in EPSG:4326 degrees. Nothing here writes
tiles or manifests — that is ``bundle.py``'s job. Every returned raster carries a
:class:`SourceProvenance` recording where the bytes came from and under what licence.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from datetime import date as _date
from typing import Any
from urllib.parse import urlsplit

import numpy as np
import tifffile
from PIL import Image

from ..overpass_source import OVERPASS_ENDPOINTS
from .scene.vendor import MAPLIBRE_URL

_DEM_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
)
_NAIP_URL = (
    "https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer/exportImage"
)
_GIBS_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
_COPERNICUS_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
_OSM_USER_AGENT = "gomoveshift-route-story/1 (+https://gomoveshift.com)"

_OSM_FILTERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("natural", ("peak", "saddle", "ridge")),
    ("mountain_pass", ("yes",)),
    ("place", ("city", "town", "village")),
    ("waterway", ("river", "stream")),
)


DEM_3DEP_PROVENANCE_NAME = "USGS 3DEP DEM"
# Bumped whenever ``_grid_plan`` changes what ground an ArcGIS export lands on;
# ``prepare`` folds it into the bundle cache key. "square-degree-1": tiles are
# square in degrees, so the server returns exactly the extent asked for.
ARCGIS_EXPORT_GRID_VERSION = "square-degree-1"
NAIP_PROVENANCE_NAME = "NAIP imagery"
COPERNICUS_PROVENANCE_NAME = "Copernicus GLO-30 DEM"
OSM_PROVENANCE_NAME = "OpenStreetMap Overpass"


class SourceCoverageError(ValueError):
    """Raised when fetched source data does not adequately cover the requested bbox."""


# Every host this module contacts, as the network-allowlist doc must name them.
# Derived from the URLs above rather than retyped, so a new source URL cannot be
# added without the allowlist doc-sync test noticing.
SOURCE_HOSTS: frozenset[str] = frozenset(
    urlsplit(url).hostname or ""
    for url in (_DEM_URL, _NAIP_URL, _GIBS_URL, _COPERNICUS_BUCKET, MAPLIBRE_URL)
) | {"prd-tnm.s3.amazonaws.com"}  # gnis.py's GNIS_HOST


class SourceFetchError(ValueError):
    """Raised when an HTTP fetch to a public map source fails."""


@dataclass(frozen=True)
class SourceProvenance:
    name: str
    url: str
    licence: str
    attribution: str
    acquired: str  # ISO date of the data (not fetch time) when known, else "unknown"


@dataclass(frozen=True)
class GeoRaster:
    array: np.ndarray
    west: float
    south: float
    east: float
    north: float
    provenance: SourceProvenance


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    west, south, east, north = bbox
    if not (west < east) or not (south < north):
        raise ValueError("bbox must have positive area: west<east and south<north")


def _request(
    session: Any,
    name: str,
    url: str,
    params: dict[str, Any] | None,
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 30,
) -> bytes:
    try:
        response = session.get(url, params=params, timeout=timeout, headers=headers or {})
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - re-raised with source context below
        raise SourceFetchError(f"{name} fetch failed for {url}: {exc}") from exc
    return response.content


def _pixel_size_degrees(
    bbox: tuple[float, float, float, float], metres_per_px: float
) -> tuple[float, float]:
    _, south, _, north = bbox
    mean_lat = (south + north) / 2.0
    lat_deg_per_m = 1.0 / 111320.0
    lon_deg_per_m = 1.0 / (111320.0 * max(math.cos(math.radians(mean_lat)), 1e-6))
    return lon_deg_per_m * metres_per_px, lat_deg_per_m * metres_per_px


def _grid_plan(
    bbox: tuple[float, float, float, float], *, metres_per_px: float, max_px: int
) -> tuple[list[list[tuple[float, float, float, float, int, int]]], int, int]:
    """Split bbox into a row-major grid of (west, south, east, north, w, h) tiles.

    Every tile has the same shape in pixels as in degrees. ArcGIS
    ``exportImage`` does not stretch the data to the requested ``size``: it
    GROWS the returned extent until it matches the pixel aspect (observed
    on 3DEP: a 0.108 degree latitude span asked at 1200x1200 came back 0.132
    degrees wide). The exports use ``imageSR=4326``, so the raster is plate
    carree and the only request the server returns over exactly the extent the
    caller labels it with is one whose pixels are square in DEGREES. One step
    drives both axes -- the finer of the two, so neither axis is coarser than
    ``metres_per_px`` -- and each tile's edges fall on whole-pixel boundaries
    of that one grid, so a tile's degree span is exactly proportional to its
    pixel count.
    """
    west, south, east, north = bbox
    step = min(_pixel_size_degrees(bbox, metres_per_px))
    total_w = max(1, math.ceil((east - west) / step))
    total_h = max(1, math.ceil((north - south) / step))
    ncols = max(1, math.ceil(total_w / max_px))
    nrows = max(1, math.ceil(total_h / max_px))
    widths = [total_w // ncols + (1 if c < total_w % ncols else 0) for c in range(ncols)]
    heights = [total_h // nrows + (1 if r < total_h % nrows else 0) for r in range(nrows)]
    lon_edges = [west + (east - west) * sum(widths[:c]) / total_w for c in range(ncols + 1)]
    lat_edges = [north - (north - south) * sum(heights[:r]) / total_h for r in range(nrows + 1)]
    tiles: list[list[tuple[float, float, float, float, int, int]]] = []
    for r in range(nrows):
        tiles.append(
            [
                (
                    lon_edges[c],
                    lat_edges[r + 1],
                    lon_edges[c + 1],
                    lat_edges[r],
                    widths[c],
                    heights[r],
                )
                for c in range(ncols)
            ]
        )
    return tiles, total_w, total_h


def _decode_tiff_f32(content: bytes) -> tuple[np.ndarray, float | None]:
    with tifffile.TiffFile(io.BytesIO(content)) as tif:
        array = np.asarray(tif.asarray(), dtype=np.float32)
        nodata = None
        tags = tif.pages[0].tags
        if "GDAL_NODATA" in tags:
            try:
                nodata = float(str(tags["GDAL_NODATA"].value).strip())
            except ValueError:
                nodata = None
    return array, nodata


def _check_nodata(array: np.ndarray, nodata: float | None, name: str) -> None:
    mask = array <= -1e30
    if nodata is not None:
        mask = mask | np.isclose(array, nodata)
    fraction = float(np.count_nonzero(mask)) / array.size
    if fraction > 0:
        raise SourceCoverageError(f"{name}: {fraction:.1%} of pixels are nodata")


# 3DEP's ImageServer advertises maxImageWidth/Height 8000, but an F32 export
# of 2734x3075 answers HTTP 500 after ~16 s while 1367x1538 returns 200 in
# ~3 s (observed). Tiles stay at or under this per side.
DEM_3DEP_MAX_TILE_PX = 2048

# NAIP's ImageServer advertises maxImageWidth/Height 4000 (larger answers a
# 200 carrying a JSON "exceeds the size limit" error), and a 4000x4000 export
# took 56 s -- past the 30 s client timeout -- while 2048x2048 returned in
# ~3 s (observed).
NAIP_MAX_TILE_PX = 2048


def fetch_3dep_dem(
    session: Any,
    bbox: tuple[float, float, float, float],
    *,
    metres_per_px: float = 10.0,
    max_px: int = DEM_3DEP_MAX_TILE_PX,
) -> GeoRaster:
    """Fetch USGS 3DEP elevation covering bbox at ``metres_per_px``, stitching a grid if needed.

    Defaults to ~10 m/px, 3DEP's native resolution for the focal terrain fetch;
    a caller pulling a much wider regional footprint passes a coarser
    ``metres_per_px`` so the request grid stays sane, not the full-resolution
    pixel count a wide bbox would otherwise imply.
    """
    _validate_bbox(bbox)
    west, south, east, north = bbox
    tiles, _, _ = _grid_plan(bbox, metres_per_px=metres_per_px, max_px=max_px)
    rows = []
    for row in tiles:
        cols = []
        for tile_west, tile_south, tile_east, tile_north, tw, th in row:
            params = {
                "bbox": f"{tile_west},{tile_south},{tile_east},{tile_north}",
                "bboxSR": 4326,
                "imageSR": 4326,
                "size": f"{tw},{th}",
                "format": "tiff",
                "pixelType": "F32",
                "interpolation": "RSP_BilinearInterpolation",
                "f": "image",
            }
            content = _request(session, "USGS 3DEP DEM", _DEM_URL, params, headers={})
            array, nodata = _decode_tiff_f32(content)
            _check_nodata(array, nodata, "USGS 3DEP DEM")
            cols.append(array)
        rows.append(np.concatenate(cols, axis=1))
    array = np.concatenate(rows, axis=0).astype(np.float32)
    provenance = SourceProvenance(
        name=DEM_3DEP_PROVENANCE_NAME,
        url=_DEM_URL,
        licence="Public domain (USGS)",
        attribution="USGS 3D Elevation Program",
        acquired="unknown",
    )
    return GeoRaster(
        array=array, west=west, south=south, east=east, north=north, provenance=provenance
    )


def _decode_image_rgb(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _decode_image_rgba(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _blank_fraction(rgb: np.ndarray) -> float:
    white = np.all(rgb >= 250, axis=-1)
    black = np.all(rgb <= 5, axis=-1)
    return float(np.count_nonzero(white | black)) / (rgb.shape[0] * rgb.shape[1])


def fetch_naip_imagery(
    session: Any,
    bbox: tuple[float, float, float, float],
    *,
    metres_per_px: float = 1.0,
    max_px: int = NAIP_MAX_TILE_PX,
) -> GeoRaster:
    """Fetch USGS/USDA NAIP orthoimagery covering bbox, stitching a grid if needed."""
    _validate_bbox(bbox)
    west, south, east, north = bbox
    tiles, _, _ = _grid_plan(bbox, metres_per_px=metres_per_px, max_px=max_px)
    rows = []
    for row in tiles:
        cols = []
        for tile_west, tile_south, tile_east, tile_north, tw, th in row:
            params = {
                "bbox": f"{tile_west},{tile_south},{tile_east},{tile_north}",
                "bboxSR": 4326,
                "imageSR": 4326,
                "size": f"{tw},{th}",
                "format": "jpgpng",
                "f": "image",
            }
            content = _request(session, "NAIP imagery", _NAIP_URL, params, headers={})
            cols.append(_decode_image_rgb(content))
        rows.append(np.concatenate(cols, axis=1))
    array = np.concatenate(rows, axis=0)
    if _blank_fraction(array) > 0.02:
        raise SourceCoverageError("NAIP imagery: >2% of the requested area is blank")
    provenance = SourceProvenance(
        name=NAIP_PROVENANCE_NAME,
        url=_NAIP_URL,
        licence="Public domain (USDA NAIP via USGS)",
        attribution="USDA NAIP via USGS",
        acquired="unknown",
    )
    return GeoRaster(
        array=array, west=west, south=south, east=east, north=north, provenance=provenance
    )


def _fetch_gibs(
    session: Any,
    bbox: tuple[float, float, float, float],
    layer: str,
    date: _date | None,
    width: int,
    height: int,
) -> GeoRaster:
    _validate_bbox(bbox)
    west, south, east, north = bbox
    params: dict[str, Any] = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "CRS": "EPSG:4326",
        "BBOX": f"{south},{west},{north},{east}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/jpeg",
    }
    if date is not None:
        params["TIME"] = date.isoformat()
    content = _request(session, f"NASA GIBS ({layer})", _GIBS_URL, params, headers={})
    array = _decode_image_rgb(content)
    acquired = date.isoformat() if date is not None else "unknown"
    provenance = SourceProvenance(
        name=f"NASA GIBS {layer}",
        url=_GIBS_URL,
        licence="NASA open data (attribution requested)",
        attribution="NASA EOSDIS GIBS / Worldview",
        acquired=acquired,
    )
    return GeoRaster(
        array=array, west=west, south=south, east=east, north=north, provenance=provenance
    )


def fetch_gibs_truecolor(
    session: Any,
    bbox: tuple[float, float, float, float],
    date: _date,
    *,
    layer: str = "VIIRS_SNPP_CorrectedReflectance_TrueColor",
    width: int,
    height: int,
) -> GeoRaster:
    """Fetch a NASA GIBS true-colour daily satellite scene for bbox on date."""
    return _fetch_gibs(session, bbox, layer, date, width, height)


def fetch_gibs_blue_marble(
    session: Any,
    bbox: tuple[float, float, float, float],
    *,
    layer: str = "BlueMarble_ShadedRelief_Bathymetry",
    width: int,
    height: int,
) -> GeoRaster:
    """Fetch the dateless NASA GIBS Blue Marble fallback backdrop for bbox."""
    return _fetch_gibs(session, bbox, layer, None, width, height)


def fetch_gibs_reference_features(
    session: Any,
    bbox: tuple[float, float, float, float],
    *,
    layer: str = "Reference_Features_15m",
    width: int,
    height: int,
) -> GeoRaster:
    """Fetch the NASA GIBS transparent coastline/national/state reference overlay for bbox.

    Served as a transparent PNG from the same GIBS host `_fetch_gibs` already
    uses: only the request `FORMAT`/`TRANSPARENT` differ. Returned array is RGBA — callers
    use `borders.overlay_borders` to blend its alpha channel as coloured lines onto a
    backdrop; the overlay's own RGB is not meaningful and is ignored there.
    """
    _validate_bbox(bbox)
    west, south, east, north = bbox
    params: dict[str, Any] = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "CRS": "EPSG:4326",
        "BBOX": f"{south},{west},{north},{east}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
    }
    content = _request(session, f"NASA GIBS ({layer})", _GIBS_URL, params, headers={})
    array = _decode_image_rgba(content)
    provenance = SourceProvenance(
        name=f"NASA GIBS {layer}",
        url=_GIBS_URL,
        # The line data underlying this layer may be third-party to NASA; do not
        # invent licence terms here.
        licence="unverified",
        attribution="NASA EOSDIS GIBS / Worldview",
        acquired="unknown",
    )
    return GeoRaster(
        array=array, west=west, south=south, east=east, north=north, provenance=provenance
    )


def cloud_fraction(rgb: np.ndarray) -> float:
    """Deterministic heuristic: fraction of near-white, low-saturation cloud pixels."""
    rgb = np.asarray(rgb)
    channel_min = rgb.min(axis=-1).astype(np.int16)
    channel_max = rgb.max(axis=-1).astype(np.int16)
    cloud = (channel_min >= 200) & ((channel_max - channel_min) <= 25)
    return float(np.count_nonzero(cloud)) / (rgb.shape[0] * rgb.shape[1])


def choose_opening_scene(
    daily: GeoRaster, blue_marble: GeoRaster, *, max_cloud: float = 0.35
) -> tuple[GeoRaster, dict[str, Any]]:
    """Pick the daily scene unless it is too cloudy, falling back to Blue Marble."""
    fraction = cloud_fraction(daily.array)
    if fraction <= max_cloud:
        chosen, label = daily, "daily"
    else:
        chosen, label = blue_marble, "blue_marble"
    decision = {"chosen": label, "cloud_fraction": fraction, "threshold": max_cloud}
    return chosen, decision


def gibs_scene_for(
    session: Any,
    bbox: tuple[float, float, float, float],
    trip_date: _date,
    *,
    width: int,
    height: int,
    max_cloud: float = 0.35,
) -> tuple[GeoRaster, dict[str, Any]]:
    """Fetch the trip-date GIBS scene and Blue Marble, then pick one, in one call.

    Wraps the two-fetch-plus-decide sequence (`fetch_gibs_truecolor`,
    `fetch_gibs_blue_marble`, `choose_opening_scene`) that a per-zoom-band
    imagery planner needs as a single unit: "that day from space" only when
    it isn't too cloudy over the requested footprint.
    """
    daily = fetch_gibs_truecolor(session, bbox, trip_date, width=width, height=height)
    blue_marble = fetch_gibs_blue_marble(session, bbox, width=width, height=height)
    return choose_opening_scene(daily, blue_marble, max_cloud=max_cloud)


# The server-side budget declared in the Overpass QL query itself (how long the
# server may spend before giving up) and the client-side HTTP timeout passed to
# `_request` (how long we wait for a response) are deliberately different:
# the client timeout must exceed the server budget so a slow-but-alive server
# reports its own giveup via the Overpass response, rather than requests
# severing the connection first and masking a real answer as a client timeout.
OSM_SERVER_BUDGET_S = 60
OSM_QUERY_TIMEOUT_S = 90


def _osm_query(bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    bbox_clause = f"({south},{west},{north},{east})"
    clauses = []
    for key, values in _OSM_FILTERS:
        pattern = "|".join(values)
        clauses.append(f'  node["{key}"~"^({pattern})$"]["name"]{bbox_clause};')
        clauses.append(f'  way["{key}"~"^({pattern})$"]["name"]{bbox_clause};')
    body = "\n".join(clauses)
    return f"[out:json][timeout:{OSM_SERVER_BUDGET_S}];\n(\n{body}\n);\nout center;"


def _osm_kind(tags: dict[str, str]) -> str | None:
    for key, values in _OSM_FILTERS:
        if tags.get(key) in values:
            return "mountain_pass" if key == "mountain_pass" else tags[key]
    return None


def fetch_osm_labels(
    session: Any,
    bbox: tuple[float, float, float, float],
    *,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Fetch named peaks/passes/places/waterways for bbox as a GeoJSON FeatureCollection.

    With no ``endpoint`` the service's own mirror list is tried in order: the
    canonical Overpass host can refuse a client's IPv4 on 443, so a single
    hard-coded host would make every label fetch fail on such a network. An explicit ``endpoint`` is used alone.
    """
    _validate_bbox(bbox)
    query = _osm_query(bbox)
    headers = {"User-Agent": _OSM_USER_AGENT}
    endpoints = (endpoint,) if endpoint else OVERPASS_ENDPOINTS
    failures: list[str] = []
    content = None
    for candidate in endpoints:
        try:
            content = _request(
                session,
                "OpenStreetMap Overpass",
                candidate,
                {"data": query},
                headers=headers,
                timeout=OSM_QUERY_TIMEOUT_S,
            )
            break
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(f"{candidate}: {exc}")
    if content is None:
        raise SourceFetchError(
            "OpenStreetMap Overpass fetch failed at every endpoint: " + "; ".join(failures)
        )
    payload = json.loads(content)
    features = []
    for element in payload.get("elements", []):
        tags = element.get("tags") or {}
        name = tags.get("name")
        kind = _osm_kind(tags)
        if not name or kind is None:
            continue
        if "lat" in element and "lon" in element:
            lon, lat = element["lon"], element["lat"]
        else:
            center = element.get("center") or {}
            if "lat" not in center or "lon" not in center:
                continue
            lon, lat = center["lon"], center["lat"]
        properties: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "source": "osm",
            "osm_id": element.get("id"),
        }
        if "ele" in tags:
            properties["ele"] = tags["ele"]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": properties,
            }
        )
    features.sort(
        key=lambda f: (f["properties"]["kind"], f["properties"]["name"], f["properties"]["osm_id"])
    )
    provenance = SourceProvenance(
        name=OSM_PROVENANCE_NAME,
        url=endpoint,
        licence="ODbL 1.0",
        attribution="© OpenStreetMap contributors",
        acquired="unknown",
    )
    return {"type": "FeatureCollection", "features": features, "provenance": provenance}


def _copernicus_cell_url(cell_south: int, cell_west: int) -> str:
    ns = "N" if cell_south >= 0 else "S"
    ew = "E" if cell_west >= 0 else "W"
    lat_code = f"{ns}{abs(cell_south):02d}_00"
    lon_code = f"{ew}{abs(cell_west):03d}_00"
    name = f"Copernicus_DSM_COG_10_{lat_code}_{lon_code}_DEM"
    return f"{_COPERNICUS_BUCKET}/{name}/{name}.tif"


def fetch_copernicus_dem(session: Any, bbox: tuple[float, float, float, float]) -> GeoRaster:
    """Fetch worldwide Copernicus GLO-30 DEM cells covering bbox, cropped and stitched."""
    _validate_bbox(bbox)
    west, south, east, north = bbox
    lat_cells = sorted(range(math.floor(south), math.ceil(north)), reverse=True)
    lon_cells = sorted(range(math.floor(west), math.ceil(east)))
    urls: list[str] = []
    row_arrays = []
    cell_shape: tuple[int, int] | None = None
    for cell_south in lat_cells:
        col_arrays = []
        for cell_west in lon_cells:
            url = _copernicus_cell_url(cell_south, cell_west)
            urls.append(url)
            try:
                response = session.get(url, params=None, timeout=30, headers={})
                if getattr(response, "status_code", 200) == 404:
                    raise SourceCoverageError(f"Copernicus DEM: no coverage at cell {url}")
                response.raise_for_status()
            except SourceCoverageError:
                raise
            except Exception as exc:  # noqa: BLE001 - re-raised with source context below
                raise SourceFetchError(f"Copernicus DEM fetch failed for {url}: {exc}") from exc
            array, _ = _decode_tiff_f32(response.content)
            if cell_shape is None:
                cell_shape = array.shape
            col_arrays.append(array)
        row_arrays.append(np.concatenate(col_arrays, axis=1))
    mega = np.concatenate(row_arrays, axis=0).astype(np.float32)

    if cell_shape is None:  # pragma: no cover - unreachable for a validated bbox
        raise SourceFetchError("Copernicus DEM: bbox produced no coverage cells")
    cell_h, cell_w = cell_shape
    mega_west = lon_cells[0]
    mega_north = lat_cells[0] + 1
    deg_per_row = 1.0 / cell_h
    deg_per_col = 1.0 / cell_w
    row_top = round((mega_north - north) / deg_per_row)
    row_bottom = round((mega_north - south) / deg_per_row)
    col_left = round((west - mega_west) / deg_per_col)
    col_right = round((east - mega_west) / deg_per_col)
    array = mega[row_top:row_bottom, col_left:col_right]

    provenance = SourceProvenance(
        name=COPERNICUS_PROVENANCE_NAME,
        url=", ".join(sorted(set(urls))),
        licence="Copernicus DEM licence (free, attribution required)",
        attribution=(
            "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus "
            "Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union "
            "and ESA; all rights reserved"
        ),
        acquired="unknown",
    )
    return GeoRaster(
        array=array, west=west, south=south, east=east, north=north, provenance=provenance
    )
