"""GNIS place names for Route Story labels (peaks, gaps, streams, populated places).

Source: USGS The National Map staged products, ``prd-tnm.s3.amazonaws.com``
(``StagedProducts/GeographicNames/DomesticNames``), the download origin for the same
Geographic Names Information System the legacy ``geonames.usgs.gov`` portal serves —
public domain, no key, no account. Each state's Domestic Names
file is a small (~1 MB) pipe-delimited text file inside a zip; this module fetches only
the states a bounding box actually touches, caches each zip once, and parses it into
:class:`GnisFeature` records filtered to the feature classes Route Story labels.

Nothing here writes ``labels.geojson`` or wires into ``prepare_bundle`` — ``prepare.py``
does that, and owns the ``labels_degraded`` loud-degradation path.
:func:`to_label_geojson` matches the property shape ``prepare.py``'s OSM label writer
already uses (``name``, ``kind``, ``source``) so the two label sources can be merged
without a reshape.
"""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

# Same download origin as the 3DEP/NAIP hosts used in sources.py
# (StagedProducts under The National Map); a literal here rather than an import
# avoids a circular import with sources.py, which this module is not part of.
GNIS_HOST = "prd-tnm.s3.amazonaws.com"

_STAGED_PRODUCTS_PATH = "StagedProducts/GeographicNames/DomesticNames"

# The Route Story label ladder cares about these GNIS feature classes; every other
# class (Church, School, Bridge, Airport, ...) is dropped during parsing.
DEFAULT_CLASSES: frozenset[str] = frozenset(
    {
        "Summit",
        "Gap",
        "Ridge",
        "Stream",
        "Populated Place",
        "Trail",
        "Falls",
        "Lake",
        "Cliff",
        "Valley",
    }
)

# GNIS feature_class -> Route Story label ladder kind. A class with no explicit
# mapping falls back to a lowercased, underscore-joined version of its own name.
_KIND_LADDER: dict[str, str] = {
    "Summit": "summit",
    "Gap": "gap",
    "Stream": "stream",
    "Populated Place": "place",
    "Trail": "trail",
}

# Coarse public-domain state bounding boxes (west, south, east, north), one line per
# state so a bbox query never needs a network round trip to know which states it
# touches. Values are approximate (deliberately coarse) and are
# only used to decide which per-state GNIS file to fetch, never to place a feature.
STATE_BBOXES: dict[str, tuple[float, float, float, float]] = {
    "AL": (-88.473227, 30.223334, -84.88908, 35.008028),
    "AK": (-179.148909, 51.214183, -129.9795, 71.365162),
    "AZ": (-114.818269, 31.332177, -109.045223, 37.00426),
    "AR": (-94.617919, 33.004106, -89.644395, 36.4996),
    "CA": (-124.409591, 32.534156, -114.131211, 42.009518),
    "CO": (-109.060253, 36.992426, -102.041524, 41.003444),
    "CT": (-73.727775, 40.980144, -71.786994, 42.050587),
    "DE": (-75.788658, 38.451013, -75.048939, 39.839007),
    "DC": (-77.119759, 38.791645, -76.909395, 38.995968),
    "FL": (-87.634938, 24.523096, -80.031362, 31.000888),
    "GA": (-85.605165, 30.357851, -80.839729, 35.000659),
    "HI": (-160.2471, 18.910361, -154.806773, 22.235),
    "ID": (-117.243027, 41.988057, -111.043564, 49.001146),
    "IL": (-91.513079, 36.970298, -87.494756, 42.508481),
    "IN": (-88.09776, 37.771742, -84.784579, 41.760592),
    "IA": (-96.639704, 40.375501, -90.140061, 43.501196),
    "KS": (-102.051744, 36.993016, -94.588413, 40.003162),
    "KY": (-89.571509, 36.497129, -81.964971, 39.147458),
    "LA": (-94.043147, 28.928609, -88.817017, 33.019457),
    "ME": (-71.083924, 42.977764, -66.949895, 47.459686),
    "MD": (-79.487651, 37.911717, -75.048939, 39.723043),
    "MA": (-73.508142, 41.237964, -69.928393, 42.886589),
    "MI": (-90.418136, 41.696118, -82.413474, 48.2388),
    "MN": (-97.239209, 43.499356, -89.491739, 49.384358),
    "MS": (-91.655009, 30.173943, -88.097888, 34.996052),
    "MO": (-95.774704, 35.995683, -89.098843, 40.61364),
    "MT": (-116.050003, 44.358221, -104.039138, 49.00139),
    "NE": (-104.053514, 39.999998, -95.30829, 43.001708),
    "NV": (-120.005746, 35.001857, -114.039648, 42.002207),
    "NH": (-72.557247, 42.69699, -70.610621, 45.305476),
    "NJ": (-75.559614, 38.928519, -73.893979, 41.357423),
    "NM": (-109.050173, 31.332301, -103.001964, 37.000232),
    "NY": (-79.762152, 40.496103, -71.856214, 45.01585),
    "NC": (-84.321869, 33.842316, -75.460621, 36.588117),
    "ND": (-104.0489, 45.935054, -96.554507, 49.000574),
    "OH": (-84.820159, 38.403202, -80.518693, 41.977523),
    "OK": (-103.002565, 33.615833, -94.430662, 37.002206),
    "OR": (-124.566244, 41.991794, -116.463504, 46.292035),
    "PA": (-80.519851, 39.7198, -74.689516, 42.26986),
    "RI": (-71.862772, 41.146339, -71.12057, 42.018798),
    "SC": (-83.35391, 32.0346, -78.54203, 35.215402),
    "SD": (-104.057698, 42.479635, -96.436589, 45.94545),
    "TN": (-90.310298, 34.982972, -81.6469, 36.678118),
    "TX": (-106.645646, 25.837377, -93.508292, 36.500704),
    "UT": (-114.052962, 36.997968, -109.041058, 42.001567),
    "VT": (-73.43774, 42.726853, -71.465558, 45.016659),
    "VA": (-83.675395, 36.540738, -75.242266, 39.466012),
    "WA": (-124.848974, 45.543541, -116.916071, 49.002494),
    "WV": (-82.644739, 37.201483, -77.719519, 40.638801),
    "WI": (-92.888114, 42.491983, -86.805415, 47.080621),
    "WY": (-111.056888, 40.994746, -104.05216, 45.005904),
}


class GnisFetchError(ValueError):
    """Raised when an HTTP fetch to the GNIS host fails."""


@dataclass(frozen=True)
class GnisFeature:
    name: str
    feature_class: str
    lat: float
    lon: float
    elevation_m: float | None
    state: str
    feature_id: str


def gnis_state_url(state_abbr: str) -> str:
    """Return the download URL for a state's Domestic Names zip."""
    state = state_abbr.strip().upper()
    return f"https://{GNIS_HOST}/{_STAGED_PRODUCTS_PATH}/DomesticNames_{state}_Text.zip"


def _bbox_overlaps(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> bool:
    aw, asouth, ae, anorth = a
    bw, bsouth, be, bnorth = b
    return aw <= be and ae >= bw and asouth <= bnorth and anorth >= bsouth


def states_for_bbox(bbox: tuple[float, float, float, float]) -> list[str]:
    """Return the sorted state abbreviations whose coarse bounding box overlaps bbox."""
    return sorted(
        state for state, state_bbox in STATE_BBOXES.items() if _bbox_overlaps(bbox, state_bbox)
    )


def _cache_key(headers: dict[str, str]) -> str:
    """Pick a stable cache subdirectory name from the download response headers."""
    etag = headers.get("ETag") or headers.get("etag")
    if etag:
        return etag.strip('"').replace("/", "_")
    last_modified = headers.get("Last-Modified") or headers.get("last-modified")
    if last_modified:
        try:
            return parsedate_to_datetime(last_modified).strftime("%Y%m%d")
        except (TypeError, ValueError):
            pass
    return "unknown-date"


def fetch_gnis_state(session: Any, state_abbr: str, cache_dir: Path) -> Path:
    """Download a state's Domestic Names zip once, returning the cached path thereafter.

    A second call for the same state finds the already-cached zip under
    ``cache_dir/gnis/<ST>/`` and returns it without issuing a request — the same
    fetch-once contract as ``bundle.bundle_dir_for``'s content-addressed cache.
    """
    state = state_abbr.strip().upper()
    filename = f"DomesticNames_{state}_Text.zip"
    state_dir = Path(cache_dir) / "gnis" / state
    existing = sorted(state_dir.glob(f"*/{filename}"))
    if existing:
        return existing[0]

    url = gnis_state_url(state)
    try:
        response = session.get(url, params=None, timeout=30, headers={})
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - re-raised with source context below
        raise GnisFetchError(
            f"GNIS Domestic Names ({state}) fetch failed for {url}: {exc}"
        ) from exc

    headers = dict(getattr(response, "headers", None) or {})
    target_dir = state_dir / _cache_key(headers)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    target_path.write_bytes(response.content)
    return target_path


def parse_gnis(
    zip_path: Path,
    *,
    bbox: tuple[float, float, float, float],
    classes: frozenset[str] = DEFAULT_CLASSES,
) -> list[GnisFeature]:
    """Parse a Domestic Names zip into features of the given classes inside bbox."""
    west, south, east, north = bbox
    features: list[GnisFeature] = []
    with zipfile.ZipFile(zip_path) as archive:
        text_members = [name for name in archive.namelist() if name.lower().endswith(".txt")]
        if not text_members:
            raise ValueError(f"{zip_path}: no .txt member found in GNIS zip")
        with archive.open(text_members[0]) as raw:
            # The staged-products file opens with a UTF-8 byte-order mark; plain
            # "utf-8" would read the first column as "\ufefffeature_id".
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"), delimiter="|")
            for row in reader:
                feature_class = (row.get("feature_class") or "").strip()
                if feature_class not in classes:
                    continue
                try:
                    lat = float(row["prim_lat_dec"])
                    lon = float(row["prim_long_dec"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (west <= lon <= east and south <= lat <= north):
                    continue
                elev_raw = (row.get("elev_in_m") or "").strip()
                elevation_m = float(elev_raw) if elev_raw else None
                features.append(
                    GnisFeature(
                        name=(row.get("feature_name") or "").strip(),
                        feature_class=feature_class,
                        lat=lat,
                        lon=lon,
                        elevation_m=elevation_m,
                        state=(row.get("state_alpha") or row.get("state_name") or "").strip(),
                        feature_id=(row.get("feature_id") or "").strip(),
                    )
                )
    return features


def gnis_features_for_bbox(
    session: Any, bbox: tuple[float, float, float, float], cache_dir: Path
) -> list[GnisFeature]:
    """Fetch, parse and combine GNIS features for every state bbox touches."""
    # A state file also lists features that straddle its neighbours, so the same
    # feature arrives once per state file; an id-less row keys on its identity.
    by_id: dict[tuple[str, ...], GnisFeature] = {}
    for state in states_for_bbox(bbox):
        zip_path = fetch_gnis_state(session, state, cache_dir)
        for feature in parse_gnis(zip_path, bbox=bbox):
            key = (
                (feature.feature_id,)
                if feature.feature_id
                else (feature.name, feature.feature_class, str(feature.lat), str(feature.lon))
            )
            by_id.setdefault(key, feature)
    return sorted(by_id.values(), key=lambda feature: (feature.feature_class, feature.name))


def to_label_geojson(features: list[GnisFeature]) -> dict[str, Any]:
    """Render GNIS features as a label GeoJSON matching ``prepare.py``'s OSM shape."""
    geo_features = []
    for feature in features:
        kind = _KIND_LADDER.get(
            feature.feature_class, feature.feature_class.lower().replace(" ", "_")
        )
        geo_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [feature.lon, feature.lat]},
                "properties": {
                    "name": feature.name,
                    "kind": kind,
                    "elevation_m": feature.elevation_m,
                    "source": "gnis",
                    "feature_id": feature.feature_id,
                },
            }
        )
    return {"type": "FeatureCollection", "features": geo_features}
