"""Conservative retained-route containment using pinned official polygons:
Census TIGER 2026 for US counties and states, Tailte Eireann's 2026 statutory
county map for Ireland.

The proof bounds a vehicle's possible travel between valid fixes. It depends
on explicit speed and horizontal-error assumptions; footage itself is not a
measurement of its administrative boundary. Missing data always refuses.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import struct
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from shapely import force_2d
from shapely.geometry import MultiPolygon, Point, Polygon, shape
from shapely.ops import transform, unary_union

from .geo_constants import US_STATE_ABBREVS
from .route_claims import retained_windows_key

logger = logging.getLogger(__name__)
_BASE = "https://www2.census.gov/geo/tiger/TIGER2026"
_SOURCES = {
    "county": (
        "tl_2026_us_county.zip",
        "b527c9ba7aeb8f17ef118f026eae5e5a938aa1d00b873bc32342d0dce59321c0",  # pragma: allowlist secret (public Census SHA-256)
    ),
    "state": (
        "tl_2026_us_state.zip",
        "2e17ea40dff3e00163eec87a26efa0b0ff7e8fe364a191bf1a7754dcb49ea326",  # pragma: allowlist secret (public Census SHA-256)
    ),
}
#: A claim level proven on another level's archive: two counties are proven
#: inside the union of their county polygons.
_SOURCE_LEVEL = {"counties": "county"}
MAX_SPEED_M_S = 70.0  # 252 km/h; above any legal road speed, still an assumption.
GPS_ERROR_M = 100.0  # No DOP/fix-type in source GPX; fail near a boundary.
_MAX_ARCHIVE_BYTES = 100_000_000  # Official county ZIP is ~83 MB; reject excess.
#: Tailte Eireann "Counties - National Statutory Boundaries - Ungeneralised
#: 2026" (CC BY 4.0, (c) Tailte Eireann), GeoJSON in Irish Transverse Mercator.
_IE_SOURCE = (
    "counties-national-statutory-boundaries-ungeneralised-2026.geojson",
    "8b422a75fcac8e21f1eef2fe2480b925e7bed06e9aeed3e5daa73cfec45df881",  # pragma: allowlist secret (public Tailte SHA-256)
)
_IE_URL = (
    "https://data-osi.opendata.arcgis.com/api/download/v1/items/"
    "a2dd7924915e4c74ad6a417e33c394eb/geojson?layers=1"
)
#: The ArcGIS download answers with two redirects (replica cache, then CDN).
_IE_HOSTS = frozenset(
    {
        "data-osi.opendata.arcgis.com",
        "services-eu1.arcgis.com",
        "stg-arcgisazurecdataprodeu1.az.arcgis.com",
    }
)
_MAX_IE_BYTES = 150_000_000  # Official GeoJSON is ~92 MB; reject excess.
#: A 2026 map of an earlier recording: a boundary redrawn since could have moved
#: inward, so every allowance gains this inward margin.
IE_MARGIN_M = 250.0
_IE_BASIS = "pinned_tailte_polygon_bounded_speed_error_and_margin"
_IE_COUNTRIES = {"ireland", "ie", "éire", "eire"}


class BoundaryArchiveError(RuntimeError):
    """The pinned geometry needed for a conservative claim is unavailable."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A Census URL must not send the request to an unapproved host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class _HostRedirect(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only over https to an exact approved host."""

    max_redirections = 3

    def __init__(self, hosts):
        super().__init__()
        self.hosts = frozenset(hosts)

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        url = urllib.parse.urlsplit(newurl)
        if url.scheme != "https" or url.hostname not in self.hosts:
            return None
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def is_irish(claim: dict) -> bool:
    """A stateless county claim in Ireland (proven on the Tailte map)."""
    return (
        claim.get("level") in ("county", "counties")
        and not claim.get("state")
        and str(claim.get("country") or "").casefold() in _IE_COUNTRIES
    )


def vintage_covers(claim: dict, recorded_on) -> bool:
    """Whether a pinned map may prove a recording of this date: TIGER 2026 only
    a 2026 recording; the Tailte 2026 map an earlier one too, under its margin."""
    year = str(recorded_on or "")[:4]
    if is_irish(claim):
        return year.isdigit() and int(year) <= 2026
    return year == "2026"


def pinned_boundary_proof(proof: dict, claim: dict) -> bool:
    """Refuse a stale or weaker persisted proof before it overrides GPS gaps."""
    if not isinstance(proof, dict):
        return False
    try:
        speed = float(proof.get("max_speed_m_s"))
        error = float(proof.get("gps_error_m"))
        gap = float(proof.get("max_gap_s"))
    except (TypeError, ValueError):
        return False
    bounded = (
        proof.get("verified") is True
        and proof.get("region") == claim.get("name")
        and bool(proof.get("windows_key"))
        and proof.get("windows_key") == claim.get("windows_key")
        and math.isfinite(speed)
        and speed >= MAX_SPEED_M_S
        and math.isfinite(error)
        and error >= GPS_ERROR_M
        and math.isfinite(gap)
        and gap >= 0
    )
    if is_irish(claim):
        try:
            margin = float(proof.get("margin_m"))
        except (TypeError, ValueError):
            return False
        return (
            bounded
            and proof.get("basis") == _IE_BASIS
            and proof.get("source") == f"TAILTE2026/COUNTIES/{_IE_SOURCE[0]}"
            and proof.get("sha256") == _IE_SOURCE[1]
            and proof.get("country") == "IE"
            and vintage_covers(claim, proof.get("recording_date"))
            and math.isfinite(margin)
            and margin >= IE_MARGIN_M
        )
    level = _SOURCE_LEVEL.get(claim.get("level"), claim.get("level"))
    if level not in _SOURCES:
        return False
    filename, digest = _SOURCES[level]
    return (
        bounded
        and proof.get("basis") == "pinned_tiger_polygon_bounded_speed_and_error"
        and proof.get("source") == f"TIGER2026/{level.upper()}/{filename}"
        and proof.get("sha256") == digest
        and str(proof.get("recording_date", "")).startswith("2026-")
        and bool(proof.get("statefp"))
        and proof.get("statefp") == _state_fips(claim.get("state") or claim.get("name"))
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dbf_rows(data: bytes):
    if len(data) < 33:
        raise BoundaryArchiveError("DBF header is incomplete")
    count, header_len, record_len = struct.unpack_from("<IHH", data, 4)
    if header_len > len(data) or record_len < 2 or header_len < 33:
        raise BoundaryArchiveError("DBF header has invalid lengths")
    fields = []
    for pos in range(32, header_len - 1, 32):
        item = data[pos : pos + 32]
        if len(item) < 32:
            raise BoundaryArchiveError("DBF field is truncated")
        name = item[:11].split(b"\0", 1)[0].decode("ascii", errors="strict")
        fields.append((name, item[16]))
    if not {"NAME", "STATEFP"}.issubset({name for name, _ in fields}):
        raise BoundaryArchiveError("DBF has no NAME or STATEFP field")
    for index in range(count):
        start = header_len + index * record_len
        raw = data[start : start + record_len]
        if len(raw) != record_len:
            raise BoundaryArchiveError("DBF record is truncated")
        if raw[0] == 42:  # deleted DBF row
            continue
        values = {}
        offset = 1
        for name, size in fields:
            values[name] = raw[offset : offset + size].decode("latin-1").strip()
            offset += size
        yield index, values


def _ring_area(points) -> float:
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:]))


def _polygon_from_shp(shp: bytes, shx: bytes, row_index: int):
    index_pos = 100 + 8 * row_index
    if len(shx) < index_pos + 8:
        raise BoundaryArchiveError("SHP index is truncated")
    offset_words, length_words = struct.unpack_from(">II", shx, index_pos)
    pos = offset_words * 2
    if len(shp) < pos + 8 + length_words * 2:
        raise BoundaryArchiveError("SHP polygon record is truncated")
    record = memoryview(shp)[pos + 8 : pos + 8 + length_words * 2]
    if struct.unpack_from("<I", record, 0)[0] != 5:
        raise BoundaryArchiveError("SHP record is not an ordinary polygon")
    parts_n, points_n = struct.unpack_from("<II", record, 36)
    if not parts_n or not points_n or len(record) < 44 + 4 * parts_n + 16 * points_n:
        raise BoundaryArchiveError("SHP polygon has invalid ring lengths")
    starts = list(struct.unpack_from(f"<{parts_n}I", record, 44)) + [points_n]
    points_at = 44 + 4 * parts_n
    rings = []
    for a, b in zip(starts, starts[1:]):
        if not 0 <= a < b <= points_n:
            raise BoundaryArchiveError("SHP polygon ring offset is invalid")
        ring = [struct.unpack_from("<dd", record, points_at + 16 * i) for i in range(a, b)]
        if len(ring) < 4 or ring[0] != ring[-1]:
            raise BoundaryArchiveError("SHP polygon ring is not closed")
        rings.append(ring)
    outers = [Polygon(r) for r in rings if _ring_area(r) < 0]
    holes = [r for r in rings if _ring_area(r) > 0]
    if not outers:
        raise BoundaryArchiveError("SHP polygon has no clockwise exterior")
    grouped = [[] for _ in outers]
    for hole in holes:
        containers = [i for i, outer in enumerate(outers) if outer.covers(Point(hole[0]))]
        if not containers:
            raise BoundaryArchiveError("SHP polygon hole has no exterior")
        smallest = min(containers, key=lambda i: outers[i].area)
        grouped[smallest].append(hole)
    polygons = [Polygon(list(outer.exterior.coords), grouped[i]) for i, outer in enumerate(outers)]
    result = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
    if not result.is_valid:
        raise BoundaryArchiveError("SHP polygon is invalid")
    return result


def _region_name(name: str, level: str) -> str:
    if level == "county":
        return name.removesuffix(" County").removesuffix(" Parish").removesuffix(" Borough")
    return name


class CensusBoundaryProver:
    """Load exact pinned TIGER 2026 polygons, then bound retained route travel."""

    def __init__(self, cache_dir: Path | None = None, *, sources=None, ie_source=None):
        base = os.environ.get("XDG_CACHE_HOME")
        self.cache_dir = Path(
            cache_dir
            or ((Path(base) if base else Path.home() / ".cache") / "gomoveshift" / "census-2026")
        )
        self.sources = sources or _SOURCES
        self.ie_source = ie_source or _IE_SOURCE
        self._shapes = {}

    def _fetch_pinned(self, filename, expected, url, handler, max_bytes, what) -> bytes:
        path = self.cache_dir / filename
        if not path.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Boundary source: fetching pinned %s into %s", what, path)
            with tempfile.NamedTemporaryFile(dir=self.cache_dir, delete=False) as tmp:
                temp_path = Path(tmp.name)
                try:
                    with urllib.request.build_opener(handler).open(url, timeout=120) as response:
                        size = 0
                        while chunk := response.read(1024 * 1024):
                            size += len(chunk)
                            if size > max_bytes:
                                raise BoundaryArchiveError(f"{what} exceeds size limit")
                            tmp.write(chunk)
                except Exception as exc:
                    temp_path.unlink(missing_ok=True)
                    raise BoundaryArchiveError(f"official {what} unavailable: {exc}") from exc
            data = temp_path.read_bytes()
            if _sha256(data) != expected:
                temp_path.unlink(missing_ok=True)
                raise BoundaryArchiveError(f"official {what} checksum mismatch")
            temp_path.replace(path)
            logger.info("Boundary source: verified and cached %s (%d bytes)", filename, len(data))
        data = path.read_bytes()
        if _sha256(data) != expected:
            raise BoundaryArchiveError(f"cached {what} checksum mismatch")
        return data

    def _pinned_zip(self, level: str) -> bytes:
        filename, expected = self.sources[level]
        return self._fetch_pinned(
            filename,
            expected,
            f"{_BASE}/{level.upper()}/{filename}",
            _NoRedirect(),
            _MAX_ARCHIVE_BYTES,
            f"Census {level} archive",
        )

    def load_ie_county(self, name: str):
        """The union of every Tailte 2026 statutory polygon of ``name`` ("County
        Cork" or "Cork"), in ITM metres. Any invalid or non-areal part refuses."""
        key = ("ie", name.casefold())
        if key in self._shapes:
            return self._shapes[key]
        data = self._fetch_pinned(
            *self.ie_source, _IE_URL, _HostRedirect(_IE_HOSTS), _MAX_IE_BYTES, "Tailte county map"
        )
        doc = json.loads(data)
        if ((doc.get("crs") or {}).get("properties") or {}).get("name") != "EPSG:2157":
            raise BoundaryArchiveError("Tailte county map is not in Irish Transverse Mercator")
        wanted = name.removeprefix("County ").casefold()
        parts = []
        for feature in doc.get("features") or []:
            if (
                str((feature.get("properties") or {}).get("ENG_NAME_VALUE", "")).casefold()
                != wanted
            ):
                continue
            part = force_2d(shape(feature["geometry"]))
            if not isinstance(part, (Polygon, MultiPolygon)) or not part.is_valid:
                raise BoundaryArchiveError(f"Tailte polygon of {name!r} is invalid")
            parts.append(part)
        if not parts:
            raise BoundaryArchiveError(f"Tailte county polygon {name!r} not found")
        polygon = unary_union(parts)
        if not polygon.is_valid or polygon.is_empty:
            raise BoundaryArchiveError(f"Tailte polygon of {name!r} is invalid")
        self._shapes[key] = polygon
        return polygon

    def _prove_ie(self, windows, claim: dict) -> dict:
        names = claim["counties"] if claim["level"] == "counties" else [claim["name"]]
        try:
            polygon = unary_union([self.load_ie_county(n) for n in names])
            result = prove_windows_inside(windows, polygon, to_xy=itm, margin_m=IE_MARGIN_M)
        except (BoundaryArchiveError, OSError, ValueError, KeyError, TypeError) as exc:
            return {"verified": False, "reason": str(exc)}
        if result.get("verified") is True:
            result.update(
                basis=_IE_BASIS,
                source=f"TAILTE2026/COUNTIES/{self.ie_source[0]}",
                sha256=self.ie_source[1],
                region=claim["name"],
                country="IE",
            )
        return result

    def load_region(self, level: str, name: str, statefp: str):
        if level not in self.sources:
            raise BoundaryArchiveError(f"no Census source for {level}")
        key = (level, name.casefold(), statefp)
        if key in self._shapes:
            return self._shapes[key]
        data = self._pinned_zip(level)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            by_ext = {
                Path(n).suffix: n
                for n in z.namelist()
                if Path(n).suffix in {".dbf", ".shp", ".shx"}
            }
            if set(by_ext) != {".dbf", ".shp", ".shx"}:
                raise BoundaryArchiveError("Census archive lacks DBF/SHP/SHX")
            row = next(
                (
                    i
                    for i, values in _dbf_rows(z.read(by_ext[".dbf"]))
                    if _region_name(values["NAME"], level).casefold()
                    == _region_name(name, level).casefold()
                    and values["STATEFP"] == statefp
                ),
                None,
            )
            if row is None:
                raise BoundaryArchiveError(
                    f"Census {level} polygon {name!r} in state {statefp} not found"
                )
            polygon = _polygon_from_shp(z.read(by_ext[".shp"]), z.read(by_ext[".shx"]), row)
        self._shapes[key] = polygon
        return polygon

    def prove(self, windows, claim: dict) -> dict:
        if is_irish(claim):
            return self._prove_ie(windows, claim)
        name = claim["name"]
        level = _SOURCE_LEVEL.get(claim["level"], claim["level"])
        state = claim.get("state") if level == "county" else name
        statefp = _state_fips(state)
        if statefp is None:
            return {"verified": False, "reason": f"unknown US state {state!r}"}
        try:
            if claim["level"] == "counties":
                polygon = unary_union(
                    [self.load_region(level, c, statefp) for c in claim["counties"]]
                )
            else:
                polygon = self.load_region(level, name, statefp)
            result = prove_windows_inside(windows, polygon)
        except (
            BoundaryArchiveError,
            zipfile.BadZipFile,
            OSError,
            ValueError,
            struct.error,
            UnicodeError,
        ) as exc:
            return {"verified": False, "reason": str(exc)}
        result.update(
            source=f"TIGER2026/{level.upper()}/{self.sources[level][0]}",
            sha256=self.sources[level][1],
            region=name,
            statefp=statefp,
        )
        return result


def _state_fips(name: str | None) -> str | None:
    # Census state FIPS is stable and small; explicit mapping avoids an extra
    # archive parse merely to disambiguate county names.
    states = {
        "Alabama": "01",
        "Alaska": "02",
        "Arizona": "04",
        "Arkansas": "05",
        "California": "06",
        "Colorado": "08",
        "Connecticut": "09",
        "Delaware": "10",
        "District of Columbia": "11",
        "Florida": "12",
        "Georgia": "13",
        "Hawaii": "15",
        "Idaho": "16",
        "Illinois": "17",
        "Indiana": "18",
        "Iowa": "19",
        "Kansas": "20",
        "Kentucky": "21",
        "Louisiana": "22",
        "Maine": "23",
        "Maryland": "24",
        "Massachusetts": "25",
        "Michigan": "26",
        "Minnesota": "27",
        "Mississippi": "28",
        "Missouri": "29",
        "Montana": "30",
        "Nebraska": "31",
        "Nevada": "32",
        "New Hampshire": "33",
        "New Jersey": "34",
        "New Mexico": "35",
        "New York": "36",
        "North Carolina": "37",
        "North Dakota": "38",
        "Ohio": "39",
        "Oklahoma": "40",
        "Oregon": "41",
        "Pennsylvania": "42",
        "Rhode Island": "44",
        "South Carolina": "45",
        "South Dakota": "46",
        "Tennessee": "47",
        "Texas": "48",
        "Utah": "49",
        "Vermont": "50",
        "Virginia": "51",
        "Washington": "53",
        "West Virginia": "54",
        "Wisconsin": "55",
        "Wyoming": "56",
    }
    if not name:
        return None
    canonical = next(
        (
            full
            for full, abbr in US_STATE_ABBREVS.items()
            if name.casefold() in {full.casefold(), abbr.casefold()}
        ),
        name,
    )
    return states.get(canonical)


_GRS80_A = 6_378_137.0
_GRS80_E2 = (1 / 298.257222101) * (2 - 1 / 298.257222101)


def itm(lat: float, lon: float) -> tuple:
    """Irish Transverse Mercator (EPSG:2157) metres of a GRS80/ETRS89 fix
    (Snyder's series; millimetre-level across Ireland)."""
    e2 = _GRS80_E2
    ep2 = e2 / (1 - e2)
    k0, lat0, lon0 = 0.99982, math.radians(53.5), math.radians(-8.0)

    def meridian(phi):
        return _GRS80_A * (
            (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * phi
            - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * math.sin(2 * phi)
            + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * math.sin(4 * phi)
            - (35 * e2**3 / 3072) * math.sin(6 * phi)
        )

    phi = math.radians(lat)
    n = _GRS80_A / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = ep2 * math.cos(phi) ** 2
    a = (math.radians(lon) - lon0) * math.cos(phi)
    x = k0 * n * (a + (1 - t + c) * a**3 / 6 + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * a**5 / 120)
    y = k0 * (
        meridian(phi)
        - meridian(lat0)
        + n
        * math.tan(phi)
        * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * a**6 / 720
        )
    )
    return 600_000.0 + x, 750_000.0 + y


def prove_windows_inside(
    windows,
    polygon,
    *,
    max_speed_m_s: float = MAX_SPEED_M_S,
    gps_error_m: float = GPS_ERROR_M,
    to_xy=None,
    margin_m: float = 0.0,
) -> dict:
    """Prove each possible interval stays inside under explicit error bounds.

    A fix's distance to *any* polygon boundary must exceed the full possible
    travel since its previous fix, including retained window edges, plus
    ``margin_m``. No path is interpolated. A near-border interval refuses even
    when endpoints are in. ``to_xy(lat, lon)`` projects fixes onto a polygon
    already in metres; without it the polygon is in degrees.
    """
    windows = list(windows)
    if not windows:
        return {"verified": False, "reason": "no retained windows to prove"}
    if (
        not polygon.is_valid
        or polygon.is_empty
        or max_speed_m_s <= 0
        or gps_error_m <= 0
        or not math.isfinite(margin_m)
        or margin_m < 0
    ):
        return {"verified": False, "reason": "invalid polygon or uncertainty policy"}
    if to_xy is None:
        north = max(abs(polygon.bounds[1]), abs(polygon.bounds[3]))
        if north >= 89:
            return {"verified": False, "reason": "polar geometry needs a different distance model"}
        # Both factors understate metres per degree, making margin tests stricter.
        x_factor = 109_000 * math.cos(math.radians(north))
        y_factor = 109_000
        projected = transform(lambda x, y, z=None: (x * x_factor, y * y_factor), polygon)

        def to_xy(lat, lon):
            return lon * x_factor, lat * y_factor

    else:
        projected = polygon
    boundary = projected.boundary
    gps_error_m += margin_m
    longest = 0.0
    for window in windows:
        try:
            start_s, end_s = float(window.start_s), float(window.end_s)
        except (TypeError, ValueError):
            return {
                "verified": False,
                "reason": f"{window.segment_id}: invalid retained window span",
            }
        if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s >= end_s:
            return {
                "verified": False,
                "reason": f"{window.segment_id}: invalid retained window span",
            }
        fixes = sorted(
            (float(t), float(lat), float(lon))
            for t, lat, lon in window.fixes
            if window.start_s <= float(t) < window.end_s
        )
        if not fixes:
            anchors = (window.prev_fix, window.next_fix)
            if not all(anchors):
                return {
                    "verified": False,
                    "reason": f"{window.segment_id}: no retained GPS fix or two cut anchors",
                }
            before, after = anchors
            if not before[0] < window.start_s or not after[0] >= window.end_s:
                return {
                    "verified": False,
                    "reason": f"{window.segment_id}: cut anchors do not bracket footage",
                }
            span = after[0] - before[0]
            longest = max(longest, span)
            for _time, lat, lon in anchors:
                point = Point(*to_xy(lat, lon))
                if not projected.covers(point):
                    return {
                        "verified": False,
                        "reason": f"{window.segment_id}: cut anchor outside boundary",
                    }
                if boundary.distance(point) <= gps_error_m + max_speed_m_s * span:
                    return {
                        "verified": False,
                        "reason": f"{window.segment_id}: possible boundary crossing without retained fix",
                    }
            continue
        for index, (time_s, lat, lon) in enumerate(fixes):
            if (
                not math.isfinite(lat)
                or not math.isfinite(lon)
                or abs(lat) > 90
                or abs(lon) > 180
                or (lat, lon) == (0, 0)
            ):
                return {"verified": False, "reason": f"{window.segment_id}: invalid GPS fix"}
            point = Point(*to_xy(lat, lon))
            if not projected.covers(point):
                return {
                    "verified": False,
                    "reason": f"{window.segment_id}: retained fix outside boundary",
                }
            previous = window.start_s if index == 0 else fixes[index - 1][0]
            gap = max(0.0, time_s - previous)
            longest = max(longest, gap)
            if boundary.distance(point) <= gps_error_m + max_speed_m_s * gap:
                return {
                    "verified": False,
                    "reason": f"{window.segment_id}: possible boundary crossing in {gap:.1f} s",
                }
        last_time, lat, lon = fixes[-1]
        tail = window.end_s - last_time
        longest = max(longest, tail)
        if boundary.distance(Point(*to_xy(lat, lon))) <= gps_error_m + max_speed_m_s * tail:
            return {
                "verified": False,
                "reason": f"{window.segment_id}: possible boundary crossing at end",
            }
    result = {
        "verified": True,
        "basis": "pinned_tiger_polygon_bounded_speed_and_error",
        "max_speed_m_s": max_speed_m_s,
        "gps_error_m": gps_error_m - margin_m,
        "max_gap_s": round(longest, 3),
        "windows_key": retained_windows_key(windows),
    }
    if margin_m:
        result["margin_m"] = margin_m
    return result
