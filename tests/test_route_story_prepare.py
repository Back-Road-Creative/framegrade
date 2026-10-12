"""Tests for the Route Story bundle-preparation orchestrator.

Network access is always a fake here: production wiring (a real
``requests.Session`` and a real tarball fetch) belongs to the CLI handler,
never to this module.
"""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import shutil
import zipfile
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image

from gomoveshift.pipeline.route_story import prepare
from gomoveshift.pipeline.route_story.scene import vendor as vendor_mod
from gomoveshift.pipeline.route_story.sources import SourceCoverageError

RECORDED_ON = date(2026, 9, 16)

GPX_TEXT = """<?xml version="1.0"?>
<gpx>
<trk><trkseg>
<trkpt lat="35.25" lon="-83.50"><ele>900</ele><time>2026-09-16T14:00:00Z</time></trkpt>
<trkpt lat="35.26" lon="-83.48"><ele>950</ele><time>2026-09-16T14:05:00Z</time></trkpt>
<trkpt lat="35.27" lon="-83.46"><ele>1000</ele><time>2026-09-16T14:10:00Z</time></trkpt>
</trkseg></trk>
</gpx>
"""

# Same trip as GPX_TEXT plus a 1-point track on another date (which alone
# would blow the footprint out) and an exact duplicate of the trip
# track (which would double-draw the route line).
GPX_TEXT_MIXED_HISTORY = """<?xml version="1.0"?>
<gpx>
<trk><trkseg>
<trkpt lat="40.0" lon="-70.0"><time>2022-08-09T09:00:00Z</time></trkpt>
</trkseg></trk>
<trk><trkseg>
<trkpt lat="35.25" lon="-83.50"><ele>900</ele><time>2026-09-16T14:00:00Z</time></trkpt>
<trkpt lat="35.26" lon="-83.48"><ele>950</ele><time>2026-09-16T14:05:00Z</time></trkpt>
<trkpt lat="35.27" lon="-83.46"><ele>1000</ele><time>2026-09-16T14:10:00Z</time></trkpt>
</trkseg></trk>
<trk><trkseg>
<trkpt lat="35.25" lon="-83.50"><ele>900</ele><time>2026-09-16T14:00:00Z</time></trkpt>
<trkpt lat="35.26" lon="-83.48"><ele>950</ele><time>2026-09-16T14:05:00Z</time></trkpt>
<trkpt lat="35.27" lon="-83.46"><ele>1000</ele><time>2026-09-16T14:10:00Z</time></trkpt>
</trkseg></trk>
</gpx>
"""


def _dem_tiff_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    tifffile.imwrite(buf, array.astype(np.float32))
    return buf.getvalue()


def _image_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _rgba_bytes_with_coverage(width: int, height: int, coverage_fraction: float) -> bytes:
    """A transparent-PNG stand-in for the GIBS reference-features overlay.

    ``coverage_fraction`` of the pixels (row-major, from the top) get a
    fully-opaque line colour; the rest stay fully transparent — a coarse but
    deterministic knob for ``borders.border_coverage``.
    """
    array = np.zeros((height, width, 4), dtype=np.uint8)
    covered = max(0, min(height * width, round(height * width * coverage_fraction)))
    flat = array.reshape(-1, 4)
    flat[:covered] = (200, 50, 50, 255)
    buf = io.BytesIO()
    Image.fromarray(array, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


# A minimal Domestic Names pipe-delimited header, matching the real GNIS export
# columns `gnis.parse_gnis` reads by name (see tests/test_route_story_gnis.py).
_GNIS_HEADER = (
    "feature_id|feature_name|feature_class|state_alpha|state_numeric|county_name|"
    "county_numeric|primary_lat_dms|primary_long_dms|prim_lat_dec|prim_long_dec|"
    "source_lat_dms|source_long_dms|source_lat_dec|source_long_dec|elev_in_m|"
    "elev_in_ft|map_name|date_created|date_edited"
)


def _gnis_zip_bytes(rows: list[str] | None = None) -> bytes:
    text = "\n".join([_GNIS_HEADER, *(rows or [])]) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("ST_Text.txt", text)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, content: bytes, *, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# Regional/national imagery bands request far larger pixel grids than the
# focal band (see bands.py's national/regional margins) — capping every
# fake raster response to this size keeps the tests fast regardless of what
# width/height a band actually asked for; nothing downstream checks that a
# fetched array's shape matches the requested size.
_FAKE_IMAGE_CAP_PX = 8

_GIBS_CLEAR_RGB = (80, 120, 60)  # low-saturation, min channel < 200: not cloudy
_GIBS_CLOUDY_RGB = (220, 220, 220)  # near-white, low-saturation: reads as cloud
_GIBS_BLUE_MARBLE_RGB = (30, 60, 90)

# The regional terrain/imagery bbox (~75 km margin) is well over a degree
# wide; the focal bbox (a ~1.5 km margin around a few hundred metres of
# trail) is a small fraction of a degree — this threshold cleanly separates
# a 3DEP call's regional band request from its focal one by bbox width
# alone, without needing to thread band identity through the fake session.
# The regional DEM (one ~0.7 deg z9 tile at ~30 m) is split by the 3DEP
# tile cap into ~0.35 deg requests, so the threshold sits well below that
# and still an order of magnitude above the focal box.
_REGIONAL_BBOX_WIDTH_DEG = 0.2


def _is_regional_bbox(bbox_param: str) -> bool:
    west, _south, east, _north = (float(v) for v in bbox_param.split(","))
    return (east - west) > _REGIONAL_BBOX_WIDTH_DEG


_DEFAULT_OVERPASS_ELEMENTS = [
    {
        "type": "node",
        "id": 1,
        "lat": 35.26,
        "lon": -83.48,
        "tags": {"natural": "peak", "name": "Test Peak"},
    }
]


class FakeSession:
    """Answers 3DEP/NAIP/Overpass/GNIS calls by URL; records every request made.

    GNIS returns no rows and Overpass returns one peak by default, matching the
    pre-merge fixture's single-label behaviour, so existing prepare tests that
    don't care about labels keep the same ``label_count``.
    """

    def __init__(
        self,
        *,
        dem_coverage_error: bool = False,
        dem_coverage_error_regional: bool = False,
        gnis_rows: list[str] | None = None,
        overpass_raises: bool = False,
        overpass_elements: list[dict] | None = None,
        gibs_daily_cloudy: bool = False,
        # Well above `prepare.MIN_SHARED_BORDER_COVERAGE` by default so the
        # borders wiring is transparent to every pre-existing test that
        # doesn't care about it.
        borders_coverage_fraction: float = 0.5,
        borders_raises: bool = False,
    ):
        self.calls: list[dict] = []
        self.dem_coverage_error = dem_coverage_error
        self.dem_coverage_error_regional = dem_coverage_error_regional
        self.gnis_rows = gnis_rows if gnis_rows is not None else []
        self.overpass_raises = overpass_raises
        self.overpass_elements = (
            overpass_elements if overpass_elements is not None else _DEFAULT_OVERPASS_ELEMENTS
        )
        self.gibs_daily_cloudy = gibs_daily_cloudy
        self.borders_coverage_fraction = borders_coverage_fraction
        self.borders_raises = borders_raises

    def get(self, url, *, params=None, timeout=None, headers=None):
        self.calls.append({"url": url, "params": params, "headers": headers})
        if "3DEPElevation" in url:
            w, h = (int(x) for x in params["size"].split(","))
            # Capped like NAIP/gibs below: the regional terrain band's bbox
            # is tens of km wide, which at ~30 m/px would otherwise request
            # (and this fake would then bilinear-sample) multi-megapixel
            # arrays per test run for no assertion this file makes.
            w, h = min(w, _FAKE_IMAGE_CAP_PX), min(h, _FAKE_IMAGE_CAP_PX)
            is_regional = _is_regional_bbox(params["bbox"])
            coverage_error = self.dem_coverage_error or (
                self.dem_coverage_error_regional and is_regional
            )
            value = -1e35 if coverage_error else 950.0
            return FakeResponse(_dem_tiff_bytes(np.full((h, w), value, dtype=np.float32)))
        if "copernicus-dem-30m" in url:
            # A real Copernicus GLO-30 cell is ~3600x3600 px per degree;
            # 400x400 is a coarse-but-adequate stand-in so a small test
            # bbox still crops to a non-degenerate array.
            return FakeResponse(_dem_tiff_bytes(np.full((400, 400), 950.0, dtype=np.float32)))
        if "NAIPImagery" in url:
            w, h = (int(x) for x in params["size"].split(","))
            w, h = min(w, _FAKE_IMAGE_CAP_PX), min(h, _FAKE_IMAGE_CAP_PX)
            return FakeResponse(_image_bytes(np.full((h, w, 3), 120, dtype=np.uint8)))
        if "gibs.earthdata" in url:
            w = min(int(params["WIDTH"]), _FAKE_IMAGE_CAP_PX)
            h = min(int(params["HEIGHT"]), _FAKE_IMAGE_CAP_PX)
            if params["LAYERS"] == "Reference_Features_15m":
                if self.borders_raises:
                    raise RuntimeError("simulated GIBS reference-features outage")
                return FakeResponse(_rgba_bytes_with_coverage(w, h, self.borders_coverage_fraction))
            if params["LAYERS"] == "BlueMarble_ShadedRelief_Bathymetry":
                color = _GIBS_BLUE_MARBLE_RGB
            else:
                color = _GIBS_CLOUDY_RGB if self.gibs_daily_cloudy else _GIBS_CLEAR_RGB
            return FakeResponse(_image_bytes(np.full((h, w, 3), color, dtype=np.uint8)))
        if "prd-tnm.s3.amazonaws.com" in url:
            return FakeResponse(_gnis_zip_bytes(self.gnis_rows))
        if "overpass" in url:
            if self.overpass_raises:
                raise RuntimeError("simulated Overpass outage")
            return FakeResponse(json.dumps({"elements": self.overpass_elements}).encode())
        raise AssertionError(f"unexpected URL: {url}")


def _fake_fetch_maplibre(_url: str) -> bytes:
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name in vendor_mod._DIST_MEMBERS:
            data = name.encode()
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    data = buf.getvalue()
    return data


@pytest.fixture(autouse=True)
def _cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache"))


# Splitting the old single z2-8 gibs band into blue_marble (z2-4) + gibs
# (z5-8, snapped at z5 instead of z2) keeps a default-zoom prepare's total
# imagery tile count in the low thousands rather than the ~10.9k the old
# single national band forced — cheap enough that the full z2-15 default no
# longer needs overriding down to z9-15 to keep this file fast (see
# test_prepare_bundle_default_zoom_tile_count_is_bounded).


@pytest.fixture()
def _patched_maplibre_sri(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neuter SRI verification for the deterministic fake tarball above."""

    def _no_verify(data: bytes, sri: str) -> None:
        return None

    monkeypatch.setattr(vendor_mod, "_verify_sri", _no_verify)


def _gpx(tmp_path: Path, text: str = GPX_TEXT) -> Path:
    path = tmp_path / "trip.gpx"
    path.write_text(text, encoding="utf-8")
    return path


# The default `regional_margin_km` (75 km) snaps the regional imagery/terrain
# bbox to a 3x3 z9 tile envelope — 1+4+16+64 = 85 z9-12 tiles per zoom
# quadruple that on each finer zoom, so a bundle at the default margin writes
# on the order of 750+ tiles per kind. A test that doesn't specifically care
# about that volume passes this instead: it still snaps outward to (at
# least) a single whole z9 tile — ~85 tiles per kind — regardless of how
# small the margin is, so every zoom 9-12 still gets real tiles, just far
# fewer of them. Only `test_prepare_bundle_default_zoom_tile_count_is_bounded`
# (which exists specifically to bound the default-margin tile count) calls
# `prepare.prepare_bundle` directly, at the real default.
_SMALL_REGIONAL_MARGIN_KM = 1.0

# The margin above does not shrink the regional bands as far as it looks:
# `naip_coarse` (and so `terrain_regional`, which shares its bbox) is widened
# to its request cap, 192 z9 tiles for this file's GPX. At the default z9-13
# the regional terrain band then writes 768 z10, 3072 z11 and 12288 z12 tiles
# -- ~16.8k PNGs, ~160 s of a ~200 s prepare. Stopping at z10 writes ~960.
# The range must stay contiguous: the manifest declares min..max and
# `verify_bundle` flags every declared zoom with no tiles, so (9, 10, 13)
# would fail verification and miss the cache. That leaves the z13 focal band
# empty here; only the tests about terrain z11-13 or the focal band take
# `_full_terrain_run`, at the real `prepare.DEFAULT_TERRAIN_ZOOMS`.
_SMALL_TERRAIN_ZOOMS = (9, 10)


def _prepare(*args, **kwargs):
    kwargs.setdefault("regional_margin_km", _SMALL_REGIONAL_MARGIN_KM)
    kwargs.setdefault("terrain_zooms", _SMALL_TERRAIN_ZOOMS)
    return prepare.prepare_bundle(*args, **kwargs)


def _find_prepared_bundle(*args, **kwargs):
    """``prepare.find_prepared_bundle`` with ``_prepare``'s defaults, so a lookup
    keys on the same arguments the matching ``_prepare`` call used."""
    kwargs.setdefault("regional_margin_km", _SMALL_REGIONAL_MARGIN_KM)
    kwargs.setdefault("terrain_zooms", _SMALL_TERRAIN_ZOOMS)
    return prepare.find_prepared_bundle(*args, **kwargs)


@dataclasses.dataclass(frozen=True)
class _PreparedRun:
    """One real ``_prepare`` call, everything a sharing test may read back."""

    report: prepare.PrepareReport
    session: FakeSession
    cache_dir: Path
    gpx_path: Path


def _run_once(
    tmp_path_factory: pytest.TempPathFactory, name: str, session: FakeSession, **kwargs
) -> _PreparedRun:
    """A real ``_prepare`` into its own fresh cache dir, for a module-scoped fixture.

    ``write_tile_pyramid``'s real per-tile bilinear resample + PNG/JPG
    encode (profiled at ~60-70s wall per call, even at
    this file's already-reduced ``_SMALL_REGIONAL_MARGIN_KM``) was otherwise
    repeated, byte-for-byte identically, by every test sharing these inputs:
    same GPX, same ``recorded_on``, same session responses.

    Sharing tests treat the result as read-only. One that needs a second
    ``_prepare`` against this cache takes ``_private_copy`` of it, and one that
    appends to the recorded calls takes ``copy.deepcopy`` of ``session``.
    """
    mp = pytest.MonkeyPatch()
    root = tmp_path_factory.mktemp(name)
    cache_dir = root / "cache"
    mp.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(cache_dir))
    mp.setattr(vendor_mod, "_verify_sri", lambda data, sri: None)
    try:
        gpx_path = _gpx(root)
        report = _prepare(
            gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON, **kwargs
        )
    finally:
        mp.undo()
    return _PreparedRun(report=report, session=session, cache_dir=cache_dir, gpx_path=gpx_path)


def _private_copy(run: _PreparedRun, tmp_path: Path) -> tuple[prepare.PrepareReport, Path]:
    """This test's own copy of ``run``'s whole cache, as though it had run the first prepare.

    Copied into ``tmp_path / "cache"`` — the dir the autouse ``_cache_dir``
    already points ``GMS_ROUTE_STORY_CACHE_DIR`` at — so a later ``_prepare``
    in the test hits (or extends, or repairs) its own copy, never the shared
    one. Returns the report rebased onto the copy and a GPX path of the same
    content (the cache key is the GPX's content, not its path).
    """
    cache_dir = tmp_path / "cache"
    shutil.copytree(run.cache_dir, cache_dir, symlinks=True)
    report = dataclasses.replace(
        run.report, bundle_dir=cache_dir / run.report.bundle_dir.relative_to(run.cache_dir)
    )
    return report, _gpx(tmp_path)


@pytest.fixture(scope="module")
def _default_run(tmp_path_factory: pytest.TempPathFactory) -> _PreparedRun:
    """One ``_prepare`` with a fully-default ``FakeSession()`` and default kwargs."""
    return _run_once(tmp_path_factory, "default-prepared", FakeSession())


@pytest.fixture(scope="module")
def _default_prepared(_default_run: _PreparedRun) -> prepare.PrepareReport:
    return _default_run.report


@pytest.fixture(scope="module")
def _full_terrain_run(tmp_path_factory: pytest.TempPathFactory) -> _PreparedRun:
    """``_default_run`` at the real ``prepare.DEFAULT_TERRAIN_ZOOMS`` (see
    ``_SMALL_TERRAIN_ZOOMS``) -- only for tests about terrain z11-13 or the focal band."""
    return _run_once(
        tmp_path_factory,
        "full-terrain-prepared",
        FakeSession(),
        terrain_zooms=prepare.DEFAULT_TERRAIN_ZOOMS,
    )


def test_shared_prepare_fixture_terrain_tile_count_is_bounded(_default_prepared) -> None:
    """Every module-scoped ``_run_once`` fixture is re-run on each xdist worker,
    and terrain tile writing dominates each run. At the default z9-13 the
    cap-widened regional band wrote ~16.8k terrain tiles (~160 s of a ~200 s
    fixture), which is too slow for the default suite."""
    assert 0 < _default_prepared.terrain_tiles_written < 2000


def test_prepare_bundle_writes_verified_bundle(_default_prepared) -> None:
    report = _default_prepared

    assert report.verify_problems == []
    assert report.terrain_fallback_used is False
    assert report.imagery_source == "NAIP imagery"
    assert report.terrain_source == "USGS 3DEP DEM"
    assert report.label_count == 1
    assert report.imagery_tiles_written > 0
    assert report.terrain_tiles_written > 0
    assert report.excluded_other_date_count == 0
    assert report.excluded_duplicate_count == 0
    assert (report.bundle_dir / "manifest.json").is_file()
    assert (report.bundle_dir / "route.geojson").is_file()
    assert (report.bundle_dir / "labels.geojson").is_file()
    assert (report.bundle_dir / "vendor" / "maplibre-gl-6.10.0").is_dir()

    route = json.loads((report.bundle_dir / "route.geojson").read_text())
    assert route["features"][0]["geometry"]["coordinates"][0] == [-83.50, 35.25]
    assert route["properties"]["excluded_other_date_count"] == 0
    assert route["properties"]["excluded_duplicate_count"] == 0

    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    assert manifest["terrain"]["encoding"] == "mapbox"
    assert manifest["terrain"]["exaggeration"] == 1.0
    # The manifest may only advertise zooms the bundle holds: at a small
    # footprint the wide requested zooms have no fully covered tile, and a
    # zoom the scene cannot fetch leaves MapLibre permanently unsettled.
    for kind, suffix in (("imagery", "jpg"), ("terrain", "png")):
        on_disk = sorted(
            int(path.name) for path in (report.bundle_dir / kind).iterdir() if path.is_dir()
        )
        assert manifest[kind]["minzoom"] == on_disk[0]
        assert manifest[kind]["maxzoom"] == on_disk[-1]
        assert on_disk == list(range(on_disk[0], on_disk[-1] + 1))
        assert any((report.bundle_dir / kind).glob(f"*/*/*.{suffix}"))

    # One tile grid for both layers — imagery is never written wider
    # than the terrain that actually holds a mesh under it.
    imagery_extent = manifest["imagery"]["extent"]
    terrain_extent = manifest["terrain"]["extent"]
    assert imagery_extent["west"] >= terrain_extent["west"]
    assert imagery_extent["east"] <= terrain_extent["east"]
    assert imagery_extent["south"] >= terrain_extent["south"]
    assert imagery_extent["north"] <= terrain_extent["north"]


def test_prepare_bundle_refuses_a_footprint_that_covers_no_whole_tile(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    with pytest.raises(ValueError, match="no tile was written"):
        _prepare(
            _gpx(tmp_path),
            FakeSession(),
            _fake_fetch_maplibre,
            recorded_on=RECORDED_ON,
            imagery_zooms=(15,),
            terrain_zooms=(3,),
        )


def test_prepare_bundle_requires_at_least_one_focal_zoom(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    """Edge case: imagery_zooms excluding the focal (`naip`, z15) band would
    leave `_kind_extent_fields` falling back to the raw focal bbox, which can
    reach past the terrain extent — refused up front, before any fetch."""
    session = FakeSession()
    with pytest.raises(ValueError, match="focal zooms 15-15"):
        _prepare(
            _gpx(tmp_path),
            session,
            _fake_fetch_maplibre,
            recorded_on=RECORDED_ON,
            imagery_zooms=tuple(range(2, 13)),
        )
    assert session.calls == []


def test_prepare_bundle_falls_back_to_copernicus_on_dem_coverage_error(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    session = FakeSession(dem_coverage_error=True)
    report = _prepare(_gpx(tmp_path), session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    assert report.terrain_fallback_used is True
    assert report.terrain_source == "Copernicus GLO-30 DEM"
    assert report.verify_problems == []


def test_prepare_bundle_single_overpass_query(_default_run: _PreparedRun) -> None:
    session = _default_run.session
    overpass_calls = [c for c in session.calls if "overpass" in c["url"]]
    assert len(overpass_calls) == 1


def test_prepare_bundle_is_idempotent_and_skips_refetch(
    tmp_path: Path, _patched_maplibre_sri: None, _merged_labels_run: _PreparedRun
) -> None:
    # Every label source answers: a DEGRADED bundle retries its labels on a hit
    # (by design), and this test is about the hit that must fetch nothing.
    first, gpx_path = _private_copy(_merged_labels_run, tmp_path)
    session = copy.deepcopy(_merged_labels_run.session)
    calls_after_first = len(session.calls)

    second = _prepare(gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    assert second.bundle_dir == first.bundle_dir
    assert len(session.calls) == calls_after_first  # no re-fetch
    assert second.verify_problems == []


def test_prepare_bundle_rejects_gpx_with_no_points(tmp_path: Path) -> None:
    empty = tmp_path / "empty.gpx"
    empty.write_text(
        '<?xml version="1.0"?><gpx><trk><trkseg></trkseg></trk></gpx>', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        _prepare(empty, FakeSession(), _fake_fetch_maplibre, recorded_on=RECORDED_ON)


def test_prepare_bundle_propagates_naip_coverage_error(tmp_path: Path) -> None:
    class BlankNaipSession(FakeSession):
        def get(self, url, *, params=None, timeout=None, headers=None):
            if "NAIPImagery" in url:
                self.calls.append({"url": url, "params": params, "headers": headers})
                w, h = (int(x) for x in params["size"].split(","))
                return FakeResponse(_image_bytes(np.full((h, w, 3), 255, dtype=np.uint8)))
            return super().get(url, params=params, timeout=timeout, headers=headers)

    with pytest.raises(SourceCoverageError):
        _prepare(_gpx(tmp_path), BlankNaipSession(), _fake_fetch_maplibre, recorded_on=RECORDED_ON)


def test_prepare_bundle_selects_only_the_recorded_day(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    """Another-date and duplicate tracks never widen the footprint or the route."""
    session = FakeSession()
    report = _prepare(
        _gpx(tmp_path, GPX_TEXT_MIXED_HISTORY),
        session,
        _fake_fetch_maplibre,
        recorded_on=RECORDED_ON,
    )

    assert report.excluded_other_date_count == 1
    assert report.excluded_duplicate_count == 1
    # The 2022 test point (lon -70) must never reach the footprint.
    assert report.footprint.east < -80.0

    route = json.loads((report.bundle_dir / "route.geojson").read_text())
    assert len(route["features"]) == 1
    assert route["properties"]["excluded_other_date_count"] == 1
    assert route["properties"]["excluded_duplicate_count"] == 1


def test_prepare_bundle_default_exaggeration_is_one(_default_prepared) -> None:
    """The default is 1.0, not the old 1.3."""
    report = _default_prepared
    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    assert manifest["terrain"]["exaggeration"] == 1.0


def test_prepare_bundle_clamps_exaggeration_to_the_editor_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patched_maplibre_sri: None
) -> None:
    # A fresh cache dir per call: the bundle cache key is footprint +
    # source versions, not exaggeration, so two calls that resolve to the
    # same footprint would otherwise share a cached manifest.
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache-high"))
    too_high = _prepare(
        _gpx(tmp_path, GPX_TEXT),
        FakeSession(),
        _fake_fetch_maplibre,
        recorded_on=RECORDED_ON,
        terrain_exaggeration=2.0,
    )
    manifest = json.loads((too_high.bundle_dir / "manifest.json").read_text())
    assert manifest["terrain"]["exaggeration"] == 1.25

    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache-low"))
    too_low = _prepare(
        _gpx(tmp_path, GPX_TEXT),
        FakeSession(),
        _fake_fetch_maplibre,
        recorded_on=RECORDED_ON,
        terrain_exaggeration=0.5,
    )
    manifest = json.loads((too_low.bundle_dir / "manifest.json").read_text())
    assert manifest["terrain"]["exaggeration"] == 1.0


# --- Label merging and loud degradation ------------------------------

_MERGE_GNIS_ROWS = [
    # A "Stream" GNIS row that collides (case-insensitive name + matching
    # ``waterway`` kind) with an OSM feature below — GNIS must win the merge.
    "3000001|Deep Creek|Stream|NC|37|Swain|173|||35.26|-83.475|||||600||TestMap|01/01/1980|",
    # A GNIS-only feature, no OSM counterpart.
    "3000002|Solo Summit|Summit|NC|37|Swain|173|||35.255|-83.49|||||1500||TestMap|01/01/1980|",
]

_MERGE_OSM_ELEMENTS = [
    {
        "type": "node",
        "id": 10,
        "lat": 35.26,
        "lon": -83.475,
        "tags": {"waterway": "stream", "name": "Deep Creek"},
    },
    {
        "type": "node",
        "id": 11,
        "lat": 35.27,
        "lon": -83.46,
        "tags": {"natural": "peak", "name": "Solo Peak"},
    },
]


@pytest.fixture(scope="module")
def _merged_labels_run(tmp_path_factory: pytest.TempPathFactory) -> _PreparedRun:
    """One ``_prepare`` where both label sources answer (an undegraded bundle)."""
    return _run_once(
        tmp_path_factory,
        "merged-labels-prepared",
        FakeSession(gnis_rows=_MERGE_GNIS_ROWS, overpass_elements=_MERGE_OSM_ELEMENTS),
    )


@pytest.fixture(scope="module")
def _degraded_labels_run(tmp_path_factory: pytest.TempPathFactory) -> _PreparedRun:
    """One ``_prepare`` with GNIS answering and Overpass down (a degraded bundle)."""
    return _run_once(
        tmp_path_factory,
        "degraded-labels-prepared",
        FakeSession(gnis_rows=_MERGE_GNIS_ROWS, overpass_raises=True),
    )


def test_prepare_bundle_merges_gnis_and_osm_labels_gnis_preferred(
    _merged_labels_run: _PreparedRun,
) -> None:
    report = _merged_labels_run.report

    assert report.labels_degraded is False
    assert report.label_problems == ()
    assert report.label_count == 3

    labels = json.loads((report.bundle_dir / "labels.geojson").read_text())
    by_key = {
        (f["properties"]["name"], f["properties"]["kind"]): f["properties"]
        for f in labels["features"]
    }
    assert by_key[("Deep Creek", "stream")]["source"] == "gnis"
    assert by_key[("Solo Peak", "peak")]["source"] == "osm"
    assert by_key[("Solo Summit", "summit")]["source"] == "gnis"

    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    assert manifest["labels"]["degraded"] is False
    assert manifest["labels"]["problems"] == []
    assert manifest["labels"]["count"] == 3
    assert len(manifest["labels"]["sources"]) == 2  # one provenance per successful source


def test_prepare_bundle_osm_failure_degrades_loudly_but_keeps_gnis(
    _degraded_labels_run: _PreparedRun,
) -> None:
    report = _degraded_labels_run.report

    assert report.labels_degraded is True
    assert any(p.startswith("osm:") for p in report.label_problems)
    assert report.label_count == 2  # both GNIS rows, no OSM contribution

    labels = json.loads((report.bundle_dir / "labels.geojson").read_text())
    sources = {f["properties"]["source"] for f in labels["features"]}
    assert sources == {"gnis"}


def test_prepare_bundle_both_label_sources_empty_records_both_problems(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    session = FakeSession(gnis_rows=[], overpass_elements=[])
    report = _prepare(_gpx(tmp_path), session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    assert report.labels_degraded is True
    assert report.label_count == 0
    prefixes = {p.split(":", 1)[0] for p in report.label_problems}
    assert prefixes == {"osm", "gnis"}


def test_prepare_bundle_cached_bundle_reports_the_same_degradation_flags(
    tmp_path: Path, _patched_maplibre_sri: None, _degraded_labels_run: _PreparedRun
) -> None:
    """A retry that fails again leaves the cached bundle exactly as it was.

    The labels ARE re-fetched (see the label-repair section below) — but a
    second failure must not touch the bundle, so every flag the first run
    recorded still reads back identically.
    """
    first, gpx_path = _private_copy(_degraded_labels_run, tmp_path)
    session = copy.deepcopy(_degraded_labels_run.session)
    labels_before = (first.bundle_dir / "labels.geojson").read_bytes()
    manifest_before = (first.bundle_dir / "manifest.json").read_bytes()

    second = _prepare(gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    assert second.bundle_dir == first.bundle_dir
    assert second.labels_degraded == first.labels_degraded is True
    assert second.label_problems == first.label_problems
    assert second.label_count == first.label_count
    assert (first.bundle_dir / "labels.geojson").read_bytes() == labels_before
    assert (first.bundle_dir / "manifest.json").read_bytes() == manifest_before


# --- A degraded bundle repairs its labels on the next cache hit ---
#
# Labels are fetched from Overpass and GNIS, and Overpass is the flaky half:
# when all three endpoints fail in one prepare (two 90 s read timeouts and a
# "Network is unreachable"), the bundle is written with GNIS-only labels and
# `labels.degraded: true`. Nothing about that is visible to the cache: the
# label sources are not part of the bundle's address and `verify_bundle` calls
# a GNIS-only bundle perfectly clean, so the degraded bundle becomes the
# permanent answer for that footprint and every later film renders without OSM
# labels until someone deletes the directory by hand.
#
# Refusing the cache hit outright would re-fetch ~5000 terrain and imagery
# tiles to repair a single small JSON file. So the repair is scoped to what
# actually failed: on a cache hit whose manifest records the degradation, the
# LABEL fetch alone is retried over the bbox the manifest itself recorded, and
# the bundle is upgraded in place if — and only if — the retry comes back
# undegraded and no smaller than what is already on disk.


def test_a_degraded_cached_bundle_refetches_only_its_labels(
    tmp_path: Path, _patched_maplibre_sri: None, _degraded_labels_run: _PreparedRun
) -> None:
    first, gpx_path = _private_copy(_degraded_labels_run, tmp_path)
    assert first.labels_degraded is True

    repaired_session = FakeSession(
        gnis_rows=_MERGE_GNIS_ROWS, overpass_elements=_MERGE_OSM_ELEMENTS
    )
    second = _prepare(gpx_path, repaired_session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    assert second.bundle_dir == first.bundle_dir
    assert second.labels_degraded is False
    assert second.label_problems == ()
    assert second.label_count == 3

    # Only the labels were re-fetched: no imagery, terrain or backdrop call.
    refetched = " ".join(call["url"] for call in repaired_session.calls)
    assert "NAIPImagery" not in refetched
    assert "3DEPElevation" not in refetched
    assert "gibs.earthdata" not in refetched


def test_the_repaired_bundle_is_written_through_and_still_verifies(
    tmp_path: Path, _patched_maplibre_sri: None, _degraded_labels_run: _PreparedRun
) -> None:
    first, gpx_path = _private_copy(_degraded_labels_run, tmp_path)
    _prepare(
        gpx_path,
        FakeSession(gnis_rows=_MERGE_GNIS_ROWS, overpass_elements=_MERGE_OSM_ELEMENTS),
        _fake_fetch_maplibre,
        recorded_on=RECORDED_ON,
    )

    labels = json.loads((first.bundle_dir / "labels.geojson").read_text())
    assert {f["properties"]["source"] for f in labels["features"]} == {"gnis", "osm"}

    manifest = json.loads((first.bundle_dir / "manifest.json").read_text())
    assert manifest["labels"]["degraded"] is False
    assert manifest["labels"]["problems"] == []
    assert manifest["labels"]["count"] == 3
    assert len(manifest["labels"]["sources"]) == 2

    # The repair rewrote a file the manifest checksums, so the manifest's own
    # entry for it has to move with it or every later cache probe misses.
    from gomoveshift.pipeline.route_story.bundle import verify_bundle

    assert verify_bundle(first.bundle_dir) == []


def test_an_undegraded_cached_bundle_fetches_nothing_at_all(
    tmp_path: Path, _patched_maplibre_sri: None, _merged_labels_run: _PreparedRun
) -> None:
    """The repair is keyed on the recorded degradation, not run on every hit."""
    first, gpx_path = _private_copy(_merged_labels_run, tmp_path)
    assert first.labels_degraded is False

    quiet_session = FakeSession(gnis_rows=_MERGE_GNIS_ROWS, overpass_elements=_MERGE_OSM_ELEMENTS)
    _prepare(gpx_path, quiet_session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    assert quiet_session.calls == []


def test_the_repair_never_replaces_a_label_set_with_a_smaller_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _patched_maplibre_sri: None,
    _degraded_labels_run: _PreparedRun,
) -> None:
    """A retry can come back clean and still be worse.

    An undegraded fetch only means every configured source answered — it does
    not mean the answer is as complete as the one already on disk (a thinned
    Overpass response, a GNIS extract mid-rebuild). Trading a bigger label set
    for a smaller one silently is the one outcome the repair must not have, so
    the count is the gate, not the degradation flag alone.
    """
    first, gpx_path = _private_copy(_degraded_labels_run, tmp_path)
    assert first.label_count == 2
    labels_before = (first.bundle_dir / "labels.geojson").read_bytes()

    def _one_clean_label(*args: object, **kwargs: object):
        return ([json.loads(labels_before)["features"][0]], [], False, ())

    monkeypatch.setattr(prepare, "_fetch_labels", _one_clean_label)
    second = _prepare(
        gpx_path,
        FakeSession(gnis_rows=_MERGE_GNIS_ROWS),
        _fake_fetch_maplibre,
        recorded_on=RECORDED_ON,
    )

    assert second.label_count == 2
    assert second.labels_degraded is True
    assert (first.bundle_dir / "labels.geojson").read_bytes() == labels_before


# --- Labels cover the regional context -----------------


def test_prepare_bundle_fetches_labels_over_the_naip_coarse_band_bbox(
    _default_run: _PreparedRun,
) -> None:
    """Labels are fetched over the regional naip_coarse band bbox, not the
    small route footprint bbox — otherwise a town or peak just outside the
    route margin never becomes a label."""
    session = _default_run.session
    report = _default_run.report

    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    naip_coarse_bbox = next(
        b["bbox"] for b in manifest["imagery"]["bands"] if b["source"] == "naip_coarse"
    )
    west, south, east, north = naip_coarse_bbox
    expected_clause = f"({south},{west},{north},{east})"

    overpass_calls = [c for c in session.calls if "overpass" in c["url"]]
    assert overpass_calls
    for call in overpass_calls:
        assert expected_clause in call["params"]["data"]


# Within the (tile-snapped) naip_coarse regional bbox but outside the small
# route footprint bbox — the regional context ring this filter exists for.
_CONTEXT_GNIS_ROWS = [
    # A "place" ladder kind, kept as regional context.
    "4000001|Far Place|Populated Place|NC|37|Swain|173|||35.35|-83.30|||||600||TestMap|01/01/1980|",
    # A non-ladder "stream" kind, dropped.
    "4000002|Far Stream|Stream|NC|37|Swain|173|||35.35|-83.31|||||600||TestMap|01/01/1980|",
]

_CONTEXT_OSM_ELEMENTS = [
    {
        "type": "node",
        "id": 20,
        # Exactly a route track point — well inside the footprint.
        "lat": 35.26,
        "lon": -83.48,
        "tags": {"waterway": "stream", "name": "Near Stream"},
    }
]


def test_prepare_bundle_keeps_only_ladder_kinds_outside_the_footprint(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    session = FakeSession(gnis_rows=_CONTEXT_GNIS_ROWS, overpass_elements=_CONTEXT_OSM_ELEMENTS)
    report = _prepare(_gpx(tmp_path), session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    labels = json.loads((report.bundle_dir / "labels.geojson").read_text())
    names = {f["properties"]["name"] for f in labels["features"]}

    assert names == {"Far Place", "Near Stream"}
    assert report.label_count == 2


def test_prepare_bundle_manifest_records_the_label_bbox(_default_prepared) -> None:
    report = _default_prepared

    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    naip_coarse_bbox = next(
        b["bbox"] for b in manifest["imagery"]["bands"] if b["source"] == "naip_coarse"
    )
    assert manifest["labels"]["bbox"] == naip_coarse_bbox


# --- Per-zoom-band imagery --------------------------------------


def test_prepare_bundle_fetches_six_imagery_bands(_default_run: _PreparedRun) -> None:
    """blue_marble (z2-4), gibs (z5-8), naip_coarse (z9-10), naip_mid
    (z11-12), naip_fine (z13-14) and naip (z15) are each fetched over their
    own band bbox, not one focal-sized raster for every zoom.

    ``trip_date_imagery`` defaults off, so the gibs band renders Blue Marble
    directly — one Blue Marble request for the gibs band's own bbox, plus
    one for the shared blue_marble band's cache fill, and no daily
    true-colour request at all (cloud-free by default). Each of those two
    Blue Marble fetches is paired with one reference-features (borders)
    fetch over the same bbox, so the gibs.earthdata
    call count doubled from 2 to 4 and now carries both LAYERS values."""
    session = _default_run.session
    report = _default_run.report

    assert report.verify_problems == []
    gibs_calls = [c for c in session.calls if "gibs.earthdata" in c["url"]]
    naip_calls = [c for c in session.calls if "NAIPImagery" in c["url"]]
    assert len(gibs_calls) == 4
    assert {c["params"]["LAYERS"] for c in gibs_calls} == {
        "BlueMarble_ShadedRelief_Bathymetry",
        "Reference_Features_15m",
    }
    # naip_coarse, naip_mid, naip_fine and naip each issue at least one NAIP
    # request, over bboxes that differ (naip_coarse is the widest of the four).
    assert len(naip_calls) >= 4

    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    band_sources = {b["source"] for b in manifest["imagery"]["bands"]}
    assert band_sources == {
        "blue_marble",
        "gibs",
        "naip_coarse",
        "naip_mid",
        "naip_fine",
        "naip",
    }
    for band in manifest["imagery"]["bands"]:
        if band["source"] == "blue_marble":
            assert band["zooms"] == [2, 4]
            assert band["shared_cache"] is True
        elif band["source"] == "gibs":
            assert band["zooms"] == [5, 8]
        elif band["source"] == "naip_coarse":
            assert band["zooms"] == [9, 10]
        elif band["source"] == "naip_mid":
            assert band["zooms"] == [11, 12]
        elif band["source"] == "naip_fine":
            assert band["zooms"] == [13, 14]
        else:
            assert band["zooms"] == [15, 15]


def test_prepare_bundle_writes_zoom_9_to_12_tiles(_default_prepared) -> None:
    """The bug this fixes: z9-12 used to be silently skipped."""
    report = _default_prepared

    for zoom in (9, 10, 11, 12):
        assert any((report.bundle_dir / "imagery" / str(zoom)).glob("*/*.jpg")), (
            f"no imagery tile written at zoom {zoom}"
        )


@pytest.mark.tmp_disk_budget_mib(160)  # ~5900 real tiles, about 91 MiB by design
def test_prepare_bundle_default_zoom_tile_count_is_bounded(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    """The bug this fixes: one z2-8 gibs band snapped at z2 forced ~10.9k
    tiles per bundle. Splitting blue_marble (z2-4) from gibs (z5-8, snapped
    at z5) keeps a default full z2-15 prepare in the low thousands.

    The upper bound grew from 3000 to 4500 when naip_coarse/naip_mid/naip
    started being floored at their own finest zoom's opening-mosaic cap
    width (`bands._fit_snapped_bbox_to_cap`) -- more real tile fetches, in
    exchange for the opening camera never handing off to a raster it must
    upscale past the sharpness floor.

    It grew again, 4500 -> 6500, when `naip_fine` (z13-14) was added so the
    bands' finest zooms step at most 2 apart (`bands._BAND_SPECS`). A whole
    extra band is a whole extra pyramid: this fixture measures 5943 tiles
    where it measured 4263 before. That is the price of the two spans that
    previously had no sharp raster at all, and it is a real one -- a full
    regional-sized prepare goes from roughly 240 to roughly 330 minutes. The
    bound stays a bound: raised deliberately, with the measured number
    recorded, never widened until the suite goes quiet."""
    session = FakeSession()
    report = prepare.prepare_bundle(
        _gpx(tmp_path), session, _fake_fetch_maplibre, recorded_on=RECORDED_ON
    )

    assert 0 < report.imagery_tiles_written < 6500


# A second trip a few hundred metres from GPX_TEXT's, on a different day —
# close enough that its blue_marble band (national margin, snapped at z2-4)
# lands in exactly the same shared tiles, but a different footprint/date so
# the bundle itself is never a cache hit (_find_cached_bundle) and genuinely
# refetches everything except the shared Blue Marble tiles.
GPX_TEXT_NEARBY_TRIP = """<?xml version="1.0"?>
<gpx>
<trk><trkseg>
<trkpt lat="35.2510" lon="-83.5010"><ele>900</ele><time>2026-09-17T14:00:00Z</time></trkpt>
<trkpt lat="35.2610" lon="-83.4810"><ele>950</ele><time>2026-09-17T14:05:00Z</time></trkpt>
<trkpt lat="35.2710" lon="-83.4610"><ele>1000</ele><time>2026-09-17T14:10:00Z</time></trkpt>
</trkseg></trk>
</gpx>
"""


def test_prepare_bundle_shares_blue_marble_tiles_across_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _patched_maplibre_sri: None
) -> None:
    fetch_calls: list[tuple] = []
    real_fetch = prepare.fetch_gibs_blue_marble

    def _counting_fetch(*args, **kwargs):
        fetch_calls.append((args, kwargs))
        return real_fetch(*args, **kwargs)

    monkeypatch.setattr(prepare, "fetch_gibs_blue_marble", _counting_fetch)

    reference_calls: list[tuple] = []
    real_reference_fetch = prepare.fetch_gibs_reference_features

    def _counting_reference_fetch(*args, **kwargs):
        reference_calls.append((args, kwargs))
        return real_reference_fetch(*args, **kwargs)

    monkeypatch.setattr(prepare, "fetch_gibs_reference_features", _counting_reference_fetch)

    def _shared_calls() -> list[tuple]:
        # Both the shared z2-4 blue_marble band and the per-bundle z5-8
        # gibs band (cloud-free by default) call `fetch_gibs_blue_marble`;
        # only the former's bbox spans the whole hemisphere-wide national
        # margin snapped at z2, so a wide longitude span picks it out.
        return [c for c in fetch_calls if (c[0][1][2] - c[0][1][0]) > 90.0]

    def _shared_reference_calls() -> list[tuple]:
        return [c for c in reference_calls if (c[0][1][2] - c[0][1][0]) > 90.0]

    session = FakeSession()
    first = _prepare(
        _gpx(tmp_path, GPX_TEXT), session, _fake_fetch_maplibre, recorded_on=RECORDED_ON
    )
    assert len(_shared_calls()) == 1  # the shared cache was empty: one fetch
    assert len(_shared_reference_calls()) == 1  # one borders fetch to bake into the shared tiles

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _prepare(
        _gpx(second_dir, GPX_TEXT_NEARBY_TRIP),
        session,
        _fake_fetch_maplibre,
        recorded_on=date(2026, 9, 17),
    )

    assert second.bundle_dir != first.bundle_dir  # genuinely a different bundle
    assert len(_shared_calls()) == 1  # no new shared Blue Marble HTTP call for the second trip
    assert len(_shared_reference_calls()) == 1  # nor a new shared borders fetch: tiles are cached
    for zoom in (2, 3, 4):
        assert any((second.bundle_dir / "imagery" / str(zoom)).glob("*/*.jpg")), (
            f"no shared blue_marble tile in the second bundle at zoom {zoom}"
        )

    # The shared tiles are hard-linked into each bundle, not copied: the two
    # bundles' z2 tile at the same x/y is the very same inode.
    first_tile = next((first.bundle_dir / "imagery" / "2").glob("*/*.jpg"))
    relative = first_tile.relative_to(first.bundle_dir)
    second_tile = second.bundle_dir / relative
    assert second_tile.is_file()
    assert second_tile.stat().st_ino == first_tile.stat().st_ino
    assert second_tile.stat().st_nlink > 1

    second_manifest = json.loads((second.bundle_dir / "manifest.json").read_text())
    blue_marble_entry = next(
        b for b in second_manifest["imagery"]["bands"] if b["source"] == "blue_marble"
    )
    assert blue_marble_entry["tiles_fetched"] == 0
    assert blue_marble_entry["tiles_linked"] > 0
    assert second.verify_problems == []


def test_prepare_bundle_manifest_records_gibs_default_is_cloud_free_blue_marble(
    _default_run: _PreparedRun,
) -> None:
    """``trip_date_imagery`` defaults off: the gibs band always renders Blue
    Marble, whatever the (unrequested) daily scene would have looked like.

    ``_default_run``'s ``FakeSession()`` already has ``gibs_daily_cloudy=False``."""
    assert _default_run.session.gibs_daily_cloudy is False
    session = _default_run.session
    report = _default_run.report
    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    gibs_band = next(b for b in manifest["imagery"]["bands"] if b["source"] == "gibs")
    assert gibs_band["cloud_decision"]["chosen"] == "blue_marble"
    assert gibs_band["cloud_decision"]["trip_date_imagery"] is False
    assert gibs_band["cloud_decision"]["borders"]["coverage"] > 0.0
    gibs_calls = [c for c in session.calls if "gibs.earthdata" in c["url"]]
    assert {c["params"]["LAYERS"] for c in gibs_calls} == {
        "BlueMarble_ShadedRelief_Bathymetry",
        "Reference_Features_15m",
    }


@pytest.fixture(scope="module")
def _trip_date_daily_run(tmp_path_factory: pytest.TempPathFactory) -> _PreparedRun:
    """One ``_prepare`` opted into ``trip_date_imagery`` on a clear (not cloudy) day."""
    return _run_once(
        tmp_path_factory,
        "trip-date-daily-prepared",
        FakeSession(gibs_daily_cloudy=False),
        trip_date_imagery=True,
    )


def test_prepare_bundle_manifest_records_gibs_cloud_decision_daily(
    _trip_date_daily_run: _PreparedRun,
) -> None:
    session = _trip_date_daily_run.session
    report = _trip_date_daily_run.report
    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    gibs_band = next(b for b in manifest["imagery"]["bands"] if b["source"] == "gibs")
    assert gibs_band["cloud_decision"]["chosen"] == "daily"
    assert gibs_band["cloud_decision"]["trip_date_imagery"] is True
    # A chosen trip-date daily scene gets no borders overlay and makes no
    # reference-features request for the gibs (z5-8) band -- the shared
    # z2-4 blue_marble band still fetches its own borders regardless, so
    # only the gibs band's (narrower, non-national) bbox is checked here.
    assert gibs_band["cloud_decision"]["borders"] is None
    reference_calls = [
        c
        for c in session.calls
        if "gibs.earthdata" in c["url"] and c["params"]["LAYERS"] == "Reference_Features_15m"
    ]
    gibs_band_reference_calls = [
        c
        for c in reference_calls
        if (float(c["params"]["BBOX"].split(",")[3]) - float(c["params"]["BBOX"].split(",")[1]))
        <= 90.0
    ]
    assert gibs_band_reference_calls == []


def test_prepare_bundle_manifest_records_gibs_cloud_decision_blue_marble(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    session = FakeSession(gibs_daily_cloudy=True)
    report = _prepare(
        _gpx(tmp_path),
        session,
        _fake_fetch_maplibre,
        recorded_on=RECORDED_ON,
        trip_date_imagery=True,
    )
    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    gibs_band = next(b for b in manifest["imagery"]["bands"] if b["source"] == "gibs")
    assert gibs_band["cloud_decision"]["chosen"] == "blue_marble"
    assert gibs_band["cloud_decision"]["trip_date_imagery"] is True
    # Falling back to Blue Marble (too cloudy for the daily scene) still
    # draws the borders overlay.
    assert gibs_band["cloud_decision"]["borders"]["coverage"] > 0.0


def test_prepare_bundle_focal_imagery_extent_stays_inside_terrain_extent(_default_prepared) -> None:
    """The shared-grid rule still holds once the manifest imagery.extent is band-scoped: the
    top-level extent is the focal band's, which must stay inside terrain's."""
    report = _default_prepared
    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    imagery_extent = manifest["imagery"]["extent"]
    terrain_extent = manifest["terrain"]["extent"]
    eps = 1e-6
    assert imagery_extent["west"] >= terrain_extent["west"] - eps
    assert imagery_extent["east"] <= terrain_extent["east"] + eps
    assert imagery_extent["south"] >= terrain_extent["south"] - eps
    assert imagery_extent["north"] <= terrain_extent["north"] + eps


def _terrain_tile_count(bundle_dir: Path, zoom: int) -> int:
    return len(list((bundle_dir / "terrain" / str(zoom)).glob("*/*.png")))


def _imagery_tile_keys(bundle_dir: Path, zoom: int) -> set[tuple[int, int]]:
    return {
        (int(path.parent.name), int(path.stem))
        for path in (bundle_dir / "imagery" / str(zoom)).glob("*/*.jpg")
    }


def _terrain_tile_keys(bundle_dir: Path, zoom: int) -> set[tuple[int, int]]:
    return {
        (int(path.parent.name), int(path.stem))
        for path in (bundle_dir / "terrain" / str(zoom)).glob("*/*.png")
    }


def test_prepare_bundle_writes_terrain_at_every_default_zoom(
    _full_terrain_run: _PreparedRun,
) -> None:
    """The core regression: a focal-sized DEM raster can never fully cover a
    regional-sized z9-12 tile, so before the regional terrain band existed
    those coarse zooms silently wrote no tiles at all (the same bug class as the
    skipped imagery zooms, for terrain instead of imagery)."""
    report = _full_terrain_run.report

    for zoom in range(9, 14):
        count = _terrain_tile_count(report.bundle_dir, zoom)
        assert count > 0, f"terrain zoom {zoom} has no tiles"


def test_prepare_bundle_regional_dem_fetched_at_coarse_resolution(
    _default_run: _PreparedRun,
) -> None:
    session = _default_run.session

    dem_calls = [c for c in session.calls if "3DEPElevation" in c["url"]]
    assert dem_calls  # sanity: 3DEP was actually called

    def implied_m_per_px(call: dict) -> float:
        west, south, east, _north = (float(v) for v in call["params"]["bbox"].split(","))
        w, _h = (int(x) for x in call["params"]["size"].split(","))
        mean_lat_cos = 0.819  # ~35 deg N, matches the trip's fake GPX
        width_m = (east - west) * 111_320.0 * mean_lat_cos
        return width_m / w

    regional_calls = [c for c in dem_calls if _is_regional_bbox(c["params"]["bbox"])]
    focal_calls = [c for c in dem_calls if not _is_regional_bbox(c["params"]["bbox"])]
    assert regional_calls, "expected at least one regional 3DEP request"
    assert focal_calls, "expected at least one focal 3DEP request"

    # Focal stays at 3DEP's native ~10 m/px; the regional request must not
    # reuse that resolution over its far wider bbox.
    assert all(implied_m_per_px(c) < 15.0 for c in focal_calls)
    assert all(implied_m_per_px(c) > 20.0 for c in regional_calls)


def test_prepare_bundle_regional_imagery_tiles_subset_of_regional_terrain(
    _full_terrain_run: _PreparedRun,
) -> None:
    """z9-12 imagery must never extend past the real relief under it (the
    per-band shared-grid idea) — every naip_coarse tile is also a terrain tile."""
    report = _full_terrain_run.report

    for zoom in range(9, 13):
        imagery_keys = _imagery_tile_keys(report.bundle_dir, zoom)
        terrain_keys = _terrain_tile_keys(report.bundle_dir, zoom)
        assert imagery_keys, f"expected regional imagery tiles at zoom {zoom}"
        assert imagery_keys <= terrain_keys


def test_prepare_bundle_verify_bundle_clean_with_regional_terrain(_default_prepared) -> None:
    report = _default_prepared

    from gomoveshift.pipeline.route_story.bundle import verify_bundle

    assert verify_bundle(report.bundle_dir) == []


def test_prepare_bundle_manifest_terrain_bands_block(_full_terrain_run: _PreparedRun) -> None:
    report = _full_terrain_run.report
    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())

    bands = manifest["terrain"]["bands"]
    sources = {b["source"] for b in bands}
    assert sources == {"terrain_regional", "terrain_focal"}
    for band in bands:
        assert band["tiles_written"] > 0
        assert "provenance" in band
    assert len(manifest["terrain"]["sources"]) >= 1


def test_prepare_bundle_regional_only_copernicus_fallback_is_recorded(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    """3DEP can cover the focal box but not the wider regional one — the two
    terrain fallbacks are decided independently."""
    session = FakeSession(dem_coverage_error_regional=True)
    report = _prepare(_gpx(tmp_path), session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    assert report.terrain_fallback_used is True
    assert report.verify_problems == []
    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    bands = {b["source"]: b for b in manifest["terrain"]["bands"]}
    assert bands["terrain_regional"]["provenance"]["name"] == "Copernicus GLO-30 DEM"
    assert bands["terrain_focal"]["provenance"]["name"] == "USGS 3DEP DEM"
    source_names = {s["name"] for s in manifest["terrain"]["sources"]}
    assert source_names == {"Copernicus GLO-30 DEM", "USGS 3DEP DEM"}


def test_prepare_bundle_different_exaggeration_uses_a_different_bundle_dir(
    tmp_path: Path, _patched_maplibre_sri: None, _default_run: _PreparedRun
) -> None:
    # `_default_run` is the terrain_exaggeration=1.0 call (the default).
    assert prepare.DEFAULT_EXAGGERATION == 1.0
    default_exaggeration, gpx_path = _private_copy(_default_run, tmp_path)
    session = FakeSession()
    higher_exaggeration = _prepare(
        gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON, terrain_exaggeration=1.2
    )
    assert default_exaggeration.bundle_dir != higher_exaggeration.bundle_dir


def test_prepare_bundle_different_regional_margin_uses_a_different_bundle_dir(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    """`regional_margin_km` changes the regional bands' bbox (and so their
    tile content) without changing the overall route footprint, so it must
    be part of the cache key or two differently-sized regional fetches would
    collide on one bundle directory. Whatever it is, `naip_coarse` and
    `terrain_regional` must still share the exact same bbox (the guarantee
    `plan_imagery_bands`/`plan_terrain_bands` are built to give)."""
    session = FakeSession()
    gpx_path = _gpx(tmp_path)
    small = _prepare(
        gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON, regional_margin_km=2.0
    )
    larger = _prepare(
        gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON, regional_margin_km=6.0
    )
    assert small.bundle_dir != larger.bundle_dir

    for report in (small, larger):
        manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
        naip_coarse_bbox = next(
            b["bbox"] for b in manifest["imagery"]["bands"] if b["source"] == "naip_coarse"
        )
        terrain_regional_bbox = next(
            b["bbox"] for b in manifest["terrain"]["bands"] if b["source"] == "terrain_regional"
        )
        assert naip_coarse_bbox == terrain_regional_bbox


def test_prepare_bundle_repeat_call_same_exaggeration_hits_the_cache(
    tmp_path: Path, _patched_maplibre_sri: None
) -> None:
    # Undegraded on purpose -- see test_prepare_bundle_is_idempotent_and_skips_refetch.
    session = FakeSession(gnis_rows=_MERGE_GNIS_ROWS, overpass_elements=_MERGE_OSM_ELEMENTS)
    gpx_path = _gpx(tmp_path)
    first = _prepare(
        gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON, terrain_exaggeration=1.2
    )
    calls_after_first = len(session.calls)
    second = _prepare(
        gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON, terrain_exaggeration=1.2
    )

    assert second.bundle_dir == first.bundle_dir
    assert len(session.calls) == calls_after_first  # no re-fetch


def test_find_prepared_bundle_matches_prepare_bundle(
    monkeypatch: pytest.MonkeyPatch, _default_run: _PreparedRun
) -> None:
    """The route-story-proof CLI resolves a bundle through this lookup, never
    its own re-derived footprint/key — so it must return exactly the same
    directory a matching ``prepare_bundle`` call already wrote."""
    # A read-only lookup: point it at the shared run's own cache.
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(_default_run.cache_dir))
    gpx_path = _default_run.gpx_path
    report = _default_run.report

    found = _find_prepared_bundle(
        gpx_path, recorded_on=RECORDED_ON, regional_margin_km=_SMALL_REGIONAL_MARGIN_KM
    )

    assert found == report.bundle_dir


def test_find_prepared_bundle_returns_none_for_a_different_exaggeration(
    monkeypatch: pytest.MonkeyPatch, _default_run: _PreparedRun
) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(_default_run.cache_dir))
    gpx_path = _default_run.gpx_path

    found = _find_prepared_bundle(
        gpx_path,
        recorded_on=RECORDED_ON,
        regional_margin_km=_SMALL_REGIONAL_MARGIN_KM,
        terrain_exaggeration=1.2,
    )

    assert found is None


def test_find_prepared_bundle_returns_none_for_a_different_regional_margin(
    monkeypatch: pytest.MonkeyPatch, _default_run: _PreparedRun
) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(_default_run.cache_dir))
    gpx_path = _default_run.gpx_path

    found = _find_prepared_bundle(gpx_path, recorded_on=RECORDED_ON, regional_margin_km=6.0)

    assert found is None


def test_cli_handler_reports_labels_degraded_but_still_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from gomoveshift.cli.handlers import route_story_prepare as handler_mod

    fake_report = prepare.PrepareReport(
        bundle_dir=tmp_path,
        footprint=prepare.Footprint(west=-1, south=-1, east=1, north=1),
        imagery_source="NAIP imagery",
        terrain_source="USGS 3DEP DEM",
        terrain_fallback_used=False,
        imagery_zooms=(11,),
        terrain_zooms=(9,),
        imagery_tiles_written=1,
        imagery_tiles_skipped=0,
        terrain_tiles_written=1,
        terrain_tiles_skipped=0,
        label_count=2,
        excluded_other_date_count=0,
        excluded_duplicate_count=0,
        verify_problems=[],
        labels_degraded=True,
        label_problems=("osm: Overpass fetch failed at every endpoint: boom",),
    )
    monkeypatch.setattr(prepare, "prepare_bundle", lambda *a, **k: fake_report)

    args = SimpleNamespace(
        gpx=tmp_path / "trip.gpx",
        recorded_on=RECORDED_ON,
        margin_m=prepare.DEFAULT_MARGIN_M,
        overpass_endpoint=None,
        exaggeration=prepare.DEFAULT_EXAGGERATION,
        trip_date_imagery=prepare.DEFAULT_TRIP_DATE_IMAGERY,
    )
    exit_code = handler_mod.cmd_route_story_prepare(args, config=None)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "LABELS DEGRADED:" in captured.err
    assert "osm: Overpass fetch failed at every endpoint: boom" in captured.err


def test_install_blue_marble_tiles_twice_into_one_bundle_relinks(tmp_path):
    """A re-prepare into a bundle dir that failed verification re-installs
    the shared tiles over the hard links the first run left: link then copy
    both refused and copy2 raised SameFileError."""
    shared = tmp_path / "shared"
    bundle_root = tmp_path / "bundle"
    tile = (2, 0, 1)
    source = prepare._imagery_tile_path(shared, *tile)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"blue-marble")

    prepare._install_blue_marble_tiles(bundle_root, shared, [tile])
    report, linked, copied = prepare._install_blue_marble_tiles(bundle_root, shared, [tile])

    destination = prepare._imagery_tile_path(bundle_root, *tile)
    assert report.written == [tile]
    assert (linked, copied) == (1, 0)
    assert destination.read_bytes() == b"blue-marble"


def test_the_cache_key_carries_the_arcgis_export_grid_version():
    """A bundle fetched on the old, extent-shifting export grid must never be a cache hit."""
    from datetime import date as _date

    from gomoveshift.pipeline.route_story import prepare as prepare_mod
    from gomoveshift.pipeline.route_story import sources as sources_mod

    versions = prepare_mod._band_source_versions(
        gibs_chosen="daily",
        recorded_on=_date(2026, 9, 16),
        terrain_focal_name=sources_mod.DEM_3DEP_PROVENANCE_NAME,
        terrain_regional_name=sources_mod.DEM_3DEP_PROVENANCE_NAME,
        regional_margin_km=25.0,
    )
    assert versions["arcgis_export"] == sources_mod.ARCGIS_EXPORT_GRID_VERSION
    assert sources_mod.ARCGIS_EXPORT_GRID_VERSION != ""


# --- Coastline/border overlay on the Blue Marble backdrop ----

from gomoveshift.pipeline.route_story.bands import plan_imagery_bands  # noqa: E402
from gomoveshift.pipeline.route_story.borders import BORDERS_RECIPE_VERSION  # noqa: E402

# Real planner output (not a hand-built ZoomBand): plan_imagery_bands snaps
# every band's bbox outward to a whole-tile envelope at its own coarsest
# zoom, which guarantees `write_tile_pyramid` fully covers every tile in
# that envelope -- a hand-picked bbox near the poles (Web Mercator's
# ~85.05 deg limit) can leave a few edge tiles permanently partially
# covered and never written, which looks like a spurious cache miss on
# every call.
_TEST_IMAGERY_BANDS = plan_imagery_bands(
    (-83.50, 35.25, -83.46, 35.27), focal_margin_m=prepare.DEFAULT_MARGIN_M
)
_SHARED_BLUE_MARBLE_BAND = next(b for b in _TEST_IMAGERY_BANDS if b.source == "blue_marble")
_GIBS_BAND = next(b for b in _TEST_IMAGERY_BANDS if b.source == "gibs")


def test_ensure_blue_marble_tiles_applies_borders_after_grading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared z2-4 tiles are graded, THEN bordered -- never the reverse,
    or the line colour would itself get re-graded."""
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache"))
    order: list[str] = []
    real_grade = prepare.grade_backdrop
    real_overlay = prepare.overlay_borders

    def _spy_grade(array):
        order.append("grade")
        return real_grade(array)

    def _spy_overlay(rgb, overlay_rgba, **kwargs):
        order.append("overlay")
        # The overlay must be blending onto the *graded* output, not the
        # raw fetched raster -- grade_backdrop is not a no-op on the fake's
        # flat Blue Marble colour, so an unequal array here would mean the
        # overlay ran before grading.
        assert np.array_equal(rgb, real_grade(np.full_like(rgb, _GIBS_BLUE_MARBLE_RGB)))
        return real_overlay(rgb, overlay_rgba, **kwargs)

    monkeypatch.setattr(prepare, "grade_backdrop", _spy_grade)
    monkeypatch.setattr(prepare, "overlay_borders", _spy_overlay)

    session = FakeSession()
    provenance, borders_provenance, needed, fetched = prepare._ensure_blue_marble_tiles(
        session, _SHARED_BLUE_MARBLE_BAND, list(_SHARED_BLUE_MARBLE_BAND.zooms)
    )

    assert fetched > 0
    assert order == ["grade", "overlay"]
    assert borders_provenance is not None
    assert borders_provenance.name == "NASA GIBS Reference_Features_15m"


def test_ensure_blue_marble_tiles_borders_provenance_persists_across_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second call against an already-warm shared cache must still return the
    borders provenance -- read from disk, not re-fetched -- and must make zero
    HTTP calls."""
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache"))
    session = FakeSession()
    first_provenance, first_borders_provenance, _needed, first_fetched = (
        prepare._ensure_blue_marble_tiles(
            session, _SHARED_BLUE_MARBLE_BAND, list(_SHARED_BLUE_MARBLE_BAND.zooms)
        )
    )
    assert first_fetched > 0
    calls_after_first = len(session.calls)

    second_provenance, second_borders_provenance, _needed2, second_fetched = (
        prepare._ensure_blue_marble_tiles(
            session, _SHARED_BLUE_MARBLE_BAND, list(_SHARED_BLUE_MARBLE_BAND.zooms)
        )
    )

    assert second_fetched == 0
    assert len(session.calls) == calls_after_first  # no new HTTP calls at all
    assert second_provenance == first_provenance
    assert second_borders_provenance == first_borders_provenance


def test_ensure_blue_marble_tiles_refetches_when_only_borders_provenance_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache hit with the borders-provenance file missing must count as
    missing and refetch, same as a missing ``provenance.json`` does."""
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache"))
    session = FakeSession()
    prepare._ensure_blue_marble_tiles(
        session, _SHARED_BLUE_MARBLE_BAND, list(_SHARED_BLUE_MARBLE_BAND.zooms)
    )
    shared_dir = prepare._blue_marble_shared_dir()
    borders_provenance_path = prepare._blue_marble_borders_provenance_path(shared_dir)
    assert borders_provenance_path.is_file()
    borders_provenance_path.unlink()

    _provenance, borders_provenance, _needed, fetched = prepare._ensure_blue_marble_tiles(
        session, _SHARED_BLUE_MARBLE_BAND, list(_SHARED_BLUE_MARBLE_BAND.zooms)
    )

    assert fetched > 0  # refetched, not silently treated as a hit
    assert borders_provenance_path.is_file()
    assert borders_provenance is not None


def test_ensure_blue_marble_tiles_raises_before_writing_on_low_coverage(tmp_path: Path) -> None:
    """Near-zero coverage over the whole-globe shared bbox is treated as a
    failed fetch: it must never ship a borderless backdrop as though it
    were correct, and must never poison the shared cache."""
    cache_dir = tmp_path / "cache"
    session = FakeSession(borders_coverage_fraction=0.0)

    with pytest.raises(prepare.SourceFetchError):
        prepare._ensure_blue_marble_tiles(
            session, _SHARED_BLUE_MARBLE_BAND, list(_SHARED_BLUE_MARBLE_BAND.zooms)
        )

    assert not cache_dir.exists()


def test_ensure_blue_marble_tiles_raises_before_writing_on_fetch_error(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    session = FakeSession(borders_raises=True)

    with pytest.raises(Exception):
        prepare._ensure_blue_marble_tiles(
            session, _SHARED_BLUE_MARBLE_BAND, list(_SHARED_BLUE_MARBLE_BAND.zooms)
        )

    assert not cache_dir.exists()


def test_gibs_band_raster_overlays_borders_on_blue_marble_path(tmp_path: Path) -> None:
    session = FakeSession(borders_coverage_fraction=0.5)
    raster, decision, borders_provenance = prepare._gibs_band_raster(
        session, _GIBS_BAND, RECORDED_ON, trip_date_imagery=False
    )
    assert decision["chosen"] == "blue_marble"
    assert decision["borders"]["coverage"] > 0.0
    assert borders_provenance is not None


def test_gibs_band_raster_records_zero_coverage_without_raising(tmp_path: Path) -> None:
    """A small inland footprint can legitimately hold no coastline/border at
    all -- unlike the shared whole-globe band, this must be recorded, not
    treated as a failure."""
    session = FakeSession(borders_coverage_fraction=0.0)
    raster, decision, borders_provenance = prepare._gibs_band_raster(
        session, _GIBS_BAND, RECORDED_ON, trip_date_imagery=False
    )
    assert decision["chosen"] == "blue_marble"
    assert decision["borders"]["coverage"] == 0.0


def _reference_features_sources(manifest: dict) -> list[dict]:
    return [
        s for s in manifest["imagery"]["sources"] if s["name"] == "NASA GIBS Reference_Features_15m"
    ]


def test_prepare_bundle_manifest_lists_borders_source_once_in_default_mode(
    _default_prepared,
) -> None:
    """Both the shared z2-4 band and the gibs z5-8 band draw borders by
    default, but they're the same source -- must not be listed twice."""
    report = _default_prepared
    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    assert len(_reference_features_sources(manifest)) == 1


def test_prepare_bundle_manifest_lists_borders_source_once_with_daily_scene_chosen(
    _trip_date_daily_run: _PreparedRun,
) -> None:
    """A chosen trip-date daily scene draws no borders on the gibs band, but
    the shared z2-4 band's own borders overlay must still be named as a
    source -- previously dropped entirely."""
    report = _trip_date_daily_run.report
    manifest = json.loads((report.bundle_dir / "manifest.json").read_text())
    gibs_band = next(b for b in manifest["imagery"]["bands"] if b["source"] == "gibs")
    assert gibs_band["cloud_decision"]["chosen"] == "daily"
    assert len(_reference_features_sources(manifest)) == 1


def test_prepare_bundle_second_bundle_off_warm_shared_cache_still_lists_borders_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _patched_maplibre_sri: None,
    _default_run: _PreparedRun,
) -> None:
    """A second, different bundle whose blue_marble band lands in the
    already-cached shared tiles must still name the borders source in its
    own manifest, and must make zero reference-features HTTP calls."""
    first, _gpx_path = _private_copy(_default_run, tmp_path)
    session = copy.deepcopy(_default_run.session)
    assert (
        len(
            _reference_features_sources(
                json.loads((first.bundle_dir / "manifest.json").read_text())
            )
        )
        == 1
    )
    calls_after_first = len(session.calls)

    second_dir = tmp_path / "second"
    second_dir.mkdir()
    second = _prepare(
        _gpx(second_dir, GPX_TEXT_NEARBY_TRIP),
        session,
        _fake_fetch_maplibre,
        recorded_on=date(2026, 9, 17),
    )

    reference_calls_between = [
        c
        for c in session.calls[calls_after_first:]
        if "gibs.earthdata" in c["url"] and c["params"]["LAYERS"] == "Reference_Features_15m"
    ]
    shared_reference_calls_between = [
        c
        for c in reference_calls_between
        if (float(c["params"]["BBOX"].split(",")[3]) - float(c["params"]["BBOX"].split(",")[1]))
        > 90.0
    ]
    assert shared_reference_calls_between == []  # no new shared borders fetch: cache is warm

    second_manifest = json.loads((second.bundle_dir / "manifest.json").read_text())
    assert len(_reference_features_sources(second_manifest)) == 1


def test_band_source_versions_includes_borders_recipe() -> None:
    versions = prepare._band_source_versions(
        gibs_chosen="blue_marble",
        recorded_on=RECORDED_ON,
        terrain_focal_name="x",
        terrain_regional_name="y",
        regional_margin_km=1.0,
    )
    assert versions["borders_recipe"] == BORDERS_RECIPE_VERSION


def test_prepare_bundle_old_borders_recipe_is_not_a_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _patched_maplibre_sri: None,
    _default_run: _PreparedRun,
) -> None:
    """A bundle prepared under a stale borders recipe (or stale
    BLUE_MARBLE_CACHE_VERSION) must never be returned as a cache hit for the
    current recipe -- otherwise an old, borderless bundle looks prepared."""
    first, gpx_path = _private_copy(_default_run, tmp_path)
    session = FakeSession()
    # The copy really is a hit under the current recipe, so a changed bundle
    # dir below is the recipe's doing, not a broken copy.
    assert (
        _find_prepared_bundle(
            gpx_path, recorded_on=RECORDED_ON, regional_margin_km=_SMALL_REGIONAL_MARGIN_KM
        )
        == first.bundle_dir
    )

    monkeypatch.setattr(prepare, "BORDERS_RECIPE_VERSION", "borders-0")
    second = _prepare(gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    assert second.bundle_dir != first.bundle_dir


def test_route_story_prepare_cli_flag_trip_date_imagery_defaults_off() -> None:
    from gomoveshift.cli.parser import _build_parser

    args = _build_parser().parse_args(
        ["route-story-prepare", "trip.gpx", "--recorded-on", "2026-09-16"]
    )
    assert args.trip_date_imagery is False


def test_route_story_prepare_cli_flag_trip_date_imagery_reaches_prepare_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gomoveshift.cli.handlers import route_story_prepare as handler_mod
    from gomoveshift.cli.parser import _build_parser

    captured: dict = {}
    fake_report = prepare.PrepareReport(
        bundle_dir=tmp_path,
        footprint=prepare.Footprint(west=-1, south=-1, east=1, north=1),
        imagery_source="NAIP imagery",
        terrain_source="USGS 3DEP DEM",
        terrain_fallback_used=False,
        imagery_zooms=(11,),
        terrain_zooms=(9,),
        imagery_tiles_written=1,
        imagery_tiles_skipped=0,
        terrain_tiles_written=1,
        terrain_tiles_skipped=0,
        label_count=0,
        excluded_other_date_count=0,
        excluded_duplicate_count=0,
        verify_problems=[],
        labels_degraded=False,
        label_problems=(),
    )

    def _fake_prepare_bundle(*args, **kwargs):
        captured.update(kwargs)
        return fake_report

    monkeypatch.setattr(prepare, "prepare_bundle", _fake_prepare_bundle)

    args = _build_parser().parse_args(
        [
            "route-story-prepare",
            str(tmp_path / "trip.gpx"),
            "--recorded-on",
            "2026-09-16",
            "--trip-date-imagery",
        ]
    )
    exit_code = handler_mod.cmd_route_story_prepare(args, config=None)

    assert exit_code == 0
    assert captured["trip_date_imagery"] is True
