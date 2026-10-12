"""Red-first tests for the Route Story public map source fetchers."""

from __future__ import annotations

import io
import json
from datetime import date

import numpy as np
import pytest
import tifffile
from PIL import Image

from gomoveshift.pipeline.route_story import sources as src


def _dem_tiff_bytes(array: np.ndarray, *, nodata: str | None = None) -> bytes:
    buf = io.BytesIO()
    extratags = [(42113, "s", 0, nodata.encode(), True)] if nodata is not None else None
    tifffile.imwrite(buf, array.astype(np.float32), extratags=extratags)
    return buf.getvalue()


def _image_bytes(array: np.ndarray, *, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(buf, format=fmt)
    return buf.getvalue()


class FakeResponse:
    def __init__(self, content: bytes, *, status_code: int = 200, headers: dict | None = None):
        self.content = content
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Records every .get() call and answers from a handler callback."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    def get(self, url, *, params=None, timeout=None, headers=None):
        call = {"url": url, "params": params, "timeout": timeout, "headers": headers}
        self.calls.append(call)
        assert timeout is not None
        return self.handler(call)


# --- fetch_3dep_dem -----------------------------------------------------

DEM_BBOX = (-105.0, 40.0, -104.995, 40.004)


def test_fetch_3dep_dem_single_tile():
    def handler(call):
        assert call["url"] == src._DEM_URL
        assert call["params"]["format"] == "tiff"
        assert call["params"]["pixelType"] == "F32"
        assert call["params"]["interpolation"] == "RSP_BilinearInterpolation"
        w, h = (int(x) for x in call["params"]["size"].split(","))
        array = np.full((h, w), 1500.0, dtype=np.float32)
        return FakeResponse(_dem_tiff_bytes(array))

    session = FakeSession(handler)
    raster = src.fetch_3dep_dem(session, DEM_BBOX)

    assert len(session.calls) == 1
    assert raster.array.dtype == np.float32
    assert np.all(raster.array == 1500.0)
    assert raster.west == DEM_BBOX[0]
    assert raster.north == DEM_BBOX[3]
    assert raster.provenance.name == "USGS 3DEP DEM"
    assert raster.provenance.licence == "Public domain (USGS)"
    assert raster.provenance.attribution == "USGS 3D Elevation Program"


def test_fetch_3dep_dem_stitches_grid_north_to_south():
    def handler(call):
        w, h = (int(x) for x in call["params"]["size"].split(","))
        value = float(len(session.calls))  # 1-indexed by call count
        return FakeResponse(_dem_tiff_bytes(np.full((h, w), value, dtype=np.float32)))

    session = FakeSession(handler)
    # 56x45 px now that the grid is square in degrees (was 43x45): 30 keeps this a 2x2 stitch.
    raster = src.fetch_3dep_dem(session, DEM_BBOX, max_px=30)

    assert len(session.calls) == 4  # 2x2 grid
    # first call is the north-west tile, so top-left corner carries value 1.0
    assert raster.array[0, 0] == 1.0
    # last call is the south-east tile, so bottom-right corner carries value 4.0
    assert raster.array[-1, -1] == 4.0


def test_fetch_3dep_dem_nodata_raises_coverage_error():
    def handler(call):
        w, h = (int(x) for x in call["params"]["size"].split(","))
        array = np.full((h, w), -1e35, dtype=np.float32)
        return FakeResponse(_dem_tiff_bytes(array))

    session = FakeSession(handler)
    with pytest.raises(src.SourceCoverageError, match="nodata"):
        src.fetch_3dep_dem(session, DEM_BBOX)


def test_fetch_3dep_dem_tagged_nodata_raises():
    def handler(call):
        w, h = (int(x) for x in call["params"]["size"].split(","))
        array = np.full((h, w), -9999.0, dtype=np.float32)
        return FakeResponse(_dem_tiff_bytes(array, nodata="-9999"))

    session = FakeSession(handler)
    with pytest.raises(src.SourceCoverageError):
        src.fetch_3dep_dem(session, DEM_BBOX)


def test_fetch_3dep_dem_http_error_wraps_as_fetch_error():
    def handler(call):
        return FakeResponse(b"", status_code=500)

    session = FakeSession(handler)
    with pytest.raises(src.SourceFetchError, match="USGS 3DEP DEM"):
        src.fetch_3dep_dem(session, DEM_BBOX)


def test_fetch_3dep_dem_rejects_degenerate_bbox():
    session = FakeSession(lambda call: FakeResponse(b""))
    with pytest.raises(ValueError, match="positive area"):
        src.fetch_3dep_dem(session, (-104.995, 40.0, -105.0, 40.004))


def test_fetch_3dep_dem_default_metres_per_px_is_unchanged():
    """Every existing caller omits ``metres_per_px`` — the default must still
    request the same ~10 m/px grid as before the keyword was added."""

    def handler(call):
        w, h = (int(x) for x in call["params"]["size"].split(","))
        return FakeResponse(_dem_tiff_bytes(np.full((h, w), 1500.0, dtype=np.float32)))

    default_session = FakeSession(handler)
    src.fetch_3dep_dem(default_session, DEM_BBOX)
    default_size = default_session.calls[0]["params"]["size"]

    explicit_session = FakeSession(handler)
    src.fetch_3dep_dem(explicit_session, DEM_BBOX, metres_per_px=10.0)
    explicit_size = explicit_session.calls[0]["params"]["size"]

    assert default_size == explicit_size


def test_fetch_3dep_dem_honours_metres_per_px():
    """A coarser ``metres_per_px`` requests a proportionally smaller pixel grid."""

    def handler(call):
        w, h = (int(x) for x in call["params"]["size"].split(","))
        return FakeResponse(_dem_tiff_bytes(np.full((h, w), 1500.0, dtype=np.float32)))

    fine_session = FakeSession(handler)
    src.fetch_3dep_dem(fine_session, DEM_BBOX, metres_per_px=10.0)
    fine_w, fine_h = (int(x) for x in fine_session.calls[0]["params"]["size"].split(","))

    coarse_session = FakeSession(handler)
    src.fetch_3dep_dem(coarse_session, DEM_BBOX, metres_per_px=30.0)
    coarse_w, coarse_h = (int(x) for x in coarse_session.calls[0]["params"]["size"].split(","))

    assert coarse_w < fine_w
    assert coarse_h < fine_h
    assert fine_w / coarse_w == pytest.approx(3.0, rel=0.2)


def test_fetch_3dep_dem_default_tiles_stay_within_what_3dep_serves():
    """3DEP advertises maxImageWidth/Height 8000 but answers HTTP 500 for a
    regional bbox whose export exceeds the per-request limit while a half-size
    one returns 200. The default grid must split that request into tiles no
    larger than 2048 px a side."""
    regional = (-85.0, 35.0, -84.0, 36.0)

    def handler(call):
        w, h = (int(x) for x in call["params"]["size"].split(","))
        return FakeResponse(_dem_tiff_bytes(np.full((h, w), 1500.0, dtype=np.float32)))

    session = FakeSession(handler)
    raster = src.fetch_3dep_dem(session, regional, metres_per_px=30.0)

    sizes = [tuple(int(x) for x in c["params"]["size"].split(",")) for c in session.calls]
    assert len(sizes) > 1
    assert max(max(w, h) for w, h in sizes) <= src.DEM_3DEP_MAX_TILE_PX == 2048
    assert raster.array.size == sum(w * h for w, h in sizes)


# --- fetch_naip_imagery ---------------------------------------------------


def test_fetch_naip_imagery_success():
    def handler(call):
        assert call["params"]["format"] == "jpgpng"
        w, h = (int(x) for x in call["params"]["size"].split(","))
        array = np.full((h, w, 3), 100, dtype=np.uint8)
        return FakeResponse(_image_bytes(array))

    session = FakeSession(handler)
    raster = src.fetch_naip_imagery(session, DEM_BBOX)

    assert raster.array.dtype == np.uint8
    assert raster.array.shape[2] == 3
    assert raster.provenance.licence == "Public domain (USDA NAIP via USGS)"


def test_fetch_naip_imagery_blank_raises_coverage_error():
    def handler(call):
        w, h = (int(x) for x in call["params"]["size"].split(","))
        array = np.full((h, w, 3), 255, dtype=np.uint8)
        return FakeResponse(_image_bytes(array))

    session = FakeSession(handler)
    with pytest.raises(src.SourceCoverageError, match="blank"):
        src.fetch_naip_imagery(session, DEM_BBOX)


def test_fetch_naip_imagery_default_tiles_stay_within_what_naip_serves():
    """NAIP advertises maxImageWidth/Height 4000 (the old 4096 cap got a
    200-with-JSON "exceeds the size limit" error above it), and a 4000x4000
    export took 56 s -- past the 30 s client timeout -- while 2048x2048
    returned in ~3 s."""
    wide = (-85.0, 35.0, -84.6, 35.6)

    def handler(call):
        w, h = (int(x) for x in call["params"]["size"].split(","))
        return FakeResponse(_image_bytes(np.full((h, w, 3), 100, dtype=np.uint8)))

    session = FakeSession(handler)
    raster = src.fetch_naip_imagery(session, wide, metres_per_px=10.0)

    sizes = [tuple(int(x) for x in c["params"]["size"].split(",")) for c in session.calls]
    assert len(sizes) > 1
    assert max(max(w, h) for w, h in sizes) <= src.NAIP_MAX_TILE_PX == 2048
    assert raster.array.shape[0] * raster.array.shape[1] == sum(w * h for w, h in sizes)


# --- GIBS -----------------------------------------------------------------


def test_fetch_gibs_truecolor_uses_1_3_0_axis_order_and_time():
    def handler(call):
        assert call["params"]["CRS"] == "EPSG:4326"
        west, south, east, north = DEM_BBOX
        assert call["params"]["BBOX"] == f"{south},{west},{north},{east}"
        assert call["params"]["TIME"] == "2026-06-01"
        assert "User-Agent" not in (call["headers"] or {})
        array = np.full((10, 12, 3), 30, dtype=np.uint8)
        return FakeResponse(_image_bytes(array, fmt="JPEG"))

    session = FakeSession(handler)
    raster = src.fetch_gibs_truecolor(session, DEM_BBOX, date(2026, 6, 1), width=12, height=10)

    assert raster.provenance.acquired == "2026-06-01"
    assert raster.provenance.attribution == "NASA EOSDIS GIBS / Worldview"
    assert raster.array.shape == (10, 12, 3)


def test_fetch_gibs_blue_marble_has_no_time_param():
    def handler(call):
        assert "TIME" not in call["params"]
        array = np.full((5, 5, 3), 40, dtype=np.uint8)
        return FakeResponse(_image_bytes(array, fmt="JPEG"))

    session = FakeSession(handler)
    raster = src.fetch_gibs_blue_marble(session, DEM_BBOX, width=5, height=5)

    assert raster.provenance.acquired == "unknown"
    assert raster.provenance.licence == "NASA open data (attribution requested)"


def test_fetch_gibs_reference_features_requests_transparent_png():
    def handler(call):
        assert call["url"] == src._GIBS_URL
        assert call["params"]["FORMAT"] == "image/png"
        assert call["params"]["TRANSPARENT"] == "TRUE"
        assert call["params"]["LAYERS"] == "Reference_Features_15m"
        array = np.zeros((5, 6, 4), dtype=np.uint8)
        array[..., 3] = 128
        buf = io.BytesIO()
        Image.fromarray(array, mode="RGBA").save(buf, format="PNG")
        return FakeResponse(buf.getvalue())

    session = FakeSession(handler)
    raster = src.fetch_gibs_reference_features(session, DEM_BBOX, width=6, height=5)

    assert raster.array.shape == (5, 6, 4)
    assert raster.array.dtype == np.uint8
    assert raster.provenance.name == "NASA GIBS Reference_Features_15m"
    assert raster.provenance.url == src._GIBS_URL
    assert raster.provenance.licence == "unverified"
    assert raster.provenance.attribution == "NASA EOSDIS GIBS / Worldview"


def test_fetch_gibs_reference_features_shares_bbox_and_crs_semantics():
    """Reference-features params differ from the base GIBS fetch only by FORMAT/TRANSPARENT."""
    captured = {}

    def handler(call):
        captured.update(call["params"])
        array = np.zeros((5, 5, 4), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(array, mode="RGBA").save(buf, format="PNG")
        return FakeResponse(buf.getvalue())

    session = FakeSession(handler)
    src.fetch_gibs_reference_features(session, DEM_BBOX, width=5, height=5)

    west, south, east, north = DEM_BBOX
    assert captured["SERVICE"] == "WMS"
    assert captured["VERSION"] == "1.3.0"
    assert captured["REQUEST"] == "GetMap"
    assert captured["CRS"] == "EPSG:4326"
    assert captured["BBOX"] == f"{south},{west},{north},{east}"


def test_fetch_gibs_truecolor_params_unchanged():
    """Adding fetch_gibs_reference_features must not alter ``_fetch_gibs``'s own request shape."""
    west, south, east, north = DEM_BBOX

    def handler(call):
        assert call["params"] == {
            "SERVICE": "WMS",
            "VERSION": "1.3.0",
            "REQUEST": "GetMap",
            "LAYERS": "VIIRS_SNPP_CorrectedReflectance_TrueColor",
            "CRS": "EPSG:4326",
            "BBOX": f"{south},{west},{north},{east}",
            "WIDTH": 12,
            "HEIGHT": 10,
            "FORMAT": "image/jpeg",
            "TIME": "2026-06-01",
        }
        array = np.full((10, 12, 3), 30, dtype=np.uint8)
        return FakeResponse(_image_bytes(array, fmt="JPEG"))

    session = FakeSession(handler)
    src.fetch_gibs_truecolor(session, DEM_BBOX, date(2026, 6, 1), width=12, height=10)


# --- cloud_fraction / choose_opening_scene --------------------------------


def test_cloud_fraction_all_cloud():
    rgb = np.full((4, 4, 3), 255, dtype=np.uint8)
    assert src.cloud_fraction(rgb) == 1.0


def test_cloud_fraction_no_cloud():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 120
    rgb[..., 2] = 5
    assert src.cloud_fraction(rgb) == 0.0


def _raster(array, provenance_name="x"):
    provenance = src.SourceProvenance(
        name=provenance_name, url="u", licence="l", attribution="a", acquired="unknown"
    )
    return src.GeoRaster(array=array, west=0, south=0, east=1, north=1, provenance=provenance)


def test_choose_opening_scene_picks_daily_when_clear():
    daily = _raster(np.zeros((4, 4, 3), dtype=np.uint8), "daily")
    blue_marble = _raster(np.full((4, 4, 3), 255, dtype=np.uint8), "bm")

    chosen, decision = src.choose_opening_scene(daily, blue_marble)

    assert chosen is daily
    assert decision == {"chosen": "daily", "cloud_fraction": 0.0, "threshold": 0.35}


def test_choose_opening_scene_falls_back_when_cloudy():
    daily = _raster(np.full((4, 4, 3), 255, dtype=np.uint8), "daily")
    blue_marble = _raster(np.zeros((4, 4, 3), dtype=np.uint8), "bm")

    chosen, decision = src.choose_opening_scene(daily, blue_marble)

    assert chosen is blue_marble
    assert decision["chosen"] == "blue_marble"
    assert decision["cloud_fraction"] == 1.0


# --- gibs_scene_for ---------------------------------------------------------


def test_gibs_scene_for_picks_daily_scene_when_clear():
    calls = []

    def handler(call):
        calls.append(call["params"].get("TIME"))
        clear = np.zeros((5, 5, 3), dtype=np.uint8)
        return FakeResponse(_image_bytes(clear, fmt="JPEG"))

    session = FakeSession(handler)

    chosen, decision = src.gibs_scene_for(session, DEM_BBOX, date(2026, 6, 1), width=5, height=5)

    assert decision["chosen"] == "daily"
    assert chosen.provenance.acquired == "2026-06-01"
    # one truecolor (dated) fetch, one blue marble (dateless) fetch
    assert calls == ["2026-06-01", None]


def test_gibs_scene_for_falls_back_to_blue_marble_when_cloudy():
    def handler(call):
        if call["params"].get("TIME") == "2026-06-01":
            cloudy = np.full((5, 5, 3), 255, dtype=np.uint8)
            return FakeResponse(_image_bytes(cloudy, fmt="JPEG"))
        clear = np.zeros((5, 5, 3), dtype=np.uint8)
        return FakeResponse(_image_bytes(clear, fmt="JPEG"))

    session = FakeSession(handler)

    chosen, decision = src.gibs_scene_for(session, DEM_BBOX, date(2026, 6, 1), width=5, height=5)

    assert decision["chosen"] == "blue_marble"
    assert chosen.provenance.acquired == "unknown"


# --- fetch_osm_labels -------------------------------------------------------

OSM_PAYLOAD = {
    "elements": [
        {
            "type": "node",
            "id": 1,
            "lat": 40.1,
            "lon": -105.1,
            "tags": {"natural": "peak", "name": "Mount Alpha", "ele": "4200"},
        },
        {
            "type": "way",
            "id": 2,
            "center": {"lat": 40.2, "lon": -105.2},
            "tags": {"waterway": "river", "name": "Beta River"},
        },
        {
            "type": "node",
            "id": 3,
            "lat": 40.3,
            "lon": -105.3,
            "tags": {"mountain_pass": "yes", "name": "Gamma Pass"},
        },
        {
            "type": "node",
            "id": 4,
            "lat": 40.4,
            "lon": -105.4,
            "tags": {"place": "town", "name": "Deltaville"},
        },
        # no name -> excluded
        {"type": "node", "id": 5, "lat": 40.5, "lon": -105.5, "tags": {"natural": "peak"}},
        # not a matching tag -> excluded
        {"type": "node", "id": 6, "lat": 40.6, "lon": -105.6, "tags": {"name": "Nowhere"}},
        # way missing a center -> excluded
        {"type": "way", "id": 7, "tags": {"waterway": "stream", "name": "No Center Creek"}},
    ]
}


def test_fetch_osm_labels_filters_sorts_and_tags_provenance():
    def handler(call):
        assert call["headers"]["User-Agent"] == src._OSM_USER_AGENT
        assert "natural" in call["params"]["data"]
        return FakeResponse(json.dumps(OSM_PAYLOAD).encode())

    session = FakeSession(handler)
    result = src.fetch_osm_labels(session, DEM_BBOX)

    # sorted by (kind, name, osm_id): mountain_pass < peak < river < town
    names = [f["properties"]["name"] for f in result["features"]]
    assert names == ["Gamma Pass", "Mount Alpha", "Beta River", "Deltaville"]
    alpha = result["features"][1]
    assert alpha["properties"]["ele"] == "4200"
    assert alpha["geometry"]["coordinates"] == [-105.1, 40.1]
    beta = result["features"][2]
    assert beta["geometry"]["coordinates"] == [-105.2, 40.2]
    assert result["provenance"].licence == "ODbL 1.0"
    assert result["provenance"].attribution == "© OpenStreetMap contributors"


def test_osm_query_declares_the_extended_server_budget():
    query = src._osm_query(DEM_BBOX)
    assert "[timeout:60]" in query
    assert f"[timeout:{src.OSM_SERVER_BUDGET_S}]" in query


def test_fetch_osm_labels_uses_the_extended_client_timeout():
    def handler(call):
        assert call["timeout"] == src.OSM_QUERY_TIMEOUT_S
        return FakeResponse(json.dumps({"elements": []}).encode())

    session = FakeSession(handler)
    src.fetch_osm_labels(session, DEM_BBOX)


def test_fetch_osm_labels_wraps_http_errors():
    def handler(call):
        return FakeResponse(b"", status_code=503)

    session = FakeSession(handler)
    with pytest.raises(src.SourceFetchError, match="Overpass"):
        src.fetch_osm_labels(session, DEM_BBOX)


# --- fetch_copernicus_dem ---------------------------------------------------


def test_fetch_copernicus_dem_single_cell():
    cell = np.arange(16, dtype=np.float32).reshape(4, 4)

    def handler(call):
        assert call["url"] == src._copernicus_cell_url(40, -105)
        return FakeResponse(_dem_tiff_bytes(cell))

    session = FakeSession(handler)
    bbox = (-105.0, 40.0, -104.5, 40.5)
    raster = src.fetch_copernicus_dem(session, bbox)

    assert len(session.calls) == 1
    assert raster.array.shape == (2, 2)
    assert raster.provenance.licence == "Copernicus DEM licence (free, attribution required)"


def test_fetch_copernicus_dem_multi_cell_stitch():
    def handler(call):
        if "N40" in call["url"] and "W105" in call["url"]:
            value = 1.0
        elif "N40" in call["url"] and "W104" in call["url"]:
            value = 2.0
        elif "N39" in call["url"] and "W105" in call["url"]:
            value = 3.0
        else:
            value = 4.0
        return FakeResponse(_dem_tiff_bytes(np.full((4, 4), value, dtype=np.float32)))

    session = FakeSession(handler)
    bbox = (-104.9, 39.5, -103.1, 40.5)
    raster = src.fetch_copernicus_dem(session, bbox)

    assert len(session.calls) == 4
    assert raster.array[0, 0] == 1.0  # north-west cell
    assert raster.array[-1, -1] == 4.0  # south-east cell


def test_fetch_copernicus_dem_missing_cell_raises_coverage_error():
    def handler(call):
        return FakeResponse(b"", status_code=404)

    session = FakeSession(handler)
    bbox = (-105.0, 40.0, -104.5, 40.5)
    with pytest.raises(src.SourceCoverageError, match="no coverage"):
        src.fetch_copernicus_dem(session, bbox)


def test_fetch_copernicus_dem_http_error_wraps():
    def handler(call):
        return FakeResponse(b"", status_code=500)

    session = FakeSession(handler)
    bbox = (-105.0, 40.0, -104.5, 40.5)
    with pytest.raises(src.SourceFetchError, match="Copernicus"):
        src.fetch_copernicus_dem(session, bbox)


def test_copernicus_cell_url_south_and_east_hemispheres():
    url = src._copernicus_cell_url(-10, 20)
    assert "S10_00" in url
    assert "E020_00" in url


class TestOverpassFailover:
    """The canonical Overpass host can be unreachable from some networks,
    so the labels fetch must try the service's mirror list, not one hard-coded host."""

    def test_falls_through_to_the_next_endpoint_when_one_is_unreachable(self):
        from gomoveshift.pipeline.chapters.overpass_source import OVERPASS_ENDPOINTS

        tried: list[str] = []

        class FailFirstSession:
            def get(self, url, *, params=None, timeout=None, headers=None):
                tried.append(url)
                if len(tried) == 1:
                    raise OSError("[Errno 101] Network is unreachable")
                return FakeResponse(json.dumps({"elements": []}).encode())

        result = src.fetch_osm_labels(FailFirstSession(), (-83.6, 35.2, -83.4, 35.3))

        assert result["features"] == []
        assert tried == list(OVERPASS_ENDPOINTS[:2])

    def test_an_explicit_endpoint_is_used_alone(self):
        tried: list[str] = []

        class RecordingSession:
            def get(self, url, *, params=None, timeout=None, headers=None):
                tried.append(url)
                return FakeResponse(json.dumps({"elements": []}).encode())

        src.fetch_osm_labels(
            RecordingSession(), (-83.6, 35.2, -83.4, 35.3), endpoint="https://mirror.example/api"
        )

        assert tried == ["https://mirror.example/api"]

    def test_every_endpoint_failing_names_them_all(self):
        class DeadSession:
            def get(self, url, *, params=None, timeout=None, headers=None):
                raise OSError("[Errno 101] Network is unreachable")

        with pytest.raises(src.SourceFetchError, match="every endpoint"):
            src.fetch_osm_labels(DeadSession(), (-83.6, 35.2, -83.4, 35.3))


@pytest.mark.parametrize(
    ("bbox", "metres_per_px", "max_px"),
    [
        # A focal and a regional terrain fetch, and a NAIP-scale one.
        ((-83.6, 35.2, -83.5, 35.3), 10.0, 2048),
        ((-85.0, 34.5, -82.0, 36.0), 30.0, 2048),
        ((-83.6, 35.2, -83.5, 35.3), 1.0, 2048),
    ],
)
def test_every_export_tile_has_the_same_shape_in_pixels_as_in_degrees(bbox, metres_per_px, max_px):
    """ArcGIS exportImage GROWS the returned extent until it matches the pixel aspect.

    Observed against 3DEP: bbox lat 35.174..35.282 at size 1200x1200
    came back as lat 35.162..35.294. With imageSR=4326 the raster is plate
    carree, so a tile whose pixel aspect differs from its degree aspect is
    returned over a different extent than the one the caller then labels it
    with -- the ground lands up to kilometres from where it really is.
    """
    tiles, _, _ = src._grid_plan(bbox, metres_per_px=metres_per_px, max_px=max_px)
    for row in tiles:
        for west, south, east, north, width, height in row:
            assert max(width, height) <= max_px
            degree_aspect = (east - west) / (north - south)
            # Integer pixel counts cannot match exactly; allow one pixel of slack.
            assert abs(width - height * degree_aspect) <= 1.0, (width, height, degree_aspect)


def test_export_resolution_is_never_coarser_than_asked():
    bbox = (-83.6, 35.2, -83.5, 35.3)
    _, total_w, total_h = src._grid_plan(bbox, metres_per_px=10.0, max_px=2048)
    lon_step, lat_step = src._pixel_size_degrees(bbox, 10.0)
    assert total_w >= round((bbox[2] - bbox[0]) / lon_step)
    assert total_h >= round((bbox[3] - bbox[1]) / lat_step)
