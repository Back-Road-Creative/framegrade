"""The Route Story bundle's offline OSM road layer (``roads.geojson``)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from gomoveshift.pipeline.route_story import bundle, prepare
from gomoveshift.pipeline.route_story import sources as src
from gomoveshift.pipeline.route_story.bundle import verify_bundle
from tests.test_route_story_prepare import (  # noqa: F401 - fixtures re-exported for pytest
    RECORDED_ON,
    FakeSession,
    _cache_dir,
    _fake_fetch_maplibre,
    _PreparedRun,
    _prepare,
    _private_copy,
    _run_once,
)


@pytest.fixture(autouse=True)
def _neuter_sri(monkeypatch: pytest.MonkeyPatch) -> None:
    from gomoveshift.pipeline.route_story.scene import vendor

    monkeypatch.setattr(vendor, "_verify_sri", lambda data, sri: None)


BBOX = (-105.0, 40.0, -104.995, 40.004)


class _Response:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, handler):
        self.handler = handler
        self.calls: list[dict] = []

    def get(self, url, *, params=None, timeout=None, headers=None):
        call = {"url": url, "params": params, "timeout": timeout, "headers": headers}
        self.calls.append(call)
        return self.handler(call)


def _way(osm_id, highway, points, name=None):
    tags = {"highway": highway}
    if name:
        tags["name"] = name
    return {
        "type": "way",
        "id": osm_id,
        "tags": tags,
        "geometry": [{"lat": lat, "lon": lon} for lon, lat in points],
    }


PAYLOAD = {
    "elements": [
        _way(30, "secondary", [(-105.0, 40.0), (-104.99, 40.01)], name="Second Street"),
        _way(10, "residential", [(-105.1, 40.1), (-105.2, 40.2), (-105.3, 40.3)]),
        _way(20, "track", [(-105.4, 40.4), (-105.5, 40.5)]),  # class not kept: dropped
        _way(25, "tertiary", [(-105.6, 40.6)]),  # one point: dropped
        {"type": "way", "id": 26, "tags": {"highway": "primary"}},  # no geometry: dropped
    ]
}


# --- fetch_osm_roads --------------------------------------------------------


def test_roads_query_filters_the_highway_classes_in_one_constant():
    query = src._osm_roads_query(BBOX)
    pattern = "|".join(src.OSM_ROAD_CLASSES)
    assert f'way[highway~"^({pattern})$"](40.0,-105.0,40.004,-104.995);' in query
    assert query.rstrip().endswith("out geom;")
    assert f"[timeout:{src.OSM_SERVER_BUDGET_S}]" in query
    classes = set(src.OSM_ROAD_CLASSES)
    trunk_to_tertiary = ["motorway", "trunk", "primary", "secondary", "tertiary"]
    expected = set(trunk_to_tertiary) | {"unclassified", "residential"}
    expected |= {f"{c}_link" for c in trunk_to_tertiary}  # OSM has no residential_link
    assert classes == expected
    for banned in ("service", "track", "path", "footway", "cycleway"):
        assert banned not in classes


def test_fetch_osm_roads_builds_sorted_linestrings_and_drops_thin_ways():
    def handler(call):
        assert call["headers"]["User-Agent"] == src._OSM_USER_AGENT
        assert call["timeout"] == src.OSM_QUERY_TIMEOUT_S
        return _Response(json.dumps(PAYLOAD).encode())

    result = src.fetch_osm_roads(_Session(handler), BBOX)

    assert result["type"] == "FeatureCollection"
    assert [f["properties"]["osm_id"] for f in result["features"]] == [10, 30]
    first, second = result["features"]
    assert first == {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[-105.1, 40.1], [-105.2, 40.2], [-105.3, 40.3]],
        },
        "properties": {"highway": "residential", "osm_id": 10},
    }
    assert "name" not in first["properties"]
    assert second["properties"] == {"highway": "secondary", "osm_id": 30, "name": "Second Street"}
    assert result["provenance"].licence == "ODbL 1.0"
    assert result["provenance"].attribution == "© OpenStreetMap contributors"


def test_fetch_osm_roads_falls_back_across_endpoints():
    seen: list[str] = []

    def handler(call):
        seen.append(call["url"])
        if len(seen) == 1:
            return _Response(b"", status_code=503)
        return _Response(json.dumps(PAYLOAD).encode())

    result = src.fetch_osm_roads(_Session(handler), BBOX)
    assert seen == list(src.OVERPASS_ENDPOINTS[:2])
    assert len(result["features"]) == 2


def test_fetch_osm_roads_raises_when_every_endpoint_fails():
    session = _Session(lambda call: _Response(b"", status_code=503))
    with pytest.raises(src.SourceFetchError, match="every endpoint"):
        src.fetch_osm_roads(session, BBOX)
    assert len(session.calls) == len(src.OVERPASS_ENDPOINTS)


def test_fetch_osm_roads_explicit_endpoint_is_used_alone():
    session = _Session(lambda call: _Response(b"", status_code=503))
    with pytest.raises(src.SourceFetchError):
        src.fetch_osm_roads(session, BBOX, endpoint="https://example.test/api")
    assert [c["url"] for c in session.calls] == ["https://example.test/api"]


def test_fetch_osm_roads_rejects_a_zero_area_bbox():
    with pytest.raises(ValueError):
        src.fetch_osm_roads(_Session(lambda call: _Response(b"{}")), (1.0, 1.0, 1.0, 2.0))


# --- verify_bundle schema handling -----------------------------------------


def _bundle(root: Path, *, schema: str, roads: bool, roads_key: bool) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "route.geojson").write_text('{"type": "FeatureCollection", "features": []}\n')
    fields: dict = {}
    if roads:
        (root / "roads.geojson").write_text('{"type": "FeatureCollection", "features": []}\n')
    if roads_key:
        fields["roads"] = {
            "bbox": [0, 0, 1, 1],
            "count": 0,
            "degraded": True,
            "problems": ["x"],
            "provenance": "p",
        }
    bundle.write_manifest(root, fields)
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["schema"] = schema
    bundle.rewrite_manifest(root, manifest)


def test_the_accepted_schemas_are_exported_next_to_the_current_one():
    assert bundle.MANIFEST_SCHEMA == "route-story-bundle/2"
    assert bundle.MANIFEST_SCHEMAS == ("route-story-bundle/1", "route-story-bundle/2")


def test_verify_accepts_a_legacy_v1_bundle_without_roads(tmp_path: Path):
    _bundle(tmp_path, schema="route-story-bundle/1", roads=False, roads_key=False)
    assert verify_bundle(tmp_path) == []


def test_verify_accepts_a_v2_bundle_with_roads(tmp_path: Path):
    _bundle(tmp_path, schema="route-story-bundle/2", roads=True, roads_key=True)
    assert verify_bundle(tmp_path) == []


def test_verify_flags_a_v2_bundle_with_no_roads_file(tmp_path: Path):
    _bundle(tmp_path, schema="route-story-bundle/2", roads=False, roads_key=True)
    assert any("roads.geojson" in p for p in verify_bundle(tmp_path))


def test_verify_flags_a_v2_bundle_with_no_roads_manifest_key(tmp_path: Path):
    _bundle(tmp_path, schema="route-story-bundle/2", roads=True, roads_key=False)
    assert any("roads" in p and "manifest" in p for p in verify_bundle(tmp_path))


def test_verify_still_rejects_an_unknown_schema(tmp_path: Path):
    _bundle(tmp_path, schema="route-story-bundle/9", roads=True, roads_key=True)
    assert any("unexpected schema" in p for p in verify_bundle(tmp_path))


# --- prepare ---------------------------------------------------------------


def _roads(run_or_dir: Path) -> dict:
    return json.loads((run_or_dir / "roads.geojson").read_text())


def _manifest(bundle_dir: Path) -> dict:
    return json.loads((bundle_dir / "manifest.json").read_text())


@pytest.fixture(scope="module")
def _roads_run(tmp_path_factory: pytest.TempPathFactory) -> _PreparedRun:
    return _run_once(tmp_path_factory, "roads-ok", FakeSession())


@pytest.fixture(scope="module")
def _roads_failed_run(tmp_path_factory: pytest.TempPathFactory) -> _PreparedRun:
    return _run_once(tmp_path_factory, "roads-failed", FakeSession(roads_raises=True))


@pytest.fixture(scope="module")
def _roads_empty_run(tmp_path_factory: pytest.TempPathFactory) -> _PreparedRun:
    return _run_once(tmp_path_factory, "roads-empty", FakeSession(roads_elements=[]))


def test_prepare_writes_roads_and_the_manifest_roads_block(_roads_run: _PreparedRun):
    bundle_dir = _roads_run.report.bundle_dir
    roads = _roads(bundle_dir)
    assert [f["properties"]["osm_id"] for f in roads["features"]] == [10]
    manifest = _manifest(bundle_dir)
    assert manifest["schema"] == "route-story-bundle/2"
    block = manifest["roads"]
    assert block["count"] == 1
    assert block["degraded"] is False
    assert block["problems"] == []
    assert "OpenStreetMap" in block["provenance"] and "ODbL" in block["provenance"]
    assert len(block["bbox"]) == 4
    assert "roads.geojson" in manifest["files"]
    assert _roads_run.report.verify_problems == []


def test_prepare_fetches_roads_over_the_route_bbox_not_the_regional_one(_roads_run: _PreparedRun):
    queries = [
        c["params"]["data"]
        for c in _roads_run.session.calls
        if "overpass" in c["url"] and "way[highway" in c["params"]["data"]
    ]
    assert len(queries) == 1
    south, west, north, east = (
        float(v)
        for v in re.search(r"\(([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+)\);", queries[0]).groups()
    )
    manifest = _manifest(_roads_run.report.bundle_dir)
    assert manifest["roads"]["bbox"] == pytest.approx([west, south, east, north])
    footprint = manifest["footprint"]
    assert west == pytest.approx(footprint["west"])
    assert north == pytest.approx(footprint["north"])
    label_west = manifest["labels"]["bbox"][0]
    assert west > label_west  # strictly tighter than the regional label bbox


def test_a_failed_roads_fetch_degrades_but_prepare_succeeds(_roads_failed_run: _PreparedRun):
    bundle_dir = _roads_failed_run.report.bundle_dir
    assert _roads(bundle_dir) == {"type": "FeatureCollection", "features": []}
    block = _manifest(bundle_dir)["roads"]
    assert block["degraded"] is True and block["count"] == 0
    assert block["problems"] and "roads" in block["problems"][0]
    assert _roads_failed_run.report.verify_problems == []


def test_an_empty_roads_fetch_is_degraded_never_silently_fine(_roads_empty_run: _PreparedRun):
    bundle_dir = _roads_empty_run.report.bundle_dir
    assert _roads(bundle_dir) == {"type": "FeatureCollection", "features": []}
    block = _manifest(bundle_dir)["roads"]
    assert block["degraded"] is True
    assert block["problems"] == ["roads: no highway ways in bbox"]


def test_the_roads_recipe_is_part_of_the_bundle_cache_key():
    versions = prepare._band_source_versions(
        gibs_chosen="blue_marble",
        recorded_on=RECORDED_ON,
        terrain_focal_name="a",
        terrain_regional_name="b",
        regional_margin_km=1.0,
    )
    assert versions["roads"] == prepare.ROADS_SOURCE_VERSION


def test_a_degraded_cached_bundle_refetches_only_its_roads(
    tmp_path: Path,
    _roads_failed_run: _PreparedRun,
):
    first, gpx_path = _private_copy(_roads_failed_run, tmp_path)
    session = FakeSession()
    second = _prepare(gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)

    assert second.bundle_dir == first.bundle_dir
    urls = " ".join(c["url"] for c in session.calls)
    assert "NAIPImagery" not in urls and "3DEPElevation" not in urls
    road_queries = [c for c in session.calls if "way[highway" in c["params"]["data"]]
    assert len(road_queries) == 1
    assert [f["properties"]["osm_id"] for f in _roads(first.bundle_dir)["features"]] == [10]
    block = _manifest(first.bundle_dir)["roads"]
    assert block["degraded"] is False and block["count"] == 1 and block["problems"] == []
    assert verify_bundle(first.bundle_dir) == []


def test_a_repair_that_is_still_degraded_leaves_the_bundle_untouched(
    tmp_path: Path,
    _roads_failed_run: _PreparedRun,
):
    first, gpx_path = _private_copy(_roads_failed_run, tmp_path)
    before = (first.bundle_dir / "manifest.json").read_bytes()
    _prepare(
        gpx_path, FakeSession(roads_raises=True), _fake_fetch_maplibre, recorded_on=RECORDED_ON
    )
    assert (first.bundle_dir / "manifest.json").read_bytes() == before


def test_an_undegraded_cached_bundle_fetches_no_roads(
    tmp_path: Path,
    _roads_run: _PreparedRun,
):
    _first, gpx_path = _private_copy(_roads_run, tmp_path)
    session = FakeSession()
    _prepare(gpx_path, session, _fake_fetch_maplibre, recorded_on=RECORDED_ON)
    assert [c for c in session.calls if "way[highway" in c["params"]["data"]] == []
