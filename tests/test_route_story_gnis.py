"""Red-first tests for the GNIS place-names source for Route Story labels."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from gomoveshift.pipeline.route_story import gnis


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


# A realistic 6-row sample of the pipe-delimited Domestic Names text file, header
# names as documented by the real GNIS Domestic Names export.
_GNIS_HEADER = (
    "feature_id|feature_name|feature_class|state_alpha|state_numeric|county_name|"
    "county_numeric|primary_lat_dms|primary_long_dms|prim_lat_dec|prim_long_dec|"
    "source_lat_dms|source_long_dms|source_lat_dec|source_long_dec|elev_in_m|"
    "elev_in_ft|map_name|date_created|date_edited"
)
_GNIS_ROWS = [
    # feature_id, feature_name, feature_class, state_alpha, state_numeric, county_name,
    # county_numeric, primary_lat_dms, primary_long_dms, prim_lat_dec, prim_long_dec,
    # source_lat_dms, source_long_dms, source_lat_dec, source_long_dec, elev_in_m,
    # elev_in_ft, map_name, date_created, date_edited
    "1000001|Clingmans Dome|Summit|NC|37|Swain|173|354338N|0833001W|35.7272|-83.5003|||||2025|6643|Clingmans Dome|01/01/1980|",
    "1000002|Newfound Gap|Gap|TN|47|Sevier|155|352643N|0832655W|35.4453|-83.4487|||||1539|5049|Mount Le Conte|01/01/1980|",
    "1000003|Oconaluftee River|Stream|NC|37|Swain|173|353159N|0832408W|35.5331|-83.4022|||||533||Bryson City|01/01/1980|",
    "1000004|Cherokee|Populated Place|NC|37|Swain|173|352947N|0831924W|35.4964|-83.3233|||||655|2149|Whittier|01/01/1980|",
    "1000005|Appalachian Trail|Trail|NC|37|Swain|173|354338N|0833001W|35.7272|-83.5003|||||||Clingmans Dome|01/01/1980|",
    "1000006|Sherman Elementary School|School|NC|37|Swain|173|353500N|0832800W|35.5833|-83.4667|||||600||Bryson City|01/01/1980|",
]


def _gnis_zip_bytes(rows: list[str] | None = None) -> bytes:
    text = "\n".join([_GNIS_HEADER, *(rows if rows is not None else _GNIS_ROWS)]) + "\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("NC_Text.txt", text)
    return buf.getvalue()


# --- gnis_state_url -------------------------------------------------------


def test_gnis_state_url_uses_signed_off_host():
    url = gnis.gnis_state_url("nc")
    assert url == (
        "https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/"
        "DomesticNames/DomesticNames_NC_Text.zip"
    )


# --- fetch_gnis_state -------------------------------------------------------


def test_fetch_gnis_state_downloads_and_caches(tmp_path: Path):
    zip_bytes = _gnis_zip_bytes()
    session = FakeSession(lambda call: FakeResponse(zip_bytes, headers={"ETag": '"abc123"'}))

    path = gnis.fetch_gnis_state(session, "NC", tmp_path)

    assert path.exists()
    assert path.read_bytes() == zip_bytes
    assert len(session.calls) == 1
    assert session.calls[0]["url"] == gnis.gnis_state_url("NC")


def test_fetch_gnis_state_second_call_uses_cache_without_a_request(tmp_path: Path):
    zip_bytes = _gnis_zip_bytes()
    session = FakeSession(lambda call: FakeResponse(zip_bytes, headers={"ETag": '"abc123"'}))

    first = gnis.fetch_gnis_state(session, "NC", tmp_path)
    second = gnis.fetch_gnis_state(session, "NC", tmp_path)

    assert second == first
    assert len(session.calls) == 1


def test_fetch_gnis_state_raises_on_http_error(tmp_path: Path):
    session = FakeSession(lambda call: FakeResponse(b"", status_code=500))

    with pytest.raises(gnis.GnisFetchError):
        gnis.fetch_gnis_state(session, "NC", tmp_path)


# --- states_for_bbox ---------------------------------------------------------


def test_states_for_bbox_blue_ridge_touches_nc_and_tn():
    bbox = (-83.0, 35.0, -82.0, 36.0)
    states = gnis.states_for_bbox(bbox)
    assert {"NC", "TN"} <= set(states)


def test_states_for_bbox_four_corners_touches_all_four_states():
    bbox = (-109.06, 36.98, -109.02, 37.02)
    states = gnis.states_for_bbox(bbox)
    assert {"UT", "CO", "AZ", "NM"} <= set(states)


# --- parse_gnis ---------------------------------------------------------------


def test_parse_gnis_filters_by_class_and_bbox(tmp_path: Path):
    zip_path = tmp_path / "DomesticNames_NC_Text.zip"
    zip_path.write_bytes(_gnis_zip_bytes())

    features = gnis.parse_gnis(zip_path, bbox=(-84.0, 35.0, -83.0, 36.0))

    names = {feature.name for feature in features}
    # The School row is not in DEFAULT_CLASSES and must be dropped.
    assert "Sherman Elementary School" not in names
    assert "Clingmans Dome" in names
    assert "Cherokee" in names


def test_parse_gnis_excludes_features_outside_bbox(tmp_path: Path):
    zip_path = tmp_path / "DomesticNames_NC_Text.zip"
    zip_path.write_bytes(_gnis_zip_bytes())

    # Newfound Gap is at -83.4487, outside this narrow eastward box.
    features = gnis.parse_gnis(zip_path, bbox=(-83.3, 35.0, -83.0, 36.0))

    names = {feature.name for feature in features}
    assert "Newfound Gap" not in names


def test_parse_gnis_reads_elevation_and_feature_id(tmp_path: Path):
    zip_path = tmp_path / "DomesticNames_NC_Text.zip"
    zip_path.write_bytes(_gnis_zip_bytes())

    features = gnis.parse_gnis(zip_path, bbox=(-84.0, 35.0, -83.0, 36.0))
    dome = next(f for f in features if f.name == "Clingmans Dome")

    assert dome.feature_id == "1000001"
    assert dome.feature_class == "Summit"
    assert dome.state == "NC"
    assert dome.elevation_m == pytest.approx(2025.0)

    trail = next(f for f in features if f.name == "Appalachian Trail")
    assert trail.elevation_m is None


# --- gnis_features_for_bbox ---------------------------------------------------


def test_gnis_features_for_bbox_dedupes_and_sorts(tmp_path: Path):
    zip_bytes = _gnis_zip_bytes()
    session = FakeSession(lambda call: FakeResponse(zip_bytes, headers={"ETag": '"abc123"'}))
    bbox = (-84.0, 35.0, -83.0, 36.0)

    features = gnis.gnis_features_for_bbox(session, bbox, tmp_path)

    ids = [f.feature_id for f in features]
    assert len(ids) == len(set(ids))
    classes_and_names = [(f.feature_class, f.name) for f in features]
    assert classes_and_names == sorted(classes_and_names)


# --- to_label_geojson ---------------------------------------------------------


def test_to_label_geojson_matches_osm_label_shape():
    features = [
        gnis.GnisFeature(
            name="Clingmans Dome",
            feature_class="Summit",
            lat=35.7272,
            lon=-83.5003,
            elevation_m=2025.0,
            state="NC",
            feature_id="1000001",
        ),
        gnis.GnisFeature(
            name="Cherokee",
            feature_class="Populated Place",
            lat=35.4964,
            lon=-83.3233,
            elevation_m=None,
            state="NC",
            feature_id="1000004",
        ),
    ]

    collection = gnis.to_label_geojson(features)

    assert collection["type"] == "FeatureCollection"
    summit = collection["features"][0]
    assert summit["type"] == "Feature"
    assert summit["geometry"] == {"type": "Point", "coordinates": [-83.5003, 35.7272]}
    assert summit["properties"] == {
        "name": "Clingmans Dome",
        "kind": "summit",
        "elevation_m": 2025.0,
        "source": "gnis",
        "feature_id": "1000001",
    }
    place = collection["features"][1]
    assert place["properties"]["kind"] == "place"


def test_gnis_host_is_registered_in_source_hosts():
    from gomoveshift.pipeline.route_story.sources import SOURCE_HOSTS

    assert gnis.GNIS_HOST in SOURCE_HOSTS


# --- the real staged-products file shape ------------------------------------

# Byte-for-byte shape of DomesticNames_NC.txt as published by the
# staged-products service: a UTF-8 byte-order mark, CRLF line ends,
# ``state_name`` instead of ``state_alpha``, and no elevation columns at all.
_REAL_GNIS_HEADER = (
    "feature_id|feature_name|feature_class|state_name|state_numeric|county_name|"
    "county_numeric|map_name|date_created|date_edited|bgn_type|bgn_authority|bgn_date|"
    "prim_lat_dms|prim_long_dms|prim_lat_dec|prim_long_dec|source_lat_dms|"
    "source_long_dms|source_lat_dec|source_long_dec"
)
_REAL_GNIS_ROWS = [
    "1017791|Burningtown Gap|Gap|North Carolina|37|Macon|113|Wesser|09/25/1979|||||"
    "351312N|0833411W|35.2201|-83.5697|||0.0|0.0",
    "1018250|Copper Ridge Bald|Summit|North Carolina|37|Macon|113|Wesser|09/25/1979|||||"
    "351418N|0833344W|35.2384|-83.5622|||0.0|0.0",
    "1019090|Tellico Bald|Summit|North Carolina|37|Macon|113|Hewitt|09/25/1979|||||"
    "351534N|0833326W|35.2594|-83.5572|||0.0|0.0",
]


def _real_gnis_zip_bytes() -> bytes:
    text = "\r\n".join([_REAL_GNIS_HEADER, *_REAL_GNIS_ROWS]) + "\r\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("Text/DomesticNames_NC.txt", "﻿" + text)
    return buf.getvalue()


def test_parse_gnis_reads_the_real_file_header_with_a_byte_order_mark(tmp_path: Path):
    zip_path = tmp_path / "DomesticNames_NC_Text.zip"
    zip_path.write_bytes(_real_gnis_zip_bytes())

    features = gnis.parse_gnis(zip_path, bbox=(-84.0, 35.0, -83.0, 36.0))

    by_name = {feature.name: feature for feature in features}
    assert by_name["Burningtown Gap"].feature_id == "1017791"
    assert by_name["Burningtown Gap"].state == "North Carolina"
    assert by_name["Tellico Bald"].elevation_m is None


def test_gnis_features_for_bbox_keeps_every_feature_of_the_real_file(tmp_path: Path):
    zip_bytes = _real_gnis_zip_bytes()
    session = FakeSession(lambda call: FakeResponse(zip_bytes, headers={"ETag": '"real"'}))

    features = gnis.gnis_features_for_bbox(session, (-84.0, 35.0, -83.0, 36.0), tmp_path)

    assert sorted(f.name for f in features) == [
        "Burningtown Gap",
        "Copper Ridge Bald",
        "Tellico Bald",
    ]


def test_gnis_features_without_an_id_are_not_collapsed_into_one(tmp_path: Path):
    rows = [
        "|Flint Rock Gap|Gap|NC|37|Macon|113|351400N|0833400W|35.2333|-83.5667|||||||x||",
        "|Otter Mountain|Summit|NC|37|Macon|113|351500N|0833500W|35.2500|-83.5833|||||||x||",
    ]
    zip_bytes = _gnis_zip_bytes(rows)
    session = FakeSession(lambda call: FakeResponse(zip_bytes, headers={"ETag": '"noid"'}))

    features = gnis.gnis_features_for_bbox(session, (-84.0, 35.0, -83.0, 36.0), tmp_path)

    assert sorted(f.name for f in features) == ["Flint Rock Gap", "Otter Mountain"]
