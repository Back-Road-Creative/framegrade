"""Retained-route boundary proof uses pinned polygons and bounded uncertainty."""

from __future__ import annotations

import hashlib
import io
import struct
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from gomoveshift.pipeline.boundary_proof import (
    _SOURCES,
    BoundaryArchiveError,
    CensusBoundaryProver,
    prove_windows_inside,
)
from gomoveshift.pipeline.route_claims import (
    RouteClaimRefusedError,
    RouteWindow,
    _region_ok,
    retained_windows_key,
    measure_route,
)
from gomoveshift.pipeline.route_claims import enforce_route_claims


def test_boundary_geometry_library_is_a_direct_runtime_dependency():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert any(spec.startswith("shapely>=") for spec in project["project"]["dependencies"])


def _window(lat: float, *, gap: float = 2.0) -> RouteWindow:
    return RouteWindow("lf1", 0.0, gap, [(0.0, lat, -80.0), (gap, lat, -80.0)])


def test_gap_deep_inside_polygon_is_bounded_without_interpolating_a_path():
    polygon = Polygon([(-81, 30), (-79, 30), (-79, 32), (-81, 32)])
    proof = prove_windows_inside([_window(31.0, gap=30)], polygon, max_speed_m_s=70)
    assert proof["verified"] is True
    assert proof["max_gap_s"] == 30
    assert proof["gps_error_m"] > 0


def test_no_retained_windows_cannot_prove_a_region():
    polygon = Polygon([(-81, 30), (-79, 30), (-79, 32), (-81, 32)])
    proof = prove_windows_inside([], polygon)
    assert proof == {"verified": False, "reason": "no retained windows to prove"}


@pytest.mark.parametrize("start,end", [(10.0, 10.0), (10.0, 9.0)])
def test_invalid_retained_window_span_cannot_prove_a_region(start, end):
    polygon = Polygon([(-81, 30), (-79, 30), (-79, 32), (-81, 32)])
    window = RouteWindow(
        "lf1", start, end, [], prev_fix=(8.0, 31.0, -80.0), next_fix=(11.0, 31.0, -80.0)
    )
    proof = prove_windows_inside([window], polygon)
    assert proof == {"verified": False, "reason": "lf1: invalid retained window span"}


def test_gap_near_boundary_refuses_even_when_both_fixes_are_inside():
    polygon = Polygon([(-81, 30), (-79, 30), (-79, 32), (-81, 32)])
    proof = prove_windows_inside([_window(30.002, gap=30)], polygon, max_speed_m_s=70)
    assert proof["verified"] is False
    assert "boundary" in proof["reason"]


def test_observed_outside_fix_and_missing_edge_fix_refuse():
    polygon = Polygon([(-81, 30), (-79, 30), (-79, 32), (-81, 32)])
    assert not prove_windows_inside([_window(29.0)], polygon)["verified"]
    window = RouteWindow("lf1", 0.0, 2000.0, [(1900.0, 31.0, -80.0)])
    assert not prove_windows_inside([window], polygon)["verified"]


def test_short_retained_window_with_no_fix_uses_nearest_source_anchors():
    polygon = Polygon([(-81, 30), (-79, 30), (-79, 32), (-81, 32)])
    window = RouteWindow(
        "lf1",
        7.0,
        10.0,
        [],
        prev_fix=(6.0, 31.0, -80.0),
        next_fix=(10.1, 31.0, -80.0),
    )
    assert prove_windows_inside([window], polygon)["verified"]
    window.next_fix = (10.1, 29.0, -80.0)
    assert not prove_windows_inside([window], polygon)["verified"]


def test_pipeline_uses_bounded_county_proof_before_burn_and_records_its_source():
    class Geo:
        def reverse_geocode_detailed(self, lat, lon):
            return {
                "town": "Alder" if lat < 31.1 else "Birch",
                "county": "Ash County",
                "state": "South Carolina",
            }

    class Proof:
        def prove(self, windows, claim):
            return {
                "verified": True,
                "basis": "pinned_tiger_polygon_bounded_speed_and_error",
                "source": f"TIGER2026/{claim['level'].upper()}/tl_2026_us_{claim['level']}.zip",
                "sha256": _SOURCES[claim["level"]][1],
                "region": claim["name"],
                "statefp": "45",
                "max_gap_s": 30.0,
                "max_speed_m_s": 70.0,
                "gps_error_m": 100.0,
                "windows_key": retained_windows_key(windows),
            }

    windows = [
        RouteWindow(
            "lf1",
            0.0,
            180.0,
            [(float(t), 31.0 if t < 90 else 31.2, -80.0) for t in range(180) if not 70 <= t < 100],
        )
    ]
    text = {"2026-02-20": {1: "Alder, SC"}}
    measured = enforce_route_claims(
        text,
        {1: windows},
        Geo(),
        boundary_prover=Proof(),
        require_review_support=True,
        review_support_available=True,
    )
    assert text["2026-02-20"][1] == "Ash County, SC"
    claim = measured[1].region_claim("county")
    assert claim["boundary_proof"]["verified"] is True
    assert measured[1].record()["boundary_proofs"]["county"]["sha256"].startswith("b527")
    assert measured[1].record()["boundary_proofs"]["county"]["recording_date"] == "2026-02-20"


def test_forged_or_weaker_proof_does_not_override_sampled_gap():
    class Geo:
        def reverse_geocode_detailed(self, lat, lon):
            return {"town": "Alder", "county": "Ash County", "state": "South Carolina"}

    window = RouteWindow(
        "lf1", 0, 180, [(float(t), 31.0, -80.0) for t in range(180) if not 50 <= t < 130]
    )
    ev = measure_route([window], Geo())
    ev.boundary_proofs["county"] = {
        "windows_key": ev.windows_key,
        "verified": True,
        "basis": "pinned_tiger_polygon_bounded_speed_and_error",
        "source": "TIGER2026/COUNTY/tl_2026_us_county.zip",
        "sha256": "0" * 64,
        "region": "Ash County",
        "statefp": "45",
        "recording_date": "2026-02-20",
        "max_speed_m_s": 1.0,
        "gps_error_m": 0.1,
    }
    ev.boundary_proofs["state"] = {
        "windows_key": ev.windows_key,
        "verified": True,
        "basis": "pinned_tiger_polygon_bounded_speed_and_error",
        "source": "TIGER2026/STATE/tl_2026_us_state.zip",
        "sha256": "2e17ea40dff3e00163eec87a26efa0b0ff7e8fe364a191bf1a7754dcb49ea326",  # pragma: allowlist secret (public Census SHA-256)
        "region": "South Carolina",
        "recording_date": "2026-02-20",
        "statefp": "45",
        "max_speed_m_s": 70.0,
        "gps_error_m": 100.0,
        "max_gap_s": 81.0,
    }
    assert _region_ok(ev.region_claim("county"), ev, 1.0)
    ev.boundary_proofs["county"].update(
        sha256="b527c9ba7aeb8f17ef118f026eae5e5a938aa1d00b873bc32342d0dce59321c0",  # pragma: allowlist secret (public Census SHA-256)
        max_speed_m_s=70.0,
        gps_error_m=100.0,
        max_gap_s=81.0,
    )
    state_proof = ev.boundary_proofs.pop("state")
    assert _region_ok(ev.region_claim("county"), ev, 1.0)
    ev.boundary_proofs["state"] = state_proof
    assert _region_ok(ev.region_claim("county"), ev, 1.0) == []


def test_pipeline_refuses_before_encode_when_waypost_lacks_boundary_review():
    class Geo:
        def reverse_geocode_detailed(self, lat, lon):
            return {"town": "Alder", "county": "Ash County", "state": "South Carolina"}

    class Proof:
        def prove(self, windows, claim):
            raise AssertionError("must not fetch polygons without a matching reviewer")

    window = RouteWindow(
        "lf1", 0, 180, [(float(t), 31.0, -80.0) for t in range(180) if not 50 <= t < 130]
    )
    with pytest.raises(RouteClaimRefusedError, match="Waypost bounded-region review unavailable"):
        enforce_route_claims(
            {"2026-02-20": {1: "Alder, SC"}},
            {1: [window]},
            Geo(),
            boundary_prover=Proof(),
            require_review_support=True,
            review_support_available=False,
        )


def test_2026_polygon_cannot_approve_a_different_recording_year():
    class Geo:
        def reverse_geocode_detailed(self, lat, lon):
            return {"town": "Alder", "county": "Ash County", "state": "South Carolina"}

    class Proof:
        def prove(self, windows, claim):
            raise AssertionError("wrong-year boundary must never be used")

    window = RouteWindow(
        "lf1", 0, 180, [(float(t), 31.0, -80.0) for t in range(180) if not 50 <= t < 130]
    )
    with pytest.raises(Exception, match="boundary year"):
        enforce_route_claims(
            {"2025-02-20": {1: "Alder, SC"}},
            {1: [window]},
            Geo(),
            boundary_prover=Proof(),
        )


def _pinned_zip(name: str, statefp: str, polygon: list[tuple[float, float]]) -> bytes:
    """One-record TIGER-shaped fixture; no downloaded data in unit tests."""
    fields = [("NAME", 32), ("STATEFP", 2)]
    header_len = 32 + 32 * len(fields) + 1
    record_len = 1 + sum(size for _, size in fields)
    dbf = bytearray(header_len + record_len + 1)
    dbf[0] = 3
    struct.pack_into("<IHH", dbf, 4, 1, header_len, record_len)
    for i, (field, size) in enumerate(fields):
        start = 32 + 32 * i
        dbf[start : start + len(field)] = field.encode()
        dbf[start + 11] = ord("C")
        dbf[start + 16] = size
    dbf[header_len - 1] = 13
    row = (" " + name.ljust(32) + statefp).encode()
    dbf[header_len : header_len + len(row)] = row
    dbf[-1] = 26

    points = polygon + [polygon[0]]
    signed = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:]))
    if signed > 0:  # SHP outer rings are clockwise.
        points = list(reversed(points))
    body = struct.pack("<i4dii", 5, -81, 30, -79, 32, 1, len(points))
    body += struct.pack("<i", 0) + b"".join(struct.pack("<dd", *point) for point in points)
    shp = bytearray(100)
    shx = bytearray(100)
    for header, length in ((shp, 100 + 8 + len(body)), (shx, 108)):
        struct.pack_into(">i", header, 0, 9994)
        struct.pack_into(">i", header, 24, length // 2)
        struct.pack_into("<ii", header, 28, 1000, 5)
    shp += struct.pack(">ii", 1, len(body) // 2) + body
    shx += struct.pack(">ii", 50, len(body) // 2)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("tl_test.dbf", dbf)
        z.writestr("tl_test.shp", shp)
        z.writestr("tl_test.shx", shx)
    return out.getvalue()


def test_pinned_zip_reads_named_shape_from_cache_without_network(tmp_path):
    data = _pinned_zip("South Carolina", "45", [(-81, 30), (-81, 32), (-79, 32), (-79, 30)])
    cache = tmp_path / "state.zip"
    cache.write_bytes(data)
    prover = CensusBoundaryProver(
        cache_dir=tmp_path, sources={"state": ("state.zip", hashlib.sha256(data).hexdigest())}
    )
    polygon = prover.load_region("state", "South Carolina", "45")
    assert polygon.covers(Polygon([(-80, 31), (-80, 31.1), (-79.9, 31)]))
    with pytest.raises(BoundaryArchiveError, match="checksum"):
        CensusBoundaryProver(
            cache_dir=tmp_path, sources={"state": ("state.zip", "0" * 64)}
        ).load_region("state", "South Carolina", "45")


def test_census_download_refuses_http_redirects(monkeypatch, tmp_path):
    def fake_opener(handler):
        assert handler.redirect_request(None, None, 302, None, {}, None) is None

        class Opener:
            def open(self, url, timeout):
                raise urllib.error.HTTPError(
                    url, 302, "redirect", {"Location": "https://elsewhere.invalid/data"}, None
                )

        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_opener)
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: pytest.fail("redirect-following urlopen was used"),
    )
    prover = CensusBoundaryProver(cache_dir=tmp_path)
    with pytest.raises(BoundaryArchiveError, match="archive unavailable"):
        prover._pinned_zip("county")


def test_census_download_caps_zip_size(monkeypatch, tmp_path):
    from gomoveshift.pipeline import boundary_proof

    class Opener:
        def open(self, url, timeout):
            return io.BytesIO(b"five!")

    monkeypatch.setattr(boundary_proof, "_MAX_ARCHIVE_BYTES", 4, raising=False)
    monkeypatch.setattr(urllib.request, "build_opener", lambda handler: Opener())
    with pytest.raises(BoundaryArchiveError, match="size limit"):
        CensusBoundaryProver(cache_dir=tmp_path)._pinned_zip("county")
    assert not list(tmp_path.iterdir())


def test_census_download_host_is_in_service_allowlist():
    from urllib.parse import urlparse

    from gomoveshift.pipeline import boundary_proof

    host = urlparse(boundary_proof._BASE).hostname
    allowlist = (
        Path(__file__).parents[1] / "docs/pipeline/84-network-allowlist-credential-stores.md"
    ).read_text()
    assert host == "www2.census.gov"
    assert f"**{host}**" in allowlist


# -- Ireland: Tailte Eireann's pinned 2026 statutory county map (ITM metres) --

from gomoveshift.pipeline.boundary_proof import (  # noqa: E402
    IE_MARGIN_M,
    _IE_HOSTS,
    _IE_SOURCE,
    _HostRedirect,
    itm,
    pinned_boundary_proof,
)

_CORK_FIX = (51.93, -9.10)  # a stateless County Cork retained fix


def test_itm_projection_matches_its_false_origin_and_scale():
    x, y = itm(53.5, -8.0)
    assert abs(x - 600000.0) < 0.001 and abs(y - 750000.0) < 0.001
    east, west = itm(53.5, -7.0), itm(53.5, -9.0)
    assert abs((east[0] - 600000.0) - (600000.0 - west[0])) < 0.001
    assert abs(east[1] - west[1]) < 0.001
    # One degree of meridian near 54 N is ~111,305 m, times k0 = 0.99982.
    assert abs(itm(54.5, -8.0)[1] - y - 111_285) < 30


def _tailte_geojson(name: str, half_m: float, centre=_CORK_FIX) -> bytes:
    import json

    cx, cy = itm(*centre)
    ring = [
        [cx - half_m, cy - half_m, -100000.0],
        [cx + half_m, cy - half_m, -100000.0],
        [cx + half_m, cy + half_m, -100000.0],
        [cx - half_m, cy + half_m, -100000.0],
        [cx - half_m, cy - half_m, -100000.0],
    ]
    return json.dumps(
        {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:2157"}},
            "features": [
                {
                    "type": "Feature",
                    "properties": {"ENG_NAME_VALUE": name, "BDY_TYPE_VALUE": "County Statutory"},
                    "geometry": {"type": "Polygon", "coordinates": [ring]},
                }
            ],
        }
    ).encode()


def _ie_prover(tmp_path, data: bytes) -> CensusBoundaryProver:
    # The real file name with the fixture's own digest (the real map is ~92 MB).
    (tmp_path / _IE_SOURCE[0]).write_bytes(data)
    return CensusBoundaryProver(
        cache_dir=tmp_path, ie_source=(_IE_SOURCE[0], hashlib.sha256(data).hexdigest())
    )


def _ie_window(gap: float = 1.0) -> RouteWindow:
    lat, lon = _CORK_FIX
    return RouteWindow("lf1", 0.0, 2 * gap, [(0.0, lat, lon), (gap, lat, lon)])


_IE_CLAIM = {"level": "county", "name": "County Cork", "state": None, "country": "Ireland"}


def test_irish_county_is_proven_on_the_tailte_map_with_an_inward_margin(tmp_path):
    prover = _ie_prover(tmp_path, _tailte_geojson("CORK", 5_000))
    proof = prover.prove([_ie_window(gap=17.0)], dict(_IE_CLAIM))
    assert proof["verified"] is True, proof
    assert proof["basis"] == "pinned_tailte_polygon_bounded_speed_error_and_margin"
    assert proof["source"] == f"TAILTE2026/COUNTIES/{_IE_SOURCE[0]}"
    assert proof["region"] == "County Cork" and proof["country"] == "IE"
    assert proof["margin_m"] == IE_MARGIN_M == 250.0
    assert proof["max_gap_s"] == 17.0


def test_irish_proof_refuses_a_fix_inside_the_vintage_margin(tmp_path):
    # 300 m inside: enough for the TIGER rule at 1 s (100 + 70 = 170 m), not
    # once the 250 m vintage margin is added (420 m).
    prover = _ie_prover(tmp_path, _tailte_geojson("CORK", 300))
    proof = prover.prove([_ie_window()], dict(_IE_CLAIM))
    assert proof["verified"] is False and "boundary" in proof["reason"]


def test_irish_proof_refuses_an_unknown_county_or_wrong_projection(tmp_path):
    prover = _ie_prover(tmp_path, _tailte_geojson("KERRY", 5_000))
    assert prover.prove([_ie_window()], dict(_IE_CLAIM))["verified"] is False
    other = tmp_path / "other"
    other.mkdir()
    bad = _tailte_geojson("CORK", 5_000).replace(b"EPSG:2157", b"EPSG:4326")
    assert _ie_prover(other, bad).prove([_ie_window()], dict(_IE_CLAIM))["verified"] is False


def test_pinned_irish_proof_needs_its_source_margin_country_and_vintage(tmp_path):
    prover = _ie_prover(tmp_path, _tailte_geojson("CORK", 5_000))
    windows = [_ie_window()]
    claim = {**_IE_CLAIM, "windows_key": retained_windows_key(windows)}
    proof = prover.prove(windows, dict(claim))
    proof["sha256"] = _IE_SOURCE[1]  # what the real pinned map carries
    proof["recording_date"] = "2025-09-15"
    assert pinned_boundary_proof(proof, claim)
    for change in (
        {"margin_m": 249.0},
        {"country": None},
        {"basis": "pinned_tiger_polygon_bounded_speed_and_error"},
        {"recording_date": "2027-01-04"},
        {"sha256": "0" * 64},
    ):
        assert not pinned_boundary_proof({**proof, **change}, claim), change
    assert not pinned_boundary_proof(proof, {**claim, "country": "United Kingdom"})


def test_tailte_download_redirects_only_to_its_https_hosts():
    handler = _HostRedirect(_IE_HOSTS)
    req = urllib.request.Request(
        "https://data-osi.opendata.arcgis.com/api/download/v1/items/x/geojson"
    )
    for host in _IE_HOSTS:
        assert handler.redirect_request(req, None, 302, "Found", {}, f"https://{host}/f")
    assert (
        handler.redirect_request(req, None, 302, "Found", {}, "https://elsewhere.invalid/f") is None
    )
    assert (
        handler.redirect_request(req, None, 302, "Found", {}, f"http://{next(iter(_IE_HOSTS))}/f")
        is None
    )


def test_tailte_hosts_are_in_service_allowlist():
    allowlist = (
        Path(__file__).parents[1] / "docs/pipeline/84-network-allowlist-credential-stores.md"
    ).read_text()
    for host in _IE_HOSTS:
        assert f"**{host}**" in allowlist


def test_gapped_irish_route_gets_a_proof_even_when_the_county_label_passes(tmp_path):
    class Geo:
        def reverse_geocode_detailed(self, lat, lon):
            return {"town": "Ballyvourney", "county": "County Cork", "country": "Ireland"}

    lat, lon = _CORK_FIX
    window = RouteWindow(
        "lf1", 0, 180, [(float(t), lat, lon) for t in range(180) if not 50 <= t < 62]
    )
    prover = _ie_prover(tmp_path, _tailte_geojson("CORK", 5_000))
    text = {"2025-09-15": {1: "County Cork"}}
    measured = enforce_route_claims(
        text,
        {1: [window]},
        Geo(),
        boundary_prover=prover,
        require_review_support=True,
        review_support_available=True,
    )
    assert text["2025-09-15"][1] == "County Cork"
    claim = measured[1].region_claim("county")
    assert claim["country"] == "Ireland"
    proof = claim["boundary_proof"]
    assert proof["basis"] == "pinned_tailte_polygon_bounded_speed_error_and_margin"
    assert proof["recording_date"] == "2025-09-15"
    assert proof["max_gap_s"] >= 12.0
