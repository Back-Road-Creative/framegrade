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
