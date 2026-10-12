"""Drive-path wiring of the dominant-county reducer.

Asserts derive_per_lf_locations and the shorts OST closure label by the
modal county, not the single centroid — so a Fripp-dominant (Beaufort)
LF that also clips a few Edisto (Colleton) points is labeled Beaufort.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.gps_enrichment import (
    _make_short_ost_text_fn,
    derive_per_lf_locations,
)

# Real cached coords (geocoding_cache.json).
_FRIPP = (32.3095, -80.6332)  # Beaufort County
_EDISTO = (32.6591, -80.6646)  # Colleton County

_BEAUFORT = {"county": "Beaufort County", "state": "South Carolina", "hamlet": "Fort Fremont"}
_COLLETON = {"county": "Colleton County", "state": "South Carolina", "town": "Edisto Beach"}


class _Geo:
    """Mock geocoder keyed by nearest cached point."""

    def reverse_geocode_detailed(self, lat, lon):
        return dict(_BEAUFORT) if abs(lat - _FRIPP[0]) < 0.2 else dict(_COLLETON)

    def reverse_geocode(self, lat, lon):  # unused now that the fallback is deleted
        d = self.reverse_geocode_detailed(lat, lon)
        return SimpleNamespace(city=None, locality=None, name=None, suburb=None, **d)


def _seg(lat, lon):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon)


def _pts(latlon, n):
    return [{"lat": latlon[0], "lon": latlon[1]} for _ in range(n)]


class TestDerivePerLfDominant:
    def test_beaufort_dominant_lf_not_mislabeled_colleton(self):
        lf = SimpleNamespace(index=1, segments=[_seg(*_FRIPP), _seg(*_EDISTO)])
        # Plurality of on-screen GPS is Fripp/Beaufort; a couple of Edisto fixes.
        gps_points_by_lf = {1: _pts(_FRIPP, 8) + _pts(_EDISTO, 2)}
        result = derive_per_lf_locations([lf], geocoder=_Geo(), gps_points_by_lf=gps_points_by_lf)
        assert "Beaufort County" in result[1]
        assert "Colleton" not in result[1]

    def test_genuine_colleton_lf_stays_colleton(self):
        lf = SimpleNamespace(index=1, segments=[_seg(*_EDISTO)])
        result = derive_per_lf_locations(
            [lf], geocoder=_Geo(), gps_points_by_lf={1: _pts(_EDISTO, 6)}
        )
        assert "Colleton County" in result[1]

    def test_segment_centroid_vote_without_point_trace(self):
        # No gps_points_by_lf (resume path) → vote over the per-segment
        # centroids; two Fripp segments meet the geocode quorum.
        lf = SimpleNamespace(index=1, segments=[_seg(*_FRIPP), _seg(*_FRIPP)])
        result = derive_per_lf_locations([lf], geocoder=_Geo())
        assert "Beaufort County" in result[1]


_BLANK = (10.0, -40.0)  # mid-Atlantic; geocodes to no county/state


class _GeoQuorum:
    """Geocoder where only points near Fripp resolve a county; far/blank
    points return no county/state so a quorum miss can be forced."""

    def reverse_geocode_detailed(self, lat, lon):
        if abs(lat - _FRIPP[0]) < 0.2 and abs(lon - _FRIPP[1]) < 0.2:
            return dict(_BEAUFORT)
        return {"county": None, "state": None}  # geocode miss → no vote

    def reverse_geocode(self, lat, lon):  # deleted bare-centroid fallback
        # Regression canary: if the fallback is reintroduced this stub
        # RETURNS a usable label, so derive stamps it instead of raising
        # and the hard-fail assertion below goes red.
        return SimpleNamespace(
            city=None, locality=None, name="Open Water", suburb=None, state=None, county=None
        )


class TestDerivePerLfQuorumFloor:
    def test_weak_quorum_lf_hard_fails_not_bare_centroid(self):
        """A quorum-degraded LF joins the aggregated ambiguous-input
        raise (operator supplies ``--per-lf-location``) — it never ships a
        bare-centroid-geocoded label ("GPS labels vote the dominant county,
        never the bare centroid")."""
        lf = SimpleNamespace(index=1, segments=[_seg(*_BLANK)])
        # 1 Fripp/Beaufort fix + 8 blank (no-county) fixes → quorum = 1 < 2.
        gps_points_by_lf = {1: _pts(_FRIPP, 1) + _pts(_BLANK, 8)}
        with pytest.raises(ValueError) as exc:
            derive_per_lf_locations([lf], geocoder=_GeoQuorum(), gps_points_by_lf=gps_points_by_lf)
        msg = str(exc.value)
        assert "ambiguous input" in msg
        assert "[1]" in msg
        assert "--per-lf-location" in msg


_MIZEN = (51.6, -8.9)  # West Cork, County Cork, Ireland
_CORK_IE = {
    "county": "County Cork",
    "state": None,
    "country": "Ireland",
    "locality": "West Cork",
    "road": "N71",
}


class _IrishGeo:
    """Stateless-county geocoder (Irish counties carry no ``state``): every
    sample resolves to County Cork / West Cork / N71, matching the real
    resolved-location shape for that area."""

    def reverse_geocode_detailed(self, lat, lon):
        return dict(_CORK_IE)

    def reverse_geocode(self, lat, lon):  # county-shaped only; no town field
        return SimpleNamespace(city=None, locality=None, name=None, suburb=None)


class TestDerivePerLfStatelessCountryTail:
    """A stateless county's label must carry its geocoded
    country. ``build_lf_title``'s ``region_label`` is derived from the
    label's tail after the first comma; a bare ``"County Cork"`` (no comma)
    yields an EMPTY region_label, which gates OFF every cascade branch
    (settlement anchor, via-road, A-to-B) and collapses straight to the
    Level-5 raw-label fallback — the "County Cork in 4K" defect."""

    def test_stateless_county_carries_country_tail_not_bare(self):
        lf = SimpleNamespace(index=1, segments=[_seg(*_MIZEN)])
        result = derive_per_lf_locations(
            [lf], geocoder=_IrishGeo(), gps_points_by_lf={1: _pts(_MIZEN, 8)}
        )
        assert result[1] == "County Cork, Ireland"

    def test_stateless_county_no_country_signal_stays_bare(self):
        """No country signal to append — degrade to the pre-fix bare label,
        never a fabricated tail."""

        class _NoCountryGeo:
            def reverse_geocode_detailed(self, lat, lon):
                return {"county": "County Cork", "state": None}

            def reverse_geocode(self, lat, lon):
                return SimpleNamespace(city=None, locality=None, name=None, suburb=None)

        lf = SimpleNamespace(index=1, segments=[_seg(*_MIZEN)])
        result = derive_per_lf_locations(
            [lf], geocoder=_NoCountryGeo(), gps_points_by_lf={1: _pts(_MIZEN, 8)}
        )
        assert result[1] == "County Cork"

    def test_stated_us_county_label_byte_identical(self):
        """Regression guard: a US (stated) county's label is untouched —
        the country tail only applies to the stateless branch."""
        lf = SimpleNamespace(index=1, segments=[_seg(*_FRIPP)])
        result = derive_per_lf_locations(
            [lf], geocoder=_Geo(), gps_points_by_lf={1: _pts(_FRIPP, 8)}
        )
        assert result[1] == "Beaufort County, South Carolina"

    def test_mizen_fixture_end_to_end_anchors_on_place_not_bare_county(self):
        """Reproduces a past defect end to end: one recognized
        place (West Cork) and a unanimous road (N71), no endpoints. The
        title must anchor on the place, never collapse to the bare county."""
        from gomoveshift.pipeline.title_builder import build_lf_title

        lf = SimpleNamespace(index=1, segments=[_seg(*_MIZEN)])
        gps_points_by_lf = {1: _pts(_MIZEN, 8)}
        locations = derive_per_lf_locations(
            [lf], geocoder=_IrishGeo(), gps_points_by_lf=gps_points_by_lf
        )
        title = build_lf_title(
            lf.segments,
            _IrishGeo(),
            locations[1],
            gps_points=gps_points_by_lf[1],
        )
        assert title != "County Cork"
        assert "West Cork" in title


class TestShortsOstDominant:
    # No clip destination resolves here — keeps the resolver offline and asserts
    # the unchanged dominant_location label (regression guard).
    def test_short_labels_dominant_county(self):
        fn = _make_short_ost_text_fn(_Geo())
        # (t, lat, lon) tuples — the shorts shape.
        pts = [(0.0, *_FRIPP)] * 8 + [(1.0, *_EDISTO)] * 2
        with patch(
            "gomoveshift.pipeline.gps_enrichment._resolve_short_destination",
            return_value=None,
        ):
            label = fn(pts, SimpleNamespace(rank=1))
        assert label is not None
        assert "South Carolina" in label
        assert "Colleton" not in label

    def test_short_empty_points_returns_none(self):
        fn = _make_short_ost_text_fn(_Geo())
        assert fn([], SimpleNamespace(rank=1)) is None
