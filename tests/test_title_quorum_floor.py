"""Title in-place vote must honour the geocode quorum floor.

Every other production caller of ``dominant_location`` (the GPS-enrichment
label path, the shorts region vote, the cross-surface content gate) passes
``min_votes=GEOCODE_QUORUM_MIN`` so a plurality off a single successfully
geocoded sample is rejected. ``build_lf_title``'s in-place vote was the last
site left on the default ``min_votes=1``: a single flaky sample could anchor
a wrong settlement into the burned OST/title.

These tests exercise the vote THROUGH ``build_lf_title``:
  * one geocoded sample (below quorum) must NOT anchor its settlement;
  * two geocoded samples (at quorum) still anchor as before.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.gps_enrichment import GEOCODE_QUORUM_MIN
from gomoveshift.pipeline.title_builder import build_lf_title

# Cashiers, NC (a recognizable settlement) and a nearby "gap" fix ~14 km
# away — close enough to share the spatial cluster (so neither is dropped as
# a GPS outlier) but far enough that only the exact Cashiers coord
# reverse-geocodes to a county/state. The gap fix models a geocode miss, so
# it does NOT count toward the quorum.
_CASHIERS = (35.10, -83.10)
_GAP = (35.20, -83.20)


def _geo():
    """Geocoder whose endpoints never resolve to a town (forcing the
    settlement backfill from the vote), and whose only county/state +
    settlement sample sits at Cashiers."""

    class _Geo:
        def reverse_geocode(self, lat, lon):
            # No settlement anywhere → start/end towns are None, so the ONLY
            # way "Cashiers" reaches the title is the in-place vote's sequence.
            return SimpleNamespace(
                city=None,
                town=None,
                village=None,
                hamlet=None,
                locality=None,
                suburb=None,
                name=None,
            )

        def reverse_geocode_detailed(self, lat, lon):
            if abs(lat - _CASHIERS[0]) + abs(lon - _CASHIERS[1]) < 0.05:
                return {"county": "Jackson County", "state": "North Carolina", "town": "Cashiers"}
            return {}

    return _Geo()


def _seg(lat, lon):
    return SimpleNamespace(gps_center_lat=lat, gps_center_lon=lon)


def _pt(lat, lon):
    return {"lat": lat, "lon": lon}


def test_quorum_constant_is_a_real_floor():
    """Guard the premise: the floor other callers pass is >= 2, so a single
    sample is genuinely below it."""
    assert GEOCODE_QUORUM_MIN >= 2


def test_single_flaky_sample_does_not_anchor():
    """One geocoded sample (< GEOCODE_QUORUM_MIN) is a weak plurality — the
    title must NOT anchor on its settlement. With the pre-fix ``min_votes=1``
    default the lone Cashiers sample wins and burns "Cashiers"; the quorum
    floor degrades the vote to None so the title falls back to the county."""
    gps_points = [_pt(*_CASHIERS)] + [_pt(*_GAP)] * 2
    title = build_lf_title(
        segments=[_seg(*_CASHIERS), _seg(*_GAP)],
        geocoder=_geo(),
        per_lf_location="Jackson County, North Carolina",
        gps_points=gps_points,
    )
    assert "Cashiers" not in title


def test_quorum_sample_still_anchors():
    """Two geocoded samples meet the floor — the settlement still anchors, so
    the fix rejects only the below-quorum case, never a genuine plurality."""
    gps_points = [_pt(*_CASHIERS)] * 2 + [_pt(*_GAP)]
    title = build_lf_title(
        segments=[_seg(*_CASHIERS), _seg(*_GAP)],
        geocoder=_geo(),
        per_lf_location="Jackson County, North Carolina",
        gps_points=gps_points,
    )
    assert "Cashiers" in title
