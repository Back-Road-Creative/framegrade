"""A short's own clip GPS is routed through the destination
resolver so a short shot *at* a named destination (WMA / reserve /
protected-area / viewpoint) names that destination instead of the nearest
town — mirroring the LF surface.

All offline: the POI source is injected and ``dominant_location`` is patched,
so no live Overpass / Nominatim call. The no-destination path must preserve
today's ``dominant_location`` behaviour exactly (regression).
"""

from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.pipeline.gps_enrichment import (
    _make_short_ost_text_fn,
    _resolve_short_destination,
)
from gomoveshift.pipeline.title_builder import DominantLocation

# A clip whose on-screen GPS sits inside Donnelley WMA (smaller polygon),
# itself inside the broad ACE Basin protected area.
_TRIMMED = [(50.0, 32.62, -80.50), (60.0, 32.625, -80.495)]
_ACE = {
    "type": "relation",
    "bounds": {"minlat": 32.40, "minlon": -80.80, "maxlat": 32.90, "maxlon": -80.20},
    "tags": {"boundary": "protected_area", "name": "ACE Basin"},
}
_WMA = {
    "type": "relation",
    "bounds": {"minlat": 32.58, "minlon": -80.56, "maxlat": 32.66, "maxlon": -80.44},
    "tags": {"leisure": "nature_reserve", "name": "Donnelley Wildlife Management Area"},
}
_SHORT = SimpleNamespace(rank=1, start_sec=40.0, end_sec=70.0)


def _dom(name="Green Pond", county="Colleton County", state="SC"):
    return DominantLocation(county=county, state=state, name=name)


def test_resolver_helper_votes_clip_fixes():
    """The single-clip helper votes one featured segment per trimmed fix and
    resolves the smallest enclosing destination (both fixes inside the WMA)."""
    assert (
        _resolve_short_destination(_TRIMMED, poi_source=lambda _b: [_ACE, _WMA])
        == "Donnelley Wildlife Management Area"
    )


def test_resolver_helper_empty_and_nullisland_degrade():
    assert _resolve_short_destination([], poi_source=lambda _b: [_WMA]) is None
    # Null-island points are dropped → no usable fix → None.
    assert _resolve_short_destination([(1.0, 0.0, 0.0)], poi_source=lambda _b: [_WMA]) is None


def test_resolver_helper_exception_degrades():
    def _boom(_b):
        raise RuntimeError("overpass down")

    assert _resolve_short_destination(_TRIMMED, poi_source=_boom) is None


# ── in-polygon SAMPLE QUORUM (no centroid-of-arc containment) ───────────
#
# A small reserve; the clip is a curved arc that bows AROUND it. The arc's mean
# point (centroid) lands inside the reserve, but the footage only grazes it.
_GRAZED = {
    "type": "relation",
    "bounds": {"minlat": 32.58, "minlon": -80.52, "maxlat": 32.62, "maxlon": -80.48},
    "tags": {"leisure": "nature_reserve", "name": "Grazed Reserve"},
}


def test_curved_clip_centroid_inside_not_selected():
    """A curved clip whose CENTROID falls inside a reserve the footage
    never entered must NOT name it. Four arc fixes sit outside the reserve on
    each side plus one grazing fix inside — their mean is the reserve centre, so
    the pre-fix centroid-of-arc containment named it. Voting the ACTUAL fixes,
    the reserve holds only 1/5 (< quorum) of the clip → not selected."""
    arc = [
        (10.0, 32.60, -80.50),  # the single grazing fix (inside)
        (20.0, 32.55, -80.50),  # south, outside
        (30.0, 32.65, -80.50),  # north, outside
        (40.0, 32.60, -80.45),  # east, outside
        (50.0, 32.60, -80.55),  # west, outside
    ]
    assert _resolve_short_destination(arc, poi_source=lambda _b: [_GRAZED]) is None


def test_clip_meeting_sample_quorum_still_resolves():
    """Guard: a clip whose footage actually crosses the reserve (majority of
    fixes inside) still names it — the quorum never suppresses a real crossing."""
    crossing = [
        (10.0, 32.60, -80.50),  # inside
        (20.0, 32.61, -80.49),  # inside
        (30.0, 32.59, -80.51),  # inside
        (40.0, 32.65, -80.50),  # outside
    ]
    assert _resolve_short_destination(crossing, poi_source=lambda _b: [_GRAZED]) == "Grazed Reserve"


def test_destination_preferred_over_town_with_state():
    """When a destination resolves, it is the burned label, with the state from
    the voted dominant county appended."""
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [_ACE, _WMA])
    with patch(
        "gomoveshift.pipeline.gps_enrichment.dominant_location",
        return_value=_dom(),
    ):
        assert fn(_TRIMMED, _SHORT) == "Donnelley Wildlife Management Area, SC"


def test_destination_without_state_returns_bare_name():
    """A destination resolves but the county vote yields no state → bare
    destination (never a trailing comma)."""
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [_WMA])
    with patch(
        "gomoveshift.pipeline.gps_enrichment.dominant_location",
        return_value=_dom(state=None),
    ):
        assert fn(_TRIMMED, _SHORT) == "Donnelley Wildlife Management Area"


def test_geocoder_failure_suppresses_uncorroborated_destination():
    """A dominant_location exception leaves the destination with ZERO GPS
    corroboration — and the short region gate no-ops on the same miss
    (``short_regions_fn`` returns ``[]``) — so the closure degrades to the
    caller fallback instead of burning ungated text."""
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [_WMA])
    with patch(
        "gomoveshift.pipeline.gps_enrichment.dominant_location",
        side_effect=RuntimeError("geocoder down"),
    ):
        assert fn(_TRIMMED, _SHORT) is None


def test_quorum_miss_suppresses_uncorroborated_destination():
    """A county vote that missed quorum (``dom is None``) must not burn a bare
    destination: the short region gate would no-op off the same quorum miss,
    shipping burned text with zero GPS corroboration."""
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [_ACE, _WMA])
    with patch(
        "gomoveshift.pipeline.gps_enrichment.dominant_location",
        return_value=None,
    ):
        assert fn(_TRIMMED, _SHORT) is None


def test_no_destination_falls_back_to_dominant_location():
    """REGRESSION: with no enclosing destination POI the closure returns the
    existing place + state label voted across the trimmed points — behaviour is
    unchanged from before destination routing."""
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [])
    with patch(
        "gomoveshift.pipeline.gps_enrichment.dominant_location",
        return_value=_dom(name="Port Royal", state="SC"),
    ):
        assert fn(_TRIMMED, _SHORT) == "Port Royal, SC"


def test_no_destination_and_no_geocode_returns_none():
    """REGRESSION: no destination and no dominant location → None, so the shorts
    loop falls back to the drive-wide location string."""
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [])
    with patch(
        "gomoveshift.pipeline.gps_enrichment.dominant_location",
        return_value=None,
    ):
        assert fn(_TRIMMED, _SHORT) is None
    # Empty trimmed short-circuits before any resolver/geocoder call.
    assert fn([], _SHORT) is None


# ── geocode quorum floor on the shorts vote ───────────────────────
#
# The shorts county vote now passes ``min_votes=GEOCODE_QUORUM_MIN`` so a
# single outlier point can no longer determine the short's county — the same
# quorum the per-LF vote already enforces. These use the REAL
# ``dominant_location`` (no patch) so the kwarg is actually exercised end to
# end; the destination resolver is patched off to isolate the county path.

_BEAUFORT_DETAIL = {"county": "Beaufort County", "state": "South Carolina", "city": "Port Royal"}


class _GeoFirstOnly:
    """Geocodes only the first sampled coord to a county; the rest yield no
    county/state. Reproduces the single-outlier case: exactly one usable vote."""

    def __init__(self):
        self.calls = 0

    def reverse_geocode_detailed(self, lat, lon):
        self.calls += 1
        return dict(_BEAUFORT_DETAIL) if self.calls == 1 else {}


class _GeoAll:
    """Every sampled coord geocodes to the same county (a normal dense clip)."""

    def reverse_geocode_detailed(self, lat, lon):
        return dict(_BEAUFORT_DETAIL)


def test_single_outlier_vote_degrades_to_none():
    """One geocoded sample (< quorum) no longer stamps the short's county — the
    closure degrades to None so the per-day loop falls back to the drive-wide
    location, instead of trusting a single-point plurality."""
    fn = _make_short_ost_text_fn(_GeoFirstOnly(), poi_source=lambda _b: [])
    with patch(
        "gomoveshift.pipeline.gps_enrichment._resolve_short_destination",
        return_value=None,
    ):
        assert fn([(0.0, 32.3095, -80.6332)], _SHORT) is None


def test_multi_point_clip_still_resolves_county():
    """A normal clip with several geocoded fixes meets the quorum and still
    resolves its place + state label — the quorum is safe for real shorts."""
    fn = _make_short_ost_text_fn(_GeoAll(), poi_source=lambda _b: [])
    pts = [(float(i), 32.3095 + i * 0.001, -80.6332 + i * 0.001) for i in range(6)]
    with patch(
        "gomoveshift.pipeline.gps_enrichment._resolve_short_destination",
        return_value=None,
    ):
        assert fn(pts, _SHORT) == "Port Royal, South Carolina"
