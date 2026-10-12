"""A short's voted settlement must lie on the clip's own GPS trace.

The reverse geocoder can name a hamlet in its address although the clip never
passes it, so the voted name is verified against the clip's fixes. The LF
surface proves an endpoint town with a forward lookup and the 2 km trace rule;
the short surface does the same, else burns the voted county.

All offline: ``dominant_location`` is patched and ``places_fn`` is injected.
"""

from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.pipeline.gps_enrichment import _make_short_ost_text_fn
from gomoveshift.pipeline.title_builder import DominantLocation

# Synthetic on-screen fixes (t, lat, lon) for the clip.
_TRIMMED = [
    (640.0, 37.1630, -80.3200),
    (660.0, 37.1670, -80.3160),
    (678.0, 37.1690, -80.3110),
]
_SHORT = SimpleNamespace(rank=1, start_sec=640.0, end_sec=678.0)
_BRUSH_HARBOR = ("Brush Harbor", 37.1843, -80.2909)  # 2.46 km from the nearest fix
_DEN_HILL = ("Den Hill", 37.1557, -80.3131)  # ~1 km from the nearest fix


def _dom(name):
    return DominantLocation(county="Montgomery County", state="Virginia", name=name)


def _run(name, places_fn):
    fn = _make_short_ost_text_fn(object(), poi_source=lambda _b: [], places_fn=places_fn)
    with patch("gomoveshift.pipeline.gps_enrichment.dominant_location", return_value=_dom(name)):
        return fn(_TRIMMED, _SHORT)


def test_settlement_off_the_trace_burns_the_county():
    assert _run("Brush Harbor", lambda _q: [_BRUSH_HARBOR]) == "Montgomery County, Virginia"


def test_settlement_on_the_trace_is_kept():
    assert _run("Den Hill", lambda _q: [_DEN_HILL]) == "Den Hill, Virginia"


def test_nearest_same_name_candidate_wins():
    far_twin = ("Den Hill", 38.5, -78.0)
    assert _run("Den Hill", lambda _q: [far_twin, _DEN_HILL]) == "Den Hill, Virginia"


def test_other_names_never_verify():
    assert _run("Brush Harbor", lambda _q: [_DEN_HILL]) == "Montgomery County, Virginia"


def test_lookup_scoped_to_voted_county():
    seen = []
    _run("Den Hill", lambda q: seen.append(q) or [_DEN_HILL])
    assert seen == ["Den Hill, Montgomery County, Virginia"]


def test_geocoder_failure_fails_closed_to_county():
    def _boom(_q):
        raise RuntimeError("nominatim down")

    assert _run("Den Hill", _boom) == "Montgomery County, Virginia"


def test_county_only_vote_needs_no_lookup():
    def _never(_q):
        raise AssertionError("county label must not be looked up")

    assert _run(None, _never) == "Montgomery County, Virginia"


def test_no_places_fn_keeps_the_vote():
    """Callers that do not wire a forward geocoder keep today's label."""
    assert _run("Brush Harbor", None) == "Brush Harbor, Virginia"


def test_drive_and_single_video_callers_wire_the_forward_geocoder():
    """Both production call sites pass ``places_fn`` so the check is live."""
    import inspect

    from gomoveshift.pipeline import shorts_extractor
    from gomoveshift.pipeline.drive_phases import phase_bc_day_first

    for mod in (phase_bc_day_first, shorts_extractor):
        src = inspect.getsource(mod)
        assert "_make_short_ost_text_fn(geocoder, places_fn=" in src, mod.__name__
