"""Unit tests for chapter_town_events -- persisted chapter list ->
recurring LOCATION lower-third (place_label, event_time_sec) events."""

from __future__ import annotations

from gomoveshift.pipeline.chapters.marker_merger import (
    _TERRAIN_BAND_SUFFIX,
    chapter_town_events,
)


def test_strips_route_suffix():
    # Two commas: _format_location's "Town, County" plus the route suffix.
    chapters = [{"timestamp": "5:00", "title": "Entering Riverside, Blue Ridge County, Highway 9"}]
    assert chapter_town_events(chapters) == [("Riverside, Blue Ridge County", 300.0)]


def test_keeps_county_when_no_route_suffix_present():
    """Regression: a first-comma split would wrongly drop the county here --
    _format_location's "Town, County" is the ENTIRE label, no route suffix.
    """
    chapters = [{"timestamp": "5:00", "title": "Entering Beaufort, Beaufort County"}]
    assert chapter_town_events(chapters) == [("Beaufort, Beaufort County", 300.0)]


def test_strips_route_suffix_after_county():
    chapters = [{"timestamp": "5:00", "title": "Entering Beaufort, Beaufort County, US-17"}]
    assert chapter_town_events(chapters) == [("Beaufort, Beaufort County", 300.0)]


def test_town_with_no_county_or_route_is_unchanged():
    chapters = [{"timestamp": "5:00", "title": "Entering Yemassee"}]
    assert chapter_town_events(chapters) == [("Yemassee", 300.0)]


def test_strips_each_terrain_band_suffix():
    chapters = [
        {"timestamp": "1:00", "title": "Entering Alpine Town — Mountain Climb"},
        {"timestamp": "2:00", "title": "Entering Speedville — Motorway"},
        {"timestamp": "3:00", "title": "Entering Shoreburg — Coastal"},
    ]
    assert chapter_town_events(chapters) == [
        ("Alpine Town", 60.0),
        ("Speedville", 120.0),
        ("Shoreburg", 180.0),
    ]


def test_strips_route_suffix_and_terrain_band_together():
    chapters = [
        {"timestamp": "4:00", "title": "Entering Beaufort, Beaufort County, US-17 — Coastal"}
    ]
    assert chapter_town_events(chapters) == [("Beaufort, Beaufort County", 240.0)]


def test_excludes_start_and_finish_chapters():
    chapters = [
        {"timestamp": "0:00", "title": "Start - Somewhere"},
        {"timestamp": "10:00", "title": "Finish - Elsewhere"},
    ]
    assert chapter_town_events(chapters) == []


def test_excludes_en_route_padding_chapters():
    chapters = [
        {"timestamp": "1:00", "title": "En route"},
        {"timestamp": "2:00", "title": "En route — Somewhere"},
    ]
    assert chapter_town_events(chapters) == []


def test_parses_hms_and_ms_timestamps():
    chapters = [
        {"timestamp": "1:02:03", "title": "Entering Farville"},
        {"timestamp": "4:05", "title": "Entering Nearville"},
    ]
    assert chapter_town_events(chapters) == [
        ("Farville", 3723.0),
        ("Nearville", 245.0),
    ]


def test_empty_none_and_garbage_input_returns_empty_list():
    assert chapter_town_events([]) == []
    assert chapter_town_events(None) == []
    assert chapter_town_events("garbage") == []
    assert chapter_town_events([{"title": "Entering Nowhere", "timestamp": "not-a-time"}]) == []
    assert chapter_town_events([{"no_title_key": True}]) == []


# ── county-less labels: the route suffix must be dropped structurally ────────
#
# The comma-COUNTING rule above assumes _format_location always contributes a
# county, but it returns a bare name whenever location.county is falsy — every
# synthetic interval-fill Location, and any ungeocoded POI. Then parts[:2] keeps
# the ROUTE as if it were the county and burns "Town, US-17".


def test_strips_route_ref_suffix_from_county_less_label():
    chapters = [{"timestamp": "5:00", "title": "Entering Yemassee, US-17"}]
    assert chapter_town_events(chapters) == [("Yemassee", 300.0)]


def test_strips_named_route_suffix_from_county_less_label():
    chapters = [{"timestamp": "5:00", "title": "Entering Yemassee, Blue Ridge Parkway"}]
    assert chapter_town_events(chapters) == [("Yemassee", 300.0)]


def test_county_is_never_mistaken_for_a_route_suffix():
    # The structural drop must not fire on a real county: only the last part,
    # and only when it passes the road-quality gate.
    chapters = [
        {"timestamp": "5:00", "title": "Entering Kingsport, Sullivan County"},
        {"timestamp": "9:00", "title": "Entering Dingle, County Kerry"},
    ]
    assert chapter_town_events(chapters) == [
        ("Kingsport, Sullivan County", 300.0),
        ("Dingle, County Kerry", 540.0),
    ]


# ── adjacent duplicates ─────────────────────────────────────────────────────
#
# MarkerMerger._collapse_consecutive_duplicates compares the DECORATED title,
# and the decorations (route suffix, terrain band) are exactly what this
# function strips — so two chapters that differ only by decoration survive it
# and then reduce to the same burned label.


def test_collapses_adjacent_duplicate_labels_after_decoration_strip():
    # The production pair: Johnson City burned twice, 76s apart. The band
    # suffix is read from the module rather than re-literalled.
    band = _TERRAIN_BAND_SUFFIX["motorway"]
    chapters = [
        {"timestamp": "6:44", "title": f"Entering Johnson City, Washington County{band}"},
        {"timestamp": "8:00", "title": "Entering Johnson City, Washington County"},
    ]
    assert chapter_town_events(chapters) == [("Johnson City, Washington County", 404.0)]


def test_non_adjacent_repeat_is_kept():
    # Adjacent-only: a drive genuinely re-entering a town later must re-label.
    chapters = [
        {"timestamp": "5:00", "title": "Entering Abingdon, Washington County"},
        {"timestamp": "10:00", "title": "Entering Marion, Smyth County"},
        {"timestamp": "15:00", "title": "Entering Abingdon, Washington County"},
    ]
    labels = [label for label, _ in chapter_town_events(chapters)]
    assert labels == [
        "Abingdon, Washington County",
        "Marion, Smyth County",
        "Abingdon, Washington County",
    ]
