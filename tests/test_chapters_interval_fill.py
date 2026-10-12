"""Interval-fill pass validation + real-time stamping.

`NominatimChapterSource.fill_interval_chapters` re-reads the geocoded pool to
fill chapter gaps. Two properties the earlier passes guarantee must also hold
here:

  1. The chosen placename must survive `is_valid_chapter_placename` — a
     tourism/business name (e.g. "...Diner", "...Hotel") must not leak back in
     as an "Entering X" chapter.
  2. The stamped time must be the chosen nearest sample's REAL video time, not
     the synthetic evenly-spaced grid time.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from gps_journey.core.models import Location

from gomoveshift.pipeline.chapters.nominatim_source import NominatimChapterSource


def _source(detailed: dict | None = None) -> NominatimChapterSource:
    geocoder = MagicMock()
    # Default: no road available at the fallback coordinate, so a rejected/
    # unusable pick is dropped rather than substituted by a synthetic road
    # chapter. Pass ``detailed`` to exercise the road-substitute branch.
    geocoder.reverse_geocode_detailed.return_value = detailed if detailed is not None else {}
    return NominatimChapterSource(geocoder=geocoder, cache=None)


def test_interval_fill_rejects_business_placename():
    source = _source()
    # Single geocoded sample whose "city" is a business name the validator
    # rejects (contains the "diner" tourism keyword).
    all_geocoded = [(80.0, Location(latitude=52.0, longitude=-9.5, city="Riverside Diner"))]

    result = source.fill_interval_chapters(
        named=[],
        all_geocoded=all_geocoded,
        video_duration=100.0,
        target=1,
    )

    names = {loc.city or loc.locality or loc.name or loc.suburb for _, loc in result}
    assert "Riverside Diner" not in names


def test_interval_fill_stamps_real_sample_time_not_grid_time():
    source = _source()
    # Grid time for target=1 over a 100s video is 50.0; the real sample sits at
    # 80.0. The added chapter must carry 80.0.
    real_time = 80.0
    grid_time = 50.0
    all_geocoded = [(real_time, Location(latitude=52.0, longitude=-9.5, city="Killarney"))]

    result = source.fill_interval_chapters(
        named=[],
        all_geocoded=all_geocoded,
        video_duration=100.0,
        target=1,
    )

    killarney_times = [t for t, loc in result if loc.city == "Killarney"]
    assert killarney_times == [real_time]
    assert grid_time not in killarney_times


# ── road-substitute branch: quality gate + OSM concurrency normalization ─────
#
# A burned superlong lower-third gets its road string from exactly here, and
# `is_valid_chapter_placename` does NOT screen it — it accepts "I 81",
# "I 81;US 58" and "Deerfield Lane" alike.


def _road_fill(detailed: dict, *, samples: int = 1, target: int = 1):
    """Run the interval fill with every sample forced down the road branch."""
    source = _source(detailed)
    # "Riverside Diner" is rejected by is_valid_chapter_placename, so the
    # placename pick always fails and the road substitute is consulted.
    all_geocoded = [
        (80.0 + i, Location(latitude=52.0, longitude=-9.5, city="Riverside Diner"))
        for i in range(samples)
    ]
    return source.fill_interval_chapters(
        named=[], all_geocoded=all_geocoded, video_duration=100.0, target=target
    )


def test_interval_fill_drops_residential_road_instead_of_labelling_it():
    # A minor residential way is not a chapter. The title path omits rather
    # than stamps one; the chapter path must agree instead of burning
    # "Deerfield Lane" as a lower third.
    result = _road_fill(
        {"road": "Deerfield Lane", "place_class": "highway", "match_type": "residential"}
    )
    assert result == []


def test_interval_fill_normalizes_osm_concurrency_ref():
    # "I 81;US 58" is OSM's concurrency join, not a display string. The ";" is
    # also the ffmpeg filtergraph chain separator and escape_ffmpeg_text does
    # not escape it, so it must never reach a drawtext label.
    result = _road_fill({"road": "I 81;US 58"})

    assert len(result) == 1
    name = result[0][1].name
    assert name == "I-81"
    assert ";" not in name


def test_interval_fill_rejected_road_does_not_consume_a_label_slot():
    # A road the quality gate rejects must not be recorded in used_labels: a
    # later sample resolving to a genuinely different label would otherwise be
    # blocked by a label that was never burned.
    result = _road_fill(
        {"road": "Deerfield Lane", "place_class": "highway", "match_type": "residential"},
        samples=2,
        target=2,
    )
    assert result == []
