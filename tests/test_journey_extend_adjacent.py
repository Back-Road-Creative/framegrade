"""A drive on the day after a trip ends joins that trip's journey.

Ace Basin (2026-02-21) was the second day of the trip already journaled as
``sc-2026`` (trip 2026-02-20..2026-02-20). No trip range held the run date, so
the drive hook created a separate draft ``usa-2026-02-21`` instead of adding
day 2 to ``sc-2026``.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.journal_publisher import load_journey
from gomoveshift.pipeline.site_sync.journey_by_date import (
    AmbiguousJourneyError,
    extend_adjacent_journey,
)

_DAY1 = """---
title: "Lowcountry, 2026"
draft: false
trip:
  start_date: '2026-02-20'
  end_date: '2026-02-20'
  duration_days: 1
days:
- day_number: 1
  date: '2026-02-20'
  day_of_week: Friday
  heading: "Beaufort"
  narrative: ""
  videos:
  - youtube_id: GOhM9X45DHQ
    title: Beaufort County Scenic Loop, SC
    duration: '13:22'
video_count: 1
---
body text
"""


def _site(tmp_path, **journeys):
    site = tmp_path / "site"
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    for slug, text in journeys.items():
        (jdir / f"{slug}.md").write_text(text)
    return site


def test_trip_ending_the_day_before_is_extended_by_one_day(tmp_path):
    site = _site(tmp_path, **{"sc-2026": _DAY1})
    assert extend_adjacent_journey(site, "2026-02-21") == "sc-2026"
    doc, body = load_journey(site / "content" / "journeys" / "sc-2026.md")
    assert doc["trip"]["start_date"] == "2026-02-20"
    assert doc["trip"]["end_date"] == "2026-02-21"
    assert doc["trip"]["duration_days"] == 2
    assert [d["date"] for d in doc["days"]] == ["2026-02-20", "2026-02-21"]
    day2 = doc["days"][1]
    assert day2["day_number"] == 2
    assert day2["day_of_week"] == "Saturday"
    assert (day2["heading"], day2["narrative"], day2["videos"]) == ("", "", [])
    # Day 1 and everything around it is untouched.
    assert doc["days"][0]["videos"][0]["youtube_id"] == "GOhM9X45DHQ"
    assert doc["title"] == "Lowcountry, 2026"
    assert doc["video_count"] == 1
    assert body == "body text\n"


def test_no_adjacent_trip_returns_none_and_writes_nothing(tmp_path):
    site = _site(tmp_path, **{"sc-2026": _DAY1})
    path = site / "content" / "journeys" / "sc-2026.md"
    assert extend_adjacent_journey(site, "2026-02-22") is None
    assert path.read_text() == _DAY1


def test_trip_starting_the_day_after_is_not_extended_backwards(tmp_path):
    # Prepending would renumber the published day 1.
    site = _site(tmp_path, **{"sc-2026": _DAY1})
    assert extend_adjacent_journey(site, "2026-02-19") is None
    assert (site / "content" / "journeys" / "sc-2026.md").read_text() == _DAY1


def test_two_trips_ending_the_day_before_are_refused(tmp_path):
    other = _DAY1.replace("Lowcountry", "Other")
    site = _site(tmp_path, **{"sc-2026": _DAY1, "other-2026": other})
    with pytest.raises(AmbiguousJourneyError) as exc:
        extend_adjacent_journey(site, "2026-02-21")
    assert "sc-2026" in str(exc.value) and "other-2026" in str(exc.value)
    assert (site / "content" / "journeys" / "sc-2026.md").read_text() == _DAY1


def test_trip_without_duration_days_gains_none(tmp_path):
    text = _DAY1.replace("  duration_days: 1\n", "")
    site = _site(tmp_path, **{"sc-2026": text})
    assert extend_adjacent_journey(site, "2026-02-21") == "sc-2026"
    doc, _ = load_journey(site / "content" / "journeys" / "sc-2026.md")
    assert doc["trip"]["end_date"] == "2026-02-21"
    assert "duration_days" not in doc["trip"]


def test_unparsable_run_date_returns_none(tmp_path):
    site = _site(tmp_path, **{"sc-2026": _DAY1})
    assert extend_adjacent_journey(site, "not-a-date") is None
