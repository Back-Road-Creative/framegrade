"""The longitude-local "day" helper is single-sourced from gps_journey.

The video pipeline used to reimplement the ``longitude / 15° → local solar day``
algorithm in three places (``_journey_filter.local_day``, ``day_grouper``'s
``_DEGREES_LONGITUDE_PER_HOUR`` shift, and ``generator._longitude_local_start``).
These tests fail if any of them stops importing the canonical
``gps_journey.core.models`` helpers and reintroduces its own copy — including a
copy that rounds differently — so the algorithm stays defined in exactly one
place.
"""

from datetime import date, datetime

import gomoveshift.pipeline.chapters._journey_filter as journey_filter_mod
import gomoveshift.pipeline.chapters.generator as generator_mod
import gomoveshift.pipeline.day_grouper as day_grouper_mod
from gps_journey.core.models import Journey, Waypoint, local_day, to_local

# (timestamp UTC, longitude, expected local date, expected local datetime).
# Spans positive/negative longitudes and UTC day boundaries. A reimplementation
# using truncation / floor instead of banker's ``round(lon / 15)`` — or a wrong
# sign — diverges on at least one row (e.g. -82.5/15 = -5.5 rounds to the even
# -6, not -5; 174/15 = 11.6 rounds to 12 and tips the day forward).
_SAMPLES = [
    (datetime(2026, 2, 21, 3, 30), -82.5, date(2026, 2, 20), datetime(2026, 2, 20, 21, 30)),
    (datetime(2026, 2, 20, 23, 45), -82.5, date(2026, 2, 20), datetime(2026, 2, 20, 17, 45)),
    (datetime(2026, 6, 14, 20, 0), 174.0, date(2026, 6, 15), datetime(2026, 6, 15, 8, 0)),
    (datetime(2026, 6, 15, 12, 0), 0.0, date(2026, 6, 15), datetime(2026, 6, 15, 12, 0)),
]


def test_journey_filter_local_day_is_canonical():
    """``_journey_filter`` re-exports the canonical helper, not its own copy."""
    assert journey_filter_mod.local_day is local_day


def test_day_grouper_references_canonical_to_local():
    """``day_grouper`` shifts via the canonical primitive, not a private const."""
    assert getattr(day_grouper_mod, "to_local", None) is to_local
    assert not hasattr(day_grouper_mod, "_DEGREES_LONGITUDE_PER_HOUR")


def test_generator_references_canonical_to_local():
    """``generator`` shifts the recording datetime via the canonical primitive."""
    assert getattr(generator_mod, "to_local", None) is to_local


def test_canonical_helpers_lock_expected_semantics():
    """Expected local date/datetime for each sample (guards a rounding change)."""
    for ts, lon, want_date, want_dt in _SAMPLES:
        assert local_day(ts, lon) == want_date
        assert to_local(ts, lon) == want_dt


def test_journey_filter_local_day_matches_canonical():
    for ts, lon, want_date, _want_dt in _SAMPLES:
        assert journey_filter_mod.local_day(ts, lon) == want_date == local_day(ts, lon)


def test_generator_longitude_local_start_matches_canonical():
    for ts, lon, _want_date, want_dt in _SAMPLES:
        wp = Waypoint(latitude=0.0, longitude=lon, timestamp=ts)
        journey = Journey(start_date=ts, end_date=ts, waypoints=(wp,))
        assert generator_mod._longitude_local_start(journey) == want_dt == to_local(ts, lon)
