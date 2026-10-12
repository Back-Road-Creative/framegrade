"""Chapter time-base correctness.

``Journey.with_waypoints`` keeps the ORIGINAL start/end dates, so after a
head/tail strip or an ``included_segments`` trim the journey's span stayed the
raw, untrimmed one. ``gps_duration = end_date - start_date`` was then measured
against footage no longer on-screen, and the ``video_duration / gps_duration``
scale that places every chapter drifted; Overpass ``num_samples`` (sized
off ``gps_duration``) was likewise off.

``_rebuild_journey`` recomputes the dates from the surviving waypoints while
preserving every other field, and the generator recomputes ``gps_duration``
after the trim. These tests lock that behavior.
"""

from datetime import datetime, timedelta

from gps_journey.core.models import Journey, Waypoint

from gomoveshift.pipeline.chapters.generator import (
    _rebuild_journey,
    _strip_null_island_waypoints,
    _strip_teleport_runs,
)

_START = datetime(2025, 1, 1, 0, 0, 0)


def _wp(lat: float, lon: float, secs: float) -> Waypoint:
    return Waypoint(
        latitude=lat, longitude=lon, timestamp=_START + timedelta(seconds=secs), location=None
    )


def _continuous_run(lat0, lon0, lat1, lon1, n, t0=0.0):
    out = []
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 0.0
        out.append(_wp(lat0 + (lat1 - lat0) * frac, lon0 + (lon1 - lon0) * frac, t0 + i))
    return out


class TestRebuildJourney:
    def test_dates_recomputed_from_surviving_waypoints(self):
        wps = _continuous_run(32.4, -80.4, 32.5, -80.5, 50, t0=100.0)
        # Original journey claims a much wider span than its waypoints cover.
        j = Journey(
            start_date=_START,
            end_date=_START + timedelta(seconds=9999),
            waypoints=tuple(wps),
            statistics=None,
            title="My Drive",
        )
        out = _rebuild_journey(j, wps[10:40])  # keep an interior slice
        assert out.start_date == wps[10].timestamp
        assert out.end_date == wps[39].timestamp
        assert len(out.waypoints) == 30
        # All non-waypoint/date fields are preserved (strictly safer than the
        # lossy _journey_filter constructor).
        assert out.title == "My Drive"

    def test_empty_keeps_original_dates(self):
        j = Journey(
            start_date=_START,
            end_date=_START + timedelta(seconds=60),
            waypoints=tuple(_continuous_run(1.0, 1.0, 2.0, 2.0, 5)),
            statistics=None,
        )
        out = _rebuild_journey(j, ())
        assert out.waypoints == ()
        assert out.start_date == _START  # unchanged — caller falls back to basic


class TestStripUpdatesDates:
    def test_teleport_strip_updates_start_date(self):
        boone = _continuous_run(36.20, -81.70, 36.21, -81.69, 15, t0=0.0)
        coast = _continuous_run(32.40, -80.40, 32.55, -80.55, 400, t0=15.0)
        j = Journey(
            start_date=_START,
            end_date=_START + timedelta(seconds=414),
            waypoints=tuple(boone + coast),
            statistics=None,
        )
        out = _strip_teleport_runs(j)
        # The 15 stripped leading fixes no longer anchor start_date.
        assert out.start_date == out.waypoints[0].timestamp
        assert out.start_date > _START
        # gps_duration now measures only the surviving coastal body.
        assert (out.end_date - out.start_date).total_seconds() == 399  # 400 pts, 1s apart

    def test_null_island_strip_updates_dates(self):
        wps = (
            _wp(0.0, 0.0, 0.0),  # null-island lead
            _wp(0.0, 0.0, 1.0),
            _wp(32.4, -80.4, 2.0),
            _wp(32.5, -80.5, 3.0),
        )
        j = Journey(
            start_date=_START,
            end_date=_START + timedelta(seconds=3),
            waypoints=wps,
            statistics=None,
        )
        out = _strip_null_island_waypoints(j)
        assert len(out.waypoints) == 2
        assert out.start_date == out.waypoints[0].timestamp  # the first real fix
        assert out.start_date == _START + timedelta(seconds=2)


class TestTemporalTeleportStrip:
    """The seam discriminator is km/SEC vs a physical ceiling, not raw
    km. A big jump over a long dropout (ferry/tunnel/bridge) is a legit traverse;
    the same km over ~1 s is a teleport."""

    def test_long_dropout_jump_is_not_stripped(self):
        # 25+ km gap, but the trace pauses 10 min (a ferry/tunnel dropout). The
        # km-only rule would strip the head; the temporal rule keeps it because
        # the implied speed is a sane ~150 km/h.
        head = _continuous_run(36.20, -81.70, 36.30, -81.60, 15, t0=0.0)
        # 25 km jump over a 600 s dropout, then a continuous body.
        body = _continuous_run(36.55, -81.30, 36.70, -81.15, 400, t0=614.0)
        j = Journey(
            start_date=_START,
            end_date=_START + timedelta(seconds=1014),
            waypoints=tuple(head + body),
            statistics=None,
        )
        out = _strip_teleport_runs(j)
        assert len(out.waypoints) == len(head) + len(body)  # nothing stripped
        assert out.start_date == _START  # head preserved

    def test_sub_second_teleport_is_stripped(self):
        # Same 25+ km jump, but at 1 s spacing → physically impossible speed →
        # a true cold-start stale lock, stripped.
        boone = _continuous_run(36.20, -81.70, 36.21, -81.69, 15, t0=0.0)
        coast = _continuous_run(32.40, -80.40, 32.55, -80.55, 400, t0=15.0)
        j = Journey(
            start_date=_START,
            end_date=_START + timedelta(seconds=414),
            waypoints=tuple(boone + coast),
            statistics=None,
        )
        out = _strip_teleport_runs(j)
        assert out.start_date == out.waypoints[0].timestamp
        assert out.start_date > _START
