"""Tests for day_grouper — splits an IngestPlan's episodes into per-day
batches so the day-first pipeline architecture can process each day
independently.

A "day" is the calendar date of the episode's GPX first fix, converted to
longitude-local time (the same satellite-UTC clock every LOCATION surface
keys off). The naive container ``creation_time`` is only a fallback when no
GPX is available. Episodes that cross midnight stay with their start date —
they are not split. Day batches are returned in chronological order, and
episodes within each batch preserve their original order.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock


def _write_gpx(path: Path, first_iso: str, lon: float, lat: float = 34.0) -> Path:
    """Write a minimal one-fix GPX file and return its path."""
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<gpx version="1.1"><trk><trkseg>'
        f'<trkpt lat="{lat}" lon="{lon}"><time>{first_iso}</time></trkpt>'
        "</trkseg></trk></gpx>\n",
        encoding="utf-8",
    )
    return path


def _make_episode(
    index: int,
    start: datetime,
    duration_min: float,
    gpx_path: Path = None,
    concat_path: Path = None,
):
    """Build an Episode-like object with just enough fields for day_grouper."""
    rec = MagicMock()
    rec.creation_time = start
    rec.total_duration = duration_min * 60

    ep = MagicMock()
    ep.index = index
    ep.recordings = [rec]
    ep.creation_time = start
    ep.total_duration = duration_min * 60
    ep.total_duration_min = duration_min
    ep.date = start.strftime("%Y-%m-%d")
    ep.gpx_path = gpx_path
    ep.concat_path = concat_path
    return ep


def _make_plan(episodes: list):
    plan = MagicMock()
    plan.episodes = episodes
    return plan


class TestGroupEpisodesByDay:
    def test_empty_plan_returns_empty_list(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        plan = _make_plan([])
        assert group_episodes_by_day(plan) == []

    def test_single_episode_single_day(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep = _make_episode(1, datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc), 30)
        batches = group_episodes_by_day(_make_plan([ep]))

        assert len(batches) == 1
        assert batches[0].date == "2026-02-20"
        assert batches[0].episodes == [ep]

    def test_two_episodes_same_day_single_batch(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep1 = _make_episode(1, datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc), 30)
        ep2 = _make_episode(2, datetime(2026, 2, 20, 14, 0, tzinfo=timezone.utc), 45)
        batches = group_episodes_by_day(_make_plan([ep1, ep2]))

        assert len(batches) == 1
        assert batches[0].date == "2026-02-20"
        assert batches[0].episodes == [ep1, ep2]

    def test_episodes_across_three_days(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep1 = _make_episode(1, datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc), 30)
        ep2 = _make_episode(2, datetime(2026, 2, 21, 9, 0, tzinfo=timezone.utc), 45)
        ep3 = _make_episode(3, datetime(2026, 2, 21, 15, 0, tzinfo=timezone.utc), 20)
        ep4 = _make_episode(4, datetime(2026, 2, 22, 11, 0, tzinfo=timezone.utc), 60)

        batches = group_episodes_by_day(_make_plan([ep1, ep2, ep3, ep4]))

        assert len(batches) == 3
        assert [b.date for b in batches] == ["2026-02-20", "2026-02-21", "2026-02-22"]
        assert batches[0].episodes == [ep1]
        assert batches[1].episodes == [ep2, ep3]
        assert batches[2].episodes == [ep4]

    def test_batches_are_in_chronological_order(self):
        """Even if episodes are passed out of order, batches come back sorted."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep_late = _make_episode(3, datetime(2026, 2, 22, 10, 0, tzinfo=timezone.utc), 30)
        ep_early = _make_episode(1, datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc), 30)
        ep_mid = _make_episode(2, datetime(2026, 2, 21, 10, 0, tzinfo=timezone.utc), 30)

        batches = group_episodes_by_day(_make_plan([ep_late, ep_early, ep_mid]))

        assert [b.date for b in batches] == ["2026-02-20", "2026-02-21", "2026-02-22"]

    def test_midnight_spanning_episode_belongs_to_start_date(self):
        """An episode that starts on Feb 20 at 23:45 and runs 30 min into
        Feb 21 must be assigned to Feb 20 (its start date), not split."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep_spanning = _make_episode(
            1,
            datetime(2026, 2, 20, 23, 45, tzinfo=timezone.utc),
            30,
        )
        ep_next_day = _make_episode(
            2,
            datetime(2026, 2, 21, 10, 0, tzinfo=timezone.utc),
            30,
        )
        batches = group_episodes_by_day(_make_plan([ep_spanning, ep_next_day]))

        assert len(batches) == 2
        assert batches[0].date == "2026-02-20"
        assert batches[0].episodes == [ep_spanning]
        assert batches[1].date == "2026-02-21"
        assert batches[1].episodes == [ep_next_day]

    def test_episode_order_preserved_within_batch(self):
        """Episodes within a day batch stay in their input order (which is
        also their chronological order from the ingest plan)."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep1 = _make_episode(1, datetime(2026, 2, 20, 9, 0, tzinfo=timezone.utc), 15)
        ep2 = _make_episode(2, datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc), 15)
        ep3 = _make_episode(3, datetime(2026, 2, 20, 11, 0, tzinfo=timezone.utc), 15)

        batches = group_episodes_by_day(_make_plan([ep1, ep2, ep3]))

        assert len(batches) == 1
        assert batches[0].episodes == [ep1, ep2, ep3]

    def test_episode_without_creation_time_uses_date_string(self):
        """If an episode has no creation_time (edge case in older data),
        fall back to parsing the `date` field."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep = MagicMock()
        ep.index = 1
        ep.creation_time = None
        ep.date = "2026-02-20"
        ep.total_duration = 1800
        ep.total_duration_min = 30
        ep.gpx_path = None
        ep.concat_path = None

        batches = group_episodes_by_day(_make_plan([ep]))
        assert len(batches) == 1
        assert batches[0].date == "2026-02-20"

    def test_episode_with_neither_creation_time_nor_date_is_skipped(self):
        """Defensive: if an episode has no date information at all, it's
        skipped (with a warning) rather than crashing the grouper."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep_bad = MagicMock()
        ep_bad.index = 1
        ep_bad.creation_time = None
        ep_bad.date = ""
        ep_bad.total_duration = 1800

        ep_good = _make_episode(
            2,
            datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc),
            30,
        )
        batches = group_episodes_by_day(_make_plan([ep_bad, ep_good]))

        # Only the good episode is batched; the bad one is silently skipped
        assert len(batches) == 1
        assert batches[0].episodes == [ep_good]


class TestDayBatchProperties:
    def test_total_duration_sums_episodes(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep1 = _make_episode(1, datetime(2026, 2, 20, 9, 0, tzinfo=timezone.utc), 30)
        ep2 = _make_episode(2, datetime(2026, 2, 20, 11, 0, tzinfo=timezone.utc), 45)
        batches = group_episodes_by_day(_make_plan([ep1, ep2]))

        # 30 + 45 = 75 min = 4500 sec
        assert batches[0].total_duration == 4500

    def test_start_time_is_first_episode(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        start1 = datetime(2026, 2, 20, 9, 0, tzinfo=timezone.utc)
        ep1 = _make_episode(1, start1, 30)
        ep2 = _make_episode(2, datetime(2026, 2, 20, 15, 0, tzinfo=timezone.utc), 30)
        batches = group_episodes_by_day(_make_plan([ep1, ep2]))
        assert batches[0].start_time == start1

    def test_end_time_is_last_episode_start_plus_duration(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep1 = _make_episode(1, datetime(2026, 2, 20, 9, 0, tzinfo=timezone.utc), 30)
        last_start = datetime(2026, 2, 20, 15, 0, tzinfo=timezone.utc)
        ep2 = _make_episode(2, last_start, 45)
        batches = group_episodes_by_day(_make_plan([ep1, ep2]))

        expected_end = last_start + timedelta(minutes=45)
        assert batches[0].end_time == expected_end

    def test_gpx_paths_preserve_episode_order(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep1 = _make_episode(
            1,
            datetime(2026, 2, 20, 9, 0, tzinfo=timezone.utc),
            30,
            gpx_path=Path("/fake/ep1.gpx"),
        )
        ep2 = _make_episode(
            2,
            datetime(2026, 2, 20, 11, 0, tzinfo=timezone.utc),
            30,
            gpx_path=Path("/fake/ep2.gpx"),
        )
        batches = group_episodes_by_day(_make_plan([ep1, ep2]))

        assert batches[0].gpx_paths == [Path("/fake/ep1.gpx"), Path("/fake/ep2.gpx")]

    def test_concat_paths_preserve_episode_order(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep1 = _make_episode(
            1,
            datetime(2026, 2, 20, 9, 0, tzinfo=timezone.utc),
            30,
            concat_path=Path("/fake/ep1.mp4"),
        )
        ep2 = _make_episode(
            2,
            datetime(2026, 2, 20, 11, 0, tzinfo=timezone.utc),
            30,
            concat_path=Path("/fake/ep2.mp4"),
        )
        batches = group_episodes_by_day(_make_plan([ep1, ep2]))

        assert batches[0].concat_paths == [Path("/fake/ep1.mp4"), Path("/fake/ep2.mp4")]

    def test_gpx_paths_filters_none(self):
        """Episodes without a gpx_path don't appear in the batch's gpx_paths list."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep1 = _make_episode(
            1,
            datetime(2026, 2, 20, 9, 0, tzinfo=timezone.utc),
            30,
            gpx_path=Path("/fake/ep1.gpx"),
        )
        ep2 = _make_episode(
            2,
            datetime(2026, 2, 20, 11, 0, tzinfo=timezone.utc),
            30,
            gpx_path=None,
        )
        batches = group_episodes_by_day(_make_plan([ep1, ep2]))

        assert batches[0].gpx_paths == [Path("/fake/ep1.gpx")]

    def test_summary_includes_date_and_episode_count(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep1 = _make_episode(1, datetime(2026, 2, 20, 9, 0, tzinfo=timezone.utc), 30)
        ep2 = _make_episode(2, datetime(2026, 2, 20, 11, 0, tzinfo=timezone.utc), 45)
        batches = group_episodes_by_day(_make_plan([ep1, ep2]))

        s = batches[0].summary()
        assert "2026-02-20" in s
        assert "2" in s  # episode count
        assert "75" in s or "75.0" in s  # total duration in minutes


class TestMultiDayDaySplit:
    """Realistic fixture: 8 episodes span Feb 20, 21, 22, 24 (4 days).
    Validates the grouper produces the expected day batches for a known drive."""

    def test_multi_day_split(self):
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        # Episode dates span four days (Feb 20, 21, 22, 24)
        episodes = [
            _make_episode(1, datetime(2026, 2, 20, 16, 26, tzinfo=timezone.utc), 27),
            _make_episode(2, datetime(2026, 2, 20, 17, 48, tzinfo=timezone.utc), 48),
            _make_episode(3, datetime(2026, 2, 21, 16, 3, tzinfo=timezone.utc), 44),
            _make_episode(4, datetime(2026, 2, 21, 17, 11, tzinfo=timezone.utc), 95),
            _make_episode(5, datetime(2026, 2, 21, 18, 48, tzinfo=timezone.utc), 58),
            _make_episode(6, datetime(2026, 2, 22, 19, 29, tzinfo=timezone.utc), 25),
            _make_episode(7, datetime(2026, 2, 24, 15, 0, tzinfo=timezone.utc), 117),
            _make_episode(8, datetime(2026, 2, 24, 17, 7, tzinfo=timezone.utc), 46),
        ]
        batches = group_episodes_by_day(_make_plan(episodes))

        assert len(batches) == 4
        assert [b.date for b in batches] == [
            "2026-02-20",
            "2026-02-21",
            "2026-02-22",
            "2026-02-24",
        ]
        assert len(batches[0].episodes) == 2  # Feb 20: ep 1, 2
        assert len(batches[1].episodes) == 3  # Feb 21: ep 3, 4, 5
        assert len(batches[2].episodes) == 1  # Feb 22: ep 6
        assert len(batches[3].episodes) == 2  # Feb 24: ep 7, 8


class TestGpxLocalDayKey:
    """The day key comes from the GPX first fix shifted to longitude-local
    time, not the naive camera-clock creation_time."""

    def test_us_east_evening_drive_groups_into_local_day_not_utc(self, tmp_path):
        """A US-East evening drive: GPX/creation_time are 00:30 UTC (19:30
        local on Feb 20). The naive UTC creation_time would misgroup it into
        Feb 21; GPX-local time keeps it on Feb 20."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        gpx = _write_gpx(tmp_path / "evening.gpx", "2026-02-21T00:30:00Z", lon=-80.5)
        ep = _make_episode(
            1,
            datetime(2026, 2, 21, 0, 30, tzinfo=timezone.utc),
            30,
            gpx_path=gpx,
        )
        batches = group_episodes_by_day(_make_plan([ep]))

        assert len(batches) == 1
        assert batches[0].date == "2026-02-20"

    def test_gpx_local_key_preferred_over_creation_time(self, tmp_path):
        """When a GPX fix is present it wins over creation_time entirely."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        gpx = _write_gpx(tmp_path / "a.gpx", "2026-03-10T02:00:00Z", lon=-75.0)
        # creation_time says Mar 11 (a wrong camera clock); GPX-local = Mar 9 21:00
        ep = _make_episode(
            1,
            datetime(2026, 3, 11, 8, 0, tzinfo=timezone.utc),
            20,
            gpx_path=gpx,
        )
        batches = group_episodes_by_day(_make_plan([ep]))
        assert batches[0].date == "2026-03-09"

    def test_clock_disagreement_emits_loud_warning(self, tmp_path, caplog):
        """A GPX-local vs creation_time date mismatch logs a hard warning."""
        import logging

        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        gpx = _write_gpx(tmp_path / "b.gpx", "2026-02-21T00:30:00Z", lon=-80.5)
        ep = _make_episode(
            1,
            datetime(2026, 2, 21, 0, 30, tzinfo=timezone.utc),
            30,
            gpx_path=gpx,
        )
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.day_grouper"):
            group_episodes_by_day(_make_plan([ep]))

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings, "expected a clock-disagreement warning"
        assert any(
            "2026-02-20" in r.getMessage() and "2026-02-21" in r.getMessage() for r in warnings
        )

    def test_no_warning_when_dates_agree(self, tmp_path, caplog):
        """No warning when GPX-local and creation_time land on the same day."""
        import logging

        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        # Midday UTC at lon 0 → local == UTC → both Feb 20.
        gpx = _write_gpx(tmp_path / "c.gpx", "2026-02-20T12:00:00Z", lon=0.0)
        ep = _make_episode(
            1,
            datetime(2026, 2, 20, 12, 0, tzinfo=timezone.utc),
            30,
            gpx_path=gpx,
        )
        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.day_grouper"):
            batches = group_episodes_by_day(_make_plan([ep]))
        assert batches[0].date == "2026-02-20"
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_near_midnight_first_fix_emits_warning(self, tmp_path, caplog):
        """A first fix within 2 h of local midnight warns — the whole-hour
        longitude→offset approximation can tip the calendar day there."""
        import logging

        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        # lon -75 → round(-75/15) = -5h. 04:30 UTC → 23:30 local Feb 20 (30 min
        # before midnight). creation_time is the SAME local day, so no
        # clock-disagreement warning fires — isolating the near-midnight warn.
        gpx = _write_gpx(tmp_path / "nm.gpx", "2026-02-21T04:30:00Z", lon=-75.0)
        ep = _make_episode(1, datetime(2026, 2, 20, 23, 30), 30, gpx_path=gpx)

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.day_grouper"):
            batches = group_episodes_by_day(_make_plan([ep]))

        assert batches[0].date == "2026-02-20"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("NEAR-MIDNIGHT" in r.getMessage() for r in warnings)

    def test_no_near_midnight_warning_for_midday_fix(self, tmp_path, caplog):
        """A comfortably-midday local fix emits no near-midnight warning."""
        import logging

        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        # lon -75 → -5h. 17:00 UTC → 12:00 local — far from any midnight.
        gpx = _write_gpx(tmp_path / "mid.gpx", "2026-02-20T17:00:00Z", lon=-75.0)
        ep = _make_episode(1, datetime(2026, 2, 20, 12, 0), 30, gpx_path=gpx)

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.day_grouper"):
            group_episodes_by_day(_make_plan([ep]))

        assert not any(
            "NEAR-MIDNIGHT" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        )

    def test_falls_back_to_creation_time_without_gpx(self):
        """No GPX → the naive creation_time date is used as before."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep = _make_episode(1, datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc), 30)
        batches = group_episodes_by_day(_make_plan([ep]))
        assert batches[0].date == "2026-02-20"

    def test_unreadable_gpx_falls_back_to_creation_time(self, tmp_path):
        """A missing/garbage GPX degrades to creation_time, never crashes."""
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day

        ep = _make_episode(
            1,
            datetime(2026, 2, 20, 10, 0, tzinfo=timezone.utc),
            30,
            gpx_path=tmp_path / "does-not-exist.gpx",
        )
        batches = group_episodes_by_day(_make_plan([ep]))
        assert batches[0].date == "2026-02-20"
