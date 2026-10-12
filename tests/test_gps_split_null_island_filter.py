"""Regression coverage for GPS-split candidates produced by null-island and stale-clock waypoints.

A 111-min source file produced split candidates at 92h and 5y because
GoPro GPS lock loss wrote (0,0) waypoints AND a stale-clock waypoint
landed years away from its neighbours. ffmpeg then sliced past EOF and
hit the 300s timeout on cross-mount I/O — Phase A aborted.

Three layers of defence are exercised here:
  1. ``_is_invalid_coord`` flags null-island and out-of-range coords.
  2. ``_sanitize_waypoints`` drops invalid coords and truncates the
     track at any inter-waypoint gap > 2h.
  3. ``find_gps_split_points`` accepts ``max_total_duration`` and
     discards candidates with elapsed beyond the source MP4 duration.
  4. ``_split_episode_at_points`` clamps + dedupes split offsets at the
     slice site so corrupt offsets can't reach ffmpeg.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.ingest.concatenator import _split_episode_at_points
from gomoveshift.pipeline.ingest.gopro_parser import (
    GPS_MAX_INTERWAYPOINT_GAP_SEC,
    _is_invalid_coord,
    _sanitize_waypoints,
    find_gps_split_points,
)


def _wp(lat, lon, t):
    return SimpleNamespace(latitude=lat, longitude=lon, timestamp=t)


T0 = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)


class TestIsInvalidCoord:
    def test_null_island_rejected(self):
        assert _is_invalid_coord(0.0, 0.0)
        assert _is_invalid_coord(0.0005, -0.0005)

    def test_out_of_range_rejected(self):
        assert _is_invalid_coord(91.0, 0.0)
        assert _is_invalid_coord(0.0, 181.0)
        assert _is_invalid_coord(-90.5, 0.0)

    def test_none_rejected(self):
        assert _is_invalid_coord(None, 0.0)
        assert _is_invalid_coord(0.0, None)

    def test_valid_coords_accepted(self):
        assert not _is_invalid_coord(35.5, -82.5)  # WNC
        assert not _is_invalid_coord(-33.86, 151.21)  # Sydney


class TestSanitizeWaypoints:
    def test_drops_null_island(self):
        wps = [
            _wp(35.5, -82.5, T0),
            _wp(0.0, 0.0, T0 + timedelta(seconds=10)),
            _wp(35.51, -82.51, T0 + timedelta(seconds=20)),
        ]
        out = _sanitize_waypoints(wps)
        assert len(out) == 2
        assert all(wp.latitude != 0.0 for wp in out)

    def test_truncates_at_large_time_gap(self):
        wps = [_wp(35.5, -82.5, T0 + timedelta(seconds=i * 60)) for i in range(5)]
        # Stale-clock waypoint 5 years in the future
        wps.append(_wp(35.6, -82.6, T0 + timedelta(days=365 * 5)))
        wps.append(_wp(35.7, -82.7, T0 + timedelta(days=365 * 5, seconds=60)))
        out = _sanitize_waypoints(wps)
        assert len(out) == 5  # later corrupt cluster discarded
        assert (out[-1].timestamp - out[0].timestamp).total_seconds() < 600

    def test_no_op_on_clean_track(self):
        wps = [_wp(35.5, -82.5, T0 + timedelta(seconds=i * 30)) for i in range(20)]
        assert _sanitize_waypoints(wps) == wps

    def test_gap_constant_under_3h(self):
        # Documented bound — anything looser would split lunch stops.
        assert GPS_MAX_INTERWAYPOINT_GAP_SEC <= 3 * 3600


class TestFindGpsSplitPointsCapping:
    """Drive find_gps_split_points end-to-end via a stubbed GPXParser
    so we exercise the real candidate-collection + clamping logic."""

    def _drive(self, waypoints, max_total_duration=None):
        journey = SimpleNamespace(waypoints=waypoints)
        with patch("gps_journey.core.gpx_parser.GPXParser") as parser_cls:
            parser_cls.return_value.parse.return_value = journey
            return find_gps_split_points(
                Path("/dev/null"),
                max_total_duration=max_total_duration,
            )

    def test_corrupt_cluster_no_split_past_eof(self):
        """Reproduction: a failing run produced split
        candidates at 92h and 5y on a 6650s source. With the fix, the
        sanitizer drops the corrupt cluster AND the max_total_duration
        cap discards anything that survives."""
        # 30 valid waypoints across the real ~111min recording
        valid = [
            _wp(35.5 + i * 0.001, -82.5 + i * 0.001, T0 + timedelta(seconds=i * 200))
            for i in range(30)
        ]
        # Inject the failure-mode waypoints that fired in production:
        # null-island + stale-clock cluster
        valid.append(_wp(0.0, 0.0, T0 + timedelta(seconds=5527 * 60)))
        valid.append(_wp(0.0, 0.0, T0 + timedelta(minutes=2_626_327)))
        splits = self._drive(valid, max_total_duration=6650.0)
        # Every returned split must fit inside the source duration
        assert all(0 <= s <= 6650.0 for s in splits), splits

    def test_max_total_duration_filters_surviving_outliers(self):
        # All-valid coords but one timestamp slightly over 2h gap — sanitizer
        # truncates at the gap, so result is naturally bounded.
        wps = [_wp(35.5, -82.5, T0 + timedelta(minutes=i)) for i in range(80)]
        wps.append(_wp(35.6, -82.6, T0 + timedelta(hours=10)))
        splits = self._drive(wps, max_total_duration=4800.0)
        assert all(s <= 4800.0 for s in splits)


class TestSplitEpisodeClamp:
    def test_corrupt_offset_clamped_to_episode_duration(self, tmp_path):
        source = tmp_path / "ep01.mp4"
        source.write_bytes(b"x")  # exists
        episode = SimpleNamespace(
            index=1,
            concat_path=source,
            total_duration=6650.0,
            recordings=[],
            date="2026-03-04",
            gpx_path=None,
            creation_time=None,
        )
        captured_cmds = []

        def fake_run(cmd, **_kw):
            # The split path also probes the source's keyframes; patching
            # concatenator.subprocess.run patches the shared stdlib module, so
            # keycut's ffprobe arrives here too. Refuse it — this test is about
            # the clamp, and unknown keyframes leave the offsets where it put them.
            if not cmd or cmd[0] != "ffmpeg":
                return SimpleNamespace(returncode=1, stdout=b"", stderr="")
            captured_cmds.append(cmd)
            return SimpleNamespace(returncode=0, stderr="")

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=fake_run,
        ):
            _split_episode_at_points(
                episode,
                split_points=[331_635.41, 9_999_999.0],  # both past EOF
                output_dir=tmp_path,
                stages_dir=tmp_path,
            )
        # Every -t value handed to ffmpeg must be inside the source.
        for cmd in captured_cmds:
            t_idx = cmd.index("-t")
            assert float(cmd[t_idx + 1]) <= 6650.0, cmd

    def test_subprocess_timeout_matches_concat_path(self, tmp_path):
        """A 220min, 150GB episode was correctly
        identified for splitting, but the legacy 300s timeout tripped
        before the legitimate 48-min slice could write its ~33GB to the
        cross-mount destination at ~100 MB/s. The split path now uses
        the same 3600s ceiling as the concat path above it."""
        source = tmp_path / "ep04.mp4"
        source.write_bytes(b"x")
        episode = SimpleNamespace(
            index=4,
            concat_path=source,
            total_duration=13200.0,  # 220 min
            recordings=[],
            date="2026-03-05",
            gpx_path=None,
            creation_time=None,
        )
        captured_timeouts = []

        def fake_run(cmd, **kw):
            # Only the ffmpeg CUT calls carry the split path's timeout ceiling;
            # keycut's keyframe ffprobe (routed here because patching
            # concatenator.subprocess.run patches the shared stdlib module)
            # carries its own and is not what this test pins.
            if not cmd or cmd[0] != "ffmpeg":
                return SimpleNamespace(returncode=1, stdout=b"", stderr="")
            captured_timeouts.append(kw.get("timeout"))
            return SimpleNamespace(returncode=0, stderr="")

        with patch(
            "gomoveshift.pipeline.ingest.concatenator.subprocess.run",
            side_effect=fake_run,
        ):
            _split_episode_at_points(
                episode,
                split_points=[2880.0, 5760.0, 8640.0],  # 48/96/144 min
                output_dir=tmp_path,
                stages_dir=tmp_path,
            )
        assert captured_timeouts, "expected at least one ffmpeg invocation"
        assert all(t == 3600 for t in captured_timeouts), (
            f"split subprocess timeout regressed: {captured_timeouts}"
        )

    def test_all_collapsed_returns_empty(self, tmp_path):
        source = tmp_path / "ep01.mp4"
        source.write_bytes(b"x")
        episode = SimpleNamespace(
            index=2,
            concat_path=source,
            total_duration=0.0,
            recordings=[],
            date="2026-03-04",
        )
        with patch("gomoveshift.pipeline.ingest.concatenator.subprocess.run") as run:
            out = _split_episode_at_points(episode, [], tmp_path, tmp_path)
        assert out == []
        run.assert_not_called()
