"""Insta360 / generic-video-ingest tests.

Covers:

- Generic-mode scan: each accepted file becomes a singleton episode;
  ffprobe ``creation_time`` honored when present; when absent, a GPX
  sidecar dates the recording, else the scan hard-fails (file mtime is
  copy time, never the recording day).
- Mixed-mode refusal in both directions.
- ``video_extensions`` filtering: files outside the list are ignored.
- GoPro-mode regression: ``scan_video_directory(mode="gopro")`` still
  groups by recording number.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Minimal valid GPX sidecar: one track segment, US-East coords (lon ≈ -80.8 →
# whole-hour offset -5h), mid-afternoon UTC so the longitude-local day is
# unambiguous (13:00 local, nowhere near midnight).
_SIDECAR_GPX = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<gpx version="1.1" creator="test">\n'
    "  <trk><trkseg>\n"
    '    <trkpt lat="35.50" lon="-80.80"><time>2026-03-05T18:00:00Z</time></trkpt>\n'
    '    <trkpt lat="35.60" lon="-80.90"><time>2026-03-05T18:05:00Z</time></trkpt>\n'
    "  </trkseg></trk>\n"
    "</gpx>\n"
)


# -----------------------------------------------------------------
# Generic mode — singleton episode per file
# -----------------------------------------------------------------


class TestScanGenericMode:
    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_each_file_becomes_singleton_recording(self, mock_ct, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_dur.return_value = 600.0
        mock_ct.return_value = datetime(2026, 5, 1, 12, 0, 0)

        for name in ["VID_20260501_120000.mp4", "drone_clip.mov", "phone.mkv"]:
            (tmp_path / name).touch()

        recs, filtered, unrec = scan_video_directory(
            tmp_path,
            extensions=[".mp4", ".mov", ".mkv"],
            mode="generic",
        )

        assert len(recs) == 3
        assert len(filtered) == 0
        assert len(unrec) == 0
        for rec in recs.values():
            assert len(rec.chapters) == 1
            assert rec.chapters[0].chapter_num == 1

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_creation_time_from_ffprobe_when_present(self, mock_ct, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_dur.return_value = 600.0
        expected = datetime(2026, 5, 1, 9, 30, 0)
        mock_ct.return_value = expected

        (tmp_path / "x.mp4").touch()

        recs, _, _ = scan_video_directory(tmp_path, extensions=[".mp4"], mode="generic")

        chapter = list(recs.values())[0].chapters[0]
        assert chapter.creation_time == expected

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_no_creation_time_and_no_gpx_hard_fails(self, mock_ct, mock_dur, tmp_path):
        """No container creation_time AND no GPX sidecar → loud hard-fail.

        The old behavior stamped the file's mtime (its copy/offload time) as
        the recording day, silently misgrouping a day-N drive offloaded on
        day M into day M's LF/journal slot. Refuse rather than stamp copy time.
        """
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_dur.return_value = 600.0
        mock_ct.return_value = None  # ffprobe failed / not present

        vid = tmp_path / "x.mp4"
        vid.touch()
        # An offload/copy mtime that is NOT the recording day.
        offload = datetime(2026, 3, 15, 10, 0, 0).timestamp()
        os.utime(vid, (offload, offload))

        with pytest.raises(ValueError, match="x.mp4"):
            scan_video_directory(tmp_path, extensions=[".mp4"], mode="generic")

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_gpx_sidecar_satisfies_when_no_creation_time(self, mock_ct, mock_dur, tmp_path):
        """A GPX sidecar dates the recording — no fail, and mtime is never stamped.

        creation_time stays None so the day key derives from the GPX first
        fix (longitude-local) downstream, not from the file's copy time.
        """
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_dur.return_value = 600.0
        mock_ct.return_value = None

        vid = tmp_path / "x.mp4"
        vid.touch()
        offload = datetime(2026, 3, 15, 10, 0, 0).timestamp()
        os.utime(vid, (offload, offload))
        # Operator-supplied GPX sidecar next to the source file.
        (tmp_path / "x.gpx").write_text("<gpx></gpx>")

        recs, _, _ = scan_video_directory(tmp_path, extensions=[".mp4"], mode="generic")

        chapter = list(recs.values())[0].chapters[0]
        # The file's copy mtime is never stamped as the recording day.
        assert chapter.creation_time is None

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_offload_mtime_never_becomes_episode_date(self, mock_ct, mock_dur, tmp_path):
        """A day-N drive offloaded on day M must not group into day M.

        create_ingest_plan must refuse rather than dating the episode to the
        file's copy (offload) mtime.
        """
        from gomoveshift.pipeline.ingest import create_ingest_plan

        mock_dur.return_value = 600.0
        mock_ct.return_value = None

        vid = tmp_path / "drive.mp4"
        vid.touch()
        offload = datetime(2026, 3, 15, 10, 0, 0).timestamp()  # copy day M
        os.utime(vid, (offload, offload))

        with pytest.raises(ValueError, match="drive.mp4"):
            create_ingest_plan(tmp_path, mode="generic", extensions=[".mp4"])

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_duration_honored_from_ffprobe(self, mock_ct, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_ct.return_value = datetime(2026, 5, 1, 12, 0, 0)
        mock_dur.return_value = 1234.5

        (tmp_path / "x.mp4").touch()

        recs, _, _ = scan_video_directory(tmp_path, extensions=[".mp4"], mode="generic")
        chapter = list(recs.values())[0].chapters[0]
        assert chapter.duration == 1234.5


# -----------------------------------------------------------------
# Mixed-mode refusal
# -----------------------------------------------------------------


class TestMixedModeRefusal:
    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    def test_generic_mode_rejects_gopro_named_files(self, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_dur.return_value = 600.0
        (tmp_path / "GX010046.MP4").touch()
        (tmp_path / "insta_clip.mp4").touch()

        with pytest.raises(ValueError, match="GX010046"):
            scan_video_directory(
                tmp_path,
                extensions=[".mp4"],
                mode="generic",
            )

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    def test_gopro_mode_rejects_non_gopro_named_files(self, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_dur.return_value = 600.0
        (tmp_path / "GX010046.MP4").touch()
        (tmp_path / "random.mp4").touch()

        with pytest.raises(ValueError, match="random.mp4"):
            scan_video_directory(
                tmp_path,
                extensions=[".mp4"],
                mode="gopro",
            )


# -----------------------------------------------------------------
# Extension list honored
# -----------------------------------------------------------------


class TestExtensionFiltering:
    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_files_outside_extension_list_ignored(self, mock_ct, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_dur.return_value = 600.0
        mock_ct.return_value = datetime(2026, 5, 1, 12, 0, 0)

        (tmp_path / "ok.mp4").touch()
        (tmp_path / "skipme.txt").touch()
        (tmp_path / "skipme.avi").touch()  # not in our list

        recs, _, _ = scan_video_directory(
            tmp_path,
            extensions=[".mp4"],
            mode="generic",
        )

        # Only the .mp4 should make it through.
        assert len(recs) == 1
        assert list(recs.values())[0].chapters[0].path.name == "ok.mp4"

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_case_insensitive_extension_match(self, mock_ct, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_dur.return_value = 600.0
        mock_ct.return_value = datetime(2026, 5, 1, 12, 0, 0)

        (tmp_path / "a.MP4").touch()
        (tmp_path / "b.Mov").touch()

        recs, _, _ = scan_video_directory(
            tmp_path,
            extensions=[".mp4", ".mov"],
            mode="generic",
        )
        assert len(recs) == 2


# -----------------------------------------------------------------
# GoPro-mode regression
# -----------------------------------------------------------------


class TestGoProModeUnchanged:
    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.gopro_gps.extract_recording_date")
    def test_gopro_mode_groups_by_recording_number(self, mock_date, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import scan_video_directory

        mock_dur.return_value = 1025.0
        mock_date.return_value = datetime(2026, 2, 20, 16, 0)

        for name in ["GX010046.MP4", "GX020046.MP4", "GX010047.MP4"]:
            (tmp_path / name).touch()

        recs, _, _ = scan_video_directory(
            tmp_path,
            extensions=[".mp4"],
            mode="gopro",
        )

        assert 46 in recs
        assert 47 in recs
        assert len(recs[46].chapters) == 2
        assert len(recs[47].chapters) == 1


# -----------------------------------------------------------------
# create_ingest_plan generic-mode integration
# -----------------------------------------------------------------


class TestCreateIngestPlanGenericMode:
    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_each_file_becomes_episode(self, mock_ct, mock_dur, tmp_path):
        from gomoveshift.pipeline.ingest import create_ingest_plan

        mock_dur.return_value = 600.0
        mock_ct.return_value = datetime(2026, 5, 1, 12, 0, 0)

        (tmp_path / "a.mp4").touch()
        (tmp_path / "b.mov").touch()

        plan = create_ingest_plan(
            tmp_path,
            mode="generic",
            extensions=[".mp4", ".mov"],
        )

        assert len(plan.episodes) == 2
        for ep in plan.episodes:
            assert ep.date == "2026-05-01"
            assert ep.total_duration == pytest.approx(600.0)

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_sidecar_gpx_attached_in_generic_ingest(self, mock_ct, mock_dur, tmp_path):
        """An operator ``<stem>.gpx`` sidecar must land on ``episode.gpx_path``.

        A generic (Insta360/DJI/phone) source has no GoPro GPMF telemetry, so
        the sidecar is its only GPS. Before the fix nothing attached it →
        ``gpx_path`` stayed None → silent GPS-less labeling.
        """
        from gomoveshift.pipeline.ingest import create_ingest_plan

        mock_dur.return_value = 600.0
        mock_ct.return_value = None  # creation_time-less generic source

        vid = tmp_path / "insta360.mp4"
        vid.touch()
        sidecar = tmp_path / "insta360.gpx"
        sidecar.write_text(_SIDECAR_GPX)

        plan = create_ingest_plan(tmp_path, mode="generic", extensions=[".mp4"])

        assert len(plan.episodes) == 1
        assert plan.episodes[0].gpx_path == sidecar

    @patch("gomoveshift.pipeline.ingest.gopro_parser.get_clip_duration")
    @patch("gomoveshift.pipeline.ingest.gopro_parser._ffprobe_creation_time")
    def test_sidecar_only_episode_reaches_a_day_batch(self, mock_ct, mock_dur, tmp_path):
        """The attached sidecar lets a creation_time-less clip key its day.

        The scan accepts a creation_time-less clip *because* the sidecar
        exists, but before the fix the sidecar was never attached to the
        episode, so ``day_grouper`` had no date key and silently dropped it
        (0 day batches). Attaching it lets the longitude-local day resolve.
        """
        from gomoveshift.pipeline.day_grouper import group_episodes_by_day
        from gomoveshift.pipeline.ingest import create_ingest_plan

        mock_dur.return_value = 600.0
        mock_ct.return_value = None

        vid = tmp_path / "insta360.mp4"
        vid.touch()
        (tmp_path / "insta360.gpx").write_text(_SIDECAR_GPX)

        plan = create_ingest_plan(tmp_path, mode="generic", extensions=[".mp4"])
        batches = group_episodes_by_day(plan)

        assert len(batches) == 1
        assert batches[0].date == "2026-03-05"


# -----------------------------------------------------------------
# _ffprobe_creation_time — timezone awareness
# -----------------------------------------------------------------


class TestFfprobeCreationTimeTzAware:
    @patch("gomoveshift.pipeline.ingest.gopro_parser.subprocess.run")
    def test_utc_creation_time_is_timezone_aware(self, mock_run):
        """A ``Z``-suffixed (UTC) creation_time must stay tz-aware, not naive.

        Generic-mode Insta360/DJI/phone sources stamp true UTC. Parsing it
        naive silently drops the zone, so a US-East evening drive
        (04:00Z = 23:00 local the previous day) keys into the wrong calendar
        day. A tz-aware value keeps the real instant and lets the local day
        be recovered.
        """
        from gomoveshift.pipeline.ingest.gopro_parser import _ffprobe_creation_time

        mock_run.return_value = MagicMock(returncode=0, stdout="2026-03-05T04:00:00.000000Z\n")

        dt = _ffprobe_creation_time(Path("insta360.mp4"))

        assert dt is not None
        assert dt.tzinfo is not None
        assert dt.utcoffset() == timedelta(0)
        # 04:00Z → 23:00 the PREVIOUS day at US-East (-05:00).
        est = timezone(timedelta(hours=-5))
        assert dt.astimezone(est).date() == date(2026, 3, 4)


# -----------------------------------------------------------------
# Config plumbing
# -----------------------------------------------------------------


class TestDriveConfigIngestFields:
    def test_default_ingest_mode_is_gopro(self):
        from gomoveshift.config import DriveConfig

        cfg = DriveConfig()
        assert cfg.ingest_mode == "gopro"

    def test_default_extensions_include_mp4(self):
        from gomoveshift.config import DriveConfig

        cfg = DriveConfig()
        assert ".mp4" in cfg.video_extensions

    def test_extensions_normalized_to_lowercase(self):
        from gomoveshift.config import DriveConfig

        cfg = DriveConfig(video_extensions=[".MP4", ".MOV"])
        assert cfg.video_extensions == [".mp4", ".mov"]

    def test_invalid_extension_rejected(self):
        from gomoveshift.config import DriveConfig

        with pytest.raises(Exception):
            DriveConfig(video_extensions=["mp4"])  # missing leading dot
