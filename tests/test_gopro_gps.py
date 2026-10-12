"""Tests for GoPro GPMF GPS detection and extraction."""

import datetime
import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.gopro_gps import (
    detect_gpmf_stream,
    extract_gopro_gpx,
    extract_recording_date,
)


class TestGPMFDetection:
    """Detect GoPro GPMF telemetry streams in MP4 files."""

    def test_detect_returns_true_for_gpmd_stream(self, tmp_path):
        """Should return True when ffprobe finds a gpmd codec stream."""
        ffprobe_output = json.dumps(
            {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "codec_tag_string": "avc1"},
                    {"codec_type": "data", "codec_name": "gpmd", "codec_tag_string": "gpmd"},
                ]
            }
        )
        mock_result = MagicMock(returncode=0, stdout=ffprobe_output)

        with patch("subprocess.run", return_value=mock_result):
            assert detect_gpmf_stream(tmp_path / "gopro.mp4") is True

    def test_detect_returns_false_for_non_gopro(self, tmp_path):
        """Should return False for regular MP4 without GPMF."""
        ffprobe_output = json.dumps(
            {
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "codec_tag_string": "avc1"},
                    {"codec_type": "audio", "codec_name": "aac", "codec_tag_string": "mp4a"},
                ]
            }
        )
        mock_result = MagicMock(returncode=0, stdout=ffprobe_output)

        with patch("subprocess.run", return_value=mock_result):
            assert detect_gpmf_stream(tmp_path / "regular.mp4") is False

    def test_detect_returns_false_on_ffprobe_error(self, tmp_path):
        """Should return False gracefully if ffprobe fails."""
        mock_result = MagicMock(returncode=1, stdout="", stderr="error")

        with patch("subprocess.run", return_value=mock_result):
            assert detect_gpmf_stream(tmp_path / "bad.mp4") is False

    def test_detect_returns_false_on_exception(self, tmp_path):
        """Should return False gracefully on any exception."""
        with patch("subprocess.run", side_effect=OSError("ffprobe not found")):
            assert detect_gpmf_stream(tmp_path / "missing.mp4") is False


class TestGPXExtraction:
    """Extract GPMF telemetry to GPX format."""

    def test_extract_returns_path_on_success(self, tmp_path):
        """Should return GPX path when gopro-overlay extraction succeeds."""
        import json as _json

        video_path = tmp_path / "gopro.mp4"
        video_path.touch()
        output_dir = tmp_path / "stages"

        stream_json = _json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "index": 0,
                        "disposition": {"default": 1},
                        "width": 1920,
                        "height": 1080,
                        "duration": "30.0",
                        "nb_frames": "900",
                        "avg_frame_rate": "30/1",
                    },
                    {
                        "codec_type": "audio",
                        "index": 1,
                        "disposition": {"default": 1},
                    },
                    {
                        "codec_type": "data",
                        "codec_tag_string": "gpmd",
                        "index": 2,
                        "time_base": "1/600",
                    },
                ]
            }
        )

        mock_ffmpeg_inst = MagicMock()
        mock_ffmpeg_inst.ffprobe.return_value.invoke.return_value.stdout = stream_json

        mock_gpx_obj = MagicMock()
        mock_gpx_obj.to_xml.return_value = '<?xml version="1.0"?><gpx><trk/></gpx>'
        # "Success" means at least one point survived the GPS lock filter.
        mock_gpx_obj.get_points_no.return_value = 1

        mock_ffmpeg_mod = MagicMock()
        mock_ffmpeg_mod.FFMPEG.return_value = mock_ffmpeg_inst

        mock_ffmpeg_gopro_mod = MagicMock()
        mock_ffmpeg_gopro_mod.FFMPEGGoPro.return_value.find_frame_duration.return_value = (
            MagicMock()
        )

        mock_framemeta_gpx_mod = MagicMock()
        mock_framemeta_gpx_mod.framemeta_to_gpx.return_value = mock_gpx_obj

        mock_modules = {
            "gopro_overlay": MagicMock(),
            "gopro_overlay.ffmpeg": mock_ffmpeg_mod,
            "gopro_overlay.ffmpeg_gopro": mock_ffmpeg_gopro_mod,
            "gopro_overlay.framemeta_gpmd": MagicMock(),
            "gopro_overlay.framemeta_gpx": mock_framemeta_gpx_mod,
            "gopro_overlay.gpmf": MagicMock(),
            "gopro_overlay.gpmd_filters": MagicMock(),
            "gopro_overlay.counter": MagicMock(),
            "gopro_overlay.dimensions": MagicMock(),
            "gopro_overlay.timeunits": MagicMock(),
            "gopro_overlay.units": MagicMock(),
        }

        with (
            patch.dict("sys.modules", mock_modules),
            patch("gomoveshift.pipeline.gopro_gps.build_stamped_gpx", return_value=mock_gpx_obj),
        ):
            result = extract_gopro_gpx(video_path, output_dir)

        assert result is not None
        assert result.suffix == ".gpx"

    def test_extract_returns_none_when_zero_locked_points(self, tmp_path):
        """A recording with no GPS lock (all readings filtered out) must return
        None and must NOT leave a GPX file behind — a written empty GPX would
        make a GPS-less LF look GPS-having downstream and trip the
        GPS-required chapter-density gate instead of the GPS-less skip path.
        """
        import json as _json

        video_path = tmp_path / "gopro.mp4"
        video_path.touch()
        output_dir = tmp_path / "stages"
        gpx_path = output_dir / f"{video_path.stem}_gopro.gpx"
        output_dir.mkdir(parents=True)
        # Simulate a stale GPX from a prior run — must be removed, not left behind.
        gpx_path.write_text("<gpx><trk/></gpx>")

        stream_json = _json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "index": 0,
                        "disposition": {"default": 1},
                        "width": 1920,
                        "height": 1080,
                        "duration": "30.0",
                        "nb_frames": "900",
                        "avg_frame_rate": "30/1",
                    },
                    {
                        "codec_type": "audio",
                        "index": 1,
                        "disposition": {"default": 1},
                    },
                    {
                        "codec_type": "data",
                        "codec_tag_string": "gpmd",
                        "index": 2,
                        "time_base": "1/600",
                    },
                ]
            }
        )

        mock_ffmpeg_inst = MagicMock()
        mock_ffmpeg_inst.ffprobe.return_value.invoke.return_value.stdout = stream_json

        mock_gpx_obj = MagicMock()
        mock_gpx_obj.to_xml.return_value = '<?xml version="1.0"?><gpx></gpx>'
        # No reading ever achieved a GPS lock — zero points survive the filter.
        mock_gpx_obj.get_points_no.return_value = 0

        mock_ffmpeg_mod = MagicMock()
        mock_ffmpeg_mod.FFMPEG.return_value = mock_ffmpeg_inst

        mock_ffmpeg_gopro_mod = MagicMock()
        mock_ffmpeg_gopro_mod.FFMPEGGoPro.return_value.find_frame_duration.return_value = (
            MagicMock()
        )

        mock_framemeta_gpx_mod = MagicMock()
        mock_framemeta_gpx_mod.framemeta_to_gpx.return_value = mock_gpx_obj

        mock_modules = {
            "gopro_overlay": MagicMock(),
            "gopro_overlay.ffmpeg": mock_ffmpeg_mod,
            "gopro_overlay.ffmpeg_gopro": mock_ffmpeg_gopro_mod,
            "gopro_overlay.framemeta_gpmd": MagicMock(),
            "gopro_overlay.framemeta_gpx": mock_framemeta_gpx_mod,
            "gopro_overlay.gpmf": MagicMock(),
            "gopro_overlay.gpmd_filters": MagicMock(),
            "gopro_overlay.counter": MagicMock(),
            "gopro_overlay.dimensions": MagicMock(),
            "gopro_overlay.timeunits": MagicMock(),
            "gopro_overlay.units": MagicMock(),
        }

        with (
            patch.dict("sys.modules", mock_modules),
            patch("gomoveshift.pipeline.gopro_gps.build_stamped_gpx", return_value=mock_gpx_obj),
        ):
            result = extract_gopro_gpx(video_path, output_dir)

        assert result is None
        assert not gpx_path.exists()

    def test_extract_returns_none_on_failure(self, tmp_path):
        """Should return None, not raise, when extraction fails."""
        video_path = tmp_path / "bad.mp4"
        video_path.touch()
        output_dir = tmp_path / "stages"

        mock_result = MagicMock(returncode=1, stderr="extraction failed")
        with patch("subprocess.run", return_value=mock_result):
            result = extract_gopro_gpx(video_path, output_dir)
            assert result is None

    def test_extract_returns_none_when_tool_missing(self, tmp_path):
        """Should return None when gopro2gpx is not installed."""
        video_path = tmp_path / "gopro.mp4"
        video_path.touch()

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = extract_gopro_gpx(video_path, tmp_path / "stages")
            assert result is None


def _extraction_stream_json() -> str:
    """Minimal ffprobe -show_streams payload with a GPMF data stream."""
    return json.dumps(
        {
            "streams": [
                {
                    "codec_type": "video",
                    "index": 0,
                    "disposition": {"default": 1},
                    "width": 1920,
                    "height": 1080,
                    "duration": "30.0",
                    "nb_frames": "900",
                    "avg_frame_rate": "30/1",
                },
                {"codec_type": "audio", "index": 1, "disposition": {"default": 1}},
                {
                    "codec_type": "data",
                    "codec_tag_string": "gpmd",
                    "index": 2,
                    "time_base": "1/600",
                },
            ]
        }
    )


def _framemeta_with_mixed_fixes():
    """A real FrameMeta whose points span pre-lock and locked GPS fixes.

    Pre-lock rows carry the camera's stale-cache coordinates (fix NO/UNKNOWN,
    non-(0,0) — invisible to a null-island guard); locked rows are the genuine
    drive. The lock filter only *relabels* an entry's ``gpsfix`` — it never
    drops the point — so without a locked-only ``filter_fn`` every one of these
    reaches the canonical GPX.
    """
    from gopro_overlay.entry import Entry
    from gopro_overlay.framemeta import FrameMeta
    from gopro_overlay.point import Point
    from gopro_overlay.timeunits import timeunits
    from gopro_overlay.units import units

    base = datetime.datetime(2025, 9, 13, 11, 0, 0, tzinfo=datetime.timezone.utc)
    # (gpsfix, lat): 0=NO, 1=UNKNOWN (pre-lock stale cache); 2=LOCK_2D, 3=LOCK_3D.
    rows = [(0, 10.0), (1, 10.1), (2, 53.0), (3, 53.1)]
    fm = FrameMeta()
    for i, (fix, lat) in enumerate(rows):
        fm.add(
            timeunits(seconds=i),
            Entry(
                dt=base + datetime.timedelta(seconds=i),
                point=Point(lat, lat),
                alt=units.Quantity(100, "m"),
                gpsfix=fix,
            ),
        )
    return fm


class TestLockedFixOnly:
    """Only locked GPS fixes are written to the canonical per-episode GPX.

    The GPX carries no fix/DOP channel, so a downstream consumer can't recover
    the classification — a pre-lock stale-cache point that slips in silently
    pollutes county voting / chapters / endpoints. The real vendored
    ``framemeta_to_gpx`` runs here so the filter is exercised end-to-end.
    """

    def test_prelock_points_absent_from_canonical_gpx(self, tmp_path):
        # End-to-end through the real vendored framemeta_to_gpx. The library is
        # not installable in CI (see the sys.modules-mocked test above), so skip
        # there and exercise the real filter wherever gopro_overlay is present.
        pytest.importorskip("gopro_overlay")
        import gpxpy

        video_path = tmp_path / "gopro.mp4"
        video_path.touch()
        output_dir = tmp_path / "stages"

        mock_ffmpeg_inst = MagicMock()
        mock_ffmpeg_inst.ffprobe.return_value.invoke.return_value.stdout = _extraction_stream_json()
        fm = _framemeta_with_mixed_fixes()

        with (
            patch("gopro_overlay.ffmpeg.FFMPEG", return_value=mock_ffmpeg_inst),
            patch("gopro_overlay.ffmpeg_gopro.FFMPEGGoPro"),
            patch("gopro_overlay.ffmpeg_gopro.GoproRecording"),
            patch("gopro_overlay.ffmpeg_gopro.filestat"),
            patch("gopro_overlay.framemeta_gpmd.parse_gopro", return_value=fm),
        ):
            result = extract_gopro_gpx(video_path, output_dir)

        assert result is not None
        with open(result) as f:
            gpx = gpxpy.parse(f)
        lats = [p.latitude for p in gpx.tracks[0].segments[0].points]

        # Pre-lock stale-cache coordinates must NOT enter the canonical GPX.
        assert 10.0 not in lats
        assert 10.1 not in lats
        # Genuine locked fixes survive.
        assert 53.0 in lats
        assert 53.1 in lats
        assert len(lats) == 2

    def test_reason_counter_is_reported(self, tmp_path, caplog):
        """The lock filter's ReasonCounter (write-only today) is surfaced: a
        relabelled reading logs its reason so the operator sees why points were
        dropped rather than the tally vanishing silently."""
        # Drives the real vendored lock filter; skip in CI where the library is
        # not installable (see the sys.modules-mocked test above).
        pytest.importorskip("gopro_overlay")
        video_path = tmp_path / "gopro.mp4"
        video_path.touch()
        output_dir = tmp_path / "stages"

        mock_ffmpeg_inst = MagicMock()
        mock_ffmpeg_inst.ffprobe.return_value.invoke.return_value.stdout = _extraction_stream_json()
        fm = _framemeta_with_mixed_fixes()

        def _parse_and_reject(*args, **kwargs):
            # Drive one reading through the real lock filter so its report
            # callback populates the counter (mirrors a high-DOP reading).
            from gopro_overlay.gpmd_filters import GPSLockComponents
            from gopro_overlay.gpmf import GPSFix
            from gopro_overlay.point import Point

            kwargs["gps_lock_filter"].submit(
                GPSLockComponents(fix=GPSFix.LOCK_3D, point=Point(53.0, 53.0), speed=0.0, dop=999)
            )
            return fm

        with (
            patch("gopro_overlay.ffmpeg.FFMPEG", return_value=mock_ffmpeg_inst),
            patch("gopro_overlay.ffmpeg_gopro.FFMPEGGoPro"),
            patch("gopro_overlay.ffmpeg_gopro.GoproRecording"),
            patch("gopro_overlay.ffmpeg_gopro.filestat"),
            patch("gopro_overlay.framemeta_gpmd.parse_gopro", side_effect=_parse_and_reject),
            caplog.at_level(logging.INFO, logger="gomoveshift.pipeline.gopro_gps"),
        ):
            result = extract_gopro_gpx(video_path, output_dir)

        assert result is not None
        assert "DOP > 10" in caplog.text


def _parse_through_lock_filter(*args, **kwargs):
    """Mirror the real GPMF visitor: drive each reading through the actual
    ``gps_lock_filter`` the code under test built, then stamp the resulting
    ``gpsfix`` onto the entry (the library does exactly this — see
    ``GPS9EntryConverter``). Two clean LOCK_3D fixes with good DOP: a ~40 kph
    city fix and a ~110 kph highway fix. Only the speed cap can relabel them.
    """
    from gopro_overlay.entry import Entry
    from gopro_overlay.framemeta import FrameMeta
    from gopro_overlay.gpmd_filters import GPSLockComponents
    from gopro_overlay.gpmf import GPSFix
    from gopro_overlay.point import Point
    from gopro_overlay.timeunits import timeunits
    from gopro_overlay.units import units

    lock = kwargs["gps_lock_filter"]
    base = datetime.datetime(2025, 9, 13, 11, 0, 0, tzinfo=datetime.timezone.utc)
    # (lat, speed_mps): ~40 kph city, ~110 kph highway.
    rows = [(53.0, 11.1), (53.5, 30.56)]
    fm = FrameMeta()
    for i, (lat, speed_mps) in enumerate(rows):
        point = Point(lat, lat)
        calculated = lock.submit(
            GPSLockComponents(fix=GPSFix.LOCK_3D, point=point, speed=speed_mps, dop=1.0)
        )
        fm.add(
            timeunits(seconds=i),
            Entry(
                dt=base + datetime.timedelta(seconds=i),
                point=point,
                alt=units.Quantity(100, "m"),
                gpsfix=calculated.value,
            ),
        )
    return fm


class TestHighwaySpeedFixes:
    """Highway-speed GPS fixes survive GPMF extraction into the GPX.

    The lock filter's ``speed_max`` is a glitch bound (discards teleport
    readings), not a cruising-speed cap. A 60 kph cycling default relabelled
    every faster fix NO-lock, so highway stretches vanished from the canonical
    GPX — skewing county voting, distance, chapters and endpoints. The real
    vendored filter runs here so the bound is exercised end-to-end.
    """

    def test_highway_speed_fix_survives_extraction(self, tmp_path):
        # Drives the real vendored lock filter; skip in CI where the library is
        # not installable (see the sys.modules-mocked test above).
        pytest.importorskip("gopro_overlay")
        import gpxpy

        video_path = tmp_path / "gopro.mp4"
        video_path.touch()
        output_dir = tmp_path / "stages"

        mock_ffmpeg_inst = MagicMock()
        mock_ffmpeg_inst.ffprobe.return_value.invoke.return_value.stdout = _extraction_stream_json()

        with (
            patch("gopro_overlay.ffmpeg.FFMPEG", return_value=mock_ffmpeg_inst),
            patch("gopro_overlay.ffmpeg_gopro.FFMPEGGoPro"),
            patch("gopro_overlay.ffmpeg_gopro.GoproRecording"),
            patch("gopro_overlay.ffmpeg_gopro.filestat"),
            patch(
                "gopro_overlay.framemeta_gpmd.parse_gopro",
                side_effect=_parse_through_lock_filter,
            ),
        ):
            result = extract_gopro_gpx(video_path, output_dir)

        assert result is not None
        with open(result) as f:
            gpx = gpxpy.parse(f)
        lats = [p.latitude for p in gpx.tracks[0].segments[0].points]

        # The ~110 kph highway fix must reach the canonical GPX; the 60 kph
        # cycling cap dropped it.
        assert 53.5 in lats
        # The slow city fix survives regardless — a sanity anchor.
        assert 53.0 in lats


class TestRecordingDate:
    """Extract recording date from ffprobe format metadata."""

    def test_extracts_creation_time_iso(self):
        """Should parse ISO creation_time from ffprobe output."""
        ffprobe_output = json.dumps(
            {"format": {"tags": {"creation_time": "2025-09-13T11:12:19.000000Z"}}}
        )
        mock_result = MagicMock(returncode=0, stdout=ffprobe_output)

        with patch("subprocess.run", return_value=mock_result):
            result = extract_recording_date(Path("/fake/video.mp4"))
            assert result is not None
            assert result.year == 2025
            assert result.month == 9
            assert result.day == 13

    def test_returns_none_when_no_creation_time(self):
        """Should return None when creation_time tag is missing."""
        ffprobe_output = json.dumps({"format": {"tags": {}}})
        mock_result = MagicMock(returncode=0, stdout=ffprobe_output)

        with patch("subprocess.run", return_value=mock_result):
            result = extract_recording_date(Path("/fake/video.mp4"))
            assert result is None

    def test_returns_none_on_ffprobe_failure(self):
        """Should return None on ffprobe failure."""
        mock_result = MagicMock(returncode=1, stdout="")
        with patch("subprocess.run", return_value=mock_result):
            result = extract_recording_date(Path("/fake/video.mp4"))
            assert result is None

    @pytest.mark.parametrize(
        "creation_time",
        [
            "2026-04-01T18:00:00.000000Z",  # ISO with fractional seconds + Z
            "2026-04-01T18:00:00Z",  # ISO with Z, no fractional
            "2026-04-01 18:00:00",  # space form, no zone (container UTC)
        ],
    )
    def test_creation_time_is_utc_aware_and_host_tz_invariant(self, creation_time):
        """The container ``creation_time`` tag is UTC — the parsed value
        must be tz-aware so ``.timestamp()`` yields the true UTC epoch on any
        host TZ. With the old naive parse, ``.timestamp()`` reinterpreted the
        wall clock as host-local, skewing the GPX-correspondence gate by the
        host's UTC offset (4-5 h on US-East) and false-blocking a correct GPX.
        """
        import os
        import time
        from datetime import datetime, timezone

        # 18:00 UTC on 2026-04-01 — the true instant the tag denotes.
        expected_epoch = datetime(2026, 4, 1, 18, 0, 0, tzinfo=timezone.utc).timestamp()

        ffprobe_output = json.dumps({"format": {"tags": {"creation_time": creation_time}}})
        mock_result = MagicMock(returncode=0, stdout=ffprobe_output)

        old_tz = os.environ.get("TZ")
        try:
            # A deliberately non-UTC host: naive .timestamp() would shift 4-5h.
            os.environ["TZ"] = "America/New_York"
            time.tzset()
            with patch("subprocess.run", return_value=mock_result):
                result = extract_recording_date(Path("/fake/video.mp4"))
            assert result is not None
            assert result.tzinfo is not None, "parsed datetime must be tz-aware"
            assert result.utcoffset() == timezone.utc.utcoffset(None)
            assert result.timestamp() == expected_epoch
        finally:
            if old_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = old_tz
            time.tzset()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
