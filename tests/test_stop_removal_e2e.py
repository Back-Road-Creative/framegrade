"""Real-footage end-to-end regression tests for stop removal.

The existing stop tests (test_motion_analysis, test_motion_ffmpeg, test_crossfade)
are synthetic-only: they feed pre-built motion arrays or mock ffmpeg. None drives a
real clip through the actual StopProcessor and asserts a known stationary span is
gone end to end. These do:

  1. Stationary-span removal e2e — synthesize a clip whose middle holds a constant
     (zero-motion) span, run the REAL detect/cut path, assert the span is detected,
     excluded from keep_segments, and the output is materially shorter.
  2. Dark-span false-cut guard — a dark/low-texture span WITH GPS speed clearly
     nonzero must NOT be cut; GPS-absent clips keep optical-only behavior.
  3. ffprobe-failure — the duration / trailing-stop path raises ProbeFailedError
     loudly instead of silently falling back to a fabricated value.

Clips are tiny (sub-10s, 320x240) so the suite stays fast.
"""

import subprocess

import pytest

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.config import Config
from gomoveshift.pipeline.exceptions import ProbeFailedError
from gomoveshift.pipeline.stop_processor import StopProcessor


def _stop_config(**stop_overrides) -> Config:
    """Config tuned for short test clips: sensitive threshold, low min duration."""
    base = {"min_stop_duration": 2.0, "motion_threshold": 0.008}
    base.update(stop_overrides)
    return Config(stop_detection=base)


def _probe_duration(path) -> float:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return float(out.stdout.strip())


def _synthesize_clip_with_stationary_span(path, *, move_s=3, still_s=4) -> None:
    """Build move -> constant-color (zero motion) -> move, concatenated.

    The middle segment is a single solid color, so consecutive frames are
    identical and motion energy there is ~0 — a real stationary span the
    detector should remove. Outer segments are testsrc (a moving counter)
    so they read as motion. 1-second keyframes keep the decode dense.
    """
    fps = 30
    common = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", str(fps), "-v", "quiet"]
    move_a = path.parent / "move_a.mp4"
    still = path.parent / "still.mp4"
    move_b = path.parent / "move_b.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={move_s}:size=320x240:rate={fps}", *common, str(move_a)],
        check=True, timeout=30,
    )  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"color=c=gray:duration={still_s}:size=320x240:rate={fps}", *common, str(still)],
        check=True, timeout=30,
    )  # fmt: skip
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"testsrc=duration={move_s}:size=320x240:rate={fps}", *common, str(move_b)],
        check=True, timeout=30,
    )  # fmt: skip
    concat = path.parent / "concat.txt"
    concat.write_text("".join(f"file '{p}'\n" for p in (move_a, still, move_b)))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
         "-c", "copy", "-v", "quiet", str(path)],
        check=True, timeout=30,
    )  # fmt: skip


class TestStationarySpanRemovalE2E:
    """A known zero-motion span is detected and removed end to end."""

    def test_stationary_span_detected_and_removed(self, tmp_path):
        clip = tmp_path / "drive.mp4"
        _synthesize_clip_with_stationary_span(clip, move_s=3, still_s=4)
        input_dur = _probe_duration(clip)

        proc = StopProcessor(_stop_config(), crossfade_duration=0.0, cache_dir=tmp_path / "cache")

        # (a) the real motion path detects the constant middle span as a stop
        stops = proc.detect_stops(clip)
        assert stops, "expected the zero-motion middle span to be detected as a stop"
        biggest = max(stops, key=lambda s: s.duration)
        # The still span is ~4s and sits after the ~3s moving intro.
        assert biggest.start_time == pytest.approx(3.0, abs=1.5)
        assert biggest.duration >= 2.0

        out = tmp_path / "no_stops.mp4"
        result = proc.process(clip, out, stops=stops)

        # (b) keep_segments straddle the stop — none lies inside it
        keep = result["keep_segments"]
        assert keep, "expected keep segments after removing the stop"
        for start, end in keep:
            assert not (start >= biggest.start_time and end <= biggest.end_time), (
                f"keep segment {(start, end)} lies inside removed stop "
                f"{(biggest.start_time, biggest.end_time)}"
            )

        # End-to-end: the rendered output is materially shorter than the input,
        # by roughly the removed span (the span really left the pixels).
        out_dur = _probe_duration(out)
        assert out_dur < input_dur - 1.0
        assert input_dur - out_dur == pytest.approx(result["total_removed_duration"], abs=1.5)


class TestDarkSpanFalseCutGuard:
    """A dark span the camera reads as motionless but GPS shows is still moving
    must NOT be cut (GPS corroboration); GPS-absent clips keep optical behavior."""

    # 0..5s zero optical motion (dark/tunnel), then motion — one optical "stop".
    _TS = [float(i) for i in range(12)]
    _ENERGIES = [0.0] * 6 + [0.05] * 6

    def _gps_moving(self, kmh: float) -> GPSData:
        """Trackpoints over 0..12s moving steadily north at ~``kmh``."""
        deg_per_s = (kmh / 3.6) / 111_320.0  # metres/s -> degrees latitude
        pts = [
            GPSTrackpoint(lat=53.0 + deg_per_s * t, lon=-9.0, elevation=None, timestamp=float(t))
            for t in range(13)
        ]
        return GPSData(trackpoints=pts)

    def _processor(self, tmp_path, gps_data):
        """StopProcessor whose decode + duration are stubbed with our series.

        A placeholder source file lets the (cache-miss) hash step run, then the
        stubbed ``_analyze_motion_from_video`` feeds the canned motion series so
        we exercise the real detect + corroboration path without ffmpeg.
        """
        cfg = (
            _stop_config(gps_speed_corroboration={"enabled": True, "stop_speed_kmh": 10.0})
            if gps_data is not None
            else _stop_config()
        )
        proc = StopProcessor(cfg, cache_dir=tmp_path / "cache", gps_data=gps_data)
        src = tmp_path / "fake.mp4"
        src.write_bytes(b"\x00" * 4096)
        proc._analyze_motion_from_video = lambda _p: (self._TS, self._ENERGIES)
        proc._get_duration = lambda _p: 12.0
        return proc, src

    def test_moving_dark_span_not_cut_when_gps_corroborates(self, tmp_path):
        proc, src = self._processor(tmp_path, self._gps_moving(60.0))
        stops = proc.detect_stops(src)
        assert stops == [], "GPS speed 60 km/h should drop the false dark stop"

    def test_gps_absent_keeps_optical_only_behavior(self, tmp_path):
        # No GPS — corroboration is a strict no-op, the optical stop survives.
        proc, src = self._processor(tmp_path, None)
        stops = proc.detect_stops(src)
        assert len(stops) == 1, "GPS-absent clip must keep the optical stop unchanged"
        assert stops[0].start_time == pytest.approx(0.0)

    def test_stopped_dark_span_is_kept_when_gps_confirms(self, tmp_path):
        # GPS ~0 km/h over the span — a real stop; corroboration confirms, keeps it.
        proc, src = self._processor(tmp_path, self._gps_moving(0.0))
        stops = proc.detect_stops(src)
        assert len(stops) == 1, "a genuinely stopped span must be kept"


class TestProbeFailureRaises:
    """The keep-segment / trailing-stop duration path raises loudly on ffprobe
    failure rather than silently substituting a fabricated duration."""

    def test_get_duration_raises_probe_failed(self, tmp_path):
        bad = tmp_path / "not_a_video.mp4"
        bad.write_text("definitely not a video container")
        proc = StopProcessor(_stop_config(), cache_dir=tmp_path / "c")
        with pytest.raises(ProbeFailedError):
            proc._get_duration(bad)

    def test_calculate_keep_segments_propagates_probe_failure(self, tmp_path):
        from gomoveshift.models import DeadSpot

        bad = tmp_path / "not_a_video.mp4"
        bad.write_text("definitely not a video container")
        proc = StopProcessor(_stop_config(), cache_dir=tmp_path / "c")
        stops = [DeadSpot(start_time=2.0, end_time=5.0, motion_energy=0.0, suggested_trim=True)]
        # No silent fallback to a fabricated duration — the failure surfaces.
        with pytest.raises(ProbeFailedError):
            proc._calculate_keep_segments(stops, bad)

    def test_trailing_stop_without_duration_raises(self, tmp_path):
        # A trailing stop with no probe-able duration must hard-fail, not truncate.
        proc = StopProcessor(_stop_config(), cache_dir=tmp_path / "c")
        ts = [float(i) for i in range(12)]
        energies = [0.05] * 6 + [0.0] * 6  # stop runs to the end of the data
        with pytest.raises(ProbeFailedError):
            proc._find_stops_in_motion_data(ts, energies, video_duration=None)


class TestKeepSegmentBoundaryBuffer:
    """dead_spots.trim_buffer_seconds widens each stop cut on both sides."""

    def _proc(self, tmp_path, buffer):
        from gomoveshift.config import Config

        cfg = Config()
        cfg.dead_spots.trim_buffer_seconds = buffer
        proc = StopProcessor(cfg, cache_dir=tmp_path / "c")
        proc._get_duration = lambda _p: 60.0
        return proc

    def test_default_buffer_widens_the_cut(self, tmp_path):
        from gomoveshift.config import Config
        from gomoveshift.models import DeadSpot

        assert Config().dead_spots.trim_buffer_seconds == 2  # wired, not dead
        proc = self._proc(tmp_path, 2)
        keep = proc._calculate_keep_segments(
            [DeadSpot(start_time=10.0, end_time=20.0, motion_energy=0.0)], tmp_path / "x.mp4"
        )
        assert keep == [(0.0, 8.0), (22.0, 60.0)]

    def test_zero_buffer_snaps_to_boundary(self, tmp_path):
        from gomoveshift.models import DeadSpot

        proc = self._proc(tmp_path, 0)
        keep = proc._calculate_keep_segments(
            [DeadSpot(start_time=10.0, end_time=20.0, motion_energy=0.0)], tmp_path / "x.mp4"
        )
        assert keep == [(0.0, 10.0), (20.0, 60.0)]
