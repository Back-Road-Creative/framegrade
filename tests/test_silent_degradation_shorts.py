"""Regression tests for silent fallbacks in the shorts path.

Each test asserts both:
  (a) the fallback path is actually taken (behavior preserved), and
  (b) a WARNING is logged describing the degradation.

Covers:
- gps_parser.parse_gpx_file: missing <time> element → point dropped + warn.
- pipeline.shorts_extractor.render_short_ffmpeg: skip_audio / skip_audio_norm warn.
- pipeline.stop_processor._find_stops_in_motion_data: trailing stop with no probed
  duration now HARD-FAILS (ProbeFailedError) — the silent last-sample fallback was removed.
- analysis.shorts_candidate_selector.select_best_shorts: GPS-less segments warn.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.analysis.gps_parser import parse_gpx_file
from gomoveshift.analysis.shorts_candidate_selector import ShortsCandidateSelector
from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint, ScoringWeights
from gomoveshift.models import HighMotionSegment
from gomoveshift.pipeline.exceptions import ProbeFailedError
from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg
from gomoveshift.pipeline.stop_processor import StopProcessor


# ---------------------------------------------------------------------------
# gps_parser.py — missing <time> element
# ---------------------------------------------------------------------------


class TestGPXMissingTimestamp:
    """GPX trackpoints without <time> should log a warning and be dropped."""

    GPX_NO_TIME = """<?xml version='1.0' encoding='UTF-8'?>
<gpx version='1.1' creator='test'>
  <trk><trkseg>
    <trkpt lat='52.0' lon='-8.0'><ele>10</ele></trkpt>
    <trkpt lat='52.1' lon='-8.1'><ele>12</ele></trkpt>
  </trkseg></trk>
</gpx>"""

    def test_missing_time_dropped_and_warns(self, tmp_path, caplog):
        gpx = tmp_path / "no_time.gpx"
        gpx.write_text(self.GPX_NO_TIME)

        with caplog.at_level(logging.WARNING, logger="gomoveshift.analysis.gps_parser"):
            data = parse_gpx_file(gpx)

        # (a) timeless points are dropped, not stamped 0.0
        assert data.trackpoints == []

        # (b) warning emitted for each bad trackpoint
        warnings = [
            r
            for r in caplog.records
            if r.levelno == logging.WARNING and "no <time>" in r.getMessage()
        ]
        assert len(warnings) == 2, f"expected 2 warnings, got {len(warnings)}"


# ---------------------------------------------------------------------------
# pipeline/shorts_extractor.py — skip_audio / skip_audio_norm
# ---------------------------------------------------------------------------


class _FakeCompleted:
    """Stand-in for subprocess.CompletedProcess."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def _stub_ffmpeg(monkeypatch, tmp_path):
    """Stub NVENC detect, ffprobe dimension probe, and ffmpeg subprocess calls."""
    from gomoveshift.pipeline import shorts_extractor as se
    from gomoveshift.pipeline import ffmpeg_runner as fr

    monkeypatch.setattr(se, "_detect_nvenc_cached", lambda: False)
    monkeypatch.setattr(se, "get_video_dimensions", lambda p: (3840, 2160))
    monkeypatch.setattr(se, "build_crop_filter", lambda w, h: "crop=1080:1920")
    monkeypatch.setattr(se, "build_ost_filter", lambda loc, sub, dur: "drawtext=_OST_")

    def _fake_run(cmd, *a, **kw):
        # Emulate ffmpeg creating the output file so render_short_ffmpeg returns True.
        out = Path(cmd[-1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"\x00" * 16)
        return _FakeCompleted(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(se.subprocess, "run", _fake_run)

    # validate_video is imported lazily inside render_short_ffmpeg; stub the
    # source module so the post-encode validity check passes for our fake file.
    _valid_result = MagicMock()
    _valid_result.valid = True
    _valid_result.reason = ""
    _valid_result.duration = 5.0
    monkeypatch.setattr(fr, "validate_video", lambda *a, **kw: _valid_result)

    return se


class TestShortsAudioMuted:
    """Shorts render MUTED unconditionally (`-an`, no
    loudnorm). The near-silent policy is not a degradation, so no warning is
    emitted; the retained skip_audio / skip_audio_norm params are no-ops."""

    def _capture_fake_run(self, captured):
        def _fake_run(cmd, *a, **kw):
            captured["cmd"] = list(cmd)
            out = Path(cmd[-1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b"\x00" * 16)
            return _FakeCompleted(returncode=0, stdout="", stderr="")

        return _fake_run

    def test_skip_audio_true_renders_muted(self, _stub_ffmpeg, tmp_path, monkeypatch):
        se = _stub_ffmpeg
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr(se.subprocess, "run", self._capture_fake_run(captured))

        ok = render_short_ffmpeg(
            source_video=tmp_path / "src.mp4",
            output_path=tmp_path / "short_0.mp4",
            start_time=0.0,
            duration=5.0,
            location="Test",
            skip_audio=True,
        )

        assert ok is True
        assert "-an" in captured["cmd"]
        assert not any("loudnorm" in str(c) for c in captured["cmd"])

    def test_skip_audio_norm_true_still_muted(self, _stub_ffmpeg, tmp_path, monkeypatch):
        se = _stub_ffmpeg
        captured: dict[str, list[str]] = {}
        monkeypatch.setattr(se.subprocess, "run", self._capture_fake_run(captured))

        ok = render_short_ffmpeg(
            source_video=tmp_path / "src.mp4",
            output_path=tmp_path / "short_1.mp4",
            start_time=0.0,
            duration=5.0,
            location="Test",
            skip_audio_norm=True,
        )

        assert ok is True
        # No loudnorm, and audio is stripped (-an) even though skip_audio_norm=True.
        assert not any("loudnorm" in str(c) for c in captured["cmd"])
        assert "-an" in captured["cmd"]
        assert "-c:a" not in captured["cmd"]


# ---------------------------------------------------------------------------
# pipeline/stop_processor.py — trailing stop fallback
# ---------------------------------------------------------------------------


def _make_stop_processor() -> StopProcessor:
    config = MagicMock()
    config.stop_detection.motion_threshold = 0.02
    config.stop_detection.min_stop_duration = 1.0
    return StopProcessor(config)


class TestTrailingStopFallback:
    def test_missing_duration_hard_fails_no_silent_fallback(self, caplog):
        """A real trailing stop with no probed duration must HARD-FAIL, not
        silently degrade to the last motion sample (the formerly silent path)."""
        proc = _make_stop_processor()

        # Build a motion sequence that ends in a stop run (low energy).
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        # First three samples are "moving", last three are "stopped".
        energies = [0.5, 0.5, 0.5, 0.001, 0.001, 0.001]

        with caplog.at_level(logging.ERROR, logger="gomoveshift.pipeline.stop_processor"):
            with pytest.raises(ProbeFailedError):
                proc._find_stops_in_motion_data(timestamps, energies, video_duration=None)

        # A loud ERROR is logged naming the refused fallback.
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("last motion sample" in m for m in msgs), msgs

    def test_known_duration_does_not_warn(self, caplog):
        proc = _make_stop_processor()
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        energies = [0.5, 0.5, 0.5, 0.001, 0.001, 0.001]

        with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.stop_processor"):
            stops = proc._find_stops_in_motion_data(timestamps, energies, video_duration=8.0)

        assert len(stops) == 1
        assert stops[0].end_time == pytest.approx(8.0)
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("last motion sample" in m for m in msgs)


# ---------------------------------------------------------------------------
# analysis/shorts_candidate_selector.py — GPS-less segments bypass geo-dedup
# ---------------------------------------------------------------------------


def _seg(start: float, end: float, avg: float = 0.05) -> HighMotionSegment:
    return HighMotionSegment(
        start_time=start,
        end_time=end,
        avg_motion_energy=avg,
        peak_motion_energy=avg * 2,
    )


class _IdentityMapper:
    """Stand-in for VideoToGpsTimeMapper that treats video time as GPS time."""

    def video_to_gps_time(self, t: float) -> float:
        return t


class TestGeoDedupGpsMissingWarning:
    def test_segments_without_gps_warn_and_remain_selectable(self, caplog):
        # Build GPSData containing only points in the range [0, 20].
        # Any segment requesting points beyond 20s will get an empty list
        # → segment bypasses geo-dedup → should trigger our warning.
        trackpoints = [
            GPSTrackpoint(lat=52.0 + i * 0.0001, lon=-8.0, elevation=None, timestamp=float(i))
            for i in range(21)
        ]
        gps_data = GPSData(trackpoints=trackpoints)

        scenic_scorer = MagicMock()
        scenic_scorer.gps_data = gps_data
        scenic_scorer.score_segment.return_value = 50.0

        selector = ShortsCandidateSelector(
            scoring_weights=ScoringWeights(),
            min_gap=5.0,
            min_duration=5.0,
            max_duration=60.0,
            target_duration=30.0,
            time_mapper=_IdentityMapper(),
            scenic_scorer=scenic_scorer,
            geo_dedup_meters=500.0,
        )

        candidates = [
            _seg(0.0, 10.0),  # has GPS points
            _seg(100.0, 120.0),  # outside GPS coverage → bypasses geo-dedup
            _seg(200.0, 220.0),  # outside GPS coverage → bypasses geo-dedup
        ]

        with caplog.at_level(
            logging.WARNING, logger="gomoveshift.analysis.shorts_candidate_selector"
        ):
            result = selector.select_best_shorts(
                candidates, dead_spots=[], video_duration=300.0, num_shorts=3
            )

        # (a) fallback path exercised: all 3 candidates were selectable
        # (GPS-less ones not rejected by geo-dedup since no center computed)
        assert len(result) >= 2

        # (b) WARNING emitted reporting the count of GPS-less segments
        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("Geo-dedup" in m and "no GPS points" in m for m in msgs), msgs

    def test_all_segments_have_gps_does_not_warn(self, caplog):
        trackpoints = [
            GPSTrackpoint(lat=52.0 + i * 0.01, lon=-8.0, elevation=None, timestamp=float(i))
            for i in range(301)
        ]
        gps_data = GPSData(trackpoints=trackpoints)
        scenic_scorer = MagicMock()
        scenic_scorer.gps_data = gps_data
        scenic_scorer.score_segment.return_value = 50.0

        selector = ShortsCandidateSelector(
            scoring_weights=ScoringWeights(),
            min_gap=5.0,
            min_duration=5.0,
            max_duration=60.0,
            target_duration=30.0,
            time_mapper=_IdentityMapper(),
            scenic_scorer=scenic_scorer,
            geo_dedup_meters=100.0,
        )

        candidates = [_seg(0.0, 10.0), _seg(100.0, 110.0), _seg(200.0, 210.0)]

        with caplog.at_level(
            logging.WARNING, logger="gomoveshift.analysis.shorts_candidate_selector"
        ):
            selector.select_best_shorts(
                candidates, dead_spots=[], video_duration=300.0, num_shorts=3
            )

        msgs = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert not any("Geo-dedup" in m and "no GPS points" in m for m in msgs)
