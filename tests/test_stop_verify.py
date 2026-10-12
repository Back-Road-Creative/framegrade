"""Tests for the post-stop-removal verification gate.

Covers ``stop_verify.verify_stops_removed``:
  (a) a "cut output" that still contains a >= min_stop_duration low-motion span
      -> the gate raises;
  (b) the duration reconciliation (hard cuts + crossfade overlap math);
  (c) keep_segments monotonicity / bounds.

The residual-stop re-scan is driven through an injected StopProcessor seam so
the assertions are deterministic and don't depend on ffmpeg motion-detection
output. One end-to-end case probes a real testsrc clip through ffprobe.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from gomoveshift.config import Config
from gomoveshift.models import DeadSpot
from gomoveshift.pipeline.stop_verify import (
    DEFAULT_DURATION_TOLERANCE_S,
    StopVerificationError,
    _expected_output_duration,
    verify_stops_removed,
)


@pytest.fixture
def cfg():
    return Config()


@pytest.fixture
def out(tmp_path):
    p = tmp_path / "no_stops.mp4"
    p.write_bytes(b"\x00")
    return p


def _proc(stops, duration):
    """StopProcessor seam: detect_stops()->``stops``; _get_duration()->``duration``."""
    proc = MagicMock()
    proc.detect_stops.return_value = stops
    proc._get_duration.return_value = duration
    return proc


def _stop(start, end):
    return DeadSpot(start_time=start, end_time=end, motion_energy=0.001, suggested_trim=True)


# ── (a) Residual-stop re-scan ────────────────────────────────────────


class TestResidualStopCheck:
    def test_residual_stop_raises(self, cfg, out):
        """A cut output the re-scan still finds a >= min_stop span in -> raises."""
        m = cfg.stop_detection.min_stop_duration
        proc = _proc([_stop(40.0, 40.0 + m + 2.0)], 100.0)
        with pytest.raises(StopVerificationError, match="residual stop"):
            verify_stops_removed(out, [(0.0, 100.0)], 100.0, cfg, _stop_processor=proc)

    def test_short_residual_below_threshold_passes(self, cfg, out):
        """A low-motion span SHORTER than min_stop_duration does not qualify."""
        m = cfg.stop_detection.min_stop_duration
        proc = _proc([_stop(40.0, 40.0 + m - 1.0)], 100.0)
        verify_stops_removed(out, [(0.0, 100.0)], 100.0, cfg, _stop_processor=proc)

    def test_clean_output_passes(self, cfg, out):
        """Zero residual stops + reconciling duration -> passes."""
        proc = _proc([], 100.0)
        verify_stops_removed(out, [(0.0, 50.0), (70.0, 120.0)], 120.0, cfg, _stop_processor=proc)


# ── (b) Duration reconciliation ──────────────────────────────────────


class TestDurationReconciliation:
    def test_hard_cut_total_is_sum(self):
        assert _expected_output_duration([(0.0, 50.0), (70.0, 120.0)], 0.0) == pytest.approx(100.0)

    def test_crossfade_subtracts_overlap(self):
        # 3 segments, 2 transitions, 0.5s crossfade each -> -1.0s.
        keep = [(0.0, 50.0), (70.0, 120.0), (130.0, 160.0)]
        assert _expected_output_duration(keep, 0.5) == pytest.approx(130.0 - 1.0)

    def test_duration_mismatch_raises(self, cfg, out):
        # Output probes far off (residual idle never cut -> too long).
        proc = _proc([], 100.0 + DEFAULT_DURATION_TOLERANCE_S + 5.0)
        with pytest.raises(StopVerificationError, match="differs from the expected"):
            verify_stops_removed(out, [(0.0, 100.0)], 100.0, cfg, _stop_processor=proc)

    def test_within_tolerance_passes(self, cfg, out):
        proc = _proc([], 100.0 + DEFAULT_DURATION_TOLERANCE_S - 0.5)
        verify_stops_removed(out, [(0.0, 100.0)], 100.0, cfg, _stop_processor=proc)

    def test_real_clip_duration_probe(self, cfg, tmp_path):
        """Real testsrc clip: the real ffprobe duration reconciles a keep-segment."""
        from gomoveshift.pipeline.stop_processor import StopProcessor

        clip = tmp_path / "no_stops.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=30",
             "-c:v", "libx264", "-g", "30", "-v", "quiet", str(clip)],
            check=True, timeout=30,
        )  # fmt: skip
        proc = _proc([], None)
        proc._get_duration.side_effect = lambda p: StopProcessor(cfg)._get_duration(p)
        verify_stops_removed(clip, [(0.0, 6.0)], 6.0, cfg, _stop_processor=proc)


# ── (c) keep_segments structural checks ──────────────────────────────


class TestKeepSegmentBounds:
    def test_non_increasing_segment_raises(self, cfg, out):
        proc = _proc([], 100.0)
        with pytest.raises(StopVerificationError, match="non-increasing"):
            verify_stops_removed(out, [(50.0, 40.0)], 100.0, cfg, _stop_processor=proc)

    def test_out_of_bounds_segment_raises(self, cfg, out):
        proc = _proc([], 100.0)
        with pytest.raises(StopVerificationError, match="outside"):
            verify_stops_removed(out, [(0.0, 500.0)], 100.0, cfg, _stop_processor=proc)

    def test_overlapping_segments_raise(self, cfg, out):
        proc = _proc([], 100.0)  # second segment starts before the first ends
        with pytest.raises(StopVerificationError, match="overlaps"):
            verify_stops_removed(out, [(0.0, 60.0), (40.0, 90.0)], 100.0, cfg, _stop_processor=proc)

    def test_none_keep_segments_only_runs_rescan(self, cfg, out):
        """keep_segments=None (legacy/no-stops) -> only the residual scan runs."""
        proc = _proc([], 0.0)
        verify_stops_removed(out, None, 0.0, cfg, _stop_processor=proc)
        proc.detect_stops.assert_called_once()
        proc._get_duration.assert_not_called()


# ── independent GPS-speed check of the RETAINED timeline ────────


class TestRetainedTimelineGPS:
    """``verify_retained_timeline_moving`` runs regardless of stops_found and
    uses the GPS trace, not the blind optical detector."""

    @staticmethod
    def _parked_processor(cfg):
        from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
        from gomoveshift.pipeline.stop_processor import StopProcessor

        tps = [
            GPSTrackpoint(lat=10.0, lon=10.0, elevation=None, timestamp=float(i)) for i in range(31)
        ]
        return StopProcessor(cfg, gps_data=GPSData(trackpoints=tps))

    def test_parked_span_in_whole_timeline_raises(self, cfg):
        from gomoveshift.pipeline.stop_verify import verify_retained_timeline_moving

        proc = self._parked_processor(cfg)
        # keep_segments=None -> whole [0, 30] retained; the parked span survives.
        with pytest.raises(StopVerificationError, match=r"STOP-VERIFY\(GPS\)"):
            verify_retained_timeline_moving(None, 30.0, cfg, proc)

    def test_parked_span_excluded_by_keep_segments_passes(self, cfg):
        from gomoveshift.pipeline.stop_verify import verify_retained_timeline_moving

        proc = self._parked_processor(cfg)
        # The 0-30 parked span is fully cut; nothing parked survives.
        verify_retained_timeline_moving([(30.0, 60.0)], 60.0, cfg, proc)

    def test_gps_absent_is_noop(self, cfg):
        from gomoveshift.pipeline.stop_processor import StopProcessor
        from gomoveshift.pipeline.stop_verify import verify_retained_timeline_moving

        verify_retained_timeline_moving(None, 30.0, cfg, StopProcessor(cfg))

    def test_disabled_is_noop(self, cfg):
        from gomoveshift.pipeline.stop_verify import verify_retained_timeline_moving

        cfg.stop_detection.gps_speed_corroboration.enabled = False
        proc = self._parked_processor(cfg)
        verify_retained_timeline_moving(None, 30.0, cfg, proc)
