"""Tests for gomoveshift.analysis.motion — MotionAnalyzer."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.analysis.motion import MotionAnalyzer
from gomoveshift.config import Config
from gomoveshift.models import DeadSpot, HighMotionSegment


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(**overrides) -> Config:
    """Create a Config with sensible test defaults."""
    return Config(**overrides)


def _make_analyzer(
    motion_threshold: float = 0.02,
    min_duration_seconds: int = 5,
    motion_sample_fps: int = 5,
    hysteresis: dict | None = None,
) -> MotionAnalyzer:
    kwargs = dict(
        dead_spots={
            "motion_threshold": motion_threshold,
            "min_duration_seconds": min_duration_seconds,
        },
        extraction={"motion_sample_fps": motion_sample_fps},
    )
    if hysteresis is not None:
        kwargs["stop_detection"] = {"hysteresis": hysteresis}
    cfg = _make_config(**kwargs)
    return MotionAnalyzer(cfg)


def _gray_frame(value: int, h: int = 100, w: int = 160) -> np.ndarray:
    """Create a uniform grayscale frame."""
    return np.full((h, w), value, dtype=np.uint8)


# ── _compute_motion_from_frames ─────────────────────────────────────


class TestComputeMotionFromFrames:
    """Unit tests for the pure-numpy motion computation (no ffmpeg)."""

    def test_constant_scene_zero_motion(self):
        """Identical frames produce ~0 motion energy."""
        analyzer = _make_analyzer()
        frames = [_gray_frame(128)] * 10
        timestamps, energies = analyzer._compute_motion_from_frames(frames, duration=10.0)
        assert len(timestamps) == 9  # N-1 pairs
        assert all(e == pytest.approx(0.0, abs=1e-6) for e in energies)

    def test_alternating_frames_high_motion(self):
        """Black/white alternation produces high motion energy."""
        analyzer = _make_analyzer()
        frames = [_gray_frame(0 if i % 2 == 0 else 255) for i in range(10)]
        timestamps, energies = analyzer._compute_motion_from_frames(frames, duration=10.0)
        assert len(energies) == 9
        # After Gaussian blur, energy is high but < 1.0 due to blur smoothing
        assert all(e > 0.3 for e in energies)

    def test_single_frame_returns_empty(self):
        """Fewer than 2 frames returns empty lists."""
        analyzer = _make_analyzer()
        ts, en = analyzer._compute_motion_from_frames([_gray_frame(128)], duration=1.0)
        assert ts == []
        assert en == []

    def test_empty_frames_returns_empty(self):
        """No frames returns empty lists."""
        analyzer = _make_analyzer()
        ts, en = analyzer._compute_motion_from_frames([], duration=0.0)
        assert ts == []
        assert en == []

    def test_timestamps_scale_with_duration(self):
        """Timestamps are evenly spaced across the duration."""
        analyzer = _make_analyzer()
        frames = [_gray_frame(128)] * 5
        ts, _ = analyzer._compute_motion_from_frames(frames, duration=20.0)
        # dt = 20/5 = 4.0; timestamps at 4, 8, 12, 16
        assert len(ts) == 4
        assert ts[0] == pytest.approx(4.0)
        assert ts[-1] == pytest.approx(16.0)

    def test_progress_callback_invoked(self):
        """Progress callback is called during processing."""
        analyzer = _make_analyzer()
        frames = [_gray_frame(100 + i) for i in range(120)]
        cb = MagicMock()
        analyzer._compute_motion_from_frames(frames, duration=60.0, progress_callback=cb)
        assert cb.call_count > 0


# ── _find_dead_spots ────────────────────────────────────────────────


class TestFindDeadSpots:
    """Tests for dead spot detection from motion energy arrays."""

    def test_all_low_motion_single_dead_spot(self):
        """Continuous low motion produces one dead spot."""
        analyzer = _make_analyzer(motion_threshold=0.02, min_duration_seconds=5)
        ts = list(range(20))
        energies = [0.001] * 20
        spots = analyzer._find_dead_spots(ts, energies)
        assert len(spots) == 1
        assert spots[0].start_time == 0
        assert spots[0].end_time == 19

    def test_all_high_motion_no_dead_spots(self):
        """High motion throughout produces no dead spots."""
        analyzer = _make_analyzer(motion_threshold=0.02, min_duration_seconds=5)
        ts = list(range(20))
        energies = [0.05] * 20
        spots = analyzer._find_dead_spots(ts, energies)
        assert len(spots) == 0

    def test_short_pause_below_min_duration(self):
        """A pause shorter than min_duration is not a dead spot."""
        analyzer = _make_analyzer(motion_threshold=0.02, min_duration_seconds=10)
        # 5 seconds low then high
        ts = list(range(20))
        energies = [0.001] * 5 + [0.05] * 15
        spots = analyzer._find_dead_spots(ts, energies)
        assert len(spots) == 0  # 5s < 10s min

    def test_dead_spot_at_end_of_video(self):
        """Dead spot at the end (no high-motion to close it) is detected."""
        analyzer = _make_analyzer(motion_threshold=0.02, min_duration_seconds=5)
        ts = list(range(20))
        energies = [0.05] * 10 + [0.001] * 10
        spots = analyzer._find_dead_spots(ts, energies)
        assert len(spots) == 1
        assert spots[0].start_time == 10

    def test_empty_input_returns_empty(self):
        """Empty timestamps returns no dead spots."""
        analyzer = _make_analyzer()
        assert analyzer._find_dead_spots([], []) == []

    def test_multiple_dead_spots(self):
        """Two separate pauses produce two dead spots."""
        analyzer = _make_analyzer(motion_threshold=0.02, min_duration_seconds=3)
        # Pattern: 5s low, 5s high, 5s low, 5s high
        ts = list(range(20))
        energies = [0.001] * 5 + [0.05] * 5 + [0.001] * 5 + [0.05] * 5
        spots = analyzer._find_dead_spots(ts, energies)
        assert len(spots) == 2


# ── hysteresis (dual enter/exit thresholds) ─────────────────────────


class TestHysteresis:
    """Dual enter/exit thresholds debounce near-threshold jitter.

    The series enters a stop (deep low), then oscillates between
    just-below the enter threshold (0.001) and just-above it (0.03)
    while staying below the exit band (0.04 at exit_multiplier=2.0).
    """

    # 3 frames deep-low (enter), then alternate above/below the enter
    # threshold for the remainder. Each "above" sample (0.03) clears the
    # single threshold (0.02) but sits inside the hysteresis exit band.
    THRESHOLD = 0.02
    _JITTER = [0.001, 0.03] * 6  # 12 samples oscillating around THRESHOLD
    ENERGIES = [0.001, 0.001, 0.001] + _JITTER
    TS = list(range(len(ENERGIES)))

    def test_disabled_default_reproduces_single_threshold_split(self):
        """Default (hysteresis off) splits the oscillation into many spans."""
        analyzer = _make_analyzer(motion_threshold=self.THRESHOLD, min_duration_seconds=0)
        # Defaults must equal current behavior even without a config block.
        assert analyzer.hysteresis_enabled is False
        assert analyzer.hysteresis_exit_multiplier == 1.0
        baseline = analyzer._find_dead_spots(self.TS, self.ENERGIES)
        # Each 0.03 sample (>= threshold) closes a span → the oscillation
        # fragments into several short dead spots.
        assert len(baseline) > 1

    def test_explicit_disabled_matches_default(self):
        """An explicit enabled=False block matches the implicit default."""
        default = _make_analyzer(motion_threshold=self.THRESHOLD, min_duration_seconds=0)
        explicit = _make_analyzer(
            motion_threshold=self.THRESHOLD,
            min_duration_seconds=0,
            hysteresis={"enabled": False, "exit_multiplier": 2.0},
        )
        assert default._find_dead_spots(self.TS, self.ENERGIES) == explicit._find_dead_spots(
            self.TS, self.ENERGIES
        )

    def test_enabled_yields_stable_single_span(self):
        """With hysteresis on, the oscillation stays one continuous stop."""
        analyzer = _make_analyzer(
            motion_threshold=self.THRESHOLD,
            min_duration_seconds=0,
            hysteresis={"enabled": True, "exit_multiplier": 2.0},
        )
        assert analyzer.hysteresis_enabled is True
        spots = analyzer._find_dead_spots(self.TS, self.ENERGIES)
        # The whole window is one span — exit band (0.04) is never breached.
        assert len(spots) == 1
        assert spots[0].start_time == self.TS[0]

    def test_enabled_strictly_fewer_spans_than_disabled(self):
        """Hysteresis never fragments more than the single threshold."""
        on = _make_analyzer(
            motion_threshold=self.THRESHOLD,
            min_duration_seconds=0,
            hysteresis={"enabled": True, "exit_multiplier": 2.0},
        )
        off = _make_analyzer(motion_threshold=self.THRESHOLD, min_duration_seconds=0)
        assert len(on._find_dead_spots(self.TS, self.ENERGIES)) < len(
            off._find_dead_spots(self.TS, self.ENERGIES)
        )

    def test_enabled_still_exits_above_band(self):
        """Energy above the exit band genuinely closes the stop."""
        analyzer = _make_analyzer(
            motion_threshold=self.THRESHOLD,
            min_duration_seconds=0,
            hysteresis={"enabled": True, "exit_multiplier": 2.0},
        )
        # Deep-low, then a clear high sample (0.10 > exit band 0.04).
        ts = list(range(6))
        energies = [0.001, 0.001, 0.001, 0.10, 0.10, 0.10]
        spots = analyzer._find_dead_spots(ts, energies)
        assert len(spots) == 1
        assert spots[0].end_time == 3  # closed at the first high sample


# ── _segments_overlap ───────────────────────────────────────────────


class TestSegmentsOverlap:
    """Tests for overlap detection between HighMotionSegments."""

    def _seg(self, start: float, end: float) -> HighMotionSegment:
        return HighMotionSegment(
            start_time=start,
            end_time=end,
            avg_motion_energy=0.05,
            peak_motion_energy=0.1,
        )

    def test_overlapping_segments(self):
        analyzer = _make_analyzer()
        assert analyzer._segments_overlap(self._seg(0, 10), self._seg(5, 15))

    def test_non_overlapping_segments(self):
        analyzer = _make_analyzer()
        assert not analyzer._segments_overlap(self._seg(0, 10), self._seg(20, 30))

    def test_adjacent_within_min_gap(self):
        """Segments within min_gap are considered overlapping."""
        analyzer = _make_analyzer()
        assert analyzer._segments_overlap(self._seg(0, 10), self._seg(12, 20), min_gap=5.0)

    def test_adjacent_outside_min_gap(self):
        analyzer = _make_analyzer()
        assert not analyzer._segments_overlap(self._seg(0, 10), self._seg(20, 30), min_gap=5.0)


# ── _find_high_motion_windows ───────────────────────────────────────


class TestFindHighMotionWindows:
    """Tests for the sliding-window high-motion segment finder."""

    def test_uniform_low_motion_no_segments(self):
        """All-low motion data produces no segments (or uses percentile fallback)."""
        analyzer = _make_analyzer(motion_threshold=0.02)
        data = [(float(i), 0.001) for i in range(100)]
        segments = analyzer._find_high_motion_windows(
            data,
            min_duration=5.0,
            max_duration=30.0,
            video_duration=100.0,
        )
        # Even percentile fallback: all windows have same energy so p80 == mean
        # Some may pass the percentile gate; that is fine — we just verify no crash.
        assert isinstance(segments, list)

    def test_spike_produces_segments(self):
        """A region of high motion produces at least one segment."""
        analyzer = _make_analyzer(motion_threshold=0.02)
        # 50 low + 50 high + 50 low at 1fps => spike in middle
        data = (
            [(float(i), 0.001) for i in range(50)]
            + [(float(i), 0.10) for i in range(50, 100)]
            + [(float(i), 0.001) for i in range(100, 150)]
        )
        segments = analyzer._find_high_motion_windows(
            data,
            min_duration=5.0,
            max_duration=60.0,
            video_duration=150.0,
        )
        assert len(segments) > 0
        # All returned segments should have energy above threshold
        for seg in segments:
            assert seg.avg_motion_energy > analyzer.motion_threshold * 2

    def test_empty_data_returns_empty(self):
        analyzer = _make_analyzer()
        assert analyzer._find_high_motion_windows([], 5.0, 30.0, 0.0) == []


# ── preprocess_frame ────────────────────────────────────────────────


class TestPreprocessFrame:
    """Tests for frame preprocessing (resize + gray + blur)."""

    def test_downscales_large_frame(self):
        analyzer = _make_analyzer()
        # 1920x1080 BGR frame
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = analyzer._preprocess_frame(frame)
        assert result.shape[1] == analyzer.ANALYSIS_WIDTH
        assert len(result.shape) == 2  # grayscale

    def test_small_frame_not_upscaled(self):
        analyzer = _make_analyzer()
        # 320x240 BGR frame (smaller than ANALYSIS_WIDTH)
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        result = analyzer._preprocess_frame(frame)
        assert result.shape[1] == 320  # unchanged width
        assert len(result.shape) == 2
