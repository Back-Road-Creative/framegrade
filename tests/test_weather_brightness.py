"""Tests for WeatherDetector brightness measurement and timeout handling.

TDD tests written before fixing:
  - FFmpeg timeout handling (currently returns 490.0 sentinel → hardcoded overcast)
  - Fallback behavior (should return None, not a bogus weather value)
  - Pre-input -ss placement for fast seek on large files
  - Timeout value adequate for large files (120s minimum)
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline.weather import (
    WeatherCondition,
    WeatherDetector,
    WeatherResult,
    _parse_yavg_lines,
)


# ── timeout value ─────────────────────────────────────────────────────────────


class TestFFmpegTimeoutValue:
    """Per-chunk FFmpeg timeout must be proportional to chunk duration
    with a 30s floor — superseded the legacy 120s monolithic timeout
    once measurement became chunked."""

    def test_per_chunk_timeout_proportional_with_floor(self):
        """``_measure_chunk`` must pass ``subprocess.run`` a timeout that
        scales with chunk duration (2x), floored at 30s and ceilinged at
        90s. A 26s chunk gets ~52s timeout; a 5s probe gets 30s; a 120s
        chunk gets 90s."""
        detector = WeatherDetector()

        captured: dict = {}

        def fake_run(cmd, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            mock_result = MagicMock()
            mock_result.stderr = "[Parsed_signalstats_1] YAVG:128.0"
            mock_result.returncode = 0
            return mock_result

        with patch("subprocess.run", side_effect=fake_run):
            detector._measure_chunk(Path("/fake/v.mp4"), 0.0, 26.0, 52.0)
        assert captured["timeout"] == 52.0

    def test_aggregate_wall_clock_exceeds_120s_for_long_segments(self):
        """For a 30-min episode, the cumulative tolerable wall clock is
        N_chunks * per_chunk_timeout >> 120s. This is the regression
        guard against the legacy single-shot 120s cap that failed on a
        27 GB Insta360 on WSL2."""
        detector = WeatherDetector()  # default chunk_max_sec=120

        chunks = detector._plan_chunks(0.0, 1800.0, detector.chunk_max_sec)
        assert len(chunks) == 15
        # Per-chunk timeout for 120s chunks = 2*120 = 240, clamped to 90.
        total_tolerable = sum(max(30.0, min(90.0, c[1] * 2.0)) for c in chunks)
        assert total_tolerable >= 120.0 * 10, (
            f"Aggregate wall budget {total_tolerable}s is not generous enough; "
            "long sources will still time out under the chunked design."
        )


# ── pre-input seek placement ──────────────────────────────────────────────────


class TestFFmpegSeekPlacement:
    """The -ss flag must come BEFORE -i for fast keyframe seek."""

    def test_ss_before_i_in_ffmpeg_command(self):
        """Verify -ss appears before -i in the constructed FFmpeg command.

        Post-input -ss forces FFmpeg to decode every frame up to the
        timestamp, making seeks on 64-min videos take 10+ minutes.
        Pre-input -ss uses keyframe index for near-instant seek.
        """
        detector = WeatherDetector()

        captured_cmd = []

        def fake_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            mock_result = MagicMock()
            mock_result.stderr = "[Parsed_metadata_2 @ 0x0] lavfi.signalstats.YAVG=128.0\n"
            mock_result.returncode = 0
            return mock_result

        with patch("subprocess.run", side_effect=fake_run):
            detector._get_avg_brightness_ffmpeg(Path("/fake/video.mp4"), 192.0, 26.0)

        assert "ffmpeg" in captured_cmd, "Command should start with ffmpeg"
        ss_idx = captured_cmd.index("-ss") if "-ss" in captured_cmd else -1
        i_idx = captured_cmd.index("-i") if "-i" in captured_cmd else -1
        assert ss_idx != -1, "-ss flag not found in FFmpeg command"
        assert i_idx != -1, "-i flag not found in FFmpeg command"
        assert ss_idx < i_idx, (
            f"-ss at index {ss_idx} must come BEFORE -i at index {i_idx}. "
            f"Post-input seek decodes entire file up to timestamp (very slow)."
        )


# ── timeout fallback returns None ─────────────────────────────────────────────


class TestTimeoutFallbackNone:
    """When FFmpeg times out, _get_avg_brightness_ffmpeg must return None."""

    def test_timeout_returns_none_not_sentinel(self):
        """FFmpeg timeout must return None, not 490.0 sentinel.

        Returning 490.0 maps to 'overcast' with confidence=1.00, producing
        incorrect weather data that propagates to all social posts.
        Returning None allows callers to handle the missing measurement.
        """
        detector = WeatherDetector()

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 60)):
            result = detector._get_avg_brightness_ffmpeg(Path("/fake/video.mp4"), 192.0, 26.0)

        assert result is None, (
            f"Expected None on timeout, got {result!r}. "
            f"490.0 sentinel maps to overcast with confidence=1.00 — incorrect."
        )

    def test_no_yavg_data_raises_measurement_error(self):
        """When FFmpeg exits clean (returncode 0) but produces no YAVG
        samples, the chunk parser raises WeatherMeasurementError rather
        than silently returning None.

        Before the parser fix, the parser silently
        returned [] when the modern ffmpeg metadata format went
        unrecognised, the chunk aggregator counted that as a "degraded"
        chunk, and ultimately Phase A failed with a misleading
        "slow source IO" diagnostic when the real cause was a parser
        contract mismatch. Now: an empty parse on a clean ffmpeg exit
        raises so the operator is pointed at the actual cause."""
        import pytest

        from gomoveshift.pipeline.exceptions import WeatherMeasurementError

        detector = WeatherDetector()

        mock_result = MagicMock()
        mock_result.stderr = "Some output without YAVG data\nNo brightness lines here"
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            with pytest.raises(WeatherMeasurementError, match="no YAVG samples"):
                detector._get_avg_brightness_ffmpeg(Path("/fake/video.mp4"), 192.0, 26.0)


# ── detect_segment returns Optional[WeatherResult] on failure ─────────────────


class TestDetectSegmentNoneOnTimeout:
    """detect_segment must return None when brightness measurement fails."""

    def test_detect_segment_returns_none_on_ffmpeg_timeout(self, tmp_path):
        """detect_segment returns None when FFmpeg times out.

        Previously: returned WeatherResult(condition='overcast', brightness=490.0,
        confidence=1.00) which is incorrect and misleads social post generation.
        """
        # Create a fake video file so exists() check passes
        fake_video = tmp_path / "test_video.mp4"
        fake_video.write_bytes(b"fake")

        detector = WeatherDetector()

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 60)):
            result = detector.detect_segment(fake_video, start_time=192.0, end_time=218.0)

        assert result is None, (
            f"Expected None on timeout, got {result!r}. "
            f"detect_segment must not return bogus WeatherResult when measurement fails."
        )

    def test_detect_segment_returns_result_on_success(self, tmp_path):
        """detect_segment returns WeatherResult when FFmpeg succeeds."""
        fake_video = tmp_path / "test_video.mp4"
        fake_video.write_bytes(b"fake")

        detector = WeatherDetector()

        # Simulate FFmpeg output with YAVG=180 (scaled: 180 * 1000/255 ≈ 706).
        # Brightness used to classify this as SUNNY; the demotion now
        # reports OVERCAST — brightness cannot verify clear skies.
        mock_result = MagicMock()
        mock_result.stderr = (
            "[Parsed_signalstats_1 @ 0x...] YAVG:180.00 YDIF:5.00\n"
            "[Parsed_signalstats_1 @ 0x...] YAVG:182.00 YDIF:4.50\n"
        )
        mock_result.returncode = 0

        with patch("subprocess.run", return_value=mock_result):
            result = detector.detect_segment(fake_video, start_time=192.0, end_time=218.0)

        assert result is not None
        assert isinstance(result, WeatherResult)
        assert result.condition == WeatherCondition.OVERCAST
        assert result.segment_start == 192.0
        assert result.segment_end == 218.0


# ── detect_all_segments skips None results ────────────────────────────────────


class TestDetectAllSegmentsSkipsNone:
    """detect_all_segments must skip segments where brightness detection failed."""

    def test_detect_all_returns_only_successful_results(self, tmp_path):
        """Segments with no brightness data are omitted from the result list.

        With 13 segments all timing out, the old code returned 13 overcast results.
        The new code should return 0 results (or only successful ones).
        """
        fake_video = tmp_path / "test_video.mp4"
        fake_video.write_bytes(b"fake")

        detector = WeatherDetector()

        segments = [
            {"start_time": 192.0, "end_time": 218.0},
            {"start_time": 456.0, "end_time": 490.0},
            {"start_time": 826.0, "end_time": 852.0},
        ]

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 60)):
            results = detector.detect_all_segments(fake_video, segments)

        assert results == [], (
            f"Expected [] when all segments time out, got {results!r}. "
            f"Timed-out segments must not produce overcast results."
        )

    def test_detect_all_returns_mixed_results(self, tmp_path):
        """detect_all returns only the successfully-measured segments."""
        fake_video = tmp_path / "test_video.mp4"
        fake_video.write_bytes(b"fake")

        detector = WeatherDetector()

        segments = [
            {"start_time": 192.0, "end_time": 218.0},  # will time out
            {"start_time": 456.0, "end_time": 490.0},  # will succeed
        ]

        success_result = MagicMock()
        success_result.stderr = "[Parsed_signalstats_1 @ 0x...] YAVG:100.00 YDIF:5.00\n"
        success_result.returncode = 0

        # With hwaccel='auto' (default) the detector probes
        # the codec via ffprobe once per video, cached on the detector.
        # Call sequence for this test: (1) ffprobe — codec probe for
        # segment 1; (2) ffmpeg chunk for segment 1 — must time out;
        # (3) ffmpeg chunk for segment 2 — succeeds (decoder is cached
        # so segment 2 does NOT re-probe).
        call_count = [0]

        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise subprocess.TimeoutExpired("ffmpeg", 60)
            return success_result

        with patch("subprocess.run", side_effect=side_effect):
            results = detector.detect_all_segments(fake_video, segments)

        assert len(results) == 1, f"Expected 1 result (timeout skipped), got {len(results)}"
        assert results[0].segment_start == 456.0


# ── 490.0 is not a valid brightness sentinel ───────────────────────────────────


class TestNoSentinelValue:
    """490.0 must not appear as a brightness return value from the detector."""

    def test_classify_490_is_overcast_but_should_never_reach_it(self):
        """490.0 falls in overcast range — confirm it classifies as overcast.

        This documents WHY using 490.0 as a fallback is wrong: it maps
        to overcast with moderate confidence, silently corrupting results.
        The fix removes this sentinel entirely.
        """
        detector = WeatherDetector()
        condition, confidence = detector._classify(490.0)
        assert condition == WeatherCondition.OVERCAST
        # Confirm it's deceptively high-confidence
        assert confidence > 0.5, "490.0 sentinel produces high-confidence overcast (wrong)"


class TestWeatherDemotedFromBrightness:
    """Brightness alone must never assert RAINY or SUNNY (demoted).

    Brightness cannot verify precipitation or clear skies, so both extreme
    bands collapse to OVERCAST — a light-level fallback, not a weather claim.
    These regression tests lock that contract so the fabricated-weather class
    (false "Rainy Day Drive" / "Sunny Day Drive" claims) cannot silently return.
    """

    def test_dark_band_reports_overcast_not_rainy(self):
        detector = WeatherDetector()
        condition, _ = detector._classify(100.0)  # well below the rainy floor
        assert condition == WeatherCondition.OVERCAST
        assert condition != WeatherCondition.RAINY

    def test_bright_band_reports_overcast_not_sunny(self):
        detector = WeatherDetector()
        condition, _ = detector._classify(900.0)  # well above the sunny ceiling
        assert condition == WeatherCondition.OVERCAST
        assert condition != WeatherCondition.SUNNY

    def test_classify_never_emits_rainy_or_sunny_across_range(self):
        detector = WeatherDetector()
        for b in range(0, 1001, 25):
            condition, _ = detector._classify(float(b))
            assert condition == WeatherCondition.OVERCAST, (
                f"brightness {b} classified {condition}; brightness must not "
                f"assert a specific weather condition"
            )


# ── chunked measurement ─────────────────────────────────────────────


class TestChunkPlanner:
    """``_plan_chunks`` splits long segments at ``chunk_max_sec`` boundaries."""

    def test_zero_duration_yields_no_chunks(self):
        assert WeatherDetector._plan_chunks(0.0, 0.0, 120.0) == []

    def test_short_segment_is_one_chunk(self):
        chunks = WeatherDetector._plan_chunks(100.0, 30.0, 120.0)
        assert chunks == [(100.0, 30.0)]

    def test_exact_boundary_is_one_chunk(self):
        chunks = WeatherDetector._plan_chunks(0.0, 120.0, 120.0)
        assert chunks == [(0.0, 120.0)]

    def test_long_segment_splits_into_chunks(self):
        chunks = WeatherDetector._plan_chunks(0.0, 1800.0, 120.0)
        assert len(chunks) == 15
        assert chunks[0] == (0.0, 120.0)
        assert chunks[-1] == (1680.0, 120.0)
        # No gaps, no overlaps
        for i in range(1, len(chunks)):
            assert chunks[i][0] == chunks[i - 1][0] + chunks[i - 1][1]

    def test_non_multiple_duration_handles_remainder(self):
        chunks = WeatherDetector._plan_chunks(10.0, 250.0, 100.0)
        assert chunks == [(10.0, 100.0), (110.0, 100.0), (210.0, 50.0)]


class TestChunkedAggregation:
    """``_get_avg_brightness_ffmpeg`` aggregates YAVG values across chunks
    and tolerates partial chunk failures up to ``min_success_ratio``."""

    def test_all_chunks_succeed_returns_mean_scaled(self):
        """Two chunks each return YAVG=128.0 → aggregate mean = 128.0
        → scaled = 128.0 * 1000/255 ≈ 501.96."""
        detector = WeatherDetector(chunk_max_sec=100.0)
        chunk_call_count = [0]

        def fake_run(cmd, **kwargs):
            chunk_call_count[0] += 1
            mock = MagicMock()
            mock.stderr = "[Parsed_signalstats_1] YAVG:128.0"
            mock.returncode = 0
            return mock

        with patch("subprocess.run", side_effect=fake_run):
            result = detector._get_avg_brightness_ffmpeg(Path("/fake/v.mp4"), 0.0, 200.0)
        # With hwaccel="auto" (default) the detector probes the
        # source codec once per video via ffprobe (cached per-path on the
        # detector), then runs N chunks. 200s / 100s chunk_max = 2 chunks +
        # 1 codec probe = 3 subprocess.run calls.
        assert chunk_call_count[0] == 3
        assert result is not None
        assert abs(result - 128.0 * 1000.0 / 255.0) < 0.01

    def test_partial_failure_above_min_ratio_still_returns(self):
        """4 chunks: 3 succeed (with YAVG=100), 1 times out. Default
        min_success_ratio=0.5 means we need 2/4 — 3/4 passes."""
        detector = WeatherDetector(chunk_max_sec=100.0, min_success_ratio=0.5)
        call_count = [0]

        def fake_run(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise subprocess.TimeoutExpired("ffmpeg", 30)
            mock = MagicMock()
            mock.stderr = "[Parsed_signalstats_1] YAVG:100.0"
            mock.returncode = 0
            return mock

        with patch("subprocess.run", side_effect=fake_run):
            result = detector._get_avg_brightness_ffmpeg(Path("/fake/v.mp4"), 0.0, 400.0)
        assert result is not None  # 3/4 chunks ok ≥ 2 required
        assert abs(result - 100.0 * 1000.0 / 255.0) < 0.01

    def test_below_min_ratio_returns_none_and_logs_error(self, caplog):
        """3 chunks, only 1 returns data (1/3 < 0.5 floor → 1 < ceil(3*0.5)=2).
        Method returns None and logs ERROR (not WARNING) to keep it
        visible to the operator."""
        import logging

        detector = WeatherDetector(chunk_max_sec=100.0, min_success_ratio=0.5)
        call_count = [0]

        def fake_run(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] >= 2:
                raise subprocess.TimeoutExpired("ffmpeg", 30)
            mock = MagicMock()
            mock.stderr = "[Parsed_signalstats_1] YAVG:90.0"
            mock.returncode = 0
            return mock

        with caplog.at_level(logging.ERROR, logger="gomoveshift.pipeline.weather"):
            with patch("subprocess.run", side_effect=fake_run):
                result = detector._get_avg_brightness_ffmpeg(Path("/fake/v.mp4"), 0.0, 300.0)
        assert result is None
        assert any(
            "degraded" in rec.message and rec.levelno == logging.ERROR for rec in caplog.records
        ), "expected ERROR-level log naming the degradation"

    def test_min_success_ratio_validation(self):
        """Constructor rejects out-of-range min_success_ratio."""
        import pytest

        with pytest.raises(ValueError):
            WeatherDetector(min_success_ratio=0.0)
        with pytest.raises(ValueError):
            WeatherDetector(min_success_ratio=1.5)
        with pytest.raises(ValueError):
            WeatherDetector(chunk_max_sec=0.0)


# ── signalstats parser: legacy + modern ffmpeg formats ────


class TestYavgParser:
    """Modern ffmpeg (≥ 5.x) emits signalstats values as frame metadata
    (``lavfi.signalstats.YAVG=132.295``) via the ``metadata=print`` filter.
    Legacy builds emitted ``[Parsed_signalstats_*] YAVG:128.00``. The
    parser must accept both — the legacy shape stays in the wild on
    older CI hosts and historical fixtures."""

    def test_parses_modern_metadata_format(self):
        stderr = (
            "[Parsed_metadata_2 @ 0x...] lavfi.signalstats.YMIN=0\n"
            "[Parsed_metadata_2 @ 0x...] lavfi.signalstats.YAVG=132.295\n"
            "[Parsed_metadata_2 @ 0x...] lavfi.signalstats.YHIGH=213\n"
            "[Parsed_metadata_2 @ 0x...] lavfi.signalstats.YAVG=100.0\n"
        )
        assert _parse_yavg_lines(stderr) == [132.295, 100.0]

    def test_parses_legacy_stderr_format(self):
        stderr = (
            "[Parsed_signalstats_1 @ 0x...] YAVG:128.0 YDIF:5.0\n"
            "[Parsed_signalstats_1 @ 0x...] YAVG:130.5 YDIF:5.5\n"
        )
        assert _parse_yavg_lines(stderr) == [128.0, 130.5]

    def test_parses_mixed_format(self):
        stderr = (
            "[Parsed_signalstats_1 @ 0x...] YAVG:128.0 YDIF:5.0\n"
            "[Parsed_metadata_2 @ 0x...] lavfi.signalstats.YAVG=200.5\n"
        )
        assert _parse_yavg_lines(stderr) == [128.0, 200.5]

    def test_empty_input_yields_empty_list(self):
        assert _parse_yavg_lines("") == []
        assert _parse_yavg_lines("no relevant lines\nnothing here") == []

    def test_filter_chain_includes_metadata_print(self):
        """_measure_chunk must pipe signalstats into `metadata=print` so
        the YAVG metadata reaches stderr on modern ffmpeg builds. This
        is the root-cause fix for a failure mode
        (0/15 chunks parsed despite clean ffmpeg exits)."""
        detector = WeatherDetector(hwaccel="none")
        captured: list[list[str]] = []

        def fake_run(cmd, **kwargs):
            captured.append(cmd)
            mock = MagicMock()
            mock.stderr = "[Parsed_metadata_2 @ 0x...] lavfi.signalstats.YAVG=128.0"
            mock.returncode = 0
            return mock

        with patch("subprocess.run", side_effect=fake_run):
            detector._measure_chunk(Path("/fake.mp4"), 0.0, 30.0, 60.0)
        assert captured
        cmd = captured[0]
        assert "-vf" in cmd
        vf_value = cmd[cmd.index("-vf") + 1]
        assert "signalstats" in vf_value
        assert "metadata=print" in vf_value, (
            f"filter chain must include metadata=print; got {vf_value!r}"
        )

    def test_clean_exit_with_zero_yavg_raises(self):
        """ffmpeg returncode=0 but zero YAVG samples = filter/parser
        contract mismatch, NOT slow IO. Must raise WeatherMeasurementError."""
        import pytest

        from gomoveshift.pipeline.exceptions import WeatherMeasurementError

        detector = WeatherDetector(hwaccel="none")

        def fake_run(cmd, **kwargs):
            mock = MagicMock()
            mock.stderr = "frame=120 fps=3.3 ... time=00:02:00 ..."  # no YAVG
            mock.returncode = 0
            return mock

        with patch("subprocess.run", side_effect=fake_run):
            with pytest.raises(WeatherMeasurementError, match="no YAVG samples"):
                detector._measure_chunk(Path("/fake.mp4"), 0.0, 30.0, 60.0)

    def test_timeout_still_returns_empty_not_raise(self):
        """A genuine ffmpeg timeout is a different failure class — it
        returns [] (degraded-chunk signal) so the aggregator can apply
        the min_success_ratio policy."""
        import subprocess as sp

        detector = WeatherDetector(hwaccel="none")
        with patch("subprocess.run", side_effect=sp.TimeoutExpired("ffmpeg", 60)):
            result = detector._measure_chunk(Path("/fake.mp4"), 0.0, 30.0, 60.0)
        assert result == []
