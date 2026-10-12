"""
Tests for batched shorts rendering optimization.

Verifies that render_shorts_batched() produces the same number of outputs
as individual rendering, that outputs are valid, and that fallback to
individual rendering works when batching fails.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.shorts_extractor import (
    DEFAULT_BATCH_SIZE,
    extract_segment_streamcopy,
    render_short_ffmpeg,
    render_shorts_batched,
)


# ── Fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def fake_source(tmp_path):
    """Create a fake source video file."""
    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00" * 1024)
    return src


@pytest.fixture
def sample_specs(tmp_path):
    """Build specs for 6 shorts (enough to test batching)."""
    output_dir = tmp_path / "shorts"
    output_dir.mkdir()
    specs = []
    for i in range(6):
        specs.append(
            {
                "index": i,
                "start_time": 60.0 * i,
                "duration": 35.0,
                "location": f"Location {i}",
                "subscribe_text": "Subscribe for more",
                "output_path": str(output_dir / f"short_{i:02d}.mp4"),
            }
        )
    return specs


# ── extract_segment_streamcopy ───────────────────────────────────────────


class TestNoLoudnormRetry:
    """render_short_ffmpeg must NOT retry on failure (the retry path was removed).

    Any FFmpeg failure returns False immediately after a single
    invocation. Under the muted-shorts policy there is no separate audio
    MEASUREMENT pass anymore either, so a failing render is exactly ONE ffmpeg
    call regardless of skip_audio_norm.
    """

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor._detect_nvenc_cached", return_value=False)
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    def test_no_retry_when_audio_norm_requested(
        self, mock_dims, mock_nvenc, mock_run, fake_source, tmp_path
    ):
        """A failing encode returns False after ONE call — no retry, no measurement pass."""
        output = tmp_path / "short.mp4"
        mock_run.return_value = MagicMock(returncode=1, stderr="Conversion failed!", stdout="")

        result = render_short_ffmpeg(
            source_video=fake_source,
            output_path=output,
            start_time=10.0,
            duration=35.0,
            location="Test",
            skip_audio_norm=False,
        )

        assert result is False, "Should fail without retry"
        # Muted shorts: a single encode call (no measurement pass), no retry.
        assert mock_run.call_count == 1, "one encode, no measurement, no retry"

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor._detect_nvenc_cached", return_value=False)
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    def test_no_retry_when_loudnorm_already_skipped(
        self, mock_dims, mock_nvenc, mock_run, fake_source, tmp_path
    ):
        """When skip_audio_norm=True and FFmpeg fails, don't retry (same as before)."""
        output = tmp_path / "short.mp4"
        mock_run.return_value = MagicMock(returncode=1, stderr="Some error")

        result = render_short_ffmpeg(
            source_video=fake_source,
            output_path=output,
            start_time=10.0,
            duration=35.0,
            location="Test",
            skip_audio_norm=True,
        )

        assert result is False
        assert mock_run.call_count == 1, "Should not retry when loudnorm already skipped"


class TestShortFailureCapture:
    """On FFmpeg failure, full stderr + command line are written to _short_failures/."""

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor._detect_nvenc_cached", return_value=False)
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    def test_failure_writes_full_stderr_and_cmd_logs(
        self, mock_dims, mock_nvenc, mock_run, fake_source, tmp_path
    ):
        """On non-zero returncode, write untruncated stderr and exact command line."""
        output = tmp_path / "out" / "short_07_rank3.mp4"
        # Use a stderr longer than 500 chars to verify no truncation
        big_stderr = "line:" + ("A" * 1000) + "\nerror tail\n"
        mock_run.return_value = MagicMock(returncode=1, stderr=big_stderr, stdout="")

        result = render_short_ffmpeg(
            source_video=fake_source,
            output_path=output,
            start_time=10.0,
            duration=35.0,
            location="Test",
        )

        assert result is False
        failures_dir = output.parent / "_short_failures"
        stderr_log = failures_dir / f"{output.stem}.stderr.log"
        cmd_log = failures_dir / f"{output.stem}.cmd.log"

        assert failures_dir.is_dir(), "_short_failures/ directory must be created on failure"
        assert stderr_log.exists(), "stderr.log must be written"
        assert cmd_log.exists(), "cmd.log must be written"

        stderr_contents = stderr_log.read_text()
        # Full stderr — no [-500:] truncation
        assert big_stderr == stderr_contents, "stderr must be written in full, not truncated"
        assert len(stderr_contents) > 500

        cmd_contents = cmd_log.read_text()
        assert "ffmpeg" in cmd_contents
        assert str(fake_source) in cmd_contents

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    @patch("gomoveshift.pipeline.shorts_extractor._detect_nvenc_cached", return_value=False)
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions", return_value=(1920, 1080))
    def test_success_does_not_create_failures_dir(
        self, mock_dims, mock_nvenc, mock_run, fake_source, tmp_path
    ):
        """Golden/baseline: successful render writes output and creates no _short_failures/."""
        output = tmp_path / "out" / "short_00_rank1.mp4"

        def _side_effect(*args, **kwargs):
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"\x00" * 200_000)
            return MagicMock(returncode=0, stdout="", stderr="")

        mock_run.side_effect = _side_effect

        with patch("gomoveshift.pipeline.ffmpeg_runner.subprocess") as mock_ffr_sub:
            mock_ffr_sub.run.return_value = MagicMock(
                returncode=0,
                stdout='{"streams":[{"codec_type":"video","duration":"35.0"},'
                '{"codec_type":"audio","duration":"35.0"}],'
                '"format":{"duration":"35.0"}}',
                stderr="",
            )
            mock_ffr_sub.TimeoutExpired = TimeoutError

            result = render_short_ffmpeg(
                source_video=fake_source,
                output_path=output,
                start_time=10.0,
                duration=35.0,
                location="Test",
                skip_audio_norm=False,
            )

        assert result is True, "Golden path — render should succeed"
        # Muted shorts: a single encode call (no measurement pass) on success.
        assert mock_run.call_count == 1, "one encode on success (no measurement pass)"
        assert output.exists()
        # Success path must not leave a _short_failures/ dir behind
        assert not (output.parent / "_short_failures").exists()


class TestExtractSegmentStreamcopy:
    """Tests for the stream-copy segment extraction."""

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    def test_success_returns_true(self, mock_run, fake_source, tmp_path):
        """Successful stream-copy returns True and creates output."""
        output = tmp_path / "seg.mp4"

        def _side_effect(*args, **kwargs):
            # Simulate ffmpeg creating the file
            output.write_bytes(b"\x00" * 512)
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = _side_effect

        result = extract_segment_streamcopy(
            source_video=fake_source,
            output_path=output,
            start_time=10.0,
            duration=35.0,
        )

        assert result is True
        assert output.exists()

        # Verify ffmpeg was called with stream-copy flags
        cmd = mock_run.call_args[0][0]
        assert "ffmpeg" in cmd[0]
        assert "-c" in cmd
        assert "copy" in cmd
        assert "-ss" in cmd

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    def test_failure_returns_false(self, mock_run, fake_source, tmp_path):
        """Failed ffmpeg returns False."""
        output = tmp_path / "seg.mp4"
        mock_run.return_value = MagicMock(returncode=1, stderr="error")

        result = extract_segment_streamcopy(
            source_video=fake_source,
            output_path=output,
            start_time=10.0,
            duration=35.0,
        )

        assert result is False

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    def test_timeout_returns_false(self, mock_run, fake_source, tmp_path):
        """Timed-out ffmpeg returns False."""
        output = tmp_path / "seg.mp4"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=120)

        result = extract_segment_streamcopy(
            source_video=fake_source,
            output_path=output,
            start_time=10.0,
            duration=35.0,
        )

        assert result is False

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    def test_empty_output_returns_false(self, mock_run, fake_source, tmp_path):
        """Empty output file is treated as failure."""
        output = tmp_path / "seg.mp4"

        def _side_effect(*args, **kwargs):
            output.write_bytes(b"")  # empty file
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = _side_effect

        result = extract_segment_streamcopy(
            source_video=fake_source,
            output_path=output,
            start_time=10.0,
            duration=35.0,
        )

        assert result is False

    @patch("gomoveshift.pipeline.shorts_extractor.subprocess.run")
    def test_creates_parent_directories(self, mock_run, fake_source, tmp_path):
        """Output parent directories are created automatically."""
        output = tmp_path / "deep" / "nested" / "seg.mp4"

        def _side_effect(*args, **kwargs):
            output.write_bytes(b"\x00" * 512)
            return MagicMock(returncode=0, stderr="")

        mock_run.side_effect = _side_effect

        result = extract_segment_streamcopy(
            source_video=fake_source,
            output_path=output,
            start_time=10.0,
            duration=35.0,
        )

        assert result is True
        assert output.parent.exists()


# ── render_shorts_batched ────────────────────────────────────────────────


class TestRenderShortsBatched:
    """Tests for the two-phase batched rendering."""

    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor.extract_segment_streamcopy")
    def test_output_count_matches_input(
        self, mock_extract, mock_render, fake_source, sample_specs, tmp_path
    ):
        """Batched rendering produces a result for every input spec."""
        # Phase 1: all extractions succeed
        mock_extract.return_value = True
        # Phase 2: all encodes succeed
        mock_render.return_value = True

        results = render_shorts_batched(
            source_video=fake_source,
            shorts_specs=sample_specs,
            output_dir=tmp_path / "shorts",
            src_width=3840,
            src_height=2160,
        )

        assert len(results) == len(sample_specs)
        assert all(v is True for v in results.values())

    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor.extract_segment_streamcopy")
    def test_extract_failure_marks_short_as_failed(
        self, mock_extract, mock_render, fake_source, sample_specs, tmp_path
    ):
        """If stream-copy fails for a short, that short is marked failed."""
        # First extract fails, rest succeed
        mock_extract.side_effect = [False] + [True] * 5
        mock_render.return_value = True

        results = render_shorts_batched(
            source_video=fake_source,
            shorts_specs=sample_specs,
            output_dir=tmp_path / "shorts",
            src_width=3840,
            src_height=2160,
        )

        assert results[0] is False
        for i in range(1, 6):
            assert results[i] is True

    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor.extract_segment_streamcopy")
    def test_encode_failure_marks_short_as_failed(
        self, mock_extract, mock_render, fake_source, sample_specs, tmp_path
    ):
        """If encode fails for a short, that short is marked failed."""
        mock_extract.return_value = True
        # Fail the third encode
        mock_render.side_effect = [True, True, False, True, True, True]

        results = render_shorts_batched(
            source_video=fake_source,
            shorts_specs=sample_specs,
            output_dir=tmp_path / "shorts",
            src_width=3840,
            src_height=2160,
        )

        failed_count = sum(1 for v in results.values() if not v)
        assert failed_count == 1

    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor.extract_segment_streamcopy")
    def test_encode_uses_segment_not_source(self, mock_extract, mock_render, fake_source, tmp_path):
        """Phase 2 encode reads from the extracted segment, not the source."""
        mock_extract.return_value = True
        mock_render.return_value = True

        specs = [
            {
                "index": 0,
                "start_time": 60.0,
                "duration": 35.0,
                "location": "Test Location",
                "subscribe_text": "Subscribe",
                "output_path": str(tmp_path / "short_00.mp4"),
            }
        ]

        render_shorts_batched(
            source_video=fake_source,
            shorts_specs=specs,
            output_dir=tmp_path / "shorts",
            src_width=3840,
            src_height=2160,
        )

        # render_short_ffmpeg should receive a segment path, not the source
        render_call = mock_render.call_args
        source_arg = render_call.kwargs.get(
            "source_video", render_call.args[0] if render_call.args else None
        )
        # The source should be in _segments dir, not the original source
        assert "_segments" in str(source_arg)
        # start_time should be 0 (segment already trimmed)
        assert render_call.kwargs.get("start_time", None) == 0.0

    @patch("gomoveshift.pipeline.shorts_extractor._probe_video_stream_duration")
    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor.extract_segment_streamcopy")
    def test_shorts_past_video_duration_skipped(
        self, mock_extract, mock_render, mock_probe, fake_source, tmp_path
    ):
        """Shorts whose time range exceeds video stream duration are skipped.

        When an LF has A/V drift (audio longer than video), shorts near the
        end would get zero video frames causing encoder EOF errors.
        """
        mock_extract.return_value = True
        mock_render.return_value = True
        # Source video stream is only 200s long
        mock_probe.return_value = 200.0

        specs = [
            {
                "index": 0,
                "start_time": 10.0,
                "duration": 35.0,
                "location": "OK",
                "subscribe_text": "Sub",
                "output_path": str(tmp_path / "short_00.mp4"),
            },
            {
                "index": 1,
                "start_time": 180.0,
                "duration": 35.0,  # ends at 215s > 200s video
                "location": "Past end",
                "subscribe_text": "Sub",
                "output_path": str(tmp_path / "short_01.mp4"),
            },
        ]

        results = render_shorts_batched(
            source_video=fake_source,
            shorts_specs=specs,
            output_dir=tmp_path / "shorts",
            src_width=3840,
            src_height=2160,
        )

        # Short 0 should succeed, short 1 should be skipped (marked False)
        assert results[0] is True
        assert results[1] is False
        # Only 1 extraction should happen (short 1 skipped before extraction)
        assert mock_extract.call_count == 1

    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor.extract_segment_streamcopy")
    def test_empty_specs_returns_empty(self, mock_extract, mock_render, fake_source, tmp_path):
        """Empty input produces empty results."""
        results = render_shorts_batched(
            source_video=fake_source,
            shorts_specs=[],
            output_dir=tmp_path / "shorts",
            src_width=3840,
            src_height=2160,
        )

        assert results == {}
        mock_extract.assert_not_called()
        mock_render.assert_not_called()

    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor.extract_segment_streamcopy")
    def test_progress_callback_called(
        self, mock_extract, mock_render, fake_source, sample_specs, tmp_path
    ):
        """Progress callback is invoked for each completed short."""
        mock_extract.return_value = True
        mock_render.return_value = True
        progress = MagicMock()

        render_shorts_batched(
            source_video=fake_source,
            shorts_specs=sample_specs,
            output_dir=tmp_path / "shorts",
            src_width=3840,
            src_height=2160,
            progress_callback=progress,
        )

        # Should be called once per short during encode phase
        assert progress.call_count == len(sample_specs)

    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor.extract_segment_streamcopy")
    def test_intermediate_segments_cleaned_up(
        self, mock_extract, mock_render, fake_source, sample_specs, tmp_path
    ):
        """Intermediate segment files are removed after encoding."""
        output_dir = tmp_path / "shorts"
        segments_dir = output_dir / "_segments"

        mock_extract.return_value = True
        mock_render.return_value = True

        render_shorts_batched(
            source_video=fake_source,
            shorts_specs=sample_specs,
            output_dir=output_dir,
            src_width=3840,
            src_height=2160,
        )

        # _segments dir should be cleaned up (or empty)
        if segments_dir.exists():
            remaining = list(segments_dir.iterdir())
            assert len(remaining) == 0, f"Leftover segments: {remaining}"


# ── PipelineShortsExtractor fallback integration ─────────────────────────


class TestBatchedFallback:
    """Tests that PipelineShortsExtractor falls back to individual rendering."""

    @patch("gomoveshift.pipeline.shorts_extractor.render_shorts_batched")
    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor._detect_nvenc_cached")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions")
    def test_fallback_on_batch_exception(
        self, mock_dims, mock_nvenc, mock_render, mock_batch, tmp_path
    ):
        """If batched rendering raises, falls back to individual rendering."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        mock_dims.return_value = (3840, 2160)
        mock_nvenc.return_value = False
        mock_batch.side_effect = RuntimeError("GPU memory error")
        mock_render.return_value = True

        mock_short = MagicMock()
        mock_short.index = 0
        mock_short.rank = 1
        mock_short.start_time = 10.0
        mock_short.end_time = 45.0
        mock_short.duration = 35.0
        mock_short.avg_motion_energy = 0.05
        mock_short.peak_motion_energy = 0.08

        mock_extraction = MagicMock()
        mock_extraction.shorts = [mock_short]
        mock_extraction.num_shorts_found = 1
        mock_extraction.source_duration = 300.0
        mock_extraction.dead_spots_avoided = 0

        video_path = tmp_path / "source.mp4"
        video_path.write_bytes(b"\x00" * 1024)
        output_dir = tmp_path / "shorts"

        with (
            patch(
                "gomoveshift.pipeline.cache.is_analysis_cached",
                return_value=False,
            ),
            patch(
                "gomoveshift.pipeline.cache.is_output_fresh",
                return_value=False,
            ),
            patch(
                "gomoveshift.analysis.shorts_extractor.ShortsExtractor.extract_shorts",
                return_value=mock_extraction,
            ),
        ):
            extractor = PipelineShortsExtractor()
            extractor.config = MagicMock()
            extractor.config.drive.skip_audio = False

            result = extractor.extract_and_render(
                video_path=str(video_path),
                output_dir=str(output_dir),
                location="Test Location",
                num_shorts=1,
            )

        # Batch was attempted and failed
        mock_batch.assert_called_once()
        # Individual render was used as fallback
        mock_render.assert_called()
        # The short was rendered
        assert result.num_rendered == 1

    @patch("gomoveshift.pipeline.shorts_extractor.render_shorts_batched")
    @patch("gomoveshift.pipeline.shorts_extractor.render_short_ffmpeg")
    @patch("gomoveshift.pipeline.shorts_extractor._detect_nvenc_cached")
    @patch("gomoveshift.pipeline.shorts_extractor.get_video_dimensions")
    def test_partial_batch_failure_falls_back_for_failed_shorts(
        self, mock_dims, mock_nvenc, mock_render, mock_batch, tmp_path
    ):
        """Shorts that fail in batch are retried individually."""
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        mock_dims.return_value = (3840, 2160)
        mock_nvenc.return_value = False
        # Batch succeeds for index 0, fails for index 1
        mock_batch.return_value = {0: True, 1: False}
        mock_render.return_value = True

        mock_shorts = []
        for i in range(2):
            s = MagicMock()
            s.index = i
            s.rank = i + 1
            s.start_time = 60.0 * i
            s.end_time = 60.0 * i + 35.0
            s.duration = 35.0
            s.avg_motion_energy = 0.05
            s.peak_motion_energy = 0.08
            mock_shorts.append(s)

        mock_extraction = MagicMock()
        mock_extraction.shorts = mock_shorts
        mock_extraction.num_shorts_found = 2
        mock_extraction.source_duration = 600.0
        mock_extraction.dead_spots_avoided = 0

        video_path = tmp_path / "source.mp4"
        video_path.write_bytes(b"\x00" * 1024)
        output_dir = tmp_path / "shorts"

        with (
            patch(
                "gomoveshift.pipeline.cache.is_analysis_cached",
                return_value=False,
            ),
            patch(
                "gomoveshift.pipeline.cache.is_output_fresh",
                return_value=False,
            ),
            patch(
                "gomoveshift.analysis.shorts_extractor.ShortsExtractor.extract_shorts",
                return_value=mock_extraction,
            ),
        ):
            extractor = PipelineShortsExtractor()
            extractor.config = MagicMock()
            extractor.config.drive.skip_audio = False

            result = extractor.extract_and_render(
                video_path=str(video_path),
                output_dir=str(output_dir),
                location="Test Location",
                num_shorts=2,
            )

        # Batch was attempted
        mock_batch.assert_called_once()
        # Individual render was called for the failed short (#1)
        mock_render.assert_called()
        # Both shorts should be rendered (one via batch, one via fallback)
        assert result.num_rendered == 2


# ── OST-fingerprint-keyed render cache ─────────────


def _mock_extraction_one_short():
    s = MagicMock()
    s.index = 0
    s.rank = 1
    s.start_time = 10.0
    s.end_time = 45.0
    s.duration = 35.0
    s.avg_motion_energy = 0.05
    s.peak_motion_energy = 0.08
    ext = MagicMock()
    ext.shorts = [s]
    ext.num_shorts_found = 1
    ext.source_duration = 300.0
    ext.dead_spots_avoided = 0
    return ext


class TestOstFingerprintCache:
    """Render cache is keyed on the burned OST text, not just mtime."""

    DRIVE_LOCATION = "County Tipperary, Ireland"
    SUBSCRIBE = "Subscribe for more"

    def _run(self, tmp_path, mock_batch, location_map=None):
        from gomoveshift.pipeline.shorts_extractor import PipelineShortsExtractor

        with (
            patch(
                "gomoveshift.pipeline.cache.is_analysis_cached",
                return_value=False,
            ),
            patch(
                "gomoveshift.analysis.shorts_extractor.ShortsExtractor.extract_shorts",
                return_value=_mock_extraction_one_short(),
            ),
            patch(
                "gomoveshift.pipeline.shorts_extractor._detect_nvenc_cached",
                return_value=False,
            ),
            patch(
                "gomoveshift.pipeline.shorts_extractor.get_video_dimensions",
                return_value=(3840, 2160),
            ),
            patch(
                "gomoveshift.pipeline.shorts_extractor.render_shorts_batched",
                mock_batch,
            ),
        ):
            extractor = PipelineShortsExtractor()
            extractor.config = MagicMock()
            extractor.config.drive.skip_audio = False
            return extractor.extract_and_render(
                video_path=str(tmp_path / "source.mp4"),
                output_dir=str(tmp_path / "shorts"),
                location=self.DRIVE_LOCATION,
                num_shorts=1,
                location_map=location_map,
            )

    def _seed_rendered_short(self, tmp_path):
        """Create source + an mtime-fresh rendered short on disk."""
        video_path = tmp_path / "source.mp4"
        video_path.write_bytes(b"\x00" * 1024)
        output_dir = tmp_path / "shorts"
        output_dir.mkdir()
        rendered = output_dir / "short_00_rank1.mp4"
        rendered.write_bytes(b"\x00" * (200 * 1024))
        return output_dir

    def test_cache_hit_needs_matching_fingerprint_and_stamps_override(self, tmp_path):
        """Matching sidecar → reuse; metadata carries the per-short override."""
        from gomoveshift.pipeline.ffmpeg_filters import shorten_ost_for_shorts
        from gomoveshift.pipeline.shorts_extractor import _short_ost_fingerprint

        output_dir = self._seed_rendered_short(tmp_path)
        override = "Healy Pass, County Cork"
        burned = shorten_ost_for_shorts(override)
        (output_dir / "short_00_rank1.ost.sha256").write_text(
            _short_ost_fingerprint(burned, self.SUBSCRIBE, 10.0, 45.0)
        )

        mock_batch = MagicMock()
        result = self._run(tmp_path, mock_batch, location_map={0: override})

        mock_batch.assert_not_called()
        assert result.num_rendered == 1
        # location_map applied to CACHED shorts too — not the drive-wide text
        assert result.shorts[0].location_text == burned

    def test_corrected_ost_text_invalidates_mtime_fresh_output(self, tmp_path):
        """Sidecar for the OLD text mismatches → re-render, sidecar updated."""
        from gomoveshift.pipeline.ffmpeg_filters import shorten_ost_for_shorts
        from gomoveshift.pipeline.shorts_extractor import _short_ost_fingerprint

        output_dir = self._seed_rendered_short(tmp_path)
        old_burned = shorten_ost_for_shorts(self.DRIVE_LOCATION)
        fp_path = output_dir / "short_00_rank1.ost.sha256"
        fp_path.write_text(_short_ost_fingerprint(old_burned, self.SUBSCRIBE, 10.0, 45.0))

        override = "Healy Pass, County Cork"
        mock_batch = MagicMock(return_value={0: True})
        result = self._run(tmp_path, mock_batch, location_map={0: override})

        mock_batch.assert_called_once()
        assert result.num_rendered == 1
        new_burned = shorten_ost_for_shorts(override)
        assert fp_path.read_text().strip() == _short_ost_fingerprint(
            new_burned, self.SUBSCRIBE, 10.0, 45.0
        )

    def test_missing_fingerprint_sidecar_forces_rerender(self, tmp_path):
        """mtime-fresh output without a sidecar cannot prove its burned text."""
        from gomoveshift.pipeline.ffmpeg_filters import shorten_ost_for_shorts
        from gomoveshift.pipeline.shorts_extractor import _short_ost_fingerprint

        output_dir = self._seed_rendered_short(tmp_path)

        mock_batch = MagicMock(return_value={0: True})
        result = self._run(tmp_path, mock_batch)

        mock_batch.assert_called_once()
        assert result.num_rendered == 1
        burned = shorten_ost_for_shorts(self.DRIVE_LOCATION)
        fp_path = output_dir / "short_00_rank1.ost.sha256"
        assert fp_path.read_text().strip() == _short_ost_fingerprint(
            burned, self.SUBSCRIBE, 10.0, 45.0
        )
