"""
Regression tests for silent degradation in single_pass.

Each fallback path that previously logged at INFO/DEBUG — or logged no
self-signalling warning at the fallback site — must now emit a
``logger.warning`` that describes what was tried, the trigger, and the
fallback taken. These tests lock in that contract.
"""

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.single_pass import SinglePassProcessor


# ── single_pass.SinglePassProcessor._measure_audio ─────────────────────


class TestSinglePassMeasureAudioWarnings:
    """_measure_audio must loudly warn whenever the two-pass measurement
    silently degrades to the single-pass fallback."""

    def _make_processor(self):
        with patch("gomoveshift.pipeline.single_pass.detect_nvenc", return_value=False):
            return SinglePassProcessor(Config())

    def test_missing_json_block_logs_warning(self, caplog):
        processor = self._make_processor()
        fake_result = CompletedProcess(args=[], returncode=0, stdout="", stderr="no json here")

        with (
            caplog.at_level("WARNING", logger="gomoveshift.pipeline.single_pass"),
            patch(
                "gomoveshift.pipeline.single_pass.run_ffmpeg_quick",
                return_value=fake_result,
            ),
        ):
            measured = processor._measure_audio(Path("/tmp/fake.mp4"), duration=600.0, height=1080)

        assert measured == {}, "missing JSON must fall back to empty measured dict"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "expected at least one WARNING on missing JSON"
        msg = " ".join(r.getMessage() for r in warnings)
        assert "single_pass" in msg
        assert "single-pass loudnorm" in msg

    def test_malformed_json_logs_warning(self, caplog):
        processor = self._make_processor()
        bad_json = '{\n  "input_i": not-a-number }'
        fake_result = CompletedProcess(args=[], returncode=0, stdout="", stderr=bad_json)

        with (
            caplog.at_level("WARNING", logger="gomoveshift.pipeline.single_pass"),
            patch(
                "gomoveshift.pipeline.single_pass.run_ffmpeg_quick",
                return_value=fake_result,
            ),
        ):
            measured = processor._measure_audio(Path("/tmp/fake.mp4"), duration=600.0, height=1080)

        assert measured == {}, "malformed JSON must fall back to empty measured dict"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "expected a WARNING on JSONDecodeError (previously silent)"
        msg = " ".join(r.getMessage() for r in warnings)
        assert "single-pass loudnorm" in msg


# ── single_pass.SinglePassProcessor._run_audio_step ────────────────────


class TestSinglePassAudioStepFallbackWarning:
    """Empty measurement dict must raise a WARNING (previously INFO)."""

    def _make_processor(self):
        with patch("gomoveshift.pipeline.single_pass.detect_nvenc", return_value=False):
            return SinglePassProcessor(Config())

    def test_empty_measurement_emits_warning(self, caplog):
        processor = self._make_processor()

        with (
            caplog.at_level("WARNING", logger="gomoveshift.pipeline.single_pass"),
            patch.object(processor, "_measure_audio", return_value={}),
        ):
            result = processor._run_audio_step(
                Path("/tmp/fake.mp4"),
                duration=600.0,
                height=1080,
                skip_audio=False,
            )

        assert result == {}, "must return the empty dict so callers fall back to single-pass"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "audio fallback must WARN, not INFO"
        msg = " ".join(r.getMessage() for r in warnings)
        assert "single-pass loudnorm" in msg or "falling back" in msg


# The former ``TestAudioMeasureLoudnessWarnings`` and
# ``TestAudioApplyNormalizationFallbackWarning`` classes were removed with
# their subject: ``gomoveshift/pipeline/audio.py``'s ``AudioProcessor`` (a dead
# two-pass ``loudnorm`` module with no production importer, and the only path
# still emitting the forbidden ``linear=true``) has since been deleted.
# The live audio path is single_pass /
# ost_applier's attenuate-only filter, covered above and in test_ost_applier.py.


# ── ffmpeg_runner moov-rewrite stall-timeout extension ────────────────


class TestFfmpegRunnerMoovStallWarning:
    """The moov-rewrite stall-timeout extension previously logged at
    INFO; it must now log at WARNING so operators can correlate ffmpeg
    stalls with the suppressed detection window."""

    def test_moov_rewrite_logs_warning(self, tmp_path, caplog):
        import textwrap

        from gomoveshift.pipeline.ffmpeg_runner import run_ffmpeg_encode

        output = tmp_path / "out.mp4"
        output.write_bytes(b"fake video data")

        # Stub ffmpeg: emit the trigger line, then a progress line, then exit.
        cmd = [
            "python3",
            "-c",
            textwrap.dedent("""\
            import sys
            print("moving the moov atom", file=sys.stderr, flush=True)
            print("frame=  100 fps= 60.0 size=   1024kB time=00:00:10.00 speed=1.0x",
                  file=sys.stderr, flush=True)
            """),
        ]

        with caplog.at_level("WARNING", logger="gomoveshift.pipeline.ffmpeg_runner"):
            result = run_ffmpeg_encode(
                cmd,
                output,
                expected_duration=30.0,
                description="test moov warn",
                # A real-clock stub: the window must outlast interpreter startup
                # on a loaded machine (a short window would kill it as a stall).
                # The extension itself is proven with a scripted clock
                # in test_ffmpeg_runner.py::test_moov_rewrite_extends_stall.
                stall_timeout=300,
                log_interval=1,
            )

        assert result.success, f"killed_reason={result.killed_reason}"
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "moov-rewrite stall-timeout extension must WARN"
        msg = " ".join(r.getMessage() for r in warnings)
        assert "moov" in msg.lower()
        assert "stall" in msg.lower()


# The former ``TestSinglePassEncoderOutputLoudnessDrift`` class was removed:
# its subject ``single_pass._check_encoded_loudness_drift`` (and the
# ``_OUTPUT_INTEGRATED_RE`` it used) has since been deleted.
# The day master no longer runs loudnorm, so there is no output
# loudness "target" to drift from — the attenuate-only filter can only reduce
# loudness, and the ceiling-only phase_d gate covers the remaining risk.


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
