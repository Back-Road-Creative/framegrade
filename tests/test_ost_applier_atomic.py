"""A killed OST encode must never leave a finished-looking artifact.

apply_ost() used to hand ffmpeg the FINAL output path directly. Cleanup
(_discard_partial_output / run_ffmpeg_encode's own cleanup-on-failure) only
runs if the Python process survives to reach it — a SIGKILL, OOM, power
loss, or an operator Ctrl-C mid-encode leaves whatever ffmpeg had flushed
sitting at the FINAL path with nothing to remove it. Resume then adopts a
truncated file as a finished LF (day_pipeline.py's superlong resume site
checks only ``size > 0``).

These tests prove the property, not the mechanism: no failure mode —
including one where every in-process cleanup handler is bypassed — can
leave bytes at the final path. Only a validated artifact ever lands there.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.ffmpeg_runner import FFmpegResult, VideoValidation

from ost_subprocess_harness import patch_ost_run


def _stub_probe(duration: float = 600.0) -> dict:
    return {
        "width": 3840,
        "height": 2160,
        "fps": 29.97,
        "bitrate_kbps": 45_000,
        "duration": duration,
    }


@pytest.fixture
def paths(tmp_path):
    source = tmp_path / "lf_01_with_montage.mp4"
    source.write_bytes(b"\x00" * 4096)
    font = tmp_path / "font.ttf"
    font.write_bytes(b"fake font")
    return source, tmp_path / "lf_01.mp4", font


def _writes_to_cmd_target(payload: bytes = b"\xff" * 4096):
    """Write ``payload`` to wherever the real cmd told ffmpeg to write —
    NOT to a path the test hardcodes. Proves apply_ost's actual encode
    target, whatever it is, rather than asserting on the implementation."""

    def _side_effect(cmd, *_a, **_kw):
        target = cmd[-1]
        # Same stdout guard as test_ost_applier_temp_muxer: "-" is not a path.
        if target == "-":
            return MagicMock(returncode=0, stdout=b"", stderr=b"")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(payload)
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    return _side_effect


def _run_apply_ost(paths, *, run_side_effect, validation: VideoValidation):
    source, output, font = paths
    with (
        patch(
            "gomoveshift.pipeline.ost_applier._probe_video_info",
            return_value=_stub_probe(),
        ),
        patch("gomoveshift.pipeline.ost_applier._has_audio_stream", return_value=True),
        patch("gomoveshift.pipeline.ost_applier._measure_audio_loudnorm", return_value={}),
        patch_ost_run(side_effect=run_side_effect),
        patch("gomoveshift.pipeline.ost_applier.validate_video", return_value=validation),
        patch("gomoveshift.pipeline.color.detect_nvenc", return_value=True),
    ):
        result = ost_applier.apply_ost(
            input_path=source,
            output_path=output,
            location="Healy Pass, County Cork",
            layout="regular_lf",
            font_path=str(font),
            prefer_nvenc=False,
        )
    return result


def _stray_files(output: Path):
    """Anything left behind under output's own name (the final artifact or
    any working file for it), excluding the final path itself."""
    if not output.parent.exists():
        return []
    return [p for p in output.parent.iterdir() if p.name.startswith(output.name) and p != output]


class TestFailedValidationLeavesNoFinal:
    def test_invalid_artifact_leaves_nothing_at_the_final_path(self, paths):
        _, output, _ = paths
        result = _run_apply_ost(
            paths,
            run_side_effect=_writes_to_cmd_target(),
            validation=VideoValidation(valid=False, reason="moov atom not found"),
        )
        assert result is None
        assert not output.exists(), "a failed-validation encode must not occupy the final path"
        assert _stray_files(output) == [], "no working file should survive a failed encode either"


class TestKilledEncodeLeavesNoFinal:
    def test_a_kill_that_skips_every_cleanup_handler_still_leaves_no_final(self, paths):
        """Simulate the worst case directly: run_ffmpeg_encode is replaced
        entirely, so NONE of its own cleanup-on-failure code runs (mirrors
        a SIGKILL that takes the whole interpreter down mid-encode). ffmpeg
        still flushed real bytes to whatever it was told to write to. The
        final path must be untouched anyway — because it was never the
        write target, not because a handler cleaned it up after the fact.
        """
        source, output, font = paths

        def _kill_no_cleanup(cmd, **_kw):
            target = Path(cmd[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00" * 1024)  # what a kill mid-write leaves
            return FFmpegResult(
                success=False,
                returncode=-9,
                wall_time=1.0,
                stderr_tail="",
                killed_reason="killed",
            )

        with (
            patch("gomoveshift.pipeline.ost_applier._probe_video_info", return_value=_stub_probe()),
            patch("gomoveshift.pipeline.ost_applier._has_audio_stream", return_value=True),
            patch("gomoveshift.pipeline.ost_applier._measure_audio_loudnorm", return_value={}),
            patch(
                "gomoveshift.pipeline.ost_applier.run_ffmpeg_encode", side_effect=_kill_no_cleanup
            ),
            patch("gomoveshift.pipeline.color.detect_nvenc", return_value=True),
        ):
            result = ost_applier.apply_ost(
                input_path=source,
                output_path=output,
                location="Healy Pass, County Cork",
                layout="regular_lf",
                font_path=str(font),
                prefer_nvenc=False,
            )

        assert result is None
        assert not output.exists(), (
            "even with zero in-process cleanup, a killed encode must not leave "
            "a finished-looking artifact at the final path"
        )


class TestSuccessPromotesToFinal:
    def test_valid_artifact_lands_at_final_with_no_leftover_working_file(self, paths):
        _, output, _ = paths
        payload = b"\xaa" * 4096
        result = _run_apply_ost(
            paths,
            run_side_effect=_writes_to_cmd_target(payload),
            validation=VideoValidation(valid=True, duration=600.0, file_size=4096),
        )
        assert result == output
        assert output.exists()
        assert output.read_bytes() == payload
        assert _stray_files(output) == [], "the working file must not survive a successful encode"


class TestPreExistingGoodFinalSurvivesAFailedReencode:
    def test_good_final_untouched_by_a_later_failed_reencode(self, paths):
        _, output, _ = paths
        good_bytes = b"GOOD-PREVIOUS-ARTIFACT" * 100
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(good_bytes)

        result = _run_apply_ost(
            paths,
            run_side_effect=_writes_to_cmd_target(b"\x00" * 16),
            validation=VideoValidation(valid=False, reason="truncated re-encode"),
        )

        assert result is None
        assert output.exists(), "a pre-existing good final must survive a failed re-encode"
        assert output.read_bytes() == good_bytes, "the good final's bytes must be untouched"
