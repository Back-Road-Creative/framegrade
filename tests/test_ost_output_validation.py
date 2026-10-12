"""apply_ost must validate the artifact it just burned.

``apply_ost`` is the ONLY re-encode that produces a drive-path LF final, and it
used to check ``returncode`` alone. Every other encode in the pipeline runs
``validate_video`` (moov atom present, video stream present, duration within
tolerance) and deletes a partial file. Without that, an ffmpeg that exits 0
after writing a truncated / moov-less MP4 — or one killed at the wall-clock
timeout mid-write — leaves a half-written file on disk that the resume logic
and the uploader both treat as a finished LF.

These tests pin: reject-and-delete on an invalid artifact, delete on timeout,
delete on a non-zero exit, and no audio requirement on a ``--no-audio`` LF.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.ffmpeg_runner import VideoValidation

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


def _run_apply_ost(paths, *, run_side_effect, validation: VideoValidation):
    source, output, font = paths
    with (
        patch(
            "gomoveshift.pipeline.ost_applier._probe_video_info",
            return_value=_stub_probe(),
        ),
        patch("gomoveshift.pipeline.ost_applier._has_audio_stream", return_value=True),
        patch(
            "gomoveshift.pipeline.ost_applier._measure_audio_loudnorm",
            return_value={},
        ),
        patch_ost_run(side_effect=run_side_effect),
        patch(
            "gomoveshift.pipeline.ost_applier.validate_video",
            return_value=validation,
        ) as mock_validate,
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
    return result, mock_validate


def _encode_writes(size: int = 4096):
    """Write to wherever the real cmd targets (cmd[-1]) — apply_ost hands
    ffmpeg a temp path, never the final path, so the fake must follow the
    same argument rather than a hardcoded final path."""

    def _side_effect(cmd, *_a, **_kw):
        # "-" is ffmpeg's stdout, not a path: never create a file named "-".
        if cmd[-1] != "-":
            target = Path(cmd[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00" * size)
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    return _side_effect


class TestBurnedArtifactIsValidated:
    def test_valid_artifact_is_returned(self, paths):
        _, output, _ = paths
        result, mock_validate = _run_apply_ost(
            paths,
            run_side_effect=_encode_writes(),
            validation=VideoValidation(valid=True, duration=600.0, file_size=4096),
        )
        assert result == output
        assert output.exists()
        mock_validate.assert_called_once()

    def test_corrupt_artifact_is_rejected_and_deleted(self, paths):
        """rc==0 but no moov atom — the exact truncated-encode failure mode."""
        _, output, _ = paths
        result, _ = _run_apply_ost(
            paths,
            run_side_effect=_encode_writes(),
            validation=VideoValidation(valid=False, reason="ffprobe failed: moov atom not found"),
        )
        assert result is None, "a corrupt burn must not be reported as a finished LF"
        assert not output.exists(), "the partial artifact must be deleted, not left for resume"

    def test_validation_uses_the_probed_input_duration(self, paths):
        _, output, _ = paths
        _, mock_validate = _run_apply_ost(
            paths,
            run_side_effect=_encode_writes(),
            validation=VideoValidation(valid=True, duration=600.0, file_size=4096),
        )
        assert mock_validate.call_args.kwargs["expected_duration"] == pytest.approx(600.0)
        assert mock_validate.call_args.kwargs["require_audio"] is True


class TestPartialOutputCleanup:
    def test_timeout_deletes_the_partial_file(self, paths):
        import subprocess as _sp

        _, output, _ = paths

        def _timeout(cmd, *_a, **_kw):
            target = Path(cmd[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00" * 128)  # half-written when the kill lands
            raise _sp.TimeoutExpired(cmd="ffmpeg", timeout=7200)

        result, _ = _run_apply_ost(
            paths,
            run_side_effect=_timeout,
            validation=VideoValidation(valid=True),
        )
        assert result is None
        assert not output.exists(), "a timed-out encode must not leave a partial artifact"

    def test_nonzero_exit_deletes_the_partial_file(self, paths):
        _, output, _ = paths

        def _fail(cmd, *_a, **_kw):
            target = Path(cmd[-1])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\x00" * 128)
            return MagicMock(returncode=1, stdout=b"", stderr=b"Conversion failed!")

        result, _ = _run_apply_ost(
            paths,
            run_side_effect=_fail,
            validation=VideoValidation(valid=True),
        )
        assert result is None
        assert not output.exists()


class TestSilentLongform:
    def test_no_audio_input_does_not_require_an_audio_stream(self, paths):
        """A --no-audio LF is burned with -an; requiring audio would fail it."""
        source, output, font = paths
        with (
            patch(
                "gomoveshift.pipeline.ost_applier._probe_video_info",
                return_value=_stub_probe(),
            ),
            patch("gomoveshift.pipeline.ost_applier._has_audio_stream", return_value=False),
            patch_ost_run(side_effect=_encode_writes()),
            patch(
                "gomoveshift.pipeline.ost_applier.validate_video",
                return_value=VideoValidation(valid=True, duration=600.0, file_size=4096),
            ) as mock_validate,
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
        assert result == output
        assert mock_validate.call_args.kwargs["require_audio"] is False
