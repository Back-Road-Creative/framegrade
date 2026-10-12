"""The missing-font stream-copy fallback must validate its output too.

The post-encode ``validate_video`` gate added to ``apply_ost`` covered the drawtext
re-encode ONLY. The missing-font fallback returned ``output_path`` the moment ffmpeg exited 0, before the input was ever
probed — so there was no expected duration to check the copy against and the
artifact shipped unchecked into an irreversible upload. A stream copy that
exits 0 having written a truncated / moov-less MP4 (output mount lost
mid-write, process killed at the timeout) is the exact failure mode.

These tests pin the structural fix: every artifact ``apply_ost`` produces
leaves through one validated exit, so the fallback cannot return an unchecked
path.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.exceptions import ProbeFailedError
from gomoveshift.pipeline.ffmpeg_runner import VideoValidation


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
    """Source video, output path, and a font path that does NOT exist."""
    source = tmp_path / "lf_01_with_montage.mp4"
    source.write_bytes(b"\x00" * 4096)
    font = tmp_path / "no-such-font.ttf"
    assert not font.exists()
    return source, tmp_path / "lf_01.mp4", font


def _copy_writes(output: Path, size: int = 4096):
    def _side_effect(*_a, **_kw):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"\x00" * size)
        return True

    return _side_effect


def _run_fallback(paths, *, validation, copy_ok=True, probe_fails=False):
    source, output, font = paths
    probe_kw = (
        {"side_effect": ProbeFailedError(source, "moov atom not found")}
        if probe_fails
        else {"return_value": _stub_probe()}
    )
    copy_impl = _copy_writes(output) if copy_ok else (lambda *_a, **_kw: False)
    with (
        patch("gomoveshift.pipeline.ost_applier._probe_video_info", **probe_kw),
        patch("gomoveshift.pipeline.ost_applier._has_audio_stream", return_value=True),
        patch("gomoveshift.pipeline.ost_applier._stream_copy", side_effect=copy_impl) as mock_copy,
        patch(
            "gomoveshift.pipeline.ost_applier.validate_video", return_value=validation
        ) as mock_validate,
    ):
        result = ost_applier.apply_ost(
            input_path=source,
            output_path=output,
            location="Healy Pass, County Cork",
            layout="regular_lf",
            font_path=str(font),
        )
    return result, mock_copy, mock_validate


class TestMissingFontFallbackIsValidated:
    def test_truncated_copy_is_rejected_and_deleted(self, paths):
        _, output, _ = paths
        result, _, _ = _run_fallback(
            paths,
            validation=VideoValidation(valid=False, reason="ffprobe failed: moov atom not found"),
        )
        assert result is None, "a truncated stream copy must not be reported as a finished LF"
        assert not output.exists(), "the bad copy must be deleted, not left for resume/upload"

    def test_validation_uses_the_probed_input_duration(self, paths):
        _, output, _ = paths
        result, _, mock_validate = _run_fallback(
            paths,
            validation=VideoValidation(valid=True, duration=600.0, file_size=4096),
        )
        assert result == output
        mock_validate.assert_called_once()
        assert mock_validate.call_args.kwargs["expected_duration"] == pytest.approx(600.0)
        assert mock_validate.call_args.kwargs["require_audio"] is True

    def test_unprobeable_input_never_reaches_the_stream_copy(self, paths):
        result, mock_copy, mock_validate = _run_fallback(
            paths,
            validation=VideoValidation(valid=True),
            probe_fails=True,
        )
        assert result is None, "an unreadable input must abort the fallback, not copy it blind"
        mock_copy.assert_not_called()
        mock_validate.assert_not_called()

    def test_failed_copy_still_returns_none_without_validating(self, paths):
        result, _, mock_validate = _run_fallback(
            paths,
            validation=VideoValidation(valid=True),
            copy_ok=False,
        )
        assert result is None
        mock_validate.assert_not_called()
