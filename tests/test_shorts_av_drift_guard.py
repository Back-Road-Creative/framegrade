"""
Muted shorts never carry the ``-shortest`` audio guard.

Historically ``render_short_ffmpeg`` added ``-shortest`` whenever audio was
encoded, to stop stream-copy A/V length drift from leaking into the final MP4.
Under the near-silent audio policy shorts render MUTED (``-an``) — there is no
audio stream, so there is no A/V drift and ``-shortest`` is meaningless. These
tests guard that ``-shortest`` never appears, regardless of the retained
``skip_audio`` flag.
"""

from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.shorts_extractor import render_short_ffmpeg


@pytest.fixture
def fake_source(tmp_path):
    """Create a fake source video file."""
    src = tmp_path / "source.mp4"
    src.write_bytes(b"\x00" * 1024)
    return src


def _captured_argv(mock_run):
    """Return the argv list passed to the subprocess.run call."""
    assert mock_run.call_count >= 1, "subprocess.run was never invoked"
    return mock_run.call_args.args[0]


def _run_render(fake_source, output, *, skip_audio):
    def _side_effect(*args, **kwargs):
        output.write_bytes(b"\x00" * 200_000)
        return MagicMock(returncode=0, stdout="", stderr="")

    # The post-encode validate_video is stubbed valid so the test exercises
    # command construction (the -shortest question), not the real probe of a
    # muted fake file.
    with (
        patch("gomoveshift.pipeline.shorts_extractor.subprocess.run") as mock_run,
        patch("gomoveshift.pipeline.shorts_extractor._detect_nvenc_cached", return_value=False),
        patch(
            "gomoveshift.pipeline.shorts_extractor.get_video_dimensions",
            return_value=(1920, 1080),
        ),
        patch(
            "gomoveshift.pipeline.ffmpeg_runner.validate_video",
            return_value=MagicMock(valid=True, reason="", duration=35.0),
        ),
    ):
        mock_run.side_effect = _side_effect

        result = render_short_ffmpeg(
            source_video=fake_source,
            output_path=output,
            start_time=10.0,
            duration=35.0,
            location="Test",
            skip_audio=skip_audio,
        )
        return result, _captured_argv(mock_run)


def test_render_short_ffmpeg_omits_shortest_muted_default(fake_source, tmp_path):
    """A default render is muted, so -shortest must not appear."""
    output = tmp_path / "short.mp4"
    result, argv = _run_render(fake_source, output, skip_audio=False)
    assert result is True
    assert "-shortest" not in argv, (
        "muted shorts encode no audio, so the -shortest A/V guard must not appear"
    )
    assert "-an" in argv


def test_render_short_ffmpeg_omits_shortest_when_audio_skipped(fake_source, tmp_path):
    """When skip_audio=True the render is muted too, so -shortest is absent."""
    output = tmp_path / "short.mp4"
    result, argv = _run_render(fake_source, output, skip_audio=True)
    assert result is True
    assert "-shortest" not in argv, "-shortest is meaningless without audio and must not appear"
    assert "-an" in argv
