"""Stop-removal concat workspace must stay off tmpfs /tmp and its concat
list must POSIX-escape apostrophes like the five twin concat sites.

The multi-segment path stream-copies every retained segment (~source size,
multi-GB 4K) into a scratch dir before concatenating. That scratch dir has to
live under the operator-controlled staging base (the same dir the cut output
lands in), never default tmpfs /tmp, or a long run ENOSPCs or gets reboot-wiped
mid-flight. FFmpeg is mocked throughout.
"""

import tempfile as _tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.stop_processor import StopProcessor


@pytest.fixture
def config():
    return Config()


def test_concat_workspace_rooted_at_staging_base_not_tmp(config, tmp_path):
    """The multi-segment scratch dir is created under the output's staging
    dir (output_path.parent), never the default tmpfs /tmp."""
    staging = tmp_path / "_stages"
    staging.mkdir()
    output = staging / "output.mp4"
    segments = [(0, 10), (20, 30), (40, 50)]

    captured = {}
    real_tempdir = _tempfile.TemporaryDirectory

    def spy(*args, **kwargs):
        captured["dir"] = kwargs.get("dir")
        return real_tempdir(*args, **kwargs)

    proc = StopProcessor(config, crossfade_duration=0)
    with (
        patch(
            "gomoveshift.pipeline.stop_processor.tempfile.TemporaryDirectory",
            side_effect=spy,
        ),
        patch.object(proc, "_run_ffmpeg"),
    ):
        proc._cut_and_stitch(staging / "input.mp4", output, segments)

    assert captured.get("dir") is not None, (
        "TemporaryDirectory must receive dir= rooted at the staging base to stay off /tmp"
    )
    assert Path(captured["dir"]) == output.parent


def test_concat_list_escapes_apostrophes(config, tmp_path):
    """An apostrophe in a segment path is POSIX-escaped in the concat list
    (' -> '\\'') exactly like the twin concat sites."""
    staging = tmp_path / "O'Brien" / "_stages"
    staging.mkdir(parents=True)
    output = staging / "output.mp4"
    segments = [(0, 10), (20, 30), (40, 50)]

    captured = {}

    def fake_run(cmd, description, timeout=3600):
        if "concat" in description:
            list_path = Path(cmd[cmd.index("-i") + 1])
            captured["content"] = list_path.read_text()

    proc = StopProcessor(config, crossfade_duration=0)
    with patch.object(proc, "_run_ffmpeg", side_effect=fake_run):
        proc._cut_and_stitch(staging / "input.mp4", output, segments)

    content = captured["content"]
    assert "seg_000.mp4" in content  # we read the real concat list
    assert "'\\''" in content, "apostrophes must be POSIX-escaped like the twin concat sites"
