"""ffmpeg must be TOLD the container when it writes the ``.ost-partial`` temp.

The atomic-output design moves every OST write off the final path onto a
sibling ending ``.ost-partial``, so a killed encode cannot strand a
finished-looking ``lf_01.mp4``. That suffix deliberately trails the real
extension — and ffmpeg picks its muxer from the output EXTENSION. With nothing
recognizable to infer from it aborts before writing a byte:

    Unable to find a suitable output format for
    '…/lf_01.mp4.ost-partial'; use a standard extension for the filename or
    specify the format manually.  (rc 234)

That once killed a multi-day run after a
16-hour master encode, on the first LF burn it reached. Every existing OST test
stubs the subprocess, so the muxer was never exercised: the burn "succeeded"
against a fake that happily wrote to any path. These tests cover both halves —
the flag is present in the command, and a real ffmpeg actually produces a
playable file.
"""

import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import ost_applier
from gomoveshift.pipeline.ffmpeg_runner import VideoValidation

from ost_subprocess_harness import patch_ost_run

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not on PATH")


def _stub_probe(duration: float = 4.0) -> dict:
    return {
        "width": 640,
        "height": 360,
        "fps": 30.0,
        "bitrate_kbps": 2_000,
        "duration": duration,
    }


class TestTempOutputCarriesAnExplicitMuxer:
    def test_burn_command_names_the_container(self, tmp_path):
        """``-f mp4`` rides with the command, and the temp path stays last so
        the runner's own cleanup still targets the right file."""
        source = tmp_path / "lf_01_with_montage.mp4"
        source.write_bytes(b"\x00" * 4096)
        font = tmp_path / "font.ttf"
        font.write_bytes(b"fake font")
        output = tmp_path / "lf_01.mp4"
        seen: dict[str, list] = {}

        def _capture(cmd, *_a, **_kw):
            seen["cmd"] = list(cmd)
            target = cmd[-1]
            # "-" is ffmpeg's stdout, not a path: writing to it litters a
            # file literally named "-" in the CWD. That file is untracked and
            # not gitignored, so `git add -A` in the service dir sweeps 4 KiB
            # of binary garbage into a commit.
            if target != "-":
                Path(target).parent.mkdir(parents=True, exist_ok=True)
                Path(target).write_bytes(b"\xff" * 4096)
            return MagicMock(returncode=0, stdout=b"", stderr=b"")

        with (
            patch.object(ost_applier, "_probe_video_info", return_value=_stub_probe()),
            patch.object(ost_applier, "_has_audio_stream", return_value=True),
            patch.object(ost_applier, "_measure_audio_loudnorm", return_value={}),
            patch_ost_run(side_effect=_capture),
            patch.object(
                ost_applier, "validate_video", return_value=VideoValidation(True, 4.0, "")
            ),
        ):
            ost_applier.apply_ost(
                input_path=source,
                output_path=output,
                location="Kingsport, TN",
                layout="regular_lf",
                font_path=str(font),
                prefer_nvenc=False,
            )

        cmd = seen["cmd"]
        assert cmd[-1].endswith(ost_applier._TEMP_OUTPUT_SUFFIX), (
            "the temp path must stay the last argument — run_ffmpeg_encode and "
            f"the atomic tests both read cmd[-1] as the encode target: {cmd[-1]}"
        )
        assert "-f" in cmd, (
            "ffmpeg cannot infer a muxer from '.ost-partial' and aborts with rc 234; "
            f"the container must be named explicitly. cmd={cmd}"
        )
        assert cmd[cmd.index("-f") + 1] == "mp4"

    @needs_ffmpeg
    def test_a_real_ffmpeg_burn_lands_a_playable_file(self, tmp_path):
        """End-to-end against the real muxer — the check every stubbed test
        structurally cannot make. No fixture is committed: the clip is drawn by
        ffmpeg itself."""
        source = tmp_path / "lf_01_with_montage.mp4"
        subprocess.run(
            [
                FFMPEG,
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=320x180:rate=15:duration=2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(source),
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
        output = tmp_path / "lf_01.mp4"

        # No font on disk → the stream-copy fallback, which writes to the SAME
        # temp path and so needs the same explicit container.
        result = ost_applier.apply_ost(
            source, output, "Kingsport, TN", "regular_lf", font_path=str(tmp_path / "absent.ttf")
        )

        assert result is not None, "the burn returned failure against real ffmpeg"
        assert output.exists() and output.stat().st_size > 0
        probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "csv=p=0",
                str(output),
            ],
            capture_output=True,
            timeout=60,
        )
        assert probe.returncode == 0 and b"video" in probe.stdout, (
            f"output is not a readable video: {probe.stderr!r}"
        )
        assert not list(tmp_path.glob(f"*{ost_applier._TEMP_OUTPUT_SUFFIX}")), (
            "the temp artifact must be promoted or removed, never left behind"
        )
