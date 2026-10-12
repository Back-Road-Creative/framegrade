"""The assembly re-encode obeys the frozen constants, and lossless cuts stay lossless.

``VideoEncoder`` backs `gomoveshift analyze` / `process` and the web
`/api/approve` render. It used to emit only ``-c:v libx265 -crf 18 -preset
slow`` — no pix_fmt, no profile, no color tags, no bitrate floor — while
``config.output.profile`` / ``pix_fmt`` (both marked "Frozen — must match
LONGFORM_*") were never read by anything. It also re-encoded every trim, even
when the cut landed exactly on a keyframe and a stream copy would have been
bit-exact.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.assembly.assembler import VideoAssembler
from gomoveshift.assembly.encoder import VideoEncoder
from gomoveshift.config import Config
from gomoveshift.pipeline import encoding_constants as ec


def _run_extract(enc: VideoEncoder, *, height: int = 2160, reencode: bool = True) -> list[str]:
    """Capture the ffmpeg argv ``extract_segment`` would run."""
    seen: list[list[str]] = []

    def fake_run(cmd, **kw):
        seen.append(list(cmd))
        return MagicMock(returncode=0)

    with patch.object(enc, "get_video_info", return_value={"streams": [{"height": height}]}):
        with patch("gomoveshift.assembly.encoder.subprocess.run", side_effect=fake_run):
            enc.extract_segment(Path("/src.mp4"), Path("/out.mp4"), 0.0, 10.0, reencode=reencode)
    return seen[0]


class TestFrozenConstants:
    def test_reencode_stamps_pix_fmt_and_profile(self):
        cmd = _run_extract(VideoEncoder())
        assert "-pix_fmt" in cmd and cmd[cmd.index("-pix_fmt") + 1] == ec.LONGFORM_PIX_FMT
        assert "-profile:v" in cmd and cmd[cmd.index("-profile:v") + 1] == ec.LONGFORM_PROFILE

    def test_reencode_stamps_all_four_color_tags(self):
        cmd = _run_extract(VideoEncoder())
        for flag, want in (
            ("-color_range", ec.LONGFORM_COLOR_RANGE),
            ("-colorspace", ec.LONGFORM_COLORSPACE),
            ("-color_primaries", ec.LONGFORM_COLOR_PRIMARIES),
            ("-color_trc", ec.LONGFORM_COLOR_TRC),
        ):
            assert flag in cmd, f"{flag} missing from encode args"
            assert cmd[cmd.index(flag) + 1] == want

    @pytest.mark.parametrize(
        "height,floor",
        [(2160, ec.LONGFORM_MIN_BITRATE_4K), (1080, ec.LONGFORM_MIN_BITRATE_1080P)],
    )
    def test_reencode_carries_the_resolution_bitrate_floor(self, height, floor):
        cmd = _run_extract(VideoEncoder(), height=height)
        assert cmd[cmd.index("-b:v") + 1] == f"{floor}k"
        assert cmd[cmd.index("-maxrate") + 1] == f"{int(floor * ec.LONGFORM_MAXRATE_RATIO)}k"
        assert cmd[cmd.index("-bufsize") + 1] == f"{int(floor * ec.LONGFORM_BUFSIZE_RATIO)}k"

    def test_stream_copy_branch_stamps_nothing(self):
        """A copy is bit-exact; re-stamping quality args on it would be a lie."""
        cmd = _run_extract(VideoEncoder(), reencode=False)
        assert "-c" in cmd and cmd[cmd.index("-c") + 1] == "copy"
        for flag in ("-pix_fmt", "-profile:v", "-b:v", "-crf"):
            assert flag not in cmd

    def test_assembler_threads_the_frozen_config_fields(self):
        cfg = Config()
        enc = VideoAssembler(cfg).encoder
        assert enc.pix_fmt == cfg.output.pix_fmt == ec.LONGFORM_PIX_FMT
        assert enc.profile == cfg.output.profile == ec.LONGFORM_PROFILE


class TestKeyframeAlignedCopy:
    def _capture_modes(self, enc: VideoEncoder, keeps) -> list[bool]:
        modes: list[bool] = []

        def fake_extract(source, output, start, end, reencode=False):
            modes.append(reencode)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"x")
            return True

        with patch.object(enc, "extract_segment", side_effect=fake_extract):
            with patch.object(enc, "concatenate", return_value=True):
                enc.trim_and_encode(Path("/src.mp4"), Path("/tmp/out.mp4"), keeps)
        return modes

    def test_stream_copies_when_every_start_lands_on_a_keyframe(self, monkeypatch):
        import keycut

        monkeypatch.setattr(keycut, "probe_keyframes", lambda *a, **k: [0.0, 2.0, 4.0, 6.0])
        modes = self._capture_modes(VideoEncoder(), [(0.0, 2.0), (4.0, 6.0)])
        assert modes == [False, False]

    def test_reencodes_every_segment_when_one_start_misses(self, monkeypatch):
        """All-or-nothing: a mixed concat of copied and re-encoded parts breaks."""
        import keycut

        monkeypatch.setattr(keycut, "probe_keyframes", lambda *a, **k: [0.0, 2.0, 4.0, 6.0])
        modes = self._capture_modes(VideoEncoder(), [(0.0, 2.0), (4.7, 6.0)])
        assert modes == [True, True]

    def test_reencodes_when_the_keyframe_probe_is_unavailable(self, monkeypatch):
        import keycut

        monkeypatch.setattr(keycut, "probe_keyframes", lambda *a, **k: None)
        modes = self._capture_modes(VideoEncoder(), [(0.0, 2.0), (4.0, 6.0)])
        assert modes == [True, True]

    def test_single_segment_copies_on_a_keyframe(self, monkeypatch, tmp_path):
        import keycut

        monkeypatch.setattr(keycut, "probe_keyframes", lambda *a, **k: [0.0, 2.0])
        modes = self._capture_modes(VideoEncoder(), [(0.0, 2.0)])
        assert modes == [False]

    def test_single_segment_reencodes_off_a_keyframe(self, monkeypatch):
        import keycut

        monkeypatch.setattr(keycut, "probe_keyframes", lambda *a, **k: [0.0, 2.0])
        modes = self._capture_modes(VideoEncoder(), [(1.3, 2.0)])
        assert modes == [True]
