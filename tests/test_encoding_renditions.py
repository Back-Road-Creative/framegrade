"""Rendition matrix: RENDITIONS table + encode_rendition() command shape.

No renders here -- ffmpeg command shape assertions only (subprocess is
monkeypatched); the ceiling-refusal test proves the refusal happens before
any ffmpeg call.
"""

from pathlib import Path

import pytest

from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline.encoding_constants import RENDITIONS, Rendition
from gomoveshift.pipeline.route_story import encode as encode_module
from gomoveshift.pipeline.route_story.encode import RenditionCeilingError, encode_rendition


class TestRenditionTable:
    def test_youtube_short_matches_legacy_shorts_constants(self):
        short = RENDITIONS["youtube_short"]
        assert short.width == ec.SHORTS_WIDTH == 1080
        assert short.height == ec.SHORTS_HEIGHT == 1920
        assert short.codec == "hevc"
        assert short.faststart is True
        assert short.audio is False

    def test_youtube_longform_keeps_audio_and_has_no_fixed_resolution(self):
        longform = RENDITIONS["youtube_longform"]
        # Long-form resolution follows the source (both a 4K and a 1080p
        # bitrate floor exist) -- never a guessed fixed w/h.
        assert longform.width is None
        assert longform.height is None
        assert longform.codec == "hevc"
        assert longform.faststart is True
        assert longform.audio is True

    def test_no_platform_ceiling_is_invented(self):
        # No max_seconds/max_bytes/fps constant exists anywhere in this
        # file for either platform -- these must stay None, not guessed.
        for platform in ("youtube_longform", "youtube_short"):
            rendition = RENDITIONS[platform]
            assert rendition.max_seconds is None
            assert rendition.max_bytes is None
            assert rendition.fps is None

    def test_route_story_short_keeps_audio_and_matches_the_social_preset_ceiling(self):
        # The route-story social cut keeps its
        # attenuated audio -- it is not cabin audio, so it does NOT take the
        # drive-shorts -an rule. Its max_seconds mirrors the social preset's
        # own 90s ceiling rather than a second, possibly-disagreeing number.
        from gomoveshift.pipeline.route_story.presets import SOCIAL

        rendition = RENDITIONS["route_story_short"]
        assert (rendition.width, rendition.height) == (1080, 1920)
        assert rendition.faststart is True
        assert rendition.audio is True
        assert rendition.max_seconds == SOCIAL.max_seconds == 90.0
        # The drive-shorts entry is untouched by this rendition's addition.
        assert RENDITIONS["youtube_short"].audio is False


class TestEncodeRenditionCommandShape:
    def _capture(self, monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            import subprocess as _subprocess

            return _subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(encode_module.subprocess, "run", fake_run)
        return captured

    def test_youtube_short_scales_and_strips_audio(self, monkeypatch, tmp_path: Path):
        captured = self._capture(monkeypatch)

        encode_rendition(tmp_path / "master.mp4", "youtube_short", tmp_path / "out.mp4")

        cmd = captured["cmd"]
        assert "-vf" in cmd
        vf = cmd[cmd.index("-vf") + 1]
        assert "scale=1080:1920" in vf
        assert "-an" in cmd
        assert "-movflags" in cmd
        assert cmd[cmd.index("-movflags") + 1] == "+faststart"

    def test_youtube_longform_keeps_audio_and_has_no_scale_filter(
        self, monkeypatch, tmp_path: Path
    ):
        captured = self._capture(monkeypatch)

        encode_rendition(tmp_path / "master.mp4", "youtube_longform", tmp_path / "out.mp4")

        cmd = captured["cmd"]
        assert "-an" not in cmd
        assert "-vf" not in cmd

    def test_route_story_short_scales_but_never_strips_audio(self, monkeypatch, tmp_path: Path):
        captured = self._capture(monkeypatch)
        # route_story_short has a max_seconds ceiling (unlike youtube_short),
        # so encode_rendition probes duration before building the ffmpeg cmd.
        monkeypatch.setattr(encode_module, "ffprobe_duration_seconds", lambda path: 45.0)

        encode_rendition(tmp_path / "master.mp4", "route_story_short", tmp_path / "out.mp4")

        cmd = captured["cmd"]
        assert "-vf" in cmd
        vf = cmd[cmd.index("-vf") + 1]
        assert "scale=1080:1920" in vf
        # Unlike the drive-shorts rendition, this one is not -an.
        assert "-an" not in cmd


class TestEncodeRenditionCeilingRefusal:
    def test_refuses_before_any_ffmpeg_call_when_master_exceeds_a_set_ceiling(
        self, monkeypatch, tmp_path: Path
    ):
        called = {"count": 0}

        def fake_run(cmd, **kwargs):
            called["count"] += 1
            raise AssertionError("ffmpeg must not be invoked past the ceiling refusal")

        monkeypatch.setattr(encode_module.subprocess, "run", fake_run)
        monkeypatch.setitem(
            encode_module.RENDITIONS,
            "test_platform",
            Rendition(
                width=None,
                height=None,
                max_seconds=5.0,
                max_bytes=None,
                fps=None,
                codec="hevc",
                faststart=False,
                audio=True,
            ),
        )
        monkeypatch.setattr(encode_module, "ffprobe_duration_seconds", lambda path: 10.0)

        with pytest.raises(RenditionCeilingError):
            encode_rendition(tmp_path / "master.mp4", "test_platform", tmp_path / "out.mp4")

        assert called["count"] == 0

    def test_a_ceiling_left_none_is_never_enforced(self, monkeypatch, tmp_path: Path):
        captured = self._capture_helper(monkeypatch)
        monkeypatch.setattr(encode_module, "ffprobe_duration_seconds", lambda path: 999999.0)

        # youtube_short.max_seconds is None -- no ceiling to violate, no probe call needed.
        encode_rendition(tmp_path / "master.mp4", "youtube_short", tmp_path / "out.mp4")

        assert captured["cmd"]

    @staticmethod
    def _capture_helper(monkeypatch):
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            import subprocess as _subprocess

            return _subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(encode_module.subprocess, "run", fake_run)
        return captured
