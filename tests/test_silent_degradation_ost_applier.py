"""Silent-degradation regression tests for ost_applier.

Each test forces a fallback path in ``ost_applier`` and asserts that
(a) the fallback behavior ran and (b) a WARNING was emitted naming both
the preferred path and the fallback, so an operator watching the
console can see the degradation.

These guard against the "silent fallback" class of bug surfaced in the
silent-degradation review.

Findings covered:
  * ffprobe subprocess exception -> hardcoded 4K/29.97 defaults
  * ffprobe JSON parse exception -> hardcoded defaults (partial probe)
  * Missing font file -> stream-copy (no OST overlay)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.ffmpeg_runner import VideoValidation


OST_LOGGER = "gomoveshift.pipeline.ost_applier"

_PROBE_JSON = json.dumps(
    {
        "streams": [{"width": 3840, "height": 2160, "r_frame_rate": "30000/1001"}],
        "format": {"duration": "600.0", "bit_rate": "45000000"},
    }
).encode("utf-8")


class TestProbeFfprobeSubprocessFailureRaises:
    """_probe_video_info must RAISE when ffprobe itself raises."""

    def test_subprocess_exception_raises_probe_failed_error(self, tmp_path, monkeypatch, caplog):
        from gomoveshift.pipeline import ost_applier
        from gomoveshift.pipeline.exceptions import ProbeFailedError

        def boom(*_a, **_kw):
            raise OSError("ffprobe binary missing")

        monkeypatch.setattr(ost_applier.subprocess, "run", boom)

        video = tmp_path / "clip.mp4"
        video.touch()

        with caplog.at_level("ERROR", logger=OST_LOGGER):
            with pytest.raises(ProbeFailedError) as exc_info:
                ost_applier._probe_video_info(video)

        # Exception carries the offending video path so the operator can
        # locate it without grepping the log.
        assert exc_info.value.video_path == video

        # ERROR was emitted at the raise site (legacy hardcoded-defaults
        # fallback removed; the message must self-signal the change).
        messages = [r.getMessage() for r in caplog.records if r.name == OST_LOGGER]
        assert any("ffprobe failed" in m and "ProbeFailedError" in m for m in messages), (
            f"expected self-signalling ffprobe-failure error, got: {messages!r}"
        )


class TestProbeJsonParseFailureRaises:
    """_probe_video_info must RAISE when ffprobe stdout is unparseable."""

    def test_bad_json_raises_probe_failed_error(self, tmp_path, monkeypatch, caplog):
        from gomoveshift.pipeline import ost_applier
        from gomoveshift.pipeline.exceptions import ProbeFailedError

        def fake_run(*_a, **_kw):
            r = MagicMock()
            r.returncode = 0
            r.stdout = b"not-valid-json{{{"
            r.stderr = b""
            return r

        monkeypatch.setattr(ost_applier.subprocess, "run", fake_run)

        video = tmp_path / "clip.mp4"
        video.touch()

        with caplog.at_level("ERROR", logger=OST_LOGGER):
            with pytest.raises(ProbeFailedError):
                ost_applier._probe_video_info(video)

        messages = [r.getMessage() for r in caplog.records if r.name == OST_LOGGER]
        assert any("JSON parse failed" in m and "ProbeFailedError" in m for m in messages), (
            f"expected JSON-parse-failure error, got: {messages!r}"
        )

    def test_bad_fps_fraction_raises(self, tmp_path, monkeypatch, caplog):
        """ZeroDivisionError inside the parse block must also raise."""
        import json

        from gomoveshift.pipeline import ost_applier
        from gomoveshift.pipeline.exceptions import ProbeFailedError

        # r_frame_rate="30/0" → ZeroDivisionError inside the parse try/except.
        payload = json.dumps(
            {
                "streams": [{"width": 1920, "height": 1080, "r_frame_rate": "30/0"}],
                "format": {"duration": "120.0", "bit_rate": "5000000"},
            }
        ).encode("utf-8")

        def fake_run(*_a, **_kw):
            r = MagicMock()
            r.returncode = 0
            r.stdout = payload
            r.stderr = b""
            return r

        monkeypatch.setattr(ost_applier.subprocess, "run", fake_run)

        video = tmp_path / "clip.mp4"
        video.touch()

        with caplog.at_level("ERROR", logger=OST_LOGGER):
            with pytest.raises(ProbeFailedError):
                ost_applier._probe_video_info(video)

        messages = [r.getMessage() for r in caplog.records if r.name == OST_LOGGER]
        assert any("JSON parse failed" in m for m in messages), messages


class TestMissingFontWarns:
    """apply_ost must WARN and stream-copy when the font file is absent."""

    def test_missing_font_logs_warning_and_stream_copies(self, tmp_path, monkeypatch, caplog):
        from gomoveshift.pipeline import ost_applier

        input_path = tmp_path / "in.mp4"
        input_path.touch()
        output_path = tmp_path / "out.mp4"

        captured_cmds: list[list[str]] = []

        # The fallback probes the input for an expected duration and then
        # ffprobe-validates the copy it produced (both apply_ost paths share one
        # validated exit), so the mock has to answer a probe. The touched empty
        # "output" no real validator would accept is stubbed valid here —
        # validation itself is covered by
        # tests/test_ost_applier_font_fallback_validation.py.
        def fake_run(cmd, *_a, **_kw):
            captured_cmds.append(list(cmd))
            r = MagicMock()
            r.returncode = 0
            r.stdout = _PROBE_JSON if "ffprobe" in str(cmd[0]) else b""
            r.stderr = b""
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.touch()
            return r

        monkeypatch.setattr(ost_applier.subprocess, "run", fake_run)
        monkeypatch.setattr(
            ost_applier, "validate_video", lambda *_a, **_kw: VideoValidation(valid=True)
        )

        missing_font = tmp_path / "does-not-exist.ttf"
        assert not missing_font.exists()

        with caplog.at_level("WARNING", logger=OST_LOGGER):
            result = ost_applier.apply_ost(
                input_path=input_path,
                output_path=output_path,
                location="Frogmore to Port Royal",
                layout="regular_lf",
                font_path=str(missing_font),
            )

        # Fallback ran: stream-copy branch invoked (ffmpeg called with `-c copy`,
        # no `-vf` drawtext filter).
        assert result == output_path
        assert captured_cmds, "expected ffmpeg stream-copy to run"
        copy_cmd = next((c for c in captured_cmds if "ffmpeg" in str(c[0])), None)
        assert copy_cmd is not None, "no ffmpeg command captured"
        assert "-c" in copy_cmd and "copy" in copy_cmd
        assert "-vf" not in copy_cmd, "missing-font fallback must stream-copy, not attempt drawtext"

        messages = [r.getMessage() for r in caplog.records if r.name == OST_LOGGER]
        assert any("font not found" in m and "stream-copy" in m for m in messages), (
            f"expected self-signalling missing-font warning, got: {messages!r}"
        )
        # Both the preferred input (the font path) and the fallback name ("stream-copy")
        # must appear in the warning so operators can act on it.
        combined = "\n".join(messages)
        assert str(missing_font) in combined
