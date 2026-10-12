"""Shorts render MUTED — near-silent policy.

Replaces the deleted two-pass loudnorm shorts path (``_measure_segment_loudnorm``
/ ``_shorts_loudnorm_filter``). Shorts now ALWAYS ship with no audio stream:
vertical platforms (TikTok/IG) may re-normalize quiet audio back UP on their
side, and a stripped stream cannot be undone — so ``-an`` is the robust choice
(see encoding_constants.py). ``render_short_ffmpeg`` still accepts
``skip_audio`` / ``skip_audio_norm`` / ``target_lufs`` for back-compat, but they
no longer gate audio: the encode is always muted.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _stub_render(tmp_path, monkeypatch):
    """Wire mocks so render_short_ffmpeg builds a command without touching
    real ffmpeg/ffprobe. Returns (source, out, captured_cmds)."""
    from gomoveshift.pipeline import shorts_extractor

    source = tmp_path / "master.mp4"
    source.touch()
    out = tmp_path / "short_01.mp4"
    captured: list[list[str]] = []

    def fake_run(cmd, *args, **kwargs):
        captured.append(list(cmd))
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"x" * 4096)
        return result

    monkeypatch.setattr(shorts_extractor.subprocess, "run", fake_run)
    # The post-encode validate_video ffprobes the output; stub it valid so the
    # test exercises command construction, not the real probe of a fake file.
    monkeypatch.setattr(
        "gomoveshift.pipeline.ffmpeg_runner.validate_video",
        lambda *a, **k: MagicMock(valid=True, reason="", duration=30.0),
    )
    return shorts_extractor, source, out, captured


def _find_encode_cmd(captured: list[list[str]]) -> list[str]:
    cmd = next((c for c in captured if "hevc_nvenc" in c or "-c:v" in c), None)
    assert cmd is not None, f"no encode command captured: {captured!r}"
    return cmd


def test_render_short_cmd_is_muted(tmp_path, monkeypatch):
    shorts_extractor, source, out, captured = _stub_render(tmp_path, monkeypatch)

    ok = shorts_extractor.render_short_ffmpeg(
        source_video=source,
        output_path=out,
        start_time=42.0,
        duration=30.0,
        location="Beaufort, SC",
        src_width=3840,
        src_height=2160,
        use_nvenc=True,
    )
    assert ok is True

    cmd = _find_encode_cmd(captured)
    assert "-an" in cmd, f"encode command is not muted: {cmd!r}"
    assert "-af" not in cmd, f"muted shorts must carry no audio filter: {cmd!r}"
    assert "-c:a" not in cmd, f"muted shorts must not encode an audio codec: {cmd!r}"
    joined = " ".join(cmd)
    assert "loudnorm" not in joined, joined
    assert "aac" not in joined, joined


def test_only_one_ffmpeg_pass_no_measurement(tmp_path, monkeypatch):
    """No windowed audio-measurement pass runs anymore — a single encode call."""
    shorts_extractor, source, out, captured = _stub_render(tmp_path, monkeypatch)

    shorts_extractor.render_short_ffmpeg(
        source_video=source,
        output_path=out,
        start_time=42.0,
        duration=30.0,
        location="Beaufort, SC",
        src_width=3840,
        src_height=2160,
        use_nvenc=True,
    )
    # Exactly one subprocess.run (the encode) — no ``-vn ... null`` measure pass.
    assert len(captured) == 1, captured
    assert not any("-vn" in c and "null" in c for c in captured)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"skip_audio_norm": True},
        {"skip_audio_norm": False},
        {"skip_audio": True},
        {"target_lufs": -20.0},
    ],
)
def test_back_compat_flags_never_reintroduce_audio(tmp_path, monkeypatch, kwargs):
    """The retained back-compat params must not re-enable an audio stream —
    the encode stays ``-an`` regardless."""
    shorts_extractor, source, out, captured = _stub_render(tmp_path, monkeypatch)

    shorts_extractor.render_short_ffmpeg(
        source_video=source,
        output_path=out,
        start_time=0.0,
        duration=30.0,
        location="Beaufort, SC",
        src_width=3840,
        src_height=2160,
        use_nvenc=True,
        **kwargs,
    )
    cmd = _find_encode_cmd(captured)
    assert "-an" in cmd
    assert "-af" not in cmd
    assert "-c:a" not in cmd


def test_deleted_symbols_are_gone():
    """The two-pass shorts loudnorm helpers were removed with the audio path."""
    from gomoveshift.pipeline import shorts_extractor

    assert not hasattr(shorts_extractor, "_shorts_loudnorm_filter")
    assert not hasattr(shorts_extractor, "_measure_segment_loudnorm")
