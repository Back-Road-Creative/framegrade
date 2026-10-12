"""The film's final encode pulls audio DOWN to the near-silent ceiling and never up.

A bare ``loudnorm`` once boosted a quiet phone clip (-51.4 LUFS going in,
-46.0 LUFS coming out) -- a boost, which the service's audio rule forbids for
any published video.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline.route_story import encode as encode_module
from gomoveshift.pipeline.route_story.encode import encode_final


def _source(path: Path, volume_db: float, duration: float = 3.0) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"color=c=gray:s=64x64:d={duration}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
            "-af", f"volume={volume_db}dB",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2",
            "-shortest", str(path),
        ],
        check=True,
        capture_output=True,
    )  # fmt: skip


def _integrated_lufs(path: Path) -> float:
    stderr = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-vn", "-af", "ebur128",
         "-f", "null", "-"],
        capture_output=True,
        text=True,
        check=True,
    ).stderr  # fmt: skip
    return float(re.findall(r"I:\s+(-?[0-9.]+) LUFS", stderr)[-1])


def _encode_command(monkeypatch, tmp_path: Path, measure_stderr: str, returncode: int = 0):
    commands = []

    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode, stdout="", stderr=measure_stderr)

    monkeypatch.setattr(encode_module.subprocess, "run", fake_run)
    encode_final(tmp_path / "in.mp4", tmp_path / "out.mp4")
    return commands[-1]


_STATS = '{{\n\t"input_i" : "{i}",\n\t"input_tp" : "{tp}",\n\t"input_lra" : "1.0"\n}}'


def test_the_encode_never_runs_a_normalising_filter(monkeypatch, tmp_path: Path):
    cmd = _encode_command(monkeypatch, tmp_path, _STATS.format(i="-51.4", tp="-25.1"))

    assert "loudnorm" not in cmd[cmd.index("-af") + 1]


def test_audio_already_under_the_ceiling_is_left_alone(monkeypatch, tmp_path: Path):
    cmd = _encode_command(monkeypatch, tmp_path, _STATS.format(i="-51.4", tp="-25.1"))

    assert cmd[cmd.index("-af") + 1] == "volume=0.00dB"


def test_loud_audio_is_pulled_down_to_the_ceiling(monkeypatch, tmp_path: Path):
    cmd = _encode_command(monkeypatch, tmp_path, _STATS.format(i="-20.0", tp="-3.0"))

    gain = float(re.fullmatch(r"volume=(-?[0-9.]+)dB", cmd[cmd.index("-af") + 1]).group(1))
    assert gain == ec.AUDIO_SILENCE_CEILING_LUFS - -20.0


def test_a_failed_measurement_attenuates_by_the_fixed_fallback(monkeypatch, tmp_path: Path):
    cmd = _encode_command(monkeypatch, tmp_path, "no stats here")

    gain = float(re.fullmatch(r"volume=(-?[0-9.]+)dB", cmd[cmd.index("-af") + 1]).group(1))
    assert gain < 0


def test_a_quiet_clip_comes_out_no_louder_than_it_went_in(tmp_path: Path):
    source = tmp_path / "quiet.mp4"
    _source(source, volume_db=-40.0)
    output = tmp_path / "out.mp4"

    encode_final(source, output)

    assert _integrated_lufs(output) <= _integrated_lufs(source) + 0.5


def test_a_loud_clip_comes_out_at_or_under_the_ceiling(tmp_path: Path):
    source = tmp_path / "loud.mp4"
    _source(source, volume_db=-6.0)
    output = tmp_path / "out.mp4"

    encode_final(source, output)

    assert _integrated_lufs(output) <= ec.AUDIO_SILENCE_CEILING_LUFS + 0.5
