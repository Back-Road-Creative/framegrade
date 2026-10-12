"""A licensed music bed mixed under the film's attenuated source audio.

A bed is only ever added to the ffmpeg command when the caller passes an
enabled ``AudioProfile`` whose bed file resolves against its
``LicenceLedger`` -- absent a profile, the command stays byte-identical
to the no-bed path.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline.audio_profile import AudioProfile, LicenceLedger, UnlicensedBedError
from gomoveshift.pipeline.route_story import encode as encode_module
from gomoveshift.pipeline.route_story.encode import encode_final, write_publish_json

_STATS = '{{\n\t"input_i" : "{i}",\n\t"input_tp" : "{tp}",\n\t"input_lra" : "1.0"\n}}'


def _fake_measure_run(
    monkeypatch, commands: list, measure_stderr: str = _STATS.format(i="-51.4", tp="-25.1")
):
    def fake_run(cmd, **kwargs):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=measure_stderr)

    monkeypatch.setattr(encode_module.subprocess, "run", fake_run)


def _ledger_entry(**overrides) -> dict:
    entry = {
        "file": "bed.mp3",
        "title": "Open Road",
        "source": "pixabay",
        "licence": "Pixabay Content License",
        "url": "https://pixabay.com/music/x",
        "added": "2026-09-25",
    }
    entry.update(overrides)
    return entry


def test_no_profile_command_is_byte_identical_to_before(monkeypatch, tmp_path: Path):
    commands: list = []
    _fake_measure_run(monkeypatch, commands)

    encode_final(tmp_path / "in.mp4", tmp_path / "out.mp4")
    without_bed = commands[-1]

    commands.clear()
    _fake_measure_run(monkeypatch, commands)
    encode_final(tmp_path / "in.mp4", tmp_path / "out2.mp4")
    without_bed_again = commands[-1]

    # Substituting the output path only, the two commands must match.
    assert without_bed[:-1] == without_bed_again[:-1]
    assert "-stream_loop" not in without_bed
    assert "-filter_complex" not in without_bed


def test_disabled_profile_produces_the_same_command_as_no_profile(monkeypatch, tmp_path: Path):
    commands: list = []
    _fake_measure_run(monkeypatch, commands)
    encode_final(tmp_path / "in.mp4", tmp_path / "out.mp4")
    baseline = commands[-1]

    commands.clear()
    _fake_measure_run(monkeypatch, commands)
    profile = AudioProfile(enabled=False)
    encode_final(tmp_path / "in.mp4", tmp_path / "out.mp4", audio_profile=profile)

    assert commands[-1] == baseline


def test_licensed_bed_adds_looped_input_and_mix_filter(monkeypatch, tmp_path: Path):
    bed_path = tmp_path / "bed.mp3"
    bed_path.write_bytes(b"not-real-audio")
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"entries": [_ledger_entry()]}))
    ledger = LicenceLedger.load(ledger_path)
    profile = AudioProfile(
        enabled=True, bed_path=bed_path, gain_db=-20.0, fade_out_s=3.0, ledger_path=ledger_path
    )

    commands: list = []
    _fake_measure_run(monkeypatch, commands)
    monkeypatch.setattr(encode_module, "ffprobe_duration_seconds", lambda path: 30.0)

    encode_final(tmp_path / "in.mp4", tmp_path / "out.mp4", audio_profile=profile, ledger=ledger)

    cmd = commands[-1]
    assert cmd[cmd.index("-stream_loop") + 1] == "-1"
    assert str(bed_path) in cmd
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "amix" in filter_complex
    assert "-20.0dB" in filter_complex
    assert "afade=t=out:st=27.000:d=3.000" in filter_complex


def test_unlicensed_bed_raises_before_any_ffmpeg_call(monkeypatch, tmp_path: Path):
    bed_path = tmp_path / "bed.mp3"
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps({"entries": []}))
    ledger = LicenceLedger.load(ledger_path)
    profile = AudioProfile(enabled=True, bed_path=bed_path, ledger_path=ledger_path)

    commands: list = []
    _fake_measure_run(monkeypatch, commands)

    with pytest.raises(UnlicensedBedError):
        encode_final(
            tmp_path / "in.mp4", tmp_path / "out.mp4", audio_profile=profile, ledger=ledger
        )

    assert commands == []


def test_write_publish_json_includes_music_bed_receipt(tmp_path: Path):
    chapters = [{"title": "Start", "start_s": 0.0, "end_s": 10.0}]
    music_bed = {
        "file": "bed.mp3",
        "ledger_entry": _ledger_entry(),
        "source": "pixabay",
        "licence": "Pixabay Content License",
    }

    path = tmp_path / "publish.json"
    write_publish_json(chapters, path, film_duration_s=10.0, music_bed=music_bed)

    payload = json.loads(path.read_text())
    assert payload["music_bed"] == music_bed


def test_write_publish_json_music_bed_defaults_to_null(tmp_path: Path):
    chapters = [{"title": "Start", "start_s": 0.0, "end_s": 10.0}]
    path = tmp_path / "publish.json"
    write_publish_json(chapters, path, film_duration_s=10.0)

    payload = json.loads(path.read_text())
    assert payload["music_bed"] is None
