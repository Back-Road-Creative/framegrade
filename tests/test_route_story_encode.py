"""Route Story final encode: delivery quality/colour/loudness pass and sidecars."""

import json
import re
import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline.route_story import encode as encode_module
from gomoveshift.pipeline.route_story.beats import Beat, chapter_rows, write_chapters_vtt
from gomoveshift.pipeline.route_story.encode import encode_final, write_publish_json


def _tiny_source_with_audio(path: Path, duration: float = 0.6) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s=64x64:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def test_encode_final_produces_bt709_colour_tags_and_an_audio_stream(tmp_path: Path):
    source = tmp_path / "source.mp4"
    _tiny_source_with_audio(source)
    output = tmp_path / "final.mp4"

    encode_final(source, output)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,color_primaries,color_transfer",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    assert video["color_primaries"] == "bt709"
    assert video["color_transfer"] == "bt709"
    assert any(s["codec_type"] == "audio" for s in streams)


def test_encode_final_uses_crf_18_preset_slow(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(encode_module.subprocess, "run", fake_run)

    encode_final(tmp_path / "in.mp4", tmp_path / "out.mp4")

    cmd = captured["cmd"]
    assert cmd[cmd.index("-crf") + 1] == "18"
    assert cmd[cmd.index("-preset") + 1] == "slow"
    # The audio filter is pinned in test_route_story_encode_audio.py.


def _beats() -> list[Beat]:
    return [
        Beat("start", 0.0, 0.0, 0.0, 100.0, "reveal", (), 7.0),
        Beat("high_point", 10.0, 0.0, 0.0, 300.0, "arc", (("Elevation", "984 ft"),), 7.0),
        Beat("finish", 20.0, 0.0, 0.0, 290.0, "pullback", (), 5.0),
    ]


def test_publish_json_and_chapters_vtt_come_from_one_chapter_list(tmp_path: Path):
    """One source: the beats' chapter rows feed both sidecars, so they cannot disagree."""
    rows = chapter_rows(_beats(), lambda t: t * 2.0, film_duration_s=50.0)
    assert rows == [
        {"title": "Start", "start_s": 0.0, "end_s": 20.0},
        {"title": "High point \u00b7 984 ft", "start_s": 20.0, "end_s": 40.0},
        {"title": "Finish", "start_s": 40.0, "end_s": 50.0},
    ]

    vtt_path = tmp_path / "chapters.vtt"
    publish_path = tmp_path / "publish.json"
    write_chapters_vtt(_beats(), lambda t: t * 2.0, vtt_path, film_duration_s=50.0)
    write_publish_json(rows, publish_path, film_duration_s=50.0)

    vtt_text = vtt_path.read_text(encoding="utf-8")
    payload = json.loads(publish_path.read_text(encoding="utf-8"))
    assert payload["duration_s"] == 50.0
    assert payload["chapters"] == rows
    for row in rows:
        assert row["title"] in vtt_text
    assert len(re.findall(r"-->", vtt_text)) == len(rows)


def test_write_publish_json_refuses_an_empty_chapter_list(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one"):
        write_publish_json([], tmp_path / "publish.json", film_duration_s=10.0)


def test_encode_module_has_no_second_chapters_writer():
    assert not hasattr(encode_module, "write_chapters_vtt")


def test_encode_final_keeps_the_audio_at_48_khz(tmp_path: Path):
    # With no explicit output rate a resampling filter hands the encoder its own
    # rate: a loudnorm-only encode once shipped 96 kHz AAC.
    source = tmp_path / "source.mp4"
    _tiny_source_with_audio(source)
    output = tmp_path / "final.mp4"

    encode_final(source, output)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=sample_rate,channels",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    (audio,) = json.loads(probe.stdout)["streams"]
    assert audio["sample_rate"] == "48000"
    assert audio["channels"] == 2
