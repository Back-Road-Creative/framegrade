"""Insert engine: stills and clips keycut into a long-form (ffmpeg mocked)."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import keycut
import pytest
from PIL import Image

from gomoveshift.pipeline import montage_builder as mb
from gomoveshift.pipeline.encoding_constants import RENDITIONS
from gomoveshift.pipeline.inserts import (
    Insert,
    MediaInfo,
    apply_inserts,
    plan_inserts,
)
from gomoveshift.pipeline.quality import UpscaleRefusedError
from gomoveshift.publishing.brand_policy import MediaProfile

LF = Path("/work/lf.mp4")
KEYFRAMES = [float(t) for t in range(0, 100, 2)]
RENDITION = replace(RENDITIONS["youtube_longform"], width=3840, height=2160)


def _probe(path: Path) -> MediaInfo:
    if Path(path) == LF:
        return MediaInfo(width=3840, height=2160, fps=30.0, duration=100.0)
    return MediaInfo(width=1920, height=1080, fps=30.0, duration=6.0)


def _still(path: Path) -> Image.Image:
    return Image.new("RGB", (1000, 2000), (255, 255, 255))  # portrait, white


def _plan(tmp_path, inserts, policy="resample"):
    return plan_inserts(
        LF,
        inserts,
        profile=MediaProfile(upscale=policy),
        rendition=RENDITION,
        probe=_probe,
        keyframes=lambda p: KEYFRAMES,
        load_still=_still,
        work_dir=tmp_path,
    )


def _recorder(calls):
    def run(argv, **kw):
        calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    return run


def test_insert_still_keeps_rest_bit_exact_outside_bounding_gops(tmp_path):
    plans = _plan(tmp_path, [Insert(at_s=10.5, media=Path("/m/a.jpg"), hold_s=4.0)])
    assert (plans[0].cut_in, plans[0].cut_out) == (10.0, 16.0)
    calls: list[list[str]] = []
    result = apply_inserts(LF, plans, tmp_path / "out.mp4", run=_recorder(calls))
    copies = [
        c for c in calls if "-c" in c and c[c.index("-c") + 1] == "copy" and "concat" not in c
    ]
    assert len(copies) == 2
    assert [(s.start, s.end) for s in result.spans if s.stream_copied] == [
        (0.0, 10.0),
        (16.0, 100.0),
    ]
    reenc = [s for s in result.spans if not s.stream_copied]
    assert [(s.start, s.end) for s in reenc] == [(10.0, 16.0)]
    encodes = [c for c in calls if "-filter_complex" in c]
    assert len(encodes) == 1
    assert "libx265" in encodes[0]
    assert "-map" in encodes[0] and "0:a" in encodes[0]  # LF audio kept under a still
    assert calls[-1][calls[-1].index("-f") + 1] == "concat"
    assert "-c" in calls[-1] and calls[-1][calls[-1].index("-c") + 1] == "copy"


def test_insert_clip_fit_recorded_as_upscale(tmp_path):
    plans = _plan(tmp_path, [Insert(at_s=20.0, media=Path("/m/phone.mp4"))])
    p = plans[0]
    assert p.kind == "clip"
    assert p.upscaled_from == [1920, 1080]
    assert p.scaler_filter.startswith("zscale=")
    assert (p.cut_in, p.cut_out) == (20.0, 26.0)  # clip's own 6 s duration, not hold_s
    calls: list[list[str]] = []
    result = apply_inserts(LF, plans, tmp_path / "out.mp4", run=_recorder(calls))
    fc = next(c for c in calls if "-filter_complex" in c)
    graph = fc[fc.index("-filter_complex") + 1]
    assert graph.index("zscale") < graph.index("overlay")
    assert result.upscaled == [[1920, 1080]]


def test_insert_clip_refused_under_policy_off(tmp_path):
    with pytest.raises(UpscaleRefusedError):
        _plan(tmp_path, [Insert(at_s=20.0, media=Path("/m/phone.mp4"))], policy="off")


def test_still_card_style_letterbox_vs_full(tmp_path):
    full = _plan(tmp_path / "f", [Insert(10.5, Path("/m/a.jpg"), style="full")])[0]
    box = _plan(tmp_path / "b", [Insert(10.5, Path("/m/a.jpg"), style="letterbox")])[0]
    with Image.open(full.card_path) as f, Image.open(box.card_path) as b:
        assert f.size == b.size == (3840, 2160)
        assert b.getpixel((2, 1080)) == (0, 0, 0)  # letterbox bars are black
        assert f.getpixel((2, 1080)) != (0, 0, 0)  # full keeps the card backdrop
    with pytest.raises(ValueError):
        _plan(tmp_path, [Insert(10.5, Path("/m/a.jpg"), style="bogus")])


def test_montage_helpers_extracted_keep_argv(tmp_path, monkeypatch):
    seen: list[list[str]] = []

    def fake_run(cmd, **kw):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr("keycut.extract.subprocess.run", fake_run)
    src, out = tmp_path / "m.mp4", tmp_path / "o.mp4"
    keycut.stream_copy_range(src, 1.0, 3.0, out)
    keycut.reencode_range(src, 1.0, 3.0, out, encode_args=mb.lf_reencode_args())
    keycut.concat_demuxer([src], out, tmp_path / "l.txt")
    assert mb.copy_range_argv(src, 1.0, 3.0, out) == seen[0]
    assert mb.reencode_range_argv(src, 1.0, 3.0, out) == seen[1]
    # keycut >= 0.1.2 seeks on the INPUT side (`-ss S -i src`), then zeroes the
    # output clock and bounds the length with `-t D` (keycut/extract.py
    # reencode_range); the pre-0.1.2 form was `-i src -ss S -to E`.
    assert seen[1] == [
        "ffmpeg", "-y", "-ss", "1.0", "-i", str(src), "-ss", "0", "-t", "2.0",
        *mb.lf_reencode_args(), str(out),
    ]  # fmt: skip
    assert mb.concat_copy_argv(tmp_path / "l.txt", out) == seen[2]
    assert mb.keyframe_span(10.5, 14.5, KEYFRAMES, 100.0) == (10.0, 16.0)
