"""Q0c: review_set_render -- paired stills, excerpts and provenance (real ffmpeg, synthetic media)."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from gomoveshift.pipeline import review_set_render as rsr

FPS = 10


def _make_synthetic(path: Path, order: str) -> None:
    """4 s, 64x48, 10 fps; one flat colour per second (synthetic, never real footage)."""
    chans = {"r": 0, "g": 1, "b": 2}
    parts = {
        c: f"if(lt(T,1),{255 if order[0] == c else 0},if(lt(T,2),{255 if order[1] == c else 0},{255 if order[2] == c else 0}))"
        for c in chans
    }  # noqa: E501
    vf = f"format=rgb24,geq=r='{parts['r']}':g='{parts['g']}':b='{parts['b']}',format=yuv420p"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i",
         f"color=c=black:s=64x48:r={FPS}:d=4", "-vf", vf, "-c:v", "libx264",
         "-crf", "12", "-g", "5", str(path)],
        check=True,
    )  # fmt: skip


def _png_rgb(path: Path) -> tuple[int, int, int]:
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vf", "scale=1:1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    assert len(raw) == 3
    return raw[0], raw[1], raw[2]


def _dominant(rgb: tuple[int, int, int]) -> str:
    return "rgb"[max(range(3), key=lambda i: rgb[i])]


def _probe(path: Path) -> tuple[float, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames:format=duration", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    j = json.loads(out)
    return float(j["format"]["duration"]), int(j["streams"][0]["nb_read_frames"])


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic_media")
    src, out = d / "src.mp4", d / "out.mp4"
    _make_synthetic(src, "rgb")  # red, green, blue
    _make_synthetic(out, "brg")  # blue, red, green
    return src, out


@pytest.fixture(scope="module")
def rendered(media, tmp_path_factory):
    src, out = media
    dest = tmp_path_factory.mktemp("review") / "set"
    result = rsr.render_review_set(
        src, out, [(0.5, 1.5), (2.5, 0.5)], dest, excerpt_seconds=1.0, synthetic=True
    )
    return dest, result


def test_stills_show_requested_frames(rendered):
    dest, result = rendered
    p0, p1 = result["pairs"]
    assert _dominant(_png_rgb(dest / p0["source_still"])) == "r"  # src 0.5 s red
    assert _dominant(_png_rgb(dest / p0["output_still"])) == "r"  # out 1.5 s: brg -> second 1 = r
    assert _dominant(_png_rgb(dest / p1["source_still"])) == "b"  # src 2.5 s blue
    assert _dominant(_png_rgb(dest / p1["output_still"])) == "b"  # out 0.5 s blue


def test_excerpts_start_and_duration(rendered):
    dest, result = rendered
    for pair, (s_col, o_col) in zip(result["pairs"], [("r", "r"), ("b", "b")]):
        for key, col in (("source_excerpt", s_col), ("output_excerpt", o_col)):
            path = dest / pair[key]
            dur, frames = _probe(path)
            assert abs(frames - FPS * 1.0) <= 1
            assert abs(dur - 1.0) <= 1.0 / FPS + 1e-6
            first = dest / f"_first_{path.stem}.png"
            subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(path),
                 "-frames:v", "1", str(first)], check=True,
            )  # fmt: skip
            assert _dominant(_png_rgb(first)) == col
            first.unlink()


def test_provenance(rendered, media):
    dest, result = rendered
    prov = json.loads((dest / "provenance.json").read_text())
    assert prov == json.loads(json.dumps(result["provenance"]))
    from gomoveshift import __version__

    assert prov["version"] == __version__
    assert "git_sha" in prov and (prov["git_sha"] is None or len(prov["git_sha"]) >= 7)
    assert prov["ffmpeg_version"].startswith("ffmpeg version")
    assert isinstance(prov["config"], dict) and prov["config"]
    assert prov["synthetic"] is True
    src, out = media
    assert prov["inputs"]["source"]["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert prov["inputs"]["output"]["size_bytes"] == out.stat().st_size
    assert prov["excerpt_args"] and "-c copy" not in " ".join(prov["excerpt_args"])
    assert prov["pairs"] == [[0.5, 1.5], [2.5, 0.5]]


def test_git_sha_null_when_unknown(media, tmp_path, monkeypatch):
    monkeypatch.setattr(rsr, "product_sha", lambda: None)
    res = rsr.render_review_set(*media, [(0.5, 0.5)], tmp_path / "o", excerpt_seconds=0.5)
    assert res["provenance"]["git_sha"] is None
    assert res["provenance"]["synthetic"] is False


def test_refuses_non_empty_dir(media, tmp_path):
    (tmp_path / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        rsr.render_review_set(*media, [(0.5, 0.5)], tmp_path, excerpt_seconds=0.5)
    assert [p.name for p in tmp_path.iterdir()] == ["keep.txt"]


def test_empty_existing_dir_ok_and_bad_inputs(media, tmp_path):
    empty = tmp_path / "e"
    empty.mkdir()
    rsr.render_review_set(*media, [(0.5, 0.5)], empty, excerpt_seconds=0.5)
    with pytest.raises(ValueError):
        rsr.render_review_set(*media, [], tmp_path / "n", excerpt_seconds=0.5)
    with pytest.raises(FileNotFoundError):
        rsr.render_review_set(tmp_path / "nope.mp4", media[1], [(0, 0)], tmp_path / "m")
