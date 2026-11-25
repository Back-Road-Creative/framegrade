"""A stream-copy join of pieces from different encoders decodes end to end.

Galley Head 2026-11-25: a montage clip re-encoded with libx265 (no keyframe at
the peak start) was prepended with ``-c copy`` to the NVENC long-form body. The
concat demuxer keeps only the FIRST file's parameter sets (VPS/SPS/PPS in the
MP4 ``hvcC`` box), so every body frame failed to decode: the container said
65 min, the OST burn decoded 12.95 s of video, and the drive failed
``OST-OUTPUT-INVALID … A/V drift`` on every heal (seven times that day).

These tests use real ffmpeg on synthetic HEVC files from two different x265
configurations, so the parameter sets differ the way libx265 and NVENC do.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import ffmpeg_runner, montage_builder

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="needs ffmpeg and ffprobe",
)

FPS = "30000/1001"


def _synth(out: Path, seconds: int, x265_params: str, preset: str, src: str) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"{src}=size=320x180:rate={FPS}",
            "-f",
            "lavfi",
            "-i",
            "sine=f=440:sample_rate=48000",
            "-t",
            str(seconds),
            "-c:v",
            "libx265",
            "-preset",
            preset,
            "-x265-params",
            f"log-level=error:{x265_params}",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            str(out),
        ],
        check=True,
    )
    return out


def _encoder_a(out: Path, seconds: int = 2) -> Path:
    """Stands in for the CPU re-encode (``lf_reencode_args``)."""
    return _synth(out, seconds, "keyint=30", "slow", "testsrc")


def _encoder_b(out: Path, seconds: int = 6) -> Path:
    """Stands in for the NVENC master (different SPS/PPS)."""
    return _synth(
        out, seconds, "keyint=60:bframes=0:no-wpp=1:aq-mode=0:ref=1", "ultrafast", "testsrc2"
    )


def _decoded_frames(path: Path) -> int:
    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    counts = [line for line in res.stdout.split() if line.strip()]
    return int(counts[-1]) if counts else 0


def _decode_errors(path: Path) -> list[str]:
    """Video decode errors (the join's AAC seams may overlap by a few ms, as a plain concat's do)."""
    res = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    return res.stderr.strip().splitlines()


def _decodes_cleanly(path: Path, frames: int) -> None:
    assert _decoded_frames(path) == frames
    assert _decode_errors(path) == []


def _frames_in_container(path: Path) -> int:
    res = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(res.stdout.strip())


@pytest.fixture
def pieces(tmp_path):
    a = _encoder_a(tmp_path / "a.mp4")
    b = _encoder_b(tmp_path / "b.mp4")
    a2 = _encoder_a(tmp_path / "a2.mp4")
    return SimpleNamespace(a=a, b=b, a2=a2)


def test_parameter_sets_tell_the_two_encoders_apart(pieces, tmp_path):
    sa = montage_builder.video_parameter_sets(pieces.a)
    assert sa is not None and sa[0] == "hevc"
    assert sa == montage_builder.video_parameter_sets(pieces.a2)
    assert sa != montage_builder.video_parameter_sets(pieces.b)
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video")
    assert montage_builder.video_parameter_sets(junk) is None


def _prepend(tmp_path, montage: Path, body: Path, **kw):
    failures: list = []

    def _built(*a, output_path, **k):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(montage, output_path)
        return output_path

    with patch.object(montage_builder, "build_montage", side_effect=_built):
        out = montage_builder.prepend_montage(
            raw_lf_path=body,
            peaks=[object()],
            output_path=tmp_path / "lf_02_with_montage.mp4",
            work_dir=tmp_path / "work",
            master_path=tmp_path / "master.mp4",
            file_boundaries=[],
            failures=failures,
            **kw,
        )
    return out, failures


def test_prepend_of_a_reencoded_montage_decodes_the_whole_body(pieces, tmp_path):
    out, failures = _prepend(tmp_path, pieces.a, pieces.b)
    assert failures == [] and out is not None
    _decodes_cleanly(out, _frames_in_container(pieces.a) + _frames_in_container(pieces.b))


def test_prepend_of_matching_pieces_keeps_the_plain_stream_copy(pieces, tmp_path):
    seen: list = []
    real = ffmpeg_runner.run_ffmpeg_encode

    def _spy(cmd, *a, **kw):
        seen.append((cmd, kw.get("pass_fds", ())))
        return real(cmd, *a, **kw)

    with patch.object(ffmpeg_runner, "run_ffmpeg_encode", side_effect=_spy):
        out, failures = _prepend(tmp_path, pieces.a, pieces.a2)
    assert failures == [] and out is not None
    ((cmd, fds),) = seen
    assert fds == () and "-protocol_whitelist" not in cmd
    _decodes_cleanly(out, _frames_in_container(pieces.a) * 2)


def test_unreadable_parameter_sets_warn_and_keep_the_plain_copy(tmp_path, caplog):
    seen: list = []

    def _encode(cmd, output_path, expected_duration, description, **kw):
        seen.append(kw.get("pass_fds", ()))
        Path(output_path).write_bytes(b"x")
        return SimpleNamespace(success=True, killed_reason=None, stderr_tail="", returncode=0)

    body = tmp_path / "body.mp4"
    body.write_bytes(b"b")
    montage = tmp_path / "m.mp4"
    montage.write_bytes(b"m")
    with (
        patch.object(ffmpeg_runner, "run_ffmpeg_encode", side_effect=_encode),
        caplog.at_level(logging.WARNING),
    ):
        out, failures = _prepend(tmp_path, montage, body)
    assert out is not None and seen == [()]
    assert any("parameter sets" in r.getMessage() for r in caplog.records)


def test_a_reencoded_clip_inside_the_montage_decodes(pieces, tmp_path):
    """A copied clip, then a re-encoded one, then a copied one (the montage's own join)."""
    order = [pieces.b, pieces.a, pieces.b]
    made = iter(order)

    def _trim(master, peak, fb, output, clip_duration, **kw):
        shutil.copy(next(made), output)
        return True

    with (
        patch.object(montage_builder, "_trim_peak_to", side_effect=_trim),
        patch.object(montage_builder, "_montage_output_is_sound", return_value=True),
    ):
        out = montage_builder.build_montage(
            master_path=tmp_path / "master.mp4",
            peaks=[object(), object(), object()],
            file_boundaries=[],
            output_path=tmp_path / "montage.mp4",
            work_dir=tmp_path / "w",
            total_duration=6.0,
            max_total_duration=6.0,
        )
    assert out is not None
    _decodes_cleanly(out, sum(_frames_in_container(p) for p in order))


def test_a_montage_with_mixed_clips_prepends_decodably_even_when_its_first_clip_matches_the_body(
    pieces, tmp_path
):
    """The montage starts with the body's parameter sets, yet its later clip carries others.

    The re-encoded clip has an open-GOP keyframe, so a second remux of an MP4
    montage would drop that keyframe's leading pictures.
    """
    order = [pieces.b, pieces.a]
    made = iter(order)

    def _trim(master, peak, fb, output, clip_duration, **kw):
        shutil.copy(next(made), output)
        return True

    failures: list = []
    with (
        patch.object(montage_builder, "_trim_peak_to", side_effect=_trim),
        patch.object(montage_builder, "_montage_output_is_sound", return_value=True),
    ):
        out = montage_builder.prepend_montage(
            raw_lf_path=pieces.b,
            peaks=[object(), object()],
            output_path=tmp_path / "lf_with_montage.mp4",
            work_dir=tmp_path / "w",
            master_path=tmp_path / "master.mp4",
            file_boundaries=[],
            failures=failures,
        )
    assert failures == [] and out is not None
    want = sum(_frames_in_container(p) for p in order) + _frames_in_container(pieces.b)
    _decodes_cleanly(out, want)


def test_a_piece_that_cannot_be_remuxed_fails_the_join_and_leaves_no_output(pieces, tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    out_path = tmp_path / "joined.mp4"
    cause = montage_builder.annexb_join(
        [pieces.a, bad],
        out_path,
        tmp_path / "list.txt",
        "hevc",
        description="test join",
    )
    assert cause and "bad.mp4" in cause
    assert not out_path.exists()
