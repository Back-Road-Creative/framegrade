"""The remaining analysis extractions state the source range.

GoPro footage is full-range YUV carrying no range tag, and every automatic
conversion assumes LIMITED on an untagged stream. The grade chain states it with
``setrange=<probed>``; these are the leftover readers. The
load-bearing test is ``test_cv2_alignment_matches_a_setrange_extraction``: the
two forms of the fix (an ffmpeg filter, and an after-the-fact correction where
OpenCV gives no range knob at all) must land pixels in the SAME numeric domain.
A half-tagged codebase is worse than an untagged one — then the readers disagree
and nothing says which is right.
"""

from __future__ import annotations

import importlib
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from gomoveshift.analysis.frame_range import (
    align_cv2_frame_range,
    probe_color_range,
    setrange_filter,
)
from gomoveshift.config import Config

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _ramp_video(path, *, tag: str | None = None) -> None:
    """A 0-255 luma ramp, so a range conversion is a measurable shift.
    ``setparams``, not ``-color_range``: frame properties beat the output flag."""
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray",
         "-video_size", "256x64", "-framerate", "5", "-i", "-",
         *(["-vf", f"setparams=range={tag}"] if tag else []),
         "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
         "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    frame = np.tile(np.arange(256, dtype=np.uint8), (64, 1)).tobytes()
    for _ in range(5):
        proc.stdin.write(frame)
    proc.stdin.close()
    assert proc.wait() == 0


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    probe_color_range.cache_clear()
    yield
    probe_color_range.cache_clear()


@pytest.fixture
def ramp(tmp_path):
    _ramp_video(src := tmp_path / "untagged.mp4")
    return src


@needs_ffmpeg
def test_untagged_source_probes_as_full(ramp):
    """No tag means full range — not the ``limited`` swscale would infer."""
    assert probe_color_range(ramp) == "full"
    assert setrange_filter(ramp) == "setrange=full"


@pytest.mark.parametrize(
    "tag,expected",
    [("tv", "limited"), ("mpeg", "limited"), ("LIMITED", "limited"),
     ("pc", "full"), ("jpeg", "full"), ("unknown", "full"), ("", "full")],
)  # fmt: skip
def test_every_ffprobe_spelling_maps_to_one_of_two_answers(tag, expected, monkeypatch, tmp_path):
    """A stub is the honest way to cover this table: libx264 will not write a
    ``tv`` VUI flag at all, so a real fixture cannot reach half of it."""
    monkeypatch.setattr(
        "gomoveshift.analysis.frame_range.subprocess.run",
        lambda *a, **k: subprocess.CompletedProcess([], 0, tag, ""),
    )
    src = tmp_path / f"{tag or 'blank'}.mp4"
    assert probe_color_range(src) == expected
    assert setrange_filter(src) == f"setrange={expected}"
    frame = np.arange(256, dtype=np.uint8).reshape(1, 256, 1).repeat(3, axis=2)
    if expected == "limited":
        # Already the domain everything assumes: not a single pixel may move.
        assert np.array_equal(align_cv2_frame_range(frame, src), frame)


def test_unreadable_source_falls_back_to_full(tmp_path):
    """A failed probe must not silently pick the other answer."""
    (bogus := tmp_path / "nope.mp4").write_bytes(b"not a video")
    assert probe_color_range(bogus) == "full"


@needs_ffmpeg
def test_cv2_alignment_matches_a_setrange_extraction(ramp, tmp_path):
    """The OpenCV correction must land on the ffmpeg readers' numbers."""
    out = tmp_path / "tagged.png"
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-v", "error", "-i", str(ramp),
         "-vf", f"{setrange_filter(ramp)},format=rgb24", "-frames:v", "1", str(out)],
        check=True,
    )  # fmt: skip
    via_ffmpeg = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)[0].astype(np.int16)

    cap = cv2.VideoCapture(str(ramp))
    ok, frame = cap.read()
    cap.release()
    assert ok
    via_cv2 = align_cv2_frame_range(frame, ramp)[0, :, 0].astype(np.int16)

    assert np.abs(via_ffmpeg - via_cv2).max() <= 2
    # Self-proving: the uncorrected decode is far enough away to fail the above.
    assert np.abs(via_ffmpeg - frame[0, :, 0].astype(np.int16)).max() > 10


def _spy(monkeypatch, module, name):
    """Record every call to ``module.<name>`` and still perform it."""
    seen = []
    real = getattr(module, name)

    def wrapped(cmd, *a, **k):
        seen.append([str(c) for c in cmd])
        return real(cmd, *a, **k)

    monkeypatch.setattr(module, name, wrapped)
    return seen


@needs_ffmpeg
@pytest.mark.parametrize("decode", ["_decode_frames_ffmpeg", "_decode_frames_ffmpeg_fixed_fps"])
def test_motion_decodes_state_the_range(ramp, monkeypatch, decode):
    """Both gray decoders, not just the one someone happened to look at."""
    from gomoveshift.analysis import motion as motion_mod
    from gomoveshift.analysis.motion import MotionAnalyzer

    seen = _spy(monkeypatch, motion_mod.subprocess, "Popen")
    getattr(MotionAnalyzer(Config()), decode)(ramp)

    assert seen, "no ffmpeg decode was launched"
    vf = seen[-1][seen[-1].index("-vf") + 1]
    assert vf.startswith("setrange=full,"), f"range must be stated first: {vf!r}"


@needs_ffmpeg
def test_sign_pass_frame_grab_states_the_range(ramp, monkeypatch):
    from gomoveshift.analysis import sign_pass

    seen = _spy(monkeypatch, sign_pass.subprocess, "run")
    assert sign_pass._default_grab_frame(ramp, 0.4) is not None
    assert seen[-1][seen[-1].index("-vf") + 1] == "setrange=full"


@needs_ffmpeg
@pytest.mark.parametrize("attr", ["SignDetector", "ShortsExtractor"])
def test_cv2_readers_align_every_decoded_frame(ramp, monkeypatch, attr):
    """A reader skipping the alignment reintroduces the split domain."""
    module = "signs" if attr == "SignDetector" else "shorts_extractor"
    mod = importlib.import_module(f"gomoveshift.analysis.{module}")
    calls = []

    def spy(frame, video_path):
        calls.append(1)
        return align_cv2_frame_range(frame, video_path)

    monkeypatch.setattr("gomoveshift.analysis.frame_range.align_cv2_frame_range", spy)
    monkeypatch.setattr(mod, "align_cv2_frame_range", spy, raising=False)

    obj = getattr(mod, attr)(Config())
    if attr == "SignDetector":
        list(obj.detect_candidates(ramp))
    else:
        obj.analyze_segment_visual_interest(ramp, 0.0, 0.8, num_samples=2)
    assert calls, "decoded frames reached the scorer without a range alignment"
