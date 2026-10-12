"""Stabilise / deflicker pass.

Filter-probe parse, argv builders and the vidstab refusal are pure-Python
unit tests with a fake ``run``. One test exercises real ffmpeg deflicker on a
tiny synthetic alternating-brightness clip and checks the jitter gate — it is
skipped only when ffmpeg is genuinely absent.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from gomoveshift.pipeline import stabilize

FFMPEG = shutil.which("ffmpeg")


class _FakeCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _filters_stdout(names):
    header = " T.. abuffer            A->A       Buffer audio frames.\n"
    body = "".join(f" T.. {n:<18} V->V       fake filter.\n" for n in names)
    return header + body


# ── probe_ffmpeg_filters ──────────────────────────────────────────────────


def test_probe_parses_present_filters():
    caps = stabilize.probe_ffmpeg_filters(
        run=lambda *a, **k: _FakeCompleted(stdout=_filters_stdout(["deflicker", "scale"]))
    )
    assert caps.has_deflicker
    assert not caps.has_vidstab


def test_probe_needs_both_vidstab_filters():
    caps = stabilize.probe_ffmpeg_filters(
        run=lambda *a, **k: _FakeCompleted(stdout=_filters_stdout(["vidstabdetect"]))
    )
    assert not caps.has_vidstab

    caps_both = stabilize.probe_ffmpeg_filters(
        run=lambda *a, **k: _FakeCompleted(
            stdout=_filters_stdout(["vidstabdetect", "vidstabtransform"])
        )
    )
    assert caps_both.has_vidstab


def test_probe_matches_expected_ffmpeg_capabilities():
    """The test environment's ffmpeg is expected to have deflicker and to
    lack vidstab. If this ever flips, the whole refusal path is untested."""
    caps = stabilize.probe_ffmpeg_filters()
    assert caps.has_deflicker
    assert not caps.has_vidstab


# ── require_vidstab / refusal ─────────────────────────────────────────────


def test_require_vidstab_raises_named_reason_when_missing():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"deflicker"}))
    with pytest.raises(stabilize.StabilizeUnavailableError, match="libvidstab"):
        stabilize.require_vidstab(caps)


def test_require_vidstab_passes_when_present():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"vidstabdetect", "vidstabtransform"}))
    stabilize.require_vidstab(caps)  # no raise


def test_vidstabdetect_argv_refuses_without_vidstab():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"deflicker"}))
    with pytest.raises(stabilize.StabilizeUnavailableError):
        stabilize.build_vidstabdetect_argv("in.mp4", "transforms.trf", caps)


def test_vidstabtransform_argv_refuses_without_vidstab():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"deflicker"}))
    with pytest.raises(stabilize.StabilizeUnavailableError):
        stabilize.build_vidstabtransform_argv("in.mp4", "transforms.trf", "out.mp4", caps)


# ── argv builders ──────────────────────────────────────────────────────────


def test_build_deflicker_argv():
    argv = stabilize.build_deflicker_argv("in.mp4", "out.mp4", mode="pm", size=5)
    assert argv[0] == "ffmpeg"
    assert "in.mp4" in argv
    assert "out.mp4" in argv
    assert "-vf" in argv
    vf = argv[argv.index("-vf") + 1]
    assert vf == "deflicker=mode=pm:size=5"


def test_build_vidstabdetect_argv_when_available():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"vidstabdetect", "vidstabtransform"}))
    argv = stabilize.build_vidstabdetect_argv("in.mp4", "t.trf", caps, shakiness=6, accuracy=10)
    vf = argv[argv.index("-vf") + 1]
    assert vf == "vidstabdetect=shakiness=6:accuracy=10:result=t.trf"


def test_build_vidstabtransform_argv_when_available():
    caps = stabilize.FfmpegFilterCapabilities(frozenset({"vidstabdetect", "vidstabtransform"}))
    argv = stabilize.build_vidstabtransform_argv(
        "in.mp4", "t.trf", "out.mp4", caps, smoothing=15, zoom=0
    )
    vf = argv[argv.index("-vf") + 1]
    assert vf == "vidstabtransform=input=t.trf:smoothing=15:zoom=0"


# ── jitter gate ─────────────────────────────────────────────────────────────


def test_gate_on_jitter_keeps_when_output_lower():
    result = stabilize.gate_on_jitter(input_jitter=5.0, output_jitter=1.0)
    assert result.passed
    assert result.reason == ""


def test_gate_on_jitter_discards_when_output_not_lower():
    result = stabilize.gate_on_jitter(input_jitter=1.0, output_jitter=5.0)
    assert not result.passed
    assert "did not drop" in result.reason


def test_measure_luma_jitter_parses_yavg_and_needs_two_samples():
    stderr = (
        "frame=1 lavfi.signalstats.YAVG=10.0\n"
        "frame=2 lavfi.signalstats.YAVG=50.0\n"
        "frame=3 lavfi.signalstats.YAVG=10.0\n"
    )
    jitter = stabilize.measure_luma_jitter(
        "clip.mp4", run=lambda *a, **k: _FakeCompleted(stderr=stderr)
    )
    assert jitter > 0

    with pytest.raises(ValueError):
        stabilize.measure_luma_jitter(
            "clip.mp4", run=lambda *a, **k: _FakeCompleted(stderr="no samples here")
        )


def test_run_deflicker_raises_on_encode_failure():
    with pytest.raises(RuntimeError):
        stabilize.run_deflicker(
            "in.mp4",
            "out.mp4",
            run=lambda *a, **k: _FakeCompleted(returncode=1, stderr="boom"),
        )


# ── real ffmpeg: deflicker on a synthetic flickering clip ─────────────────


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_deflicker_reduces_measured_jitter_on_real_ffmpeg(tmp_path):
    import numpy as np

    width, height, fps, n_frames = 64, 64, 24, 40
    input_path = tmp_path / "flicker_in.mp4"

    argv = [
        FFMPEG,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-pix_fmt",
        "yuv420p",
        str(input_path),
    ]
    frames = []
    for i in range(n_frames):
        brightness = 30 if i % 2 == 0 else 225
        frames.append(np.full((height, width, 3), brightness, dtype=np.uint8))
    raw = b"".join(f.tobytes() for f in frames)

    proc = subprocess.run(argv, input=raw, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")

    output_path = tmp_path / "flicker_out.mp4"
    gate = stabilize.run_deflicker(input_path, output_path)

    assert gate.passed, gate.reason
    assert gate.output_jitter < gate.input_jitter
    assert output_path.exists()
