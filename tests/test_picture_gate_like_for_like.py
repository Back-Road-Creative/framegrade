"""The picture gate compares each delivered frame with the frame it was made from.

Mizen (Ireland 2025-09-17, 2026-11-16): the day master's gate sampled the master
at 25 evenly spaced seconds but each source episode at its own spacing, so the
two medians came from different scenes. Sharpness varies by scene from 90 to
2614 on that drive, and a faithful master (335.8 against 405.8 on the same 25
frames, inside the 25% tolerance) failed at 335.8 < 448.1 (source 597.5).

Here ``ffmpeg`` is replaced by a fake that renders a noise frame whose texture
depends on the scene at that point of the day's timeline, so the real sampler
runs. The frame choice, and for the day master the contrast the grade left, decide
the verdict. Every name is invented.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline import day_pipeline, quality
from gomoveshift.pipeline.validation import longform_validator as lv

DATE = "2026-02-20"
EPISODES = (300.0, 100.0)


def _scene(pos: float) -> float:
    """Texture at timeline second ``pos``: busy and plain scenes alternate every 8 s."""
    return 40.0 if math.floor(pos / 8.0) % 2 == 0 else 12.0


def _png(amp: float, seed: int, blur: float = 0.0) -> bytes:
    rng = np.random.default_rng(seed)
    frame = np.clip(128.0 + amp * rng.uniform(-1.0, 1.0, (96, 96)), 0, 255).astype(np.uint8)
    if blur:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    return cv2.imencode(".png", frame)[1].tobytes()


def _fake_ffmpeg(monkeypatch, where, *, keep: float = 0.91, blur: float = 0.0):
    """Patch ffmpeg: ``where(name, second, vf)`` gives the frame's timeline second.

    A delivered file (``keep`` of the texture) and its reference render the same
    noise at the same timeline second, so a faithful file differs only by ``keep``.
    ``blur`` smears the delivered frame (a real loss of detail: the gate puts the
    reference at the delivered contrast, so a lower ``keep`` alone is a tone grade).
    """
    calls = []

    def run(cmd, capture_output=True, timeout=None):
        name = Path(cmd[cmd.index("-i") + 1]).name
        second = float(cmd[cmd.index("-ss") + 1])
        vf = cmd[cmd.index("-vf") + 1]
        calls.append((name, round(second, 3), vf))
        pos, delivered = where(name, second, vf)
        amp = _scene(pos) * (keep if delivered else 1.0)
        return SimpleNamespace(
            returncode=0, stdout=_png(amp, int(round(pos * 1000)), blur if delivered else 0.0)
        )

    monkeypatch.setattr(quality.subprocess, "run", run)
    return calls


# ── the day master: paired with its episodes on the day's timeline ──────────


def _day(tmp_path, *, amount=0.0, denoise=2.5):
    episodes = []
    for i, d in enumerate(EPISODES):
        concat = tmp_path / f"episode_{i + 1:02d}_{DATE}.mp4"
        concat.write_bytes(b"x")
        episodes.append(SimpleNamespace(concat_path=concat, total_duration=d))
    batch = SimpleNamespace(episodes=episodes, total_duration=sum(EPISODES))
    day_dir = tmp_path / f"day_{DATE}"
    day_dir.mkdir()
    master = day_dir / f"day_{DATE}_master.mp4"
    master.write_bytes(b"m")
    (day_dir / f"day_{DATE}_color_cache.json").write_text(
        json.dumps({"corrections": [{"sharpen_amount": amount, "denoise_strength": denoise}]})
    )
    return batch, master


def _on_day_timeline(name, second, vf):
    if name.endswith("_master.mp4"):
        return second, True
    offset = 0.0 if name.startswith("episode_01") else EPISODES[0]
    return offset + second, False


def test_a_faithful_master_passes_when_its_frames_are_paired(tmp_path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, _on_day_timeline)
    batch, master = _day(tmp_path)
    day_pipeline._gate_master_picture(batch, master, DATE)
    master_seconds = [s for n, s, _ in calls if n.endswith("_master.mp4")]
    source_seconds = [_on_day_timeline(n, s, v)[0] for n, s, v in calls if n.startswith("ep")]
    assert master_seconds == [round(t, 3) for t in quality.gate_timestamps(sum(EPISODES))]
    assert source_seconds == pytest.approx(master_seconds, abs=1e-3)


def test_a_soft_master_still_fails_on_the_same_frames(tmp_path, monkeypatch):
    _fake_ffmpeg(monkeypatch, _on_day_timeline, keep=1.0, blur=1.0)
    batch, master = _day(tmp_path)
    with pytest.raises(RuntimeError, match=f"Day {DATE}.*delivered sharpness"):
        day_pipeline._gate_master_picture(batch, master, DATE)


def test_a_missing_episode_defers_without_sampling(tmp_path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, _on_day_timeline)
    batch, master = _day(tmp_path)
    batch.episodes[1].concat_path.unlink()
    day_pipeline._gate_master_picture(batch, master, DATE)
    assert calls == []


def test_timeline_position_maps_a_concat_second_to_its_file():
    files = ["a.mp4", "b.mp4"]
    assert quality.timeline_position(files, EPISODES, 10.0) == ("a.mp4", 10.0)
    assert quality.timeline_position(files, EPISODES, 310.0) == ("b.mp4", pytest.approx(10.0))
    assert quality.timeline_position(files, EPISODES, 400.0) == ("b.mp4", pytest.approx(100.0))


# ── a long-form: paired with its own input, before the text burn ────────────


def _same_seconds(name, second, vf):
    return second, name.startswith("lf_01.mp4")


def test_an_lf_is_judged_against_its_pre_text_input(tmp_path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, _same_seconds)
    final, before = tmp_path / "lf_01.mp4", tmp_path / "lf_01_with_montage.mp4"
    gate = lv.paired_picture_gate(final, 1800.0, lambda t: (before, t))
    assert gate.passed, gate.message
    assert {n for n, _, _ in calls} == {final.name, before.name}

    _fake_ffmpeg(monkeypatch, _same_seconds, keep=1.0, blur=1.0)
    soft = lv.paired_picture_gate(final, 1800.0, lambda t: (before, t))
    assert not soft.passed and "delivered sharpness" in soft.message


# ── a Short: paired with the master through the Short's own crop ────────────

START = 1234.5
CROP = "crop=1214:2160,scale=w=1080:h=1920,setsar=1"


def _short_on_master(name, second, vf):
    if name == "short_01.mp4":
        return START + second, True
    assert vf.startswith(CROP + ","), vf  # the reference has the Short's geometry
    return second, False


def test_a_short_is_judged_against_the_master_through_its_crop(tmp_path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, _short_on_master)
    short, master = tmp_path / "short_01.mp4", tmp_path / f"day_{DATE}_master.mp4"
    gate = lv.paired_picture_gate(short, 45.0, lambda t: (master, START + t), reference_vf=CROP)
    assert gate.passed, gate.message
    short_vf = {v for n, _, v in calls if n == short.name}
    assert all(not v.startswith("crop") for v in short_vf)


# ── Phase D reads the render-time verdict; it never re-measures unlike frames ──


def test_phase_d_reads_the_render_time_record(tmp_path):
    lf = tmp_path / "lf_01.mp4"
    lf.write_bytes(b"lf body")
    assert lv.recorded_picture_gate(lf, changed=False).passed

    missing = lv.recorded_picture_gate(lf, changed=True)
    assert not missing.passed and "no picture record" in missing.message

    lv.write_picture_record(lf, lv.GateResult(gate="picture_quality", passed=False, message="soft"))
    assert lv.recorded_picture_gate(lf, changed=True).message == "soft"
    lv.write_picture_record(lf, lv.GateResult(gate="picture_quality", passed=True, message="ok"))
    assert lv.recorded_picture_gate(lf, changed=True).passed

    lf.write_bytes(b"a different render")
    stale = lv.recorded_picture_gate(lf, changed=True)
    assert not stale.passed and "different file" in stale.message


def test_picture_inputs_never_sample_the_source(tmp_path, monkeypatch):
    calls = _fake_ffmpeg(monkeypatch, _on_day_timeline)
    batch, master = _day(tmp_path)
    art = SimpleNamespace(date=DATE, master_path=master, regular_lf_indices=[1])
    state = SimpleNamespace(
        source_sharpness=[400.0],
        day_artifacts=[art],
        episode_files=[str(e.concat_path) for e in batch.episodes],
        file_durations=list(EPISODES),
    )
    assert quality.picture_inputs_from_state(state)[2] == {1: True}
    assert calls == []


# ── contrast: the gate judges detail, not the grade's tone curve ────────────
# Courtmacsherry (Ireland 2025-09-19, 2026-11-20): the day's grade lowered the
# master's luma spread 8.6%, Laplacian variance scales with contrast squared,
# and the gate read 240.0 < 250.3 as softness on 25 faithful frames. Only the
# day master is graded, so only it opts in (``contrast_match=True``).


def _textured(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fine = rng.uniform(0.0, 255.0, (96, 96)).astype(np.float32)
    return cv2.GaussianBlur(fine, (0, 0), 0.7)


def _halo(frame: np.ndarray, amount: float) -> np.ndarray:
    return np.clip(frame + amount * (frame - cv2.GaussianBlur(frame, (0, 0), 1.5)), 0, 255)


def _blocks(seed: int) -> np.ndarray:
    """Hard edges with a mild halo, for the ringing checks."""
    rng = np.random.default_rng(seed)
    small = rng.uniform(60, 200, (8, 8)).astype(np.float32)
    return _halo(cv2.resize(small, (96, 96), interpolation=cv2.INTER_NEAREST), 1.0)


def _tone(frame: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(128.0 + gain * (frame - 128.0), 0, 255)


def _smear(frame: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(frame, (0, 0), sigma)


def _paired(monkeypatch, make_delivered, make_ref=_textured, seeds=range(25)):
    refs = {i: make_ref(i) for i in seeds}
    state = {"i": -1}

    def run(cmd, capture_output=True, timeout=None):
        name = Path(cmd[cmd.index("-i") + 1]).name
        if name.endswith("_master.mp4"):
            state["i"] += 1
            frame = make_delivered(refs[state["i"] % len(refs)])
        else:
            frame = refs[state["i"] % len(refs)]
        png = cv2.imencode(".png", np.rint(frame).astype(np.uint8))[1].tobytes()
        return SimpleNamespace(returncode=0, stdout=png)

    monkeypatch.setattr(quality.subprocess, "run", run)


def _gate(make_delivered, monkeypatch, *, contrast_match=True, make_ref=_textured):
    _paired(monkeypatch, make_delivered, make_ref)
    return lv.paired_picture_gate(
        Path("m_master.mp4"),
        250.0,
        lambda t: (Path("ref.mp4"), t),
        contrast_match=contrast_match,
    )


def _day_gate(tmp_path, monkeypatch, make_delivered):
    """The day master path itself: ``None`` when it passes, else the failure text."""
    _paired(monkeypatch, make_delivered)
    batch, master = _day(tmp_path)
    try:
        day_pipeline._gate_master_picture(batch, master, DATE)
    except RuntimeError as exc:
        return str(exc)
    return None


def _raw_ratio(graded) -> float:
    return quality.sharpness_score(graded(_textured(0))) / quality.sharpness_score(_textured(0))


def test_a_tone_compressed_master_with_slight_encode_loss_passes(monkeypatch):
    # The incident grade: variance x0.84 (gain ~0.92) plus the slight encode loss.
    graded = lambda f: _smear(_tone(f, 0.92), 0.36)  # noqa: E731
    # The raw ratio is below the 0.75 the old gate needed.
    assert _raw_ratio(graded) < 0.75
    gate = _gate(graded, monkeypatch)
    assert gate.passed, gate.message
    assert "reference at delivered contrast \u00d70.84" in gate.message
    assert "contrast floor" not in gate.message


def test_the_strongest_legitimate_grade_passes(monkeypatch):
    # lift 20, pull 15, sat 1.25, gamma 1.3 measured variance x0.61-0.66; gain 0.79 is x0.62.
    graded = lambda f: _tone(f, 0.79)  # noqa: E731
    assert _raw_ratio(graded) < 0.75
    gate = _gate(graded, monkeypatch)
    assert gate.passed, gate.message
    assert "contrast floor" not in gate.message


def test_a_crushed_grey_master_fails_instead_of_shrinking_the_reference(monkeypatch):
    grain = np.random.default_rng(7)
    crushed = lambda f: _tone(f, 0.10) + grain.normal(0.0, 0.3, f.shape)  # noqa: E731
    gate = _gate(crushed, monkeypatch)
    assert not gate.passed and "delivered sharpness" in gate.message
    assert "reference at delivered contrast \u00d70.55" in gate.message
    assert "; reference held at the contrast floor on 25 pairs" in gate.message


def test_a_tone_compressed_master_that_is_also_smeared_still_fails(monkeypatch):
    gate = _gate(lambda f: _smear(_tone(f, 0.79), 1.2), monkeypatch)
    assert not gate.passed and "delivered sharpness" in gate.message


def test_added_contrast_cannot_hide_softening(monkeypatch):
    soft_and_punchy = lambda f: _tone(_smear(f, 0.5), 1.5)  # noqa: E731
    _paired(monkeypatch, soft_and_punchy)
    raw = quality.sample_paired_picture(
        Path("m_master.mp4"), 250.0, lambda t: (Path("r.mp4"), t), contrast_match=True
    )
    # The old raw comparison would have passed this pair...
    assert statistics.median(raw[0]) >= 0.75 * statistics.median(
        [r / k for r, k in zip(raw[2], raw[4])]
    )
    # ...the contrast-matched gate does not, and no upper clamp softens it.
    assert max(raw[4]) > 1.0
    gate = _gate(soft_and_punchy, monkeypatch)
    assert not gate.passed and "delivered sharpness" in gate.message


def test_flat_frames_do_not_divide_by_zero(monkeypatch):
    def run(cmd, capture_output=True, timeout=None):
        png = cv2.imencode(".png", np.full((96, 96), 90, np.uint8))[1].tobytes()
        return SimpleNamespace(returncode=0, stdout=png)

    monkeypatch.setattr(quality.subprocess, "run", run)
    out = quality.sample_paired_picture(
        Path("m.mp4"), 100.0, lambda t: (Path("r.mp4"), t), contrast_match=True
    )
    assert out[4] and set(out[4]) == {1.0}
    assert all(math.isfinite(v) for v in out[2])


def test_a_tiny_frame_measures_variance_at_full_resolution():
    tiny = np.arange(36, dtype=np.uint8).reshape(6, 6) * 4
    expected = float(quality._bounded_luma(tiny, quality.SHARPEN_DETAIL_MAX_DIM).var())
    assert quality.luma_variance(tiny) == pytest.approx(expected)


# ── the strict edge of the gate, on the day master path ─────────────────────


@pytest.mark.parametrize("sigma", [0.41, 0.6])
def test_a_fine_texture_smear_at_the_fail_onset_fails_the_day_master(tmp_path, monkeypatch, sigma):
    # sigma 0.41 scores ~0.73 against the 0.75 line. Measured at full resolution the smear would
    # shave the variance the factor divides by and pass (~0.83); a 0.30 tolerance would pass it too.
    failure = _day_gate(tmp_path, monkeypatch, lambda f: _smear(f, sigma))
    assert failure and "delivered sharpness" in failure


def test_a_fine_texture_smear_just_inside_the_line_passes_the_day_master(tmp_path, monkeypatch):
    # sigma 0.39 scores ~0.79: inside the 0.75 line, outside a 0.20 tolerance.
    assert _day_gate(tmp_path, monkeypatch, lambda f: _smear(f, 0.39)) is None


# ── LF and Short are not graded: they never adjust ──────────────────────────


def test_lf_and_short_gates_do_not_adjust_the_reference(monkeypatch):
    graded = lambda f: _tone(f, 0.6)  # noqa: E731
    plain = _gate(graded, monkeypatch, contrast_match=False)
    assert not plain.passed and "delivered sharpness" in plain.message
    assert "contrast" not in plain.message
    _paired(monkeypatch, graded)
    out = quality.sample_paired_picture(Path("m_master.mp4"), 250.0, lambda t: (Path("r.mp4"), t))
    assert set(out[4]) == {1.0}
    # the reference frames are the unadjusted ones: sharpness equal to the raw score
    expected = [quality.sharpness_score(np.rint(_textured(i))) for i in range(25)]
    assert out[2] == pytest.approx(expected)


def test_only_the_day_master_opts_in(tmp_path, monkeypatch):
    seen = {}

    def spy(video, duration, reference_at, **kwargs):
        seen.update(kwargs)
        return lv.GateResult(gate=lv.PICTURE_GATE, passed=True, message="ok")

    monkeypatch.setattr(lv, "paired_picture_gate", spy)
    batch, master = _day(tmp_path)
    day_pipeline._gate_master_picture(batch, master, DATE)
    assert seen == {"contrast_match": True}


def test_a_contrast_lowering_grade_does_not_hide_added_halos(monkeypatch):
    # Ringing scales with contrast (linearly), so the reference's ringing moves by
    # sqrt of the variance ratio; left alone it would excuse the master's halos.
    graded = lambda f: _tone(_halo(f, 0.2), 0.8)  # noqa: E731
    ring_max = max(
        quality.RINGING_MAX,
        quality.ringing_score(_blocks(0)) * (1 + quality.RINGING_SOURCE_MARGIN),
    )
    assert quality.ringing_score(graded(_blocks(0))) <= ring_max  # unadjusted, it would pass
    gate = _gate(graded, monkeypatch, make_ref=_blocks)
    assert not gate.passed and "ringing" in gate.message
    assert "delivered sharpness" not in gate.message
    # The same master with no extra halos still passes on its own ringing.
    clean = _gate(lambda f: _tone(f, 0.8), monkeypatch, make_ref=_blocks)
    assert clean.passed, clean.message
