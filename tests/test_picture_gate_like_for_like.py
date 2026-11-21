"""The picture gate compares each delivered frame with the frame it was made from.

Mizen (Ireland 2025-09-17, 2026-11-16): the day master's gate sampled the master
at 25 evenly spaced seconds but each source episode at its own spacing, so the
two medians came from different scenes. Sharpness varies by scene from 90 to
2614 on that drive, and a faithful master (335.8 against 405.8 on the same 25
frames, inside the 25% tolerance) failed at 335.8 < 448.1 (source 597.5).

Here ``ffmpeg`` is replaced by a fake that renders a noise frame whose texture
depends on the scene at that point of the day's timeline, so the real sampler
runs and only the frame choice decides the verdict. Every name is invented.
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
# and the gate read 240.0 < 250.3 as softness on 25 faithful frames.


def _textured(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fine = rng.uniform(0.0, 255.0, (96, 96)).astype(np.float32)
    return cv2.GaussianBlur(fine, (0, 0), 0.7)


def _tone(frame: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(128.0 + gain * (frame - 128.0), 0, 255)


def _smear(frame: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(frame, (0, 0), sigma)


def _paired(monkeypatch, make_delivered, seeds=range(25)):
    refs = {i: _textured(i) for i in seeds}
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


def _gate(make_delivered, monkeypatch):
    _paired(monkeypatch, make_delivered)
    return lv.paired_picture_gate(Path("m_master.mp4"), 250.0, lambda t: (Path("ref.mp4"), t))


def test_a_tone_compressed_master_with_no_detail_loss_passes(monkeypatch):
    graded = lambda f: _tone(f, 0.6)  # noqa: E731
    # The raw ratio is below the 0.75 the old gate needed.
    raw = quality.sharpness_score(graded(_textured(0))) / quality.sharpness_score(_textured(0))
    assert raw < 0.75
    gate = _gate(graded, monkeypatch)
    assert gate.passed, gate.message
    assert "reference at delivered contrast ×0.36" in gate.message


def test_a_tone_compressed_master_that_is_also_smeared_still_fails(monkeypatch):
    gate = _gate(lambda f: _smear(_tone(f, 0.6), 1.2), monkeypatch)
    assert not gate.passed and "delivered sharpness" in gate.message


def test_added_contrast_cannot_hide_softening(monkeypatch):
    soft_and_punchy = lambda f: _tone(_smear(f, 0.5), 1.5)  # noqa: E731
    _paired(monkeypatch, soft_and_punchy)
    raw = quality.sample_paired_picture(Path("m_master.mp4"), 250.0, lambda t: (Path("r.mp4"), t))
    # The old raw comparison would have passed this pair...
    assert statistics.median(raw[0]) >= 0.75 * statistics.median(
        [r / k for r, k in zip(raw[2], raw[4])]
    )
    # ...the contrast-matched gate does not.
    gate = lv.paired_picture_gate(Path("m_master.mp4"), 250.0, lambda t: (Path("ref.mp4"), t))
    assert not gate.passed and "delivered sharpness" in gate.message


def test_flat_frames_do_not_divide_by_zero(monkeypatch):
    def run(cmd, capture_output=True, timeout=None):
        png = cv2.imencode(".png", np.full((96, 96), 90, np.uint8))[1].tobytes()
        return SimpleNamespace(returncode=0, stdout=png)

    monkeypatch.setattr(quality.subprocess, "run", run)
    out = quality.sample_paired_picture(Path("m.mp4"), 100.0, lambda t: (Path("r.mp4"), t))
    assert out[4] and set(out[4]) == {1.0}
    assert all(math.isfinite(v) for v in out[2])
