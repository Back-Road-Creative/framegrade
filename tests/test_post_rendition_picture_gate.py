"""The post-rendition picture gate fails a delivery that softened or rang.

ffmpeg is never run: the delivered-frame sampler is monkeypatched.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from gomoveshift.pipeline import quality
from gomoveshift.pipeline.drive_phases.phase_d_gates import _run_final_validation_gate
from gomoveshift.pipeline.validation.longform_validator import (
    PICTURE_SKIPPED_MESSAGE,
    LongformValidator,
)
from gomoveshift.pipeline.validation.shorts_validator import ShortsValidator

LOW, HIGH = 60.0, 190.0


def _step() -> np.ndarray:
    row = np.where(np.arange(64) < 32, LOW, HIGH)
    return np.tile(row, (64, 1))


def _haloed() -> np.ndarray:
    """The same edge with a Gibbs-style +/-24 ripple decaying on both sides."""
    frame = _step()
    for dx, k in ((0, 24), (1, -10), (2, 6)):
        frame[:, 32 + dx] += k
        frame[:, 31 - dx] -= k
    return frame


def _validator() -> LongformValidator:
    ffprobe = MagicMock()
    ffprobe.probe_duration.return_value = 100.0
    return LongformValidator(ffprobe)


def _sampler(monkeypatch, sharp, ring):
    calls = []

    def fake(path, duration, n=5):
        calls.append((path, n))
        return sharp, ring

    monkeypatch.setattr(quality, "sample_delivered_picture", fake)
    return calls


def test_ringing_score_orders_clean_and_haloed_edges():
    clean, halo = quality.ringing_score(_step()), quality.ringing_score(_haloed())
    assert clean == pytest.approx(0.0, abs=1e-3)
    assert quality.ringing_score(np.full((64, 64), 128.0)) == 0.0
    assert clean < quality.RINGING_MAX < halo


def test_picture_changed_reads_future_markers():
    assert not quality.picture_changed({"sharpen_amount": 0.0, "denoise_strength": 0.0})
    assert not quality.picture_changed({})
    assert quality.picture_changed({"sharpen_amount": 0.3})
    assert quality.picture_changed(SimpleNamespace(denoise_strength=1.5))
    assert quality.picture_changed({"stabilise": True})
    assert quality.picture_changed(SimpleNamespace(upscale="2x"))
    assert quality.picture_changed({"some_future_step": True}) is False


def test_rendition_that_softens_or_rings_hard_fails(monkeypatch):
    _sampler(monkeypatch, [100.0, 110.0, 105.0], [0.0, 0.0, 0.0])
    soft = _validator().check_picture_quality(Path("f.mp4"), source_sharpness=400.0, changed=True)
    assert not soft.passed
    assert "105.0" in soft.message and "400.0" in soft.message

    _sampler(monkeypatch, [390.0, 400.0, 410.0], [0.05, 0.05, 0.05])
    rang = _validator().check_picture_quality(Path("f.mp4"), source_sharpness=400.0, changed=True)
    assert not rang.passed
    assert "0.0500" in rang.message

    _sampler(monkeypatch, [390.0, 400.0, 410.0], [0.0, 0.0, 0.0])
    ok = _validator().check_picture_quality(Path("f.mp4"), source_sharpness=400.0, changed=True)
    assert ok.passed


def test_gate_skipped_when_no_rendition_step(monkeypatch):
    calls = _sampler(monkeypatch, [1.0], [1.0])
    v = _validator()
    for kwargs in (
        {"source_sharpness": 400.0, "changed": False},
        {"source_sharpness": None, "changed": True},
    ):
        gate = v.check_picture_quality(Path("f.mp4"), **kwargs)
        assert gate.gate == "picture_quality" and gate.passed
        assert gate.message == PICTURE_SKIPPED_MESSAGE
    assert calls == []


def test_shorts_validator_records_picture_gate_hard(monkeypatch):
    _sampler(monkeypatch, [50.0], [0.0])
    sv = ShortsValidator(MagicMock())
    sv._validate_single_short = lambda p: []
    skipped = sv.validate_short_files([Path("s.mp4")])
    assert [g.gate for g in skipped.gates] == ["picture_quality"] and skipped.passed
    bad = sv.validate_short_files([Path("s.mp4")], source_sharpness=400.0, picture_changed=True)
    assert not bad.passed


def _run_gate(tmp_path, picture_check_fn, inputs):
    big = tmp_path / "lf.mp4"
    big.write_bytes(b"x" * (1024 * 1024 + 1))
    ok = SimpleNamespace(passed=True, gates=[], summary=lambda: "ok")
    return _run_final_validation_gate(
        validator_cls=lambda **kw: SimpleNamespace(validate=lambda **k: ok),
        config=SimpleNamespace(
            drive=SimpleNamespace(skip_audio=True),
            output=SimpleNamespace(audio_max_lufs=-18.0, audio_target_tp=-1.5),
        ),
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1, total_duration=1800.0)]),
        regular_lf_paths=[big],
        metadata_per_lf={1: {}},
        perceptual_check_fn=lambda p, d: None,
        picture_inputs=inputs,
        picture_check_fn=picture_check_fn,
    )


def test_phase_d_aborts_on_failed_picture_quality(tmp_path):
    seen = []

    def check(path, duration, source_sharpness, changed, source_ringing=None):
        seen.append((source_sharpness, source_ringing, changed))
        return SimpleNamespace(gate="picture_quality", passed=False, message="delivered 10 < 300")

    with pytest.raises(RuntimeError, match="delivered 10 < 300"):
        _run_gate(tmp_path, check, {1: (400.0, 0.012, True)})
    assert seen == [(400.0, 0.012, True)]


def test_phase_d_passes_when_no_inputs(tmp_path):
    seen = []

    def check(path, duration, source_sharpness, changed, source_ringing=None):
        seen.append((source_sharpness, source_ringing, changed))
        return SimpleNamespace(gate="picture_quality", passed=True, message="skipped")

    _run_gate(tmp_path, check, None)
    assert seen == [(None, None, False)]


# ── production plumbing ───────────────────────────────────────────────────


def _state(tmp_path, amount, sharpness=(400.0, None, 600.0)):
    import json

    master = tmp_path / "day_2026-01-01_master.mp4"
    (tmp_path / "day_2026-01-01_color_cache.json").write_text(
        json.dumps({"corrections": [{"sharpen_amount": amount, "denoise_strength": 0.0}]})
    )
    art = SimpleNamespace(date="2026-01-01", master_path=master, regular_lf_indices=[1])
    return SimpleNamespace(source_sharpness=list(sharpness), day_artifacts=[art])


def test_picture_inputs_from_state_reads_cache_and_median(tmp_path):
    sharp, ring, changed = quality.picture_inputs_from_state(_state(tmp_path, 0.4))
    assert sharp == 500.0 and ring is None and changed == {1: True}
    assert quality.picture_inputs_from_state(_state(tmp_path, 0.0))[2] == {1: False}
    assert quality.picture_changed_for_day(tmp_path / "absent.json") is False


def test_final_gate_gets_real_inputs_and_fails_on_soft_delivery(tmp_path, monkeypatch):
    from gomoveshift.pipeline.drive_phases.phase_d_gates import _lf_picture_inputs

    inputs = _lf_picture_inputs(_state(tmp_path, 0.4))
    assert inputs == {1: (500.0, None, True)}
    _sampler(monkeypatch, [50.0, 60.0], [0.0, 0.0])
    with pytest.raises(RuntimeError, match="delivered sharpness"):
        _run_gate(tmp_path, None, inputs)


def test_soft_short_fails_through_the_facade(monkeypatch):
    from gomoveshift.pipeline.pipeline_validator import PipelineValidator

    _sampler(monkeypatch, [50.0], [0.0])
    pv = PipelineValidator.__new__(PipelineValidator)
    sv = ShortsValidator(MagicMock())
    sv._validate_single_short = lambda p: []
    pv._shorts = sv
    result = pv.validate_short_files([Path("s.mp4")], source_sharpness=400.0, picture_changed=True)
    assert not result.passed


# ── judged against the source, measured the same way (Kingsport 2026-10-13) ──
# Real numbers: 25 frames of the three raw GoPro sources gave sharpness 1450.1 and
# ringing 0.0152; 25 frames of the delivered LF gave 1289.6 and 0.0151. The old
# gate compared a 5-frame source figure (2056.2) and an absolute ringing max that
# the untouched source itself exceeds, and failed a faithful encode.


def test_ringing_is_judged_against_the_source(monkeypatch):
    _sampler(monkeypatch, [1289.6] * 3, [0.0151] * 3)
    v = _validator()
    kingsport = v.check_picture_quality(
        Path("f.mp4"), source_sharpness=1450.1, source_ringing=0.0152, changed=True
    )
    assert kingsport.passed, kingsport.message

    added = v.check_picture_quality(
        Path("f.mp4"), source_sharpness=1450.1, source_ringing=0.005, changed=True
    )
    assert not added.passed and "ringing 0.0151" in added.message

    unmeasured = v.check_picture_quality(Path("f.mp4"), source_sharpness=1450.1, changed=True)
    assert not unmeasured.passed and "ringing 0.0151" in unmeasured.message


def test_halo_on_a_ringing_source_still_fails(monkeypatch):
    limit = 0.0152 * (1 + quality.RINGING_SOURCE_MARGIN)
    _sampler(monkeypatch, [1400.0] * 3, [limit + 0.002] * 3)
    gate = _validator().check_picture_quality(
        Path("f.mp4"), source_sharpness=1450.1, source_ringing=0.0152, changed=True
    )
    assert not gate.passed and "ringing" in gate.message


def test_delivered_side_samples_the_gate_frame_count(monkeypatch):
    calls = _sampler(monkeypatch, [1400.0], [0.0])
    _validator().check_picture_quality(Path("f.mp4"), source_sharpness=1450.1, changed=True)
    assert calls == [(Path("f.mp4"), quality.PICTURE_GATE_SAMPLES)]
    assert quality.PICTURE_GATE_SAMPLES >= 25


def _episode_state(tmp_path, amount, durations=(300.0, 100.0)):
    state = _state(tmp_path, amount, sharpness=(2056.2,))
    files = []
    for i, _ in enumerate(durations):
        f = tmp_path / f"episode_{i:02d}.mp4"
        f.write_bytes(b"x")
        files.append(str(f))
    state.episode_files = files
    state.file_durations = list(durations)
    return state


def test_source_is_remeasured_from_the_episodes_with_the_gate_sampler(tmp_path, monkeypatch):
    seen = []

    def fake(path, duration, n=5):
        seen.append((Path(path).name, duration, n))
        return [1450.1] * n, [0.0152] * n

    monkeypatch.setattr(quality, "sample_delivered_picture", fake)
    sharp, ring, changed = quality.picture_inputs_from_state(_episode_state(tmp_path, 0.011))
    assert (sharp, ring, changed) == (1450.1, 0.0152, {1: True})
    # Duration-weighted share of the gate's frame count, every episode sampled.
    assert [s[0] for s in seen] == ["episode_00.mp4", "episode_01.mp4"]
    assert sum(s[2] for s in seen) >= quality.PICTURE_GATE_SAMPLES
    assert seen[0][2] > seen[1][2]


def test_unchanged_picture_never_samples_the_source(tmp_path, monkeypatch):
    calls = _sampler(monkeypatch, [1.0], [1.0])
    sharp, ring, changed = quality.picture_inputs_from_state(_episode_state(tmp_path, 0.0))
    assert calls == [] and changed == {1: False}
    assert (sharp, ring) == (2056.2, None)


def test_missing_episode_falls_back_to_the_recorded_sharpness(tmp_path, monkeypatch):
    calls = _sampler(monkeypatch, [1.0], [1.0])
    state = _episode_state(tmp_path, 0.4)
    state.episode_files = [str(tmp_path / "gone.mp4")]
    sharp, ring, _ = quality.picture_inputs_from_state(state)
    assert calls == [] and (sharp, ring) == (2056.2, None)
