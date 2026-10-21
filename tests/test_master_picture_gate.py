"""KP-GATE-MOVE: the picture gate runs on the day master straight after its encode.

The Kingsport 2026-03-04 drive learned of a picture problem only at Phase D's
final gate, after the LFs and shorts were cut and burned. The master is what a
sharpen or denoise step touched, so it is judged there, against the day's own
sources, before any downstream work. ffmpeg is never run: the sampler is patched.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.pipeline import day_pipeline, quality
from test_day_pipeline import TestRunDayPipeline as _StageMocks  # aliased: not a test class here
from test_day_pipeline import _make_day_batch, _make_lf_plan, _make_scores, _make_seg

DATE = "2026-02-20"


def _batch(tmp_path, durations=(300.0, 100.0)):
    batch = _make_day_batch(DATE, len(durations))
    for ep, d in zip(batch.episodes, durations):
        ep.concat_path = tmp_path / f"ep_{ep.index:02d}.mp4"
        ep.concat_path.write_bytes(b"x")
        ep.total_duration = d
    return batch


def _master(tmp_path, amount):
    day_dir = tmp_path / f"day_{DATE}"
    day_dir.mkdir(parents=True, exist_ok=True)
    master = day_dir / f"day_{DATE}_master.mp4"
    master.write_bytes(b"m")
    (day_dir / f"day_{DATE}_color_cache.json").write_text(
        json.dumps({"corrections": [{"sharpen_amount": amount, "denoise_strength": 0.0}]})
    )
    return master


def _sampler(monkeypatch, *, source, master):
    """Source episodes return ``source``, the master returns ``master`` (sharp, ring)."""
    calls = []

    def fake(path, duration, n=5):
        calls.append((Path(path).name, n))
        sharp, ring = master if Path(path).name.endswith("_master.mp4") else source
        return [sharp] * n, [ring] * n

    monkeypatch.setattr(quality, "sample_delivered_picture", fake)
    return calls


def test_soft_master_fails_straight_after_encode(tmp_path, monkeypatch):
    _sampler(monkeypatch, source=(1450.0, 0.015), master=(600.0, 0.015))
    with pytest.raises(RuntimeError, match=f"Day {DATE}.*delivered sharpness 600.0"):
        day_pipeline._gate_master_picture(_batch(tmp_path), _master(tmp_path, 0.3), DATE)


def test_master_ringing_above_its_source_fails(tmp_path, monkeypatch):
    _sampler(monkeypatch, source=(1450.0, 0.015), master=(1400.0, 0.03))
    with pytest.raises(RuntimeError, match="ringing 0.0300"):
        day_pipeline._gate_master_picture(_batch(tmp_path), _master(tmp_path, 0.3), DATE)


def test_faithful_master_passes_kingsport_numbers(tmp_path, monkeypatch):
    calls = _sampler(monkeypatch, source=(1450.1, 0.0152), master=(1289.6, 0.0151))
    day_pipeline._gate_master_picture(_batch(tmp_path), _master(tmp_path, 0.011), DATE)
    names = [c[0] for c in calls]
    assert names[-1] == f"day_{DATE}_master.mp4" and "ep_01.mp4" in names


def test_unchanged_picture_samples_nothing(tmp_path, monkeypatch):
    calls = _sampler(monkeypatch, source=(1.0, 1.0), master=(1.0, 1.0))
    day_pipeline._gate_master_picture(_batch(tmp_path), _master(tmp_path, 0.0), DATE)
    assert calls == []


def test_unmeasurable_source_defers_to_phase_d(tmp_path, monkeypatch):
    calls = _sampler(monkeypatch, source=(1.0, 1.0), master=(1.0, 1.0))
    batch = _batch(tmp_path)
    batch.episodes[0].concat_path = tmp_path / "gone.mp4"
    day_pipeline._gate_master_picture(batch, _master(tmp_path, 0.3), DATE)
    assert calls == []


def test_run_day_pipeline_gates_before_any_lf_is_cut(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        day_pipeline,
        "_gate_master_picture",
        lambda batch, master, date: (
            seen.append((master.name, date)) or (_ for _ in ()).throw(RuntimeError("picture gate"))
        ),
    )
    _, mocks = _StageMocks()._patch_all_stages(tmp_path)
    with (
        mocks["encode_day_master"],
        patch("gomoveshift.pipeline.day_pipeline.extract_and_concat_segments") as cut,
        pytest.raises(RuntimeError, match="picture gate"),
    ):
        day_pipeline.run_day_pipeline(
            day_batch=_make_day_batch(DATE, 3, per_ep_duration=1200.0),
            day_scores=_make_scores(3600, score=0.6),
            output_dir=tmp_path / "out",
            location="Test Location, SC",
            config=None,
            lf_plans=[_make_lf_plan(1, [_make_seg(0, 100, 200)], [])],
        )
    assert seen == [(f"day_{DATE}_master.mp4", DATE)]
    cut.assert_not_called()


def test_no_colour_records_is_quiet(tmp_path, monkeypatch, caplog):
    calls = _sampler(monkeypatch, source=(1.0, 1.0), master=(1.0, 1.0))
    master = tmp_path / f"day_{DATE}_master.mp4"
    master.write_bytes(b"m")
    day_pipeline._gate_master_picture(_batch(tmp_path), master, DATE)
    assert calls == [] and not [r for r in caplog.records if r.levelname == "WARNING"]
