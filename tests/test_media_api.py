"""RunSpec / Deliverables typed API around the day render."""

from __future__ import annotations

import dataclasses
import inspect
import json
from pathlib import Path

import numpy as np

from gomoveshift.pipeline import media_api
from gomoveshift.pipeline.day_pipeline import DayArtifacts, run_day_pipeline
from gomoveshift.pipeline.media_api import Deliverables, Measurements, RunSpec


class _Batch:
    date = "2026-05-01"


def _spec(tmp_path: Path) -> RunSpec:
    return RunSpec(
        day_batch=_Batch(),
        day_scores=np.zeros(3),
        output_dir=tmp_path,
        location="Somewhere, OR",
        lf_plans=[],
    )


def test_render_day_returns_deliverables_with_measurements(tmp_path, monkeypatch):
    day_dir = tmp_path / "day_2026-05-01"
    day_dir.mkdir()
    sidecar = day_dir / "ost_audit_2026-05-01.json"
    sidecar.write_text(json.dumps({}))
    art = DayArtifacts(date="2026-05-01", master_path=day_dir / "master.mp4")
    monkeypatch.setattr(media_api, "run_day_pipeline", lambda **kw: art)

    out = media_api.render_day(_spec(tmp_path))

    assert isinstance(out, Deliverables)
    assert isinstance(out, DayArtifacts)
    assert out.date == "2026-05-01"
    assert out.sidecars == {"ost_audit": sidecar}
    assert isinstance(out.measurements, Measurements)
    assert out.measurements.sharpen_amount == {}
    assert out.measurements.source_picture_scores == {}
    assert out.measurements.upscale_provenance == {}


def test_render_day_none_stays_none(tmp_path, monkeypatch):
    monkeypatch.setattr(media_api, "run_day_pipeline", lambda **kw: None)
    assert media_api.render_day(_spec(tmp_path)) is None


def test_run_all_days_yields_deliverables():
    from gomoveshift.pipeline import day_pipeline

    # Drive the per-day body through the public seam: run_all_days must go
    # through render_day, never call run_day_pipeline directly.
    src = inspect.getsource(day_pipeline.run_all_days)
    assert "render_day(" in src
    assert "run_day_pipeline(" not in src


def test_run_spec_mirrors_run_day_pipeline_signature():
    params = list(inspect.signature(run_day_pipeline).parameters)
    assert [f.name for f in dataclasses.fields(RunSpec)] == params
