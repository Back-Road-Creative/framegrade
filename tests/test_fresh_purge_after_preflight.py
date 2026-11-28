"""``--fresh`` must not clear the prior checkpoint until pre-flight passes.

A run that pre-flight refuses (for example for staging disk space) renders
nothing, so it must leave the prior run's ``drive_plan.json``,
``pipeline_state.json`` and ``*.confirmed`` markers untouched. A run whose
pre-flight passes still purges them before any phase reads them.
"""

from __future__ import annotations

import json
from pathlib import Path

from gomoveshift.pipeline import drive_orchestrator as mod
from gomoveshift.models import DrivePipelineResult

_SENTINEL = {"sentinel": "prior-run"}
_NAMES = ("drive_plan.json", "pipeline_state.json", "x.confirmed")


def _seed(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name in _NAMES:
        (d / name).write_text(json.dumps(_SENTINEL))


def _run(tmp_path, monkeypatch, preflight):
    from gomoveshift.config import Config

    src = tmp_path / "source"
    src.mkdir()
    out = tmp_path / "out"
    stages = tmp_path / "stages"
    _seed(stages)
    monkeypatch.setattr(mod, "acquire_gpu_lock", lambda *a, **k: (None, None))
    monkeypatch.setattr(
        mod.DrivePipelineOrchestrator,
        "_resolve_paths",
        lambda self, s, o: (o / "proj", stages),
    )
    monkeypatch.setattr(mod, "run_preflight_and_prompt", preflight)
    orch = mod.DrivePipelineOrchestrator(config=Config())
    try:
        orch.run(source_dir=src, output_base=out, region="usa", fresh=True, force=True)
    except Exception:  # noqa: BLE001 - a later phase may fail; only the files matter
        pass
    return stages


def test_refused_preflight_keeps_prior_checkpoint(tmp_path, monkeypatch):
    def refuse(**kw):
        return DrivePipelineResult(
            success=False,
            source_dir=kw["source_dir"],
            output_dir=kw["project_dir"],
            drive_plan=None,
            regular_lf_paths=[],
            shorts_per_lf={},
            stages_completed=[],
            errors=["RENDER BLOCKED: insufficient cache space"],
        )

    stages = _run(tmp_path, monkeypatch, refuse)
    for name in _NAMES:
        assert json.loads((stages / name).read_text()) == _SENTINEL, name


def test_passing_preflight_still_purges_before_phases(tmp_path, monkeypatch):
    seen = {}

    def ok(**kw):
        seen["at_preflight"] = {n: (kw["stages_dir"] / n).exists() for n in _NAMES}
        return None

    def stop(self, state_file):
        raise RuntimeError("stop after pre-flight")

    monkeypatch.setattr(mod.DrivePipelineOrchestrator, "_select_phase_manifest", stop)
    stages = _run(tmp_path, monkeypatch, ok)
    assert all(seen["at_preflight"].values()), "purge ran before pre-flight"
    assert not (stages / "drive_plan.json").exists()
    assert not (stages / "x.confirmed").exists()
