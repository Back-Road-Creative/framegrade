"""Regression: ``--fresh`` must purge a prior run's stage artifacts.

Phase BC loads
``drive_plan.json`` by path from ``stages_dir`` (and the legacy
``output_base/_stages`` mirror), so a stale plan left from a DIFFERENT
source silently reassembles LFs from the wrong drive's segments.
``_try_resume`` early-returns on ``fresh`` — without an explicit purge the
orphan files survive. Without it, a ``--fresh`` run reuses the prior
plan until the orphan ``drive_plan.json`` is deleted by hand.
"""

from __future__ import annotations

import json
from pathlib import Path


def _seed(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "drive_plan.json").write_text(
        json.dumps({"source": "/OTHER/drive", "regular_lfs": [1, 2]})
    )
    (d / "pipeline_state.json").write_text(json.dumps({"phases_completed": []}))
    return d


def _orch():
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

    return DrivePipelineOrchestrator(config=Config())


def test_fresh_purges_plan_and_state_from_both_locations(tmp_path):
    from gomoveshift.pipeline.drive_state_model import PipelineState

    output_base = tmp_path / "out"
    stages_dir = tmp_path / "stages"
    mirror = output_base / "_stages"
    # A plan from a DIFFERENT drive sits in both the stages dir and the
    # legacy output-side mirror that Phase BC / resume read from.
    _seed(stages_dir)
    _seed(mirror)

    state = PipelineState(source_dir=tmp_path / "newsrc", output_base=output_base)
    state_file = stages_dir / "pipeline_state.json"

    resumed = _orch()._try_resume(state, state_file, stages_dir, [], fresh=True)

    assert resumed == []
    for d in (stages_dir, mirror):
        assert not (d / "drive_plan.json").exists(), f"{d}: stale plan survived --fresh"
        assert not (d / "pipeline_state.json").exists(), f"{d}: stale checkpoint survived --fresh"


def test_non_fresh_resume_keeps_artifacts(tmp_path):
    """Inverse guard: a normal (non-fresh) resume must NOT purge the plan —
    the resuming run needs ``drive_plan.json`` to load."""
    from gomoveshift.pipeline.drive_state_model import PipelineState

    output_base = tmp_path / "out"
    stages_dir = tmp_path / "stages"
    _seed(stages_dir)

    state = PipelineState(source_dir=tmp_path / "src", output_base=output_base)
    state_file = stages_dir / "pipeline_state.json"

    _orch()._try_resume(state, state_file, stages_dir, [], fresh=False)

    assert (stages_dir / "drive_plan.json").exists(), "non-fresh resume wrongly purged the plan"
