"""Regression test: ScheduleGenerationStage must not report success=True
after swallowing a schedule-write exception.

The stage wrapped its whole body in a bare ``try/except Exception`` that
logged a warning and fell through to ``return StageResult(success=True)``.
A truncated/unwritten ``release_schedule.yaml`` was therefore reported as a
successful stage, and the Metricool CSV generator downstream had no way to
know slots were missing. Fix: mirror the ``AnalyticsSyncStage`` /
``ShortsStage`` convention — on a genuine write failure, return
``StageResult(success=False, error=..., fatal=False)`` so the orchestrator
logs a visible non-fatal failure and does not mark the stage complete.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gomoveshift.pipeline.context import PipelineContext, PipelinePaths
from gomoveshift.pipeline.stages.schedule import ScheduleGenerationStage


def _make_ctx(tmp_path: Path, **overrides) -> PipelineContext:
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    paths = PipelinePaths(
        output_base=tmp_path / "out",
        project_dir=project_dir,
        stages_dir=tmp_path / "stages",
        state_file=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
    )
    paths.output_base.mkdir(parents=True, exist_ok=True)
    from gomoveshift.config import Config

    cfg = Config()  # the default publish date asks seo_manager over the SEO-storage ledger
    cfg.seo.storage_path = str(tmp_path / "seo")
    Path(cfg.seo.storage_path).mkdir(exist_ok=True)
    defaults = dict(
        project_name="testproj",
        source_video=tmp_path / "src.mp4",
        source_hash="abc123",
        config=cfg,
        location="A to B via Road",
        publish_day="wed",
        paths=paths,
    )
    defaults.update(overrides)
    return PipelineContext(**defaults)


def test_schedule_write_failure_returns_non_fatal_failure(tmp_path, monkeypatch):
    """A ReleaseSchedule.save() failure must surface as success=False,
    fatal=False — never a silently-swallowed success=True.
    """
    from gomoveshift.pipeline import orchestrator as _orch
    from gomoveshift.pipeline.scheduler import ReleaseSchedule

    def _boom_save(self, path):
        raise OSError("simulated disk-full mid-write")

    monkeypatch.setattr(ReleaseSchedule, "save", _boom_save)
    monkeypatch.setattr(_orch, "PipelineScheduler", _orch.PipelineScheduler)

    stage = ScheduleGenerationStage()
    ctx = _make_ctx(tmp_path)

    result = stage.run(ctx)

    assert result.success is False, (
        "Stage swallowed the schedule-write OSError and returned success=True. "
        "stages/schedule.py must propagate a genuine write failure as "
        "StageResult(success=False)."
    )
    assert "simulated disk-full mid-write" in result.error
    assert result.fatal is False, (
        "A missing/truncated shorts schedule should not abort the whole "
        "pipeline run (it's recoverable/regenerable) — but it must not be "
        "silently reported as success either."
    )
