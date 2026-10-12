"""The single-video stage and the ingest batch pick default publish dates through
seo_manager's occupancy, never their own weekday arithmetic: a ledger-booked date is skipped."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gomoveshift.config import Config
from gomoveshift.pipeline import seo_slots
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.context import PipelineContext, PipelinePaths
from gomoveshift.pipeline.ingest import Episode, IngestPlan
from gomoveshift.pipeline.stages.schedule import ScheduleGenerationStage


def _config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.seo.storage_path = str(tmp_path / "seo")
    Path(cfg.seo.storage_path).mkdir(exist_ok=True)
    return cfg


def _book(cfg: Config, *dates: str) -> None:
    entries = [
        ContentEntry(
            date=d,
            time="16:00:00",
            content_type="longform",
            title="Booked",
            project_name="rival",
            status="scheduled",
            source="pipeline",
            region=None,
        )
        for d in dates
    ]
    ContentLedger(entries=entries).save(Path(cfg.seo.storage_path) / ".content_ledger.json")


def _first_free(cfg: Config, day: str) -> str:
    return seo_slots.default_publish_slot(cfg, day).strftime("%Y-%m-%d")


def test_stage_default_date_skips_a_ledger_booked_date(tmp_path, monkeypatch):
    from gomoveshift.pipeline import orchestrator as _orch

    cfg = _config(tmp_path)
    free = _first_free(cfg, "sat")
    _book(cfg, free)
    seen = {}

    class _Sched:
        def generate_schedule(self, **kw):
            seen.update(kw)
            return SimpleNamespace(slots=[], save=lambda p: None)

    monkeypatch.setattr(_orch, "PipelineScheduler", _Sched)
    project = tmp_path / "project"
    project.mkdir()
    paths = PipelinePaths(
        output_base=tmp_path,
        project_dir=project,
        stages_dir=tmp_path / "stages",
        state_file=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
    )
    ctx = PipelineContext(
        project_name="p",
        source_video=tmp_path / "s.mp4",
        source_hash="h",
        config=cfg,
        location="A to B",
        publish_day="sat",
        paths=paths,
    )
    assert ScheduleGenerationStage().run(ctx).success is True
    assert seen["publish_date"] != free
    assert seen["publish_date"] > free
    assert seen["publish_day"] == "sat"


def test_ingest_batch_default_dates_skip_ledger_booked_dates(tmp_path):
    from gomoveshift.pipeline.ingest import run_batch_pipeline

    cfg = _config(tmp_path)
    first = _first_free(cfg, "sat")
    _book(cfg, first)
    episodes = []
    for i in (1, 2):
        concat = tmp_path / f"ep{i}.mp4"
        concat.write_bytes(b"\x00")
        ep = Episode(index=i, recordings=[], date="2026-02-20")
        ep.concat_path = concat
        ep.location = "SC"
        episodes.append(ep)
    plan = IngestPlan(source_dir=tmp_path, episodes=episodes)
    ok = MagicMock(success=True, output_dir="/out", final_video=None, errors=[])
    with (
        patch("gomoveshift.pipeline.orchestrator.PipelineOrchestrator") as orch,
        patch("gomoveshift.config.get_config", return_value=cfg),
    ):
        orch.return_value.run.return_value = ok
        results = run_batch_pipeline(plan, publish_day="sat", output_base=tmp_path / "out")
    dates = [r["publish_date"] for r in results]
    assert first not in dates
    assert dates[0] > first and dates[1] > dates[0]
