"""Single-video publish ordering: validation gates the irreversible upload.

Two invariants:

1. ``PublishAndArchive`` runs ``validation_gate`` BEFORE ``drive_upload`` — the
   audio/content gate must be able to block the irreversible Drive/YouTube
   upload, not merely the local archive copy.
2. ``ValidationGateStage`` failure is ``fatal=True`` so ``SubPipeline.run``
   aborts the sequence — a failed validation must NOT warn-continue into
   ``drive_upload`` / ``d_drive_copy``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

from gomoveshift.pipeline.context import PipelineContext, PipelinePaths
from gomoveshift.pipeline.stage_protocol import BaseStage, StageResult
from gomoveshift.pipeline.stages.validation_gate import ValidationGateStage
from gomoveshift.pipeline.sub_pipelines import PublishAndArchive, SubPipeline


def test_publish_runs_validation_before_upload():
    names = [s.name for s in PublishAndArchive().stages]
    assert names.index("validation_gate") < names.index("drive_upload"), (
        "validation_gate must precede drive_upload so it can block the "
        "irreversible upload, not just the local archive copy"
    )


# ── SubPipeline abort behaviour with a fatal validation ─────────────────


@dataclass
class _FakeCtx:
    skip_stages: set = field(default_factory=set)
    stages_completed: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    video_properties: Optional[object] = None


class _Spy(BaseStage):
    def __init__(self, name, result=None):
        self.name = name
        self._result = result or StageResult(success=True)
        self.ran = False

    def run(self, ctx):  # type: ignore[override]
        self.ran = True
        return self._result


def test_fatal_validation_aborts_before_upload_and_copy():
    """A fatal validation_gate failure stops the sub-pipeline — drive_upload
    and d_drive_copy never run (no irreversible upload, no archive copy)."""
    val = _Spy(
        "validation_gate",
        StageResult(success=False, error="audio too loud", fatal=True),
    )
    upload = _Spy("drive_upload")
    copy = _Spy("d_drive_copy")

    sp = SubPipeline()
    sp.name = "publish_test"
    sp.stages = [val, upload, copy]

    result = sp.run(_FakeCtx(), lambda ctx: None)

    assert result.success is False and result.fatal is True
    assert val.ran is True
    assert upload.ran is False, "drive_upload ran after a fatal validation failure"
    assert copy.ran is False, "d_drive_copy ran after a fatal validation failure"


# ── ValidationGateStage returns fatal=True on failure ───────────────────


def _make_ctx(tmp_path: Path) -> PipelineContext:
    project_dir = tmp_path / "project"
    project_dir.mkdir(parents=True, exist_ok=True)
    final = project_dir / "FINAL.mp4"
    final.write_bytes(b"x" * (2 * 1024 * 1024))  # > 1 MiB so should_run() is True
    paths = PipelinePaths(
        output_base=tmp_path / "out",
        project_dir=project_dir,
        stages_dir=tmp_path / "stages",
        state_file=tmp_path / "state.json",
        staging_dir=tmp_path / "staging",
    )
    paths.output_base.mkdir(parents=True, exist_ok=True)
    return PipelineContext(
        project_name="p",
        source_video=tmp_path / "src.mp4",
        source_hash="h",
        config=MagicMock(),
        location="A to B",
        publish_day="wed",
        paths=paths,
        final_video=final,
    )


def test_validation_gate_stage_failure_is_fatal(tmp_path, monkeypatch):
    from gomoveshift.pipeline import orchestrator as _orch

    failed = MagicMock(passed=False, summary=lambda: "audio_loudness FAIL")

    class _Validator:
        def __init__(self, **_kw):
            pass

        def validate(self, **_kw):
            return failed

    monkeypatch.setattr(_orch, "PipelineValidator", _Validator)

    ctx = _make_ctx(tmp_path)
    assert ValidationGateStage().should_run(ctx) is True
    result = ValidationGateStage().run(ctx)
    assert result.success is False
    assert result.fatal is True, (
        "ValidationGateStage failure must be fatal so SubPipeline.run aborts "
        "before the irreversible drive_upload"
    )
