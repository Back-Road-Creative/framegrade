"""Single-video DriveUploadStage ships the PLANNED shorts, not a glob.

The stage used to glob ``short_*.mp4`` and ship whatever sat on disk with only
a corruption check. It now reconciles the on-disk ``*.mp4`` against the plan the
run produced (``ctx.shorts_data`` segments' ``output_path``): an unplanned /
stale short on disk hard-fails rather than glob-shipping. With no plan present
(legacy path) it falls back to the glob so those callers are unchanged.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from gomoveshift.pipeline.context import PipelineContext, PipelinePaths
from gomoveshift.pipeline.stages.upload import DriveUploadStage


def _make_ctx(tmp_path, shorts_data=None):
    project_dir = tmp_path / "project"
    (project_dir / "shorts").mkdir(parents=True, exist_ok=True)
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
        shorts_data=shorts_data,
    )


def _planned(project_dir, names):
    return {"segments": [{"output_path": str(project_dir / "shorts" / n)} for n in names]}


def test_stale_unplanned_short_hard_fails(tmp_path, monkeypatch):
    """A stale short_02 on disk that the plan never produced must NOT ship."""
    from gomoveshift.pipeline import orchestrator as _orch

    monkeypatch.setattr(
        _orch, "validate_video", lambda p, require_audio=False: MagicMock(valid=True, reason="")
    )
    uploader_inst = MagicMock()
    monkeypatch.setattr(_orch, "PipelineUploader", MagicMock(return_value=uploader_inst))

    ctx = _make_ctx(tmp_path, shorts_data=_planned(tmp_path / "project", ["short_01.mp4"]))
    shorts = ctx.paths.project_dir / "shorts"
    (shorts / "short_01.mp4").write_bytes(b"x")
    (shorts / "short_02.mp4").write_bytes(b"stale")  # unplanned leftover

    result = DriveUploadStage().run(ctx)

    assert result.success is False
    assert "unplanned" in result.error.lower()
    uploader_inst.upload_shorts.assert_not_called()


def test_plan_matches_disk_uploads(tmp_path, monkeypatch):
    """Disk exactly matches the plan → upload proceeds."""
    from gomoveshift.pipeline import orchestrator as _orch

    monkeypatch.setattr(
        _orch, "validate_video", lambda p, require_audio=False: MagicMock(valid=True, reason="")
    )
    uploader_inst = MagicMock()
    uploader_inst.upload_shorts.return_value = MagicMock(files=[MagicMock()])
    monkeypatch.setattr(_orch, "PipelineUploader", MagicMock(return_value=uploader_inst))

    ctx = _make_ctx(
        tmp_path, shorts_data=_planned(tmp_path / "project", ["short_01.mp4", "short_02.mp4"])
    )
    shorts = ctx.paths.project_dir / "shorts"
    (shorts / "short_01.mp4").write_bytes(b"x")
    (shorts / "short_02.mp4").write_bytes(b"y")

    result = DriveUploadStage().run(ctx)

    assert result.success is True
    uploader_inst.upload_shorts.assert_called_once()


def test_no_plan_falls_back_to_glob(tmp_path, monkeypatch):
    """No ctx.shorts_data (legacy) → glob fallback still uploads on-disk shorts."""
    from gomoveshift.pipeline import orchestrator as _orch

    monkeypatch.setattr(
        _orch, "validate_video", lambda p, require_audio=False: MagicMock(valid=True, reason="")
    )
    uploader_inst = MagicMock()
    uploader_inst.upload_shorts.return_value = MagicMock(files=[MagicMock()])
    monkeypatch.setattr(_orch, "PipelineUploader", MagicMock(return_value=uploader_inst))

    ctx = _make_ctx(tmp_path, shorts_data=None)
    (ctx.paths.project_dir / "shorts" / "short_01.mp4").write_bytes(b"x")

    result = DriveUploadStage().run(ctx)

    assert result.success is True
    uploader_inst.upload_shorts.assert_called_once()


def test_skip_stops_warns_upload_ungated(tmp_path, monkeypatch, caplog):
    """--skip-stops (stop_removal in skip_stages) makes the residual risk
    EXPLICIT with a loud warning — shorts derive from raw video — but still ships
    (manual/local surface; the sensitive dead-spot avoidance is the real guard).

    The warning is the WHOLE protection: nothing reads a marker off the context,
    so the stage must not set an undeclared ``upload_ungated_reason`` attribute
    that silently reaches no consumer."""
    import logging

    from gomoveshift.pipeline import orchestrator as _orch

    monkeypatch.setattr(
        _orch, "validate_video", lambda p, require_audio=False: MagicMock(valid=True, reason="")
    )
    uploader_inst = MagicMock()
    uploader_inst.upload_shorts.return_value = MagicMock(files=[MagicMock()])
    monkeypatch.setattr(_orch, "PipelineUploader", MagicMock(return_value=uploader_inst))

    ctx = _make_ctx(tmp_path, shorts_data=None)
    ctx.skip_stages = {PipelineContext.STAGE_STOP_REMOVAL}
    (ctx.paths.project_dir / "shorts" / "short_01.mp4").write_bytes(b"x")

    with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.stages.upload"):
        result = DriveUploadStage().run(ctx)

    assert result.success is True
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("UNGATED (--skip-stops)" in r.getMessage() for r in warnings)
    assert not hasattr(ctx, "upload_ungated_reason")
    uploader_inst.upload_shorts.assert_called_once()
