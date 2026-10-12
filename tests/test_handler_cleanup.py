"""Handler-level tests for ``cmd_cleanup``.

``cmd_cleanup`` lives in ``gomoveshift/cli/handlers/cleanup.py`` and dispatches
to ``PipelineCleanup``. The handler itself does no try/except, so an
underlying-pipeline exception propagates to the CLI top-level. We pin three
behaviours:

1. Happy path — dry-run lists files, returns 0.
2. Missing project — argparse-accepted name that doesn't exist on disk → exit 1.
3. ``PipelineCleanup`` raising mid-plan propagates (contract pin — a future
   refactor that adds a swallow-and-return-1 should re-evaluate this test).
4. Confirm flag triggers execute path.
5. No-projects discovery short-circuits to exit 0.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.cli.handlers import cleanup as cleanup_handler


@pytest.fixture
def fake_config() -> MagicMock:
    return MagicMock()


def _args(project=None, confirm=False, min_age=7, archive_base=None) -> SimpleNamespace:
    return SimpleNamespace(
        project=project, confirm=confirm, min_age=min_age, archive_base=archive_base
    )


def _patch_paths(tmp_path: Path):
    """Patch ``gms_output_dir`` and ``gms_staging_dir`` symbols at the import
    site (``gomoveshift.paths``) since ``cmd_cleanup`` imports them lazily."""
    output = tmp_path / "output"
    staging = tmp_path / "staging"
    output.mkdir()
    staging.mkdir()
    return (
        patch("gomoveshift.paths.gms_output_dir", return_value=output),
        patch("gomoveshift.paths.gms_staging_dir", return_value=staging),
        output,
        staging,
    )


class TestCmdCleanup:
    def test_dry_run_lists_and_returns_zero(self, tmp_path, fake_config, capsys):
        out_p, stg_p, output, _ = _patch_paths(tmp_path)
        plan = SimpleNamespace(
            files=[{"path": str(output / "x.mp4"), "size": 1024 * 1024, "reason": "old"}],
            skipped=[],
        )
        with (
            out_p,
            stg_p,
            patch("gomoveshift.pipeline.cleanup.PipelineCleanup") as mock_cleanup_cls,
        ):
            mock_cleanup = mock_cleanup_cls.return_value
            mock_cleanup.discover_projects.return_value = [output / "proj1"]
            mock_cleanup.plan_cleanup.return_value = plan
            rc = cleanup_handler.cmd_cleanup(_args(), fake_config)
        assert rc == 0
        text = capsys.readouterr().out
        assert "DRY RUN" in text
        assert "WOULD DELETE" in text

    def test_named_project_missing_returns_one(self, tmp_path, fake_config, capsys):
        out_p, stg_p, output, _ = _patch_paths(tmp_path)
        with (
            out_p,
            stg_p,
            patch("gomoveshift.pipeline.cleanup.PipelineCleanup"),
        ):
            rc = cleanup_handler.cmd_cleanup(_args(project="no-such-project"), fake_config)
        assert rc == 1
        assert "Project not found" in capsys.readouterr().err

    def test_no_projects_discovered_returns_zero(self, tmp_path, fake_config, capsys):
        out_p, stg_p, _, _ = _patch_paths(tmp_path)
        with (
            out_p,
            stg_p,
            patch("gomoveshift.pipeline.cleanup.PipelineCleanup") as mock_cleanup_cls,
        ):
            mock_cleanup = mock_cleanup_cls.return_value
            mock_cleanup.discover_projects.return_value = []
            rc = cleanup_handler.cmd_cleanup(_args(), fake_config)
        assert rc == 0
        assert "No projects found" in capsys.readouterr().out

    def test_confirm_flag_executes_plan(self, tmp_path, fake_config, capsys):
        out_p, stg_p, output, _ = _patch_paths(tmp_path)
        plan = SimpleNamespace(
            files=[{"path": str(output / "x.mp4"), "size": 2048, "reason": "stale"}],
            skipped=[],
        )
        with (
            out_p,
            stg_p,
            patch("gomoveshift.pipeline.cleanup.PipelineCleanup") as mock_cleanup_cls,
        ):
            mock_cleanup = mock_cleanup_cls.return_value
            mock_cleanup.discover_projects.return_value = [output / "proj1"]
            mock_cleanup.plan_cleanup.return_value = plan
            mock_cleanup.execute_plan.return_value = 1
            rc = cleanup_handler.cmd_cleanup(_args(confirm=True), fake_config)
        assert rc == 0
        mock_cleanup.execute_plan.assert_called_once_with(plan)
        text = capsys.readouterr().out
        assert "DELETE" in text
        assert "Deleted 1" in text

    def test_plan_raises_propagates_to_caller(self, tmp_path, fake_config):
        """``cmd_cleanup`` has no top-level try/except: a ``PipelineCleanup``
        error propagates to the CLI dispatcher. Pin this so a quiet
        ``except Exception`` insert later doesn't go unnoticed."""
        out_p, stg_p, output, _ = _patch_paths(tmp_path)
        with (
            out_p,
            stg_p,
            patch("gomoveshift.pipeline.cleanup.PipelineCleanup") as mock_cleanup_cls,
        ):
            mock_cleanup = mock_cleanup_cls.return_value
            mock_cleanup.discover_projects.return_value = [output / "proj1"]
            mock_cleanup.plan_cleanup.side_effect = RuntimeError("plan-blew-up")
            with pytest.raises(RuntimeError, match="plan-blew-up"):
                cleanup_handler.cmd_cleanup(_args(), fake_config)

    def test_no_archive_base_leaves_d_drive_base_unset(self, tmp_path, fake_config, capsys):
        """Default CLI invocation (no ``--archive-base``) must construct
        ``PipelineCleanup`` with ``d_drive_base=None`` — the D: reclamation
        feature stays off unless an operator explicitly opts in."""
        out_p, stg_p, output, _ = _patch_paths(tmp_path)
        plan = SimpleNamespace(files=[], skipped=[])
        with (
            out_p,
            stg_p,
            patch("gomoveshift.pipeline.cleanup.PipelineCleanup") as mock_cleanup_cls,
        ):
            mock_cleanup = mock_cleanup_cls.return_value
            mock_cleanup.discover_projects.return_value = [output / "proj1"]
            mock_cleanup.plan_cleanup.return_value = plan
            cleanup_handler.cmd_cleanup(_args(), fake_config)
        _, kwargs = mock_cleanup_cls.call_args
        assert kwargs.get("d_drive_base") is None

    def test_archive_base_flag_threads_into_pipeline_cleanup(self, tmp_path, fake_config, capsys):
        """``--archive-base`` supplies a distinct archive location, threaded
        through to ``PipelineCleanup(d_drive_base=...)``."""
        out_p, stg_p, output, _ = _patch_paths(tmp_path)
        archive_base = tmp_path / "archive"
        plan = SimpleNamespace(files=[], skipped=[])
        with (
            out_p,
            stg_p,
            patch("gomoveshift.pipeline.cleanup.PipelineCleanup") as mock_cleanup_cls,
        ):
            mock_cleanup = mock_cleanup_cls.return_value
            mock_cleanup.discover_projects.return_value = [output / "proj1"]
            mock_cleanup.plan_cleanup.return_value = plan
            cleanup_handler.cmd_cleanup(_args(archive_base=archive_base), fake_config)
        _, kwargs = mock_cleanup_cls.call_args
        assert kwargs.get("d_drive_base") == archive_base

    def test_empty_plan_short_circuits(self, tmp_path, fake_config, capsys):
        out_p, stg_p, output, _ = _patch_paths(tmp_path)
        plan = SimpleNamespace(files=[], skipped=[])
        with (
            out_p,
            stg_p,
            patch("gomoveshift.pipeline.cleanup.PipelineCleanup") as mock_cleanup_cls,
        ):
            mock_cleanup = mock_cleanup_cls.return_value
            mock_cleanup.discover_projects.return_value = [output / "proj1"]
            mock_cleanup.plan_cleanup.return_value = plan
            rc = cleanup_handler.cmd_cleanup(_args(), fake_config)
        assert rc == 0
        assert "Nothing to clean up" in capsys.readouterr().out
