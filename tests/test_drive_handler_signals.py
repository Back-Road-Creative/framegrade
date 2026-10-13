"""Tests for cmd_drive graceful shutdown & resource wiring.

The drive pipeline must install SIGTERM/SIGHUP handlers so that a reboot
or tmux-kill-session triggers an orderly SystemExit (letting orchestrator
state checkpoints land) rather than a hard SIGKILL that leaves
half-written MP4s and an un-checkpointed pipeline. The registration lives in ``DrivePipelineOrchestrator.run()`` so any entry
point that drives the orchestrator inherits the contract — these tests
exercise the orchestrator-side handler directly.
"""

from __future__ import annotations

import signal
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.cli.handlers import drive as drive_handler


def _args(**overrides):
    defaults = dict(
        directory="/tmp/does-not-matter",
        region="usa",
        # cmd_drive now refuses when
        # neither --publish nor --lf-day is given.
        publish="tue",
        location=None,
        start_date=None,
        shorts_per_day=None,
        shorts_target=None,
        shorts_gap=None,
        per_day=False,
        day_first=False,
        preview=False,
        no_review=True,
        no_audio=False,
        skip_validation=False,
        phase=None,
        fresh=False,
        skip_upload=True,
        skip_csv=True,
        youtube_upload=False,
        no_tmux=True,
        output_dir="/tmp/out",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def in_tmux(monkeypatch):
    monkeypatch.setattr(drive_handler, "is_in_tmux", lambda: True)


class TestSignalHandlerInstalled:
    def test_sigterm_and_sighup_installed_by_orchestrator_run(self):
        """SIGTERM/SIGHUP handler registration lives on
        ``DrivePipelineOrchestrator.run()``, not on ``cmd_drive``. Use the
        nonexistent-source-dir early-return path so ``run()`` exits cleanly
        after the signal-handler registration block but before any phase
        work.
        """
        from pathlib import Path

        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        config = MagicMock()
        orch = DrivePipelineOrchestrator(config)

        with patch("gomoveshift.pipeline.drive_orchestrator.signal.signal") as mock_signal:
            result = orch.run(
                source_dir=Path("/tmp/does-not-exist-on-purpose"),
                output_base=Path("/tmp/out"),
            )

        installed = {call.args[0] for call in mock_signal.call_args_list}
        assert signal.SIGTERM in installed
        assert signal.SIGHUP in installed
        # The early-return path reports the missing source, but the
        # signal-registration block still fired — that's the contract.
        assert result.success is False

    def test_handler_registration_tolerates_non_main_thread(self):
        """signal.signal() raises ValueError off the main thread. The
        defensive try/except in ``run()`` must swallow that so unit tests
        and other non-main-thread invocations don't crash.
        """
        from pathlib import Path

        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        config = MagicMock()
        orch = DrivePipelineOrchestrator(config)

        with patch(
            "gomoveshift.pipeline.drive_orchestrator.signal.signal",
            side_effect=ValueError("signal only works in main thread"),
        ):
            # Should not raise; the early-return path completes.
            result = orch.run(
                source_dir=Path("/tmp/does-not-exist-on-purpose"),
                output_base=Path("/tmp/out"),
            )
        assert result.success is False


class TestGracefulShutdown:
    def test_sigterm_systemexit_returns_143(self, in_tmux):
        args = _args()
        config = MagicMock()
        orch = MagicMock()
        orch.return_value.run.side_effect = SystemExit(143)

        with (
            patch.object(drive_handler, "configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator", orch),
        ):
            rc = drive_handler.cmd_drive(args, config)

        assert rc == 143

    def test_ctrl_c_returns_130(self, in_tmux):
        args = _args()
        config = MagicMock()
        orch = MagicMock()
        orch.return_value.run.side_effect = KeyboardInterrupt()

        with (
            patch.object(drive_handler, "configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator", orch),
        ):
            rc = drive_handler.cmd_drive(args, config)

        assert rc == 130


class TestCorruptStateHandling:
    """When pipeline_state.json is unreadable, load_drive_state raises
    CorruptStateError. cmd_drive must catch it, print a loud banner
    to stderr, and return a non-zero exit code. An operator tailing
    stdout must NOT miss the failure and must be pointed at --fresh
    as the explicit recovery path.
    """

    def test_cli_corrupt_state_exits_loudly(self, in_tmux, capsys):
        from gomoveshift.pipeline.exceptions import CorruptStateError

        args = _args()
        config = MagicMock()
        orch = MagicMock()
        orch.return_value.run.side_effect = CorruptStateError(
            "/tmp/stages/pipeline_state.json", "Expecting value: line 1 column 1"
        )

        with (
            patch.object(drive_handler, "configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator", orch),
        ):
            rc = drive_handler.cmd_drive(args, config)

        # Non-zero exit: operator's shell, CI, or wrapper script sees a
        # failure. 2 is distinct from PipelineValidationError's 1 and
        # SIGTERM's 143.
        assert rc == 2

        captured = capsys.readouterr()
        # Banner must be on stderr (operator tailing stdout gets nothing;
        # stderr goes to the shell by default and is hard to miss).
        assert "CORRUPT" in captured.err
        assert "pipeline_state.json" in captured.err
        # Explicit recovery instruction — don't leave the operator guessing.
        assert "--fresh" in captured.err
        # Path reported so operator can inspect it.
        assert "/tmp/stages/pipeline_state.json" in captured.err

    def test_cli_corrupt_state_does_not_silently_succeed(self, in_tmux, capsys):
        """Negative assertion: we must never see success messaging when
        the state file is corrupt. (This is the whole point of the fix —
        the previous silent fresh-start could be mistaken for success.)
        """
        from gomoveshift.pipeline.exceptions import CorruptStateError

        args = _args()
        config = MagicMock()
        orch = MagicMock()
        orch.return_value.run.side_effect = CorruptStateError("/tmp/stages/pipeline_state.json")

        with (
            patch.object(drive_handler, "configure_logging"),
            patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator", orch),
        ):
            rc = drive_handler.cmd_drive(args, config)

        captured = capsys.readouterr()
        assert rc != 0
        assert "Drive pipeline complete" not in captured.out
        assert "Drive pipeline complete" not in captured.err
