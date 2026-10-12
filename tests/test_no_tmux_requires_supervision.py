"""``--no-tmux`` asserts supervision, and the assertion is verified.

A long encode launched with ``--no-tmux`` from a non-interactive parent can die
with that parent, unnoticed, leaving a huge raw concat and no master on disk. A
printed warning would be a late gate on irreversible work, so the handlers
enforce the requirement instead: tmux missing is a hard error, tmux present
means a detached re-exec, and ``--no-tmux`` means "I am already supervised"
and that claim is verified.

The flag stays, because CI, containers and the test suite legitimately want a
foreground process whose exit code they read — but it now means "I am already
supervised", and a process that is neither a session leader nor explicitly
declared supervised is refused instead of warned.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.cli import tmux as tmux_mod
from gomoveshift.cli.handlers import drive as drive_handler


@pytest.fixture(autouse=True)
def _not_in_tmux(monkeypatch):
    """Every case here is the "operator is not inside tmux" situation."""
    monkeypatch.setattr(drive_handler, "is_in_tmux", lambda: False)
    monkeypatch.delenv("GMS_SUPERVISED", raising=False)


class TestProcessIsSupervised:
    def test_session_leader_counts_as_supervised(self, monkeypatch):
        """setsid/daemonised: the process already outlives its parent."""
        monkeypatch.setattr(os, "getsid", lambda _pid: os.getpid())
        assert tmux_mod.process_is_supervised() is True

    def test_env_declaration_counts_as_supervised(self, monkeypatch):
        monkeypatch.setattr(os, "getsid", lambda _pid: os.getpid() + 1)
        monkeypatch.setenv("GMS_SUPERVISED", "1")
        assert tmux_mod.process_is_supervised() is True

    def test_plain_child_of_a_shell_is_not_supervised(self, monkeypatch):
        monkeypatch.setattr(os, "getsid", lambda _pid: os.getpid() + 1)
        assert tmux_mod.process_is_supervised() is False

    def test_empty_env_value_does_not_count(self, monkeypatch):
        """An empty GMS_SUPERVISED must not read as an assertion."""
        monkeypatch.setattr(os, "getsid", lambda _pid: os.getpid() + 1)
        monkeypatch.setenv("GMS_SUPERVISED", "")
        assert tmux_mod.process_is_supervised() is False


class TestDriveRefusesUnsupervisedNoTmux:
    def test_unsupervised_no_tmux_is_refused(self, monkeypatch, capsys):
        monkeypatch.setattr(os, "getsid", lambda _pid: os.getpid() + 1)
        rc = drive_handler.cmd_drive(SimpleNamespace(no_tmux=True, publish="mon"), MagicMock())
        assert rc == 2, "an unsupervised --no-tmux run must not start"
        err = capsys.readouterr().err
        assert "GMS_SUPERVISED" in err, "the error must name the supported escape hatch"

    def test_supervised_no_tmux_proceeds(self, monkeypatch):
        """Declared supervision is honoured — the flag still works for CI.

        Proof of passage is an ``AttributeError`` from the real work that
        follows the gate: this ``SimpleNamespace`` carries only ``no_tmux``
        plus a schedule flag (needed so the schedule-flag
        guard doesn't fire first), so reaching any later field means the
        supervision gate let it by. A ``return 2`` would end the call cleanly
        instead.
        """
        monkeypatch.setattr(os, "getsid", lambda _pid: os.getpid() + 1)
        monkeypatch.setenv("GMS_SUPERVISED", "1")
        with pytest.raises(AttributeError):
            drive_handler.cmd_drive(SimpleNamespace(no_tmux=True, publish="mon"), MagicMock())

    def test_session_leader_no_tmux_proceeds(self, monkeypatch):
        monkeypatch.setattr(os, "getsid", lambda _pid: os.getpid())
        with pytest.raises(AttributeError):
            drive_handler.cmd_drive(SimpleNamespace(no_tmux=True, publish="mon"), MagicMock())
