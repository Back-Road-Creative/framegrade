"""``cmd_drive`` tmux re-exec appends ``--force``.

When a drive is launched outside tmux, the handler re-execs its own
``sys.argv`` inside a DETACHED tmux pane. That pane is a tty, so the
orchestrator's preflight ``Proceed? [y/N]`` would block INVISIBLY there —
no operator is watching the detached pane. The re-exec must append
``--force`` so the detached run skips the prompt.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

from gomoveshift.cli.handlers import drive as drive_handler


def _run_reexec(monkeypatch, argv):
    """Drive cmd_drive down the tmux re-exec branch and capture the argv it
    hands to ``launch_in_tmux``."""
    captured = {}

    def fake_launch(command_args, session_name="gms-pipeline"):
        captured["argv"] = list(command_args)
        captured["session"] = session_name
        return 0

    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.setattr(drive_handler, "is_in_tmux", lambda: False)
    monkeypatch.setattr(drive_handler, "tmux_available", lambda: True)
    monkeypatch.setattr(drive_handler, "launch_in_tmux", fake_launch)
    # cmd_drive now refuses when neither
    # --publish nor --lf-day is given, before the re-exec branch this test
    # exercises — give it a harmless default publish day.
    rc = drive_handler.cmd_drive(SimpleNamespace(no_tmux=False, publish="mon"), MagicMock())
    return rc, captured


def test_tmux_reexec_appends_force(monkeypatch):
    rc, captured = _run_reexec(monkeypatch, ["gomoveshift", "drive", "/src", "--publish", "tue"])
    assert rc == 0
    assert "--force" in captured["argv"]
    # Re-exec targets the drive session name (unchanged behavior).
    assert captured["session"] == "gms-drive"


def test_tmux_reexec_does_not_duplicate_force(monkeypatch):
    rc, captured = _run_reexec(monkeypatch, ["gomoveshift", "drive", "/src", "--force"])
    assert rc == 0
    assert captured["argv"].count("--force") == 1
