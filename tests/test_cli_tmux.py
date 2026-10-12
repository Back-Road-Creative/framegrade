"""Tests for ``gomoveshift/cli/tmux.py`` tmux session helpers.

Covers the env-var detection (``is_in_tmux``), the ``tmux -V`` availability
probe (``tmux_available``), and the launch/verify helper (``launch_in_tmux``)
including its post-launch ``has-session`` verification gate. ``subprocess`` and
``os.environ`` are mocked so the suite never spawns a real tmux server.
"""

import subprocess
from unittest.mock import MagicMock, patch

from gomoveshift.cli import tmux


class TestIsInTmux:
    """``is_in_tmux`` reads TMUX / TMUX_PANE / TERM_PROGRAM."""

    def test_tmux_env_var_present(self, monkeypatch):
        monkeypatch.setenv("TMUX", "/tmp/tmux-1000/default,123,0")
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert tmux.is_in_tmux() is True

    def test_tmux_pane_present_when_tmux_unset(self, monkeypatch):
        """Wrappers that unset TMUX leave TMUX_PANE — still inside tmux."""
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.setenv("TMUX_PANE", "%3")
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert tmux.is_in_tmux() is True

    def test_term_program_tmux(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "tmux")
        assert tmux.is_in_tmux() is True

    def test_no_tmux_signals(self, monkeypatch):
        monkeypatch.delenv("TMUX", raising=False)
        monkeypatch.delenv("TMUX_PANE", raising=False)
        monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
        assert tmux.is_in_tmux() is False
        # TERM_PROGRAM absent (not just non-tmux) is also False.
        monkeypatch.delenv("TERM_PROGRAM", raising=False)
        assert tmux.is_in_tmux() is False


class TestTmuxAvailable:
    """``tmux_available`` probes ``tmux -V`` and degrades gracefully."""

    def test_available_when_version_succeeds(self):
        with patch.object(tmux.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert tmux.tmux_available() is True
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            assert args[0] == ["tmux", "-V"]
            assert kwargs.get("timeout") == 2

    def test_unavailable_on_nonzero_returncode(self):
        with patch.object(tmux.subprocess, "run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            assert tmux.tmux_available() is False

    def test_unavailable_when_binary_missing(self):
        with patch.object(tmux.subprocess, "run", side_effect=FileNotFoundError):
            assert tmux.tmux_available() is False

    def test_unavailable_on_timeout(self):
        with patch.object(
            tmux.subprocess,
            "run",
            side_effect=subprocess.TimeoutExpired(cmd="tmux", timeout=2),
        ):
            assert tmux.tmux_available() is False


class TestLaunchInTmux:
    """``launch_in_tmux`` creates a detached session and verifies it persists."""

    def test_successful_launch_returns_zero(self, capsys):
        live: set[str] = set()

        def fake_run(cmd, *a, **k):
            op = cmd[1]
            if op == "has-session":
                target = cmd[cmd.index("-t") + 1]
                return MagicMock(returncode=0 if target in live else 1)
            if op == "new-session":
                live.add(cmd[cmd.index("-s") + 1])
            return MagicMock(returncode=0)

        with (
            patch.object(tmux.subprocess, "run", side_effect=fake_run) as mock_run,
            patch.object(tmux.time, "sleep"),
        ):
            rc = tmux.launch_in_tmux(["gomoveshift", "drive", "/src"], session_name="gms-test")
        assert rc == 0
        # the launcher must NEVER kill an existing session — no kill-session op.
        assert not any(c.args[0][:2] == ["tmux", "kill-session"] for c in mock_run.call_args_list)

    def test_existing_live_session_is_not_killed(self, capsys):
        """A live session of the same name is refused, never killed —
        starting a second drive must not terminate an in-flight run."""
        live = {"gms-drive"}
        calls = []

        def fake_run(cmd, *a, **k):
            calls.append(cmd)
            op = cmd[1]
            target = cmd[cmd.index("-t") + 1] if "-t" in cmd else None
            if op == "has-session":
                return MagicMock(returncode=0 if target in live else 1)
            if op == "new-session":
                name = cmd[cmd.index("-s") + 1]
                if name in live:  # tmux refuses a duplicate session name
                    return MagicMock(returncode=1)
                live.add(name)
                return MagicMock(returncode=0)
            if op == "kill-session":
                live.discard(target)
                return MagicMock(returncode=0)
            return MagicMock(returncode=0)

        with (
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
            patch.object(tmux.time, "sleep"),
        ):
            rc = tmux.launch_in_tmux(["gomoveshift", "drive"], session_name="gms-drive")

        assert rc != 0  # refused, not launched
        assert not any(c[:2] == ["tmux", "kill-session"] for c in calls)
        assert "gms-drive" in live  # the in-flight session survived
        assert "already running" in capsys.readouterr().err

    def test_new_session_quotes_args(self):
        captured = {}

        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["tmux", "new-session"]:
                captured["cmd"] = cmd
            return MagicMock(returncode=0)

        with (
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
            patch.object(tmux.time, "sleep"),
        ):
            tmux.launch_in_tmux(["gomoveshift", "a b"], session_name="s")
        # The shell command is the last element after bash -c; shlex.quote
        # wraps the arg containing a space.
        shell_command = captured["cmd"][-1]
        assert "'a b'" in shell_command

    def test_failed_new_session_returns_its_code(self):
        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["tmux", "new-session"]:
                return MagicMock(returncode=2)
            return MagicMock(returncode=0)

        with (
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
            patch.object(tmux.time, "sleep"),
        ):
            rc = tmux.launch_in_tmux(["x"], session_name="s")
        assert rc == 2

    def test_session_did_not_persist_returns_one(self, capsys):
        """new-session succeeds but has-session fails -> verification gate fires."""

        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["tmux", "has-session"]:
                return MagicMock(returncode=1)
            return MagicMock(returncode=0)

        with (
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
            patch.object(tmux.time, "sleep"),
        ):
            rc = tmux.launch_in_tmux(["x"], session_name="ghost")
        assert rc == 1
        err = capsys.readouterr().err
        assert "did not persist" in err

    def test_exception_during_launch_returns_one(self, capsys):
        def fake_run(cmd, *args, **kwargs):
            if cmd[:2] == ["tmux", "new-session"]:
                raise OSError("boom")
            return MagicMock(returncode=0)

        with (
            patch.object(tmux.subprocess, "run", side_effect=fake_run),
            patch.object(tmux.time, "sleep"),
        ):
            rc = tmux.launch_in_tmux(["x"], session_name="s")
        assert rc == 1
        err = capsys.readouterr().err
        assert "Failed to launch tmux" in err
