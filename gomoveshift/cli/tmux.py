"""tmux session management helpers for long-running pipeline commands."""

import os
import shlex
import subprocess
import sys
import time


def is_in_tmux() -> bool:
    """Check if currently running inside a tmux session.

    Accept ``TMUX``, ``TMUX_PANE``, or ``TERM_PROGRAM=tmux``. Wrappers
    that unset ``TMUX`` (some ssh-forwarders) leave the other two intact;
    the legacy ``TMUX``-only check produced a false-negative that re-launched
    a nested session.
    """
    if "TMUX" in os.environ or "TMUX_PANE" in os.environ:
        return True
    return os.environ.get("TERM_PROGRAM") == "tmux"


def process_is_supervised() -> bool:
    """Whether this process already survives its parent exiting.

    Two ways to be safe without tmux, and both are checked rather than
    trusted:

    * **Session leader** — ``setsid``/daemonised, so no parent shell owns it.
    * **``GMS_SUPERVISED``** — the caller asserts an external supervisor
      (systemd, a CI runner, the test harness). Non-empty values only; an
      empty string is not an assertion.
    """
    try:
        if os.getsid(0) == os.getpid():
            return True
    except OSError:  # pragma: no cover - platform without getsid
        pass
    return bool(os.environ.get("GMS_SUPERVISED", "").strip())


def require_supervision(what: str = "pipeline") -> "int | None":
    """Gate ``--no-tmux``: ``None`` to proceed, or an exit code to return.

    ``--no-tmux`` once printed a warning and ran anyway, so a long drive
    encode launched from a non-interactive parent could die with its parent
    unnoticed. A warning arrives too late for irreversible work, so the flag
    means "I am already supervised" and the claim is verified.
    """
    if process_is_supervised():
        return None
    print(
        f"ERROR: --no-tmux would leave this {what} unsupervised — it dies "
        "when its parent shell or ssh session exits.\n"
        "  Run it under tmux (drop --no-tmux and it re-execs detached), or\n"
        "  detach it yourself (setsid/systemd-run), or\n"
        "  set GMS_SUPERVISED=1 to assert an external supervisor (CI, tests).",
        file=sys.stderr,
    )
    return 2


def tmux_available() -> bool:
    """Check if tmux is available on the system."""
    try:
        result = subprocess.run(["tmux", "-V"], capture_output=True, timeout=2)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def launch_in_tmux(command_args: list[str], session_name: str = "gms-pipeline") -> int:
    """
    Launch command in a tmux session.

    Args:
        command_args: Command and arguments to run
        session_name: Name for the tmux session

    Returns:
        Exit code from tmux command (0 = success, verified session running)
    """
    print(f"Long-running pipeline — launching in tmux session '{session_name}'")
    print(f"To attach: tmux attach -t {session_name}")
    print()

    # Never kill an existing session -- an in-flight run (e.g. a multi-hour
    # drive) must not be terminated by launching a second one under the same
    # name. tmux's own new-session atomically refuses a duplicate name; when it
    # does, surface a clear "attach instead" hint rather than clobbering the run.
    shell_command = " ".join(shlex.quote(arg) for arg in command_args)
    cmd = ["tmux", "new-session", "-d", "-s", session_name, "bash", "-c", shell_command]

    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            already = subprocess.run(
                ["tmux", "has-session", "-t", session_name], capture_output=True, check=False
            )
            if already.returncode == 0:
                print(
                    f"\u2717 A session named '{session_name}' is already running \u2014 refusing "
                    f"to kill it; attach with: tmux attach -t {session_name}",
                    file=sys.stderr,
                )
            else:
                print("\u2717 Failed to create tmux session", file=sys.stderr)
            return result.returncode

        # ENFORCE: Verify session actually exists and is running
        time.sleep(0.5)  # Give it a moment to start
        verify = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True)
        if verify.returncode != 0:
            print(f"\u2717 FAILED: tmux session '{session_name}' did not persist!", file=sys.stderr)
            print("  This indicates a bug in command execution.", file=sys.stderr)
            print(f"  Command was: {shell_command[:100]}...", file=sys.stderr)
            return 1

        print(f"\u2713 Pipeline started in tmux session '{session_name}'")
        print(f"  Attach with: tmux attach -t {session_name}")
        return 0
    except Exception as e:
        print(f"Failed to launch tmux: {e}", file=sys.stderr)
        return 1
