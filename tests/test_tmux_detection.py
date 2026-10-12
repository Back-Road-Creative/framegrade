"""``is_in_tmux`` must accept TMUX, TMUX_PANE, or TERM_PROGRAM=tmux.

Wrappers that unset TMUX (e.g. ssh-into-tmux forwarders) leave TMUX_PANE
or TERM_PROGRAM intact. The legacy TMUX-only check produced a false-negative
that re-launched a nested tmux session.
"""

import pytest


@pytest.fixture
def clean_env(monkeypatch):
    for var in ("TMUX", "TMUX_PANE", "TERM_PROGRAM"):
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


def test_is_in_tmux_via_tmux_env(clean_env):
    clean_env.setenv("TMUX", "/tmp/tmux-1000/default,1234,0")
    from gomoveshift.cli.tmux import is_in_tmux

    assert is_in_tmux() is True


def test_is_in_tmux_via_tmux_pane(clean_env):
    clean_env.setenv("TMUX_PANE", "%0")
    from gomoveshift.cli.tmux import is_in_tmux

    assert is_in_tmux() is True


def test_is_in_tmux_via_term_program(clean_env):
    clean_env.setenv("TERM_PROGRAM", "tmux")
    from gomoveshift.cli.tmux import is_in_tmux

    assert is_in_tmux() is True


def test_is_not_in_tmux_when_none_set(clean_env):
    from gomoveshift.cli.tmux import is_in_tmux

    assert is_in_tmux() is False
