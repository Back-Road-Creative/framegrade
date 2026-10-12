"""Every CLI subcommand at least builds and answers ``--help``.

A subcommand whose parser is built at import time but never exercised can
die on a typo, a bad ``type=``, a missing default, or a broken import, and
nothing notices until an operator runs it. ``--help`` forces argparse to
construct every parser node (including nested ``photos``/``metrics``/
``comments`` subcommands) without running any handler.

Discovery walks ``parser._actions`` for ``argparse._SubParsersAction``
instances and reads their ``.choices`` dict, recursing into any subcommand
that itself owns nested subparsers (``photos``, ``metrics``, ``comments``).
``parser._subparsers`` (the private attribute named in early drafts of this
plan) is NOT the right handle here -- it is the group container, not the
action; ``.choices`` on the ``_SubParsersAction`` found via ``_actions`` is
what actually gives the name -> sub-parser mapping.
"""

import argparse

import pytest

from gomoveshift.cli.parser import _build_parser


def _discover_subcommands() -> list[tuple[str, ...]]:
    """Recursively collect every subcommand path as a tuple of tokens."""
    paths: list[tuple[str, ...]] = []

    def walk(parser: argparse.ArgumentParser, prefix: tuple[str, ...]) -> None:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    path = prefix + (name,)
                    paths.append(path)
                    walk(sub, path)

    walk(_build_parser(), ())
    return paths


SUBCOMMAND_PATHS = _discover_subcommands()
SUBCOMMAND_IDS = [" ".join(path) for path in SUBCOMMAND_PATHS]


def test_discovery_found_subcommands():
    """Sanity check: discovery itself isn't silently returning nothing."""
    assert len(SUBCOMMAND_PATHS) > 30


@pytest.mark.parametrize("path", SUBCOMMAND_PATHS, ids=SUBCOMMAND_IDS)
def test_subcommand_help_builds_and_exits_zero(path, capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([*path, "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert out.strip() != ""
