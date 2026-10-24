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


# --- UI-6: --publish is validated at parse time ---------------------------------------------


def _publish_argv(command: str, value: str, tmp_path) -> list[str]:
    if command == "pipeline":
        return ["pipeline", str(tmp_path / "v.mp4"), "--publish", value]
    return [command, str(tmp_path), "--publish", value]


@pytest.mark.parametrize("command", ["pipeline", "batch", "drive"])
def test_publish_rejects_an_unknown_day_at_parse_time(command, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc_info:
        _build_parser().parse_args(_publish_argv(command, "funday", tmp_path))
    assert exc_info.value.code == 2
    assert "mon/tue/wed/thu/fri/sat/sun" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["pipeline", "batch", "drive"])
@pytest.mark.parametrize("raw,expected", [("mon", "mon"), ("Fri", "fri"), ("SUN", "sun")])
def test_publish_accepts_a_weekday_in_any_case(command, raw, expected, tmp_path):
    args = _build_parser().parse_args(_publish_argv(command, raw, tmp_path))
    assert args.publish == expected


# --- UI-7: the top-level help is discoverable -----------------------------------------------


def _top_level_help() -> str:
    return _build_parser().format_help()


def test_top_level_help_has_an_epilog_naming_every_command():
    """The epilog groups the commands; a command added without a group fails here."""
    parser = _build_parser()
    assert parser.epilog
    epilog_words = set(parser.epilog.replace(",", " ").split())
    commands = {path[0] for path in SUBCOMMAND_PATHS}
    assert commands - epilog_words == set()


def test_top_level_usage_does_not_dump_every_command():
    usage = _build_parser().format_usage()
    assert "<command>" in usage
    assert "{analyze," not in usage


def test_server_help_says_what_the_server_is():
    assert "review" in _server_help().lower()


def _server_help() -> str:
    for action in _build_parser()._actions:
        if isinstance(action, argparse._SubParsersAction):
            return next(a.help for a in action._choices_actions if a.dest == "server")
    raise AssertionError("no subparsers")


# --- PROD-4: --version reads the package version ---------------------------------------------


def test_version_flag_prints_the_package_version(capsys):
    import gomoveshift

    with pytest.raises(SystemExit) as exc_info:
        _build_parser().parse_args(["--version"])
    assert exc_info.value.code == 0
    assert capsys.readouterr().out.startswith(f"gomoveshift {gomoveshift.__version__}")


def test_version_flag_appends_the_commit_when_known(capsys, monkeypatch):
    monkeypatch.setattr("gomoveshift.health.store.product_sha", lambda *a, **k: "abc1234deadbeef")
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--version"])
    assert "abc1234" in capsys.readouterr().out
