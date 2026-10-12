"""``gomoveshift route-story-film --profile <yaml>`` CLI adapter."""

import argparse

import pytest

from gomoveshift.cli.handlers import route_story_film as route_story_film_module
from gomoveshift.cli.handlers.route_story_film import (
    build_route_story_film_argv,
    cmd_route_story_film,
)
from gomoveshift.pipeline.route_story.profile import ProfileError


def _args(profile, dry_run=False):
    return argparse.Namespace(profile=str(profile), dry_run=dry_run)


def _stub_mapping(monkeypatch, *, load_returns=None, argv_returns=None):
    """Stub load_profile/film_parser/build_film_argv and capture film.main() calls."""
    monkeypatch.setattr(route_story_film_module, "load_profile", lambda path: load_returns)
    monkeypatch.setattr(route_story_film_module, "film_parser", lambda: object())
    monkeypatch.setattr(
        route_story_film_module, "build_film_argv", lambda parser, profile: argv_returns
    )
    called = []
    monkeypatch.setattr(route_story_film_module.film, "main", lambda argv: called.append(argv) or 0)
    return called


def test_build_route_story_film_argv_shape():
    argv = build_route_story_film_argv("route_story_profiles/trip.yaml")

    assert argv == [
        argv[0],
        "-m",
        "gomoveshift",
        "route-story-film",
        "--profile",
        "route_story_profiles/trip.yaml",
    ]


def test_build_route_story_film_argv_dry_run_appends_flag():
    argv = build_route_story_film_argv("route_story_profiles/trip.yaml", dry_run=True)

    assert argv[-1] == "--dry-run"


def test_dry_run_prints_argv_exits_zero_and_never_calls_film_main(tmp_path, monkeypatch, capsys):
    profile_path = tmp_path / "trip.yaml"
    called = _stub_mapping(
        monkeypatch, load_returns={"output": "film.mp4"}, argv_returns=["--output", "film.mp4"]
    )

    rc = cmd_route_story_film(_args(profile_path, dry_run=True), config=None)

    assert rc == 0
    assert called == []
    assert "--output film.mp4" in capsys.readouterr().out


def test_unverified_value_refuses_before_launch(tmp_path, monkeypatch, capsys):
    profile_path = tmp_path / "trip.yaml"
    called = _stub_mapping(
        monkeypatch,
        load_returns={"photos-dir": "x"},
        argv_returns=["--photos-dir", "UNVERIFIED-example/x"],
    )

    rc = cmd_route_story_film(_args(profile_path), config=None)

    assert rc == 2
    assert called == []
    err = capsys.readouterr().err
    assert "UNVERIFIED-example/x" in err


def test_profile_error_refuses_with_exit_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        route_story_film_module,
        "load_profile",
        lambda path: (_ for _ in ()).throw(ProfileError("boom")),
    )

    rc = cmd_route_story_film(_args(tmp_path / "trip.yaml"), config=None)

    assert rc == 1
    assert "boom" in capsys.readouterr().err


def test_real_run_calls_film_main_with_the_built_argv(tmp_path, monkeypatch):
    profile_path = tmp_path / "trip.yaml"
    called = _stub_mapping(
        monkeypatch, load_returns={"output": "film.mp4"}, argv_returns=["--output", "film.mp4"]
    )

    rc = cmd_route_story_film(_args(profile_path), config=None)

    assert rc == 0
    assert called == [["--output", "film.mp4"]]
