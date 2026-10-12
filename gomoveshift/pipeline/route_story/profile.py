"""Load a route-story launch profile (a YAML mapping of film.py flag -> value)
and map it onto film.py's own argparse parser.

The mapping lets a launch happen in-process as a console job, without a
separate launcher subprocess. See docs/pipeline/16-route-story-film.md.
"""

from __future__ import annotations

import argparse
import shlex
from datetime import date
from pathlib import Path

import yaml

from . import film
from .prepare import find_prepared_bundle

#: --regional-raster/-credit/-bounds are film.py's three parallel
#: action="append" options for repeatable regional imagery; a profile that
#: lists any of them must give all three the same number of entries.
REGIONAL_TRIPLE = ("regional-raster", "regional-credit", "regional-bounds")

#: A profile value with no grounding in a delivery receipt is written with
#: this prefix; a real launch refuses it.
UNVERIFIED_PREFIX = "UNVERIFIED-"


class ProfileError(RuntimeError):
    """A clear, user-facing failure -- never a bare traceback."""


def load_profile(path: "str | Path") -> dict:
    path = Path(path)
    if not path.is_file():
        raise ProfileError(f"profile not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ProfileError(f"profile must be a YAML mapping of flag -> value: {path}")
    return data


def film_parser() -> argparse.ArgumentParser:
    """film.py's real, live parser, captured without running a render.

    film.main() builds the parser inline and calls parser.parse_args(argv)
    immediately; a temporary monkeypatch of ArgumentParser.parse_args
    captures ``self`` and raises a sentinel first, so nothing past that
    point runs.
    """
    captured: dict[str, argparse.ArgumentParser] = {}
    real_parse_args = argparse.ArgumentParser.parse_args

    class _Captured(Exception):  # noqa: N818 -- a sentinel, not an error
        pass

    def _capture(self, *_args, **_kwargs):
        captured["parser"] = self
        raise _Captured()

    argparse.ArgumentParser.parse_args = _capture
    try:
        try:
            film.main([])
        except _Captured:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse_args

    parser = captured.get("parser")
    if parser is None:
        raise ProfileError(
            "could not capture film.py's argparse parser (main() no longer "
            "calls parse_args() the way this module expects)"
        )
    return parser


def _argv_value(argv: list[str], flag: str) -> str | None:
    try:
        return argv[argv.index(flag) + 1]
    except ValueError:
        return None


def _resolve_scene_bundle(argv: list[str]) -> str:
    gpx_value = _argv_value(argv, "--gpx")
    recorded_on_value = _argv_value(argv, "--recorded-on")
    if gpx_value is None or recorded_on_value is None:
        raise ProfileError(
            "--renderer scene needs --gpx and --recorded-on in the profile to "
            "resolve a scene bundle"
        )
    try:
        recorded_on = date.fromisoformat(recorded_on_value)
    except ValueError as exc:
        raise ProfileError(f"--recorded-on {recorded_on_value!r} is not a YYYY-MM-DD date") from exc

    bundle_dir = find_prepared_bundle(Path(gpx_value), recorded_on=recorded_on)
    if bundle_dir is not None:
        return str(bundle_dir)

    prepare_cmd = [
        "gomoveshift",
        "route-story-prepare",
        gpx_value,
        "--recorded-on",
        recorded_on_value,
    ]
    raise ProfileError(
        "no prepared Route Story bundle found for this GPX/date; run first, "
        "then re-launch: " + " ".join(shlex.quote(part) for part in prepare_cmd)
    )


def build_film_argv(parser: argparse.ArgumentParser, profile: dict) -> list[str]:
    """Map every profile key to a film.py parser option, validated against
    the real parser so an unknown/misspelled key fails here -- before
    anything launches -- rather than being silently dropped."""
    dest_by_flag: dict[str, argparse.Action] = {}
    for action in parser._actions:  # noqa: SLF001 -- introspection, no public API for this
        for opt in action.option_strings:
            if opt.startswith("--"):
                dest_by_flag[opt[2:]] = action

    unknown = sorted(set(profile) - set(dest_by_flag))
    if unknown:
        raise ProfileError(f"unknown profile key(s) not in film.py's parser: {', '.join(unknown)}")

    lengths = {k: len(profile[k]) for k in REGIONAL_TRIPLE if k in profile}
    if lengths and len(set(lengths.values())) != 1:
        raise ProfileError(
            f"regional-raster/-credit/-bounds must all be the same length, got {lengths}"
        )

    argv: list[str] = []
    for key, value in profile.items():
        action = dest_by_flag[key]
        flag = f"--{key}"
        is_append = action.__class__.__name__ == "_AppendAction"
        is_store_true = action.__class__.__name__ == "_StoreTrueAction"

        if is_store_true:
            if value:
                argv.append(flag)
            continue

        if is_append:
            for item in value:
                argv.append(flag)
                if isinstance(item, list):
                    argv.extend(str(v) for v in item)
                else:
                    argv.append(str(item))
            continue

        if isinstance(value, list):
            argv.append(flag)
            argv.extend(str(v) for v in value)
            continue

        argv.append(flag)
        argv.append(str(value))

    if _argv_value(argv, "--renderer") == "scene" and "--bundle" not in argv:
        argv += ["--bundle", _resolve_scene_bundle(argv)]

    return argv


def unverified_values(argv: list[str]) -> list[str]:
    """Any argv value refusing a real launch -- see UNVERIFIED_PREFIX."""
    return [value for value in argv if value.startswith(UNVERIFIED_PREFIX)]
