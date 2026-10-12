"""CLI adapter: launch a Route Story film render from a per-trip
launch profile YAML passed with ``--profile``.

The console job IS the supervised process here: ``jobs.launch()`` logs,
reaps and fingerprints it the same way it does every other console job.
An external job supervisor may keep its own supervision -- this handler is
a second, in-process door onto ``film.main()``, not a replacement for it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ...pipeline.route_story import film
from ...pipeline.route_story.profile import (
    ProfileError,
    build_film_argv,
    film_parser,
    load_profile,
    unverified_values,
)


def build_route_story_film_argv(profile_path: "str | Path", *, dry_run: bool = False) -> list[str]:
    """Argv for ``gomoveshift route-story-film --profile <yaml>``."""
    argv = [
        sys.executable,
        "-m",
        "gomoveshift",
        "route-story-film",
        "--profile",
        str(profile_path),
    ]
    if dry_run:
        argv.append("--dry-run")
    return argv


def cmd_route_story_film(args, config) -> int:
    try:
        profile = load_profile(Path(args.profile))
        parser = film_parser()
        film_argv = build_film_argv(parser, profile)
    except ProfileError as exc:
        print(f"route-story-film: {exc}", file=sys.stderr)
        return 1

    unverified = unverified_values(film_argv)
    if unverified:
        print("route-story-film: profile has unverified value(s), refusing:", file=sys.stderr)
        for value in unverified:
            print(f"  {value}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(" ".join(film_argv))
        return 0

    return film.main(film_argv)
