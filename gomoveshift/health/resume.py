"""The resume recipe a drive writes for itself (``<stages dir>/resume.json``, never tracked).

``health heal`` re-launches a failed drive from it and never guesses an argv. Writing it must
never fail a drive: every function here swallows its own errors (logged as a warning)."""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RECIPE_NAME = "resume.json"
#: An operator's stop: this file beside ``resume.json`` means ``health heal`` never relaunches the
#: drive, on any mirror SHA (a one-attempt validation run, say). Its contents are free-form notes.
HOLD_NAME = "no-auto-resume"
SCHEMA = "resume-recipe/1"
LINE_PREFIX = "resume recipe: "
_LINE_RE = re.compile(re.escape(LINE_PREFIX) + r"(\S.*?)\s*$")
#: Environment a resume must keep: the staging override decides where the saved state lives, so a
#: resume without it stages somewhere empty. Only these keys are recorded or replayed.
RECIPE_ENV_KEYS = ("GMS_INTERMEDIATE_DIR",)


def recipe_line(path: Path) -> str:
    """The one stable log line that links a drive's log to its recipe (parsed by ``drives``)."""
    return f"{LINE_PREFIX}{path}"


def parse_recipe_path(lines: List[str]) -> Optional[str]:
    found: Optional[str] = None
    for line in lines:
        if m := _LINE_RE.search(line):
            found = m.group(1)
    return found


#: Destinations of the flags that are one run's own act, never a standing choice: ``fresh``
#: discards state and ``ost_audit_confirmed`` is an operator's yes to the OST text they just
#: reviewed. A recipe replayed by ``health heal`` must never re-issue either, so a changed burn
#: stays unconfirmed. They are matched by DESTINATION against the drive parser, never by a
#: spelling: argparse resolves any unique prefix (and ``name=value``) to the same option.
_ONE_SHOT_DESTS = frozenset({"fresh", "ost_audit_confirmed"})


@functools.lru_cache(maxsize=1)
def _drive_options() -> Dict[str, str]:
    """``{option string: dest}`` of the real ``drive`` subparser (``--`` options only)."""
    import argparse

    from ..cli.parser import _build_parser

    parser = _build_parser()
    drive = next(
        a.choices["drive"]
        for a in parser._actions
        if isinstance(a, argparse._SubParsersAction) and "drive" in a.choices
    )
    return {
        opt: action.dest
        for action in drive._actions
        for opt in action.option_strings
        if opt.startswith("--")
    }


def _is_one_shot(token: str) -> bool:
    """Whether the drive parser resolves ``token`` (exact, unique prefix or ``=value``) to a
    one-shot flag. An ambiguous prefix counts when any candidate is one: fail closed."""
    if not token.startswith("--"):
        return False
    key = token.split("=", 1)[0]
    options = _drive_options()
    if key in options:
        return options[key] in _ONE_SHOT_DESTS
    return any(dest in _ONE_SHOT_DESTS for opt, dest in options.items() if opt.startswith(key))


def strip_one_shot(argv: List[str]) -> List[str]:
    """``argv`` without any token that resolves to a one-shot flag. A bare ``--`` ends the
    options: it is kept, and nothing after it is an option (so nothing after it is stripped)."""
    out: List[str] = []
    for i, a in enumerate(argv):
        if a == "--":
            out.extend(argv[i:])
            break
        if not _is_one_shot(a):
            out.append(a)
    return out


def write_recipe(
    stages_dir: Path, argv: List[str], *, cwd: str, sha: Optional[str], now_iso: str
) -> Optional[Path]:
    """Write the recipe: the drive's own CLI args (``argv`` excludes the program name) with the
    one-shot flags removed (see :func:`strip_one_shot`) -- a resume must keep the prior state,
    and a confirmation is one run's, never replayed -- plus the ``RECIPE_ENV_KEYS`` set in
    this process's environment. When the one-shot flags cannot be resolved, or the write fails,
    nothing is written and any recipe an earlier run left is removed (heal then finds none)."""
    path = Path(stages_dir) / RECIPE_NAME
    try:
        body = {
            "schema": SCHEMA,
            "argv": [
                sys.executable,
                "-m",
                "gomoveshift",
                *strip_one_shot(argv),
            ],
            "cwd": cwd,
            "env": {k: os.environ[k] for k in RECIPE_ENV_KEYS if os.environ.get(k)},
            "sha": sha,
            "started_at": now_iso,
            "phase_reached": None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 -- never fail a drive; an unstrippable argv is unwritten
        logger.warning("could not write resume recipe %s: %s", path, exc)
        # A recipe an earlier run left would replay THAT run's argv: no readable recipe is safer.
        for stale in (path, path.with_name(path.name + ".tmp")):
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove stale resume recipe %s", stale)
        return None
    return path


def is_held(recipe_path: Path) -> bool:
    """True when the operator's ``no-auto-resume`` marker sits beside this recipe."""
    return (Path(recipe_path).parent / HOLD_NAME).exists()


def read_recipe(path: Path) -> Optional[Dict[str, Any]]:
    """The recipe, or None when absent, unreadable or without a usable argv/cwd."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    argv = data.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        return None
    if not isinstance(data.get("cwd"), str):
        return None
    # A recipe written before the flag was one-shot may still carry it: never replay it.
    try:
        data["argv"] = strip_one_shot(argv)
    except Exception as exc:  # noqa: BLE001 -- cannot tell what is one-shot: replay nothing
        logger.warning("could not strip one-shot flags from recipe %s: %s", path, exc)
        return None
    return data


def recipe_env(recipe: Dict[str, Any]) -> Dict[str, str]:
    """The recorded ``RECIPE_ENV_KEYS`` a resume must run with; anything else is ignored."""
    env = recipe.get("env")
    if not isinstance(env, dict):
        return {}
    return {k: env[k] for k in RECIPE_ENV_KEYS if isinstance(env.get(k), str) and env[k]}


def mark_phase(stages_dir: Path, phase: str) -> None:
    """Record the latest completed phase; silent when the dir has no recipe."""
    path = Path(stages_dir) / RECIPE_NAME
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["phase_reached"] = phase
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except FileNotFoundError:
        return
    except (OSError, ValueError) as exc:
        logger.warning("could not update resume recipe %s: %s", path, exc)
