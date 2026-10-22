"""The resume recipe a drive writes for itself (``<stages dir>/resume.json``, never tracked).

``health heal`` re-launches a failed drive from it and never guesses an argv. Writing it must
never fail a drive: every function here swallows its own errors (logged as a warning)."""

from __future__ import annotations

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


def write_recipe(
    stages_dir: Path, argv: List[str], *, cwd: str, sha: Optional[str], now_iso: str
) -> Optional[Path]:
    """Write the recipe: the drive's own CLI args (``argv`` excludes the program name) with
    ``--fresh`` removed -- a resume must keep the prior state, not clear it -- plus the
    ``RECIPE_ENV_KEYS`` set in this process's environment."""
    path = Path(stages_dir) / RECIPE_NAME
    try:
        body = {
            "schema": SCHEMA,
            "argv": [sys.executable, "-m", "gomoveshift", *[a for a in argv if a != "--fresh"]],
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
    except OSError as exc:
        logger.warning("could not write resume recipe %s: %s", path, exc)
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
