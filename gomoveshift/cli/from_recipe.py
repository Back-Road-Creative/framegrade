"""``gomoveshift drive --from-recipe <path-or-project>``: relaunch from the run's own recipe.

Every drive writes ``<stages>/resume.json`` (``health.resume.write_recipe``): its argv with
``--fresh`` stripped and the ``GMS_INTERMEDIATE_DIR`` it ran under. ``health heal`` relaunches
from it and never guesses; only a hand-typed ``drive`` ignored it. 2026-11-08: a manual
relaunch of nctopa-2026-03-05-rec0063 retyped the env wrong, the drive chose a FRESH stage
dir and left the earlier 139 GB stage cache orphaned.

:func:`expand` runs BEFORE argparse: the recipe's argv becomes the argv, the caller may add
only :data:`OVERRIDE_FLAGS`, and the recipe's env is handed back to be applied before anything
resolves a stage dir. The same stage dir is reached because the same env is set — robust by
construction, no second lookup path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from ..health.launch import SUPERSEDED_DIR
from ..health.resume import RECIPE_NAME, read_recipe, recipe_env

FLAG = "--from-recipe"

#: The only flags allowed beside ``--from-recipe`` (``name -> values it takes``): what a
#: resume legitimately changes. Everything else the recipe carries, and a different value
#: would be a different run.
OVERRIDE_FLAGS: Dict[str, int] = {
    "--phase": 1,
    "--publish": 1,
    "-p": 1,
    "--lf-day": 1,
    "--lf-publish-date": 1,
    "--skip-upload": 0,
    "--no-youtube-upload": 0,
    "--force": 0,
    "--no-tmux": 0,
}


class FromRecipeError(ValueError):
    """Why the relaunch is refused; the CLI prints it and exits 2."""


def default_stages_root() -> Path:
    """Where a bare project name is looked up (``paths.gms_intermediate_base``)."""
    from ..paths import gms_intermediate_base

    return gms_intermediate_base()


def find_recipes(project: str, stages_root: Path) -> List[Path]:
    """Every live recipe for ``project`` under ``stages_root``: ``<root>/<project>/_stages`` and
    ``<root>/<run>/<project>/_stages`` (the layout ``GMS_INTERMEDIATE_DIR=<root>/<run>`` gives),
    never under ``_superseded``."""
    root = Path(stages_root)
    found = []
    for pattern in (f"{project}/_stages/{RECIPE_NAME}", f"*/{project}/_stages/{RECIPE_NAME}"):
        for path in sorted(root.glob(pattern)):
            if path.relative_to(root).parts[0] == SUPERSEDED_DIR:
                continue
            found.append(path)
    return found


def resolve_recipe(target: str, stages_root: Path) -> Path:
    """The one ``resume.json`` ``target`` names: a recipe file, a ``_stages`` or project dir
    holding one, or a bare project name with exactly one live recipe under ``stages_root``."""
    p = Path(target).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        for candidate in (p / RECIPE_NAME, p / "_stages" / RECIPE_NAME):
            if candidate.is_file():
                return candidate
        raise FromRecipeError(
            f"no recipe at {p}: expected {RECIPE_NAME} or _stages/{RECIPE_NAME} in it"
        )
    if os.sep in target or p.suffix:
        raise FromRecipeError(f"no recipe at {p}")
    hits = find_recipes(target, stages_root)
    if not hits:
        raise FromRecipeError(
            f"no recipe under {Path(stages_root) / target} (or */{target}/_stages)"
        )
    if len(hits) > 1:
        listed = "\n  ".join(str(h) for h in hits)
        raise FromRecipeError(
            f"{len(hits)} recipes for {target}; pass the one to relaunch:\n  {listed}"
        )
    return hits[0]


def _split_overrides(tokens: List[str]) -> Tuple[Optional[str], List[str]]:
    """``(target, override argv)`` from the tokens after ``drive``; refuses anything else."""
    target: Optional[str] = None
    overrides: List[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == FLAG or tok.startswith(FLAG + "="):
            if tok == FLAG:
                if i + 1 >= len(tokens):
                    raise FromRecipeError(f"{FLAG} needs a recipe path or project name")
                target = tokens[i + 1]
                i += 2
            else:
                target = tok.split("=", 1)[1]
                i += 1
            continue
        if tok == "--fresh":
            raise FromRecipeError(
                f"--fresh beside {FLAG}: a relaunch keeps the run's state; drop --fresh, or run "
                "`drive` without --from-recipe to start over"
            )
        name = tok.split("=", 1)[0] if tok.startswith("--") else tok
        if name in OVERRIDE_FLAGS:
            n = OVERRIDE_FLAGS[name]
            if "=" in tok and tok.startswith("--"):
                n = 0
            overrides.extend(tokens[i : i + 1 + n])
            i += 1 + n
            continue
        allowed = ", ".join(k for k in OVERRIDE_FLAGS if k != "-p")
        if tok.startswith("-"):
            raise FromRecipeError(
                f"{tok} beside {FLAG}: the recipe carries it; only {allowed} may override"
            )
        raise FromRecipeError(
            f"{tok} beside {FLAG}: the directory comes from the recipe; only {allowed} may be added"
        )
    return target, overrides


def expand(
    argv: List[str], *, environ: Mapping[str, str], stages_root: Optional[Path] = None
) -> Tuple[List[str], Dict[str, str], Optional[Path]]:
    """``(argv, env to apply, recipe path)``. Untouched ``(argv, {}, None)`` unless ``argv`` is
    a ``drive`` with ``--from-recipe``. Raises :class:`FromRecipeError`."""
    if (
        not argv
        or argv[0] != "drive"
        or not any(a == FLAG or a.startswith(FLAG + "=") for a in argv)
    ):
        return list(argv), {}, None
    target, overrides = _split_overrides(list(argv[1:]))
    if target is None:  # unreachable: the FLAG scan above found one
        raise FromRecipeError(f"{FLAG} needs a recipe path or project name")
    root = Path(stages_root) if stages_root is not None else default_stages_root()
    recipe_path = resolve_recipe(target, root)
    recipe = read_recipe(recipe_path)
    if recipe is None:
        raise FromRecipeError(f"unreadable recipe {recipe_path}: no usable argv/cwd")
    recipe_argv = recipe["argv"]
    if len(recipe_argv) < 4 or recipe_argv[1:4] != ["-m", "gomoveshift", "drive"]:
        raise FromRecipeError(
            f"recipe {recipe_path} is not a `<python> -m gomoveshift drive ...` run; cannot relaunch it"
        )
    env = recipe_env(recipe)
    for key in ("GMS_INTERMEDIATE_DIR",):
        current = environ.get(key, "")
        recorded = env.get(key, "")
        if current and current != recorded:
            where = f"recipe stages at {recorded}" if recorded else "recipe recorded no " + key
            raise FromRecipeError(
                f"{where}, environment says {key}={current}; unset it or drop {FLAG}"
            )
    return [*recipe_argv[3:], *overrides], env, recipe_path
