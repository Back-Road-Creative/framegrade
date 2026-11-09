"""``gomoveshift drive --from-recipe <path-or-project>``: a relaunch never retypes the env.

2026-11-08: a manual relaunch of nctopa-2026-03-05-rec0063 ran with a different
``GMS_INTERMEDIATE_DIR`` than the staged run, so the drive chose a FRESH stage dir
and left the earlier 139 GB stage cache orphaned. The operator had to re-type the
env by hand from the earlier job's argv — while every drive already writes
``<stages>/resume.json`` with that argv and env.

Rules pinned here: the recipe's argv IS the argv (``--fresh`` already stripped),
its env is applied before anything resolves a stage dir, and the caller may add
only the override flags a resume legitimately changes. Anything else beside
``--from-recipe`` refuses, naming the flag. A bare project name resolves to the
one recipe under the stages root; none or several refuse. A conflicting
``GMS_INTERMEDIATE_DIR`` in the environment refuses: the same stage dir is reached
because the same env is set, never through a second lookup path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.cli import from_recipe
from gomoveshift.cli.from_recipe import FromRecipeError, expand
from gomoveshift.cli.parser import _build_parser
from gomoveshift.health.launch import SUPERSEDED_DIR

PY = "/opt/venv/bin/python"
SRC = "/mnt/d/_gms-sources/nctopa-2026-03-05-rec0063"
OUT = "/mnt/d/GMSVidOutput/nctopa-2026-03-05-rec0063"
DRIVE_ARGS = [
    "drive", SRC, "--region", "usa", "--ingest-mode", "gopro", "--lf-mode", "full_drive",
    "--output-dir", OUT, "--skip-upload", "--no-youtube-upload", "--no-review", "--no-tmux", "--force",
]  # fmt: skip


def _recipe(stages: Path, *, env=None, argv=None) -> Path:
    stages.mkdir(parents=True, exist_ok=True)
    path = stages / "resume.json"
    path.write_text(
        json.dumps(
            {
                "schema": "resume-recipe/1",
                "argv": argv if argv is not None else [PY, "-m", "gomoveshift", *DRIVE_ARGS],
                "cwd": "/srv/code/_active/svc-gomoveshift-video",
                "env": {"GMS_INTERMEDIATE_DIR": str(stages.parent.parent)} if env is None else env,
                "sha": "abc1234",
            }
        )
    )
    return path


def _root(tmp_path: Path) -> Path:
    return tmp_path / "stages-root"


def _one(tmp_path: Path, run: str = "rec0063-signs-resume3") -> Path:
    """``<root>/<run>/<project>/_stages/resume.json`` — the layout GMS_INTERMEDIATE_DIR=<root>/<run> gives."""
    return _recipe(_root(tmp_path) / run / "nctopa-2026-03-05-rec0063" / "_stages")


def _expand(tmp_path: Path, argv, environ=None):
    return expand(argv, environ={} if environ is None else environ, stages_root=_root(tmp_path))


def test_other_commands_and_a_plain_drive_pass_through_untouched(tmp_path):
    for argv in (["cleanup", "x"], ["drive", SRC, "--region", "usa"], []):
        assert _expand(tmp_path, argv) == (argv, {}, None)


def test_a_recipe_file_supplies_the_argv_and_the_env(tmp_path):
    recipe = _one(tmp_path)
    argv, env, used = _expand(tmp_path, ["drive", "--from-recipe", str(recipe)])
    assert argv == DRIVE_ARGS
    assert env == {"GMS_INTERMEDIATE_DIR": str(_root(tmp_path) / "rec0063-signs-resume3")}
    assert used == recipe


def test_override_flags_are_appended_after_the_recipe_argv(tmp_path):
    recipe = _one(tmp_path)
    argv, _env, _ = _expand(
        tmp_path,
        [
            "drive",
            "--from-recipe",
            str(recipe),
            "--phase",
            "A",
            "--lf-day=wed",
            "--force",
            "--publish",
            "wed",
        ],
    )
    assert argv == [*DRIVE_ARGS, "--phase", "A", "--lf-day=wed", "--force", "--publish", "wed"]
    # the result is something the real parser accepts
    ns = _build_parser().parse_args(argv)
    assert ns.phase == "A" and ns.lf_day == "wed" and ns.directory == Path(SRC)


@pytest.mark.parametrize(
    "extra, named",
    [
        (["--region", "usa"], "--region"),
        (["--output-dir", OUT], "--output-dir"),
        (["--label-dry-run"], "--label-dry-run"),
        ([SRC], SRC),
    ],
)
def test_anything_the_recipe_carries_refuses_naming_it(tmp_path, extra, named):
    recipe = _one(tmp_path)
    with pytest.raises(FromRecipeError) as exc:
        _expand(tmp_path, ["drive", "--from-recipe", str(recipe), *extra])
    assert named in str(exc.value) and "--from-recipe" in str(exc.value)


def test_fresh_beside_from_recipe_refuses(tmp_path):
    recipe = _one(tmp_path)
    with pytest.raises(FromRecipeError, match="--fresh"):
        _expand(tmp_path, ["drive", "--from-recipe", str(recipe), "--fresh"])


def test_a_bare_project_name_resolves_the_one_recipe_under_the_root(tmp_path):
    recipe = _one(tmp_path)
    argv, env, used = _expand(tmp_path, ["drive", "--from-recipe", "nctopa-2026-03-05-rec0063"])
    assert used == recipe and argv == DRIVE_ARGS
    assert env["GMS_INTERMEDIATE_DIR"] == str(recipe.parent.parent.parent)


def test_a_one_level_layout_resolves_too(tmp_path):
    recipe = _recipe(_root(tmp_path) / "clonakilty-2025-09-17" / "_stages", env={})
    _argv, env, used = _expand(tmp_path, ["drive", "--from-recipe", "clonakilty-2025-09-17"])
    assert used == recipe and env == {}


def test_no_recipe_under_the_root_refuses(tmp_path):
    _root(tmp_path).mkdir()
    with pytest.raises(FromRecipeError) as exc:
        _expand(tmp_path, ["drive", "--from-recipe", "nctopa-2026-03-05-rec0063"])
    assert "no recipe" in str(exc.value) and str(_root(tmp_path)) in str(exc.value)


def test_two_recipes_refuse_naming_both(tmp_path):
    a = _one(tmp_path, "run-a")
    b = _one(tmp_path, "run-b")
    with pytest.raises(FromRecipeError) as exc:
        _expand(tmp_path, ["drive", "--from-recipe", "nctopa-2026-03-05-rec0063"])
    assert str(a) in str(exc.value) and str(b) in str(exc.value)


def test_a_superseded_dir_is_not_a_candidate(tmp_path):
    live = _one(tmp_path, "run-live")
    _one(tmp_path, SUPERSEDED_DIR)
    _argv, _env, used = _expand(tmp_path, ["drive", "--from-recipe", "nctopa-2026-03-05-rec0063"])
    assert used == live


def test_a_stages_dir_or_project_dir_path_resolves_its_recipe(tmp_path):
    recipe = _one(tmp_path)
    for target in (recipe.parent, recipe.parent.parent):
        assert _expand(tmp_path, ["drive", "--from-recipe", str(target)])[2] == recipe


def test_a_conflicting_environment_refuses(tmp_path):
    recipe = _one(tmp_path)
    with pytest.raises(FromRecipeError) as exc:
        _expand(
            tmp_path,
            ["drive", "--from-recipe", str(recipe)],
            environ={"GMS_INTERMEDIATE_DIR": "/mnt/c/Elsewhere"},
        )
    msg = str(exc.value)
    assert "/mnt/c/Elsewhere" in msg and str(_root(tmp_path) / "rec0063-signs-resume3") in msg


def test_an_environment_set_when_the_recipe_recorded_none_refuses(tmp_path):
    recipe = _recipe(_root(tmp_path) / "clonakilty-2025-09-17" / "_stages", env={})
    with pytest.raises(FromRecipeError, match="GMS_INTERMEDIATE_DIR"):
        _expand(
            tmp_path,
            ["drive", "--from-recipe", str(recipe)],
            environ={"GMS_INTERMEDIATE_DIR": "/mnt/c/Elsewhere"},
        )


def test_the_same_environment_is_not_a_conflict(tmp_path):
    recipe = _one(tmp_path)
    same = str(_root(tmp_path) / "rec0063-signs-resume3")
    _argv, env, _ = _expand(
        tmp_path, ["drive", "--from-recipe", str(recipe)], environ={"GMS_INTERMEDIATE_DIR": same}
    )
    assert env == {"GMS_INTERMEDIATE_DIR": same}


def test_a_recipe_whose_argv_is_not_a_gomoveshift_drive_refuses(tmp_path):
    for argv in (["bash", "run.sh"], [PY, "-m", "gomoveshift", "cleanup", "x"]):
        recipe = _recipe(_root(tmp_path) / "r" / "p" / "_stages", argv=argv)
        with pytest.raises(FromRecipeError, match="drive"):
            _expand(tmp_path, ["drive", "--from-recipe", str(recipe)])


def test_an_unreadable_recipe_refuses(tmp_path):
    stages = _root(tmp_path) / "r" / "p" / "_stages"
    stages.mkdir(parents=True)
    (stages / "resume.json").write_text("{not json")
    with pytest.raises(FromRecipeError, match="unreadable"):
        _expand(tmp_path, ["drive", "--from-recipe", str(stages / "resume.json")])


def test_the_recipe_env_reaches_the_stage_dir_the_run_used(tmp_path, monkeypatch):
    """Robust by construction: ``_stages_base`` lands on the recipe's dir because the env is set."""
    from gomoveshift.cli.handlers.drive import _stages_base

    recipe = _one(tmp_path)
    monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
    argv, env, _ = _expand(tmp_path, ["drive", "--from-recipe", str(recipe)])
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    ns = _build_parser().parse_args(argv)
    config = SimpleNamespace(paths=SimpleNamespace(intermediate_folder=""))
    assert _stages_base(ns, config) == recipe.parent


def test_main_expands_applies_the_env_and_dispatches_the_recipe_argv(tmp_path, monkeypatch, capsys):
    from gomoveshift import cli

    recipe = _one(tmp_path)
    seen = {}

    def fake_drive(args, config):
        seen["directory"] = args.directory
        seen["phase"] = args.phase
        seen["env"] = os.environ.get("GMS_INTERMEDIATE_DIR")
        return 0

    monkeypatch.setattr(
        "sys.argv", ["gomoveshift", "drive", "--from-recipe", str(recipe), "--phase", "A"]
    )
    # ``main()`` applies the recipe env with ``os.environ.update`` — by design, so the
    # drive's children inherit it. In the test process that write must not outlive the
    # test: ``monkeypatch.delenv`` on an UNSET name records nothing to restore, and the
    # leaked ``GMS_INTERMEDIATE_DIR`` sent every later orchestrator test on the same
    # xdist worker to the wrong stage base (#258 CI shard 2; .data/logs/u5-leak-red.log).
    # ``patch.dict(os.environ)`` snapshots and restores the whole mapping.
    with (
        patch.dict(os.environ),
        patch.dict(cli._COMMAND_MAP, {"drive": fake_drive}),
        patch.object(cli, "get_config", return_value=SimpleNamespace()),
        patch("gomoveshift.pipeline.schedule_constants.apply_config"),
        patch.object(from_recipe, "default_stages_root", return_value=_root(tmp_path)),
    ):
        os.environ.pop("GMS_INTERMEDIATE_DIR", None)
        rc = cli.main()
    assert rc == 0
    assert seen == {
        "directory": Path(SRC),
        "phase": "A",
        "env": str(_root(tmp_path) / "rec0063-signs-resume3"),
    }
    assert str(recipe) in capsys.readouterr().out
    assert (
        "GMS_INTERMEDIATE_DIR" not in os.environ
        or os.environ["GMS_INTERMEDIATE_DIR"] != seen["env"]
    ), "the recipe env leaked out of the test"


def test_main_refuses_with_exit_2_and_the_reason(tmp_path, monkeypatch, capsys):
    from gomoveshift import cli

    recipe = _one(tmp_path)
    monkeypatch.setattr(
        "sys.argv", ["gomoveshift", "drive", "--from-recipe", str(recipe), "--fresh"]
    )
    with patch.object(from_recipe, "default_stages_root", return_value=_root(tmp_path)):
        rc = cli.main()
    assert rc == 2
    assert "--fresh" in capsys.readouterr().err


def test_the_parser_documents_the_flag():
    helptext = _build_parser()._subparsers._group_actions[0].choices["drive"].format_help()
    assert "--from-recipe" in helptext
