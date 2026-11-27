"""Every replay of a saved ``resume.json`` re-resolves its input files first (``heal.refresh_recipe``).

2026-11-26: gomoveshiftsite#334 renamed the Ireland GPX files and Galley Head's retries kept
replaying a recipe naming ``ireland-2025-09-enhanced-day14.gpx``: each launch died in the drive's
pre-flight (exit 2). #614 and #618 fixed two of the replays; ``health publish`` and
``drive --from-recipe`` still replayed the recipe as written. This guard runs the same two cases
through every replay path, so a new one that skips the shared step fails here:

* a recipe GPX that is gone, whose backlog row names one that exists: launched with the row's file,
  and the recipe rewritten to name it;
* a recipe GPX that is gone with no replacement: refused with the reason, and nothing launched.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.cli import from_recipe
from gomoveshift.health import heal, launch, resume
from tests.test_publish_command import Rig as PublishRig

PROJECT = "day-a"
NOW = datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)


def _recipe(tmp: Path, stages_root: Path, gpx: Path) -> Path:
    """``<stages root>/<project>/<project>/_stages/resume.json`` naming ``gpx``, with a live source
    dir and service interpreter under ``tmp`` (the code root)."""
    src = tmp / "src" / PROJECT
    src.mkdir(parents=True, exist_ok=True)
    python = tmp / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    stages = stages_root / PROJECT / PROJECT / "_stages"
    stages.mkdir(parents=True, exist_ok=True)
    path = stages / "resume.json"
    argv = ["/old/py", "-m", "gomoveshift", "drive", str(src), "--region", "ireland",
            "--output-dir", str(tmp / "out" / PROJECT), "--skip-upload", "--no-youtube-upload",
            "--day-gpx", str(gpx)]  # fmt: skip
    body = {"argv": argv, "cwd": "/old", "env": {"GMS_INTERMEDIATE_DIR": str(stages_root / PROJECT)}}  # fmt: skip
    path.write_text(json.dumps(body))
    return path


def _heal(tmp, gpx, row):
    recipe = _recipe(tmp, tmp / "stages", gpx)
    inc, launches = tmp / "inc", []
    key = heal.register_recipe(inc, recipe, now_iso="2026-11-26T14:00:00Z")
    out = heal.run_heal(
        incidents_dir=inc, jobs_dir=tmp / "jobs", code_root=tmp, mirror_sha="a" * 40, now=NOW,
        only_key=key, launcher=lambda argv, cwd: launches.append(argv) or (True, ""),
        notify_fn=lambda *a: True, git_changed=lambda *a: [], conflict_fn=lambda j, rp, recipe: None,
        slot_fn=lambda r, rc: None, installed_version_fn=lambda p: None,
        pinned_version_fn=lambda c: None, sibling_versions_fn=lambda p: {},
        manifest_sha_fn=lambda r: None, manifest_row_fn=lambda r: row, config_sha_fn=lambda c: None,
    )  # fmt: skip
    return recipe, launches, out["attempts"][-1]["reason"]


def _backlog(tmp, gpx, row):
    recipe, launches = _recipe(tmp, tmp / "stages", gpx), []
    res = launch._resume_confirmed(
        {}, PROJECT, tmp / "stages", tmp / "out", tmp, tmp / "jobs", False,
        lambda argv, cwd, env: launches.append(argv) or (True, ""),
        lambda p: "1.0.0", lambda: "1.0.0", manifest_row=row,
    )  # fmt: skip
    return recipe, launches, res["reason"]


def _publish(tmp, gpx, row):
    rig = PublishRig(tmp)
    rig.audit(PROJECT)
    recipe = _recipe(tmp, rig.stages, gpx)
    rig.manifest_row = row
    res = rig.run(PROJECT)
    return recipe, [argv for argv, _cwd, _env in rig.launches], res["reason"]


def _from_recipe(tmp, gpx, row):
    recipe = _recipe(tmp, tmp / "stages", gpx)
    try:
        argv, _env, _path = from_recipe.expand(
            ["drive", "--from-recipe", str(recipe)], environ={}, manifest_row_fn=lambda r: row
        )
    except from_recipe.FromRecipeError as exc:
        return recipe, [], str(exc)
    return recipe, [argv], "relaunched"


REPLAYS = {"heal": _heal, "backlog": _backlog, "publish": _publish, "from-recipe": _from_recipe}


@pytest.mark.parametrize("replay", sorted(REPLAYS))
def test_a_gone_recipe_gpx_replays_with_the_backlog_rows_current_file(tmp_path, replay):
    gone, current = tmp_path / "gpx" / "enhanced-day14.gpx", tmp_path / "gpx" / "day15.gpx"
    current.parent.mkdir()
    current.write_text("<gpx/>")
    row = SimpleNamespace(project=PROJECT, day_gpx=str(current), bracket_gpx=())
    recipe, launches, reason = REPLAYS[replay](tmp_path, gone, row)
    assert len(launches) == 1, reason
    argv = launches[0]
    assert argv[argv.index("--day-gpx") + 1] == str(current)
    assert str(gone) not in argv
    saved = resume.read_recipe(recipe)["argv"]  # the next replay needs no refresh
    assert saved[saved.index("--day-gpx") + 1] == str(current)


@pytest.mark.parametrize("replay", sorted(REPLAYS))
@pytest.mark.parametrize("row", ["no row", "row file gone too"])
def test_a_gone_recipe_gpx_with_no_replacement_is_refused_and_never_launched(tmp_path, replay, row):
    gone = tmp_path / "gpx" / "enhanced-day14.gpx"
    found = None if row == "no row" else SimpleNamespace(day_gpx=str(tmp_path / "x.gpx"))
    recipe, launches, reason = REPLAYS[replay](tmp_path, gone, found)
    assert launches == []
    assert f"--day-gpx {gone} no longer exists" in reason
    assert str(gone) in resume.read_recipe(recipe)["argv"]  # nothing rewritten


def test_a_gone_source_dir_or_other_input_is_refused():
    argv = ["/py", "-m", "gomoveshift", "drive", "/gone/src"]
    assert heal.refresh_inputs(argv, None)[2] == "the drive's source directory /gone/src no longer exists"  # fmt: skip
    argv = ["/py", "-m", "gomoveshift", "drive", "/", "--trail-profile=/gone/p.yaml"]
    fresh, _refreshed, why = heal.refresh_inputs(argv, None)
    assert fresh is None and "--trail-profile /gone/p.yaml no longer exists" in why
