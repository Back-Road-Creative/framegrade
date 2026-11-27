"""A project whose queued Short may still need a route re-measure keeps its stages.

``dispatch-queue`` re-measures a queued Short whose route record has no ``bins`` from the
project's ``_stages`` (``gps_points_<date>.json``, ``shorts_ost_audit_<date>.json``).
2026-11-22: Early Clonakilty's six queued Shorts lost those files to stage cleanup and were
refused at every tick with nothing raised. ``stage_owners.remeasure_holds`` is the one rule the
releases here consult; ``health check`` rule 17 raises when the files are already gone.
Synthetic data, ``tmp_path`` only.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from test_queue_repair import DATE, _repair, _short, env  # noqa: F401 -- env is a fixture

from gomoveshift.cli.handlers import cleanup as cleanup_handler
from gomoveshift.health import check as check_mod
from gomoveshift.health import launch
from gomoveshift.health import stage_owners as so
from gomoveshift.pipeline import cleanup as cleanup_mod
from gomoveshift.publishing.route_remeasure import remeasure_inputs


def _db(env):  # noqa: F811
    return env.tmp / "q" / "social_queue.sqlite3"


def _cfg(env):  # noqa: F811
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(env.tmp / "q")))


def test_a_queued_short_with_no_bins_holds_its_project(env):  # noqa: F811
    _, (rid,) = env.add(_short(3))
    assert so.remeasure_holds(_db(env)) == {
        "proj": [so.RemeasureHold(rid, str(env.tmp / "out" / "proj_publish_plan.json"), DATE)]
    }


def test_a_measured_row_holds_nothing(env):  # noqa: F811
    env.add(_short(3))
    _repair(env, apply=True)  # now has bins: never re-measured again
    assert so.remeasure_holds(_db(env)) == {}


def test_a_published_row_holds_nothing(env):  # noqa: F811
    env.add(_short(3), status="published")
    assert so.remeasure_holds(_db(env)) == {}


def test_an_unreadable_sidecar_holds(env):  # noqa: F811
    sidecar, (rid,) = env.add(_short(3))
    sidecar.unlink()  # proves nothing either way
    [hold] = so.remeasure_holds(_db(env))["proj"]
    assert (hold.delivery_id, hold.date) == (rid, None)


def test_a_missing_queue_raises_so_nothing_is_released(tmp_path):
    with pytest.raises(FileNotFoundError):
        so.remeasure_holds(tmp_path / "nope.sqlite3")


def test_the_cleanup_name_pattern_matches_the_remeasure_inputs():
    for path in remeasure_inputs(Path("/s"), "2025-09-19").values():
        assert cleanup_mod._REMEASURE_INPUT_RE.match(path.name)


def _main_pipeline_project(out: Path) -> Path:
    stages = out / "proj" / "_stages"
    stages.mkdir(parents=True)
    (stages / "pipeline_state.json").write_text(json.dumps({"stages_completed": ["csv_generation"]}))  # fmt: skip
    (out / "proj" / "proj_FINAL.mp4").write_bytes(b"f" * 100)
    (stages / "x_color.mp4").write_bytes(b"c" * 100)
    for path in remeasure_inputs(stages, DATE).values():
        path.write_text("{}")
    return stages


def _cli(env, **flags):  # noqa: F811
    out, stg, cache = env.tmp / "o", env.tmp / "stg", env.tmp / "cache"
    for d in (out, stg, cache):
        d.mkdir(exist_ok=True)
    args = SimpleNamespace(project="proj", confirm=True, min_age=0, archive_base=None)
    args.__dict__.update(flags)
    with (
        patch("gomoveshift.paths.gms_output_dir", return_value=out),
        patch("gomoveshift.paths.gms_staging_dir", return_value=stg),
        patch("gomoveshift.paths.gms_intermediate_base", return_value=cache),
        patch("gomoveshift.health.heal.live_drive_cmdlines", return_value=[]),
        patch(
            "gomoveshift.pipeline.ffmpeg_runner.validate_video",
            return_value=SimpleNamespace(valid=True),
        ),
    ):
        return cleanup_handler.cmd_cleanup(args, _cfg(env))


def test_cleanup_keeps_the_remeasure_inputs_of_a_held_project(env, capsys):  # noqa: F811
    _, (rid,) = env.add(_short(3))
    stages = _main_pipeline_project(env.tmp / "o")
    assert _cli(env) == 0
    assert not (stages / "x_color.mp4").exists()  # the rest is still reclaimed
    for path in remeasure_inputs(stages, DATE).values():
        assert path.exists(), path
    _repair(env, apply=True)  # measured: the hold ends
    assert _cli(env) == 0
    assert not any(p.exists() for p in remeasure_inputs(stages, DATE).values())


def test_cleanup_never_removes_a_held_abandoned_run(env, capsys):  # noqa: F811
    _, (rid,) = env.add(_short(3))
    stages = env.tmp / "cache" / "proj" / "_stages"
    stages.mkdir(parents=True)
    (stages / "pipeline_state.json").write_text(json.dumps({"phases_completed": ["phase_a"]}))
    assert _cli(env, abandoned=True) == 2
    assert stages.is_dir()
    assert f"queued Short(s) {rid} may still need a route re-measure" in capsys.readouterr().out


def test_an_unreadable_queue_cleans_nothing(env, capsys):  # noqa: F811
    stages = _main_pipeline_project(env.tmp / "o")
    _db(env).unlink()
    assert _cli(env) == 2
    assert (stages / "x_color.mp4").exists()
    assert "social queue unreadable" in capsys.readouterr().out


def test_backlog_never_lists_a_held_root_for_release(env):  # noqa: F811
    from tests.test_health_backlog_launch import Rig  # noqa: PLC0415

    env.add(_short(3))
    rig = Rig(env.tmp / "rig")
    (rig.stages / "proj").mkdir()
    owners = [so.Owner("proj", "JP", "r", "framegrade#1", date(2026, 1, 1))]
    lst = env.tmp / "rel" / so.STAGES_RELEASE_LIST_NAME
    kw = {**rig.kwargs(), "stage_owners": owners, "release_list": lst, "queue_db": _db(env)}
    res = launch.run_backlog(**kw)
    assert res["stage_roots"]["listed"] == [] and res["stage_roots"]["held"] == ["proj"]
    assert not lst.exists()
    _repair(env, apply=True)
    assert launch.run_backlog(**kw)["stage_roots"]["listed"] == ["proj"]


def test_health_raises_when_a_held_projects_stages_are_gone(env):  # noqa: F811
    _, (a, b) = env.add(_short(3), _short(4))
    gone = env.tmp / "gone" / "_stages"
    rows = check_mod.stages_gone_incidents(_db(env), lambda s, p: gone, "2026-11-27T00:00:00Z")
    assert [r["key"] for r in rows] == ["queued_short_stages_gone:proj"]
    assert rows[0]["delivery_ids"] == [a, b]
    assert rows[0]["missing"] == [f"gps_points_{DATE}.json", f"shorts_ost_audit_{DATE}.json"]
    line = check_mod._line(rows[0])
    assert "proj" in line and "supersede re-render" in line and str(gone) in line
    assert check_mod.stages_gone_incidents(_db(env), lambda s, p: env.stages, "t") == []


def test_health_check_files_and_resolves_the_incident(env, tmp_path):  # noqa: F811
    env.add(_short(3))
    cfg = SimpleNamespace(
        seo=SimpleNamespace(storage_path=str(env.tmp / "q")),
        paths=SimpleNamespace(intermediate_folder=""),
    )
    notes: list = []
    with patch("gomoveshift.paths.gms_intermediate_dir", return_value=tmp_path / "none"):
        found = check_mod._cadence_rows(cfg, None, check_mod.datetime.now(check_mod.timezone.utc), "t", notes)  # fmt: skip
    assert [r["project"] for r in found["queued_short_stages_gone"]] == ["proj"], notes
    with patch("gomoveshift.paths.gms_intermediate_dir", return_value=env.stages):
        found = check_mod._cadence_rows(cfg, None, check_mod.datetime.now(check_mod.timezone.utc), "t", notes)  # fmt: skip
    assert found["queued_short_stages_gone"] == []
