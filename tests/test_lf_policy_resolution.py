"""LF duration policy resolves by construction (synthetic data only)."""

import json
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from gomoveshift.config import (
    DriveConfig,
    lf_policy_banner,
    lf_policy_snapshot,
    resolve_lf_mode,
)
from gomoveshift.pipeline.drive_orchestrator import (
    LF_POLICY_STAMP,
    reconcile_lf_policy_on_resume,
    write_lf_policy_stamp,
)
from gomoveshift.pipeline.regular_lf_planner import (
    InsufficientFootageError,
    plan_full_drive_parts,
)

BOUNDS = {"full_drive_part_min_sec": 3600, "full_drive_part_max_sec": 5400}


def _scores(n: int) -> np.ndarray:
    arr = np.zeros((n, 9))
    arr[:, 0] = np.arange(n)
    arr[:, 1] = 0.5
    return arr


# (a) unset mode + both bounds -> full_drive, source default
def test_unset_mode_with_bounds_resolves_full_drive():
    d = DriveConfig(**BOUNDS)
    assert d.lf_selection_mode == "full_drive"
    assert d.lf_mode_source == "default"
    assert resolve_lf_mode(d) == "full_drive"


def test_unset_mode_without_bounds_stays_best_moments():
    d = DriveConfig()
    assert d.lf_selection_mode == "best_moments"
    assert d.lf_mode_source == "default"


def test_explicit_yaml_mode_is_kept_and_sourced():
    d = DriveConfig(lf_selection_mode="full_drive")
    assert (d.lf_selection_mode, d.lf_mode_source) == ("full_drive", "yaml")


def test_resolver_covers_standins_without_a_mode_field():
    ns = SimpleNamespace(**BOUNDS)
    assert resolve_lf_mode(ns) == "full_drive"
    assert resolve_lf_mode(SimpleNamespace()) == "best_moments"


# (b) explicit best_moments + bounds -> named warning naming both
def test_explicit_best_moments_with_bounds_warns_by_name(caplog):
    with caplog.at_level(logging.WARNING):
        d = DriveConfig(lf_selection_mode="best_moments", **BOUNDS)
    assert d.lf_selection_mode == "best_moments"
    msg = next(r.getMessage() for r in caplog.records if "LF-POLICY-OVERRIDE" in r.getMessage())
    assert "best_moments" in msg and "full_drive_part_min_sec=3600" in msg
    assert "full_drive_part_max_sec=5400" in msg


def test_cli_override_to_best_moments_warns_and_sets_source(caplog):
    d = DriveConfig(**BOUNDS)
    with caplog.at_level(logging.WARNING):
        d.set_lf_mode("best_moments", "cli")
    assert d.lf_mode_source == "cli"
    assert any("LF-POLICY-OVERRIDE" in r.getMessage() for r in caplog.records)


def test_no_override_warning_when_consistent(caplog):
    with caplog.at_level(logging.WARNING):
        DriveConfig(**BOUNDS)
        DriveConfig()
    assert not any("LF-POLICY-OVERRIDE" in r.getMessage() for r in caplog.records)


# (c) banner
def test_banner_shows_mode_source_and_bounds():
    line = lf_policy_banner(DriveConfig(**BOUNDS))
    assert "mode=full_drive" in line and "source=default" in line
    assert "part_min=3600s" in line and "part_max=5400s" in line
    d = DriveConfig(**BOUNDS)
    d.set_lf_mode("full_drive", "cli")
    assert "source=cli" in lf_policy_banner(d)


# (d) short day
def test_short_full_drive_day_raises_naming_day_seconds_and_minimum():
    with pytest.raises(InsufficientFootageError) as ei:
        plan_full_drive_parts(
            _scores(1620),
            min_duration_sec=480,
            full_drive_max_duration_sec=7200,
            stop_mask=np.zeros(1620, dtype=bool),
            part_min_sec=3600,
            part_max_sec=5400,
            day="2026-02-20",
        )
    msg = str(ei.value)
    assert "2026-02-20" in msg and "1620" in msg and "3600" in msg
    assert isinstance(ei.value, ValueError)


def test_stop_masked_footage_counts_as_not_kept():
    mask = np.zeros(4000, dtype=bool)
    mask[1000:3000] = True  # only 2000 s kept of 4000
    with pytest.raises(InsufficientFootageError, match="2000"):
        plan_full_drive_parts(
            _scores(4000),
            min_duration_sec=480,
            full_drive_max_duration_sec=7200,
            stop_mask=mask,
            part_min_sec=3600,
            part_max_sec=5400,
            day="d",
        )


def test_each_short_day_refuses_on_its_own_never_joined():
    """Two short consecutive days are two refusals, not one joined LF."""
    for day in ("2026-02-20", "2026-02-21"):
        with pytest.raises(InsufficientFootageError, match=day):
            plan_full_drive_parts(
                _scores(1800),
                min_duration_sec=480,
                full_drive_max_duration_sec=7200,
                stop_mask=np.zeros(1800, dtype=bool),
                part_min_sec=3600,
                part_max_sec=5400,
                day=day,
            )


def test_no_escape_flag_exists():
    from gomoveshift.cli.parser import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["drive", "/tmp/x", "--allow-short-lf"])
    assert not hasattr(DriveConfig(), "allow_short_lf")


def test_migration_is_named_when_mode_inferred(caplog):
    with caplog.at_level(logging.WARNING):
        DriveConfig(**BOUNDS)
    assert any(
        "LF-POLICY-MIGRATION: mode inferred full_drive from part bounds" in r.getMessage()
        for r in caplog.records
    )


def test_long_enough_day_does_not_raise():
    parts = plan_full_drive_parts(
        _scores(4000),
        min_duration_sec=480,
        full_drive_max_duration_sec=7200,
        stop_mask=np.zeros(4000, dtype=bool),
        part_min_sec=3600,
        part_max_sec=5400,
        day="d",
    )
    assert len(parts) == 1


# (e) resume
PHASES = ["phase_a", "phase_bc_day_first", "phase_d"]


def test_resume_with_changed_policy_reruns_plan_and_names_it(tmp_path, caplog):
    write_lf_policy_stamp(tmp_path, DriveConfig())  # plan built under best_moments
    with caplog.at_level(logging.WARNING):
        kept = reconcile_lf_policy_on_resume(tmp_path, list(PHASES), DriveConfig(**BOUNDS))
    assert kept == ["phase_a"]
    assert any("LF-POLICY-RESUME-MISMATCH" in r.getMessage() for r in caplog.records)


def test_resume_with_changed_bounds_reruns_plan(tmp_path):
    write_lf_policy_stamp(tmp_path, DriveConfig(**BOUNDS))
    other = DriveConfig(full_drive_part_min_sec=1800, full_drive_part_max_sec=3600)
    assert reconcile_lf_policy_on_resume(tmp_path, list(PHASES), other) == ["phase_a"]


def test_resume_with_same_policy_keeps_everything(tmp_path, caplog):
    write_lf_policy_stamp(tmp_path, DriveConfig(**BOUNDS))
    with caplog.at_level(logging.WARNING):
        kept = reconcile_lf_policy_on_resume(tmp_path, list(PHASES), DriveConfig(**BOUNDS))
    assert kept == PHASES
    assert not any("LF-POLICY-RESUME" in r.getMessage() for r in caplog.records)


def test_resume_without_stamp_warns_by_name_and_keeps(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        kept = reconcile_lf_policy_on_resume(tmp_path, list(PHASES), DriveConfig())
    assert kept == PHASES
    assert any("LF-POLICY-RESUME-UNSTAMPED" in r.getMessage() for r in caplog.records)


def test_resume_before_plan_exists_is_untouched(tmp_path):
    assert reconcile_lf_policy_on_resume(tmp_path, ["phase_a"], DriveConfig()) == ["phase_a"]


def test_stamp_roundtrips_the_snapshot(tmp_path):
    write_lf_policy_stamp(tmp_path, DriveConfig(**BOUNDS))
    assert json.loads((tmp_path / LF_POLICY_STAMP).read_text()) == lf_policy_snapshot(
        DriveConfig(**BOUNDS)
    )
