"""The hand-set long-form day/time defaults are gone; Waypost's cadence answers.

``DriveConfig.long_form_publish_days`` / ``long_form_publish_time`` survive only as the
per-run ``--lf-day`` / ``--lf-time`` transport (empty / unset by default). A config-file
value is dropped at load with ``RETIRED_CONFIG_KEY``.
"""

from __future__ import annotations

import ast
import logging
from datetime import datetime
from datetime import time as dt_time
from pathlib import Path

import pytest
import yaml

from gomoveshift.config import Config, DriveConfig
from gomoveshift.pipeline import schedule_constants as sc

from test_launch_anchor_gate import restore_schedule_module  # noqa: F401,E402  (fixture)

SEED_CADENCE = True  # opt out of conftest.legacy_test_cadence


PKG = Path(__file__).resolve().parents[1] / "gomoveshift"

# Waypost's seed (brand.yaml): ireland sun 14:00; usa tue 16:00 + fri 15:00.
SEED_DAYS = frozenset({1, 4, 6})


def test_the_drive_defaults_carry_no_long_form_day_or_time():
    d = DriveConfig()
    assert d.long_form_publish_days == []
    assert d.long_form_publish_time is None


def test_long_form_days_without_a_transport_are_waypost_days():
    assert sc.long_form_days(Config()) == SEED_DAYS


def test_a_bare_config_never_reads_a_decision_store_under_the_cwd(tmp_path, monkeypatch):
    """A bare ``Config()`` carries the relative ``./data/seo``: a decision store that happens
    to sit under the working directory must not decide a test's cadence."""
    from seo_manager.analytics import cadence_decisions

    from gomoveshift.pipeline import seo_slots
    from gomoveshift.pipeline.volume_source import DB_NAME

    store = tmp_path / "host-cwd" / "data" / "seo"
    store.mkdir(parents=True)
    cadence_decisions.append(
        store / DB_NAME,
        knob="lf_days",
        region="usa",
        value=2,
        prev=2,
        status="ok",
        source="policy",
        reason_code="TAIL_UP",
        basis="a host's own store",
        applied_at=datetime(2026, 1, 1, 12, 0),
        evidence={"dow": [1, 3]},
    )
    monkeypatch.chdir(tmp_path / "host-cwd")
    assert seo_slots.cadence_for(Config(), "usa").region("usa").dow == (1, 4)


def test_long_form_days_follow_a_waypost_decision(waypost_lf_days):
    config = Config()
    waypost_lf_days({"ireland": (6,), "usa": (2,)}, config)
    assert sc.long_form_days(config) == frozenset({2, 6})


def test_the_runs_transport_wins_over_waypost():
    config = Config()
    config.drive.long_form_publish_days = [4]
    assert sc.long_form_days(config) == frozenset({4})


def test_long_form_time_is_waypost_time_for_the_runs_day():
    config = Config()
    config.drive.long_form_publish_days = [4]  # usa fri
    assert sc.long_form_time(config) == dt_time(15, 0)
    config.drive.long_form_publish_days = [6]  # ireland sun
    assert sc.long_form_time(config) == dt_time(14, 0)


def test_long_form_time_without_a_transport_is_the_earliest_waypost_days_time():
    assert sc.long_form_time(Config()) == dt_time(16, 0)  # tue, the earliest Waypost day


def test_long_form_time_off_waypost_days_falls_back_to_the_first_regions_default():
    config = Config()
    config.drive.long_form_publish_days = [2]  # wed: nobody's day
    assert sc.long_form_time(config) == dt_time(14, 0)  # ireland, first region by name


def test_the_runs_time_transport_wins():
    config = Config()
    config.drive.long_form_publish_time = "09:30"
    assert sc.long_form_time(config) == dt_time(9, 30)


def test_resolve_lf_time_last_fallback_is_waypost_time():
    config = Config()
    assert sc.resolve_lf_time(config, "wed", None) == "16:00"  # earliest Waypost day, tue
    assert sc.resolve_lf_time(config, "tue", "08:00") == "08:00"


def test_import_time_constants_are_the_seed_not_a_config_default():
    assert sc.LONG_FORM_DAYS == SEED_DAYS
    assert sc.LONG_FORM_TIME == dt_time(16, 0)


@pytest.mark.usefixtures("restore_schedule_module")
def test_apply_config_mirrors_the_accessors(waypost_lf_days):
    config = Config()
    waypost_lf_days({"ireland": (6,), "usa": (2,)}, config)
    sc.apply_config(config)
    assert sc.LONG_FORM_DAYS == frozenset({2, 6})
    config.drive.long_form_publish_days = [6]
    sc.apply_config(config)
    assert sc.LONG_FORM_DAYS == frozenset({6})
    assert sc.LONG_FORM_TIME == dt_time(14, 0)


@pytest.mark.parametrize(
    "key,value",
    [("long_form_publish_days", [3]), ("long_form_publish_time", "08:00")],
)
def test_a_config_file_value_is_dropped_with_a_warning(tmp_path, caplog, key, value):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"drive": {key: value}}))
    with caplog.at_level(logging.WARNING, logger="gomoveshift.config"):
        cfg = Config.load(path)
    hits = [
        r
        for r in caplog.records
        if f"RETIRED_CONFIG_KEY key={key} (retired: Waypost decides)" in r.getMessage()
    ]
    assert len(hits) == 1
    assert getattr(cfg.drive, key) == ([] if key.endswith("days") else None)
    assert sc.long_form_days(cfg) == SEED_DAYS


def test_the_fingerprint_reads_the_accessors(waypost_lf_days):
    from types import SimpleNamespace

    from gomoveshift.pipeline.orchestrator_fingerprint import build_phase_d_fingerprint_inputs

    config = Config()
    state = SimpleNamespace(source_dir=None)
    before = build_phase_d_fingerprint_inputs(state, config=config)
    waypost_lf_days({"ireland": (6,), "usa": (2,)}, config)
    after = build_phase_d_fingerprint_inputs(state, config=config)
    assert before != after


# --- guard: nothing in GMS reads the field as a default ---------------------

_FIELDS = ("long_form_publish_days", "long_form_publish_time")
# The per-run transport is the field's only job: the drive handler writes it, Phase D reads
# the run's one picked day, and schedule_constants' accessors read it first.
_READERS = {
    "cli/handlers/drive.py",
    "pipeline/schedule_constants.py",
    "pipeline/drive_phases/phase_d_scheduling.py",
}


def _field_reads(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FIELDS:
            out.append(node.lineno)
        elif (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in _FIELDS
        ):
            out.append(node.lineno)
    return out


def test_guard_fails_if_gms_reads_long_form_publish_days_default(tmp_path):
    planted = tmp_path / "x.py"
    planted.write_text(
        "from gomoveshift.config import DriveConfig\n"
        "days = DriveConfig().long_form_publish_days\n"
        "t = getattr(cfg.drive, 'long_form_publish_time', None)\n",
        encoding="utf-8",
    )
    assert _field_reads(planted) == [2, 3]


def test_no_gms_module_outside_the_transport_readers_reads_the_fields():
    offenders = {}
    for path in sorted(PKG.rglob("*.py")):
        rel = path.relative_to(PKG).as_posix()
        if rel in _READERS:
            continue
        lines = _field_reads(path)
        if lines:
            offenders[rel] = lines
    assert not offenders, f"read Waypost's cadence via schedule_constants instead: {offenders}"


def test_the_guard_allowlist_has_no_stale_entries():
    for rel in _READERS:
        assert _field_reads(PKG / rel), f"{rel} no longer reads the fields; drop it from _READERS"
