"""A day just under the part minimum ships as ONE long-form instead of being refused.

``full_drive_short_day_min_sec`` lowers the per-day floor: a day whose kept
footage is in ``[short_day_min, part_min)`` is one part holding all its kept
ranges. Below it the day is still refused; with the key unset nothing changes.
The planner, the phase-A GPS-gap pre-gate and the preflight refusal message all
use the same floor so a pre-gate never refuses a day the planner would accept.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.config import Config, DriveConfig
from gomoveshift.pipeline import orchestrator_preflight as op
from gomoveshift.pipeline.drive_phases.phase_a import _assert_days_clear_gps_gap_floor
from gomoveshift.pipeline.regular_lf_planner import (
    InsufficientFootageError,
    plan_full_drive_parts,
)

DAY = "2026-03-04"
PART_MIN, PART_MAX, SHORT_MIN = 3600, 5400, 2700


def _scores(n):
    arr = np.zeros((n, 11))
    arr[:, 0] = np.arange(n)
    arr[:, 1] = 0.5
    arr[:, 2] = 0.6
    return arr


def _plan(n, short_min):
    return plan_full_drive_parts(
        _scores(n),
        min_duration_sec=480,
        full_drive_max_duration_sec=7200,
        stop_mask=np.zeros(n, dtype=bool),
        part_min_sec=PART_MIN,
        part_max_sec=PART_MAX,
        short_day_min_sec=short_min,
        day=DAY,
    )


# --- planner ---------------------------------------------------------------


def test_planner_short_day_at_floor_is_one_part():
    parts = _plan(2889, SHORT_MIN)
    assert len(parts) == 1
    assert sum(r.duration_sec for r in parts[0]) == pytest.approx(2889)


def test_planner_day_exactly_at_short_floor_is_one_part():
    assert len(_plan(SHORT_MIN, SHORT_MIN)) == 1


def test_planner_day_under_short_floor_still_raises():
    with pytest.raises(InsufficientFootageError, match=DAY) as exc:
        _plan(2600, SHORT_MIN)
    msg = str(exc.value)
    assert "2600s" in msg and "full_drive_short_day_min_sec=2700s" in msg
    assert "never padded" in msg


def test_planner_key_unset_still_raises_at_2889():
    with pytest.raises(InsufficientFootageError, match="full_drive_part_min_sec=3600s"):
        _plan(2889, None)


def test_planner_short_day_does_not_call_the_splitter(monkeypatch):
    from gomoveshift.pipeline import regular_lf_planner as rlp

    def boom(*a, **k):
        raise AssertionError("split_ranges_balanced must not run for a short day")

    monkeypatch.setattr(rlp, "split_ranges_balanced", boom)
    assert len(_plan(2889, SHORT_MIN)) == 1


def test_planner_short_day_still_honours_the_480s_minimum():
    # 400 s floor-lowered day: under the 480 s LF minimum, so still an error.
    with pytest.raises(ValueError, match="minimum LF duration"):
        plan_full_drive_parts(
            _scores(400),
            min_duration_sec=480,
            full_drive_max_duration_sec=7200,
            stop_mask=np.zeros(400, dtype=bool),
            part_min_sec=PART_MIN,
            part_max_sec=PART_MAX,
            short_day_min_sec=300,
            day=DAY,
        )


def test_planner_longer_days_split_as_before():
    parts = _plan(9000, SHORT_MIN)
    assert len(parts) == 2


# --- phase A GPS-gap pre-gate ----------------------------------------------


def _drive(short_min):
    return SimpleNamespace(
        lf_selection_mode="full_drive",
        full_drive_part_min_sec=PART_MIN,
        full_drive_part_max_sec=PART_MAX,
        full_drive_short_day_min_sec=short_min,
    )


def _gps(seconds, start=1_000_000.0):
    tps = [GPSTrackpoint(35.0 + 1e-5 * i, -80.0, None, start + i) for i in range(int(seconds))]
    return (GPSData(trackpoints=tps, creator=None), start)


def _gate(short_min, seconds):
    _assert_days_clear_gps_gap_floor(
        _drive(short_min),
        [(0.0, float(seconds))],
        [float(seconds)],
        [DAY],
        [_gps(seconds)],
    )


def test_gps_gate_passes_a_day_the_planner_accepts():
    _gate(SHORT_MIN, 2889)


def test_gps_gate_refuses_under_the_short_floor():
    with pytest.raises(InsufficientFootageError, match="full_drive_short_day_min_sec=2700s"):
        _gate(SHORT_MIN, 2600)


def test_gps_gate_key_unset_refuses_at_2889():
    with pytest.raises(InsufficientFootageError, match="full_drive_part_min_sec=3600s"):
        _gate(None, 2889)


# --- preflight / backlog refusal message ------------------------------------


def _cfg(short_min):
    cfg = Config()
    cfg.drive.full_drive_part_min_sec = PART_MIN
    cfg.drive.full_drive_part_max_sec = PART_MAX
    cfg.drive.full_drive_short_day_min_sec = short_min
    cfg.drive.lf_selection_mode = None
    cfg.drive._resolve_lf_policy()
    return cfg


def _message(monkeypatch, tmp_path, secs, short_min):
    ep = SimpleNamespace(
        index=1,
        creation_time=datetime(2026, 3, 4, 12, 0),
        date=DAY,
        total_duration=secs,
        gpx_path=None,
    )
    monkeypatch.setattr(op, "create_ingest_plan", lambda *a, **k: SimpleNamespace(episodes=[ep]))
    return op.day_under_part_min_message(_cfg(short_min), tmp_path)


def test_preflight_accepts_a_short_day_at_or_above_the_short_floor(monkeypatch, tmp_path):
    assert _message(monkeypatch, tmp_path, 2889.0, SHORT_MIN) is None


def test_preflight_refuses_under_the_short_floor(monkeypatch, tmp_path):
    msg = _message(monkeypatch, tmp_path, 2600.0, SHORT_MIN)
    assert msg and "full_drive_short_day_min_sec=2700s" in msg and "never padded" in msg


def test_preflight_key_unset_refuses_at_2889(monkeypatch, tmp_path):
    msg = _message(monkeypatch, tmp_path, 2889.0, None)
    assert msg and "full_drive_part_min_sec=3600s" in msg


# --- config ----------------------------------------------------------------


@pytest.mark.parametrize("bad", [0, -1, PART_MIN + 1])
def test_driveconfig_rejects_bad_short_day_min(bad):
    with pytest.raises(ValueError, match="full_drive_short_day_min_sec"):
        DriveConfig(
            full_drive_part_min_sec=PART_MIN,
            full_drive_part_max_sec=PART_MAX,
            full_drive_short_day_min_sec=bad,
        )


def test_short_day_min_without_part_bounds_is_inert():
    # "Comment both part bounds out for one LF per day" must still load: the
    # short-day floor only lowers a part floor, so with none set it does nothing.
    d = DriveConfig(full_drive_short_day_min_sec=SHORT_MIN)
    assert d.full_drive_short_day_min_sec == SHORT_MIN


def test_clearing_part_bounds_keeps_the_shipped_drive_config_valid():
    from pathlib import Path

    cfg = Config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    drive = cfg.drive
    drive.full_drive_part_min_sec = drive.full_drive_part_max_sec = None
    drive.__class__.model_validate(drive.model_dump())  # the drive CLI's re-validate


def test_driveconfig_accepts_valid_short_day_min():
    d = DriveConfig(
        full_drive_part_min_sec=PART_MIN,
        full_drive_part_max_sec=PART_MAX,
        full_drive_short_day_min_sec=PART_MIN,
    )
    assert d.full_drive_short_day_min_sec == PART_MIN
    assert DriveConfig().full_drive_short_day_min_sec is None


def test_shipped_config_ships_a_45_to_59_min_day_as_one_part():
    from pathlib import Path

    cfg = Config.load(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg.drive.full_drive_short_day_min_sec == 2700
