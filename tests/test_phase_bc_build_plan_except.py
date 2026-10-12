"""``_build_drive_plan_from_state`` narrows its rebuild ``except``.

The bare ``except Exception`` used to swallow programming errors
(``AttributeError`` / ``TypeError`` / ``AssertionError``) at WARN and
rebuild LFs from possibly-corrupt segments. The except is now narrowed to
the expected transient/IO/parse/validation failures
(``FileNotFoundError`` / ``json.JSONDecodeError`` / ``ValueError``), which
degrade gracefully to ``None``; programming errors propagate.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.config import Config, DriveConfig
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _build_drive_plan_from_state,
)
from gomoveshift.pipeline.longform_assembler import LongFormAssembler

from _state_helpers import make_state


def _dense_scores(duration_sec: int) -> np.ndarray:
    n = int(duration_sec)
    arr = np.zeros((n, 11), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64)
    arr[:, 1] = 0.6
    arr[:, 2] = 0.6
    return arr


def _config() -> Config:
    cfg = Config()
    cfg.drive = DriveConfig(
        lf_selection_mode="best_moments",
        adaptive_threshold_enabled=False,
        min_lf_duration=480.0,
        min_segment_duration=30.0,
        min_motion_energy=0.0,
    )
    return cfg


def _state(tmp_path: Path):
    ep = tmp_path / "ep_00.mp4"
    ep.write_bytes(b"")
    return make_state(
        drive_scores=_dense_scores(1800),
        file_boundaries=[(0.0, 1800.0)],
        episode_files=[str(ep)],
        episode_dates=["2025-09-14"],
        region="ireland",
        gps_per_episode=None,
    )


@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError("missing score cache"),
        json.JSONDecodeError("bad", "doc", 0),
        ValueError("episode_dates length must match file_boundaries length"),
    ],
)
def test_expected_failures_degrade_to_none(tmp_path: Path, monkeypatch, exc):
    """An expected transient/IO/parse/validation failure returns ``None``
    (graceful) and is logged at WARN, not raised."""

    def _boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(LongFormAssembler, "build_drive_plan", _boom)
    state = _state(tmp_path)
    assert _build_drive_plan_from_state(_config(), state) is None


@pytest.mark.parametrize(
    "exc",
    [
        TypeError("unexpected keyword"),
        AttributeError("'NoneType' object has no attribute 'x'"),
        AssertionError("invariant broken"),
    ],
)
def test_programming_errors_propagate(tmp_path: Path, monkeypatch, exc):
    """A programming error must NOT be swallowed — rebuilding LFs from a
    state that triggered an ``AttributeError`` / ``TypeError`` /
    ``AssertionError`` would mask a real defect."""

    def _boom(*_a, **_k):
        raise exc

    monkeypatch.setattr(LongFormAssembler, "build_drive_plan", _boom)
    state = _state(tmp_path)
    with pytest.raises(type(exc)):
        _build_drive_plan_from_state(_config(), state)
