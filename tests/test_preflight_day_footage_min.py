"""Upstream gate: a full_drive day under the part minimum is refused before ingest."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from gomoveshift.config import Config
from gomoveshift.pipeline import orchestrator_preflight as op


def _ep(day: str, secs: float):
    y, m, d = (int(x) for x in day.split("-"))
    return SimpleNamespace(
        index=1,
        creation_time=datetime(y, m, d, 12, 0),
        date=day,
        total_duration=secs,
        gpx_path=None,
    )


def _cfg(lo, hi):
    cfg = Config()
    cfg.drive.full_drive_part_min_sec = lo
    cfg.drive.full_drive_part_max_sec = hi
    return cfg


def _run(tmp_path, monkeypatch, episodes, cfg):
    monkeypatch.setattr(
        op, "create_ingest_plan", lambda *a, **k: SimpleNamespace(episodes=episodes)
    )
    monkeypatch.setattr(op, "print_plan", lambda *a, **k: None)
    pf = MagicMock()
    monkeypatch.setattr(op, "run_preflight", pf)
    cal = MagicMock()
    monkeypatch.setattr(op, "_calibration_for_plan", cal)
    monkeypatch.setattr("gomoveshift.pipeline.preflight.cleanup_stale_staging", MagicMock())
    res = op.run_preflight_and_prompt(
        state=MagicMock(phases_completed=[]),
        stages_dir=tmp_path,
        dry_run=False,
        force=True,
        source_dir=tmp_path,
        project_dir=tmp_path,
        start_time=0.0,
        stages_completed=[],
        config=cfg,
        emit_summary=MagicMock(),
    )
    return res, pf, cal


def test_day_under_part_min_is_refused_before_ingest(tmp_path, monkeypatch):
    eps = [_ep("2026-03-04", 1025.0), _ep("2026-03-04", 1025.0)]
    res, _pf, cal = _run(tmp_path, monkeypatch, eps, _cfg(3600, 5400))
    assert res is not None and res.success is False
    msg = res.errors[0]
    assert "2026-03-04" in msg and "2050s" in msg and "full_drive_part_min_sec=3600s" in msg
    assert "never padded" in msg
    # Refused before the calibration encode, the run's first GPU work; the
    # cheap pre-flight checks run too so every blocker is reported together.
    cal.assert_not_called()


def test_day_at_or_above_part_min_passes(tmp_path, monkeypatch):
    eps = [_ep("2026-03-04", 1800.0), _ep("2026-03-04", 1800.0)]
    res, pf, _cal = _run(tmp_path, monkeypatch, eps, _cfg(3600, 5400))
    assert res is None
    pf.assert_called_once()


def test_gate_inactive_when_bounds_unset(tmp_path, monkeypatch):
    eps = [_ep("2026-03-04", 100.0)]
    res, pf, _cal = _run(tmp_path, monkeypatch, eps, _cfg(None, None))
    assert res is None
    pf.assert_called_once()


def test_adjacent_day_clock_skew_is_not_refused(tmp_path, monkeypatch):
    # One real day split across two camera-clock dates must not be refused.
    eps = [_ep("2026-03-04", 2000.0), _ep("2026-03-05", 2000.0)]
    res, _pf, _cal = _run(tmp_path, monkeypatch, eps, _cfg(3600, 5400))
    assert res is None
