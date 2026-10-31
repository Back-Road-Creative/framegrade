"""Render-readiness gate: every cheap blocker is reported together, before any GPU work.

Reproduces two failed evidence attempts (synthetic layout, no private data):
attempt a burned a calibration encode and then refused on a missing GoPro parser;
attempt d burned one and then refused on the source-filesystem check. Both must
now refuse before ``ensure_calibrated`` runs, in one aggregated report.
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import orchestrator_preflight as op
from gomoveshift.pipeline import preflight


def _ep(day: str, secs: float):
    y, m, d = (int(x) for x in day.split("-"))
    return SimpleNamespace(
        index=1,
        creation_time=datetime(y, m, d, 12, 0),
        date=day,
        total_duration=secs,
        gpx_path=None,
    )


def _run(tmp_path, monkeypatch, *, preflight_fn, episodes=None, cfg=None, dry_run=False):
    calls: list[str] = []
    cfg = cfg or Config()
    episodes = episodes or [_ep("2026-03-04", 4000.0)]
    monkeypatch.setattr(
        op, "create_ingest_plan", lambda *a, **k: SimpleNamespace(episodes=episodes)
    )
    monkeypatch.setattr(op, "print_plan", lambda *a, **k: calls.append("plan"))

    def calibrate(**k):
        calls.append("calibrate-dry" if k["dry_run"] else "calibrate")

    monkeypatch.setattr(op, "_calibration_for_plan", calibrate)
    monkeypatch.setattr(preflight, "cleanup_stale_staging", lambda *a, **k: None)

    def pf(**k):
        calls.append("preflight")
        return preflight_fn(**k)

    monkeypatch.setattr(op, "run_preflight", pf)
    stages = tmp_path / "proj" / "_stages"
    stages.mkdir(parents=True)
    res = op.run_preflight_and_prompt(
        state=MagicMock(phases_completed=[]),
        stages_dir=stages,
        dry_run=dry_run,
        force=True,
        source_dir=tmp_path,
        project_dir=tmp_path,
        start_time=0.0,
        stages_completed=[],
        config=cfg,
        emit_summary=MagicMock(),
    )
    return res, calls


def _raise(msg):
    def fn(**k):
        raise RuntimeError(msg)

    return fn


class TestRefusalBeforeGpuWork:
    def test_missing_gopro_parser_refuses_before_calibration(self, tmp_path, monkeypatch):
        # Attempt a: calibration encode ran, then "gopro-overlay not installed".
        res, calls = _run(
            tmp_path, monkeypatch, preflight_fn=_raise("gopro-overlay is not installed")
        )
        assert res is not None and res.success is False
        assert "gopro-overlay" in res.errors[0]
        assert "calibrate" not in calls

    def test_unreadable_source_refuses_before_calibration(self, tmp_path, monkeypatch):
        # Attempt d: calibration encode ran, then the source-filesystem refusal.
        res, calls = _run(
            tmp_path, monkeypatch, preflight_fn=_raise("Source footage is not readable")
        )
        assert res.success is False
        assert calls == ["preflight"]

    def test_part_min_refusal_aggregates_with_preflight_blockers(self, tmp_path, monkeypatch):
        cfg = Config()
        cfg.drive.full_drive_part_min_sec = 3600
        cfg.drive.full_drive_part_max_sec = 5400
        # Config() resolved the LF policy before the bounds existed; re-resolve so
        # the operator's bounds select full_drive, where the floor is active.
        cfg.drive.lf_selection_mode = None
        cfg.drive._resolve_lf_policy()
        res, calls = _run(
            tmp_path,
            monkeypatch,
            preflight_fn=_raise("gopro-overlay is not installed"),
            episodes=[_ep("2026-03-04", 1025.0)],
            cfg=cfg,
        )
        assert res.success is False
        msg = res.errors[0]
        assert "RENDER BLOCKED" in msg
        assert "2026-03-04" in msg and "gopro-overlay" in msg
        assert "calibrate" not in calls


class TestReadyReport:
    def test_ready_report_precedes_calibration(self, tmp_path, monkeypatch, caplog):
        caplog.set_level("INFO")
        res, calls = _run(tmp_path, monkeypatch, preflight_fn=lambda **k: None)
        assert res is None
        assert calls[:2] == ["preflight", "calibrate"]
        assert any("READY TO RENDER" in r.getMessage() for r in caplog.records)

    def test_dry_run_still_never_encodes(self, tmp_path, monkeypatch):
        res, calls = _run(tmp_path, monkeypatch, preflight_fn=lambda **k: None, dry_run=True)
        assert res.success is True
        assert "calibrate" not in calls and "preflight" not in calls


class TestRunPreflightAggregates:
    def test_independent_blockers_are_reported_together(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        src.mkdir()
        (src / "GX010001.MP4").symlink_to(tmp_path / "gone.MP4")
        (src / "GX020001.MP4").write_bytes(b"x" * 4096)
        monkeypatch.setattr(
            preflight,
            "check_gopro_telemetry_parser",
            _raise_any("gopro-overlay is not installed"),
        )
        monkeypatch.setattr(preflight, "check_disk_space", _raise_any("Insufficient disk space"))
        _stub_host_measurements(monkeypatch)
        with pytest.raises(preflight.PreflightBlockedError) as exc:
            preflight.run_preflight(
                source_dir=src,
                output_dir=tmp_path / "out",
                staging_dir=tmp_path / "stg",
                require_tools=None,
                ingest_mode="gopro",
            )
        msg = str(exc.value)
        assert msg.startswith("RENDER BLOCKED: 3 blockers")
        assert "gopro-overlay" in msg and "missing target" in msg and "Insufficient disk" in msg
        assert len(exc.value.blockers) == 3

    def test_blocked_is_a_runtime_error(self):
        assert issubclass(preflight.PreflightBlockedError, RuntimeError)

    def test_clear_run_returns_checked_steps(self, tmp_path, monkeypatch):
        monkeypatch.setattr(preflight, "check_disk_space", lambda *a, **k: None)
        _stub_host_measurements(monkeypatch)
        src = tmp_path / "src"
        src.mkdir()
        (src / "GX010001.MP4").write_bytes(b"x" * 4096)
        report = preflight.run_preflight(
            source_dir=src,
            output_dir=tmp_path / "out",
            staging_dir=tmp_path / "stg",
            require_tools=None,
        )
        assert "source footage" in report.checked and "output disk" in report.checked


def _stub_host_measurements(monkeypatch):
    """Cache-disk, host-volume and RAM figures are the host's, not the test's.
    Without this stub a CI runner whose tmp volume is below the staging floor
    adds an "Insufficient cache space" blocker the test never set up."""
    monkeypatch.setattr(preflight, "check_wsl_host_volume_space", lambda *a, **k: None)
    monkeypatch.setattr(preflight, "estimate_peak_cache_gb", lambda *a, **k: 0.0)
    monkeypatch.setattr(preflight, "check_ram", lambda *a, **k: None)


def _raise_any(msg):
    def fn(*a, **k):
        raise RuntimeError(msg)

    return fn
