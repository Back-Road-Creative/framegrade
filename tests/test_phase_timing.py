"""A5: phase-timing recorder and preflight's use of recorded rates (tmp records only)."""

from __future__ import annotations

import json

from gomoveshift.pipeline import phase_timing, preflight_plan
from gomoveshift.pipeline.preflight_plan import PlanRow


def _write(path, phases):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"phase_timings": phases}))


def _rows():
    return [
        PlanRow("Phase A", "RUN", "fresh"),
        PlanRow("Day 2026-02-22", "RUN", "fresh"),
        PlanRow("Day 2026-02-23", "RUN", "fresh"),
        PlanRow("Phase D", "RUN", "fresh"),
    ]


class TestRecorder:
    def test_record_shape(self, tmp_path):
        path = tmp_path / "_stages" / phase_timing.RECORD_NAME
        rec = phase_timing.PhaseTimingRecorder(path)
        rec.add("phase_a", 1000.0, 1010.0, frames=5000)
        rec.add("phase_bc_day_first", 1010.0, 1310.0, units=3, unit="day")
        data = json.loads(path.read_text())
        a = data["phase_timings"]["phase_a"]
        assert a["wall_s"] == 10.0
        assert a["frames"] == 5000
        assert a["frames_per_s"] == 500.0
        assert a["start"].startswith("1970-01-01T00:16:40")
        assert a["end"].startswith("1970-01-01T00:16:50")
        bc = data["phase_timings"]["phase_bc_day_first"]
        assert (bc["units"], bc["unit"], bc["wall_s"]) == (3, "day", 300.0)
        assert "frames_per_s" not in bc

    def test_context_manager_records_success_not_failure(self, tmp_path):
        path = tmp_path / phase_timing.RECORD_NAME
        rec = phase_timing.PhaseTimingRecorder(path)
        with rec.phase("phase_a") as extra:
            extra["frames"] = 10
        try:
            with rec.phase("phase_d"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        names = set(json.loads(path.read_text())["phase_timings"])
        assert names == {"phase_a"}

    def test_resume_merges_existing_record(self, tmp_path):
        path = tmp_path / phase_timing.RECORD_NAME
        phase_timing.PhaseTimingRecorder(path).add("phase_a", 0.0, 5.0)
        phase_timing.PhaseTimingRecorder(path).add("phase_d", 5.0, 9.0)
        assert set(json.loads(path.read_text())["phase_timings"]) == {"phase_a", "phase_d"}


class TestRecordedRates:
    def test_median_per_unit(self, tmp_path):
        for i, wall in enumerate((100.0, 300.0, 200.0)):
            _write(
                tmp_path / f"d{i}" / "_stages" / phase_timing.RECORD_NAME,
                {"phase_bc_day_first": {"wall_s": wall, "units": 2}},
            )
        rates = phase_timing.recorded_rates(
            phase_timing.discover_record_paths(tmp_path / "d0" / "_stages", siblings=True)
        )
        assert len(phase_timing.discover_record_paths(tmp_path / "d0" / "_stages")) == 1
        assert rates["phase_bc_day_first"] == {"sec_per_unit": 100.0, "samples": 3}

    def test_no_records_no_rates(self, tmp_path):
        assert phase_timing.recorded_rates([tmp_path / "missing.json"]) == {}


class TestPreflightUsesRates:
    def test_recorded_rates_replace_constants(self):
        rates = {
            "phase_a": {"sec_per_unit": 1000.0, "samples": 2},
            "phase_bc_day_first": {"sec_per_unit": 40000.0, "samples": 2},
            "phase_d": {"sec_per_unit": 500.0, "samples": 2},
        }
        total, bd = preflight_plan._estimate_wall_clock(_rows(), None, None, rates)
        assert bd["phase_a"] == 1000
        assert bd["phase_d"] == 500
        assert bd["master"] == 80000
        assert bd["lfs"] == bd["shorts"] == bd["superlong"] == 0
        assert total == 81500

    def test_fallback_without_rates(self):
        with_none = preflight_plan._estimate_wall_clock(_rows(), None, None, None)
        default = preflight_plan._estimate_wall_clock(_rows(), None, None)
        assert with_none == default
        assert with_none[1]["phase_a"] == preflight_plan._PHASE_A_RUN_SEC

    def test_build_plan_reports_source(self, tmp_path):
        stages = tmp_path / "drive" / "_stages"
        stages.mkdir(parents=True)
        plan = preflight_plan.build_plan(None, stages, calibration=None)
        assert plan.rate_source == "built-in constants"
        assert any("built-in constants" in ln for ln in preflight_plan._render_plan_lines(plan))
        _write(
            stages / phase_timing.RECORD_NAME,
            {"phase_a": {"wall_s": 90.0, "units": None}},
        )
        plan = preflight_plan.build_plan(None, stages, calibration=None)
        assert plan.rate_source == "recorded rates (1 phase, 1 run min)"
        assert any("recorded rates" in ln for ln in preflight_plan._render_plan_lines(plan))
