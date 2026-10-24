"""An under-spec machine is told so beside the plan's ETA (advisory, never a refusal)."""

from unittest.mock import MagicMock

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import orchestrator_preflight as op


def test_advisory_names_each_shortfall():
    msg = op.machine_spec_advisory(cpu_cores=2, total_ram_gb=4.0)
    assert msg is not None
    assert "2 CPU cores" in msg and "4.0 GB RAM" in msg
    assert f"{op.RECOMMENDED_CPU_CORES}+" in msg and f"{op.RECOMMENDED_RAM_GB:g}+" in msg


def test_advisory_names_only_the_resource_that_falls_short():
    msg = op.machine_spec_advisory(cpu_cores=16, total_ram_gb=4.0)
    assert msg is not None and "RAM" in msg and "CPU" not in msg


def test_no_advisory_at_or_above_the_recommendation():
    assert (
        op.machine_spec_advisory(
            cpu_cores=op.RECOMMENDED_CPU_CORES, total_ram_gb=op.RECOMMENDED_RAM_GB
        )
        is None
    )


def test_no_advisory_for_an_8gb_machine_that_reports_slightly_less():
    # Firmware, kernel and an integrated GPU reserve some; "8 GB" reports about 7.6-7.8 GiB.
    assert op.machine_spec_advisory(cpu_cores=8, total_ram_gb=7.7) is None


def test_advisory_still_fires_for_a_clearly_smaller_machine():
    assert op.machine_spec_advisory(cpu_cores=8, total_ram_gb=6.0) is not None


def test_no_advisory_when_the_machine_cannot_be_read():
    assert op.machine_spec_advisory(cpu_cores=None, total_ram_gb=None) is None


def _run(tmp_path, monkeypatch, *, spec, dry_run):
    monkeypatch.setattr(op, "print_plan", lambda *a, **k: print("PLAN-ETA-LINE"))
    monkeypatch.setattr(op, "_calibration_for_plan", MagicMock(return_value=None))
    monkeypatch.setattr(op, "_phase_a_done", lambda state: True)
    monkeypatch.setattr(op, "_readiness_blockers", lambda **k: [])
    monkeypatch.setattr(op, "_machine_spec", lambda: spec)
    return op.run_preflight_and_prompt(
        state=MagicMock(phases_completed=["A"]),
        stages_dir=tmp_path,
        dry_run=dry_run,
        force=True,
        source_dir=tmp_path,
        project_dir=tmp_path,
        start_time=0.0,
        stages_completed=[],
        config=Config(),
        emit_summary=MagicMock(),
    )


@pytest.mark.parametrize("dry_run", [True, False])
def test_low_spec_advisory_is_printed_after_the_plan(tmp_path, monkeypatch, capsys, dry_run):
    _run(tmp_path, monkeypatch, spec=(2, 4.0), dry_run=dry_run)
    out = capsys.readouterr().out
    assert "below the recommended" in out
    assert out.index("PLAN-ETA-LINE") < out.index("below the recommended")


def test_adequate_machine_prints_no_advisory(tmp_path, monkeypatch, capsys):
    _run(tmp_path, monkeypatch, spec=(16, 64.0), dry_run=True)
    assert "below the recommended" not in capsys.readouterr().out
