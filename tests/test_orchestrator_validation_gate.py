"""Tests for the DoD validator as the terminal gate of the drive orchestrator.

The pipeline wires
``tools.validate_pipeline_output.run_validation`` into the outermost
``DrivePipelineOrchestrator.run()`` boundary. These tests pin the contract:

* When the validator returns 13/13 PASS, ``run()`` returns a result with
  ``success=True``.
* When the validator reports any failure, ``run()`` raises
  :class:`PipelineValidationError` (which is NOT caught internally — the
  CLI entry point catches it and exits non-zero).
* ``--skip-validation`` (``config.drive.skip_validation=True``) bypasses
  the gate with a WARNING log and never calls the validator.
* Default for ``skip_validation`` is ``False`` — the gate is opt-out, not
  opt-in.

There is a single orchestrator path, so one wire-up at ``run()``'s end covers
every drive run.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.exceptions import PipelineValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_phase(orch: DrivePipelineOrchestrator) -> None:
    """Stub every phase method so run() doesn't do real work.

    Also disables the interactive ReviewGate so we don't trip on an empty
    synthetic workspace. We want the validator gate under test, not the
    upstream human review gates.
    """
    orch.config.drive.no_review = True

    def _noop(*_args, **_kwargs):
        return None

    for name in (
        "_phase_a_ingest_and_analyze",
        "_phase_b_process_files",
        "_phase_c_assemble",
        "_phase_d_metadata_and_publish",
    ):
        setattr(orch, name, _noop)


def _stub_review_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace ReviewGate methods with no-ops so synthetic tests don't
    trip on the "no episode files found" automated check.
    """
    from gomoveshift.pipeline import review_gate as rg

    monkeypatch.setattr(rg.ReviewGate, "gate_a", lambda *a, **kw: None)
    monkeypatch.setattr(rg.ReviewGate, "gate_b", lambda *a, **kw: None)
    monkeypatch.setattr(rg.ReviewGate, "gate_cd", lambda *a, **kw: None)


def _make_report(*, passed: int, failed: int):
    """Build a fake ValidationReport-alike stub."""
    report = MagicMock()
    report.passed_count = passed
    report.failed_count = failed
    report.checks = [MagicMock(passed=True) for _ in range(passed)] + [
        MagicMock(passed=False) for _ in range(failed)
    ]
    report.format_table.return_value = f"== DoD Validation == (passed={passed}, failed={failed})"
    return report


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------


def test_orchestrator_passes_when_validator_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """13/13 PASS → run() returns success=True."""
    _stub_review_gate(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    orch = DrivePipelineOrchestrator()
    _mock_phase(orch)

    with patch(
        "gomoveshift.pipeline.drive_orchestrator.run_validation",
        return_value=_make_report(passed=13, failed=0),
    ) as mock_v:
        result = orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="ireland",
        )
    assert mock_v.called, "run_validation must be invoked by the orchestrator"
    assert result.success is True
    assert result.errors == []


def test_orchestrator_raises_when_validator_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """9/13 PASS → run() raises PipelineValidationError."""
    _stub_review_gate(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    orch = DrivePipelineOrchestrator()
    _mock_phase(orch)

    with (
        patch(
            "gomoveshift.pipeline.drive_orchestrator.run_validation",
            return_value=_make_report(passed=9, failed=4),
        ),
        pytest.raises(PipelineValidationError),
    ):
        orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="ireland",
        )


def test_orchestrator_raise_message_includes_report_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The PipelineValidationError message contains the formatted table."""
    _stub_review_gate(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    orch = DrivePipelineOrchestrator()
    _mock_phase(orch)

    fake = _make_report(passed=10, failed=3)
    fake.format_table.return_value = (
        "== DoD Validation: demo ==\n[FAIL] 15 recording_date_per_lf — missing\nSummary: 10/13"
    )

    with (
        patch(
            "gomoveshift.pipeline.drive_orchestrator.run_validation",
            return_value=fake,
        ),
        pytest.raises(PipelineValidationError) as exc_info,
    ):
        orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="ireland",
        )

    msg = str(exc_info.value)
    assert "15 recording_date_per_lf" in msg
    assert "3 check" in msg  # count surfaced in the message


def test_skip_validation_flag_bypasses_gate_with_warning(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
):
    """skip_validation=True → validator NOT called, WARNING logged."""
    _stub_review_gate(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    cfg = Config()
    cfg.drive.skip_validation = True
    orch = DrivePipelineOrchestrator(config=cfg)
    _mock_phase(orch)

    with (
        caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_orchestrator"),
        patch("gomoveshift.pipeline.drive_orchestrator.run_validation") as mock_v,
    ):
        result = orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="ireland",
        )
    mock_v.assert_not_called()
    assert result.success is True
    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "validation" in r.message.lower()
    ]
    assert warnings, "expected a WARNING log when skip_validation=True"


def test_skip_validation_false_is_default():
    """Default config has skip_validation=False (gate is opt-out)."""
    cfg = Config()
    assert cfg.drive.skip_validation is False


# test_per_day_orchestrator_uses_same_gate removed — per_day path was
# deleted. The single surviving path is already covered by
# test_validation_gate_not_swallowed_by_run_try_except and
# test_skip_validation_is_not_default_opt_in below.


def test_validation_gate_not_swallowed_by_run_try_except(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """PipelineValidationError must propagate out of run() — never suppressed.

    The orchestrator has a broad ``except Exception`` around the phase loop
    for checkpointing. This test pins that the validator gate is OUTSIDE
    that block so the exception reaches the caller.
    """
    _stub_review_gate(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    orch = DrivePipelineOrchestrator()
    _mock_phase(orch)

    with (
        patch(
            "gomoveshift.pipeline.drive_orchestrator.run_validation",
            return_value=_make_report(passed=12, failed=1),
        ),
        pytest.raises(PipelineValidationError),
    ):
        orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="ireland",
        )


def test_skip_validation_is_not_default_opt_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A vanilla orchestrator invocation with no skip flags calls the validator."""
    _stub_review_gate(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    orch = DrivePipelineOrchestrator()
    _mock_phase(orch)

    with patch(
        "gomoveshift.pipeline.drive_orchestrator.run_validation",
        return_value=_make_report(passed=13, failed=0),
    ) as mock_v:
        orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="ireland",
        )
    mock_v.assert_called_once()


def test_orchestrator_threads_superlong_ceiling_into_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Superlong ceiling is computed from live config, not hardcoded."""
    _stub_review_gate(monkeypatch)
    source = tmp_path / "source"
    source.mkdir()
    cfg = Config()
    cfg.drive.superlong_part_max_duration_sec = 100
    cfg.drive.superlong_split_grace = 0.5
    orch = DrivePipelineOrchestrator(config=cfg)
    _mock_phase(orch)

    with patch(
        "gomoveshift.pipeline.drive_orchestrator.run_validation",
        return_value=_make_report(passed=13, failed=0),
    ) as mock_v:
        orch.run(source_dir=source, output_base=tmp_path / "output", region="ireland")
    assert mock_v.call_args.kwargs["superlong_max_lf_duration"] == pytest.approx(150.0)
