"""Tests for the eager-import contract on state_invariants.

The orchestrator's :meth:`DrivePipelineOrchestrator.run` MUST resolve every
``gomoveshift.pipeline.state_invariants`` symbol at startup — before
phase_a runs — so that a stale ``sys.modules`` entry (left behind by a
mid-run ``.py`` merge) raises ``ImportError`` immediately rather than
mid-phase, after hours of compute.

Turns a runtime crash into a startup error.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import drive_orchestrator
from gomoveshift.pipeline.drive_orchestrator import (
    DrivePipelineOrchestrator,
    _eager_import_state_invariants,
)


# ── Pure unit tests on the helper ──────────────────────────────────────


def test_eager_import_resolves_all_required_symbols():
    """Happy path: every expected symbol is importable from state_invariants."""
    # Must not raise. Asserts the public-symbol contract is satisfied.
    _eager_import_state_invariants()


def test_eager_import_raises_on_missing_symbol(monkeypatch):
    """Simulate a stale sys.modules where a required symbol vanished."""
    from gomoveshift.pipeline import state_invariants as _si

    # Drop one of the required symbols and ensure the helper raises
    # ImportError (not AttributeError, not silent).
    monkeypatch.delattr(_si, "assert_state_invariants", raising=True)
    with pytest.raises(ImportError) as excinfo:
        _eager_import_state_invariants()
    assert "assert_state_invariants" in str(excinfo.value)
    assert "stale" in str(excinfo.value).lower()


def test_eager_import_lists_all_missing_symbols(monkeypatch):
    """When multiple required symbols are missing, all are reported."""
    from gomoveshift.pipeline import state_invariants as _si

    monkeypatch.delattr(_si, "assert_state_invariants", raising=True)
    monkeypatch.delattr(_si, "STATE_INVARIANT_ERROR_PREFIX", raising=True)
    with pytest.raises(ImportError) as excinfo:
        _eager_import_state_invariants()
    msg = str(excinfo.value)
    assert "assert_state_invariants" in msg
    assert "STATE_INVARIANT_ERROR_PREFIX" in msg


# ── Orchestrator wiring: helper runs BEFORE phase_a ─────────────────────


def test_run_calls_eager_import_before_phase_a(tmp_path):
    """Orchestrator.run() invokes _eager_import_state_invariants
    before any phase work. We assert call ordering: the eager-import
    helper must be invoked AT LEAST ONCE, and PhaseA's run must not
    have been called before it."""
    call_order: list[str] = []

    source_dir = tmp_path / "src"
    source_dir.mkdir()
    output_base = tmp_path / "out"
    output_base.mkdir()

    def _record_eager():
        call_order.append("eager_import")

    def _record_phase_a(self, state):
        call_order.append("phase_a")

    with (
        patch.object(
            drive_orchestrator,
            "_eager_import_state_invariants",
            side_effect=_record_eager,
        ),
        patch.object(
            DrivePipelineOrchestrator,
            "_phase_a_ingest_and_analyze",
            autospec=True,
            side_effect=_record_phase_a,
        ),
        patch.object(
            DrivePipelineOrchestrator,
            "_phase_bc_day_first",
            autospec=True,
        ),
        patch.object(
            DrivePipelineOrchestrator,
            "_phase_d_metadata_and_publish",
            autospec=True,
        ),
        patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate"),
        patch("gomoveshift.pipeline.orchestrator_preflight.print_plan"),
        patch("gomoveshift.pipeline.orchestrator_preflight.run_preflight"),
        patch(
            "gomoveshift.pipeline.drive_orchestrator.run_validation",
            return_value=MagicMock(failures=[], passes=[]),
        ),
    ):
        orch = DrivePipelineOrchestrator()
        try:
            orch.run(
                source_dir=source_dir,
                output_base=output_base,
                fresh=True,
                force=True,
                skip_upload=True,
                skip_csv=True,
            )
        except Exception:
            # We only care about the ordering of the early-startup
            # eager-import call vs phase_a; downstream failures (e.g.
            # missing real ingest data) are not the contract under test.
            pass

    assert "eager_import" in call_order, (
        f"_eager_import_state_invariants was never called; observed: {call_order!r}"
    )
    if "phase_a" in call_order:
        assert call_order.index("eager_import") < call_order.index("phase_a"), (
            f"eager_import must run before phase_a; observed: {call_order!r}"
        )


def test_run_fails_fast_when_state_invariants_import_broken(tmp_path):
    """Negative test: when the eager-import raises ImportError (simulating
    a stale sys.modules from a mid-run code merge), orchestrator.run()
    surfaces it at startup — phase_a is never reached."""
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    output_base = tmp_path / "out"
    output_base.mkdir()

    phase_a_called = False

    def _record_phase_a(self, state):
        nonlocal phase_a_called
        phase_a_called = True

    with (
        patch.object(
            drive_orchestrator,
            "_eager_import_state_invariants",
            side_effect=ImportError(
                "state_invariants module is missing required symbols "
                "['assert_state_invariants'] — sys.modules cache is stale"
            ),
        ),
        patch.object(
            DrivePipelineOrchestrator,
            "_phase_a_ingest_and_analyze",
            autospec=True,
            side_effect=_record_phase_a,
        ),
    ):
        orch = DrivePipelineOrchestrator()
        with pytest.raises(ImportError) as excinfo:
            orch.run(
                source_dir=source_dir,
                output_base=output_base,
                fresh=True,
                force=True,
            )

    assert "stale" in str(excinfo.value).lower()
    assert not phase_a_called, (
        "phase_a must NOT run when the eager-import contract fails — "
        "the whole point of the contract is to surface the ImportError at "
        "startup, not after hours of phase work."
    )
