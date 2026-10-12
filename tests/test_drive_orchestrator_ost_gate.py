"""Tests for the OST audit gate's interaction with DrivePipelineOrchestrator.

When ``run_day_pipeline`` raises :class:`OstAuditPendingError` from inside
``phase_bc_day_first``, the orchestrator must catch it cleanly:

* exit with ``success=True`` (the operator was told what to review),
* leave ``phase_bc_day_first`` OUT of ``stages_completed`` (so resume
  re-checks the gate), but
* keep ``phase_a`` in ``stages_completed`` (already done, idempotent).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError


@patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
@patch(
    "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_d_metadata_and_publish"
)
@patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
@patch(
    "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_a_ingest_and_analyze"
)
def test_orchestrator_catches_ost_audit_pending_exits_success_true(
    mock_a, mock_bc, mock_d, mock_rg, tmp_path
):
    """OstAuditPendingError → success=True, phase_a in stages_completed,
    phase_bc_day_first NOT in stages_completed, phase_d NEVER called."""

    def _a_side_effect(state):
        state["episode_files"] = ["/mock/ep0.mp4"]
        # Parallel to episode_files — empty-with-episodes now hard-fails
        # the OST gate.
        state["episode_locations"] = ["Mock Town, SC"]
        return {}

    def _bc_side_effect(state):
        # Simulate run_day_pipeline raising the audit-pending exception.
        raise OstAuditPendingError(
            "OST audit pending — review /tmp/out/ost_audit_2025-09-14.json "
            "and create ost_audit_2025-09-14.confirmed (or pass "
            "--ost-audit-confirmed) to proceed."
        )

    mock_a.side_effect = _a_side_effect
    mock_bc.side_effect = _bc_side_effect

    source = tmp_path / "source"
    source.mkdir()

    orch = DrivePipelineOrchestrator()
    result = orch.run(
        source_dir=source,
        output_base=tmp_path / "output",
        region="ireland",
        force=True,  # skip preflight prompt
    )

    assert result.success is True
    assert "phase_a" in result.stages_completed
    assert "phase_bc_day_first" not in result.stages_completed
    assert "phase_d" not in result.stages_completed
    mock_d.assert_not_called()


@patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
@patch(
    "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_d_metadata_and_publish"
)
@patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
@patch(
    "gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_a_ingest_and_analyze"
)
def test_orchestrator_does_not_treat_ost_audit_pending_as_error(
    mock_a, mock_bc, mock_d, mock_rg, tmp_path
):
    """``errors`` must remain empty — the gate is not a failure."""

    def _a_side_effect(state):
        state["episode_files"] = ["/mock/ep0.mp4"]
        state["episode_locations"] = ["Mock Town, SC"]
        return {}

    mock_a.side_effect = _a_side_effect
    mock_bc.side_effect = OstAuditPendingError("audit pending")

    source = tmp_path / "source"
    source.mkdir()

    orch = DrivePipelineOrchestrator()
    result = orch.run(
        source_dir=source,
        output_base=tmp_path / "output",
        region="ireland",
        force=True,
    )

    assert result.success is True
    assert result.errors == []


class TestRunPhaseAOstGate:
    """The OST gate must not silently no-op."""

    def test_empty_locations_with_episodes_raises(self):
        import pytest

        from gomoveshift.pipeline.drive_orchestrator import run_phase_a_ost_gate

        state = SimpleNamespace(episode_locations=[], episode_files=["/e0.mp4"])
        with pytest.raises(RuntimeError, match="state drift"):
            run_phase_a_ost_gate(state)

    def test_no_episodes_is_noop(self):
        from gomoveshift.pipeline.drive_orchestrator import run_phase_a_ost_gate

        run_phase_a_ost_gate(SimpleNamespace(episode_locations=[], episode_files=[]))

    def test_duplicate_locations_raise(self):
        import pytest

        from gomoveshift.pipeline.drive_orchestrator import run_phase_a_ost_gate

        state = SimpleNamespace(
            episode_locations=["Town A", "Town A"],
            episode_files=["/e0.mp4", "/e1.mp4"],
        )
        with pytest.raises(RuntimeError, match="duplicate"):
            run_phase_a_ost_gate(state)

    def test_episode_titles_threaded_into_env_preflight(self):
        """Resume path: _episode_titles exposes saved episode_locations so
        run_preflight's step 8 re-runs the OST check on resume."""
        from gomoveshift.pipeline.orchestrator_preflight import _episode_titles

        assert _episode_titles(SimpleNamespace(episode_locations=["A", "B"])) == ["A", "B"]
        assert _episode_titles(SimpleNamespace(episode_locations=None)) is None
        assert _episode_titles({"episode_locations": ["C"]}) == ["C"]
