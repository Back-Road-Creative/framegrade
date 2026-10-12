"""Gate C/D human review runs BEFORE phase_d's irreversible uploads.

A rejection at Gate C/D must block publication ENTIRELY: no Drive shorts
upload, no YouTube LF upload, no Metricool CSV. The orchestrator injects the
review INTO phase_d (ahead of the upload and CSV blocks) so those
blocks are unreachable once ``gate_cd`` raises ``ReviewFailedError``
(the gate runs before the irreversible upload).

Regression guard: if ``gate_cd`` ran AFTER phase_d completed (after the
YouTube upload + Metricool CSV write), a rejection could no longer stop
publication.
"""

from __future__ import annotations

import time
from pathlib import Path

from gomoveshift.config import get_config
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.review_gate import ReviewFailedError

from test_phase_d_split_skip_upload import (
    _build_state,
    _install_fake_youtube_uploader,
    _phase_d_patches,
    _setup_mocks,
)


def test_gate_cd_rejection_blocks_all_phase_d_uploads(tmp_path, monkeypatch):
    """A Gate C/D rejection means NO Drive/YouTube upload is attempted.

    Driven through the real ``_execute_phases`` run loop so the whole wiring
    is exercised: the loop sets ``_pending_gate_cd``, phase_d honours it, and
    the gate rejection fires before the upload blocks.
    """
    state = _build_state(tmp_path, num_lfs=2)

    def _reject(*_a, **_kw):
        raise ReviewFailedError("human rejected at Gate C/D")

    monkeypatch.setattr("gomoveshift.pipeline.review_gate.ReviewGate.gate_cd", _reject)

    upload_tracker: dict = {}
    _install_fake_youtube_uploader(monkeypatch, upload_tracker)

    orch = DrivePipelineOrchestrator(get_config())
    orch._skip_invariants = True
    errors: list = []
    with (
        _phase_d_patches()[0] as m0,
        _phase_d_patches()[1] as m1,
        _phase_d_patches()[2] as m2,
        _phase_d_patches()[3] as m3,
        _phase_d_patches()[4] as m4,
        _phase_d_patches()[5] as m5,
        _phase_d_patches()[6] as m6,
        _phase_d_patches()[7] as m7,
    ):
        uploader = _setup_mocks((m0, m1, m2, m3, m4, m5, m6, m7), num_lfs=2)
        orch._execute_phases(
            state=state,
            state_file=tmp_path / "pipeline_state.json",
            phase_defs=[("phase_d", orch._phase_d_metadata_and_publish)],
            resumed_phases=[],
            source_dir=Path(state.source_dir),
            project_dir=Path(state.output_base),
            location="Asheville, NC",
            list_lfs=False,
            label_dry_run=False,
            start_time=time.time(),
            stages_completed=[],
            errors=errors,
            timings={},
        )

    assert uploader.upload_shorts.call_count == 0, (
        "Drive shorts upload must NOT run when Gate C/D rejects; "
        f"got {uploader.upload_shorts.call_count} call(s)"
    )
    assert upload_tracker.get("calls", []) == [], (
        "YouTube LF upload must NOT run when Gate C/D rejects; "
        f"got calls for {upload_tracker.get('calls')}"
    )
    assert any("ReviewFailedError" in e for e in errors), (
        "the Gate C/D rejection must be recorded as a run error, not swallowed"
    )
