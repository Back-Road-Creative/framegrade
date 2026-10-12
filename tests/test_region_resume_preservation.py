"""Region resume-preservation — an ABSENT ``--region`` must not stomp the saved
region on resume.

``--region`` used to default to "ireland", so every resume without the flag
silently overwrote a USA drive's region back to ireland — which reset the
publish schedule to the ireland/Wed default and confused the cadence (the SC
re-burn scheduling bug). ``--region`` is now an argparse sentinel (default
None); ``_try_resume`` keeps the saved region when this run's is None, and
``run()`` resolves a still-None region to "ireland" after the merge.
"""

import json
from pathlib import Path

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.drive_state_model import PipelineState


def _write_state(state_file: Path, saved: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(saved))


class TestRegionResumePreservation:
    def test_absent_region_preserves_saved_usa(self, tmp_path):
        orch = DrivePipelineOrchestrator(config=Config())
        state_file = tmp_path / "pipeline_state.json"
        _write_state(state_file, {"phases_completed": ["phase_a"], "region": "usa"})
        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        # Operator did NOT pass --region → orchestrator passes None (the sentinel
        # _build_initial_state now forwards).
        state["region"] = None
        orch._try_resume(state, state_file, tmp_path, [], fresh=False)
        assert state.region == "usa", (
            "absent --region stomped the saved USA region on resume — the "
            "ireland default would reset the publish schedule"
        )

    def test_explicit_region_wins_over_saved(self, tmp_path):
        orch = DrivePipelineOrchestrator(config=Config())
        state_file = tmp_path / "pipeline_state.json"
        _write_state(state_file, {"phases_completed": ["phase_a"], "region": "ireland"})
        state = PipelineState(source_dir=Path("/tmp/src"), output_base=tmp_path)
        state["region"] = "usa"  # operator passed --region usa this invocation
        orch._try_resume(state, state_file, tmp_path, [], fresh=False)
        assert state.region == "usa", "explicit --region usa did not win over saved ireland"
