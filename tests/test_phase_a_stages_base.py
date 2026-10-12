"""Phase A must route its heavy working files (concatenated per-episode
.mp4s + the drive_scores cache) to the ext4 intermediate dir
``state.stages_base`` — never the output dir, which may be a slow/unstable
v9fs/drvfs mount.

This is the Phase A twin of the Phase D / phase_bc_day_first stages_base
fixes (see test_phase_d_stages_bugs.py). Hardcoding ``output_base/_stages``
instead of reading ``stages_base`` would put hundreds of GB of episode concats
on the slow mount, and would also miss the orchestrator's cache reload (which
reads from stages_base first).
"""

from pathlib import Path

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases import phase_a as phase_a_mod
from gomoveshift.pipeline.drive_phases.phase_a import PhaseA
from gomoveshift.pipeline.drive_state_model import PipelineState


class _StopAfterIngestError(Exception):
    """Sentinel to halt PhaseA.run right after the ingest call."""


def _state(tmp_path, *, stages_base):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    return PipelineState(
        source_dir=src,
        output_base=out,
        stages_base=stages_base,
        region="ireland",
        publish_day="wed",
    )


def _patch_ingest(monkeypatch, captured):
    """Capture the dir PhaseA hands to execute_ingest, then halt."""
    monkeypatch.setattr(phase_a_mod, "create_ingest_plan", lambda *a, **k: object())

    def _fake_execute_ingest(plan, output_dir, **kwargs):
        captured["dir"] = Path(output_dir)
        raise _StopAfterIngestError()

    monkeypatch.setattr(phase_a_mod, "execute_ingest", _fake_execute_ingest)


class TestPhaseAStagesBase:
    def test_ingest_routed_to_stages_base(self, tmp_path, monkeypatch):
        """With stages_base set, episode concats go to stages_base — NOT
        output_base — and no fallback output_base/_stages dir is created."""
        custom_stages = tmp_path / "custom_stages"
        custom_stages.mkdir()
        state = _state(tmp_path, stages_base=custom_stages)
        captured: dict = {}
        _patch_ingest(monkeypatch, captured)

        with pytest.raises(_StopAfterIngestError):
            PhaseA(Config()).run(state)

        assert captured["dir"] == custom_stages, (
            f"ingest must write to stages_base ({custom_stages}), got {captured['dir']}"
        )
        assert not (state.output_base / "_stages").exists(), (
            "Phase A created the legacy output_base/_stages dir instead of using stages_base"
        )

    def test_falls_back_to_output_stages_when_unset(self, tmp_path, monkeypatch):
        """Back-compat: when stages_base is None, the single-FS layout
        (output_base/_stages) is preserved."""
        state = _state(tmp_path, stages_base=None)
        captured: dict = {}
        _patch_ingest(monkeypatch, captured)

        with pytest.raises(_StopAfterIngestError):
            PhaseA(Config()).run(state)

        assert captured["dir"] == state.output_base / "_stages", (
            f"with no stages_base, ingest should use output_base/_stages, got {captured['dir']}"
        )
