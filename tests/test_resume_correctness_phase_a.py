"""Resume-correctness regression tests for Phase A (ingest + scoring).

Simulates SIGKILL-style crashes mid-Phase-A and asserts the resume path
either recovers cleanly or fails loudly — never silently proceeds with
stale/partial data.

Scope:
  - Phase A outputs: drive_plan.json (written later by phase_bc but
    loaded on resume), drive_scores.npz, episode_locations,
    episode_profiles, gpx_files, gps_per_episode, episode_dates,
    file_boundaries, file_durations, episode_files, ingest_plan,
    location.
  - Resume contract: pipeline_state.json's phases_completed list is the
    source of truth; artifacts must be consistent with it.

Each test pins the safe behaviour: a resume either recovers or drops the
phase so it re-runs.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline._drive_state import load_drive_state, save_drive_state
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.drive_state_model import PipelineState


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


def _make_source_dir(tmp_path: Path) -> Path:
    """Create a synthetic source dir with two empty .mp4 placeholders."""
    src = tmp_path / "source"
    src.mkdir()
    (src / "GX010001.MP4").write_bytes(b"")
    (src / "GX010002.MP4").write_bytes(b"")
    return src


def _make_state_dict(
    source_dir: Path,
    output_base: Path,
    phases_completed=None,
) -> dict:
    """Minimal pipeline_state.json body simulating a completed Phase A."""
    return {
        "source_dir": str(source_dir),
        "output_base": str(output_base),
        "region": "ireland",
        "publish_day": "wed",
        "location": "Test Route",
        "phases_completed": phases_completed if phases_completed is not None else ["phase_a"],
        "episode_locations": ["Cork → Kinsale", "Kinsale → Cork"],
        "episode_profiles": [{"title": "Cork → Kinsale"}, {"title": "Kinsale → Cork"}],
        "episode_files": ["/fake/ep1.mp4", "/fake/ep2.mp4"],
        "gpx_files": ["/fake/ep1.gpx", "/fake/ep2.gpx"],
        "episode_dates": ["2026-03-24", "2026-03-25"],
        "file_durations": [300.0, 400.0],
        "file_boundaries": [[0.0, 300.0], [300.0, 700.0]],
        "run_start": "2026-04-19T08:00:00+00:00",
    }


def _write_fake_scores_npz(path: Path) -> None:
    """Write a minimal drive_scores.npz that UnifiedSegmentScorer.load_scores can read."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    # UnifiedSegmentScorer.save_scores writes at least a 'scores' array and metadata
    scores = np.zeros((10, 9), dtype=np.float32)
    np.savez(
        path,
        scores=scores,
        num_files=2,
        total_duration=700.0,
        file_durations=np.array([300.0, 400.0]),
    )


# ─────────────────────────────────────────────────────────────────────────
# Test suite
# ─────────────────────────────────────────────────────────────────────────


class TestResumeCorrectnessPhaseA:
    """Regression tests for Phase A resume behavior under crash scenarios."""

    def test_resume_from_complete_phase_a_state(self, tmp_path):
        """Happy-path baseline — state says phase_a done + drive_scores.npz present.

        Expectation: _try_resume reports phase_a in resumed_phases, reloads
        drive_scores into state, and does NOT re-run Phase A.
        """
        source_dir = _make_source_dir(tmp_path)
        output_base = tmp_path / "output"
        output_base.mkdir()
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        state_file.write_text(json.dumps(_make_state_dict(source_dir, output_base)))
        _write_fake_scores_npz(stages_dir / "drive_scores.npz")

        orch = DrivePipelineOrchestrator(config=Config())
        state = PipelineState(source_dir=source_dir, output_base=output_base)
        resumed = orch._try_resume(state, state_file, stages_dir, stages_completed=[], fresh=False)

        assert "phase_a" in resumed, "phase_a must be detected as resumed"
        # drive_scores must be reloaded from disk
        assert state.get("drive_scores") is not None, (
            "drive_scores.npz exists — resume must reload it into state"
        )
        # Episode metadata must be preserved from checkpoint
        assert state.get("episode_locations") == ["Cork → Kinsale", "Kinsale → Cork"]
        assert state.get("file_durations") == [300.0, 400.0]

    def test_resume_with_truncated_drive_plan_json(self, tmp_path):
        """Simulate crash mid-save of drive_plan.json: truncate to 50% length.

        drive_plan.json is loaded by phase_bc on resume via DrivePlan.load().
        A truncated file must NOT cause a silent-proceed with stale data;
        DrivePlan.load raises JSONDecodeError, and phase_bc catches it and
        rebuilds from state.

        Here we pin the low-level contract: DrivePlan.load on a truncated
        file raises — the orchestrator's wrapper must not eat that error
        without fallback.
        """
        from gomoveshift.models import DrivePlan, RegularLFPlan

        # Build a valid plan, save it, truncate it
        plan = DrivePlan(
            source_files=[Path("/fake/a.mp4")],
            processed_files=[Path("/fake/a.mp4")],
            regular_lfs=[
                RegularLFPlan(
                    index=1,
                    segments=[],
                    montage_peaks=[],
                    shorts_target=3,
                    region="ireland",
                    total_duration=600.0,
                )
            ],
            score_threshold=0.5,
            total_source_duration=600.0,
            total_highlight_duration=600.0,
        )
        drive_plan_path = tmp_path / "drive_plan.json"
        plan.save(drive_plan_path)
        raw = drive_plan_path.read_text()
        truncated = raw[: len(raw) // 2]
        drive_plan_path.write_text(truncated)

        # Direct load must raise — resume code MUST handle this, not silently proceed.
        with pytest.raises((json.JSONDecodeError, ValueError, KeyError)):
            DrivePlan.load(drive_plan_path)

        # Sanity: phase_bc_day_first wraps .load in try/except and rebuilds on
        # failure. We assert the wrapping behavior exists in source to pin
        # the contract — if removed, this test fails.
        import gomoveshift.pipeline.drive_phases.phase_bc_day_first as pbc_mod

        source = Path(pbc_mod.__file__).read_text()
        assert "DrivePlan.load(drive_plan_path)" in source
        assert "rebuilding" in source, (
            "phase_bc_day_first must have a fallback path for corrupt drive_plan.json"
        )

    def test_resume_with_state_says_phase_a_done_but_scores_missing(self, tmp_path):
        """pipeline_state.json lists phase_a done, but drive_scores.npz is absent.

        _try_resume validates artifacts listed in PHASE_ARTIFACTS. When drive_scores.npz
        is missing, phase_a is DROPPED from resumed_phases so Phase A re-runs
        on this invocation — preventing downstream NoneAttributeError.

        The check reads ``PHASE_ARTIFACTS`` in drive_orchestrator.
        """
        source_dir = _make_source_dir(tmp_path)
        output_base = tmp_path / "output"
        output_base.mkdir()
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        state_file.write_text(json.dumps(_make_state_dict(source_dir, output_base)))
        # NOTE: no drive_scores.npz written — simulates SIGKILL after state
        # save but before / during npz save, or .npz deleted manually.

        orch = DrivePipelineOrchestrator(config=Config())
        state = PipelineState(source_dir=source_dir, output_base=output_base)
        resumed = orch._try_resume(state, state_file, stages_dir, stages_completed=[], fresh=False)

        # phase_a is dropped from resumed_phases because its
        # declared artifact (drive_scores.npz) is missing. Re-run is
        # idempotent (phase_a re-ingests from source_dir).
        assert "phase_a" not in resumed, (
            "phase_a must be dropped when drive_scores.npz is missing so it re-runs; "
            "trusting the stale checkpoint caused downstream NoneAttributeError."
        )
        # drive_scores should NOT be populated — phase_a will repopulate it.
        assert state.get("drive_scores") is None, (
            "drive_scores must remain unset so the re-run can populate it fresh"
        )

    def test_resume_finds_scores_npz_via_output_base_fallback(self, tmp_path):
        """drive_scores.npz absent from stages_dir but present in output_base/_stages/.

        Historical artifact of legacy runs: the authoritative drive_scores.npz
        sometimes lives at ``output_base/_stages/drive_scores.npz`` (where
        Phase A originally wrote it in older versions) rather than under the
        cache-side ``stages_dir``. A strict stages_dir-only check drops phase_a
        and forces a ~30-minute Phase A rerun on every resume — even though
        the authoritative scores file is on disk and loadable.

        The artifact-drift check must consult the output_base/_stages/ fallback
        before declaring an artifact missing. The phase_a reload in
        ``DrivePipelineOrchestrator._try_resume`` already uses this fallback;
        the drop check must too.
        """
        source_dir = _make_source_dir(tmp_path)
        output_base = tmp_path / "output"
        output_base.mkdir()
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        state_file.write_text(json.dumps(_make_state_dict(source_dir, output_base)))
        # drive_scores.npz NOT in stages_dir — but present in output_base/_stages/.
        _write_fake_scores_npz(output_base / "_stages" / "drive_scores.npz")

        orch = DrivePipelineOrchestrator(config=Config())
        state = PipelineState(source_dir=source_dir, output_base=output_base)
        resumed = orch._try_resume(state, state_file, stages_dir, stages_completed=[], fresh=False)

        assert "phase_a" in resumed, (
            "phase_a must be kept resumed when drive_scores.npz exists in the "
            "output_base/_stages/ fallback — else every resume costs a 30-min "
            "Phase A rerun for scores that are already on disk."
        )

    def test_resume_drops_phase_bc_when_drive_plan_json_missing(self, tmp_path):
        """phase_bc_day_first marked complete but drive_plan.json is absent.

        Regression test for PHASE_ARTIFACTS validation on phase_bc_day_first.
        Simulates SIGKILL between checkpoint write and drive_plan.json save:
        the phase must be dropped so run_all_days re-runs and produces the
        expected artifact rather than trusting stale completion.
        """
        source_dir = _make_source_dir(tmp_path)
        output_base = tmp_path / "output"
        output_base.mkdir()
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        state_dict = _make_state_dict(
            source_dir,
            output_base,
            phases_completed=["phase_a", "phase_bc_day_first"],
        )
        state_file.write_text(json.dumps(state_dict))
        # phase_a artifact present so it stays resumed; phase_bc artifact missing.
        _write_fake_scores_npz(stages_dir / "drive_scores.npz")

        orch = DrivePipelineOrchestrator(config=Config())
        state = PipelineState(source_dir=source_dir, output_base=output_base)
        resumed = orch._try_resume(state, state_file, stages_dir, stages_completed=[], fresh=False)

        assert "phase_a" in resumed, "phase_a with valid artifact must stay resumed"
        assert "phase_bc_day_first" not in resumed, (
            "phase_bc_day_first must be dropped when drive_plan.json is missing "
            "so the partition/plan-build path re-runs on this invocation"
        )

    def test_resume_drops_phase_d_when_drive_schedule_json_missing(self, tmp_path):
        """phase_d marked complete but drive_schedule.json is absent.

        Regression test for PHASE_ARTIFACTS validation on phase_d. Phase D
        re-runs are safe because the youtube_uploads sidecar guards against
        duplicate uploads (see drive_orchestrator.run — sidecar merge
        defaults skip_upload=True when prior uploads exist).
        """
        source_dir = _make_source_dir(tmp_path)
        output_base = tmp_path / "output"
        output_base.mkdir()
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        state_dict = _make_state_dict(
            source_dir,
            output_base,
            phases_completed=["phase_a", "phase_bc_day_first", "phase_d"],
        )
        state_file.write_text(json.dumps(state_dict))
        # Present: phase_a + phase_bc artifacts. Missing: phase_d's schedule.
        _write_fake_scores_npz(stages_dir / "drive_scores.npz")
        (stages_dir / "drive_plan.json").write_text("{}")

        orch = DrivePipelineOrchestrator(config=Config())
        state = PipelineState(source_dir=source_dir, output_base=output_base)
        resumed = orch._try_resume(state, state_file, stages_dir, stages_completed=[], fresh=False)

        assert "phase_a" in resumed
        assert "phase_bc_day_first" in resumed
        assert "phase_d" not in resumed, (
            "phase_d must be dropped when drive_schedule.json is missing "
            "so metadata/schedule/social_posts regenerate on re-run"
        )

    def test_resume_drops_multiple_phases_with_missing_artifacts(self, tmp_path):
        """Both phase_a and phase_bc_day_first artifacts missing — both dropped.

        Pins the multi-phase cascade: manifest validation fires per-phase
        independently, not all-or-nothing.
        """
        source_dir = _make_source_dir(tmp_path)
        output_base = tmp_path / "output"
        output_base.mkdir()
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        state_dict = _make_state_dict(
            source_dir,
            output_base,
            phases_completed=["phase_a", "phase_bc_day_first"],
        )
        state_file.write_text(json.dumps(state_dict))
        # Deliberately write no artifacts.

        orch = DrivePipelineOrchestrator(config=Config())
        state = PipelineState(source_dir=source_dir, output_base=output_base)
        resumed = orch._try_resume(state, state_file, stages_dir, stages_completed=[], fresh=False)

        assert resumed == [], f"All phases with missing artifacts must drop; got {resumed}"

    def test_resume_keeps_phase_b_preview_without_manifest_entry(self, tmp_path):
        """phase_b_preview has an empty PHASE_ARTIFACTS entry → always kept.

        Pins the "no authoritative gate file" opt-out: empty manifest list
        means the phase trusts the checkpoint (the phase's own fan-out
        artifacts are validated by downstream consumers, not here).
        """
        source_dir = _make_source_dir(tmp_path)
        output_base = tmp_path / "output"
        output_base.mkdir()
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        state_dict = _make_state_dict(
            source_dir,
            output_base,
            phases_completed=["phase_a", "phase_b_preview"],
        )
        state_file.write_text(json.dumps(state_dict))
        _write_fake_scores_npz(stages_dir / "drive_scores.npz")

        orch = DrivePipelineOrchestrator(config=Config())
        state = PipelineState(source_dir=source_dir, output_base=output_base)
        resumed = orch._try_resume(state, state_file, stages_dir, stages_completed=[], fresh=False)

        assert "phase_b_preview" in resumed, (
            "phase_b_preview with empty manifest entry must stay resumed "
            "(no per-phase gate file to validate)"
        )

    def test_resume_from_corrupt_pipeline_state_json(self, tmp_path):
        """pipeline_state.json is malformed (unclosed brace).

        load_drive_state catches JSONDecodeError, attempts a .tmp
        sibling recovery, and if that fails, **raises CorruptStateError**
        rather than silently returning None, which would cause a silent
        fresh start that could discard hours of prior work.

        The CLI entrypoint is expected to catch CorruptStateError and
        surface a loud banner + non-zero exit (see
        test_cli_corrupt_state_exits_loudly).
        """
        from gomoveshift.pipeline.exceptions import CorruptStateError

        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        # Malformed JSON — unclosed brace + trailing comma
        state_file.write_text('{"phases_completed": ["phase_a",],')

        # Corrupt state with no .tmp sibling MUST raise — no silent fallback.
        with pytest.raises(CorruptStateError) as excinfo:
            load_drive_state(state_file)
        # Error carries the offending path so operators can repair it.
        assert Path(excinfo.value.state_file) == state_file
        assert str(state_file) in str(excinfo.value)

    def test_resume_recovers_from_tmp_sibling_after_corrupt_main(self, tmp_path):
        """If main state file is corrupt but a .tmp sibling from an atomic
        write is present and valid, load_drive_state recovers from it.

        This is the safety net that makes the atomic-write pattern
        meaningful — pin it so it doesn't regress.
        """
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        state_file.write_text("{ not valid json")

        # Write a valid .tmp sibling (simulates mid-replace crash)
        tmp_sibling = stages_dir / "pipeline_state_abc123.tmp"
        source_dir = _make_source_dir(tmp_path)
        output_base = tmp_path / "output"
        output_base.mkdir()
        tmp_sibling.write_text(json.dumps(_make_state_dict(source_dir, output_base)))

        result = load_drive_state(state_file)
        assert result is not None, (
            "A valid .tmp sibling must be used to recover from corrupt main state"
        )
        assert result.get("phases_completed") == ["phase_a"]

    def test_resume_ignores_artifact_when_state_says_not_done(self, tmp_path):
        """Inverse check: drive_scores.npz exists on disk, but state does not
        list phase_a as completed → resume must NOT pretend phase_a is done.

        This guards against the reverse bug of the "state says done but
        artifact missing" case — orphaned artifacts from a prior fresh run
        must not short-circuit a new run.
        """
        source_dir = _make_source_dir(tmp_path)
        output_base = tmp_path / "output"
        output_base.mkdir()
        stages_dir = tmp_path / "_stages"
        stages_dir.mkdir()
        state_file = stages_dir / "pipeline_state.json"
        # State has no phase_a in phases_completed
        state_dict = _make_state_dict(source_dir, output_base, phases_completed=[])
        state_file.write_text(json.dumps(state_dict))
        _write_fake_scores_npz(stages_dir / "drive_scores.npz")

        orch = DrivePipelineOrchestrator(config=Config())
        state = PipelineState(source_dir=source_dir, output_base=output_base)
        resumed = orch._try_resume(state, state_file, stages_dir, stages_completed=[], fresh=False)

        assert "phase_a" not in resumed, (
            "Orphan drive_scores.npz without matching phases_completed entry "
            "must not cause phase_a to be skipped"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
