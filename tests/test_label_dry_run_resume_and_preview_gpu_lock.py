"""``--label-dry-run``/``--list-lfs`` GPU-lock skip must not leak past its
own guarantee: a resumed phase_a, ``--preview``, or a ``--phase`` filter can
all still reach a real nvenc encode without the GPU lock.

Regression: the fresh-phase_a branch printed labels and returned before
phase_bc/phase_d, but a RESUMED phase_a's branch unconditionally
``continue``d past that check, so a resumed ``--label-dry-run`` invocation
ran the full encode + upload WITHOUT ever taking the GPU lock. Separately,
``--preview`` runs a real nvenc encode before the labels-only short-circuit
fires, and an explicit ``--phase`` filter bypasses it entirely — both must
still take the GPU lock even when ``label_dry_run``/``list_lfs`` is set.
"""

from types import SimpleNamespace

import pytest

from gomoveshift.config import Config


class _StopEncodeError(RuntimeError):
    pass


def _patch_common(monkeypatch, mod):
    def _a_side_effect(self, state):
        state["episode_files"] = ["/mock/ep0.mp4"]
        state["episode_locations"] = ["Mock Town, SC"]

    monkeypatch.setattr(
        mod.DrivePipelineOrchestrator, "_phase_a_ingest_and_analyze", _a_side_effect
    )
    monkeypatch.setattr(mod.ReviewGate, "gate_a", lambda self, **k: None)
    monkeypatch.setattr(mod.ReviewGate, "gate_b", lambda self, **k: None)
    from gomoveshift.pipeline.drive_phases import lf_listing

    monkeypatch.setattr(lf_listing, "print_label_dry_run", lambda config, state: None)
    monkeypatch.setattr(lf_listing, "print_lf_listing", lambda config, state: None)


def _fake_resume_phase_a_only(self, state, state_file, stages_dir, stages_completed, fresh):
    stages_completed.extend(["phase_a"])
    state.phases_completed = ["phase_a"]
    return ["phase_a"]


def _patch_resume_scaffolding(monkeypatch, mod):
    monkeypatch.setattr(mod.DrivePipelineOrchestrator, "_try_resume", _fake_resume_phase_a_only)
    for name in (
        "invalidate_phase_d_if_schedule_region_stale",
        "invalidate_phase_d_if_fingerprint_stale",
        "invalidate_phase_d_if_outputs_missing",
        "invalidate_phase_bc_if_superlong_incomplete",
    ):
        monkeypatch.setattr(mod, name, lambda state, rp, sc, **k: rp)
    monkeypatch.setattr(mod, "auto_populate_per_day_completed", lambda *a, **k: None)
    monkeypatch.setattr(mod, "assert_resume_state_coherent", lambda *a, **k: None)
    monkeypatch.setattr(mod, "resume_invariants", lambda *a, **k: None)
    monkeypatch.setattr(mod, "run_phase_a_ost_gate", lambda *a, **k: None)
    monkeypatch.setattr(mod, "run_preflight_and_prompt", lambda *a, **k: None)
    monkeypatch.setattr(mod.DrivePipelineOrchestrator, "_run_dod_validator", lambda self, **k: None)
    monkeypatch.setattr(mod.DrivePipelineOrchestrator, "_save_state", lambda self, f, s: None)


class TestResumedLabelDryRunNeverEncodes:
    def test_resumed_phase_a_label_dry_run_never_runs_phase_bc_or_d(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import drive_orchestrator as mod

        monkeypatch.setattr(mod, "acquire_gpu_lock", lambda *a, **k: (None, None))
        _patch_resume_scaffolding(monkeypatch, mod)
        _patch_common(monkeypatch, mod)

        calls = {"bc": 0, "d": 0}

        def _bc(self, state):
            calls["bc"] += 1

        def _d(self, state, drive_plan=None):
            calls["d"] += 1

        monkeypatch.setattr(mod.DrivePipelineOrchestrator, "_phase_bc_day_first", _bc)
        monkeypatch.setattr(mod.DrivePipelineOrchestrator, "_phase_d_metadata_and_publish", _d)

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        cfg.drive.no_review = True
        orch = mod.DrivePipelineOrchestrator(config=cfg)
        src = tmp_path / "source"
        src.mkdir()

        result = orch.run(
            source_dir=src,
            output_base=tmp_path / "out",
            region="usa",
            label_dry_run=True,
            force=True,
        )

        assert calls["bc"] == 0, "a resumed --label-dry-run must never reach phase_bc"
        assert calls["d"] == 0, "a resumed --label-dry-run must never reach phase_d"
        assert result.success is True

    def test_resumed_list_lfs_never_calls_acquire_gpu_lock(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import drive_orchestrator as mod

        def _boom(*a, **k):
            raise AssertionError("acquire_gpu_lock must not be called for a resumed --list-lfs")

        monkeypatch.setattr(mod, "acquire_gpu_lock", _boom)
        _patch_resume_scaffolding(monkeypatch, mod)
        _patch_common(monkeypatch, mod)
        monkeypatch.setattr(
            mod.DrivePipelineOrchestrator, "_phase_bc_day_first", lambda self, s: None
        )
        monkeypatch.setattr(
            mod.DrivePipelineOrchestrator,
            "_phase_d_metadata_and_publish",
            lambda self, s, drive_plan=None: None,
        )

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        cfg.drive.no_review = True
        orch = mod.DrivePipelineOrchestrator(config=cfg)
        src = tmp_path / "source"
        src.mkdir()

        result = orch.run(
            source_dir=src,
            output_base=tmp_path / "out",
            region="usa",
            list_lfs=True,
            force=True,
        )
        assert result.success is True


class TestPreviewAndPhaseFilterStillTakeGpuLock:
    def test_preview_with_label_dry_run_still_takes_gpu_lock(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import drive_orchestrator as mod

        def _boom(*a, **k):
            raise _StopEncodeError("acquire_gpu_lock reached (preview must still take the lock)")

        monkeypatch.setattr(mod, "acquire_gpu_lock", _boom)
        _patch_common(monkeypatch, mod)

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        cfg.drive.preview = True
        orch = mod.DrivePipelineOrchestrator(config=cfg)
        src = tmp_path / "source"
        src.mkdir()

        with pytest.raises(_StopEncodeError):
            orch.run(
                source_dir=src,
                output_base=tmp_path / "out",
                region="usa",
                label_dry_run=True,
                force=True,
            )

    def test_phase_filter_with_label_dry_run_still_takes_gpu_lock(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import drive_orchestrator as mod

        def _boom(*a, **k):
            raise _StopEncodeError(
                "acquire_gpu_lock reached (--phase filter must still take the lock)"
            )

        monkeypatch.setattr(mod, "acquire_gpu_lock", _boom)
        _patch_common(monkeypatch, mod)

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        orch = mod.DrivePipelineOrchestrator(config=cfg)
        src = tmp_path / "source"
        src.mkdir()

        with pytest.raises(_StopEncodeError):
            orch.run(
                source_dir=src,
                output_base=tmp_path / "out",
                region="usa",
                label_dry_run=True,
                phase="phase_a",
                force=True,
            )
