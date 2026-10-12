"""``--label-dry-run`` / ``--list-lfs`` must never wait on the shared GPU lock.

Regression: a ``--label-dry-run`` invocation used to log "Waiting up to 3600s
for the shared GPU lock" because a live drive held it — making the pre-encode
label-verification gate the service contract requires impossible while any
other drive runs.

Two contracts pinned here:

1. ``drive_orchestrator.run()`` must never call ``acquire_gpu_lock()`` for a
   ``--label-dry-run``/``--list-lfs`` invocation in ``full_drive`` mode — the
   scoring pass it would otherwise wait for is not needed to resolve labels.
2. ``PhaseA.run(state, labels_only=True)`` in ``full_drive`` mode must never
   call the GPU-bound ``UnifiedSegmentScorer.score_file`` (real frame
   decode), and must never persist a synthetic ``drive_scores.npz`` cache
   that would fool a later LIVE run's score-cache resume gate into skipping real
   scoring.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases import phase_a as phase_a_mod
from gomoveshift.pipeline.drive_phases.phase_a import PhaseA
from gomoveshift.pipeline.drive_state_model import PipelineState


def _episode(idx, concat_path, duration=120.0):
    return SimpleNamespace(
        index=idx,
        concat_path=str(concat_path),
        gpx_path=None,
        date="2026-09-27",
        total_duration=duration,
        creation_time=None,
        gpx_source="gpmf",
    )


class _BoomScorer:
    """Stands in for ``UnifiedSegmentScorer`` — any GPU-bound call is a bug."""

    def __init__(self, *a, **k):
        pass

    def score_file(self, *a, **k):
        raise AssertionError(
            "score_file (GPU frame decode) must not run for a labels_only full_drive Phase A"
        )

    def merge_file_scores(self, file_scores, file_durations):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        return UnifiedSegmentScorer.merge_file_scores(self, file_scores, file_durations)

    def save_scores(self, *a, **k):
        raise AssertionError(
            "save_scores must not persist a synthetic drive_scores.npz — a later "
            "LIVE run would wrongly reuse it via the score-cache resume gate"
        )

    def load_scores(self, path):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        return UnifiedSegmentScorer.load_scores(self, path)


def _patch_phase_a(monkeypatch, episodes):
    monkeypatch.setattr(phase_a_mod, "UnifiedSegmentScorer", _BoomScorer)
    monkeypatch.setattr(phase_a_mod, "create_ingest_plan", lambda *a, **k: object())
    monkeypatch.setattr(
        phase_a_mod,
        "execute_ingest",
        lambda plan, out, **k: SimpleNamespace(episodes=episodes),
    )
    monkeypatch.setattr(phase_a_mod, "_get_shared_geocoder", lambda: None)
    monkeypatch.setattr(phase_a_mod, "_get_route_analyzer", lambda g: None)
    monkeypatch.setattr(
        phase_a_mod,
        "build_route_profile",
        lambda **k: SimpleNamespace(to_dict=lambda: {}, title=lambda: ""),
    )
    monkeypatch.setattr(phase_a_mod, "_extract_episode_ost_meta", lambda p: {})
    monkeypatch.setattr(phase_a_mod, "differentiate_duplicate_ost", lambda locs, meta: locs)
    monkeypatch.setattr(phase_a_mod, "derive_drive_ost_text", lambda *a, **k: "")
    monkeypatch.setattr(phase_a_mod, "load_overrides", lambda base: None)
    monkeypatch.setattr(phase_a_mod, "validate_ost", lambda t, fmt: (True, ""))
    monkeypatch.setattr(phase_a_mod, "classify_video", lambda *a: phase_a_mod.VideoType.REGULAR_LF)


def _make_episode_files(tmp_path, n=2):
    files = []
    for i in range(n):
        p = tmp_path / f"ep_{i}.mp4"
        p.write_bytes(b"x" * (1000 + i))
        files.append(p)
    return files


class TestPhaseALabelsOnlyFullDriveNeverScores:
    def test_never_calls_score_file_or_save_scores(self, tmp_path, monkeypatch):
        stages = tmp_path / "stages"
        stages.mkdir()
        files = _make_episode_files(tmp_path)
        episodes = [_episode(i, f) for i, f in enumerate(files)]
        _patch_phase_a(monkeypatch, episodes)

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        state = PipelineState(
            source_dir=src,
            output_base=out,
            stages_base=stages,
            region="usa",
            publish_day="fri",
        )

        # Must not raise -- _BoomScorer blows up on any GPU-bound call.
        PhaseA(cfg).run(state, labels_only=True)

        assert state.drive_scores is not None
        assert state.drive_scores.shape[0] > 0, "synthetic scores must span real duration"
        assert not (stages / "drive_scores.npz").exists(), (
            "a labels_only run must not write a cache a later LIVE run could reuse"
        )

    def test_labels_only_false_still_scores_for_real(self, tmp_path, monkeypatch):
        """Sanity: the normal (encode) path is untouched by the new flag."""
        stages = tmp_path / "stages"
        stages.mkdir()
        files = _make_episode_files(tmp_path)
        episodes = [_episode(i, f) for i, f in enumerate(files)]

        class _SpyScorer(_BoomScorer):
            calls = 0

            def score_file(self, *a, **k):
                type(self).calls += 1
                import numpy as np

                return np.zeros((5, 11), dtype=np.float64)

            def save_scores(self, scores, path, metadata):
                from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

                UnifiedSegmentScorer.save_scores(self, scores, path, metadata)

        monkeypatch.setattr(phase_a_mod, "UnifiedSegmentScorer", _SpyScorer)
        monkeypatch.setattr(phase_a_mod, "create_ingest_plan", lambda *a, **k: object())
        monkeypatch.setattr(
            phase_a_mod,
            "execute_ingest",
            lambda plan, out, **k: SimpleNamespace(episodes=episodes),
        )
        monkeypatch.setattr(phase_a_mod, "_get_shared_geocoder", lambda: None)
        monkeypatch.setattr(phase_a_mod, "_get_route_analyzer", lambda g: None)
        monkeypatch.setattr(
            phase_a_mod,
            "build_route_profile",
            lambda **k: SimpleNamespace(to_dict=lambda: {}, title=lambda: ""),
        )
        monkeypatch.setattr(phase_a_mod, "_extract_episode_ost_meta", lambda p: {})
        monkeypatch.setattr(phase_a_mod, "differentiate_duplicate_ost", lambda locs, meta: locs)
        monkeypatch.setattr(phase_a_mod, "derive_drive_ost_text", lambda *a, **k: "")
        monkeypatch.setattr(phase_a_mod, "load_overrides", lambda base: None)
        monkeypatch.setattr(phase_a_mod, "validate_ost", lambda t, fmt: (True, ""))
        monkeypatch.setattr(
            phase_a_mod, "classify_video", lambda *a: phase_a_mod.VideoType.REGULAR_LF
        )

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()
        state = PipelineState(
            source_dir=src,
            output_base=out,
            stages_base=stages,
            region="usa",
            publish_day="fri",
        )

        PhaseA(cfg).run(state)  # labels_only defaults False

        assert _SpyScorer.calls == len(files)
        assert (stages / "drive_scores.npz").exists()


class TestOrchestratorSkipsGpuLockForFullDriveLabelDryRun:
    def _patch_common(self, monkeypatch, mod):
        def _a_side_effect(self, state):
            state["episode_files"] = ["/mock/ep0.mp4"]
            state["episode_locations"] = ["Mock Town, SC"]

        monkeypatch.setattr(
            mod.DrivePipelineOrchestrator, "_phase_a_ingest_and_analyze", _a_side_effect
        )
        monkeypatch.setattr(mod.ReviewGate, "gate_a", lambda self, **k: None)
        from gomoveshift.pipeline.drive_phases import lf_listing

        monkeypatch.setattr(lf_listing, "print_label_dry_run", lambda config, state: None)
        monkeypatch.setattr(lf_listing, "print_lf_listing", lambda config, state: None)

    def test_label_dry_run_full_drive_never_calls_acquire_gpu_lock(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import drive_orchestrator as mod

        def _boom(*a, **k):
            raise AssertionError(
                "acquire_gpu_lock must not be called for --label-dry-run in full_drive mode"
            )

        monkeypatch.setattr(mod, "acquire_gpu_lock", _boom)
        self._patch_common(monkeypatch, mod)

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        orch = mod.DrivePipelineOrchestrator(config=cfg)
        source = tmp_path / "source"
        source.mkdir()

        result = orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="usa",
            label_dry_run=True,
        )
        assert result.success is True

    def test_list_lfs_full_drive_never_calls_acquire_gpu_lock(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import drive_orchestrator as mod

        def _boom(*a, **k):
            raise AssertionError(
                "acquire_gpu_lock must not be called for --list-lfs in full_drive mode"
            )

        monkeypatch.setattr(mod, "acquire_gpu_lock", _boom)
        self._patch_common(monkeypatch, mod)

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        orch = mod.DrivePipelineOrchestrator(config=cfg)
        source = tmp_path / "source"
        source.mkdir()

        result = orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="usa",
            list_lfs=True,
        )
        assert result.success is True

    def test_normal_run_still_takes_the_gpu_lock(self, tmp_path, monkeypatch):
        """Sanity: only the labels_only + full_drive combination is exempt."""
        from gomoveshift.pipeline import drive_orchestrator as mod

        class _SentinelError(RuntimeError):
            pass

        def _boom(*a, **k):
            raise _SentinelError("acquire_gpu_lock reached")

        monkeypatch.setattr(mod, "acquire_gpu_lock", _boom)
        self._patch_common(monkeypatch, mod)

        cfg = Config()
        cfg.drive.lf_selection_mode = "full_drive"
        orch = mod.DrivePipelineOrchestrator(config=cfg)
        source = tmp_path / "source"
        source.mkdir()

        with pytest.raises(_SentinelError):
            orch.run(
                source_dir=source,
                output_base=tmp_path / "output",
                region="usa",
            )
