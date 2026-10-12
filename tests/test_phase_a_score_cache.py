"""Phase A must reload the cached ``drive_scores.npz`` on resume instead
of re-running the 3-4h scoring pass, when the episode source files it scores
are unchanged.

The validity key is each episode concat path + size + mtime, persisted in the
.npz metadata sidecar (``score_inputs``) at save time. A cache HIT skips the
expensive ``scorer.score_file`` calls entirely; an input change (added /
removed / replaced episode) forces a fresh re-score; ``--fresh`` always
re-scores (it does not delete the .npz).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases import phase_a as phase_a_mod
from gomoveshift.pipeline.drive_phases.phase_a import PhaseA
from gomoveshift.pipeline.drive_state_model import PipelineState


class _SpyScorer:
    """Stand-in scorer that records every score_file call (the expensive pass)
    and reuses the real save/load round-trip for the .npz cache."""

    score_file_calls = 0
    score_file_kwargs: list = []

    def __init__(self, *a, **k):
        pass

    def score_file(self, concat_path, **kwargs):
        type(self).score_file_calls += 1
        type(self).score_file_kwargs.append(kwargs)
        return np.zeros((5, 9), dtype=np.float64)

    def merge_file_scores(self, file_scores, file_durations):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        return UnifiedSegmentScorer.merge_file_scores(self, file_scores, file_durations)

    def save_scores(self, scores, path, metadata):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        UnifiedSegmentScorer.save_scores(self, scores, path, metadata)

    def load_scores(self, path):
        from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

        return UnifiedSegmentScorer.load_scores(self, path)


def _episode(idx, concat_path):
    return SimpleNamespace(
        index=idx,
        concat_path=str(concat_path),
        gpx_path=None,
        date="2026-05-24",
        total_duration=120.0,
        creation_time=None,
        gpx_source="gpmf",
    )


def _patch_phase_a(monkeypatch, episodes):
    """Neutralise every Phase A collaborator except the scoring gate."""
    monkeypatch.setattr(_SpyScorer, "score_file_calls", 0)
    monkeypatch.setattr(_SpyScorer, "score_file_kwargs", [])
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
    # Dead-zone probe shells out to ffprobe — short-circuit the existence
    # check so it never runs.
    monkeypatch.setattr(phase_a_mod, "classify_video", lambda *a: phase_a_mod.VideoType.REGULAR_LF)


def _state(tmp_path, *, stages_base, fresh=False):
    src = tmp_path / "src"
    out = tmp_path / "out"
    src.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    return PipelineState(
        source_dir=src,
        output_base=out,
        stages_base=stages_base,
        region="ireland",
        publish_day="wed",
        fresh=fresh,
    )


def _make_episode_files(tmp_path, n=2):
    files = []
    for i in range(n):
        p = tmp_path / f"ep_{i}.mp4"
        p.write_bytes(b"x" * (1000 + i))
        files.append(p)
    return files


class TestPhaseAScoreCache:
    def test_resume_reuses_cache_skips_rescore(self, tmp_path, monkeypatch):
        stages = tmp_path / "stages"
        stages.mkdir()
        files = _make_episode_files(tmp_path)
        episodes = [_episode(i, f) for i, f in enumerate(files)]
        _patch_phase_a(monkeypatch, episodes)

        # First run: populates drive_scores.npz + score_inputs key.
        state1 = _state(tmp_path, stages_base=stages)
        PhaseA(Config()).run(state1)
        assert _SpyScorer.score_file_calls == len(files), "first run must score every episode"
        assert (stages / "drive_scores.npz").exists()
        first_scores = state1.drive_scores

        # Second run (resume): same inputs → cache HIT, NO re-score.
        monkeypatch.setattr(_SpyScorer, "score_file_calls", 0)
        state2 = _state(tmp_path, stages_base=stages)
        PhaseA(Config()).run(state2)
        assert _SpyScorer.score_file_calls == 0, "resume with unchanged inputs must NOT re-score"
        assert state2.drive_scores is not None
        np.testing.assert_array_equal(state2.drive_scores, first_scores)
        assert len(state2.file_durations) == len(files)

    def test_changed_episode_list_forces_rescore(self, tmp_path, monkeypatch):
        stages = tmp_path / "stages"
        stages.mkdir()
        files = _make_episode_files(tmp_path, n=2)
        episodes = [_episode(i, f) for i, f in enumerate(files)]
        _patch_phase_a(monkeypatch, episodes)

        state1 = _state(tmp_path, stages_base=stages)
        PhaseA(Config()).run(state1)
        assert _SpyScorer.score_file_calls == 2

        # Add a third episode → cache key changes → MUST re-score.
        extra = _make_episode_files(tmp_path, n=3)[2:]
        new_files = files + extra
        new_episodes = [_episode(i, f) for i, f in enumerate(new_files)]
        _patch_phase_a(monkeypatch, new_episodes)
        monkeypatch.setattr(_SpyScorer, "score_file_calls", 0)

        state2 = _state(tmp_path, stages_base=stages)
        PhaseA(Config()).run(state2)
        assert _SpyScorer.score_file_calls == len(new_files), (
            "changed episode set must invalidate the cache and re-score"
        )

    def test_fresh_run_ignores_cache(self, tmp_path, monkeypatch):
        stages = tmp_path / "stages"
        stages.mkdir()
        files = _make_episode_files(tmp_path)
        episodes = [_episode(i, f) for i, f in enumerate(files)]
        _patch_phase_a(monkeypatch, episodes)

        state1 = _state(tmp_path, stages_base=stages)
        PhaseA(Config()).run(state1)
        assert (stages / "drive_scores.npz").exists()

        # --fresh does not delete the .npz, so the gate must re-score anyway.
        monkeypatch.setattr(_SpyScorer, "score_file_calls", 0)
        state2 = _state(tmp_path, stages_base=stages, fresh=True)
        PhaseA(Config()).run(state2)
        assert _SpyScorer.score_file_calls == len(files), "--fresh must always re-score"


class TestPhaseADoesNotMeasureWeather:
    """The ``weather`` scoring weight is gone (its mass moved to
    ``brightness``), and the classifier behind it reports one constant class
    for every input. Phase A's per-episode weather pass therefore decodes
    every episode with ffmpeg to produce a column that ranks nothing — pin
    that it no longer runs.

    The detector itself must SURVIVE: the single-video social stage still
    builds one.
    """

    def test_phase_a_never_builds_or_passes_a_weather_detector(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import weather as weather_mod

        stages = tmp_path / "stages"
        stages.mkdir()
        files = _make_episode_files(tmp_path)
        episodes = [_episode(i, f) for i, f in enumerate(files)]
        _patch_phase_a(monkeypatch, episodes)

        built: list = []
        monkeypatch.setattr(
            weather_mod.WeatherDetector,
            "__init__",
            lambda self, **k: built.append(k),
        )

        PhaseA(Config()).run(_state(tmp_path, stages_base=stages))

        assert _SpyScorer.score_file_calls == len(files), "sanity: episodes must still be scored"
        assert built == [], (
            "Phase A built a WeatherDetector; the weather weight was removed so "
            f"the per-episode ffmpeg brightness scan ranks nothing (got {built!r})"
        )
        assert all(kw.get("weather_detector") is None for kw in _SpyScorer.score_file_kwargs), (
            "Phase A still hands a weather_detector to score_file — that is the "
            "call that decodes every episode a second time"
        )

    def test_weather_detector_survives_for_the_social_stage(self):
        from gomoveshift.pipeline.stages import social as social_mod
        from gomoveshift.pipeline.weather import WeatherDetector

        assert callable(getattr(WeatherDetector, "detect_segment", None)), (
            "WeatherDetector must keep its measurement API — only Phase A's use of it goes"
        )
        social_src = Path(social_mod.__file__).read_text(encoding="utf-8")
        assert "WeatherDetector" in social_src, (
            "the social stage is the surviving WeatherDetector consumer; dropping "
            "Phase A's weather pass must not over-delete the class"
        )


class TestScoredGapGuard:
    """A failed per-episode score (0 rows) hard-fails instead of merging
    a silent scored-timeline gap."""

    def test_empty_score_for_real_episode_raises(self):
        episodes = [
            SimpleNamespace(index=0, total_duration=120.0),
            SimpleNamespace(index=1, total_duration=90.0),
        ]
        file_scores = [
            np.zeros((120, 11), dtype=np.float64),
            np.zeros((0, 11), dtype=np.float64),  # episode 1 scored nothing
        ]
        with pytest.raises(RuntimeError, match="PHASE-A"):
            phase_a_mod._assert_no_scored_gaps(episodes, file_scores)

    def test_all_episodes_scored_is_noop(self):
        episodes = [SimpleNamespace(index=0, total_duration=120.0)]
        file_scores = [np.zeros((120, 11), dtype=np.float64)]
        phase_a_mod._assert_no_scored_gaps(episodes, file_scores)

    def test_zero_duration_episode_is_allowed_empty(self):
        # A degenerate 0s episode legitimately has no rows — not a gap.
        episodes = [SimpleNamespace(index=0, total_duration=0.0)]
        file_scores = [np.zeros((0, 11), dtype=np.float64)]
        phase_a_mod._assert_no_scored_gaps(episodes, file_scores)
