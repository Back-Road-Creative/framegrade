"""Regression test for the resume-path episode-index convention.

``_synthesize_ingest_plan_from_state`` (phase_bc_day_first) must produce
1-based ``ep.index`` values to match the convention established by
``execute_ingest`` (``ingest/__init__.py``) and ``episode_grouper``
(both use ``enumerate(..., 1)``).

``day_pipeline.run_day_pipeline`` computes
``day_episode_indices = [int(ep.index) - 1 for ep in day_batch.episodes]``,
which silently produces negative values if indices are 0-based. The
downstream segment remap then raises ``IndexError`` inside a broad
except → silent LF drop on every resumed run with non-trivial days.

Covers the 1-based ingest-index contract on resume.
"""

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _synthesize_ingest_plan_from_state,
)

from _state_helpers import make_state


def _make_state(num_files: int = 3):
    return make_state(
        episode_files=[f"/tmp/ep{i}.mp4" for i in range(num_files)],
        episode_dates=["2026-04-01"] * num_files,
        file_durations=[60.0] * num_files,
        gpx_files=[None] * num_files,
    )


def test_synthesized_episodes_are_one_based():
    plan = _synthesize_ingest_plan_from_state(_make_state(3))
    assert plan is not None
    assert [ep.index for ep in plan.episodes] == [1, 2, 3], (
        "Resume path must emit 1-based indices matching execute_ingest / "
        "episode_grouper. 0-based indices break day_pipeline's -1 subtract."
    )


def test_single_episode_resume_is_one_based():
    plan = _synthesize_ingest_plan_from_state(_make_state(1))
    assert plan is not None
    assert [ep.index for ep in plan.episodes] == [1]


def test_day_pipeline_index_subtraction_never_negative_on_resume():
    """End-to-end invariant: after -1 subtract, no index is negative."""
    plan = _synthesize_ingest_plan_from_state(_make_state(5))
    assert plan is not None
    day_episode_indices = [int(ep.index) - 1 for ep in plan.episodes]
    assert all(i >= 0 for i in day_episode_indices), (
        f"day_pipeline would produce negative indices: {day_episode_indices}"
    )
