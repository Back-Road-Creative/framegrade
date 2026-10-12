"""Regression tests for the resume rehydrate of ``IngestPlan`` episodes.

Rehydrate — ``IngestPlan.from_dict`` is a shallow rehydrate: ``episodes`` come back
as the raw ``list[dict]`` it was serialized from, not ``Episode`` instances.
``group_episodes_by_day`` does ``getattr(ep, "date", None)`` /
``getattr(ep, "creation_time", None)`` which returns ``None`` for a ``dict``
on every probe → every episode is silently dropped → 0 day batches.
``phase_bc_day_first`` then runs to completion processing nothing, and the
orchestrator records ``phase_bc_day_first`` in ``phases_completed``.

Fix: ``phase_bc_day_first`` detects dict-children at the head of ``run()``
and synthesizes from state-derived parallel arrays (which already exist
for resume) before grouping.

Defense-in-depth: if grouping still produces zero
batches while episodes exist, the phase must raise rather than silently
mark itself complete on no work.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    PhaseBCDayFirst,
    _resolve_ingest_plan,
)

from _state_helpers import make_state


def _resume_state_arrays(num_files: int = 3, date: str = "2026-02-20"):
    return make_state(
        episode_files=[f"/tmp/ep{i}.mp4" for i in range(num_files)],
        episode_dates=[date] * num_files,
        file_durations=[60.0] * num_files,
        gpx_files=[None] * num_files,
    )


class TestResolveIngestPlan:
    def test_synthesizes_when_ingest_plan_missing(self):
        state = _resume_state_arrays(2)
        plan = _resolve_ingest_plan(state)
        assert plan is not None
        assert len(plan.episodes) == 2
        assert plan.episodes[0].date == "2026-02-20"

    def test_synthesizes_when_episodes_are_dicts(self):
        """Rehydrate regression — IngestPlan.from_dict yields list[dict]."""
        state = _resume_state_arrays(3)
        state.ingest_plan = SimpleNamespace(
            episodes=[
                {"index": 1, "date": "2026-02-20", "concat_path": "/tmp/ep0.mp4"},
                {"index": 2, "date": "2026-02-20", "concat_path": "/tmp/ep1.mp4"},
                {"index": 3, "date": "2026-02-20", "concat_path": "/tmp/ep2.mp4"},
            ]
        )
        plan = _resolve_ingest_plan(state)
        assert plan is not None
        assert len(plan.episodes) == 3
        assert plan.episodes[0].date == "2026-02-20"
        assert hasattr(plan.episodes[0], "concat_path")

    def test_passthrough_when_episodes_are_attribute_objects(self):
        attr_eps = [SimpleNamespace(index=1, date="2026-02-20", concat_path=None)]
        existing = SimpleNamespace(episodes=attr_eps)
        state = make_state(ingest_plan=existing)
        plan = _resolve_ingest_plan(state)
        assert plan is existing

    def test_synthesizes_when_episodes_list_empty(self):
        state = _resume_state_arrays(1)
        state.ingest_plan = SimpleNamespace(episodes=[])
        plan = _resolve_ingest_plan(state)
        assert plan is not None
        assert len(plan.episodes) == 1


class TestZeroBatchGuard:
    """Defense-in-depth regression — refuse to record phase_bc_day_first complete
    when episodes exist but grouping yielded zero batches."""

    def _config(self):
        from gomoveshift.config import Config

        return Config()

    def test_raises_when_episodes_exist_but_no_date_attribute(self, tmp_path):
        config = self._config()
        phase = PhaseBCDayFirst(config)

        # Episodes are SimpleNamespace (passes the dict-detect rehydrate
        # guard) but have no date or creation_time → group_episodes_by_day
        # returns 0 batches.
        ingest_plan = SimpleNamespace(
            episodes=[
                SimpleNamespace(index=1, date=None, creation_time=None, concat_path=None),
            ]
        )
        state = make_state(
            ingest_plan=ingest_plan,
            output_base=str(tmp_path),
            stages_base=str(tmp_path / "_stages"),
            location="",
            episode_locations=[],
            episode_files=[],
        )

        with pytest.raises(RuntimeError, match="0 day batches"):
            phase.run(state)

    def test_no_raise_when_no_episodes_at_all(self, tmp_path):
        """Empty input is the legitimate no-op path — must not raise."""
        config = self._config()
        phase = PhaseBCDayFirst(config)

        state = make_state(
            ingest_plan=None,
            output_base=str(tmp_path),
            stages_base=str(tmp_path / "_stages"),
            location="",
        )
        # No raise; state is populated with empty defaults.
        phase.run(state)
        assert state.get("regular_lf_paths") == []
        assert state.get("shorts_per_lf") == {}
