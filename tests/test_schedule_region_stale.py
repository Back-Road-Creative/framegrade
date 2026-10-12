"""invalidate_phase_d_if_schedule_region_stale — force a schedule recompute on
resume when the saved drive_schedule's region no longer matches the resolved
region.

The flag fingerprint re-runs phase_d only when CLI flags CHANGE; a saved
schedule that is wrong-but-flag-consistent (a stale region/weekday schedule) was reused
as-is. This drops phase_d so the schedule recomputes against the resolved
region.
"""

from types import SimpleNamespace

from gomoveshift.pipeline.orchestrator_fingerprint import (
    invalidate_phase_d_if_schedule_region_stale,
)


def _state(region, ds_region):
    ds = {"region": ds_region, "lf_slots": []} if ds_region is not None else None
    return SimpleNamespace(
        region=region, drive_schedule=ds, phases_completed=["phase_a", "phase_d"]
    )


class TestScheduleRegionStale:
    def test_region_mismatch_drops_phase_d(self):
        state = _state(region="usa", ds_region="ireland")
        resumed = ["phase_a", "phase_d"]
        stages = ["phase_a", "phase_d"]
        out = invalidate_phase_d_if_schedule_region_stale(state, resumed, stages)
        assert "phase_d" not in out
        assert "phase_d" not in stages
        assert state.phases_completed == ["phase_a"]

    def test_region_match_keeps_phase_d(self):
        state = _state(region="usa", ds_region="usa")
        resumed = ["phase_a", "phase_d"]
        stages = ["phase_a", "phase_d"]
        out = invalidate_phase_d_if_schedule_region_stale(state, resumed, stages)
        assert out == ["phase_a", "phase_d"]
        assert "phase_d" in stages

    def test_no_saved_schedule_is_noop(self):
        state = _state(region="usa", ds_region=None)
        resumed = ["phase_a", "phase_d"]
        out = invalidate_phase_d_if_schedule_region_stale(state, resumed, ["phase_a", "phase_d"])
        assert out == ["phase_a", "phase_d"]

    def test_phase_d_not_in_resume_is_noop(self):
        state = _state(region="usa", ds_region="ireland")
        resumed = ["phase_a"]
        out = invalidate_phase_d_if_schedule_region_stale(state, resumed, ["phase_a"])
        assert out == ["phase_a"]

    def test_dataclass_schedule_region(self):
        # drive_schedule round-trips to a dataclass with a .region attr.
        state = SimpleNamespace(
            region="usa",
            drive_schedule=SimpleNamespace(region="ireland", lf_slots=[]),
            phases_completed=["phase_a", "phase_d"],
        )
        out = invalidate_phase_d_if_schedule_region_stale(
            state, ["phase_a", "phase_d"], ["phase_a", "phase_d"]
        )
        assert "phase_d" not in out
