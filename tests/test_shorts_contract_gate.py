"""SHORTS-CONTRACT gate.

The target is a maximum: a plan with fewer shorts (quality bar, gap, per-LF
window) is valid and the planned count becomes the contract. A short that was
planned but is missing after render (or on resume) must still hard-fail. The
master-deletion gate must count-match against the planned count, never
produced-vs-produced. ``--allow-short-count`` is the only escape.
"""

import pytest

from gomoveshift.pipeline.day_pipeline import DayArtifacts, _enforce_shorts_contract
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _expected_shorts_for_master_drop,
)


class TestEnforceShortsContract:
    def test_shortfall_raises(self):
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT: rendered 2 < required 3"):
            _enforce_shorts_contract(2, 3, "2026-06-01", False, stage="rendered")

    def test_rendered_below_planned_still_raises_when_plan_was_quality_limited(self):
        # The plan returned 3 of a target of 14; one of the 3 then fails to render.
        planned = 3
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT: rendered 2 < required 3"):
            _enforce_shorts_contract(2, planned, "2026-06-01", False, stage="rendered")
        _enforce_shorts_contract(3, planned, "2026-06-01", False, stage="rendered")

    def test_message_names_day_and_escape_flag(self):
        with pytest.raises(RuntimeError, match=r"2026-06-01.*--allow-short-count"):
            _enforce_shorts_contract(9, 14, "2026-06-01", False, stage="rendered")

    def test_at_target_passes(self):
        _enforce_shorts_contract(14, 14, "2026-06-01", False)

    def test_no_target_passes(self):
        _enforce_shorts_contract(0, None, "2026-06-01", False)

    def test_allow_short_count_suppresses(self):
        _enforce_shorts_contract(2, 3, "2026-06-01", True, stage="rendered")

    def test_zero_planned_zero_rendered_passes(self):
        # A day with nothing above its timeline mean plans zero shorts: valid.
        _enforce_shorts_contract(0, 0, "2026-06-01", False, stage="rendered")

    def test_rendered_stage_in_message(self):
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT: rendered 8 < required 9"):
            _enforce_shorts_contract(8, 9, "2026-06-01", False, stage="rendered")


class TestMasterDropUsesPlannedTarget:
    def test_quality_limited_day_expects_its_planned_count(self):
        art = DayArtifacts(
            date="2026-06-01",
            master_path=None,
            short_paths=["a.mp4", "b.mp4", "c.mp4"],
            planned_shorts_count=3,
        )
        assert _expected_shorts_for_master_drop(art, allow_short_count=False) == 3

    def _art(self, produced: int, planned):
        return DayArtifacts(
            date="2026-06-01",
            master_path=None,
            short_paths=[f"short_{i:02d}.mp4" for i in range(produced)],
            planned_shorts_count=planned,
        )

    def test_shortfall_day_expects_planned_not_produced(self):
        # 9-of-14 day: gate must demand 14, so drop_day_master preserves
        # the master — produced-vs-produced (the old bug) passed at 9.
        art = self._art(produced=9, planned=14)
        assert _expected_shorts_for_master_drop(art, allow_short_count=False) == 14

    def test_allow_short_count_falls_back_to_actual(self):
        art = self._art(produced=9, planned=14)
        assert _expected_shorts_for_master_drop(art, allow_short_count=True) == 9

    def test_no_planned_target_falls_back_to_actual(self):
        # Disk-reconstructed artifacts carry no planned count.
        art = self._art(produced=9, planned=None)
        assert _expected_shorts_for_master_drop(art, allow_short_count=False) == 9

    def test_no_planned_no_shorts_is_none(self):
        art = self._art(produced=0, planned=None)
        assert _expected_shorts_for_master_drop(art, allow_short_count=False) is None
