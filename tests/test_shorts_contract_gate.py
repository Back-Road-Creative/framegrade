"""SHORTS-CONTRACT gate.

A day that plans or renders fewer shorts than the required target must
hard-fail BEFORE any shorts encode/CSV/upload — not log the shortfall and
continue (a silent ceiling below the target). The master-deletion gate must
count-match against the planned target, never produced-vs-produced.
``--allow-short-count`` is the only escape.
"""

import pytest

from gomoveshift.pipeline.day_pipeline import DayArtifacts, _enforce_shorts_contract
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _expected_shorts_for_master_drop,
)


class TestEnforceShortsContract:
    def test_shortfall_raises_before_encode(self):
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT: planned 9 < required 14"):
            _enforce_shorts_contract(9, 14, "2026-06-01", False)

    def test_message_names_day_and_escape_flag(self):
        with pytest.raises(RuntimeError, match=r"2026-06-01.*--allow-short-count"):
            _enforce_shorts_contract(9, 14, "2026-06-01", False)

    def test_at_target_passes(self):
        _enforce_shorts_contract(14, 14, "2026-06-01", False)

    def test_no_target_passes(self):
        _enforce_shorts_contract(0, None, "2026-06-01", False)

    def test_allow_short_count_suppresses(self):
        _enforce_shorts_contract(9, 14, "2026-06-01", True)

    def test_rendered_stage_in_message(self):
        with pytest.raises(RuntimeError, match="SHORTS-CONTRACT: rendered 8 < required 9"):
            _enforce_shorts_contract(8, 9, "2026-06-01", False, stage="rendered")


class TestMasterDropUsesPlannedTarget:
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
