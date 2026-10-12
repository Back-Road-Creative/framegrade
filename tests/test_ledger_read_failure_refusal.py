"""A content ledger that EXISTS but can't be read/parsed must never degrade to
"proceed collision-blind" — it must refuse, not silently schedule over a
booking it couldn't see. Covers the two callers that turn a ledger read into
an irreversible/collision-sensitive decision (Phase D scheduling, longform
metadata init) plus ``ContentLedger.reload()``'s own merge step."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.content_ledger import ContentLedger
from gomoveshift.pipeline.drive_phases.phase_d_scheduling import (
    _build_and_persist_drive_schedule,
)
from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

from _state_helpers import make_state


def _corrupt_ledger(tmp_path):
    ledger_path = tmp_path / ".content_ledger.json"
    ledger_path.write_text("{not valid json")
    return ledger_path


class TestPhaseDRefusesOnUnreadableLedger:
    def _call(self, tmp_path, state):
        scheduler = MagicMock()
        scheduler.generate_drive_schedule.return_value = SimpleNamespace(to_dict=lambda: {})
        scheduler_cls = MagicMock()
        scheduler_cls.from_config.return_value = scheduler
        config = SimpleNamespace(
            drive=SimpleNamespace(shorts_publish_times=["15:00"], shorts_slot_pool_by_region={}),
            seo=SimpleNamespace(storage_path=str(tmp_path)),
        )
        return _build_and_persist_drive_schedule(
            scheduler_cls=scheduler_cls,
            config=config,
            drive_plan=SimpleNamespace(regular_lfs=[]),
            state=state,
            region="usa",
            stages_dir=tmp_path,
        )

    def test_corrupt_ledger_raises_instead_of_scheduling_blind(self, tmp_path):
        _corrupt_ledger(tmp_path)
        with pytest.raises(RuntimeError, match="Content ledger unreadable"):
            self._call(tmp_path, make_state())


class TestLedgerReloadReportsFailure:
    def test_reload_raises_on_corrupt_file_instead_of_silent_merge(self, tmp_path):
        ledger_path = _corrupt_ledger(tmp_path)
        ledger = ContentLedger(_path=ledger_path)
        with pytest.raises(Exception):
            ledger.reload(ledger_path)


class TestLongFormMetadataRefusesOnUnreadableLedger:
    def test_init_raises_instead_of_falling_back_to_naive_calc(self, tmp_path):
        _corrupt_ledger(tmp_path)
        with patch.object(LongFormMetadataGenerator, "_init_seo_manager"):
            with pytest.raises(RuntimeError, match="Content ledger unreadable"):
                LongFormMetadataGenerator(seo_storage_path=tmp_path)
