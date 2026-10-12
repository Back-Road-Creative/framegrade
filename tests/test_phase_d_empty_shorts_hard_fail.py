"""Empty/absent ``shorts_per_lf`` at Phase D is a HARD FAIL.

The legacy fallback used to (a) return from the shorts quality gate WITHOUT
validating anything and (b) glob ``output_base/shorts/lf_NN/`` and ship whatever
sat on disk with ZERO LUFS/geometry/OST gating. Both are killed: an empty
``shorts_per_lf`` while regular LFs exist now raises rather than glob-ships. A
genuine zero-shorts day carries per-LF empty lists (``{i: []}`` — truthy), which
remains a benign no-op.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gomoveshift.pipeline.drive_phases.external_systems_coordinator import (
    _upload_all_lfs_shorts_to_drive,
)
from gomoveshift.pipeline.drive_phases.phase_d_gates import _assert_shorts_quality_gate

from _state_helpers import make_state


class TestShortsQualityGateEmpty:
    def test_empty_with_lfs_hard_fails(self):
        state = make_state(regular_lf_paths=["/x/lf_01.mp4"], shorts_per_lf={})
        with pytest.raises(RuntimeError, match="shorts_per_lf"):
            _assert_shorts_quality_gate(state, validator_cls=MagicMock())

    def test_empty_without_lfs_is_noop(self):
        state = make_state(regular_lf_paths=[], shorts_per_lf={})
        # No LFs → nothing to ship → no raise.
        _assert_shorts_quality_gate(state, validator_cls=MagicMock())

    def test_zero_shorts_day_with_empty_lists_is_noop(self):
        # {i: []} is truthy — the real zero-shorts-day shape — must NOT raise.
        state = make_state(
            regular_lf_paths=["/x/lf_01.mp4", "/x/lf_02.mp4"],
            shorts_per_lf={1: [], 2: []},
        )
        _assert_shorts_quality_gate(state, validator_cls=MagicMock())

    def test_allow_bad_shorts_escape_bypasses_empty_hard_fail(self):
        state = make_state(regular_lf_paths=["/x/lf_01.mp4"], shorts_per_lf={})
        state.skip_shorts_quality_gate = True
        # Operator escape (--allow-bad-shorts) suppresses the hard fail.
        _assert_shorts_quality_gate(state, validator_cls=MagicMock())


class TestUploadLoopEmpty:
    def _drive_plan(self):
        return SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)])

    def test_empty_shorts_per_lf_refuses_glob_ship(self, tmp_path):
        state = make_state(regular_lf_paths=["/x/lf_01.mp4"], shorts_per_lf={})
        factory = MagicMock()
        with pytest.raises(RuntimeError, match="shorts_per_lf"):
            _upload_all_lfs_shorts_to_drive(
                drive_plan=self._drive_plan(),
                state=state,
                project_base="proj",
                output_base=tmp_path,
                uploader_factory=factory,
                skip_drive_upload=False,
            )
        # The killed fallback must never build an uploader / glob the legacy dir.
        factory.assert_not_called()

    def test_skip_drive_upload_short_circuits_before_hard_fail(self, tmp_path):
        state = make_state(regular_lf_paths=["/x/lf_01.mp4"], shorts_per_lf={})
        factory = MagicMock()
        # skip_drive_upload returns early — an audit run must not hard-fail here.
        manifests, uploader = _upload_all_lfs_shorts_to_drive(
            drive_plan=self._drive_plan(),
            state=state,
            project_base="proj",
            output_base=tmp_path,
            uploader_factory=factory,
            skip_drive_upload=True,
        )
        assert uploader is None
        factory.assert_not_called()
