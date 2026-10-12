"""Phase D threads per-LF recording-date map into the consolidated CSV writer.

The consolidated-CSV writer makes
``parent_recording_date_by_short_index`` a *required* parameter on
``build_consolidated_shorts_csv``. Phase D must build the map keyed by
short_index from each short's parent LF index → ``state["per_lf_recording_date"]``.

These tests cover the pure mapping helper that Phase D delegates to. The
Phase BC invariant is the
upstream guarantee that the map is non-empty when Phase BC has run; we
verify here only that Phase D constructs the per-short map correctly
from the per-LF state dict.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import (
    _build_recording_dates_by_short_index,
)


def _post(short_index: int):
    return SimpleNamespace(short_index=short_index)


class TestRecordingDatesByShortIndex:
    def test_builds_map_per_short_from_parent_lf_index(self):
        shorts_by_lf = {
            1: [_post(101), _post(102), _post(103)],
            2: [_post(201), _post(202)],
        }
        per_lf_recording_date = {1: "2025-09-14", 2: "2025-09-15"}

        result = _build_recording_dates_by_short_index(
            shorts_by_lf=shorts_by_lf,
            per_lf_recording_date=per_lf_recording_date,
        )

        assert result == {
            101: "2025-09-14",
            102: "2025-09-14",
            103: "2025-09-14",
            201: "2025-09-15",
            202: "2025-09-15",
        }

    def test_str_keys_raise_after_load_normalization_contract(self):
        # State-shape normalization at the load boundary: per-LF maps
        # are canonical int-keyed in ``state`` because ``PipelineState.from_dict``
        # coerces them via ``_assert_canonical_key_shape``. Helpers
        # downstream of the load boundary therefore drop their str-key
        # fallback — passing a str-keyed map is now a contract violation
        # that surfaces as a missing-LF ValueError (defensive layer ahead
        # of the CSV writer's own check). This test pins the new contract:
        # str keys are NOT silently tolerated after normalization; they raise, which
        # tells the operator the load-boundary normalization failed.
        shorts_by_lf = {1: [_post(101)]}
        per_lf_recording_date = {"1": "2026-01-01"}  # str key, NOT normalized

        with pytest.raises(ValueError, match="LF index 1"):
            _build_recording_dates_by_short_index(
                shorts_by_lf=shorts_by_lf,
                per_lf_recording_date=per_lf_recording_date,
            )

    def test_non_empty_map_when_state_populated(self):
        shorts_by_lf = {1: [_post(101)]}
        result = _build_recording_dates_by_short_index(
            shorts_by_lf=shorts_by_lf,
            per_lf_recording_date={1: "2025-09-14"},
        )
        assert result, "result must be non-empty when state is populated"

    def test_raises_when_parent_lf_missing_from_state(self):
        # Defensive: if Phase BC invariant somehow failed to populate an LF,
        # Phase D's pre-CSV-build assertion must catch it before the CSV
        # writer raises (which it also would, by the writer's own contract).
        shorts_by_lf = {1: [_post(101)], 2: [_post(201)]}
        per_lf_recording_date = {1: "2025-09-14"}  # LF 2 missing

        with pytest.raises(ValueError, match="LF index 2"):
            _build_recording_dates_by_short_index(
                shorts_by_lf=shorts_by_lf,
                per_lf_recording_date=per_lf_recording_date,
            )

    def test_skips_lfs_with_zero_shorts(self):
        # An LF with zero shorts has no per-short rows to map a date to.
        # The invariant should only apply to LFs that actually parent shorts.
        # Pre-fix this raised ValueError on the zero-shorts LF even though
        # the empty posts list meant the map result would be unaffected.
        shorts_by_lf = {1: [], 2: [_post(201)]}
        per_lf_recording_date = {2: "2025-09-15"}  # LF 1 absent — fine, it has no shorts

        result = _build_recording_dates_by_short_index(
            shorts_by_lf=shorts_by_lf,
            per_lf_recording_date=per_lf_recording_date,
        )
        assert result == {201: "2025-09-15"}

    def test_all_lfs_zero_shorts_yields_empty_map(self):
        # Edge case: a Phase D pass with no shorts at all (e.g. an LF-only
        # run, or a test fixture). Must not raise; must return empty map.
        shorts_by_lf = {1: [], 2: []}
        result = _build_recording_dates_by_short_index(
            shorts_by_lf=shorts_by_lf,
            per_lf_recording_date={},
        )
        assert result == {}
