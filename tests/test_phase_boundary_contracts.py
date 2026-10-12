"""TDD tests: aggregated BC→D per-LF boundary contract.

Two symmetric aggregators wrap the existing ``_assert_per_lf_*_complete``
family into a single named contract:

- ``assert_phase_bc_per_lf_complete(state, skeleton_lfs)`` runs as the LAST
  statement of ``PhaseBCDayFirst.run()`` — it walks every populated per_lf_*
  field and raises ``PhaseBCExitContractError`` listing ALL missing-by-field
  with LF indices.
- ``assert_phase_d_per_lf_inputs_present(state)`` runs as the FIRST statement
  of ``PhaseD.run()`` — symmetrical check from D's POV (uses
  ``drive_plan.regular_lfs`` for the index roster). Raises
  ``PhaseDEntryContractError``.

Removing the contract makes Phase D's defensive ``.get(...) or {}`` reads
unnecessary; the contract guarantees presence by construction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from _state_helpers import make_state


def _lf(idx: int, segments=None, total_duration: float = 300.0):
    return SimpleNamespace(
        index=idx,
        segments=segments or [SimpleNamespace(source_file_index=0)],
        montage_peaks=[],
        total_duration=total_duration,
    )


def _populated_state(lf_indices=(1, 2)):
    """A minimal state dict that satisfies the BC-exit per-LF contract.

    Mirrors what ``_assert_post_encoding_invariants`` leaves on state after
    a successful Phase BC run on a GPS-bearing drive.
    """
    return {
        "per_lf_recording_date": {i: "2026-02-20" for i in lf_indices},
        "per_lf_ost_text": {i: f"Drive LF {i}" for i in lf_indices},
        "per_lf_titles": {i: f"Drive LF {i}" for i in lf_indices},
        "per_lf_locations": {i: "Killarney, Kerry, Ireland" for i in lf_indices},
        "per_lf_trim_ratio": {i: 0.75 for i in lf_indices},
        # GPX-keyed fields populate only on GPS-bearing drives; we declare
        # per_lf_gpx_path here so the GPX-keyed completeness checks fire.
        "per_lf_gpx_path": {i: f"/tmp/lf_{i}.gpx" for i in lf_indices},
        "per_lf_distance_km": {i: 42.0 for i in lf_indices},
        "per_lf_elevation_gain_m": {i: 150.0 for i in lf_indices},
        "per_lf_regions": {i: ["County Kerry"] for i in lf_indices},
    }


class TestPhaseBCExitAggregator:
    def test_passes_on_fully_populated_state(self):
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_bc_per_lf_complete,
        )

        skeleton = [_lf(1), _lf(2)]
        state = _populated_state(lf_indices=(1, 2))
        # Should not raise.
        assert_phase_bc_per_lf_complete(state, skeleton)

    def test_passes_on_gps_less_drive_with_no_gpx_paths(self):
        """GPS-less drives leave per_lf_gpx_path empty by design; the
        GPX-keyed sub-checks (distance/elevation/regions) must be skipped."""
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_bc_per_lf_complete,
        )

        skeleton = [_lf(1)]
        state = _populated_state(lf_indices=(1,))
        state["per_lf_gpx_path"] = {}
        # Wipe GPX-derived maps too — they should be empty on a GPS-less drive.
        state["per_lf_distance_km"] = {}
        state["per_lf_elevation_gain_m"] = {}
        state["per_lf_regions"] = {}
        assert_phase_bc_per_lf_complete(state, skeleton)

    def test_aggregates_all_missing_fields_in_single_raise(self):
        """Single-shot aggregate error: missing fields are listed together,
        not raised one-by-one."""
        from gomoveshift.pipeline.exceptions import PhaseBCExitContractError
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_bc_per_lf_complete,
        )

        skeleton = [_lf(1), _lf(2), _lf(3)]
        state = _populated_state(lf_indices=(1, 2, 3))
        # Drop entries from multiple independent fields.
        del state["per_lf_recording_date"][2]
        del state["per_lf_trim_ratio"][3]
        del state["per_lf_distance_km"][1]
        del state["per_lf_distance_km"][2]

        with pytest.raises(PhaseBCExitContractError) as exc:
            assert_phase_bc_per_lf_complete(state, skeleton)

        msg = str(exc.value)
        # All three field names must appear in the single message.
        assert "per_lf_recording_date" in msg
        assert "per_lf_trim_ratio" in msg
        assert "per_lf_distance_km" in msg
        # Missing LF indices surfaced.
        assert "2" in msg and "3" in msg and "1" in msg

    def test_raises_when_per_lf_recording_date_entirely_missing(self):
        from gomoveshift.pipeline.exceptions import PhaseBCExitContractError
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_bc_per_lf_complete,
        )

        skeleton = [_lf(1)]
        state = _populated_state(lf_indices=(1,))
        state["per_lf_recording_date"] = {}
        with pytest.raises(PhaseBCExitContractError) as exc:
            assert_phase_bc_per_lf_complete(state, skeleton)
        assert "per_lf_recording_date" in str(exc.value)


class TestPhaseDEntryAggregator:
    def test_passes_on_fully_populated_state(self):
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_d_per_lf_inputs_present,
        )

        regular_lfs = [_lf(1), _lf(2)]
        state = _populated_state(lf_indices=(1, 2))
        state["drive_plan"] = SimpleNamespace(regular_lfs=regular_lfs)
        state["regular_lf_paths"] = [f"/tmp/lf_{i:02d}.mp4" for i in (1, 2)]
        assert_phase_d_per_lf_inputs_present(state)

    def test_raises_when_resumed_state_missing_fields(self):
        """Simulates resume from a corrupted state that BC-exit would have
        flagged: the D-entry aggregator must independently catch it."""
        from gomoveshift.pipeline.exceptions import PhaseDEntryContractError
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_d_per_lf_inputs_present,
        )

        regular_lfs = [_lf(1), _lf(2)]
        state = _populated_state(lf_indices=(1, 2))
        state["drive_plan"] = SimpleNamespace(regular_lfs=regular_lfs)
        state["regular_lf_paths"] = [f"/tmp/lf_{i:02d}.mp4" for i in (1, 2)]
        # Strip critical entries to simulate a stale / corrupted state file.
        del state["per_lf_recording_date"][1]
        del state["per_lf_ost_text"][2]

        with pytest.raises(PhaseDEntryContractError) as exc:
            assert_phase_d_per_lf_inputs_present(state)
        msg = str(exc.value)
        assert "per_lf_recording_date" in msg
        assert "per_lf_ost_text" in msg

    def test_empty_drive_plan_skips_check(self):
        """A drive_plan with no regular_lfs (zero-qualifying-footage drive)
        has nothing for D to validate — the contract is vacuous."""
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_d_per_lf_inputs_present,
        )

        state = make_state(
            drive_plan=SimpleNamespace(regular_lfs=[]),
            regular_lf_paths=[],
        )
        # Must not raise; per_lf_* dicts can legitimately be empty here.
        assert_phase_d_per_lf_inputs_present(state)


class TestGpxOmittedIndicesTolerated:
    """A stationary LF whose cleaned trace yields no
    usable distance/elevation is dropped from per_lf_distance_km/
    per_lf_elevation_gain_m ON PURPOSE and recorded in per_lf_gpx_omitted. Both
    boundary contracts must exclude those indices from the GPX-keyed coverage
    check — else the aggregate re-fails on an LF the per-field invariant already
    tolerated."""

    def test_bc_exit_excludes_omitted_distance_index(self):
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_bc_per_lf_complete,
        )

        skeleton = [_lf(1), _lf(2)]
        state = _populated_state(lf_indices=(1, 2))
        # LF2 is a stationary LF: distance + elevation legitimately omitted.
        del state["per_lf_distance_km"][2]
        del state["per_lf_elevation_gain_m"][2]
        state["per_lf_gpx_omitted"] = {
            "per_lf_distance_km": [2],
            "per_lf_elevation_gain_m": [2],
        }
        # Must not raise — index 2 is a deliberate omission, not missing data.
        assert_phase_bc_per_lf_complete(state, skeleton)

    def test_bc_exit_still_raises_on_non_omitted_gap(self):
        from gomoveshift.pipeline.exceptions import PhaseBCExitContractError
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_bc_per_lf_complete,
        )

        skeleton = [_lf(1), _lf(2)]
        state = _populated_state(lf_indices=(1, 2))
        # LF2 omitted (recorded), but LF1 is a genuine gap — must still raise.
        del state["per_lf_distance_km"][2]
        del state["per_lf_distance_km"][1]
        state["per_lf_gpx_omitted"] = {"per_lf_distance_km": [2]}
        with pytest.raises(PhaseBCExitContractError) as exc:
            assert_phase_bc_per_lf_complete(state, skeleton)
        msg = str(exc.value)
        assert "per_lf_distance_km" in msg and "1" in msg

    def test_d_entry_excludes_omitted_index(self):
        from gomoveshift.pipeline.state_invariants import (
            assert_phase_d_per_lf_inputs_present,
        )

        regular_lfs = [_lf(1), _lf(2)]
        state = _populated_state(lf_indices=(1, 2))
        state["drive_plan"] = SimpleNamespace(regular_lfs=regular_lfs)
        state["regular_lf_paths"] = [f"/tmp/lf_{i:02d}.mp4" for i in (1, 2)]
        del state["per_lf_distance_km"][2]
        del state["per_lf_elevation_gain_m"][2]
        state["per_lf_gpx_omitted"] = {
            "per_lf_distance_km": [2],
            "per_lf_elevation_gain_m": [2],
        }
        # Resume reads omitted indices identically — must not raise.
        assert_phase_d_per_lf_inputs_present(state)


class TestPhaseDRunRaisesContractWhenStateIsMissing:
    """Removal-of-defensive-read regression: with per_lf_ost_text missing,
    ``PhaseD.run`` must surface ``PhaseDEntryContractError`` BEFORE any
    defensive read silently swallows the absence."""

    def test_phase_d_run_raises_d_entry_contract_on_partial_per_lf_ost_text(self):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_d import PhaseD
        from gomoveshift.pipeline.exceptions import PhaseDEntryContractError

        # Two LFs rendered, but per_lf_ost_text only covers one — exactly the
        # partial-coverage bug the D-entry contract is built to catch.
        regular_lfs = [_lf(1), _lf(2)]
        populated = _populated_state(lf_indices=(1, 2))
        # Partial coverage: LF 2 is missing — must fail-fast at D entry,
        # BEFORE any defensive `.get(...) or {}` read silently swallows it.
        del populated["per_lf_ost_text"][2]
        state = make_state(
            drive_plan=SimpleNamespace(regular_lfs=regular_lfs),
            regular_lf_paths=["/tmp/lf_01.mp4", "/tmp/lf_02.mp4"],
            region="ireland",
            publish_day="wed",
            location="Killarney",
            output_base="/tmp/gms_test_pr2",
            source_dir="/tmp/source",
            skip_upload=True,
            skip_csv=True,
            **populated,
        )

        phase = PhaseD(config=Config())
        with pytest.raises(PhaseDEntryContractError) as exc:
            phase.run(state)
        assert "per_lf_ost_text" in str(exc.value)
        assert "2" in str(exc.value)
