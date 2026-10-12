"""Observability tests for the resume-time backstops.

Each backstop used to recover silently, so operators had no signal when a
phase boundary was breached on resume. Three production fire-points report themselves:

1. ``per_lf_trim_ratio`` ffprobe (phase_bc_day_first)
2. ``per_lf_recording_date`` path derivation (phase_bc_day_first)
3. ``file_boundaries`` auto-compute (drive_orchestrator)

``test_backstop_shorts_per_lf_logs_warn_and_records_recovery`` pins the helper's
whole-state ``[-1]`` sentinel for a ``shorts_per_lf`` reason, because the upload step
refuses an absent ``shorts_per_lf`` instead of falling back to a legacy layout.

Contract per fire-point:

- ``logger.warning("backstop fired: ...")`` with field name + LF indices +
  resume-context reason.
- ``state["per_lf_state_recovered_from_disk"][<field>] = [<indices>]``
  with ``[-1]`` sentinel for whole-state recoveries like ``file_boundaries``.

Recovery itself is unchanged; these tests pin only the visibility.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from _state_helpers import make_state


# ─── Backstop #1: per_lf_trim_ratio path derivation ─────────────────────


def _seg(source_file_index: int):
    return SimpleNamespace(
        source_file_index=source_file_index,
        source_path=f"/tmp/ep_{source_file_index}.mp4",
        start_time=0.0,
        end_time=10.0,
        original_start_time=0.0,
        original_end_time=10.0,
    )


def _lf(index: int, segments, total_duration: float = 0.0):
    return SimpleNamespace(
        index=index,
        segments=segments,
        montage_peaks=[],
        total_duration=total_duration,
    )


def test_backstop_trim_ratio_logs_warn_and_records_recovery(tmp_path, caplog):
    """Backstop #1: when the fresh-compute path returns an incomplete map
    AND the path-derivation fills the gap, a WARN log fires and the
    recovery dict captures the recovered indices.
    """
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        _assert_post_encoding_invariants,
    )

    # Both LFs have on-disk finals → path derivation fills via ffprobe.
    # Fresh-compute path returns EMPTY because the LF's total_duration is
    # 0 (the fresh-compute branch requires final_duration > 0 to produce
    # an entry, simulating a resume from a state whose skeleton LFs were
    # rehydrated without total_duration populated).
    lf1_path = tmp_path / "day_2026-02-20" / "regular_lfs" / "lf_01.mp4"
    lf2_path = tmp_path / "day_2026-02-20" / "regular_lfs" / "lf_02.mp4"
    lf1_path.parent.mkdir(parents=True)
    lf1_path.write_bytes(b"x")
    lf2_path.write_bytes(b"x")

    # total_duration=0 → fresh-compute drops both LFs (resume-style state).
    skeleton_lfs = [_lf(1, [_seg(0)], 0.0), _lf(2, [_seg(1)], 0.0)]
    state = make_state(file_durations=[1000.0, 1000.0])

    # Monkeypatch ffprobe so path-derivation produces non-zero durations.
    import gomoveshift.pipeline.drive_phases.phase_bc_day_first as _mod

    original_probe = _mod._probe_duration
    _mod._probe_duration = lambda p: 300.0 if "lf_01" in str(p) else 200.0
    try:
        with caplog.at_level(logging.WARNING):
            _assert_post_encoding_invariants(
                state=state,
                skeleton_lfs=skeleton_lfs,
                regular_lf_paths=[lf1_path, lf2_path],
            )
    finally:
        _mod._probe_duration = original_probe

    recovery = state.get("per_lf_state_recovered_from_disk") or {}
    assert "per_lf_trim_ratio" in recovery
    assert sorted(recovery["per_lf_trim_ratio"]) == [1, 2]
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("backstop fired:" in m and "per_lf_trim_ratio" in m for m in warn_msgs)


# ─── Backstop #2: per_lf_recording_date path derivation ─────────────────


def test_backstop_recording_date_logs_warn_and_records_recovery(tmp_path, caplog):
    """Backstop #2: ``_derive_per_lf_recording_date_from_paths`` fills
    entries that ``_build_per_lf_recording_date_map`` missed (resume
    fast-path, no DrivePlan). When ANY index is path-derived, fire WARN
    and record the indices.
    """
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        _record_recording_date_recovery,
    )

    state = make_state()
    partition_map = {1: "2026-02-20"}  # LF 1 from partition
    path_map = {2: "2026-02-21", 3: "2026-02-21"}  # LFs 2, 3 derived
    with caplog.at_level(logging.WARNING):
        _record_recording_date_recovery(state, partition_map, path_map)

    recovery = state.get("per_lf_state_recovered_from_disk") or {}
    assert "per_lf_recording_date" in recovery
    assert sorted(recovery["per_lf_recording_date"]) == [2, 3]
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("backstop fired:" in m and "per_lf_recording_date" in m for m in warn_msgs)


def test_backstop_recording_date_silent_when_partition_covers_all(caplog):
    """When the partition-derived map already covers every LF the
    path-derived map references, no backstop fired."""
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        _record_recording_date_recovery,
    )

    state = make_state()
    partition_map = {1: "2026-02-20", 2: "2026-02-21"}
    path_map = {1: "2026-02-20", 2: "2026-02-21"}  # already covered
    with caplog.at_level(logging.WARNING):
        _record_recording_date_recovery(state, partition_map, path_map)

    recovery = state.get("per_lf_state_recovered_from_disk") or {}
    assert "per_lf_recording_date" not in recovery
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("backstop fired:" in m for m in warn_msgs)


# ─── Backstop #3: file_boundaries auto-compute ─────────────────────────


def test_backstop_file_boundaries_logs_warn_and_records_recovery(caplog):
    """Backstop #3: drive_orchestrator reconstructs file_boundaries from
    file_durations when missing (legacy state files). Helper records a
    whole-state recovery with the ``[-1]`` sentinel.
    """
    from gomoveshift.pipeline.state_invariants import record_backstop_recovery

    state = make_state()
    with caplog.at_level(logging.WARNING):
        record_backstop_recovery(
            state,
            "file_boundaries",
            [-1],
            "legacy state file: reconstructed from file_durations",
        )

    recovery = state.get("per_lf_state_recovered_from_disk") or {}
    assert recovery["file_boundaries"] == [-1]
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("backstop fired:" in m and "file_boundaries" in m for m in warn_msgs)
    assert any("legacy state file" in m for m in warn_msgs)


# ─── Whole-state sentinel: shorts_per_lf ────────────────────────────


def test_backstop_shorts_per_lf_logs_warn_and_records_recovery(caplog):
    """Whole-state sentinel: a ``shorts_per_lf`` recovery is recorded as ``[-1]``
    (the entire map is missing), not a per-LF gap fill. The helper is called
    directly; no production path falls back to the legacy directory layout.
    """
    from gomoveshift.pipeline.state_invariants import record_backstop_recovery

    state = make_state()
    with caplog.at_level(logging.WARNING):
        record_backstop_recovery(
            state,
            "shorts_per_lf",
            [-1],
            "legacy-layout fallback: legacy output_base/shorts/lf_NN/ directory layout",
        )

    recovery = state.get("per_lf_state_recovered_from_disk") or {}
    assert recovery["shorts_per_lf"] == [-1]
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert any("backstop fired:" in m and "shorts_per_lf" in m for m in warn_msgs)
    assert any("legacy-layout fallback" in m for m in warn_msgs)


# ─── Fresh-path: no backstop fires, recovery dict stays empty ──────────


def test_no_backstop_fires_on_complete_fresh_state(tmp_path, caplog):
    """Fresh-encode path with full fresh-compute coverage → recovery
    dict stays empty and no WARN logs for the trim_ratio backstop."""
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        _assert_post_encoding_invariants,
    )

    lf1_path = tmp_path / "day_2026-02-20" / "regular_lfs" / "lf_01.mp4"
    lf1_path.parent.mkdir(parents=True)
    lf1_path.write_bytes(b"x")

    skeleton_lfs = [_lf(1, [_seg(0)], 300.0)]
    state = make_state(file_durations=[1000.0])  # fresh compute: trim_ratio = 0.3
    with caplog.at_level(logging.WARNING):
        _assert_post_encoding_invariants(
            state=state,
            skeleton_lfs=skeleton_lfs,
            regular_lf_paths=[lf1_path],
        )

    recovery = state.get("per_lf_state_recovered_from_disk") or {}
    assert "per_lf_trim_ratio" not in recovery
    warn_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("backstop fired:" in m for m in warn_msgs)


# ─── state_invariants: per_lf_state_recovered_from_disk shape gate ─────


def test_recovery_dict_int_list_value_passes():
    """Well-formed recovery dict: str keys, list[int] values."""
    from gomoveshift.pipeline.state_invariants import assert_state_invariants

    state = make_state(per_lf_state_recovered_from_disk={"per_lf_trim_ratio": [1, 2, 3]})
    assert_state_invariants(state)  # no raise


def test_recovery_dict_non_int_value_raises():
    """Recovery dict values must be lists of int — catches bug class."""
    from gomoveshift.pipeline.exceptions import StateInvariantError
    from gomoveshift.pipeline.state_invariants import (
        STATE_INVARIANT_ERROR_PREFIX,
        assert_state_invariants,
    )

    state = make_state(per_lf_state_recovered_from_disk={"per_lf_trim_ratio": ["1"]})
    with pytest.raises(StateInvariantError) as exc_info:
        assert_state_invariants(state)
    msg = str(exc_info.value)
    assert msg.startswith(STATE_INVARIANT_ERROR_PREFIX)
    assert "per_lf_state_recovered_from_disk" in msg


def test_recovery_dict_non_list_value_raises():
    """Recovery dict values must be lists, not bare ints."""
    from gomoveshift.pipeline.exceptions import StateInvariantError
    from gomoveshift.pipeline.state_invariants import assert_state_invariants

    state = make_state(per_lf_state_recovered_from_disk={"per_lf_trim_ratio": 1})
    with pytest.raises(StateInvariantError):
        assert_state_invariants(state)
