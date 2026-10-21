"""Tests for the OST audit gate inside ``run_day_pipeline``.

Burning OST pixels onto an LF without an operator-visible audit
artefact is unrepresentable. The gate writes
``<output_base>/ost_audit_<date>.json`` and raises
``OstAuditPendingError`` until the operator either

* creates ``<output_base>/ost_audit_<date>.confirmed`` (manual review),
* sets ``state["ost_audit_passed"][date] = True`` (resume marker), or
* re-runs with ``--ost-audit-confirmed`` (non-interactive bypass).

Hash mismatch (operator edits per-LF labels and re-runs phase_bc) re-
raises so the operator re-confirms against the new OST text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gomoveshift.pipeline.day_grouper import DayBatch
from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError


# --- fixtures (mirrors test_day_pipeline.py) --------------------------

NUM_COLUMNS = 9


@dataclass
class _FakeEpisode:
    index: int
    concat_path: Path
    total_duration: float
    date: str
    creation_time: object = None


def _make_scores(duration_sec: int, score: float = 0.5) -> np.ndarray:
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, 0] = np.arange(duration_sec, dtype=np.float64)
    arr[:, 1] = score
    return arr


def _make_day_batch(date: str, episode_count: int = 1, per_ep: float = 600.0):
    eps = [
        _FakeEpisode(
            index=i + 1,
            concat_path=Path(f"/fake/ep_{i:03d}.mp4"),
            total_duration=per_ep,
            date=date,
        )
        for i in range(episode_count)
    ]
    return DayBatch(date=date, episodes=eps)


def _make_lf_plan(index, segments, peaks):
    return SimpleNamespace(index=index, segments=segments, montage_peaks=peaks)


def _make_seg(ep_idx, start, end):
    return SimpleNamespace(
        source_file_index=ep_idx,
        start_time=float(start),
        end_time=float(end),
        duration=float(end - start),
        gps_center_lat=0.0,
        gps_center_lon=0.0,
    )


def _patch_heavy(tmp_path):
    """Mock encode_day_master, extract, render_short, ffmpeg helpers."""
    master = tmp_path / "out" / "day_2025-09-14" / "day_2025-09-14_master.mp4"
    master.parent.mkdir(parents=True, exist_ok=True)
    master.write_bytes(b"x")

    def _fake_concat(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"x")
        return output_path

    def _fake_montage(raw_lf_path, peaks, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"x")
        return output_path

    def _fake_render(source_video, output_path, start_time, duration, location, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"render")
        return True

    return master, [
        patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master),
        patch(
            "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
            side_effect=_fake_concat,
        ),
        patch("gomoveshift.pipeline.day_pipeline.prepend_montage", side_effect=_fake_montage),
        patch("gomoveshift.pipeline.day_pipeline._render_short_ffmpeg", side_effect=_fake_render),
    ]


def _state_with_ost_text(date: str, lf_index: int = 1):
    """Minimal state dict supplying everything format_ost_audit reads."""
    return {
        "per_lf_ost_text": {lf_index: f"Place A to Place B via R{lf_index}, County Kerry"},
        "per_lf_titles": {lf_index: f"Driving — Place A to Place B (Day {date})"},
        "per_lf_recording_date": {lf_index: date},
        "per_lf_locations": {lf_index: "County Kerry"},
    }


def _run(tmp_path, state, date="2025-09-14", lf_indices=(1,)):
    """Invoke run_day_pipeline with mocks; returns (result, mock_apply_ost)."""
    from gomoveshift.pipeline import day_pipeline

    day_batch = _make_day_batch(date, 1, per_ep=600.0)
    day_scores = _make_scores(600, 0.6)
    output_base = tmp_path / "out"
    lf_plans = [_make_lf_plan(i, [_make_seg(0, 100 * i, 100 * i + 100)], []) for i in lf_indices]

    _, patches = _patch_heavy(tmp_path)

    fake_apply = patch(
        "gomoveshift.pipeline.day_pipeline.apply_ost",
        side_effect=lambda input_path, output_path, location, layout, **kw: (
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            or Path(output_path).write_bytes(b"ost")
            or output_path
        ),
    )

    cms = [p.__enter__() for p in patches]
    apply_mock = fake_apply.__enter__()
    try:
        result = day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=day_scores,
            output_dir=output_base,
            location="County Kerry",
            config=None,
            lf_plans=lf_plans,
            per_lf_ost_text=state["per_lf_ost_text"],
            per_lf_locations=state["per_lf_locations"],
            state=state,
        )
    finally:
        fake_apply.__exit__(None, None, None)
        for p in patches:
            p.__exit__(None, None, None)
    return result, apply_mock, output_base


def test_first_run_raises_pending_writes_sidecar_skips_apply_ost(tmp_path):
    state = _state_with_ost_text("2025-09-14", 1)
    with pytest.raises(OstAuditPendingError):
        _run(tmp_path, state)
    # Sidecar landed at output_base/ost_audit_<date>.json
    sidecar = tmp_path / "out" / "ost_audit_2025-09-14.json"
    assert sidecar.exists()


def test_first_run_does_not_call_apply_ost(tmp_path):
    """The gate must fire BEFORE apply_ost — we must not commit pixels."""
    from gomoveshift.pipeline import day_pipeline

    state = _state_with_ost_text("2025-09-14", 1)
    day_batch = _make_day_batch("2025-09-14", 1, per_ep=600.0)
    day_scores = _make_scores(600, 0.6)
    output_base = tmp_path / "out"
    lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]

    _, patches = _patch_heavy(tmp_path)
    cms = [p.__enter__() for p in patches]
    fake_apply = patch("gomoveshift.pipeline.day_pipeline.apply_ost")
    apply_mock = fake_apply.__enter__()
    try:
        with pytest.raises(OstAuditPendingError):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="County Kerry",
                config=None,
                lf_plans=lf_plans,
                per_lf_ost_text=state["per_lf_ost_text"],
                per_lf_locations=state["per_lf_locations"],
                state=state,
            )
        assert apply_mock.call_count == 0
    finally:
        fake_apply.__exit__(None, None, None)
        for p in patches:
            p.__exit__(None, None, None)


def test_confirmed_file_passes_gate_and_calls_apply_ost(tmp_path):
    """When ost_audit_<date>.confirmed exists, the LF gate passes.

    The shorts pre-burn gate is a separate confirmation, so the run also
    needs ``shorts_ost_audit_<date>.confirmed`` to complete past the shorts.
    """
    state = _state_with_ost_text("2025-09-14", 1)
    output_base = tmp_path / "out"
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "ost_audit_2025-09-14.confirmed").write_text("ok")
    (output_base / "shorts_ost_audit_2025-09-14.confirmed").write_text("ok")

    result, apply_mock, _ = _run(tmp_path, state)
    assert result is not None
    assert apply_mock.call_count == 1


def test_state_flag_passes_gate(tmp_path):
    """When state['ost_audit_passed'][date] is True, the LF gate passes.

    The shorts pre-burn gate is independent, so the run also drops a
    ``shorts_ost_audit_<date>.confirmed`` to complete past the shorts.
    """
    state = _state_with_ost_text("2025-09-14", 1)
    state["ost_audit_passed"] = {"2025-09-14": True}
    # Need matching hash too — the gate compares hashes when state says passed.
    from gomoveshift.pipeline.ost_audit_sidecar import (
        compute_ost_audit_hash,
        format_ost_audit,
    )

    payload = format_ost_audit("2025-09-14", state, [1])
    state["ost_audit_hash"] = {"2025-09-14": compute_ost_audit_hash(payload)}
    output_base = tmp_path / "out"
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "shorts_ost_audit_2025-09-14.confirmed").write_text("ok")

    result, apply_mock, _ = _run(tmp_path, state)
    assert result is not None
    assert apply_mock.call_count == 1


def test_stale_confirmed_binding_reopens_lf_gate(tmp_path):
    """A ``.confirmed`` bound (stamped) to a DIFFERENT audit hash must reopen
    the LF gate — ``exists()`` alone can't validate a never-reviewed burn.

    A valid shorts confirmation is supplied so ONLY the LF gate can raise."""
    from gomoveshift.pipeline.ost_audit_sidecar import stamp_confirmed

    state = _state_with_ost_text("2025-09-14", 1)
    output_base = tmp_path / "out"
    output_base.mkdir(parents=True, exist_ok=True)
    stamp_confirmed(output_base / "ost_audit_2025-09-14.confirmed", "0" * 64)
    (output_base / "shorts_ost_audit_2025-09-14.confirmed").write_text("ok")

    with pytest.raises(OstAuditPendingError):
        _run(tmp_path, state)


def test_hash_mismatch_re_raises_after_state_confirmation(tmp_path):
    """If state says passed but the hash differs (operator changed labels),
    re-prompt by re-raising OstAuditPendingError."""
    state = _state_with_ost_text("2025-09-14", 1)
    state["ost_audit_passed"] = {"2025-09-14": True}
    state["ost_audit_hash"] = {"2025-09-14": "stale-hash-from-prior-run"}

    with pytest.raises(OstAuditPendingError):
        _run(tmp_path, state)


def test_ost_audit_confirmed_flag_in_state_bypasses_and_persists(tmp_path):
    """state['ost_audit_confirmed_this_run'] = True (set when --ost-audit-confirmed
    is passed on THIS run) bypasses the file/hash check and persists the pass
    marker. A persisted-only confirmation no longer bypasses."""
    state = _state_with_ost_text("2025-09-14", 1)
    state["ost_audit_confirmed_this_run"] = True

    result, apply_mock, _ = _run(tmp_path, state)
    assert result is not None
    assert apply_mock.call_count == 1
    # Persistence: the gate must record the pass + hash for resume.
    assert state.get("ost_audit_passed", {}).get("2025-09-14") is True
    assert "2025-09-14" in state.get("ost_audit_hash", {})


# --- OstAuditPendingError propagation through run_all_days ------
#
# `run_all_days` wraps per-day `run_day_pipeline` calls in a generic
# `try/except Exception` (see day_pipeline.py). That wrapper converts any
# exception into a logged
# "continue to next day" event, which silently swallows the OST audit
# gate's clean-exit signal: `OstAuditPendingError` never reaches the
# orchestrator's `except OstAuditPendingError` clause
# (drive_orchestrator.py), so an audit-pending exit shows up as
# `success=False, errors=[RuntimeError]` instead of `success=True,
# stages_completed=[phase_a]`.
#
# The fix is to let `OstAuditPendingError` propagate unwrapped. All OTHER
# exceptions keep the existing crash-recovery behaviour.


def _make_minimal_day_batch_with_concat(date: str):
    """A DayBatch whose single episode has a concat_path that matches
    the file_boundaries map passed to run_all_days."""
    ep = _FakeEpisode(
        index=1,
        concat_path=Path(f"/fake/ep_{date}.mp4"),
        total_duration=600.0,
        date=date,
    )
    return DayBatch(date=date, episodes=[ep])


def test_run_all_days_propagates_ost_audit_pending_unwrapped(tmp_path):
    """`OstAuditPendingError` raised inside `run_day_pipeline` must
    bubble out of `run_all_days` with its type preserved, so the
    orchestrator's `except OstAuditPendingError` branch fires."""
    from gomoveshift.pipeline import day_pipeline

    date = "2025-09-14"
    day_batch = _make_minimal_day_batch_with_concat(date)
    drive_scores = _make_scores(600, 0.6)
    # One episode at drive offset [0, 600).
    file_boundaries = [(0.0, 600.0)]
    episode_paths = [str(day_batch.episodes[0].concat_path)]

    def _raise_pending(**kwargs):
        raise OstAuditPendingError(f"OST audit pending for day {date}: hash mismatch")

    with patch(
        "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
        side_effect=_raise_pending,
    ):
        with pytest.raises(OstAuditPendingError):
            day_pipeline.run_all_days(
                day_batches=[day_batch],
                drive_scores=drive_scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path / "out",
                location="County Kerry",
                episode_paths=episode_paths,
            )


def test_run_all_days_still_wraps_generic_exceptions(tmp_path):
    """Regression guard: a generic exception from `run_day_pipeline`
    keeps the existing crash-recovery behaviour — it is caught,
    logged, and the day is skipped so the outer loop continues. Only
    `OstAuditPendingError` is exempt from this wrapping."""
    from gomoveshift.pipeline import day_pipeline

    date = "2025-09-14"
    day_batch = _make_minimal_day_batch_with_concat(date)
    drive_scores = _make_scores(600, 0.6)
    file_boundaries = [(0.0, 600.0)]
    episode_paths = [str(day_batch.episodes[0].concat_path)]

    def _raise_value_error(**kwargs):
        raise ValueError("some unrelated failure")

    with patch(
        "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
        side_effect=_raise_value_error,
    ):
        # Must NOT raise — the generic-exception path stays wrapped.
        results = day_pipeline.run_all_days(
            day_batches=[day_batch],
            drive_scores=drive_scores,
            file_boundaries=file_boundaries,
            output_base=tmp_path / "out",
            location="County Kerry",
            episode_paths=episode_paths,
        )
    # Day crashed → no artifacts collected, but the call returned cleanly.
    assert results == []
