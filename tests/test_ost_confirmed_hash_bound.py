"""A persisted ``--ost-audit-confirmed`` must not auto-pass a changed,
never-reviewed OST burn after a hard kill lost the audit hash.

The bug: ``state["ost_audit_confirmed"]`` persists across resume, but the
audit hash only landed at day-complete. A ``SIGKILL`` mid-encode + a label
edit before resume → the persisted confirmation blanket-bypassed the gate for
materially changed burn text (``confirmation_is_stale(None, x)`` is False when
the hash was lost). Two robust guards:

* The confirmation bypass is honored ONLY when the flag was supplied on THIS
  invocation (``ost_audit_confirmed_this_run`` — never persisted). A
  persisted-only confirmation carries no bypass; a resume re-passes through the
  hash-bound ``ost_audit_passed`` + ``ost_audit_hash`` markers instead.
* The front-load gate persists its hash markers immediately (via
  ``state_save_fn``) so a mid-encode kill can no longer lose them, and the
  existing stale-reopen logic sees the real prior hash on resume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gomoveshift.pipeline.day_grouper import DayBatch
from gomoveshift.pipeline.day_pipeline import run_all_days
from gomoveshift.pipeline.ost_audit_sidecar import (
    OstAuditPendingError,
    compute_ost_audit_hash,
    format_ost_audit,
    lf_gate_or_raise,
    shorts_gate_or_raise,
)

DATE = "2025-09-14"
D1, D2 = "2026-02-20", "2026-02-21"


def _lf_state(text="Killarney to Dingle via R560, County Kerry"):
    return {"per_lf_ost_text": {1: text}}


def _shorts_entries(text="Killarney to Dingle"):
    return [
        {
            "rank": 1,
            "parent_lf_index": 0,
            "ost_text": text,
            "start_sec": 10.0,
            "end_sec": 40.0,
        }
    ]


# ── Fix (1): persisted confirmation is hash-bound; fresh confirmation isn't ──


def test_persisted_confirmation_without_hash_reopens_lf_gate(tmp_path):
    """A persisted ``ost_audit_confirmed=True`` with NO matching stored hash
    (the lost-hash-after-kill state) must NOT auto-pass — re-review required."""
    state = _lf_state()
    state["ost_audit_confirmed"] = True  # persisted from a prior run; hash lost
    with pytest.raises(OstAuditPendingError):
        lf_gate_or_raise(tmp_path, DATE, [1], state)


def test_fresh_confirmation_bypasses_lf_gate(tmp_path):
    """A confirmation supplied THIS invocation still bypasses first-time
    (no stored hash) — the operator is confirming the current text now."""
    state = _lf_state()
    state["ost_audit_confirmed_this_run"] = True
    lf_gate_or_raise(tmp_path, DATE, [1], state)  # must NOT raise
    assert state["ost_audit_passed"][DATE] is True
    assert DATE in state["ost_audit_hash"]


def test_matching_stored_hash_resume_passes_lf_gate(tmp_path):
    """A resume with a hash-bound pass marker (no live flag) still passes when
    the burn text is unchanged."""
    payload = format_ost_audit(DATE, _lf_state(), [1])
    good_hash = compute_ost_audit_hash(payload)
    state = _lf_state()
    state["ost_audit_passed"] = {DATE: True}
    state["ost_audit_hash"] = {DATE: good_hash}
    lf_gate_or_raise(tmp_path, DATE, [1], state)  # must NOT raise
    assert state["ost_audit_passed"][DATE] is True


def test_persisted_confirmation_with_changed_text_reopens_lf_gate(tmp_path):
    """Persisted confirmation + a stored hash for the OLD text + a changed
    burn text → stale reopen (belt to the fresh-only braces)."""
    old_hash = compute_ost_audit_hash(format_ost_audit(DATE, _lf_state("Old label"), [1]))
    state = _lf_state("New label via R1, County Clare")  # materially changed
    state["ost_audit_confirmed"] = True
    state["ost_audit_hash"] = {DATE: old_hash}
    with pytest.raises(OstAuditPendingError):
        lf_gate_or_raise(tmp_path, DATE, [1], state)


def test_persisted_confirmation_without_hash_reopens_shorts_gate(tmp_path):
    """Shorts analog: a persisted-only confirmation carries no bypass."""
    state = {"ost_audit_confirmed": True}
    with pytest.raises(OstAuditPendingError):
        shorts_gate_or_raise(tmp_path, DATE, _shorts_entries(), state)


def test_fresh_confirmation_bypasses_shorts_gate(tmp_path):
    """Shorts analog: a THIS-run confirmation bypasses first-time."""
    state = {"ost_audit_confirmed_this_run": True}
    shorts_gate_or_raise(tmp_path, DATE, _shorts_entries(), state)  # must NOT raise
    assert state["shorts_ost_audit_passed"][DATE] is True
    assert DATE in state["shorts_ost_audit_hash"]


# ── Fix (2): front-load gate persists the hash markers immediately ──


@dataclass
class _Ep:
    index: int
    concat_path: Path
    total_duration: float
    date: str
    creation_time: object = None


def _run_frontload_with_save(tmp_path, state, spy, save_fn):
    ep1 = _Ep(1, Path("/fake/ep_000.mp4"), 600.0, D1)
    ep2 = _Ep(2, Path("/fake/ep_001.mp4"), 600.0, D2)
    scores = np.zeros((1200, 9), dtype=np.float64)
    scores[:, 0] = np.arange(1200, dtype=np.float64)
    scores[:, 1] = 0.6
    out = tmp_path / "stages"
    out.mkdir(parents=True, exist_ok=True)
    with patch("gomoveshift.pipeline.day_pipeline.run_day_pipeline", side_effect=spy):
        run_all_days(
            day_batches=[DayBatch(date=D1, episodes=[ep1]), DayBatch(date=D2, episodes=[ep2])],
            drive_scores=scores,
            file_boundaries=[(0.0, 600.0), (600.0, 1200.0)],
            output_base=out,
            location="County Kerry",
            episode_paths=[str(ep1.concat_path), str(ep2.concat_path)],
            lf_plans_by_date={D1: [SimpleNamespace(index=1)], D2: [SimpleNamespace(index=2)]},
            per_lf_ost_text_by_date={
                D1: {1: "Alpha to Bravo via R1, County Kerry"},
                D2: {2: "Delta to Echo via R3, County Clare"},
            },
            state=state,
            state_save_fn=save_fn,
        )


def test_frontload_persists_hash_before_first_encode(tmp_path):
    """A confirmed front-load run must persist the OST hash markers BEFORE the
    first ``run_day_pipeline`` (encode), so a mid-encode kill can't lose them."""
    encode_calls: list = []
    save_snapshots: list = []

    def spy(**kw):
        # At the first encode, the hash markers must already be persisted.
        encode_calls.append(kw.get("day_batch").date)

    def save_fn(st):
        save_snapshots.append(dict(st.get("ost_audit_hash") or {}))

    state = {"ost_audit_confirmed_this_run": True}
    _run_frontload_with_save(tmp_path, state, spy, save_fn)

    assert save_snapshots, "state_save_fn was never called after the front-load gate"
    # The first save carries BOTH days' hashes, and happens before any encode.
    assert set(save_snapshots[0]) == {D1, D2}
    assert encode_calls == [D1, D2]
