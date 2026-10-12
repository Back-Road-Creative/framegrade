"""Front-load ALL days' LF OST audits before the first master encode.

Every day's per-LF burn text is fully derivable BEFORE any master encode, so
``run_all_days`` runs the LF OST-audit gate for every day UPFRONT — writing each
``ost_audit_<date>.json`` and raising ``OstAuditPendingError`` ONCE before the
first ``run_day_pipeline`` (hence before the first ``encode_day_master``).
Previously the gate raised per-day AFTER that day's multi-hour master encode
(one stop/resume cycle per day).
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
from gomoveshift.pipeline.ost_audit_sidecar import OstAuditPendingError

D1, D2 = "2026-02-20", "2026-02-21"


@dataclass
class _Ep:
    index: int
    concat_path: Path
    total_duration: float
    date: str
    creation_time: object = None


def _lf(index: int):
    return SimpleNamespace(index=index, segments=[], montage_peaks=[])


def _call(tmp_path, state, spy):
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
            lf_plans_by_date={D1: [_lf(1), _lf(2)], D2: [_lf(3), _lf(4)]},
            per_lf_ost_text_by_date={
                D1: {1: "Alpha to Bravo via R1, County Kerry", 2: "Bravo to Charlie via R2, Kerry"},
                D2: {3: "Delta to Echo via R3, County Clare", 4: "Echo to Foxtrot via R4, Clare"},
            },
            state=state,
        )
    return out


def test_all_days_audits_written_before_first_encode(tmp_path):
    """Unconfirmed multi-day run: raise ONCE upfront, ZERO encodes, and BOTH
    days' sidecars on disk before the first ``run_day_pipeline``."""
    calls: list = []
    out = None
    with pytest.raises(OstAuditPendingError):
        out = _call(tmp_path, {}, lambda **kw: calls.append(kw.get("day_batch")))
    out = tmp_path / "stages"
    assert calls == []  # run_day_pipeline (owns encode_day_master) never ran
    assert (out / f"ost_audit_{D1}.json").exists()
    assert (out / f"ost_audit_{D2}.json").exists()


def test_confirmed_run_clears_all_days_and_reaches_encode(tmp_path):
    """A ``--ost-audit-confirmed`` run clears every day upfront (no raise) and
    proceeds into ``run_day_pipeline`` for all days, persisting the markers the
    per-day gate fast-passes on."""
    calls: list = []
    state: dict = {"ost_audit_confirmed_this_run": True}
    _call(tmp_path, state, lambda **kw: calls.append(kw.get("day_batch").date) or None)
    assert set(calls) == {D1, D2}
    assert state.get("ost_audit_passed") == {D1: True, D2: True}
    assert set(state.get("ost_audit_hash") or {}) == {D1, D2}


def test_lf_gate_reopens_on_burn_text_change(tmp_path):
    """A burn-text change on a confirmed day still reopens the audit — the
    front-load never weakens the hash-bound reopen-on-change behaviour."""
    from gomoveshift.pipeline.ost_audit_sidecar import lf_gate_or_raise

    state: dict = {
        "per_lf_ost_text": {1: "Alpha to Bravo via R1, Kerry"},
        "ost_audit_confirmed_this_run": True,
    }
    lf_gate_or_raise(tmp_path, D1, [1], state)
    assert state["ost_audit_passed"][D1] is True
    state["ost_audit_confirmed_this_run"] = False
    state["per_lf_ost_text"] = {1: "Alpha to Zulu via R9, Clare"}  # material change
    with pytest.raises(OstAuditPendingError):
        lf_gate_or_raise(tmp_path, D1, [1], state)
