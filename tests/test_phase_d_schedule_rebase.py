"""drive_schedule rebase-on-stale-resume contract.

Three guards + two backstops in ``_maybe_rebase_drive_schedule``. Clock
injection via ``now=``; never patch ``datetime``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gomoveshift.pipeline.drive_phases.phase_d import _maybe_rebase_drive_schedule
from gomoveshift.pipeline.scheduler import DriveReleaseSchedule, ScheduledSlot

from _state_helpers import make_state

NOW = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
STALE = NOW - timedelta(hours=20)
RECENT = NOW - timedelta(hours=2)


def _sched(first_lf_date: str, computed_at: datetime | None) -> DriveReleaseSchedule:
    base = datetime.fromisoformat(f"{first_lf_date}T11:00:00+00:00")
    return DriveReleaseSchedule(
        region="ireland",
        lf_slots=[
            ScheduledSlot(0, first_lf_date, "11:00:00", base.isoformat(), "Mon"),
            ScheduledSlot(
                0,
                (base + timedelta(weeks=1)).date().isoformat(),
                "11:00:00",
                (base + timedelta(weeks=1)).isoformat(),
                "Mon",
            ),
        ],
        shorts_slots=[],
        computed_at=computed_at.isoformat() if computed_at else None,
    )


@pytest.fixture
def output_base(tmp_path: Path) -> Path:
    return tmp_path


def test_no_old_schedule_skips(output_base):
    state = make_state()
    info = _maybe_rebase_drive_schedule(state, output_base, now=NOW)
    assert info["action"] == "no_old_schedule"


def test_recent_schedule_not_rebased(output_base):
    state = make_state(drive_schedule=_sched("2026-05-11", RECENT), start_date="2026-05-25")
    info = _maybe_rebase_drive_schedule(state, output_base, now=NOW)
    assert info["action"] == "fresh"
    assert state["start_date"] == "2026-05-25"


def test_stale_but_future_publish_not_rebased(output_base):
    state = make_state(drive_schedule=_sched("2026-05-25", STALE), start_date="2026-05-25")
    info = _maybe_rebase_drive_schedule(state, output_base, now=NOW)
    assert info["action"] == "future"
    assert state["start_date"] == "2026-05-25"


def test_three_guards_pass_rebases_and_logs(output_base, caplog):
    """All three guards pass → start_date cleared, INFO log names old slots."""
    state = make_state(drive_schedule=_sched("2026-05-11", STALE), start_date="2026-05-11")
    with caplog.at_level(logging.INFO, logger="gomoveshift.pipeline.drive_phases.phase_d"):
        info = _maybe_rebase_drive_schedule(state, output_base, now=NOW)
    assert info["action"] == "rebase"
    assert state["start_date"] is None
    assert info["old_first_publish"] is not None and "2026-05-11" in info["old_first_publish"]
    joined = " ".join(rec.getMessage() for rec in caplog.records)
    assert "REBASED" in joined and "OLD lf_slots dates" in joined and "2026-05-11" in joined


def test_youtube_uploads_present_no_rebase(output_base):
    """Any uploaded LF blocks rebase — splitting publish across weeks is worse than stale."""
    (output_base / "youtube_uploads.json").write_text(
        json.dumps({"1": {"video_id": "abc", "publish_at": "2026-05-11T11:00"}})
    )
    state = make_state(drive_schedule=_sched("2026-05-11", STALE), start_date="2026-05-11")
    info = _maybe_rebase_drive_schedule(state, output_base, now=NOW)
    assert info["action"] == "uploads_present"
    assert state["start_date"] == "2026-05-11"


def test_missing_computed_at_treated_stale(output_base):
    """Legacy schedule (no computed_at) + past publish + no uploads → rebase."""
    state = make_state(drive_schedule=_sched("2026-05-11", None), start_date="2026-05-11")
    info = _maybe_rebase_drive_schedule(state, output_base, now=NOW)
    assert info["action"] == "rebase"
    assert state["start_date"] is None


def test_lf_publish_date_explicit_wins(output_base):
    """--lf-publish-date wins over auto-rebase AND over --force-no-rebase."""
    state = make_state(
        drive_schedule=_sched("2026-05-11", STALE),
        start_date="2026-05-11",
    )
    state.lf_publish_date = "2026-06-01"
    state.force_no_rebase = True
    info = _maybe_rebase_drive_schedule(state, output_base, now=NOW)
    assert info["action"] == "explicit"
    assert state.start_date == "2026-06-01"


def test_force_no_rebase_preserves_stale_schedule(output_base):
    """--force-no-rebase short-circuits before guard evaluation."""
    state = make_state(
        drive_schedule=_sched("2026-05-11", STALE),
        start_date="2026-05-11",
    )
    state.force_no_rebase = True
    info = _maybe_rebase_drive_schedule(state, output_base, now=NOW)
    assert info["action"] == "force_no_rebase"
    assert state.start_date == "2026-05-11"


def test_dataclass_shape_schedule_after_json_round_trip_rebases(output_base):
    """Resume hands back drive_schedule as a ``DriveReleaseSchedule``
    dataclass (``_deserialize_dataclass`` rehydrates on load). The helper now
    uses direct attribute access — the legacy dict-shape tolerance shim
    (``_attr_or_key``) has been deleted.
    """
    state = make_state(
        drive_schedule=_sched("2026-05-11", STALE),
        start_date="2026-05-11",
    )
    info = _maybe_rebase_drive_schedule(state, output_base, now=NOW)
    assert info["action"] == "rebase"
    assert state["start_date"] is None
