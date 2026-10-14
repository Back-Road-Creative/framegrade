"""A regular long-form is never inserted with a publish time that has passed.

Phase D builds the drive schedule at its start; thumbnails and the shorts Drive
upload run before the long-form loop, and a long encode can outlast the
``--lf-publish-date`` the launcher passed. ``videos.insert`` with a past
``publishAt`` cannot be taken back, so the loop checks each PENDING LF's slot
against the clock right before the insert:

* nothing of the run live -> the slots are re-resolved through the same
  Waypost resolver a fresh run uses (``LF_SLOT_RERESOLVED``);
* otherwise -> ``LF_SLOT_PASSED`` and no insert.

This module runs on the real clock (the suite-wide pin in ``conftest`` skips
it) and injects ``clock=`` where a test needs a fixed time.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline import seo_slots
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.drive_phases import phase_d as phase_d_mod
from gomoveshift.pipeline.drive_phases.phase_d_scheduling import (
    _build_and_persist_drive_schedule,
    _reresolve_passed_lf_slots,
)
from gomoveshift.pipeline.scheduler import PipelineScheduler
from gomoveshift.pipeline.youtube_sidecar import save_sidecar

from _state_helpers import make_state
from test_phase_d_split_skip_upload import _build_state, _run_phase_d

REAL_LF_INSERT_CLOCK = True

PASSED = "2030-01-08T16:00:00-05:00"
LATER = "2030-01-15T16:00:00-05:00"
NOW = datetime(2030, 1, 9, 12, 0, tzinfo=timezone.utc)


def _schedule(*slots):
    return SimpleNamespace(lf_slots=[SimpleNamespace(datetime_est=s, date=s[:10]) for s in slots])


def _run_loop(tmp_path, schedule, *, state=None, skip=False, insert=None, **kwargs):
    """Drive the loop over ``len(schedule.lf_slots)`` LFs; return the insert mock."""
    n = len(schedule.lf_slots)
    if insert is None:
        insert = MagicMock(side_effect=lambda **kw: {"video_id": f"v{kw['lf'].index}"})
    with (
        patch.object(esc, "_resolve_youtube_uploader_class", return_value=MagicMock()),
        patch.object(esc, "_upload_lf_to_youtube", insert),
        patch.object(esc, "_book_lf_upload_in_ledger"),
    ):
        esc._upload_lfs_to_youtube(
            drive_plan=SimpleNamespace(
                regular_lfs=[SimpleNamespace(index=i) for i in range(1, n + 1)]
            ),
            regular_lf_paths=[f"/v/lf{i}.mp4" for i in range(1, n + 1)],
            metadata_per_lf={i: {"title": f"T{i}"} for i in range(1, n + 1)},
            drive_schedule=schedule,
            youtube_uploads={},
            state=state if state is not None else make_state(),
            output_base=tmp_path,
            config=MagicMock(),
            save_state=lambda s: None,
            skip_youtube_upload=skip,
            youtube_upload_enabled=True,
            **kwargs,
        )
    return insert


def _utc(iso):
    return datetime.fromisoformat(iso).astimezone(timezone.utc)


# ─── the insert loop ────────────────────────────────────────────────────


def test_passed_slot_is_never_inserted(tmp_path):
    insert = MagicMock()
    with pytest.raises(RuntimeError, match=r"LF_SLOT_PASSED lf=1") as exc:
        _run_loop(tmp_path, _schedule(PASSED), insert=insert, clock=lambda: NOW)
    assert "2030-01-08" in str(exc.value)
    insert.assert_not_called()


def test_slot_equal_to_now_counts_as_passed(tmp_path):
    with pytest.raises(RuntimeError, match="LF_SLOT_PASSED"):
        _run_loop(tmp_path, _schedule(PASSED), clock=lambda: _utc(PASSED))


def test_passed_slot_is_reresolved_before_the_insert(tmp_path):
    seen = []

    def reschedule(now):
        seen.append(now)
        return _schedule(LATER)

    insert = _run_loop(tmp_path, _schedule(PASSED), clock=lambda: NOW, reschedule=reschedule)

    assert seen == [NOW]
    assert insert.call_args.kwargs["publish_at"] == _utc(LATER)


def test_resolver_that_declines_refuses(tmp_path):
    """``None`` from the resolver = an LF of the run is live: no heal here."""
    with pytest.raises(RuntimeError, match="LF_SLOT_PASSED"):
        _run_loop(tmp_path, _schedule(PASSED), clock=lambda: NOW, reschedule=lambda now: None)


def test_reresolved_slot_still_passed_refuses(tmp_path):
    with pytest.raises(RuntimeError, match="LF_SLOT_PASSED"):
        _run_loop(
            tmp_path,
            _schedule(PASSED),
            clock=lambda: NOW,
            reschedule=lambda now: _schedule(PASSED),
        )


def test_resolver_halt_propagates(tmp_path):
    def reschedule(now):
        raise seo_slots.CadenceSourceUnreadableError("store unreadable")

    with pytest.raises(seo_slots.CadenceSourceUnreadableError):
        _run_loop(tmp_path, _schedule(PASSED), clock=lambda: NOW, reschedule=reschedule)


def test_future_slot_inserts_without_the_resolver(tmp_path):
    reschedule = MagicMock()
    insert = _run_loop(tmp_path, _schedule(LATER), clock=lambda: NOW, reschedule=reschedule)
    reschedule.assert_not_called()
    assert insert.call_args.kwargs["publish_at"] == _utc(LATER)


def test_live_lf_with_an_aired_slot_is_not_checked(tmp_path):
    """LF 1 aired last week; the retry of LF 2 must not trip over LF 1's slot."""
    state = make_state(youtube_uploads={1: {"video_id": "live1"}})
    reschedule = MagicMock()
    insert = _run_loop(
        tmp_path, _schedule(PASSED, LATER), state=state, clock=lambda: NOW, reschedule=reschedule
    )
    reschedule.assert_not_called()
    assert insert.call_count == 2


def test_skipped_upload_never_reads_the_clock(tmp_path):
    def clock():
        raise AssertionError("clock read on a skipped upload")

    _run_loop(tmp_path, _schedule(PASSED), skip=True, clock=clock)


def test_default_clock_is_the_real_utc_clock(tmp_path):
    """No ``clock=``: the real clock refuses a year-2000 slot, passes a year-2999 one."""
    assert esc._utc_now().utcoffset().total_seconds() == 0
    with pytest.raises(RuntimeError, match="LF_SLOT_PASSED"):
        _run_loop(tmp_path, _schedule("2000-01-04T16:00:00-05:00"))
    assert _run_loop(tmp_path, _schedule("2999-01-01T16:00:00-05:00")).call_count == 1


# ─── the resolver Phase D hands the loop ────────────────────────────────


def _cfg(tmp_path):
    return SimpleNamespace(
        drive=SimpleNamespace(
            shorts_publish_times=["14:00", "18:00"],
            long_form_publish_days=[1],  # Tuesday
            long_form_publish_time="16:00",
        ),
        seo=SimpleNamespace(storage_path=str(tmp_path)),
    )


def _builder_kwargs(tmp_path, state):
    return dict(
        scheduler_cls=PipelineScheduler,
        config=_cfg(tmp_path),
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=0)]),
        state=state,
        region="usa",
        stages_dir=tmp_path,
    )


def _passed_run(tmp_path, **state_kwargs):
    """A run whose launcher anchor (Tue 2030-01-08) is the schedule it built."""
    state = make_state(
        publish_day="tue",
        shorts_per_day=2,
        no_ledger_reconcile=True,
        lf_publish_date="2030-01-08",
        start_date="2030-01-08",
        **state_kwargs,
    )
    current = _build_and_persist_drive_schedule(**_builder_kwargs(tmp_path, state))
    assert current.lf_slots[0].date == "2030-01-08"
    return state, current


def _reresolve(tmp_path, state, current):
    return _reresolve_passed_lf_slots(
        current=current, output_base=tmp_path, now=NOW, **_builder_kwargs(tmp_path, state)
    )


def test_nothing_live_reresolves_through_the_fresh_run_resolver(tmp_path, caplog):
    state, current = _passed_run(tmp_path)

    with caplog.at_level(logging.WARNING, logger="gomoveshift.pipeline.drive_phases.phase_d"):
        fresh = _reresolve(tmp_path, state, current)

    assert fresh.lf_slots[0].date == "2030-01-15"
    assert _utc(fresh.lf_slots[0].datetime_est) > NOW
    assert fresh.identity_times == current.identity_times
    # The passed anchor is gone from state, so a resume does not rebuild on it.
    assert state.lf_publish_date is None
    assert state.start_date is None
    assert state.drive_schedule is fresh
    on_disk = json.loads((tmp_path / "drive_schedule.json").read_text())
    assert on_disk["lf_slots"][0]["date"] == "2030-01-15"
    marker = [r.getMessage() for r in caplog.records if "LF_SLOT_RERESOLVED" in r.getMessage()]
    assert len(marker) == 1
    assert "2030-01-08" in marker[0] and "2030-01-15" in marker[0]


def test_live_lf_in_state_declines(tmp_path):
    state, current = _passed_run(tmp_path, youtube_uploads={0: {"video_id": "live"}})

    assert _reresolve(tmp_path, state, current) is None
    assert state.lf_publish_date == "2030-01-08"
    assert state.start_date == "2030-01-08"


def test_live_lf_in_sidecar_declines(tmp_path):
    state, current = _passed_run(tmp_path)
    save_sidecar(tmp_path, {"0": {"video_id": "live"}})

    assert _reresolve(tmp_path, state, current) is None
    assert state.lf_publish_date == "2030-01-08"


# ─── Phase D hands the loop its resolver ────────────────────────────────


def _far_future_schedule(num_lfs):
    day = date(2999, 1, 1)
    while day.weekday() != 2:  # the harness publishes on Wednesdays
        day += timedelta(days=1)
    days = [(day + timedelta(weeks=i)).isoformat() for i in range(num_lfs)]
    fresh = MagicMock()
    fresh.lf_slots = [
        MagicMock(
            date=d, time="11:00:00", datetime_est=f"{d}T11:00:00-05:00", day_of_week="Wednesday"
        )
        for d in days
    ]
    fresh.shorts_slots = []
    fresh.to_dict.return_value = {"region": "usa"}
    return fresh


def test_phase_d_reresolves_and_carries_the_new_schedule_forward(tmp_path, monkeypatch):
    """The harness schedule (2026-04-29) has passed on the real clock."""
    state = _build_state(tmp_path, num_lfs=2)
    fresh = _far_future_schedule(2)
    reresolve = MagicMock(return_value=fresh)
    monkeypatch.setattr(phase_d_mod, "_reresolve_passed_lf_slots", reresolve)

    _uploader, tracker = _run_phase_d(state, num_lfs=2, monkeypatch=monkeypatch)

    # Asked once (LF 1); LF 2 reads its slot off the same rebuilt schedule.
    assert reresolve.call_count == 1
    kw = reresolve.call_args.kwargs
    assert [s.date for s in kw["current"].lf_slots] == ["2026-04-29", "2026-05-06"]
    assert kw["state"] is state
    assert kw["region"] == "usa"
    assert kw["output_base"] == tmp_path / "output"
    assert kw["now"].utcoffset().total_seconds() == 0
    assert tracker["calls"] == [1, 2]
    assert [state["youtube_uploads"][i]["publish_at"].year for i in (1, 2)] == [2999, 2999]
    # Everything after the loop (shorts CSV, publish plan, state) uses it too.
    assert state.drive_schedule is fresh
