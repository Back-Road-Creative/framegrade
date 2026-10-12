"""LF 2..N of one run must not land on an occupied ledger date.

``generate_drive_schedule`` anchors LF 1 at the base date and adds
``LF_PUBLISH_INTERVAL_DAYS`` per LF, but only the base date was ever checked
against the content ledger. Phase D now checks EVERY LF's date with the same
occupancy ``next_free_slot`` uses and refuses (never silently shifts).
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest
from freezegun import freeze_time

from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.drive_phases.phase_d import _build_and_persist_drive_schedule
from gomoveshift.pipeline.schedule_constants import EST
from gomoveshift.pipeline.scheduler import PipelineScheduler

from _state_helpers import make_state


def _cfg(tmp_path):
    return SimpleNamespace(
        drive=SimpleNamespace(
            shorts_publish_times=["14:00", "18:00"],
            shorts_slot_pool_by_region={"usa": ["14:00", "18:00", "10:00", "16:00"]},
            long_form_publish_days=[1],
            long_form_publish_time="16:00",
        ),
        seo=SimpleNamespace(storage_path=str(tmp_path)),
    )


#: A Saturday. Every date below is relative to it, and the clock is frozen
#: there so the schedule builder's own "today" agrees with the test's.
_TODAY = dt.date(2026, 10, 3)


@pytest.fixture(autouse=True)
def _frozen_clock():
    with freeze_time(dt.datetime(2026, 10, 3, 12, 0, tzinfo=EST)):
        yield


def _tuesday(weeks: int) -> dt.date:
    day = _TODAY + dt.timedelta(days=1)
    while day.weekday() != 1:
        day += dt.timedelta(days=1)
    return day + dt.timedelta(weeks=weeks)


def _book(tmp_path, date, region="usa"):
    ledger = ContentLedger.load(tmp_path / ".content_ledger.json")
    ledger.entries.append(
        ContentEntry(
            date=date.strftime("%Y-%m-%d"),
            time="16:00:00",
            content_type="longform",
            project_name="rival",
            source="pipeline",
            region=region,
            status="scheduled",
        )
    )
    ledger.save(tmp_path / ".content_ledger.json")


def _build(tmp_path, n_lfs, base, **state_kwargs):
    return _build_and_persist_drive_schedule(
        scheduler_cls=PipelineScheduler,
        config=_cfg(tmp_path),
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=i) for i in range(n_lfs)]),
        state=make_state(
            publish_day="tue",
            shorts_per_day=2,
            no_ledger_reconcile=True,
            start_date=base.strftime("%Y-%m-%d"),
            **state_kwargs,
        ),
        region="usa",
        stages_dir=tmp_path,
    )


def test_week_two_occupied_refuses_naming_date_and_index(tmp_path):
    base = _tuesday(1)
    week2 = base + dt.timedelta(days=7)
    _book(tmp_path, week2)
    with pytest.raises(RuntimeError) as exc:
        _build(tmp_path, 2, base)
    msg = str(exc.value)
    assert week2.strftime("%Y-%m-%d") in msg
    assert "LF 2" in msg


def test_all_free_keeps_base_and_base_plus_seven(tmp_path):
    base = _tuesday(1)
    out = _build(tmp_path, 2, base)
    assert [s.date for s in out.lf_slots] == [
        base.strftime("%Y-%m-%d"),
        (base + dt.timedelta(days=7)).strftime("%Y-%m-%d"),
    ]


def test_other_region_booking_does_not_block(tmp_path):
    base = _tuesday(1)
    _book(tmp_path, base + dt.timedelta(days=7), region="ireland")
    out = _build(tmp_path, 2, base)
    assert len(out.lf_slots) == 2


def test_single_lf_run_unchanged(tmp_path):
    base = _tuesday(1)
    _book(tmp_path, base + dt.timedelta(days=7))
    out = _build(tmp_path, 1, base)
    assert [s.date for s in out.lf_slots] == [base.strftime("%Y-%m-%d")]


def test_three_lfs_only_week_three_booked_names_lf_three(tmp_path):
    base = _tuesday(1)
    week3 = base + dt.timedelta(days=14)
    _book(tmp_path, week3)
    with pytest.raises(RuntimeError) as exc:
        _build(tmp_path, 3, base)
    msg = str(exc.value)
    assert "LF 3 of 3" in msg
    assert week3.strftime("%Y-%m-%d") in msg


def test_two_lfs_booking_beyond_last_lf_passes(tmp_path):
    base = _tuesday(1)
    _book(tmp_path, base + dt.timedelta(days=14))
    out = _build(tmp_path, 2, base)
    assert len(out.lf_slots) == 2


def test_resumed_run_with_uploads_skips_guard(tmp_path):
    base = _tuesday(1)
    _book(tmp_path, base + dt.timedelta(days=7))
    uploads = {
        "0": {"publish_at": dt.datetime.combine(base, dt.time(16, 0), tzinfo=EST).isoformat()}
    }
    out = _build(tmp_path, 2, base, youtube_uploads=uploads)
    assert len(out.lf_slots) == 2
