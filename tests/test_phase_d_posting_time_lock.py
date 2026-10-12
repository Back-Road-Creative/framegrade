"""The run-wide shorts posting-time lock survives resume.

Contract (docs/pipeline/79-publish-cadence-resolution.md, run-wide posting-time lock):
a drive locks ONE consistent pair of shorts posting times for its whole run.
``_build_and_persist_drive_schedule`` re-runs on every Phase-D invocation
(including a resumed run). The first build
resolves the pair against the shared ledger; every later build must READ BACK
that persisted pair instead of re-resolving — because by then the ledger holds
the drive's OWN booked shorts, which would otherwise contest its own identity
times and flip a resumed run onto a DIFFERENT pair.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.drive_phases.phase_d import _build_and_persist_drive_schedule
from gomoveshift.pipeline.scheduler import PipelineScheduler

from _state_helpers import make_state


def _lf(idx):
    return SimpleNamespace(index=idx)


def _shorts_times(schedule):
    return sorted({s.time for s in schedule.shorts_slots})


def _cfg(tmp_path):
    """Real-shaped config stub: USA prefers 14:00/18:00, pool falls to 10:00/16:00."""
    return SimpleNamespace(
        drive=SimpleNamespace(
            shorts_publish_times=["14:00", "18:00"],
            shorts_slot_pool_by_region={"usa": ["14:00", "18:00", "10:00", "16:00"]},
            long_form_publish_days=[1],  # Tuesday
            long_form_publish_time="16:00",
        ),
        seo=SimpleNamespace(storage_path=str(tmp_path)),
    )


def _future_start():
    """A start date safely in the real future (ledger occupancy filters on now)."""
    return (dt.datetime.now().date() + dt.timedelta(days=60)).strftime("%Y-%m-%d")


def _book_own_shorts(tmp_path, start_date):
    """Book this drive's OWN shorts at 14:00/18:00 across the run window."""
    ledger = ContentLedger()
    base = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
    for k in range(14):
        day = (base + dt.timedelta(days=k)).strftime("%Y-%m-%d")
        for t in ("14:00:00", "18:00:00"):
            ledger.entries.append(
                ContentEntry(
                    date=day,
                    time=t,
                    content_type="shorts",
                    project_name="drive_alpha",
                    source="pipeline",
                    region="usa",
                    status="scheduled",
                )
            )
    ledger.save(tmp_path / ".content_ledger.json")


def _build(config, tmp_path, start_date, *, drive_schedule=None):
    return _build_and_persist_drive_schedule(
        scheduler_cls=PipelineScheduler,
        config=config,
        drive_plan=SimpleNamespace(regular_lfs=[_lf(0)]),
        state=make_state(
            start_date=start_date,
            publish_day="tue",
            shorts_per_day=2,
            no_ledger_reconcile=True,
            drive_schedule=drive_schedule,
        ),
        region="usa",
        stages_dir=tmp_path,
    )


def test_resumed_run_keeps_its_posting_times(tmp_path):
    """Observable contract: the shorts posting times a resumed run schedules are
    the SAME pair the first build locked — not the fenced fallback (10:00/16:00)
    it would flip to when re-resolving against the drive's OWN booked shorts."""
    config = _cfg(tmp_path)
    start_date = _future_start()

    # First build: empty ledger → the drive posts at the top-priority pair.
    first = _build(config, tmp_path, start_date)
    assert _shorts_times(first) == ["14:00:00", "18:00:00"]

    # The drive books its own shorts, then resumes with the persisted schedule.
    _book_own_shorts(tmp_path, start_date)
    second = _build(config, tmp_path, start_date, drive_schedule=first)

    assert _shorts_times(second) == ["14:00:00", "18:00:00"]


def test_locked_pair_is_persisted_for_readback(tmp_path):
    """The lock itself: the first build stamps ``identity_times`` and a resumed
    build reads it back verbatim (the field is what survives a state round-trip)."""
    config = _cfg(tmp_path)
    start_date = _future_start()

    first = _build(config, tmp_path, start_date)
    assert first.identity_times == ["14:00:00", "18:00:00"]

    _book_own_shorts(tmp_path, start_date)
    second = _build(config, tmp_path, start_date, drive_schedule=first)
    assert second.identity_times == ["14:00:00", "18:00:00"]


def test_locked_identity_overrides_ledger_resolution():
    """The mechanism: a supplied lock wins over contested ledger slots and is
    echoed on the schedule so it round-trips into the next resume."""
    sched = PipelineScheduler(shorts_times=[dt.time(14, 0), dt.time(18, 0)])
    occupied = set()
    d = dt.date(2099, 7, 7)
    for _ in range(60):
        occupied.add((d.strftime("%Y-%m-%d"), "14:00:00"))
        occupied.add((d.strftime("%Y-%m-%d"), "18:00:00"))
        d += dt.timedelta(days=1)
    out = sched.generate_drive_schedule(
        regular_lfs=[_lf(0)],
        region="usa",
        start_date="2099-07-07",
        occupied_slots=occupied,
        slot_pool=[dt.time(14, 0), dt.time(18, 0), dt.time(10, 0), dt.time(16, 0)],
        shorts_per_day=2,
        locked_identity_times=[dt.time(14, 0), dt.time(18, 0)],
    )
    assert out.identity_times == ["14:00:00", "18:00:00"]
    assert _shorts_times(out) == ["14:00:00", "18:00:00"]


def test_unreadable_store_stops_phase_d_without_picking_a_date(tmp_path):
    """Fail closed: a corrupt schedule.sqlite3 beside the ledger must make the seo_manager
    slot call raise out of Phase D. It must NOT fall back to a ledger-blind base date."""
    import sqlite3

    import pytest

    (tmp_path / "schedule.sqlite3").write_bytes(b"this is not a database" * 50)
    called = []

    class _Spy(PipelineScheduler):
        def generate_drive_schedule(self, *a, **k):
            called.append(k)
            return super().generate_drive_schedule(*a, **k)

    with pytest.raises(sqlite3.DatabaseError):
        _build_and_persist_drive_schedule(
            scheduler_cls=_Spy,
            config=_cfg(tmp_path),
            drive_plan=SimpleNamespace(regular_lfs=[_lf(0)]),
            state=make_state(
                start_date=None,
                publish_day="tue",
                shorts_per_day=2,
                no_ledger_reconcile=True,
            ),
            region="usa",
            stages_dir=tmp_path,
        )
    assert called == []
