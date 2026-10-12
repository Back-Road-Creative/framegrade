"""The drive's long-form BASE DATE is locked once and read back on resume.

``_build_and_persist_drive_schedule`` runs on every Phase-D entry. When no
explicit operator date is set and no LF has uploaded yet, it used to re-resolve
the base date through ``next_free_slot`` on EVERY build — against the ledger's
current occupancy and the current clock — so a plain re-run or a crash-resume
could silently rebuild the schedule on a DIFFERENT base than the one already
persisted in ``drive_schedule.json``.

The lock is the persisted schedule's own ``lf_slots[0].date`` (no new state
field). Priority: explicit start date > earliest already-uploaded LF >
locked ``lf_slots[0].date`` > ``next_free_slot``. A rebase pass that decided
to rebase passes ``rebased=True`` and skips the read-back — it cleared
``start_date`` on purpose.
"""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.drive_phases.phase_d import _build_and_persist_drive_schedule
from gomoveshift.pipeline.schedule_constants import EST
from gomoveshift.pipeline.scheduler import DriveReleaseSchedule, PipelineScheduler

from _state_helpers import make_state


def _cfg(tmp_path):
    """Real-shaped config stub: USA long-form on Tuesdays at 16:00."""
    return SimpleNamespace(
        drive=SimpleNamespace(
            shorts_publish_times=["14:00", "18:00"],
            shorts_slot_pool_by_region={"usa": ["14:00", "18:00", "10:00", "16:00"]},
            long_form_publish_days=[1],  # Tuesday
            long_form_publish_time="16:00",
        ),
        seo=SimpleNamespace(storage_path=str(tmp_path)),
    )


def _future_tuesday(weeks: int = 0) -> dt.date:
    """The next Tuesday strictly after today, plus ``weeks``."""
    day = dt.datetime.now(EST).date() + dt.timedelta(days=1)
    while day.weekday() != 1:
        day += dt.timedelta(days=1)
    return day + dt.timedelta(weeks=weeks)


def _book_longform(tmp_path, date_str):
    """Another drive books the long-form slot on ``date_str`` (same region)."""
    ledger = ContentLedger.load(tmp_path / ".content_ledger.json")
    ledger.entries.append(
        ContentEntry(
            date=date_str,
            time="16:00:00",
            content_type="longform",
            project_name="drive_rival",
            source="pipeline",
            region="usa",
            status="scheduled",
        )
    )
    ledger.save(tmp_path / ".content_ledger.json")


def _build(config, tmp_path, *, rebased=False, **state_kwargs):
    return _build_and_persist_drive_schedule(
        scheduler_cls=PipelineScheduler,
        config=config,
        drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=0)]),
        state=make_state(
            publish_day="tue",
            shorts_per_day=2,
            no_ledger_reconcile=True,
            **state_kwargs,
        ),
        region="usa",
        stages_dir=tmp_path,
        rebased=rebased,
    )


def test_rebuild_keeps_locked_base_date_after_ledger_moves(tmp_path):
    """The defect: a second build with NO uploads must reuse the persisted base
    date even though the ledger has since booked that date."""
    config = _cfg(tmp_path)

    first = _build(config, tmp_path)
    base = first.lf_slots[0].date

    _book_longform(tmp_path, base)
    second = _build(config, tmp_path, drive_schedule=first)

    assert second.lf_slots[0].date == base


def test_explicit_start_date_overrides_lock(tmp_path):
    """An operator date still wins over the persisted lock."""
    config = _cfg(tmp_path)
    first = _build(config, tmp_path)
    base = first.lf_slots[0].date

    explicit = _future_tuesday(3).strftime("%Y-%m-%d")
    assert explicit != base

    out = _build(config, tmp_path, drive_schedule=first, start_date=explicit)
    assert out.lf_slots[0].date == explicit


def test_uploaded_lf_anchor_overrides_lock(tmp_path):
    """An already-uploaded LF (the irreversible act) outranks the lock."""
    config = _cfg(tmp_path)
    first = _build(config, tmp_path)
    base = first.lf_slots[0].date

    anchor = _future_tuesday(2)
    assert anchor.strftime("%Y-%m-%d") != base
    uploads = {
        "0": {"publish_at": dt.datetime.combine(anchor, dt.time(16, 0), tzinfo=EST).isoformat()}
    }

    out = _build(config, tmp_path, drive_schedule=first, youtube_uploads=uploads)
    assert out.lf_slots[0].date == anchor.strftime("%Y-%m-%d")


def test_rebased_build_ignores_lock(tmp_path):
    """A deliberate rebase must NOT read the stale base back."""
    config = _cfg(tmp_path)
    first = _build(config, tmp_path)
    base = first.lf_slots[0].date

    _book_longform(tmp_path, base)
    out = _build(config, tmp_path, drive_schedule=first, rebased=True)

    assert out.lf_slots[0].date != base


def test_legacy_schedule_without_lf_slots_resolves_fresh(tmp_path):
    """A legacy/empty persisted schedule falls through to a fresh resolve."""
    config = _cfg(tmp_path)
    legacy = DriveReleaseSchedule(region="usa", lf_slots=[], shorts_slots=[])

    out = _build(config, tmp_path, drive_schedule=legacy)
    fresh = _build(config, tmp_path)

    assert out.lf_slots[0].date == fresh.lf_slots[0].date
