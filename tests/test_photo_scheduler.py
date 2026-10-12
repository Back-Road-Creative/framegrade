"""Tests for photo post scheduler."""

from datetime import datetime

import pytest
from freezegun import freeze_time

from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.photos.scheduler import EST, PhotoScheduler


class TestPhotoSchedulerCanonicalLedger:
    @pytest.mark.real_content_ledger
    def test_no_arg_init_loads_canonical_seo_ledger(self, tmp_path, monkeypatch):
        """PhotoScheduler() must resolve the SEO-storage ledger — the single
        store — never a gms_output_dir orphan.

        Opts out of the conftest hermetic-ledger redirect because it tests
        the real default resolution; still hermetic via the get_config patch.
        """
        from types import SimpleNamespace

        seo_dir = tmp_path / "seo"
        ledger_path = seo_dir / ".content_ledger.json"
        seeded = ContentLedger()
        seeded.add(
            ContentEntry(
                date="2027-03-01",
                time="10:00:00",
                content_type="photo",
                title="Fable Pass Sunrise",
                source="photo_pipeline",
            )
        )
        seeded.save(path=ledger_path)
        monkeypatch.setattr(
            "gomoveshift.config.get_config",
            lambda config_path=None: SimpleNamespace(
                seo=SimpleNamespace(storage_path=str(seo_dir))
            ),
        )

        scheduler = PhotoScheduler()
        assert [e.title for e in scheduler.ledger.entries] == ["Fable Pass Sunrise"]


@freeze_time("2026-01-15")
class TestPhotoScheduler:
    def test_schedules_correct_count(self):
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(count=7, start_date="2027-03-01")
        assert len(slots) == 7

    def test_schedules_at_09am(self):
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(count=3, start_date="2027-03-01")
        for slot in slots:
            assert slot.time == "09:00:00"

    def test_avoids_occupied_slots(self):
        ledger = ContentLedger()
        ledger.add(
            ContentEntry(
                date="2027-04-01",
                time="09:00:00",
                content_type="photo",
                title="Existing photo",
            )
        )
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(count=3, start_date="2027-04-01")

        dates = [s.date for s in slots]
        assert "2027-04-01" not in dates

    def test_consecutive_days(self):
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(count=5, start_date="2027-03-01")

        dates = [s.date for s in slots]
        assert dates == [
            "2027-03-01",
            "2027-03-02",
            "2027-03-03",
            "2027-03-04",
            "2027-03-05",
        ]

    def test_includes_day_of_week(self):
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(count=1, start_date="2027-03-01")
        assert slots[0].day_of_week == "Monday"


@freeze_time("2026-01-15")
class TestScheduleBoundaryEnforcement:
    """TDD tests for end_date boundary enforcement."""

    def test_schedule_respects_end_date(self):
        """7 slots from Feb 23 to Mar 1 — all dates <= Mar 1."""
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(
            count=7,
            start_date="2027-02-23",
            end_date="2027-03-01",
        )
        assert len(slots) == 7
        for slot in slots:
            assert slot.date <= "2027-03-01"
            assert slot.date >= "2027-02-23"

    def test_schedule_reduces_count_at_boundary(self):
        """10 slots in 5-day window → only 5 returned."""
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(
            count=10,
            start_date="2027-02-23",
            end_date="2027-02-27",
        )
        assert len(slots) == 5
        for slot in slots:
            assert slot.date >= "2027-02-23"
            assert slot.date <= "2027-02-27"

    def test_schedule_with_conflicts_stays_in_range(self):
        """3 of 7 days occupied — remaining 4 within range, not overflowing."""
        ledger = ContentLedger()
        for date in ["2027-04-02", "2027-04-04", "2027-04-06"]:
            ledger.add(
                ContentEntry(
                    date=date,
                    time="09:00:00",
                    content_type="photo",
                )
            )
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(
            count=7,
            start_date="2027-04-01",
            end_date="2027-04-07",
        )
        # 7 day range minus 3 occupied = 4 available
        assert len(slots) == 4
        for slot in slots:
            assert slot.date >= "2027-04-01"
            assert slot.date <= "2027-04-07"
            assert slot.date not in ["2027-04-02", "2027-04-04", "2027-04-06"]

    def test_schedule_all_occupied_returns_empty(self):
        """All days occupied → empty list, not scanning 60 days forward."""
        ledger = ContentLedger()
        for day in range(1, 7):
            ledger.add(
                ContentEntry(
                    date=f"2027-04-{day:02d}",
                    time="09:00:00",
                    content_type="photo",
                )
            )
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(
            count=7,
            start_date="2027-04-01",
            end_date="2027-04-06",
        )
        assert len(slots) == 0

    def test_schedule_without_end_date_reasonable_cap(self):
        """Without end_date, doesn't scan past start + count + 14 days."""
        ledger = ContentLedger()
        for day in range(1, 31):
            ledger.add(
                ContentEntry(
                    date=f"2027-04-{day:02d}",
                    time="09:00:00",
                    content_type="photo",
                )
            )
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(count=3, start_date="2027-04-01")
        # With cap of count+14=17 days, can only scan Apr 1-17 (all occupied)
        assert len(slots) == 0


@freeze_time("2026-01-15")
class TestPastDateGuard:
    """Tests for past-date scheduling guard."""

    def test_past_start_date_clamps_instead_of_raising(self, caplog):
        """A past start_date is clamped forward, never raised.

        ``publish_batch`` passes ``batch.week_start`` (the generate day), so an
        overnight review — generate day N, publish day N+1 — always arrives with
        a past start. Raising there crashed AFTER the Drive upload.
        """
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        with caplog.at_level("WARNING"):
            slots = scheduler.schedule_batch(count=3, start_date="2020-01-01")
        assert len(slots) == 3
        today_str = datetime.now(EST).strftime("%Y-%m-%d")
        for slot in slots:
            assert slot.date >= today_str
        assert any("2020-01-01" in r.message for r in caplog.records)

    def test_today_start_date_allowed(self):
        """start_date == today does not raise."""
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        today_str = datetime.now(EST).strftime("%Y-%m-%d")
        slots = scheduler.schedule_batch(count=1, start_date=today_str)
        assert len(slots) == 1


class TestPastTimeSkip:
    """A candidate slot at-or-before ``now`` is never booked (mirrors the
    ledger's photo branch) — Metricool rejects a past scheduled time."""

    @freeze_time("2026-01-15 20:00:00")  # 15:00 EST — today's 09:00 has passed
    def test_same_day_slot_after_post_time_is_skipped(self):
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(count=1, start_date="2026-01-15")
        assert slots[0].date == "2026-01-16"

    @freeze_time("2026-01-15 12:00:00")  # 07:00 EST — today's 09:00 is still ahead
    def test_same_day_slot_before_post_time_is_kept(self):
        ledger = ContentLedger()
        scheduler = PhotoScheduler(ledger=ledger)
        slots = scheduler.schedule_batch(count=1, start_date="2026-01-15")
        assert slots[0].date == "2026-01-15"


@freeze_time("2026-01-15")
class TestSchedulerLedgerFreshness:
    def test_schedule_batch_reloads_ledger_from_disk(self, tmp_path):
        """A slot booked by another process after __init__ is still avoided."""
        ledger_path = tmp_path / ".content_ledger.json"
        ledger = ContentLedger()
        ledger.save(path=ledger_path)
        scheduler = PhotoScheduler(ledger=ledger)

        concurrent = ContentLedger.load(ledger_path)
        concurrent.add(
            ContentEntry(
                date="2027-04-01",
                time="09:00:00",
                content_type="photo",
                title="Booked by a concurrent run",
            )
        )
        concurrent.save(path=ledger_path)

        slots = scheduler.schedule_batch(count=2, start_date="2027-04-01")
        assert "2027-04-01" not in [s.date for s in slots]


def test_scheduler_reuses_canonical_schedule_constants():
    """No local redefinition — one source of truth for EST/PHOTO_POST_TIME."""
    from gomoveshift.pipeline import schedule_constants
    from gomoveshift.pipeline.photos import scheduler as photo_scheduler

    assert photo_scheduler.PHOTO_POST_TIME is schedule_constants.PHOTO_POST_TIME
    assert photo_scheduler.EST is schedule_constants.EST


def test_photo_post_hour_does_not_collide_with_video_slots():
    """09:00 must stay clear of every configured shorts/long-form slot."""
    from gomoveshift.config import DriveConfig
    from gomoveshift.pipeline import schedule_constants

    drive = DriveConfig()
    photo_hhmm = schedule_constants.PHOTO_POST_TIME.strftime("%H:%M")
    booked = set(drive.shorts_publish_times) | {drive.long_form_publish_time}
    for value in (drive.shorts_publish_times_by_region or {}).values():
        windows = value.values() if isinstance(value, dict) else [value]
        for window in windows:
            booked |= set(window)
    for pool in (drive.shorts_slot_pool_by_region or {}).values():
        booked |= set(pool)
    assert photo_hhmm not in booked


@freeze_time("2026-01-15")
class TestContentLedgerPhotoType:
    def test_photo_content_type_accepted(self):
        ledger = ContentLedger()
        ledger.add(
            ContentEntry(
                date="2027-03-01",
                time="10:00:00",
                content_type="photo",
                title="Daily photo",
            )
        )
        assert len(ledger.entries) == 1
        assert ledger.entries[0].content_type == "photo"

    def test_get_next_available_photo_slot(self):
        ledger = ContentLedger()
        from zoneinfo import ZoneInfo

        now = datetime(2027, 3, 1, 8, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        slot = ledger.get_next_available_slot(content_type="photo", after=now)
        assert slot is not None
        assert slot["content_type"] == "photo"
        assert slot["time"] == "09:00:00"

    def test_photo_slot_skips_occupied(self):
        ledger = ContentLedger()
        ledger.add(
            ContentEntry(
                date="2027-03-02",
                time="09:00:00",
                content_type="photo",
            )
        )
        from zoneinfo import ZoneInfo

        now = datetime(2027, 3, 1, 8, 0, 0, tzinfo=ZoneInfo("America/New_York"))

        slot = ledger.get_next_available_slot(content_type="photo", after=now)
        assert slot["date"] != "2027-03-02"
