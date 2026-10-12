"""``next_free_slot`` — config cadence, region-aware, ledger-checked slot contract."""

from datetime import datetime, time as dt_time
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import content_ledger, schedule_constants
from gomoveshift.pipeline.content_ledger import (
    ContentEntry,
    ContentLedger,
    next_free_slot,
    next_free_superlong_slot,
)
from gomoveshift.pipeline.drive_phases import phase_d_scheduling
from gomoveshift.pipeline.schedule_constants import EST

ANCHOR = datetime(2026, 6, 9, 9, 0, tzinfo=EST)  # Tuesday; next Sundays: 06-14, 06-21


def _ireland_sunday_config() -> Config:
    config = Config()  # the cadence resolved for --lf-day sun --region ireland
    config.drive.long_form_publish_days = [6]  # Sunday
    config.drive.long_form_publish_time = "14:00"
    return config


def _lf_entry(date: str, region=None) -> ContentEntry:
    return ContentEntry(
        date=date,
        time="14:00:00",
        content_type="longform",
        title="Booked Drive",
        project_name="booked_drive",
        status="scheduled",
        source="pipeline",
        region=region,
    )


class TestNextFreeSlot:
    def test_ireland_sunday_cadence_from_config(self):
        # Empty ledger → next config-cadence Sunday 14:00, not Wed 11:00.
        slot = next_free_slot(ContentLedger(), _ireland_sunday_config(), "ireland", after=ANCHOR)
        assert slot == datetime(2026, 6, 14, 14, 0, tzinfo=EST)  # a Sunday

    def test_occupied_slot_skips_to_next_week(self):
        """A same-region LF on the candidate DATE blocks it (any time-of-day)."""
        ledger = ContentLedger()
        ledger.entries.append(_lf_entry("2026-06-14", region="ireland"))
        slot = next_free_slot(ledger, _ireland_sunday_config(), "ireland", after=ANCHOR)
        assert slot == datetime(2026, 6, 21, 14, 0, tzinfo=EST)

    def test_occupancy_is_date_granular_not_time_keyed(self):
        ledger = ContentLedger()  # an entry booked at a DIFFERENT time still occupies the date
        entry = _lf_entry("2026-06-14", region="ireland")
        entry.time = "11:00:00"  # legacy grid time, not the cadence 14:00
        ledger.entries.append(entry)
        slot = next_free_slot(ledger, _ireland_sunday_config(), "ireland", after=ANCHOR)
        assert slot == datetime(2026, 6, 21, 14, 0, tzinfo=EST)

    def test_region_isolation_default(self):
        ledger = ContentLedger()  # a USA LF doesn't block Ireland — unless block_cross_region
        ledger.entries.append(_lf_entry("2026-06-14", region="usa"))
        config = _ireland_sunday_config()
        assert next_free_slot(ledger, config, "ireland", after=ANCHOR) == datetime(
            2026, 6, 14, 14, 0, tzinfo=EST
        )
        assert next_free_slot(
            ledger, config, "ireland", after=ANCHOR, block_cross_region=True
        ) == datetime(2026, 6, 21, 14, 0, tzinfo=EST)

    def test_anonymous_entry_blocks_all_regions(self):
        ledger = ContentLedger()  # a legacy no-region entry conservatively occupies for everyone
        ledger.entries.append(_lf_entry("2026-06-14", region=None))
        slot = next_free_slot(ledger, _ireland_sunday_config(), "ireland", after=ANCHOR)
        assert slot == datetime(2026, 6, 21, 14, 0, tzinfo=EST)

    def test_publish_day_arg_narrows_candidate_days(self):
        config = _ireland_sunday_config()  # explicit publish_day narrows the candidate days
        config.drive.long_form_publish_days = [2, 6]
        slot = next_free_slot(ContentLedger(), config, "ireland", after=ANCHOR, publish_day="sun")
        assert slot.weekday() == 6

    def test_non_longform_rejected(self):
        with pytest.raises(ValueError, match="long-form slots only"):
            next_free_slot(ContentLedger(), None, "ireland", content_type="shorts")


class TestNextFreeSuperlongSlot:
    """Superlong upload slot: config-driven day, default Saturday,
    reusing next_free_slot's region-aware ledger occupancy — no parallel
    scheduler."""

    def test_default_day_is_saturday(self):
        slot = next_free_superlong_slot(ContentLedger(), Config(), "usa", after=ANCHOR)
        assert slot.weekday() == 5  # Saturday

    def test_honours_overridden_config_value(self):
        config = Config()
        config.drive.superlong_publish_day = "mon"
        slot = next_free_superlong_slot(ContentLedger(), config, "usa", after=ANCHOR)
        assert slot.weekday() == 0  # Monday

    def test_does_not_collide_with_existing_lf_booking_same_region(self):
        """A same-region LF already booked on the candidate Saturday pushes
        the superlong part to the following Saturday — same occupancy rule
        as a regular LF, never a second ledger/scheduler."""
        ledger = ContentLedger()
        ledger.entries.append(_lf_entry("2026-06-13", region="usa"))  # a Saturday
        slot = next_free_superlong_slot(ledger, Config(), "usa", after=ANCHOR)
        assert slot.date().isoformat() == "2026-06-20"

    def test_region_isolation_is_the_opposite_of_next_free_slot_by_default(self):
        """Unlike a regular LF, every region resolves the SAME superlong day
        absent a per-region override — so a different region's Saturday
        booking DOES block: the superlong slots are one shared weekly queue,
        not a per-region reservation."""
        ledger = ContentLedger()
        ledger.entries.append(_lf_entry("2026-06-13", region="ireland"))
        slot = next_free_superlong_slot(ledger, Config(), "usa", after=ANCHOR)
        assert slot.date().isoformat() == "2026-06-20"


class TestSingleVideoPathUsesConfigCadence:
    def test_get_next_available_slot_follows_module_cadence(self, monkeypatch):
        monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({6}))
        monkeypatch.setattr(schedule_constants, "LONG_FORM_TIME", dt_time(14, 0))
        slot = ContentLedger().get_next_available_slot("longform", after=ANCHOR)
        assert slot is not None
        assert slot["date"] == "2026-06-14"
        assert slot["time"] == "14:00:00"
        assert slot["day_name"] == "Sunday"

    def test_get_next_available_slot_skips_occupied_region_aware(self, monkeypatch):
        monkeypatch.setattr(schedule_constants, "LONG_FORM_DAYS", frozenset({6}))
        monkeypatch.setattr(schedule_constants, "LONG_FORM_TIME", dt_time(14, 0))
        ledger = ContentLedger()
        ledger.entries.append(_lf_entry("2026-06-14", region="usa"))
        slot = ledger.get_next_available_slot("longform", after=ANCHOR)  # legacy: any-entry view
        assert slot is not None and slot["date"] == "2026-06-21"
        slot = ledger.get_next_available_slot("longform", after=ANCHOR, region="ireland")
        assert slot is not None and slot["date"] == "2026-06-14"


class TestDriveDefaultBaseDate:
    """Drive DEFAULT base date routes through next_free_slot; explicit bypasses."""

    def _run(self, tmp_path, start_date, youtube_uploads=None):
        from gomoveshift.pipeline.scheduler import PipelineScheduler

        config = _ireland_sunday_config()
        config.seo.storage_path = str(tmp_path)
        ledger = ContentLedger()
        ledger.entries.append(_lf_entry("2026-06-14", region="ireland"))
        ledger.save(Path(tmp_path) / ".content_ledger.json")
        state = SimpleNamespace(
            start_date=start_date,
            publish_day=None,
            shorts_per_day=None,
            youtube_uploads=youtube_uploads,
        )
        schedule = phase_d_scheduling._build_and_persist_drive_schedule(
            scheduler_cls=PipelineScheduler,
            config=config,
            drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=1)]),
            state=state,
            region="ireland",
            stages_dir=tmp_path,
            now=ANCHOR,
        )
        return schedule

    def test_default_base_date_skips_ledger_occupied_sunday(self, tmp_path):
        schedule = self._run(tmp_path, start_date=None)
        assert schedule.lf_slots[0].date == "2026-06-21"

    def test_explicit_operator_date_bypasses_ledger(self, tmp_path):
        schedule = self._run(tmp_path, start_date="2026-06-14")
        assert schedule.lf_slots[0].date == "2026-06-14"

    def test_reconcile_runs_before_slot_selection(self, tmp_path, monkeypatch):
        """Ordering: slot selection must see the phantom-cleaned ledger."""
        calls = []
        real_slot = content_ledger.next_free_slot
        monkeypatch.setattr(phase_d_scheduling, "_build_readonly_youtube_service", object)
        monkeypatch.setattr(
            content_ledger,
            "reconcile_ledger_with_youtube",
            lambda *a, **k: calls.append("reconcile"),
        )
        monkeypatch.setattr(
            content_ledger,
            "next_free_slot",
            lambda *a, **k: (calls.append("next_free_slot"), real_slot(*a, **k))[1],
        )
        self._run(tmp_path, start_date=None)
        assert calls == ["reconcile", "next_free_slot"]


class TestDriveScheduleAnchorsToUploadedLfs:
    """A resumed drive with no explicit start_date must anchor to its OWN
    already-uploaded LFs instead of next_free_slot — once those LFs are
    booked into the ledger under this drive's own region, next_free_slot's
    date-granular/region-only occupancy check walks straight past them."""

    _run = TestDriveDefaultBaseDate._run

    def _spy_next_free_slot(self, monkeypatch):
        calls = []
        real_slot = content_ledger.next_free_slot
        monkeypatch.setattr(
            content_ledger,
            "next_free_slot",
            lambda *a, **k: (calls.append(1), real_slot(*a, **k))[1],
        )
        return calls

    def test_anchors_to_earliest_uploaded_lf_and_skips_next_free_slot(self, tmp_path, monkeypatch):
        calls = self._spy_next_free_slot(monkeypatch)
        uploads = {
            1: {"publish_at": "2026-06-14T14:00:00-04:00"},
            2: {"publish_at": "2026-06-21T14:00:00-04:00"},
        }
        schedule = self._run(tmp_path, start_date=None, youtube_uploads=uploads)
        assert schedule.lf_slots[0].date == "2026-06-14"
        assert calls == []

    def test_no_uploads_still_uses_next_free_slot(self, tmp_path, monkeypatch):
        calls = self._spy_next_free_slot(monkeypatch)
        schedule = self._run(tmp_path, start_date=None, youtube_uploads=None)
        assert schedule.lf_slots[0].date == "2026-06-21"
        assert len(calls) == 1

    def test_explicit_start_date_still_wins_over_uploads(self, tmp_path, monkeypatch):
        calls = self._spy_next_free_slot(monkeypatch)
        uploads = {1: {"publish_at": "2026-06-14T14:00:00-04:00"}}
        schedule = self._run(tmp_path, start_date="2026-06-28", youtube_uploads=uploads)
        assert schedule.lf_slots[0].date == "2026-06-28"
        assert calls == []

    def test_unparseable_upload_record_is_skipped_not_fatal(self, tmp_path, monkeypatch):
        calls = self._spy_next_free_slot(monkeypatch)
        uploads = {
            1: {"publish_at": "not-a-date"},
            2: {"publish_at": "2026-06-14T14:00:00-04:00"},
        }
        schedule = self._run(tmp_path, start_date=None, youtube_uploads=uploads)
        assert schedule.lf_slots[0].date == "2026-06-14"
        assert calls == []

    def test_no_parseable_uploads_falls_through_to_next_free_slot(self, tmp_path, monkeypatch):
        calls = self._spy_next_free_slot(monkeypatch)
        uploads = {1: {"publish_at": None}, 2: {"publish_at": "garbage"}}
        schedule = self._run(tmp_path, start_date=None, youtube_uploads=uploads)
        assert schedule.lf_slots[0].date == "2026-06-21"
        assert len(calls) == 1
