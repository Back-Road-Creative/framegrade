"""Superlong publish day is REGION-AWARE, like every other location-dependent
drive setting (``active_journey_by_region``). The flat ``superlong_publish_day`` stays the
fallback for a region absent from the map — the same shape
``resolve_shorts_times_for_region`` uses."""

from __future__ import annotations

from datetime import datetime

import pytest

from gomoveshift.config import Config, DriveConfig
from gomoveshift.pipeline.content_ledger import (
    ContentEntry,
    ContentLedger,
    next_free_superlong_slot,
)
from gomoveshift.pipeline.schedule_constants import EST, resolve_superlong_publish_day

ANCHOR = datetime(2026, 8, 10, 9, 0, tzinfo=EST)  # a Monday


def _config(**drive_kwargs) -> Config:
    config = Config()
    for key, value in drive_kwargs.items():
        setattr(config.drive, key, value)
    return config


def _superlong_entry(date: str, region: str, project_name: str, title: str) -> ContentEntry:
    return ContentEntry(
        date=date,
        time="11:00:00",
        content_type="longform",
        title=title,
        project_name=project_name,
        status="scheduled",
        source="pipeline",
        region=region,
    )


class TestResolver:
    def test_region_in_map_wins_over_the_flat_default(self):
        config = _config(
            superlong_publish_day="sat",
            superlong_publish_day_by_region={"ireland": "sun", "usa": "wed"},
        )
        assert resolve_superlong_publish_day(config, "ireland") == "sun"
        assert resolve_superlong_publish_day(config, "usa") == "wed"

    @pytest.mark.parametrize("region", ["france", None])
    def test_region_absent_or_none_falls_back_to_the_flat_field(self, region):
        config = _config(
            superlong_publish_day="sat",
            superlong_publish_day_by_region={"ireland": "sun"},
        )
        assert resolve_superlong_publish_day(config, region) == "sat"

    def test_empty_map_is_the_default_so_behaviour_is_unchanged(self):
        config = _config()
        assert config.drive.superlong_publish_day_by_region == {}
        assert resolve_superlong_publish_day(config, "usa") == "sat"


class TestPrecedenceOverLedgerHistory:
    """The superlong publish day derives from operator intent
    first, then previous related videos (an established cadence), then
    location, then the next open day — never a bare hardcoded Saturday."""

    def test_explicit_override_wins_over_contradicting_ledger_history(self):
        config = _config(superlong_publish_day="mon")
        ledger = ContentLedger()
        ledger.entries.append(
            _superlong_entry("2026-08-08", "usa", "roadtrip", "Big Trip — Part 1 of 3")
        )  # a Saturday — contradicts the explicit "mon" override
        day = resolve_superlong_publish_day(config, "usa", ledger=ledger, project_name="roadtrip")
        assert day == "mon"

    def test_ledger_history_wins_over_a_contradicting_region_entry(self):
        config = _config(superlong_publish_day_by_region={"usa": "tue"})
        ledger = ContentLedger()
        ledger.entries.append(
            _superlong_entry("2026-08-08", "usa", "roadtrip", "Big Trip — Part 1 of 3")
        )  # a Saturday — an established cadence now outranks the region map
        day = resolve_superlong_publish_day(config, "usa", ledger=ledger, project_name="roadtrip")
        assert day == "sat"

    def test_no_related_history_falls_through_to_the_region_entry(self):
        config = _config(superlong_publish_day_by_region={"usa": "tue"})
        ledger = ContentLedger()  # no related history at all
        day = resolve_superlong_publish_day(config, "usa", ledger=ledger, project_name="roadtrip")
        assert day == "tue"

    def test_explicit_override_beats_both_history_and_the_region_entry(self):
        config = _config(
            superlong_publish_day="mon",
            superlong_publish_day_by_region={"usa": "tue"},
        )
        ledger = ContentLedger()
        ledger.entries.append(
            _superlong_entry("2026-08-08", "usa", "roadtrip", "Big Trip — Part 1 of 3")
        )  # a Saturday — contradicts both the region map and the explicit override
        day = resolve_superlong_publish_day(config, "usa", ledger=ledger, project_name="roadtrip")
        assert day == "mon"

    def test_related_ledger_history_sets_the_day(self):
        config = _config()  # no explicit override, no per-region map entry
        ledger = ContentLedger()
        ledger.entries.append(
            _superlong_entry("2026-08-04", "usa", "roadtrip", "Big Trip — Part 1 of 3")
        )  # a Tuesday
        day = resolve_superlong_publish_day(config, "usa", ledger=ledger, project_name="roadtrip")
        assert day == "tue"
        assert day != "sat"  # must not silently fall through to the old default

    def test_no_related_history_returns_none_for_the_caller_to_search_any_day(self):
        config = _config()  # no explicit override, no per-region map entry
        ledger = ContentLedger()  # no related history at all
        day = resolve_superlong_publish_day(config, "usa", ledger=ledger, project_name="roadtrip")
        assert day is None

    def test_unrelated_superlong_part_does_not_influence_the_day(self):
        config = _config()
        ledger = ContentLedger()
        ledger.entries.append(
            _superlong_entry("2026-08-08", "usa", "some_other_trip", "Other Trip — Part 1 of 2")
        )  # a Saturday, but a DIFFERENT project — must not count as related
        day = resolve_superlong_publish_day(config, "usa", ledger=ledger, project_name="roadtrip")
        assert day is None


class TestNextFreeSuperlongSlotUsesRelatedHistory:
    def test_related_history_picks_the_next_free_slot_on_that_weekday(self):
        config = _config()
        ledger = ContentLedger()
        ledger.entries.append(
            _superlong_entry("2026-08-04", "usa", "roadtrip", "Big Trip — Part 1 of 3")
        )  # a Tuesday
        slot = next_free_superlong_slot(
            ledger, config, "usa", after=ANCHOR, project_name="roadtrip"
        )
        assert slot.weekday() == 1  # Tuesday
        assert slot.weekday() != 5  # never the old hardcoded Saturday

    def test_no_history_searches_any_day_and_lands_before_the_old_saturday_default(self):
        config = _config()
        ledger = ContentLedger()  # nothing booked at all
        slot = next_free_superlong_slot(
            ledger, config, "usa", after=ANCHOR, project_name="roadtrip"
        )
        # ANCHOR is a Monday with nothing booked anywhere — the earliest open
        # day is well before the following Saturday.
        assert slot.date() < datetime(2026, 8, 15, tzinfo=EST).date()
        assert slot.weekday() != 5


class TestValidation:
    """Validation runs in ``DriveConfig.model_post_init``, so these construct
    the model directly rather than mutating an already-built ``Config``."""

    def test_a_junk_weekday_in_the_map_hard_fails(self):
        with pytest.raises(ValueError, match="superlong_publish_day_by_region"):
            DriveConfig(superlong_publish_day_by_region={"usa": "caturday"})

    def test_a_non_string_day_in_the_map_hard_fails(self):
        with pytest.raises(ValueError, match="superlong_publish_day_by_region"):
            DriveConfig(superlong_publish_day_by_region={"usa": 5})

    def test_a_valid_map_constructs(self):
        drive = DriveConfig(superlong_publish_day_by_region={"ireland": "sun", "usa": "wed"})
        assert drive.superlong_publish_day_by_region == {"ireland": "sun", "usa": "wed"}

    def test_the_flat_field_keeps_its_own_validation(self):
        with pytest.raises(ValueError, match="superlong_publish_day"):
            DriveConfig(superlong_publish_day="caturday")


class TestSlotBooking:
    """The booking path must consume the RESOLVED per-region day, not the flat
    field — otherwise the map exists but changes nothing."""

    @pytest.mark.parametrize(
        "region,expected_weekday",
        [("ireland", 6), ("usa", 2)],  # Mon=0 .. Sun=6 -> sun, wed
    )
    def test_slot_lands_on_the_regions_own_day(self, region, expected_weekday):
        config = _config(
            superlong_publish_day="sat",
            superlong_publish_day_by_region={"ireland": "sun", "usa": "wed"},
        )
        slot = next_free_superlong_slot(ContentLedger(), config, region, after=ANCHOR)
        assert slot.weekday() == expected_weekday

    def test_unmapped_region_still_lands_on_the_flat_default(self):
        config = _config(
            superlong_publish_day="sat",
            superlong_publish_day_by_region={"ireland": "sun"},
        )
        slot = next_free_superlong_slot(ContentLedger(), config, "usa", after=ANCHOR)
        # "usa" is not in the by-region map (only a region Waypost knows can be searched)
        assert slot.weekday() == 5  # sat
