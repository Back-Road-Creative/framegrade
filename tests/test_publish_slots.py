"""Tests for LF-day publish slots + shorts-time resolution.

Single source of truth:
- **LF time** is a property of the LF day → ``publish_slots[day].lf_time``.
- **Shorts times** are a property of the REGION →
  ``shorts_publish_times_by_region``. ``publish_slots`` does not carry (or
  override) shorts times.

So ``--lf-day tue`` determines the long-form time; ``--region usa`` determines
the shorts window; an explicit ``--shorts-times`` overrides the region.
"""

import pytest

from gomoveshift.config import Config, PublishSlot
from gomoveshift.pipeline import schedule_constants


def _cfg():
    return Config()


# --- publish_slots config defaults (lf_time only) ---------------------


def test_default_slots_match_cadence_table():
    drive = _cfg().drive
    assert drive.publish_slots["tue"].lf_time == "16:00"
    assert drive.publish_slots["fri"].lf_time == "15:00"
    assert drive.publish_slots["sun"].lf_time == "14:00"
    # Shorts are NOT on the slot anymore — they live in the region map.
    assert not hasattr(drive.publish_slots["tue"], "shorts_times")


def test_publish_slot_rejects_bad_hhmm():
    with pytest.raises(ValueError):
        PublishSlot(lf_time="25:00")


# --- resolve_publish_slot ---------------------------------------------


def test_resolve_slot_by_day():
    slot = schedule_constants.resolve_publish_slot(_cfg(), "tue")
    assert slot is not None
    assert slot.lf_time == "16:00"


def test_resolve_slot_absent_returns_none():
    # Wednesday has no designed slot (it's the day to avoid).
    assert schedule_constants.resolve_publish_slot(_cfg(), "wed") is None
    assert schedule_constants.resolve_publish_slot(_cfg(), None) is None


# --- resolve_shorts_times: region is the single source ----------------


def test_shorts_explicit_flag_wins():
    cfg = _cfg()
    out = schedule_constants.resolve_shorts_times(
        cfg, lf_day="tue", shorts_times_arg="09:00,21:00", region="usa"
    )
    assert out == ["09:00", "21:00"]


def test_shorts_come_from_region_map_with_per_day_usa_stagger():
    # No explicit --shorts-times: the REGION map decides.
    # USA maps to a per-LF-day window so Tue and Fri drives don't share a slot
    # (anti-cannibalization): Tue 14:00/18:00, Fri 10:00/16:00. An undesigned
    # USA day uses _default (14:00/18:00). Shorts never come from publish_slots.
    cfg = _cfg()
    assert schedule_constants.resolve_shorts_times(cfg, "tue", None, "usa") == ["14:00", "18:00"]
    assert schedule_constants.resolve_shorts_times(cfg, "fri", None, "usa") == ["10:00", "16:00"]
    assert schedule_constants.resolve_shorts_times(cfg, "wed", None, "usa") == ["14:00", "18:00"]


def test_shorts_ireland_region_window():
    # Ireland is a flat (non-per-day) window — same times every day.
    cfg = _cfg()
    assert schedule_constants.resolve_shorts_times(cfg, "sun", None, "ireland") == [
        "08:00",
        "12:00",
    ]
    assert schedule_constants.resolve_shorts_times(cfg, None, None, "ireland") == ["08:00", "12:00"]


def test_shorts_no_region_falls_back_to_flat():
    cfg = _cfg()
    cfg.drive.shorts_publish_times = ["12:00"]
    out = schedule_constants.resolve_shorts_times(
        cfg, lf_day="wed", shorts_times_arg=None, region="atlantis"
    )
    assert out == ["12:00"]


# --- resolve_lf_time: precedence (unchanged) --------------------------


def test_lf_time_explicit_flag_wins():
    cfg = _cfg()
    out = schedule_constants.resolve_lf_time(cfg, lf_day="tue", lf_time_arg="09:30")
    assert out == "09:30"


def test_lf_time_derives_from_slot():
    cfg = _cfg()
    out = schedule_constants.resolve_lf_time(cfg, lf_day="tue", lf_time_arg=None)
    assert out == "16:00"


def test_lf_time_no_slot_keeps_config_default():
    cfg = _cfg()  # long_form_publish_time default "11:00"
    out = schedule_constants.resolve_lf_time(cfg, lf_day="wed", lf_time_arg=None)
    assert out == "11:00"
