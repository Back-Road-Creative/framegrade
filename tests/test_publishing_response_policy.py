"""Tests for the bounded-response policy math (pure, no DB/clock)."""

from gomoveshift.publishing.response_policy import (
    DEFAULT_LADDER,
    consecutive_healthy_days,
    daily_cap_for,
    should_auto_pause,
)

LADDER = (1, 2, 3)


def test_suppressed_caps_at_one_regardless_of_ladder_or_healthy_days():
    assert daily_cap_for("SUPPRESSED", LADDER, healthy_days=0) == 1
    assert daily_cap_for("SUPPRESSED", (5, 10), healthy_days=99) == 1


def test_healthy_uses_the_ladder_ceiling():
    assert daily_cap_for("HEALTHY", LADDER, healthy_days=0) == 3
    assert daily_cap_for("HEALTHY", None, healthy_days=0) == DEFAULT_LADDER[-1]


def test_recovering_steps_one_rung_per_seven_healthy_days():
    assert daily_cap_for("RECOVERING", LADDER, healthy_days=0) == 1
    assert daily_cap_for("RECOVERING", LADDER, healthy_days=6) == 1
    assert daily_cap_for("RECOVERING", LADDER, healthy_days=7) == 2
    assert daily_cap_for("RECOVERING", LADDER, healthy_days=13) == 2
    assert daily_cap_for("RECOVERING", LADDER, healthy_days=14) == 3
    assert daily_cap_for("RECOVERING", LADDER, healthy_days=999) == 3


def test_recovering_ladder_none_falls_back_to_default():
    assert daily_cap_for("RECOVERING", None, healthy_days=7) == DEFAULT_LADDER[1]


def test_should_auto_pause_true_for_seven_consecutive_suppressed_days():
    states = {f"2026-09-{d:02d}": "SUPPRESSED" for d in range(11, 18)}
    assert should_auto_pause(states, today="2026-09-17") is True


def test_should_auto_pause_false_at_six_days():
    states = {f"2026-09-{d:02d}": "SUPPRESSED" for d in range(12, 18)}
    assert should_auto_pause(states, today="2026-09-17") is False


def test_should_auto_pause_gap_resets_the_streak():
    states = {f"2026-09-{d:02d}": "SUPPRESSED" for d in range(11, 18)}
    states["2026-09-14"] = "RECOVERING"
    assert should_auto_pause(states, today="2026-09-17") is False


def test_should_auto_pause_missing_day_breaks_the_streak():
    states = {f"2026-09-{d:02d}": "SUPPRESSED" for d in range(11, 18)}
    del states["2026-09-13"]
    assert should_auto_pause(states, today="2026-09-17") is False


def test_consecutive_healthy_days_counts_backward_from_the_day_before_today():
    states = {
        "2026-09-15": "HEALTHY",
        "2026-09-16": "RECOVERING",
        "2026-09-17": "SUPPRESSED",  # today -- not counted
    }
    assert consecutive_healthy_days(states, today="2026-09-17") == 2


def test_consecutive_healthy_days_stops_at_a_suppressed_or_missing_day():
    states = {
        "2026-09-16": "HEALTHY",
        "2026-09-14": "HEALTHY",  # gap at 09-15 -- not counted
    }
    assert consecutive_healthy_days(states, today="2026-09-17") == 1
