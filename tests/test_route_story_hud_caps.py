"""HUD speed/grade caps and units follow the resolved activity profile.

Absent an explicit ``DisplayCaps``/units call, the HUD draws exactly as it
always has -- an 8.0 m/s speed cap, a 30.0% grade cap, imperial units. A film
whose activity resolved to something other than driving (or whose region
resolved to metric) sets those once for the whole render.
"""

from __future__ import annotations

import pytest

from gomoveshift.pipeline.route_story import hud as hud_module


@pytest.fixture(autouse=True)
def _reset_active_display_state():
    hud_module.set_active_display_caps(None)
    hud_module.set_active_display_units(None)
    yield
    hud_module.set_active_display_caps(None)
    hud_module.set_active_display_units(None)


def test_default_caps_still_hide_a_speed_above_the_legacy_8_mps_ceiling():
    assert hud_module._safe_speed(9.0) == "—"


def test_cycling_caps_show_a_speed_the_legacy_ceiling_would_hide():
    hud_module.set_active_display_caps(
        hud_module.DisplayCaps(speed_cap_mps=25.0, grade_cap_percent=20.0)
    )

    assert hud_module._safe_speed(12.0) == "26.8 mph"


def test_grade_hidden_under_cycling_cap_but_shown_under_hiking_cap():
    hud_module.set_active_display_caps(
        hud_module.DisplayCaps(speed_cap_mps=25.0, grade_cap_percent=20.0)
    )
    assert hud_module._safe_grade(32.0) == "—"

    hud_module.set_active_display_caps(
        hud_module.DisplayCaps(speed_cap_mps=4.0, grade_cap_percent=35.0)
    )
    assert hud_module._safe_grade(32.0) == "+32.0%"


def test_metric_units_format_speed_as_km_per_hour():
    hud_module.set_active_display_units("metric")

    assert hud_module._speed(12.0) == "43.2 km/h"


def test_metric_units_format_pace_as_min_per_km():
    hud_module.set_active_display_units("metric")

    assert hud_module._pace(4.0) == "4:10/km"


def test_metric_units_format_elevation_in_meters():
    hud_module.set_active_display_units("metric")

    assert hud_module._feet(1234.0) == "1,234 m"


def test_resetting_caps_and_units_to_none_restores_legacy_display():
    hud_module.set_active_display_caps(
        hud_module.DisplayCaps(speed_cap_mps=25.0, grade_cap_percent=20.0)
    )
    hud_module.set_active_display_units("metric")

    hud_module.set_active_display_caps(None)
    hud_module.set_active_display_units(None)

    assert hud_module._safe_speed(9.0) == "—"
    assert hud_module._speed(9.0) == "20.1 mph"
