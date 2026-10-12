"""HUD text formatting for Route Story dashboards."""

from gomoveshift.pipeline.route_story import hud as hud_module
from gomoveshift.pipeline.route_story.dashboard import DashboardState
from gomoveshift.pipeline.route_story.metrics import RoutePoint


def _dashboard(**overrides) -> DashboardState:
    base = dict(
        data_available=True,
        unobserved_gap=False,
        distance_traveled_m=0.0,
        distance_remaining_m=0.0,
        distance_total_m=0.0,
        completion_fraction=0.0,
        elapsed_seconds=0.0,
        moving_seconds=0.0,
        current_elevation_m=None,
        current_speed_mps=None,
        current_grade_percent=None,
        ascent_m=None,
        descent_m=None,
        moving_average_speed_mps=None,
    )
    base.update(overrides)
    return DashboardState(**base)


def test_pace_reports_stopped_below_the_moving_threshold():
    assert hud_module._pace(0.3) == "stopped"


def test_speed_reports_stopped_below_the_moving_threshold():
    assert hud_module._safe_speed(0.3) == "stopped"


def test_pace_still_rejects_an_unrealistic_gps_jump():
    assert hud_module._pace(12.0) == "—"


def test_speed_still_rejects_an_unrealistic_gps_jump():
    assert hud_module._safe_speed(12.0) == "—"


def test_pace_still_reports_a_normal_moving_speed():
    assert hud_module._pace(2.0) == "13:25/mi"


def test_dashboard_cells_never_show_a_warp_badge():
    names = [name for name, _ in hud_module._dashboard_cells(_dashboard())]
    assert "WARP" not in names


# ---------------------------------------------------------------------------
# Frozen field set -- decided once per film from segment capabilities, never
# per frame from whatever the current frame's telemetry happens to carry.
# ---------------------------------------------------------------------------


def _point(time_s: float, **overrides) -> RoutePoint:
    base = dict(latitude=0.0, longitude=0.0)
    base.update(overrides)
    return RoutePoint(time_s, **base)


def test_dashboard_fields_omit_body_columns_when_no_segment_carries_them():
    segments = [[_point(0.0), _point(1.0)]]
    fields = hud_module.dashboard_fields_for(segments)
    assert "HR" not in fields
    assert "CAD" not in fields
    assert "TEMP" not in fields


def test_dashboard_fields_include_heart_rate_when_any_point_in_any_segment_carries_it():
    segments = [
        [_point(0.0), _point(1.0)],
        [_point(2.0, heart_rate_bpm=142.0), _point(3.0)],
    ]
    fields = hud_module.dashboard_fields_for(segments)
    assert "HR" in fields
    assert "CAD" not in fields
    assert "TEMP" not in fields


def test_dashboard_fields_never_include_a_warp_badge():
    fields = hud_module.dashboard_fields_for([[_point(0.0), _point(1.0)]])
    assert "WARP" not in fields


def test_dashboard_cells_render_a_placeholder_for_a_frozen_field_missing_on_this_frame():
    # HR is frozen present for the film, but this particular frame's rider
    # dropped their strap -- the column stays, its value reads "--".
    fields = ("SPEED", "ELEVATION", "GRADE", "DISTANCE", "HR")
    cells = dict(hud_module._dashboard_cells(_dashboard(), fields=fields))
    assert cells["HR"] == "—"
    assert list(dict(hud_module._dashboard_cells(_dashboard(), fields=fields))) == list(fields)
