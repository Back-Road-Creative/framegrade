"""GRADE on the rail only means something in motion."""

"""HUD text formatting for Route Story dashboards."""

from gomoveshift.pipeline.route_story import hud as hud_module
from gomoveshift.pipeline.route_story.dashboard import DashboardState


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


def test_grade_is_blank_while_stopped():
    # Grade is rise over the last edge's distance; standing still that distance
    # is GPS jitter, and an earlier film read "stopped ... GRADE -18.0%".
    cells = dict(
        hud_module._dashboard_cells(_dashboard(current_speed_mps=0.1, current_grade_percent=-18.0))
    )

    assert cells["SPEED"] == "stopped"
    assert cells["GRADE"] == "—"


def test_grade_is_blank_when_the_speed_is_unknown():
    cells = dict(
        hud_module._dashboard_cells(_dashboard(current_speed_mps=None, current_grade_percent=4.0))
    )

    assert cells["GRADE"] == "—"


def test_grade_still_shows_while_moving():
    cells = dict(
        hud_module._dashboard_cells(_dashboard(current_speed_mps=1.0, current_grade_percent=7.1))
    )

    assert cells["GRADE"] == "+7.1%"
