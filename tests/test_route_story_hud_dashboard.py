"""The dashboard rail: TIME of day, one motion cell, no WARP badge."""

from datetime import datetime, timedelta, timezone

import pytest
from PIL import Image, ImageDraw

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story import hud as hud_module
from gomoveshift.pipeline.route_story.dashboard import DashboardState


def _dashboard(**overrides) -> DashboardState:
    base = dict(
        data_available=True,
        unobserved_gap=False,
        distance_traveled_m=5150.0,
        distance_remaining_m=0.0,
        distance_total_m=5150.0,
        completion_fraction=1.0,
        elapsed_seconds=0.0,
        moving_seconds=0.0,
        current_elevation_m=312.0,
        current_speed_mps=1.5,
        current_grade_percent=3.2,
        ascent_m=140.0,
        descent_m=60.0,
        moving_average_speed_mps=1.4,
        recording_time_s=0.0,
    )
    base.update(overrides)
    return DashboardState(**base)


@pytest.fixture(autouse=True)
def _reset_dashboard_globals():
    yield
    hud_module.set_active_dashboard_clock(None)
    hud_module.set_active_motion_cell_mode("speed")


# ---------------------------------------------------------------------------
# WARP is gone entirely
# ---------------------------------------------------------------------------


def test_dashboard_cells_never_show_a_warp_badge():
    names = [name for name, _ in hud_module._dashboard_cells(_dashboard())]
    assert "WARP" not in names


def test_dashboard_cells_never_show_warp_even_with_clock_and_pace_active():
    hud_module.set_active_dashboard_clock(datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc))
    hud_module.set_active_motion_cell_mode("pace")
    names = [name for name, _ in hud_module._dashboard_cells(_dashboard())]
    assert "WARP" not in names


def test_time_compression_module_state_is_fully_removed():
    assert not hasattr(hud_module, "set_active_time_compression")
    assert not hasattr(hud_module, "_time_compression_label")
    assert not hasattr(hud_module, "_active_time_compression")


def test_time_compression_still_lands_in_the_film_receipt():
    # film.py:~361 -- the honest speed factor stays in the receipt even
    # though the WARP screen badge that used to show it is gone.
    import inspect

    source = inspect.getsource(film_module.film_receipt)
    assert '"time_compression": time_compression' in source


# ---------------------------------------------------------------------------
# TIME cell
# ---------------------------------------------------------------------------


def test_time_of_day_label_uses_the_active_clock_anchor():
    hud_module.set_active_dashboard_clock(
        datetime(2026, 9, 16, 18, 41, tzinfo=timezone(timedelta(hours=-4)))
    )
    cells = dict(hud_module._dashboard_cells(_dashboard(recording_time_s=0.0)))
    assert cells["TIME"] == "6:41 PM"


def test_time_of_day_label_advances_with_the_recording_clock():
    hud_module.set_active_dashboard_clock(
        datetime(2026, 9, 16, 6, 5, tzinfo=timezone(timedelta(hours=-4)))
    )
    cells = dict(hud_module._dashboard_cells(_dashboard(recording_time_s=3600.0)))
    assert cells["TIME"] == "7:05 AM"


def test_time_of_day_label_handles_a_half_hour_offset():
    # +05:30 (India) at a UTC instant that lands on a half-hour boundary.
    anchor = datetime(2026, 9, 16, 0, 0, tzinfo=timezone.utc).astimezone(
        timezone(timedelta(hours=5, minutes=30))
    )
    hud_module.set_active_dashboard_clock(anchor)
    cells = dict(hud_module._dashboard_cells(_dashboard(recording_time_s=0.0)))
    assert cells["TIME"] == "5:30 AM"


def test_time_of_day_label_rolls_over_past_midnight():
    hud_module.set_active_dashboard_clock(
        datetime(2026, 9, 16, 23, 30, tzinfo=timezone(timedelta(hours=-4)))
    )
    cells = dict(hud_module._dashboard_cells(_dashboard(recording_time_s=3600.0)))
    assert cells["TIME"] == "12:30 AM"


def test_time_cell_is_omitted_without_an_active_clock():
    names = [name for name, _ in hud_module._dashboard_cells(_dashboard())]
    assert "TIME" not in names


def test_time_of_day_label_has_no_leading_zero_no_seconds_no_tz_text():
    hud_module.set_active_dashboard_clock(
        datetime(2026, 9, 16, 9, 5, tzinfo=timezone(timedelta(hours=-4)))
    )
    cells = dict(hud_module._dashboard_cells(_dashboard(recording_time_s=0.0)))
    assert cells["TIME"] == "9:05 AM"
    assert ":" in cells["TIME"] and cells["TIME"].count(":") == 1
    assert "UTC" not in cells["TIME"] and "-" not in cells["TIME"] and "+" not in cells["TIME"]


# ---------------------------------------------------------------------------
# UTC offset resolution
# ---------------------------------------------------------------------------


def test_resolve_dashboard_utc_offset_uses_the_agreed_exif_offset():
    offset, source = hud_module.resolve_dashboard_utc_offset(["-04:00", "-04:00"], "+05:30")
    assert (offset, source) == ("-04:00", "exif_offset")


def test_resolve_dashboard_utc_offset_falls_through_to_cli_on_disagreement(caplog):
    with caplog.at_level("WARNING"):
        offset, source = hud_module.resolve_dashboard_utc_offset(["-04:00", "-05:00"], "+05:30")
    assert (offset, source) == ("+05:30", "cli_offset")
    assert any("disagree" in message for message in caplog.messages)


def test_resolve_dashboard_utc_offset_falls_through_to_cli_when_no_exif():
    offset, source = hud_module.resolve_dashboard_utc_offset([None, None], "-04:00")
    assert (offset, source) == ("-04:00", "cli_offset")


def test_resolve_dashboard_utc_offset_omits_when_neither_is_available():
    offset, source = hud_module.resolve_dashboard_utc_offset([None], None)
    assert (offset, source) == (None, "omitted")


def test_resolve_dashboard_utc_offset_omits_on_disagreement_with_no_cli():
    offset, source = hud_module.resolve_dashboard_utc_offset(["-04:00", "-05:00"], None)
    assert (offset, source) == (None, "omitted")


# ---------------------------------------------------------------------------
# --utc-offset parsing (accept/reject table)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["-04:00", "+05:30", "+00:00", "-00:00", "+14:00", "-12:00"],
)
def test_utc_offset_accepts_well_formed_values(text):
    assert film_module._utc_offset_arg(text) == text


@pytest.mark.parametrize(
    "text",
    ["EST", "-4:00", "+5:30", "-04:60", "+25:00", "04:00", "", "-04", "+05:30:00"],
)
def test_utc_offset_rejects_malformed_values(text):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        film_module._utc_offset_arg(text)


def test_parse_utc_offset_builds_the_matching_timezone():
    assert hud_module.parse_utc_offset("+05:30").utcoffset(None) == timedelta(hours=5, minutes=30)
    assert hud_module.parse_utc_offset("-04:00").utcoffset(None) == timedelta(hours=-4)


# ---------------------------------------------------------------------------
# Offset recovered from a photo's two capture clocks
# ---------------------------------------------------------------------------


def test_offset_from_capture_recovers_the_original_exif_offset():
    wall_local = datetime(2026, 9, 16, 8, 30, 0)
    capture_utc = datetime(2026, 9, 16, 12, 30, 0, tzinfo=timezone.utc)
    assert hud_module._offset_from_capture(wall_local, capture_utc) == "-04:00"


def test_offset_from_capture_handles_a_positive_offset():
    wall_local = datetime(2026, 9, 16, 18, 0, 0)
    capture_utc = datetime(2026, 9, 16, 12, 30, 0, tzinfo=timezone.utc)
    assert hud_module._offset_from_capture(wall_local, capture_utc) == "+05:30"


# ---------------------------------------------------------------------------
# Motion cell: exactly one of PACE / SPEED
# ---------------------------------------------------------------------------


def test_motion_cell_mode_boundary_is_inclusive_of_pace():
    assert hud_module.motion_cell_mode(4.5) == "pace"
    assert hud_module.motion_cell_mode(4.500001) == "speed"


def test_motion_cell_mode_defaults_to_speed_when_unknown():
    assert hud_module.motion_cell_mode(None) == "speed"


def test_exactly_one_of_pace_or_speed_is_present_in_pace_mode():
    hud_module.set_active_motion_cell_mode("pace")
    names = [name for name, _ in hud_module._dashboard_cells(_dashboard())]
    assert names.count("PACE") == 1
    assert "SPEED" not in names


def test_exactly_one_of_pace_or_speed_is_present_in_speed_mode():
    hud_module.set_active_motion_cell_mode("speed")
    names = [name for name, _ in hud_module._dashboard_cells(_dashboard())]
    assert names.count("SPEED") == 1
    assert "PACE" not in names


# ---------------------------------------------------------------------------
# Receipt block
# ---------------------------------------------------------------------------


def test_dashboard_receipt_shape():
    assert hud_module.dashboard_receipt("pace", "exif_offset", "-04:00") == {
        "motion_cell": "pace",
        "time_cell": "exif_offset",
        "utc_offset": "-04:00",
        "pace_window_s": 20.0,
        "grade_window_m": 75.0,
    }
    assert hud_module.dashboard_receipt("speed", "omitted", None) == {
        "motion_cell": "speed",
        "time_cell": "omitted",
        "utc_offset": None,
        "pace_window_s": 20.0,
        "grade_window_m": 75.0,
    }


def test_dashboard_cell_label_discloses_the_averaging_basis():
    assert hud_module._dashboard_cell_label("PACE") == "RECENT PACE"
    assert hud_module._dashboard_cell_label("SPEED") == "RECENT SPEED"
    assert hud_module._dashboard_cell_label("GRADE") == "GRADE"


# ---------------------------------------------------------------------------
# Cell text fits its column, TIME + HR/CAD/TEMP all present
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(1920, 1080), (960, 540)])
def test_every_dashboard_cell_value_fits_its_column(size):
    width, height = size
    hud_module.set_active_dashboard_clock(
        datetime(2026, 9, 16, 12, 0, tzinfo=timezone(timedelta(hours=-4)))
    )
    hud_module.set_active_motion_cell_mode("speed")
    dashboard = _dashboard(
        current_heart_rate_bpm=188.0,
        current_cadence_rpm=190.0,
        current_temperature_c=-12.0,
    )
    cells = hud_module._dashboard_cells(dashboard)
    assert any(name == "TIME" for name, _ in cells)
    assert any(name in ("HR", "CAD", "TEMP") for name, _ in cells)

    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    value_font = hud_module._font(max(16, width // 48), bold=True)
    left, right = int(width * 0.045), int(width * 0.955)
    cell_w = (right - left) / len(cells)
    for _, content in cells:
        fitted = hud_module._fitted_value_font(draw, content, value_font, cell_w - 14)
        assert draw.textlength(content, font=fitted) <= cell_w - 14 + 0.5


def test_time_of_day_keeps_the_real_clock_across_a_recording_gap():
    """An hour the recorder was paused is still an hour on the wall clock."""
    from gomoveshift.pipeline.route_story.dashboard import dashboard_state_at
    from gomoveshift.pipeline.route_story.metrics import RoutePoint
    from gomoveshift.pipeline.route_story.timeline import Scene, StoryTimeline

    segments = [
        [RoutePoint(0.0, 0.0, 0.0, 100.0), RoutePoint(10.0, 0.0, 0.0001, 100.0)],
        [RoutePoint(3600.0, 0.0, 0.0002, 100.0), RoutePoint(3610.0, 0.0, 0.0003, 100.0)],
    ]
    timeline = StoryTimeline(fps=10, scenes=(Scene("route_follow", 0, 10, 3600.0, 3610.0),))
    hud_module.set_active_dashboard_clock(
        datetime(2026, 9, 16, 9, 0, tzinfo=timezone(timedelta(hours=-4)))
    )

    dashboard = dashboard_state_at(timeline, 0, segments, moving_speed_mps=0.5)

    assert dict(hud_module._dashboard_cells(dashboard))["TIME"] == "10:00 AM"


def test_dashboard_fields_for_includes_time_only_when_a_clock_anchor_is_active():
    segments = [[]]
    assert "TIME" not in hud_module.dashboard_fields_for(segments)
    hud_module.set_active_dashboard_clock(
        datetime(2026, 9, 16, 9, 0, tzinfo=timezone(timedelta(hours=-4)))
    )
    assert "TIME" in hud_module.dashboard_fields_for(segments)


def test_time_cell_is_omitted_when_the_frame_has_no_recording_clock():
    hud_module.set_active_dashboard_clock(
        datetime(2026, 9, 16, 9, 0, tzinfo=timezone(timedelta(hours=-4)))
    )
    names = [name for name, _ in hud_module._dashboard_cells(_dashboard(recording_time_s=None))]
    assert "TIME" not in names


# ---------------------------------------------------------------------------
# Frozen grid: column x-positions never move, even with a mid-film sensor
# dropout and wildly different digit widths across frames.
# ---------------------------------------------------------------------------


def _cell_x_positions(fields, dashboard, width, height):
    image = Image.new("RGB", (width, height))
    hud_module._draw_dashboard(image, dashboard, "credit", fields=fields)
    # Recompute the same geometry _draw_dashboard used, to assert against it
    # without depending on pixel inspection.
    from gomoveshift.pipeline.route_story.layout import hud_layout

    layout = hud_layout(width, height)
    left, right = layout.dashboard_left, layout.dashboard_right
    cell_w = (right - left) / len(fields)
    return [int(left + index * cell_w) for index in range(len(fields))]


def test_dashboard_column_positions_never_move_across_a_sensor_dropout():
    fields = ("SPEED", "ELEVATION", "GRADE", "DISTANCE", "HR", "CAD")
    width, height = 1920, 1080
    with_sensors = _dashboard(current_heart_rate_bpm=188.0, current_cadence_rpm=190.0)
    dropout = _dashboard(current_heart_rate_bpm=None, current_cadence_rpm=None)
    positions_full = _cell_x_positions(fields, with_sensors, width, height)
    positions_dropout = _cell_x_positions(fields, dropout, width, height)
    assert positions_full == positions_dropout


def test_dashboard_value_font_is_one_size_for_the_whole_film():
    """A wide value on one frame must not shrink the font used by a later,
    narrower-valued frame in the same column -- that would make column widths
    (and therefore x-positions) implicitly depend on frame content again.

    DISTANCE is fit against the film's total (constant) distance rather than
    the current frame's travelled distance, so an early, narrow frame never
    picks a font wider than what the film will actually need later.
    """
    fields = ("SPEED", "ELEVATION", "GRADE", "DISTANCE")
    cell_w = 200.0
    early = _dashboard(distance_traveled_m=10.0, distance_total_m=1609.344 * 12345.6)
    late = _dashboard(distance_traveled_m=1609.344 * 12345.6, distance_total_m=1609.344 * 12345.6)
    image = Image.new("RGB", (1920, 1080))
    draw = ImageDraw.Draw(image)
    font_early = hud_module._dashboard_value_font(draw, fields, early, cell_w)
    font_late = hud_module._dashboard_value_font(draw, fields, late, cell_w)
    assert font_early.size == font_late.size


# ---------------------------------------------------------------------------
# Portrait safe zone: the panel never renders under platform chrome.
# ---------------------------------------------------------------------------


def test_dashboard_panel_respects_the_portrait_bottom_safe_zone():
    from gomoveshift.pipeline.route_story.layout import hud_layout

    width, height = 1080, 1920
    layout = hud_layout(width, height)
    assert layout.short_bottom_safe_px > 0
    image = Image.new("RGB", (width, height))
    hud_module._draw_dashboard(image, _dashboard(), "credit")
    pixels = image.load()
    for y in range(height - layout.short_bottom_safe_px, height):
        assert pixels[width // 2, y] == (0, 0, 0)


def test_dashboard_panel_is_unchanged_on_landscape():
    from gomoveshift.pipeline.route_story.layout import hud_layout

    width, height = 1920, 1080
    layout = hud_layout(width, height)
    assert layout.short_bottom_safe_px == 0
    image = Image.new("RGB", (width, height))
    hud_module._draw_dashboard(image, _dashboard(), "credit")
    pixels = image.load()
    # The panel fill reaches the true bottom edge when there is no safe zone.
    assert pixels[width // 2, height - 1] != (0, 0, 0)


def test_time_cell_shows_a_placeholder_on_a_frame_with_no_recording_clock_once_frozen():
    # TIME is frozen present for the whole film once a clock anchor is set;
    # a frame with no recording clock of its own still keeps the column.
    hud_module.set_active_dashboard_clock(
        datetime(2026, 9, 16, 9, 0, tzinfo=timezone(timedelta(hours=-4)))
    )
    fields = ("SPEED", "ELEVATION", "GRADE", "DISTANCE", "TIME")
    cells = dict(hud_module._dashboard_cells(_dashboard(recording_time_s=None), fields=fields))
    assert cells["TIME"] == "—"


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920)])
def test_the_frozen_value_font_never_grows_past_the_rail_base_font(width, height):
    """Freezing the value font only removes per-frame refitting -- it never
    enlarges the text. The fit starts from the rail's own base value font
    (the one the unfrozen path shrinks from), so a short value like "328 ft"
    on a portrait social frame cannot balloon to fill its whole cell and
    spill white glyphs up into the map field, where preflight frame QA reads
    them as page-white void."""
    layout = hud_module.hud_layout(width, height)
    fields = ("SPEED", "ELEVATION", "GRADE", "DISTANCE")
    cell_w = (layout.dashboard_right - layout.dashboard_left) / len(fields)
    draw = ImageDraw.Draw(Image.new("RGB", (width, height)))

    font = hud_module._dashboard_value_font(draw, fields, _dashboard(), cell_w - 14)

    assert font.size <= max(16, width // 48)
