"""Aspect-aware HUD anchor geometry (portrait 9:16 HUD layout)."""

from PIL import Image

from gomoveshift.pipeline.route_story import hud as hud_module
from gomoveshift.pipeline.route_story.layout import SHORT_BOTTOM_SAFE_PX, hud_layout

LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT = 1920, 1080
PORTRAIT_WIDTH, PORTRAIT_HEIGHT = 1080, 1920


# ---------------------------------------------------------------------------
# Landscape: exactly today's literal positions (locked in BEFORE the hud.py
# draw functions were refactored to read from this module).
# ---------------------------------------------------------------------------


def test_landscape_layout_matches_the_historical_literal_positions():
    layout = hud_layout(LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT)
    assert layout.title_y == LANDSCAPE_HEIGHT * 0.13
    assert layout.title_safe_width == LANDSCAPE_WIDTH * 0.88
    assert layout.dashboard_panel_height == max(72, int(LANDSCAPE_HEIGHT * 0.135))
    assert layout.dashboard_left == int(LANDSCAPE_WIDTH * 0.045)
    assert layout.dashboard_right == int(LANDSCAPE_WIDTH * 0.955)
    assert layout.badge_summit_top == LANDSCAPE_HEIGHT * 0.80
    assert layout.badge_arrival_top == LANDSCAPE_HEIGHT * 0.12
    assert layout.watermark_x == LANDSCAPE_WIDTH - LANDSCAPE_WIDTH * 0.06
    assert layout.watermark_y == LANDSCAPE_HEIGHT - LANDSCAPE_HEIGHT * 0.02
    assert layout.watermark_anchor == "rs"
    assert layout.photo_label_x == int(LANDSCAPE_WIDTH * 0.055)
    assert layout.photo_label_y == int(LANDSCAPE_HEIGHT * 0.84)
    assert layout.short_bottom_safe_px == 0


def test_a_square_frame_is_treated_as_landscape():
    # width >= height is the landscape branch, exactly per hud_layout's contract.
    layout = hud_layout(1080, 1080)
    assert layout.watermark_anchor == "rs"
    assert layout.short_bottom_safe_px == 0


def test_landscape_badge_render_lands_at_the_historical_pixel_row():
    image = Image.new("RGB", (LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT), "#101010")
    hud_module.draw_high_point(image, "HIGH POINT · 1,000 ft", LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT)
    # The historical formula: y0 = height * 0.80.
    expected_top = int(LANDSCAPE_HEIGHT * 0.80)
    pixels = image.load()
    row_has_ink = any(pixels[x, expected_top] != (16, 16, 16) for x in range(0, LANDSCAPE_WIDTH, 4))
    assert row_has_ink


def test_landscape_watermark_render_lands_at_the_historical_bottom_right_box():
    image = Image.new("RGB", (LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT), "#123456")
    box = hud_module.draw_watermark(
        image, "Made with GoMoveShift", LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT
    )
    layout = hud_layout(LANDSCAPE_WIDTH, LANDSCAPE_HEIGHT)
    # A little slack for the text's stroke, which extends past the anchor.
    assert box[2] <= layout.watermark_x + 20
    assert box[3] <= layout.watermark_y + 20


# ---------------------------------------------------------------------------
# Portrait: badges/watermark move to bottom-centre, clear of the platform
# safe zone; nothing renders outside the canvas.
# ---------------------------------------------------------------------------


def test_portrait_watermark_is_bottom_centered_above_the_safe_zone():
    layout = hud_layout(PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    assert layout.watermark_anchor == "ms"
    assert layout.watermark_x == PORTRAIT_WIDTH / 2
    assert layout.watermark_y <= PORTRAIT_HEIGHT - SHORT_BOTTOM_SAFE_PX
    assert layout.short_bottom_safe_px == SHORT_BOTTOM_SAFE_PX


def test_portrait_badges_stack_above_the_safe_zone_and_dashboard():
    layout = hud_layout(PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    slot = max(15, PORTRAIT_WIDTH // 70) + 18
    assert layout.badge_summit_top + slot <= PORTRAIT_HEIGHT - SHORT_BOTTOM_SAFE_PX
    assert layout.badge_summit_top + slot <= PORTRAIT_HEIGHT - layout.dashboard_panel_height
    assert layout.badge_arrival_top + slot < layout.badge_summit_top


def test_portrait_dashboard_and_title_scale_from_width_not_height():
    layout = hud_layout(PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    # Scaling from width (the short side in portrait) instead of height keeps
    # the bar's on-screen thickness comparable to the landscape cut, rather
    # than the much taller panel a height-based formula would give here.
    assert layout.dashboard_panel_height == max(72, int(PORTRAIT_WIDTH * 0.135))
    assert layout.dashboard_panel_height < int(PORTRAIT_HEIGHT * 0.135)


def test_portrait_high_point_badge_renders_within_the_canvas_and_safe_zone():
    image = Image.new("RGBA", (PORTRAIT_WIDTH, PORTRAIT_HEIGHT), (16, 16, 16, 255))
    hud_module.draw_high_point(image, "HIGH POINT · 1,000 ft", PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    draw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw(image)
    font = hud_module._font(max(15, PORTRAIT_WIDTH // 70), bold=True)
    bbox = draw.textbbox(
        (PORTRAIT_WIDTH / 2, hud_layout(PORTRAIT_WIDTH, PORTRAIT_HEIGHT).badge_summit_top + 9),
        "HIGH POINT · 1,000 ft",
        font=font,
        anchor="ma",
    )
    assert 0 <= bbox[0]
    assert bbox[2] <= PORTRAIT_WIDTH
    assert bbox[3] <= PORTRAIT_HEIGHT - SHORT_BOTTOM_SAFE_PX


def test_portrait_media_arrival_badge_is_horizontally_centered():
    layout = hud_layout(PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    image = Image.new("RGBA", (PORTRAIT_WIDTH, PORTRAIT_HEIGHT), (16, 16, 16, 255))
    hud_module._draw_media_arrival(image, "VIDEO ARRIVAL", PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    # The badge box is centred on width / 2 regardless of orientation --
    # _badge's own x0 formula already centres on text width, unchanged here.
    assert layout.badge_arrival_top < PORTRAIT_HEIGHT - SHORT_BOTTOM_SAFE_PX


def test_portrait_watermark_render_stays_inside_the_canvas_and_safe_zone():
    image = Image.new("RGB", (PORTRAIT_WIDTH, PORTRAIT_HEIGHT), "#123456")
    box = hud_module.draw_watermark(image, "Made with GoMoveShift", PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    assert box[0] >= 0
    assert box[2] <= PORTRAIT_WIDTH
    assert box[3] <= PORTRAIT_HEIGHT - SHORT_BOTTOM_SAFE_PX


def test_portrait_dashboard_panel_top_is_the_panel_height_above_the_safe_zone():
    layout = hud_layout(PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    # One resolved row shared by the painter and frame QA: the panel's top
    # edge, so what the rail covers and what the map-field detectors skip
    # can never drift apart again.
    assert layout.dashboard_panel_top == (
        PORTRAIT_HEIGHT - SHORT_BOTTOM_SAFE_PX - layout.dashboard_panel_height
    )


def test_landscape_dashboard_panel_top_runs_to_the_frame_bottom():
    layout = hud_layout(1920, 1080)
    assert layout.dashboard_panel_top == 1080 - layout.dashboard_panel_height
