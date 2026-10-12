"""The route's strokes are sized off the frame, in BOTH renderers.

``layers.py`` has always scaled its route strokes with frame width; ``scene.js``
-- the production renderer -- did not. Its pins scaled (``height * 6 / 1080``,
``height * 8 / 1080``) while the line through them stayed a flat 3px, so at a
larger master the pins grew and the line they sit on did not: at 2560px wide the
travelled line is the same 3px it was at 1920, a quarter thinner against
everything around it.

The divisors live in Python and the test derives the expected JavaScript from
them, so the two renderers cannot drift apart without this file going red.
"""

from __future__ import annotations

from pathlib import Path

from gomoveshift.pipeline.route_story import layers
from gomoveshift.pipeline.route_story.frame_qa import FrameQaThresholds

PROXY_WIDTH = 960
MASTER_1080P_WIDTH = 1920
MASTER_1440P_WIDTH = 2560


def _scene_js() -> str:
    return (Path(layers.__file__).parent / "scene" / "scene.js").read_text(encoding="utf-8")


class TestPythonWidths:
    def test_every_stroke_grows_with_the_frame(self) -> None:
        """The whole point: a 1440p master must not ship the 1080p line."""
        for measure in (
            layers.route_line_width_px,
            layers.travelled_line_width_px,
            layers.route_halo_width_px,
        ):
            assert measure(MASTER_1440P_WIDTH) > measure(MASTER_1080P_WIDTH)

    def test_the_halo_is_always_wider_than_the_line_it_haloes(self) -> None:
        for width in (PROXY_WIDTH, MASTER_1080P_WIDTH, MASTER_1440P_WIDTH):
            assert layers.route_halo_width_px(width) > layers.route_line_width_px(width)

    def test_the_travelled_line_clears_the_render_s_own_floor_at_every_size(self) -> None:
        """Including the 960px proxy. A stroke that scaled below the gate's
        floor would make the proxy pass fail on its own thinness rather than
        on anything wrong with the film."""
        floor = FrameQaThresholds().min_travelled_line_width_px
        for width in (PROXY_WIDTH, MASTER_1080P_WIDTH, MASTER_1440P_WIDTH):
            assert layers.travelled_line_width_px(width) >= floor

    def test_a_tiny_frame_still_gets_a_visible_stroke(self) -> None:
        """The floors are what stop a thumbnail-sized render drawing nothing."""
        assert layers.route_line_width_px(1) >= 2
        assert layers.travelled_line_width_px(1) >= 4
        assert layers.route_halo_width_px(1) >= 5

    def test_the_1080p_widths_are_unchanged(self) -> None:
        """The 1440p scaling must not move 1080p: that would silently restyle
        every film already approved at that size."""
        assert layers.route_halo_width_px(MASTER_1080P_WIDTH) == 8
        assert layers.route_line_width_px(MASTER_1080P_WIDTH) == 3
        assert layers.travelled_line_width_px(MASTER_1080P_WIDTH) == 6


class TestSceneJsParity:
    """scene.js is the production renderer; a width fixed only in Pillow never
    reaches a frame."""

    def test_scene_js_derives_each_stroke_from_the_same_divisor(self) -> None:
        scene_js = _scene_js()
        for name, floor, divisor in (
            ("routeLineWidthPx", 2, layers.ROUTE_LINE_WIDTH_DIVISOR),
            ("travelledLineWidthPx", 4, layers.TRAVELLED_LINE_WIDTH_DIVISOR),
            ("routeHaloWidthPx", 5, layers.ROUTE_HALO_WIDTH_DIVISOR),
        ):
            expected = f"Math.max({floor}, Math.floor(width / {divisor}))"
            assert expected in scene_js, f"scene.js {name} must compute {expected}"

    def test_scene_js_no_longer_carries_a_flat_route_stroke_constant(self) -> None:
        """The constants are gone, not shadowed. A leftover
        ``ROUTE_LINE_WIDTH_PX = 3`` beside the new helper is the exact shape
        that lets a later edit quietly go back to the flat width."""
        scene_js = _scene_js()
        for dead in ("OVERLAY_LINE_WIDTH_PX", "ROUTE_LINE_WIDTH_PX", "ROUTE_HALO_EXTRA_PX"):
            assert dead not in scene_js, f"{dead} should be gone from scene.js"
