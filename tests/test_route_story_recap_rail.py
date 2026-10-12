"""The recap's numbers sit in the film's bottom rail, never over the map."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image, ImageChops

from gomoveshift.pipeline.route_story import hud as hud_module

_MAP = (40, 110, 60)
_RAIL_EMPTY = (7, 18, 25)


def _summary() -> SimpleNamespace:
    return SimpleNamespace(
        distance_m=10782.0,
        ascent_m=443.5,
        elapsed_seconds=13500.0,
        moving_average_speed_mps=0.98,
    )


def _recap_frame(width: int = 1920, height: int = 1080) -> tuple[Image.Image, int]:
    """A frame as ``_render_recap_map`` hands it over: map on top, empty rail below."""
    map_height = int(height * 0.855)
    image = Image.new("RGB", (width, height), _RAIL_EMPTY)
    image.paste(Image.new("RGB", (width, map_height), _MAP), (0, 0))
    return image, map_height


class TestRecapRail:
    def test_nothing_is_drawn_over_the_map(self):
        image, map_height = _recap_frame()
        before = image.crop((0, 0, image.width, map_height)).copy()

        hud_module._draw_recap(image, _summary(), 1.0)

        after = image.crop((0, 0, image.width, map_height))
        assert ImageChops.difference(before, after).getbbox() is None

    def test_the_rail_carries_the_numbers(self):
        image, map_height = _recap_frame()

        hud_module._draw_recap(image, _summary(), 1.0)

        rail = image.crop((0, map_height, image.width, image.height)).convert("L")
        bright = rail.point(lambda value: 255 if value > 200 else 0).getbbox()
        assert bright is not None
        # Text spans the rail like the dashboard does, not one corner of it.
        assert bright[2] - bright[0] > image.width * 0.5

    def test_the_rail_cells_name_the_day(self):
        cells = dict(hud_module._recap_cells(_summary()))

        assert cells == {
            "DISTANCE": "6.7 mi",
            "ASCENT": "1,455 ft",
            "TIME": "3h 45m",
            "MOVING AVG": "2.2 mph",
        }

    def test_the_numbers_fade_in_rather_than_pop(self):
        start, map_height = _recap_frame()
        end, _ = _recap_frame()

        hud_module._draw_recap(start, _summary(), 0.0)
        hud_module._draw_recap(end, _summary(), 1.0)

        def peak(image: Image.Image) -> int:
            return (
                image.crop((0, map_height, image.width, image.height)).convert("L").getextrema()[1]
            )

        assert peak(start) < peak(end)

    def test_the_numbers_are_fully_lit_for_most_of_the_recap(self):
        # A real 1080p film faded the rail across the whole recap: the numbers
        # reached full brightness on the last frame, with no time left to read them.
        early, map_height = _recap_frame()
        end, _ = _recap_frame()

        hud_module._draw_recap(early, _summary(), 0.2)
        hud_module._draw_recap(end, _summary(), 1.0)

        box = (0, map_height, early.width, early.height)
        assert ImageChops.difference(early.crop(box), end.crop(box)).getbbox() is None
