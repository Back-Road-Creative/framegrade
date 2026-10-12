"""Where the day started and where it ended are pinned on the map.

Two fixed pins, unlike the photo pins that accumulate as their beats play:
the first recorded fix and the last, present from the first map frame to the
last. Both come off the track -- there is no manual override and nothing is
inferred -- and both are coloured well away from the travelled line, the
marker and the white photo pin so ``frame_qa``'s colour detectors can never
mistake one for another.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from gomoveshift.pipeline.route_story import layers
from gomoveshift.pipeline.route_story.frame_qa import TRAVELLED_LINE_RGB
from gomoveshift.pipeline.route_story.layers import (
    ROUTE_FINISH_RGB,
    ROUTE_START_RGB,
    _render_map,
    route_endpoints,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster
from gomoveshift.pipeline.route_story.scene_renderer import SceneMapRenderer

WIDTH, HEIGHT = 320, 180


def _rasters(tmp_path: Path) -> list[SatelliteRaster]:
    coarse = tmp_path / "coarse.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(coarse)
    return [SatelliteRaster(coarse, -86.0, 33.0, -80.0, 37.5, "coarse")]


def _segments() -> list[list[RoutePoint]]:
    """Two segments -- a recorder pause splits the day, and the finish pin
    belongs to the LAST point of the LAST segment, not the first segment's.

    The endpoints are deliberately ~0.4 deg apart. ``_render_map`` frames the
    whole route, so a few-metre test route renders sub-pixel and the marker's
    own disc covers both pins -- the assertions would then fail on framing,
    not on the pins.
    """
    return [
        [
            RoutePoint(0, 35.0, -80.0, 100),
            RoutePoint(10, 35.1, -80.1, 100),
        ],
        [
            RoutePoint(60, 35.3, -80.3, 100),
            RoutePoint(70, 35.4, -80.4, 100),
        ],
    ]


def _nearest_colour_distance(image: Image.Image, colour: tuple[int, int, int]) -> int:
    return min(
        sum(abs(a - b) for a, b in zip(pixel, colour)) for pixel in image.convert("RGB").getdata()
    )


class TestRouteEndpoints:
    def test_the_endpoints_are_the_first_and_last_recorded_fixes(self) -> None:
        assert route_endpoints(_segments()) == ((-80.0, 35.0), (-80.4, 35.4))

    def test_a_track_with_no_points_has_no_endpoints(self) -> None:
        assert route_endpoints([]) is None
        assert route_endpoints([[]]) is None

    def test_a_single_fix_is_both_the_start_and_the_finish(self) -> None:
        """A one-point track is degenerate but not a crash: the day started
        and ended in the same place, and the map says so rather than raising."""
        lone = [[RoutePoint(0, 35.0, -80.0, 100)]]

        assert route_endpoints(lone) == ((-80.0, 35.0), (-80.0, 35.0))


class TestEndpointPinsOnTheMap:
    def test_the_finish_pin_is_there_from_the_very_first_frame(self, tmp_path: Path) -> None:
        """Unlike a photo pin, the finish is not a spoiler to withhold -- the
        untravelled route line already runs to it on frame one. Fading it in
        later would make the map's first frame disagree with its own line."""
        frame = _render_map(_rasters(tmp_path), _segments(), 0.0, WIDTH, HEIGHT)

        assert _nearest_colour_distance(frame, ROUTE_FINISH_RGB) <= 12

    def test_the_start_pin_is_still_there_once_the_marker_has_moved_on(
        self, tmp_path: Path
    ) -> None:
        """The whole point of a start pin is later in the day: it says where
        this all began while the marker is somewhere else entirely.

        Sampled mid-route, where the marker is in the recorder gap between the
        two segments and both pins sit clear of the dashboard rail. The rail
        is translucent over the lower map field and dims anything beneath it,
        pins included -- longstanding behaviour these pins share with the
        photo pins, not something the endpoint pins change.
        """
        frame = _render_map(_rasters(tmp_path), _segments(), 35.0, WIDTH, HEIGHT)

        assert _nearest_colour_distance(frame, ROUTE_START_RGB) <= 12

    def test_the_marker_covers_a_pin_it_is_standing_on(self, tmp_path: Path) -> None:
        """Deliberate z-order, not a defect. At route-second 0 the marker IS
        the start, and a pin drawn over the marker would hide the one thing
        the frame most needs to show -- where the rider is right now.
        """
        frame = _render_map(_rasters(tmp_path), _segments(), 0.0, WIDTH, HEIGHT)

        assert _nearest_colour_distance(frame, ROUTE_START_RGB) > 12


class TestEndpointPinColours:
    def test_neither_pin_can_be_mistaken_for_the_travelled_line(self) -> None:
        """``frame_qa`` measures the travelled line by colour proximity. A pin
        sitting on the line in a near-enough yellow would be counted as line
        width, and a gate built to catch a hairline route would pass on the
        pin instead."""
        for pin in (ROUTE_START_RGB, ROUTE_FINISH_RGB):
            distance = sum(abs(a - b) for a, b in zip(pin, TRAVELLED_LINE_RGB))
            assert distance >= 150, f"{pin} is too close to the travelled line"

    def test_the_two_pins_cannot_be_mistaken_for_each_other(self) -> None:
        distance = sum(abs(a - b) for a, b in zip(ROUTE_START_RGB, ROUTE_FINISH_RGB))
        assert distance >= 150

    def test_neither_pin_is_the_white_photo_pin(self) -> None:
        for pin in (ROUTE_START_RGB, ROUTE_FINISH_RGB):
            assert sum(abs(a - b) for a, b in zip(pin, (255, 255, 255))) >= 150


class _FakeCapture:
    """Records every pose it is asked to render and returns a solid frame."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.poses: list[dict] = []

    def render(self, pose, light=None):
        self.poses.append(pose)
        return np.full((self.height, self.width, 3), (80, 90, 100), dtype=np.uint8), []


class TestSceneRendererEndpoints:
    """The scene renderer is the production path for the route-story film, so
    a pin that exists only in the Pillow renderer does not ship at all."""

    def _renderer(self, capture: _FakeCapture) -> SceneMapRenderer:
        return SceneMapRenderer(
            capture=capture,
            trip_epoch=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            max_zoom=16.0,
            follow_zoom=14.0,
        )

    def test_the_pose_carries_the_tracks_endpoints(self) -> None:
        capture = _FakeCapture(WIDTH, int(HEIGHT * 0.855))

        self._renderer(capture).render(_segments(), 35.0, WIDTH, HEIGHT)

        assert capture.poses[0]["endpoints"] == [[-80.0, 35.0], [-80.4, 35.4]]

    def test_an_empty_track_leaves_the_pose_byte_identical(self) -> None:
        """A pose that predates endpoints must stay exactly what it was, or
        every frame already in the frame cache is invalidated for nothing."""
        capture = _FakeCapture(WIDTH, int(HEIGHT * 0.855))

        self._renderer(capture).render([[RoutePoint(0, 35.0, -80.0, 100)]], 0.0, WIDTH, HEIGHT)

        assert capture.poses[0]["endpoints"] == [[-80.0, 35.0], [-80.0, 35.0]]

    def test_scene_js_pins_the_endpoints_in_the_same_colours_as_pillow(self) -> None:
        """Two renderers, one film. If they disagreed, a story that switched
        renderers mid-way would change the colour of the same place.
        """
        scene_js = (Path(layers.__file__).parent / "scene" / "scene.js").read_text(encoding="utf-8")

        for name, rgb in (
            ("ROUTE_START_COLOR", ROUTE_START_RGB),
            ("ROUTE_FINISH_COLOR", ROUTE_FINISH_RGB),
        ):
            expected = "#%02x%02x%02x" % rgb
            assert f'var {name} = "{expected}";' in scene_js, (
                f"scene.js {name} must be {expected} to match layers.py"
            )
