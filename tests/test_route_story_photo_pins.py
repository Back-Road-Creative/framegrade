"""A pin stays on the route where each played photo/video beat happened.

Covers the pure filter/collapse helper (``layers.visible_pins``), both
``MapRenderer`` implementations drawing from it identically, the recap map,
the film receipt count, and that the new pin colour never confuses
``frame_qa``'s travelled-line-width detector.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import PhotoMoment, film_receipt, render_route_story_film
from gomoveshift.pipeline.route_story.frame_qa import FrameQaThresholds, check_frame
from gomoveshift.pipeline.route_story.layers import PillowMapRenderer, _render_map, visible_pins
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster
from gomoveshift.pipeline.route_story.scene_renderer import SceneMapRenderer

WIDTH, HEIGHT = 320, 180
MAP_HEIGHT = int(HEIGHT * 0.855)
TRIP_EPOCH = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


class _FakeCapture:
    """Records every pose it is asked to render and returns a solid frame."""

    def __init__(self, width: int, height: int, color: tuple[int, int, int] = (80, 90, 100)):
        self.width = width
        self.height = height
        self.color = color
        self.poses: list[dict] = []

    def render(self, pose, light=None):
        self.poses.append(pose)
        array = np.full((self.height, self.width, 3), self.color, dtype=np.uint8)
        return array, []


def _straight_segments(count: int = 60, step_s: float = 1.0) -> list[list[RoutePoint]]:
    return [
        [
            RoutePoint(i * step_s, 35.0 + i * 0.0004, -80.0 + i * 0.0006, 100.0 + i)
            for i in range(count)
        ]
    ]


def _rasters(tmp_path: Path) -> list[SatelliteRaster]:
    coarse, fine = tmp_path / "coarse.png", tmp_path / "fine.png"
    Image.new("RGB", (64, 64), (200, 30, 30)).save(coarse)
    Image.new("RGB", (64, 64), (30, 160, 60)).save(fine)
    return [
        SatelliteRaster(coarse, -86.0, 33.0, -80.0, 37.5, "coarse"),
        SatelliteRaster(fine, -80.05, 34.95, -79.95, 35.05, "fine"),
    ]


def _segments() -> list[list[RoutePoint]]:
    return [
        [
            RoutePoint(0, 35.0, -80.0, 100),
            RoutePoint(10, 35.0002, -80.0002, 100),
            RoutePoint(20, 35.0004, -80.0004, 100),
        ]
    ]


class TestVisiblePins:
    def test_default_empty_beats_yield_no_pins(self):
        assert visible_pins((), 42.0) == []

    def test_a_beat_is_invisible_strictly_before_its_own_time(self):
        beats = [(10.0, -80.0, 35.0)]
        assert visible_pins(beats, 9.999) == []

    def test_a_beat_becomes_visible_at_exactly_its_own_time(self):
        beats = [(10.0, -80.0, 35.0)]
        assert visible_pins(beats, 10.0) == [(-80.0, 35.0)]

    def test_a_beat_stays_visible_after_its_own_time(self):
        beats = [(10.0, -80.0, 35.0)]
        assert visible_pins(beats, 999.0) == [(-80.0, 35.0)]

    def test_beats_within_the_collapse_radius_keep_only_the_first(self):
        # ~5 m apart at this latitude -- well inside the 15 m collapse radius.
        beats = [(1.0, -80.0, 35.0), (2.0, -80.00004, 35.0)]
        assert visible_pins(beats, 100.0) == [(-80.0, 35.0)]

    def test_beats_beyond_the_collapse_radius_both_survive(self):
        # ~1.1 km apart -- far outside the collapse radius.
        beats = [(1.0, -80.0, 35.0), (2.0, -79.99, 35.0)]
        assert visible_pins(beats, 100.0) == [(-80.0, 35.0), (-79.99, 35.0)]

    def test_only_already_played_beats_count_toward_collapse(self):
        # The second beat has not played yet -- it must neither appear NOR
        # suppress a later, farther-apart beat at the same nominal spot.
        beats = [(1.0, -80.0, 35.0), (500.0, -80.00004, 35.0)]
        assert visible_pins(beats, 100.0) == [(-80.0, 35.0)]


class TestSceneMapRendererPins:
    def test_pose_carries_only_already_played_pins(self):
        segments = _straight_segments()
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture,
            trip_epoch=TRIP_EPOCH,
            max_zoom=16.0,
            follow_zoom=14.0,
            pins=[(5.0, -80.0, 35.0), (50.0, -79.98, 35.02)],
        )

        renderer.render(segments, 10.0, WIDTH, HEIGHT)

        assert capture.poses[0]["pins"] == [[-80.0, 35.0]]

    def test_default_pins_field_is_empty_and_pose_carries_no_pins(self):
        segments = _straight_segments()
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture, trip_epoch=TRIP_EPOCH, max_zoom=16.0, follow_zoom=14.0
        )

        renderer.render(segments, 10.0, WIDTH, HEIGHT)

        # No pin yet: the pose is the very dict it was before pins existed, so a
        # frame cache keyed on the pose keeps every pin-less frame it already has.
        assert "pins" not in capture.poses[0]

    def test_a_dataclasses_replace_clone_still_carries_the_pins_on_its_pose(self):
        # The per-worker clone `_capture_leg_parallel` makes via
        # `dataclasses.replace` must carry `pins` like every other field --
        # each worker's own page needs the already-played pins too.
        segments = _straight_segments()
        capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        renderer = SceneMapRenderer(
            capture=capture,
            trip_epoch=TRIP_EPOCH,
            max_zoom=16.0,
            follow_zoom=14.0,
            pins=[(1.0, -80.0, 35.0)],
        )

        worker_capture = _FakeCapture(WIDTH, MAP_HEIGHT)
        worker = replace(renderer, capture=worker_capture)
        worker.render(segments, 10.0, WIDTH, HEIGHT)

        assert worker_capture.poses[0]["pins"] == [[-80.0, 35.0]]

    def test_multi_worker_render_leg_carries_pins_on_every_frame(self):
        # Exercises the real parallel path end to end: every worker opens its
        # own capture via capture_factory (SceneMapRenderer._capture_leg_parallel),
        # and every one of them must see the already-played pins.
        segments = _straight_segments(count=200)
        captures: list[_FakeCapture] = []

        def factory():
            capture = _FakeCapture(WIDTH, MAP_HEIGHT)
            captures.append(capture)
            return _NullContext(capture)

        renderer = SceneMapRenderer(
            capture=_FakeCapture(WIDTH, MAP_HEIGHT),
            trip_epoch=TRIP_EPOCH,
            max_zoom=16.0,
            follow_zoom=14.0,
            pins=[(1.0, -80.0, 35.0)],
            workers=3,
            capture_factory=factory,
        )
        times = [float(i) for i in range(1, 13)]

        list(renderer.render_leg(segments, times, WIDTH, HEIGHT))

        all_poses = [pose for capture in captures for pose in capture.poses]
        assert len(all_poses) == len(times)
        assert all(pose["pins"] == [[-80.0, 35.0]] for pose in all_poses)


class _NullContext:
    def __init__(self, value) -> None:
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *exc_info) -> bool:
        return False


class TestPillowRendererPins:
    def test_zero_visible_pins_renders_byte_identical_to_no_pins_argument(self, tmp_path):
        rasters = _rasters(tmp_path)
        segments = _segments()

        without_pins = _render_map(rasters, segments, 5.0, WIDTH, HEIGHT, show_dashboard=False)
        # A beat far in the future is present but never visible at this
        # at_time -- must draw nothing, exactly like leaving pins out.
        with_future_pin = _render_map(
            rasters,
            segments,
            5.0,
            WIDTH,
            HEIGHT,
            show_dashboard=False,
            pins=[(999.0, -80.0, 35.0)],
        )

        assert np.array_equal(np.asarray(without_pins), np.asarray(with_future_pin))

    def test_a_pin_appears_only_once_its_beat_has_played(self, tmp_path):
        rasters = _rasters(tmp_path)
        segments = _segments()
        # Well clear of the route/marker itself so the marker disc can never
        # simply paint over the pin and hide a real difference.
        pins = [(10.0, -80.003, 35.003)]

        before = _render_map(rasters, segments, 5.0, WIDTH, HEIGHT, show_dashboard=False, pins=pins)
        before_baseline = _render_map(rasters, segments, 5.0, WIDTH, HEIGHT, show_dashboard=False)
        after = _render_map(rasters, segments, 15.0, WIDTH, HEIGHT, show_dashboard=False, pins=pins)
        after_baseline = _render_map(rasters, segments, 15.0, WIDTH, HEIGHT, show_dashboard=False)

        assert np.array_equal(np.asarray(before), np.asarray(before_baseline))
        assert not np.array_equal(np.asarray(after), np.asarray(after_baseline))

    def test_render_leg_thread_the_renderer_own_pins_field(self, tmp_path):
        rasters = _rasters(tmp_path)
        segments = _segments()
        renderer = PillowMapRenderer(rasters=tuple(rasters), pins=[(10.0, -80.0002, 35.0002)])

        via_renderer = renderer.render(segments, 15.0, WIDTH, HEIGHT)
        via_helper = _render_map(
            rasters,
            segments,
            15.0,
            WIDTH,
            HEIGHT,
            pins=[(10.0, -80.0002, 35.0002)],
        )

        assert np.array_equal(np.asarray(via_renderer), np.asarray(via_helper))


class TestFilmPhotoPinsWiring:
    def _photo(self, source: Path, anchor: MediaAnchor) -> PhotoMoment:
        return PhotoMoment(source, anchor)

    def test_receipt_counts_one_pin_per_mapped_photo(self, tmp_path):
        national = tmp_path / "n.jpg"
        local = tmp_path / "l.jpg"
        photo_a = tmp_path / "a.jpg"
        photo_b = tmp_path / "b.jpg"
        for path in (national, local, photo_a, photo_b):
            Image.new("RGB", (16, 16), "#334455").save(path)

        receipt = film_receipt(
            segments=_segments(),
            national=SatelliteRaster(national, -180, -90, 180, 90, "n"),
            local=SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "l"),
            photos=[
                self._photo(
                    photo_a,
                    MediaAnchor("a.jpg", "timestamp_interpolated", 5.0, 35.00005, -79.99995),
                ),
                self._photo(
                    photo_b,
                    MediaAnchor("b.jpg", "timestamp_interpolated", 15.0, 35.0003, -80.0003),
                ),
            ],
        )

        assert receipt["photo_pins"] == 2

    def test_receipt_has_no_pins_with_no_segments(self, tmp_path):
        national = tmp_path / "n.jpg"
        local = tmp_path / "l.jpg"
        Image.new("RGB", (16, 16), "#334455").save(national)
        Image.new("RGB", (16, 16), "#334455").save(local)

        receipt = film_receipt(
            segments=[],
            national=SatelliteRaster(national, -180, -90, 180, 90, "n"),
            local=SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "l"),
            photos=[],
        )

        assert receipt["photo_pins"] == 0

    def test_render_route_story_film_leaves_a_pin_on_the_recap(self, tmp_path):
        overview = tmp_path / "overview.jpg"
        local = tmp_path / "local.jpg"
        photo = tmp_path / "photo.jpg"
        Image.new("RGB", (720, 360), "#3c6f83").save(overview)
        Image.new("RGB", (640, 360), "#4e7540").save(local)
        Image.new("RGB", (640, 360), "#aa7744").save(photo)
        output = tmp_path / "film.mp4"
        seen_recap_pins: list[list] = []
        real_render_recap_map = film_module._render_recap_map

        def spy_render_recap_map(*args, **kwargs):
            seen_recap_pins.append(kwargs.get("pins"))
            return real_render_recap_map(*args, **kwargs)

        film_module._render_recap_map = spy_render_recap_map
        try:
            render_route_story_film(
                segments=[
                    [RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]
                ],
                national=SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
                local=SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
                photos=[
                    PhotoMoment(
                        photo,
                        MediaAnchor(
                            "photo.jpg", "timestamp_interpolated", 5.0, 35.00005, -79.99995
                        ),
                    )
                ],
                output=output,
                fps=2,
                width=320,
                height=180,
                intro_seconds=1,
                map_seconds=1,
                photo_seconds=1,
                recap_seconds=1,
            )
        finally:
            film_module._render_recap_map = real_render_recap_map

        assert output.is_file()
        assert seen_recap_pins
        assert seen_recap_pins[0] == [(5.0, -79.99995, 35.00005)]


class TestFrameQaPinColour:
    """The pin's white fill/dark outline must never read as the travelled
    line, and must not change the travelled-line-width measurement near the
    marker (`frame_qa.travelled_line_min_width`)."""

    def _frame_with_travelled_stripe(self) -> np.ndarray:
        height, width = 400, 400
        frame = np.full((height, width, 3), (80, 90, 100), dtype=np.uint8)
        # A 6px-wide vertical travelled-line stripe well clear of the
        # marker's own exclusion radius (12px) but inside the sampling
        # radius (200px).
        frame[150:250, 257:263] = (0xFF, 0xCF, 0x3F)
        return frame

    def _draw_pin(self, frame: np.ndarray, x: int, y: int, radius: int = 6) -> np.ndarray:
        frame = frame.copy()
        yy, xx = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
        dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
        frame[dist <= radius] = (255, 255, 255)
        outline = (dist > radius) & (dist <= radius + 2)
        frame[outline] = (0x07, 0x12, 0x19)
        return frame

    def test_pin_beside_the_marker_does_not_change_travelled_line_width(self):
        marker_xy = (200.0, 200.0)
        thresholds = FrameQaThresholds()
        base = self._frame_with_travelled_stripe()
        # ~42px from the marker: outside the 12px exclusion zone, and clear
        # of the stripe at x=257-263.
        with_pin = self._draw_pin(base, x=170, y=170)

        without_pin_result = check_frame(0, base, thresholds, marker_xy=marker_xy)
        with_pin_result = check_frame(0, with_pin, thresholds, marker_xy=marker_xy)

        assert without_pin_result.travelled_line_width_px is not None
        assert with_pin_result.travelled_line_width_px == without_pin_result.travelled_line_width_px
        assert with_pin_result.problems == without_pin_result.problems

    def test_pin_white_fill_is_far_from_the_travelled_line_colour_by_tolerance(self):
        # The exact per-channel test travelled_line_min_width uses: white vs
        # ROUTE_BRIGHT_COLOR must differ by more than its 40-tolerance on at
        # least one channel, so a pin pixel is never mistaken for the line.
        pin_rgb = np.array((255, 255, 255))
        travelled_rgb = np.array((0xFF, 0xCF, 0x3F))
        assert int(np.abs(pin_rgb - travelled_rgb).max()) > 40
