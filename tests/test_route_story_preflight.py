"""Pre-render QA gate: a handful of frames, checked and tiled, before the
long film render starts."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.frame_qa import (
    TRAVELLED_LINE_RGB,
    FrameQaError,
    check_frame,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.preflight import (
    CONTACT_SHEET_COLUMNS,
    THUMB_WIDTH,
    RoutePreflightError,
    preflight_times,
    render_preflight_frames,
    run_preflight,
)
from gomoveshift.pipeline.route_story.scene_renderer import SceneMapRenderer, scene_map_height

SEGMENTS = [[RoutePoint(0.0, 35.0, -80.0, 100.0), RoutePoint(10.0, 35.001, -79.999, 110.0)]]


class _FakeRenderer:
    """Records every call; can be made to return a bad frame at one index."""

    name = "fake"

    def __init__(self, fail_index: int | None = None, ok_color=(80, 90, 100)):
        self._fail_index = fail_index
        self._ok_color = ok_color
        self.render_calls: list[float] = []
        self.leg_calls = 0

    def render(self, segments, at_time, width, height):
        index = len(self.render_calls)
        self.render_calls.append(at_time)
        color = (255, 255, 255) if index == self._fail_index else self._ok_color
        return Image.new("RGB", (width, height), color)

    def render_leg(self, segments, times, width, height):
        self.leg_calls += 1
        for at_time in times:
            yield self.render(segments, at_time, width, height)


def test_preflight_times_are_evenly_spaced_and_include_endpoints():
    times = preflight_times(0.0, 10.0, 5)

    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(10.0)
    assert times == pytest.approx([0.0, 2.5, 5.0, 7.5, 10.0])


def test_render_preflight_frames_with_zero_count_skips_the_renderer():
    renderer = _FakeRenderer()

    frames = render_preflight_frames(
        renderer, SEGMENTS, start_time=0.0, end_time=10.0, width=64, height=36, count=0
    )

    assert frames == []
    assert renderer.render_calls == []


def test_a_qa_failing_frame_raises_before_any_leg_is_rendered(tmp_path: Path):
    renderer = _FakeRenderer(fail_index=2)

    with pytest.raises(RoutePreflightError, match="page-white"):
        run_preflight(
            renderer,
            SEGMENTS,
            start_time=0.0,
            end_time=10.0,
            width=64,
            height=36,
            count=5,
            output=tmp_path / "out-preflight.jpg",
        )

    assert renderer.leg_calls == 0


class _HudRailRenderer(_FakeRenderer):
    """A good map field over the film's dark dashboard rail, like a composed film frame."""

    def __init__(self, map_color=(80, 90, 100)):
        super().__init__(ok_color=map_color)

    def render(self, segments, at_time, width, height):
        image = super().render(segments, at_time, width, height)
        rail_top = scene_map_height(height)
        # The rail's colour over dark imagery lands within the detector's
        # tolerance of scene.js's #101418 "unloaded tile" background.
        image.paste((14, 19, 23), (0, rail_top, width, height))
        return image


def test_the_dashboard_rail_is_not_mistaken_for_unloaded_tiles(tmp_path: Path):
    result = run_preflight(
        _HudRailRenderer(),
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=64,
        height=200,
        count=3,
        output=tmp_path / "out-preflight.jpg",
    )
    assert result.passed


def test_an_unloaded_map_field_still_fails_above_the_dashboard_rail(tmp_path: Path):
    with pytest.raises(RoutePreflightError, match="unloaded-tile"):
        run_preflight(
            _HudRailRenderer(map_color=(16, 20, 24)),
            SEGMENTS,
            start_time=0.0,
            end_time=10.0,
            width=64,
            height=200,
            count=3,
            output=tmp_path / "out-preflight.jpg",
        )


class _SelfCheckingRenderer(_FakeRenderer):
    """Like the scene renderer: QA's its own raw capture, then draws dark label pills."""

    checks_raw_frames = True

    def __init__(self, raw_fails_at: int | None = None):
        super().__init__()
        self._raw_fails_at = raw_fails_at

    def render(self, segments, at_time, width, height):
        index = len(self.render_calls)
        image = super().render(segments, at_time, width, height)
        if index == self._raw_fails_at:
            raw = np.full((height, width, 3), (16, 20, 24), dtype=np.uint8)
            raise FrameQaError(check_frame(index, raw), {"bearing": 12.0}, rgb=raw)
        # Label pills: 5% of the map field, inside the unloaded-tile tolerance.
        image.paste((10, 20, 30), (0, 0, width, max(1, height // 20)))
        return image


def test_a_renderer_that_checks_its_raw_frames_is_not_rejudged_on_its_label_pills(tmp_path: Path):
    result = run_preflight(
        _SelfCheckingRenderer(),
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=64,
        height=200,
        count=3,
        output=tmp_path / "out-preflight.jpg",
    )
    assert result.passed and result.frames == 3


def test_a_raw_frame_qa_failure_stops_preflight_and_the_sheet_shows_the_bad_frame(tmp_path: Path):
    renderer = _SelfCheckingRenderer(raw_fails_at=1)
    output = tmp_path / "out-preflight.jpg"

    with pytest.raises(RoutePreflightError, match="unloaded-tile") as excinfo:
        run_preflight(
            renderer,
            SEGMENTS,
            start_time=0.0,
            end_time=10.0,
            width=64,
            height=200,
            count=3,
            output=output,
        )

    assert "route time 5.0s" in str(excinfo.value)
    assert len(renderer.render_calls) == 2  # stopped at the bad frame
    assert renderer.leg_calls == 0
    # one good frame + the failing raw frame, tiled side by side
    assert Image.open(output).size[0] == THUMB_WIDTH * 2


class _MarkerAwareRenderer(_FakeRenderer):
    """Like SceneMapRenderer: exposes the last frame's marker position via
    ``last_marker_xy`` so run_preflight can thread it into check_frame's
    travelled-line-width floor."""

    def __init__(self, line_width_px: int, marker_xy=(50.0, 20.0)):
        super().__init__(ok_color=(30, 40, 50))
        self._line_width_px = line_width_px
        self._marker_xy = marker_xy
        self.last_marker_xy = None

    def render(self, segments, at_time, width, height):
        image = super().render(segments, at_time, width, height)
        pixels = image.load()
        marker_x = int(self._marker_xy[0])
        half = self._line_width_px // 2
        for y in range(height):
            for dx in range(-half, self._line_width_px - half):
                x = marker_x + dx
                if 0 <= x < width:
                    pixels[x, y] = TRAVELLED_LINE_RGB
        self.last_marker_xy = self._marker_xy
        return image


def test_run_preflight_records_travelled_line_width_for_a_marker_aware_renderer(
    tmp_path: Path,
):
    result = run_preflight(
        _MarkerAwareRenderer(line_width_px=3),
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=100,
        height=40,
        count=2,
        output=tmp_path / "out-preflight.jpg",
    )

    assert result.passed
    assert all(r.travelled_line_width_px is not None for r in result.results)
    assert all(r.travelled_line_width_px >= 2.0 for r in result.results)


def test_run_preflight_fails_a_hairline_travelled_line_near_the_marker(tmp_path: Path):
    with pytest.raises(RoutePreflightError, match="travelled"):
        run_preflight(
            _MarkerAwareRenderer(line_width_px=1),
            SEGMENTS,
            start_time=0.0,
            end_time=10.0,
            width=100,
            height=40,
            count=2,
            output=tmp_path / "out-preflight.jpg",
        )


def test_the_marker_sheet_is_written_from_the_preflight_walks_own_frames(tmp_path: Path):
    """One walk, not two. The close-ups an operator approves on are crops of
    the very frames the gate judged, captioned with the gate's own numbers."""
    sheet = tmp_path / "marker-sheet.jpg"

    result = run_preflight(
        _MarkerAwareRenderer(line_width_px=3),
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=100,
        height=40,
        count=2,
        output=tmp_path / "out-preflight.jpg",
        marker_sheet=sheet,
    )

    assert result.passed
    assert sheet.exists()
    assert result.marker_sheet == sheet


def test_the_marker_sheet_is_written_even_when_a_frame_fails_its_gate(tmp_path: Path):
    """The failing case is the one the sheet exists for. A pack that appears
    only on a clean run shows the operator nothing on the run they must judge.
    """
    sheet = tmp_path / "marker-sheet.jpg"

    with pytest.raises(RoutePreflightError, match="travelled"):
        run_preflight(
            _MarkerAwareRenderer(line_width_px=1),
            SEGMENTS,
            start_time=0.0,
            end_time=10.0,
            width=100,
            height=40,
            count=2,
            output=tmp_path / "out-preflight.jpg",
            marker_sheet=sheet,
        )

    assert sheet.exists()


def test_no_marker_sheet_is_written_when_none_is_asked_for(tmp_path: Path):
    result = run_preflight(
        _MarkerAwareRenderer(line_width_px=3),
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=100,
        height=40,
        count=2,
        output=tmp_path / "out-preflight.jpg",
    )

    assert result.marker_sheet is None
    assert not list(tmp_path.glob("marker*"))


def test_contact_sheet_is_written_with_the_expected_tile_grid(tmp_path: Path):
    renderer = _FakeRenderer()
    output = tmp_path / "out-preflight.jpg"
    count = 6

    result = run_preflight(
        renderer,
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=64,
        height=36,
        count=count,
        output=output,
    )

    assert result.passed
    assert result.sheet == output
    assert output.is_file()
    sheet = Image.open(output)
    columns = min(CONTACT_SHEET_COLUMNS, count)
    rows = math.ceil(count / columns)
    thumb_height = round(THUMB_WIDTH * 36 / 64)
    assert sheet.size == (THUMB_WIDTH * columns, thumb_height * rows)


def test_preflight_frames_zero_disables_the_gate_in_the_cli(tmp_path: Path, monkeypatch):
    gpx, photos, national, local, output = _cli_fixture(tmp_path)
    rendered: dict[str, object] = {}

    def fake_render(**kwargs):
        rendered.update(kwargs)
        Path(kwargs["output"]).write_bytes(b"fake film bytes")

    monkeypatch.setattr(film_module, "render_route_story_film", fake_render)
    # The fake film is not a real video; the delivery encode is not under test here.
    monkeypatch.setattr(
        film_module,
        "encode_final",
        lambda source, destination: Path(destination).write_bytes(Path(source).read_bytes()),
    )
    called = {"n": 0}
    real_run_preflight = film_module.run_preflight

    def counting_run_preflight(*args, **kwargs):
        called["n"] += 1
        return real_run_preflight(*args, **kwargs)

    monkeypatch.setattr(film_module, "run_preflight", counting_run_preflight)

    exit_code = film_module.main(
        _cli_argv(gpx, photos, national, local, output, "--preflight-frames", "0")
    )

    assert exit_code == 0
    assert called["n"] == 0
    assert rendered  # the long render still ran


def test_preflight_only_never_calls_the_film_render(tmp_path: Path, monkeypatch):
    gpx, photos, national, local, output = _cli_fixture(tmp_path)

    def fake_render(**kwargs):
        raise AssertionError("the long film render must not run under --preflight-only")

    monkeypatch.setattr(film_module, "render_route_story_film", fake_render)

    exit_code = film_module.main(
        _cli_argv(gpx, photos, national, local, output, "--preflight-only")
    )

    assert exit_code == 0
    assert not output.exists()
    sheet = output.with_name(output.stem + "-preflight.jpg")
    assert sheet.is_file()


def _cli_argv(gpx, photos, national, local, output, *extra):
    return [
        "--gpx",
        str(gpx),
        "--recorded-on",
        "2026-09-16",
        "--photos-dir",
        str(photos),
        "--national-raster",
        str(national),
        "--national-credit",
        "Test overview",
        "--local-raster",
        str(local),
        "--local-credit",
        "Test local",
        "--local-bounds",
        "-80.1",
        "34.9",
        "-79.9",
        "35.1",
        "--output",
        str(output),
        *extra,
    ]


def _cli_fixture(tmp_path: Path):
    gpx = tmp_path / "day.gpx"
    gpx.write_text(
        """<gpx><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.001" lon="-79.999"><ele>110</ele><time>2026-09-16T12:00:10Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    photos = tmp_path / "photos"
    photos.mkdir()
    national = tmp_path / "national.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    output = tmp_path / "film.mp4"
    return gpx, photos, national, local, output


def test_single_frame_render_does_not_disturb_the_scene_follow_camera_inertia():
    width, height = 64, 36
    map_height = scene_map_height(height)
    segments = [
        [RoutePoint(i * 1.0, 35.0 + i * 0.0004, -80.0 + i * 0.0006, 100.0 + i) for i in range(30)]
    ]
    epoch = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    times = [float(i + 1) for i in range(20)]

    baseline = SceneMapRenderer(
        capture=_FakeSceneCapture(width, map_height), trip_epoch=epoch, max_zoom=16.0
    )
    baseline_frames = list(baseline.render_leg(segments, times, width, height))

    warmed = SceneMapRenderer(
        capture=_FakeSceneCapture(width, map_height), trip_epoch=epoch, max_zoom=16.0
    )
    for at_time in preflight_times(segments[0][0].time_s, segments[0][-1].time_s, 6):
        warmed.render(segments, at_time, width, height)
    warmed_frames = list(warmed.render_leg(segments, times, width, height))

    assert [frame.tobytes() for frame in baseline_frames] == [
        frame.tobytes() for frame in warmed_frames
    ]


class _FakeSceneCapture:
    """Returns a solid frame for any pose; the scene camera never sees a real tile."""

    def __init__(self, width: int, height: int, color: tuple[int, int, int] = (80, 90, 100)):
        self._shape = (height, width, 3)
        self._color = color

    def render(self, pose, light=None):
        return np.full(self._shape, self._color, dtype=np.uint8), []


class _PortraitRailRenderer(_FakeRenderer):
    """A composed portrait frame: the dashboard rail sits ABOVE the bottom safe
    zone, so its top rows overlap the map field, and its value text is white."""

    def __init__(self, *, white_rows: tuple[int, int]):
        super().__init__(ok_color=(80, 90, 100))
        self.white_rows = white_rows

    def render(self, segments, at_time, width, height):
        image = super().render(segments, at_time, width, height)
        top, bottom = self.white_rows
        image.paste((255, 255, 255), (0, top, width, bottom))
        return image


def test_white_dashboard_text_inside_the_portrait_map_field_is_not_page_white(tmp_path: Path):
    from gomoveshift.pipeline.route_story.layout import hud_layout

    width, height = 1080, 1920
    layout = hud_layout(width, height)
    rail_top = layout.dashboard_panel_top
    rail_bottom = height - layout.short_bottom_safe_px
    # The portrait rail is lifted clear of platform chrome, so it starts
    # inside the map field (its white value text alone read as
    # 0.60% page-white on a clean frame and failed every social preflight).
    assert rail_top < scene_map_height(height)
    result = run_preflight(
        _PortraitRailRenderer(white_rows=(rail_top, rail_bottom)),
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=width,
        height=height,
        count=1,
        output=tmp_path / "out-preflight.jpg",
    )
    assert result.passed


def test_page_white_just_above_the_portrait_rail_still_fails(tmp_path: Path):
    from gomoveshift.pipeline.route_story.layout import hud_layout

    width, height = 1080, 1920
    rail_top = hud_layout(width, height).dashboard_panel_top
    with pytest.raises(RoutePreflightError, match="page-white"):
        run_preflight(
            _PortraitRailRenderer(white_rows=(rail_top - 40, rail_top)),
            SEGMENTS,
            start_time=0.0,
            end_time=10.0,
            width=width,
            height=height,
            count=1,
            output=tmp_path / "out-preflight.jpg",
        )


class _WhiteBoxRenderer(_FakeRenderer):
    """A composed frame with one white box inside the map field."""

    def __init__(self, *, box: tuple[int, int, int, int]):
        super().__init__(ok_color=(80, 90, 100))
        self.box = box

    def render(self, segments, at_time, width, height):
        image = super().render(segments, at_time, width, height)
        image.paste((255, 255, 255), self.box)
        return image


def test_a_small_landscape_frame_keeps_the_map_field_as_the_cap_denominator(tmp_path: Path):
    from gomoveshift.pipeline.route_story.layout import hud_layout

    width, height = 640, 360
    map_rows = scene_map_height(height)
    rail_top = hud_layout(width, height).dashboard_panel_top
    # The landscape rail's 72 px floor puts it inside the map field on a
    # small frame (masking those rows shrank the denominator and
    # tipped a 0.48% frame over the 0.50% cap in test_route_story_outputs).
    assert rail_top < map_rows
    white_px = 950
    assert white_px / (map_rows * width) < 0.005 < white_px / (rail_top * width)
    result = run_preflight(
        _WhiteBoxRenderer(box=(0, 100, 475, 102)),
        SEGMENTS,
        start_time=0.0,
        end_time=10.0,
        width=width,
        height=height,
        count=1,
        output=tmp_path / "out-preflight.jpg",
    )
    assert result.passed
