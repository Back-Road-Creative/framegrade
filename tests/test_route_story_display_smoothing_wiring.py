"""Wiring: the drawn Route Story line may be display-smoothed, but every
measurement -- distance, pace, the dashboard rail, preflight sampling -- must
stay keyed to the recorded points regardless of the flag."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator, Sequence

from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.dashboard import dashboard_state_at
from gomoveshift.pipeline.route_story.film import drawn_segments, render_route_story_film
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster
from gomoveshift.pipeline.route_story.timeline import Scene, StoryTimeline

METERS_PER_DEGREE_LAT = math.pi / 180 * 6_371_008.8


def _offset_point(
    east_m: float, north_m: float, time_s: float, lat0: float = 40.0, lon0: float = -74.0
):
    lat = lat0 + north_m / METERS_PER_DEGREE_LAT
    lon = lon0 + east_m / (METERS_PER_DEGREE_LAT * math.cos(math.radians(lat0)))
    return RoutePoint(time_s=time_s, latitude=lat, longitude=lon, elevation_m=100.0)


def _jittered_segment(n: int = 11, spacing_m: float = 5.0, jitter_m: float = 4.0):
    """Enough zig-zag for smooth_display_segments to actually move points."""
    points = []
    for i in range(n):
        east = jitter_m if i % 2 == 0 else -jitter_m
        points.append(_offset_point(east, i * spacing_m, i * 2.0))
    return points


class _SpyRenderer:
    """Records every segments list it's handed; draws a flat frame."""

    name = "spy"

    def __init__(self) -> None:
        self.rendered_segments: list[Sequence[Sequence[RoutePoint]]] = []
        self.leg_segments: list[Sequence[Sequence[RoutePoint]]] = []
        self.measured: Sequence[Sequence[RoutePoint]] | None = None

    def measuring(self, segments):
        self.measured = segments
        return self

    def render(self, segments, at_time, width, height) -> Image.Image:
        self.rendered_segments.append(segments)
        return Image.new("RGB", (width, height), "#223344")

    def render_leg(self, segments, times, width, height) -> Iterator[Image.Image]:
        self.leg_segments.append(segments)
        for _ in times:
            yield Image.new("RGB", (width, height), "#223344")


def _rasters(tmp_path: Path):
    overview = tmp_path / "overview.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(overview)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    return (
        SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
        SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
    )


def _render(tmp_path: Path, renderer, *, segments, display_smoothing: bool) -> Path:
    national, local = _rasters(tmp_path)
    output = tmp_path / f"film-{'on' if display_smoothing else 'off'}.mp4"
    render_route_story_film(
        segments=segments,
        national=national,
        local=local,
        photos=[],
        output=output,
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        photo_seconds=1,
        recap_seconds=1,
        map_renderer=renderer,
        display_smoothing=display_smoothing,
    )
    return output


def test_renderer_draws_smoothed_line_while_summary_stays_unsmoothed(tmp_path, monkeypatch):
    segment = _jittered_segment()
    captured_summary_segments = []
    real_summarize_track = film_module.summarize_track

    def spy_summarize_track(segments, **kwargs):
        captured_summary_segments.append(segments)
        return real_summarize_track(segments, **kwargs)

    monkeypatch.setattr(film_module, "summarize_track", spy_summarize_track)

    spy = _SpyRenderer()
    _render(tmp_path, spy, segments=[segment], display_smoothing=True)

    assert captured_summary_segments, "summarize_track was never called"
    measured = captured_summary_segments[0][0]
    drawn = spy.leg_segments[0][0]
    assert [p.latitude for p in measured] == [p.latitude for p in segment]
    assert [p.longitude for p in measured] == [p.longitude for p in segment]
    # The drawn line moved -- smoothing actually changed the geometry.
    assert any(
        d.latitude != m.latitude or d.longitude != m.longitude for d, m in zip(drawn, measured)
    )
    # time_s is preserved exactly on every drawn point either way.
    assert [p.time_s for p in drawn] == [p.time_s for p in segment]


def test_no_display_smoothing_draws_the_cleaned_segments_unchanged(tmp_path):
    segment = _jittered_segment()
    spy = _SpyRenderer()
    _render(tmp_path, spy, segments=[segment], display_smoothing=False)

    drawn = spy.leg_segments[0][0]
    assert [(p.latitude, p.longitude) for p in drawn] == [
        (p.latitude, p.longitude) for p in segment
    ]


def _rail(image: Image.Image) -> bytes:
    """The dashboard rail's pixels -- the bottom band ``hud._draw_dashboard`` paints."""
    width, height = image.size
    return image.crop((0, height - max(72, int(height * 0.135)), width, height)).tobytes()


def test_the_dashboard_rail_is_pixel_identical_with_smoothing_on_and_off(tmp_path):
    """The rail fill is translucent, so numbers can never be re-stamped over a
    wrong rail afterwards -- the renderer itself must measure the recorded points."""
    from gomoveshift.pipeline.route_story.layers import PillowMapRenderer

    segment = _jittered_segment(n=41)
    cleaned, track = drawn_segments([segment], smooth=True)
    assert track is not None
    at_time = segment[-1].time_s
    plain = PillowMapRenderer(rasters=_rasters(tmp_path))

    recorded = plain.render(cleaned, at_time, 640, 360)
    drawn_only = plain.render(track.segments, at_time, 640, 360)
    measuring = plain.measuring(cleaned)
    measured = measuring.render(track.segments, at_time, 640, 360)
    (leg_frame,) = measuring.render_leg(track.segments, [at_time], 640, 360)

    assert _rail(drawn_only) != _rail(recorded)  # the jitter really changes the numbers
    assert _rail(measured) == _rail(recorded)
    assert _rail(leg_frame) == _rail(recorded)


def test_a_renderer_that_cannot_measure_the_recorded_points_is_refused(tmp_path):
    import pytest

    class _DrawOnly:
        name = "draw-only"

        def render(self, segments, at_time, width, height):
            return Image.new("RGB", (width, height))

        def render_leg(self, segments, times, width, height):
            return iter(())

    with pytest.raises(ValueError, match="recorded"):
        _render(tmp_path, _DrawOnly(), segments=[_jittered_segment()], display_smoothing=True)


def test_the_film_tells_its_renderer_to_measure_the_recorded_points(tmp_path):
    segment = _jittered_segment()
    spy = _SpyRenderer()
    _render(tmp_path, spy, segments=[segment], display_smoothing=True)

    assert spy.measured is not None
    assert [(p.latitude, p.longitude) for p in spy.measured[0]] == [
        (p.latitude, p.longitude) for p in segment
    ]


def test_drawn_segments_helper_never_smooths_the_measurement_element(tmp_path):
    segment = _jittered_segment()
    cleaned, smoothed_track = drawn_segments([segment], smooth=True)

    assert [(p.latitude, p.longitude) for p in cleaned[0]] == [
        (p.latitude, p.longitude) for p in segment
    ]
    assert smoothed_track is not None
    assert any(
        (d.latitude, d.longitude) != (c.latitude, c.longitude)
        for d, c in zip(smoothed_track.segments[0], cleaned[0])
    )


def test_drawn_segments_smooth_false_returns_no_track():
    segment = _jittered_segment()
    cleaned, smoothed_track = drawn_segments([segment], smooth=False)
    assert smoothed_track is None
    assert [(p.latitude, p.longitude) for p in cleaned[0]] == [
        (p.latitude, p.longitude) for p in segment
    ]


# ---------------------------------------------------------------------------
# main() wiring: preflight sampling and both receipts.
# ---------------------------------------------------------------------------


def _gpx_text() -> str:
    return """<gpx><trk><trkseg>
    <trkpt lat="35.0000" lon="-80.0000"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
    <trkpt lat="35.0001" lon="-80.0002" ><ele>100</ele><time>2026-09-16T12:00:03Z</time></trkpt>
    <trkpt lat="35.0002" lon="-80.0000" ><ele>100</ele><time>2026-09-16T12:00:06Z</time></trkpt>
    <trkpt lat="35.0003" lon="-80.0002" ><ele>100</ele><time>2026-09-16T12:00:09Z</time></trkpt>
    <trkpt lat="35.0004" lon="-80.0000" ><ele>100</ele><time>2026-09-16T12:00:12Z</time></trkpt>
    <trkpt lat="35.0005" lon="-80.0002" ><ele>100</ele><time>2026-09-16T12:00:15Z</time></trkpt>
    </trkseg></trk></gpx>"""


def _cli_fixture(tmp_path: Path):
    gpx = tmp_path / "day.gpx"
    gpx.write_text(_gpx_text(), encoding="utf-8")
    photos = tmp_path / "photos"
    photos.mkdir()
    national = tmp_path / "national.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    output = tmp_path / "film.mp4"
    return gpx, photos, national, local, output


def _cli_argv(gpx, photos, national, local, output, receipt, *extra):
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
        "--receipt",
        str(receipt),
        *extra,
    ]


def test_preflight_samples_the_same_segments_the_film_draws(tmp_path, monkeypatch):
    gpx, photos, national, local, output = _cli_fixture(tmp_path)
    receipt = tmp_path / "receipt.json"

    monkeypatch.setattr(
        film_module,
        "render_route_story_film",
        lambda **kwargs: Path(kwargs["output"]).write_bytes(b"fake film bytes"),
    )
    monkeypatch.setattr(
        film_module,
        "encode_final",
        lambda source, destination: Path(destination).write_bytes(Path(source).read_bytes()),
    )

    captured_preflight_segments = []
    captured_preflight_measured = []
    real_run_preflight = film_module.run_preflight

    def spy_run_preflight(renderer, segments, **kwargs):
        captured_preflight_segments.append(segments)
        captured_preflight_measured.append(renderer.measured_segments)
        return real_run_preflight(renderer, segments, **kwargs)

    monkeypatch.setattr(film_module, "run_preflight", spy_run_preflight)

    exit_code = film_module.main(
        _cli_argv(gpx, photos, national, local, output, receipt, "--preflight-frames", "4")
    )
    assert exit_code == 0
    assert captured_preflight_segments

    from datetime import date

    from gomoveshift.pipeline.route_story.gpx import select_gpx_day

    day = select_gpx_day(gpx, date.fromisoformat("2026-09-16"))
    expected_cleaned, expected_smoothed = drawn_segments(day.segments, smooth=True)
    expected = expected_smoothed.segments if expected_smoothed is not None else expected_cleaned

    got = captured_preflight_segments[0]
    assert len(got) == len(expected)
    for got_segment, expected_segment in zip(got, expected):
        assert [(p.latitude, p.longitude) for p in got_segment] == [
            (p.latitude, p.longitude) for p in expected_segment
        ]

    # The preflight sheet's dashboard reads the recorded points, like the film's.
    measured = captured_preflight_measured[0]
    assert measured is not None
    assert [[(p.latitude, p.longitude) for p in s] for s in measured] == [
        [(p.latitude, p.longitude) for p in s] for s in expected_cleaned
    ]


def test_receipt_has_display_smoothing_block(tmp_path, monkeypatch):
    gpx, photos, national, local, output = _cli_fixture(tmp_path)
    receipt = tmp_path / "receipt.json"

    monkeypatch.setattr(
        film_module,
        "render_route_story_film",
        lambda **kwargs: Path(kwargs["output"]).write_bytes(b"fake film bytes"),
    )
    monkeypatch.setattr(
        film_module,
        "encode_final",
        lambda source, destination: Path(destination).write_bytes(Path(source).read_bytes()),
    )

    exit_code = film_module.main(_cli_argv(gpx, photos, national, local, output, receipt))
    assert exit_code == 0

    saved = json.loads(receipt.read_text())
    block = saved["display_smoothing"]
    assert block["enabled"] is True
    assert block["recipe"] == "display-smooth-1"
    assert block["window_m"] == film_module.DISPLAY_SMOOTHING_WINDOW_M
    assert block["max_shift_m"] == film_module.DISPLAY_SMOOTHING_MAX_SHIFT_M
    assert isinstance(block["max_shift_m_observed"], float)
    assert isinstance(block["mean_shift_m"], float)
    assert isinstance(block["points_clamped"], int)


def test_preflight_only_receipt_also_has_display_smoothing_block(tmp_path, monkeypatch):
    gpx, photos, national, local, output = _cli_fixture(tmp_path)
    receipt = tmp_path / "receipt.json"

    def fake_render(**kwargs):
        raise AssertionError("the long film render must not run under --preflight-only")

    monkeypatch.setattr(film_module, "render_route_story_film", fake_render)

    exit_code = film_module.main(
        _cli_argv(gpx, photos, national, local, output, receipt, "--preflight-only")
    )
    assert exit_code == 0

    saved = json.loads(receipt.read_text())
    assert saved["display_smoothing"]["enabled"] is True


def test_no_display_smoothing_flag_disables_the_block(tmp_path, monkeypatch):
    gpx, photos, national, local, output = _cli_fixture(tmp_path)
    receipt = tmp_path / "receipt.json"

    monkeypatch.setattr(
        film_module,
        "render_route_story_film",
        lambda **kwargs: Path(kwargs["output"]).write_bytes(b"fake film bytes"),
    )
    monkeypatch.setattr(
        film_module,
        "encode_final",
        lambda source, destination: Path(destination).write_bytes(Path(source).read_bytes()),
    )

    exit_code = film_module.main(
        _cli_argv(gpx, photos, national, local, output, receipt, "--no-display-smoothing")
    )
    assert exit_code == 0

    saved = json.loads(receipt.read_text())
    block = saved["display_smoothing"]
    assert block["enabled"] is False
    assert block["recipe"] is None
    assert block["window_m"] is None
    assert block["max_shift_m"] is None
    assert block["max_shift_m_observed"] is None
