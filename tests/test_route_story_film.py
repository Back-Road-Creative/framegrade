"""End-to-end local Route Story film composition."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import activity as activity_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.continuity import CameraState
from gomoveshift.pipeline.route_story import frame_qa
from gomoveshift.pipeline.route_story import hud as hud_module
from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story import metadata as metadata_module
from gomoveshift.pipeline.route_story.film import (
    MapLabel,
    PhotoMoment,
    _review_motion_clips,
    clean_display_segments,
    crop_black_border,
    drawn_segments,
    film_receipt,
    label_visibility_intervals,
    render_route_story_film,
    load_map_labels,
    route_end_time,
    select_visible_labels,
    map_leg_seconds,
)
from gomoveshift.pipeline.route_story.layers import _render_map
from gomoveshift.pipeline.route_story.hud import (
    _duration,
    _feet,
    _fit_contain,
    _fit_cover,
    _interpolate,
    _pace,
    _percent,
    _photo_frame,
    _speed,
    _to_display_rgb,
    _unlocated_video_card,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster


def test_film_includes_real_satellite_map_and_a_matched_photo(tmp_path: Path):
    overview = tmp_path / "overview.jpg"
    local = tmp_path / "local.jpg"
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (720, 360), "#3c6f83").save(overview)
    Image.new("RGB", (640, 360), "#4e7540").save(local)
    Image.new("RGB", (640, 360), "#aa7744").save(photo)
    output = tmp_path / "film.mp4"

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]],
        national=SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
        local=SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
        photos=[
            PhotoMoment(
                photo,
                MediaAnchor("photo.jpg", "timestamp_interpolated", 5.0, 35.00005, -79.99995),
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

    assert output.is_file()
    assert output.stat().st_size > 0

    receipt = film_receipt(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]],
        national=SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
        local=SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
        photos=[
            PhotoMoment(
                photo,
                MediaAnchor("photo.jpg", "timestamp_interpolated", 5.0, 35.00005, -79.99995),
            )
        ],
    )
    assert receipt["photos"] == [
        {
            "media_id": "photo.jpg",
            "anchor_status": "timestamp_interpolated",
            "placement": "route_moment",
            "interpolation_method": None,
            "rejected_gps": None,
            "required_speed_mps": None,
        }
    ]


def test_a_qa_refused_render_keeps_its_frames_until_the_next_success(tmp_path, monkeypatch):
    overview = tmp_path / "overview.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (720, 360), "#3c6f83").save(overview)
    Image.new("RGB", (640, 360), "#4e7540").save(local)
    output = tmp_path / "film.mp4"
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "staging"))

    def render_kwargs():
        return dict(
            segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]],
            national=SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
            local=SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
            photos=[],
            output=output,
            fps=2,
            width=320,
            height=180,
            intro_seconds=1,
            map_seconds=1,
            photo_seconds=1,
            recap_seconds=1,
        )

    fake_result = frame_qa.FrameQaResult(
        frame=0,
        void_fraction=1.0,
        unloaded_fraction=0.0,
        black_fraction=0.0,
        problems=("frame 0: fake refusal",),
    )
    fake_error = frame_qa.FrameQaError(fake_result, {"pose": "fake"})

    with monkeypatch.context() as patched:
        patched.setattr(film_module, "_report_qa_faults", lambda ledger, out, **kwargs: fake_error)
        with pytest.raises(frame_qa.FrameQaError):
            render_route_story_film(**render_kwargs())

    kept_root = film_module._kept_frames_root(output)
    assert kept_root.is_dir()
    kept_files = [path for path in kept_root.rglob("*") if path.is_file()]
    assert kept_files, "a QA-refused render must keep its composited frames"

    render_route_story_film(**render_kwargs())

    assert output.is_file()
    assert not kept_root.exists(), "a successful render must clear the previous refusal's frames"


def test_film_converts_16_bit_raw_display_values_without_wrapping():
    pixels = np.array([[[0, 32768, 65535]]], dtype=np.uint16)

    converted = _to_display_rgb(pixels)

    assert converted.tolist() == [[[0, 128, 255]]]


def test_display_rgb_scales_float_and_uint16_sources():
    assert _to_display_rgb(np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)).max() == 255
    wide = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    assert _to_display_rgb(wide).tolist() == [[[0, 128, 255]]]


def test_dashboard_numbers_read_in_imperial_units_and_absent_values_dash():
    assert _speed(None) == "—"
    assert _speed(26.8224) == "60.0 mph"
    assert _feet(None) == "—"
    assert _feet(304.8) == "1,000 ft"
    assert _percent(None) == "—"
    assert _percent(-4.25) == "-4.2%"
    assert _duration(3720.0) == "1h 02m"


def test_interpolation_refuses_to_invent_a_missing_endpoint():
    assert _interpolate(None, 10.0, 0.5) is None
    assert _interpolate(10.0, None, 0.5) is None
    assert _interpolate(10.0, 20.0, 0.25) == 12.5


def test_an_unplaced_video_says_so_on_screen():
    card = _unlocated_video_card(320, 180)

    assert card.size == (320, 180)
    # The card is not a flat fill: the disclosure text is drawn onto it.
    assert len(card.convert("L").getcolors(maxcolors=65536) or []) > 1


def test_a_photo_frame_keeps_the_whole_still_inside_the_frame(tmp_path: Path):
    still = Image.new("RGB", (400, 100), "#c08040")
    background = Image.new("RGB", (320, 180), "#101820")

    contained = _fit_contain(still, 320, 180)
    frame = _photo_frame(contained, background, 320, 180)

    assert frame.size == (320, 180)
    assert contained.width <= 320 and contained.height <= 180
    covered = _fit_cover(still, 320, 180)
    assert covered.size == (320, 180)


def test_a_still_that_pillow_cannot_open_falls_back_to_the_raw_decoder(tmp_path: Path, monkeypatch):
    broken = tmp_path / "frame.nef"
    broken.write_bytes(b"not an image")
    decoded = np.zeros((4, 6, 3), dtype=np.uint16)
    decoded[..., 0] = 65535

    class _Decoded:
        rgb = decoded

    monkeypatch.setattr(hud_module, "develop", lambda path: _Decoded())

    still = hud_module._load_still(broken)

    assert still.size == (6, 4)
    assert still.getpixel((0, 0)) == (255, 0, 0)


def test_the_export_command_refuses_a_photo_with_no_zoned_capture_time(tmp_path: Path):
    gpx = tmp_path / "day.gpx"
    gpx.write_text(
        """<gpx><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T12:00:10Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    photos = tmp_path / "photos"
    photos.mkdir()
    Image.new("RGB", (64, 48), "#aa7744").save(photos / "no-exif.jpg")
    national = tmp_path / "national.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)

    argv = [
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
        str(tmp_path / "film.mp4"),
    ]

    with pytest.raises(ValueError, match="no explicitly zoned capture time"):
        film_module.main(argv)

    assert not (tmp_path / "film.mp4").exists()


def test_crop_black_border_removes_only_an_unambiguously_black_edge():
    image = Image.new("RGB", (20, 14), "black")
    image.paste(Image.new("RGB", (16, 10), "#204060"), (2, 2))

    cropped, box = crop_black_border(image)

    assert cropped.size == (16, 10)
    assert box == (2, 2, 18, 12)


def test_clean_display_segments_removes_an_implausible_terminal_jump():
    segments = [
        [
            RoutePoint(0, 35.0, -80.0, 100),
            RoutePoint(30, 35.0001, -80.0001, 101),
            RoutePoint(35, 35.01, -79.99, 102),
        ]
    ]

    cleaned = clean_display_segments(segments, max_speed_mps=8.0, max_gap_s=90.0)

    assert cleaned == [[segments[0][0], segments[0][1]]]


def test_drawn_segments_accepts_a_custom_max_speed_cap():
    segments = [
        [
            RoutePoint(0, 35.0, -80.0, 100),
            RoutePoint(10, 35.001347, -80.0, 101),
        ]
    ]

    default_cleaned, _ = drawn_segments(segments, smooth=False)
    assert default_cleaned == []

    widened_cleaned, _ = drawn_segments(segments, smooth=False, max_speed_mps=20.0)
    assert widened_cleaned == [[segments[0][0], segments[0][1]]]


def test_route_end_time_stops_at_the_last_accepted_edge_not_a_raw_trailing_jump():
    segments = [
        [
            RoutePoint(0, 35.0, -80.0, 100),
            RoutePoint(30, 35.0001, -80.0001, 101),
            RoutePoint(35, 35.01, -79.99, 102),
        ]
    ]

    assert route_end_time(segments) == 30


def test_show_gap_badge_flag_suppresses_the_live_gps_alert_overlay(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (64, 32), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.001, -79.999, 100)]]

    with_badge = _render_map(
        (raster,), segments, at_time=999.0, width=64, height=64, show_dashboard=False
    )
    without_badge = _render_map(
        (raster,),
        segments,
        at_time=999.0,
        width=64,
        height=64,
        show_dashboard=False,
        show_gap_badge=False,
    )

    assert (np.array(with_badge) != np.array(without_badge)).any()


def test_map_labels_are_scale_aware_prioritized_and_deduplicated(tmp_path: Path):
    labels = [
        MapLabel("Trailhead", 35.25, -83.50, "trailhead", max_span_lon=1.0),
        MapLabel("Franklin", 35.18, -83.30, "town", max_span_lon=25.0),
        MapLabel("Franklin", 35.18, -83.30, "town", max_span_lon=25.0),
        MapLabel("Hidden Creek", 35.25, -83.50, "water", max_span_lon=0.5),
    ]

    visible = select_visible_labels(
        labels,
        west=-83.62,
        south=35.20,
        east=-83.37,
        north=35.31,
        span_lon=0.25,
    )

    assert [label.name for label in visible] == ["Trailhead", "Hidden Creek"]


def test_map_label_loader_preserves_kind_and_uses_default_visibility(tmp_path: Path):
    source = tmp_path / "labels.json"
    source.write_text(
        '[{"name":"Franklin","latitude":35.18,"longitude":-83.38,"kind":"town"}]\n',
        encoding="utf-8",
    )

    labels = load_map_labels(source)

    assert labels == (MapLabel("Franklin", 35.18, -83.38, "town"),)


def test_pace_is_minutes_per_mile_and_never_rolls_seconds_over():
    assert _pace(3 * 0.44704) == "20:00/mi"
    assert _pace(6 * 0.44704) == "10:00/mi"
    assert _pace(0.1 * 0.44704) == "stopped"


def test_map_leg_duration_gives_long_legs_time_to_read():
    assert map_leg_seconds(0, 30) >= 3.0
    assert map_leg_seconds(0, 3600) > map_leg_seconds(0, 30)
    assert map_leg_seconds(0, 100_000) <= 14.0


def test_plan_map_legs_is_proportional_to_real_route_speed():
    from gomoveshift.pipeline.route_story.assembly import plan_map_legs

    factor, counts = plan_map_legs([(0, 100), (100, 1100)], fps=30, map_seconds=111)
    assert factor == pytest.approx(1100 / 111)
    ratio = counts[1] / counts[0]
    assert ratio == pytest.approx(10.0, rel=0.15)


def test_plan_map_legs_default_total_matches_the_old_leg_clock_sum():
    from gomoveshift.pipeline.route_story.assembly import plan_map_legs

    leg_bounds = [(0, 30), (30, 3630), (3630, 103630)]
    expected_total = sum(map_leg_seconds(start, end) for start, end in leg_bounds)
    factor, counts = plan_map_legs(leg_bounds, fps=30)
    total_route = sum(end - start for start, end in leg_bounds)
    assert factor == pytest.approx(total_route / expected_total)
    expected_frames = round(expected_total * 30)
    assert abs(sum(counts) - expected_frames) <= len(leg_bounds)


def test_plan_map_legs_override_is_respected():
    from gomoveshift.pipeline.route_story.assembly import plan_map_legs

    factor, counts = plan_map_legs([(0, 200)], fps=10, map_seconds=20)
    assert factor == pytest.approx(10.0)
    assert counts == [200]


def test_plan_map_legs_zero_length_leg_gets_one_frame_no_division_error():
    from gomoveshift.pipeline.route_story.assembly import plan_map_legs

    factor, counts = plan_map_legs([(5.0, 5.0), (0, 100)], fps=30, map_seconds=10)
    assert counts[0] == 1
    assert factor == pytest.approx(10.0)


def test_plan_map_legs_all_zero_length_uses_factor_one():
    from gomoveshift.pipeline.route_story.assembly import plan_map_legs

    factor, counts = plan_map_legs([(5.0, 5.0), (5.0, 5.0)], fps=30)
    assert factor == 1.0
    assert counts == [1, 1]


def test_plan_map_legs_single_leg():
    from gomoveshift.pipeline.route_story.assembly import plan_map_legs

    factor, counts = plan_map_legs([(0, 500)], fps=24, map_seconds=5)
    assert factor == pytest.approx(100.0)
    assert counts == [round(5 * 24)]


def test_plan_map_legs_rejects_a_non_positive_override():
    from gomoveshift.pipeline.route_story.assembly import plan_map_legs

    with pytest.raises(ValueError):
        plan_map_legs([(0, 10)], fps=30, map_seconds=0)


def test_label_bands_allow_cities_at_regional_zoom_and_peaks_at_local_zoom():
    city = MapLabel("Asheville", 35.6, -82.55, "city", max_span_lon=90.0, min_span_lon=2.0)
    peak = MapLabel("Mount Mitchell", 35.76, -82.27, "peak", max_span_lon=1.0)
    regional = select_visible_labels(
        [city, peak], west=-84, south=34, east=-81, north=37, span_lon=3.0
    )
    local = select_visible_labels(
        [city, peak], west=-83, south=35.7, east=-82, north=36.0, span_lon=1.0
    )
    assert [label.name for label in regional] == ["Asheville"]
    assert [label.name for label in local] == ["Mount Mitchell"]


def test_film_receipt_lists_outputs_with_hashes_and_beats(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    output = tmp_path / "film.mp4"
    output.write_bytes(b"not really a video, just bytes to hash")
    import hashlib

    expected_sha = hashlib.sha256(output.read_bytes()).hexdigest()
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]]

    receipt = film_receipt(
        segments=segments,
        national=raster,
        local=raster,
        photos=[],
        output=output,
    )

    assert receipt["outputs"] == [{"path": "film.mp4", "sha256": expected_sha}]
    assert receipt["beats"][0]["kind"] == "start"
    assert receipt["beats"][-1]["kind"] == "finish"


def test_film_receipt_records_the_time_compression_factor_and_override(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [
        [
            RoutePoint(step * 50, 35.0 + step * 0.0005, -80.0 + step * 0.0005, 100)
            for step in range(21)
        ]
    ]

    default_receipt = film_receipt(segments=segments, national=raster, local=raster, photos=[])
    assert default_receipt["map_seconds"] is None
    assert default_receipt["time_compression"] == pytest.approx(1000 / map_leg_seconds(0, 1000))

    override_receipt = film_receipt(
        segments=segments, national=raster, local=raster, photos=[], map_seconds=100.0
    )
    assert override_receipt["map_seconds"] == 100.0
    assert override_receipt["time_compression"] == pytest.approx(10.0)


def test_film_receipt_has_no_outputs_when_output_is_not_supplied(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.01, -79.99, 110)]]

    receipt = film_receipt(segments=segments, national=raster, local=raster, photos=[])

    assert receipt["outputs"] == []


def test_film_receipt_display_filter_defaults_to_no_activity(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.01, -79.99, 110)]]

    receipt = film_receipt(segments=segments, national=raster, local=raster, photos=[])

    assert receipt["display_filter"] == {
        "max_speed_mps": 8.0,
        "max_gap_s": 90.0,
        "activity": None,
        "activity_source": None,
        "units": None,
        "units_source": None,
    }
    assert receipt["mapping_version"] is None


def test_film_receipt_display_filter_records_the_resolved_activity(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.01, -79.99, 110)]]

    receipt = film_receipt(
        segments=segments,
        national=raster,
        local=raster,
        photos=[],
        max_speed_mps=20.0,
        activity="cycling",
        activity_source="explicit",
        units="imperial",
        units_source="explicit",
        mapping_version="2026-09-25.1",
    )

    assert receipt["display_filter"] == {
        "max_speed_mps": 20.0,
        "max_gap_s": 90.0,
        "activity": "cycling",
        "activity_source": "explicit",
        "units": "imperial",
        "units_source": "explicit",
    }
    assert receipt["mapping_version"] == "2026-09-25.1"


def test_film_receipt_pacing_mode_defaults_to_recorded(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.01, -79.99, 110)]]

    receipt = film_receipt(segments=segments, national=raster, local=raster, photos=[])

    assert receipt["pacing_mode"] == "recorded"


def test_film_receipt_presentation_mapping_version_is_a_distinct_constant(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.01, -79.99, 110)]]

    receipt = film_receipt(segments=segments, national=raster, local=raster, photos=[])

    assert isinstance(receipt["presentation_mapping_version"], str)
    assert receipt["presentation_mapping_version"]
    assert film_module.PRESENTATION_MAPPING_VERSION == receipt["presentation_mapping_version"]
    assert film_module.PRESENTATION_MAPPING_VERSION is not activity_module.MAPPING_VERSION


def test_pacing_mode_flag_rejects_an_unrecognized_value():
    with pytest.raises(SystemExit):
        film_module.main(["--pacing-mode", "not-a-real-mode"])


def test_label_visibility_intervals_close_when_a_label_leaves_the_map_viewport(tmp_path: Path):
    raster_path = tmp_path / "local.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    local = SatelliteRaster(raster_path, -80.1, 34.9, -79.9, 35.1, "Test local")
    segments = [
        [
            RoutePoint(0, 35.0, -80.0, 100),
            RoutePoint(10, 35.0005, -80.0005, 100),
            RoutePoint(20, 36.0, -81.0, 100),
        ]
    ]
    labels = [MapLabel("Trailhead", 35.0, -80.0, "trailhead", max_span_lon=1.0)]

    intervals = label_visibility_intervals(segments, labels, local)

    assert intervals == {"Trailhead": [{"start_s": 0.0, "end_s": 10.0}]}


def test_video_coordinate_clock_conflict_requires_review_in_receipt(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    receipt = film_receipt(
        segments=[[RoutePoint(0, 0, 0), RoutePoint(10, 0, 0.01)]],
        national=raster,
        local=raster,
        photos=[],
        source_video=tmp_path / "clip.mp4",
        source_video_anchor=MediaAnchor("clip.mp4", "coordinate_time_conflict", 5, 40, 40),
    )
    assert receipt["source_video"]["placement"] == "review_required"


class _FakeRendererNoPoses:
    name = "fake-scene"


class _FakeRendererOnePose:
    name = "fake-scene"
    fps = 24.0
    rendered_poses = [
        CameraState(center_lat=35.0, center_lon=-80.0, bearing_deg=0.0, zoom=14.0, pitch_deg=60.0)
    ]


class _FakeRendererWithPoses:
    name = "fake-scene"
    fps = 24.0
    rendered_poses = [
        CameraState(center_lat=35.0, center_lon=-80.0, bearing_deg=0.0, zoom=14.0, pitch_deg=60.0),
        CameraState(
            center_lat=35.0001, center_lon=-80.0001, bearing_deg=6.0, zoom=14.2, pitch_deg=60.0
        ),
        CameraState(
            center_lat=35.0002, center_lon=-80.0002, bearing_deg=12.0, zoom=14.4, pitch_deg=60.0
        ),
    ]


def test_film_receipt_omits_motion_and_route_trim_when_not_given(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]]

    receipt = film_receipt(segments=segments, national=raster, local=raster, photos=[])

    assert "motion" not in receipt
    assert "route_trim" not in receipt


def test_film_receipt_omits_motion_when_map_renderer_has_no_rendered_poses(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]]

    no_renderer = film_receipt(
        segments=segments,
        national=raster,
        local=raster,
        photos=[],
        map_renderer=_FakeRendererNoPoses(),
    )
    one_pose = film_receipt(
        segments=segments,
        national=raster,
        local=raster,
        photos=[],
        map_renderer=_FakeRendererOnePose(),
    )

    assert "motion" not in no_renderer
    assert "motion" not in one_pose


def test_film_receipt_computes_motion_from_the_rendered_camera_path(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]]
    renderer = _FakeRendererWithPoses()

    receipt = film_receipt(
        segments=segments,
        national=raster,
        local=raster,
        photos=[],
        map_renderer=renderer,
    )

    assert receipt["motion"]["frame_count"] == len(renderer.rendered_poses)
    assert receipt["motion"]["fps"] == renderer.fps
    assert receipt["motion"]["bearing_speed"]["mean_abs"] == pytest.approx(6.0 * renderer.fps)


def test_film_receipt_carries_motion_and_route_trim_when_given(tmp_path: Path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]]
    motion = {"bearing_speed": {"mean_abs": 1.0}}
    route_trim = {"start_m": 5.0, "end_m": 0.0, "removed_start_points": 1}

    receipt = film_receipt(
        segments=segments,
        national=raster,
        local=raster,
        photos=[],
        motion=motion,
        route_trim=route_trim,
    )

    assert receipt["motion"] == motion
    assert receipt["route_trim"] == route_trim


def _route_story_gpx(tmp_path: Path, *, n: int = 5, step_deg: float = 0.0010) -> Path:
    """5 points ~111m apart, one minute per point -- 2026-09-16."""
    gpx = tmp_path / "day.gpx"
    points_xml = "\n".join(
        f'<trkpt lat="{35.0 + index * step_deg}" lon="-80.0"><ele>100</ele>'
        f"<time>2026-09-16T12:{index:02d}:00Z</time></trkpt>"
        for index in range(n)
    )
    gpx.write_text(
        f"<gpx><trk><trkseg>{points_xml}</trkseg></trk></gpx>",
        encoding="utf-8",
    )
    return gpx


def _base_argv(tmp_path: Path, gpx: Path, extra: list | None = None) -> list:
    photos = tmp_path / "photos"
    photos.mkdir(exist_ok=True)
    national = tmp_path / "national.jpg"
    local = tmp_path / "local.jpg"
    if not national.exists():
        Image.new("RGB", (64, 32), "#3c6f83").save(national)
    if not local.exists():
        Image.new("RGB", (64, 32), "#4e7540").save(local)
    argv = [
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
        str(tmp_path / "film.mp4"),
    ]
    return argv + (extra or [])


def test_duplicate_title_fails_before_the_long_render(tmp_path: Path, monkeypatch):
    gpx = _route_story_gpx(tmp_path)
    metadata_path = tmp_path / "meta.json"

    def _boom(**kwargs):
        raise AssertionError("render should not run")

    def _raise_duplicate(**kwargs):
        raise metadata_module.DuplicateTitleError("title already published: 'Fine Title'")

    monkeypatch.setattr(film_module, "render_route_story_film", _boom)
    monkeypatch.setattr(metadata_module, "preflight_title", _raise_duplicate)

    argv = _base_argv(
        tmp_path, gpx, extra=["--metadata", str(metadata_path), "--title", "Fine Title"]
    )

    with pytest.raises(SystemExit):
        film_module.main(argv)

    assert not metadata_path.exists()


def test_trim_flags_reduce_the_segments_the_render_receives(tmp_path: Path, monkeypatch):
    gpx = _route_story_gpx(tmp_path)
    captured: dict = {}

    class _StopBeforeRenderError(Exception):
        pass

    def _capture_render(**kwargs):
        captured["segments"] = kwargs["segments"]
        raise _StopBeforeRenderError()

    monkeypatch.setattr(film_module, "render_route_story_film", _capture_render)

    untrimmed_argv = _base_argv(tmp_path, gpx)
    with pytest.raises(_StopBeforeRenderError):
        film_module.main(untrimmed_argv)
    untrimmed_points = sum(len(segment) for segment in captured["segments"])

    trimmed_argv = _base_argv(tmp_path, gpx, extra=["--trim-start-m", "150"])
    with pytest.raises(_StopBeforeRenderError):
        film_module.main(trimmed_argv)
    trimmed_points = sum(len(segment) for segment in captured["segments"])

    assert trimmed_points < untrimmed_points


def test_privacy_zone_reduces_the_segments_the_render_receives(tmp_path: Path, monkeypatch):
    gpx = _route_story_gpx(tmp_path)
    captured: dict = {}

    class _StopBeforeRenderError(Exception):
        pass

    def _capture_render(**kwargs):
        captured["segments"] = kwargs["segments"]
        raise _StopBeforeRenderError()

    monkeypatch.setattr(film_module, "render_route_story_film", _capture_render)

    untrimmed_argv = _base_argv(tmp_path, gpx, extra=["--preflight-frames", "0"])
    with pytest.raises(_StopBeforeRenderError):
        film_module.main(untrimmed_argv)
    untrimmed_points = sum(len(segment) for segment in captured["segments"])

    # _route_story_gpx points sit at lat 35.0 + index * 0.001 (~111m apart),
    # lon -80.0. A 50m-radius zone on the third point (35.002) reaches only
    # that one point -- its neighbours are ~111m away. --preflight-frames 0
    # skips real map-tile QA, irrelevant to this test's own concern (that
    # the zoned segments, not the raw ones, reach the render).
    zoned_argv = _base_argv(
        tmp_path,
        gpx,
        extra=["--privacy-zone", "35.002,-80.0,50", "--preflight-frames", "0"],
    )
    with pytest.raises(_StopBeforeRenderError):
        film_module.main(zoned_argv)
    zoned_points = sum(len(segment) for segment in captured["segments"])

    assert zoned_points < untrimmed_points


def test_over_trim_refuses_with_the_route_trim_error_text(tmp_path: Path, capsys):
    gpx = _route_story_gpx(tmp_path)
    argv = _base_argv(tmp_path, gpx, extra=["--trim-start-m", "100000"])

    with pytest.raises(SystemExit):
        film_module.main(argv)

    assert "would remove the entire route" in capsys.readouterr().err


def test_public_receipt_with_zero_trim_refuses_before_writing_anything(
    tmp_path: Path, monkeypatch, capsys
):
    gpx = _route_story_gpx(tmp_path)
    receipt = tmp_path / "receipt.json"
    public_receipt_path = tmp_path / "public-receipt.json"

    def _boom(**kwargs):
        raise AssertionError("nothing should render for a refused --public-receipt run")

    monkeypatch.setattr(film_module, "render_route_story_film", _boom)

    argv = _base_argv(
        tmp_path,
        gpx,
        extra=["--receipt", str(receipt), "--public-receipt", str(public_receipt_path)],
    )

    with pytest.raises(SystemExit):
        film_module.main(argv)

    assert "--public-receipt" in capsys.readouterr().err
    assert not receipt.exists()
    assert not public_receipt_path.exists()
    assert not (tmp_path / "film.mp4").exists()


def test_public_receipt_without_receipt_flag_refuses(tmp_path: Path, monkeypatch, capsys):
    gpx = _route_story_gpx(tmp_path)
    public_receipt_path = tmp_path / "public-receipt.json"

    def _boom(**kwargs):
        raise AssertionError("nothing should render for a refused --public-receipt run")

    monkeypatch.setattr(film_module, "render_route_story_film", _boom)

    argv = _base_argv(
        tmp_path,
        gpx,
        extra=["--trim-start-m", "150", "--public-receipt", str(public_receipt_path)],
    )

    with pytest.raises(SystemExit):
        film_module.main(argv)

    stderr = capsys.readouterr().err
    assert "--public-receipt" in stderr
    assert "--receipt" in stderr
    assert not public_receipt_path.exists()


def test_public_receipt_with_nonzero_trim_writes_sanitized_json_alongside_receipt(
    tmp_path: Path, monkeypatch
):
    gpx = _route_story_gpx(tmp_path)
    receipt = tmp_path / "receipt.json"
    public_receipt_path = tmp_path / "public-receipt.json"

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

    argv = _base_argv(
        tmp_path,
        gpx,
        extra=[
            "--trim-start-m",
            "150",
            "--receipt",
            str(receipt),
            "--public-receipt",
            str(public_receipt_path),
        ],
    )

    exit_code = film_module.main(argv)
    assert exit_code == 0

    private = json.loads(receipt.read_text())
    assert private["route_trim"]["start_m"] == 150.0
    public = json.loads(public_receipt_path.read_text())

    assert public["route_trim"] == private["route_trim"]
    assert public["route_summary"] == private["route_summary"]
    assert public["outputs"] == private["outputs"]
    assert public["national_imagery"] == {"credit": "Test overview"}
    assert public["local_imagery"] == {"credit": "Test local"}
    for block in (public["national_imagery"], public["local_imagery"]):
        assert "bounds" not in block
        assert "sha256" not in block
        assert "dimensions" not in block
    assert "names" not in public["map_labels"]

    # The private receipt is unaffected by requesting a public one alongside it.
    argv_private_only = _base_argv(
        tmp_path,
        gpx,
        extra=["--trim-start-m", "150", "--receipt", str(receipt)],
    )
    exit_code = film_module.main(argv_private_only)
    assert exit_code == 0
    private_only = receipt.read_text()
    assert private_only == json.dumps(private, indent=2, sort_keys=True) + "\n"


def _review_pack_kwargs() -> dict:
    return {
        "opening": Path("opening-sheet.jpg"),
        "stills": Path("stills-sheet.jpg"),
        "placements": Path("placement-table.png"),
        "selection": {"kept": (), "groups": (), "not_considered": ()},
        "preflight_sheet": None,
        "marker_sheet": None,
    }


class TestReviewMotionClips:
    def test_no_op_when_the_pack_was_never_requested(self, tmp_path: Path) -> None:
        poses = [CameraState(0.0, 0.0, float(i), 15.0, 0.0) for i in range(10)]
        calls = []

        _review_motion_clips(
            None,
            None,
            poses,
            10.0,
            tmp_path / "film.mp4",
            runner=lambda cmd, **kwargs: calls.append(cmd),
        )

        assert calls == []
        assert not (tmp_path / "review-pack.json").exists()

    def test_no_op_with_fewer_than_two_rendered_poses(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        calls = []

        _review_motion_clips(
            pack_dir,
            _review_pack_kwargs(),
            [CameraState(0.0, 0.0, 0.0, 15.0, 0.0)],
            10.0,
            tmp_path / "film.mp4",
            runner=lambda cmd, **kwargs: calls.append(cmd),
        )

        assert calls == []
        assert not (pack_dir / "review-pack.json").exists()

    def test_writes_motion_clips_into_the_review_pack(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        fps = 10.0
        poses = [CameraState(0.0, 0.0, bearing, 15.0, 0.0) for bearing in range(200)]
        # A single sharp turn makes one obviously hectic frame.
        poses[100] = CameraState(0.0, 0.0, 90.0, 15.0, 0.0)
        calls = []

        def fake_runner(cmd, **kwargs):
            calls.append((cmd, kwargs))

        _review_motion_clips(
            pack_dir,
            _review_pack_kwargs(),
            poses,
            fps,
            tmp_path / "film.mp4",
            runner=fake_runner,
        )

        assert len(calls) >= 1
        pack = json.loads((pack_dir / "review-pack.json").read_text())
        assert pack["motion_clips"]
        assert pack["motion_clips"][0]["name"] == "motion-clip-1.mp4"

    def test_a_runner_failure_is_recorded_never_raised(self, tmp_path: Path) -> None:
        pack_dir = tmp_path / "pack"
        pack_dir.mkdir()
        fps = 10.0
        poses = [CameraState(0.0, 0.0, bearing, 15.0, 0.0) for bearing in range(200)]
        poses[100] = CameraState(0.0, 0.0, 90.0, 15.0, 0.0)

        def failing_runner(cmd, **kwargs):
            raise RuntimeError("ffmpeg exploded")

        _review_motion_clips(
            pack_dir,
            _review_pack_kwargs(),
            poses,
            fps,
            tmp_path / "film.mp4",
            runner=failing_runner,
        )

        pack = json.loads((pack_dir / "review-pack.json").read_text())
        assert pack["motion_clips"] == [{"motion_clips_error": "ffmpeg exploded"}]


class TestReviewPackFlag:
    def test_review_pack_flag_writes_the_pack_before_the_long_render(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        gpx = _route_story_gpx(tmp_path)

        class _StopBeforeRenderError(Exception):
            pass

        def _boom(**kwargs):
            raise _StopBeforeRenderError()

        monkeypatch.setattr(film_module, "render_route_story_film", _boom)

        argv = _base_argv(tmp_path, gpx, extra=["--review-pack"])
        with pytest.raises(_StopBeforeRenderError):
            film_module.main(argv)

        pack_dir = tmp_path / "film-review-pack"
        assert (pack_dir / "review-pack.json").exists()

    def test_without_the_flag_no_pack_is_written_on_a_full_render(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        gpx = _route_story_gpx(tmp_path)

        class _StopBeforeRenderError(Exception):
            pass

        def _boom(**kwargs):
            raise _StopBeforeRenderError()

        monkeypatch.setattr(film_module, "render_route_story_film", _boom)

        argv = _base_argv(tmp_path, gpx)
        with pytest.raises(_StopBeforeRenderError):
            film_module.main(argv)

        pack_dir = tmp_path / "film-review-pack"
        assert not pack_dir.exists()

    def test_review_pack_without_preflight_frames_or_preflight_only_fails_fast(
        self, tmp_path: Path, capsys
    ) -> None:
        gpx = _route_story_gpx(tmp_path)
        argv = _base_argv(tmp_path, gpx, extra=["--review-pack", "--preflight-frames", "0"])

        with pytest.raises(SystemExit):
            film_module.main(argv)

        assert "--review-pack" in capsys.readouterr().err


@pytest.mark.parametrize("text", ["95,10,100", "10,200,100", "10,10,0"])
def test_privacy_zone_arg_refuses_an_impossible_zone_as_a_usage_error(text):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        film_module._privacy_zone_arg(text)
