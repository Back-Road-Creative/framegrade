"""Route Story film: map label validation, terrain overlay, and video evidence."""

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import assembly as assembly_module
from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.assembly import _video_evidence
from gomoveshift.pipeline.route_story.film import (
    MapLabel,
    PhotoMoment,
    load_map_labels,
    map_leg_seconds,
    render_route_story_film,
    select_visible_labels,
)
from gomoveshift.pipeline.route_story.layers import _draw_map_labels, _render_map
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster
from gomoveshift.pipeline.route_story.terrain import TerrainGrid


def _tiny_gray_clip(path: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=gray:s=64x64:d=0.5",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _sine_clip_with_audio(path: Path, duration: float = 1.0) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=gray:s=64x64:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=880:duration={duration}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _silent_frames_clip(tmp_path: Path, name: str, fps: int, frame_count: int) -> Path:
    frames_dir = tmp_path / f"{name}-frames"
    frames_dir.mkdir()
    for index in range(frame_count):
        Image.new("RGB", (64, 64), "#222222").save(frames_dir / f"{index:06d}.jpg")
    output = tmp_path / f"{name}.mp4"
    assembly_module._encode_frames(frames_dir, output, fps)
    return output


def _mean_volume_db(path: Path, start: float, duration: float) -> float:
    result = subprocess.run(
        [
            "ffmpeg",
            "-ss",
            str(start),
            "-t",
            str(duration),
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
    )
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", result.stderr)
    assert match, result.stderr
    return float(match.group(1))


def test_source_clip_audio_lands_in_place_between_silent_legs(tmp_path: Path):
    """The clip's own audio must land exactly at the clip's chapter in the
    assembled timeline, with the silent map legs before and after it staying
    silent -- never boosted, never bled across chapters."""
    fps = 4
    leg_in = _silent_frames_clip(tmp_path, "leg-in", fps, frame_count=4)  # ~1.0s
    source = tmp_path / "source.mp4"
    _sine_clip_with_audio(source, duration=1.0)
    interlude = tmp_path / "interlude.mp4"
    assembly_module._encode_source_video(source, interlude, fps, 64, 64)
    recap = _silent_frames_clip(tmp_path, "recap", fps, frame_count=4)  # ~1.0s

    output = tmp_path / "assembled.mp4"
    assembly_module._concat([leg_in, interlude, recap], output)

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream_types = {stream["codec_type"] for stream in json.loads(probe.stdout)["streams"]}
    assert "audio" in stream_types

    before = _mean_volume_db(output, 0.1, 0.6)
    during = _mean_volume_db(output, 1.2, 0.6)
    after = _mean_volume_db(output, 2.2, 0.6)

    # The source clip's chapter (the middle window) carries the sine tone;
    # the silent legs before and after it (recap's chapter) stay near-silent.
    assert during > before + 20
    assert during > after + 20


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "  "}, "name"),
        ({"latitude": 200.0}, "geographic"),
        ({"max_span_lon": 0.0}, "visibility span must be positive"),
        ({"min_span_lon": -1.0}, "minimum visibility span must be positive"),
        ({"max_span_lon": 1.0, "min_span_lon": 2.0}, "inverted"),
    ],
)
def test_map_label_refuses_bad_input(kwargs, match):
    base = {"name": "Peak", "latitude": 35.0, "longitude": -80.0}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        MapLabel(**base)


@pytest.mark.parametrize(
    "payload, match",
    [({"name": "Franklin"}, "JSON list"), (["Franklin"], "object")],
)
def test_load_map_labels_refuses_a_malformed_manifest(tmp_path, payload, match):
    source = tmp_path / "labels.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_map_labels(source)


def test_select_visible_labels_returns_nothing_for_a_degenerate_viewport():
    labels = (MapLabel("Franklin", 35.18, -83.30, "town"),)

    assert select_visible_labels(labels, west=-84, south=35, east=-83, north=36, span_lon=0) == ()


@pytest.mark.parametrize("kwargs", [{}, {"minimum": 0.0}, {"minimum": 5.0, "maximum": 3.0}])
def test_map_leg_seconds_refuses_invalid_bounds(kwargs):
    start, end = (10.0, 5.0) if not kwargs else (0.0, 10.0)
    with pytest.raises(ValueError, match="ordered|invalid"):
        map_leg_seconds(start, end, **kwargs)


def test_draw_map_labels_skips_off_map_and_overlapping_names_and_sizes_town_dots():
    image = Image.new("RGB", (320, 180), "#071219")
    labels = [
        MapLabel("TownCenter", 1.0, 1.0, "town", max_span_lon=5.0),
        MapLabel("NearPoi", 1.333, 0.6875, "poi", max_span_lon=5.0),
        MapLabel("FarPoi", 1.333, 0.625, "poi", max_span_lon=5.0),
        MapLabel("OffMapPoi", 1.95, 1.0, "poi", max_span_lon=5.0),
    ]

    _draw_map_labels(image, labels, west=0.0, north=2.0, span_lon=2.0, span_lat=2.0, map_height=180)

    pixels = np.asarray(image)
    background = np.asarray(Image.new("RGB", (320, 180), "#071219"))
    assert not np.array_equal(pixels, background)


def test_render_map_falls_back_to_the_nearest_point_in_a_display_gap_and_applies_terrain(tmp_path):
    raster_path = tmp_path / "raster.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [
        [RoutePoint(0, 0.0, 0.0, 10), RoutePoint(10, 0.0, 0.01, 15)],
        [RoutePoint(100, 0.0, 0.5, 20), RoutePoint(110, 0.0, 0.51, 25)],
    ]
    terrain = TerrainGrid(-1.0, -1.0, 1.0, 1.0, np.linspace(0, 100, 16).reshape(4, 4))

    image = _render_map([raster], segments, 50.0, 320, 180, terrain=terrain)

    assert image.size == (320, 180)


@pytest.mark.parametrize(
    "tags, expected",
    [
        (
            {"creation_time": "2026-09-16T12:00:10Z", "location": "+40.1234-075.5678/"},
            (10.0, 40.1234, -75.5678),
        ),
        ({}, (None, None, None)),
    ],
)
def test_video_evidence_reads_optional_creation_time_and_gps_tags(monkeypatch, tags, expected):
    payload = json.dumps({"format": {"tags": tags}})

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=payload, stderr="")

    monkeypatch.setattr(assembly_module.subprocess, "run", fake_run)

    evidence = _video_evidence(
        Path("clip.mov"), datetime(2026, 9, 16, 12, 0, 0, tzinfo=timezone.utc)
    )

    assert (evidence.captured_at_s, evidence.latitude, evidence.longitude) == expected


@pytest.mark.parametrize(
    "anchor_status",
    ["embedded_coordinates", "coordinate_time_conflict", None],
)
def test_source_video_arrival_leg_and_unlocated_card_variants(tmp_path: Path, anchor_status):
    overview = tmp_path / "overview.jpg"
    local = tmp_path / "local.jpg"
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (720, 360), "#3c6f83").save(overview)
    Image.new("RGB", (640, 360), "#4e7540").save(local)
    Image.new("RGB", (640, 360), "#aa7744").save(photo)
    clip = tmp_path / "clip.mp4"
    _tiny_gray_clip(clip)
    output = tmp_path / "film.mp4"

    source_video_anchor = (
        MediaAnchor("clip.mp4", anchor_status, 15.0, 35.0002, -79.9998)
        if anchor_status is not None
        else None
    )

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
        local=SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
        photos=[
            PhotoMoment(
                photo,
                MediaAnchor("photo.jpg", "timestamp_interpolated", 5.0, 35.00005, -79.99995),
            )
        ],
        output=output,
        source_video=clip,
        source_video_anchor=source_video_anchor,
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


def test_main_happy_path_renders_and_writes_a_receipt(tmp_path: Path, monkeypatch):
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
    national = tmp_path / "national.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    output = tmp_path / "film.mp4"
    receipt = tmp_path / "receipt.json"

    rendered: dict[str, object] = {}

    def fake_render(**kwargs):
        rendered.update(kwargs)
        Path(kwargs["output"]).write_bytes(b"fake film bytes")

    monkeypatch.setattr(film_module, "render_route_story_film", fake_render)

    encoded: dict[str, Path] = {}

    def fake_encode_final(source: Path, destination: Path) -> Path:
        encoded["source"], encoded["destination"] = Path(source), Path(destination)
        Path(destination).write_bytes(b"delivery:" + Path(source).read_bytes())
        return Path(destination)

    monkeypatch.setattr(film_module, "encode_final", fake_encode_final)

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
        str(output),
        "--receipt",
        str(receipt),
    ]

    exit_code = film_module.main(argv)

    assert exit_code == 0
    assert rendered["photos"] == []
    assert receipt.is_file()
    # The product is 1080p30; a run that names no size must not quietly ship 720p24.
    assert (rendered["width"], rendered["height"], rendered["fps"]) == (1920, 1080, 30)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["export"]["file"] == output.name
    # The film is assembled to a scratch file beside the output, then the
    # delivery encode (crf 18, bt709, loudnorm) writes the named output; the
    # receipt hashes what was delivered, and no scratch file is left behind.
    assert rendered["output"] != output
    assert Path(rendered["output"]).parent == output.parent
    assert encoded == {"source": Path(rendered["output"]), "destination": output}
    assert output.read_bytes() == b"delivery:fake film bytes"
    assert not Path(rendered["output"]).exists()
    assert payload["outputs"][0]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_main_passes_the_map_seconds_override_through_to_render_and_receipt(
    tmp_path: Path, monkeypatch
):
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
    national = tmp_path / "national.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    output = tmp_path / "film.mp4"

    rendered: dict[str, object] = {}
    receipted: dict[str, object] = {}

    def fake_render(**kwargs):
        rendered.update(kwargs)
        Path(kwargs["output"]).write_bytes(b"fake film bytes")

    def fake_receipt(**kwargs):
        receipted.update(kwargs)
        return {"export": {"file": Path(kwargs["output"]).name}}

    monkeypatch.setattr(film_module, "render_route_story_film", fake_render)
    monkeypatch.setattr(film_module, "film_receipt", fake_receipt)
    monkeypatch.setattr(
        film_module,
        "encode_final",
        lambda source, destination: Path(destination).write_bytes(Path(source).read_bytes()),
    )

    receipt = tmp_path / "receipt.json"
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
        str(output),
        "--receipt",
        str(receipt),
        "--map-seconds",
        "42.0",
    ]

    exit_code = film_module.main(argv)

    assert exit_code == 0
    assert rendered["map_seconds"] == 42.0
    assert receipted["map_seconds"] == 42.0


def test_main_rejects_a_non_positive_map_seconds_override():
    with pytest.raises(SystemExit):
        film_module.main(
            [
                "--gpx",
                "/nonexistent.gpx",
                "--recorded-on",
                "2026-09-16",
                "--photos-dir",
                "/nonexistent",
                "--national-raster",
                "/nonexistent.jpg",
                "--national-credit",
                "x",
                "--local-raster",
                "/nonexistent.jpg",
                "--local-credit",
                "x",
                "--local-bounds",
                "-80.1",
                "34.9",
                "-79.9",
                "35.1",
                "--output",
                "/nonexistent-out.mp4",
                "--map-seconds",
                "0",
            ]
        )
