"""A located video plays on the route, in true route-clock order -- not as a
trailing interlude -- and unplaced media never reaches the screen."""

from pathlib import Path

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import PhotoMoment, render_route_story_film
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.privacy import PrivacyResult
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster


def _rasters(tmp_path: Path):
    overview = tmp_path / "overview.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (720, 360), "#3c6f83").save(overview)
    Image.new("RGB", (640, 360), "#4e7540").save(local)
    return (
        SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
        SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
    )


def _photo(tmp_path: Path, name: str, track_time_s: float) -> PhotoMoment:
    source = tmp_path / name
    Image.new("RGB", (640, 360), "#aa7744").save(source)
    return PhotoMoment(
        source, MediaAnchor(name, "timestamp_interpolated", track_time_s, 35.0001, -79.9999)
    )


def _patch_encoders(monkeypatch):
    """Fast, content-free stand-ins for ffmpeg so a structural test runs in ms."""
    concat_calls: dict[str, list[Path]] = {}

    def fake_encode_frames(frames, output, fps):
        Path(output).write_bytes(b"frames:" + Path(frames).name.encode())

    def fake_encode_source_video(source, output, fps, width, height, **_):
        Path(output).write_bytes(b"video:" + Path(source).name.encode())

    def fake_concat(clips, output):
        concat_calls["clips"] = [Path(clip) for clip in clips]
        Path(output).write_bytes(b"final")

    monkeypatch.setattr(film_module, "_encode_frames", fake_encode_frames)
    monkeypatch.setattr(film_module, "_encode_source_video", fake_encode_source_video)
    monkeypatch.setattr(film_module, "_concat", fake_concat)
    return concat_calls


class _RecordingRenderer:
    """A fake ``MapRenderer`` (the ``layers.MapRenderer`` protocol) that records every ``render`` call."""

    name = "fake"

    def __init__(self):
        self.render_calls: list[float] = []

    def render(self, segments, at_time, width, height):
        self.render_calls.append(at_time)
        return Image.new("RGB", (width, height), "#123456")

    def measuring(self, segments):
        # The film asks every renderer to measure the recorded points.
        return self

    def render_leg(self, segments, times, width, height):
        for at_time in times:
            yield self.render(segments, at_time, width, height)


def test_video_arrival_hold_uses_the_active_renderer_not_the_flat_map(tmp_path, monkeypatch):
    """The hold must come from the SAME renderer every map leg uses -- a
    scene renderer's held frame must never pop to the unrelated flat Pillow
    map right before the clip plays."""
    national, local = _rasters(tmp_path)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    output = tmp_path / "film.mp4"
    _patch_encoders(monkeypatch)

    render_map_calls: list[float] = []
    real_render_map = film_module._render_map

    def spy_render_map(*args, **kwargs):
        render_map_calls.append(args[2])  # at_time is the third positional arg
        return real_render_map(*args, **kwargs)

    monkeypatch.setattr(film_module, "_render_map", spy_render_map)

    fake_renderer = _RecordingRenderer()
    target = 10.0

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[],
        output=output,
        source_video=clip,
        source_video_anchor=MediaAnchor(
            "clip.mp4", "embedded_coordinates", target, 35.0001, -79.9999
        ),
        map_renderer=fake_renderer,
        fps=10,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        recap_seconds=1,
    )

    # The leg into the video's moment ends exactly at ``target`` -- one render
    # call, from render_leg. The arrival hold reuses that same frame; it never
    # asks the renderer for a second, separately-rendered frame at the same
    # time (that second call is a camera snap -- see the settle test below).
    assert fake_renderer.render_calls.count(target) == 1
    # The flat Pillow map (module-level ``_render_map``) is never asked for
    # the arrival hold; only the recap (a different, unrelated map view) uses
    # it directly.
    assert target not in render_map_calls


def test_the_settle_hold_reuses_the_legs_own_last_frame_no_snap(tmp_path, monkeypatch):
    """The map must not jump to a freshly, un-rate-limited rendered bearing
    right before a photo cuts in -- the held frame is the SAME image the
    leg's own ``render_leg`` last produced, never a second render() call."""
    national, local = _rasters(tmp_path)
    output = tmp_path / "film.mp4"
    _patch_encoders(monkeypatch)

    class _LegOnlyRenderer(_RecordingRenderer):
        """Records each ``render_leg`` call's own images separately -- the
        film makes several leg calls (into the beat, then on to the end of
        the route), so "the leg's last frame" means the last frame of ITS
        OWN call, never whatever a later, unrelated leg produced."""

        def __init__(self):
            super().__init__()
            self.leg_calls: list[list[Image.Image]] = []

        def render_leg(self, segments, times, width, height):
            images: list[Image.Image] = []
            self.leg_calls.append(images)
            for at_time in times:
                self.render_calls.append(at_time)
                image = Image.new("RGB", (width, height), "#654321")
                images.append(image)
                yield image

    fake_renderer = _LegOnlyRenderer()
    held_images: list[Image.Image] = []
    real_append_hold = film_module._append_hold

    def spy_append_hold(frames, frame, image, fps, seconds, **kwargs):
        held_images.append(image)
        return real_append_hold(frames, frame, image, fps, seconds, **kwargs)

    monkeypatch.setattr(film_module, "_append_hold", spy_append_hold)

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[_photo(tmp_path, "p.jpg", 10.0)],
        output=output,
        map_renderer=fake_renderer,
        fps=10,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        photo_seconds=1,
        recap_seconds=1,
    )

    assert held_images, "the photo beat never held on the settled map frame"
    # Every render_leg call's images are accounted for; a call count mismatch
    # means something outside render_leg asked the renderer for an extra
    # frame -- exactly the bug this fixes.
    assert len(fake_renderer.render_calls) == sum(len(images) for images in fake_renderer.leg_calls)
    first_leg_last_image = fake_renderer.leg_calls[0][-1]
    assert list(held_images[0].getdata()) == list(first_leg_last_image.getdata())


def test_a_placed_video_splits_the_film_around_itself(tmp_path, monkeypatch):
    national, local = _rasters(tmp_path)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    output = tmp_path / "film.mp4"
    concat_calls = _patch_encoders(monkeypatch)

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[],
        output=output,
        source_video=clip,
        source_video_anchor=MediaAnchor(
            "clip.mp4", "embedded_coordinates", 10.0, 35.0001, -79.9999
        ),
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        recap_seconds=1,
    )

    clip_names = [path.name for path in concat_calls["clips"]]
    assert clip_names == ["main-1.mp4", "video.mp4", "main-2.mp4", "recap.mp4"]
    assert not any("interlude" in name for name in clip_names)


def test_no_source_video_keeps_the_old_two_clip_shape(tmp_path, monkeypatch):
    national, local = _rasters(tmp_path)
    output = tmp_path / "film.mp4"
    concat_calls = _patch_encoders(monkeypatch)

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[],
        output=output,
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        recap_seconds=1,
    )

    clip_names = [path.name for path in concat_calls["clips"]]
    assert clip_names == ["main-1.mp4", "recap.mp4"]


def test_a_video_between_two_photos_splits_photos_by_route_time(tmp_path, monkeypatch):
    national, local = _rasters(tmp_path)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"clip")
    output = tmp_path / "film.mp4"
    _patch_encoders(monkeypatch)

    calls: list[tuple[str, str, bool]] = []

    def fake_append_photo(
        frames, frame, photo, fps, seconds, width, height, *, mapped, still_source="developed", **_
    ):
        calls.append((photo.source.name, Path(frames).name, mapped))
        return frame + 1

    monkeypatch.setattr(film_module, "_append_photo", fake_append_photo)

    photo_a = _photo(tmp_path, "a.jpg", 5.0)
    photo_b = _photo(tmp_path, "b.jpg", 15.0)

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[photo_a, photo_b],
        output=output,
        source_video=clip,
        source_video_anchor=MediaAnchor(
            "clip.mp4", "embedded_coordinates", 10.0, 35.0001, -79.9999
        ),
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        photo_seconds=1,
        recap_seconds=1,
    )

    # Every entry is mapped (route-anchored); no unplaced photo ever reaches
    # ``_append_photo``, and photo A lands in the part before the video's
    # split while photo B lands in the part after it.
    assert [call[2] for call in calls] == [True, True]
    assert calls[0][0:2] == ("a.jpg", "main-1-frames")
    assert calls[1][0:2] == ("b.jpg", "main-2-frames")


def test_an_unmapped_photo_never_reaches_append_photo(tmp_path, monkeypatch):
    national, local = _rasters(tmp_path)
    output = tmp_path / "film.mp4"
    _patch_encoders(monkeypatch)

    calls = []
    monkeypatch.setattr(
        film_module,
        "_append_photo",
        lambda frames, frame, photo, *a, **k: calls.append(photo.source.name) or frame + 1,
    )

    unmapped_source = tmp_path / "unmapped.jpg"
    Image.new("RGB", (640, 360), "#aa7744").save(unmapped_source)
    unmapped = PhotoMoment(
        unmapped_source, MediaAnchor("unmapped.jpg", "unanchored_outside_track", None, None, None)
    )

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[unmapped],
        output=output,
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        recap_seconds=1,
    )

    assert calls == []


def test_receipt_carries_rejected_gps_and_required_speed(tmp_path):
    national, local = _rasters(tmp_path)
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]]
    photo = _photo(tmp_path, "photo.jpg", 5.0)
    conflicted = PhotoMoment(
        photo.source,
        MediaAnchor(
            "photo.jpg",
            "timestamp_interpolated",
            5.0,
            35.0001,
            -79.9999,
            interpolation_method="stationary_edge",
            rejected_latitude=40.0,
            rejected_longitude=-70.0,
            required_speed_mps=999.0,
        ),
    )

    receipt = film_module.film_receipt(
        segments=segments, national=national, local=local, photos=[conflicted]
    )

    assert receipt["photos"][0]["interpolation_method"] == "stationary_edge"
    assert receipt["photos"][0]["rejected_gps"] == {"latitude": 40.0, "longitude": -70.0}
    assert receipt["photos"][0]["required_speed_mps"] == 999.0


def test_main_refuses_unplaced_media_before_any_render(monkeypatch, tmp_path):
    gpx = tmp_path / "day.gpx"
    gpx.write_text(
        """<gpx><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T12:00:10Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "orphan.jpg").write_bytes(b"not a real jpeg, evidence is faked below")
    national = tmp_path / "national.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    output = tmp_path / "film.mp4"
    receipt_path = tmp_path / "receipt.json"

    from datetime import datetime, timezone

    monkeypatch.setattr(
        film_module,
        "extract_exif_capture_metadata",
        lambda source: type("M", (), {"capture_utc": datetime(2099, 1, 1, tzinfo=timezone.utc)})(),
    )
    monkeypatch.setattr(film_module, "extract_exif_gps", lambda source: None)
    # The placeholder bytes are unreadable, so the real privacy gate would withhold
    # the orphan before the unplaced check; the gate has its own tests.
    monkeypatch.setattr(
        film_module,
        "apply_privacy_gate",
        lambda photos, *, blur, work_dir: PrivacyResult(
            photos=tuple(photos), attestation={"classes": {}, "blur": blur}
        ),
    )

    render_called = []
    monkeypatch.setattr(
        film_module,
        "render_route_story_film",
        lambda **kwargs: render_called.append(True),
    )
    monkeypatch.setattr(
        film_module,
        "encode_final",
        lambda source, destination: Path(destination).write_bytes(b"x"),
    )

    argv = [
        "--gpx",
        str(gpx),
        "--recorded-on",
        "2026-09-16",
        "--photos-dir",
        str(photos_dir),
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
        str(receipt_path),
    ]

    with pytest.raises(SystemExit):
        film_module.main(argv)
    assert render_called == []

    exit_code = film_module.main(argv + ["--allow-unplaced"])
    assert exit_code == 0
    assert render_called == [True]
    import json

    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["unplaced_media"] == [
        {
            "media_id": "orphan.jpg",
            "anchor_status": "unanchored_outside_track",
            "reason": "no defensible route position",
        }
    ]


def test_staging_root_honors_gms_intermediate_dir_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "custom"))

    root = film_module._staging_root()

    assert root.is_relative_to(tmp_path / "custom")
    assert root.is_dir()


def test_staging_root_default_never_starts_with_tmp(monkeypatch):
    """Pure path resolution only -- must not create anything under the real
    home cache. ``_staging_root`` (which does `mkdir`) is covered by the
    override test above, against ``tmp_path``."""
    monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)

    base = film_module._staging_base()

    assert not str(base).startswith("/tmp")


def test_staging_base_default_follows_the_cache_home(monkeypatch, tmp_path):
    monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
    monkeypatch.setenv("GMS_CACHE_HOME", str(tmp_path / "cache-home"))

    assert film_module._staging_base() == tmp_path / "cache-home" / "route-story-film"

    # The intermediate-dir override still wins over the cache home.
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "custom"))
    assert film_module._staging_base() == tmp_path / "custom" / "route-story-film"
