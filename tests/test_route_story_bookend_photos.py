"""A same-day photo shot just outside the GPX recording is a "bookend": it
plays in the film with no claimed route position, instead of being refused
or silently dropped."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import (
    BOOKEND_AFTER_CAPTION,
    BOOKEND_BEFORE_CAPTION,
    BOOKEND_MAX_OUTSIDE_S,
    PhotoMoment,
    classify_bookend,
    film_receipt,
    render_route_story_film,
)
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.privacy import PrivacyResult
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster
from gomoveshift.pipeline.route_story.select import GroupResult, PhotoScore, SelectionResult

TRACK_START_S = 0.0
TRACK_END_S = 7200.0


# --- classify_bookend (pure) ---


def _photo(
    tmp_path: Path, name: str, *, status: str, captured_at_exif_s: float | None
) -> PhotoMoment:
    source = tmp_path / name
    Image.new("RGB", (16, 16), "#334455").save(source)
    anchor = MediaAnchor(name, status, None, None, None)
    return PhotoMoment(source, anchor, captured_at_exif_s=captured_at_exif_s)


def test_just_inside_before_limit_is_a_before_bookend(tmp_path):
    photo = _photo(
        tmp_path,
        "a.jpg",
        status="unanchored_outside_track",
        captured_at_exif_s=TRACK_START_S - BOOKEND_MAX_OUTSIDE_S,
    )
    assert classify_bookend(photo, TRACK_START_S, TRACK_END_S) == "before_recording"


def test_just_outside_before_limit_is_not_a_bookend(tmp_path):
    photo = _photo(
        tmp_path,
        "a.jpg",
        status="unanchored_outside_track",
        captured_at_exif_s=TRACK_START_S - BOOKEND_MAX_OUTSIDE_S - 1,
    )
    assert classify_bookend(photo, TRACK_START_S, TRACK_END_S) is None


def test_just_inside_after_limit_is_an_after_bookend(tmp_path):
    photo = _photo(
        tmp_path,
        "a.jpg",
        status="unanchored_outside_track",
        captured_at_exif_s=TRACK_END_S + BOOKEND_MAX_OUTSIDE_S,
    )
    assert classify_bookend(photo, TRACK_START_S, TRACK_END_S) == "after_recording"


def test_just_outside_after_limit_is_not_a_bookend(tmp_path):
    photo = _photo(
        tmp_path,
        "a.jpg",
        status="unanchored_outside_track",
        captured_at_exif_s=TRACK_END_S + BOOKEND_MAX_OUTSIDE_S + 1,
    )
    assert classify_bookend(photo, TRACK_START_S, TRACK_END_S) is None


def test_exactly_at_track_start_or_end_is_not_a_bookend(tmp_path):
    at_start = _photo(
        tmp_path, "a.jpg", status="unanchored_outside_track", captured_at_exif_s=TRACK_START_S
    )
    at_end = _photo(
        tmp_path, "b.jpg", status="unanchored_outside_track", captured_at_exif_s=TRACK_END_S
    )
    assert classify_bookend(at_start, TRACK_START_S, TRACK_END_S) is None
    assert classify_bookend(at_end, TRACK_START_S, TRACK_END_S) is None


def test_no_capture_time_is_not_a_bookend(tmp_path):
    photo = _photo(tmp_path, "a.jpg", status="unanchored_outside_track", captured_at_exif_s=None)
    assert classify_bookend(photo, TRACK_START_S, TRACK_END_S) is None


def test_other_unplaced_status_is_never_a_bookend_even_if_outside_window(tmp_path):
    """Only ``unanchored_outside_track`` counts -- a gap inside the recording
    or a coordinate conflict stays ordinary unplaced media."""
    photo = _photo(
        tmp_path,
        "a.jpg",
        status="unanchored_gap",
        captured_at_exif_s=TRACK_START_S - 60.0,
    )
    assert classify_bookend(photo, TRACK_START_S, TRACK_END_S) is None


# --- render_route_story_film sequencing ---


def _rasters(tmp_path: Path):
    overview = tmp_path / "overview.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (720, 360), "#3c6f83").save(overview)
    Image.new("RGB", (640, 360), "#4e7540").save(local)
    return (
        SatelliteRaster(overview, -180, -90, 180, 90, "Test overview"),
        SatelliteRaster(local, -80.1, 34.9, -79.9, 35.1, "Test local"),
    )


class _RecordingRenderer:
    """A fake ``MapRenderer`` that records every leg render as one event."""

    name = "fake"

    def render(self, segments, at_time, width, height):
        return Image.new("RGB", (width, height), "#123456")

    def measuring(self, segments):
        # The film asks every renderer to measure the recorded points.
        return self

    def render_leg(self, segments, times, width, height):
        for at_time in times:
            yield self.render(segments, at_time, width, height)


def _patch_encoders(monkeypatch):
    monkeypatch.setattr(
        film_module,
        "_encode_frames",
        lambda frames, output, fps: Path(output).write_bytes(b"frames"),
    )
    monkeypatch.setattr(
        film_module,
        "_encode_source_video",
        lambda source, output, fps, width, height: Path(output).write_bytes(b"video"),
    )
    monkeypatch.setattr(
        film_module, "_concat", lambda clips, output: Path(output).write_bytes(b"final")
    )


def test_before_bookend_renders_between_opening_and_first_leg(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "staging"))
    national, local = _rasters(tmp_path)
    output = tmp_path / "film.mp4"
    _patch_encoders(monkeypatch)

    events: list[tuple] = []
    monkeypatch.setattr(
        film_module,
        "_append_photo",
        lambda frames, frame, photo, *a, mapped, caption=None, **k: (
            events.append(("photo", photo.source.name, mapped, caption)) or frame + 1
        ),
    )
    real_render_leg = _RecordingRenderer.render_leg

    def spy_render_leg(self, segments, times, width, height):
        events.append(("leg",))
        return real_render_leg(self, segments, times, width, height)

    monkeypatch.setattr(_RecordingRenderer, "render_leg", spy_render_leg)

    before = _photo(
        tmp_path, "before.jpg", status="unanchored_outside_track", captured_at_exif_s=-10.0
    )

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[],
        output=output,
        map_renderer=_RecordingRenderer(),
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        photo_seconds=1,
        recap_seconds=1,
        before_bookends=[before],
    )

    kinds = [event[0] for event in events]
    assert kinds == ["photo", "leg"]
    assert events[0][1] == "before.jpg"
    assert events[0][2] is False
    assert events[0][3] == film_module.BOOKEND_BEFORE_CAPTION


def test_after_bookend_renders_between_last_leg_and_recap(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "staging"))
    national, local = _rasters(tmp_path)
    output = tmp_path / "film.mp4"
    _patch_encoders(monkeypatch)

    events: list[tuple] = []
    monkeypatch.setattr(
        film_module,
        "_append_photo",
        lambda frames, frame, photo, *a, mapped, caption=None, **k: (
            events.append(("photo", photo.source.name, mapped, caption)) or frame + 1
        ),
    )
    real_render_leg = _RecordingRenderer.render_leg

    def spy_render_leg(self, segments, times, width, height):
        events.append(("leg",))
        return real_render_leg(self, segments, times, width, height)

    monkeypatch.setattr(_RecordingRenderer, "render_leg", spy_render_leg)

    monkeypatch.setattr(
        film_module,
        "_render_recap_map",
        lambda *a, **k: events.append(("recap",)) or Image.new("RGB", (320, 180), "#000000"),
    )

    after = _photo(
        tmp_path, "after.jpg", status="unanchored_outside_track", captured_at_exif_s=99999.0
    )

    render_route_story_film(
        segments=[[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(20, 35.0002, -79.9998, 110)]],
        national=national,
        local=local,
        photos=[],
        output=output,
        map_renderer=_RecordingRenderer(),
        fps=2,
        width=320,
        height=180,
        intro_seconds=1,
        map_seconds=1,
        photo_seconds=1,
        recap_seconds=1,
        after_bookends=[after],
    )

    kinds = [event[0] for event in events]
    # One leg, then the bookend photo, then one "recap" event per recap
    # frame -- the photo must land strictly between the leg and the first
    # recap frame.
    assert kinds[0] == "leg"
    assert kinds[1] == "photo"
    assert all(kind == "recap" for kind in kinds[2:])
    assert events[1][1] == "after.jpg"
    assert events[1][2] is False
    assert events[1][3] == film_module.BOOKEND_AFTER_CAPTION


# --- caption wiring (assembly._append_photo) ---


def test_before_bookend_still_card_carries_its_own_caption_not_the_unplaced_one(
    tmp_path, monkeypatch
):
    from gomoveshift.pipeline.route_story import assembly as assembly_module
    from gomoveshift.pipeline.route_story.hud import UNPLACED_PHOTO_CAPTION

    drawn: list[str] = []
    monkeypatch.setattr(
        assembly_module, "_draw_photo_label", lambda image, label: drawn.append(label)
    )
    frames = tmp_path / "frames"
    frames.mkdir()
    photo = _photo(
        tmp_path, "before.jpg", status="unanchored_outside_track", captured_at_exif_s=-10.0
    )

    assembly_module._append_photo(
        frames, 0, photo, 2, 1.0, 320, 180, mapped=False, caption=BOOKEND_BEFORE_CAPTION
    )

    assert drawn == [BOOKEND_BEFORE_CAPTION] * 2
    assert UNPLACED_PHOTO_CAPTION not in drawn


def test_after_bookend_still_card_carries_its_own_caption_not_the_unplaced_one(
    tmp_path, monkeypatch
):
    from gomoveshift.pipeline.route_story import assembly as assembly_module
    from gomoveshift.pipeline.route_story.hud import UNPLACED_PHOTO_CAPTION

    drawn: list[str] = []
    monkeypatch.setattr(
        assembly_module, "_draw_photo_label", lambda image, label: drawn.append(label)
    )
    frames = tmp_path / "frames"
    frames.mkdir()
    photo = _photo(
        tmp_path, "after.jpg", status="unanchored_outside_track", captured_at_exif_s=99999.0
    )

    assembly_module._append_photo(
        frames, 0, photo, 2, 1.0, 320, 180, mapped=False, caption=BOOKEND_AFTER_CAPTION
    )

    assert drawn == [BOOKEND_AFTER_CAPTION] * 2
    assert UNPLACED_PHOTO_CAPTION not in drawn


def test_default_caption_is_none_and_unplaced_photo_keeps_the_old_caption(tmp_path, monkeypatch):
    """An ordinary unplaced photo (caption left unset) is untouched by this
    change -- see also tests/test_route_story_photo_caption.py, which pins
    the same default end to end."""
    from gomoveshift.pipeline.route_story import assembly as assembly_module
    from gomoveshift.pipeline.route_story.hud import UNPLACED_PHOTO_CAPTION

    drawn: list[str] = []
    monkeypatch.setattr(
        assembly_module, "_draw_photo_label", lambda image, label: drawn.append(label)
    )
    frames = tmp_path / "frames"
    frames.mkdir()
    photo = _photo(tmp_path, "orphan.jpg", status="unanchored_gap", captured_at_exif_s=None)

    assembly_module._append_photo(frames, 0, photo, 2, 1.0, 320, 180, mapped=False)

    assert drawn == [UNPLACED_PHOTO_CAPTION] * 2


# --- receipt fields ---


def test_receipt_records_bookend_placement_and_seconds_outside(tmp_path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]]
    photo = _photo(
        tmp_path, "before.jpg", status="unanchored_outside_track", captured_at_exif_s=-3600.0
    )

    receipt = film_receipt(
        segments=segments,
        national=raster,
        local=raster,
        photos=[photo],
        bookend_info={"before.jpg": ("before_recording", 3600.0)},
        bookend_play_order=["before.jpg"],
    )

    assert receipt["photos"] == [
        {
            "media_id": "before.jpg",
            "anchor_status": "unanchored_outside_track",
            "placement": "before_recording",
            "interpolation_method": None,
            "rejected_gps": None,
            "required_speed_mps": None,
            "seconds_outside_track": 3600.0,
        }
    ]
    assert receipt["bookend_photos"] == ["before.jpg"]


def test_receipt_bookend_photos_defaults_to_empty(tmp_path):
    raster_path = tmp_path / "map.jpg"
    Image.new("RGB", (32, 16), "#446677").save(raster_path)
    raster = SatelliteRaster(raster_path, -180, -90, 180, 90, "test")
    segments = [[RoutePoint(0, 35.0, -80.0, 100), RoutePoint(10, 35.0001, -79.9999, 110)]]

    receipt = film_receipt(segments=segments, national=raster, local=raster, photos=[])

    assert receipt["bookend_photos"] == []


# --- main() integration ---


def _main_rasters(tmp_path: Path):
    national = tmp_path / "national.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    return national, local


def _main_gpx(tmp_path: Path) -> Path:
    gpx = tmp_path / "day.gpx"
    gpx.write_text(
        """<gpx><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T14:00:00Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    return gpx


def _main_argv(tmp_path: Path, *, photos_dir: Path, output: Path, receipt: Path | None = None):
    national, local = _main_rasters(tmp_path)
    argv = [
        "--gpx",
        str(_main_gpx(tmp_path)),
        "--recorded-on",
        "2026-09-16",
        "--photos-dir",
        str(photos_dir),
        "--national-raster",
        str(national),
        "--national-credit",
        "Test national",
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
    ]
    if receipt is not None:
        argv += ["--receipt", str(receipt)]
    return argv


def _patch_main_common(
    monkeypatch, *, capture_utc: dict[str, datetime], anchors: dict[str, MediaAnchor]
):
    """Skip real EXIF/anchoring; hand back a per-file capture time and anchor.

    The fixtures are placeholder bytes, not images, so the privacy gate is
    stubbed to a pass-through (it has its own tests with real JPEGs in
    ``test_route_story_privacy_gate.py``). Returns the list of gate calls so a
    test can assert ``main`` routed every still through it."""
    gate_calls: list[dict] = []

    def _gate(photos, *, blur, work_dir):
        gate_calls.append({"names": [p.source.name for p in photos], "blur": blur})
        return PrivacyResult(photos=tuple(photos), attestation={"classes": {}, "blur": blur})

    monkeypatch.setattr(film_module, "apply_privacy_gate", _gate)
    monkeypatch.setattr(
        film_module,
        "extract_exif_capture_metadata",
        lambda source: type("M", (), {"capture_utc": capture_utc[source.name]})(),
    )
    monkeypatch.setattr(film_module, "extract_exif_gps", lambda source: None)
    monkeypatch.setattr(
        film_module,
        "anchor_media_to_segments",
        lambda evidence, segments, **kwargs: anchors[evidence.media_id],
    )
    monkeypatch.setattr(
        film_module, "encode_final", lambda source, destination: Path(destination).write_bytes(b"x")
    )
    monkeypatch.setattr(film_module, "render_route_story_film", lambda **kwargs: None)
    return gate_calls


def _fake_selection_result(kept: tuple[str, ...], all_names: tuple[str, ...]) -> SelectionResult:
    losers = tuple(name for name in all_names if name not in kept)
    score = PhotoScore(name="x", sharpness=1.0, exposure_badness=0.0, color_vibrancy=1.0)
    group = GroupResult(
        members=all_names,
        winner=kept[0] if kept else all_names[0],
        kept=kept,
        losers=losers,
        scores={name: score for name in all_names},
        reason="test group",
    )
    return SelectionResult(groups=(group,), kept=kept)


def test_main_never_refuses_when_only_off_track_photos_are_bookends(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    names = ["before_ok.jpg", "after_ok.jpg", "placed.jpg"]
    for name in names:
        (photos_dir / name).write_bytes(b"fake")

    capture_utc = {
        # track: 2026-09-16 12:00:00Z (0s) .. 14:00:00Z (7200s)
        "before_ok.jpg": datetime(2026, 9, 16, 11, 0, 0, tzinfo=timezone.utc),  # -3600s
        "after_ok.jpg": datetime(2026, 9, 16, 15, 0, 0, tzinfo=timezone.utc),  # 10800s
        "placed.jpg": datetime(2026, 9, 16, 13, 0, 0, tzinfo=timezone.utc),  # 3600s
    }
    anchors = {
        "before_ok.jpg": MediaAnchor("before_ok.jpg", "unanchored_outside_track", None, None, None),
        "after_ok.jpg": MediaAnchor("after_ok.jpg", "unanchored_outside_track", None, None, None),
        "placed.jpg": MediaAnchor(
            "placed.jpg", "timestamp_interpolated", 3600.0, 35.0001, -79.9999
        ),
    }
    _patch_main_common(monkeypatch, capture_utc=capture_utc, anchors=anchors)
    monkeypatch.setattr(
        film_module,
        "select_kept_photos",
        lambda photos, *, still_source, options: (
            list(photos),
            _fake_selection_result(
                tuple(p.source.name for p in photos), tuple(p.source.name for p in photos)
            ),
        ),
    )

    output = tmp_path / "film.mp4"
    receipt = tmp_path / "receipt.json"
    # --pre-trip-photos: without it, before_ok.jpg would be excluded by
    # default (see test_before_bookend_excluded_by_default_* below).
    argv = _main_argv(tmp_path, photos_dir=photos_dir, output=output, receipt=receipt) + [
        "--pre-trip-photos"
    ]

    exit_code = film_module.main(argv)
    assert exit_code == 0

    import json

    saved = json.loads(receipt.read_text())
    assert saved["unplaced_media"] == []
    assert set(saved["bookend_photos"]) == {"before_ok.jpg", "after_ok.jpg"}
    assert saved["excluded_pretrip_photos"] == []


def test_main_still_refuses_a_photo_7_hours_outside(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "way_off.jpg").write_bytes(b"fake")

    capture_utc = {
        # 7 hours before the track start (12:00:00Z)
        "way_off.jpg": datetime(2026, 9, 16, 5, 0, 0, tzinfo=timezone.utc),
    }
    anchors = {
        "way_off.jpg": MediaAnchor("way_off.jpg", "unanchored_outside_track", None, None, None),
    }
    _patch_main_common(monkeypatch, capture_utc=capture_utc, anchors=anchors)

    output = tmp_path / "film.mp4"
    argv = _main_argv(tmp_path, photos_dir=photos_dir, output=output)

    with pytest.raises(SystemExit):
        film_module.main(argv)


def test_before_bookend_excluded_by_default_renders_without_allow_unplaced(tmp_path, monkeypatch):
    """A driving film shouldn't open with
    pre-trip photos. Leaving one out is a deliberate, receipted exclusion,
    not unplaced media -- the default run proceeds with no flag, and
    --allow-unplaced (which would also hide a genuinely unplaced photo) is
    never needed to get there."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "before_ok.jpg").write_bytes(b"fake")

    capture_utc = {
        "before_ok.jpg": datetime(2026, 9, 16, 11, 0, 0, tzinfo=timezone.utc),  # -3600s
    }
    anchors = {
        "before_ok.jpg": MediaAnchor("before_ok.jpg", "unanchored_outside_track", None, None, None),
    }
    _patch_main_common(monkeypatch, capture_utc=capture_utc, anchors=anchors)

    output = tmp_path / "film.mp4"
    receipt = tmp_path / "receipt.json"
    argv = _main_argv(tmp_path, photos_dir=photos_dir, output=output, receipt=receipt)

    exit_code = film_module.main(argv)
    assert exit_code == 0

    import json

    saved = json.loads(receipt.read_text())
    assert saved["excluded_pretrip_photos"] == ["before_ok.jpg"]
    assert saved["bookend_photos"] == []
    assert saved["unplaced_media"] == []


def test_pre_trip_photos_flag_restores_it_as_a_bookend(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "before_ok.jpg").write_bytes(b"fake")

    capture_utc = {
        "before_ok.jpg": datetime(2026, 9, 16, 11, 0, 0, tzinfo=timezone.utc),  # -3600s
    }
    anchors = {
        "before_ok.jpg": MediaAnchor("before_ok.jpg", "unanchored_outside_track", None, None, None),
    }
    _patch_main_common(monkeypatch, capture_utc=capture_utc, anchors=anchors)
    monkeypatch.setattr(
        film_module,
        "select_kept_photos",
        lambda photos, *, still_source, options: (
            list(photos),
            _fake_selection_result(
                tuple(p.source.name for p in photos), tuple(p.source.name for p in photos)
            ),
        ),
    )

    output = tmp_path / "film.mp4"
    receipt = tmp_path / "receipt.json"
    argv = _main_argv(tmp_path, photos_dir=photos_dir, output=output, receipt=receipt) + [
        "--pre-trip-photos"
    ]

    exit_code = film_module.main(argv)
    assert exit_code == 0

    import json

    saved = json.loads(receipt.read_text())
    assert saved["excluded_pretrip_photos"] == []
    assert saved["bookend_photos"] == ["before_ok.jpg"]


def test_no_bookends_flag_restores_refusal(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "before_ok.jpg").write_bytes(b"fake")

    capture_utc = {
        "before_ok.jpg": datetime(
            2026, 9, 16, 11, 0, 0, tzinfo=timezone.utc
        ),  # -3600s, would be a bookend
    }
    anchors = {
        "before_ok.jpg": MediaAnchor("before_ok.jpg", "unanchored_outside_track", None, None, None),
    }
    _patch_main_common(monkeypatch, capture_utc=capture_utc, anchors=anchors)

    output = tmp_path / "film.mp4"
    argv = _main_argv(tmp_path, photos_dir=photos_dir, output=output) + ["--no-bookends"]

    with pytest.raises(SystemExit):
        film_module.main(argv)

    # --pre-trip-photos, not just dropping --no-bookends, is what makes the
    # same photo a bookend and never refused (before-recording bookends are
    # excluded by default either way).
    monkeypatch.setattr(
        film_module,
        "select_kept_photos",
        lambda photos, *, still_source, options: (
            list(photos),
            _fake_selection_result(
                tuple(p.source.name for p in photos), tuple(p.source.name for p in photos)
            ),
        ),
    )
    exit_code = film_module.main(argv[:-1] + ["--pre-trip-photos"])
    assert exit_code == 0


def test_lookalike_bookends_collapse_to_one_via_best_pick(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    names = ["dup1.jpg", "dup2.jpg"]
    for name in names:
        (photos_dir / name).write_bytes(b"fake")

    capture_utc = {
        "dup1.jpg": datetime(2026, 9, 16, 11, 0, 0, tzinfo=timezone.utc),
        "dup2.jpg": datetime(2026, 9, 16, 11, 0, 1, tzinfo=timezone.utc),
    }
    anchors = {
        name: MediaAnchor(name, "unanchored_outside_track", None, None, None) for name in names
    }
    _patch_main_common(monkeypatch, capture_utc=capture_utc, anchors=anchors)

    select_calls = []

    def fake_select_kept_photos(photos, *, still_source, options):
        select_calls.append(tuple(p.source.name for p in photos))
        return [photos[0]], _fake_selection_result(("dup1.jpg",), tuple(names))

    monkeypatch.setattr(film_module, "select_kept_photos", fake_select_kept_photos)

    output = tmp_path / "film.mp4"
    receipt = tmp_path / "receipt.json"
    argv = _main_argv(tmp_path, photos_dir=photos_dir, output=output, receipt=receipt) + [
        "--pre-trip-photos"
    ]

    exit_code = film_module.main(argv)
    assert exit_code == 0
    assert set(select_calls[0]) == {"dup1.jpg", "dup2.jpg"}

    import json

    saved = json.loads(receipt.read_text())
    assert saved["bookend_photos"] == ["dup1.jpg"]


def test_pacing_kept_count_includes_bookends(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    names = ["mapped.jpg", "before.jpg", "after.jpg"]
    for name in names:
        (photos_dir / name).write_bytes(b"fake")

    capture_utc = {
        "mapped.jpg": datetime(2026, 9, 16, 13, 0, 0, tzinfo=timezone.utc),
        "before.jpg": datetime(2026, 9, 16, 11, 0, 0, tzinfo=timezone.utc),
        "after.jpg": datetime(2026, 9, 16, 15, 0, 0, tzinfo=timezone.utc),
    }
    anchors = {
        "mapped.jpg": MediaAnchor(
            "mapped.jpg", "timestamp_interpolated", 3600.0, 35.0001, -79.9999
        ),
        "before.jpg": MediaAnchor("before.jpg", "unanchored_outside_track", None, None, None),
        "after.jpg": MediaAnchor("after.jpg", "unanchored_outside_track", None, None, None),
    }
    _patch_main_common(monkeypatch, capture_utc=capture_utc, anchors=anchors)
    monkeypatch.setattr(
        film_module,
        "select_kept_photos",
        lambda photos, *, still_source, options: (
            list(photos),
            _fake_selection_result(
                tuple(p.source.name for p in photos), tuple(p.source.name for p in photos)
            ),
        ),
    )

    pacing_calls = []
    real_resolve = film_module.resolve_photo_seconds

    def spy_resolve(**kwargs):
        pacing_calls.append(kwargs)
        return real_resolve(**kwargs)

    monkeypatch.setattr(film_module, "resolve_photo_seconds", spy_resolve)

    output = tmp_path / "film.mp4"
    argv = _main_argv(tmp_path, photos_dir=photos_dir, output=output) + ["--pre-trip-photos"]
    exit_code = film_module.main(argv)
    assert exit_code == 0
    assert pacing_calls[0]["kept_count"] == 3


@pytest.mark.parametrize(("extra", "expected_blur"), [([], None), (["--skip-blur"], False)])
def test_main_routes_every_still_through_the_privacy_gate(
    tmp_path, monkeypatch, extra, expected_blur
):
    """Every photo reaches the privacy gate before selection or compositing,
    blur defaults on (``None`` = config default), ``--skip-blur`` passes
    ``False``, and the receipt carries the gate's attestation."""
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    names = ["before_ok.jpg", "after_ok.jpg", "placed.jpg"]
    for name in names:
        (photos_dir / name).write_bytes(b"fake")
    capture_utc = {
        "before_ok.jpg": datetime(2026, 9, 16, 11, 0, 0, tzinfo=timezone.utc),
        "after_ok.jpg": datetime(2026, 9, 16, 15, 0, 0, tzinfo=timezone.utc),
        "placed.jpg": datetime(2026, 9, 16, 13, 0, 0, tzinfo=timezone.utc),
    }
    anchors = {
        "before_ok.jpg": MediaAnchor("before_ok.jpg", "unanchored_outside_track", None, None, None),
        "after_ok.jpg": MediaAnchor("after_ok.jpg", "unanchored_outside_track", None, None, None),
        "placed.jpg": MediaAnchor(
            "placed.jpg", "timestamp_interpolated", 3600.0, 35.0001, -79.9999
        ),
    }
    gate_calls = _patch_main_common(monkeypatch, capture_utc=capture_utc, anchors=anchors)
    monkeypatch.setattr(
        film_module,
        "select_kept_photos",
        lambda photos, *, still_source, options: (
            list(photos),
            _fake_selection_result(
                tuple(p.source.name for p in photos), tuple(p.source.name for p in photos)
            ),
        ),
    )
    receipt = tmp_path / "receipt.json"
    argv = _main_argv(
        tmp_path, photos_dir=photos_dir, output=tmp_path / "film.mp4", receipt=receipt
    )

    assert film_module.main(argv + extra) == 0

    assert len(gate_calls) == 1
    assert sorted(gate_calls[0]["names"]) == sorted(names)
    assert gate_calls[0]["blur"] is expected_blur
    import json

    privacy = json.loads(receipt.read_text())["privacy"]
    assert privacy["blur"] is expected_blur
    assert privacy["withheld"] == []
