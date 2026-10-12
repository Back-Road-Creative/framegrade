"""A Route Story film shows only the best photo of each look-alike group, and
stills never crowd out the map."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import PhotoMoment
from gomoveshift.pipeline.route_story.privacy import PrivacyResult
from gomoveshift.pipeline.route_story.select import SelectionOptions, SelectionResult
from gomoveshift.pipeline.route_story.still_selection import (
    PHOTO_SECONDS_FLOOR,
    STILLS_SHARE_CAP,
    build_photo_candidates,
    max_stills_for_cap,
    resolve_photo_seconds,
)


def _rasters(tmp_path: Path):
    national = tmp_path / "national.jpg"
    local = tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    return national, local


def _photo(tmp_path: Path, name: str) -> PhotoMoment:
    source = tmp_path / name
    Image.new("RGB", (64, 32), "#aa7744").save(source)
    return PhotoMoment(source, MediaAnchor(name, "timestamp_interpolated", None, None, None))


def _gpx(tmp_path: Path) -> Path:
    gpx = tmp_path / "day.gpx"
    gpx.write_text(
        """<gpx><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T12:01:00Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    return gpx


def _base_argv(tmp_path: Path, *, photos_dir: Path, output: Path, receipt: Path | None = None):
    national, local = _rasters(tmp_path)
    argv = [
        "--gpx",
        str(_gpx(tmp_path)),
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


def _patch_common(monkeypatch, *, anchors: dict[str, MediaAnchor]):
    """Skip real EXIF/anchoring; hand back a fixed, named anchor per file."""
    monkeypatch.setattr(
        film_module,
        "extract_exif_capture_metadata",
        lambda source: type(
            "M", (), {"capture_utc": datetime(2026, 9, 16, 12, 0, 5, tzinfo=timezone.utc)}
        )(),
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
    # Placeholder bytes are unreadable, so the real privacy gate would withhold every
    # photo; the gate has its own tests (test_route_story_privacy_gate.py).
    monkeypatch.setattr(
        film_module,
        "apply_privacy_gate",
        lambda photos, *, blur, work_dir: PrivacyResult(
            photos=tuple(photos), attestation={"classes": {}, "blur": blur}
        ),
    )


def _fake_selection_result(kept: tuple[str, ...], all_names: tuple[str, ...]) -> SelectionResult:
    from gomoveshift.pipeline.route_story.select import GroupResult, PhotoScore

    losers = tuple(name for name in all_names if name not in kept)
    score = PhotoScore(name="x", sharpness=1.0, exposure_badness=0.0, color_vibrancy=1.0)
    group = GroupResult(
        members=all_names,
        winner=kept[0],
        kept=kept,
        losers=losers,
        scores={name: score for name in all_names},
        reason="test group",
    )
    return SelectionResult(groups=(group,), kept=kept)


def test_main_renders_only_kept_photos_and_names_losers_in_receipt(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    names = ["group1.jpg", "group2.jpg", "group3.jpg", "distinct.jpg"]
    for name in names:
        (photos_dir / name).write_bytes(b"fake")

    anchors = {
        name: MediaAnchor(name, "timestamp_interpolated", 5.0 + index, 35.0001, -79.9999)
        for index, name in enumerate(names)
    }
    _patch_common(monkeypatch, anchors=anchors)

    captured_kwargs: dict = {}
    monkeypatch.setattr(
        film_module, "render_route_story_film", lambda **kwargs: captured_kwargs.update(kwargs)
    )

    fake_result = _fake_selection_result(("group2.jpg", "distinct.jpg"), tuple(names))
    select_calls = []

    def fake_select_kept_photos(mapped_photos, *, still_source, options):
        select_calls.append((tuple(p.source.name for p in mapped_photos), options))
        kept_names = set(fake_result.kept)
        kept = [p for p in mapped_photos if p.source.name in kept_names]
        return kept, fake_result

    monkeypatch.setattr(film_module, "select_kept_photos", fake_select_kept_photos)

    output = tmp_path / "film.mp4"
    receipt = tmp_path / "receipt.json"
    argv = _base_argv(tmp_path, photos_dir=photos_dir, output=output, receipt=receipt)

    exit_code = film_module.main(argv)
    assert exit_code == 0

    assert set(select_calls[0][0]) == set(names)
    assert select_calls[0][1].enabled is True

    rendered_names = {photo.source.name for photo in captured_kwargs["photos"]}
    assert rendered_names == {"group2.jpg", "distinct.jpg"}

    import json

    saved = json.loads(receipt.read_text())
    assert saved["photo_selection"]["enabled"] is True
    assert saved["photo_selection"]["capture_time_source"] == "exif"
    group = saved["photo_selection"]["groups"][0]
    assert group["winner"] == "group2.jpg"
    assert set(group["losers"]) == {"group1.jpg", "group3.jpg"}
    assert "stills_pacing" in saved


def test_pacing_call_agrees_with_render_defaults_for_intro_and_recap(tmp_path, monkeypatch):
    """The pacing cap must use the SAME intro/recap seconds the render call
    actually gets -- not a copy of the numbers that can silently drift."""
    import inspect

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "a.jpg").write_bytes(b"fake")
    anchors = {"a.jpg": MediaAnchor("a.jpg", "timestamp_interpolated", 5.0, 35.0001, -79.9999)}
    _patch_common(monkeypatch, anchors=anchors)
    real_render = film_module.render_route_story_film
    monkeypatch.setattr(film_module, "render_route_story_film", lambda **kwargs: None)
    monkeypatch.setattr(
        film_module,
        "select_kept_photos",
        lambda mapped_photos, *, still_source, options: (
            list(mapped_photos),
            _fake_selection_result(("a.jpg",), ("a.jpg",)),
        ),
    )

    pacing_calls = []
    real_resolve = film_module.resolve_photo_seconds

    def spy_resolve(**kwargs):
        pacing_calls.append(kwargs)
        return real_resolve(**kwargs)

    monkeypatch.setattr(film_module, "resolve_photo_seconds", spy_resolve)

    output = tmp_path / "film.mp4"
    argv = _base_argv(tmp_path, photos_dir=photos_dir, output=output)
    exit_code = film_module.main(argv)
    assert exit_code == 0

    render_defaults = inspect.signature(real_render).parameters
    assert pacing_calls[0]["intro_seconds"] == render_defaults["intro_seconds"].default
    assert pacing_calls[0]["recap_seconds"] == render_defaults["recap_seconds"].default
    assert render_defaults["intro_seconds"].default == film_module.DEFAULT_INTRO_SECONDS
    assert render_defaults["recap_seconds"].default == film_module.DEFAULT_RECAP_SECONDS


def test_no_best_pick_keeps_every_photo(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    names = ["a.jpg", "b.jpg"]
    for name in names:
        (photos_dir / name).write_bytes(b"fake")
    anchors = {
        name: MediaAnchor(name, "timestamp_interpolated", 5.0, 35.0001, -79.9999) for name in names
    }
    _patch_common(monkeypatch, anchors=anchors)
    monkeypatch.setattr(film_module, "render_route_story_film", lambda **kwargs: None)

    select_calls = []

    def fake_select_kept_photos(mapped_photos, *, still_source, options):
        select_calls.append(options)
        return list(mapped_photos), _fake_selection_result(tuple(names), tuple(names))

    monkeypatch.setattr(film_module, "select_kept_photos", fake_select_kept_photos)

    output = tmp_path / "film.mp4"
    receipt = tmp_path / "receipt.json"
    argv = _base_argv(tmp_path, photos_dir=photos_dir, output=output, receipt=receipt) + [
        "--no-best-pick"
    ]

    exit_code = film_module.main(argv)
    assert exit_code == 0
    assert select_calls[0].enabled is False

    import json

    saved = json.loads(receipt.read_text())
    assert saved["photo_selection"]["enabled"] is False


def test_keep_flag_passes_case_insensitive_name_through(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    names = ["Group1.jpg", "group2.jpg"]
    for name in names:
        (photos_dir / name).write_bytes(b"fake")
    anchors = {
        name: MediaAnchor(name, "timestamp_interpolated", 5.0, 35.0001, -79.9999) for name in names
    }
    _patch_common(monkeypatch, anchors=anchors)
    monkeypatch.setattr(film_module, "render_route_story_film", lambda **kwargs: None)

    select_calls = []

    def fake_select_kept_photos(mapped_photos, *, still_source, options):
        select_calls.append(options)
        return list(mapped_photos), _fake_selection_result(tuple(names), tuple(names))

    monkeypatch.setattr(film_module, "select_kept_photos", fake_select_kept_photos)

    output = tmp_path / "film.mp4"
    argv = _base_argv(tmp_path, photos_dir=photos_dir, output=output) + ["--keep", "group1.jpg"]

    exit_code = film_module.main(argv)
    assert exit_code == 0
    assert select_calls[0].keep == frozenset({"Group1.jpg"})


def test_keep_flag_unknown_name_is_a_parser_error(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "a.jpg").write_bytes(b"fake")
    anchors = {"a.jpg": MediaAnchor("a.jpg", "timestamp_interpolated", 5.0, 35.0001, -79.9999)}
    _patch_common(monkeypatch, anchors=anchors)

    output = tmp_path / "film.mp4"
    argv = _base_argv(tmp_path, photos_dir=photos_dir, output=output) + ["--keep", "nope.jpg"]

    with pytest.raises(SystemExit):
        film_module.main(argv)


def test_capture_time_fed_to_selector_is_exif_route_seconds_not_mtime(tmp_path):
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16), "#334455").save(source)
    import os

    # A misleading, fixed mtime, far from either clock below -- never the
    # real clock, so this stays deterministic.
    misleading = 1_000_000.0
    os.utime(source, (misleading, misleading))

    # anchor.track_time_s is the resolved ROUTE POSITION time (where the
    # nearest track point is), deliberately different from the EXIF capture
    # time -- the selector must be fed the EXIF value, not the anchor's.
    anchor = MediaAnchor("photo.jpg", "timestamp_interpolated", 42.5, 35.0, -80.0)
    photo = PhotoMoment(source, anchor, captured_at_exif_s=37.1)

    candidates = build_photo_candidates([photo], still_source="developed")
    assert candidates[0].captured_at == 37.1
    assert candidates[0].placement == (35.0, -80.0)


def test_missing_exif_capture_time_raises_instead_of_defaulting(tmp_path):
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16), "#334455").save(source)
    anchor = MediaAnchor("photo.jpg", "timestamp_interpolated", 42.5, 35.0, -80.0)
    photo = PhotoMoment(source, anchor)  # captured_at_exif_s left unset

    with pytest.raises(ValueError, match="photo.jpg"):
        build_photo_candidates([photo], still_source="developed")


def test_unplaced_photo_never_reaches_the_selector(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "placed.jpg").write_bytes(b"fake")
    (photos_dir / "orphan.jpg").write_bytes(b"fake")
    anchors = {
        "placed.jpg": MediaAnchor("placed.jpg", "timestamp_interpolated", 5.0, 35.0001, -79.9999),
        "orphan.jpg": MediaAnchor("orphan.jpg", "unanchored_outside_track", None, None, None),
    }
    _patch_common(monkeypatch, anchors=anchors)

    select_calls = []

    def fake_select_kept_photos(mapped_photos, *, still_source, options):
        select_calls.append(tuple(p.source.name for p in mapped_photos))
        return list(mapped_photos), _fake_selection_result(
            tuple(p.source.name for p in mapped_photos), tuple(p.source.name for p in mapped_photos)
        )

    monkeypatch.setattr(film_module, "select_kept_photos", fake_select_kept_photos)

    output = tmp_path / "film.mp4"
    argv = _base_argv(tmp_path, photos_dir=photos_dir, output=output)

    with pytest.raises(SystemExit):
        film_module.main(argv)
    assert select_calls == []

    monkeypatch.setattr(film_module, "render_route_story_film", lambda **kwargs: None)
    receipt = tmp_path / "receipt.json"
    exit_code = film_module.main(argv + ["--allow-unplaced", "--receipt", str(receipt)])
    assert exit_code == 0
    assert select_calls == [("placed.jpg",)]

    import json

    saved = json.loads(receipt.read_text())
    not_considered = saved["photo_selection"]["not_considered"]
    assert not_considered == [{"name": "orphan.jpg", "anchor_status": "unanchored_outside_track"}]


def test_preflight_only_run_writes_photo_selection_receipt(tmp_path, monkeypatch):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "a.jpg").write_bytes(b"fake")
    anchors = {"a.jpg": MediaAnchor("a.jpg", "timestamp_interpolated", 5.0, 35.0001, -79.9999)}
    _patch_common(monkeypatch, anchors=anchors)
    monkeypatch.setattr(
        film_module,
        "run_preflight",
        lambda *a, **k: type("R", (), {"sheet": None, "passed": True, "frames": 0})(),
    )
    monkeypatch.setattr(film_module, "render_route_story_film", lambda **kwargs: None)

    fake_result = _fake_selection_result(("a.jpg",), ("a.jpg",))
    monkeypatch.setattr(
        film_module,
        "select_kept_photos",
        lambda mapped_photos, *, still_source, options: (list(mapped_photos), fake_result),
    )

    output = tmp_path / "film.mp4"
    receipt = tmp_path / "receipt.json"
    argv = _base_argv(tmp_path, photos_dir=photos_dir, output=output, receipt=receipt) + [
        "--preflight-only",
        "--preflight-frames",
        "1",
    ]

    exit_code = film_module.main(argv)
    assert exit_code == 0

    import json

    saved = json.loads(receipt.read_text())
    assert saved["photo_selection"]["groups"][0]["winner"] == "a.jpg"
    assert "stills_pacing" in saved


# --- pacing (pure function) ---


def test_pacing_under_cap_leaves_seconds_unchanged():
    pacing = resolve_photo_seconds(
        requested_seconds=2.5, kept_count=3, intro_seconds=12.0, map_seconds=20.0, recap_seconds=6.0
    )
    assert pacing["used_seconds"] == 2.5
    assert pacing["floor_hit"] is False
    assert pacing["stills_share"] < STILLS_SHARE_CAP


def test_pacing_over_cap_shrinks_evenly():
    pacing = resolve_photo_seconds(
        requested_seconds=20.0,
        kept_count=5,
        intro_seconds=12.0,
        map_seconds=10.0,
        recap_seconds=6.0,
    )
    assert pacing["used_seconds"] < 20.0
    assert pacing["used_seconds"] > PHOTO_SECONDS_FLOOR
    assert pacing["stills_share"] == pytest.approx(STILLS_SHARE_CAP, abs=1e-6)
    assert pacing["floor_hit"] is False


def test_pacing_never_shrinks_below_floor_and_reports_floor_hit():
    pacing = resolve_photo_seconds(
        requested_seconds=20.0,
        kept_count=1000,
        intro_seconds=1.0,
        map_seconds=1.0,
        recap_seconds=1.0,
    )
    assert pacing["used_seconds"] == PHOTO_SECONDS_FLOOR
    assert pacing["floor_hit"] is True
    assert pacing["stills_share"] > STILLS_SHARE_CAP


# --- max_stills_for_cap (pure function) ---


def test_max_stills_for_cap_with_no_other_scenes_is_zero():
    assert max_stills_for_cap(other_seconds=0.0) == 0


def test_max_stills_for_cap_matches_the_cap_formula():
    # STILLS_SHARE_CAP=0.35, PHOTO_SECONDS_FLOOR=2.5: raw = 0.35*100/(2.5*0.65) = 21.53...
    assert max_stills_for_cap(other_seconds=100.0) == 21


def test_max_stills_for_cap_never_exceeds_the_cap_at_the_floor():
    for other_seconds in (5.0, 37.5, 90.0, 250.0, 1000.0):
        count = max_stills_for_cap(other_seconds=other_seconds)
        pacing = resolve_photo_seconds(
            requested_seconds=PHOTO_SECONDS_FLOOR,
            kept_count=count,
            intro_seconds=0.0,
            map_seconds=other_seconds,
            recap_seconds=0.0,
        )
        assert pacing["floor_hit"] is False
        assert pacing["stills_share"] <= STILLS_SHARE_CAP + 1e-9

        pacing_one_more = resolve_photo_seconds(
            requested_seconds=PHOTO_SECONDS_FLOOR,
            kept_count=count + 1,
            intro_seconds=0.0,
            map_seconds=other_seconds,
            recap_seconds=0.0,
        )
        assert pacing_one_more["stills_share"] > STILLS_SHARE_CAP
