"""An operator review pack: a fast pre-render look before the long film render."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.review_pack import (
    PLACEMENT_TABLE_NAME,
    TABLE_HEADER_HEIGHT,
    TABLE_ROW_HEIGHT,
    TILE_WIDTH,
    WARNING_RGB,
    PlacementRow,
    classify_placement,
    format_route_clock,
    gather_review_inputs,
    opening_sheet,
    placement_table,
    stills_sheet,
    write_motion_clips,
    write_review_pack,
)
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster


def test_opening_sheet_tiles_frames_labelled_by_fraction(tmp_path):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    frames = [Image.new("RGB", (100, 60), color) for color in colors]

    output = opening_sheet(frames, tmp_path / "opening.jpg")

    sheet = Image.open(output)
    assert sheet.size == (TILE_WIDTH * 3, round(TILE_WIDTH * 60 / 100))
    for index, color in enumerate(colors):
        x = index * TILE_WIDTH + TILE_WIDTH // 2
        y = sheet.height - 5
        pixel = sheet.getpixel((x, y))
        assert all(abs(actual - expected) <= 3 for actual, expected in zip(pixel, color))


def test_stills_sheet_places_original_and_as_shown_side_by_side(tmp_path):
    original = Image.new("RGB", (200, 150), (10, 20, 30))
    as_shown = Image.new("RGB", (300, 200), (200, 100, 50))

    output = stills_sheet([("photo.jpg", original, as_shown)], tmp_path / "stills.jpg")

    sheet = Image.open(output)
    # Well inside each tile, clear of the label strip's JPEG colour bleed.
    left = sheet.getpixel((sheet.width // 4, sheet.height // 2))
    right = sheet.getpixel((3 * sheet.width // 4, sheet.height // 2))
    # JPEG is lossy: the colour survives to within a few levels, never exactly.
    assert all(abs(a - e) <= 4 for a, e in zip(left, (10, 20, 30)))
    assert all(abs(a - e) <= 4 for a, e in zip(right, (200, 100, 50)))


def test_placement_table_marks_an_unplaced_row_with_the_warning_colour(tmp_path):
    rows = [
        PlacementRow("a.jpg", "photo", "placed on route", "01:20"),
        PlacementRow("b.jpg", "photo", "not placed", "—", "no defensible route position"),
    ]

    output = placement_table(rows, tmp_path / PLACEMENT_TABLE_NAME)

    sheet = Image.open(output)
    placed_y = TABLE_HEADER_HEIGHT + 5
    unplaced_y = TABLE_HEADER_HEIGHT + TABLE_ROW_HEIGHT + 5
    assert sheet.getpixel((5, placed_y)) != WARNING_RGB
    assert sheet.getpixel((5, unplaced_y)) == WARNING_RGB


def test_classify_placement_follows_the_films_own_bookend_rule():
    """Before/after the trail is what the film PLAYS as a bookend, never a second clock rule."""
    window = {"track_start_s": 0.0, "track_end_s": 100.0}
    before = classify_placement(
        is_mapped=False, captured_at_s=-5.0, bookend="before_recording", **window
    )
    after = classify_placement(
        is_mapped=False, captured_at_s=150.0, bookend="after_recording", **window
    )
    assert before[0] == "before the trail" and "before the first map leg" in before[1]
    assert after[0] == "after the trail" and "after the last map leg" in after[1]
    # Outside the track but NOT a bookend (too far outside, or bookends off):
    # the film leaves it out, so the table must not dress it as a notice.
    assert classify_placement(is_mapped=False, captured_at_s=-5.0, **window) == (
        "not placed",
        "captured before the recorded track started; not a bookend",
    )
    assert classify_placement(is_mapped=False, captured_at_s=150.0, **window) == (
        "not placed",
        "captured after the recorded track ended; not a bookend",
    )
    assert classify_placement(is_mapped=False, captured_at_s=None, **window) == (
        "not placed",
        "no defensible route position",
    )
    assert classify_placement(is_mapped=True, captured_at_s=50.0, **window) == (
        "placed on route",
        "",
    )


def test_every_reason_the_table_can_show_fits_its_column():
    from gomoveshift.pipeline.route_story import review_pack as pack_module

    window = {"track_start_s": 0.0, "track_end_s": 100.0}
    cases = [(-5.0, "before_recording"), (150.0, "after_recording"), (-5.0, None), (150.0, None)]
    cases.append((None, None))
    font = pack_module._font(18)
    for captured_at_s, bookend in cases:
        _status, reason = classify_placement(
            is_mapped=False, captured_at_s=captured_at_s, bookend=bookend, **window
        )
        assert font.getlength(reason) + 10 <= pack_module.TABLE_COLUMN_WIDTHS[-1], reason


def test_bookends_show_in_play_order_on_the_stills_sheet_and_as_notices(tmp_path, monkeypatch):
    from gomoveshift.pipeline.route_story import review_pack as pack_module

    def photo(name, mapped, clock):
        path = tmp_path / name
        Image.new("RGB", (400, 300), (30, 60, 90)).save(path)
        return _FakePhoto(path, mapped, clock if mapped else None, clock)

    early, kept, late = (
        photo("early.jpg", False, -60.0),
        photo("kept.jpg", True, 42.0),
        photo("late.jpg", False, 160.0),
    )
    dropped_bookend = photo("early-twin.jpg", False, -61.0)
    far = photo("yesterday.jpg", False, -90000.0)
    shown: list[str] = []
    rows: list[PlacementRow] = []
    real_pair = pack_module._still_pair
    monkeypatch.setattr(
        pack_module,
        "_still_pair",
        lambda source, *a, **k: shown.append(Path(source).name) or real_pair(source, *a, **k),
    )
    real_table = pack_module.placement_table
    monkeypatch.setattr(
        pack_module,
        "placement_table",
        lambda table_rows, output: rows.extend(table_rows) or real_table(table_rows, output),
    )

    gather_review_inputs(
        tmp_path / "pack",
        rasters=(_raster(tmp_path),),
        first_point=RoutePoint(0.0, 35.2, -83.5, 800.0),
        labels=(),
        points=[RoutePoint(0.0, 35.2, -83.5, 800.0), RoutePoint(100.0, 35.25, -83.4, 820.0)],
        photos=[kept, late, far, dropped_bookend, early],
        kept_mapped=[kept],
        before_bookends=[early],
        after_bookends=[late],
        dropped_bookend_names={"early-twin.jpg"},
        source_video=None,
        video_placed=False,
        video_clock_s=None,
        width=320,
        height=180,
    )

    assert shown == ["early.jpg", "kept.jpg", "late.jpg"]
    status = {row.file_name: row.status for row in rows}
    assert status == {
        "early.jpg": "before the trail",
        "kept.jpg": "placed on route",
        "late.jpg": "after the trail",
        "yesterday.jpg": "not placed",
    }


def test_format_route_clock():
    assert format_route_clock(90.0) == "01:30"
    assert format_route_clock(None) == "—"


def _pack_inputs(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    opening = directory / "opening-sheet.jpg"
    stills = directory / "stills-sheet.jpg"
    placements = directory / PLACEMENT_TABLE_NAME
    Image.new("RGB", (10, 10)).save(opening)
    Image.new("RGB", (10, 10)).save(stills)
    Image.new("RGB", (10, 10)).save(placements)
    return {"opening": opening, "stills": stills, "placements": placements}


def test_write_review_pack_counts_kept_dropped_and_unplaced_and_stays_in_directory(tmp_path):
    directory = tmp_path / "pack"
    inputs = _pack_inputs(directory)
    selection = {
        "kept": ("a.jpg", "b.jpg"),
        "groups": ({"losers": ("c.jpg",)},),
        "not_considered": ({"name": "d.jpg", "anchor_status": "unanchored_gap"},),
    }

    result = write_review_pack(directory, **inputs, selection=selection, preflight_sheet=None)

    assert result["counts"] == {"kept": 2, "dropped": 1, "unplaced": 1}
    on_disk = json.loads((directory / "review-pack.json").read_text())
    assert on_disk == json.loads(json.dumps(result))
    assert sorted(p.name for p in directory.iterdir()) == sorted(
        ["opening-sheet.jpg", "stills-sheet.jpg", PLACEMENT_TABLE_NAME, "review-pack.json"]
    )


def test_the_pack_names_the_marker_sheet_the_preflight_walk_will_write(tmp_path):
    """The pack JSON is written before the preflight walk runs, so it records
    the name the sheet will land under -- the same way it already does for the
    contact sheet. An operator reading the JSON finds every sheet listed."""
    directory = tmp_path / "pack"
    inputs = _pack_inputs(directory)
    selection = {"kept": (), "groups": (), "not_considered": ()}

    result = write_review_pack(
        directory,
        **inputs,
        selection=selection,
        preflight_sheet=None,
        marker_sheet=directory / "marker-sheet.jpg",
    )

    assert result["marker_sheet"] == str(directory / "marker-sheet.jpg")


def test_a_pack_with_no_marker_sheet_says_so_rather_than_omitting_the_key(tmp_path):
    directory = tmp_path / "pack"
    inputs = _pack_inputs(directory)

    result = write_review_pack(
        directory,
        **inputs,
        selection={"kept": (), "groups": (), "not_considered": ()},
        preflight_sheet=None,
    )

    assert result["marker_sheet"] is None


def test_write_review_pack_is_deterministic(tmp_path):
    directory = tmp_path / "pack"
    inputs = _pack_inputs(directory)
    selection = {"kept": (), "groups": (), "not_considered": ()}

    first = write_review_pack(directory, **inputs, selection=selection, preflight_sheet=None)
    second = write_review_pack(directory, **inputs, selection=selection, preflight_sheet=None)

    assert first == second


class _FakeAnchor:
    def __init__(self, status: str, track_time_s: float | None):
        self.status = status
        self.track_time_s = track_time_s


class _FakePhoto:
    def __init__(self, source: Path, is_mapped: bool, track_time_s, captured_at_exif_s):
        self.source = source
        self._is_mapped = is_mapped
        self.anchor = _FakeAnchor(
            "embedded_coordinates" if is_mapped else "unanchored_gap", track_time_s
        )
        self.captured_at_exif_s = captured_at_exif_s

    @property
    def is_mapped(self) -> bool:
        return self._is_mapped


def _raster(tmp_path: Path) -> SatelliteRaster:
    path = tmp_path / "national.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(path)
    return SatelliteRaster(path, -130.0, 20.0, -60.0, 55.0, "National agency")


def test_gather_review_inputs_writes_a_full_pack_the_operator_can_open(tmp_path):
    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    kept_path = photos_dir / "kept.jpg"
    Image.new("RGB", (400, 300), (30, 60, 90)).save(kept_path)
    unplaced_path = photos_dir / "unplaced.jpg"
    Image.new("RGB", (400, 300), (90, 30, 60)).save(unplaced_path)

    kept_photo = _FakePhoto(kept_path, True, track_time_s=42.0, captured_at_exif_s=42.0)
    unplaced_photo = _FakePhoto(unplaced_path, False, track_time_s=None, captured_at_exif_s=-10.0)
    points = [RoutePoint(0.0, 35.2, -83.5, 800.0), RoutePoint(100.0, 35.25, -83.4, 820.0)]
    first_point = points[0]

    pack_dir = tmp_path / "pack"
    inputs = gather_review_inputs(
        pack_dir,
        rasters=(_raster(tmp_path),),
        first_point=first_point,
        labels=(),
        points=points,
        photos=[kept_photo, unplaced_photo],
        kept_mapped=[kept_photo],
        source_video=None,
        video_placed=False,
        video_clock_s=None,
        width=320,
        height=180,
        still_source="developed",
    )
    pack = write_review_pack(
        pack_dir,
        **inputs,
        selection={"kept": ("kept.jpg",), "groups": (), "not_considered": ()},
        preflight_sheet=None,
    )

    for key in ("opening", "stills", "placements"):
        assert inputs[key].exists()
        assert inputs[key].parent == pack_dir
    assert pack["counts"]["kept"] == 1
    assert (pack_dir / "review-pack.json").exists()


def test_a_failed_preflight_still_leaves_the_pack_behind(tmp_path, monkeypatch):
    """The run the operator most needs to look at is the one the checks refused."""
    import pytest

    from gomoveshift.pipeline.route_story import film as film_module
    from gomoveshift.pipeline.route_story.preflight import RoutePreflightError
    import test_route_story_best_pick_wiring as wiring

    photos_dir = tmp_path / "photos"
    photos_dir.mkdir()
    (photos_dir / "a.jpg").write_bytes(b"fake")
    anchors = {
        "a.jpg": wiring.MediaAnchor("a.jpg", "timestamp_interpolated", 5.0, 35.0001, -79.9999)
    }
    wiring._patch_common(monkeypatch, anchors=anchors)
    fake_result = wiring._fake_selection_result(("a.jpg",), ("a.jpg",))
    monkeypatch.setattr(
        film_module,
        "select_kept_photos",
        lambda mapped_photos, *, still_source, options: (list(mapped_photos), fake_result),
    )
    written: list[Path] = []
    monkeypatch.setattr(
        film_module,
        "gather_review_inputs",
        lambda directory, **kwargs: (
            written.append(Path(directory))
            or {
                "opening": directory / "o",
                "stills": directory / "s",
                "placements": directory / "p",
            }
        ),
    )

    def refuse(*args, **kwargs):
        raise RoutePreflightError("frame 0: 40.00% unloaded-tile")

    monkeypatch.setattr(film_module, "run_preflight", refuse)

    output = tmp_path / "film.mp4"
    argv = wiring._base_argv(tmp_path, photos_dir=photos_dir, output=output) + [
        "--preflight-only",
        "--preflight-frames",
        "1",
    ]
    with pytest.raises(RoutePreflightError):
        film_module.main(argv)

    assert written == [tmp_path / "film-review-pack"]
    assert (tmp_path / "film-review-pack" / "review-pack.json").exists()


def test_an_unreadable_photo_is_shown_as_unreadable_not_raised(tmp_path):
    from gomoveshift.pipeline.route_story.review_pack import WARNING_RGB, _still_pair

    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")

    name, original, as_shown = _still_pair(broken, 320, 180, "developed")

    assert "COULD NOT BE READ" in name
    assert original.getpixel((5, 5)) == WARNING_RGB == as_shown.getpixel((5, 5))


def test_before_and_after_the_trail_rows_are_a_notice_not_a_warning(tmp_path):
    from gomoveshift.pipeline.route_story.review_pack import NOTICE_RGB, WARNING_RGB

    rows = [
        PlacementRow("before.jpg", "photo", "before the trail", "—", "captured before"),
        PlacementRow("lost.jpg", "photo", "not placed", "—", "no defensible route position"),
    ]
    sheet = Image.open(placement_table(rows, tmp_path / "table.png")).convert("RGB")

    assert sheet.getpixel((5, 44 + 28)) == NOTICE_RGB
    assert sheet.getpixel((5, 44 + 56 + 28)) == WARNING_RGB


def test_a_trip_with_no_media_still_gets_sheets_that_say_so(tmp_path):
    stills = stills_sheet([], tmp_path / "stills.jpg")
    table = placement_table([], tmp_path / "table.png")

    assert Image.open(stills).width > 0
    assert Image.open(table).width > 0


def test_the_pack_shows_a_still_exactly_as_the_film_renders_its_first_frame(tmp_path):
    """The pack is the surface the operator approves a trip on, so a still in
    it must be pixel-identical to the film's own first presentation frame.
    Two implementations of "the still card" drift apart silently, and the
    operator then signs off on a frame the render never produces -- the pack
    showed a 0.91 letterboxed inset while the film had moved to a full-bleed
    push-in."""
    from gomoveshift.pipeline.route_story.hud import (
        _load_still,
        prepare_still,
        still_card_frame,
    )
    from gomoveshift.pipeline.route_story.review_pack import render_still_as_shown

    source = tmp_path / "wide.jpg"
    Image.new("RGB", (1200, 800), (40, 120, 200)).save(source)
    width, height = 640, 360

    as_shown = render_still_as_shown(source, width, height, still_source="camera")
    expected = still_card_frame(
        prepare_still(_load_still(source, source="camera"), width, height), width, height, 0.0
    )

    assert as_shown.size == (width, height)
    assert as_shown.convert("RGB").tobytes() == expected.convert("RGB").tobytes()


class _FakeCompletedProcess:
    def __init__(self, args):
        self.args = args
        self.returncode = 0


def test_write_motion_clips_runs_ffmpeg_per_window_and_returns_rows(tmp_path):
    video = tmp_path / "film.mp4"
    video.write_bytes(b"not a real video")
    directory = tmp_path / "pack"
    calls = []

    def fake_runner(cmd, **kwargs):
        calls.append((list(cmd), kwargs))
        return _FakeCompletedProcess(cmd)

    windows = [(1.0, 5.0), (10.0, 14.0)]

    clips = write_motion_clips(video, directory, windows, runner=fake_runner)

    assert clips == [
        {"name": "motion-clip-1.mp4", "start_s": 1.0, "end_s": 5.0},
        {"name": "motion-clip-2.mp4", "start_s": 10.0, "end_s": 14.0},
    ]
    assert len(calls) == 2
    first_cmd, first_kwargs = calls[0]
    assert first_cmd == [
        "ffmpeg",
        "-y",
        "-ss",
        "1.0",
        "-t",
        "4.0",
        "-i",
        str(video),
        "-c",
        "copy",
        str(directory / "motion-clip-1.mp4"),
    ]
    assert first_kwargs["check"] is True


def test_write_motion_clips_empty_windows_returns_empty_list(tmp_path):
    video = tmp_path / "film.mp4"
    video.write_bytes(b"x")
    directory = tmp_path / "pack"
    calls = []

    clips = write_motion_clips(video, directory, [], runner=lambda cmd, **kwargs: calls.append(cmd))

    assert clips == []
    assert calls == []


def test_write_review_pack_records_motion_clips_and_defaults_to_empty(tmp_path):
    directory = tmp_path / "pack"
    inputs = _pack_inputs(directory)
    selection = {"kept": (), "groups": (), "not_considered": ()}

    default_result = write_review_pack(
        directory, **inputs, selection=selection, preflight_sheet=None
    )
    assert default_result["motion_clips"] == []

    clips = [{"name": "motion-clip-1.mp4", "start_s": 1.0, "end_s": 5.0}]
    result = write_review_pack(
        directory, **inputs, selection=selection, preflight_sheet=None, motion_clips=clips
    )
    assert result["motion_clips"] == clips
    on_disk = json.loads((directory / "review-pack.json").read_text())
    assert on_disk["motion_clips"] == clips
