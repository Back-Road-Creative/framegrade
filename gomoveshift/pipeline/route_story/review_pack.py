"""An operator review pack: a small set of glanceable images plus one JSON.

Written by ``--preflight-only`` so a human can approve or reject a Route
Story film in seconds, before the multi-hour long render starts. A full
render takes hours; the things an operator rejects a finished film for --
an unrecognisable opening shot, a dull photo, a photo or video landing in
the wrong place -- are all visible in this pack in seconds.

Every builder here is pure: it takes already-rendered/loaded inputs and
writes files. It does no CLI parsing and no GPX/EXIF loading of its own.
``film.py`` computes those inputs and calls ``gather_review_inputs``, whose
only job is to keep that call out of the tight ``--preflight-only`` budget.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Collection, Sequence

from PIL import Image, ImageDraw

from ..photos.develop import PhotoDecodeError
from .frame_qa import FrameQaThresholds
from .hud import (
    _camera_preview,
    _draw_opening,
    _fit_cover,
    _font,
    _load_still,
    prepare_still,
    still_card_frame,
)
from .layers import _render_opening_map, opening_credit

Runner = Callable[..., Any]

OPENING_SHEET_NAME = "opening-sheet.jpg"
STILLS_SHEET_NAME = "stills-sheet.jpg"
MARKER_SHEET_NAME = "marker-sheet.jpg"
PLACEMENT_TABLE_NAME = "placement-table.png"
REVIEW_PACK_JSON = "review-pack.json"

# Opening-frame fractions the film itself sweeps across (continental overview
# to trailhead); the operator gets one tile per fraction.
OPENING_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)

# Sized to judge colour and sharpness on, not just to recognise the shot: a
# thumbnail hides exactly the dullness and blur the pack exists to catch.
TILE_WIDTH = 640
STILL_TILE_WIDTH = 800
STILL_TILE_HEIGHT = 533

# Marker close-ups. A four-pixel gap between the dot and the end of its own
# bright line is invisible in a 640px-wide map tile and obvious in a 320px
# crop, which is the whole reason this sheet is separate from the opening one.
# The tiles come from the preflight walk's OWN frames and its own measured
# gaps (preflight.run_preflight), never a second render at second sample
# times: a sheet built from its own walk could show a clean marker on a film
# the gate then refuses, or the reverse.
MARKER_CROP_PX = 320
MARKER_SHEET_COLUMNS = 4
# The render's own ceiling, imported rather than restated: a sheet flagging at
# its own threshold would show green on a film the render then refuses.
MARKER_GAP_CEILING_PX = FrameQaThresholds().max_line_end_to_marker_px
LABEL_HEIGHT = 44  # taller than the stills sheet's 33px label type, which a 34px strip clipped
TABLE_HEADER_HEIGHT = 44
TABLE_ROW_HEIGHT = 56
TABLE_COLUMNS = ("File", "Kind", "Status", "Route clock", "Reason")
TABLE_COLUMN_WIDTHS = (360, 90, 220, 170, 660)  # the longest reason must fit
# A rust-red distinct from the dashboard/HUD palette, read at a glance on a phone.
WARNING_RGB = (196, 60, 44)
# Amber: worth a look, not a defect -- a photo taken before or after the
# recording has no place on the map but is still a real moment of the trip.
NOTICE_RGB = (214, 158, 46)
ROW_FILL_BY_STATUS = {
    "not placed": WARNING_RGB,
    "before the trail": NOTICE_RGB,
    "after the trail": NOTICE_RGB,
}


def opening_sheet(frames: Sequence[Image.Image], output: Path) -> Path:
    """Tile opening frames, each labelled with its fraction, in one strip."""
    if not frames:
        raise ValueError("opening_sheet needs at least one frame")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    thumb_height = round(TILE_WIDTH * frames[0].height / frames[0].width)
    sheet = Image.new("RGB", (TILE_WIDTH * len(frames), thumb_height), "#101418")
    draw = ImageDraw.Draw(sheet)
    font = _font(max(11, TILE_WIDTH // 24), bold=True)
    for index, frame in enumerate(frames):
        fraction = index / (len(frames) - 1) if len(frames) > 1 else 0.0
        x = index * TILE_WIDTH
        sheet.paste(frame.convert("RGB").resize((TILE_WIDTH, thumb_height)), (x, 0))
        draw.rectangle((x, 0, x + TILE_WIDTH, font.size + 6), fill=(0, 0, 0))
        draw.text((x + 4, 2), f"{fraction:.2f}", fill="#ffffff", font=font)
    sheet.save(output)
    return output


def render_still_as_shown(
    source: Path, width: int, height: int, *, still_source: str = "developed"
) -> Image.Image:
    """Composite a photo exactly as the film's still card shows it.

    Calls the same ``prepare_still`` / ``still_card_frame`` pair
    ``assembly._append_photo`` drives, at ``progress=0.0`` -- the film's first
    presentation frame of that photo. This is deliberately a call into the
    film's own recipe rather than a second composite kept in step by hand: the
    pack is what an operator approves a trip on, so a pack frame that differs
    from the render is worse than no pack at all. It drifted exactly that way
    once, still showing a letterboxed inset after the film moved to a
    full-bleed push-in.
    """
    still = prepare_still(_load_still(source, source=still_source), width, height)
    return still_card_frame(still, width, height, 0.0)


def _empty_sheet(message: str, output: Path) -> Path:
    """A trip with nothing to show still gets a sheet that SAYS so -- a missing
    file would read as a pack that failed to build."""
    sheet = Image.new("RGB", (sum(TABLE_COLUMN_WIDTHS), TABLE_ROW_HEIGHT * 2), "#101418")
    ImageDraw.Draw(sheet).text((10, TABLE_ROW_HEIGHT - 12), message, fill="#ffffff", font=_font(20))
    sheet.save(output)
    return output


def original_still(source: Path, *, still_source: str = "developed") -> Image.Image:
    """The photo as the camera left it -- the "before" half of a stills row.

    A RAW file is not something Pillow can open, so it takes the same camera
    preview the film's own loader starts from; only when there is none does it
    fall back to the film's rendering (a row whose halves match, never a crash).
    """
    try:
        with Image.open(source) as opened:
            return opened.convert("RGB")
    except (OSError, SyntaxError):
        preview = _camera_preview(source)
        return preview if preview is not None else _load_still(source, source=still_source)


def _still_pair(
    source: Path, width: int, height: int, still_source: str
) -> tuple[str, Image.Image, Image.Image]:
    """One stills row. A photo that cannot be read is SHOWN as such, not raised:
    the pack is evidence for a decision, and one bad file must not cost the
    operator every other sheet (the render itself still refuses it later)."""
    try:
        return (
            source.name,
            original_still(source, still_source=still_source),
            render_still_as_shown(source, width, height, still_source=still_source),
        )
    except (OSError, SyntaxError, ValueError, PhotoDecodeError):
        unreadable = Image.new("RGB", (STILL_TILE_WIDTH, STILL_TILE_HEIGHT), WARNING_RGB)
        return (f"{source.name} -- COULD NOT BE READ", unreadable, unreadable)


def stills_sheet(pairs: Sequence[tuple[str, Image.Image, Image.Image]], output: Path) -> Path:
    """One row per kept photo: its name, the original, and the film's own rendering."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not pairs:
        return _empty_sheet("No photos are placed on this route.", output)
    row_height = STILL_TILE_HEIGHT + LABEL_HEIGHT
    sheet = Image.new("RGB", (STILL_TILE_WIDTH * 2, row_height * len(pairs)), "#101418")
    draw = ImageDraw.Draw(sheet)
    font = _font(max(11, STILL_TILE_WIDTH // 24), bold=True)
    for index, (name, original, as_shown) in enumerate(pairs):
        y = index * row_height
        draw.rectangle((0, y, STILL_TILE_WIDTH * 2, y + LABEL_HEIGHT), fill=(0, 0, 0))
        draw.text((4, y + 3), name, fill="#ffffff", font=font)
        sheet.paste(
            _fit_cover(original.convert("RGB"), STILL_TILE_WIDTH, STILL_TILE_HEIGHT),
            (0, y + LABEL_HEIGHT),
        )
        sheet.paste(
            _fit_cover(as_shown.convert("RGB"), STILL_TILE_WIDTH, STILL_TILE_HEIGHT),
            (STILL_TILE_WIDTH, y + LABEL_HEIGHT),
        )
    sheet.save(output)
    return output


def marker_closeup(
    image: Image.Image, marker_xy: tuple[float, float], *, size: int = MARKER_CROP_PX
) -> Image.Image:
    """A ``size``x``size`` crop centred on the marker.

    Clamped to the frame rather than shrunk when the marker is near an edge:
    a ragged tile would make the sheet's rows stop lining up exactly where the
    map is most cramped. A frame smaller than the crop comes back whole.
    """
    width, height = image.size
    if width <= size or height <= size:
        return image.convert("RGB")
    marker_x, marker_y = marker_xy
    left = min(max(round(marker_x) - size // 2, 0), width - size)
    top = min(max(round(marker_y) - size // 2, 0), height - size)
    return image.convert("RGB").crop((left, top, left + size, top + size))


def _marker_gap_label(gap_px: float | None) -> tuple[str, bool]:
    """The tile's gap caption and whether it is a warning.

    ``None`` is shown as unknown, never as zero: ``line_end_to_marker_px``
    returns it when no line pixel is in range at all (the film's first frame,
    before any line is drawn), and a printed 0.0 would read as a perfect
    result on a frame nothing was measured on.
    """
    if gap_px is None:
        return "gap —", False
    return f"gap {gap_px:.1f}px", gap_px > MARKER_GAP_CEILING_PX


def marker_sheet(tiles: Sequence[tuple[str, Image.Image, float | None]], output: Path) -> Path:
    """A grid of marker close-ups, each captioned with its route clock and the
    measured dot-to-line gap, the over-ceiling ones in the warning colour."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not tiles:
        return _empty_sheet("This route has no marker samples to show.", output)

    columns = min(MARKER_SHEET_COLUMNS, len(tiles))
    rows = math.ceil(len(tiles) / columns)
    cell_height = MARKER_CROP_PX + LABEL_HEIGHT
    sheet = Image.new("RGB", (MARKER_CROP_PX * columns, cell_height * rows), "#101418")
    draw = ImageDraw.Draw(sheet)
    font = _font(max(11, MARKER_CROP_PX // 20), bold=True)
    for index, (clock, image, gap_px) in enumerate(tiles):
        x = (index % columns) * MARKER_CROP_PX
        y = (index // columns) * cell_height
        caption, warn = _marker_gap_label(gap_px)
        draw.rectangle(
            (x, y, x + MARKER_CROP_PX, y + LABEL_HEIGHT),
            fill=WARNING_RGB if warn else (0, 0, 0),
        )
        draw.text((x + 6, y + 6), f"{clock} · {caption}", fill="#ffffff", font=font)
        sheet.paste(
            _fit_cover(image.convert("RGB"), MARKER_CROP_PX, MARKER_CROP_PX),
            (x, y + LABEL_HEIGHT),
        )
    sheet.save(output)
    return output


@dataclass(frozen=True)
class PlacementRow:
    """One row of the placement table: what a file is, where it landed, and why."""

    file_name: str
    kind: str
    status: str
    route_clock: str
    reason: str = ""


def format_route_clock(seconds: float | None) -> str:
    """``mm:ss`` for a route-clock time, or an em dash when there isn't one."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "—"
    total = round(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def classify_placement(
    *,
    is_mapped: bool,
    captured_at_s: float | None,
    track_start_s: float,
    track_end_s: float,
    bookend: str | None = None,
) -> tuple[str, str]:
    """Where a piece of media sits relative to the recorded track, and why.

    ``bookend`` is ``film.classify_bookend``'s own verdict for the photo --
    the single rule for what plays before/after the trail. This function
    never re-derives it from the clock: a photo outside the track that the
    film does NOT play as a bookend is simply not placed.
    """
    if is_mapped:
        return "placed on route", ""
    if bookend == "before_recording":
        return "before the trail", "plays before the first map leg; no route position claimed"
    if bookend == "after_recording":
        return "after the trail", "plays after the last map leg; no route position claimed"
    if captured_at_s is not None and captured_at_s < track_start_s:
        return "not placed", "captured before the recorded track started; not a bookend"
    if captured_at_s is not None and captured_at_s > track_end_s:
        return "not placed", "captured after the recorded track ended; not a bookend"
    return "not placed", "no defensible route position"


def placement_table(rows: Sequence[PlacementRow], output: Path) -> Path:
    """A plain, large-type table image: file, kind, status, clock, reason."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return _empty_sheet("This trip has no photos or video.", output)
    width = sum(TABLE_COLUMN_WIDTHS)
    height = TABLE_HEADER_HEIGHT + TABLE_ROW_HEIGHT * len(rows)
    sheet = Image.new("RGB", (width, height), "#101418")
    draw = ImageDraw.Draw(sheet)
    header_font = _font(20, bold=True)
    body_font = _font(18)
    x = 0
    for name, col_width in zip(TABLE_COLUMNS, TABLE_COLUMN_WIDTHS):
        draw.text((x + 10, 12), name, fill="#ffffff", font=header_font)
        x += col_width
    for row_index, row in enumerate(rows):
        y = TABLE_HEADER_HEIGHT + row_index * TABLE_ROW_HEIGHT
        row_fill = ROW_FILL_BY_STATUS.get(row.status)
        if row_fill is not None:
            draw.rectangle((0, y, width, y + TABLE_ROW_HEIGHT), fill=row_fill)
        text_fill = "#101418" if row_fill is not None else "#ffffff"
        cells = (row.file_name, row.kind, row.status, row.route_clock, row.reason or "—")
        x = 0
        for cell, col_width in zip(cells, TABLE_COLUMN_WIDTHS):
            draw.text((x + 10, y + 16), cell, fill=text_fill, font=body_font)
            x += col_width
    sheet.save(output)
    return output


def write_motion_clips(
    video: Path,
    directory: Path,
    windows: Sequence[tuple[float, float]],
    *,
    runner: Runner = subprocess.run,
) -> list[dict]:
    """Cut one continuous clip per hectic window with a lossless stream copy.

    Each window becomes ``motion-clip-<n>.mp4`` under ``directory`` (``n``
    1-based, in ``windows`` order) via ffmpeg's ``-c copy`` path -- fast, no
    re-encode. ``runner`` is the same injectable-subprocess seam
    ``delivery.check_delivery`` uses, so a test never needs a real ffmpeg.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    for index, (start_s, end_s) in enumerate(windows, start=1):
        name = f"motion-clip-{index}.mp4"
        clip_path = directory / name
        runner(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(start_s),
                "-t",
                str(end_s - start_s),
                "-i",
                str(video),
                "-c",
                "copy",
                str(clip_path),
            ],
            check=True,
            capture_output=True,
        )
        clips.append({"name": name, "start_s": start_s, "end_s": end_s})
    return clips


def write_review_pack(
    directory: Path,
    *,
    opening: Path,
    stills: Path,
    placements: Path,
    selection: dict,
    preflight_sheet: Path | None,
    marker_sheet: Path | None = None,
    motion_clips: Sequence[dict] | None = None,
) -> dict:
    """Write ``review-pack.json`` beside the already-written sheets, and return it.

    Never writes anywhere but ``directory`` -- the sheets are written by the
    caller (typically via ``gather_review_inputs``), and this only records
    their names plus the selection block and derived counts.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    kept_count = len(selection.get("kept", ()))
    dropped_count = sum(len(group.get("losers", ())) for group in selection.get("groups", ()))
    unplaced_count = len(selection.get("not_considered", ()))
    pack = {
        "opening_sheet": Path(opening).name,
        "stills_sheet": Path(stills).name,
        "placement_table": Path(placements).name,
        "preflight_sheet": str(preflight_sheet) if preflight_sheet else None,
        # A full path like preflight_sheet's, not a bare name: both are
        # written by the preflight walk AFTER this JSON, so neither is on
        # disk yet to resolve relative to.
        "marker_sheet": str(marker_sheet) if marker_sheet else None,
        "selection": selection,
        "counts": {"kept": kept_count, "dropped": dropped_count, "unplaced": unplaced_count},
        "motion_clips": list(motion_clips) if motion_clips else [],
    }
    (directory / REVIEW_PACK_JSON).write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return pack


def gather_review_inputs(
    directory: Path,
    *,
    rasters: Sequence[Any],
    first_point: Any,
    labels: Sequence[Any],
    points: Sequence[Any],
    photos: Sequence[Any],
    kept_mapped: Sequence[Any],
    source_video: Path | None,
    video_placed: bool,
    video_clock_s: float | None,
    width: int,
    height: int,
    still_source: str = "developed",
    before_bookends: Sequence[Any] = (),
    after_bookends: Sequence[Any] = (),
    dropped_bookend_names: Collection[str] = (),
) -> dict[str, Path]:
    """Build the three sheets from the film's own already-computed inputs.

    Kept out of ``film.py`` because assembling pairs and placement rows is
    bulky enough to live in its own module;
    ``film.py`` passes through the objects it already has in scope
    (``PhotoMoment``-like photos, ``RoutePoint``-like points, rasters).
    """
    directory = Path(directory)
    frames = []
    for fraction in OPENING_FRACTIONS:
        image = _render_opening_map(rasters, first_point, fraction, width, height, labels)
        _draw_opening(
            image, fraction, opening_credit(rasters, first_point, fraction, width, height)
        )
        frames.append(image)
    opening = opening_sheet(frames, directory / OPENING_SHEET_NAME)

    kept_names = {photo.source.name for photo in kept_mapped}
    # The film's own play order: bookends before, the route's photos, bookends after.
    shown = [*before_bookends, *kept_mapped, *after_bookends]
    pairs = [_still_pair(photo.source, width, height, still_source) for photo in shown]
    bookend_of = {photo.source.name: "before_recording" for photo in before_bookends}
    bookend_of.update({photo.source.name: "after_recording" for photo in after_bookends})
    stills = stills_sheet(pairs, directory / STILLS_SHEET_NAME)

    track_start_s = min(point.time_s for point in points)
    track_end_s = max(point.time_s for point in points)
    rows: list[PlacementRow] = []
    for photo in photos:
        if photo.is_mapped:
            if photo.source.name in kept_names:
                rows.append(
                    PlacementRow(
                        photo.source.name,
                        "photo",
                        "placed on route",
                        format_route_clock(photo.anchor.track_time_s),
                    )
                )
            continue
        if photo.source.name in dropped_bookend_names:
            continue  # a best-pick loser, like a dropped route photo: the receipt names it
        status, reason = classify_placement(
            is_mapped=False,
            captured_at_s=photo.captured_at_exif_s,
            track_start_s=track_start_s,
            track_end_s=track_end_s,
            bookend=bookend_of.get(photo.source.name),
        )
        rows.append(PlacementRow(photo.source.name, "photo", status, "—", reason))
    if source_video is not None:
        if video_placed:
            rows.append(
                PlacementRow(
                    source_video.name,
                    "video",
                    "placed on route",
                    format_route_clock(video_clock_s),
                )
            )
        else:
            status, reason = classify_placement(
                is_mapped=False,
                captured_at_s=video_clock_s,
                track_start_s=track_start_s,
                track_end_s=track_end_s,
            )
            rows.append(PlacementRow(source_video.name, "video", status, "—", reason))
    placements = placement_table(rows, directory / PLACEMENT_TABLE_NAME)

    return {"opening": opening, "stills": stills, "placements": placements}
