"""A minutes-fast QA gate that runs before a Route Story film's long render.

Renders a handful of frames spread across the whole story through the SAME
renderer object and single-frame ``render`` call the long render later drives
-- the real bundle, tiles, labels, marker, HUD path -- checks each through
``frame_qa``'s detectors, and tiles them into one contact-sheet JPEG, so a
visual or tile defect is caught in seconds instead of after hours of encoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .frame_qa import (
    FrameQaError,
    FrameQaResult,
    FrameQaThresholds,
    check_frame,
    check_opening_frame,
    check_opening_ladder,
    line_end_to_marker_px,
)
from .hud import _font
from .layers import MapRenderer
from .layout import hud_layout
from .metrics import RoutePoint
from .review_pack import marker_closeup
from .review_pack import marker_sheet as write_marker_sheet
from .scene_renderer import scene_map_height

THUMB_WIDTH = 320
CONTACT_SHEET_COLUMNS = 4


def map_field_rows(width: int, height: int) -> int:
    """How many rows from the top of a composed frame frame QA judges as map.

    Everything from this row down is HUD -- the dashboard rail, and on
    portrait the platform bottom safe zone under it. The rail's top row comes
    from the same ``hud_layout`` the painter uses, so the judged field and
    the painted rail can never drift apart.
    """
    map_rows = scene_map_height(height)
    rail_top = hud_layout(width, height).dashboard_panel_top
    if rail_top <= 0:
        # A frame shorter than the rail's minimum height (tiny probe frames):
        # the layout is degenerate there, so judge the plain map field rather
        # than masking the whole frame and passing anything.
        return map_rows
    return min(map_rows, rail_top)


class RoutePreflightError(RuntimeError):
    """A preflight frame failed frame QA before the long render started."""


@dataclass(frozen=True)
class PreflightResult:
    frames: int
    results: tuple[FrameQaResult, ...]
    times: tuple[float, ...]
    sheet: Path | None
    # None when run_preflight was not asked for one -- appended last so
    # existing positional callers are unaffected.
    marker_sheet: Path | None = None

    @property
    def passed(self) -> bool:
        return all(result.ok for result in self.results)


def preflight_times(start_time: float, end_time: float, count: int) -> list[float]:
    """Evenly spaced route-clock times across the whole story, endpoints included."""
    if count <= 0:
        return []
    if count == 1:
        return [start_time]
    step = (end_time - start_time) / (count - 1)
    return [start_time + step * index for index in range(count)]


def render_preflight_frames(
    renderer: MapRenderer,
    segments: Sequence[Sequence[RoutePoint]],
    *,
    start_time: float,
    end_time: float,
    width: int,
    height: int,
    count: int,
) -> list[tuple[float, Image.Image]]:
    """Render ``count`` frames through the renderer's single-frame ``render``.

    Deliberately calls ``render`` rather than ``render_leg``: a preflight
    sample is not a contiguous leg, and ``render`` never touches a
    follow-camera's inertia state (only ``render_leg`` carries that across
    calls), so these calls cannot disturb the long render that follows.
    """
    times = preflight_times(start_time, end_time, count)
    return [(at_time, renderer.render(segments, at_time, width, height)) for at_time in times]


def write_contact_sheet(frames: Sequence[tuple[float, Image.Image]], output: Path) -> Path:
    """Tile preflight frames into one captioned JPEG for a human/dispatcher look."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = min(CONTACT_SHEET_COLUMNS, max(1, len(frames)))
    rows = math.ceil(len(frames) / columns) if frames else 1
    thumb_height = 0
    thumbs: list[Image.Image] = []
    for _, image in frames:
        thumb_height = round(THUMB_WIDTH * image.height / image.width)
        thumbs.append(image.convert("RGB").resize((THUMB_WIDTH, thumb_height)))
    sheet = Image.new("RGB", (THUMB_WIDTH * columns, thumb_height * rows), "#101418")
    draw = ImageDraw.Draw(sheet)
    font = _font(max(11, THUMB_WIDTH // 24), bold=True)
    for index, (at_time, _) in enumerate(frames):
        column, row = index % columns, index // columns
        x, y = column * THUMB_WIDTH, row * thumb_height
        sheet.paste(thumbs[index], (x, y))
        caption = f"t={at_time:.1f}s"
        draw.rectangle((x, y, x + THUMB_WIDTH, y + font.size + 6), fill=(0, 0, 0))
        draw.text((x + 4, y + 2), caption, fill="#ffffff", font=font)
    sheet.save(output, "JPEG", quality=88)
    return output


def _write_marker_sheet(
    frames: Sequence[tuple[float, Image.Image]],
    marker_positions: Sequence[tuple[float, float] | None],
    output: Path,
) -> Path:
    """Crop each sampled frame around its marker and tile the close-ups.

    The gap under each tile is measured here by calling the same
    ``line_end_to_marker_px`` ``check_frame`` itself calls, on the same
    frame, rather than being read out of a result object -- so the sheet is
    writable on the failing path too, where no result exists yet, and the
    number still cannot drift from the gate's.

    A frame whose renderer reported no marker position is shown whole and
    captioned as unmeasured; dropping it would silently shorten the walk.
    """
    tiles: list[tuple[str, Image.Image, float | None]] = []
    for index, (at_time, image) in enumerate(frames):
        clock = f"t={at_time:.1f}s"
        marker_xy = marker_positions[index] if index < len(marker_positions) else None
        if marker_xy is None:
            tiles.append((clock, image, None))
            continue
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        tiles.append(
            (clock, marker_closeup(image, marker_xy), line_end_to_marker_px(rgb, marker_xy))
        )
    return write_marker_sheet(tiles, output)


def run_preflight(
    renderer: MapRenderer,
    segments: Sequence[Sequence[RoutePoint]],
    *,
    start_time: float,
    end_time: float,
    width: int,
    height: int,
    count: int,
    output: Path,
    thresholds: FrameQaThresholds | None = None,
    opening_frames: Sequence[Image.Image] = (),
    marker_sheet: Path | None = None,
) -> PreflightResult:
    """Render, QA and tile a preflight sample. Raises before any leg render.

    The contact sheet is written before any QA failure is raised, so a human
    or the dispatcher can see what tripped the gate.

    ``opening_frames``, when given, are checked with
    ``check_opening_frame`` (mean luma / colourfulness / sharpness floors)
    and then, as a set, with ``check_opening_ladder`` (no rung far softer
    than the ladder's own median) after the contact sheet is written -- a
    viewer-legible check for the "dark, fuzzy, unrecognisable opening shot"
    defect that no pixel-fraction detector below catches.

    ``marker_sheet``, when given, also writes a grid of close-ups cropped
    around the marker, one per sampled frame, each captioned with that
    frame's own measured dot-to-line gap. A four-pixel gap -- the marker
    trailing behind the end of its own bright travelled line -- is invisible
    in a whole-frame contact-sheet tile and obvious in a crop. It is written
    from THESE frames and THESE measurements, and it is written whether or
    not the gate passes: the failing run is the one an operator most needs
    to look at.
    """
    frames: list[tuple[float, Image.Image]] = []
    # Captured right after each render() call, never re-read at the end of
    # the loop: last_marker_xy tracks the RENDERER's most recent frame, and
    # the frames it renders are not re-checked until after every one of them
    # has already overwritten it.
    marker_positions: list[tuple[float, float] | None] = []
    for at_time in preflight_times(start_time, end_time, count):
        try:
            image = renderer.render(segments, at_time, width, height)
            frames.append((at_time, image))
            marker_positions.append(getattr(renderer, "last_marker_xy", None))
        except FrameQaError as exc:
            # The renderer judged its own raw capture. Show what it rejected.
            if exc.rgb is not None:
                frames.append((at_time, Image.fromarray(exc.rgb, mode="RGB")))
            sheet = write_contact_sheet(frames, output) if frames else None
            if marker_sheet is not None and frames:
                _write_marker_sheet(frames, marker_positions, marker_sheet)
            raise RoutePreflightError(
                f"preflight frame at route time {at_time:.1f}s failed frame QA: {exc} "
                f"(sheet: {sheet})"
            ) from exc
    if not frames:
        return PreflightResult(frames=0, results=(), times=(), sheet=None)

    sheet = write_contact_sheet(frames, output)
    # Written before the opening-frame and per-frame gates below, for the same
    # reason the contact sheet is: the run an operator most needs to look at
    # is the one that is about to raise.
    written_marker_sheet = (
        _write_marker_sheet(frames, marker_positions, marker_sheet)
        if marker_sheet is not None
        else None
    )
    thresholds = thresholds or FrameQaThresholds()

    opening_rgb = [np.array(image.convert("RGB"), dtype=np.uint8) for image in opening_frames]
    for rgb in opening_rgb:
        problems = check_opening_frame(rgb, thresholds)
        if problems:
            raise RoutePreflightError(
                f"opening frame failed QA: {'; '.join(problems)} (sheet: {sheet})"
            )
    # Per-frame floors pass a ladder whose rungs are individually acceptable
    # and wildly inconsistent with each other -- the "fuzzy when zooming in"
    # defect. Checked after them so a ladder that is uniformly dark or blurred
    # reports that plainer problem first.
    ladder_problems = check_opening_ladder(opening_rgb, thresholds)
    if ladder_problems:
        raise RoutePreflightError(
            f"opening zoom goes soft partway down the imagery ladder: "
            f"{'; '.join(ladder_problems)} (sheet: {sheet})"
        )

    results = []
    # A renderer that checks its own raw captures has already judged
    # void/unloaded/flat-black on these frames; its label pills are a dark
    # fill inside the unloaded-tile tolerance (0.52% of a clean
    # frame), so those detectors are never re-run on a composed frame of
    # its. The marker-geometry checks (line width / line-end distance) DO
    # still run here -- they look only near the marker, not at the whole
    # frame, so the label-pill/rail pixels can't spuriously trip them, and
    # skipping them entirely would mean the one check that catches "the
    # marker lags behind the travelled line" never runs on this renderer.
    checks_raw = getattr(renderer, "checks_raw_frames", False)
    for index, ((at_time, image), marker_xy) in enumerate(zip(frames, marker_positions)):
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        if checks_raw:
            result = check_frame(
                index, rgb, thresholds, marker_xy=marker_xy, marker_geometry_only=True
            )
        else:
            # The detectors judge the MAP field. A renderer hands back the
            # whole composed frame, and the dashboard rail is a dark fill
            # within tolerance of scene.js's "unloaded tile" background,
            # so it alone would read as unloaded on a clean frame. On portrait
            # the rail is lifted clear of the bottom safe zone, so its top rows
            # (and their white value text) sit INSIDE the map field: mask from whichever comes
            # first, the map field's end or the rail's own top row. The
            # denominator stays the map field's own area, so the caps keep
            # their calibration and a masked rail can only ever lower a
            # fraction, never raise one.
            height_px, width_px = rgb.shape[0], rgb.shape[1]
            hud_mask = np.zeros(rgb.shape[:2], dtype=bool)
            hud_mask[map_field_rows(width_px, height_px) :] = True
            result = check_frame(
                index,
                rgb,
                thresholds,
                hud_mask=hud_mask,
                marker_xy=marker_xy,
                field_pixels=scene_map_height(height_px) * width_px,
            )
        results.append(result)
        if not result.ok:
            problems = "; ".join(result.problems)
            raise RoutePreflightError(
                f"preflight frame at route time {at_time:.1f}s failed frame QA: {problems} "
                f"(sheet: {sheet})"
            )
    return PreflightResult(
        frames=len(frames),
        results=tuple(results),
        times=tuple(at_time for at_time, _ in frames),
        sheet=sheet,
        marker_sheet=written_marker_sheet,
    )
