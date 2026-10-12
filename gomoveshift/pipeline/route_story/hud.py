"""Route Story film HUD: overlays, typography, and still-photo preparation.

This module owns everything drawn on top of a frame that is not the map
itself — the dashboard, title cards, recap panel, and photo/video labels —
plus the small unit-conversion formatters those overlays share and the still
loading/fitting helpers (RAW fallback decode, black-border crop, cover/contain
resizing) that turn a source photo into displayable pixels. It does not know
about the map projection, satellite rasters, or frame sequencing — those
belong to ``layers`` and ``assembly``.
"""

from __future__ import annotations

import functools
import io
import logging
import math
import re
import sys
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from ..photos.develop import _ROT90_BY_FLIP, develop
from . import grade, graded_cache
from .dashboard import GRADE_WINDOW_M, PACE_WINDOW_S, DashboardState
from .layout import HudLayout, hud_layout
from .metrics import RoutePoint
from .profile_geometry import ProfileGeometry, ProfilePoint

logger = logging.getLogger(__name__)

_M_PER_MILE = 1609.344
_M_PER_FOOT = 0.3048
_MPS_PER_MPH = 0.44704
_STOPPED_SPEED_MPS = 0.5
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_BOLD_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# The dashboard's clock anchor and motion-cell mode are read here as module
# state, rather than threaded onto ``DashboardState``, mirroring the
# time-compression badge this replaced: the film assembly loop (``film.py``)
# scopes each to the whole render without touching the renderer call chain in
# ``layers.py`` that builds the dashboard for every ``_render_map`` call.
# ``None``/the default draws exactly as before either was ever set.
_active_dashboard_clock_anchor: datetime | None = None
_active_motion_cell_mode: str = "speed"
# The frozen dashboard column set for the whole render -- same module-state
# reasoning as the two above: ``render_leg`` (scene_renderer.py) sets this
# once per leg from that leg's own segment capabilities without threading a
# new parameter through its worker/job call chain. ``None`` (the default,
# and what a lone ``render()``/single-frame call leaves it at) falls back to
# ``_dashboard_cells``'s legacy per-frame dynamic columns.
_active_dashboard_fields: tuple[str, ...] | None = None

# "The trip's moving-average speed" decides PACE vs SPEED once for the whole
# film, not per frame -- an on-foot trip that briefly coasts downhill should
# not flip the label mid-film. 4.5 m/s (~10 mph) is a brisk-run ceiling.
_MOTION_PACE_THRESHOLD_MPS = 4.5

# The speed/grade caps and display units are the same module-state shape as
# the three settings above: the resolved ``ActivityProfile``'s caps and the
# film's resolved units (``route_story/film.py``) are set once for the whole
# render rather than threaded through the renderer call chain. ``None`` (the
# default, and what the lone ``render()``/single-frame path leaves both at)
# is today's behaviour -- an 8.0 m/s speed cap, a 30.0% grade cap, imperial
# units -- unchanged from before either setter existed.
_DEFAULT_SPEED_CAP_MPS = 8.0
_DEFAULT_GRADE_CAP_PERCENT = 30.0


@dataclass(frozen=True)
class DisplayCaps:
    """The HUD's speed/grade display caps for the whole render."""

    speed_cap_mps: float
    grade_cap_percent: float


_active_display_caps: DisplayCaps | None = None
_active_display_units: str | None = None


def set_active_display_caps(caps: DisplayCaps | None) -> None:
    """Set (or clear, with ``None``) the HUD's speed/grade caps for the whole render."""
    global _active_display_caps
    _active_display_caps = caps


def set_active_display_units(units: str | None) -> None:
    """Set (or clear, with ``None``) the HUD's display units: ``"imperial"`` or ``"metric"``."""
    global _active_display_units
    _active_display_units = units


def _speed_cap_mps() -> float:
    return _active_display_caps.speed_cap_mps if _active_display_caps else _DEFAULT_SPEED_CAP_MPS


def _grade_cap_percent() -> float:
    return (
        _active_display_caps.grade_cap_percent
        if _active_display_caps
        else _DEFAULT_GRADE_CAP_PERCENT
    )


# Opening-title text is fitted to this fraction of the frame width. The
# remainder is the safe margin: it absorbs the glyph stroke (drawn outside the
# measured advance) and keeps the card off the edge on a title-safe TV crop.
_TITLE_SAFE_WIDTH_FRACTION = 0.88

_UTC_OFFSET_RE = re.compile(r"([+-])(\d{2}):(\d{2})")


def set_active_dashboard_clock(anchor: datetime | None) -> None:
    """Set (or clear, with ``None``) the local wall-clock time at route-second 0."""
    global _active_dashboard_clock_anchor
    _active_dashboard_clock_anchor = anchor


def set_active_motion_cell_mode(mode: str) -> None:
    """Set the dashboard's single motion cell for the whole render: ``pace`` or ``speed``."""
    global _active_motion_cell_mode
    _active_motion_cell_mode = mode


def set_active_dashboard_fields(fields: tuple[str, ...] | None) -> None:
    """Set (or clear, with ``None``) the frozen dashboard column set for the whole render."""
    global _active_dashboard_fields
    _active_dashboard_fields = fields


def motion_cell_mode(moving_average_speed_mps: float | None) -> str:
    """``"pace"`` for an on-foot trip, else ``"speed"`` -- unknown reads ``"speed"``."""
    if (
        moving_average_speed_mps is not None
        and moving_average_speed_mps <= _MOTION_PACE_THRESHOLD_MPS
    ):
        return "pace"
    return "speed"


_DASHBOARD_CELL_LABELS = {
    "PACE": "RECENT PACE",
    "SPEED": "RECENT SPEED",
}


def _dashboard_cell_label(name: str) -> str:
    """The on-screen label drawn for a dashboard cell's field key.

    Discloses that PACE/SPEED are eased over a trailing window (see
    ``PACE_WINDOW_S`` in ``dashboard.py``) rather than the instantaneous
    edge value. Only the drawn text changes -- ``name`` itself stays the
    field-key string used for column freezing and ``_FIELD_VALUES`` lookups.
    """
    return _DASHBOARD_CELL_LABELS.get(name, name)


def _primary_motion_cell(dashboard: DashboardState) -> tuple[str, str]:
    if _active_motion_cell_mode == "pace":
        return ("PACE", _pace(dashboard.current_speed_mps))
    return ("SPEED", _safe_speed(dashboard.current_speed_mps))


def _time_of_day_label(dashboard: DashboardState) -> str | None:
    """Local time of day at this frame's moment, or ``None`` when no clock is set."""
    if _active_dashboard_clock_anchor is None or dashboard.recording_time_s is None:
        return None
    # The recording clock, never ``elapsed_seconds``: that one skips recorder
    # pauses, so the readout would fall behind the real time after every gap.
    moment = _active_dashboard_clock_anchor + timedelta(seconds=dashboard.recording_time_s)
    hour12 = moment.hour % 12 or 12
    return f"{hour12}:{moment.minute:02d} {'AM' if moment.hour < 12 else 'PM'}"


def parse_utc_offset(text: str) -> timezone:
    """Parse a strict ``+HH:MM``/``-HH:MM`` UTC offset; raise ``ValueError`` otherwise."""
    match = _UTC_OFFSET_RE.fullmatch(text.strip())
    if match is None:
        raise ValueError(f"UTC offset must look like +05:30 or -04:00, got {text!r}")
    sign, hours, minutes = match.group(1), int(match.group(2)), int(match.group(3))
    if hours > 23 or minutes > 59:
        raise ValueError(f"UTC offset out of range: {text!r}")
    delta = timedelta(hours=hours, minutes=minutes)
    return timezone(-delta if sign == "-" else delta)


def _offset_from_capture(wall_local: datetime, capture_utc: datetime) -> str:
    """The raw ``+HH:MM``/``-HH:MM`` EXIF offset implied by a photo's two capture clocks."""
    total_minutes = round((wall_local - capture_utc.replace(tzinfo=None)).total_seconds() / 60)
    sign = "-" if total_minutes < 0 else "+"
    total_minutes = abs(total_minutes)
    return f"{sign}{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def resolve_dashboard_utc_offset(
    exif_offsets: Sequence[str | None], cli_offset: str | None
) -> tuple[str | None, str]:
    """Resolve the dashboard TIME cell's UTC offset, never guessing.

    Prefers the offset every zoned photo agrees on; a disagreement is logged
    and falls through to an explicit ``--utc-offset``; absent both, the TIME
    cell is omitted. Returns ``(offset, source)`` where ``source`` is one of
    ``"exif_offset"``, ``"cli_offset"``, ``"omitted"``.
    """
    known = {offset for offset in exif_offsets if offset is not None}
    if len(known) == 1:
        return next(iter(known)), "exif_offset"
    if len(known) > 1:
        logger.warning("Route Story photos disagree on UTC offset: %s", sorted(known))
    if cli_offset is not None:
        return cli_offset, "cli_offset"
    return None, "omitted"


def dashboard_receipt(
    motion_mode: str, time_cell_source: str, utc_offset: str | None
) -> dict[str, object]:
    """The export receipt's ``"dashboard"`` block, matching what was actually drawn."""
    return {
        "motion_cell": motion_mode,
        "time_cell": time_cell_source,
        "utc_offset": utc_offset,
        "pace_window_s": PACE_WINDOW_S,
        "grade_window_m": GRADE_WINDOW_M,
    }


def crop_black_border(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop an unambiguously black frame edge without fabricating replacement pixels.

    A photograph with merely dark content must remain intact.  Therefore an
    edge is removable only when virtually every pixel on the full row or
    column is near digital black.  The returned box is relative to the source
    image and is retained by the caller for the export receipt.
    """
    rgb = np.asarray(image.convert("RGB"))
    luma = (
        rgb[:, :, 0].astype(np.uint16) * 54
        + rgb[:, :, 1].astype(np.uint16) * 183
        + rgb[:, :, 2].astype(np.uint16) * 19
    ) // 256
    dark = luma <= 8

    def edge_count(rows: np.ndarray) -> int:
        count = 0
        for row in rows:
            if float(np.mean(row)) < 0.99:
                break
            count += 1
        return count

    top = edge_count(dark)
    bottom = edge_count(dark[::-1])
    left = edge_count(dark.T)
    right = edge_count(dark.T[::-1])
    if top + bottom >= image.height - 2 or left + right >= image.width - 2:
        return image, (0, 0, image.width, image.height)
    box = (left, top, image.width - right, image.height - bottom)
    if box == (0, 0, image.width, image.height):
        return image, box
    return image.crop(box), box


def _smoothstep(value: float) -> float:
    value = min(1.0, max(0.0, value))
    return value * value * (3.0 - 2.0 * value)


def _draw_opening(image: Image.Image, fraction: float, credit: str) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    draw.rectangle((0, 0, width, height), fill=(3, 12, 17, int(105 * (1 - fraction))))
    subfont = _font(max(12, width // 70))
    # Outlined: trip-day imagery can put white cloud under the credit.
    draw.text(
        (width * 0.07, height - subfont.size * 2),
        credit,
        font=subfont,
        fill=(255, 255, 255, 235),
        stroke_width=max(1, subfont.size // 9),
        stroke_fill=(3, 12, 17, 235),
    )


# A place name only captions the ride when the track actually ran near it.
# The labels file carries every name in the regional bbox, so "nearest label"
# with no ceiling would caption a Smokies ride with a town two counties away.
OPENING_TITLE_MAX_LABEL_KM = 25.0
# The title is clear of the screen before the opening camera lands, so the
# first map leg never starts underneath it.
OPENING_TITLE_FADE_IN_END = 0.10
OPENING_TITLE_FADE_OUT_START = 0.72
OPENING_TITLE_FADE_OUT_END = 0.95
_EARTH_RADIUS_KM = 6371.0088


def _km_between(first: tuple[float, float], second: tuple[float, float]) -> float:
    """Great-circle kilometres between two (latitude, longitude) pairs."""
    lat1, lon1 = math.radians(first[0]), math.radians(first[1])
    lat2, lon2 = math.radians(second[0]), math.radians(second[1])
    haversine = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(haversine)))


def _nearest_place(labels: Sequence[Any], where: tuple[float, float]) -> str:
    """The closest label name within the ceiling, or ``""`` when none is."""
    best_name, best_km = "", OPENING_TITLE_MAX_LABEL_KM
    for label in labels:
        km = _km_between(where, (label.latitude, label.longitude))
        if km <= best_km:
            best_name, best_km = label.name, km
    return best_name


def resolve_opening_title(
    operator_title: str | None,
    labels: Sequence[Any],
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    recorded_on: date | None,
) -> tuple[str, str]:
    """The opening card's two lines: ``(title, subtitle)``.

    The operator's own ``--title`` is the trip's name and is used verbatim --
    never edited, appended to, or reordered. Absent one, the title is derived
    from places the track demonstrably ran near plus the day it was recorded.
    Nothing here is invented: no close place means no place line at all, and
    an unknown recording date means no date line, rather than a guess.
    """
    subtitle = (
        f"{recorded_on.day} {recorded_on.strftime('%B %Y')}" if recorded_on is not None else ""
    )
    if operator_title and operator_title.strip():
        return operator_title.strip(), subtitle
    first, last = _nearest_place(labels, start), _nearest_place(labels, end)
    if first and last and first != last:
        return f"{first} → {last}", subtitle
    return (first or last), subtitle


def draw_opening_title(image: Image.Image, fraction: float, title: str, subtitle: str) -> None:
    """Fade the trip's name over the opening camera, clear before it lands."""
    if not title and not subtitle:
        return
    if fraction <= 0.0:
        alpha = 0.0
    elif fraction < OPENING_TITLE_FADE_IN_END:
        alpha = _smoothstep(fraction / OPENING_TITLE_FADE_IN_END)
    elif fraction < OPENING_TITLE_FADE_OUT_START:
        alpha = 1.0
    else:
        span = OPENING_TITLE_FADE_OUT_END - OPENING_TITLE_FADE_OUT_START
        alpha = 1.0 - _smoothstep((fraction - OPENING_TITLE_FADE_OUT_START) / span)
    opacity = int(round(255 * max(0.0, min(1.0, alpha))))
    if opacity == 0:
        return
    # The card is drawn opaque onto its own transparent layer and that whole
    # layer is faded, rather than passing an alpha in ``fill``: on an RGB
    # image ``ImageDraw.text`` ignores the fill's alpha channel outright and
    # draws at full strength, which turned this fade into a hard cut. (Filled
    # rectangles, like the opening's own dimming wash, DO honour it -- so the
    # bug is invisible until a text overlay is the thing being faded.)
    width, height = image.size
    layout = hud_layout(width, height)
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    y = layout.title_y
    # The card is drawn centred, so text wider than the frame is clipped at BOTH
    # edges and reads as a mangled title rather than a long one. Fit to the safe
    # width before drawing; the stroke sits outside the measured advance, so the
    # margin below leaves room for it at the largest stroke the title can take.
    safe_width = layout.title_safe_width
    if title:
        font = _fitted_value_font(
            draw, title, _font(max(24, width // 22), bold=True), safe_width, bold=True
        )
        draw.text(
            (width / 2, y),
            title,
            font=font,
            fill=(255, 255, 255, 255),
            anchor="ma",
            stroke_width=max(2, font.size // 12),
            stroke_fill=(3, 12, 17, 255),
        )
        y += font.size * 1.25
    if subtitle:
        # Measured uppercased, because that is what gets drawn.
        subfont = _fitted_value_font(
            draw, subtitle.upper(), _font(max(14, width // 52)), safe_width, bold=False
        )
        draw.text(
            (width / 2, y),
            subtitle.upper(),
            font=subfont,
            fill=(245, 183, 59, 255),
            anchor="ma",
            stroke_width=max(1, subfont.size // 10),
            stroke_fill=(3, 12, 17, 255),
        )
    if opacity < 255:
        overlay.putalpha(overlay.getchannel("A").point(lambda value: value * opacity // 255))
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert(image.mode))


# The summit is a moment on the route, not permanent chrome: the badge rides
# with the marker for a beat either side of the highest recorded fix.
HIGH_POINT_CALLOUT_SECONDS = 2.0


def high_point_callout(points: Sequence[RoutePoint]) -> tuple[float, str] | None:
    """``(time_s, text)`` for the track's highest recorded fix, or ``None``.

    A track with no elevation data gets no callout rather than a zero-foot
    one -- the film reports what the GPS recorded or says nothing.
    """
    known = [point for point in points if point.elevation_m is not None]
    if not known:
        return None
    highest = max(known, key=lambda point: (point.elevation_m or 0.0, -point.time_s))
    feet = round((highest.elevation_m or 0.0) / _M_PER_FOOT)
    return highest.time_s, f"HIGH POINT · {feet:,} ft"


def _badge(
    image: Image.Image,
    label: str,
    width: int,
    height: int,
    *,
    top: float,
    outline: str,
) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    font = _font(max(15, width // 70), bold=True)
    text_width = draw.textlength(label, font=font)
    x0 = (width - text_width) / 2 - 18
    y0 = top
    draw.rounded_rectangle(
        (x0, y0, x0 + text_width + 36, y0 + font.size + 18),
        radius=10,
        fill=(4, 14, 18, 215),
        outline=outline,
        width=2,
    )
    draw.text((width / 2, y0 + 9), label, font=font, fill="white", anchor="ma")


def draw_high_point(image: Image.Image, label: str, width: int, height: int) -> None:
    """The summit badge -- a different accent from a media arrival, and lower
    on the frame, so the two never read as the same event."""
    layout = hud_layout(width, height)
    _badge(image, label, width, height, top=layout.badge_summit_top, outline="#7fd4e8")


def _draw_media_arrival(image: Image.Image, label: str, width: int, height: int) -> None:
    layout = hud_layout(width, height)
    _badge(image, label, width, height, top=layout.badge_arrival_top, outline="#f5b73b")


# The maker's mark burned on every delivered frame. Small and translucent on
# purpose -- an unintrusive credit, not a second logo competing with the map
# or the photo it sits over.
DEFAULT_WATERMARK_TEXT = "Made with GoMoveShift"
_WATERMARK_HEIGHT_FRACTION = 0.016
_WATERMARK_OPACITY = 140  # ~55% of 255


def draw_watermark(
    image: Image.Image, text: str, width: int, height: int
) -> tuple[int, int, int, int]:
    """Burn ``text`` bottom-right, inside the title-safe width, and return the
    pixel box it occupies (left, top, right, bottom) so a caller can exclude
    it from frame QA. A falsy ``text`` draws nothing and returns an empty box.
    """
    if not text:
        return (0, 0, 0, 0)
    layout = hud_layout(width, height)
    draw = ImageDraw.Draw(image, "RGBA")
    size = max(9, round(height * _WATERMARK_HEIGHT_FRACTION))
    font = _fitted_value_font(draw, text, _font(size), layout.title_safe_width, bold=False)
    stroke_width = max(1, font.size // 10)
    x, y = layout.watermark_x, layout.watermark_y
    draw.text(
        (x, y),
        text,
        font=font,
        fill=(255, 255, 255, _WATERMARK_OPACITY),
        stroke_width=stroke_width,
        stroke_fill=(3, 12, 17, _WATERMARK_OPACITY),
        anchor=layout.watermark_anchor,
    )
    return draw.textbbox(
        (x, y), text, font=font, stroke_width=stroke_width, anchor=layout.watermark_anchor
    )


def _draw_dashboard(
    image: Image.Image,
    dashboard: DashboardState,
    credit: str,
    fields: Sequence[str] | None = None,
) -> None:
    """Paint the dashboard rail.

    ``fields`` is the film-frozen column set from ``dashboard_fields_for``:
    every column stays in the same order and at the same x-position for
    every frame of the film, and a single value font is fit once for all of
    them rather than per cell per frame. Left ``None``, this falls back to
    ``set_active_dashboard_fields``'s module-level value (set once per leg by
    ``render_leg`` -- see the module docstring by ``_active_dashboard_fields``
    for why that is module state rather than a threaded parameter here); a
    lone ``render()``/single-frame call that never set it gets the legacy
    per-frame dynamic columns.
    """
    resolved_fields = fields if fields is not None else _active_dashboard_fields
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    layout = hud_layout(width, height)
    panel_h = layout.dashboard_panel_height
    # ``dashboard_panel_top`` is the one resolved row shared with preflight's
    # frame-QA mask: the rail is painted from exactly the row the map-field
    # detectors stop judging, so its white value text can never read as
    # page-white again.
    top = layout.dashboard_panel_top
    bottom = top + panel_h
    # ``rectangle`` draws its bottom coordinate INCLUSIVE, so ``bottom - 1``
    # is the panel's true last row -- ``bottom`` itself (and everything below
    # it, on portrait) is the platform-chrome safe zone and must stay untouched.
    draw.rectangle((0, top, width, bottom - 1), fill=(5, 15, 20, 235))
    label = _font(max(10, width // 95), bold=True)
    small = _font(max(9, width // 112))
    cells = _dashboard_cells(dashboard, resolved_fields)
    left, right = layout.dashboard_left, layout.dashboard_right
    cell_w = (right - left) / len(cells)
    frozen_value_font = (
        _dashboard_value_font(draw, resolved_fields, dashboard, cell_w - 14)
        if resolved_fields is not None
        else None
    )
    value = _dashboard_base_value_font(width)
    for index, (name, content) in enumerate(cells):
        x = int(left + index * cell_w)
        draw.text((x, top + 11), _dashboard_cell_label(name), font=label, fill="#9ab0ba")
        fitted = frozen_value_font or _fitted_value_font(draw, content, value, cell_w - 14)
        draw.text((x, top + 12 + label.size), content, font=fitted, fill="#ffffff")
    lower = f"AVG {_safe_speed(dashboard.moving_average_speed_mps)}  •  ASCENT {_feet(dashboard.ascent_m)}  •  DESCENT {_feet(dashboard.descent_m)}  •  ELAPSED {_duration(dashboard.elapsed_seconds)}"
    draw.text((left, bottom - small.size - 11), lower, font=small, fill="#d8e2e6")
    draw.text((right, bottom - small.size - 11), credit, font=small, fill="#9ab0ba", anchor="ra")


def _dashboard_base_value_font(width: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """The rail's value font before any fit-to-cell shrinking."""
    return _font(max(16, width // 48), bold=True)


def _dashboard_value_font(
    draw: ImageDraw.ImageDraw,
    fields: Sequence[str],
    dashboard: DashboardState,
    max_width: float,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """One value font for every frozen column, for the whole film.

    Fit once against the widest content each column will plausibly need,
    never against a single frame's actual value -- a frame refitting its own
    font is what let columns visually reflow mid-film. DISTANCE is measured
    against the film's total (constant) distance rather than the current
    frame's travelled distance, since travelled distance only grows.

    The fit starts from the rail's base value font -- the same one the
    unfrozen path shrinks from -- and only ever shrinks: freezing the size
    must never enlarge the text past the panel it is drawn in.
    """
    base = _dashboard_base_value_font(draw.im.size[0])
    widest = base
    for name in fields:
        content = _dashboard_cells(
            replace(dashboard, distance_traveled_m=dashboard.distance_total_m), (name,)
        )[0][1]
        fitted = _fitted_value_font(draw, content, base, max_width)
        if fitted.size < widest.size:
            widest = fitted
    return widest


def _fitted_value_font(
    draw: ImageDraw.ImageDraw,
    content: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: float,
    *,
    bold: bool = True,
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """``font``, shrunk just enough that ``content`` fits within ``max_width``.

    ``bold`` must match the weight of the font handed in: a shrunk font is built
    fresh from ``_font``, so fitting a regular-weight caption with the default
    would silently re-draw it bold.
    """
    measured = draw.textlength(content, font=font)
    if measured <= max_width:
        return font
    # Width does not scale perfectly linearly with point size (hinting, integer
    # advances), so the linear estimate is a starting point that is then walked
    # down until it genuinely measures short enough.
    fitted = _font(max(13, int(font.size * max_width / measured)), bold=bold)
    while fitted.size > 13 and draw.textlength(content, font=fitted) > max_width:
        fitted = _font(fitted.size - 1, bold=bold)
    return fitted


_FIELD_VALUES: dict[str, Callable[[DashboardState], str]] = {
    "SPEED": lambda d: _safe_speed(d.current_speed_mps),
    "PACE": lambda d: _pace(d.current_speed_mps),
    "ELEVATION": lambda d: _feet(d.current_elevation_m),
    "GRADE": lambda d: _moving_grade(d.current_grade_percent, d.current_speed_mps),
    "DISTANCE": lambda d: f"{d.distance_traveled_m / _M_PER_MILE:.1f} mi",
    "HR": lambda d: _heart_rate(d.current_heart_rate_bpm),
    "CAD": lambda d: _cadence(d.current_cadence_rpm),
    "TEMP": lambda d: _temperature(d.current_temperature_c),
    "TIME": lambda d: _time_of_day_label(d) or "—",
}


# Which of HR/CAD/TEMP a given segments object carries, keyed by identity --
# a film calls this once per rendered frame (the Pillow ``_render_map`` path
# has no per-leg object to cache it on itself), and the same segments object
# is reused across thousands of frames, so scanning every point again each
# time would be a real per-frame cost for a long day's track. Bounded like
# ``dashboard._INDEX_CACHE`` so a long export cycling many distinct legs
# can't grow this without limit, and -- like it -- each entry keeps the
# segments object and a hit requires ``is`` identity, because a bare
# ``id()`` key would serve a freed leg's answer to the next leg allocated
# at the same address.
_SEGMENT_CAPABILITY_CACHE: dict[
    int, tuple[Sequence[Sequence[RoutePoint]], tuple[bool, bool, bool]]
] = {}
_SEGMENT_CAPABILITY_CACHE_MAX = 8


def _segment_capabilities(segments: Sequence[Sequence[RoutePoint]]) -> tuple[bool, bool, bool]:
    key = id(segments)
    cached = _SEGMENT_CAPABILITY_CACHE.get(key)
    if cached is not None and cached[0] is segments:
        return cached[1]
    has_hr = has_cad = has_temp = False
    for segment in segments:
        for point in segment:
            has_hr = has_hr or point.heart_rate_bpm is not None
            has_cad = has_cad or point.cadence_rpm is not None
            has_temp = has_temp or point.temperature_c is not None
    result = (has_hr, has_cad, has_temp)
    if key not in _SEGMENT_CAPABILITY_CACHE and (
        len(_SEGMENT_CAPABILITY_CACHE) >= _SEGMENT_CAPABILITY_CACHE_MAX
    ):
        _SEGMENT_CAPABILITY_CACHE.pop(next(iter(_SEGMENT_CAPABILITY_CACHE)))
    _SEGMENT_CAPABILITY_CACHE[key] = (segments, result)
    return result


def dashboard_fields_for(segments: Sequence[Sequence[RoutePoint]]) -> tuple[str, ...]:
    """The dashboard's column set for a whole film, decided ONCE.

    A column is decided from what the film's segments actually carry, never
    from a single frame's telemetry: HR/CAD/TEMP appear when ANY point in ANY
    segment carries that reading, so a sensor dropout on one frame never
    removes or reflows a column (``_dashboard_cells`` renders "—" there
    instead). TIME appears when a wall-clock anchor is active for this film
    (``set_active_dashboard_clock``) -- not a segment capability, but frozen
    the same way for the same reason.
    """
    fields = [
        "PACE" if _active_motion_cell_mode == "pace" else "SPEED",
        "ELEVATION",
        "GRADE",
        "DISTANCE",
    ]
    has_hr, has_cad, has_temp = _segment_capabilities(segments)
    if has_hr:
        fields.append("HR")
    if has_cad:
        fields.append("CAD")
    if has_temp:
        fields.append("TEMP")
    if _active_dashboard_clock_anchor is not None:
        fields.append("TIME")
    return tuple(fields)


def _dashboard_cells(
    dashboard: DashboardState, fields: Sequence[str] | None = None
) -> list[tuple[str, str]]:
    """Telemetry cells for one frame.

    ``fields`` is the film-frozen column set from ``dashboard_fields_for``:
    every named column renders in that exact order, "—" standing in for a
    frozen column with no value on this particular frame. Without it (legacy
    / single-frame callers with no frozen set) a column is included only when
    THIS frame's own telemetry carries it -- the per-frame dynamic behavior a
    frozen film call site must not use.
    """
    if fields is not None:
        return [(name, _FIELD_VALUES[name](dashboard)) for name in fields]
    cells = [
        _primary_motion_cell(dashboard),
        ("ELEVATION", _feet(dashboard.current_elevation_m)),
        ("GRADE", _moving_grade(dashboard.current_grade_percent, dashboard.current_speed_mps)),
        ("DISTANCE", f"{dashboard.distance_traveled_m / _M_PER_MILE:.1f} mi"),
    ]
    if dashboard.current_heart_rate_bpm is not None:
        cells.append(("HR", _heart_rate(dashboard.current_heart_rate_bpm)))
    if dashboard.current_cadence_rpm is not None:
        cells.append(("CAD", _cadence(dashboard.current_cadence_rpm)))
    if dashboard.current_temperature_c is not None:
        cells.append(("TEMP", _temperature(dashboard.current_temperature_c)))
    time_label = _time_of_day_label(dashboard)
    if time_label is not None:
        cells.append(("TIME", time_label))
    return cells


def draw_profile(
    image: Image.Image,
    geometry: ProfileGeometry,
    at_time_s: float,
    layout: HudLayout,
) -> None:
    """Draw the segmented distance/elevation profile and its synced cursor.

    Each ``ProfileSegment`` is its own polyline -- a gap between source
    segments (or an unknown-elevation stretch) is a visible break, never a
    joined synthetic climb. The portion already travelled (distance <= the
    cursor) gets a filled area under it; the remainder is line only. The
    cursor position comes from ``geometry.cursor_distance_m`` -- the same
    recording clock the map marker uses -- so the two never drift apart.
    """
    if not geometry.segments or geometry.total_distance_m <= 0:
        return
    draw = ImageDraw.Draw(image, "RGBA")
    x, y = layout.profile_left, layout.profile_top
    width = layout.profile_right - layout.profile_left
    height = layout.profile_height
    low, span = geometry.min_elevation_m, geometry.elevation_span_m
    total = geometry.total_distance_m
    cursor_distance = geometry.cursor_distance_m(at_time_s)

    def _xy(point: ProfilePoint) -> tuple[float, float]:
        return (
            x + point.distance_m / total * width,
            y + height - (point.elevation_m - low) / span * height,
        )

    for segment in geometry.segments:
        coords = [_xy(point) for point in segment.points]
        draw.line(coords, fill=(117, 158, 169, 180), width=2)
        completed = [
            (px, py)
            for (px, py), point in zip(coords, segment.points)
            if point.distance_m <= cursor_distance
        ]
        if len(completed) >= 2:
            area = [(px, y + height) for px, _ in completed] + [
                (px, py) for px, py in reversed(completed)
            ]
            draw.polygon(area, fill=(117, 158, 169, 60))

    cursor_x = x + min(1.0, max(0.0, cursor_distance / total)) * width
    draw.line((cursor_x, y, cursor_x, y + height), fill=(245, 183, 59, 255), width=2)

    small = _font(max(9, layout.width // 112))
    draw.text(
        (x + 4, y + 2), _feet(geometry.max_elevation_m), font=small, fill="#9ab0ba", anchor="la"
    )
    draw.text(
        (x + 4, y + height - 2),
        _feet(geometry.min_elevation_m),
        font=small,
        fill="#9ab0ba",
        anchor="lb",
    )


# Shown on a still whose clock falls where the GPS track has nothing to anchor
# it to. It stays off the map rather than borrow the nearest point, and the
# caption says so in the film's own voice -- "UNLOCATED" was an audit status.
UNPLACED_PHOTO_CAPTION = "From the trip • not placed on the route map"


def _draw_photo_label(image: Image.Image, label: str) -> None:
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    layout = hud_layout(width, height)
    x, y = layout.photo_label_x, layout.photo_label_y
    font = _font(max(12, width // 70), bold=True)
    draw.rounded_rectangle(
        (x, y, x + font.getlength(label) + 28, y + font.size + 18),
        radius=10,
        fill=(4, 14, 18, 205),
    )
    draw.text((x + 14, y + 8), label, font=font, fill="#f5d48b")


RECAP_TITLE = "THE DAY, AT A GLANCE"
# The share of the recap spent fading the numbers in. Fading across all of it
# left them fully lit on the last frame only, with no time to read them.
RECAP_FADE_SHARE = 0.15


def _recap_cells(summary) -> list[tuple[str, str]]:
    return [
        ("DISTANCE", f"{summary.distance_m / _M_PER_MILE:.1f} mi"),
        ("ASCENT", _feet(summary.ascent_m)),
        ("TIME", _duration(summary.elapsed_seconds)),
        ("MOVING AVG", _speed(summary.moving_average_speed_mps)),
    ]


def _draw_recap(image: Image.Image, summary, fraction: float) -> None:
    """The day's numbers, in the rail the dashboard used for the whole film.

    The recap map has no dashboard, so its rail was an empty bar while a panel
    laid over the map hid the start of the route. The rail holds the numbers
    instead: the map stays clear edge to edge and the frame keeps the film's
    one layout. ``fraction`` is the recap's progress: the text fades in over
    its first ``RECAP_FADE_SHARE`` and then holds fully lit to the end.
    """
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    layout = hud_layout(width, height)
    top = height - layout.dashboard_panel_height
    alpha = max(0, min(255, round(255 * fraction / RECAP_FADE_SHARE)))
    left, right = layout.dashboard_left, layout.dashboard_right
    middle = (top + height) // 2

    title = _font(max(16, width // 48), bold=True)
    title_room = width * 0.31
    measured = draw.textlength(RECAP_TITLE, font=title)
    if measured > title_room:
        title = _font(max(12, int(title.size * title_room / measured)), bold=True)
    draw.text((left, middle), RECAP_TITLE, font=title, fill=(245, 212, 139, alpha), anchor="lm")

    label = _font(max(10, width // 95), bold=True)
    value = _font(max(16, width // 48), bold=True)
    cells = _recap_cells(summary)
    cells_left = int(width * 0.40)
    cell_w = (right - cells_left) / len(cells)
    block_top = middle - (label.size + 4 + value.size) // 2
    for index, (name, content) in enumerate(cells):
        x = int(cells_left + index * cell_w)
        draw.text((x, block_top), name, font=label, fill=(154, 176, 186, alpha))
        fitted = value
        content_w = draw.textlength(content, font=value)
        if content_w > cell_w - 14:
            fitted = _font(max(13, int(value.size * (cell_w - 14) / content_w)), bold=True)
        draw.text(
            (x, block_top + label.size + 4), content, font=fitted, fill=(255, 255, 255, alpha)
        )

    rail = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(rail).rectangle((0, top, width, height), fill=(5, 15, 20, 235))
    composed = Image.alpha_composite(Image.alpha_composite(image.convert("RGBA"), rail), overlay)
    image.paste(composed.convert("RGB"))


UNPLACED_VIDEO_SUBTITLE = "Original video • not placed on the route map"


def _unlocated_video_card(
    width: int,
    height: int,
    subtitle: str = UNPLACED_VIDEO_SUBTITLE,
    backdrop: Image.Image | None = None,
) -> Image.Image:
    """The title card ahead of the clip; ``backdrop`` is a frame-sized picture to set it on."""
    if backdrop is None:
        image = Image.new("RGB", (width, height), "#09141a")
    else:
        image = backdrop.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    title = _font(max(22, width // 32), bold=True)
    text = _font(max(13, width // 65))
    outline = (3, 12, 17, 235)
    draw.text(
        (width * 0.09, height * 0.38),
        "TRIP INTERLUDE",
        font=title,
        fill="#f5d48b",
        stroke_width=max(1, title.size // 12),
        stroke_fill=outline,
    )
    draw.text(
        (width * 0.09, height * 0.38 + title.size + 12),
        subtitle,
        font=text,
        fill="white",
        stroke_width=max(1, text.size // 9),
        stroke_fill=outline,
    )
    return image


def _photo_frame(
    foreground: Image.Image, background: Image.Image, width: int, height: int
) -> Image.Image:
    shade = Image.new("RGBA", (width, height), (2, 9, 12, 105))
    image = Image.alpha_composite(background.convert("RGBA"), shade).convert("RGB")
    image.paste(foreground, ((width - foreground.width) // 2, (height - foreground.height) // 2))
    return image


# A still whose aspect ratio is close enough to the frame's goes full-bleed
# instead of behind a letterbox card. Measured as min(ratio)/max(ratio), so
# 1.0 is an exact match; a 3:2 or 4:3 still in a 16:9 frame both clear this,
# a portrait one does not.
STILL_FULL_BLEED_MIN_ASPECT_MATCH = 0.7

# The delivered push-in grows from 1.0x to this across a still's whole hold.
STILL_PUSH_IN_MAX_SCALE = 1.06

# The backdrop blur radius scales with frame width the same way the old
# per-photo blur did.
_BACKDROP_BLUR_WIDTH_DIVISOR = 80


def _ease_in_out(fraction: float) -> float:
    """Smoothstep: slow at both ends of the hold, fastest in the middle."""
    clamped = min(1.0, max(0.0, fraction))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def _zoomed_resize(source: Image.Image, width: int, height: int, scale: float) -> Image.Image:
    """Cover-fit ``source`` to ``(width, height)`` at push-in ``scale``.

    A single resize with a float ``box`` -- the crop region shrinks toward
    the centre as ``scale`` grows -- so the zoom is sub-pixel accurate from
    one decoded source instead of stepping through prefiltered sizes.
    """
    source_width, source_height = source.size
    target_ratio = width / height
    source_ratio = source_width / source_height
    if source_ratio > target_ratio:
        base_height = float(source_height)
        base_width = source_height * target_ratio
    else:
        base_width = float(source_width)
        base_height = source_width / target_ratio
    crop_width = base_width / scale
    crop_height = base_height / scale
    center_x, center_y = source_width / 2, source_height / 2
    box = (
        center_x - crop_width / 2,
        center_y - crop_height / 2,
        center_x + crop_width / 2,
        center_y + crop_height / 2,
    )
    return source.resize((width, height), Image.Resampling.LANCZOS, box=box)


# A still is bounded once per photo to this multiple of the pixels its card
# actually uses -- headroom over ``STILL_PUSH_IN_MAX_SCALE`` so even the fully
# pushed-in frame is a downscale, never a stretch.
STILL_PREPARE_MARGIN = 1.25


def _is_full_bleed(still_ratio: float, frame_ratio: float) -> bool:
    match = min(still_ratio, frame_ratio) / max(still_ratio, frame_ratio)
    return match >= STILL_FULL_BLEED_MIN_ASPECT_MATCH


def prepare_still(still: Image.Image, width: int, height: int) -> Image.Image:
    """Bound a decoded still to the pixels ``still_card_frame`` will sample.

    Sized to what the card COVERS (the full frame for a full-bleed still,
    the full frame height for a portrait one), not what fits inside a box --
    a contain-fit would leave a 4:3 photo short of the frame's width and
    stretch it. Never enlarges a still smaller than that.
    """
    if _is_full_bleed(still.width / still.height, width / height):
        scale = max(width / still.width, height / still.height)
    else:
        scale = height / still.height
    scale = min(1.0, scale * STILL_PREPARE_MARGIN)
    if scale >= 1.0:
        return still
    return still.resize(
        (max(1, round(still.width * scale)), max(1, round(still.height * scale))),
        Image.Resampling.LANCZOS,
    )


def still_card_frame(still: Image.Image, width: int, height: int, progress: float) -> Image.Image:
    """The single recipe for what an inserted still looks like at ``progress``
    (0.0 -> 1.0) through its on-screen hold.

    A landscape-ish still (aspect ratio close to the frame's, per
    ``STILL_FULL_BLEED_MIN_ASPECT_MATCH``) fills the frame edge to edge --
    no blurred backdrop, no shade. A portrait/narrow still keeps the
    blurred, shaded backdrop, but the foreground fills the full frame
    height (no inset). Either way the still pushes in smoothly from 1.0x to
    ``STILL_PUSH_IN_MAX_SCALE`` across the hold, eased in and out.
    """
    scale = 1.0 + (STILL_PUSH_IN_MAX_SCALE - 1.0) * _ease_in_out(progress)
    frame_ratio = width / height
    still_ratio = still.width / still.height
    if _is_full_bleed(still_ratio, frame_ratio):
        return _zoomed_resize(still, width, height, scale)

    foreground_height = height
    foreground_width = max(1, round(height * still_ratio))
    foreground = _zoomed_resize(still, foreground_width, foreground_height, scale)
    background = _fit_cover(still, width, height).filter(
        ImageFilter.GaussianBlur(radius=max(8, width // _BACKDROP_BLUR_WIDTH_DIVISOR))
    )
    return _photo_frame(foreground, background, width, height)


STILL_SOURCES = ("developed", "camera")
# A camera preview narrower than this is a thumbnail, not a picture to hold on screen.
CAMERA_PREVIEW_MIN_WIDTH = 1920
# Every inserted still is graded (haze removal, exposure, vibrance, lens-corner
# crop) by default; a test of unrelated behaviour may disable it for exact
# ungraded pixels.
_GRADE_STILLS = True


def _load_still(path: Path, *, source: str = "developed") -> Image.Image:
    """The graded still for ``path``, from the disk cache when this machine has
    graded these exact bytes before (see ``graded_cache``): the grade costs
    ~55 s for a 20 MP camera still, and a render asks for each photo twice --
    once to choose between look-alikes, once for its beat."""
    if source not in STILL_SOURCES:
        raise ValueError(f"unknown still source {source!r}; expected one of {STILL_SOURCES}")
    cached_path = graded_cache.cache_path(
        path,
        f"still:{source}:{_GRADE_STILLS}",
        graded_cache.code_identity(grade, sys.modules[__name__], sys.modules[develop.__module__]),
    )
    cached = graded_cache.load(cached_path)
    if cached is not None:
        return cached
    graded = _grade_still_from_file(path, source)
    graded_cache.store(cached_path, graded)
    return graded


def _grade_still_from_file(path: Path, source: str) -> Image.Image:
    try:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
    except (OSError, SyntaxError):
        preview = _camera_preview(path) if source == "camera" else None
        if source == "camera" and preview is None:
            logger.warning("%s: no usable camera preview; developing the RAW instead", path)
        if preview is not None:
            image = preview
        else:
            # The grade is conventional RAW processing: camera-white-balance
            # decode, measured denoise/colour correction and sharpening.  It does
            # not generate or paint scene detail.
            image = Image.fromarray(_to_display_rgb(np.asarray(develop(path).rgb)), "RGB")
    cropped, _ = crop_black_border(image)
    cropped = grade.crop_lens_corners(cropped)
    return grade.grade_still(cropped, enabled=_GRADE_STILLS)


def _embedded_thumb(path: Path) -> SimpleNamespace | None:
    """The RAW file's embedded preview: ``format``, ``data`` and LibRaw's ``flip``."""
    import rawpy

    try:
        with rawpy.imread(str(path)) as raw:
            thumb = raw.extract_thumb()
            flip = int(getattr(raw.sizes, "flip", 0))
    except Exception:
        return None
    return SimpleNamespace(format=thumb.format, data=thumb.data, flip=flip)


def _camera_preview(path: Path) -> Image.Image | None:
    """The camera's own full-size rendering of a RAW frame, upright, or None."""
    thumb = _embedded_thumb(path)
    if thumb is None or getattr(thumb.format, "name", str(thumb.format)).lower() != "jpeg":
        return None
    try:
        with Image.open(io.BytesIO(thumb.data)) as opened:
            oriented = opened.getexif().get(274, 1) != 1
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, SyntaxError):
        return None
    if max(image.size) < CAMERA_PREVIEW_MIN_WIDTH:
        return None
    # Nikon previews carry no EXIF orientation; the camera's flip lives in
    # LibRaw's sizes. Apply it only when the JPEG did not orient itself.
    turns = 0 if oriented else _ROT90_BY_FLIP.get(int(getattr(thumb, "flip", 0)), 0)
    return image.rotate(90 * turns, expand=True) if turns else image


def _to_display_rgb(pixels: np.ndarray) -> np.ndarray:
    """Convert a decoded sRGB image to 8-bit without wrapping RAW values."""
    if pixels.dtype == np.uint8:
        return pixels
    if np.issubdtype(pixels.dtype, np.integer):
        maximum = np.iinfo(pixels.dtype).max
        return np.rint(pixels.astype(np.float32) * (255.0 / maximum)).clip(0, 255).astype(np.uint8)
    maximum = 1.0 if float(np.nanmax(pixels)) <= 1.0 else 255.0
    return np.rint(pixels * (255.0 / maximum)).clip(0, 255).astype(np.uint8)


def _fit_cover(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS
    )
    return resized.crop(
        (
            (resized.width - width) // 2,
            (resized.height - height) // 2,
            (resized.width + width) // 2,
            (resized.height + height) // 2,
        )
    )


def _fit_contain(image: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / image.width, height / image.height)
    return image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )


@functools.lru_cache(maxsize=256)
def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(_BOLD_FONT if bold else _FONT, size)
    except OSError:
        return ImageFont.load_default()


def _interpolate(first: float | None, second: float | None, fraction: float) -> float | None:
    if first is None or second is None:
        return None
    return first + (second - first) * fraction


def _speed(value: float | None) -> str:
    if value is None:
        return "—"
    if _active_display_units == "metric":
        return f"{value * 3.6:.1f} km/h"
    return f"{value / _MPS_PER_MPH:.1f} mph"


def _pace(value: float | None) -> str:
    if value is None or not 0 <= value <= _speed_cap_mps():
        return "—"
    if value < _STOPPED_SPEED_MPS:
        return "stopped"
    if _active_display_units == "metric":
        total_seconds = round(1000.0 / value)
        unit = "km"
    else:
        total_seconds = round(3600.0 / (value / _MPS_PER_MPH))
        unit = "mi"
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}/{unit}"


def _safe_speed(value: float | None) -> str:
    """Do not promote a rejected GPS jump into an instantaneous display value."""
    if value is None or not 0 <= value <= _speed_cap_mps():
        return "—"
    return "stopped" if value < _STOPPED_SPEED_MPS else _speed(value)


def _feet(value: float | None) -> str:
    if value is None:
        return "—"
    if _active_display_units == "metric":
        return f"{value:,.0f} m"
    return f"{value / _M_PER_FOOT:,.0f} ft"


def _heart_rate(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} bpm"


def _cadence(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f} rpm"


def _temperature(value: float | None) -> str:
    return "—" if value is None else f"{value:.0f}°C"


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:+.1f}%"


def _safe_grade(value: float | None) -> str:
    return _percent(value) if value is not None and abs(value) <= _grade_cap_percent() else "—"


def _moving_grade(grade: float | None, speed_mps: float | None) -> str:
    """Grade is rise over the last edge's distance, so it only means something in motion.

    Standing still that distance is GPS jitter: a film once read "stopped" beside
    a grade of -18.0%. The same goes for a speed the dashboard would not show.
    """
    if speed_mps is None or not _STOPPED_SPEED_MPS <= speed_mps <= _speed_cap_mps():
        return "—"
    return _safe_grade(grade)


def _duration(value: float) -> str:
    minutes = round(value / 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"
