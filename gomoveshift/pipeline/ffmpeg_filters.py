"""
Shared FFmpeg filter utilities for video processing.

Extracted from pipeline/shorts_extractor.py to eliminate duplication.
Used by shorts rendering, and available for any FFmpeg filter construction.
"""

import json
import logging
import subprocess
from pathlib import Path

from .encoding_constants import (
    SHORTS_AUDIO_BITRATE,
    SHORTS_AUDIO_LUFS,
    SHORTS_CRF,
    SHORTS_HEIGHT,
    SHORTS_PRESET,
    SHORTS_WIDTH,
)
from .quality import UPSCALE_KERNEL, UPSCALE_SCALER

logger = logging.getLogger(__name__)

# Rendering defaults — sourced from encoding_constants (single source of truth)

DEFAULT_OUTPUT_WIDTH = SHORTS_WIDTH
DEFAULT_OUTPUT_HEIGHT = SHORTS_HEIGHT
DEFAULT_CRF = SHORTS_CRF
DEFAULT_PRESET = SHORTS_PRESET
DEFAULT_AUDIO_BITRATE = SHORTS_AUDIO_BITRATE
DEFAULT_LUFS = SHORTS_AUDIO_LUFS
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _validate_font_path(path: str) -> str:
    """Reject font paths containing ffmpeg-special chars.

    ``font_path`` is interpolated bare into ``drawtext=fontfile='{font_path}'``
    at many call sites; a ``:`` ``'`` or ``\\`` would break the filter graph or
    inject filter options. The path is a hardcoded constant (not injectable),
    so we validate it once at module load — robust-by-construction beats
    escaping ~10 call sites.
    """
    for ch in (":", "'", "\\"):
        if ch in path:
            raise ValueError(
                f"font path {path!r} contains ffmpeg-special char {ch!r}; "
                "it is interpolated bare into drawtext=fontfile='...'"
            )
    return path


# Fail fast at import if the canonical constant is ever changed to an unsafe value.
# This validates ffmpeg-special CHARS only — it does NOT check the file exists
# (the filesystem isn't touched at import so a font-less test/CI host can still
# import). Existence is gated at runtime by ``preflight.check_ost_font`` before
# any encode; do not mistake this call for proof the font is on disk.
_validate_font_path(FONT_PATH)

LOC_FONT_SIZE = 48
LOC_LINE_HEIGHT = 58  # fontsize + spacing
LOC_MAX_WIDTH = DEFAULT_OUTPUT_WIDTH - 80  # 40px margin each side

# Canonical on-screen brand / CTA text. SINGLE source so the three burn
# surfaces — long-form (ost_applier), shorts (build_ost_filter below) and the
# single-video path (single_pass) — cannot drift apart on the literal strings.
# Per-surface POSITION / timing legitimately differ and stay in each surface; only the
# words are shared here.
BRAND_TEXT = "GoMoveShift"
WEBSITE_TEXT = "gomoveshift.com"
YOUTUBE_CTA_TEXT = "Full drive on YouTube"


SHORTS_MAX_TEXT_LEN = 40  # max chars for mobile readability at fontsize=48

# Connector tokens title assembly joins endpoints/roads with — "{start} to
# {end} via {roadA} + {roadB}" (``title_builder``). A truncation that cuts
# right after one burns an orphaned arc ("Killarney to"), so every word-level
# chop strips them from the tail; ``ost_validator`` rejects any that
# slip through, upstream of the irreversible burn.
TITLE_CONNECTOR_TOKENS = frozenset({"to", "via", "+"})


def strip_trailing_connectors(text: str) -> str:
    """Drop orphaned trailing connector tokens left by a chop ("Killarney to"
    -> "Killarney"). Never strips to empty: a text that is ONLY a connector is
    returned unchanged for ``validate_ost`` to reject.
    """
    words = text.split()
    while len(words) > 1 and words[-1].lower() in TITLE_CONNECTOR_TOKENS:
        words = words[:-1]
    return " ".join(words) if words else text


def _truncate_region_preserving(text: str, budget: int) -> str:
    """Truncate ``text`` to ``budget`` chars, region-preserving.

    Drops whole trailing comma-separated clauses (", County Kerry", ", SC") one
    at a time from the right — keeping the distinguishing leading clause (the
    town) — before any character-level cut: "Kenmare, County Kerry" over budget
    degrades to "Kenmare", never "Kenmare, Count". Only when the leading clause
    alone still overflows does it fall back to a word-boundary cut, then a hard
    char cut.
    """
    if len(text) <= budget:
        return text
    parts = [p.strip() for p in text.split(",")]
    while len(parts) > 1 and len(", ".join(parts)) > budget:
        parts = parts[:-1]
    text = ", ".join(parts)
    if len(text) <= budget:
        return text
    # Leading clause alone still too long — prefer a word-boundary cut.
    truncated = text[:budget]
    last_space = truncated.rfind(" ")
    if last_space > 10:
        return strip_trailing_connectors(truncated[:last_space])
    # No usable word boundary (a single over-budget token): a bare hard cut chops
    # mid-word ("Newtownmountkennedy" -> "Newtownmountken"). Reserve one char for
    # a U+2026 ellipsis so the truncation reads as intentional, not accidental.
    return truncated[: budget - 1].rstrip() + "…"


def shorten_ost_for_shorts(location: str) -> str:
    """Shorten location text for shorts (vertical mobile video).

    Strips verbose clauses that don't help on a short mobile video:
    - "via ..." clause from route descriptions
    - "Scenic Drive" suffix (redundant for shorts)
    Region-preserving truncation if still over SHORTS_MAX_TEXT_LEN.
    """
    if not location:
        return ""
    text = location
    # Strip "via ..." clause — too verbose for mobile
    if " via " in text:
        text = text[: text.index(" via ")]
    # Strip "Scenic Drive" suffix — redundant for shorts context
    if text.endswith(" Scenic Drive"):
        text = text[: -len(" Scenic Drive")]
    # Drop the whole trailing ", County X" clause before any char-chop.
    text = _truncate_region_preserving(text, SHORTS_MAX_TEXT_LEN)
    return text.strip()


def escape_ffmpeg_text(text: str) -> str:
    """Escape text for FFmpeg drawtext filter.

    Escapes the five characters drawtext treats specially:
      * ``\\`` — escape char itself (must be first to avoid double-escaping)
      * ``:``  — option separator
      * ``'``  — string delimiter
      * ``%``  — format-specifier prefix (``%{pts}``, ``%{n}``, etc.)
      * ``{`` / ``}`` — specifier framing

    Without escaping ``%`` / ``{`` / ``}``, a literal like ``"100%"`` or
    ``"Route {A}"`` in a location/caption would be interpreted as a render-time
    specifier and burn as a timestamp or frame number.

    Brings this helper into parity with
    ``ost_applier._escape_drawtext``.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "'\\''")
    text = text.replace("%", "\\%")
    text = text.replace("{", "\\{").replace("}", "\\}")
    return text


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Get video width and height via ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "stream=width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)
    for stream in data.get("streams", []):
        if "width" in stream and "height" in stream:
            return int(stream["width"]), int(stream["height"])

    raise RuntimeError(f"No video stream found in {video_path}")


def crop_upscaled_from(
    src_w: int,
    src_h: int,
    dst_w: int = DEFAULT_OUTPUT_WIDTH,
    dst_h: int = DEFAULT_OUTPUT_HEIGHT,
) -> list[int] | None:
    """``[crop_w, crop_h]`` when the 9:16 crop is smaller than the output (an upscale)."""
    crop_w, crop_h = _crop_size(src_w, src_h, dst_w, dst_h)
    return [crop_w, crop_h] if crop_w < dst_w or crop_h < dst_h else None


def _crop_size(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int]:
    """Even-rounded centre-crop window for ``dst_w:dst_h`` out of the source."""
    dst_aspect = dst_w / dst_h  # 0.5625

    src_aspect = src_w / src_h
    if src_aspect > dst_aspect:
        # Source wider — crop sides
        crop_h = src_h
        crop_w = int(src_h * dst_aspect)
    else:
        # Source taller — crop top/bottom
        crop_w = src_w
        crop_h = int(src_w / dst_aspect)

    # Round DOWN to even. 4:2:0 subsamples chroma 2x on both axes, so an odd
    # crop dimension is not representable and ffmpeg silently rounds it —
    # meaning the effective crop window stops being the one computed above.
    # Every common source hit this: 3840x2160 produced crop_w=1215.
    # Down, never up: a crop larger than the source makes ffmpeg fail outright.
    crop_w -= crop_w % 2
    crop_h -= crop_h % 2
    return crop_w, crop_h


def build_crop_filter(
    src_w: int,
    src_h: int,
    dst_w: int = DEFAULT_OUTPUT_WIDTH,
    dst_h: int = DEFAULT_OUTPUT_HEIGHT,
) -> str:
    """Build center-crop filter for vertical 9:16 output."""
    crop_w, crop_h = _crop_size(src_w, src_h, dst_w, dst_h)
    # One scaler for every resize (quality.UPSCALE_SCALER): swscale's default
    # bicubic softens detail-heavy driving footage on the 4K→1080 drop, and a crop narrower than the output is a declared upscale
    # that goes through the same filter. No dither flag is needed: shorts
    # encode 10-bit end-to-end (SHORTS_PIX_FMT p010le), so no 10→8-bit depth
    # reduction remains here.
    return f"crop={crop_w}:{crop_h},{UPSCALE_SCALER}=w={dst_w}:h={dst_h}:filter={UPSCALE_KERNEL}"


def measure_text_width(
    text: str,
    font_size: int = LOC_FONT_SIZE,
    font_path: str = FONT_PATH,
) -> int:
    """Rendered pixel width of ``text`` at ``font_size`` in the EXACT ffmpeg font.

    Pinned to ``FONT_PATH`` (DejaVuSans-Bold) — the same hardcoded font both OST
    render surfaces burn with — NOT a Liberation/Ubuntu ``_find_font`` fallback,
    or the measurement would describe a different glyph set than ffmpeg renders.
    Returns ``-1`` when PIL is unavailable so callers can degrade to
    the codepoint proxy rather than silently passing wide text.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(font_path, font_size)
        draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        bbox = draw.textbbox((0, 0), text, font=font)
        return int(bbox[2] - bbox[0])
    except Exception:
        return -1


def wrap_text_for_ffmpeg(
    text: str,
    font_size: int = LOC_FONT_SIZE,
    max_width: int = LOC_MAX_WIDTH,
    font_path: str = FONT_PATH,
) -> list:
    """Word-wrap text using PIL font metrics. Returns list of lines."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(font_path, font_size)
        img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(img)

        words = text.split()
        lines = []
        current_line = ""
        for word in words:
            test = f"{current_line} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line = test
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines or [text]
    except Exception as exc:
        # PIL/font unavailable — wrapping is DISABLED, not merely approximated:
        # the whole title comes back as one unwrapped line and can overflow the
        # frame. WARN loudly (this was a silent bare except) so the fallback
        # leaves a breadcrumb on both the LF burn and the shorts path.
        logger.warning(
            "wrap_text_for_ffmpeg: PIL wrap unavailable (%s) — returning the "
            "text as one unwrapped line; long titles may overflow the frame",
            exc,
        )
        return [text]


def build_ost_filter(
    location: str,
    subscribe_text: str,
    duration: float,
    font_path: str = FONT_PATH,
) -> str:
    """
    Build drawtext filter for on-screen text (shorts path).

    Location text: shown for the first 5 seconds, centered near top.
    Wraps long text onto multiple lines using PIL font metrics.
    YouTube CTA: "Full drive on YouTube" shown mid-video (upper area).
    Website CTA: "gomoveshift.com" shown in final seconds (upper area).

    The ``subscribe_text`` parameter is accepted for API compatibility
    but ignored — CTAs are now hardcoded to the YouTube cross-promo
    and website URL.

    Raises ``ValueError`` on an empty/whitespace ``location`` — a short must
    carry a title (mirrors ``ost_applier.apply_ost``'s ``OST-EMPTY-LOCATION``
    contract). The upstream ``day_pipeline._gate_short_ost_text`` already
    guarantees a non-empty title; this is the burn-site regression guard so a
    future caller can never silently emit a degenerate empty-text drawtext.
    """
    if not location or not location.strip():
        raise ValueError(
            "build_ost_filter: refusing to render a short with an empty location "
            "title — a short must carry a title. The "
            "_gate_short_ost_text cascade should have resolved or hard-failed "
            "upstream; inspect the short's OST text resolution."
        )
    ost_show_duration = 5.0
    fade_duration = 0.5

    # Location text: show during first 5s with fade in/out
    loc_alpha = (
        f"if(lt(t,{fade_duration}),t/{fade_duration},"
        f"if(gt(t,{ost_show_duration - fade_duration}),"
        f"({ost_show_duration}-t)/{fade_duration},1))"
    )

    # Wrap location text into multiple lines
    loc_lines = wrap_text_for_ffmpeg(location, LOC_FONT_SIZE, LOC_MAX_WIDTH, font_path)

    # Build one drawtext filter per line, vertically stacked from y=280
    location_filters = []
    for i, line in enumerate(loc_lines):
        line_escaped = escape_ffmpeg_text(line)
        y_pos = 280 + i * LOC_LINE_HEIGHT
        location_filters.append(
            f"drawtext=fontfile='{font_path}':"
            f"text='{line_escaped}':"
            f"fontsize={LOC_FONT_SIZE}:"
            f"fontcolor=white:"
            f"borderw=3:bordercolor=black:"
            f"shadowcolor=black@0.6:shadowx=4:shadowy=4:"
            f"x=(w-text_w)/2:y={y_pos}:"
            f"alpha='{loc_alpha}'"
        )

    cta_filters = []

    # YouTube CTA — cross-promote long-form (upper area, mid-video)
    yt_cta_text = YOUTUBE_CTA_TEXT
    yt_cta_duration = 6.0
    yt_cta_end_offset = 7.0  # ends 7s before video end
    yt_end = duration - yt_cta_end_offset
    yt_start = yt_end - yt_cta_duration
    # Only fire the YT CTA once the location-title window (t∈[0, ost_show_duration],
    # y=280) has cleared — the CTA shares y=280, so starting it before the title
    # fades double-stacks two texts at the same spot. This subsumes the old
    # `duration > 14` guard: yt_start >= ost_show_duration implies duration >= 18,
    # closing the 14-18s simultaneous-render overlap.
    if yt_start >= ost_show_duration:
        yt_escaped = escape_ffmpeg_text(yt_cta_text)
        yt_alpha = (
            f"if(lt(t,{yt_start}),0,"
            f"if(lt(t,{yt_start + fade_duration}),"
            f"(t-{yt_start})/{fade_duration},"
            f"if(gt(t,{yt_end - fade_duration}),"
            f"({yt_end}-t)/{fade_duration},1)))"
        )
        cta_filters.append(
            f"drawtext=fontfile='{font_path}':"
            f"text='{yt_escaped}':"
            f"fontsize=38:"
            f"fontcolor=white:"
            f"borderw=2:bordercolor=black:"
            f"shadowcolor=black@0.6:shadowx=3:shadowy=3:"
            f"x=(w-text_w)/2:y=280:"
            f"alpha='{yt_alpha}'"
        )

    # Website CTA — gomoveshift.com (upper area, final seconds)
    ws_text = WEBSITE_TEXT
    ws_duration = 3.5
    ws_end_margin = 0.5
    if duration > (ws_duration + ws_end_margin + 1.0):
        ws_end = duration - ws_end_margin
        ws_start = ws_end - ws_duration
        ws_escaped = escape_ffmpeg_text(ws_text)
        ws_alpha = (
            f"if(lt(t,{ws_start}),0,"
            f"if(lt(t,{ws_start + fade_duration}),"
            f"(t-{ws_start})/{fade_duration},"
            f"if(gt(t,{ws_end - fade_duration}),"
            f"({ws_end}-t)/{fade_duration},1)))"
        )
        cta_filters.append(
            f"drawtext=fontfile='{font_path}':"
            f"text='{ws_escaped}':"
            f"fontsize=38:"
            f"fontcolor=white:"
            f"borderw=2:bordercolor=black:"
            f"shadowcolor=black@0.6:shadowx=3:shadowy=3:"
            f"x=(w-text_w)/2:y=280:"
            f"alpha='{ws_alpha}'"
        )

    all_filters = location_filters + cta_filters
    return ",".join(all_filters)
