"""
Thumbnail Text Overlay for GoMoveShift Videos

Adds branded text overlays to video thumbnails for YouTube/social media.
Supports 3 styles for A/B testing:
- MINIMAL: Clean text at bottom with gradient overlay
- BOLD: High contrast with border and background box
- IMMERSIVE: Subtle text in corner, focus on scenery

Usage:
    from thumbnail_text_overlay import generate_styled_thumbnails
    results = generate_styled_thumbnails(
        background_path="frame.jpg",
        title="Tipperary, Ireland",
        subtitle="4K POV",
        output_dir="./thumbnails"
    )
"""

import logging
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Absolute (not relative) so `python thumbnail_text_overlay.py` still runs.
from gomoveshift.pipeline.encoding_constants import LONGFORM_COLOR_RANGE
from gomoveshift.pipeline.ffmpeg_filters import FONT_PATH
from gomoveshift.pipeline.quality import (
    encode_thumbnail_jpeg,
    finish_deltas_in_band,
    thumbnail_finish_measured,
)

logger = logging.getLogger(__name__)


class ThumbnailStyle(Enum):
    """Thumbnail text overlay styles for A/B testing."""

    MINIMAL = "minimal"  # Clean, subtle text at bottom
    BOLD = "bold"  # High contrast text with border
    IMMERSIVE = "immersive"  # Very subtle text, focus on image


# Windows font paths (in order of preference)
WINDOWS_FONTS = [
    r"C:\Windows\Fonts\arialbd.ttf",  # Arial Bold
    r"C:\Windows\Fonts\arial.ttf",  # Arial
    r"C:\Windows\Fonts\calibrib.ttf",  # Calibri Bold
    r"C:\Windows\Fonts\segoeui.ttf",  # Segoe UI
    r"C:\Windows\Fonts\verdana.ttf",  # Verdana
]

# Linux font paths (fallback)
LINUX_FONTS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

# Smallest plausible styled thumbnail. Anything under this is a blank or
# truncated write, not a real 1920x1080 JPEG.
MIN_THUMBNAIL_BYTES = 10 * 1024

# A correctly sized font renders "Ag" at ~0.87-0.94x its nominal point size.
# The broken unsized fallback lands near 0.14x, so half is a wide, safe margin.
MIN_FONT_SIZE_RATIO = 0.5


def _rendered_text_height(font) -> int:
    """Height in pixels that `font` actually draws, ascender to descender."""
    bbox = font.getbbox("Ag")
    return bbox[3] - bbox[1]


def _font_search_paths() -> List[str]:
    """Font candidates, canonical burn font first.

    ``FONT_PATH`` is the constant every ffmpeg ``drawtext`` burn uses. The
    thumbnail used to carry its own literal list and try Arial before it, so
    the same title could render in two different faces on two surfaces — the
    exact drift that constant exists to prevent. Platform lists stay as
    fallbacks for a host without DejaVu.
    """
    return [FONT_PATH, *LINUX_FONTS, *WINDOWS_FONTS]


def _get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    """Get a font, preferring the canonical burn font."""
    for path in _font_search_paths():
        try:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        except Exception:
            continue

    # Ultimate fallback: Pillow's bundled font, asked for the size we actually
    # want. An unsized load_default() yields ~10px text on a 1920x1080
    # thumbnail — unreadable, and (on Pillow 12) still a FreeTypeFont, so a
    # type check would not catch it. Verify the rendered size instead.
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:  # Pillow too old to accept size=
        font = ImageFont.load_default()

    rendered = _rendered_text_height(font)
    if rendered < size * MIN_FONT_SIZE_RATIO:
        raise RuntimeError(
            f"No usable font: fallback renders {rendered}px for a requested size of "
            f"{size}px. Install a TrueType font (e.g. fonts-dejavu) rather than "
            f"burning unreadable thumbnail text."
        )
    return font


def extract_frame_from_video(
    video_path: str, output_path: str, timestamp: int = 60, width: int = 1920, height: int = 1080
) -> Optional[str]:
    """
    Extract a single frame from video using ffmpeg.

    Args:
        video_path: Path to source video
        output_path: Path for output image
        timestamp: Time in seconds to extract frame
        width: Output width
        height: Output height

    Returns:
        Path to generated frame, or None on failure
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        video_path,
        "-vframes",
        "1",
        "-vf",
        # flags=lanczos: swscale's default bicubic softens a 4K→1080 downscale
        # (same fix as the shorts crop). in_range/out_range: our finals are
        # stamped limited-range and a JPEG is full-range, so state the
        # conversion instead of leaving it to implicit swscale behaviour.
        f"scale={width}:{height}:force_original_aspect_ratio=decrease"
        f":flags=lanczos:in_range={LONGFORM_COLOR_RANGE}:out_range=pc,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "-q:v",
        "2",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and Path(output_path).exists():
            return output_path
        else:
            print(f"FFmpeg error: {result.stderr[:200] if result.stderr else 'unknown'}")
            return None
    except subprocess.TimeoutExpired:
        print("FFmpeg timed out")
        return None
    except Exception as e:
        print(f"Error extracting frame: {e}")
        return None


def _wrap_text(
    draw: ImageDraw.Draw, text: str, font: ImageFont.FreeTypeFont, max_width: int
) -> list:
    """Word-wrap text to fit within max_width. Returns list of lines."""
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


def _draw_minimal_style(image: Image.Image, title: str, subtitle: Optional[str]) -> Image.Image:
    """
    Minimal style: Clean text at bottom with gradient overlay.
    Best for: Letting scenery speak, subtle branding.
    """
    width, height = image.size
    margin = 60

    # Create gradient overlay at bottom
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)

    overlay_height = 300
    for y in range(overlay_height):
        alpha = int((y / overlay_height) * 180)
        overlay_draw.line(
            [(0, height - overlay_height + y), (width, height - overlay_height + y)],
            fill=(0, 0, 0, alpha),
        )

    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    max_text_width = width - (margin * 2)

    # Draw title — word wrap if needed
    title_font = _get_font(72, bold=True)
    lines = _wrap_text(draw, title, title_font, max_text_width)

    # Position from bottom up: subtitle at -80, title lines above that
    sub_height = 0
    if subtitle:
        sub_font = _get_font(40, bold=False)
        sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sub_height = sub_bbox[3] - sub_bbox[1] + 20  # height + gap

    line_height = draw.textbbox((0, 0), "Ag", font=title_font)[3] + 10
    total_title_height = line_height * len(lines)

    # Start Y: position so subtitle sits ~40px from bottom
    base_y = height - 40 - sub_height - total_title_height

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        lw = bbox[2] - bbox[0]
        lx = (width - lw) // 2
        ly = base_y + i * line_height
        draw.text((lx + 3, ly + 3), line, fill=(0, 0, 0), font=title_font)
        draw.text((lx, ly), line, fill=(255, 255, 255), font=title_font)

    # Subtitle
    if subtitle:
        sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sub_width = sub_bbox[2] - sub_bbox[0]
        sub_x = (width - sub_width) // 2
        sub_y = height - 40 - sub_height + 10
        draw.text((sub_x, sub_y), subtitle, fill=(200, 200, 200), font=sub_font)

    return image


def _draw_bold_style(image: Image.Image, title: str, subtitle: Optional[str]) -> Image.Image:
    """
    Bold style: High contrast with border and background box.
    Best for: Maximum visibility, standout in search results.
    """
    width, height = image.size
    margin = 60  # Minimum margin from image edges
    max_text_width = width - (margin * 2)
    padding = 35

    draw = ImageDraw.Draw(image, "RGBA")

    # Scale font, then wrap if still too wide
    title_font_size = 85
    title_font = _get_font(title_font_size, bold=True)
    title_lines = _wrap_text(draw, title, title_font, max_text_width)

    # If wrapping produced too many lines, scale down font
    while len(title_lines) > 2 and title_font_size > 52:
        title_font_size -= 4
        title_font = _get_font(title_font_size, bold=True)
        title_lines = _wrap_text(draw, title, title_font, max_text_width)

    line_height = draw.textbbox((0, 0), "Ag", font=title_font)[3] + 10
    total_title_height = line_height * len(title_lines)

    # Measure widest line for box sizing
    max_line_width = 0
    for line in title_lines:
        bbox = draw.textbbox((0, 0), line, font=title_font)
        lw = bbox[2] - bbox[0]
        if lw > max_line_width:
            max_line_width = lw

    # Subtitle measurement
    sub_space = 0
    sub_font = None
    if subtitle:
        sub_font_size = 48
        while sub_font_size >= 24:
            sub_font = _get_font(sub_font_size, bold=True)
            sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
            sub_width = sub_bbox[2] - sub_bbox[0]
            if sub_width <= max_text_width:
                break
            sub_font_size -= 4
        else:
            sub_font = _get_font(24, bold=True)
            sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
            sub_width = sub_bbox[2] - sub_bbox[0]
        sub_space = 60
        if sub_width > max_line_width:
            max_line_width = sub_width

    # Center the block vertically
    block_height = total_title_height + sub_space
    title_y = (height - block_height) // 2 - 10
    box_x = (width - max_line_width) // 2

    box_coords = [
        (box_x - padding, title_y - padding),
        (box_x + max_line_width + padding, title_y + block_height + padding),
    ]

    # Semi-transparent black background
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(box_coords, fill=(0, 0, 0, 200))
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Draw yellow border
    for i in range(5):
        draw.rectangle(
            [
                (box_coords[0][0] - i, box_coords[0][1] - i),
                (box_coords[1][0] + i, box_coords[1][1] + i),
            ],
            outline=(255, 200, 0),
        )

    # Draw title lines centered
    for i, line in enumerate(title_lines):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        lw = bbox[2] - bbox[0]
        lx = (width - lw) // 2
        ly = title_y + i * line_height
        draw.text((lx + 4, ly + 4), line, fill=(0, 0, 0), font=title_font)
        draw.text((lx, ly), line, fill=(255, 255, 255), font=title_font)

    # Subtitle in yellow
    if subtitle and sub_font:
        sub_bbox = draw.textbbox((0, 0), subtitle, font=sub_font)
        sub_width = sub_bbox[2] - sub_bbox[0]
        sub_x = (width - sub_width) // 2
        sub_y = title_y + total_title_height + 20
        draw.text((sub_x, sub_y), subtitle, fill=(255, 200, 0), font=sub_font)

    return image


def _draw_immersive_style(image: Image.Image, title: str, subtitle: Optional[str]) -> Image.Image:
    """
    Immersive style: Subtle text in corner, focus on scenery.
    Best for: Cinematic feel, when the image is the star.
    """
    width, _height = image.size
    draw = ImageDraw.Draw(image)

    title_font = _get_font(56, bold=True)
    max_text_width = width - 100  # 50px margin each side

    # Word wrap title
    lines = _wrap_text(draw, title, title_font, max_text_width)
    line_height = draw.textbbox((0, 0), "Ag", font=title_font)[3] + 8

    title_x = 50
    title_y = 50

    # Black stroke around the glyphs (not a box — keeps the immersive,
    # box-free look) so white text stays legible over a bright frame/sky.
    # The old 2px drop-shadow offset gave no contrast on light backgrounds.
    for i, line in enumerate(lines):
        ly = title_y + i * line_height
        draw.text(
            (title_x, ly),
            line,
            fill=(255, 255, 255),
            font=title_font,
            stroke_width=3,
            stroke_fill=(0, 0, 0),
        )

    # Subtitle below last title line
    if subtitle:
        sub_font = _get_font(36, bold=False)
        sub_y = title_y + len(lines) * line_height + 10
        draw.text(
            (title_x, sub_y),
            subtitle,
            fill=(230, 230, 230),
            font=sub_font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    return image


def finish_background(background_path: str) -> Image.Image:
    """Load the background and finish it once, before any text is drawn.

    A frame whose finish measures out of band is returned unfinished with a warning.
    """
    bg = Image.open(background_path).convert("RGB")
    before = np.asarray(bg)
    after, d = thumbnail_finish_measured(before)
    if not finish_deltas_in_band(before, d):
        logger.warning(
            "thumbnail finish cannot reach the design-spec bands for %s "
            "(finish contrast %+.1f %%, saturation %+.1f %%); shipping unfinished",
            background_path,
            d["contrast"] * 100,
            d["saturation"] * 100,
        )
        return bg
    return Image.fromarray(after)


def generate_single_style(
    background_path: str,
    title: str,
    subtitle: Optional[str],
    style: ThumbnailStyle,
    output_path: str,
    bg: Optional[Image.Image] = None,
) -> str:
    """
    Generate a single styled thumbnail.

    Args:
        background_path: Path to background image
        title: Main title text
        subtitle: Optional subtitle
        style: Which style to apply
        output_path: Where to save
        bg: Already-finished background (see finish_background); finished here when omitted

    Returns:
        Path to generated thumbnail
    """
    if bg is None:
        bg = finish_background(background_path)
    else:
        bg = bg.copy()  # the style drawers paint in place; never leak text between styles

    # Apply style
    if style == ThumbnailStyle.MINIMAL:
        img = _draw_minimal_style(bg, title, subtitle)
    elif style == ThumbnailStyle.BOLD:
        img = _draw_bold_style(bg, title, subtitle)
    else:  # IMMERSIVE
        img = _draw_immersive_style(bg, title, subtitle)

    # Save
    # <=2 MB gate: refuses before anything is written (YouTube rejects larger).
    encode_thumbnail_jpeg(img, output_path)

    return output_path


def generate_styled_thumbnails(
    background_path: str,
    title: str,
    subtitle: Optional[str] = None,
    output_dir: str = "./thumbnails/styled",
    base_name: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate all 3 thumbnail style variations for A/B testing.

    Args:
        background_path: Path to background image (extracted video frame)
        title: Main title text (e.g., "Tipperary, Ireland")
        subtitle: Optional subtitle (e.g., "4K POV")
        output_dir: Directory to save thumbnails
        base_name: Base filename (defaults to background image stem)

    Returns:
        Dict mapping style name to file path:
        {
            "minimal": "/path/to/thumbnail_minimal.jpg",
            "bold": "/path/to/thumbnail_bold.jpg",
            "immersive": "/path/to/thumbnail_immersive.jpg"
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base_name is None:
        base_name = Path(background_path).stem

    results = {}

    # One finish per background, reused by every style.
    try:
        finished = finish_background(background_path)
    except Exception as e:
        print(f"  X Failed to finish background: {e}")
        return results

    for style in ThumbnailStyle:
        output_path = output_dir / f"{base_name}_{style.value}.jpg"

        try:
            generate_single_style(
                background_path=background_path,
                title=title,
                subtitle=subtitle,
                style=style,
                output_path=str(output_path),
                bg=finished,
            )
            results[style.value] = str(output_path)
            print(f"  + Generated {style.value}: {output_path.name}")
        except Exception as e:
            print(f"  X Failed to generate {style.value}: {e}")

    return results


def generate_thumbnails_from_video(
    video_path: str,
    title: str,
    subtitle: Optional[str] = None,
    output_dir: str = "./thumbnails/styled",
    frame_timestamps: List[int] = [60, 300, 600],
    base_name: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    """
    Extract frames from video and generate styled thumbnails for each.

    This is the main entry point for the pipeline.

    Args:
        video_path: Path to source video
        title: Main title text
        subtitle: Optional subtitle
        output_dir: Directory to save thumbnails
        frame_timestamps: Seconds at which to extract frames
        base_name: Base filename (defaults to video stem)

    Returns:
        Nested dict: {frame_index: {style: path}}
        Example:
        {
            "frame1": {"minimal": "...", "bold": "...", "immersive": "..."},
            "frame2": {"minimal": "...", "bold": "...", "immersive": "..."},
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if base_name is None:
        base_name = Path(video_path).stem

    results = {}

    for i, timestamp in enumerate(frame_timestamps, 1):
        print(f"\nExtracting frame {i} at {timestamp}s...")

        # Extract frame
        frame_path = output_dir / f"{base_name}_frame{i}.jpg"
        extracted = extract_frame_from_video(
            video_path=video_path, output_path=str(frame_path), timestamp=timestamp
        )

        if not extracted:
            print(f"  X Failed to extract frame at {timestamp}s")
            continue

        # Generate styled versions
        print(f"  Generating styled thumbnails for frame {i}...")
        styled = generate_styled_thumbnails(
            background_path=str(frame_path),
            title=title,
            subtitle=subtitle,
            output_dir=str(output_dir),
            base_name=f"{base_name}_frame{i}",
        )

        results[f"frame{i}"] = styled

    return results


# Legacy single-style function for backwards compatibility
def add_text_overlay(
    input_path: str,
    output_path: str,
    main_text: str,
    sub_text: Optional[str] = None,
    style: str = "minimal",
) -> str:
    """
    Add text overlay to a thumbnail image (legacy interface).

    Args:
        input_path: Source thumbnail image
        output_path: Where to save the result
        main_text: Primary text (location name)
        sub_text: Secondary text (drive type/vibe)
        style: "minimal", "bold", or "immersive"

    Returns:
        Path to output file
    """
    style_enum = ThumbnailStyle(style)
    return generate_single_style(
        background_path=input_path,
        title=main_text,
        subtitle=sub_text,
        style=style_enum,
        output_path=output_path,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python thumbnail_text_overlay.py <input_image> <title> <subtitle> [output_dir]"
        )
        print("\nExample:")
        print('  python thumbnail_text_overlay.py frame.jpg "Tipperary, Ireland" "4K POV"')
        print("\nThis generates 3 styled thumbnails for A/B testing:")
        print("  - minimal: Clean text at bottom")
        print("  - bold: High contrast box with border")
        print("  - immersive: Subtle corner text")
        sys.exit(1)

    input_path = sys.argv[1]
    title = sys.argv[2]
    subtitle = sys.argv[3] if len(sys.argv) > 3 else None
    output_dir = sys.argv[4] if len(sys.argv) > 4 else "./thumbnails/styled"

    print(f"Input: {input_path}")
    print(f"Title: {title}")
    print(f"Subtitle: {subtitle}")
    print(f"Output: {output_dir}")
    print()

    results = generate_styled_thumbnails(
        background_path=input_path, title=title, subtitle=subtitle, output_dir=output_dir
    )

    print(f"\nGenerated {len(results)} thumbnails:")
    for style, path in results.items():
        print(f"  {style}: {path}")
