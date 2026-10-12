"""Tests for gomoveshift.pipeline.ffmpeg_filters.escape_ffmpeg_text.

These tests pin the escape behavior for FFmpeg drawtext. FFmpeg's drawtext
filter interprets `%{pts}`, `%{n}`, etc. as format specifiers at render time,
and `{` / `}` frame the specifier. Literal `%`, `{`, `}` in a location or
caption must be escaped so they burn as-is rather than rendering as a
timestamp / frame number.
"""

from gomoveshift.pipeline.ffmpeg_filters import build_crop_filter, escape_ffmpeg_text


def test_escape_percent():
    """% must be escaped so FFmpeg doesn't parse `%{...}` specifiers."""
    result = escape_ffmpeg_text("Hwy 17 at 100%")
    assert "\\%" in result
    # And not an unescaped `%` remaining
    assert "100\\%" in result


def test_escape_braces():
    """{ and } must be escaped so they don't frame a specifier."""
    result = escape_ffmpeg_text("{curly} text")
    assert "\\{" in result
    assert "\\}" in result


def test_escape_combined():
    """All five special chars escape correctly with no double-escaping."""
    # Realistic-ish mixed string touching every special char the escaper handles.
    src = "A\\B:C'D%E{F}"
    result = escape_ffmpeg_text(src)
    # Backslash first so subsequent escapes aren't re-escaped.
    assert "\\\\" in result  # the original `\` became `\\`
    assert "\\:" in result
    assert "'\\''" in result
    assert "\\%" in result
    assert "\\{" in result
    assert "\\}" in result
    # No triple-backslash before %, { or } (would indicate double-escape).
    assert "\\\\%" not in result
    assert "\\\\{" not in result
    assert "\\\\}" not in result


def test_no_false_positives_on_normal_text():
    """Plain ASCII without any specials must pass through unchanged."""
    src = "Santa Cruz to Monterey via Highway 1"
    assert escape_ffmpeg_text(src) == src


def test_crop_filter_downscale_uses_lanczos():
    """The shorts 4K→1080 downscale must use lanczos, not swscale's default
    bicubic, which softens detail-heavy driving footage."""
    result = build_crop_filter(3840, 2160)
    assert result.endswith("scale=1080:1920:flags=lanczos"), (
        f"Expected lanczos-flagged scale, got '{result}'"
    )
    # 1214, not the raw 1215 this pinned before: int(2160 * 1080/1920) is odd,
    # and 4:2:0 subsamples chroma 2x horizontally, so ffmpeg cannot honour an
    # odd width and silently rounds it. The crop is now evened in code, so the
    # emitted window is the one actually applied.
    assert result.startswith("crop=1214:2160,"), f"Center-crop miscomputed: '{result}'"
    crop_w, crop_h = (int(v) for v in result.split(",")[0].removeprefix("crop=").split(":"))
    assert crop_w % 2 == 0 and crop_h % 2 == 0, f"crop dims must be even: '{result}'"
