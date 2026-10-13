"""Post-assembly OST text application for the day-first pipeline.

Single authoritative place where OST text gets burned onto a finished
video artifact (day master, regular LF, or short). Replaces the per-
episode drawtext path that used to live inside `single_pass.py`.

Why post-assembly: the OST is a property of the FINAL output, not the
source footage. Burning it here means:
  * The timing is relative to the final artifact's timeline (after the
    montage, in the correct position for the clip's structure).
  * Each output can show a different OST — the regular LF's title can
    differ from the super LF's title, shorts can have their own text.
  * No dependency on segment selection: the OST is always visible at
    the right moment because it's applied AFTER everything else.

## Layouts

Each layout specifies where the location overlay fires and which CTAs
are shown:

  | Layout      | Location window          | Brand                     | YouTube CTA | Website CTA |
  |-------------|--------------------------|---------------------------|-------------|-------------|
  | regular_lf  | [0, M] + crossfade out   | [0, M] + crossfade out    | No          | Last 6s     |
  | short       | 0s - 5s                  | 0s - 5s                   | Yes (mid)   | Last 3.5s   |

NOTE: in the day-first drive pipeline ``regular_lf`` is the live long-form
path (the only production ``apply_ost`` call). Production SHORTS are NOT burned
here — they go through ``shorts_extractor.render_short_ffmpeg`` ->
``ffmpeg_filters.build_ost_filter`` (crop + scale + OST in one pass). The
``short`` layout above is a still-tested ``apply_ost`` capability, not the live
shorts renderer; the two are kept aligned but build_ost_filter owns shorts.

For ``regular_lf`` both the LOCATION title and the BRAND (``GoMoveShift``)
text are visible from ``t=0`` through the end of the montage ``M``
(``config.drive.montage_max_duration_sec``, threaded in as the
``montage_duration`` kwarg), crossfading linearly to zero over the last
``config.drive.ost_title_crossfade_sec`` seconds as the main video begins.
The bottom-right logo watermark and the top-right persistent
``gomoveshift.com`` URL remain full-video — those are corner watermarks,
not center-screen text. A persistent center-screen brand was
reverted (a content-integrity violation visible to every viewer for the
entire LF).

Shorts show "Full drive on YouTube" to cross-promote long-form content
(works whether the viewer is on YouTube, TikTok, or Instagram), then
the website URL at the end. Long-form shows only the website URL (the
viewer is already on YouTube; native end-screen cards handle subscribe).

## Encoding

Drawtext modifies video pixels, so the input must be re-encoded. We use
`hevc_nvenc -preset p7 -cq 18` for GPU-accelerated high-quality output,
and pass audio through via `-c:a copy` (drawtext doesn't touch audio).

When `location` is empty or the font file is missing, we fall back to a
pure stream-copy so the pipeline still produces an output file (it just
won't have a title overlay — logged clearly). That copy is ffprobe-checked
against the input's probed duration exactly like a burn, so a truncated
fallback output is rejected rather than uploaded.
"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

from gomoveshift.pipeline.exceptions import ProbeFailedError
from gomoveshift.pipeline.ffmpeg_filters import (
    BRAND_TEXT,
    WEBSITE_TEXT,
    YOUTUBE_CTA_TEXT,
    _validate_font_path,
    escape_ffmpeg_text,
    wrap_text_for_ffmpeg,
)
from gomoveshift.pipeline.ffmpeg_filters import (
    FONT_PATH as _DEFAULT_FONT_PATH,
)
from gomoveshift.pipeline.ffmpeg_runner import run_ffmpeg_encode, validate_video
from gomoveshift.pipeline.loudnorm_measure import (
    build_loudnorm_measure_cmd,
    parse_loudnorm_stats,
)

logger = logging.getLogger(__name__)


# loudnorm pass-1 measurement timeout. Audio-only decode runs ~100× faster
# than the video re-encode, so a generous default rides out long LFs without
# risking an indefinite hang.
_AUDIO_MEASURE_TIMEOUT_SEC = 1800


def _measure_audio_loudnorm(input_path: Path) -> dict:
    """Run a loudnorm measurement pass (audio-only decode) over ``input_path``.

    Mirrors ``single_pass.SinglePassProcessor._measure_audio`` (shared origin —
    kept as a small module-level twin rather than importing the method, which
    would couple ``ost_applier`` to ``SinglePassProcessor``/``Config``). A
    single-pass dynamic ``loudnorm`` WITHOUT measured input stats mis-targets
    very quiet ambient driving audio (far below target, where single-pass
    loudnorm over-amplifies), so the
    per-LF OST re-encode needs the same two-pass measure→apply contract the
    day-master path uses. Returns the parsed ``input_i/input_tp/input_lra/
    input_thresh`` (+ ``target_offset``) dict, or ``{}`` on any failure (caller
    falls back to a single-pass target-only filter — degrade, never fail).
    """
    cmd = build_loudnorm_measure_cmd(input_path)
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=_AUDIO_MEASURE_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001 — measurement is best-effort; degrade not fail
        logger.warning("OST audio measurement failed for %s: %s", input_path.name, e)
        return {}
    if result.returncode != 0:
        logger.warning(
            "⚠ ost_applier: loudnorm measurement returned rc=%d for %s; "
            "falling back to single-pass loudnorm (output loudness may undershoot)",
            result.returncode,
            input_path.name,
        )
        return {}
    stderr = (result.stderr or b"").decode("utf-8", errors="replace")
    try:
        measured = parse_loudnorm_stats(stderr)
    except json.JSONDecodeError as e:
        logger.warning(
            "⚠ ost_applier: loudnorm measurement JSON failed to parse for %s (%s); "
            "falling back to single-pass loudnorm (output loudness may undershoot)",
            input_path.name,
            e,
        )
        return {}
    if measured is None:
        logger.warning(
            "⚠ ost_applier: two-pass loudnorm measurement attempted for %s; "
            "no JSON stats block in ffmpeg stderr; falling back to single-pass "
            "loudnorm (output loudness may undershoot)",
            input_path.name,
        )
        return {}
    return measured


def _audio_attenuate_filter(measured: Optional[dict] = None) -> str:
    """Build the ATTENUATE-ONLY per-LF OST re-encode audio filter.

    Policy: every published
    video must be practically silent. The prior two-pass ``loudnorm=I=-20``
    boosted quiet cab audio *up* to target — audible speech.
    ``build_attenuate_filter`` instead computes a single non-positive
    ``volume=`` gain that pulls the measured source loudness DOWN to the
    near-silent integrated ceiling and honours the true-peak cap; positive
    gain is impossible by construction (asserted in the helper).

    ``aresample=async=1000`` still prefixes the gain to re-clock the audio
    stream before it is written, preventing non-monotonic DTS warnings from
    source files whose AAC packets have minor timestamp discontinuities
    (common after NVENC re-encodes).
    """
    from .loudnorm_measure import build_attenuate_filter

    volume = build_attenuate_filter(measured)
    # Defensive invariant: a loudnorm normalization (which boosts quiet content
    # toward a target) is forbidden on this attenuate-only path.
    if "loudnorm" in volume:
        raise RuntimeError(
            "OST audio filter must be attenuate-only (no loudnorm boost) — "
            f"got: {volume!r}. See build_attenuate_filter and the "
            "near-silent policy in encoding_constants.py."
        )
    return f"aresample=async=1000,{volume}"


# Font path — DejaVu Sans Bold is the canonical GoMoveShift brand font.
# Validated at load: it is interpolated bare into drawtext=fontfile='...'.
FONT_PATH = _validate_font_path(_DEFAULT_FONT_PATH)

# ffmpeg timeout for the OST re-encode step. This is the ONLY re-encode that
# produces a drive-path LF FINAL, and it owns a whole-file ``+faststart``
# moov atom rewrite on top of the NVENC re-encode (measured wall time ~1.10x
# input duration). The prior flat 2h (7200s) default was fine for a ~14min
# LF but starves a multi-hour superlong part (the part planner allows up to
# a 4h cap) — this is the absolute ceiling (``max_timeout``) passed to
# ``run_ffmpeg_encode``, which also stall-detects and cleans up a partial
# output on any kill, so a real hang is caught by the stall window below,
# not this ceiling. Scale with the probed input duration, mirroring
# single_pass._calc_encode_timeout's shape (multiplier * duration + an
# explicit faststart buffer), floored at the flat default so no input gets a
# tighter timeout than it had; short LFs are unaffected. Env override always wins
# (manual tuning), capped at 24h so an unbounded value can't hang a run indefinitely.
_OST_ENCODE_TIMEOUT_SEC_DEFAULT = 7200  # floor — unchanged from the prior flat timeout
_OST_ENCODE_TIMEOUT_MULTIPLIER = 2.0  # generous margin over the ~1.10x measured cost
_OST_FASTSTART_BUFFER_FRAC = 0.3
_OST_FASTSTART_BUFFER_FLOOR_SEC = 120


def _ost_encode_timeout_sec(duration_sec: float) -> int:
    raw = os.environ.get("GMS_OST_ENCODE_TIMEOUT_SEC", "")
    if raw.isdigit():
        # Cap at 24h — an unbounded env value could hang a run indefinitely.
        return min(int(raw), 86400)
    faststart_buffer = max(
        _OST_FASTSTART_BUFFER_FLOOR_SEC, int(duration_sec * _OST_FASTSTART_BUFFER_FRAC)
    )
    scaled = int(duration_sec * _OST_ENCODE_TIMEOUT_MULTIPLIER) + faststart_buffer
    return max(_OST_ENCODE_TIMEOUT_SEC_DEFAULT, scaled)


def _resolve_encoding_args(
    info: dict,
    prefer_nvenc: bool,
    input_name: str,
) -> list:
    """Pick NVENC encoding args for the OST burn-in re-encode.

    When ``prefer_nvenc=True`` and the first ``detect_nvenc()`` probe
    raises ``NVENCUnavailableError``, retry once after a short sleep —
    NVENC sessions can take a few seconds to release after a large
    encode finishes, and a bare probe during that window can spuriously
    fail. Second probe failure propagates: there is no libx265
    fallback path, so any unrecoverable probe failure aborts
    the run rather than silently CPU-encoding for hours.
    """
    from .color import detect_nvenc, get_encoding_args
    from .color_filters import NVENCUnavailableError

    # First probe. detect_nvenc() raises NVENCUnavailableError on any
    # probe failure. When
    # prefer_nvenc=True (the master encode used NVENC, so we expect it
    # here too), allow ONE retry after a short sleep to ride out
    # transient post-encode session release before propagating the error.
    try:
        has_nvenc = detect_nvenc()
    except NVENCUnavailableError:
        if not prefer_nvenc:
            raise
        logger.warning(
            "OST apply: NVENC probe failed for %s despite prefer_nvenc; "
            "retrying after 5s (NVENC sessions may be settling)",
            input_name,
        )
        time.sleep(5)
        # Second probe — let NVENCUnavailableError propagate.
        has_nvenc = detect_nvenc()
    return get_encoding_args(
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        has_nvenc=has_nvenc,
        source_bitrate_kbps=info["bitrate_kbps"],
        normalize_audio=True,
    )


# Reference video width for font scaling. At 1920, the base sizes apply.
# Above 1920, sizes scale up linearly with a 1.5x cap. Below 1920, sizes
# are floored at the base (don't shrink — readability matters more than
# proportionality at low resolutions).
_REF_WIDTH = 1920
_SCALE_CAP = 1.5

# Base font sizes (at the reference width)
_BRAND_FONTSIZE = 64
_LOCATION_FONTSIZE = 48
_CTA_FONTSIZE = 48

# Drawtext y-positions as fractions of video height
_BRAND_Y_FRAC = 0.68
_LOCATION_Y_FRAC = 0.76

# Long-form burned-location wrapping. Lines are stacked at
# _LOCATION_LINE_SPACING × fontsize; the wrapped block is measured against this
# fraction of the frame width, and a wrap beyond _LOCATION_MAX_LINES logs a WARN
# (too long to sit cleanly over the footage).
_LOCATION_LINE_SPACING = 1.25
_LOCATION_WIDTH_FRAC = 0.85
_LOCATION_MAX_LINES = 3

# Brand / CTA text — re-exported from ffmpeg_filters as the single source so
# the LF and shorts surfaces can't drift on the literal strings.
# Private aliases keep the existing call sites in this module unchanged.
_BRAND_TEXT = BRAND_TEXT
_YOUTUBE_CTA_TEXT = YOUTUBE_CTA_TEXT
_WEBSITE_TEXT = WEBSITE_TEXT
_FADE_DURATION = 1.0  # seconds, for drawtext alpha fade in/out

# Contrast treatment. Every burned text entry gets a
# border AND a drop shadow. The shorts renderer
# (``ffmpeg_filters.build_ost_filter``) has always drawn both; this surface drew
# the border only, so the SAME brand text read differently on a long-form than
# on a short — a thin dark border alone loses the glyph edge against bright sky.
# Values mirror the shorts surface: black@0.6, 4px at 48pt, 3px at 38pt.
_SHADOW_COLOR = "black@0.6"
_SHADOW_OFFSET_RATIO = 1 / 12

# CTA timing — offsets from end of video
_YOUTUBE_CTA_DURATION = 6.0  # seconds visible
_YOUTUBE_CTA_END_OFFSET = 7.0  # gap before website CTA starts
_WEBSITE_DURATION_SHORT = 3.5
_WEBSITE_DURATION_LF = 6.0
_CTA_END_MARGIN = 0.5  # breathing room before video ends

# Superlong recurring LOCATION lower-thirds (additive — no-op unless a
# caller passes ``lower_thirds``). Defaults mirror
# config.drive.superlong_lower_third_{duration,min_gap}_sec.
_LOWER_THIRD_DURATION_SEC_DEFAULT = 8.0
_LOWER_THIRD_MIN_GAP_SEC_DEFAULT = 45.0
_LOWER_THIRD_FONTSIZE = 40
_LOWER_THIRD_Y_FRAC = 0.90


LAYOUTS = {
    "regular_lf": {
        # DEPRECATED for regular_lf — kept for back-compat with tests /
        # YAML configs. The regular_lf LOCATION window is now [0, M]
        # (full-montage with tail crossfade) where M =
        # ``config.drive.montage_max_duration_sec``, threaded into
        # ``_build_drawtext_filter`` via the ``montage_duration`` kwarg.
        # BRAND shares that same [0, M] window and crossfades out WITH the
        # location — it is NOT persistent full-video. The old
        # "persistent, no enable, no alpha fade" note described earlier
        # behaviour and reintroducing it burns the brand over the whole video.
        "location_start_sec": 2.0,
        "location_duration_sec": 8.0,
        "show_youtube_cta": False,
        "show_website": True,
        "website_duration_sec": _WEBSITE_DURATION_LF,
        "cta_y_frac": 0.75,  # lower area — standard for landscape
        "show_persistent_url": True,  # top-right watermark for full LF
        # Brand logo overlay disabled for regular_lf — collides with YouTube's
        # channel watermark on the YouTube surface.
        # Drive/embed surfaces accept no-logo for now. The overlay capability is
        # preserved in apply_ost() — any layout that opts in via show_logo=True
        # still gets the bottom-right 240px PNG.
        "show_logo": False,
    },
    "short": {
        # Aligned to the LIVE shorts window (ffmpeg_filters.build_ost_filter:
        # ost_show_duration = 5.0 → title shown t∈[0, 5]). This is the
        # still-tested apply_ost capability, NOT the live renderer; keeping the
        # windows equal is what "the two are kept aligned" (module docstring)
        # actually requires — they had drifted to 0.5-8s.
        "location_start_sec": 0.0,
        "location_duration_sec": 5.0,
        "show_youtube_cta": True,
        "youtube_cta_duration_sec": _YOUTUBE_CTA_DURATION,
        "youtube_cta_end_offset_sec": _YOUTUBE_CTA_END_OFFSET,
        "show_website": True,
        "website_duration_sec": _WEBSITE_DURATION_SHORT,
        "cta_y_frac": 0.25,  # upper third — avoids mobile platform UI chrome
    },
}


def _escape_drawtext(text: str) -> str:
    """Escape a string for use inside an ffmpeg drawtext filter's `text=` field.

    Thin delegate to the shared ``escape_ffmpeg_text`` (``ffmpeg_filters``)
    so the long-form and shorts surfaces cannot diverge again. The long-form
    path historically escaped neither ``{``/``}`` (letting a literal ``"Route
    {A}"`` burn as a render-time specifier) and used a different ``'`` escape
    than the shorts path. Both are now the single documented form.
    """
    return escape_ffmpeg_text(text)


def _font_scale(video_width: int) -> float:
    """Scale factor for fonts based on video width (1.0x floor, 1.5x cap)."""
    return max(1.0, min(video_width / _REF_WIDTH, _SCALE_CAP))


def _shadow_offset(fontsize: int) -> int:
    """Drop-shadow offset in pixels for a drawtext entry at ``fontsize``.

    Proportional rather than fixed so a 4K LF (fonts scaled up to 1.5x) keeps
    the shadow in the same visual relationship to the glyph as a 1080p frame.
    The ratio reproduces the shorts burn surface exactly at scale 1.0: 4px at
    48pt (the location title) and 3px at 38pt (the CTA). Never rounds to zero.
    """
    return max(1, round(fontsize * _SHADOW_OFFSET_RATIO))


def _shadow_tokens(fontsize: int) -> str:
    """``shadowcolor``/``shadowx``/``shadowy`` drawtext tokens, no outer colons.

    Callers place this AFTER the entry's ``y=`` token. drawtext options are
    order-independent, but ``shadowy=`` contains the substring ``y=``, so a
    reader splitting an entry on ``y=`` to recover its vertical anchor must
    meet the real one first.
    """
    offset = _shadow_offset(fontsize)
    return f"shadowcolor={_SHADOW_COLOR}:shadowx={offset}:shadowy={offset}"


def _alpha_expr(start: float, end: float, fade: float) -> str:
    """Build a drawtext `alpha` expression for smooth fade in/out."""
    return (
        f"if(lt(t\\,{start})\\,0\\,"
        f"if(lt(t\\,{start + fade})\\,"
        f"(t-{start})/{fade}\\,"
        f"if(lt(t\\,{end - fade})\\,1\\,"
        f"if(lt(t\\,{end})\\,"
        f"({end}-t)/{fade}\\,0))))"
    )


def _build_location_crossfade_alpha(montage_duration: float, crossfade: float) -> str:
    """Linear crossfade-out alpha for the LOCATION title.

    Full opacity from t=0 until t = ``montage_duration - crossfade``;
    linear fade to 0 over the last ``crossfade`` seconds of the
    montage; then 0 for the remainder of the video.

    Piecewise:
      * ``t < fade_start``              → 1
      * ``fade_start <= t < M``         → 1 - (t - fade_start) / crossfade
      * ``t >= M``                      → 0
    """
    fade_start = montage_duration - crossfade
    return (
        f"if(lt(t,{fade_start}),1,if(lt(t,{montage_duration}),1-(t-({fade_start}))/{crossfade},0))"
    )


# Default crossfade (seconds) for the LOCATION title tail — callers
# normally override via ``config.drive.ost_title_crossfade_sec``.
_DEFAULT_OST_TITLE_CROSSFADE_SEC = 0.5


def _prepare_lower_thirds(
    entries: Optional[list],
    *,
    video_duration: float,
    opener_end_sec: float,
    cta_start_sec: Optional[float],
    duration_sec: float,
    min_gap_sec: float,
) -> list:
    """Raw ``(place_label, event_time_sec)`` events -> clamped, non-
    overlapping ``(label, start, end)`` lower-third windows. Hard
    fail on an empty/whitespace label. Ascending event-time order; a window
    is dropped when it overlaps the opener (``[0, opener_end_sec]``) or the
    end-card CTA (``[cta_start_sec, video_duration]``), or starts within
    ``min_gap_sec`` of the previously KEPT window's end."""
    entries = list(entries or [])
    for label, _t in entries:
        if not label or not str(label).strip():
            raise ValueError(
                "OST-EMPTY-LOWER-THIRD-LABEL: refusing to burn a superlong "
                f"lower-third with an empty place label (events: {entries!r}) "
                "— an upstream state defect; inspect the town-event source."
            )
        if ";" in str(label):
            raise ValueError(
                "OST-INTERNAL-SEPARATOR-LOWER-THIRD-LABEL: refusing to burn a "
                f"superlong lower-third whose label contains ';' ({label!r}) — "
                'an OSM concurrency join ("I 81;US 58") that leaked past the '
                "chapter road filter. ';' also separates drawtext chains in the "
                "filtergraph and escape_ffmpeg_text does not escape it."
            )
    kept: list = []
    last_end = float("-inf")
    for label, event_time in sorted(entries, key=lambda e: e[1]):
        start = max(0.0, float(event_time))
        end = min(video_duration, start + duration_sec)
        if end <= start or start < opener_end_sec:
            continue
        if cta_start_sec is not None and end > cta_start_sec:
            continue
        if start - last_end < min_gap_sec:
            continue
        # A repeat of the previously KEPT label re-burns one place. Upstream
        # collapses adjacent duplicates too, but cannot see a repeat whose
        # intervening label the min-gap rule drops HERE. ``last_end`` is left
        # untouched so the earlier window survives.
        if kept and kept[-1][0] == str(label):
            continue
        kept.append((str(label), start, end))
        last_end = end
    return kept


def _kit_text_color(brand_kit: Optional[dict], alpha: str = "1.0") -> str:
    """``fontcolor=`` value for an OST drawtext entry.

    ``brand_kit["colors"]["text"]`` (a ``#RRGGBB`` hex, already validated by
    ``console.settings.brand_kit``) when present, else the pre-kit default
    ``white`` — so a caller that never passes ``brand_kit`` (every production
    call site today, and every OST test written before the kit existed) gets
    the byte-identical ``white@{alpha}`` string.
    """
    hexval = None
    if brand_kit:
        hexval = (brand_kit.get("colors") or {}).get("text")
    return f"{hexval or 'white'}@{alpha}"


def _location_drawtext_filters(
    location: str,
    font_path: str,
    location_fontsize: int,
    video_width: int,
    y_frac: float,
    alpha_expr: str,
    enable_expr: str,
    fontcolor: str = "white@1.0",
) -> list:
    """Build one centered drawtext per wrapped line of the burned location title.

    The long-form OST overlay had NO wrapping: a single centered drawtext
    overflowed the frame on long ``"A to B via Road, ST"`` titles validated
    under the 100-char ``lf_title`` budget. Wrap on PIL metrics pinned to the
    EXACT ffmpeg ``FONT_PATH`` (not a Liberation/Ubuntu fallback),
    then stack the lines vertically centered on ``y_frac`` so the block keeps
    the same anchor as the old single line. Every line shares the title's alpha
    crossfade + enable window. A wrap beyond ``_LOCATION_MAX_LINES`` logs a WARN
    (the burn proceeds — wrapped text fits by construction).

    ``fontcolor`` defaults to the pre-brand-kit ``white@1.0`` so a caller that
    never passes it keeps the original literal string.
    """
    max_width = int(video_width * _LOCATION_WIDTH_FRAC)
    lines = wrap_text_for_ffmpeg(location, location_fontsize, max_width, font_path) or [location]
    if len(lines) > _LOCATION_MAX_LINES:
        logger.warning(
            "OST: burned location %r wrapped to %d lines (> %d) at %dpx — title crowds the frame",
            location,
            len(lines),
            _LOCATION_MAX_LINES,
            video_width,
        )
    line_height = int(location_fontsize * _LOCATION_LINE_SPACING)
    n = len(lines)
    out = []
    for i, line in enumerate(lines):
        esc = _escape_drawtext(line)
        offset = int((i - (n - 1) / 2) * line_height)
        sign = "+" if offset >= 0 else "-"
        out.append(
            f"drawtext="
            f"fontfile='{font_path}':"
            f"text='{esc}':"
            f"fontsize={location_fontsize}:"
            f"fontcolor={fontcolor}:"
            f"borderw=3:"
            f"bordercolor=black@0.8:"
            f"x=(w-text_w)/2:"
            f"y=h*{y_frac}{sign}{abs(offset)}:"
            f"{_shadow_tokens(location_fontsize)}:"
            f"alpha='{alpha_expr}':"
            f"enable='{enable_expr}'"
        )
    return out


def _build_drawtext_filter(
    location: str,
    layout: dict,
    font_path: str,
    video_width: int,
    video_duration: float,
    *,
    montage_duration: Optional[float] = None,
    crossfade_sec: float = _DEFAULT_OST_TITLE_CROSSFADE_SEC,
    lower_thirds: Optional[list] = None,
    lower_third_duration_sec: float = _LOWER_THIRD_DURATION_SEC_DEFAULT,
    lower_third_min_gap_sec: float = _LOWER_THIRD_MIN_GAP_SEC_DEFAULT,
    brand_kit: Optional[dict] = None,
    media_profile: Optional[object] = None,
) -> str:
    """Build the ffmpeg `-vf` filter chain for the OST overlay.

    The chain has three drawtext entries (brand, location, CTAs) each
    positioned relative to the FINAL output's timeline.

    When ``montage_duration`` is provided:
      * LOCATION and BRAND share an ``enable=between(t,0,M)`` gate and
        the same alpha crossfade-out over the last ``crossfade_sec``
        seconds of the montage. Both fade to zero together at end of M.
      * Logo + URL watermarks (corner overlays, applied elsewhere) remain
        full-video — those are NOT center-screen text and don't apply
        here.

    When ``montage_duration`` is ``None`` (shorts + legacy callers):
      * The layout's ``location_start_sec`` / ``location_duration_sec``
        fields drive a symmetric fade-in / fade-out window, and BRAND
        shares that window.

    ``lower_thirds``: optional ``(place_label, event_time_sec)`` town-
    transition events (superlong parts only). ``None``/empty is a no-op —
    byte-identical to before this parameter existed. See
    :func:`_prepare_lower_thirds` for the clamp/overlap/gap contract.

    ``brand_kit``: optional dict from ``console.settings.brand_kit()``.
    ``None`` (every call site today) is byte-identical to the pre-kit
    filter — text stays ``white`` at every entry. When present, its
    ``colors.text`` overrides every ``fontcolor=`` in this filter.
    """
    scale = _font_scale(video_width)
    brand_fontsize = int(_BRAND_FONTSIZE * scale)
    location_fontsize = int(_LOCATION_FONTSIZE * scale)
    cta_fontsize = int(_CTA_FONTSIZE * scale)
    lower_third_fontsize = int(_LOWER_THIRD_FONTSIZE * scale)

    text_color = _kit_text_color(brand_kit)
    brand_text = getattr(media_profile, "brand_text", None) or _BRAND_TEXT
    website_text = getattr(media_profile, "website_text", None) or _WEBSITE_TEXT
    brand_esc = _escape_drawtext(brand_text)

    if montage_duration is not None:
        # regular_lf path (brand shares montage gate + crossfade with location).
        # Clamp the (config-fixed ~15s) montage_duration to the actual video
        # length. A sub-15s LF would otherwise hold the title at full opacity to
        # t=15 — past EOF — never reaching its crossfade-out. Clamping keeps the
        # title fading out within the clip.
        if montage_duration > video_duration:
            logger.info(
                "OST: clamping montage_duration %.3gs -> video_duration %.3gs "
                "(sub-montage LF) so the title fades out within the clip",
                montage_duration,
                video_duration,
            )
            montage_duration = video_duration
        loc_enable = f"between(t,0,{montage_duration})"
        loc_alpha_expr = _build_location_crossfade_alpha(montage_duration, crossfade_sec)

        # Brand — gated to montage window, crossfades out with location.
        filters = [
            f"drawtext="
            f"fontfile='{font_path}':"
            f"text='{brand_esc}':"
            f"fontsize={brand_fontsize}:"
            f"fontcolor={text_color}:"
            f"borderw=3:"
            f"bordercolor=black@0.8:"
            f"x=(w-text_w)/2:"
            f"y=h*{_BRAND_Y_FRAC}:"
            f"{_shadow_tokens(brand_fontsize)}:"
            f"alpha='{loc_alpha_expr}':"
            f"enable='{loc_enable}'"
        ]

        # Location — crossfades out at the end of the montage. Wrapped.
        filters.extend(
            _location_drawtext_filters(
                location,
                font_path,
                location_fontsize,
                video_width,
                _LOCATION_Y_FRAC,
                loc_alpha_expr,
                loc_enable,
                fontcolor=text_color,
            )
        )
    else:
        # Shorts / legacy callers — fall back to LAYOUTS timing fields.
        loc_start = float(layout["location_start_sec"])
        loc_duration = float(layout["location_duration_sec"])
        loc_end = loc_start + loc_duration
        loc_alpha = _alpha_expr(loc_start, loc_end, _FADE_DURATION)

        # Brand (GoMoveShift) — above the location text, same enable window
        filters = [
            f"drawtext="
            f"fontfile='{font_path}':"
            f"text='{brand_esc}':"
            f"fontsize={brand_fontsize}:"
            f"fontcolor={text_color}:"
            f"borderw=3:"
            f"bordercolor=black@0.8:"
            f"x=(w-text_w)/2:"
            f"y=h*{_BRAND_Y_FRAC}:"
            f"{_shadow_tokens(brand_fontsize)}:"
            f"alpha='{loc_alpha}':"
            f"enable='between(t,{loc_start},{loc_end})'"
        ]

        # Location text — below the brand. Wrapped.
        filters.extend(
            _location_drawtext_filters(
                location,
                font_path,
                location_fontsize,
                video_width,
                _LOCATION_Y_FRAC,
                loc_alpha,
                f"between(t,{loc_start},{loc_end})",
                fontcolor=text_color,
            )
        )

    cta_y = layout.get("cta_y_frac", 0.75)

    # YouTube CTA — cross-promote long-form content (shorts only)
    if layout.get("show_youtube_cta"):
        yt_duration = layout.get("youtube_cta_duration_sec", _YOUTUBE_CTA_DURATION)
        yt_end_offset = layout.get("youtube_cta_end_offset_sec", _YOUTUBE_CTA_END_OFFSET)
        min_for_yt = yt_duration + yt_end_offset + 1.0
        if video_duration > min_for_yt:
            yt_end = video_duration - yt_end_offset
            yt_start = yt_end - yt_duration
            yt_alpha = _alpha_expr(yt_start, yt_end, _FADE_DURATION)
            yt_esc = _escape_drawtext(_YOUTUBE_CTA_TEXT)
            filters.append(
                f"drawtext="
                f"fontfile='{font_path}':"
                f"text='{yt_esc}':"
                f"fontsize={cta_fontsize}:"
                f"fontcolor={text_color}:"
                f"borderw=3:"
                f"bordercolor=black@0.8:"
                f"x=(w-text_w)/2:"
                f"y=h*{cta_y}:"
                f"{_shadow_tokens(cta_fontsize)}:"
                f"alpha='{yt_alpha}':"
                f"enable='between(t,{yt_start},{yt_end})'"
            )

    # Persistent URL watermark — top-right, visible for full video.
    # Separate from the end-card CTA below; this is the always-on brand
    # attribution (mirror of the old non-day-first filtergraph._apply_url_text).
    if layout.get("show_persistent_url"):
        url_esc = _escape_drawtext(website_text)
        # Smaller than end-card (persistent watermark shouldn't dominate)
        url_fontsize = int(_CTA_FONTSIZE * 0.6 * scale)
        filters.append(
            f"drawtext="
            f"fontfile='{font_path}':"
            f"text='{url_esc}':"
            f"fontsize={url_fontsize}:"
            f"fontcolor={_kit_text_color(brand_kit, '0.85')}:"
            f"borderw=2:"
            f"bordercolor=black@0.7:"
            f"x=w-text_w-40:"
            f"y=40:"
            f"{_shadow_tokens(url_fontsize)}"
        )

    # Website CTA — drive traffic to gomoveshift.com
    if layout.get("show_website"):
        ws_duration = layout.get("website_duration_sec", _WEBSITE_DURATION_LF)
        min_for_ws = ws_duration + _CTA_END_MARGIN + 1.0
        if video_duration > min_for_ws:
            ws_end = video_duration - _CTA_END_MARGIN
            ws_start = ws_end - ws_duration
            ws_alpha = _alpha_expr(ws_start, ws_end, _FADE_DURATION)
            ws_esc = _escape_drawtext(website_text)
            filters.append(
                f"drawtext="
                f"fontfile='{font_path}':"
                f"text='{ws_esc}':"
                f"fontsize={cta_fontsize}:"
                f"fontcolor={text_color}:"
                f"borderw=3:"
                f"bordercolor=black@0.8:"
                f"x=(w-text_w)/2:"
                f"y=h*{cta_y}:"
                f"{_shadow_tokens(cta_fontsize)}:"
                f"alpha='{ws_alpha}':"
                f"enable='between(t,{ws_start},{ws_end})'"
            )

    # Superlong recurring LOCATION lower-thirds — additive, no-op when
    # ``lower_thirds`` is None/empty (every regular_lf / short call today).
    if lower_thirds:
        opener_end_sec = montage_duration if montage_duration is not None else loc_end
        cta_start_sec = None
        if layout.get("show_website"):
            ws_duration = layout.get("website_duration_sec", _WEBSITE_DURATION_LF)
            min_for_ws = ws_duration + _CTA_END_MARGIN + 1.0
            if video_duration > min_for_ws:
                cta_start_sec = video_duration - _CTA_END_MARGIN - ws_duration
        prepared = _prepare_lower_thirds(
            lower_thirds,
            video_duration=video_duration,
            opener_end_sec=opener_end_sec,
            cta_start_sec=cta_start_sec,
            duration_sec=lower_third_duration_sec,
            min_gap_sec=lower_third_min_gap_sec,
        )
        for lt_label, lt_start, lt_end in prepared:
            lt_alpha = _alpha_expr(lt_start, lt_end, _FADE_DURATION)
            filters.extend(
                _location_drawtext_filters(
                    lt_label,
                    font_path,
                    lower_third_fontsize,
                    video_width,
                    _LOWER_THIRD_Y_FRAC,
                    lt_alpha,
                    f"between(t,{lt_start},{lt_end})",
                    fontcolor=text_color,
                )
            )

    return ",".join(filters)


def _probe_video_info(video_path: Path) -> dict:
    """Run ffprobe to get video width, height, fps, bitrate, and duration.

    Raises :class:`ProbeFailedError` if ffprobe cannot be invoked, returns
    a non-zero exit, or emits stdout that does not parse into a stream +
    format payload. The legacy behaviour returned hardcoded 4K@29.97
    defaults on any failure, which silently accepted truncated /
    half-written outputs on resume and let downstream encode + validation
    steps treat fabricated values as ground truth.

    Callers must trap explicitly when they want a soft fallback. The
    single in-tree caller (:func:`apply_ost`) traps and returns ``None``
    so its own failure contract (``Returns: ... None on any failure``)
    is preserved without forging dimensions or duration.

    Uses keyed ffprobe output to avoid index-order dependency.
    """
    import json as _json

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate:format=duration,bit_rate",
        "-of",
        "json",
        str(video_path),
    ]
    # A TimeoutExpired here is transient load (a heavy concurrent encode
    # pins the CPU past the 30s probe window), NOT corruption \u2014 retry it before
    # aborting the OST burn. Without this, a valid LF probed while another
    # heavy encode is running can abort its OST burn. Mirrors the
    # FFprobeClient contract: retry ONLY TimeoutExpired (linear backoff);
    # every NON-timeout failure (rc!=0, OSError, bad JSON) stays fatal on the
    # first attempt and is never retried.
    from .validation.ffprobe_client import (
        FFPROBE_TIMEOUT_BACKOFF_SEC,
        FFPROBE_TIMEOUT_RETRIES,
    )

    result = None
    last_timeout: Optional[subprocess.TimeoutExpired] = None
    for attempt in range(1, FFPROBE_TIMEOUT_RETRIES + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            break
        except subprocess.TimeoutExpired as exc:
            last_timeout = exc
            if attempt < FFPROBE_TIMEOUT_RETRIES:
                logger.warning(
                    "OST: ffprobe timed out for %s (attempt %d/%d) \u2014 transient "
                    "load, not corruption; retrying",
                    video_path,
                    attempt,
                    FFPROBE_TIMEOUT_RETRIES,
                )
                time.sleep(FFPROBE_TIMEOUT_BACKOFF_SEC * attempt)
        except Exception as e:  # noqa: BLE001 \u2014 non-timeout failures stay fatal
            logger.error(
                "\u26a0 OST: ffprobe failed for %s (%s) \u2014 raising "
                "ProbeFailedError (legacy hardcoded-defaults fallback removed; "
                "callers must decide explicitly)",
                video_path,
                e,
            )
            raise ProbeFailedError(video_path, f"subprocess error: {e}") from e
    if result is None:
        logger.error(
            "\u26a0 OST: ffprobe timed out for %s after %d attempts \u2014 raising "
            "ProbeFailedError (transient load, not corruption)",
            video_path,
            FFPROBE_TIMEOUT_RETRIES,
        )
        raise ProbeFailedError(video_path, f"subprocess error: {last_timeout}") from last_timeout

    if result.returncode != 0:
        stderr_tail = (result.stderr or b"").decode("utf-8", errors="replace")[-500:]
        logger.error(
            "\u26a0 OST: ffprobe exited rc=%d for %s \u2014 raising "
            "ProbeFailedError (stderr tail: %s)",
            result.returncode,
            video_path,
            stderr_tail,
        )
        raise ProbeFailedError(
            video_path,
            f"ffprobe rc={result.returncode}; stderr: {stderr_tail}",
        )

    info: dict = {
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "bitrate_kbps": 0,
        "duration": 0.0,
    }
    try:
        data = _json.loads(result.stdout)
        streams = data.get("streams", [])
        fmt = data.get("format", {})
        if not streams or not fmt:
            raise ValueError(
                f"ffprobe payload missing streams/format (streams={len(streams)}, "
                f"format_keys={list(fmt.keys()) if fmt else []})"
            )
        s = streams[0]
        info["width"] = int(s["width"])
        info["height"] = int(s["height"])
        fps_str = s.get("r_frame_rate", "")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            info["fps"] = float(num) / float(den)
        elif fps_str:
            info["fps"] = float(fps_str)
        else:
            raise ValueError("ffprobe stream missing r_frame_rate")
        info["duration"] = float(fmt["duration"])
        info["bitrate_kbps"] = int(fmt.get("bit_rate", 0)) // 1000
    except (ValueError, KeyError, ZeroDivisionError, _json.JSONDecodeError) as e:
        logger.error(
            "\u26a0 OST: ffprobe JSON parse failed for %s \u2014 raising "
            "ProbeFailedError (parse error: %s)",
            video_path,
            e,
        )
        raise ProbeFailedError(video_path, f"parse error: {e}") from e

    return info


def _has_audio_stream(video_path: Path) -> bool:
    """True when ``video_path`` carries at least one audio stream.

    A ``--no-audio`` drive encodes the day master ``-an``, so the LF segments
    stream-cut from it have NO audio stream. This probe lets ``apply_ost``
    emit ``-an`` instead of an ``-af`` audio filter that would fail (or attach
    to nothing) on a silent input — the burn re-encode must stay silent, not
    re-introduce an empty/erroring audio graph. Best-effort: a probe failure
    returns ``True`` (assume audio) so the existing attenuate path runs — the
    silent-master case is the one this guards, and it probes cleanly.
    """
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:  # noqa: BLE001 — degrade to "has audio" on any probe error
        logger.warning(
            "OST: audio-stream probe failed for %s (%s) — assuming audio present",
            video_path.name,
            e,
        )
        return True
    if result.returncode != 0:
        return True
    return bool((result.stdout or "").strip())


# ffmpeg writes HERE, never to the caller's final path, so a hard kill can
# only strand partial bytes at this sibling. Same directory (os.replace is
# atomic same-filesystem only); always trails the real extension
# (``lf_01.mp4.ost-partial``) so no `*.mp4` glob can adopt it.
_TEMP_OUTPUT_SUFFIX = ".ost-partial"

# ffmpeg selects its muxer from the output EXTENSION, and the temp name above
# ends in ``.ost-partial`` precisely so it is NOT a recognizable one — leaving
# ffmpeg to abort before writing a byte ("Unable to find a suitable output
# format … specify the format manually", rc 234). So every write to the temp
# path has to name the container the FINAL path implies. Derived here, in one
# place, rather than hardcoded at each call site: a fourth encode added later
# gets it by construction instead of rediscovering the failure in production.
_CONTAINER_BY_SUFFIX = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".mkv": "matroska"}


def _muxer_args(final_path: Path) -> list[str]:
    suffix = Path(final_path).suffix.lower()
    return ["-f", _CONTAINER_BY_SUFFIX.get(suffix, suffix.lstrip(".") or "mp4")]


def _discard_partial_output(output_path: Path, reason: str) -> None:
    """Delete a half-written burn artifact so nothing downstream can adopt it.

    A failed or timed-out ffmpeg still leaves whatever it had flushed on disk.
    Left there, the resume path sees a plausible-looking ``lf_NN.mp4`` and the
    uploader ships it. Every other encode in the pipeline unlinks on failure;
    this is the same rule for the OST burn.
    """
    try:
        if Path(output_path).exists():
            Path(output_path).unlink()
            logger.warning("OST: removed partial output %s (%s)", output_path, reason)
    except OSError as e:
        logger.error(
            "OST: could NOT remove partial output %s (%s) — delete it by hand "
            "before resuming, it is not a finished LF: %s",
            output_path,
            reason,
            e,
        )


def _stream_copy(input_path: Path, output_path: Path, final_path: Path | None = None) -> bool:
    """Stream-copy the input to the output with no re-encode. Used as the
    fallback when there's no OST to apply (empty location or missing font)
    so the pipeline still produces an output file.

    ``final_path`` names the container when ``output_path`` is the extensionless
    temp target (see ``_muxer_args``); it defaults to ``output_path`` for callers
    writing straight to a real filename."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        *_muxer_args(final_path or output_path),
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
    except Exception as e:
        logger.error("Stream copy failed for %s: %s", input_path, e)
        return False
    if result.returncode != 0:
        logger.error(
            "Stream copy returned rc=%d: %s",
            result.returncode,
            (result.stderr or b"").decode("utf-8", errors="replace")[-500:],
        )
        return False
    return True


def _validated_output(
    input_path: Path,
    temp_path: Path,
    final_path: Path,
    expected_duration: float,
    require_audio: bool,
    kind: str,
    *,
    final_pre_existed: bool,
) -> Optional[Path]:
    """The single exit through which every artifact ``apply_ost`` produces returns.

    A zero exit code does not prove a complete file: an ffmpeg killed after the
    last progress line — or one that lost its output mount mid-write — can exit
    0 having left a moov-less, short, or empty MP4 behind, and both resume and
    the uploader would treat it as a finished LF. Routing BOTH the drawtext
    re-encode and the missing-font stream-copy fallback through here means no
    path can hand back an output that was never checked against the input's
    probed duration (the fallback used to return before the input was even
    probed, so its copy shipped unchecked into an irreversible upload).

    ``temp_path`` is what ffmpeg actually wrote (never ``final_path`` — see
    ``_TEMP_OUTPUT_SUFFIX``). A valid ``temp_path`` is promoted onto
    ``final_path`` with ``os.replace`` (atomic, same filesystem) so a killed
    or invalid encode can never leave anything at ``final_path``: either the
    promotion never runs, or it runs only after validation passed.
    """
    validation = validate_video(
        temp_path,
        expected_duration=expected_duration,
        require_audio=require_audio,
    )
    if not validation.valid:
        logger.error(
            "OST-OUTPUT-INVALID: %s artifact for %s failed validation (%s) — "
            "discarding the partial file so resume redoes it instead of uploading it",
            kind,
            input_path.name,
            validation.reason,
        )
        _discard_partial_output(temp_path, "failed post-encode validation")
        if not final_pre_existed and final_path.exists():
            # Defensive: production ffmpeg only ever targets temp_path, so
            # this fires for a test stand-in that wrote straight to
            # final_path. A pre-existing final is never touched.
            _discard_partial_output(final_path, "failed post-encode validation")
        return None

    try:
        os.replace(temp_path, final_path)
    except FileNotFoundError:
        if not final_path.exists():
            logger.error(
                "OST-OUTPUT-MISSING: %s artifact for %s validated at %s but "
                "neither it nor %s exists on disk — refusing to report success",
                kind,
                input_path.name,
                temp_path,
                final_path,
            )
            return None
        # Nothing staged at temp_path to promote, but final_path already
        # holds the artifact — nothing to do.

    logger.info("OST %s on %s → %s", kind, input_path.name, final_path)
    return final_path


def apply_ost(
    input_path: Path,
    output_path: Path,
    location: str,
    layout: str,
    font_path: Optional[str] = None,
    brand_logo_path: Optional[str] = None,
    prefer_nvenc: bool = True,
    montage_duration: Optional[float] = None,
    crossfade_sec: float = _DEFAULT_OST_TITLE_CROSSFADE_SEC,
    lower_thirds: Optional[list] = None,
    lower_third_duration_sec: float = _LOWER_THIRD_DURATION_SEC_DEFAULT,
    lower_third_min_gap_sec: float = _LOWER_THIRD_MIN_GAP_SEC_DEFAULT,
    brand_kit: Optional[dict] = None,
    media_profile: Optional[object] = None,
) -> Optional[Path]:
    """Burn an OST location overlay onto a video with layout-specific timing.

    Args:
        input_path: Source video (day master, regular LF, or short).
        output_path: Where to write the OST-overlaid output.
        location: Title text from `title_builder`. Empty string means
            no overlay; the input is stream-copied to the output.
        layout: One of the keys in `LAYOUTS` — `"regular_lf"` or
            `"short"`. Invalid layouts raise ValueError.
        font_path: Optional override for the drawtext font file. Defaults
            to the module's FONT_PATH constant (DejaVu Sans Bold).
        brand_logo_path: Optional path to a PNG logo to overlay in the
            bottom-right corner at 240px wide, 85% opacity. Honored only
            for layouts with ``show_logo=True`` (16:9 LF formats — shorts
            skip to keep 9:16 frames clean). Missing file logs a warning
            and the encode proceeds without the logo.
        montage_duration: Total montage length in seconds. When provided
            (regular_lf path) the LOCATION title is visible from
            ``t=0`` through ``t=montage_duration`` with a linear
            crossfade out over the last ``crossfade_sec`` seconds, and
            the BRAND text shares that window and crossfade (only the logo
            and URL corner watermarks are full-video). When ``None`` (shorts or
            legacy callers), the layout's ``location_start_sec`` /
            ``location_duration_sec`` fields drive the window.
        crossfade_sec: Linear fade-out duration (seconds) at the tail of
            the montage for the LOCATION title. Default ``0.5`` —
            override via ``config.drive.ost_title_crossfade_sec``.
        lower_thirds: Optional ``(place_label, event_time_sec)`` town-
            transition events — superlong parts only. ``None``/empty is a
            no-op. See :func:`_prepare_lower_thirds` for the clamp contract.
        lower_third_duration_sec / lower_third_min_gap_sec: passed through
            to :func:`_prepare_lower_thirds`.
        brand_kit: Optional dict from ``console.settings.brand_kit()``.
            ``None`` (every call site today, since gomoveshift ships no
            brand.yaml kit) is byte-identical to the filter this function
            emitted before the brand kit existed. When present and
            ``font_path`` is not explicitly given, its ``font`` overrides
            the module default; its ``colors.text`` overrides every burned
            entry's ``fontcolor=``.

    Returns:
        ``output_path`` on success, or ``None`` on any failure (missing
        input, ffprobe failure — input is unreadable / truncated —,
        ffmpeg encode failure, or an output that fails ``_validated_output``
        — which now covers the missing-font stream-copy fallback as well as
        the drawtext re-encode). The legacy "ffprobe failure is tolerated"
        behaviour was removed (``_probe_video_info`` no longer falls back to
        silent defaults) — a truncated input now aborts the encode
        instead of being re-encoded with fabricated 4K@29.97 dimensions.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown layout {layout!r}; expected one of {sorted(LAYOUTS)}")

    input_path = Path(input_path)
    output_path = Path(output_path)
    # ffmpeg always writes here, never to output_path — see _TEMP_OUTPUT_SUFFIX.
    temp_output_path = output_path.with_name(output_path.name + _TEMP_OUTPUT_SUFFIX)
    final_pre_existed = output_path.exists()

    if not input_path.exists():
        logger.error("OST apply: input file not found: %s", input_path)
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # font_path (explicit override) wins; else the brand kit's font; else the
    # module default. brand_kit is None at every call site today, so this is
    # identical to the pre-kit `font_path or FONT_PATH`.
    font = (
        font_path
        or (brand_kit or {}).get("font")
        or getattr(media_profile, "font", None)
        or FONT_PATH
    )
    if not brand_logo_path:
        brand_logo_path = getattr(media_profile, "logo", None)

    # An empty/whitespace location reaching the burn is an UPSTREAM state
    # defect (the OST/title cascade should always resolve a label, gated by
    # validate_burned_ost_text), not graceful degradation. The legacy path
    # stream-copied a title-less video and reported SUCCESS (info log), so a
    # missing title shipped silently. Hard-fail loud with a greppable prefix.
    if not location or not location.strip():
        raise ValueError(
            f"OST-EMPTY-LOCATION: refusing to burn {input_path.name} with an "
            f"empty location title \u2014 the per-LF OST text cascade resolved to "
            f"{location!r}. This is an upstream state defect; inspect the "
            f"per_lf_ost_text / per_lf_locations cascade for this LF."
        )

    # Probe the input for width, height, fps, bitrate, and duration so the
    # filter can scale fonts and we can build proper encoder args. It runs
    # ahead of the font check below so the stream-copy fallback shares the
    # same probed expected duration to validate its own output against — it
    # used to return before this probe, with nothing to check the copy
    # against. A ProbeFailedError here means the input is not a usable video
    # — most commonly a half-written / truncated artifact left behind by a
    # crashed encode that resume is now revisiting. We refuse to forge
    # default dimensions (the legacy 4K@29.97 fallback was a source of
    # silent corruption) and return None so the
    # orchestrator surfaces a hard failure instead of producing a bad
    # output.
    try:
        info = _probe_video_info(input_path)
    except ProbeFailedError as e:
        logger.error(
            "OST apply: aborting encode for %s — ffprobe could not read the "
            "input (%s). Inspect / delete the file and re-run; the legacy "
            "hardcoded 3840x2160@29.97 fallback was removed because it let "
            "truncated artifacts pass through resume.",
            input_path,
            e,
        )
        return None

    # Graceful degradation: if the font file is missing, stream-copy the input
    # to the output. The pipeline still produces a file; there's just no
    # overlay. (Distinct from the empty-location case above \u2014 a missing font is an environment
    # issue, not a missing-title state defect.)
    if not Path(font).exists():
        logger.warning(
            "\u26a0 OST: font not found at %s \u2014 falling back to "
            "stream-copy (no title overlay) for %s",
            font,
            input_path.name,
        )
        if not _stream_copy(input_path, temp_output_path, output_path):
            _discard_partial_output(temp_output_path, "stream-copy failed")
            return None
        return _validated_output(
            input_path,
            temp_output_path,
            output_path,
            info["duration"],
            _has_audio_stream(input_path),
            "stream-copied",
            final_pre_existed=final_pre_existed,
        )

    layout_conf = LAYOUTS[layout]

    vfilter = _build_drawtext_filter(
        location=location,
        layout=layout_conf,
        font_path=font,
        video_width=info["width"],
        video_duration=info["duration"],
        montage_duration=montage_duration,
        crossfade_sec=crossfade_sec,
        lower_thirds=lower_thirds,
        lower_third_duration_sec=lower_third_duration_sec,
        lower_third_min_gap_sec=lower_third_min_gap_sec,
        brand_kit=brand_kit,
        media_profile=media_profile,
    )

    # Build encoder args. _resolve_encoding_args retries the NVENC probe
    # once when prefer_nvenc is True — see its docstring.
    enc_args = _resolve_encoding_args(info, prefer_nvenc, input_path.name)

    # --no-audio: a silent day master (encoded -an) yields LF segments with no
    # audio stream. Attaching an -af/-map 0:a filter to a stream that isn't
    # there would error or re-introduce an empty graph, so detect the absence
    # and burn the OST with -an — the LF stays truly silent through the burn.
    has_audio = _has_audio_stream(input_path)

    # Per-LF audio attenuation: when audio IS
    # present, run a loudness MEASUREMENT pass first, then feed the measured
    # stats into an ATTENUATE-ONLY ``volume=`` gain so each LF is pulled DOWN to
    # the near-silent ceiling — never boosted toward a target (audible
    # speech came from the old -20 LUFS boost). The measurement pass is
    # an audio-only decode (~100× faster than the NVENC video re-encode). On
    # measurement failure ``build_attenuate_filter`` applies a conservative
    # fixed attenuation (still never a boost). See the near-silent policy
    # in encoding_constants.py and the Phase D loudness ceiling gate.
    if has_audio:
        measured_stats = _measure_audio_loudnorm(input_path)
        audio_filter = _audio_attenuate_filter(measured_stats)
    else:
        audio_filter = None

    # Logo overlay: only for layouts that opt in (16:9 LFs), and only if
    # the PNG is on disk. Missing file logs a warning and drops the
    # watermark instead of failing — regression guard from the silent
    # drop in the old day-first path.
    use_logo = bool(brand_logo_path) and bool(layout_conf.get("show_logo"))
    logo_exists = False
    if use_logo:
        logo_exists = Path(brand_logo_path).is_file()
        if not logo_exists:
            logger.warning(
                "OST apply: brand_logo_path=%s not found — encoding without logo "
                "watermark (check config.brand_logo_path)",
                brand_logo_path,
            )

    if use_logo and logo_exists:
        # Two-input filter_complex: source video + logo PNG.
        # Chain: drawtext OST on source → overlay scaled logo → [vout].
        # Audio is mapped from stream 0; -af applies loudnorm independently
        # of the video filter_complex (valid ffmpeg: simple audio filtergraph
        # on the mapped audio stream).
        fc = (
            f"[0:v]{vfilter}[v1];"
            f"[1:v]scale=240:-1:flags=lanczos,format=rgba[logo];"
            f"[v1][logo]overlay=x=W-w-40:y=H-h-40[vout]"
        )
        audio_args = ["-map", "0:a?", "-af", audio_filter] if has_audio else ["-an"]
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-i",
            str(brand_logo_path),
            "-filter_complex",
            fc,
            "-map",
            "[vout]",
            *audio_args,
            *enc_args,
            "-movflags",
            "+faststart",
            *_muxer_args(output_path),
            str(temp_output_path),
        ]
    else:
        audio_args = ["-af", audio_filter] if has_audio else ["-an"]
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-vf",
            vfilter,
            *enc_args,
            *audio_args,
            "-movflags",
            "+faststart",
            *_muxer_args(output_path),
            str(temp_output_path),
        ]
    # run_ffmpeg_encode (the same Popen-based, stall-detecting runner
    # color_filters.encode_video uses) replaces a raw subprocess.run guarded
    # only by the multi-hour max_timeout above — a genuinely stalled ffmpeg
    # (process alive, no progress) is now killed within stall_timeout instead
    # of burning most of the wall-clock budget, and it cleans up its own
    # partial output on any failure (kill or non-zero exit). It targets
    # temp_output_path, same as cmd — a kill strands bytes only there.
    encode_result = run_ffmpeg_encode(
        cmd,
        output_path=temp_output_path,
        expected_duration=info["duration"],
        description=f"OST burn ({input_path.name})",
        stall_timeout=600,
        max_timeout=_ost_encode_timeout_sec(info["duration"]),
    )
    if not encode_result.success:
        reason = encode_result.killed_reason or f"rc={encode_result.returncode}"
        logger.error(
            "OST encode failed for %s (%s): %s",
            input_path.name,
            reason,
            encode_result.stderr_tail[-500:],
        )
        return None

    # This is the ONLY re-encode that produces a drive-path LF final, so it
    # gets the same post-encode integrity check every other encode in the
    # pipeline runs (single_pass, render_short_ffmpeg) — see _validated_output.
    return _validated_output(
        input_path,
        temp_output_path,
        output_path,
        info["duration"],
        has_audio,
        "burned",
        final_pre_existed=final_pre_existed,
    )
