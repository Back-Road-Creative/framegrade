"""
Single-Pass Pipeline — combines Color + OST + Audio into one FFmpeg encode.

Replaces the current 3-encode pipeline (Stages 3-5) with:
  Pass 1: Audio loudness measurement (audio-only decode, fast)
  Pass 2: Color correction + OST overlays + audio normalization in ONE encode

Performance impact for 4K@60fps 44-min video:
  Current:  Color encode (~5min) + OST encode (~70min) + Audio mux (~5min) = ~80min
  Proposed: Audio measure (~5min) + Single encode (~15min NVENC) = ~20min
  With RTX 4070 NVENC: ~20 minutes total vs ~80+ minutes current

Also eliminates generational quality loss from double video re-encode and
resolves the codec mismatch (color outputs HEVC, OST re-encodes to H.264).

Downstream stages unaffected:
  - Shorts extraction: operates on the final output, re-encodes to 9:16 anyway
  - Weather detection: uses signalstats on final output, read-only analysis
  - Thumbnails: frame extraction from final output
  - Metadata/chapters/social/schedule: no video processing

"""

import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import Config
from . import encoding_constants as ec
from .cache import (
    TRANSIENT_COLOR_CACHE_SCHEMA_VERSION,
    is_analysis_cached,
    load_cached_json,
    save_cache_json,
)
from .color import (
    CorrectionParams,
    FrameStats,
    _average_params,
    aggregate_segment_stats,
    analyze_segments,
    build_filter_chain,
    compute_correction,
    detect_nvenc,
    get_encoding_args,
    merge_adjacent_segments,
    params_to_lut3d_cube,
    sample_frames,
    scale_range_token,
)
from .color_filters import OUTPUT_FPS, cfr_filter, check_output_fps
from .ffmpeg_filters import (
    BRAND_TEXT,
    FONT_PATH,
    WEBSITE_TEXT,
    _validate_font_path,
    escape_ffmpeg_text,
)
from .ffmpeg_runner import run_ffmpeg_encode, run_ffmpeg_quick, validate_video
from .loudnorm_measure import (
    build_attenuate_filter,
    build_loudnorm_measure_cmd,
    parse_loudnorm_stats,
)
from .ost_validator import validate_ost
from .quality import rendition_for, upscale_plan

logger = logging.getLogger(__name__)

# One range-token mapper for the whole pipeline, re-exported under the historical
# private name. The grade chain and the analysis sampling must never drift onto
# two different definitions of "full" — see docs/pipeline-internals.md.
_scale_range_token = scale_range_token
# Working format for the grade chain. Nothing pinned one,
# so ffmpeg negotiated from the SOURCE depth: a standard 8-bit GoPro file put
# the curve bend, gamma stretch and range squeeze through `rgb24`, quantising
# every intermediate before the 10-bit encode ever saw the frame. Grading at
# >=10 bits and letting the encoder do one dithered downconvert is the standard
# anti-banding fix; it costs CPU in the filter chain and nothing else.
#
# gbrp16le is the domain the RGB grading filters (curves/colorbalance/lut3d)
# work in anyway, so pinning it adds headroom without adding a conversion.
# Verified not to disturb the downstream full->limited squeeze.
GRADE_WORKING_FORMAT = "gbrp16le"

# Pixel formats pinned through the privacy-blur composite.
#
# ``alphamerge`` is 8-bit-only in this build's default negotiation, and ffmpeg
# negotiates formats BACKWARDS: that requirement propagated up through the
# ``overlay``, through the ``split``, and into the base chain's ``scale``, which
# then emitted yuva420p. So the untouched 10-bit base was flattened to 8 bits
# before it ever reached the blur — the whole frame, not just masked regions,
# on every day master (blur is default-on) right before a 10-bit encode.
#
# BLUR_BASE_FORMAT is the load-bearing one: pinning the base BEFORE the split
# stops the backward propagation, so the masked path leaves pixels outside the
# mask bit-identical.
# The blur branch itself still passes through alphamerge's 8-bit surface, which
# is unavoidable here and immaterial — those pixels are boxblurred anyway.
BLUR_BASE_FORMAT = "yuv420p10le"
BLUR_ALPHA_FORMAT = "yuva420p10le"
BLUR_OVERLAY_FORMAT = "yuv420p10"

# The blur branch ([b2]) is boxblurred on a 1/BLUR_BRANCH_DOWNSCALE-size copy and
# scaled back up. Measured: the
# full-frame boxblur is the wall of the composite (3840x2160 grade + blur
# 2.20 fps with the branch discarded); at 1/4 resolution the composite ran 3.89x
# faster, pixels more than 3 px from the mask were identical and the masked
# area was at least as blurred. The [b1] base branch is untouched.
BLUR_BRANCH_DOWNSCALE = 4


def _hwdownload_format_for_pix_fmt(pix_fmt: Optional[str]) -> str:
    """Pick the ``hwdownload`` output format that MATCHES the source bit depth.

    When NVDEC decodes on the GPU (``-hwaccel_output_format cuda``) the
    CUDA frame surface carries the source's native bit depth. ``hwdownload``
    cannot widen/narrow during the copy, so its ``format=`` MUST match:
    a 10-bit (Main 10) CUDA surface downloads to ``p010le``; an 8-bit one
    to ``nv12``. The old ``format=nv12|p010le`` alternation did NOT solve
    this — ffmpeg constrains ``hwdownload`` to the FIRST listed format it
    can satisfy (``nv12``), so a 10-bit source aborted with
    ``[hwdownload] Invalid output format nv12 for hwframe download`` →
    ``hevc_nvenc … -22``. Probing the real ``pix_fmt`` and emitting the
    single correct format makes the chain work for ANY source by
    construction.

    Any >8-bit pixel format (``yuv420p10le``, ``yuv422p10le``, ``p010le``,
    ``yuv420p12le``, Main 10 …) → ``p010le``; everything else (incl.
    unknown/empty, the safe 8-bit default) → ``nv12``.
    """
    pf = (pix_fmt or "").lower()
    # 10/12-bit little-endian planar fmts and the 10-bit NV surface fmt.
    if "p010" in pf or "10le" in pf or "12le" in pf or "10be" in pf or "12be" in pf:
        return "p010le"
    return "nv12"


_COUNTRY_SUFFIXES = (
    ", United States",
    ", Ireland",
    ", UK",
    ", United Kingdom",
    ", Canada",
    ", Australia",
    ", New Zealand",
)


GRADE_FALLBACK_EQ = "eq_filter_chain"


def _bake_global_filter(
    corrections: list[CorrectionParams],
    *,
    strict: bool = False,
    fallback: Optional[list] = None,
) -> tuple[str, str, Optional[Path]]:
    """Bake ONE uniform grade to a LUT so the chain never hits ``eq``.

    Returns (filter_str, method, lut_dir); the caller owns ``lut_dir``. It is
    None — and nothing is left in /tmp — for an identity grade (nothing worth
    a LUT pass) or a failed bake, which degrades to the legacy 8-bit string
    chain rather than aborting a multi-hour master encode. That fallback is
    appended to ``fallback`` (when given) as ``GRADE_FALLBACK_EQ`` so the
    caller can record it; with ``strict`` it raises ``ValueError`` instead.
    """
    avg = corrections[0] if len(corrections) == 1 else _average_params(corrections)
    lut_dir = Path(tempfile.mkdtemp(prefix="gms_span_luts_"))
    try:
        lut = params_to_lut3d_cube(avg, output_path=lut_dir / "global.cube")
        filter_str = build_filter_chain([avg], lut_paths=[lut])
    except (ValueError, TypeError, OSError) as e:
        shutil.rmtree(lut_dir, ignore_errors=True)
        if strict:
            raise ValueError(
                f"GRADE-FALLBACK-REFUSED: the global LUT bake failed ({e}) and strict_grade "
                "is on, so the 8-bit eq filter chain will not be used. Nothing was encoded."
            ) from e
        logger.warning(f"  Color: global LUT bake failed ({e}) — using the 8-bit filter chain")
        if fallback is not None:
            fallback.append(GRADE_FALLBACK_EQ)
        return (build_filter_chain([avg]), "auto_global", None)
    if "lut3d=" not in filter_str:
        shutil.rmtree(lut_dir, ignore_errors=True)
        return (filter_str, "auto_global", None)
    return (filter_str, "auto_global", lut_dir)


def _bake_recording_fallback(
    proc: "SinglePassProcessor", corrections: list[CorrectionParams]
) -> tuple[str, str, Optional[Path]]:
    """``_bake_global_filter`` that records a fallback on ``proc`` (and honours
    ``config.drive.strict_grade``)."""
    reasons: list[str] = []
    strict = getattr(proc.config.drive, "strict_grade", False) is True
    result = _bake_global_filter(corrections, strict=strict, fallback=reasons)
    if reasons:
        proc._grade_fallback = reasons[0]
    return result


def _clean_location(location: str) -> str:
    """Strip the country suffix from an on-screen location overlay.

    "Not all footage is scenic drives" — the overlay makes no genre claim. A
    single location is shown as-is (no "Scenic Drive" appended); route-style
    descriptions (containing ' to ') from RouteAnalyzer already describe the
    journey.
    """
    for suffix in _COUNTRY_SUFFIXES:
        if location.endswith(suffix):
            location = location[: -len(suffix)]
            break
    return location


# Last format of every HDR / gamut conversion chain, shared by the encode
# `-vf`, the blur `filter_complex` and the colour-analysis sampler (all take
# the one resolved chain string, so they cannot diverge). It must keep at least
# the 10 bits the master is encoded at: the chains used to end `yuv420p`, an
# 8-bit hop (<=256 levels per channel, banding in skies) ahead of the 16-bit
# grade. 10-bit 4:2:0 is the grade's own input depth class, so `format=` into
# GRADE_WORKING_FORMAT is a lossless widening, and the chain stays YUV so the
# bt709/tv tags probed by `_probe_converted_sample` still describe it. The
# sampler converts to 8-bit RGB only at its own end, after this chain.
TONEMAP_OUTPUT_FORMAT = "yuv420p10le"

# HDR -> BT.709 tonemap chains, tried in order; the first whose filters all exist
# in the installed ffmpeg wins (probed by filter NAME, never by ffmpeg version).
# (label, operator, required filters, chain). Each chain ends limited-range
# BT.709 in TONEMAP_OUTPUT_FORMAT, so the SDR gate's re-check of the tagged result
# is truthful.
_HDR_TONEMAP_CHAINS = (
    (
        "zscale",
        "hable",
        frozenset({"zscale", "tonemap"}),
        "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
        "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=limited,"
        f"format={TONEMAP_OUTPUT_FORMAT}",
    ),
    (
        "libplacebo",
        "bt.2390",
        frozenset({"libplacebo"}),
        "libplacebo=tonemapping=bt.2390:colorspace=bt709:color_primaries=bt709:"
        f"color_trc=bt709:range=tv:format={TONEMAP_OUTPUT_FORMAT}",
    ),
)

# Wide-gamut (BT.2020 / P3) source with a KNOWN SDR transfer: same curve as
# BT.709, so only the primaries move. No tonemap (that would darken it).
_GAMUT_CHAIN = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    f"zscale=t=bt709:m=bt709:r=limited,format={TONEMAP_OUTPUT_FORMAT}"
)
_GAMUT_OPERATOR = "zscale:gamut"

_filter_names_cache: Optional[frozenset] = None
_libplacebo_ok_cache: Optional[bool] = None


def _ffmpeg_filter_names() -> frozenset:
    """Names of the filters the installed ffmpeg offers (cached; empty on failure)."""
    global _filter_names_cache
    if _filter_names_cache is None:
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30
            ).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        names = set()
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2 and len(parts[0]) <= 3:
                names.add(parts[1])
        _filter_names_cache = frozenset(names)
    return _filter_names_cache


def _libplacebo_usable() -> bool:
    """True when libplacebo can actually run here (it needs a Vulkan device).

    ``ffmpeg -filters`` lists libplacebo whether or not a GPU device exists, so
    the name alone is not enough: run it once on a tiny synthetic frame.
    """
    global _libplacebo_ok_cache
    if _libplacebo_ok_cache is None:
        try:
            rc = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=black:s=64x64:d=0.1",
                    "-vf",
                    "libplacebo=tonemapping=bt.2390:format=yuv420p",
                    "-frames:v",
                    "1",
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                timeout=60,
            ).returncode
        except (OSError, subprocess.SubprocessError):
            rc = 1
        _libplacebo_ok_cache = rc == 0
    return _libplacebo_ok_cache


def _probe_converted_sample(trc: str, primaries: str, chain: str) -> dict:
    """Run ``chain`` on a tiny synthetic frame tagged like the source and
    ffprobe the RESULT; returns ``color_trc``/``color_primaries``/``color_range``
    (empty dict when ffmpeg or ffprobe fails). This is what makes the SDR
    re-check real: it reads what the chain actually emits, not what we hoped.
    """
    matrix = "bt2020nc" if primaries == "bt2020" else "bt709"
    src = (
        "color=c=gray:s=64x64:d=0.1,format=yuv420p,"
        f"setparams=color_trc={trc}:color_primaries={primaries}:colorspace={matrix}:range=tv"
    )
    with tempfile.TemporaryDirectory(prefix="gms_tm_probe_") as tmp:
        out = Path(tmp) / "sample.mkv"
        try:
            enc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", src]
                + ["-vf", chain, "-frames:v", "1", "-c:v", "ffv1", str(out)],
                capture_output=True,
                timeout=60,
            )
            if enc.returncode != 0:
                return {}
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries"]
                + ["stream=color_transfer,color_primaries,color_range", "-of", "json", str(out)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            stream = json.loads(probe.stdout)["streams"][0]
        except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
            return {}
    return {
        "color_trc": str(stream.get("color_transfer", "")),
        "color_primaries": str(stream.get("color_primaries", "")),
        "color_range": str(stream.get("color_range", "")),
    }


def _tonemap_sidecar(output_path: Path) -> Path:
    stem = output_path.stem.removesuffix(".tmp")
    return output_path.with_name(f"{stem}.tonemap.json")


def _clear_tonemap_record(output_path: Path) -> None:
    """Drop a stale sidecar when an SDR master is written to the same stem."""
    _tonemap_sidecar(output_path).unlink(missing_ok=True)


def _record_tonemap(
    output_path: Path, operator: str, source: dict, grade_fallback: Optional[str] = None
) -> None:
    """Persist the tonemap operator next to the master (``<final stem>.tonemap.json``).

    Callers encode to ``<stem>.tmp<ext>`` and rename, so the sidecar is keyed on
    the final stem and survives the rename.
    """
    sidecar = _tonemap_sidecar(output_path)
    payload = {
        "tonemap": operator,
        "source_color_trc": str(source.get("color_trc", "")),
        "source_color_primaries": str(source.get("color_primaries", "")),
        "grade_fallback": grade_fallback,
    }
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")


@dataclass
class SinglePassResult:
    """Result of combined color + OST + audio processing."""

    success: bool
    output_path: Optional[Path] = None
    color_method: str = "none"
    ost_applied: bool = False
    audio_normalized: bool = False
    nvenc_used: bool = False
    encoding_time: float = 0.0
    total_time: float = 0.0
    tonemap: str = ""  # "<filter>:<operator>" when an HDR source was tonemapped
    # Why the grade degraded to the 8-bit eq chain (e.g. "eq_filter_chain"), else None.
    grade_fallback: Optional[str] = None
    # Declared resampling upscale applied by this encode (quality.UpscalePlan), else None.
    upscale_plan: Optional[object] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "output_path": str(self.output_path) if self.output_path else None,
            "color_method": self.color_method,
            "ost_applied": self.ost_applied,
            "audio_normalized": self.audio_normalized,
            "nvenc_used": self.nvenc_used,
            "encoding_time": round(self.encoding_time, 1),
            "total_time": round(self.total_time, 1),
            "tonemap": self.tonemap,
            "grade_fallback": self.grade_fallback,
        }


class SinglePassProcessor:
    """
    Combines Color Correction + OST + Audio Normalization into a single FFmpeg
    encode pass, eliminating two full re-encodes from the pipeline.

    Architecture:
        1. Color analysis (sample frames, compute corrections) — no encode
        2. Audio measurement (loudnorm pass 1) — audio-only decode
        3. Single FFmpeg encode with combined -vf and -af filters
           using NVENC when available

    The video filter chain order:
        [color_curves/eq] → [video_fade_in/out] → [drawtext_location] → [drawtext_subscribe]

    The audio filter:
        loudnorm with measured stats from step 2

    Downstream stages (shorts, weather, thumbnails) are unaffected — they
    operate on the final output just as before.
    """

    # Validated at load: it is interpolated bare into drawtext=fontfile='...'.
    DEFAULT_FONT = _validate_font_path(FONT_PATH)

    def __init__(self, config: Config, profile: Optional[object] = None):
        self.config = config
        # Why the last grade fell back to the 8-bit eq chain (None = it did not).
        self._grade_fallback: Optional[str] = None
        # Per-tenant MediaProfile (brand_policy.load_media_profile), set once;
        # None = engine defaults; the upscaler reads upscale/claim_native_resolution_only.
        self.profile = profile
        # Set True whenever _wrap_text falls back from PIL metrics to the
        # 0.55×fontsize char approximation, so the fallback is recorded (queryable
        # on the instance) rather than silently taking over the legacy burn path.
        self._wrap_pil_fallback_used: bool = False
        # detect_nvenc() raises NVENCUnavailableError on probe failure.
        # If construction returns, NVENC is available; downstream
        # `if not has_nvenc: libx265` branches are unreachable by design.
        self._nvenc_available: bool = detect_nvenc()

    def _has_nvenc(self) -> bool:
        return self._nvenc_available

    def _find_font(self, font_path: Optional[str] = None) -> Optional[str]:
        font = font_path or self.DEFAULT_FONT
        if Path(font).exists():
            return font
        fonts_root = Path(FONT_PATH).parent.parent
        fallbacks = [
            str(fonts_root / "liberation" / "LiberationSans-Bold.ttf"),
            str(fonts_root / "ubuntu" / "Ubuntu-Bold.ttf"),
        ]
        for fb in fallbacks:
            if Path(fb).exists():
                return fb
        return None

    def _escape_text(self, text: str) -> str:
        """Escape text for FFmpeg drawtext filter.

        Delegates to the canonical ``escape_ffmpeg_text`` so the day-master
        long-form overlay shares one escaper with ``ost_applier``. The shared
        helper escapes the five drawtext-special characters — ``\\ : ' % { }``
        — including the braces this method previously omitted (an unescaped
        ``{``/``}`` in a location label burns as a render-time specifier).
        """
        return escape_ffmpeg_text(text)

    @staticmethod
    def _pack_last_line(words: list[str], fits) -> str:
        """Pack as many ``words`` onto the final wrapped line as ``fits`` allows.

        ``max_lines`` caps the number of lines, so the final line must hold the
        remaining words — but a bare ``" ".join(words)`` burns one off-frame line
        when they don't all fit. Append U+2026 when words are dropped so the
        overflow is visible rather than silently clipped by the frame edge. A
        lone over-width word is returned intact (no mid-word break).
        """
        line = words[0]
        for w in words[1:]:
            candidate = f"{line} {w}"
            if fits(candidate):
                line = candidate
            else:
                return f"{line}…"
        return line

    def _wrap_text(
        self,
        text: str,
        fontsize: int,
        max_width_px: int,
        max_lines: int = 3,
        font_path: Optional[str] = None,
    ) -> list[str]:
        """
        Word-wrap text to fit within max_width_px at the given fontsize.

        Uses PIL ``ImageDraw.textbbox()`` for pixel-accurate measurement against
        the actual TrueType font (DejaVu Sans Bold). This matches what FFmpeg
        drawtext will render and avoids overflow/incorrect wrapping on variable-
        width glyphs that the old character-average approximation produced.

        Falls back to an approximate character width (0.55 * fontsize) only when
        PIL or the font file are unavailable (e.g. minimal test environments).

        Returns:
            List of text lines (1 to max_lines).
        """
        words = text.split()
        if not words:
            return []

        resolved_font = self._find_font(font_path)
        pil_lines: Optional[list[str]] = None
        if resolved_font is not None:
            try:
                from PIL import Image, ImageDraw, ImageFont

                font = ImageFont.truetype(resolved_font, fontsize)
                img = Image.new("RGB", (1, 1))
                draw = ImageDraw.Draw(img)

                def _fits(s: str) -> bool:
                    bbox = draw.textbbox((0, 0), s, font=font)
                    return (bbox[2] - bbox[0]) <= max_width_px

                lines: list[str] = []
                current_line = ""
                for word_idx, word in enumerate(words):
                    test_line = f"{current_line} {word}".strip() if current_line else word
                    if _fits(test_line):
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                        if len(lines) >= max_lines - 1:
                            # Last line: pack only what fits; a bare join of every
                            # remaining word burns one off-frame line.
                            current_line = self._pack_last_line(words[word_idx:], _fits)
                            break
                if current_line:
                    lines.append(current_line)
                pil_lines = lines[:max_lines]
            except Exception as e:  # noqa: BLE001 — fall back to approximation
                # The 0.55×fontsize char approximation silently takes over the
                # legacy burn path when PIL/font is unavailable — escalate to
                # WARNING and record it on the instance so it isn't invisible.
                logger.warning(
                    "single_pass._wrap_text: PIL font measurement failed (%s) — "
                    "using the 0.55×fontsize approximation, which may mis-wrap",
                    e,
                )
                self._wrap_pil_fallback_used = True
                pil_lines = None

        if pil_lines is not None:
            return pil_lines

        # Fallback: approximate average character width for DejaVu Sans Bold
        avg_char_width = fontsize * 0.55
        max_chars_per_line = max(10, int(max_width_px / avg_char_width))

        lines = []
        current_line = ""

        for word_idx, word in enumerate(words):
            test_line = f"{current_line} {word}".strip() if current_line else word
            if len(test_line) <= max_chars_per_line:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                if len(lines) >= max_lines - 1:
                    # Last line: pack only what fits (char-budget variant of the PIL branch above).
                    current_line = self._pack_last_line(
                        words[word_idx:],
                        lambda s: len(s) <= max_chars_per_line,
                    )
                    break

        if current_line:
            lines.append(current_line)

        return lines[:max_lines]

    # ── Timeout Calculations ─────────────────────────────────────────

    def _calc_audio_measurement_timeout(self, duration: float, height: int = 1080) -> int:
        """Calculate timeout for audio loudness measurement pass.

        Reads the full file for audio-only decode. 4K files are much larger
        on disk so I/O is slower even though no video decode occurs.

        The timeout scales with duration so a long 4K input does not time out.
        The pass uses -vn to skip video decode; audio-only runs well under
        realtime.
        """
        multiplier = 1.0 if height < 2160 else 1.5
        return max(300, int(duration * multiplier))

    def _calc_encode_timeout(self, duration: float, use_nvenc: bool = True) -> int:
        """Calculate the absolute wall-time ceiling for the single-pass encode.

        This is a runaway guard, not a speed estimate: ``run_ffmpeg_encode``'s
        600 s stall detector already kills a hung encode, so the ceiling must
        never fire on a progressing one. The nvenc path is CPU-bound in its
        filter chain (gbrp16le grade + lut3d + blur overlay at 4K runs 3-4 fps,
        ~0.11x wall) — on long 4K sources the wall time can reach ~8.5-9x the
        media duration, which a tighter cap (~8x) would kill near the end.
        20x nvenc / 40x x265 keeps ≥2x headroom over that worst case. Adds 30% for the faststart
        moov rewrite on large files.
        """
        timeout_mult = (
            20.0 if use_nvenc else 40.0
        )  # 4K grade+blur chain ~0.11x wall; x265 slow ~0.05x
        faststart_buffer = max(120, int(duration * 0.3))
        return max(600, int(duration * timeout_mult) + faststart_buffer)

    def _get_video_info(self, video_path: Path) -> dict:
        """Get video properties via ffprobe."""
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-select_streams",
            "v:0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return {"duration": 0, "height": 1080, "width": 1920, "fps": 30.0}

        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        duration = float(fmt.get("duration", 0))

        streams = data.get("streams", [])
        if not streams:
            return {"duration": duration, "height": 1080, "width": 1920, "fps": 30.0}

        stream = streams[0]
        height = int(stream.get("height", 1080))
        width = int(stream.get("width", 1920))
        # Source pixel format drives the NVDEC hwdownload bit depth (10-bit
        # GoPro Main 10 → p010le, 8-bit → nv12). See
        # ``_hwdownload_format_for_pix_fmt``.
        pix_fmt = str(stream.get("pix_fmt", ""))
        # Color signalling. ``color_range`` drives the
        # conditional full->limited squeeze (avoid double-compressing an
        # already-limited source); ``color_transfer``/``color_primaries`` feed
        # the SDR gate that refuses HDR sources the encoder would mis-tag as
        # BT.709. ffprobe omits these fields for untagged streams -> "".
        color_range = str(stream.get("color_range", ""))
        color_trc = str(stream.get("color_transfer", ""))
        color_primaries = str(stream.get("color_primaries", ""))
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) != 0 else 30.0
        else:
            fps = float(fps_str)

        # Extract source bitrate for quality floor enforcement
        bitrate_bps = int(fmt.get("bit_rate", 0))
        bitrate_kbps = bitrate_bps // 1000

        return {
            "duration": duration,
            "height": height,
            "width": width,
            "fps": fps,
            "bitrate_kbps": bitrate_kbps,
            "pix_fmt": pix_fmt,
            "color_range": color_range,
            "color_trc": color_trc,
            "color_primaries": color_primaries,
        }

    # Transfer functions / primaries that signal an HDR (or otherwise
    # non-BT.709) source. _resolve_hdr_source converts these to BT.709 ahead of
    # the grade; one that cannot be converted would be mis-graded and
    # mislabeled by the unconditional BT.709 SDR output tag, so it is refused.
    _HDR_TRC = frozenset({"smpte2084", "arib-std-b67", "smpte2084-10"})
    # SDR transfer curves (bt2020-10/12 are the BT.709 curve at wider bit depth).
    _SDR_TRC = frozenset(
        {"bt709", "smpte170m", "smpte240m", "bt2020-10", "bt2020-12", "iec61966-2-1"}
    )
    _NON_BT709_PRIMARIES = frozenset({"bt2020", "smpte431", "smpte432"})

    def _assert_sdr_source(self, info: dict) -> None:
        """Upstream gate: fail fast on HDR / non-BT.709.

        Nothing in the encode probes ``color_trc``/``color_primaries`` and
        with no conversion an HLG/PQ (HDR) or BT.2020 source would be graded
        as SDR and the output falsely stamped ``bt709``/``tv``, baking wrong
        colours and a wrong tag into a multi-hour irreversible master. Refusing
        the source here (before color analysis and the encode) is the upstream
        gate that keeps that from ever reaching an artifact.

        Empty/untagged fields are treated as SDR BT.709 — raw GoPro footage is
        untagged and is what this pipeline is built for.
        """
        trc = str(info.get("color_trc", "")).strip().lower()
        primaries = str(info.get("color_primaries", "")).strip().lower()
        if trc in self._HDR_TRC or primaries in self._NON_BT709_PRIMARIES:
            have = _ffmpeg_filter_names()
            gamut_only = trc not in self._HDR_TRC and trc in self._SDR_TRC
            lacking = (
                ""
                if trc in self._HDR_TRC or gamut_only
                else (
                    " The transfer characteristic is not a known curve, so no "
                    "conversion can be chosen: tag it (ffmpeg -color_trc) first."
                )
            )
            convert = (
                "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                "tonemap=hable,zscale=t=bt709:m=bt709:r=limited,format=yuv420p"
            )
            if gamut_only:
                convert = _GAMUT_CHAIN
                if "zscale" not in have:
                    lacking = (
                        " This ffmpeg lacks zscale (libzimg), so it cannot gamut-map "
                        "BT.2020/P3 to BT.709 here; run the conversion with an ffmpeg "
                        "that has it."
                    )
            elif trc in self._HDR_TRC and not (
                {"zscale", "tonemap"} <= have or ("libplacebo" in have and _libplacebo_usable())
            ):
                lacking = (
                    " This ffmpeg lacks zscale+tonemap (libzimg) and a usable libplacebo"
                    + (
                        " (it is listed but has no working Vulkan device)"
                        if "libplacebo" in have
                        else ""
                    )
                    + ", so it cannot tonemap here; run the conversion with an ffmpeg "
                    "that has them."
                )
            raise ValueError(
                "HDR / non-BT.709 source is not supported by the SDR encode "
                f"path (color_trc={trc or 'unset'!r}, "
                f"color_primaries={primaries or 'unset'!r}). The pipeline grades "
                "and tags SDR BT.709 only; encoding it untouched would mis-grade "
                f"and mislabel this master.{lacking} Convert the source to SDR "
                f"BT.709 before running the pipeline: ffmpeg -i SRC -vf {convert} "
                "-c:v libx265 -crf 14 -c:a copy OUT.mkv"
            )

    def _resolve_hdr_source(self, info: dict) -> tuple[dict, str, str]:
        """Return (info, tonemap_filter, operator) for the encode.

        An SDR source passes through untouched. An HDR one (a known HDR transfer
        curve) is tonemapped to BT.709 ahead of the grade when the installed
        ffmpeg has a working tonemap filter; the returned info is then tagged
        BT.709 limited and re-checked by ``_assert_sdr_source``. Anything else
        (no usable filter, or wide-gamut primaries with no known HDR transfer,
        which zscale cannot convert) is refused here, before color analysis.
        """
        trc = str(info.get("color_trc", "")).strip().lower()
        primaries = str(info.get("color_primaries", "")).strip().lower()
        if trc not in self._HDR_TRC and primaries not in self._NON_BT709_PRIMARIES:
            return info, "", ""
        available = _ffmpeg_filter_names()
        candidates: list[tuple[str, str]] = []  # (operator label, chain)
        if trc in self._HDR_TRC:
            for label, operator, needs, chain in _HDR_TONEMAP_CHAINS:
                if needs <= available and (label != "libplacebo" or _libplacebo_usable()):
                    candidates.append((f"{label}:{operator}", chain))
                    break
        elif trc in self._SDR_TRC and "zscale" in available:
            candidates.append((_GAMUT_OPERATOR, _GAMUT_CHAIN))
        for operator, chain in candidates:
            probed = _probe_converted_sample(trc, primaries, chain)
            if not probed:
                raise ValueError(
                    f"The {operator} chain could not be run on a converted sample, so "
                    "its BT.709 result cannot be verified; refusing the source."
                )
            self._assert_sdr_source(probed)
            if (
                probed.get("color_trc") != "bt709"
                or probed.get("color_primaries") != "bt709"
                or probed.get("color_range") != "tv"
            ):
                raise ValueError(
                    f"HDR / non-BT.709 conversion check failed: the {operator} chain "
                    f"produced {probed!r}, not bt709/bt709/tv; refusing the source."
                )
            return {**info, **probed}, chain, operator
        self._assert_sdr_source(info)  # raises: nothing can convert this source
        return info, "", ""  # pragma: no cover

    # ── Step 1: Color Analysis (no encode) ─────────────────────────────

    def _corrections_to_filter(
        self,
        corrections: list[CorrectionParams],
        span_durations: Optional[list[float]],
    ) -> tuple[str, str, Optional[Path]]:
        """Derive (filter_str, method, lut_dir) from corrections.

        Every grading path now reaches the encode as a baked .cube LUT (temp
        dir; the caller cleans it up): multi-span bakes one per span and
        stitches them with xfades, single-span bakes exactly one. The string
        chain is only the fallback for a failed bake — it ends in ``eq``, which
        drags the stream to 8-bit mid-chain.

        The method label is derived from the EMITTED filter, not the analysis
        intent: "auto_adaptive" is logged/cached only when per-span grading
        actually reaches the encode (an older label claimed adaptive while
        shipping the averaged grade).
        """
        if not corrections:
            return ("null", "none", None)
        if len(corrections) < 2 or not span_durations or len(span_durations) != len(corrections):
            return _bake_recording_fallback(self, corrections)

        lut_dir = Path(tempfile.mkdtemp(prefix="gms_span_luts_"))
        try:
            lut_paths = [
                params_to_lut3d_cube(c, output_path=lut_dir / f"span_{i:03d}.cube")
                for i, c in enumerate(corrections)
            ]
            filter_str = build_filter_chain(
                corrections, lut_paths=lut_paths, span_durations=span_durations
            )
        except (ValueError, TypeError, OSError) as e:
            logger.warning(f"  Color: adaptive filtergraph failed ({e}) — using averaged")
            shutil.rmtree(lut_dir, ignore_errors=True)
            return _bake_recording_fallback(self, corrections)
        if "xfade=" in filter_str:
            return (filter_str, "auto_adaptive", lut_dir)
        # Identity collapse — a single averaged grade was emitted.
        shutil.rmtree(lut_dir, ignore_errors=True)
        return _bake_recording_fallback(self, corrections)

    def _analyze_color(
        self,
        video_path: Path,
        tonemap_filter: str = "",
    ) -> tuple[str, str, list[CorrectionParams], Optional[Path]]:
        """
        Run histogram-based color analysis and return a filter string.

        Cache schema v3: the canonical artifact is
        ``corrections`` (one per merged lighting span) plus ``span_durations``;
        the filter is re-derived from them each call, baking per-span LUTs
        for the adaptive path. v1/v2 files invalidate on read — the first
        run after an upgrade re-analyzes the master.

        Returns:
            (filter_string, method_name, corrections_list, lut_dir)
            filter_string is "null" if no correction needed. lut_dir is a
            temp dir of baked .cube LUTs the caller must remove after the
            encode (None when no LUTs were baked).
        """
        no_sharpen = bool(self.config.drive.no_sharpen)
        no_denoise = bool(self.config.drive.no_denoise)
        # Check cache before expensive analysis
        cache_path = video_path.parent / f"{video_path.stem}_color_analysis.json"
        if is_analysis_cached(video_path, cache_path):
            cached = load_cached_json(cache_path)
            if (
                cached
                and cached.get("version") == TRANSIENT_COLOR_CACHE_SCHEMA_VERSION
                and "corrections" in cached
                and cached.get("tonemap", "") == tonemap_filter
            ):
                corrections = [CorrectionParams(**c) for c in cached["corrections"]]
                if all(
                    c.sharpen_disabled == no_sharpen and c.denoise_disabled == no_denoise
                    for c in corrections
                ):
                    filter_str, method, lut_dir = self._corrections_to_filter(
                        corrections, cached.get("span_durations")
                    )
                    logger.info(f"  Color: loaded from cache ({method})")
                    return (filter_str, method, corrections, lut_dir)
                logger.info("  Color: cached grade used different sharpen/denoise opt-outs")

        info = self._get_video_info(video_path)
        duration = info["duration"]

        # Measure in the domain the grade is applied in; the probe already ran.
        sample_kwargs: dict = {}
        source_range = scale_range_token(info.get("color_range", ""))
        if tonemap_filter:
            # HDR source: measure the post-tonemap frames (limited BT.709).
            sample_kwargs["tonemap_filter"] = tonemap_filter
            source_range = "limited"
        frames = sample_frames(
            video_path,
            duration,
            has_nvenc=detect_nvenc(),
            source_range=source_range,
            **sample_kwargs,
        )
        if not frames:
            logger.info("  Color: No frames sampled — skipping correction")
            return ("null", "none", [], None)

        segments, is_adaptive = analyze_segments(frames)
        if not segments:
            return ("null", "none", [], None)

        frame_width = info.get("width")
        if not frame_width:
            logger.info("  Color: probe lacks a source width — sharpen radius assumes 1920")
            frame_width = 1920
        grade_kwargs = {
            "frame_width": int(frame_width),
            "no_sharpen": no_sharpen,
            "no_denoise": no_denoise,
        }

        span_durations: Optional[list[float]] = None
        if is_adaptive:
            span_stats, span_durations = merge_adjacent_segments(segments, total_duration=duration)
            corrections = [compute_correction(s, **grade_kwargs) for s in span_stats]
            logger.info(f"  Color: merged {len(segments)} segments → {len(span_stats)} span(s)")
        else:
            # Aggregate all segments via shared helper (same logic as color_analysis)
            global_stats = aggregate_segment_stats(
                [
                    FrameStats(
                        black_point=s.black_point,
                        white_point=s.white_point,
                        mid_tone=s.mid_tone,
                        saturation=s.saturation,
                        r_mean=s.r_mean,
                        g_mean=s.g_mean,
                        b_mean=s.b_mean,
                        detail_score=s.detail_score,
                        noise_score=s.noise_score,
                    )
                    for s in segments
                ]
            )
            corrections = [compute_correction(global_stats, **grade_kwargs)]

        filter_str, method, lut_dir = self._corrections_to_filter(corrections, span_durations)

        # Cache result for next run (v3 schema — merged corrections + spans)
        save_cache_json(
            cache_path,
            {
                "version": TRANSIENT_COLOR_CACHE_SCHEMA_VERSION,
                "corrections": [c.__dict__ for c in corrections],
                "method": method,
                "span_durations": span_durations,
                "tonemap": tonemap_filter,
            },
        )

        logger.info(f"  Color: {method}, {len(corrections)} correction(s)")
        return (filter_str, method, corrections, lut_dir)

    # ── Step 2: Audio Measurement (no video encode) ────────────────────

    def _measure_audio(self, video_path: Path, duration: float, height: int = 1080) -> dict:
        """
        Run loudnorm measurement pass (audio-only decode).

        Returns:
            Dict with measured stats, or empty dict on failure.
        """
        cmd = build_loudnorm_measure_cmd(video_path)

        timeout = self._calc_audio_measurement_timeout(duration, height)
        logger.info(f"  Audio measurement pass (timeout={timeout}s)...")

        try:
            result = run_ffmpeg_quick(cmd, timeout=timeout, description="audio measurement")
        except RuntimeError as e:
            logger.warning(f"  Audio measurement failed: {e}")
            return {}
        stderr = result.stderr or ""

        try:
            measured = parse_loudnorm_stats(stderr)
        except json.JSONDecodeError as e:
            logger.warning(
                "⚠ single_pass: two-pass loudnorm measurement attempted; "
                "stats JSON failed to parse (%s); "
                "falling back to single-pass loudnorm (output loudness may be suboptimal)",
                e,
            )
            return {}
        if measured is None:
            logger.warning(
                "⚠ single_pass: two-pass loudnorm measurement attempted; "
                "no JSON stats block found in ffmpeg stderr; "
                "falling back to single-pass loudnorm (output loudness may be suboptimal)"
            )
            return {}
        return measured

    # ── Step 3: Build Combined Filter Chain ────────────────────────────

    @staticmethod
    def _build_fade_alpha(start: float, end: float, fade_duration: float) -> str:
        """Build an ffmpeg alpha expression that fades in/out between start and end.

        The expression evaluates to:
            0              when t < start
            ramp 0→1       when start <= t < start + fade_duration
            1              when start + fade_duration <= t < end - fade_duration
            ramp 1→0       when end - fade_duration <= t < end
            0              when t >= end
        """
        return (
            f"if(lt(t\\,{start})\\,0\\,"
            f"if(lt(t\\,{start + fade_duration})\\,"
            f"(t-{start})/{fade_duration}\\,"
            f"if(lt(t\\,{end - fade_duration})\\,1\\,"
            f"if(lt(t\\,{end})\\,"
            f"({end}-t)/{fade_duration}\\,0))))"
        )

    def _build_location_overlay(
        self,
        location: str,
        font_path: str,
        location_duration: float,
        fade_duration: float,
        brand_fontsize: int,
        loc_fontsize: int,
        video_width: int,
    ) -> list[str]:
        """Build drawtext filter fragments for the location overlay.

        Returns list of drawtext filter strings for the brand line and
        wrapped location lines.
        """
        filters = []
        loc_start = 1.0
        loc_end = loc_start + location_duration

        loc_alpha = self._build_fade_alpha(loc_start, loc_end, fade_duration)

        brand_esc = self._escape_text(getattr(self.profile, "brand_text", None) or BRAND_TEXT)
        filters.append(
            f"drawtext="
            f"fontfile='{font_path}':"
            f"text='{brand_esc}':"
            f"fontsize={brand_fontsize}:"
            f"fontcolor=white@1.0:"
            f"borderw=3:"
            f"bordercolor=black@0.8:"
            f"x=(w-text_w)/2:"
            f"y=h*0.68:"
            f"alpha='{loc_alpha}':"
            f"enable='between(t,{loc_start},{loc_end})'"
        )

        cleaned = _clean_location(location)
        max_width_px = max(video_width - 200, 800)
        loc_lines = self._wrap_text(
            cleaned,
            loc_fontsize,
            max_width_px,
            max_lines=2,
            font_path=font_path,
        )

        # Stack wrapped lines by fontsize×1.25 px (parity with ost_applier's
        # _LOCATION_LINE_SPACING) rather than 0.03·h. 0.03·h is only ~32px at
        # 1080p / ~65px at 4K against a 48/72px font, so two-line locations
        # visually overlapped; a px offset scales with the glyphs.
        line_spacing_px = int(loc_fontsize * 1.25)
        for i, line in enumerate(loc_lines):
            loc_esc = self._escape_text(line)
            y_expr = "h*0.76" if i == 0 else f"h*0.76+{i * line_spacing_px}"
            filters.append(
                f"drawtext="
                f"fontfile='{font_path}':"
                f"text='{loc_esc}':"
                f"fontsize={loc_fontsize}:"
                f"fontcolor=white@1.0:"
                f"borderw=3:"
                f"bordercolor=black@0.8:"
                f"x=(w-text_w)/2:"
                f"y={y_expr}:"
                f"alpha='{loc_alpha}':"
                f"enable='between(t,{loc_start},{loc_end})'"
            )

        return filters

    def _build_website_overlay(
        self,
        font_path: str,
        duration: float,
        website_duration: float,
        fade_duration: float,
        website_fontsize: int,
    ) -> str:
        """Build drawtext filter fragment for the website CTA overlay."""
        website_text = getattr(self.profile, "website_text", None) or WEBSITE_TEXT
        ws_end = duration - 0.5
        ws_start = ws_end - website_duration
        website_esc = self._escape_text(website_text)

        ws_alpha = self._build_fade_alpha(ws_start, ws_end, fade_duration)

        return (
            f"drawtext="
            f"fontfile='{font_path}':"
            f"text='{website_esc}':"
            f"fontsize={website_fontsize}:"
            f"fontcolor=white@1.0:"
            f"borderw=3:"
            f"bordercolor=black@0.8:"
            f"x=(w-text_w)/2:"
            f"y=h*0.75:"
            f"alpha='{ws_alpha}':"
            f"enable='between(t,{ws_start},{ws_end})'"
        )

    def _build_video_filter(
        self,
        color_filter: str,
        location: str,
        font_path: str,
        duration: float,
        fade_video: bool = True,
        video_fade_in: float = 1.0,
        video_fade_out: float = 1.0,
        location_duration: float = 8.0,
        subscribe_duration: float = 6.0,
        fade_duration: float = 1.0,
        video_width: int = 3840,
        skip_ost: bool = False,
        source_range: str = "full",
        tonemap_filter: str = "",
        upscale_filter: str = "",
        output_fps: str = OUTPUT_FPS,
    ) -> str:
        """
        Build the combined video filter chain:
            [color_corrections],[range],[video_fades],[text]

        Filter order matters:
            1. Color correction first (modifies pixel values).
            2. Video fades (modifies brightness globally at start/end)
            3. Text overlays last (drawn on top of corrected+faded frames)
        """
        filters = []

        # HDR source: tonemap to BT.709 ahead of everything.
        if tonemap_filter:
            filters.append(tonemap_filter)
        # A declared upscale runs ahead of the grade (after any HDR tonemap), so the grade, fades and
        #    text all run at the output size. Absent a plan the chain is unchanged.
        if upscale_filter:
            filters.append(upscale_filter)

        # 0. Tag the PROBED range before anything converts colour space: the
        #    YUV→RGB conversion ffmpeg auto-inserts ahead of curves/colorbalance
        #    assumes LIMITED on untagged input, which mis-grades full-range
        #    GoPro footage and clips its true shadows/highlights.
        #    Metadata only — it moves no pixel by itself.
        filters.append(f"setrange={source_range}")

        # 1. Color correction filters (full CPU grade), preceded by the
        #    high-precision working format so the grade is not computed at
        #    source depth. Skipped when there is no grade — a 16-bit RGB
        #    working copy of a 4K frame is not free, and fades/text do not
        #    need it.
        if color_filter and color_filter != "null":
            filters.append(f"format={GRADE_WORKING_FORMAT}")
            filters.append(color_filter)

        # 2. Range conversion → limited (16-235). YouTube assumes limited range
        #    for SDR BT.709 uploads. ``in_range`` follows the PROBED source
        #    range: a full-range GoPro source is squeezed
        #    once; an already-limited source becomes a no-op instead of being
        #    double-compressed to washed blacks / dull whites.
        filters.append(f"scale=in_range={source_range}:out_range=limited")

        # 3. Video fade in/out
        if fade_video:
            filters.append(f"fade=t=in:st=0:d={video_fade_in}:color=black")
            fade_out_start = duration - video_fade_out
            filters.append(f"fade=t=out:st={fade_out_start}:d={video_fade_out}:color=black")

        # Resolution-aware font scaling
        font_scale = max(1.0, min(video_width / 1920, 1.5))
        brand_fontsize = int(64 * font_scale)
        loc_fontsize = int(48 * font_scale)
        subscribe_fontsize = int(48 * font_scale)

        # 4. Location text overlay (first N seconds)
        if location and font_path:
            filters.extend(
                self._build_location_overlay(
                    location,
                    font_path,
                    location_duration,
                    fade_duration,
                    brand_fontsize,
                    loc_fontsize,
                    video_width,
                )
            )

        # 5. Website CTA (last N seconds)
        if font_path and not skip_ost:
            filters.append(
                self._build_website_overlay(
                    font_path,
                    duration,
                    subscribe_duration,
                    fade_duration,
                    subscribe_fontsize,
                )
            )

        # 6. Explicit CFR conversion, last. See color_filters.cfr_filter for why
        #    this is a filter rather than the output-side -r, and why it is last.
        filters.append(cfr_filter(output_fps))

        return ",".join(filters)

    def _build_blur_filter(self, video_width: int) -> str:
        """Boxblur radius to anonymize faces/plates; ``config.drive.blur_strength``
        is the 1080p reference, scaled by width (power 2 ≈ Gaussian)."""
        radius = max(2, round(self.config.drive.blur_strength * video_width / 1920))
        return f"boxblur={radius}:2"

    @staticmethod
    def _reduce_blur_branch(blur_filter: str, video_width: int, video_height: int) -> str:
        """Run a ``boxblur=R:P`` branch at 1/BLUR_BRANCH_DOWNSCALE resolution.

        Anything else (another filter, or a frame size the factor does not divide,
        where the scale-back would not reproduce the mask's size) is returned unchanged.
        """
        d = BLUR_BRANCH_DOWNSCALE
        m = re.fullmatch(r"boxblur=(\d+):(\d+)", blur_filter)
        if not m or video_width % d or video_height % d:
            return blur_filter
        radius = max(1, round(int(m.group(1)) / d))
        return (
            f"scale=iw/{d}:ih/{d}:flags=bilinear,boxblur={radius}:{m.group(2)},"
            f"scale=iw*{d}:ih*{d}:flags=bilinear"
        )

    def _build_blur_filter_complex(
        self,
        color_filter: str,
        location: str,
        font_path: str,
        duration: float,
        blur_filter: str,
        fade_video: bool = True,
        video_fade_in: float = 1.0,
        video_fade_out: float = 1.0,
        location_duration: float = 8.0,
        subscribe_duration: float = 6.0,
        fade_duration: float = 1.0,
        video_width: int = 3840,
        video_height: int = 2160,
        skip_ost: bool = False,
        source_range: str = "full",
        tonemap_filter: str = "",
        upscale_filter: str = "",
        output_fps: str = OUTPUT_FPS,
    ) -> str:
        """Privacy-blur ``filter_complex`` (input 0 = video, 1 = privacy mask).

        Blur lands AFTER color/scale/fade but BEFORE the text overlays so the
        OST/CTA pixels stay sharp. The gray mask drives an ``alphamerge`` +
        ``overlay`` composite, NOT ``maskedmerge``: maskedmerge promotes the
        gray mask to yuv420p with chroma pinned at 128, half-blending chroma
        across the whole frame regardless of the mask (verified via PSNR).
        """
        # Base segment: color → range-limit → fades (mirrors the -vf chain).
        # ``in_range`` follows the probed source range so a limited source is
        # not double-compressed.
        base: list[str] = []
        if tonemap_filter:
            base.append(tonemap_filter)
        # Mirrors _build_video_filter's upscale: first, so the blur composite
        # and the mask below both run at the output size.
        if upscale_filter:
            base.append(upscale_filter)
        # Mirrors _build_video_filter step 0 — same tag, or the two paths grade
        # in different domains.
        base.append(f"setrange={source_range}")
        # Mirrors _build_video_filter step 1 — same working-format pin, or the
        # blur path grades at a different precision than the -vf path.
        if color_filter and color_filter != "null":
            base.append(f"format={GRADE_WORKING_FORMAT}")
            base.append(color_filter)
        base.append(f"scale=in_range={source_range}:out_range=limited")
        if fade_video:
            base.append(f"fade=t=in:st=0:d={video_fade_in}:color=black")
            base.append(f"fade=t=out:st={duration - video_fade_out}:d={video_fade_out}:color=black")

        # Tail = text overlays (never blurred). Wiring mirrors _build_video_filter
        # steps 4-5; the drift guard test asserts the emitted drawtext is
        # byte-identical across both paths.
        font_scale = max(1.0, min(video_width / 1920, 1.5))
        tail: list[str] = []
        if location and font_path:
            tail.extend(
                self._build_location_overlay(
                    location,
                    font_path,
                    location_duration,
                    fade_duration,
                    int(64 * font_scale),
                    int(48 * font_scale),
                    video_width,
                )
            )
        if font_path and not skip_ost:
            tail.append(
                self._build_website_overlay(
                    font_path, duration, subscribe_duration, fade_duration, int(48 * font_scale)
                )
            )

        blur_filter = self._reduce_blur_branch(blur_filter, video_width, video_height)

        # Pin the base BEFORE the split so alphamerge's 8-bit-only requirement
        # cannot negotiate backwards into it — see the BLUR_*_FORMAT comment.
        base.append(f"format={BLUR_BASE_FORMAT}")
        segs = [
            f"[0:v]{','.join(base)}[base]",
            "[base]split[b1][b2]",
            f"[b2]{blur_filter}[bl]",
            f"[1:v]{upscale_filter + ',' if upscale_filter else ''}format=gray[m]",
            f"[bl][m]alphamerge,format={BLUR_ALPHA_FORMAT}[bla]",
        ]
        # CFR conversion rides the tail, strictly AFTER the composite — the
        # mask pairs 1:1 with SOURCE frames (see color_filters.cfr_filter).
        tail.append(cfr_filter(output_fps))
        overlay = f"overlay=format={BLUR_OVERLAY_FORMAT}"
        if tail:
            segs.append(f"[b1][bla]{overlay}[ov]")
            segs.append(f"[ov]{','.join(tail)}[vout]")
        else:
            segs.append(f"[b1][bla]{overlay}[vout]")
        return ";".join(segs)

    def _build_audio_filter(self, measured: dict) -> str:
        """Build the ATTENUATE-ONLY day-master audio filter.

        Standing rule: every
        published video must be practically silent — conversation and songs
        must not be interpretable at maximum playback volume. The day-first
        architecture runs ONE audio pass over the concatenated day master and
        stream-copies regular LFs out of it, so this single filter governs
        every LF's loudness. It must therefore never AMPLIFY: the prior
        two-pass ``loudnorm=I=-20`` boosted quiet cab audio *up* to target,
        which leaves audible speech in the output.

        ``build_attenuate_filter`` computes a single non-positive ``volume=``
        gain that pulls the measured source loudness DOWN to the near-silent
        integrated ceiling (``AUDIO_SILENCE_CEILING_LUFS``) and honours the
        true-peak cap. Positive gain is impossible by construction (asserted
        in the helper); the defensive invariants below trip if a future
        refactor reintroduces a loudnorm boost.
        """
        af = build_attenuate_filter(measured)
        # Defensive invariant: the day-master + stream-copy LF architecture
        # must never receive an amplifying filter. Attenuate-only means a bare
        # ``volume=<=0 dB`` gain — a loudnorm normalization (which boosts quiet
        # content up toward a target) is forbidden. Re-introducing one must
        # trip this at filter-build time.
        if "loudnorm" in af:
            raise RuntimeError(
                "day-master audio filter must be attenuate-only (no loudnorm "
                f"boost) — got: {af!r}. See build_attenuate_filter and the "
                "near-silent policy in encoding_constants.py."
            )
        if not af.startswith("volume="):
            raise RuntimeError(
                f"attenuate-only audio filter must be a 'volume=' gain — got: {af[:80]!r}"
            )
        return af

    def _get_encoding_args(
        self,
        width: int,
        height: int,
        fps: float,
        use_nvenc: bool,
        source_bitrate_kbps: int = 0,
        output_fps: str = OUTPUT_FPS,
    ) -> list[str]:
        """
        Build encoding args from frozen constants (no config overrides).
        """
        return get_encoding_args(
            width,
            height,
            fps,
            has_nvenc=use_nvenc,
            source_bitrate_kbps=source_bitrate_kbps,
            output_fps=output_fps,
        )

    # ── Main Process Method ────────────────────────────────────────────

    def _run_color_step(
        self,
        video_path: Path,
        skip_color: bool,
        lut_path: Optional[Path],
        tonemap_filter: str = "",
    ) -> tuple[str, str, Optional[Path]]:
        """Run color analysis and return (color_filter, color_method, temp_lut_path).

        Checks for LUT override first, then runs histogram-based analysis.

        ``temp_lut_path`` is non-None only when a directory of per-span LUTs
        (CPU adaptive path) was created; the caller is responsible for
        cleaning it up.

        Returns ("null", "none", None) when color is skipped.
        """
        self._grade_fallback = None
        if skip_color:
            logger.info("  Step 1/3: Color — SKIPPED")
            return ("null", "none", None)

        logger.info("  Step 1/3: Analyzing color...")
        t0 = time.time()

        if lut_path and Path(lut_path).exists():
            color_filter = f"lut3d='{lut_path}'"
            color_method = "lut3d"
            logger.info(f"  Color analysis: {color_method} ({time.time() - t0:.1f}s)")
            return (color_filter, color_method, None)

        color_filter, color_method, _corrections, temp_lut = self._analyze_color(
            video_path, tonemap_filter=tonemap_filter
        )

        logger.info(f"  Color analysis: {color_method} ({time.time() - t0:.1f}s)")
        return (color_filter, color_method, temp_lut)

    def _run_audio_step(
        self,
        video_path: Path,
        duration: float,
        height: int,
        skip_audio: bool,
    ) -> dict:
        """Run audio loudness measurement and return measured stats.

        Returns empty dict when audio is skipped or measurement fails.
        """
        if skip_audio:
            logger.info("  Step 2/3: Audio — SKIPPED")
            return {}

        logger.info("  Step 2/3: Measuring audio loudness...")
        t0 = time.time()
        measured = self._measure_audio(video_path, duration, height)

        if measured:
            logger.info(
                f"  Audio: {measured.get('input_i', '?')} LUFS → attenuated to "
                f"≤{ec.AUDIO_SILENCE_CEILING_LUFS:.0f} LUFS near-silent ceiling "
                f"({time.time() - t0:.1f}s)"
            )
        else:
            logger.warning(
                "⚠ single_pass: audio two-pass loudness measurement attempted; "
                "measurement returned no usable stats (%.1fs); "
                "falling back to single-pass loudnorm — output loudness may be suboptimal",
                time.time() - t0,
            )

        return measured

    def _run_encode_step(
        self,
        video_path: Path,
        output_path: Path,
        vf: str,
        measured: dict,
        skip_audio: bool,
        skip_ost: bool,
        use_nvenc: bool,
        info: dict,
        duration: float,
        font: Optional[str],
        filter_complex: Optional[str] = None,
        blur_mask_path: Optional[Path] = None,
        output_fps: str = OUTPUT_FPS,
    ) -> SinglePassResult:
        """Assemble FFmpeg arguments and run the single-pass encode.

        Handles encoding arg assembly, subprocess execution, output
        validation, and quality gate enforcement. Returns SinglePassResult.
        """
        width = info["width"]
        height = info["height"]
        fps = info["fps"]

        # Build encoding arguments with source-relative bitrate floor
        source_bitrate_kbps = info.get("bitrate_kbps", 0)
        logger.info(
            f"  Encoding: source_bitrate={source_bitrate_kbps}kbps, "
            f"floor_active={source_bitrate_kbps > 0}"
        )
        if source_bitrate_kbps > 0:
            target = max(int(source_bitrate_kbps * 1.0), 50000)
            logger.info(f"  VBR floor: {target}kbps (-b:v {target}k, CQ 1 quality retained)")
        enc_args = self._get_encoding_args(
            width,
            height,
            fps,
            use_nvenc,
            source_bitrate_kbps=source_bitrate_kbps,
            output_fps=output_fps,
        )

        # Remove -c:a copy and -movflags from enc_args — we set our own
        enc_args_video_only = []
        skip_next = False
        for arg in enc_args:
            if skip_next:
                skip_next = False
                continue
            if arg == "-c:a":
                skip_next = True
                continue
            if arg == "-movflags":
                skip_next = True
                continue
            enc_args_video_only.append(arg)

        cmd = ["ffmpeg", "-y"]

        # NVDEC decode path. Two cases enable NVDEC:
        #   1. No CPU filters at all (legacy fast-path): just `-hwaccel cuda`.
        #   2. CPU filters present AND ``config.drive.use_nvdec_decode=True``:
        #      `-hwaccel cuda -hwaccel_output_format cuda` so frames decode
        #      on GPU, then `hwdownload,format=<nv12|p010le>` at the head of
        #      the filter chain so CPU filters see normal frames. The
        #      download format is chosen from the source's probed `pix_fmt`
        #      bit depth (`_hwdownload_format_for_pix_fmt`): 10-bit Main 10
        #      GoPro sources → `p010le`, 8-bit → `nv12`. The old
        #      `nv12|p010le` alternation aborted every 10-bit day master
        #      (`[hwdownload] Invalid output format nv12 …` → nvenc -22).
        #      NVDEC decode is markedly faster on 4K HEVC sources than full-CPU decode;
        #      see ``config.drive.use_nvdec_decode`` docstring.
        # ``graph`` is whichever filtergraph drives this encode — the blur
        # ``filter_complex`` (input 1 = mask) when active, else the bare ``vf``.
        # NVDEC / hwdownload decisions key off it identically.
        graph = filter_complex if filter_complex else vf
        has_filters = bool(graph) and graph != "null"
        nvdec_with_filters = (
            use_nvenc and has_filters and getattr(self.config.drive, "use_nvdec_decode", False)
        )
        if use_nvenc and not has_filters:
            cmd.extend(["-hwaccel", "cuda"])
        elif nvdec_with_filters:
            cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])

        cmd.extend(["-i", str(video_path)])

        # Add video filters. When NVDEC fed the input, prepend an hwdownload
        # whose format matches the source bit depth so CPU filters see normal
        # (nv12 / p010le) frames. A bit-depth mismatch fails the download
        # (10-bit surface → nv12 = fatal -22).
        if filter_complex:
            # Blur path: the mask is a second input; the composite emits [vout].
            cmd.extend(["-i", str(blur_mask_path)])
            fc = filter_complex
            if nvdec_with_filters:
                dl_fmt = _hwdownload_format_for_pix_fmt(info.get("pix_fmt"))
                fc = fc.replace("[0:v]", f"[0:v]hwdownload,format={dl_fmt},", 1)
            cmd.extend(["-filter_complex", fc])
        elif vf and vf != "null":
            if nvdec_with_filters:
                dl_fmt = _hwdownload_format_for_pix_fmt(info.get("pix_fmt"))
                effective_vf = f"hwdownload,format={dl_fmt},{vf}"
            else:
                effective_vf = vf
            cmd.extend(["-vf", effective_vf])

        # Add audio filter or copy audio
        if not skip_audio:
            af = self._build_audio_filter(measured)
            cmd.extend(
                [
                    "-af",
                    af,
                    *enc_args_video_only,
                    "-c:a",
                    "aac",
                    "-b:a",
                    self.config.output.audio_bitrate,
                ]
            )
        else:
            cmd.extend(
                [
                    *enc_args_video_only,
                    "-an",
                ]
            )

        # filter_complex consumes auto-mapping — name the streams explicitly.
        # The composite output is [vout]; audio comes straight off input 0.
        if filter_complex:
            cmd.extend(["-map", "[vout]"])
            if not skip_audio:
                cmd.extend(["-map", "0:a"])

        cmd.extend(
            [
                "-write_tmcd",
                "0",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )

        timeout = self._calc_encode_timeout(duration, use_nvenc)
        logger.info(f"  Filter chain: {(filter_complex or vf)[:120]}...")
        logger.info(f"  Timeout: {timeout}s ({timeout / 3600:.1f}h)")

        encode_start = time.time()

        encode_result = run_ffmpeg_encode(
            cmd,
            output_path=output_path,
            expected_duration=duration,
            description="single-pass encode (color+OST+audio)",
            stall_timeout=600,
            max_timeout=timeout,
            log_interval=30,
        )

        encode_time = time.time() - encode_start

        if not encode_result.success:
            logger.error(f"  Single-pass encode failed: {encode_result.stderr_tail[-500:]}")
            return SinglePassResult(success=False)

        # Validate output with ffprobe (moov atom integrity check)
        validation = validate_video(
            output_path,
            expected_duration=duration,
            require_audio=not skip_audio,
        )
        if not validation.valid:
            logger.error(f"  Output validation failed: {validation.reason}")
            if output_path.exists():
                output_path.unlink()
            return SinglePassResult(success=False)

        out_size = validation.file_size
        logger.info(
            f"  Output: {out_size / (1024**3):.1f} GB, "
            f"{validation.video_codec}, {validation.duration:.1f}s"
        )

        # Post-encode quality gate: verify output achieved the bitrate floor.
        # NVENC CQ 18 should produce 40-50 Mbps at 4K; if it's below the
        # absolute floor, a regression (e.g. accidental CBR switch) is likely.
        if duration > 0:
            output_bitrate_kbps = (out_size * 8) // (int(duration) * 1000)
            if height >= 2160:
                abs_floor = ec.LONGFORM_MIN_BITRATE_4K
            elif height >= 1080:
                abs_floor = ec.LONGFORM_MIN_BITRATE_1080P
            else:
                abs_floor = 5000
            if output_bitrate_kbps < abs_floor:
                logger.error(
                    f"  POST-ENCODE QUALITY GATE FAILED: output {output_bitrate_kbps} kbps "
                    f"< floor {abs_floor} kbps for {height}p. "
                    f"NVENC CQ 18 should produce 40-50 Mbps at 4K — "
                    f"check get_encoding_args for CBR regression."
                )
                output_path.unlink(missing_ok=True)
                return SinglePassResult(success=False)
            logger.info(
                f"  Quality gate: {output_bitrate_kbps} kbps >= floor {abs_floor} kbps — PASSED"
            )

        return SinglePassResult(
            success=True,
            output_path=output_path,
            ost_applied=font is not None and not skip_ost,
            audio_normalized=not skip_audio,
            nvenc_used=use_nvenc,
            encoding_time=encode_time,
        )

    def process(
        self,
        video_path: Path,
        output_path: Path,
        location: str,
        video_duration: Optional[float] = None,
        lut_path: Optional[Path] = None,
        skip_color: bool = False,
        skip_ost: bool = False,
        skip_audio: bool = False,
        video_info: Optional[dict] = None,
        blur_mask_path: Optional[Path] = None,
        output_fps: str = OUTPUT_FPS,
    ) -> SinglePassResult:
        """
        Process video with combined color + OST + audio in a single encode.

        Replaces separate Stage 3 (color), Stage 4 (OST), and Stage 5 (audio)
        from the current pipeline with one FFmpeg invocation.

        Args:
            video_path: Input video (post stop-removal)
            output_path: Final output path (_FINAL.mp4)
            location: Location text for OST overlay
            video_duration: Video duration (auto-detected if None)
            lut_path: Optional LUT file (overrides auto color analysis)
            skip_color: Skip color correction (pass through)
            skip_ost: Skip text overlays
            skip_audio: Skip audio normalization
            video_info: Pre-detected video info dict (skips ffprobe if provided)
            blur_mask_path: Frame-accurate privacy mask. When set, faces/
                plates are blurred in-encode (alphamerge composite) before the
                text overlays — the master ships anonymized.

        Returns:
            SinglePassResult with processing details
        """
        total_start = time.time()
        video_path = Path(video_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        info = video_info if video_info is not None else self._get_video_info(video_path)
        duration = video_duration or info["duration"]
        height = info["height"]
        width = info["width"]
        fps = info["fps"]
        check_output_fps(output_fps, fps)

        if duration <= 0:
            return SinglePassResult(success=False)

        # Upstream SDR gate — refuse HDR/non-BT.709
        # sources BEFORE the multi-hour encode, rather than grading them as SDR
        # and falsely tagging bt709 into an irreversible master.
        info_in = info
        info, tonemap_filter, tonemap_op = self._resolve_hdr_source(info)
        if tonemap_op:
            kind = "gamut-mapped" if tonemap_op.startswith("zscale:gamut") else "tonemapped"
            logger.info(f"  Source {kind} to BT.709 ({tonemap_op})")

        # Validate-at-burn gate: the legacy `pipeline` CLI
        # hands the raw --location straight to drawtext. Refuse garbage
        # (None/null, coordinates, pure digits, HOA leakage, placeholders,
        # frame overflow) BEFORE the irreversible encode — checked on the
        # exact cleaned text the overlay burns. skip_ost callers (day master,
        # graded clips, previews without a label) burn nothing and are exempt.
        if not skip_ost and location and location.strip():
            burn_text = _clean_location(location)
            ok, reason = validate_ost(burn_text, "lf")
            if not ok:
                raise ValueError(
                    f"OST-BURN-GATE: refusing single-pass encode — overlay text "
                    f"{burn_text!r} failed validate_ost: {reason}. Fix --location "
                    f"and re-run; nothing was encoded."
                )

        # NVENC with max quality params is the only supported path. __init__ already raised NVENCUnavailableError if
        # the probe failed, so reaching here implies use_nvenc=True by
        # construction.
        use_nvenc = self._has_nvenc()
        preset_label = "p7+multipass"
        logger.info(f"  Video: {width}x{height} @ {fps:.2f}fps, {duration:.1f}s")
        logger.info("  Encoder: hevc_nvenc (GPU)")
        logger.info(f"  Preset: {preset_label}")

        # ── Step 1: Color analysis ─────────────────────────────────────
        color_filter, color_method, temp_lut = self._run_color_step(
            video_path,
            skip_color,
            lut_path,
            tonemap_filter=tonemap_filter,
        )

        try:
            # ── Step 2: Audio measurement ──────────────────────────────────
            measured = self._run_audio_step(video_path, duration, height, skip_audio)

            # ── Step 3: Combined single-pass encode ────────────────────────
            logger.info("  Step 3/3: Single-pass encode (color + OST + audio)...")
            font = self._find_font() if not skip_ost else None
            if not skip_ost and not font:
                logger.warning("  No font found — OST text will be skipped")

            # A source below the drive long-form rendition is resampled up
            # (profile.upscale); "off" refuses here, before the multi-hour
            # encode. Only a tenant-profiled encode (the day master) is
            # held to the rendition; the generic path keeps the source size.
            plan = None
            if self.profile is not None:
                plan = upscale_plan(
                    width,
                    height,
                    rendition_for("drive_longform_4k"),
                    getattr(self.profile, "upscale", "off"),
                )
            if plan is not None:
                logger.info(
                    "  Upscale: %sx%s -> %sx%s (%s)", *plan.source, *plan.target, plan.scaler
                )

            # Build combined video filter. The blur filter_complex reuses the
            # exact same chain inputs, so the two paths never drift.
            filter_kwargs = dict(
                color_filter=color_filter if not skip_color else "null",
                location=location if font and not skip_ost else "",
                font_path=font or "",
                duration=duration,
                fade_video=not skip_ost,
                video_width=plan.target[0] if plan else width,
                skip_ost=skip_ost,
                source_range=_scale_range_token(info.get("color_range", "")),
                tonemap_filter=tonemap_filter,
                output_fps=output_fps,
            )
            if plan is not None:
                filter_kwargs["upscale_filter"] = plan.filter
            vf = self._build_video_filter(**filter_kwargs)

            filter_complex = None
            if blur_mask_path is not None:
                filter_complex = self._build_blur_filter_complex(
                    **filter_kwargs,
                    video_height=plan.target[1] if plan else height,
                    blur_filter=self._build_blur_filter(filter_kwargs["video_width"]),
                )

            result = self._run_encode_step(
                video_path,
                output_path,
                vf,
                measured,
                skip_audio,
                skip_ost,
                use_nvenc,
                info,
                duration,
                font,
                filter_complex=filter_complex,
                blur_mask_path=blur_mask_path,
                output_fps=output_fps,
            )

            if not result.success:
                return result
        finally:
            # Clean up the per-span adaptive LUT dir (the only temp LUT
            # artifact the color step now produces).
            if temp_lut is not None and temp_lut.is_dir():
                shutil.rmtree(temp_lut, ignore_errors=True)

        total_time = time.time() - total_start
        encode_time = result.encoding_time
        speed = duration / encode_time if encode_time > 0 else 0

        logger.info(
            f"  Encode: {encode_time:.0f}s ({encode_time / 60:.1f}min), {speed:.2f}x realtime"
        )
        logger.info(f"  Total: {total_time:.0f}s ({total_time / 60:.1f}min)")

        result.color_method = color_method
        result.tonemap = tonemap_op
        result.grade_fallback = self._grade_fallback
        if tonemap_op or self._grade_fallback:
            _record_tonemap(output_path, tonemap_op, info_in, self._grade_fallback)
        else:
            _clear_tonemap_record(output_path)
        result.total_time = total_time
        result.upscale_plan = plan
        return result
