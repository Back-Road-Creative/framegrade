"""Color filters — ffmpeg filter strings, encoding args, video encoding.

Split from color.py.
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

from .color_correction import (
    CorrectionParams,
    _average_params,
)
from .ffmpeg_runner import run_ffmpeg_encode

logger = logging.getLogger(__name__)

# Frozen publish cadence: NTSC 29.97. Driving content always ships at this rate
# regardless of source fps, so 59.94/60 fps sources are decimated at encode time.
OUTPUT_FPS = "30000/1001"

# Opt-in source-cadence output: a caller may name one of these instead of the
# frozen default, but only when it equals the probed source rate (no upsampling).
SUPPORTED_OUTPUT_FPS = ("30000/1001", "60000/1001", "30", "60")


def _fps_value(rate: str) -> float:
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1)


def _check_supported(rate: str) -> str:
    if rate not in SUPPORTED_OUTPUT_FPS:
        raise ValueError(
            f"unsupported output rate {rate!r}; supported: {', '.join(SUPPORTED_OUTPUT_FPS)}"
        )
    return rate


def source_output_fps(source_fps: float) -> str:
    """The supported output-rate string equal to a probed source rate, else refuse."""
    for rate in SUPPORTED_OUTPUT_FPS:
        if abs(_fps_value(rate) - source_fps) < 0.01:
            return rate
    raise ValueError(
        f"source rate {source_fps:.3f} fps has no supported output rate "
        f"({', '.join(SUPPORTED_OUTPUT_FPS)})"
    )


def resolve_output_fps(output_fps: str, source_fps: float) -> str:
    """Turn the ``"source"`` cadence request into the probed source's supported rate.

    Anything else passes through untouched, so the default path never consults the source.
    """
    return source_output_fps(source_fps) if output_fps == "source" else output_fps


def cadence_kwargs(config) -> dict:
    """``{"output_fps": "source"}`` when ``--output-cadence source`` is on, else ``{}``.

    Empty by default so every encode call site receives exactly what it did before.
    """
    drive = getattr(config, "drive", None)
    return {"output_fps": "source"} if getattr(drive, "output_cadence", "fixed") == "source" else {}


def probe_source_fps(path: Path) -> float:
    """The first video stream's frame rate, from ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=avg_frame_rate", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    num, _, den = out.partition("/")
    return float(num) / float(den or 1)


def source_cadence_conflicts(drive) -> list[str]:
    """Options that stream-copy-join pieces re-encoded at the fixed rate.

    The insert splice and the trail-film opening re-encode a span at 29.97 and
    join it to the master with a stream copy, which would leave a 59.94 file at
    two cadences. ``trail`` is tri-state (``None`` = auto), so only an explicit
    ``--no-trail`` clears it.
    """
    found = []
    if getattr(drive, "insert_photos", "") or getattr(drive, "insert_clips", []):
        found.append("--insert-photos/--insert-clip")
    if getattr(drive, "trail", None) is not False:
        found.append("the trail film (pass --no-trail)")
    return found


def check_output_fps(output_fps: str, source_fps: float) -> str:
    """Validate an output rate: the default, or exactly the probed source rate."""
    _check_supported(output_fps)
    if output_fps != OUTPUT_FPS and abs(_fps_value(output_fps) - source_fps) >= 0.01:
        raise ValueError(
            f"output rate {output_fps} does not match the source rate {source_fps:.3f} fps; "
            "source-cadence output never changes the rate"
        )
    return output_fps


def cfr_filter(fps: str = OUTPUT_FPS) -> str:
    """The explicit constant-frame-rate conversion for the video chain.

    **Frames are DROPPED or DUPLICATED — never blended.** For an exact 59.94
    source that is a clean 2:1 decimation and the correct cadence; the policy
    is stated here so nobody "improves" it into a blend, which would smear
    motion on every pan.

    This must be a FILTER, not the output-side ``-r``.
    Left to ``-r`` the muxer picks frames itself, and on this build it
    over-produces: a 10.000 s / 600-frame 60 fps source came out **301 frames /
    10.043 s** instead of 300 / 10.010 s, selecting different frames than the
    filter (PSNR ~26 dB apart). ``-fps_mode cfr`` / ``-vsync cfr`` do not fix
    it. ``get_encoding_args`` still stamps ``-r``, which is a no-op once the
    stream already carries this rate.

    Placement is the chain TAIL. It must not run before the privacy-blur
    composite: the mask is materialised one frame per SOURCE frame and
    ``alphamerge`` pairs them 1:1, so decimating first would desync the mask.
    """
    return f"fps={_check_supported(fps)}"


# ── FFmpeg Filter Generation ─────────────────────────────────────────


def _curves_point(level_in: float, level_out: float) -> str:
    """Format a normalized curves control point (0-1 range)."""
    return f"{level_in:.4f}/{level_out:.4f}"


def _params_all_equal(params_list: list[CorrectionParams], *, eps: float = 1e-6) -> bool:
    """True iff all params are numerically equal across every field.

    Identity-collapse guard: when this is True the adaptive filtergraph
    builder is bit-equivalent to the legacy averaged single-filter form,
    so we short-circuit to avoid an unnecessary xfade chain.
    """
    if len(params_list) <= 1:
        return True
    p0 = params_list[0]
    fields = (
        "shadow_lift",
        "highlight_pull",
        "saturation_mult",
        "contrast_curve",
        "white_balance_r",
        "white_balance_b",
        "white_balance_r_high",
        "white_balance_b_high",
        "gamma",
        "denoise_strength",
        "sharpen_amount",
    )
    for p in params_list[1:]:
        for f in fields:
            if abs(getattr(p, f) - getattr(p0, f)) > eps:
                return False
    return True


def build_filter_chain(
    params_list: list[CorrectionParams],
    segment_duration: float = 0.0,
    *,
    lut_paths: Optional[list[Path]] = None,
    span_durations: Optional[list[float]] = None,
    branch_filter: str = "lut3d",
    xfade_duration: float = 1.0,
) -> str:
    """
    Build ffmpeg filter string from correction parameters.

    For single params (global): simple curves + eq filter.
    For multiple params with identical values: collapses to the single-filter
    form (byte-identical to the legacy averaged path — no behavior change).
    For multiple distinct params WITH ``lut_paths``: per-span LUT3D
    filtergraph stitched via 1.0s xfade transitions (measured ≤ 1.0 luma
    delta at boundaries — at/below JND).
    Spatial filters (denoise/sharpen) cannot live in a LUT, so they are
    appended once after the stitch from the averaged params.
    For multiple distinct params WITHOUT ``lut_paths``: falls back to the
    legacy averaged form so call sites that have not opted into LUT baking
    keep their current behavior.

    Args:
        params_list: One CorrectionParams for global, N for adaptive.
        segment_duration: Uniform span duration in seconds (ignored when
            ``span_durations`` is given).
        lut_paths: Optional N per-span .cube LUT paths. When provided
            and params differ, the adaptive filtergraph is emitted.
        span_durations: Optional N per-span durations in seconds, for
            variable-length spans from ``merge_adjacent_segments``.
        branch_filter: Per-branch filter name (``lut3d`` for the CPU
            adaptive path).
        xfade_duration: Transition duration in seconds (default 1.0s,
            audit-validated as at/below JND).

    Returns:
        ffmpeg -vf compatible filter string.
    """
    if not params_list:
        return "null"  # Identity filter

    lut_path = lut_paths[0] if lut_paths and len(lut_paths) == 1 else None

    if len(params_list) == 1:
        return _single_span_filter(params_list[0], lut_path, branch_filter)

    # Identity collapse: when params are numerically equal, the adaptive
    # filtergraph and the legacy averaged single-filter are equivalent.
    # Prefer the simpler form so the call site sees no behavior change.
    if _params_all_equal(params_list):
        return _single_span_filter(_average_params(params_list), lut_path, branch_filter)

    # Per-span adaptive path: requires baked LUTs (one per span).
    # Without them, preserve the legacy averaged behavior so callers that
    # have not opted in keep working.
    spans = span_durations or [float(segment_duration)] * len(params_list)
    if (
        lut_paths
        and len(lut_paths) == len(params_list)
        and len(spans) == len(params_list)
        and all(d > 0 for d in spans)
    ):
        tail_parts = _spatial_tail(_average_params(params_list))
        return _build_adaptive_filtergraph(
            lut_paths,
            span_durations=spans,
            branch_filter=branch_filter,
            xfade_duration=xfade_duration,
            spatial_tail=",".join(tail_parts) if tail_parts else None,
        )

    return _build_single_filter(_average_params(params_list))


def _build_adaptive_filtergraph(
    lut_paths: list[Path],
    *,
    span_durations: Optional[list[float]] = None,
    segment_duration: float = 0.0,
    branch_filter: str = "lut3d",
    xfade_duration: float = 1.0,
    spatial_tail: Optional[str] = None,
) -> str:
    """Build an N-branch per-span LUT3D filtergraph stitched via xfade.

    Produces (for spans [30, 30, 30], xfade=1.0)::

        split=3[s0][s1][s2];
        [s0]trim=end=30.5,setpts=PTS-STARTPTS,lut3d=<lut_0>[c0];
        [s1]trim=start=29.5:end=60.5,setpts=PTS-STARTPTS,lut3d=<lut_1>[c1];
        [s2]trim=start=59.5,setpts=PTS-STARTPTS,lut3d=<lut_2>[c2];
        [c0][c1]xfade=transition=fade:duration=1.000:offset=29.5[x1];
        [x1][c2]xfade=transition=fade:duration=1.000:offset=59.5

    Each branch is TRIMMED to its span (± the fade half-window) with
    timestamps reset before grading. This is load-bearing: xfade restarts
    its second input at the transition offset, so un-trimmed same-source
    branches REWIND the content by the boundary offset at every seam and
    stretch the output (verified: 6s two-span source → 8.5s
    output replaying from t=0 after the fade). With trim+setpts the output
    is frame-aligned with the input timeline and duration-preserving.

    The xfade is centered on each span boundary (offset = boundary - d/2)
    so each span's LUT owns its full nominal interval and only the ±d/2
    window around the seam blends. Spatial tail (hqdn3d, unsharp) rides
    the final chain once, after the stitch, so temporal-state filters do
    not cross span boundaries.

    The graph's input and final output are UNLABELED so it embeds directly
    in a plain ``-vf`` chain and composes with downstream filters via a
    comma (the single-pass builder appends scale/fade/drawtext to it).

    """
    n = len(lut_paths)
    if n < 2:
        raise ValueError(f"_build_adaptive_filtergraph requires N>=2, got {n}")
    spans = span_durations or [float(segment_duration)] * n
    if len(spans) != n or any(d <= 0 for d in spans):
        raise ValueError(f"need {n} positive span durations, got {spans}")

    half_d = xfade_duration / 2.0
    # boundaries[i] = source time where span i ends / span i+1 begins.
    boundaries: list[float] = []
    acc = 0.0
    for d in spans[:-1]:
        acc += d
        boundaries.append(acc)
    # EOF clamp: the final xfade is centered on boundaries[-1], so its window
    # ends at boundaries[-1] + half_d. A partial final segment (clamped to real
    # remaining upstream) leaves sum(spans) == the true media length; without
    # this, the last fade would run past EOF and truncate. Pin the last
    # boundary back so the fade lands inside the footage. Exact-length final
    # spans leave it untouched (min picks the nominal boundary), keeping
    # interior seams byte-identical.
    if boundaries:
        boundaries[-1] = min(boundaries[-1], sum(spans) - half_d)
    if any(b2 - b1 < xfade_duration for b1, b2 in zip(boundaries, boundaries[1:])):
        raise ValueError(f"span shorter than the xfade window: {spans}")

    parts: list[str] = []

    # 1. Split input into N parallel streams
    split_labels = "".join(f"[s{i}]" for i in range(n))
    parts.append(f"split={n}{split_labels}")

    # 2. One trimmed, timestamp-reset, graded branch per span
    for i, lut_path in enumerate(lut_paths):
        trim_args = []
        if i > 0:
            trim_args.append(f"start={boundaries[i - 1] - half_d:g}")
        if i < n - 1:
            trim_args.append(f"end={boundaries[i] + half_d:g}")
        parts.append(
            f"[s{i}]trim={':'.join(trim_args)},setpts=PTS-STARTPTS,{branch_filter}={lut_path}[c{i}]"
        )

    # 3. Stitch via N-1 xfades, each centered on a span boundary. The last
    #    chain's output stays unlabeled (graph output); the optional spatial
    #    tail rides that same chain via a comma.
    prev_label = "c0"
    for i in range(1, n):
        offset = boundaries[i - 1] - half_d
        chain = (
            f"[{prev_label}][c{i}]xfade=transition=fade:"
            f"duration={xfade_duration:.3f}:offset={offset:g}"
        )
        if i < n - 1:
            chain += f"[x{i}]"
            prev_label = f"x{i}"
        elif spatial_tail:
            chain += f",{spatial_tail}"
        parts.append(chain)

    return ";".join(parts)


def _unsharp(params: CorrectionParams) -> str:
    """unsharp fragment; matrix size follows the measured radius (5 at 1.2px, the legacy size)."""
    size = 5
    if params.sharpen_radius > 0:
        size = min(23, max(3, 2 * round(2 * params.sharpen_radius) + 1))
    return f"unsharp={size}:{size}:{params.sharpen_amount:.2f}:{size}:{size}:0"


def _spatial_tail(params: CorrectionParams) -> list[str]:
    """Denoise/sharpen fragments — spatial, so never representable in a LUT."""
    parts: list[str] = []
    if params.denoise_strength > 0:
        s = params.denoise_strength
        parts.append(f"hqdn3d={s:.1f}:{s:.1f}:{s * 0.75:.1f}:{s * 0.75:.1f}")
    if params.sharpen_amount > 0:
        parts.append(_unsharp(params))
    return parts


def _single_span_filter(
    params: CorrectionParams,
    lut_path: Optional[Path],
    branch_filter: str = "lut3d",
) -> str:
    """One uniform grade, preferring a baked LUT over the string chain.

    The string chain ends in ``eq``, which negotiates the stream to 8-bit
    yuv420p mid-chain; ``lut3d`` stays gbrp10le and ``params_to_lut3d_cube``
    mirrors curves → colorbalance → eq exactly.
    ``lut_path=None`` keeps the string chain — the documented fallback.
    """
    chain = _build_single_filter(params)
    # Identity grades emit nothing at all; never bake a pointless LUT pass.
    if lut_path is None or chain == "null":
        return chain
    return ",".join([f"{branch_filter}={lut_path}", *_spatial_tail(params)])


def _build_single_filter(params: CorrectionParams) -> str:
    """Build filter chain: curves -> colorbalance -> eq -> hqdn3d -> unsharp."""
    # If ALL corrections are negligible, return identity
    has_shadow = params.shadow_lift >= 1
    has_highlight = params.highlight_pull >= 1
    has_sat = abs(params.saturation_mult - 1.0) >= 0.01
    has_contrast = params.contrast_curve > 0.001
    has_wb = (
        abs(params.white_balance_r) > 0.001
        or abs(params.white_balance_b) > 0.001
        or abs(params.white_balance_r_high) > 0.001
        or abs(params.white_balance_b_high) > 0.001
    )
    has_gamma = abs(params.gamma - 1.0) > 0.001
    has_denoise = params.denoise_strength > 0
    has_sharpen = params.sharpen_amount > 0

    if not any(
        [
            has_shadow,
            has_highlight,
            has_sat,
            has_contrast,
            has_wb,
            has_gamma,
            has_denoise,
            has_sharpen,
        ]
    ):
        return "null"

    filters = []

    # Build curves filter for shadow lift, highlight pull, and S-curve contrast
    if has_shadow or has_highlight or has_contrast:
        bl_in = 0.0
        bl_out = params.shadow_lift / 255.0

        wh_in = 1.0
        wh_out = 1.0 - (params.highlight_pull / 255.0)

        # Highlight knee. When strong sun glare is
        # detected (highlight_pull >= 5) squeeze the TOP control point further
        # so blown skies stop short of the ceiling:
        #   pull=5  -> ceiling 0.96
        #   pull=15 -> ceiling 0.98
        # The old approach ran a SEPARATE `colorlevels=romax=ceiling` filter,
        # which is a whole-range linear gain (out = in * ceiling) — it darkened
        # shadows and midtones 2-4% on every bright-sky clip. The ceiling now
        # scales ONLY the 1.0 control point; the 0.75 knee anchor keeps the
        # un-squeezed ``wh_out``, so the curve stays identity through the
        # midtones and only the [0.75, 1.0] highlight segment bends down: a
        # real knee. The numpy mirror in color_correction._apply_curves applies
        # the identical fold so the baked GPU LUT stays in parity.
        wh_top = wh_out
        if params.highlight_pull >= 5.0:
            wh_top *= 0.95 + (params.highlight_pull / 500.0)

        # S-curve: adjust midpoint control points for contrast
        c = params.contrast_curve
        points = [
            _curves_point(bl_in, bl_out),
            _curves_point(0.25, 0.25 + bl_out * 0.5 - c),
            _curves_point(0.75, 0.75 - (1.0 - wh_out) * 0.5 + c),
            _curves_point(wh_in, wh_top),
        ]
        # NB: ffmpeg's curves filter requires SPACE-separated points. Colons
        # are silently accepted but produce a degenerate single-point curve
        # that crushes the image to near-uniform gray — see regression test
        # test_curves_points_are_space_separated (caught a silent long-running
        # quality regression).
        curves_str = f"curves=m='{' '.join(points)}'"
        filters.append(curves_str)

    # White balance: colorbalance filter. rs/rm (shadows+midtones) carry the
    # whole-frame cast; rh/bh (highlights) carry a cast concentrated in the
    # bright end. gs/gm/gh stay 0 — green is the reference channel.
    if has_wb:
        filters.append(
            f"colorbalance="
            f"rs={params.white_balance_r:.4f}:gs=0:bs={params.white_balance_b:.4f}:"
            f"rm={params.white_balance_r:.4f}:gm=0:bm={params.white_balance_b:.4f}:"
            f"rh={params.white_balance_r_high:.4f}:gh=0:bh={params.white_balance_b_high:.4f}"
        )

    # Merged eq filter: saturation + gamma in one filter
    eq_parts = []
    if abs(params.saturation_mult - 1.0) >= 0.01:
        eq_parts.append(f"saturation={params.saturation_mult:.3f}")
    if abs(params.gamma - 1.0) > 0.001:
        eq_parts.append(f"gamma={params.gamma:.4f}")
    if eq_parts:
        filters.append(f"eq={':'.join(eq_parts)}")

    # Denoise: hqdn3d (fast temporal + spatial)
    if params.denoise_strength > 0:
        s = params.denoise_strength
        filters.append(f"hqdn3d={s:.1f}:{s:.1f}:{s * 0.75:.1f}:{s * 0.75:.1f}")

    # Sharpen: unsharp (luma only, preserve chroma)
    if params.sharpen_amount > 0:
        filters.append(_unsharp(params))

    return ",".join(filters) if filters else "null"


# ── Encoding Specification ────────────────────────────────────────────


def _nvenc_args(output_fps: str, gop_size: int) -> list[str]:
    """Return NVENC-specific video encoding arguments.

    p7, CQ 18 (YouTube-optimized), multipass, lookahead,
    spatial/temporal AQ, B-frame refs — maximum hardware quality.
    """
    from . import encoding_constants as ec

    return [
        "-r",
        output_fps,
        "-c:v",
        "hevc_nvenc",
        "-preset",
        ec.LONGFORM_NVENC_PRESET,
        "-tune",
        "hq",
        "-rc",
        "vbr",
        "-cq",
        str(ec.LONGFORM_NVENC_CQ),
        "-profile:v",
        ec.LONGFORM_PROFILE,
        "-pix_fmt",
        "p010le",  # 10-bit color
        "-rc-lookahead",
        "32",  # 32-frame lookahead for rate allocation
        "-spatial-aq",
        "1",  # Adaptive QP based on spatial complexity
        "-temporal-aq",
        "1",  # Adaptive QP based on temporal complexity
        "-b_ref_mode",
        "2",  # Use B-frames as references (middle)
        "-bf",
        "4",  # 4 B-frames for better inter-frame prediction
        "-multipass",
        "2",  # Full two-pass (qres+fullres) within NVENC
        "-g",
        str(gop_size),
    ]


def _x265_args(output_fps: str, gop_size: int) -> list[str]:
    """Return x265 (CPU) video encoding arguments.

    slow preset, CRF 18, with thread tuning for multi-core CPUs.
    """
    from . import encoding_constants as ec

    return [
        "-r",
        output_fps,
        "-c:v",
        "libx265",
        "-preset",
        ec.LONGFORM_PRESET,
        "-crf",
        str(ec.LONGFORM_CRF),
        "-profile:v",
        ec.LONGFORM_PROFILE,
        "-pix_fmt",
        ec.LONGFORM_PIX_FMT,
        "-g",
        str(gop_size),
        "-x265-params",
        f"keyint={gop_size}:min-keyint={gop_size}:open-gop=0:pools=16:frame-threads=4:wpp=1",
    ]


def get_encoding_args(
    width: int,
    height: int,
    fps: float,
    has_nvenc: bool = False,
    source_bitrate_kbps: int = 0,
    normalize_audio: bool = False,
    cpu_by_choice: bool = False,
    output_fps: str = OUTPUT_FPS,
) -> list[str]:
    """
    Generate optimal ffmpeg encoding arguments from frozen constants.

    ``output_fps`` defaults to the frozen publish cadence; an explicit
    supported rate (opt-in source cadence) stamps that rate and keeps the
    GOP at 2 s of it. ``fps`` (the source rate) is not consulted.

    ``cpu_by_choice=True`` marks a caller that picks libx265 on purpose
    (not because the GPU is missing), so the "NVENC unavailable" warning —
    which operators read as a GPU fault — is not emitted.

    NVENC (when GPU available): p7, CQ 18 (YouTube-optimized), multipass,
    lookahead, spatial/temporal AQ, B-frame refs — maximum hardware quality.
    x265 fallback (no GPU): slow preset, CRF 18.

    Always adds -b:v/-maxrate/-bufsize as soft floor hints in VBR mode,
    based on fixed resolution-based absolute floors (40 Mbps for 4K,
    15 Mbps for 1080p). The ``source_bitrate_kbps`` parameter is accepted
    for API compatibility but no longer affects the floor calculation.

    All quality values come from encoding_constants — no config overrides.
    """
    from . import encoding_constants as ec

    # All output normalized to 29.97fps regardless of source fps. The real
    # conversion is the explicit ``cfr_filter()`` in the video chain; this arg
    # only stamps the rate. See OUTPUT_FPS.
    _check_supported(output_fps)
    gop_size = round(2 * _fps_value(output_fps))  # 2-second keyframe interval (60 at 29.97)

    # Pick codec-specific args
    if not has_nvenc and not cpu_by_choice:
        logger.warning(
            "⚠ color_filters: NVENC (hevc_nvenc CQ 18) unavailable; falling back to "
            "libx265 slow preset CRF 18 (CPU encode, ~10% quality loss vs NVENC)"
        )
    args = _nvenc_args(output_fps, gop_size) if has_nvenc else _x265_args(output_fps, gop_size)

    # Color metadata: TAG the output BT.709 LIMITED range (16-235), the SDR
    # standard YouTube assumes. These are metadata stamps only — they describe
    # the pixels, they do not convert them. The actual full->limited squeeze is
    # a filter, `scale=in_range=<probed>:out_range=limited` in single_pass,
    # where `in_range` follows the PROBED source range.
    # Do not read this block as "full range is preserved": stamping `tv` here
    # while feeding unsqueezed full-range pixels is exactly the range mismatch
    # that crushes blacks and clips whites.
    args.extend(
        [
            "-color_range",
            ec.LONGFORM_COLOR_RANGE,
            "-colorspace",
            ec.LONGFORM_COLORSPACE,
            "-color_primaries",
            ec.LONGFORM_COLOR_PRIMARIES,
            "-color_trc",
            ec.LONGFORM_COLOR_TRC,
        ]
    )

    # Bitrate floor: add -b:v/-maxrate/-bufsize as soft floor hints in VBR mode.
    #
    # Fixed absolute floors based on resolution — not source-relative. CQ 18
    # adapts to scene complexity; the floor just prevents the encoder from
    # going too low on simple content. Source bitrate is irrelevant to the
    # quality we want to deliver.
    #
    # Do NOT switch to CBR: CBR removes -cq, which with -tune hq causes NVENC
    # to produce a far lower bitrate (~700 kbps) than the 90 Mbps requested.
    if height >= 2160:
        target = ec.LONGFORM_MIN_BITRATE_4K
    elif height >= 1080:
        target = ec.LONGFORM_MIN_BITRATE_1080P
    else:
        target = 5000

    maxrate = int(target * ec.LONGFORM_MAXRATE_RATIO)
    bufsize = int(target * ec.LONGFORM_BUFSIZE_RATIO)

    args.extend(
        [
            "-b:v",
            f"{target}k",
            "-maxrate",
            f"{maxrate}k",
            "-bufsize",
            f"{bufsize}k",
        ]
    )

    if normalize_audio:
        # Per-LF audio encode: replace stream-copy with AAC so the
        # loudnorm filter (added by the caller) can re-normalize each LF
        # independently. The "Stage 5" comment was for the old pipeline;
        # in day-first, the day master is normalized day-wide and LF
        # slices inherit day-average loudness — quiet segments land
        # 2–4 LUFS below target without this per-LF correction.
        args.extend(["-c:a", "aac", "-b:a", ec.LONGFORM_AUDIO_BITRATE])
    else:
        # Stream-copy audio: used for color-filter and LUT passes where
        # audio normalization is handled elsewhere in the pipeline.
        args.extend(["-c:a", "copy"])

    # Container flags
    args.extend(
        [
            "-movflags",
            "+faststart",
        ]
    )

    return args


# ── NVENC Detection ───────────────────────────────────────────────────


class NVENCUnavailableError(RuntimeError):
    """NVENC probe failed — drive pipeline cannot proceed.

    Raised by ``detect_nvenc`` whenever the GPU encoder is not usable.
    No env-var or CLI override re-enables libx265 on this host: CPU
    encode is unrepresentable from the drive run path.

    Recovery procedure (a silent CPU encode can take hours and lose ~10%
    quality): kill the run, restore the GPU encoder (driver / device
    visibility), then resume.
    """


def detect_nvenc() -> bool:
    """Check if NVENC HEVC encoding is available.

    Uses 256x256 test source because NVENC rejects frames below its
    minimum resolution (~144x144 depending on driver).

    Raises ``NVENCUnavailableError`` unconditionally on probe failure
    (no env-var escape hatch). Returns ``True`` on success.
    The legacy boolean return shape is preserved for call sites that
    branch on it, but the False branch is now unreachable from any
    legitimate drive-pipeline invocation — failure paths raise.
    """
    fallback_msg = (
        "NVENC is required for the drive pipeline; CPU fallback is "
        "no longer permitted. Recovery: restore the GPU encoder "
        "(driver / device visibility) and resume."
    )
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=256x256:rate=1",
                "-c:v",
                "hevc_nvenc",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise NVENCUnavailableError(
                f"NVENC probe returned nonzero (returncode {result.returncode}). {fallback_msg}"
            )
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise NVENCUnavailableError(
            f"NVENC probe failed ({type(e).__name__}). {fallback_msg}"
        ) from e


# ── Video Encoding & Processing ──────────────────────────────────────


def encode_video(
    video_path: Path,
    output_path: Path,
    filter_str: str,
    width: int,
    height: int,
    fps: float,
    duration: float,
    has_nvenc: bool = False,
) -> bool:
    """
    Run the ffmpeg encode with color filters and optimal specs.

    Returns True on success.
    """
    enc_args = get_encoding_args(width, height, fps, has_nvenc=has_nvenc)

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    # Apply video filter
    if filter_str and filter_str != "null":
        cmd.extend(["-vf", filter_str])

    cmd.extend(enc_args)
    cmd.append(str(output_path))

    encoder = "hevc_nvenc" if has_nvenc else "libx265"
    if has_nvenc:
        logger.info(f"  Encoding: {encoder}, CQ 18, Main 10 (NVENC hardware path)")
    else:
        logger.warning(
            f"⚠ color_filters: Encoding with {encoder} CRF 18, Main 10 "
            "(CPU fallback — NVENC unavailable, ~10% quality loss)"
        )
    # Read the GOP back out of the args actually being passed to ffmpeg. It is
    # a FIXED 60 (output is normalised to 29.97fps, so 60 frames really is 2s
    # regardless of the source), and deriving the log from `fps` instead made
    # it lie for every non-29.97 source — a 59.94fps source was reported as 120.
    gop_logged = enc_args[enc_args.index("-g") + 1] if "-g" in enc_args else "unknown"
    logger.info(f"  GOP: {gop_logged} frames (2s at the normalized 29.97fps output)")
    logger.info(f"  Filter: {filter_str}")

    # Calculate timeout: ~8x duration for CPU, ~4x for NVENC (4K 60fps needs headroom)
    timeout_mult = 4.0 if has_nvenc else 8.0
    timeout = max(300, int(duration * timeout_mult))
    logger.info(f"  Timeout: {timeout}s ({timeout / 3600:.1f}h)")

    encode_result = run_ffmpeg_encode(
        cmd,
        output_path=output_path,
        expected_duration=duration,
        description=f"color correction encode ({encoder})",
        stall_timeout=600,
        max_timeout=timeout,
    )

    if not encode_result.success:
        raise RuntimeError(f"FFmpeg encode failed: {encode_result.stderr_tail[-500:]}")

    return True
