"""Stabilise / deflicker pass.

Two independent ffmpeg passes, each gated separately because they depend on
build-time filters that not every ffmpeg build has:

  * **Deflicker** — single-pass ffmpeg ``deflicker`` filter.
  * **Stabilise** — two-pass ``vidstabdetect`` / ``vidstabtransform``
    (libvidstab). Absent from many ffmpeg builds
    (check with ``ffmpeg -filters``). Requesting it
    when it's missing raises ``StabilizeUnavailableError`` — this module must
    never silently pass the input through unstabilised, which would look
    identical to a successful no-op stabilise to anything downstream.

Each pass is gated on a measurement and the output only replaces the input when
it measurably improved. Deflicker is judged on per-frame luminance jitter
(``measure_luma_jitter`` / ``gate_on_jitter``). Stabilise is judged on camera
motion (``judge_stabilisation``): shake residual must drop and black borders
stay small, because a translating image keeps its mean luma and luma jitter
cannot see shake. A pass that fails its gate is discarded and the caller is told
why — never a silent pass-through of a filter that made things worse.

``apply_picture_steps`` is the drive-path entry (``--stabilize`` /
``--deflicker`` via ``day_master_encoder.encode_day_master``); see
``docs/pipeline/142-stabilise-and-deflicker.md``.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

logger = logging.getLogger(__name__)

StrPath = Union[str, Path]

# Pixel formats libx265 takes as-is. An input outside this set (hardware
# surface formats, full-range ``yuvj``) is written as the long-form 10-bit
# format instead: a depth never lower than the master's own.
_X265_PIX_FMTS = frozenset(
    {
        "yuv420p",
        "yuv422p",
        "yuv444p",
        "yuv420p10le",
        "yuv422p10le",
        "yuv444p10le",
        "yuv420p12le",
        "yuv422p12le",
        "yuv444p12le",
    }
)
# libx265 in this ffmpeg writes only the matrix and range from ``-colorspace`` /
# ``-color_range``; primaries and transfer reach the bitstream only through
# ``-x265-params``. Values both ffmpeg and x265 spell the same way:
_X265_PRIMARIES = frozenset(
    {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "film", "bt2020", "smpte428"}
    | {"smpte431", "smpte432"}
)
_X265_TRANSFERS = frozenset(
    {"bt709", "bt470m", "bt470bg", "smpte170m", "smpte240m", "linear", "log100", "log316"}
    | {"iec61966-2-4", "bt1361e", "iec61966-2-1", "bt2020-10", "bt2020-12", "smpte2084"}
    | {"smpte428", "arib-std-b67"}
)
_X265_MATRICES = frozenset(
    {"gbr", "bt709", "fcc", "bt470bg", "smpte170m", "smpte240m", "ycgco", "bt2020nc"}
    | {"bt2020c", "smpte2085", "chroma-derived-nc", "chroma-derived-c", "ictcp"}
)
_UNSET_TAGS = frozenset({"", "unknown", "unspecified", "reserved"})

DEFLICKER_FILTER = "deflicker"
VIDSTABDETECT_FILTER = "vidstabdetect"
VIDSTABTRANSFORM_FILTER = "vidstabtransform"

#: Named reason for the vidstab refusal (never a
#: silent pass-through). Kept as a module constant so callers/tests can match
#: on it instead of a free-floating string literal.
VIDSTAB_MISSING_REASON = "ffmpeg built without libvidstab"

_FFMPEG_TIMEOUT_S = 10
_ENCODE_TIMEOUT_S = 600
_MEASURE_TIMEOUT_S = 120


class StabilizeUnavailableError(RuntimeError):
    """Raised when a vidstab pass is requested but this ffmpeg build lacks it.

    Callers that want "stabilise if possible, else skip" must catch this
    explicitly and choose a fallback (e.g. deflicker-only) — the module itself
    never degrades a stabilise request into a no-op.
    """


@dataclass(frozen=True)
class FfmpegFilterCapabilities:
    """The set of ffmpeg filter names this build has registered."""

    available: frozenset

    @property
    def has_deflicker(self) -> bool:
        return DEFLICKER_FILTER in self.available

    @property
    def has_vidstab(self) -> bool:
        return VIDSTABDETECT_FILTER in self.available and VIDSTABTRANSFORM_FILTER in self.available


def probe_ffmpeg_filters(
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    *,
    ffmpeg_bin: str = "ffmpeg",
) -> FfmpegFilterCapabilities:
    """Run ``ffmpeg -filters`` once and parse the registered filter names.

    Mirrors ``cli.handlers.doctor.check_ffmpeg_filters``'s parse (the filter
    name is the second whitespace-delimited token of each row) but returns
    the full capability set rather than a pass/fail against a hand-listed
    subset — this module needs to know WHICH of deflicker/vidstab it has, not
    just whether a fixed list is satisfied.
    """
    result = run(
        [ffmpeg_bin, "-filters"], capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S
    )
    available = set()
    for raw_line in result.stdout.splitlines():
        parts = raw_line.split()
        # Filter list rows look like: " T.. scale   V->V  Scale the input video."
        if len(parts) >= 2:
            available.add(parts[1])
    return FfmpegFilterCapabilities(frozenset(available))


def require_vidstab(capabilities: FfmpegFilterCapabilities) -> None:
    """Refuse with a named reason when this ffmpeg build lacks vidstab.

    Never returns a "degraded but okay" value — the whole point is that a
    stabilise request either gets real stabilisation or a loud refusal, not a
    quiet pass-through.
    """
    if not capabilities.has_vidstab:
        raise StabilizeUnavailableError(VIDSTAB_MISSING_REASON)


@dataclass(frozen=True)
class StreamSpec:
    """Pixel format and colour tags of an input's first video stream."""

    pix_fmt: str = ""
    color_range: str = ""
    color_space: str = ""
    color_primaries: str = ""
    color_transfer: str = ""
    height: int = 0


def nvenc_available() -> bool:
    """True when the hardware HEVC encoder works, the same probe the day master uses.

    ``color_filters.detect_nvenc`` raises on a missing GPU; the picture-step
    intermediate may fall back to x265 instead, so that is read as ``False``.
    """
    from .color_filters import NVENCUnavailableError, detect_nvenc

    try:
        return bool(detect_nvenc())
    except NVENCUnavailableError:
        return False


def probe_stream_spec(
    path: StrPath,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ffprobe_bin: str = "ffprobe",
) -> StreamSpec:
    """ffprobe ``path`` for the pixel format and colour tags the intermediate must keep.

    A failed or unparseable probe returns an empty spec (logged): the argv then
    falls back to the 10-bit long-form format and writes no tags, rather than
    guessing the input's.
    """
    argv = [
        ffprobe_bin,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=pix_fmt,height,color_range,color_space,color_primaries,color_transfer",
        "-of",
        "json",
        str(path),
    ]
    try:
        res = run(argv, capture_output=True, text=True, timeout=_FFMPEG_TIMEOUT_S * 3)
        if res.returncode != 0:
            raise ValueError(f"ffprobe rc={res.returncode}")
        stream = json.loads(res.stdout)["streams"][0]
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        logger.warning("could not probe %s for pixel format / colour tags: %s", path, exc)
        return StreamSpec()
    return StreamSpec(
        pix_fmt=str(stream.get("pix_fmt", "")),
        color_range=str(stream.get("color_range", "")),
        color_space=str(stream.get("color_space", "")),
        color_primaries=str(stream.get("color_primaries", "")),
        color_transfer=str(stream.get("color_transfer", "")),
        height=int(stream.get("height") or 0),
    )


def _nvenc_video_args(spec: StreamSpec) -> list:
    """hevc_nvenc args from the long-form NVENC constants (no ``-r``: keep input timing)."""
    from . import encoding_constants as ec

    eight_bit = spec.pix_fmt == "yuv420p"  # the one 8-bit layout NVENC takes as-is
    if spec.height >= 2160:
        target = ec.LONGFORM_MIN_BITRATE_4K
    elif spec.height >= 1080:
        target = ec.LONGFORM_MIN_BITRATE_1080P
    else:
        target = 5000
    return [
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
        "main" if eight_bit else ec.LONGFORM_PROFILE,
        "-pix_fmt",
        "yuv420p" if eight_bit else "p010le",
        # CQ alone is capped by NVENC's default bitrate; same soft floors as the master encode
        "-b:v",
        f"{target}k",
        "-maxrate",
        f"{int(target * ec.LONGFORM_MAXRATE_RATIO)}k",
        "-bufsize",
        f"{int(target * ec.LONGFORM_BUFSIZE_RATIO)}k",
    ]


def intermediate_video_args(spec: StreamSpec | None = None, *, has_nvenc: bool = False) -> list:
    """Explicit ``-c:v`` / quality / ``-pix_fmt`` / colour-tag args for an intermediate.

    Encoder selection follows ``color_filters.get_encoding_args``: hevc_nvenc
    (``LONGFORM_NVENC_*``, main10 ``p010le``) when ``has_nvenc``, else libx265
    at the long-form CRF and preset. The pixel format keeps the input's depth
    (10-bit stays 10-bit); the colour range / matrix / primaries / transfer
    are copied from the probe, and a tag the input does not carry is left
    unwritten rather than invented. No ``-r``: the input's timing is kept.
    """
    from . import encoding_constants as ec

    spec = spec or StreamSpec()
    if has_nvenc:
        args = _nvenc_video_args(spec)
        return args + _color_flags(spec)
    pix_fmt = spec.pix_fmt if spec.pix_fmt in _X265_PIX_FMTS else ec.LONGFORM_PIX_FMT
    args = [
        "-c:v",
        "libx265",
        "-preset",
        ec.LONGFORM_PRESET,
        "-crf",
        str(ec.LONGFORM_CRF),
        "-pix_fmt",
        pix_fmt,
    ]
    if pix_fmt == ec.LONGFORM_PIX_FMT:
        args += ["-profile:v", ec.LONGFORM_PROFILE]
    args += _color_flags(spec)
    vui = [
        f"{key}={value}"
        for key, value, known in (
            ("colorprim", spec.color_primaries, _X265_PRIMARIES),
            ("transfer", spec.color_transfer, _X265_TRANSFERS),
            ("colormatrix", spec.color_space, _X265_MATRICES),
        )
        if value in known
    ]
    if vui:
        args += ["-x265-params", ":".join(vui)]
    return args


def _color_flags(spec: StreamSpec) -> list:
    args: list = []
    for flag, value in (
        ("-color_range", spec.color_range),
        ("-colorspace", spec.color_space),
        ("-color_primaries", spec.color_primaries),
        ("-color_trc", spec.color_transfer),
    ):
        if value.strip().lower() not in _UNSET_TAGS:
            args += [flag, value]
    return args


def build_deflicker_argv(
    input_path: StrPath,
    output_path: StrPath,
    *,
    mode: str = "pm",
    size: int = 5,
    ffmpeg_bin: str = "ffmpeg",
    stream: StreamSpec | None = None,
    has_nvenc: bool = False,
) -> list:
    """argv for a single-pass ``deflicker`` re-encode (explicit codec, depth, tags)."""
    filter_expr = f"deflicker=mode={mode}:size={size}"
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_expr,
        *intermediate_video_args(stream, has_nvenc=has_nvenc),
        "-c:a",
        "copy",
        str(output_path),
    ]


def build_vidstabdetect_argv(
    input_path: StrPath,
    transforms_path: StrPath,
    capabilities: FfmpegFilterCapabilities,
    *,
    shakiness: int = 5,
    accuracy: int = 15,
    ffmpeg_bin: str = "ffmpeg",
) -> list:
    """argv for vidstab pass 1 (motion analysis, writes the transforms file).

    Raises ``StabilizeUnavailableError`` up front — refusing to hand back an
    argv that would just fail at ffmpeg's filter-graph parse with a much less
    legible error.
    """
    require_vidstab(capabilities)
    filter_expr = (
        f"vidstabdetect=shakiness={shakiness}:accuracy={accuracy}:result={transforms_path}"
    )
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_expr,
        "-f",
        "null",
        "-",
    ]


def build_vidstabtransform_argv(
    input_path: StrPath,
    transforms_path: StrPath,
    output_path: StrPath,
    capabilities: FfmpegFilterCapabilities,
    *,
    smoothing: int = 10,
    zoom: float = 0,
    ffmpeg_bin: str = "ffmpeg",
    stream: StreamSpec | None = None,
    has_nvenc: bool = False,
) -> list:
    """argv for vidstab pass 2 (applies the transforms from pass 1; explicit codec, depth, tags)."""
    require_vidstab(capabilities)
    filter_expr = f"vidstabtransform=input={transforms_path}:smoothing={smoothing}:zoom={zoom}"
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_expr,
        *intermediate_video_args(stream, has_nvenc=has_nvenc),
        "-c:a",
        "copy",
        str(output_path),
    ]


_YAVG_RE = re.compile(r"lavfi\.signalstats\.YAVG=([0-9.]+)")


def measure_luma_jitter(
    path: StrPath,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ffmpeg_bin: str = "ffmpeg",
) -> float:
    """Std of frame-to-frame mean-luma deltas, via ``signalstats``.

    Runs ``ffmpeg -vf signalstats -f null -`` and parses
    ``lavfi.signalstats.YAVG=<value>`` off stderr (one per frame), then
    returns the population std of the deltas between consecutive frames'
    YAVG. A flickering clip has large deltas and a high std; a stable clip
    doesn't.
    """
    argv = [
        ffmpeg_bin,
        "-i",
        str(path),
        "-vf",
        # signalstats alone only attaches per-frame metadata; metadata=print
        # is what actually emits "lavfi.signalstats.YAVG=..." lines to stderr.
        "signalstats,metadata=print",
        "-f",
        "null",
        "-",
    ]
    result = run(argv, capture_output=True, text=True, timeout=_MEASURE_TIMEOUT_S)
    means = [float(m.group(1)) for m in _YAVG_RE.finditer(result.stderr)]
    if len(means) < 2:
        raise ValueError(
            f"could not measure luma jitter for {path}: "
            f"only {len(means)} YAVG samples parsed from ffmpeg stderr"
        )
    deltas = [b - a for a, b in zip(means, means[1:])]
    return statistics.pstdev(deltas)


@dataclass(frozen=True)
class JitterGateResult:
    """Outcome of gating a stabilise/deflicker output on measured jitter."""

    input_jitter: float
    output_jitter: float
    passed: bool
    reason: str = ""


def gate_on_jitter(input_jitter: float, output_jitter: float) -> JitterGateResult:
    """Keep the output only if its measured jitter dropped below the input's."""
    if output_jitter < input_jitter:
        return JitterGateResult(input_jitter, output_jitter, True)
    reason = (
        f"output jitter {output_jitter:.4f} did not drop below input jitter "
        f"{input_jitter:.4f}; discarding stabilised output"
    )
    return JitterGateResult(input_jitter, output_jitter, False, reason)


def run_deflicker(
    input_path: StrPath,
    output_path: StrPath,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ffmpeg_bin: str = "ffmpeg",
    mode: str = "pm",
    size: int = 5,
    has_nvenc: bool = False,
) -> JitterGateResult:
    """Run the deflicker pass and gate the result on measured luma jitter.

    The gate runs before the output replaces anything: on failure,
    ``output_path`` is unlinked so the caller can never mistake a
    jitter-worsening encode for a kept result.
    """
    argv = build_deflicker_argv(
        input_path,
        output_path,
        mode=mode,
        size=size,
        ffmpeg_bin=ffmpeg_bin,
        stream=probe_stream_spec(input_path, run=run),
        has_nvenc=has_nvenc,
    )
    result = run(argv, capture_output=True, text=True, timeout=_ENCODE_TIMEOUT_S)
    if result.returncode != 0:
        raise RuntimeError(
            f"deflicker encode failed (rc={result.returncode}): {result.stderr[-2000:]}"
        )
    input_jitter = measure_luma_jitter(input_path, run=run, ffmpeg_bin=ffmpeg_bin)
    output_jitter = measure_luma_jitter(output_path, run=run, ffmpeg_bin=ffmpeg_bin)
    gate = gate_on_jitter(input_jitter, output_jitter)
    if not gate.passed:
        Path(output_path).unlink(missing_ok=True)
        logger.warning("deflicker output discarded: %s", gate.reason)
    return gate


# --- Stabilise judge: camera motion, not luma ---------------------------------
# A translating image keeps the same per-frame mean, so luma jitter cannot see
# camera shake. The stabilise step is judged on the shake residual instead: per
# consecutive frame pair the global translation (phase correlation on small grey
# frames), accumulated into a camera trajectory; the shake is that trajectory
# minus its own smoothed version, so a deliberate pan or turn is not shake.
MOTION_FRAME_W = 160  # decode size; shake is measured in pixels of this frame
MOTION_FRAME_H = 90
MOTION_MAX_FRAMES = 240  # consecutive frames from the start (8-10 s of footage)
MOTION_MIN_FRAMES = 8
# Moving-average window (frames) that defines "intentional" motion. ~0.5 s at
# 30 fps: slower than a hand tremor, faster than a pan.
MOTION_SMOOTH_WINDOW = 15
# Keep only when output residual <= input residual * this (a 25% cut at least).
MOTION_MAX_RESIDUAL_RATIO = 0.75
# An input already steadier than this (px RMS at 160 px wide) has nothing to fix.
MOTION_MIN_INPUT_RESIDUAL = 0.3
# Black edge band: outer fraction of each side, and the luma at or below which a
# pixel counts as black. Output coverage above the bound means the stabiliser
# zoomed too little and exposed its fill.
MOTION_BORDER_BAND = 0.10
MOTION_BLACK_LEVEL = 16
MOTION_MAX_BORDER_COVERAGE = 0.05


def read_grey_frames(
    path: StrPath,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ffmpeg_bin: str = "ffmpeg",
    max_frames: int = MOTION_MAX_FRAMES,
):
    """First ``max_frames`` consecutive frames as small grey uint8 arrays (ffmpeg pipe)."""
    import numpy as np

    argv = [
        ffmpeg_bin,
        "-v",
        "error",
        "-i",
        str(path),
        "-frames:v",
        str(max_frames),
        "-vf",
        f"scale={MOTION_FRAME_W}:{MOTION_FRAME_H}",
        "-pix_fmt",
        "gray",
        "-f",
        "rawvideo",
        "-",
    ]
    result = run(argv, capture_output=True, timeout=_MEASURE_TIMEOUT_S)
    if result.returncode != 0:
        raise RuntimeError(f"frame decode failed for {path} (rc={result.returncode})")
    size = MOTION_FRAME_W * MOTION_FRAME_H
    raw = result.stdout
    count = len(raw) // size
    return [
        np.frombuffer(raw, dtype=np.uint8, count=size, offset=i * size).reshape(
            MOTION_FRAME_H, MOTION_FRAME_W
        )
        for i in range(count)
    ]


def shake_residual(frames) -> float:
    """RMS (px) of the camera trajectory's deviation from its smoothed self.

    Translation only (rotation is not measured). The trajectory is the running
    sum of per-pair phase-correlation shifts; the smoothed version is a moving
    average with odd-reflected edges, so a constant-velocity pan has ~0 residual.
    """
    import cv2
    import numpy as np

    if len(frames) < MOTION_MIN_FRAMES:
        raise ValueError(f"need {MOTION_MIN_FRAMES} frames to measure shake, got {len(frames)}")
    grey = [np.asarray(f, dtype=np.float32) for f in frames]
    window = cv2.createHanningWindow((grey[0].shape[1], grey[0].shape[0]), cv2.CV_32F)
    steps = [cv2.phaseCorrelate(a, b, window)[0] for a, b in zip(grey, grey[1:])]
    traj = np.cumsum(np.asarray(steps, dtype=np.float64), axis=0)
    pad = min(MOTION_SMOOTH_WINDOW // 2, len(traj) - 1)
    padded = np.pad(traj, ((pad, pad), (0, 0)), mode="reflect", reflect_type="odd")
    kernel = np.ones(2 * pad + 1) / (2 * pad + 1)
    smooth = np.stack(
        [np.convolve(padded[:, k], kernel, mode="valid") for k in range(traj.shape[1])], axis=1
    )
    return float(np.sqrt(np.mean(np.sum((traj - smooth) ** 2, axis=1))))


def border_coverage(frames) -> float:
    """Mean fraction of the outer edge band that is near-black."""
    import numpy as np

    fractions = []
    for f in frames:
        a = np.asarray(f)
        h, w = a.shape
        bh, bw = max(1, int(h * MOTION_BORDER_BAND)), max(1, int(w * MOTION_BORDER_BAND))
        band = np.ones(a.shape, dtype=bool)
        band[bh:-bh, bw:-bw] = False
        fractions.append(float(np.mean(a[band] <= MOTION_BLACK_LEVEL)))
    return float(np.mean(fractions))


@dataclass(frozen=True)
class MotionGateResult:
    """Outcome of judging a stabilise output on camera motion."""

    input_residual: float
    output_residual: float
    border_coverage: float
    passed: bool
    reason: str = ""


def gate_on_motion(
    input_residual: float, output_residual: float, coverage: float
) -> MotionGateResult:
    """Keep the output only when shake dropped enough and borders stayed small."""
    reasons = []
    if input_residual < MOTION_MIN_INPUT_RESIDUAL:
        reasons.append(
            f"input shake residual {input_residual:.3f}px is below "
            f"{MOTION_MIN_INPUT_RESIDUAL}px; nothing to stabilise"
        )
    elif output_residual > input_residual * MOTION_MAX_RESIDUAL_RATIO:
        reasons.append(
            f"shake residual {output_residual:.3f}px did not drop to "
            f"{MOTION_MAX_RESIDUAL_RATIO:.0%} of input {input_residual:.3f}px"
        )
    if coverage > MOTION_MAX_BORDER_COVERAGE:
        reasons.append(f"black border coverage {coverage:.3f} exceeds {MOTION_MAX_BORDER_COVERAGE}")
    if not reasons:
        return MotionGateResult(input_residual, output_residual, coverage, True)
    reason = "; ".join(reasons) + "; discarding stabilised output"
    return MotionGateResult(input_residual, output_residual, coverage, False, reason)


def judge_stabilisation(
    input_path: StrPath,
    output_path: StrPath,
    *,
    run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ffmpeg_bin: str = "ffmpeg",
    read_frames: Callable[..., list] = read_grey_frames,
) -> MotionGateResult:
    """Measure shake residual (input and output) and output border coverage, then gate."""
    before = read_frames(input_path, run=run, ffmpeg_bin=ffmpeg_bin)
    after = read_frames(output_path, run=run, ffmpeg_bin=ffmpeg_bin)
    return gate_on_motion(shake_residual(before), shake_residual(after), border_coverage(after))


@dataclass(frozen=True)
class PictureStepsResult:
    """Path to grade (``src`` when nothing was kept) plus the record of what ran."""

    path: Path
    record: dict


def _step_record(gate: JitterGateResult) -> dict:
    return {
        "input_jitter": gate.input_jitter,
        "output_jitter": gate.output_jitter,
        "kept": gate.passed,
    }


def apply_picture_steps(
    src: StrPath,
    out_dir: StrPath,
    stem: str,
    *,
    deflicker: bool,
    stabilize: bool,
    capabilities: FfmpegFilterCapabilities | None = None,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
    has_nvenc: bool = False,
) -> PictureStepsResult:
    """Deflicker then stabilise ``src`` into ``out_dir``; never raises.

    Deflicker is gated on luma jitter, stabilise on camera motion: a discarded (or failed) step
    deletes its output and the chain continues with the previous input. The
    returned ``record`` carries ``deflicker_applied`` / ``stabilize_applied``
    plus per-step jitter numbers. Intermediates other than the returned path
    are removed; the caller owns removing the returned path when it is not
    ``src``.
    """
    run = run or subprocess.run
    out_dir = Path(out_dir)
    current = Path(src)
    record: dict = {"deflicker_applied": False, "stabilize_applied": False, "steps": {}}

    if deflicker:
        out = out_dir / f"{stem}.deflicker.mp4"
        try:
            gate = run_deflicker(current, out, run=run, has_nvenc=has_nvenc)
        except Exception as exc:  # encode / measure failure: keep the input
            out.unlink(missing_ok=True)
            logger.warning("deflicker failed, continuing with input: %s", exc)
            record["steps"]["deflicker"] = {"kept": False, "error": str(exc)}
        else:
            record["steps"]["deflicker"] = _step_record(gate)
            if gate.passed:
                current = out
                record["deflicker_applied"] = True

    if stabilize:
        out = out_dir / f"{stem}.stabilize.mp4"
        trf = out_dir / f"{stem}.stabilize.trf"
        try:
            caps = capabilities if capabilities is not None else probe_ffmpeg_filters()
            for argv in (
                build_vidstabdetect_argv(current, trf, caps),
                build_vidstabtransform_argv(
                    current,
                    trf,
                    out,
                    caps,
                    stream=probe_stream_spec(current, run=run),
                    has_nvenc=has_nvenc,
                ),
            ):
                res = run(argv, capture_output=True, text=True, timeout=_ENCODE_TIMEOUT_S)
                if res.returncode != 0:
                    raise RuntimeError(f"vidstab rc={res.returncode}: {res.stderr[-2000:]}")
            gate = judge_stabilisation(current, out, run=run)
        except Exception as exc:
            out.unlink(missing_ok=True)
            logger.warning("stabilise failed, continuing with input: %s", exc)
            record["steps"]["stabilize"] = {"kept": False, "error": str(exc)}
        else:
            record["steps"]["stabilize"] = {
                "input_residual": gate.input_residual,
                "output_residual": gate.output_residual,
                "border_coverage": gate.border_coverage,
                "kept": gate.passed,
            }
            if gate.passed:
                if current != Path(src):
                    current.unlink(missing_ok=True)
                current = out
                record["stabilize_applied"] = True
            else:
                out.unlink(missing_ok=True)
                logger.warning("stabilise output discarded: %s", gate.reason)
        finally:
            trf.unlink(missing_ok=True)

    return PictureStepsResult(current, record)
