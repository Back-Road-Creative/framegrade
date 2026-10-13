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

Every pass here is gated by measured per-frame luminance jitter
(``measure_luma_jitter`` / ``gate_on_jitter``): the output only replaces the
input when its jitter measurably dropped. A pass that makes jitter worse (or
doesn't improve it) is discarded and the caller is told why — never a silent
pass-through of a filter that made things worse.

``apply_picture_steps`` is the drive-path entry (``--stabilize`` /
``--deflicker`` via ``day_master_encoder.encode_day_master``); see
``docs/pipeline/142-stabilise-and-deflicker.md``.
"""

from __future__ import annotations

import logging
import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Union

logger = logging.getLogger(__name__)

StrPath = Union[str, Path]

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


def build_deflicker_argv(
    input_path: StrPath,
    output_path: StrPath,
    *,
    mode: str = "pm",
    size: int = 5,
    ffmpeg_bin: str = "ffmpeg",
) -> list:
    """argv for a single-pass ``deflicker`` re-encode."""
    filter_expr = f"deflicker=mode={mode}:size={size}"
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_expr,
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
) -> list:
    """argv for vidstab pass 2 (applies the transforms from pass 1)."""
    require_vidstab(capabilities)
    filter_expr = f"vidstabtransform=input={transforms_path}:smoothing={smoothing}:zoom={zoom}"
    return [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vf",
        filter_expr,
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
) -> JitterGateResult:
    """Run the deflicker pass and gate the result on measured luma jitter.

    The gate runs before the output replaces anything: on failure,
    ``output_path`` is unlinked so the caller can never mistake a
    jitter-worsening encode for a kept result.
    """
    argv = build_deflicker_argv(
        input_path, output_path, mode=mode, size=size, ffmpeg_bin=ffmpeg_bin
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
) -> PictureStepsResult:
    """Deflicker then stabilise ``src`` into ``out_dir``; never raises.

    Each step is gated on measured luma jitter: a discarded (or failed) step
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
            gate = run_deflicker(current, out, run=run)
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
                build_vidstabtransform_argv(current, trf, out, caps),
            ):
                res = run(argv, capture_output=True, text=True, timeout=_ENCODE_TIMEOUT_S)
                if res.returncode != 0:
                    raise RuntimeError(f"vidstab rc={res.returncode}: {res.stderr[-2000:]}")
            gate = gate_on_jitter(
                measure_luma_jitter(current, run=run), measure_luma_jitter(out, run=run)
            )
        except Exception as exc:
            out.unlink(missing_ok=True)
            logger.warning("stabilise failed, continuing with input: %s", exc)
            record["steps"]["stabilize"] = {"kept": False, "error": str(exc)}
        else:
            record["steps"]["stabilize"] = _step_record(gate)
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
