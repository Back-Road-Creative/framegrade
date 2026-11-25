"""Prepend a branded montage to a raw LF.

`regular_lf` OST layout (ost_applier.LAYOUTS) overlays the location
drawtext across the full montage window. This module builds the
montage from the LF's own `montage_peaks` (high-score clips from the
same LF's footage, per longform_assembler.select_montage_peaks) and
concatenates it in front of the assembled LF body.

The heavy filter-graph montage with crossfades + overlays is already
implemented in `drive_assembler._render_montage`. For the day-first
segment-driven path we re-use a simpler approach: stream-copy each
peak's central slice out of the already-color-corrected day master,
stream-copy-concat them into a reel, then concat that reel with the
assembled LF body. This keeps the prepend generational-loss-free
and avoids re-encoding the LF itself.

HARD CUTS BETWEEN MONTAGE CLIPS ARE DELIBERATE — not a missing
feature (a deliberate colour/visual-quality decision).
A crossfade between two clips requires decoding and re-encoding both
sides of every join, which would put a generation of lossy re-encode
on the one part of the LF made entirely of the day master's
highest-scoring footage, and would re-encode the LF body behind it.
The lossless stream-copy join IS the feature; the cut-on-the-beat
look is the house style, not a compromise. If crossfades are ever
wanted here, the cost is explicit: the montage becomes a filter-graph
render (see `drive_assembler._render_montage`) and stops being
generational-loss-free. Do not "fix" the hard cuts without taking
that trade knowingly.

Duration cap.
``MONTAGE_DURATION_SEC`` is the authoritative total montage length.
It is sourced from ``config.drive.montage_max_duration_sec`` (default
15.0s, lowered from the legacy 30.0s per operator feedback).
``ost_applier.LAYOUTS["regular_lf"]`` and ``day_pipeline`` OST calls
read the same config field so the OST title crossfade window
tracks the montage tail — single source of truth for intro timing.

The cap is enforced by ``_compute_montage_plan``: per-clip duration
is first shrunk toward a 1.5s floor (the operator explicitly rejected
1.0s as too short); only if that floor cannot satisfy the cap is the
peak count reduced. When the cap actually fires an INFO line is
logged so operators can grep it out of the drive log.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from keycut import (
    Segment as CutSegment,
)
from keycut import (
    concat_demuxer,
    keyframe_at_or_before,
    probe_keyframes,
    reencode_range,
    segment_to_master_range,
    snap_start_to_keyframe,
    stream_copy_range,
)

from ..config import get_config
from .color_filters import OUTPUT_FPS, probe_source_fps, resolve_output_fps

logger = logging.getLogger(__name__)

# Merge threshold pinned to the day master's nominal GOP (60 frames at
# 29.97 fps ≈ 2.002 s; ``color_filters.get_encoding_args`` sets
# ``gop_size = 60``) plus float margin. keycut would otherwise MEASURE the
# master's real GOP, which differs on an irregular master — this pins the
# pipeline's long-standing fixed constant.
GOP_DURATION_SEC = 2.1


_LF_REENCODE_ARGS: Optional[list[str]] = None
_LF_REENCODE_ARGS_BY_RATE: dict[str, list[str]] = {}


def reencode_args_for(source, output_fps: str = OUTPUT_FPS) -> list[str]:
    """``lf_reencode_args`` at the rate a source-cadence run keeps (``"source"`` = probe ``source``)."""
    if output_fps == "source":
        output_fps = resolve_output_fps(output_fps, probe_source_fps(Path(source)))
    return lf_reencode_args() if output_fps == OUTPUT_FPS else lf_reencode_args(output_fps)


def lf_reencode_args(output_fps: str = OUTPUT_FPS) -> list[str]:
    """ffmpeg output arguments for the accurate-seek re-encode fallback.

    keycut would otherwise derive an encoder from the master's own codec at
    CRF 18. This pipeline instead pins the shared long-form encoder args
    (libx265 slow CRF 18 + the 4K bitrate floor + BT.709 colour metadata)
    so a repaired join is parameter-compatible with the stream-copied ranges
    around it, and AAC 256k audio so the re-encoded piece carries the same
    audio parameters the rest of the pipeline emits (keycut's default is
    ``-c:a copy``).

    ``3840×2160`` only feeds the bitrate-floor hint — no scale filter is
    added, so the input resolution is preserved. ``has_nvenc=False`` keeps
    this path hermetic and deterministic: ``detect_nvenc`` raises without a
    GPU, and this join-repair path is rare enough that the CPU encode cost
    is acceptable.

    ``cpu_by_choice=True`` because the CPU encode is deliberate: the shared
    "NVENC unavailable" WARNING would falsely report a GPU fault every run.

    ``output_fps`` is the frozen cadence unless a source-cadence run names the
    master's own rate, so a repaired join matches the pieces around it.

    Cached per process (and per rate): the args are constant.
    """
    global _LF_REENCODE_ARGS
    from .color import get_encoding_args

    tail = ["-c:a", "aac", "-b:a", "256k"]
    if output_fps != OUTPUT_FPS:
        if output_fps not in _LF_REENCODE_ARGS_BY_RATE:
            _LF_REENCODE_ARGS_BY_RATE[output_fps] = [
                *get_encoding_args(
                    3840, 2160, 29.97, has_nvenc=False, cpu_by_choice=True, output_fps=output_fps
                ),
                *tail,
            ]
        return list(_LF_REENCODE_ARGS_BY_RATE[output_fps])
    if _LF_REENCODE_ARGS is None:
        _LF_REENCODE_ARGS = [
            *get_encoding_args(3840, 2160, 29.97, has_nvenc=False, cpu_by_choice=True),
            *tail,
        ]
    return list(_LF_REENCODE_ARGS)


def copy_range_argv(src, start: float, end: float, out, *, ffmpeg_bin: str = "ffmpeg") -> list[str]:
    """argv of keycut's ``stream_copy_range`` (input-side seek, ``-c copy``)."""
    return [ffmpeg_bin, "-y", "-ss", f"{float(start)}", "-to", f"{float(end)}", "-i", str(src)] + [
        "-c",
        "copy",
        str(out),
    ]


def reencode_range_argv(
    src, start: float, end: float, out, *, ffmpeg_bin: str = "ffmpeg"
) -> list[str]:
    """argv of keycut's ``reencode_range`` with the pinned join-repair args.

    An input-side seek with an output-side ``-ss 0`` and a ``-t`` length, as
    keycut 0.1.2 runs it: it decodes one GOP before the cut, not the whole
    source before it.
    """
    duration = float(end) - float(start)
    return [ffmpeg_bin, "-y", "-ss", f"{float(start)}", "-i", str(src), "-ss", "0"] + [
        "-t",
        f"{duration}",
        *lf_reencode_args(),
        str(out),
    ]


def concat_copy_argv(list_file, out, *, ffmpeg_bin: str = "ffmpeg") -> list[str]:
    """argv of keycut's ``concat_demuxer`` (stream-copy join, faststart)."""
    return [ffmpeg_bin, "-y", "-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy"] + [
        "-movflags",
        "+faststart",
        str(out),
    ]


# A stream-copy join of MP4 pieces keeps only the FIRST piece's decoder
# configuration (the ``hvcC``/``avcC`` box holding the VPS/SPS/PPS). A piece
# from another encoder (a libx265 join-repair or no-keyframe re-encode next to
# NVENC master footage) then cannot be decoded past the join: the container
# still reports the full length, so duration checks pass, and only a decoder
# finds out (Galley Head 2026-11-25: 12.95 s of a 65 min long-form decoded, the
# OST burn failed on A/V drift on every heal). Converting each piece to Annex B
# writes its parameter sets in front of its own keyframes, so the joined
# stream carries them and every piece decodes. Still lossless: no re-encode.
_ANNEXB_BSF = {"hevc": "hevc_mp4toannexb", "h264": "h264_mp4toannexb"}
# MP4 sample entries whose parameter sets already travel in-band.
_IN_BAND_TAGS = {"hev1", "avc3"}
# An Annex B join that feeds another one (a mixed montage, then its prepend)
# is written as MPEG-TS, never MP4: the remux of an MP4 piece writes that
# file's FIRST parameter sets in front of every keyframe, so a later clip's
# open-GOP keyframe would lose the reference pictures its leading frames need.
# A TS piece is already Annex B and passes through the remux untouched.


def video_parameter_sets(path, *, ffprobe_bin: str = "ffprobe") -> Optional[tuple[str, str]]:
    """``(codec, configuration)`` of the first video stream, or None when unreadable.

    The configuration is the sha256 of the decoder configuration record;
    pieces with different values cannot be joined by a plain stream copy. A
    piece whose parameter sets travel in-band (``hev1``/``avc3`` or MPEG-TS,
    e.g. the output of :func:`annexb_join`) is marked ``in-band`` so it never
    compares equal to an out-of-band piece.
    """
    try:
        res = subprocess.run(
            [ffprobe_bin, "-v", "error", "-select_streams", "v:0", "-show_data_hash", "sha256"]
            + ["-show_entries", "stream=codec_name,codec_tag_string,extradata_hash"]
            + ["-show_entries", "format=format_name"]
            + ["-of", "json", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        info = json.loads(res.stdout or "{}") if res.returncode == 0 else {}
    except Exception:  # noqa: BLE001 - any probe failure is "unreadable": the caller warns
        return None
    streams = info.get("streams") or []
    container = (info.get("format") or {}).get("format_name", "")
    if not streams or not streams[0].get("codec_name"):
        return None
    codec = streams[0]["codec_name"]
    digest = streams[0].get("extradata_hash") or ""
    if streams[0].get("codec_tag_string") in _IN_BAND_TAGS or "mpegts" in container:
        return codec, f"in-band:{digest}"
    return (codec, digest) if digest else None


def annexb_join_codec(parts, description: str) -> tuple[Optional[str], Optional[str]]:
    """How ``parts`` must be stream-copy joined: ``(codec, None)`` for
    :func:`annexb_join`, ``(None, None)`` for the plain concat, or
    ``(None, cause)`` when no stream-copy join of them can decode.

    Unreadable parameter sets keep the plain concat with a WARNING, as
    ``_montage_output_is_sound`` does when it cannot probe.
    """
    sets = [video_parameter_sets(p) for p in parts]
    unreadable = [str(p) for p, got in zip(parts, sets) if got is None]
    if unreadable:
        logger.warning(
            "⚠ %s: video parameter sets unreadable for %s — keeping the plain "
            "stream-copy join (pieces from different encoders would not decode "
            "past the first one)",
            description,
            unreadable,
        )
        return None, None
    if len(set(sets)) == 1:
        return None, None
    codecs = sorted({codec for codec, _ in sets})
    if len(codecs) != 1 or codecs[0] not in _ANNEXB_BSF:
        return (
            None,
            f"{description}: pieces cannot be joined by stream copy (video codecs {codecs})",
        )
    logger.info(
        "%s: pieces carry different %s parameter sets — joining through Annex B",
        description,
        codecs[0],
    )
    return codecs[0], None


def _format_duration(path, ffprobe_bin: str) -> float:
    try:
        res = subprocess.run(
            [ffprobe_bin, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0"]
            + [str(path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return float(res.stdout.strip()) if res.returncode == 0 else 0.0
    except (OSError, subprocess.SubprocessError, ValueError):
        return 0.0


def annexb_join(
    parts,
    output_path,
    list_file,
    codec: str,
    *,
    description: str,
    container: str = "mp4",
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
) -> Optional[str]:
    """Stream-copy join ``parts`` whose parameter sets differ; None on success, else the cause.

    Each piece is remuxed to Annex B MPEG-TS by its own ffmpeg writing to a
    pipe (no extra disk I/O on a 38 GB body), and the concat demuxer reads
    ``pipe:N`` in order, told each piece's length so the timeline stays
    contiguous. ``container`` is ``"mp4"`` (faststart) for a finished file or
    ``"mpegts"`` for one another join reads. The join runs under the
    stall-detecting runner. Any failure, of the join or of a piece's remux,
    deletes the output.
    """
    from .ffmpeg_runner import run_ffmpeg_encode

    parts = [Path(p) for p in parts]
    output_path, list_file = Path(output_path), Path(list_file)
    durations = [_format_duration(p, ffprobe_bin) for p in parts[:-1]]
    short = [str(p) for p, d in zip(parts, durations) if d <= 0]
    if short:
        return f"{description}: no duration for {short}"

    producers = []
    try:
        for i, part in enumerate(parts):
            err = list_file.with_name(f"{list_file.stem}_part{i:02d}.err")
            with open(err, "w") as err_file:
                proc = subprocess.Popen(
                    [ffmpeg_bin, "-nostdin", "-v", "error", "-i", str(part)]
                    + ["-map", "0:v:0", "-map", "0:a:0?", "-c", "copy"]
                    + ["-bsf:v", _ANNEXB_BSF[codec], "-f", "mpegts", "pipe:1"],
                    stdout=subprocess.PIPE,
                    stderr=err_file,
                )
            producers.append((part, proc, err))
        lines = []
        for i, (_part, proc, _err) in enumerate(producers):
            lines.append(f"file 'pipe:{proc.stdout.fileno()}'")
            if i < len(durations):
                lines.append(f"duration {durations[i]}")
        list_file.write_text("\n".join(lines) + "\n")
        result = run_ffmpeg_encode(
            [ffmpeg_bin, "-nostdin", "-y", "-protocol_whitelist", "file,pipe"]
            + ["-f", "concat", "-safe", "0", "-i", str(list_file), "-c", "copy"]
            + (["-movflags", "+faststart"] if container == "mp4" else ["-f", container])
            + [str(output_path)],
            output_path=output_path,
            expected_duration=0.0,
            description=description,
            pass_fds=tuple(proc.stdout.fileno() for _, proc, _ in producers),
        )
    finally:
        broken = []
        for part, proc, err in producers:
            proc.stdout.close()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if proc.returncode != 0:
                tail = err.read_text(errors="replace").strip().splitlines()[-1:]
                broken.append(f"{part} (exit {proc.returncode}{': ' + tail[0] if tail else ''})")
            err.unlink(missing_ok=True)

    cause = None
    if broken:
        cause = f"{description}: Annex B remux failed for " + "; ".join(broken)
    elif not result.success:
        tail = (result.stderr_tail or "").strip().splitlines()[-1:]
        cause = (
            f"{description} killed ({result.killed_reason})"
            if result.killed_reason
            else f"{description} failed (exit {result.returncode})"
        ) + (f": {tail[0]}" if tail else "")
    if cause:
        output_path.unlink(missing_ok=True)
        logger.error("%s", cause)
    return cause


def keyframe_span(
    start: float, end: float, keyframes: list[float], duration: float
) -> tuple[float, float]:
    """The GOP-aligned span bounding ``[start, end]``: the keyframe at or before
    ``start`` and the first keyframe at or after ``end`` (else the file end)."""
    cut_in = keyframe_at_or_before(start, keyframes)
    cut_out = next((k for k in keyframes if k >= end), duration)
    return (0.0 if cut_in is None else cut_in), cut_out


def to_cut_segment(segment) -> CutSegment:
    """Adapt an ``LFSegment``-like object to keycut's ``Segment`` tuple.

    ``source_file_index`` is DAY-LOCAL here — it indexes into the day's
    ``file_boundaries``. Callers holding drive-global indices must remap
    first (``phase_bc_day_first._partition_lf_plans_by_date``).
    """
    return CutSegment(
        int(getattr(segment, "source_file_index", 0)),
        float(segment.start_time),
        float(segment.end_time),
    )


def _authoritative_max_duration() -> float:
    """Read the current authoritative montage duration from config.

    Lazy read at call time (not import time) so per-invocation
    ``config.yaml`` overrides and test monkeypatches take effect.
    """
    return float(get_config().drive.montage_max_duration_sec)


# Authoritative montage duration. ``ost_applier.LAYOUTS["regular_lf"]``
# and ``day_pipeline`` OST calls read this via
# ``config.drive.montage_max_duration_sec`` so the OST title crossfade
# window tracks the montage tail. Module-level binding kept for
# backward compatibility with prior imports; config remains the source
# of truth at call time via ``_authoritative_max_duration()``.
MONTAGE_DURATION_SEC = _authoritative_max_duration()

# Operator-requested floor: clips shorter than this feel "too short"
# (1.0s was explicitly rejected). When the cap forces a
# trim, per-clip duration is shrunk toward this floor before peak
# count is reduced.
MIN_CLIP_DURATION_SEC = 1.5


def _compute_montage_plan(
    peak_count: int,
    requested_total_duration: float,
    max_total_duration: float,
    min_clip_duration_sec: float = MIN_CLIP_DURATION_SEC,
) -> tuple[int, float, bool]:
    """Return the (effective peak count, per-clip duration, was_capped) plan.

    The cap is enforced by:

      1. If ``requested_total_duration <= max_total_duration`` → no-op;
         return ``(peak_count, requested_total_duration/peak_count, False)``.
      2. Otherwise shrink per-clip duration toward ``min_clip_duration_sec``
         while keeping all peaks (``peak_count`` unchanged, ``D = max/N``).
      3. If that per-clip duration would fall below the floor, reduce
         ``N`` so that ``N * min_clip_duration_sec <= max_total_duration``
         and set ``D = max_total_duration / N``.

    Returns a tuple ``(n_used, clip_dur, was_capped)``. Caller is
    responsible for slicing the peaks list to ``n_used`` and passing
    ``clip_dur`` to the per-peak extractor. ``was_capped`` indicates
    whether the cap branch ran (used by the INFO log at the call site
    — this helper emits it so the log carries both requested vs
    effective totals).
    """
    if peak_count <= 0:
        return 0, 0.0, False

    was_capped = requested_total_duration > max_total_duration
    effective_total = min(requested_total_duration, max_total_duration)

    n_used = peak_count
    clip_dur = effective_total / n_used
    if clip_dur < min_clip_duration_sec:
        # Floor breached — reduce peak count so the floor holds. Even
        # when ``was_capped`` is False (caller under the cap but with
        # too many peaks) the floor is binding: clips shorter than
        # 1.5s feel like a strobe and the operator explicitly
        # rejected 1.0s.
        n_used = max(1, int(effective_total // min_clip_duration_sec))
        clip_dur = effective_total / n_used

    if was_capped:
        logger.info(
            "montage capped at %.1fs (would have been %.1fs): using %d/%d peaks at %.2fs each",
            max_total_duration,
            requested_total_duration,
            n_used,
            peak_count,
            clip_dur,
        )
    return n_used, clip_dur, was_capped


def _trim_peak_to(
    master_path: Path,
    peak,
    file_boundaries,
    output: Path,
    clip_duration: float,
    output_fps: str = OUTPUT_FPS,
) -> bool:
    """Stream-copy a `clip_duration` slice of `peak` from the master
    into `output`. Returns True on success.

    The slice is centred in the peak, then its start is snapped forward
    to the first keyframe that still fits a full clip inside the peak —
    an unaligned `-ss … -c copy` cut snaps BACK to the prior keyframe,
    replaying up to one GOP (~2s) of pre-peak footage per clip and
    ballooning the capped montage (segment-join finding). When the keyframe probe is UNAVAILABLE, a `-c copy` would
    snap back on every clip, so the centred window is RE-ENCODED with an
    accurate seek instead. The same holds when keyframes exist but none fit
    inside the peak: the start stays unaligned, so a `-c copy` would snap back up to
    one GOP per clip (+1.7 s on a 15.0 s plan, Galley Head 2026-11), and the centred
    window is re-encoded with an accurate seek too."""
    try:
        s, e = segment_to_master_range(to_cut_segment(peak), file_boundaries)
    except IndexError as exc:
        logger.error("montage peak translate failed: %s", exc)
        return False
    mid = (s + e) / 2.0
    clip_start = max(s, mid - clip_duration / 2.0)
    keyframes = probe_keyframes(master_path)
    if keyframes is None:
        # Probe unavailable — `-c copy` would snap back to the prior keyframe
        # and replay pre-peak footage. Re-encode the centred clip accurately.
        # Logged at WARNING (not cached, fires every peak) so a run that
        # silently downgrades every montage clip to a CPU re-encode is
        # visible to whoever is reading the drive log, not just DEBUG.
        logger.warning(
            "montage peak keyframe probe unavailable for %s — re-encoding "
            "centred window instead of stream-copying (CPU, generational loss)",
            master_path,
        )
        clip_end = min(e, clip_start + clip_duration)
        if clip_end <= clip_start:
            return False
        return reencode_range(
            master_path,
            clip_start,
            clip_end,
            output,
            encode_args=reencode_args_for(master_path, output_fps),
        )
    clip_start = snap_start_to_keyframe(
        clip_start,
        e - clip_duration + 1e-3,
        keyframes,
    )
    clip_end = min(e, clip_start + clip_duration)
    if clip_end <= clip_start:
        return False
    if not any(abs(k - clip_start) <= 1e-3 for k in keyframes):
        # No keyframe fits: `-c copy` would snap back to the prior keyframe and
        # replay pre-peak footage. Cut the centred window accurately instead.
        # WARNING like the probe-unavailable branch: a montage quietly downgraded
        # to a CPU re-encode (generational loss) must be visible in the drive log.
        logger.warning(
            "montage peak has no keyframe at its start in %s — re-encoding the "
            "centred window instead of stream-copying (CPU, generational loss)",
            master_path,
        )
        return reencode_range(
            master_path,
            clip_start,
            clip_end,
            output,
            encode_args=reencode_args_for(master_path, output_fps),
        )
    return stream_copy_range(master_path, clip_start, clip_end, output)


def _montage_output_is_sound(
    output_path: Path, target_duration: float, failures: Optional[list] = None
) -> bool:
    """Check the assembled montage against the duration the plan asked for.

    Uses ``PipelineValidator.validate_montage_output`` (10% tolerance — the
    keyframe snap in ``_trim_peak_to`` moves cut points by up to a GOP, so an
    exact match is not expected).

    ``probe_duration`` returns ``0.0`` as a blanket failure sentinel, which is
    NOT evidence the montage is broken — an ffprobe that cannot run says
    nothing about the file. That case warns and passes rather than failing a
    good montage, but it says plainly that the montage went out unverified.
    """
    # Local imports: pipeline_validator pulls in the content/longform/shorts
    # validators, none of which montage assembly needs at module import time.
    # subprocess is imported here too, not taken from module scope: this gate is
    # the only remaining user of it in this module, so a change that stops the
    # rest of the file shelling out directly would otherwise strand this call
    # with a NameError that no single branch's tests can see.
    import subprocess as _subprocess

    from .pipeline_validator import PipelineValidator
    from .validation.ffprobe_client import FFprobeClient

    if FFprobeClient(_subprocess).probe_duration(output_path) <= 0:
        logger.warning(
            "⚠ build_montage: could not probe %s — montage shipped UNVERIFIED "
            "(expected ~%.1fs). Check that ffprobe is on PATH.",
            output_path.name,
            target_duration,
        )
        return True

    result = PipelineValidator().validate_montage_output(
        output_path, target_duration=target_duration
    )
    if not result.passed:
        logger.error(
            "build_montage: assembled montage failed duration verification — %s. "
            "The clip extractions or the concat produced a bad file; refusing to "
            "prepend it (returning None so the caller skips this LF).",
            result.message,
        )
        if failures is not None:
            failures.append(f"assembled montage failed duration verification — {result.message}")
        return False
    return True


def build_montage(
    master_path: Path,
    peaks: list,
    file_boundaries,
    output_path: Path,
    work_dir: Path,
    total_duration: Optional[float] = None,
    max_total_duration: Optional[float] = None,
    output_fps: str = OUTPUT_FPS,
    failures: Optional[list] = None,
) -> Optional[Path]:
    """Assemble a montage from ``peaks`` capped at
    ``max_total_duration`` seconds.

    ``total_duration`` is the caller's requested total length; if
    ``None`` the value is sourced from
    ``config.drive.montage_max_duration_sec``.
    ``max_total_duration`` is the hard cap enforced by
    ``_compute_montage_plan``; it also defaults to
    ``config.drive.montage_max_duration_sec`` so by default the
    request equals the cap and no trim fires.

    When ``total_duration`` exceeds ``max_total_duration`` the plan
    shrinks per-clip duration toward a 1.5s floor and then reduces
    peak count if the floor is breached — see
    ``_compute_montage_plan``.

    Returns the montage file on success, None on failure or when peaks
    is empty. A failure's cause is appended to ``failures`` when given, so the
    caller can name it instead of logging a bare "failed".
    """
    master_path = Path(master_path)
    output_path = Path(output_path)
    work_dir = Path(work_dir)

    if not peaks:
        logger.warning(
            "⚠ build_montage: no peaks provided (master=%s, output=%s) — "
            "preferred path needs ≥1 peak; falling back to no-op (returning None, "
            "no montage will be built)",
            master_path,
            output_path,
        )
        return None

    if total_duration is None:
        total_duration = _authoritative_max_duration()
    if max_total_duration is None:
        max_total_duration = _authoritative_max_duration()

    n_used, clip_duration, _capped = _compute_montage_plan(
        peak_count=len(peaks),
        requested_total_duration=total_duration,
        max_total_duration=max_total_duration,
    )
    effective_peaks = peaks[:n_used]

    work_dir.mkdir(parents=True, exist_ok=True)

    clip_files: list[Path] = []
    for i, peak in enumerate(effective_peaks):
        clip_file = work_dir / f"montage_clip_{i:02d}.mp4"
        if _trim_peak_to(
            master_path,
            peak,
            file_boundaries,
            clip_file,
            clip_duration,
            **({} if output_fps == OUTPUT_FPS else {"output_fps": output_fps}),
        ):
            clip_files.append(clip_file)

    if not clip_files:
        logger.error("build_montage: every peak clip extraction failed")
        if failures is not None:
            failures.append("every montage peak clip extraction failed")
        return None

    list_file = work_dir / f"{output_path.stem}_montage_concat_list.txt"
    # A re-encoded clip (no keyframe at its peak) carries other parameter sets
    # than the copied master clips: join those through Annex B or the montage
    # stops decoding at the first such seam.
    codec, refused = annexb_join_codec(clip_files, "montage clip concat")
    if refused or codec:
        # Written as MPEG-TS: the prepend joins it again (see _ANNEXB_BSF).
        output_path = output_path.with_suffix(".ts")
        cause = refused or annexb_join(
            clip_files,
            output_path,
            list_file,
            codec,
            description="montage clip concat",
            container="mpegts",
        )
        if cause:
            if failures is not None:
                failures.append(cause)
            return None
    elif not concat_demuxer(clip_files, output_path, list_file):
        if failures is not None:
            failures.append("montage clip concat failed")
        return None

    # Verify what we just assembled BEFORE it is prepended to the LF and
    # encoded. A silently-empty clip extraction or a truncated concat used to
    # ride through here unchecked and only surface after an irreversible
    # multi-hour encode.
    if not _montage_output_is_sound(output_path, n_used * clip_duration, failures):
        return None

    for p in clip_files:
        try:
            p.unlink()
        except OSError:
            pass
    try:
        list_file.unlink()
    except OSError:
        pass
    return output_path


def prepend_montage(
    raw_lf_path: Path,
    peaks: list,
    output_path: Path,
    work_dir: Path,
    master_path: Optional[Path] = None,
    file_boundaries: Optional[list] = None,
    output_fps: str = OUTPUT_FPS,
    failures: Optional[list] = None,
) -> Optional[Path]:
    """Prepend a capped-duration montage built from `peaks` to `raw_lf_path`.

    Montage total length is bounded by
    ``config.drive.montage_max_duration_sec`` (default 15.0s — see
    module docstring).

    Args:
        raw_lf_path: The assembled raw LF body (output of
            `day_pipeline.extract_and_concat_segments`).
        peaks: LFSegment-like objects referencing the day master.
        output_path: Destination for the [montage + LF] concat.
        work_dir: Scratch directory for intermediates.
        master_path: Required source for peak slices. If None, the
            raw LF is returned unchanged (peak source unavailable).
        file_boundaries: Day-local file_boundaries for peak
            translation. Required when `master_path` is provided.

    Takes no text: the montage's only place text is the LF's checked OST,
    which ``apply_ost`` overlays afterwards (D8b).

    Returns the concatenated output path on success, None on failure. Every
    None return appends its cause to ``failures`` when given (the caller raises
    with it: a planned long-form is never dropped silently).
    """
    raw_lf_path = Path(raw_lf_path)
    output_path = Path(output_path)
    work_dir = Path(work_dir)

    if not raw_lf_path.exists():
        logger.error("prepend_montage: raw LF missing %s", raw_lf_path)
        if failures is not None:
            failures.append(f"raw LF missing: {raw_lf_path}")
        return None

    if master_path is None or file_boundaries is None or not peaks:
        logger.warning(
            "⚠ prepend_montage: preferred path (build+prepend montage) unavailable "
            "— master_path_present=%s, file_boundaries_present=%s, peaks=%d; "
            "falling back to no-op (returning None; OST title window tracks "
            "the montage tail so a raw LF would leave the title hanging over "
            "footage — caller must skip this LF)",
            master_path is not None,
            file_boundaries is not None,
            len(peaks) if peaks else 0,
        )
        # We still must not return the raw LF silently — that would ship
        # without a montage. Surface the failure so the caller can skip
        # this LF rather than publish a broken one.
        if failures is not None:
            failures.append(
                "montage inputs unavailable "
                f"(master={master_path is not None}, "
                f"file_boundaries={file_boundaries is not None}, peaks={len(peaks or [])})"
            )
        return None

    montage = work_dir / f"{output_path.stem}_montage.mp4"
    built = build_montage(
        master_path=master_path,
        peaks=peaks,
        file_boundaries=file_boundaries,
        output_path=montage,
        work_dir=work_dir,
        output_fps=output_fps,
        failures=failures,
    )
    if built is None:
        if failures is not None and not failures:
            failures.append("build_montage returned None")
        return None

    list_file = work_dir / f"{output_path.stem}_prepend_list.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # The montage may hold libx265 re-encodes, the body is NVENC master
    # footage: with different parameter sets the plain copy below would decode
    # only the montage (Galley Head 2026-11-25), so that pair joins through
    # Annex B instead.
    codec, refused = annexb_join_codec([built, raw_lf_path], "prepend_montage concat")
    if refused or codec:
        cause = refused or annexb_join(
            [built, raw_lf_path],
            output_path,
            list_file,
            codec,
            description="prepend_montage concat",
        )
        if cause:
            if failures is not None:
                failures.append(cause)
            return None
        for p in (built, list_file):
            try:
                p.unlink()
            except OSError:
                pass
        return output_path

    lines = []
    for p in [built, raw_lf_path]:
        escaped = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_file.write_text("\n".join(lines) + "\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    # Stall-detecting runner, not a fixed wall clock: a 38 GB `-c copy
    # +faststart` concat on a slow volume ran past the old `timeout=1800` while
    # still writing (Galley Head 2026-11). It is killed only when ffmpeg stops
    # making progress: a stats line that advanced, or, during the silent
    # moov rewrite, growth of ffmpeg's own I/O counters (`_read_io_counters`),
    # so the rewrite takes as long as the volume needs. Where those counters
    # cannot be read the rewrite window is max(size window, 2x this run's
    # elapsed time), which also scales with a slow volume. It passes NO
    # `max_timeout`: the runner's default 86400 s ceiling stands, because a
    # size-based ceiling killed a concat that was still advancing whenever the
    # volume was slower than its assumed speed, on every retry. On a stall, a
    # non-zero exit or a start failure it deletes the partial output: a
    # truncated concat left on disk is worse than none, since a resume would
    # ship a half-length long-form. The failure is reported as None with its
    # cause.
    from .ffmpeg_runner import run_ffmpeg_encode

    result = run_ffmpeg_encode(
        cmd,
        output_path=output_path,
        expected_duration=0.0,
        description="prepend_montage concat",
    )
    if not result.success:
        cause = (
            f"prepend concat killed ({result.killed_reason})"
            if result.killed_reason
            else f"prepend concat failed (exit {result.returncode})"
        )
        tail = (result.stderr_tail or "").strip().splitlines()[-1:]
        logger.error("prepend_montage concat failed: %s %s", cause, " ".join(tail))
        if failures is not None:
            failures.append(cause + (f": {tail[0]}" if tail else ""))
        return None

    try:
        built.unlink()
    except OSError:
        pass
    try:
        list_file.unlink()
    except OSError:
        pass

    return output_path
