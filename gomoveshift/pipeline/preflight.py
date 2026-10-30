"""Pre-flight validation for the Drive Pipeline.

Runs before Phase A to catch configuration and environment issues that would
waste hours of encode time. The checks are plain Python functions so the
orchestrator can call them programmatically.

Checks:
  1. Source directory exists and contains .MP4 files
  2. Disk space sufficient for processing (output + staging). The staging
     gate sizes the TRUE peak (all persisted episode concats + the busiest
     day's master working set, source × 1.5) so it agrees with the per-day
     mid-run gate and fails UPFRONT, not hours in. On WSL the Windows host
     volume backing the ext4 VHDX is checked too — the guest ext4 free
     space is blind to the host drive the VHDX expands into.
  3. Input files are non-empty and a recognized extension (a suspiciously
     small but valid clip only WARNs — ingest drops sub-minimum clips).
  4. Free RAM above a minimum threshold
  5. Required tools (ffmpeg, ffprobe) present on PATH
  6. OST titles are unique across episodes (post-Phase A hook)

Also exposes ``cleanup_stale_staging()`` — opportunistic removal of old
per-project staging dirs so the persistent intermediate cache does not grow
unbounded across runs.
"""

from __future__ import annotations

import errno
import importlib.util
import logging
import os
import shutil
import stat
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from gomoveshift.paths import gms_config_home

from .ost_validator import validate_ost, validate_ost_uniqueness

logger = logging.getLogger(__name__)

_GB = 1_073_741_824  # bytes per gibibyte

# ``--superlong`` output-space sizing, shared between the per-day gate
# (``check_superlong_output_space``, measured against the real day master)
# and the run-level upfront estimate (``estimate_peak_cache_gb``), which has
# no master to measure yet and reuses these same two factors against raw
# source bytes instead — see ``estimate_peak_cache_gb`` for why. Kept as one
# pair of constants so the two estimates can't drift apart.
_SUPERLONG_RAW_PLUS_FINAL_FACTOR = 2.0  # raw stream-copied concat + final coexist
_SUPERLONG_OUTPUT_SAFETY_FACTOR = 1.2  # margin over the sizing estimate itself


def _virtual_memory():
    """Return object with `.available` bytes. Wrapped for test patchability.

    Uses psutil when available, falls back to parsing /proc/meminfo on Linux.
    """
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory()
    except ImportError:
        pass

    meminfo = Path("/proc/meminfo")
    if sys.platform != "win32" and meminfo.exists():
        available_kb = 0
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
                break

        class _Mem:
            available = available_kb * 1024

        return _Mem()

    class _Unknown:
        available = 0

    return _Unknown()


def estimate_peak_cache_gb(
    source_files: Iterable[Path],
    *,
    safety_factor: float = 1.5,
    floor_gb: float = 30.0,
    completed_source_bytes: float = 0.0,
    superlong_enabled: bool = False,
) -> float:
    """Estimate TRUE peak intermediate-cache usage in GB from source sizes.

    Aligned with the per-day mid-run gate (``day_master_encoder.encode_day_master``,
    which checks ``day_source_bytes × 1.5`` via ``ffmpeg_runner.check_disk_space``)
    so preflight catches a shortfall UPFRONT instead of mid-encode on day N
    (an early check, not a late one). A mid-run ``Insufficient disk space``
    failure is a LATE gate firing hours in; an under-estimate here (a flat
    ``source × 1.2``) is how that happens.

    The peak the staging FS must hold at the worst moment:
      * ALL retained per-episode concats persist on the staging dir through
        scoring AND the per-day master encode (Phase A routes them to
        ``state.stages_base`` so they fit on ext4, not a v9fs output mount)
        ≈ total source bytes (×1.0), PLUS
      * the busiest day's master-encode working set — its raw concat +
        the master being written + work dir ≈ that day's source × ~0.5 on
        top of the day's own already-counted concat (the per-day gate
        expresses this same spike as ``day_source × 1.5``).

    Preflight runs BEFORE ingest groups recordings into days, so it cannot
    measure the largest single day. The fail-safe bound that always covers
    the per-day gate's worst case — a single-day trip, where largest_day ==
    total — is ``total_source × 1.5``: it dominates ``total + largest_day×0.5``
    for any day distribution and reserves conservatively (over-, never
    under-, estimating a SAFETY gate). Hence the default ``safety_factor=1.5``,
    matching the per-day formula exactly when the whole trip is one day.

    ``completed_source_bytes`` makes the gate RESUME-AWARE: on a resume
    with N/M day masters already on disk, those days' masters are cached and
    their raw episodes are purgeable, so they no longer contribute to the
    remaining staging peak. The caller passes the sum of completed days'
    recorded ``source_bytes`` (from ``per_day_completed``); it is subtracted
    from the measured source total BEFORE the safety factor, so a 3/4-done
    resume no longer demands the full ``source × 1.5`` and fails a gate the
    fresh run already cleared. Clamped at 0 (a stale/over-count never makes the
    estimate negative); the ``floor_gb`` minimum still applies, and a legacy
    record without ``source_bytes`` subtracts 0 (conservative, back-compat).

    ``floor_gb`` enforces a minimum regardless of source size — a tiny
    trip still needs OST / final workspace, and early-exiting preflight
    with a "your 1 MB test needs 0.5 MB" message helps nobody.

    ``superlong_enabled`` adds a term for planned ``--superlong``
    output space. The per-day gate (``check_superlong_output_space``) sizes
    against the REAL day master's bytes/sec, but no master exists yet at
    run-level preflight time — so this reuses that gate's own two factors
    (raw concat + final coexist, plus its safety margin) against raw SOURCE
    bytes instead, the same proxy convention the per-day master-encode gate
    already uses for its own working set (``day_source_bytes × 1.5`` in
    ``day_master_encoder``). Without this term a ``--superlong`` run could
    clear the run-level gate and then fill the disk mid-day, hours into a
    live encode, which is a late failure; the check belongs upstream of the encode.

    Missing or unreadable files are silently skipped — the caller already
    validates file presence via ``validate_input_files``.
    """
    total_bytes = 0
    for f in source_files:
        try:
            if f.is_file():
                total_bytes += f.stat().st_size
        except OSError:
            continue
    remaining_bytes = max(0.0, total_bytes - max(0.0, completed_source_bytes))
    est_gb = (remaining_bytes * safety_factor) / _GB
    if superlong_enabled:
        est_gb += (
            remaining_bytes * _SUPERLONG_RAW_PLUS_FINAL_FACTOR * _SUPERLONG_OUTPUT_SAFETY_FACTOR
        ) / _GB
    return max(est_gb, floor_gb)


def check_disk_space(path: Path, min_free_gb: int = 50) -> None:
    """Raise RuntimeError if *path*'s filesystem has < *min_free_gb* GB free.

    Walks up to the nearest existing ancestor if *path* doesn't exist yet.
    """
    target = Path(path)
    while not target.exists():
        parent = target.parent
        if parent == target:
            break
        target = parent

    usage = shutil.disk_usage(target)
    free_gb = usage.free / _GB
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient disk space at {path}: "
            f"{free_gb:.1f} GB free, need {min_free_gb} GB. "
            f"Free up space before running the pipeline."
        )


def check_superlong_output_space(
    master_path: Path,
    output_dir: Path,
    total_part_duration_sec: float,
    *,
    safety_factor: float = _SUPERLONG_OUTPUT_SAFETY_FACTOR,
    probe_fn=None,
) -> None:
    """Raise RuntimeError if *output_dir* lacks room for the planned
    superlong parts — both the raw stream-copied concat (``sl_NN_segments.mp4``)
    and the ``sl_NN.mp4`` final it feeds, which coexist for the whole encode.

    The fixed 50 GB ``check_disk_space`` floor is meaningless at superlong
    scale (a 4h part FINAL runs ~70-90 GB at the frozen 40 Mbps 4K profile),
    so this scales with the real footage instead: bytes/sec is measured from
    the ACTUAL day-master file on disk (size / duration), never an assumed
    bitrate. Called at part-production start, upstream of the irreversible
    per-part concat + OST re-encode (an early check, not a late one).

    ``probe_fn`` defaults to ``FFprobeClient(subprocess).probe_duration``
    (injected for testability). A duration that can't be measured makes the
    bytes/sec figure meaningless rather than a real space shortfall — the
    master's own integrity is gated separately in ``day_master_encoder`` —
    so this logs a warning and skips instead of raising a misleading error.
    """
    if probe_fn is None:
        import subprocess

        from .validation.ffprobe_client import FFprobeClient

        probe_fn = FFprobeClient(subprocess).probe_duration

    master_path = Path(master_path)
    master_bytes = master_path.stat().st_size
    master_duration = float(probe_fn(master_path))
    if master_duration <= 0:
        logger.warning(
            "Superlong output-space gate: could not measure %s's duration "
            "(probe returned %.2fs) — skipping the space estimate.",
            master_path,
            master_duration,
        )
        return

    bytes_per_sec = master_bytes / master_duration
    # Two artifacts co-exist for the whole OST-encode duration: the raw
    # stream-copied concat (`sl_NN_segments.mp4`, written first — a `-c copy`
    # extraction from the master, so it runs at the master's own bytes/sec)
    # AND the final it feeds, never deleted until the encode completes. One
    # finals-equivalent copy alone under-reserves by ~1.7x against that.
    raw_plus_final_bytes_per_sec = bytes_per_sec * _SUPERLONG_RAW_PLUS_FINAL_FACTOR
    needed_gb = (
        raw_plus_final_bytes_per_sec * float(total_part_duration_sec) * safety_factor
    ) / _GB
    free_gb = shutil.disk_usage(output_dir).free / _GB
    if free_gb < needed_gb:
        raise RuntimeError(
            f"Insufficient output space for superlong parts at {output_dir}: "
            f"{free_gb:.1f} GB free, need ~{needed_gb:.1f} GB (shortfall "
            f"{needed_gb - free_gb:.1f} GB) — sized from the actual day "
            f"master ({master_bytes / _GB:.2f} GB / {master_duration:.0f}s "
            f"= {bytes_per_sec * 3600 / _GB:.1f} GB/hr) x "
            f"{total_part_duration_sec / 3600:.2f}h planned x 2 (raw concat + "
            f"final coexist) x {safety_factor} safety margin. Free up space "
            f"in the output directory before producing superlong parts."
        )


# One long-form holds three artifacts on the cache volume while it burns: the
# stream-copied concat (``lf_NN_segments.mp4``), the montage-prepended copy
# (``lf_NN_with_montage.mp4``) and the final. Measured 2026-10-08 on a 30-minute
# LF: a 16.78 GB share of the day master peaked at 51.78 GB (x3.09).
_LF_WORK_FACTOR = 3.1
_LF_WORK_SAFETY_FACTOR = 1.1  # margin over the sizing estimate itself


def check_lf_work_space(
    master_path: Path,
    output_dir: Path,
    lf_duration_sec: float,
    master_duration_sec: float,
    *,
    lf_index: int | None = None,
    later_lf_secs: Sequence[float] = (),
    safety_factor: float = _LF_WORK_SAFETY_FACTOR,
) -> None:
    """Raise RuntimeError if *output_dir* lacks room for one long-form's work
    files plus its final, and for every later long-form of the run beside it.

    Sized from the ACTUAL day master on disk (size / duration), like
    ``check_superlong_output_space``, never an assumed bitrate. Called per LF
    right before its concat, upstream of the stream copy, the montage prepend
    and the OST re-encode: a shortfall fails here with the numbers instead of
    as a half-written concat or a montage step that only logs "skipping".

    ``master_duration_sec`` comes from the day's file boundaries (the caller
    already holds them), so nothing is probed. Inputs that cannot price
    anything log a warning and skip: a master that cannot be read (the concat
    then fails on its own, having written nothing), or a size or day length
    that is not positive. A part with no length prices at zero and passes.

    ``later_lf_secs`` are the lengths of the LFs this run still burns after
    this one. Each final stays on the cache volume while the next LF's three
    artifacts are written, so the peak is the worst LF's work plus the finals
    burned before it. Pricing that here stops a day before its first concat
    instead of at a later LF's gate hours in (Ace Basin 2026-10-27: LF 2
    needed 145.8 GiB beside LF 1's 47.9 GiB final, ~125 GiB left).
    """
    try:
        master_bytes = Path(master_path).stat().st_size
    except OSError as exc:
        logger.warning(
            "LF work-space gate: cannot read %s (%s) - skipping the space estimate.",
            master_path,
            exc,
        )
        return
    if master_duration_sec <= 0 or master_bytes <= 0:
        logger.warning(
            "LF work-space gate: no usable size/duration for %s (%d bytes, %.2fs) "
            "- skipping the space estimate.",
            master_path,
            master_bytes,
            master_duration_sec,
        )
        return

    bytes_per_sec = master_bytes / float(master_duration_sec)
    share_bytes = bytes_per_sec * float(lf_duration_sec)
    peak_bytes = share_bytes * _LF_WORK_FACTOR
    finals_before = share_bytes
    for sec in later_lf_secs:
        later_bytes = bytes_per_sec * float(sec)
        peak_bytes = max(peak_bytes, finals_before + later_bytes * _LF_WORK_FACTOR)
        finals_before += later_bytes
    needed_gb = peak_bytes * safety_factor / _GB
    free_gb = shutil.disk_usage(output_dir).free / _GB
    if free_gb < needed_gb:
        label = f"LF {lf_index}" if lf_index is not None else "a long-form"
        later = (
            f" and {len(later_lf_secs)} later long-form(s) burned beside the finals before them"
            if later_lf_secs
            else ""
        )
        raise RuntimeError(
            f"Insufficient work space for {label}{later} at {output_dir}: "
            f"{free_gb:.1f} GB free, need ~{needed_gb:.1f} GB (shortfall "
            f"{needed_gb - free_gb:.1f} GB) - its {share_bytes / _GB:.1f} GB share of "
            f"the day master x {_LF_WORK_FACTOR} (stream copy + montage copy + final "
            f"coexist) x {safety_factor} margin. Free space on the cache volume "
            f"(`gomoveshift cleanup <published-project> --confirm`), then resume: "
            f"finished long-forms are kept."
        )


_WSL_HOST_MIN_FREE_GB = 25.0  # absolute floor for the host volume even on resume


def _is_wsl() -> bool:
    """True when running under WSL (ext4 root lives in a host-backed VHDX)."""
    if sys.platform == "win32":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def check_wsl_host_volume_space(
    staging_dir: Path,
    needed_gb: float,
    *,
    host_volume: Path | None = None,
) -> None:
    """On WSL, verify the Windows host volume backing the ext4 VHDX has room.

    Inside WSL2 the Linux root filesystem is a dynamically-expanding
    ``ext4.vhdx`` stored on a Windows volume (C: by default). Writing into
    ext4 grows the VHDX on the host, and the VHDX never shrinks on its own.
    ``shutil.disk_usage('/')`` reports free space *inside* the guest — it is
    blind to the host volume, so a run can pass the ext4 staging gate yet
    fill the Windows drive and crash the host (the guest ext4 can report
    hundreds of GB free while the host drive is nearly full).

    This gate measures the host volume directly via its drvfs mount
    (``/mnt/c`` by default, override with ``GMS_WSL_HOST_VOLUME``). No-op
    when not on WSL. When running ON WSL but the host volume can't be
    resolved, it HARD-FAILS rather than skipping: a silent warn+return makes
    the disk-space gate a no-op exactly when it matters (the backing drive is
    unlocatable), which re-opens the host-exhaustion crash class on a non-C:
    VHDX. Point ``GMS_WSL_HOST_VOLUME`` at the backing drive to clear it.
    """
    if not _is_wsl():
        return
    if host_volume is None:
        host_volume = Path(os.environ.get("GMS_WSL_HOST_VOLUME", "/mnt/c"))
    if not host_volume.exists():
        raise RuntimeError(
            f"WSL host-volume space check cannot run: {host_volume} is not "
            f"mounted, so the drive backing the ext4 VHDX cannot be measured. "
            f"Skipping this gate would let the VHDX expand into a near-full "
            f"host drive and crash Windows. Set "
            f"GMS_WSL_HOST_VOLUME to the drive that backs the WSL VHDX "
            f"(its /mnt/<letter> mount) and retry."
        )
    free_gb = shutil.disk_usage(host_volume).free / _GB
    if free_gb < needed_gb:
        raise RuntimeError(
            f"Insufficient space on the Windows host volume {host_volume}: "
            f"{free_gb:.1f} GB free, need ~{needed_gb:.1f} GB. On WSL2 the "
            f"ext4 staging filesystem lives in a host-backed VHDX that grows "
            f"as the pipeline writes — the guest reports free space while the "
            f"host drive fills and crashes Windows. Free space on "
            f"{host_volume}, then reclaim the VHDX with `wsl --shutdown` + "
            f"diskpart `compact vdisk` (or `wsl --manage <distro> "
            f"--set-sparse true`). Override the checked volume with "
            f"GMS_WSL_HOST_VOLUME if the VHDX lives on another drive."
        )


def validate_input_files(
    files: list[Path],
    min_size_mb: int = 100,
    allowed_extensions: tuple[str, ...] | list[str] = (".mp4", ".MP4"),
    warnings: list[str] | None = None,
) -> list[str]:
    """Validate a list of input video files.

    Returns a list of error strings. Empty list means all hard checks
    passed. Non-fatal observations (a suspiciously small clip) are logged
    at WARNING and, when *warnings* is supplied, appended to it — they are
    NOT added to the returned errors and never abort the run.

    A genuinely empty / missing / wrong-extension file is still a hard
    error. A small-but-valid file is only a warning because ingest
    deliberately DROPS clips under ``MIN_EPISODE_DURATION`` (1200s) in
    ``ingest/episode_grouper.py``: small but valid GoPro clips must
    not hard-fail preflight when ingest discards them anyway, and a
    genuinely truncated/corrupt file still fails downstream at
    ffprobe/ingest — so downgrading loses no real safety.

    ``allowed_extensions`` matching is case-insensitive — pass lowercase
    entries (e.g. ``[".mp4", ".mov"]``); ``.MP4`` files match ``.mp4``.
    """
    errors: list[str] = []

    if not files:
        errors.append("No input files provided.")
        return errors

    allowed_lower = tuple(ext.lower() for ext in allowed_extensions)

    for f in files:
        f = Path(f)
        if not f.exists():
            errors.append(f"File not found: {f}")
            continue

        size = f.stat().st_size
        if size == 0:
            errors.append(f"File is empty (0 bytes): {f}")
            continue

        if f.suffix.lower() not in allowed_lower:
            errors.append(
                f"Unexpected file extension '{f.suffix}' for {f.name} "
                f"(expected {', '.join(allowed_lower)})"
            )
            continue

        size_mb = size / (1024 * 1024)
        if size_mb < min_size_mb:
            msg = (
                f"File suspiciously small ({size_mb:.1f} MB < {min_size_mb} MB): {f} "
                f"— ingest drops sub-minimum clips; not aborting."
            )
            logger.warning(msg)
            if warnings is not None:
                warnings.append(msg)

    return errors


def check_ram(min_free_gb: float = 2.0) -> None:
    """Raise RuntimeError if available RAM is below *min_free_gb* GB.

    Uses ``psutil.virtual_memory().available`` (kernel estimate of memory
    reclaimable without swapping). Falls back to /proc/meminfo.
    """
    mem = _virtual_memory()
    free_gb = mem.available / _GB
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Insufficient free RAM: {free_gb:.2f} GB available, "
            f"need {min_free_gb} GB. Close other processes and retry."
        )


def check_tools(tools: Iterable[str]) -> None:
    """Raise RuntimeError if any named executable is not on PATH."""
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise RuntimeError(
            f"Required tool(s) not found on PATH: {', '.join(missing)}. "
            f"Install them before running the pipeline."
        )


def check_picture_filters(*, stabilize: bool, deflicker: bool, capabilities=None) -> None:
    """Raise RuntimeError when a requested picture step's ffmpeg filter is absent.

    Nothing is probed unless a flag is set. ``--stabilize`` needs libvidstab
    (``StabilizeUnavailableError`` carries ``VIDSTAB_MISSING_REASON``);
    ``--deflicker`` needs the ``deflicker`` filter.
    """
    if not (stabilize or deflicker):
        return
    from . import stabilize as _stab

    caps = capabilities if capabilities is not None else _stab.probe_ffmpeg_filters()
    if stabilize:
        try:
            _stab.require_vidstab(caps)
        except _stab.StabilizeUnavailableError as exc:
            raise RuntimeError(f"--stabilize refused: {exc}") from exc
    if deflicker and not caps.has_deflicker:
        raise RuntimeError("--deflicker refused: ffmpeg has no 'deflicker' filter")


def check_ost_font() -> None:
    """Raise RuntimeError if the canonical OST burn font is missing.

    Every burned title renders with one hardcoded ``FONT_PATH`` (DejaVu Sans
    Bold) on BOTH surfaces — long-form via ``ost_applier.apply_ost`` and shorts
    via ``ffmpeg_filters.build_ost_filter``. When that file is absent the
    long-form path silently stream-copies a *title-less* video and reports
    success (the ``apply_ost`` missing-font branch), and nothing downstream
    re-verifies the title pixels — so a font-less host would ship every LF with
    no on-screen title. Assert existence here, upstream of any encode, rather
    than discover the missing title after the irreversible burn. ``_validate_font_path`` only rejects
    ffmpeg-special chars in the constant; it deliberately never touches the filesystem, so this is
    the single place the font's existence is gated.
    """
    from .ffmpeg_filters import FONT_PATH

    if not Path(FONT_PATH).is_file():
        raise RuntimeError(
            f"OST burn font not found: {FONT_PATH}. Every burned title "
            "(long-form and shorts) renders with this font; without it the "
            "long-form path silently ships a title-less video. Install the "
            "DejaVu fonts (e.g. `apt-get install fonts-dejavu-core`) or point "
            "the pipeline at an available bold TTF before running."
        )


def cleanup_stale_staging(
    base: Path,
    max_age_days: int = 14,
    protect: Iterable[Path] | None = None,
) -> list[Path]:
    """Remove per-project staging dirs under *base* older than *max_age_days*.

    Returns the list of directories that were removed. Silent no-op if *base*
    doesn't exist. Never touches paths in *protect* (pass the current run's
    project dir to avoid racing a live pipeline).
    """
    base = Path(base)
    if not base.is_dir():
        return []
    protected = {Path(p).resolve() for p in (protect or ())}
    cutoff = time.time() - (max_age_days * 86400)
    removed: list[Path] = []
    # Accumulate per-dir rmtree failures and surface them in a single
    # end-of-function WARNING instead of only emitting one log line per
    # iteration. A persistently-stuck dir (open handle, EBUSY mount) would
    # otherwise vanish into the noise across runs while the cache grows
    # unbounded — the aggregate count + bytes makes it visible.
    failed: list[Path] = []
    failed_bytes = 0
    for child in base.iterdir():
        if not child.is_dir():
            continue
        if child.resolve() in protected:
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        if mtime >= cutoff:
            continue
        # Robust blast-radius bound: the sweep may only reclaim PURE staging.
        # A candidate that is (or contains) a finished-output / resumable-state
        # artifact is a real PROJECT, never reclaimable staging, and must be
        # unreachable by construction. This blocks BOTH failure shapes: (a) an
        # ext4 --output-dir makes the sweep base == the output base, so a
        # finished sibling's output dir (FINAL.mp4 + youtube_uploads.json)
        # sits directly under it; (b) a paused-at-gate project's dir mtime
        # reflects project START, so its masters/state would be flagged and
        # destroyed, bypassing the master-deletion integrity gate. Every real
        # project carries at least one such marker, so a sibling project can
        # never be selected.
        if _holds_output_artifact(child):
            logger.info(
                "Stale-staging cleanup: keeping %s — holds an output/resumable-state "
                "artifact (FINAL/master/youtube_uploads.json/pipeline_state.json), "
                "not reclaimable staging",
                child,
            )
            continue
        try:
            child_bytes = _dir_size_bytes(child)
            shutil.rmtree(child)
        except OSError as exc:
            logger.warning("Could not remove stale staging dir %s: %s", child, exc)
            failed.append(child)
            failed_bytes += child_bytes
            continue
        removed.append(child)
        logger.info(
            "Removed stale staging dir: %s (age %.1f days)", child, (time.time() - mtime) / 86400
        )
    if failed:
        logger.warning(
            "Stale-staging cleanup: %d dir(s) could not be reclaimed (~%.2f GB still held); "
            "they will be retried next run. Stuck path(s): %s",
            len(failed),
            failed_bytes / _GB,
            ", ".join(str(p) for p in failed),
        )
    return removed


# Exact-name markers whose presence proves a candidate dir holds a finished
# output / resumable project state (not pure staging): the YouTube-upload
# dedupe sidecar and the resumable pipeline state.
_OUTPUT_ARTIFACT_NAMES = frozenset({"youtube_uploads.json", "pipeline_state.json"})
# Suffix markers for the expensive-to-reproduce media: long-form finals and
# per-day masters (~60 GB / ~6h to re-encode).
_OUTPUT_ARTIFACT_SUFFIXES = ("_FINAL.mp4", "_master.mp4")


def _holds_output_artifact(path: Path) -> bool:
    """True if *path* is or (recursively) contains a finished-output / resumable
    project-state artifact, so the stale-staging sweep must not delete it."""
    try:
        for _root, _dirs, files in os.walk(path):
            for name in files:
                if name in _OUTPUT_ARTIFACT_NAMES or name.endswith(_OUTPUT_ARTIFACT_SUFFIXES):
                    return True
    except OSError:
        # Unreadable subtree — fail safe (treat as protected, never delete).
        return True
    return False


def _dir_size_bytes(path: Path) -> int:
    """Best-effort recursive byte size of *path*; unreadable entries skipped."""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:
                    continue
    except OSError:
        return total
    return total


def run_preflight_ost_check(episode_titles: list[str]) -> None:
    """Raise RuntimeError on duplicated OR invalid episode OST titles.

    Called after Phase A derives episode locations but before Phase B
    starts encoding. Catches exact and near-duplicates, plus per-title
    garbage (placeholder literals, coordinate strings, None/null) via
    ``validate_ost`` — uniqueness alone lets unique-but-wrong titles ship.
    Empty titles are skipped
    (GPS-less episodes legitimately carry "").
    """
    invalid = []
    for i, title in enumerate(episode_titles):
        if not (title or "").strip():
            continue
        ok, reason = validate_ost(title, "lf")
        if not ok:
            invalid.append(f"  ep {i}: {title!r} — {reason}")
    if invalid:
        raise RuntimeError(
            f"OST validity check failed — {len(invalid)} invalid title(s). "
            f"Fix in ost_overrides.json before encoding:\n" + "\n".join(invalid)
        )
    dupes = validate_ost_uniqueness(episode_titles)
    if dupes:
        lines = []
        for title, indices in dupes.items():
            ep_list = ", ".join(f"ep {i}" for i in indices)
            lines.append(f"  {title!r} appears at: {ep_list}")
        detail = "\n".join(lines)
        raise RuntimeError(
            f"OST uniqueness check failed — {len(dupes)} duplicate title(s) found. "
            f"Fix in ost_overrides.json before encoding:\n{detail}"
        )


def _writable_by_current_user(st: os.stat_result) -> tuple[bool, str]:
    """Return ``(ok, reason)`` describing how the current process can write
    to a file with the given ``stat`` result.

    Reason values:
      * ``"uid_match"`` — current effective uid owns the file.
      * ``"group_write"`` — file is group-writable AND the file's gid is
        in the current process's supplementary group set. With a
        setgid directory and files mode 664, every member of the shared
        group can rewrite without chown.
      * ``"none"`` — neither path applies; file is unwritable by the
        current user without elevation.

    A strict uid-only check would demand a chown for every file another
    member of the group created, even when group-write makes that
    unnecessary.
    """
    my_uid = os.geteuid()
    if st.st_uid == my_uid:
        return (True, "uid_match")
    if (st.st_mode & stat.S_IWGRP) and st.st_gid in os.getgroups():
        return (True, "group_write")
    return (False, "none")


_SEO_STATE_FILENAMES = frozenset({".lock", "events.jsonl", "index.json"})

#: Optional environment variables naming the account that should own the SEO
#: state files. They only shape the repair hint; the writability check runs
#: whether or not they are set.
SEO_OWNER_USER_ENV = "GMS_SEO_OWNER_USER"
SEO_OWNER_GROUP_ENV = "GMS_SEO_OWNER_GROUP"


def check_seo_ownership(
    seo_storage_path: Path,
    project: str,
    *,
    target_user: str | None = None,
    target_group: str | None = None,
) -> None:
    """Detect + repair writability drift on SEO state files.

    When several accounts (for example a container user and a host user)
    write into the SEO state directory, each account's pipeline must still be
    able to rewrite the files another one created, otherwise SEO Manager
    init fails and LFs ship with ``seo_degraded=True``.

    Detection and self-repair always run. The owner account is optional and
    only shapes the fail-fast hint: ``target_user`` / ``target_group`` or the
    ``GMS_SEO_OWNER_USER`` / ``GMS_SEO_OWNER_GROUP`` environment variables
    (arguments win) name the account in the ``chown`` command. With either
    one unset the hint shows placeholders and names the variables to set.

    Coverage:
      * ``<seo_storage_path>/.content_ledger.json``
      * ``<seo_storage_path>/<project>/{.lock,events.jsonl,index.json}``

    A file passes preflight if EITHER:
      1. The current effective uid owns it (``uid_match``), OR
      2. The file is group-writable (``stat.S_IWGRP``) AND the current
         user belongs to the file's group (``group_write`` fallback —
         the typical setgid-dir + 664-files case where every group
         member can write fine without chown).

    When neither holds, the function still attempts ``os.chown`` (covers
    the case where the running user is root or owns a parent and can
    take the file). On ``PermissionError`` it raises ``RuntimeError``
    whose message lists both repair options:
      * ``sudo chown <target_user>:<target_group> <paths>``, OR
      * ``sudo chmod g+w <paths>`` plus group membership.

    Args:
        seo_storage_path: The SEO data directory (``config.seo.storage_path``).
        project: SEO project segment (e.g. ``"gomoveshift"``).
        target_user / target_group: Owner account named in the fail-fast
            command. Default to ``GMS_SEO_OWNER_USER`` / ``GMS_SEO_OWNER_GROUP``;
            when either is still unset the command shows placeholders.
    """
    if not hasattr(os, "geteuid"):
        # Windows: no uid/gid ownership to drift (NTFS uses ACLs).
        return
    target_user = target_user or os.environ.get(SEO_OWNER_USER_ENV, "").strip() or None
    target_group = target_group or os.environ.get(SEO_OWNER_GROUP_ENV, "").strip() or None
    seo_storage_path = Path(seo_storage_path).resolve()
    project_dir = seo_storage_path / project
    candidates = [seo_storage_path / ".content_ledger.json"]
    if project_dir.is_dir():
        for name in sorted(_SEO_STATE_FILENAMES):
            candidates.append(project_dir / name)

    my_uid = os.geteuid()
    my_gid = os.getegid()

    mismatched: list[Path] = []
    for path in candidates:
        try:
            st = os.stat(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            logger.warning("SEO preflight: could not stat %s: %s", path, exc)
            continue
        ok, reason = _writable_by_current_user(st)
        if ok:
            if reason == "group_write":
                logger.info(
                    "SEO preflight: %s writable via group-write fallback "
                    "(gid=%d, mode=%o) — no chown needed",
                    path,
                    st.st_gid,
                    st.st_mode & 0o777,
                )
            continue
        mismatched.append(path)

    if not mismatched:
        return

    unrepairable: list[Path] = []
    for path in mismatched:
        try:
            os.chown(path, my_uid, my_gid)
            logger.info("SEO preflight: repaired ownership on %s", path)
        except PermissionError:
            unrepairable.append(path)
        except OSError as exc:
            logger.warning("SEO preflight: chown failed for %s: %s", path, exc)
            unrepairable.append(path)

    if unrepairable:
        diag_lines = []
        for path in unrepairable:
            try:
                st = os.stat(path)
                diag_lines.append(
                    f"    {path} (uid={st.st_uid}, gid={st.st_gid}, mode={st.st_mode & 0o777:o})"
                )
            except OSError:
                diag_lines.append(f"    {path} (could not stat)")
        diag_block = "\n".join(diag_lines)
        paths_str = " ".join(str(p) for p in unrepairable)
        groups_str = ",".join(str(g) for g in sorted(os.getgroups()))
        if target_user and target_group:
            owner_hint = f"{target_user}:{target_group}"
            owner_note = ""
        else:
            owner_hint = "<owner-user>:<owner-group>"
            owner_note = (
                f"      (set {SEO_OWNER_USER_ENV} and {SEO_OWNER_GROUP_ENV} "
                "to name the accounts here)\n"
            )
        raise RuntimeError(
            "SEO state files cannot be written by the current user "
            f"(uid={my_uid}, gids=[{groups_str}]).\n"
            "Files:\n"
            f"{diag_block}\n"
            "Fix with EITHER of:\n"
            f"  (a) sudo chown {owner_hint} {paths_str}\n"
            f"{owner_note}"
            "  (b) sudo chmod g+w <files> and ensure the invoking user is a member\n"
            "      of the file's group (verify with `id` and `stat -c '%G' <file>`)."
        )


def _device_of(path: str | os.PathLike) -> int:
    """``st_dev`` of *path*, following symlinks. Seam for tests."""
    return os.stat(path).st_dev


def _read_probe(path: str | os.PathLike) -> None:
    """Read the first 4 KiB. ``stat`` alone is not proof: a drvfs
    volume can answer ``stat`` from cache while every read fails EIO."""
    with open(path, "rb") as fh:
        fh.read(4096)


def _read_mounts() -> list[tuple[str, str]]:
    """(mount point, fstype) rows from /proc/self/mounts; empty off Linux."""
    if sys.platform == "win32":
        return []
    try:
        with open("/proc/self/mounts", encoding="utf-8") as fh:
            rows = [line.split() for line in fh]
    except OSError:
        return []
    # /proc/mounts escapes spaces in paths as \040.
    return [(r[1].replace("\\040", " "), r[2]) for r in rows if len(r) >= 3]


def _mount_for(path: str, table: list[tuple[str, str]]) -> tuple[str, str]:
    """Longest mount point that contains *path*, matched on whole components."""
    best = ("?", "unknown")
    for mnt, fstype in table:
        root = mnt.rstrip("/") or "/"
        inside = root == "/" or path == root or path.startswith(root + "/")
        if inside and (best[0] == "?" or len(root) > len(best[0])):
            best = (root, fstype)
    return best


def check_source_filesystem(
    source_dir: Path,
    staging_dir: Path,
    allow_cross_fs: bool = False,
    *,
    video_extensions: Iterable[str] | None = None,
) -> None:
    """Resolve the source video files, refuse unreadable ones, report where
    the reads really happen.

    Correctness (always refuses; no flag bypasses it): a video entry whose
    symlink dangles, or a file that cannot be opened and read. Both mean the
    pipeline would encode from footage it cannot get.

    Performance (never refuses): video files whose resolved target sits on a
    different filesystem than staging. On WSL2 that is the documented
    common layout -- ``~/.cache/gomoveshift/sources/<trip>/`` symlinks
    into a Windows drive mount (drvfs) -- and its cost is read speed (~3x
    master encode wall-clock). Weather carries no score weight and
    ``allow_degraded_weather`` defaults True (config.py), so nothing that
    ranks or renders depends on slow reads. The comparison follows each
    file's symlink, not the directory, so footage symlinked from another
    volume is not hidden by a staging directory on the symlinks' own volume.
    ``allow_cross_fs`` (``--allow-cross-fs-source``) only lowers the report
    from WARNING to INFO.
    """
    src = Path(source_dir)
    stg = Path(staging_dir)
    if not src.is_dir() or not stg.exists():
        return  # other preflight checks surface the missing-path error
    exts = {e.lower() for e in (video_extensions or (".mp4",))}
    entries = sorted(e for e in src.iterdir() if e.suffix.lower() in exts)

    dangling: list[str] = []
    unreadable: list[str] = []
    resolved: list[str] = []
    for entry in entries:
        if entry.is_symlink() and not entry.exists():
            dangling.append(f"{entry.name} -> missing target {os.readlink(entry)}")
            continue
        if not entry.is_file():
            continue
        real = os.path.realpath(entry)
        try:
            _read_probe(real)
        except OSError as exc:
            code = errno.errorcode.get(exc.errno or 0, str(exc.errno))
            unreadable.append(f"{entry.name} ({real}): {code} {exc.strerror}")
            continue
        resolved.append(real)

    if dangling or unreadable:
        lines = [f"  - {d}" for d in dangling] + [f"  - {u} cannot be read" for u in unreadable]
        raise RuntimeError(
            f"Source footage in {src} is not readable:\n" + "\n".join(lines) + "\n"
            "Reattach the volume the files live on (or re-point the symlinks) and "
            "re-run. This is not a filesystem-speed check; "
            "--allow-cross-fs-source does not bypass it."
        )

    stg_real = os.path.realpath(stg)
    stg_dev = _device_of(stg_real)
    remote = [r for r in resolved if _device_of(r) != stg_dev]
    if not remote:
        return
    table = _read_mounts()
    stg_mnt, stg_fs = _mount_for(stg_real, table)
    where = sorted({_mount_for(r, table) for r in remote})
    dirs = sorted({os.path.dirname(r) for r in remote})
    logger.log(
        logging.INFO if allow_cross_fs else logging.WARNING,
        "PERFORMANCE: %d of %d source video files are read from a different "
        "filesystem than staging. Files: %s on %s. Staging: %s on %s (%s). "
        "Correct output, slower reads (~3x master encode wall-clock). To avoid "
        "it, stage on the footage volume or copy the footage next to staging.",
        len(remote),
        len(resolved),
        ", ".join(dirs),
        ", ".join(f"{m} ({f})" for m, f in where),
        stg,
        stg_mnt,
        stg_fs,
    )


ALLOW_CPU_BLUR_ENV = "GMS_ALLOW_CPU_BLUR"
_CUDA_PROVIDER = "CUDAExecutionProvider"


def _import_onnxruntime():
    """Seam for tests; the detectors import the same module."""
    import importlib

    return importlib.import_module("onnxruntime")


# Smallest valid ONNX model: one Identity node, float tensor x -> y, IR 8, opset 13.
# Hand-serialized ModelProto (protobuf wire format) so the probe needs neither the
# ``onnx`` package nor redactcam's model files; verified loadable by onnxruntime.
# Not a credential: the pragmas stop detect-secrets reading the hex as one.
_PROBE_MODEL = bytes.fromhex(
    "08083a2b0a100a017812017922084964"  # pragma: allowlist secret
    "656e746974791201675a090a01781204"  # pragma: allowlist secret
    "0a02080162090a017912040a02080142"  # pragma: allowlist secret
    "02100d"
)


def _probe_session_providers() -> list[str]:
    """Providers a real session actually binds when CUDA is requested first.

    ``get_available_providers()`` lists CUDA whenever the GPU wheel is installed,
    even when ``libcudart``/``libcudnn`` cannot be loaded; only creating a session
    reveals the silent fallback to CPU."""
    ort = _import_onnxruntime()
    session = ort.InferenceSession(_PROBE_MODEL, providers=[_CUDA_PROVIDER, "CPUExecutionProvider"])
    return list(session.get_providers())


def _gpu_onnxruntime_installed() -> bool:
    """``onnxruntime-gpu`` in the venv's dist-info is the installer's stated
    intent to run on CUDA — independent of which wheel's files currently occupy
    the ``onnxruntime/`` package directory (the CPU wheel can overwrite them and
    leave this metadata in place, which is exactly the state this gate catches)."""
    import importlib.metadata

    try:
        importlib.metadata.distribution("onnxruntime-gpu")
    except importlib.metadata.PackageNotFoundError:
        return False
    return True


def check_gopro_telemetry_parser(
    ingest_mode: str | None, *, importable: bool | None = None
) -> None:
    """Step 1c. Refuse a GoPro-ingest run whose venv cannot parse GPMF telemetry.

    Without the ``gopro`` extra ``extract_gopro_gpx`` logs an ERROR and returns
    ``None``, and the run would carry on for hours with no GPS at all.

    Args:
        ingest_mode: ``config.drive.ingest_mode``; only ``"gopro"`` reads GPMF.
            ``None`` (caller did not plumb config) skips.
        importable: ``None`` → is ``gopro_overlay`` findable in this venv.
    """
    if ingest_mode != "gopro":
        return
    if importable is None:
        importable = importlib.util.find_spec("gopro_overlay") is not None
    if importable:
        return
    raise RuntimeError(
        "--ingest-mode gopro reads GPS from GoPro GPMF telemetry, but gopro-overlay is "
        "not installed in this venv, so every clip would come out GPS-less.\n"
        "Fix with EITHER of:\n"
        "  (a) pip install -e '.[gopro]' && pip install --no-deps gopro-overlay==0.134.0 "
        "in this venv (gopro-overlay pins gpxpy==1.6.1, which this package cannot take, "
        "so it is not in the extra and must be installed with --no-deps)\n"
        "  (b) --ingest-mode generic, for footage that carries no GoPro telemetry."
    )


def check_blur_inference_provider(
    *,
    gpu_expected: bool | None = None,
    available: list[str] | None = None,
    session_providers: list[str] | None = None,
) -> None:
    """Step 7c. Refuse to start a blur run whose detectors would silently run on
    CPU when the venv installed the GPU onnxruntime build.

    A venv can have ``onnxruntime-gpu`` and later ``pip install -e`` pulls the CPU
    ``onnxruntime`` wheel over it (both wheels own ``onnxruntime/``, last install
    wins). Providers are then CPU-only, redactcam picks CPU without a word, and
    the plate/face pass runs for hours while the GPU idles. This gate fails at
    minute 0 with the repair.

    Self-contained on purpose — it does not import redactcam — so it holds
    regardless of which redactcam commit the venv carries.

    Args:
        gpu_expected: ``None`` → is the ``onnxruntime-gpu`` distribution installed.
        available: ``None`` → ``onnxruntime.get_available_providers()`` after
            ``preload_dlls()`` (the CUDA libs are pip wheels that must be on the
            loader path first; the detectors do the same).
        session_providers: ``None`` → providers bound by a real probe
            ``InferenceSession`` requesting CUDA first. onnxruntime-gpu (CUDA 12
            build) beside CUDA-13 wheels still *lists* CUDA, so the listing alone
            passes and the run falls back to CPU later. CUDA must be first here,
            i.e. actually loaded.
    """
    if gpu_expected is None:
        gpu_expected = _gpu_onnxruntime_installed()
    if available is None:
        try:
            ort = _import_onnxruntime()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Blur is enabled but onnxruntime is not importable in this venv "
                f"({exc}). Repair: tools/enable_gpu_detection.sh <venv>"
            ) from exc
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        available = list(ort.get_available_providers())
    if not gpu_expected:
        logger.info("Blur inference provider verified: %s", available)
        return
    if _CUDA_PROVIDER in available:
        if session_providers is None:
            session_providers = _probe_session_providers()
        if session_providers and session_providers[0] == _CUDA_PROVIDER:
            logger.info("Blur inference provider verified: %s", session_providers)
            return
        available = [*session_providers] or ["none (session bound no provider)"]
        cause = (
            "CUDAExecutionProvider is listed but a real session fell back to "
            f"{', '.join(available)}: the CUDA runtime libraries (libcudart/libcudnn "
            "of the major version this onnxruntime-gpu wheel was built for) are not "
            "loadable in this venv, e.g. onnxruntime-gpu built for CUDA 12 beside "
            "CUDA 13 nvidia wheels pulled in by torch."
        )
        repair = (
            "  (a) pip install 'onnxruntime-gpu[cuda,cudnn]==<pinned version>' or "
            "bash tools/enable_gpu_detection.sh <venv>\n"
        )
    else:
        cause = (
            "onnxruntime-gpu is installed in this venv but CUDAExecutionProvider is not "
            f"available (providers: {', '.join(available) or 'none'}).\n"
            "The CPU onnxruntime wheel was most likely installed over the GPU build — "
            "both own the onnxruntime/ package directory and the last install wins (a "
            "plain `pip install -e .` can do this and the blur pass then runs for hours "
            "on CPU)."
        )
        repair = (
            "  (a) bash tools/enable_gpu_detection.sh <venv>   (uninstall the CPU wheel, "
            "force-reinstall the pinned onnxruntime-gpu + nvidia-*-cu12 wheels, verify "
            "CUDA)\n"
        )
    if os.environ.get(ALLOW_CPU_BLUR_ENV, "").strip().lower() in {"1", "true", "yes"}:
        logger.warning(
            "%s set: blur detectors will run on CPU (~19x slower) although "
            "onnxruntime-gpu is installed; providers=%s",
            ALLOW_CPU_BLUR_ENV,
            available,
        )
        return
    raise RuntimeError(
        f"{cause}\n"
        "Fix with EITHER of:\n"
        f"{repair}"
        f"  (b) export {ALLOW_CPU_BLUR_ENV}=1 to accept CPU blur knowingly.\n"
        "      Expect the blur pass to take ~19x longer."
    )


def check_weather_decode_path(
    source_files: Iterable[Path],
    weather_hwaccel: str,
) -> None:
    """Refuse to start when the weather decode path won't actually take
    effect.

    The silent failure mode this guards against is ``weather_hwaccel='auto'``
    emitting bare ``-hwaccel auto`` without a ``-c:v <codec>_cuvid`` decoder
    for a 4K HEVC Insta360 source. Software signalstats decode falls below
    realtime, every chunk hits the per-chunk timeout cap, and the measurement
    degrades.

    This preflight check makes the failure unrepresentable from the
    contract side: when ``weather_hwaccel`` is non-``"none"`` and any
    source file uses a cuvid-mappable codec (HEVC / H.264 / AV1), the
    active ffmpeg build MUST advertise both ``cuda`` in ``-hwaccels`` and
    the matching ``<codec>_cuvid`` decoder. Otherwise raise with explicit
    remediation. Phase A never runs in a degraded-decode configuration.

    Args:
        source_files: All source files about to be processed. Each is
            ffprobed once to learn its codec.
        weather_hwaccel: ``config.drive.weather_hwaccel`` value. ``"none"``
            short-circuits the check (operator explicitly opted into
            software decode).

    Raises:
        RuntimeError when the configured hwaccel cannot deliver GPU
        decode for at least one source file that needs it.
    """
    # Import here so the module-level import graph stays the same;
    # weather is a leaf module that already imports from exceptions.
    from .weather import WeatherDetector, ffmpeg_supports_cuda_decode

    if not weather_hwaccel or weather_hwaccel == "none":
        return

    # Probe-only detector: it is asked for a cuvid decoder mapping and
    # nothing else, so config.drive.weather_chunk_* would be inert here.
    # They are threaded at the site that actually measures brightness
    # (pipeline/stages/social.py), which is where the degraded-brightness
    # remediation message is emitted.
    detector = WeatherDetector(hwaccel=weather_hwaccel)
    files = list(source_files)
    if not files:
        return

    mappable: list[tuple[Path, str]] = []
    for f in files:
        decoder = detector._cuvid_decoder_for(f)
        if decoder is not None:
            mappable.append((f, decoder))

    if not mappable:
        # No cuvid-mappable sources — nothing to verify. Auto mode will
        # fall through to software for all of these; that's correct.
        # `cuda` mode would raise on first chunk; but per the contract,
        # operator explicitly asked for cuda and has no cuvid codec —
        # surface that here, not after stage init.
        if weather_hwaccel == "cuda":
            raise RuntimeError(
                f"weather_hwaccel='cuda' was requested but none of the "
                f"{len(files)} source files use a cuvid-mappable codec "
                f"(HEVC / H.264 / AV1). Set "
                f"config.drive.weather_hwaccel='none' to accept software "
                f"decode, or supply sources in a NVDEC-supported codec."
            )
        return

    decoders_needed = {d for _, d in mappable}
    missing: list[str] = []
    for dec in sorted(decoders_needed):
        if not ffmpeg_supports_cuda_decode(dec):
            missing.append(dec)

    if missing:
        sample = mappable[0]
        raise RuntimeError(
            f"weather_hwaccel='{weather_hwaccel}' but the active ffmpeg "
            f"build is missing CUDA decode support for: {', '.join(missing)}.\n"
            f"Example source needing GPU decode: {sample[0]} (codec → {sample[1]}).\n"
            f"Software signalstats on 4K HEVC runs below realtime and will "
            f"degrade weather measurement.\n"
            f"Fix with EITHER of:\n"
            f"  (a) rebuild ffmpeg with --enable-cuda-llvm --enable-nvdec\n"
            f"      and verify with `ffmpeg -hwaccels | grep cuda` and\n"
            f"      `ffmpeg -decoders | grep cuvid`.\n"
            f"  (b) set config.drive.weather_hwaccel='none' in the run\n"
            f"      config to accept software decode.\n"
            f"      Expect Phase A to take ~3-5× longer on 4K HEVC."
        )

    logger.info(
        "Weather decode path verified: ffmpeg supports cuda + cuvid for "
        "%d source file(s) (decoders=%s).",
        len(mappable),
        sorted(decoders_needed),
    )


def check_concat_target_alignment(
    source_dir: Path,
    output_dir: Path,
) -> None:
    """Refuse stale concat symlinks pointing outside the current source dir.

    The concat resume shortcut (ingest/concatenator.py) reuses
    ``<output>/episode_NN_<date>.mp4`` when the file exists, even if the
    symlink's target lives under a previous source directory. Switching the
    source from a Windows drive mount to ext4 mid-rerun would otherwise leave
    a stale symlink silently re-routing Phase A back to the old location.

    Robust-by-construction fix: dereference every existing concat symlink
    and assert its target resolves under the current ``source_dir``. A
    source dir of symlinks to another volume is
    current: a link whose resolved target equals the resolved path of a
    file in the source dir passes. Any other target still raises.
    Stale targets raise — operator deletes the stale concat (or re-passes the
    original source path) rather than silently inheriting it.
    """
    out = Path(output_dir)
    if not out.exists():
        return
    src_resolved = Path(source_dir).resolve()
    # A source dir of symlinks (e.g. to another volume) resolves outside itself; its
    # files' resolved targets are still "the current source".
    src_files = (
        {f.resolve() for f in Path(source_dir).iterdir()} if Path(source_dir).is_dir() else set()
    )
    stale: list[tuple[Path, Path]] = []
    for entry in out.iterdir():
        if not entry.is_symlink():
            continue
        if not entry.name.startswith("episode_"):
            continue
        try:
            target = entry.resolve(strict=False)
        except OSError:
            continue
        if target in src_files:
            continue
        try:
            target.relative_to(src_resolved)
        except ValueError:
            stale.append((entry, target))
    if stale:
        bullets = "\n".join(f"    {link} -> {target}" for link, target in stale)
        raise RuntimeError(
            f"Stale concat symlink(s) in {out} point outside the current "
            f"source directory {src_resolved}:\n{bullets}\n"
            f"Delete these so concat re-runs from the current source, or "
            f"re-invoke pointing at the symlinks' original source dir."
        )


class PreflightBlockedError(RuntimeError):
    """Every blocker ``run_preflight`` found, reported together.

    ``blockers`` is ``[(check, message), ...]`` in check order; ``str()`` is the
    operator-facing ``RENDER BLOCKED`` report.
    """

    def __init__(self, blockers: list[tuple[str, str]]):
        self.blockers = list(blockers)
        super().__init__(format_blockers(self.blockers))


def format_blockers(blockers: list[tuple[str, str]]) -> str:
    """The ``RENDER BLOCKED`` report: one numbered line per blocker."""
    n = len(blockers)
    lines = [f"RENDER BLOCKED: {n} blocker{'s' if n != 1 else ''} found before any encode:"]
    for i, (check, message) in enumerate(blockers, 1):
        lines.append(f"  {i}. [{check}] " + str(message).replace("\n", "\n     "))
    return "\n".join(lines)


@dataclass
class PreflightReport:
    """Checks ``run_preflight`` ran and passed, plus non-blocking warnings."""

    checked: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class _Blockers:
    """Collects each failing check instead of stopping at the first."""

    def __init__(self) -> None:
        self.found: list[tuple[str, str]] = []
        self.checked: list[str] = []

    @contextmanager
    def step(self, name: str):
        try:
            yield
        except RuntimeError as exc:
            self.found.append((name, str(exc)))
        else:
            self.checked.append(name)


def run_preflight(
    source_dir: Path,
    output_dir: Path,
    min_free_gb: int = 50,
    episode_titles: list[str] | None = None,
    *,
    staging_dir: Path | None = None,
    min_staging_gb: float = 30.0,
    min_ram_gb: float = 2.0,
    require_tools: Iterable[str] | None = ("ffmpeg", "ffprobe"),
    seo_storage_path: Path | None = None,
    seo_project: str | None = None,
    video_extensions: Iterable[str] | None = None,
    allow_cross_fs_source: bool = False,
    weather_hwaccel: str | None = None,
    blur_enabled: bool | None = None,
    check_credentials: bool = False,
    completed_source_bytes: float = 0.0,
    ingest_complete: bool = False,
    superlong_enabled: bool = False,
    ingest_mode: str | None = None,
) -> PreflightReport:
    """Run every pre-flight check and report all blockers together.

    Each check that fails is collected rather than raised, so one run names
    every cheap blocker before any encode (render-readiness gate). Raises
    ``PreflightBlockedError`` (a ``RuntimeError``) listing them; returns a
    ``PreflightReport`` of the checks that passed. A missing ``source_dir``
    still raises at once — nothing else can be measured without it.

    Args:
        source_dir: Directory containing raw GoPro .MP4 files.
        output_dir: Target output directory (checked for disk space).
        min_free_gb: Minimum free disk space at *output_dir* in GB.
        episode_titles: If provided, check OST uniqueness. Typically
            passed after Phase A derives titles.
        staging_dir: Intermediate/staging directory (checked separately).
            Created if it doesn't exist.
        min_staging_gb: Minimum free disk space at *staging_dir* in GB.
        min_ram_gb: Minimum available RAM in GB.
        require_tools: Executables that must be on PATH. Pass ``None`` to
            skip the check (tests only).
        completed_source_bytes: Resume hint — total raw source bytes of
            day masters already encoded on disk, subtracted from the staging
            peak-cache estimate. Default 0 (fresh run = full estimate).
        superlong_enabled: This run's ``--superlong`` intent. Adds
            planned superlong output space to the staging peak-cache
            estimate so a superlong run that can't fit refuses here,
            upfront, instead of mid-day in ``check_superlong_output_space``.
        blur_enabled: Whether this run will execute the blur pass
            (``config.drive.blur_enabled and not skip_blur``). ``True`` runs
            step 7c, ``check_blur_inference_provider``; ``None`` (default) skips
            it, keeping call sites without a config working.
        ingest_mode: ``config.drive.ingest_mode``. ``"gopro"`` runs step 1c,
            ``check_gopro_telemetry_parser``; ``None`` (default) skips it.
    """
    logger.info("Pre-flight: validating environment...")
    gate = _Blockers()

    # 1. Source directory
    if not Path(source_dir).is_dir():
        raise RuntimeError(
            f"Source directory not found: {source_dir}. Check the path and try again."
        )

    # 1c. GPMF parser for GoPro ingest. Cheap
    # and fatal, so it runs before any disk or encode measurement.
    with gate.step("GoPro telemetry parser"):
        check_gopro_telemetry_parser(ingest_mode)

    # 1a. Source footage contract. Refuses dangling or unreadable
    # video files; reports, never refuses, files read from another filesystem
    # than ``staging_dir`` (a speed cost, see ``check_source_filesystem``).
    if staging_dir is not None:
        Path(staging_dir).mkdir(parents=True, exist_ok=True)
        with gate.step("source footage"):
            check_source_filesystem(
                Path(source_dir),
                Path(staging_dir),
                allow_cross_fs=allow_cross_fs_source,
                video_extensions=list(video_extensions) if video_extensions else None,
            )

    # 1b. Concat symlink alignment — stale symlinks point at a previous
    # source directory and cause Phase A to silently re-read the old
    # source. Phase A writes episode concats/symlinks to the ext4 staging
    # dir (state.stages_base), so scan there when set; fall back to
    # output_dir for legacy single-FS layouts. Operator deletes the stale
    # symlink or re-invokes with the original source path.
    _ingest_dir = Path(staging_dir) if staging_dir is not None else Path(output_dir)
    with gate.step("concat alignment"):
        check_concat_target_alignment(Path(source_dir), _ingest_dir)

    # 2. Find video files. ``video_extensions``
    # is the single source of truth for accepted extensions — populated
    # by callers from ``config.drive.video_extensions``. Falls back to
    # the historical ``.mp4`` / ``.MP4`` pair when unset so existing
    # call-sites (and tests) keep working without a config plumb.
    exts = list(video_extensions) if video_extensions else [".mp4", ".MP4"]
    exts_lower = {e.lower() for e in exts}
    mp4_files = sorted(
        f for f in Path(source_dir).iterdir() if f.is_file() and f.suffix.lower() in exts_lower
    )

    # 3. Validate input files. Small-but-valid clips are downgraded to
    # warnings (ingest drops sub-minimum clips), so they never abort here.
    file_warnings: list[str] = []
    file_errors = validate_input_files(
        mp4_files, allowed_extensions=tuple(exts_lower), warnings=file_warnings
    )
    with gate.step("input files"):
        if file_errors:
            raise RuntimeError(
                "Input file validation failed:\n" + "\n".join(f"  - {e}" for e in file_errors)
            )

    # 4. Disk space — output
    with gate.step("output disk"):
        check_disk_space(output_dir, min_free_gb=min_free_gb)

    # 5. Disk space — staging / intermediate cache (separate mount from
    # output in WSL drvfs setups). Estimate peak usage from source file
    # sizes so a 200 GB trip with only 40 GB cache free fails loud here
    # instead of mid-encode on day 2. The estimate uses the per-day
    # bounded cache — see ``estimate_peak_cache_gb``.
    if staging_dir is not None:
        with gate.step("staging disk"):
            staging_dir = Path(staging_dir)
            staging_dir.mkdir(parents=True, exist_ok=True)
            # Phase A writes the concatenated per-episode .mp4s to the staging
            # dir, where they PERSIST through scoring + the per-day master
            # encode. Peak ≈ all episode concats (~source size) + the busiest
            # day's master-encode working set (raw + master + work). Size at
            # source × 1.5 so the preflight gate AGREES with the per-day gate's
            # own ``day_source × 1.5`` formula (day_master_encoder) — a batch
            # that won't fit fails loud HERE instead of mid-encode on day N.
            # On a resume, subtract days whose masters are already encoded —
            # their raw episodes are purgeable, so they aren't part of the
            # remaining peak. A fresh run passes 0 and behaves as before.
            # ingest_complete: a resume with Phase A already done does NOT
            # re-create the per-episode concats (Phase A is skipped), so the whole
            # source×1.5 concat term overstates the real remaining peak (which is
            # just the downstream LF/short working set, comfortably above the
            # floor). Treat the full source as already-staged so the gate drops to
            # ``floor_gb`` — otherwise a Phase-A-complete resume on a disk that
            # held the prior run's outputs false-fails
            # (the gate demands the full concat term though no re-ingest would occur).
            effective_completed = completed_source_bytes
            if ingest_complete:
                try:
                    effective_completed = float(sum(f.stat().st_size for f in mp4_files))
                except OSError:
                    effective_completed = completed_source_bytes
            est_peak_gb = estimate_peak_cache_gb(
                mp4_files,
                safety_factor=1.5,
                floor_gb=min_staging_gb,
                completed_source_bytes=effective_completed,
                superlong_enabled=superlong_enabled,
            )
            usage = shutil.disk_usage(staging_dir)
            free_gb = usage.free / _GB
            if free_gb < est_peak_gb:
                total_source_gb = sum(f.stat().st_size for f in mp4_files) / _GB
                raise RuntimeError(
                    f"Insufficient cache space at {staging_dir}: "
                    f"{free_gb:.1f} GB free, need ~{est_peak_gb:.1f} GB "
                    f"(estimated peak from {total_source_gb:.1f} GB source). "
                    f"Free space in the intermediate cache volume, or set "
                    f"GMS_INTERMEDIATE_DIR to a volume with more headroom."
                )

            # 5a. WSL host-volume headroom. The ext4 staging FS can report
            # plenty free while the host VHDX expands into a near-full Windows
            # drive and crashes the host (ext4 can report hundreds of GB free
            # while the host drive is nearly full). The host must cover the VHDX growth beyond what
            # ext4 can already absorb (est_peak − current ext4 free), floored at
            # an absolute minimum so a brink-full host never passes. No-op off
            # WSL. See ``check_wsl_host_volume_space``.
            host_needed_gb = max(est_peak_gb - free_gb, _WSL_HOST_MIN_FREE_GB)
            check_wsl_host_volume_space(staging_dir, host_needed_gb)

    # 6. RAM
    with gate.step("RAM"):
        check_ram(min_free_gb=min_ram_gb)

    # 7. Tools
    if require_tools:
        with gate.step("tools"):
            check_tools(require_tools)

    # 7b. OST burn font. Every burned title (LF + shorts) renders with the
    # single canonical FONT_PATH; a missing font silently ships title-less
    # long-forms via the apply_ost stream-copy fallback, with no other gate to
    # catch it. Assert it upstream of any encode.
    with gate.step("OST font"):
        check_ost_font()

    # 7a. Weather decode path contract.
    # Verify the configured weather_hwaccel can actually deliver GPU
    # decode for any cuvid-mappable source. ``None`` (caller didn't pass
    # the param) skips the check — keeps existing test call-sites working
    # without a config plumb.
    if weather_hwaccel is not None:
        with gate.step("weather decode"):
            check_weather_decode_path(mp4_files, weather_hwaccel)

    # 7c. Blur inference provider (silent CPU fallback). Same
    # ``None``-skips contract as 7a; ``False`` (blur disabled / skipped) has
    # nothing to gate.
    if blur_enabled:
        with gate.step("blur provider"):
            check_blur_inference_provider()

    # 8. OST uniqueness (if titles available)
    if episode_titles is not None:
        with gate.step("OST uniqueness"):
            run_preflight_ost_check(episode_titles)

    # 9. SEO state-file ownership (cross-user drift). Only runs when caller
    # passes the SEO paths; tests / non-drive entry points can omit and skip
    # this check. ``GMS_SEO_OWNER_USER`` / ``GMS_SEO_OWNER_GROUP`` only name
    # the account in the repair hint.
    if seo_storage_path is not None and seo_project is not None:
        with gate.step("SEO ownership"):
            check_seo_ownership(seo_storage_path, seo_project)

    # 10. Drive credential / SOPS asset existence. Catch missing creds at
    # preflight instead of hours into Phase D. Each env var defaults to a file
    # under the config home but is overrideable; a blank value only skips
    # this existence check (the uploader still falls back to the default
    # file). The check itself is opt-in via ``check_credentials=True`` so
    # existing test entry points that don't exercise the Drive upload code
    # path stay green without mocking out the credential files.
    if check_credentials:
        with gate.step("credentials"):
            check_drive_credentials()

    if gate.found:
        raise PreflightBlockedError(gate.found)
    logger.info("Pre-flight: all %d checks passed.", len(gate.checked))
    return PreflightReport(checked=gate.checked, warnings=file_warnings)


def check_drive_credentials() -> None:
    """Verify resolved Drive credential paths exist on disk.

    Reads four env vars whose defaults (files under the config home) are the
    same ones ``gomoveshift.pipeline.uploader`` resolves, at call time, through
    ``gomoveshift.paths.gms_secret_path``:

    * ``GMS_DRIVE_CREDENTIALS_ENC`` — SOPS-encrypted OAuth client JSON
    * ``GMS_DRIVE_TOKEN_ENC`` — SOPS-encrypted refresh token JSON
    * ``GMS_AGE_KEY_FILE`` — age key used to decrypt the above
    * ``GMS_SOPS_CONFIG`` — sops config (creation_rules + recipients)

    A blank env value (``""``) only skips this existence check for that
    variable; the uploader treats a blank value as unset and then reads the
    default file under the config home. Otherwise a missing file raises
    ``RuntimeError`` matching the shape used by the other preflight checks
    so the orchestrator's existing error handling can surface it cleanly.
    """
    config_home = gms_config_home()
    spec = (
        ("GMS_DRIVE_CREDENTIALS_ENC", config_home / "drive_credentials.json.enc"),
        ("GMS_DRIVE_TOKEN_ENC", config_home / "drive_token.json.enc"),
        ("GMS_AGE_KEY_FILE", config_home / "age.key"),
        ("GMS_SOPS_CONFIG", config_home / ".sops.yaml"),
    )
    missing: list[str] = []
    for env_var, default in spec:
        raw = os.environ.get(env_var)
        if raw == "":
            # Blank value: skip this check (the uploader uses the default).
            continue
        # Same resolution as the uploader: strip whitespace, expand ``~``,
        # and treat a whitespace-only value as unset.
        resolved = Path(raw.strip()).expanduser() if raw and raw.strip() else default
        if not resolved.exists():
            missing.append(f"{env_var}={resolved}")
    if missing:
        raise RuntimeError(
            "Drive credential path(s) missing on disk: "
            + ", ".join(missing)
            + ". Set the env var to a valid path, or pass an empty string "
            "to disable the check."
        )


@dataclass(frozen=True)
class MlUpscaleCheck:
    ok: bool
    checked: bool
    reason: str = ""


def check_ml_upscale(profile_upscale: str, weights_path=None, providers=None) -> MlUpscaleCheck:
    """Probe the ML upscaler only when the profile asks for ``ml``. Never raises.

    A failure is advisory: stills fall back to the resampled frame, so this is
    for the run summary, not a gate. Call it where ``check_blur_inference_provider``
    runs, with ``profile.media.upscale``.
    """
    if profile_upscale != "ml":
        return MlUpscaleCheck(True, False)
    from . import quality

    problems = []
    try:
        quality.verify_ml_upscale_weights(
            weights_path if weights_path is not None else quality.ml_upscale_weights_path()
        )
    except Exception as exc:  # noqa: BLE001
        problems.append(f"weights: {exc}")
    try:
        if providers is None:
            providers = list(_import_onnxruntime().get_available_providers())
        if _CUDA_PROVIDER not in providers:
            problems.append(f"CUDA provider unavailable (providers: {', '.join(providers)})")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"CUDA provider unavailable: {exc}")
    return MlUpscaleCheck(not problems, True, "; ".join(problems))
