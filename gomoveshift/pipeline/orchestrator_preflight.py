"""Preflight prompt + run_preflight invocation, extracted from
DrivePipelineOrchestrator."""

from __future__ import annotations

import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional

from ..models import DrivePipelineResult
from .day_grouper import group_episodes_by_day
from .ingest import create_ingest_plan
from .inserts import describe_drive_inserts
from .preflight import PreflightBlockedError, format_blockers, run_preflight
from .preflight_plan import print_plan

logger = logging.getLogger("gomoveshift.pipeline.drive_orchestrator")


def _phase_a_done(state) -> bool:
    """True when a resumed checkpoint says Phase A already completed.

    Read from ``state.phases_completed`` (the persisted resume checkpoint),
    NOT the orchestrator's in-process ``stages_completed`` (which is empty at
    preflight — no phase has run yet this process). When Phase A is done the
    per-episode concats already exist and will NOT be re-created, so the
    preflight staging estimate must not demand space for a re-ingest.
    """
    pc = getattr(state, "phases_completed", None)
    if pc is None and isinstance(state, dict):
        pc = state.get("phases_completed")
    return "phase_a" in (pc or [])


def _episode_titles(state) -> Optional[list]:
    """Return ``state.episode_locations`` (attr or dict key) or ``None``.

    Gate-bypass hardening: the canonical ``run_preflight``
    call below never exercised step 8's OST uniqueness/validity check because
    ``episode_titles`` was never passed — leaving the check solely on the
    resume-skippable phase_a branch. On a resume with Phase A done the titles
    already sit in state; pass them so the env preflight re-runs the gate.
    A fresh run (no titles yet) returns ``None`` and the phase_a-side gate
    (``run_phase_a_ost_gate``) covers it after derivation.
    """
    titles = getattr(state, "episode_locations", None)
    if titles is None and isinstance(state, dict):
        titles = state.get("episode_locations")
    return list(titles) if titles else None


def _completed_source_bytes(state) -> float:
    """Sum recorded ``source_bytes`` across COMPLETED days.

    Reads ``state.per_day_completed`` (dataclass attr or dict key). Each
    record carries ``source_bytes`` once ``build_master_record`` ran for
    that day; an older record or an ``auto_populate_per_day_completed``
    recovery marker (reconstructed from disk, no byte count) simply omits it
    and contributes 0 — so the resume-aware preflight subtraction degrades to
    the conservative full estimate rather than over-subtracting.
    """
    per_day = getattr(state, "per_day_completed", None)
    if per_day is None and isinstance(state, dict):
        per_day = state.get("per_day_completed")
    if not per_day:
        return 0.0
    total = 0.0
    for rec in per_day.values():
        if isinstance(rec, dict):
            try:
                total += float(rec.get("source_bytes", 0) or 0)
            except (TypeError, ValueError):
                continue
    return total


def _source_video_files(source_dir: Path, video_extensions) -> list[Path]:
    """Sorted video files directly in ``source_dir`` (same extension rule as
    ``preflight.run_preflight`` step 2; ``.mp4``/``.MP4`` when unset)."""
    exts = {e.lower() for e in video_extensions} if video_extensions else {".mp4"}
    try:
        return sorted(
            p for p in Path(source_dir).iterdir() if p.is_file() and p.suffix.lower() in exts
        )
    except OSError:
        return []


def _calibration_for_plan(*, dry_run: bool, source_dir: Path, stages_dir: Path, config):
    """Record the plan's ETA is based on. A dry run only reads a saved record
    (it must not spend a minute encoding); a real run measures once if none."""
    from .calibration import ensure_calibrated, load_calibration

    if dry_run:
        return load_calibration()
    return ensure_calibrated(
        config,
        _source_video_files(source_dir, config.drive.video_extensions),
        blur=config.drive.blur_enabled and not config.drive.skip_blur,
        workdir=stages_dir / "_calibration",
    )


def _open_controlling_tty():
    """The operator's terminal regardless of redirection, or ``None``."""
    try:
        return open("/dev/tty", "w")  # noqa: SIM115 - closed by _ask
    except OSError:
        return None


def _prompt_channel():
    """Where 'Proceed? [y/N]' can actually be SEEN, or ``None``.

    stdin being a tty is not enough: with stdout and stderr redirected to a
    log, ``input()`` writes the prompt into the log and the run waits invisibly
    (the operator never sees it). Order: stdout tty, stderr tty, controlling tty.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        try:
            if stream is not None and stream.isatty():
                return stream
        except (AttributeError, ValueError):
            pass
    return _open_controlling_tty()


def _ask(channel) -> str:
    prompt = "Proceed? [y/N] "
    if channel is sys.stdout:
        return input(prompt)
    channel.write(prompt)
    channel.flush()
    try:
        return sys.stdin.readline()
    finally:
        if channel is not sys.stderr:
            channel.close()


def day_under_part_min_message(config, source_dir: Path) -> Optional[str]:
    """Upstream gate: a full_drive day whose RAW footage is under the part minimum.

    Kept (stop-masked) footage can never exceed a day's raw footage, so a day
    whose summed raw source duration is under ``full_drive_part_min_sec`` is
    certain to be refused by ``plan_full_drive_parts`` -- after the whole of
    Phase A has run. ``create_ingest_plan`` is a no-file-ops scan (durations
    and container ``creation_time`` only), so the refusal is knowable here.
    Inactive unless both part bounds are set. The planner's stop-masked check
    is unchanged. Returns the refusal message, or ``None``.

    Before ingest there is no GPX, so days key on ``creation_time`` (the
    camera clock), which can sit a calendar day off the GPX-local day Phase A
    later uses. To never refuse a day that the real grouping would keep, a day
    is refused only when it AND its adjacent calendar days together are still
    under the minimum.
    """
    drive = getattr(config, "drive", None)
    part_min = getattr(drive, "full_drive_part_min_sec", None)
    if part_min is None or getattr(drive, "full_drive_part_max_sec", None) is None:
        return None
    source_dir = Path(source_dir)
    if not source_dir.is_dir():
        return None  # run_preflight names the missing source dir
    plan = create_ingest_plan(
        source_dir,
        mode=drive.ingest_mode,
        extensions=drive.video_extensions,
    )
    raw = {b.date: b.total_duration for b in group_episodes_by_day(plan)}
    for day in sorted(raw):
        if raw[day] <= 0:
            continue
        d = date.fromisoformat(day)
        near = sum(raw.get((d + timedelta(days=k)).isoformat(), 0.0) for k in (-1, 0, 1))
        if near < part_min:
            return (
                f"Day {day}: raw source footage {raw[day]:.0f}s is under "
                f"full_drive_part_min_sec={part_min}s, so the day can never fill a "
                "part. Footage is never padded, repeated or joined to another day. "
                "Refused before ingest."
            )
    return None


def _readiness_blockers(*, state, stages_dir: Path, source_dir: Path, project_dir: Path, config):
    """Run ``run_preflight`` for a real run; return its blockers (empty = ready).

    Logs the ``READY TO RENDER`` report when clear. ``cleanup_stale_staging``
    runs first so the staging-space check measures what the run will have; it
    only removes other projects' staging dirs older than its age bound and
    never one holding outputs, so running it before the operator prompt is safe.
    run_preflight runs UNCONDITIONALLY: on resume, ``resume_invariants`` also checks disk/RAM via
    the same helpers; this remains the canonical entry because it also
    validates source_dir + input files + SEO ownership.
    """
    from .preflight import cleanup_stale_staging

    try:
        cleanup_stale_staging(stages_dir.parent.parent, protect={stages_dir.parent})
        report = run_preflight(
            source_dir=source_dir,
            output_dir=project_dir,
            staging_dir=stages_dir,
            # Gate-bypass hardening: resumed runs carry
            # episode_locations in state — re-run the OST check here since
            # the phase_a execution branch is skipped on resume.
            episode_titles=_episode_titles(state),
            seo_storage_path=Path(config.seo.storage_path),
            seo_project="gomoveshift",
            video_extensions=config.drive.video_extensions,
            allow_cross_fs_source=config.drive.allow_cross_fs_source,
            weather_hwaccel=config.drive.weather_hwaccel,
            # Refuse to start a blur run whose detectors would fall
            # back to CPU because the GPU onnxruntime build was clobbered.
            blur_enabled=config.drive.blur_enabled and not config.drive.skip_blur,
            # Surface missing Drive credentials at preflight instead of hours
            # into Phase D.
            check_credentials=True,
            # Resume-aware disk gate — don't demand space for days whose
            # masters are already encoded (their raw episodes are purgeable).
            completed_source_bytes=_completed_source_bytes(state),
            # A resume with Phase A already done won't re-ingest (no re-concat),
            # so the source×1.5 concat term doesn't apply — drop the gate to its
            # floor rather than false-fail on a disk holding the prior outputs.
            ingest_complete=_phase_a_done(state),
            # Same ``state.superlong_enabled`` sentinel read as
            # ``preflight_plan._estimate_wall_clock`` — by preflight time
            # ``run()`` has already resolved this run's CLI intent onto
            # state, on both a fresh run and a resume.
            superlong_enabled=bool(getattr(state, "superlong_enabled", None)),
            # A GoPro run must be able to read its own GPS.
            ingest_mode=config.drive.ingest_mode,
        )
    except PreflightBlockedError as e:
        return list(e.blockers)
    except RuntimeError as e:
        return [("pre-flight", str(e))]
    checked = list(getattr(report, "checked", None) or [])
    warnings = list(getattr(report, "warnings", None) or [])
    logger.info(
        "READY TO RENDER: %d pre-flight checks passed before any encode%s%s",
        len(checked),
        f" ({', '.join(checked)})" if checked else "",
        f"; {len(warnings)} warning(s): " + "; ".join(warnings) if warnings else "",
    )
    return []


def run_preflight_and_prompt(
    *,
    state,
    stages_dir: Path,
    dry_run: bool,
    force: bool,
    source_dir: Path,
    project_dir: Path,
    start_time: float,
    stages_completed: list,
    config,
    emit_summary: Callable[..., None],
    labels_only: bool = False,
) -> Optional[DrivePipelineResult]:
    """Render the preflight execution plan, prompt the operator, and run
    ``run_preflight``. Returns a non-None ``DrivePipelineResult`` only on
    an early-exit path (dry-run, declined prompt, preflight failure); the
    caller short-circuits ``run()`` in that case. Returns ``None`` on the
    happy path so phase work proceeds.

    Prompting policy:

    * ``--force`` → no prompt, proceed.
    * tty stdin but stdout AND stderr redirected → the prompt goes to the
      controlling terminal (/dev/tty); with none, REFUSED by name.
    * non-tty without ``--force`` → REFUSED (named message, no prompt): nobody
      can answer, and a detached run used to sit at a prompt for an hour (or
      silently proceed). Pass ``--force`` to run unattended, ``--dry-run`` to
      only see the plan.
    * tty + no ``--force`` → ``Proceed? [y/N]``; non-yes aborts.
    """
    # Render-readiness gate: every cheap prerequisite is checked, and every
    # blocker reported together, BEFORE the calibration encode (the run's
    # first GPU work). A refusal on a missing parser or an unreadable source
    # must not come after a calibration encode, or a full Phase A.
    # ``--dry-run`` keeps its plan-only, side-effect-free path: it reads a
    # saved calibration and never runs the gate.
    blockers: list[tuple[str, str]] = []
    if not _phase_a_done(state):
        refusal = day_under_part_min_message(config, source_dir)
        if refusal:
            blockers.append(("raw duration", refusal))
    if not dry_run:
        blockers.extend(
            _readiness_blockers(
                state=state,
                stages_dir=stages_dir,
                source_dir=source_dir,
                project_dir=project_dir,
                config=config,
            )
        )
    if blockers:
        report = format_blockers(blockers)
        logger.error("Pre-flight check failed: %s", report)
        emit_summary(
            source_dir=source_dir,
            state=state,
            stages_completed=stages_completed,
            phase_defs_total=0,
            total_time=time.time() - start_time,
        )
        return DrivePipelineResult(
            success=False,
            source_dir=source_dir,
            output_dir=project_dir,
            drive_plan=None,
            regular_lf_paths=[],
            shorts_per_lf={},
            stages_completed=[],
            errors=[f"Pre-flight failed: {report}"],
            timings={},
        )
    # A --label-dry-run / --list-lfs run never encodes (and in full_drive never
    # takes the GPU lock), so it only reads a saved record, like --dry-run.
    record = _calibration_for_plan(
        dry_run=dry_run or labels_only, source_dir=source_dir, stages_dir=stages_dir, config=config
    )
    fast_skipped = getattr(getattr(config, "drive", None), "fast_skipped", None)
    # No record -> print_plan's own default (load; none -> reference model).
    plan_kwargs: dict = {}
    if record is not None:
        plan_kwargs["calibration"] = record
    if fast_skipped:
        plan_kwargs["fast_skipped"] = fast_skipped
    insert_lines = describe_drive_inserts(config.drive)
    if insert_lines:
        plan_kwargs["insert_lines"] = insert_lines
    print_plan(state, stages_dir, **plan_kwargs)
    if dry_run:
        logger.info("--dry-run: plan rendered, exiting before any phase work")
        emit_summary(
            source_dir=source_dir,
            state=state,
            stages_completed=stages_completed,
            phase_defs_total=0,
            total_time=time.time() - start_time,
        )
        return DrivePipelineResult(
            success=True,
            source_dir=source_dir,
            output_dir=project_dir,
            drive_plan=None,
            regular_lf_paths=[],
            shorts_per_lf={},
            stages_completed=[],
            errors=[],
            timings={},
        )
    if not force:
        try:
            is_tty = sys.stdin.isatty()
        except (AttributeError, ValueError):
            is_tty = False
        channel = _prompt_channel() if is_tty else None
        if channel is None:
            msg = (
                "PREFLIGHT-NO-OPERATOR: refusing to start -- no operator can see and answer "
                "'Proceed? [y/N]' (stdin is not a terminal, or no visible terminal exists "
                "for the prompt: stdout/stderr are redirected and /dev/tty is unavailable). Re-run with --force to run "
                "unattended, or --dry-run to only print the plan."
            )
            logger.error(msg)
            return DrivePipelineResult(
                success=False,
                source_dir=source_dir,
                output_dir=project_dir,
                drive_plan=None,
                regular_lf_paths=[],
                shorts_per_lf={},
                stages_completed=[],
                errors=[msg],
                timings={},
            )
        if channel is not None:
            try:
                answer = _ask(channel).strip().lower()
            except (EOFError, KeyboardInterrupt):
                answer = ""
            if answer not in {"y", "yes"}:
                logger.info(
                    "Preflight declined (answer=%r); exiting without running phases",
                    answer,
                )
                emit_summary(
                    source_dir=source_dir,
                    state=state,
                    stages_completed=stages_completed,
                    phase_defs_total=0,
                    total_time=time.time() - start_time,
                )
                return DrivePipelineResult(
                    success=True,
                    source_dir=source_dir,
                    output_dir=project_dir,
                    drive_plan=None,
                    regular_lf_paths=[],
                    shorts_per_lf={},
                    stages_completed=[],
                    errors=[],
                    timings={},
                )

    return None
