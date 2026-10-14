"""Handler for the ``drive`` CLI command."""

import json
import logging
import os
import sys
from pathlib import Path

from ..logging_setup import configure_logging
from ..tmux import is_in_tmux, launch_in_tmux, require_supervision, tmux_available

logger = logging.getLogger(__name__)


def resolve_youtube_upload(args) -> bool:
    """Resolve effective YouTube-upload flag.

    Precedence: --no-youtube-upload → False; --youtube-upload → True;
    --publish set → True (implied); else False (preserves smoke flow).
    """
    if getattr(args, "no_youtube_upload", None):
        return False
    if getattr(args, "youtube_upload", None):
        return True
    if getattr(args, "publish", None):
        return True
    return False


def _stages_base(args, config) -> Path:
    """Where the run's ``pipeline_state.json`` lives; the pure twin of
    ``orchestrator._resolve_paths``, which also mkdirs ``output_base`` and the project dir."""
    from ...paths import gms_intermediate_dir

    output_base = Path(args.output_dir)
    project_name = Path(args.directory).stem.replace(" ", "_").lower()
    nested = output_base.name.lower() == project_name
    return gms_intermediate_dir(
        output_base.parent if nested else output_base,
        project_name,
        intermediate_folder=config.paths.intermediate_folder,
    )


def _region_provenance(args, config) -> str:
    """Effective region + source for the startup banner.

    ``--region`` is an argparse sentinel; on a resume without it the banner
    used to print ``Region: None`` while ``run()`` restored the saved region.
    Best-effort checkpoint peek for provenance only — never raises, never
    decides anything (``run()`` resolves the real value after the merge).
    """
    if args.region:
        return f"{args.region} (cli)"
    try:
        stages_base = _stages_base(args, config)
        saved = json.loads((stages_base / "pipeline_state.json").read_text()).get("region")
        if saved:
            return f"{saved} (resumed state)"
    except Exception:  # noqa: BLE001 — display-only; run() owns resolution
        pass
    return "ireland (default — no --region, no resumable state)"


def _effective_region(args, state_file: Path):
    """The run's resolved region: ``--region`` wins, else the value the
    orchestrator persisted to state (its authoritative resolution)."""
    if getattr(args, "region", None):
        return args.region
    try:
        return json.loads(state_file.read_text()).get("region")
    except (OSError, ValueError):
        return None


def _has_live_upload(uploads) -> bool:
    """True when saved ``youtube_uploads`` holds any LF with a YouTube video id."""
    if not isinstance(uploads, dict):
        return False
    return any(isinstance(r, dict) and r.get("video_id") for r in uploads.values())


def _launch_anchor_refusal(args, config, publish_day) -> str | None:
    """Refuse a bad LF date BEFORE any phase runs (a wrong-weekday anchor).

    The effective anchor is ``--lf-publish-date``, else the ``lf_publish_date`` saved in ``pipeline_state.json``
    (it survives a resume), else ``--start-date``; the effective day is ``--publish``/``--lf-day`` or the saved
    one. Off the run's LF day, past its slot time, or booked by another project -> the
    message (naming the next free date). An audit-only run keeps only the weekday check.
    With no anchor the run's day must still be one of Waypost's for its region
    (``seo_slots.lf_day_refusal``). An unreadable Waypost cadence refuses with
    ``CADENCE_SOURCE_UNREADABLE``. Silent when ``args`` carries no real values (a partial
    namespace).
    """
    anchor = getattr(args, "lf_publish_date", None)
    anchor = anchor if isinstance(anchor, str) and anchor else None
    saved: dict = {}
    source_dir = Path(args.directory)
    try:
        stages_base = _stages_base(args, config)
        if not getattr(args, "fresh", False):
            saved = json.loads((Path(stages_base) / "pipeline_state.json").read_text())
    except (OSError, ValueError, TypeError, AttributeError):
        saved = {}
    if not isinstance(saved, dict):
        saved = {}
    resumed = False
    flag = "--lf-publish-date"
    if anchor is None:
        # Phase D's order: --lf-publish-date, then the saved lf_publish_date (its backstop
        # keeps it over --start-date on a resume), then --start-date, taken as LF1 exactly
        # as given. A saved start_date is not an anchor: Phase D rebases it when stale.
        saved_anchor = saved.get("lf_publish_date")
        if isinstance(saved_anchor, str) and saved_anchor:
            anchor = saved_anchor
            # A saved anchor of a drive with a live LF is history, not a plan: only its
            # weekday is checked (Phase D skips its booked check on such a resume too).
            resumed = _has_live_upload(saved.get("youtube_uploads"))
        else:
            arg_start = getattr(args, "start_date", None)
            if isinstance(arg_start, str) and arg_start:
                anchor, flag = arg_start, "--start-date"
    day = publish_day if isinstance(publish_day, str) and publish_day else saved.get("publish_day")
    region = getattr(args, "region", None)
    region = region if isinstance(region, str) and region else saved.get("region")
    from ...pipeline.seo_slots import CadenceSourceUnreadableError, lf_day_refusal

    if anchor is None:
        # Nothing booked yet, but the day must still be one of Waypost's for the region.
        try:
            return lf_day_refusal(config, region if isinstance(region, str) else None, day)
        except CadenceSourceUnreadableError as exc:
            return str(exc)
    # An audit-only run (--skip-upload with --no-youtube-upload) books nothing, so a past or
    # booked date cannot hurt: only the weekday is checked, like a resumed live drive.
    if getattr(args, "skip_upload", None) and getattr(args, "no_youtube_upload", None):
        resumed = True
    from ...pipeline.content_ledger import ContentLedger
    from ...pipeline.seo_slots import anchor_refusal

    ledger_path = Path(config.seo.storage_path) / ".content_ledger.json"
    ledger = ContentLedger.load(ledger_path)
    stem = source_dir.stem
    names = list(dict.fromkeys([stem.replace(" ", "_").lower(), stem]))
    try:
        return anchor_refusal(
            config,
            ledger,
            anchor=anchor,
            publish_day=day,
            region=region if isinstance(region, str) else None,
            project_names=names,
            resumed=resumed,
            flag=flag,
        )
    except CadenceSourceUnreadableError as exc:
        return str(exc)


def _run_date_from_state(state: dict) -> str | None:
    """The run's recording day (``YYYY-MM-DD``), for date-based journey
    resolution — never today's wall clock. Day-first: every LF in a drive
    run shares one calendar day (see the drive-pipeline invariant), so any
    ``per_lf_recording_date`` value suffices; sorting picks deterministically
    if state ever carries more than one."""
    rec_dates = state.get("per_lf_recording_date")
    if not isinstance(rec_dates, dict) or not rec_dates:
        return None
    values = sorted({str(v) for v in rec_dates.values() if v})
    return values[0] if values else None


def _maybe_auto_publish_journal(args, orchestrator, config) -> None:
    """Post-Phase-D hook: merge the run's uploads into the site journey.

    Default-ON, but skipped when YouTube upload is off for the run (nothing
    was uploaded to merge). The journey slug resolves from an explicit
    ``--auto-publish-journal SLUG`` (wins), else
    ``config.drive.active_journey_by_region`` keyed by the run's region
    (wins over date inference), else the journey whose ``trip.start_date``..
    ``end_date`` range contains the run's recording day — refusing on
    cross-trip overlap rather than guessing. The site path resolves from
    ``--site-path``, else ``GMS_SITE_PATH``. ``--no-auto-publish-journal``
    force-disables.

    When a real publish run can't resolve a slug or site path, log a loud
    WARNING (the site silently going stale is the exact failure we're fixing) —
    but never raise: YouTube uploads are the irreversible work; the merge is
    recoverable via the standalone command / replay cron.
    """
    if getattr(args, "no_auto_publish_journal", False):
        return
    if not resolve_youtube_upload(args):
        logger.info(
            "auto-publish-journal: YouTube upload was off for this run; nothing to merge "
            "(standalone: gomoveshift publish-journal <state>)"
        )
        return

    # Whole body guarded: a recoverable side-quest must NEVER break the drive
    # batch (YouTube uploads are the irreversible work). _resolve_paths, the
    # state read, and the merge all run under this guard.
    try:
        project_dir, stages_base = orchestrator._resolve_paths(args.directory, args.output_dir)
        state_file = stages_base / "pipeline_state.json"

        slug = getattr(args, "auto_publish_journal", None)
        region = None
        if not slug:
            region = _effective_region(args, state_file)
            mapping = (
                getattr(getattr(config, "drive", None), "active_journey_by_region", None) or {}
            )
            slug = mapping.get(region) if region else None

        site_path = getattr(args, "site_path", None) or os.environ.get("GMS_SITE_PATH")

        if not slug and site_path and state_file.is_file():
            from ...pipeline.site_sync.journey_by_date import (
                AmbiguousJourneyError,
                JourneyAlreadyExistsError,
                create_draft_journey,
                resolve_journey_by_date,
            )

            try:
                state_for_date = json.loads(state_file.read_text())
            except (OSError, ValueError):
                state_for_date = {}
            run_date = _run_date_from_state(state_for_date)
            if run_date:
                try:
                    slug = resolve_journey_by_date(Path(site_path), run_date)
                except AmbiguousJourneyError as exc:
                    logger.error(
                        "Journal auto-publish REFUSED — the site will NOT update for this run: %s",
                        exc,
                    )
                    return
                if not slug and region:
                    try:
                        slug = create_draft_journey(Path(site_path), region, run_date)
                    except JourneyAlreadyExistsError as exc:
                        logger.error(
                            "Journal auto-publish REFUSED — the site will NOT update for "
                            "this run: %s",
                            exc,
                        )
                        return
                    logger.info(
                        "auto-publish-journal: no journey matched run date %s; created "
                        "draft journey %s (draft: true — review before it goes live).",
                        run_date,
                        slug,
                    )

        if not slug or not site_path:
            # Loud only on a real publish run; a smoke/non-publish run stays quiet.
            if getattr(args, "publish", None):
                reasons = []
                if not slug:
                    reasons.append(
                        f"no journey for region {region!r} (set "
                        "config.drive.active_journey_by_region or pass --auto-publish-journal SLUG)"
                    )
                if not site_path:
                    reasons.append("no site path (pass --site-path or set GMS_SITE_PATH)")
                logger.warning(
                    "Journal auto-publish skipped — the site will NOT update for this run: %s. "
                    "After resolving, run: gomoveshift publish-journal %s --journey <slug> "
                    "--site-path <site> --apply",
                    "; ".join(reasons),
                    state_file,
                )
            return

        if not state_file.is_file():
            logger.error(
                "auto-publish-journal: state file %s missing after run; skipping.", state_file
            )
            return

        state = json.loads(state_file.read_text())

        # The routing written below is authoritative forever, so a wrong or
        # stale --region would durably file (e.g.) a USA LF under the Ireland
        # journey. Cross-check each LF's canonical ResolvedLocation region against
        # the run's region and REFUSE the whole merge on any positive disagreement,
        # rather than routing silently. No-op for GPS-less / legacy LFs.
        from ...pipeline.site_sync.journey_reconcile import (
            JourneyRegionMismatchError,
            assert_region_consistency,
        )

        try:
            assert_region_consistency(state, getattr(args, "region", None) or state.get("region"))
        except JourneyRegionMismatchError as exc:
            logger.error(
                "Journal auto-publish REFUSED — the site will NOT update for this run: %s",
                exc,
            )
            return

        from .publish_journal import run_publish_journal

        rc = run_publish_journal(
            state_path=state_file,
            site_path=Path(site_path).resolve(),
            journey_slug=slug,
            apply=True,
            allow_day_create=False,
            # Route the deferral sidecar to the durable drive output dir, where
            # the replay cron's --scan-root looks — NOT the ephemeral ext4
            # staging dir that holds pipeline_state.json (which the cron never
            # scans). This is what lets future-dated uploads reach the site.
            pending_dir=project_dir,
        )
        if rc != 0:
            logger.error(
                "auto-publish-journal: publish-journal exited %d; run "
                "`gomoveshift publish-journal %s --journey %s --site-path %s` to retry.",
                rc,
                state_file,
                slug,
                site_path,
            )

        # Record this drive's LF routing (id -> journey + recording date) so the
        # channel-reconciliation backstop (`sync-site --journeys`) can re-embed
        # or heal these ids even if a per-drive merge ever misses one. Runs
        # regardless of the merge rc (routing is idempotent and independently
        # useful); a failure here never fails the batch.
        from ...pipeline.site_sync.journey_reconcile import (
            ROUTING_FILENAME,
            merge_routing,
            routing_from_state,
        )

        routes = routing_from_state(state, slug)
        if routes:
            merge_routing(Path(site_path).resolve() / "data" / ROUTING_FILENAME, routes)
            logger.info("auto-route: recorded %d LF route(s) -> journey %s", len(routes), slug)
    except Exception:  # never fail the drive batch on a recoverable side-quest
        logger.exception("auto-publish-journal crashed; uploads succeeded, journal not updated.")


def apply_picture_flags(config, args) -> None:
    """Merge ``--stabilize`` / ``--deflicker`` onto ``config.drive``.

    ``--fast`` wins over everything: it forces ``no_sharpen``, clears a saved
    ``stabilize`` / ``deflicker`` opt-in, and records each dropped step in
    ``config.drive.fast_skipped`` for the preflight plan.

    Sentinel rule: the flags default to ``None`` ("not supplied"); only a
    supplied flag overrides, so a value already on the config survives a run
    that omits the flag.
    """
    if getattr(args, "stabilize", None) is True:
        config.drive.stabilize = True
    if getattr(args, "deflicker", None) is True:
        config.drive.deflicker = True
    config.drive.fast_skipped = []
    if getattr(args, "fast", False):
        if not config.drive.no_sharpen:
            config.drive.fast_skipped.append("sharpen")
        config.drive.no_sharpen = True
        for step in ("stabilize", "deflicker"):
            if getattr(config.drive, step):
                config.drive.fast_skipped.append(step)
                setattr(config.drive, step, False)


def cmd_drive(args, config):
    """Run Drive Pipeline on a GoPro directory."""
    from ...config import ConfigError
    from ...pipeline.drive_orchestrator import DrivePipelineOrchestrator
    from ...pipeline.exceptions import (
        CorruptStateError,
        PhaseQualityError,
        PipelineValidationError,
        SidecarIntegrityError,
    )

    # A real drive run must not silently default/skip the publish schedule.
    # ``--publish`` stays optional at argparse level (no cross-flag
    # ``required``), so refuse here, before any Phase A work, when neither
    # schedule flag was given. Discovery-only runs (--list-lfs/--label-dry-run) do no encode/upload and are exempt.
    if (
        getattr(args, "publish", None) is None
        and getattr(args, "lf_day", None) is None
        and not getattr(args, "list_lfs", False)
        and not getattr(args, "label_dry_run", False)
    ):
        raise SystemExit(
            "error: drive requires --publish or --lf-day (schedule-critical) "
            "— pass one explicitly, or use --list-lfs/--label-dry-run for a "
            "discovery-only run."
        )

    # --day-gpx slices a whole-day GPX using the camera clock offset; that
    # offset is operator knowledge and is never defaulted.
    if getattr(args, "day_gpx", None) and getattr(args, "camera_utc_offset_hours", None) is None:
        raise SystemExit(
            "error: --day-gpx requires --camera-utc-offset-hours N "
            "(camera clock UTC offset; US Eastern DST is -4) — never guessed."
        )

    # Interim guard: ``--skip-stop-verify`` is still a NO-OP on drive runs —
    # its only consumer is the single-video stop-removal stage, which the drive path
    # doesn't run (drive stop handling is the per-day stop mask). A no-op *safety*
    # flag is worse than none (it implies a verification step was deliberately
    # skipped when nothing happened), so reject at entry. ``--skip-blur`` is NO LONGER
    # here: privacy blur IS wired into the drive day master (encode_day_master →
    # blur_gate), so the flag is functional — set on config.drive below.
    # `is True` (not mere truthiness): argparse store_true sets exactly True when
    # the flag is passed, so this fires only on real operator intent — never on a
    # MagicMock args attribute (Mock is not True), keeping existing cmd_drive mock
    # tests unaffected.
    no_op_flags = [
        flag
        for flag, attr in (("--skip-stop-verify", "skip_stop_verify"),)
        if getattr(args, attr, False) is True
    ]
    if no_op_flags:
        print(
            f"error: {', '.join(no_op_flags)} not supported on drive runs — "
            "stop-removal verification is not wired into the drive path, so the flag "
            "is a silent no-op (a no-op safety flag is worse than none). Remove it.",
            file=sys.stderr,
        )
        return 2

    # Fail-fast SEO config validation. A missing SEO service checkout
    # used to surface mid-Phase D after hours of encode; this gate moves the
    # failure to startup. Set ``seo.skip_validation: true`` in config.yaml to
    # bypass for dry-run / test invocations that don't touch SEO. Runs BEFORE
    # tmux re-exec so operators see the message in their own shell.
    try:
        config.seo.validate_paths_exist()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # ENFORCE: Long-running pipelines MUST run in tmux for persistence
    if not is_in_tmux():
        if args.no_tmux:
            # --no-tmux asserts "I am already supervised"; the claim is
            # verified, never taken on trust. See cli/tmux.require_supervision.
            unsupervised = require_supervision("drive pipeline")
            if unsupervised is not None:
                return unsupervised
        elif tmux_available():
            # Re-exec inside a DETACHED tmux pane. That pane is a tty, so the
            # orchestrator's preflight "Proceed? [y/N]" would block INVISIBLY
            # there (no operator is watching the detached pane) right after the
            # launcher prints "Pipeline started". Append --force so the detached
            # run skips the prompt; the operator reviews up front via
            # --dry-run / --label-dry-run. See preflight_plan.py.
            reexec_argv = list(sys.argv)
            if "--force" not in reexec_argv:
                reexec_argv.append("--force")
            return launch_in_tmux(reexec_argv, session_name="gms-drive")
        else:
            print(
                "ERROR: tmux is required for long-running pipelines but is not available.",
                file=sys.stderr,
            )
            return 1

    # Persist a drive.log artifact
    # inside the project dir so operators have a durable log after the
    # tty closes. Mirrors cmd_pipeline. project_dir is computed
    # the same way the orchestrator computes it (output_dir / <project>
    # unless output_dir already ends with the project name).
    source_dir = Path(args.directory)
    # --dry-run prints the preflight plan and exits before any ingest, so it
    # must not write sidecars. --list-lfs / --label-dry-run run Phase A, which
    # reads the sidecars, so they still slice (idempotent).
    if getattr(args, "day_gpx", None) and not getattr(args, "dry_run", False):
        from ...pipeline.ingest.day_gpx_slicer import DayGpxSliceError, slice_day_gpx

        try:
            slice_day_gpx(Path(args.day_gpx), source_dir, args.camera_utc_offset_hours)
        except DayGpxSliceError as exc:
            print(f"error: --day-gpx: {exc}", file=sys.stderr)
            return 2
    output_base = Path(args.output_dir)
    project_name = source_dir.stem.replace(" ", "_").lower()
    if output_base.name.lower() == project_name:
        project_dir = output_base
    else:
        project_dir = output_base / project_name
    project_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(
        log_file=project_dir / "drive.log",
        json_log=getattr(args, "log_json", False),
    )

    print("=" * 60)
    print("  GoMoveShift Drive Pipeline")
    print("=" * 60)
    print(f"  Source:   {args.directory}")
    print(f"  Region:   {_region_provenance(args, config)}")
    print(f"  Publish:  {args.publish or '(earliest configured long-form day)'}")
    if args.location:
        print(f"  Location: {args.location}")
    if args.start_date:
        print(f"  Start:    {args.start_date}")
    if args.shorts_per_day:
        print(f"  Shorts/day: {args.shorts_per_day}")
    if args.shorts_target:
        print(f"  Shorts/LF:  {args.shorts_target}")
    if args.shorts_gap:
        print(f"  Shorts gap: {args.shorts_gap}s")
    print("=" * 60)

    # Reject ALL-overrides-equal-to-location at the
    # earliest possible moment. The parser already validated each KV's
    # shape via ``_per_lf_location_kv``; this is the cross-flag rule.
    # We bail before any tmux / phase work to keep the failure cheap.
    from ..parser import _validate_per_lf_location_args

    try:
        _validate_per_lf_location_args(args)
    except Exception as exc:  # argparse.ArgumentTypeError or similar
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Reduce list-of-tuples → dict so downstream consumers see a
    # ``{lf_index: label}`` shape. Last-write-wins on duplicate indices
    # (operator passes ``--per-lf-location 1=A --per-lf-location 1=B`` →
    # B wins; cheap predictable behavior).
    per_lf_overrides: dict[int, str] = {}
    for idx, label in getattr(args, "per_lf_locations", None) or []:
        per_lf_overrides[int(idx)] = label

    config.drive.preview = args.preview
    config.drive.no_sharpen = bool(getattr(args, "no_sharpen", False))
    config.drive.no_denoise = bool(getattr(args, "no_denoise", False))
    config.drive.trail = getattr(args, "trail", None)
    config.drive.trail_profile = getattr(args, "trail_profile", "") or ""
    apply_picture_flags(config, args)
    # Refuse upstream of any work when this ffmpeg cannot do what was asked.
    want_stabilize = config.drive.stabilize is True
    want_deflicker = config.drive.deflicker is True
    if want_stabilize or want_deflicker:
        from ...pipeline.preflight import check_picture_filters

        try:
            check_picture_filters(stabilize=want_stabilize, deflicker=want_deflicker)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    # CLI override beats config.
    _ingest_mode = getattr(args, "ingest_mode", None)
    if _ingest_mode in ("gopro", "generic"):
        config.drive.ingest_mode = _ingest_mode
    # CLI override beats config for LF
    # selection mode. Sentinel-None means "not supplied", any other value
    # is operator intent.
    _lf_mode = getattr(args, "lf_mode", None)
    if _lf_mode in ("best_moments", "full_drive"):
        config.drive.lf_selection_mode = _lf_mode
    # --lf-target-duration: clamp the best_moments filter to a single-LF
    # window of EXACTLY N minutes. Incompatible with full_drive (where the
    # output is bounded by full_drive_max_duration_sec, not a target).
    _lf_target_minutes = getattr(args, "lf_target_duration", None)
    if isinstance(_lf_target_minutes, (int, float)) and not isinstance(_lf_target_minutes, bool):
        if config.drive.lf_selection_mode == "full_drive":
            print(
                "error: --lf-target-duration is incompatible with "
                "--lf-mode full_drive (full_drive returns the whole "
                "timeline; use --lf-mode best_moments for a target duration).",
                file=sys.stderr,
            )
            return 2
        if _lf_target_minutes <= 0:
            print(
                f"error: --lf-target-duration must be > 0 minutes (got {_lf_target_minutes})",
                file=sys.stderr,
            )
            return 2
        target_sec = float(_lf_target_minutes) * 60.0
        config.drive.best_moments_mode = True
        config.drive.best_moments_target_min = target_sec
        config.drive.best_moments_target_max = target_sec
        # The packer (group_into_regular_lfs) bins segments into LFs of
        # [min_lf_duration, max_lf_duration]. Default max is 720s (12 min);
        # if the target exceeds that, segments would split across two LFs.
        # Widen the bracket so the target fits in a single LF; never
        # shrink it (a user passing 5min still allows multi-LF default).
        if target_sec > config.drive.max_lf_duration:
            config.drive.max_lf_duration = target_sec * 1.2
        if target_sec < config.drive.min_lf_duration:
            config.drive.min_lf_duration = max(target_sec * 0.8, 60.0)
    # --superlong: two independent mutual-exclusion rejects, same style as
    # --lf-target-duration above. A full_drive single LF and a
    # duration-clamped best_moments LF are different outputs from the
    # whole-day sl_NN.mp4 parts superlong produces.
    if getattr(args, "superlong_enabled", None):
        if config.drive.lf_selection_mode == "full_drive":
            print(
                "error: --superlong is incompatible with --lf-mode "
                "full_drive (full_drive already spans the whole "
                "post-stop-removal timeline as a single LF; --superlong "
                "would duplicate it as sl_NN.mp4 parts).",
                file=sys.stderr,
            )
            return 2
        if isinstance(_lf_target_minutes, (int, float)) and not isinstance(
            _lf_target_minutes, bool
        ):
            print(
                "error: --superlong is incompatible with "
                "--lf-target-duration (a fixed-duration highlight LF and "
                "whole-day superlong parts are different outputs; run "
                "them in separate invocations).",
                file=sys.stderr,
            )
            return 2
    if args.no_review:
        config.drive.no_review = True
    # --no-audio is threaded through run(skip_audio=) → the flag registry,
    # which owns the config.drive.skip_audio mirror on both the fresh build
    # and every resume (a hand-wire here would drop the intent on a resume
    # that omits the flag). See flag_registry.FLAG_REGISTRY.
    if getattr(args, "skip_stop_verify", False):
        config.drive.skip_stop_verify = True
    if getattr(args, "skip_blur", False):
        config.drive.skip_blur = True
    from ...pipeline.inserts import InsertError, apply_insert_flags, describe_drive_inserts

    apply_insert_flags(args, config)
    try:
        describe_drive_inserts(config.drive)  # media must exist before any phase runs
    except InsertError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "skip_validation", False):
        config.drive.skip_validation = True
    if getattr(args, "allow_cross_fs_source", False):
        config.drive.allow_cross_fs_source = True
    if getattr(args, "allow_degraded_weather", False):
        config.drive.allow_degraded_weather = True

    def _real_number(val):
        # Robust against MagicMock-based test fixtures: an unset attribute on
        # a MagicMock args object auto-vivifies as a MagicMock (truthy, not
        # None), so an `is not None` check alone would misread "not passed"
        # as "passed". Only a genuine argparse float/int counts as supplied.
        return isinstance(val, (int, float)) and not isinstance(val, bool)

    _duration_dirty = False
    _shorts_min_duration = getattr(args, "shorts_min_duration", None)
    if _real_number(_shorts_min_duration):
        config.drive.shorts_min_duration_sec = _shorts_min_duration
        _duration_dirty = True
    _shorts_max_duration = getattr(args, "shorts_max_duration", None)
    if _real_number(_shorts_max_duration):
        config.drive.shorts_max_duration_sec = _shorts_max_duration
        _duration_dirty = True
    _shorts_target_duration = getattr(args, "shorts_target_duration", None)
    if _real_number(_shorts_target_duration):
        config.drive.shorts_target_duration_sec = _shorts_target_duration
        _duration_dirty = True
    if _duration_dirty:
        # Pydantic re-validates on assignment only when validate_assignment
        # is set; force a fresh model_validate to surface bad input early
        # (min/max/target ordering, 60s hard ceiling).
        config.drive.__class__.model_validate(config.drive.model_dump())

    # Per-invocation cadence overrides — must run BEFORE
    # schedule_constants.apply_config so module-level attrs reflect the
    # per-run values. main() already called apply_config once with the
    # config-file values; we mutate config.drive then re-apply so the
    # CLI flags win.
    from ...pipeline import schedule_constants as _sched
    from ...pipeline.seo_slots import CadenceSourceUnreadableError

    dow = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    cadence_dirty = False
    _lf_day = getattr(args, "lf_day", None)
    _lf_time = getattr(args, "lf_time", None)
    _shorts_times_arg = getattr(args, "shorts_times", None)
    _lf_day_key = _lf_day if isinstance(_lf_day, str) and _lf_day in dow else None

    if _lf_day_key is not None:
        config.drive.long_form_publish_days = [dow[_lf_day_key]]
        cadence_dirty = True

    # Consistency: --lf-day sets ``long_form_publish_days`` (the weekday the
    # schedule actually lands on), but ``publish_day`` is fed separately from
    # --publish. If ``publish_day`` could differ from the weekday in
    # ``long_form_publish_days``, the ``_check_publish_day`` state invariant
    # would trip. So when --lf-day is given and --publish is NOT, default
    # ``publish_day`` to the SAME weekday. An explicit --publish still wins; with
    # both absent it stays None and the orchestrator uses the earliest
    # configured long-form day.
    _publish_day = args.publish or _lf_day_key

    # Long-form time: explicit --lf-time wins; else the LF day's designed
    # slot decides; else the config default is left untouched. The slot path
    # is why --lf-day tue alone yields a 16:00 long-form time (cadence table).
    if isinstance(_lf_time, str) or _sched.resolve_publish_slot(config, _lf_day_key):
        config.drive.long_form_publish_time = _sched.resolve_lf_time(config, _lf_day_key, _lf_time)
        cadence_dirty = True

    # Shorts times: single precedence point — explicit --shorts-times → the
    # region's Waypost windows (this --lf-day's share of its list) → flat list
    # when there is no region. Shorts are determined per-REGION by Waypost
    # (seo_slots.shorts_windows), never by a per-day publish_slots source. An
    # unreadable Waypost source, or a region it has no hours for, stops here.
    try:
        if isinstance(_shorts_times_arg, str) or _lf_day_key is not None:
            config.drive.shorts_publish_times = _sched.resolve_shorts_times(
                config, _lf_day_key, _shorts_times_arg, args.region
            )
            cadence_dirty = True
        elif isinstance(args.region, str):
            # No explicit --shorts-times and no LF-day slot: region-aware default
            # only. Deliberately NOT cadence_dirty — the hours are Waypost's
            # (seo_manager validates them when it reads the cadence), and shorts
            # scheduling reads config.drive.shorts_publish_times directly
            # (scheduler.from_config + phase_d_scheduling), not module-level
            # SHORTS_TIMES.
            config.drive.shorts_publish_times = _sched.resolve_shorts_times(
                config, None, None, args.region
            )
        if cadence_dirty:
            # Pydantic re-validates on assignment only when validate_assignment
            # is set; force a fresh model_validate to surface bad input early.
            config.drive.__class__.model_validate(config.drive.model_dump())
            _sched.apply_config(config)

        # ── Single-source drift gate ──────────────────────────────────────────
        # Shorts times come from ONE place: Waypost's cadence
        # (seo_slots.shorts_windows) or an explicit --shorts-times. When neither an
        # explicit flag was given NOR the resolved window equals the region's
        # determined spec, a SECOND source has drifted back in (the
        # publish_slots-override regression class) — fail loud before any encode or
        # upload rather than publishing off-spec. Tautological by construction
        # today; the point is to make the single-source invariant ENFORCED so it
        # cannot silently regress.
        if not isinstance(_shorts_times_arg, str) and isinstance(args.region, str):
            _spec = _sched.resolve_shorts_times_for_region(config, args.region, _lf_day_key)
            if list(config.drive.shorts_publish_times) != list(_spec):
                raise SystemExit(
                    "SCHEDULE-SPEC-DRIFT: resolved shorts window "
                    f"{list(config.drive.shorts_publish_times)} != region "
                    f"{args.region!r} spec {list(_spec)} (lf_day={_lf_day_key!r}, no "
                    "--shorts-times). A schedule source has drifted from "
                    "Waypost's shorts hours — see "
                    "docs/pipeline/79-publish-cadence-resolution.md."
                )
    except CadenceSourceUnreadableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    orchestrator = DrivePipelineOrchestrator(config)

    refusal = _launch_anchor_refusal(args, config, _publish_day)
    if refusal:
        print(f"error: {refusal}", file=sys.stderr)
        return 2

    # Map letter to phase name
    phase = None
    if args.phase:
        phase_map = {"A": "phase_a", "BC": "phase_bc_day_first", "D": "phase_d"}
        phase = phase_map[args.phase]

    try:
        result = orchestrator.run(
            source_dir=args.directory,
            output_base=args.output_dir,
            region=args.region,
            publish_day=_publish_day,
            location=args.location,
            fresh=args.fresh,
            # Argparse-sentinel: ``None`` means the operator did NOT
            # supply the flag; ``True``/``False`` means EXPLICIT. The
            # orchestrator's sidecar auto-skip safety net steps aside
            # whenever ANY of the four upload flags was explicit.
            skip_upload=getattr(args, "skip_upload", None),
            skip_drive_upload=getattr(args, "skip_drive_upload", None),
            skip_youtube_upload=getattr(args, "skip_youtube_upload", None),
            skip_csv=args.skip_csv,
            phase=phase,
            start_date=getattr(args, "start_date", None),
            # Rebase-on-stale-resume backstops. force_no_rebase is an
            # argparse sentinel: None = absent (saved state survives resume).
            lf_publish_date=getattr(args, "lf_publish_date", None),
            force_no_rebase=getattr(args, "force_no_rebase", None),
            shorts_per_day=getattr(args, "shorts_per_day", None),
            shorts_target=getattr(args, "shorts_target", None),
            # --publish implies --youtube-upload (opt out via
            # --no-youtube-upload). See resolve_youtube_upload above.
            youtube_upload=resolve_youtube_upload(args),
            shorts_gap=getattr(args, "shorts_gap", None),
            keep_intermediates=getattr(args, "keep_intermediates", None),
            keep_masters=getattr(args, "keep_masters", None),
            # SHORTS-CONTRACT escape.
            # Argparse-sentinel: None = flag absent (state wins on resume),
            # True = explicit (CLI wins). Tri-state passed through as-is.
            allow_short_count=getattr(args, "allow_short_count", None),
            # --allow-bad-shorts: the Phase D shorts dimension-gate escape
            # its own failure message tells the operator to pass. Same
            # argparse-sentinel treatment as allow_short_count above.
            skip_shorts_quality_gate=getattr(args, "skip_shorts_quality_gate", None),
            no_ledger_reconcile=getattr(args, "no_ledger_reconcile", None),
            # Operator-driven reconciliation. Argparse-sentinel
            # ``args.force_sidecar_reset``
            # is None when the CLI did not supply the flag (state wins
            # on resume) and True when it was supplied (CLI wins).
            # Legacy callers that pass a definite True/False still work
            # because ``run()`` treats ``None`` as "not supplied" →
            # state wins and ``False``/``True`` as "explicit" → CLI wins.
            force_sidecar_reset=getattr(args, "force_sidecar_reset", None),
            # Quality-gate escape hatch. Same argparse-sentinel
            # treatment as force_sidecar_reset above — None means the
            # CLI did not pass the flag and state wins on resume.
            allow_seo_degraded=getattr(args, "allow_seo_degraded", None),
            # Preflight execution plan. --dry-run renders the plan
            # and exits 0 without running phases; --force skips the
            # interactive Proceed? prompt (required for tmux). See
            # pipeline/preflight_plan.py.
            dry_run=getattr(args, "dry_run", False),
            force=getattr(args, "force", False),
            # --skip-invariants is the
            # break-glass for a latent Layer-1 invariant bug that would
            # otherwise kill every save_drive_state call mid-run. Never
            # set in normal operation; only passed here so the CLI flag
            # reaches the orchestrator.
            skip_invariants=getattr(args, "skip_invariants", False),
            # Orchestrator-entry concurrent-run lock escape hatch.
            # Operator-only; only set when PID-probe stale-recovery has
            # misfired and the lock needs to be broken manually.
            force_unlock=getattr(args, "force_unlock", False),
            # Operator-supplied per-LF location overrides
            # for GPS-less drives. Empty dict on regular GPS-bearing
            # invocations. The post-discovery validation gate in PhaseBC
            # checks coverage against actual LF indices.
            per_lf_overrides=per_lf_overrides,
            # --list-lfs short-circuits Phase A → drive plan build
            # → print table → exit, without entering phase_bc / encoding.
            list_lfs=getattr(args, "list_lfs", False),
            # Verification harness: resolve + print each LF's corrected
            # dominant-county label, then exit 0 without encoding.
            label_dry_run=getattr(args, "label_dry_run", False),
            # --ost-audit-confirmed bypasses the per-day OST
            # audit gate (non-interactive runs). Argparse-sentinel:
            # None means "operator did NOT pass the flag" → state wins
            # on resume; True means "explicit" → CLI wins.
            ost_audit_confirmed=getattr(args, "ost_audit_confirmed", None),
            # --gate-cd-confirmed marker; argparse-sentinel.
            gate_cd_confirmed=getattr(args, "gate_cd_confirmed", None),
            # Explicit --no-youtube-upload (banner distinguishes
            # decline vs forgotten flag). tri-state passed through —
            # the registry's ALWAYS_CLI row coerces it concrete, so omitting
            # the flag clears a saved True instead of latching it forever.
            no_youtube_upload=getattr(args, "no_youtube_upload", None),
            # --no-audio (argparse-sentinel: None = flag absent → saved intent
            # survives resume; True = strip LF audio this run). The registry
            # mirrors it onto config.drive.skip_audio (the pipeline read path).
            skip_audio=getattr(args, "no_audio", None),
            # --sign-verify / --no-sign-verify / --sign-verify-strict
            # (argparse-sentinel: None = flag absent → saved intent survives
            # resume). Threaded through run() exactly like keep_masters /
            # ost_audit_confirmed: the registry owns both the state write and
            # the config.drive mirror the pipeline reads, so — like --no-audio
            # above — this handler must NOT hand-wire config.drive.sign_verify
            # (a hand-wire drops the intent on a resume that omits the flag).
            sign_verify=getattr(args, "sign_verify", None),
            sign_verify_strict=getattr(args, "sign_verify_strict", None),
            # --superlong / --superlong-day: registry-owned, same reasoning
            # as --no-audio above.
            superlong_enabled=getattr(args, "superlong_enabled", None),
            superlong_publish_day=getattr(args, "superlong_publish_day", None),
        )
    except PipelineValidationError as exc:
        # Outermost boundary: log full report and exit non-zero. The
        # orchestrator never catches this — success=True here would be
        # a lie.
        print("\nDrive pipeline failed DoD validation!", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        # This gate fires AFTER Phase D's irreversible uploads, so the
        # recoverable journal merge / deferral sidecar / routing append must
        # still run before we return — otherwise the site silently goes stale.
        # The hook never raises and never masks the exit code below.
        _maybe_auto_publish_journal(args, orchestrator, config)
        return 1
    except CorruptStateError as exc:
        # Loud failure for corrupt pipeline_state.json. Previously this
        # was swallowed into a silent fresh-start (load_drive_state
        # returned None). An operator tailing stdout could miss the
        # warning and lose hours of prior work. Now: banner + non-zero
        # exit + explicit recovery instructions. Operator must opt in
        # to the fresh run by re-invoking with --fresh.
        banner = "=" * 60
        print(f"\n{banner}", file=sys.stderr)
        print("  ERROR: CORRUPT pipeline_state.json — RESUME BLOCKED", file=sys.stderr)
        print(banner, file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print(
            "  The saved state file could not be parsed and no valid",
            file=sys.stderr,
        )
        print(
            "  .tmp sibling was recoverable. The pipeline will NOT",
            file=sys.stderr,
        )
        print(
            "  silently start over — prior work is preserved.",
            file=sys.stderr,
        )
        print("", file=sys.stderr)
        print("  To recover:", file=sys.stderr)
        print(
            "    1. Inspect / repair the state file manually, OR",
            file=sys.stderr,
        )
        print(
            "    2. Re-run with --fresh to discard prior state and start over.",
            file=sys.stderr,
        )
        print(f"{banner}\n", file=sys.stderr)
        return 2
    except SidecarIntegrityError as exc:
        # Loud failure when youtube_uploads.json disagrees with the
        # current drive_plan. The gate runs at Phase D entry — a
        # mismatch usually means phantom entries from a prior aborted
        # run or manually deleted YouTube drafts. Do NOT auto-wipe;
        # surface the diff so the operator decides.
        banner = "=" * 60
        print(f"\n{banner}", file=sys.stderr)
        print(
            "  ERROR: youtube_uploads.json INTEGRITY CHECK FAILED",
            file=sys.stderr,
        )
        print(banner, file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  To recover:", file=sys.stderr)
        print(
            "    1. Verify via YouTube Studio which video IDs still exist, OR",
            file=sys.stderr,
        )
        print(
            "    2. Re-run with --force-sidecar-reset to wipe the sidecar and upload fresh.",
            file=sys.stderr,
        )
        print(f"{banner}\n", file=sys.stderr)
        return 3
    except PhaseQualityError as exc:
        # Loud failure when Phase D's end-of-phase quality gate finds
        # semantic violations (title collisions, phantom video IDs in
        # CSV, schedule drift, drive-wide location leak, etc). The gate
        # accumulates every violation and reports them all at once; the
        # body of the exception is the full list.
        banner = "=" * 60
        print(f"\n{banner}", file=sys.stderr)
        print(
            "  ERROR: PHASE D QUALITY GATE FAILED",
            file=sys.stderr,
        )
        print(banner, file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  To recover:", file=sys.stderr)
        print(
            "    1. Inspect each violation above and fix the underlying data, OR",
            file=sys.stderr,
        )
        print(
            "    2. For seo_degraded=True violations only: re-run with",
            file=sys.stderr,
        )
        print(
            "       --allow-seo-degraded after confirming titles are acceptable.",
            file=sys.stderr,
        )
        print(f"{banner}\n", file=sys.stderr)
        # End-of-Phase-D gate fires AFTER the irreversible uploads; run
        # the recoverable journal hook before returning (see PipelineValidationError).
        _maybe_auto_publish_journal(args, orchestrator, config)
        return 4
    except KeyboardInterrupt:
        print("\n\nDrive pipeline interrupted by user.")
        return 130
    except SystemExit as e:
        print(f"\n\nDrive pipeline terminated (exit code {e.code}).")
        return int(e.code) if e.code is not None else 143

    if result.success:
        print("\nDrive pipeline complete!")
        plan = result.drive_plan
        if plan:
            # plan may be a DrivePlan dataclass or a plain dict (from checkpoint state)
            if isinstance(plan, dict):
                regular_lfs = plan.get("regular_lfs", [])
                highlight_ratio = plan.get("highlight_ratio", 0)
            else:
                regular_lfs = plan.regular_lfs
                highlight_ratio = plan.highlight_ratio
            print(f"  Regular LFs: {len(regular_lfs)}")
            print(f"  Highlight ratio: {highlight_ratio:.0%}")

        _maybe_auto_publish_journal(args, orchestrator, config)
        return 0
    else:
        print("\nDrive pipeline failed!")
        for error in result.errors:
            print(f"   Error: {error}", file=sys.stderr)
        return 1
