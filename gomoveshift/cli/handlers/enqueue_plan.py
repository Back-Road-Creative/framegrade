"""Handler for ``gomoveshift enqueue-plan``.

The missing link between Phase D and the dispatcher. Phase D writes
``<project>_publish_plan.json``; ``gomoveshift dispatch-queue`` drains the
delivery queue; until now nothing put deliveries INTO that queue, so
``enqueue_from_plan_sidecar`` was unreferenced and the chain was broken.

**Shorts publish automatically** (there are no manual uploads). Every drive runs
``--no-review``, so no human Gate C/D answer is recorded; that absence does
not block the queue. The automated pre-spool gates below (media
verification, privacy, payload limits, rolling ceiling and brand cap,
batch anti-spam, plan gates) decide what reaches the queue.

When a human DID confirm Gate C/D, ``review_gate`` records
``gate_cd_confirmation_hash`` over every ``*_publish_plan.json`` sidecar,
and this command still refuses when:

- the gate is marked confirmed but carries no hash, or
- the sidecar has changed since it was confirmed (recomputed hash !=
  the recorded one).

``gomoveshift drive`` calls this handler with ``--apply`` at the end of a
successful publish run (``pipeline/auto_enqueue.py``), so no operator step
sits between Phase D and the hourly dispatch cron.

Dry-run by default, matching ``publish-journal`` and ``dispatch-queue``:
without ``--apply`` nothing is written to the queue.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

from ...console.library import privacy_status_blocks_publish, privacy_status_for_media
from ...console.settings import current_brand
from ...pipeline.drive_state_model import PipelineState
from ...pipeline.review_gate import compute_gate_cd_hash
from ...publishing.anti_spam_gate import check_batch, recent_platform_texts
from ...publishing.brand_policy import load_brand_policy
from ...publishing.dynamic_schedule import weekly_volume_from_ladder
from ...publishing.enqueue_from_plan import (
    DeliveryPlanConflictError,
    PublishPlanSidecarError,
    enqueue_from_plan_sidecar,
    select_capped_shorts,
)
from ...publishing.media_fingerprint import MediaVerificationError, verify_media
from ...publishing.models import PayloadLimitError, validate_payloads
from ...publishing.plan_io import PublishPlanFormatError, load_publish_plan
from ...publishing.queue import get_platform_daily_cap, open_queue_db
from .pub_status import VALIDATE_COMMAND, gating_problems, record_ignore_validate

#: Deliberately NOT ``TERMINAL_STATUSES``: a cadence ceiling counts the slots a
#: window has spent, and a ``published`` delivery spent one. Only a delivery
#: that will never reach a platform frees its slot back up.
_WILL_NEVER_PUBLISH = ("failed_permanent", "cancelled")
_WILL_NEVER_PUBLISH_SQL = ", ".join(f"'{s}'" for s in _WILL_NEVER_PUBLISH)

logger = logging.getLogger(__name__)

#: Used when ``config.seo`` carries no ``rolling_week_ceiling_per_platform``
#: (a bare-bones test double, or a config predating the field) -- kept in
#: sync with ``SEOConfig.rolling_week_ceiling_per_platform``'s own default:
#: 3 long-forms/week (Ireland Sun, USA Tue, USA Fri) x 7 shorts/long-form (``DriveConfig.min_shorts_per_lf``) x 1
#: delivery/platform/short = 21, the whole-account steady state, not a
#: per-region number.
_DEFAULT_ROLLING_WEEK_CEILING = 21


class GateNotApprovedError(RuntimeError):
    """Refuses to enqueue a plan Gate C/D has not approved as-is."""


def _load_state(state_path: Path) -> PipelineState:
    try:
        raw = json.loads(state_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GateNotApprovedError(f"run state unreadable: {state_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise GateNotApprovedError(f"run state is not valid JSON: {state_path}: {exc}") from exc
    return PipelineState.from_dict(raw)


def assert_gate_cd_approved(state: PipelineState, output_dir: Path) -> None:
    """Raise when a human Gate C/D confirmation no longer matches the content.

    An unreviewed run (no confirmation recorded) passes: shorts publish
    automatically through the automated pre-spool gates. A confirmed run
    must still carry a hash, and the sidecars must be unchanged since."""
    if not getattr(state, "gate_cd_confirmed", None):
        return
    recorded = getattr(state, "gate_cd_confirmation_hash", None)
    if not recorded:
        raise GateNotApprovedError(
            "Gate C/D is marked confirmed but carries no confirmation hash, so the approval "
            "cannot be tied to any particular content — refusing to enqueue."
        )
    current = compute_gate_cd_hash(state=state, output_dir=output_dir)
    if current != recorded:
        raise GateNotApprovedError(
            "the reviewed content has CHANGED since Gate C/D was confirmed "
            f"(recorded={recorded[:12]}… current={current[:12]}…) — what was approved is not "
            "what would be published. Re-run Gate C/D before enqueuing."
        )


def _default_output_dir(state: PipelineState, state_path: Path) -> Path:
    """Resolve the project directory to search for sidecars when the
    operator did not pass ``--output-dir``.

    Phase D writes ``*_publish_plan.json`` sidecars into the project
    directory (``state.output_base``), but ``pipeline_state.json`` lives
    under the stages dir — which ``gomoveshift.paths.gms_intermediate_dir``
    can relocate entirely off the output mount (``GMS_INTERMEDIATE_DIR``,
    or automatically for a drvfs-rooted run). ``state_path.parent`` is
    therefore the wrong default whenever a state carries ``output_base``;
    prefer it. The ``_stages``-parent heuristic is a fallback for legacy
    state files that predate the field.
    """
    output_base = getattr(state, "output_base", None)
    if output_base:
        return Path(output_base)
    if state_path.parent.name == "_stages":
        return state_path.parent.parent
    return state_path.parent


def _sidecars(output_dir: Path) -> List[Path]:
    return sorted(output_dir.glob("*_publish_plan.json"))


def _healthy_days(conn, platform: str) -> int:
    """Trailing consecutive days of ``queue.response_log`` for
    ``platform`` whose state is not ``SUPPRESSED`` -- HEALTHY and
    RECOVERING both count, matching ``response_policy.
    consecutive_healthy_days``'s own "healthy days" definition (the
    ``healthy_days`` argument that steps ``daily_cap_for``'s RECOVERING
    ladder). Counted by distinct calendar day (``at[:10]``) so several
    rows logged the same day count once; stops at the first SUPPRESSED row
    or the end of the log."""
    rows = conn.execute(
        "SELECT at, state FROM response_log WHERE platform=? ORDER BY at DESC",
        (platform,),
    ).fetchall()
    days: List[str] = []
    seen = set()
    for row in rows:
        if row["state"] == "SUPPRESSED":
            break
        day = row["at"][:10]
        if day not in seen:
            seen.add(day)
            days.append(day)
    return len(days)


def _media_verification_errors(sidecars: List[Path], max_shorts: int | None = None) -> List[str]:
    """Every short across every sidecar whose media fails ``verify_media``,
    reported all at once — not just the first — so one bad run isn't
    fixed one short at a time. A delivery can outlive its own media (an
    unmounted drive, a deleted short) days before the cron fires it; this
    is the upstream gate that catches that while the operator is still at
    the terminal, instead of inside a retry loop later."""
    errors: List[str] = []
    for path in sidecars:
        plan = load_publish_plan(path)
        # Only the shorts that will actually be spooled: a capped-out short
        # is never published, so its media is not a reason to refuse the run.
        for record in select_capped_shorts(list(plan.shorts), max_shorts):
            try:
                verify_media(record)
            except MediaVerificationError as exc:
                errors.append(f"  {path.name} short_index={record.short_index}: {exc}")
    return errors


def _privacy_violations(sidecars: List[Path], max_shorts: int | None = None) -> List[str]:
    """Every media item across every sidecar whose recorded privacy
    status is unrecorded or unrecognised (``console.library.
    privacy_status_blocks_publish``), reported all at once -- the same
    whole-run-refused shape as ``_media_verification_errors``. Robust by
    construction over a late gate: this runs before any queue write, not
    after a post ships unreviewed. Scoped to the same ``select_capped_shorts``
    selection the media-verification pass already uses, so a capped-out
    item (never spooled) is never a reason to refuse the run.

    ``GMS_SITE_PATH`` is read directly rather than through
    ``console.settings.resolve_site_path`` -- this handler runs outside any
    console brand context and has no other source for it."""
    site_path = os.environ.get("GMS_SITE_PATH")
    errors: List[str] = []
    for path in sidecars:
        plan = load_publish_plan(path)
        for record in select_capped_shorts(list(plan.shorts), max_shorts):
            for item in (record.media, *record.additional_media):
                status = privacy_status_for_media(item.path, record.format, site_path=site_path)
                if privacy_status_blocks_publish(status):
                    errors.append(
                        f"  {path.name} short_index={record.short_index} "
                        f"media={Path(item.path).name}: {status}"
                    )
    return errors


def _payload_limit_violations(sidecars: List[Path], max_shorts: int | None = None) -> List[str]:
    """Every short across every sidecar whose payload text exceeds a
    platform's documented limit, reported all at once -- the same
    whole-run-refused shape as ``_media_verification_errors`` and
    ``_privacy_violations``. Robust over a late gate: this runs before any
    queue write, not after a platform's API 400s a field it already
    accepted the short's media for."""
    errors: List[str] = []
    for path in sidecars:
        plan = load_publish_plan(path)
        for record in select_capped_shorts(list(plan.shorts), max_shorts):
            try:
                validate_payloads(record.payloads)
            except PayloadLimitError as exc:
                errors.append(f"  {path.name} short_index={record.short_index}: {exc}")
    return errors


def _validate_gate_refuses(conn, args, config, sidecars, max_shorts) -> bool:
    """U-R2: refuse to spool onto a (platform, project) pair that
    ``publishing-status --validate`` names a problem on. Only the pairs this
    plan would write are judged, so one broken pair never blocks the rest.
    ``--ignore-validate --reason`` lets it through and audits the override.
    The drive's automatic enqueue passes no escape flag, so it always gates."""
    ignore = getattr(args, "ignore_validate", False)
    reason = (getattr(args, "reason", None) or "").strip()
    if ignore and not reason:
        print(
            "Error: --ignore-validate needs --reason '<why>' (written to the audit log); "
            "nothing was written to the queue."
        )
        return True
    planned = {
        (platform, project) for platform, project, _, _ in _planned_deliveries(sidecars, max_shorts)
    }
    credentials_dir = getattr(getattr(config, "seo", None), "credentials_dir", None)
    problems = [
        p
        for p in gating_problems(
            conn,
            credentials_dir,
            storage_path=getattr(getattr(config, "seo", None), "storage_path", None),
        )
        if (p.platform, p.project) in planned
    ]
    if not problems:
        return False
    if ignore:
        record_ignore_validate(conn, problems, reason=reason, command="enqueue-plan")
        print(f"warning: --ignore-validate: {len(problems)} validate problem(s) overridden")
        return False
    print(
        "Error: publishing-status --validate names a problem on a platform/project this "
        "plan would publish to; nothing was written to the queue:"
    )
    for problem in problems:
        print(problem.line())
    print(f"See: {VALIDATE_COMMAND}. Override only with --ignore-validate --reason '<why>'.")
    return True


def _planned_deliveries(
    sidecars: List[Path], max_shorts: int | None
) -> List[Tuple[str, str, int, datetime]]:
    """Every ``(platform, project_name, short_index, scheduled_at)`` this run
    would spool, across all sidecars -- scoped to the same
    ``select_capped_shorts`` selection ``_media_verification_errors`` already
    uses, so the ceiling never counts a short the cap would defer anyway."""
    planned: List[Tuple[str, str, int, datetime]] = []
    for path in sidecars:
        plan = load_publish_plan(path)
        for record in select_capped_shorts(list(plan.shorts), max_shorts):
            for platform in record.routes.selected:
                planned.append(
                    (
                        platform,
                        record.project_name,
                        record.short_index,
                        record.schedule.scheduled_at_utc,
                    )
                )
    return planned


def _planned_records(sidecars: List[Path], max_shorts: int | None) -> List:
    """Every ``PublishRecord`` this run would spool, across all sidecars --
    scoped to the same ``select_capped_shorts`` selection the other
    pre-spool gates use, so the anti-spam gate never counts a short the cap
    would defer anyway."""
    records: List = []
    for path in sidecars:
        plan = load_publish_plan(path)
        records.extend(select_capped_shorts(list(plan.shorts), max_shorts))
    return records


def _rolling_ceiling_violations(
    conn, planned: List[Tuple[str, str, int, datetime]], ceiling: int
) -> List[Dict[str, object]]:
    """Refuse-whole check: for each platform this run would deliver to,
    count what the queue already holds in the rolling 7-day window starting
    at this run's earliest scheduled_at for that platform, plus what this
    run would add. A second region's plan pushing a platform's weekly total
    past ``ceiling`` is exactly the case a per-drive gate cannot see -- this
    is counted against the queue, the only place the cross-drive total is
    known. ``status != 'failed_permanent'`` because a queued, retrying,
    awaiting-manual, or already-published row all still occupy that
    platform's weekly slot; only a permanently-failed one does not."""
    by_platform: Dict[str, List[Tuple[str, int, datetime]]] = defaultdict(list)
    for platform, project_name, short_index, scheduled_at in planned:
        by_platform[platform].append((project_name, short_index, scheduled_at))

    violations: List[Dict[str, object]] = []
    for platform, entries in by_platform.items():
        window_start = min(scheduled_at for _, _, scheduled_at in entries)
        window_end = window_start + timedelta(days=7)
        in_window = [
            (project_name, short_index)
            for project_name, short_index, scheduled_at in entries
            if window_start <= scheduled_at < window_end
        ]
        existing = conn.execute(
            "SELECT COUNT(*) AS n FROM deliveries WHERE platform=? "  # noqa: S608
            f"AND status NOT IN ({_WILL_NEVER_PUBLISH_SQL}) "
            "AND scheduled_at >= ? AND scheduled_at < ?",
            (platform, window_start.isoformat(), window_end.isoformat()),
        ).fetchone()["n"]
        total = existing + len(in_window)
        if total > ceiling:
            violations.append(
                {
                    "platform": platform,
                    "ceiling": ceiling,
                    "existing": existing,
                    "planned": len(in_window),
                    "not_spooled": in_window,
                    "window_start": window_start,
                    "window_end": window_end,
                }
            )
    return violations


def _register_enqueued_shorts_in_tracking(spooled: List) -> None:
    """Register every spooled short in the TrackingStore so
    ``update_platform_id`` -- the publish hook's only write into tracking
    (``queue.py._register_platform_id_best_effort``) -- finds a record keyed
    by ``(project_name, short_index)`` once a delivery actually publishes.
    Before this, ``register_short``/``register_shorts_batch`` had no
    production caller, so the store never got any records at all.

    Grouped by ``project_name`` because ``register_shorts_batch`` takes one
    project at a time. ``register_shorts_batch`` finds-or-updates by index,
    so re-enqueuing the same plan is idempotent by construction -- no
    duplicate-detection needed here.

    Never blocks the enqueue: a registration failure is logged as a
    WARNING and swallowed, the same non-fatal contract the publish hook
    itself uses for this same store.

    Only ``file``, ``duration_sec`` and a location label (from
    ``sampled_regions``) are grounded in the plan/sidecar data available
    here. ``ContentMeta.title_format`` and any per-short series id are left
    unset -- no publish-plan field carries either.
    """
    by_project: Dict[str, List] = defaultdict(list)
    for record in spooled:
        by_project[record.project_name].append(record)

    from ...tracking import TrackingStore

    store = TrackingStore()
    for project_name, records in by_project.items():
        shorts_data = []
        regions: List[str] = []
        for record in records:
            youtube = record.payloads.youtube
            shorts_data.append(
                {
                    "index": record.short_index,
                    "file": record.media.path,
                    "duration_sec": record.media.duration_seconds,
                    "tags": list(youtube.tags) if youtube else [],
                }
            )
            for region in record.sampled_regions:
                if region not in regions:
                    regions.append(region)
        try:
            store.register_shorts_batch(project_name, shorts_data, location=", ".join(regions))
        except Exception as exc:  # noqa: BLE001 -- must never block the enqueue
            logger.warning(
                "tracking: failed to register %d short(s) for project %s (non-fatal): %s",
                len(shorts_data),
                project_name,
                exc,
            )


def cmd_enqueue_plan(args, config) -> int:
    state_path = Path(args.state_path).resolve()

    try:
        state = _load_state(state_path)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = _default_output_dir(state, state_path)
        assert_gate_cd_approved(state, output_dir)
    except GateNotApprovedError as exc:
        print(f"Error: {exc}")
        return 1

    sidecars = _sidecars(output_dir)
    if not sidecars:
        # Not an error: a run with every route off writes no
        # sidecar at all, which is the normal state today.
        print(
            f"No *_publish_plan.json sidecar in {output_dir}; nothing to enqueue. "
            "If the plan lives elsewhere, pass --output-dir <project dir>."
        )
        return 0

    if getattr(state, "gate_cd_confirmed", None):
        print(f"Gate C/D approved, content hash matches. Sidecars: {len(sidecars)}")
    else:
        print(
            "No human Gate C/D review on record — the automated gates decide. "
            f"Sidecars: {len(sidecars)}"
        )
    for path in sidecars:
        print(f"  {path.name}")

    # GMS-17a: a dry run is the operator's check, so it reads every sidecar
    # too -- a publish preview copied over a plan's name is refused here,
    # not reported as ready to spool.
    try:
        for path in sidecars:
            load_publish_plan(path)
    except PublishPlanFormatError as exc:
        print(f"Error: {exc}")
        return 1

    max_shorts = getattr(args, "max_shorts", None)
    if max_shorts is not None:
        # The cap is applied in memory over the approved sidecar; the file is
        # never rewritten, so the Gate C/D hash keeps describing what a human
        # reviewed. Trimming the file instead flips that hash and is refused.
        print(f"Volume cap: spooling at most {max_shorts} short(s) per sidecar (plan unchanged).")

    shadow = bool(getattr(args, "shadow", False))
    if shadow:
        print("SHADOW run: these deliveries will be spooled UNPUBLISHABLE.")

    if not args.apply:
        print("\nDRY RUN - nothing was written to the queue. Re-run with --apply.")
        return 0

    try:
        media_errors = _media_verification_errors(sidecars, max_shorts)
    except PublishPlanFormatError as exc:
        print(f"Error: {exc}")
        return 1
    if media_errors:
        print(
            "Error: media verification failed for the following short(s); "
            "nothing was written to the queue:"
        )
        print("\n".join(media_errors))
        return 1

    privacy_errors = _privacy_violations(sidecars, max_shorts)
    if privacy_errors:
        print(
            "Error: privacy status is unrecorded or unrecognised for the following "
            "item(s); nothing was written to the queue:"
        )
        print("\n".join(privacy_errors))
        return 1

    payload_errors = _payload_limit_violations(sidecars, max_shorts)
    if payload_errors:
        print(
            "Error: a platform payload field exceeds its documented limit for the "
            "following short(s); nothing was written to the queue:"
        )
        print("\n".join(payload_errors))
        return 1

    conn = open_queue_db(config.seo.storage_path)
    try:
        if not shadow and _validate_gate_refuses(conn, args, config, sidecars, max_shorts):
            return 1
        ceiling = getattr(
            getattr(config, "seo", None),
            "rolling_week_ceiling_per_platform",
            _DEFAULT_ROLLING_WEEK_CEILING,
        )
        # A brand's policy.weekly_ceiling (brand_policy.py) further
        # trims the rolling-week ceiling -- never raises it above config.
        # Operator/config intent is the outer bound; the brand policy is an
        # additional, brand-scoped cap on top of it.
        _rung = "config"
        policy = load_brand_policy(current_brand())
        _policy_ceiling = policy.weekly_ceiling
        if _policy_ceiling is not None and _policy_ceiling < ceiling:
            ceiling = _policy_ceiling
            _rung = "brand-policy-cap"
        # The bounded-response ladder is a third rung. youtube is
        # the reference platform -- every delivery in this queue schedules
        # off the youtube slot (dynamic_schedule.py's own staggering
        # contract staggers facebook/instagram OFF a youtube base_utc), so
        # its bounded-response state is the one that should further trim a
        # ceiling shared across platforms. Only a "live" cap row actually
        # lowers the ceiling; a "shadow" row is logged only, matching
        # dispatch._select_due's own shadow-vs-live distinction. Never
        # raises the ceiling above what config/brand-policy already set.
        cap_row = get_platform_daily_cap(conn, "youtube")
        if cap_row is not None:
            healthy_days = _healthy_days(conn, "youtube")
            ladder_volume = weekly_volume_from_ladder(policy, cap_row["state"], healthy_days)
            if cap_row["mode"] == "live":
                if ladder_volume < ceiling:
                    ceiling = ladder_volume
                    _rung = "ladder"
            else:
                logger.info(
                    "enqueue-plan: ladder rung would set ceiling %s "
                    "(state=%s healthy_days=%s mode=shadow)",
                    ladder_volume,
                    cap_row["state"],
                    healthy_days,
                )
        logger.info("enqueue-plan: rolling week ceiling %s per platform (rung=%s)", ceiling, _rung)
        violations = _rolling_ceiling_violations(
            conn, _planned_deliveries(sidecars, max_shorts), ceiling
        )
        if violations:
            print(
                "Error: rolling 7-day per-platform ceiling exceeded; "
                "nothing was written to the queue (whole plan refused):"
            )
            for v in violations:
                not_spooled = ", ".join(f"{p}#{i}" for p, i in v["not_spooled"])
                print(
                    f"  platform={v['platform']} ceiling={v['ceiling']} "
                    f"existing_in_window={v['existing']} this_plan_would_add={v['planned']} "
                    f"window={v['window_start'].isoformat()}..{v['window_end'].isoformat()} "
                    f"NOT spooled: {not_spooled}"
                )
            return 1

        # Refuse the whole batch on templated/duplicate/over-capped
        # content -- upstream of the spool loop below, the same
        # nothing-written-on-refusal shape the other pre-spool gates use.
        # Cross-batch uniqueness/count is a property no single sidecar's
        # generator call can see, so this is the one gate that runs here
        # rather than at generation time.
        planned_records = _planned_records(sidecars, max_shorts)
        anti_spam_violations = check_batch(
            planned_records,
            per_location_cap=policy.per_location_cap,
            recent_texts=recent_platform_texts(
                conn, exclude_projects={r.project_name for r in planned_records}
            ),
        )
        if anti_spam_violations:
            print(
                "Error: batch anti-spam gate refused the following short(s); "
                "nothing was written to the queue:"
            )
            for v in anti_spam_violations:
                platform = f" platform={v.platform}" if v.platform else ""
                print(f"  index={v.index}{platform} rule={v.rule}: {v.detail}")
            return 1

        total_new = total_existing = total_deferred = 0
        for path in sidecars:
            try:
                plan = load_publish_plan(path)
                plan_size = len(plan.shorts)
                result = enqueue_from_plan_sidecar(conn, path, shadow=shadow, max_shorts=max_shorts)
            except DeliveryPlanConflictError as exc:
                # The plan moved under an in-flight delivery. Stop rather
                # than enqueue a partial run: continuing would spool some
                # shorts against a plan that conflicts with others.
                print(f"Error: {exc}")
                return 1
            except PublishPlanSidecarError as exc:
                print(f"Error: {exc}")
                return 1
            if not shadow:  # a shadow delivery can never publish; nothing to track
                _register_enqueued_shorts_in_tracking(
                    select_capped_shorts(list(plan.shorts), max_shorts)
                )
            total_new += result.enqueued_count
            total_existing += result.already_present_count
            deferred = plan_size - len(select_capped_shorts(list(range(plan_size)), max_shorts))
            total_deferred += deferred
            line = (
                f"  {path.name}: enqueued={result.enqueued_count} "
                f"already_present={result.already_present_count}"
            )
            if deferred:
                line += f" deferred={deferred}"
            print(line)
        suffix = " (SHADOW - none of these can ever publish)" if shadow else ""
        tail = f" deferred={total_deferred}" if total_deferred else ""
        print(f"\nenqueued={total_new} already_present={total_existing}{tail}{suffix}")
        return 0
    finally:
        conn.close()
