"""Handler for the ``metrics`` CLI command and its subcommands.

Wires the previously-orphaned ``gomoveshift.tracking``
module (``MetricsCollector`` / ``TrackingStore`` / ``collect_all``, zero
prior call-sites) into the CLI.

NEW OPERATOR SURFACE: ``gomoveshift metrics collect <project>`` pulls the
latest YouTube performance numbers for a project's registered shorts and
persists them. The Phase-D post-upload hook auto-registers freshly
uploaded YouTube IDs so a later ``metrics collect`` has something to
fetch.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional

from ...console.settings import current_brand
from ...pipeline.drive_phases.youtube_uploader import YouTubeUploader

# Imported at module scope (not lazily) so tests can ``patch.object`` them.
from ...publishing.brand_policy import load_brand_policy
from ...publishing.dispatch import apply_suppression_response
from ...publishing.queue import (
    open_queue_db,
    recorded_publish_time,
    registered_platform_id,
)
from ...tracking import (
    MetricRow,
    MetricsCollector,
    TrackingStore,
    upsert_audience_online,
    upsert_from_metrics,
)
from ...tracking.channel_daily import load_rows as load_channel_daily_rows
from ...tracking.channel_daily import pull_channel_daily, shorts_feed_share
from ...tracking.channel_enrol import enrol_channel
from ...tracking.collector import YouTubeCollector
from ...tracking.evidence_export import build_evidence_bundle
from ...tracking.interval import latest_interval
from ...tracking.observation import ObservationStatus
from ...tracking.slot_performance import read_sidecar_slot
from ...tracking.suppression import classify_youtube_shorts_feed
from ..logging_setup import configure_logging

logger = logging.getLogger(__name__)


def cmd_metrics(args, config):
    """Dispatch ``metrics`` subcommands."""
    if args.metrics_command == "export-evidence":
        # stdout is the JSON document: no stdout log handler, warnings go to stderr.
        return _cmd_metrics_export_evidence(config)
    configure_logging()

    if args.metrics_command == "collect":
        return _cmd_metrics_collect(args, config)
    if args.metrics_command == "backfill-ids":
        return _cmd_metrics_backfill_ids(config)
    if args.metrics_command == "best-times":
        return _cmd_metrics_best_times(args, config)
    if args.metrics_command == "report":
        return _cmd_metrics_report(args, config)
    if args.metrics_command == "analytics-probe":
        return _cmd_metrics_analytics_probe()
    if args.metrics_command == "enrol-channel":
        return _cmd_metrics_enrol_channel()
    if args.metrics_command == "feed-share":
        return _cmd_metrics_feed_share(args, config)

    print(
        "Usage: gomoveshift metrics {collect,backfill-ids,best-times,report,feed-share,analytics-probe,enrol-channel,export-evidence}",
        file=sys.stdout,
    )
    return 1


def _cmd_metrics_analytics_probe() -> int:
    """One tiny live Analytics query; prints ANALYTICS_OK or the reason line.

    Exit 0 when ok, 2 otherwise.
    """
    status = YouTubeCollector().probe_analytics()
    print(status.line())
    return 0 if status.status == "ok" else 2


def _cmd_metrics_enrol_channel() -> int:
    """Register every channel upload the tracking store lacks (idempotent).

    Prints ``EVIDENCE_COVERAGE ...``; on any listing/registration failure
    prints ``ENROL_CHANNEL_FAILED reason=...`` and exits 2.
    """
    try:
        coverage = enrol_channel(YouTubeUploader()._build_service(), TrackingStore())
    except Exception as exc:
        print(f"ENROL_CHANNEL_FAILED reason={exc}")
        return 2
    print(coverage.line())
    return 0


def _cmd_metrics_export_evidence(config) -> int:
    """Print the EvidenceBundle v1 JSON (settled day-7 facts, no thresholds).

    Shorts with neither a settled series nor a usable snapshot are excluded
    and counted on stderr as ``EVIDENCE_EXCLUDED n=<k> reason=no_settled_series``.
    """
    status = YouTubeCollector().analytics_status()
    records = TrackingStore().load_all_records()
    with _slot_source(config) as slot_for:
        bundle = build_evidence_bundle(records, status, datetime.now(), slot_for=slot_for)
    print(json.dumps(bundle, indent=2))
    excluded = bundle["health"]["excluded"]
    if excluded:
        print(f"EVIDENCE_EXCLUDED n={len(excluded)} reason=no_settled_series", file=sys.stderr)
    return 0


@contextlib.contextmanager
def _slot_source(config):
    """``slot_for(project, item)`` reading the publish-plan sidecar's region /
    weekday / hour_local, or ``None`` when the queue database is unreachable
    (the export then carries no region and derives the slot from the posting time)."""
    try:
        storage_path = Path(config.seo.storage_path)
        conn = open_queue_db(storage_path)
    except Exception as exc:  # noqa: BLE001 - region is optional evidence
        logger.warning("evidence export: no sidecar lookup (%s)", exc)
        yield None
        return
    lookup = _sidecar_lookup(conn, storage_path)

    def slot_for(project, item):
        index = getattr(item, "index", None)
        if getattr(item, "platform_ids", None) is None or index is None:
            return None
        path = lookup(project)
        return read_sidecar_slot(path, index) if path else None

    try:
        yield slot_for
    finally:
        conn.close()


def _cmd_metrics_collect(args, config) -> int:
    """Collect + persist metrics for one project's tracking record, then
    fold them into ``slot_performance``.

    Loads the project's tracking record, runs ``collect_all`` (updates each
    short's metrics in place), saves, and folds the collected numbers into
    the slot-keyed aggregate every ranking/decline surface reads. The fold
    lives HERE rather than in a ``metrics fold`` subcommand precisely so
    there is no way to collect without folding — a separate command is one
    a cron can forget. Exits 1 when the project has no tracking record yet
    (nothing to collect).
    """
    store = TrackingStore()
    project = args.project

    if not store.exists(project):
        print(
            f"No tracking record for project {project!r}. Shorts must be "
            "registered (Phase-D post-upload hook) before collecting.",
            file=sys.stderr,
        )
        return 1

    brand = current_brand()
    record = store.load(project)
    collector = MetricsCollector(brand=brand)
    updated = collector.collect_all(record)
    store.save(record)
    if collector.budget_exhausted_line:
        print(collector.budget_exhausted_line)

    # Line consumed verbatim by the metrics cron; never reshape it.
    print(f"Collected metrics for {updated} short(s) in project {project}")
    _report_observations(record)

    rows = _metric_rows(record)
    folded, skipped_reason = _fold_into_slot_performance(config, rows)
    line = f"Folded {folded} of {len(rows)} metric row(s) into slot_performance"
    if skipped_reason is not None:
        line += f" — fold skipped: {skipped_reason}"
    print(line)

    refreshed = _refresh_audience_online(config, collector)
    print(f"Refreshed audience-online: {refreshed} hour(s) written")

    pulled = _pull_channel_daily(collector, brand)
    print(f"Pulled {pulled} channel-daily row(s) for {brand}")

    _apply_bounded_response(config, brand, getattr(args, "response_mode", "shadow"))
    return 0


def _report_observations(record) -> None:
    """Print what this run learned per platform, per short: a measured
    trend when two reads exist, or the platform's own refusal reason --
    never an absent measurement printed as a silent zero.
    """
    for short in getattr(record, "shorts", []):
        for platform, series in sorted(getattr(short, "observation_log", {}).items()):
            if not series:
                continue
            latest = series[-1]
            status = latest.get("status")
            if status == ObservationStatus.MEASURED:
                trend = latest_interval(short, platform)
                if (
                    trend is not None
                    and trend.status == ObservationStatus.MEASURED
                    and trend.views is not None
                ):
                    print(
                        f"  short #{short.index} {platform}: measured "
                        f"(Δviews={trend.views} since {trend.start_at})"
                    )
                else:
                    print(f"  short #{short.index} {platform}: measured")
            else:
                detail = latest.get("detail") or ""
                suffix = f" — {detail}" if detail else ""
                print(f"  short #{short.index} {platform}: {status}{suffix}")


def _metric_rows(record) -> List[MetricRow]:
    """Project a tracking record's per-platform metrics into ``MetricRow``s.

    ``engagement_rate`` is the record's own derived property (interactions
    over views), not a stored field — nothing else here is computed."""
    rows: List[MetricRow] = []
    for short in getattr(record, "shorts", []):
        for platform, metrics in getattr(short, "metrics", {}).items():
            rows.append(
                MetricRow(
                    project_name=record.project,
                    short_index=short.index,
                    platform=platform,
                    views=metrics.views,
                    engagement_rate=metrics.engagement_rate,
                )
            )
    return rows


def _sidecar_lookup(conn, storage_path: Path) -> Callable[[str], Optional[Path]]:
    """``project_name -> publish-plan sidecar path``, or ``None``.

    Resolves the way ``dispatch_queue.SidecarRecordLoader`` does — the path
    the delivery was actually enqueued from wins, the flat
    ``<storage_path>/<project>_publish_plan.json`` convention is the
    fallback for rows predating that column — but returns ``None`` instead
    of raising: an unreachable sidecar must cost the fold one row, never
    the collection that already succeeded.
    """

    def lookup(project_name: str) -> Optional[Path]:
        row = conn.execute(
            "SELECT sidecar_path FROM deliveries "
            "WHERE project_name=? AND sidecar_path IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (project_name,),
        ).fetchone()
        path = (
            Path(row["sidecar_path"])
            if row is not None
            else storage_path / f"{project_name}_publish_plan.json"
        )
        if not path.exists():
            logger.warning(
                "no publish-plan sidecar for project %s at %s; slot fold skipped",
                project_name,
                path,
            )
            return None
        return path

    return lookup


def _fold_into_slot_performance(config, rows: List[MetricRow]) -> tuple[int, Optional[str]]:
    """Fold ``rows`` into ``slot_performance``, returning
    ``(count_folded, skipped_reason)``. ``skipped_reason`` is ``None`` on
    success (including the trivial no-rows case) and the exception's
    ``type: message`` when the fold was skipped.

    The aggregate shares the publishing spool's database
    (``<seo.storage_path>/social_queue.sqlite3``): the sidecar paths the
    fold resolves against live in that same file's ``deliveries`` table,
    and a second database would be a second thing to locate, back up and
    keep in sync. Collection is already persisted by the time this runs, so
    any failure here is reported and swallowed — a slot aggregate is
    rebuildable from the tracking records, an operator mistaking a fold
    error for a lost collection is not. The reason still rides the CLI's own
    stdout line, since the cron that runs this captures stdout, not the
    logger, and a nightly fold failure must not be invisible.
    """
    if not rows:
        return 0, None
    try:
        storage_path = Path(config.seo.storage_path)
        conn = open_queue_db(storage_path)
        try:
            return upsert_from_metrics(conn, rows, _sidecar_lookup(conn, storage_path)), None
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - never lose a collected metric to a fold
        reason = f"{type(exc).__name__}: {exc}"
        logger.warning("slot_performance fold skipped: %s", exc)
        return 0, reason


def _refresh_audience_online(config, collector: MetricsCollector) -> int:
    """Refresh ``audience_online`` from each platform collector's
    ``fetch_audience_online`` hour-of-day histogram, returning the row
    count written.

    Same standing as ``_fold_into_slot_performance``: collection is already
    persisted by the time this runs, so any failure here is reported and
    swallowed, never fatal to ``metrics collect``. A collector without a
    ``fetch_audience_online`` hook (YouTube has no hour dimension by
    design; TikTok doesn't expose one) is skipped, not treated as a fault.
    An empty histogram -- the normal result while a platform's insights
    scope hasn't been (re)granted -- writes zero rows, not an error.
    """
    try:
        storage_path = Path(config.seo.storage_path)
        conn = open_queue_db(storage_path)
        try:
            written = 0
            for platform, plat_collector in collector._collectors.items():
                fetch = getattr(plat_collector, "fetch_audience_online", None)
                if fetch is None:
                    continue
                hour_counts = fetch()
                if not hour_counts:
                    continue
                written += upsert_audience_online(conn, platform, hour_counts)
            return written
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - never fail collect on an audience-online refresh error
        logger.warning("audience_online refresh skipped: %s", exc)
        return 0


def _pull_channel_daily(collector: MetricsCollector, brand: str) -> int:
    """Best-effort daily channel-level pull: views by traffic source
    and content type, via `tracking.channel_daily.pull_channel_daily`,
    which reuses `collector.youtube`'s already-authenticated Analytics v2
    client rather than a second auth path. `brand` comes from
    `console.settings.current_brand()` -- the console's own multi-brand
    resolution (``GMS_CONSOLE_BRAND``, default "gomoveshift") -- not a
    literal here.

    Same standing as `_fold_into_slot_performance` and
    `_refresh_audience_online` above: every metric this run collected is
    already saved by the time this runs, so a failure here is reported and
    swallowed, never fatal to `metrics collect`. Every project's nightly
    `metrics collect` invocation reaches this, so the pull runs once the
    cron is enabled -- the per-day upsert makes a re-pull from a second
    project's invocation a no-op rather than a duplicate.

    Pulls only yesterday, the daily cron's own window; a wider backfill is
    a direct call to `pull_channel_daily`, not a CLI flag nothing else uses.
    """
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    try:
        return pull_channel_daily(collector.youtube, brand, yesterday, yesterday)
    except Exception as exc:  # noqa: BLE001 - never fail collect on a channel-daily pull error
        logger.warning("channel-daily pull skipped: %s", exc)
        return 0


def _apply_bounded_response(config, brand: str, mode: str) -> None:
    """Wire the already-built, shadow-first bounded response
    (`publishing.response_policy` / `publishing.dispatch.
    apply_suppression_response`) into the nightly `metrics collect` run.
    `docs/pipeline/141-bounded-response.md` has the full contract.

    Classifies YouTube's Shorts-feed share history via `tracking.
    suppression.classify_youtube_shorts_feed` over every stored
    `channel_daily` row for `brand` -- there is NO equivalent Facebook
    reel-views series in `channel_daily` yet (`grep -rn
    "classify_facebook_reel_reach" gomoveshift/tracking/suppression.py`
    exists but nothing in `channel_daily` stores reel-views rows), so
    Facebook is never classified or acted on here.

    `mode` defaults to ``"shadow"`` (``args.response_mode``, itself
    defaulting to ``"shadow"`` in the CLI parser) -- this call never
    flips it to ``"live"`` on its own; going live is a human editing the
    cron invocation. A platform with a `manual:`-reasoned override
    (console) is left untouched by `apply_suppression_response` itself,
    so nothing here needs to special-case it.

    Same standing as `_pull_channel_daily`/`_refresh_audience_online`:
    every metric this run collected is already saved by the time this
    runs, so a failure here is reported and swallowed, never fatal to
    `metrics collect`.
    """
    try:
        rows = load_channel_daily_rows(brand)
        day_states = classify_youtube_shorts_feed(rows) if rows else {}
        if not day_states:
            print(f"Bounded response: no channel-daily rows for {brand}")
            return

        today = max(day_states)
        ladder = load_brand_policy(brand).daily_ladder
        storage_path = Path(config.seo.storage_path)
        conn = open_queue_db(storage_path)
        try:
            decisions = apply_suppression_response(
                conn,
                brand=brand,
                states={"youtube": day_states},
                today=today,
                ladder=ladder,
                mode=mode,
            )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 - never fail collect on a bounded-response error
        logger.warning("bounded response skipped: %s", exc)
        return

    for decision in decisions:
        print(
            f"Bounded response ({decision['mode']}): {decision['platform']} "
            f"state={decision['state']} cap={decision['cap']} "
            f"healthy_days={decision['healthy_days']} auto_pause={decision['auto_pause']}"
        )


def _cmd_metrics_best_times(args, config) -> int:
    """``metrics best-times``: print the region x platform x top-3
    ``slot_performance`` table, with the platform's audience-online hours.

    Report-only and always exits 0 — thin data is the normal case and a
    cron must not go red for it. GMS moves no slot from the ranking.
    """
    from ...tracking.slot_performance import (
        ensure_audience_online_schema,
        slot_performance_trend,
        slots_ranked,
    )

    min_n = getattr(args, "min_n", 5)
    show_trend = getattr(args, "trend", False)
    try:
        conn = open_queue_db(config.seo.storage_path)
    except Exception as exc:  # noqa: BLE001 - never crash a report on a store error
        print(f"best-times: insufficient data — could not open the tracking store ({exc})")
        return 0

    try:
        ranked = slots_ranked(conn, min_n=min_n)
        if not ranked:
            print(
                f"best-times: insufficient data — no ranked slot_performance rows (min_n={min_n})"
            )
            return 0

        groups: dict = {}
        for row in ranked:
            groups.setdefault((row.region, row.platform), []).append(row)

        ensure_audience_online_schema(conn)

        for region, platform in sorted(groups):
            top3 = groups[(region, platform)][:3]
            print(f"{region} / {platform}:")
            for rank, row in enumerate(top3, start=1):
                print(
                    f"  #{rank} {row.hour_local:02d}:00 -- n={row.n} "
                    f"mean_views={row.mean_views:.1f} mean_engagement={row.mean_engagement:.3f}"
                )

            if show_trend:
                top = top3[0]
                trend = slot_performance_trend(
                    conn, region, platform, top.weekday, top.hour_local, weeks=4
                )
                if trend:
                    print(f"  trend ({top.hour_local:02d}:00, last {len(trend)} fold(s)):")
                    for snap in trend:
                        print(
                            f"    {snap['folded_at']}: n={snap['n']} "
                            f"mean_views={snap['mean_views']:.1f}"
                        )

            online_rows = conn.execute(
                "SELECT hour_local, SUM(n_online) AS total FROM audience_online "
                "WHERE platform = ? GROUP BY hour_local ORDER BY total DESC LIMIT 3",
                (platform,),
            ).fetchall()
            if online_rows:
                hours = ", ".join(f"{row[0]:02d}:00" for row in online_rows)
                print(f"  audience_online top-3: {hours}")
        return 0
    finally:
        conn.close()


def _cmd_metrics_report(args, config) -> int:
    """``metrics report``: print (or save) a ``PerformanceReporter``
    markdown report, wiring the previously call-site-less ``PerformanceReporter``
    into the CLI so the length experiment (and everything else the reporter
    already computes) is actually readable by an operator.

    Report-only and always exits 0 -- thin/missing data is the normal case,
    not a fault, matching ``best-times``'s standing. ``--project`` and
    ``--all`` are mutually exclusive at the parser; a project with no
    tracking record prints one line and exits 0 rather than raising.
    """
    from ...tracking.report import PerformanceReporter

    reporter = PerformanceReporter()
    output = Path(args.output) if getattr(args, "output", None) else None

    if getattr(args, "all", False):
        text = reporter.generate_cross_project_report()
        if output is not None:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text, encoding="utf-8")
            print(f"Saved cross-project report to {output}")
        else:
            print(text)
        return 0

    project = args.project
    if not reporter.store.exists(project):
        print(f"No tracking record for project {project!r}.")
        return 0

    if output is not None:
        saved = reporter.save_report(project, output_path=output)
        print(f"Saved report to {saved}")
    else:
        print(reporter.generate_project_report(project))
    return 0


def _cmd_metrics_feed_share(args, config) -> int:
    """``metrics feed-share``: the interim-cadence exit rule as a report
    line -- Shorts-feed share of channel views must be >= ``--threshold`` on
    each of the last ``--days`` measured days.

    Report-only and always exits 0, matching ``best-times``/``report``'s
    standing: thin data (fewer than ``--days`` measured days, or no
    ``channel_daily`` rows at all) is the normal case before the metrics
    cron has accumulated history, never a fault. A day is only "measured"
    when `shorts_feed_share` produced a share for it -- a day with zero
    total traffic-source views is never fabricated into a 0.0.
    """
    days = getattr(args, "days", 7)
    threshold = getattr(args, "threshold", 0.70)
    brand = current_brand()

    rows = load_channel_daily_rows(brand)
    if not rows:
        print(f"feed-share: no channel_daily rows for {brand}")
        return 0

    shares = shorts_feed_share(rows)
    if not shares:
        print(f"feed-share: no channel_daily rows for {brand}")
        return 0

    ordered_days = sorted(shares)[-days:]
    print(f"feed-share: {brand} last {days} measured days (threshold {threshold})")
    met = 0
    for day in ordered_days:
        share = shares[day]
        status = "OK" if share >= threshold else "LOW"
        if status == "OK":
            met += 1
        print(f"  {day} {share:.2f} {status}")

    measured = len(ordered_days)
    if measured < days:
        print(f"  verdict: INSUFFICIENT ({measured} measured days, need {days})")
    elif met == measured:
        print(f"  verdict: MET ({met}/{measured} days >= {threshold})")
    else:
        print(f"  verdict: NOT MET ({met}/{measured} days >= {threshold})")
    return 0


def _cmd_metrics_backfill_ids(config) -> int:
    """One-shot: register tracking ids for every ``published`` queue row.

    Covers rows that published before ``queue.transition``'s own
    registration hook existed. Idempotent -- ``TrackingStore.update_platform_id``
    overwrites in place, so re-running is safe. Best-effort per row: a short
    unregistered with the tracking store is skipped and reported, never
    fatal to the run.
    """
    conn = open_queue_db(config.seo.storage_path)
    try:
        rows = conn.execute(
            "SELECT project_name, short_index, platform, platform_upload_id, "
            "platform_post_id, published_at, published_at_source "
            "FROM deliveries WHERE status='published'"
        ).fetchall()
    finally:
        conn.close()

    store = TrackingStore()
    registered = 0
    for row in rows:
        video_id = registered_platform_id(
            row["platform"], row["platform_upload_id"], row["platform_post_id"]
        )
        if not video_id:
            continue
        try:
            # The receipt's provenance crosses with it. The queue knows
            # whether this time is a platform's confirmed second or a human's
            # late click; tracking anchors age-based judgements to it, so
            # carrying the time without the label is what let an operator's
            # catch-up click read as a publish second. Old rows here have no
            # source at all -- see ``recorded_publish_time``.
            posted_at, posted_at_source = recorded_publish_time(
                row["published_at"], row["published_at_source"]
            )
            store.update_platform_id(
                row["project_name"],
                row["short_index"],
                row["platform"],
                video_id,
                posted_at=posted_at,
                posted_at_source=posted_at_source,
            )
            registered += 1
        except ValueError as exc:
            print(f"  skip: {exc}", file=sys.stderr)

    print(f"Backfilled {registered} platform id(s) from {len(rows)} published row(s)")
    return 0
