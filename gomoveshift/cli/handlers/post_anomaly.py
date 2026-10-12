"""``post-anomaly`` — shadow-only per-post dip/spike report over already-collected metrics.

Read-only, always exits 0: thin data and a policy-infeasible epoch are the normal case and a cron
must not go red for them. Every printed line carries ``shadow=true`` — nothing reads these lines
to act."""

from __future__ import annotations

from datetime import datetime, timezone

from seo_manager.analytics.anomaly_stats import Refusal

from ...tracking.anomaly import LOOKBACK_DAYS, check_feasibility, detect_post_anomaly
from ...tracking.decline import platforms_seen
from ...tracking.store import TrackingStore  # module scope, not lazy, so tests can patch it
from ..logging_setup import configure_logging


def _run_marker(ts: str, outcome: str) -> str:
    return f"POST_ANOMALY_RUN ts={ts} window={LOOKBACK_DAYS} outcome={outcome}"


def cmd_post_anomaly(args, config) -> int:
    """Print one ``POST_ANOMALY``/``POST_ANOMALY_REFUSED`` line per candidate/refusal, plus
    exactly one ``POST_ANOMALY_RUN`` marker per invocation — the cron freshness receipt."""
    configure_logging()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        store = TrackingStore()
        records = store.load_all_records()
        platforms = platforms_seen(records)
        if not platforms:
            print("post-anomaly: insufficient data — no collected metrics yet")
            print(_run_marker(ts, "no-candidate"))
            return 0

        feasible = check_feasibility()
        if isinstance(feasible, Refusal):
            print(f"POST_ANOMALY_INFEASIBLE reason={feasible.reason} shadow=true")
            print(_run_marker(ts, "refused"))
            return 0

        found = 0
        for platform in platforms:
            result = detect_post_anomaly(records, platform, feasible.allocated_alpha)
            if result is None:
                continue
            print(result.log_line())
            found += 1
        if not found:
            print(f"post-anomaly: no candidate over {len(platforms)} platform(s)")
        print(_run_marker(ts, "ok"))
        return 0
    except Exception:
        print(_run_marker(ts, "error"))
        raise
