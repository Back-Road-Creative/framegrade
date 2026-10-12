"""``platform-decline`` — report per-platform decline in day-7 views.

Read-only, always exits 0: thin data is the normal case and a cron must not go red for it."""

from __future__ import annotations

from ...tracking.decline import detect_platform_decline, platforms_seen
from ...tracking.store import TrackingStore  # module scope, not lazy, so tests can patch it
from ..logging_setup import configure_logging


def cmd_platform_decline(args, config) -> int:
    """Print one ``PLATFORM_DECLINE`` line per declining platform."""
    configure_logging()
    window, drop = getattr(args, "window", 14), getattr(args, "min_drop_pct", 20.0)
    store = TrackingStore()
    records = store.load_all_records()
    platforms = platforms_seen(records)
    if not platforms:
        print("platform-decline: insufficient data — no collected metrics yet")
        return 0
    found = 0
    for platform in platforms:
        finding = detect_platform_decline(records, platform, window, drop, store)
        if finding is not None:
            print(finding.log_line())
            found += 1
    if not found:
        print(f"platform-decline: no decline over {len(platforms)} platform(s) (window={window})")
    return 0
