"""Rule 7, the fast guard: a bad Shorts run is named in days, not weeks.

When the newest ``GUARD_N`` settled YouTube Shorts each fall under ``LOW_FRACTION`` of the
channel's trailing median day-7 views, one ``short_bad_run`` incident lists every Short in the
run with the causes a fixed sweep finds in recorded data. Detection only: nothing here edits a
queue row, a plan or a tracking record.

Settled is ``tracking.day7.window_settled`` (posted + 7 d <= today - 2 d), and a Short's views
are ``settled_day7_views`` over its YouTube series, so the guard and the day-7 reports agree. The
channel is every tracked project: Shorts are ordered by ``posted_at``.

Cause sweep (``causes`` per Short, each ``{code, detail, c5_fixable}``):

- ``delivery_status``: the queue's newest non-shadow YouTube delivery is not ``published``.
  YouTube's own processing or privacy state is not recorded anywhere, so it is not swept.
- ``duplicate_place_text``: the Short's place text (``ContentMeta.hook``) equals, after case and
  whitespace folding, an earlier Short's in the same project. ``c5_fixable`` is True: unit C2's
  composer already fixes it for new plans, so a later unit may re-write a queued row.
- ``late_publish``: queue ``published_at`` is over ``LATE_AFTER_H`` after ``scheduled_at``. (The
  queue column is ``published_at``; there is no ``platform_published_at``.)
- ``bunched_publish``: another tracked Short was posted within ``BUNCH_WITHIN_H`` of this one.
- ``place_county_mismatch``: the sidecar's ``sampled_regions`` carry a county and the place text
  names a county whose last word is not among theirs. Skipped when the sidecar has no county.
- ``missing_core_metrics``: no settled day-7 series, or a ``CORE_METRICS`` field never reported.
  A Short with no series counts as low (the API omits zero-view days, so a dead Short has none).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..tracking.day7 import posted_date, settled_day7_views, window_settled
from .store import new_incident

KIND = "short_bad_run"
#: Consecutive settled Shorts that must all be low. The measured base rate is 4 of the last 10
#: Shorts under 5 views; 3 in a row trips it while one good Short in the window still breaks it.
GUARD_N = 3
#: "Low" is under this fraction of the trailing median: 0.25 of a median of a few dozen views is
#: well above 5, so the base-rate Shorts are low, while a normal weak Short (half the median) is not.
LOW_FRACTION = 0.25
#: The settled Shorts older than the run that form the trailing median, and the fewest measured
#: ones it needs: with less history the guard is not evaluated, never guessed.
TRAILING_N = 10
MIN_BASELINE = 5
LATE_AFTER_H = 3
BUNCH_WITHIN_H = 1
CORE_METRICS = ("views", "watch_time_seconds", "avg_view_duration_seconds")

Cause = Dict[str, Any]
QueueRows = Dict[Tuple[str, int], Dict[str, Any]]


def _moment(text: Any) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _fold(text: str) -> str:
    return " ".join((text or "").casefold().split())


def load_queue_rows(db_path: Path) -> QueueRows:
    """``(project, short_index) -> newest non-shadow YouTube delivery``, opened read-only."""
    if not db_path.is_file():
        return {}
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        found = conn.execute(
            "SELECT project_name, short_index, status, error_class, scheduled_at, published_at,"
            " sidecar_path FROM deliveries WHERE platform = 'youtube' AND shadow = 0 ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return {(r["project_name"], int(r["short_index"])): dict(r) for r in found}


def sidecar_regions(sidecar_path: Any, short_index: int) -> List[str]:
    """``sampled_regions`` of one Short in a publish-plan sidecar; ``[]`` when unreadable."""
    try:
        raw = json.loads(Path(str(sidecar_path)).read_text(encoding="utf-8"))
        for entry in raw["shorts"]:
            if entry.get("short_index") == short_index:
                return [r for r in entry.get("sampled_regions") or [] if isinstance(r, str)]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    return []


def _county_tails(text: str) -> set:
    from ..pipeline.validation.content_gates import _county_names

    # last word only: the extractor can swallow a verb phrase ahead of the name
    return {n.split()[-1] for n in _county_names(text) if n.split()}


def _causes(
    project: str,
    short: Any,
    earlier_hooks: set,
    queue: Optional[Dict[str, Any]],
    regions: List[str],
    others: List[datetime],
    views: Optional[int],
    today: date,
) -> List[Cause]:
    out: List[Cause] = []

    def add(code: str, detail: str, fixable: bool = False) -> None:
        out.append({"code": code, "detail": detail, "c5_fixable": fixable})

    if queue and queue.get("status") != "published":
        err = f" ({queue['error_class']})" if queue.get("error_class") else ""
        add("delivery_status", f"queue delivery is {queue.get('status')}{err}, not published")
    hook = _fold(short.content.hook)
    if hook and hook in earlier_hooks:
        add(
            "duplicate_place_text",
            f"place text {short.content.hook!r} repeats an earlier Short",
            True,
        )
    sched = _moment(queue.get("scheduled_at")) if queue else None
    pub = _moment(queue.get("published_at")) if queue else None
    if sched and pub and pub - sched > timedelta(hours=LATE_AFTER_H):
        add("late_publish", f"published {pub - sched} after its {sched.isoformat()} slot")
    posted = _moment(short.posted_at)
    if posted and any(abs(o - posted) < timedelta(hours=BUNCH_WITHIN_H) for o in others):
        add(
            "bunched_publish",
            f"another Short posted within {BUNCH_WITHIN_H} h of {posted.isoformat()}",
        )
    measured = _county_tails(" ".join(regions))
    foreign = _county_tails(short.content.hook) - measured
    if measured and foreign:
        add(
            "place_county_mismatch",
            f"place text names {sorted(foreign)}; the sidecar measured {sorted(measured)}",
        )
    yt = short.metrics.get("youtube")
    gaps = [m for m in CORE_METRICS if yt is None or getattr(yt, m) is None]
    if views is None:
        gaps.append("day-7 series")
    if gaps:
        add("missing_core_metrics", f"not recorded: {', '.join(gaps)}")
    return out


def bad_run_incidents(
    records: List[Any],
    queue_rows: QueueRows,
    regions_of: Callable[[str, int], List[str]],
    today: date,
    now_iso: str,
) -> List[Dict[str, Any]]:
    """The rule: ``[]`` or one ``short_bad_run`` incident (key ``short_bad_run``)."""
    tracked: List[Tuple[datetime, str, Any]] = []
    for rec in records:
        for short in rec.shorts:
            moment = _moment(short.posted_at)
            if moment and short.platform_ids.get("youtube"):
                tracked.append((moment, rec.project, short))
    tracked.sort(key=lambda t: (t[0], t[1], t[2].index), reverse=True)
    settled: List[Tuple[datetime, str, Any, Optional[int]]] = []
    for moment, project, short in tracked:
        posted = posted_date(short.posted_at)
        if posted is None or not window_settled(posted, today):
            continue
        yt = short.metrics.get("youtube")
        views = settled_day7_views(short.posted_at, yt.daily, today) if yt else None
        settled.append((moment, project, short, views))
    window, older = settled[:GUARD_N], settled[GUARD_N:]
    if len(window) < GUARD_N:
        return []
    baseline = [v for *_x, v in older[:TRAILING_N] if v is not None]
    if len(baseline) < MIN_BASELINE:
        return []
    med = float(median(baseline))
    threshold = med * LOW_FRACTION
    if any(v is not None and v >= threshold for *_x, v in window):
        return []
    instants = [t[0] for t in tracked]
    shorts: List[Dict[str, Any]] = []
    for moment, project, short, views in window:
        earlier = {
            _fold(s.content.hook)
            for r in records
            if r.project == project
            for s in r.shorts
            if s is not short
            and (s.index, s.posted_at or "") < (short.index, short.posted_at or "")
        }
        others = [o for o in instants if o != moment]
        queue = queue_rows.get((project, short.index))
        causes = _causes(
            project, short, earlier, queue, regions_of(project, short.index), others, views, today
        )
        shorts.append(
            {
                "project": project,
                "index": short.index,
                "youtube_id": short.platform_ids.get("youtube"),
                "posted_at": short.posted_at,
                "views": views,
                "causes": causes,
            }
        )
    return [
        new_incident(
            KIND,
            KIND,
            f"{KIND}.json",
            now_iso,
            n=GUARD_N,
            median_views=med,
            threshold_views=threshold,
            shorts=shorts,
        )
    ]


def bad_run_rows(
    storage: Path, tracking_root: Optional[Path], today: date, now_iso: str
) -> List[Dict[str, Any]]:
    """Load tracking records and the queue, then run the rule. Read-only."""
    from ..tracking.store import TrackingStore

    records = TrackingStore(tracking_root).load_all_records()
    queue = load_queue_rows(storage / "social_queue.sqlite3")

    def regions_of(project: str, index: int) -> List[str]:
        row = queue.get((project, index))
        return (
            sidecar_regions(row["sidecar_path"], index) if row and row.get("sidecar_path") else []
        )

    return bad_run_incidents(records, queue, regions_of, today, now_iso)
