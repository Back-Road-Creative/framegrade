"""Weekly growth review: which formats are falling behind, and which posts went wrong.

The nightly collectors store how every post did, and the hourly health check
heals failed drives. Neither says, in one place, "this kind of post is not
working" or "this post never went out". This module does, from what is
already stored. It is suggest-only: it reads the tracking store and the
social queue (opened read-only) and prints lines. It changes no setting,
cadence or post.

Two parts:

* :func:`format_scorecard` groups settled day-7 views by platform and a
  coarse format (clip length band, and the tracked place), compares each
  group's median in the last ``WINDOW_DAYS`` with the platform's median, and
  flags a group that sits far below it.
* :func:`post_problems` reads the queue for posts in the last ``WINDOW_DAYS``
  that failed for good, are past their slot and still not out, or went out
  twice to the same platform.

The two numbers below are a stated starting policy, not measured values.
Change them here; nothing else carries a copy.
"""

from __future__ import annotations

import sqlite3
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .day7 import posted_date, settled_day7_views

#: Days a review looks back over.
WINDOW_DAYS = 28
#: A group needs at least this many settled posts to be judged at all.
MIN_GROUP_POSTS = 5
#: A group whose median is below this fraction of its platform's median is flagged.
UNDERPERFORM_RATIO = 0.5
#: A queued post this many hours past its slot and still not out is stuck.
STUCK_AFTER_H = 24

_LENGTH_BANDS: Tuple[Tuple[float, str], ...] = (
    (15.0, "under15s"),
    (30.0, "15to30s"),
    (60.0, "30to60s"),
)


def length_band(seconds: Optional[float]) -> str:
    if not seconds or seconds <= 0:
        return "length_unknown"
    for upper, label in _LENGTH_BANDS:
        if seconds < upper:
            return label
    return "over60s"


@dataclass(frozen=True)
class Sample:
    platform: str
    groups: Tuple[str, ...]  # e.g. ("length:15to30s", "place:Donegal")
    day7_views: int


@dataclass(frozen=True)
class GroupScore:
    platform: str
    group: str
    n: int
    median: float
    platform_median: float

    @property
    def ratio(self) -> Optional[float]:
        return None if not self.platform_median else self.median / self.platform_median

    @property
    def underperforming(self) -> bool:
        r = self.ratio
        return self.n >= MIN_GROUP_POSTS and r is not None and r < UNDERPERFORM_RATIO

    def line(self) -> str:
        tag = "GROWTH_UNDERPERFORMER" if self.underperforming else "GROWTH_FORMAT"
        ratio = "undefined" if self.ratio is None else f"{self.ratio:.2f}"
        return (
            f"{tag} platform={self.platform} group={self.group} n={self.n} "
            f"median_day7={self.median:.0f} platform_median={self.platform_median:.0f} "
            f"ratio={ratio}"
        )


def music_grouper(test: Any, receipts: set) -> Callable[[str, int, str, date], Optional[str]]:
    """The ``music:on`` / ``music:off`` group of a Short on a platform in the
    licensed-music test (``publishing/shorts_music.py``), or ``None``.

    A Short in the music half counts as ``music:on`` only when a receipt shows
    it really went out with a track; one that posted muted after all (no track,
    a refused copy) is left out of both groups, never counted as silent."""
    from ..publishing.shorts_music import music_arm

    def group(project: str, index: int, platform: str, posted: date) -> Optional[str]:
        if test is None or platform not in test.platforms or not test.is_active(posted):
            return None
        if not music_arm(project, index):
            return "music:off"
        return "music:on" if (project, index, platform) in receipts else None

    return group


#: Project-name prefix the day-reel route gives every reel it queues
#: (``publishing.day_reels``: ``day-reel-<drive file id>``).
DAY_REEL_PREFIX = "day-reel-"


def source_group(project: str) -> str:
    """``source:day-reel`` for a reel cut from the day's own phone clip,
    ``source:drive`` for a Short cut from a recorded drive, so the review can
    compare the two."""
    return "source:day-reel" if project.startswith(DAY_REEL_PREFIX) else "source:drive"


def samples_from_records(
    records: Iterable[Any],
    today: date,
    music: Optional[Callable[[str, int, str, date], Optional[str]]] = None,
) -> List[Sample]:
    """One sample per (short, platform) with a settled day-7 figure posted in
    the window. A short with no settled figure is left out, never counted as 0.
    ``music`` (see :func:`music_grouper`) adds the Short's music-test group."""
    since = today - timedelta(days=WINDOW_DAYS + 9)  # 7-day window + 2-day settle lag
    out: List[Sample] = []
    for record in records:
        place = (getattr(record, "location", "") or "").strip()
        source = source_group(getattr(record, "project", "") or "")
        for short in getattr(record, "shorts", []) or []:
            posted = posted_date(short.posted_at)
            if posted is None or posted < since:
                continue
            band = length_band(getattr(short.content, "duration_sec", None))
            groups = [f"length:{band}", source]
            if place:
                groups.append(f"place:{place}")
            for platform, metrics in (short.metrics or {}).items():
                views = settled_day7_views(short.posted_at, getattr(metrics, "daily", []), today)
                if views is None:
                    continue
                extra = music(record.project, short.index, platform, posted) if music else None
                out.append(Sample(platform, tuple(groups) + ((extra,) if extra else ()), views))
    return out


def format_scorecard(samples: List[Sample]) -> List[GroupScore]:
    by_platform: Dict[str, List[int]] = defaultdict(list)
    by_group: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for s in samples:
        by_platform[s.platform].append(s.day7_views)
        for g in s.groups:
            by_group[(s.platform, g)].append(s.day7_views)
    scores = [
        GroupScore(
            platform,
            group,
            len(values),
            float(statistics.median(values)),
            float(statistics.median(by_platform[platform])),
        )
        for (platform, group), values in by_group.items()
    ]
    # Worst first, so a reader sees the problem before the rest.
    return sorted(scores, key=lambda s: (s.platform, s.ratio if s.ratio is not None else 1e9))


def _when(text: Optional[str]) -> Optional[datetime]:
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class PostProblem:
    kind: str  # failed | stuck | duplicate
    platform: str
    project: str
    short_index: int
    delivery_ids: Tuple[int, ...]
    detail: str

    def line(self) -> str:
        ids = ",".join(str(i) for i in self.delivery_ids)
        return (
            f"POST_{self.kind.upper()} platform={self.platform} project={self.project} "
            f"short={self.short_index} deliveries={ids} {self.detail}"
        )


def post_problems(db_path: Path, now: datetime) -> List[PostProblem]:
    """Problems in the last ``WINDOW_DAYS``, from the queue opened read-only.
    Shadow rows are rehearsals and never count. A missing queue is no problem
    found, not a crash: the caller prints that it had no queue."""
    since = now - timedelta(days=WINDOW_DAYS)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, project_name, short_index, platform, status, error_class, "
            "scheduled_at, published_at, platform_post_id, platform_upload_id, updated_at "
            "FROM deliveries WHERE shadow = 0"
        ).fetchall()
    finally:
        conn.close()

    out: List[PostProblem] = []
    published: Dict[Tuple[str, int, str], List[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        updated = _when(row["updated_at"])
        slot = _when(row["scheduled_at"])
        key = (row["project_name"], int(row["short_index"]), row["platform"])
        if row["status"] == "published":
            when = _when(row["published_at"]) or updated
            if when and when >= since:
                published[key].append(row)
            continue
        if row["status"] == "failed_permanent" and updated and updated >= since:
            out.append(
                PostProblem(
                    "failed",
                    row["platform"],
                    key[0],
                    key[1],
                    (int(row["id"]),),
                    f"error={row['error_class'] or 'unstated'}",
                )
            )
        elif (
            row["status"] not in ("cancelled", "missing_remote", "failed_permanent")
            and slot is not None
            and since <= slot <= now - timedelta(hours=STUCK_AFTER_H)
        ):
            out.append(
                PostProblem(
                    "stuck",
                    row["platform"],
                    key[0],
                    key[1],
                    (int(row["id"]),),
                    f"status={row['status']} slot={slot.isoformat()}",
                )
            )
    for key, group in published.items():
        if len(group) > 1:
            out.append(
                PostProblem(
                    "duplicate",
                    key[2],
                    key[0],
                    key[1],
                    tuple(int(r["id"]) for r in group),
                    f"published={len(group)}x",
                )
            )
    return sorted(out, key=lambda p: (p.kind, p.platform, p.project, p.short_index))


def summary_line(scores: List[GroupScore], problems: List[PostProblem]) -> str:
    flagged = sum(1 for s in scores if s.underperforming)
    kinds = defaultdict(int)
    for p in problems:
        kinds[p.kind] += 1
    return (
        f"GROWTH_REVIEW window_days={WINDOW_DAYS} groups={len(scores)} underperforming={flagged} "
        f"failed={kinds['failed']} stuck={kinds['stuck']} duplicate={kinds['duplicate']}"
    )
