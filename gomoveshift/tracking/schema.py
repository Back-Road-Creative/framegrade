"""
Data models for effectiveness tracking records.

Defines the canonical schema for tracking shorts performance alongside
content metadata (hook text, posting time, weather, OST).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Slot performance aggregate (SQLite): one CREATE TABLE IF NOT EXISTS,
# applied lazily by ensure_slot_performance_schema on every connection.
SLOT_PERFORMANCE_SCHEMA = """
CREATE TABLE IF NOT EXISTS slot_performance (
    region TEXT NOT NULL,
    platform TEXT NOT NULL,
    weekday INTEGER NOT NULL,
    hour_local INTEGER NOT NULL,
    n INTEGER NOT NULL DEFAULT 0,
    mean_views REAL NOT NULL DEFAULT 0.0,
    mean_engagement REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (region, platform, weekday, hour_local)
);
"""


def ensure_slot_performance_schema(conn: sqlite3.Connection) -> None:
    """Create the ``slot_performance`` table if it doesn't exist yet."""
    conn.executescript(SLOT_PERFORMANCE_SCHEMA)


# Slot performance history: a snapshot of ``slot_performance`` is
# APPENDED here on every fold, keyed additionally by ``folded_at`` -- unlike
# ``slot_performance`` itself, which is deleted and recomputed each fold
# (see ``_rebuild_slot_performance``), this table is never deleted from, so
# ``metrics best-times --trend`` can show how a slot moved over the last few
# folds instead of only its latest, overwritten aggregate.
SLOT_PERFORMANCE_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS slot_performance_history (
    folded_at TEXT NOT NULL,
    region TEXT NOT NULL,
    platform TEXT NOT NULL,
    weekday INTEGER NOT NULL,
    hour_local INTEGER NOT NULL,
    n INTEGER NOT NULL DEFAULT 0,
    mean_views REAL NOT NULL DEFAULT 0.0,
    mean_engagement REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (folded_at, region, platform, weekday, hour_local)
);
"""


def ensure_slot_performance_history_schema(conn: sqlite3.Connection) -> None:
    """Create the ``slot_performance_history`` table if it doesn't exist yet."""
    conn.executescript(SLOT_PERFORMANCE_HISTORY_SCHEMA)


@dataclass
class DailyMetricPoint:
    """One day's slice of a video's analytics (dimensions=day query row).

    ``day`` is the row's identity (which slice this is), always present.
    Every measured field is ``Optional[...] = None`` for the same reason
    documented on ``PlatformMetrics`` below: None means "not reported for
    this day", a distinct state from a genuine measured 0 — this is that
    same missing/measured distinction applied one level down, to the
    per-day series instead of the platform aggregate.
    """

    day: str  # YYYY-MM-DD
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    watch_time_seconds: Optional[float] = None
    #: That day's average view duration (Analytics ``averageViewDuration``);
    #: the long-form length analysis needs it per day, so it is kept rather
    #: than dropped. None = not reported, never 0.
    avg_view_duration_seconds: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DailyMetricPoint:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class PlatformMetrics:
    """Metrics from a single platform for one short."""

    # These are CUMULATIVE, lifetime-to-date counters, never an interval: a
    # per-period change is a different quantity and lives in its own type,
    # ``observation.IntervalDelta`` (derived by ``observation.interval_between``
    # from two cumulative reads), so a delta can never be written here.
    # Optional[...] = None means "not reported by this platform" — a
    # distinct state from a genuine measured 0. asdict()/JSON round-trip
    # None as null, so "missing" survives to_dict()/from_dict() intact
    # instead of being silently written as a fabricated measured zero.
    views: Optional[int] = None
    likes: Optional[int] = None
    comments: Optional[int] = None
    shares: Optional[int] = None
    watch_time_seconds: Optional[float] = None
    avg_view_duration_seconds: Optional[float] = None
    impressions: Optional[int] = None
    click_through_rate: Optional[float] = None
    #: Lifetime Analytics ``averageViewPercentage`` (0-100). None = not reported.
    avg_view_percentage: Optional[float] = None
    daily: List[DailyMetricPoint] = field(default_factory=list)
    #: Facebook Reel starts/replays count 1 ms playback, not three-second
    #: video views. Keep them distinct so existing view trends cannot blend.
    reel_initial_plays: Optional[int] = None
    reel_replays: Optional[int] = None
    reel_total_plays: Optional[int] = None

    @property
    def engagement_rate(self) -> Optional[float]:
        """Engagement = (likes + comments + shares) / views.

        Returns None ("unknown") when views or any interaction count was
        never reported — inventing 0.0 for unknown inputs is exactly the
        fabricated-zero bug this schema exists to prevent. A genuinely
        measured 0 views is a different, well-defined case (no engagement
        is possible) and still yields 0.0.
        """
        if self.views is None or self.likes is None or self.comments is None or self.shares is None:
            return None
        if self.views == 0:
            return 0.0
        return (self.likes + self.comments + self.shares) / self.views

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PlatformMetrics:
        # Only pass known fields so extra keys don't break us
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        daily_raw = filtered.pop("daily", None) or []
        obj = cls(**filtered)
        obj.daily = [DailyMetricPoint.from_dict(d) if isinstance(d, dict) else d for d in daily_raw]
        return obj


@dataclass
class ContentMeta:
    """Content metadata for one short — what was shown, when, and in what conditions."""

    hook: str = ""  # Opening text / location hook
    cta: str = ""  # Call to action
    weather: str = ""  # rainy / overcast / sunny / snowy
    duration_sec: float = 0.0
    title_format: str = ""  # e.g. "Guess the Location — {location}"
    tags: List[str] = field(default_factory=list)
    description_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ContentMeta:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


#: The provenances a posting time may claim, spelled to match
#: ``publishing/queue.py``'s ``_PUBLISHED_AT_SOURCES`` exactly. Deliberately a
#: local literal rather than an import: ``publishing`` -> ``tracking`` is the
#: only import edge between these packages (``queue.py``, and even that one is
#: function-local), and importing back the other way would make the two
#: mutually dependent. The cross-package agreement is pinned by a TEST instead
#: -- ``tests/test_tracking_posted_at_provenance.py`` imports both and fails if
#: either list grows a member the other lacks.
_PUBLISHED_AT_SOURCES = (
    "operator_confirmed",
    "platform_reconciled",
    "platform_confirmed",
    "platform_scheduled",
)


@dataclass
class ShortRecord:
    """Complete tracking record for a single short."""

    index: int
    file: str = ""
    posted_at: Optional[str] = None  # ISO 8601
    #: WHICH moment ``posted_at`` is. The publishing side already
    #: distinguishes a human's late click from a platform's own confirmed
    #: second (``deliveries.published_at_source``), and age-based judgements
    #: here -- ``decline.py``, ``collector.py`` -- anchor to ``posted_at``, so
    #: reading the two as one fact makes a short look younger than it is.
    #: ``None`` means the provenance was never recorded, which is only true of
    #: records written before this existed; it is never backfilled by a guess.
    posted_at_source: Optional[str] = None
    platform_ids: Dict[str, str] = field(default_factory=dict)  # {"youtube": "id", ...}
    content: ContentMeta = field(default_factory=ContentMeta)
    metrics: Dict[str, PlatformMetrics] = field(default_factory=dict)  # keyed by platform
    collected_at: Optional[str] = None  # ISO 8601 — when metrics were last fetched
    notes: str = ""
    #: A SERIES of raw ``observation.MetricObservation.to_dict()``
    #: entries per platform, appended rather than overwritten -- unlike
    #: ``metrics`` above, where the newest collection replaces the last and
    #: nothing persists how a post moved between two fetches. Kept as plain
    #: dicts (not typed ``MetricObservation``) so this module never imports
    #: ``observation.py``, which already imports ``schema.py``; callers use
    #: ``store.append_observation`` to add to it. A record with no key here
    #: — every record written before this existed — must read back as
    #: "no observations recorded", never as "recorded, empty": ``from_dict``
    #: below defaults it to ``{}``, same as an absent legacy field elsewhere
    #: in this class.
    observation_log: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Refuse a posting time with no provenance, and a provenance with no
        time. This is the robust tier: the gap this closes cannot reopen by
        someone later setting ``posted_at`` and forgetting the source.

        It runs on CONSTRUCTION only. ``from_dict`` assigns both fields after
        building the record, so a legacy record already on disk still loads --
        refusing to READ data that exists would turn an honest unknown into an
        outage, and that unknown is the very thing we must not overwrite.
        """
        if self.posted_at and not self.posted_at_source:
            raise ValueError(
                f"short #{self.index} has a posted_at with no posted_at_source: a "
                "posting time that does not say which moment it is cannot be "
                "compared with one that does"
            )
        if self.posted_at_source and not self.posted_at:
            raise ValueError(
                f"short #{self.index} has posted_at_source="
                f"{self.posted_at_source!r} and no posted_at: a provenance "
                "label describing no timestamp is a claim about nothing"
            )
        if self.posted_at_source and self.posted_at_source not in _PUBLISHED_AT_SOURCES:
            raise ValueError(
                f"short #{self.index} has an unrecognized posted_at_source "
                f"{self.posted_at_source!r}; the queue issues only "
                f"{list(_PUBLISHED_AT_SOURCES)}"
            )

    # Computed helpers --------------------------------------------------------

    @property
    def total_views(self) -> int:
        # A platform with no reported views contributes nothing to the
        # cross-platform total — that's an aggregate property, distinct
        # from claiming that platform's own view count IS zero.
        return sum(m.views or 0 for m in self.metrics.values())

    @property
    def total_engagement(self) -> Optional[float]:
        """Engagement across all platforms: total interactions / total views.

        Returns None ("unknown") when NO platform ever reported a view count
        (every ``PlatformMetrics.views`` is None) — a credentials failure,
        scope denial, or a collector that never ran. Inventing 0.0 here would
        be indistinguishable in a report from a short that was seen and
        ignored. This is narrower than total_views' fold-to-zero: total_views
        is a sum, where an unreported platform can validly contribute
        nothing to the total; a rate's denominator changes what the
        numerator means, so an unknown addend can't be silently absorbed the
        same way. A genuinely measured 0 total views (at least one platform
        reported, and reported zero) is unchanged: still a well-defined 0.0.
        """
        if all(m.views is None for m in self.metrics.values()):
            return None
        total_interactions = sum(
            (m.likes or 0) + (m.comments or 0) + (m.shares or 0) for m in self.metrics.values()
        )
        total_views = self.total_views
        return total_interactions / total_views if total_views > 0 else 0.0

    @property
    def posting_hour(self) -> Optional[int]:
        """Hour of day (0-23) the short was posted, or None."""
        if not self.posted_at:
            return None
        try:
            dt = datetime.fromisoformat(self.posted_at)
            return dt.hour
        except (ValueError, TypeError):
            return None

    # Serialisation -----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "file": self.file,
            "posted_at": self.posted_at,
            "posted_at_source": self.posted_at_source,
            "platform_ids": self.platform_ids,
            "content": self.content.to_dict(),
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "collected_at": self.collected_at,
            "notes": self.notes,
            "observation_log": self.observation_log,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ShortRecord:
        metrics_raw = data.get("metrics", {})
        metrics = {}
        for platform, m in metrics_raw.items():
            if isinstance(m, dict):
                metrics[platform] = PlatformMetrics.from_dict(m)
            else:
                metrics[platform] = m

        content_raw = data.get("content", {})
        content = (
            ContentMeta.from_dict(content_raw) if isinstance(content_raw, dict) else ContentMeta()
        )

        record = cls(
            index=data.get("index", 0),
            file=data.get("file", ""),
            platform_ids=data.get("platform_ids", {}),
            content=content,
            metrics=metrics,
            collected_at=data.get("collected_at"),
            notes=data.get("notes", ""),
            observation_log=data.get("observation_log") or {},
        )
        # Assigned after construction on purpose -- see ``__post_init__``. A
        # record written before ``posted_at_source`` existed has a time and no
        # source, and it has to load as the unknown it is.
        record.posted_at = data.get("posted_at")
        record.posted_at_source = data.get("posted_at_source")
        return record


#: Where a long-form's ``duration_sec`` came from. Only one source exists today.
LONG_FORM_DURATION_SOURCE_YOUTUBE = "youtube_data_api"


@dataclass
class LongFormRecord:
    """Tracking record for one long-form video (a project may have several).

    ``duration_sec`` is None until a collection reads it from the platform;
    it is never defaulted to 0 (an unknown length is not a zero-length video),
    and ``duration_source`` says where a persisted value came from.
    """

    index: int
    video_id: str = ""
    title: str = ""
    published_date: Optional[str] = None  # YYYY-MM-DD, the ledger booking date
    duration_sec: Optional[float] = None
    duration_source: Optional[str] = None
    metrics: Dict[str, PlatformMetrics] = field(default_factory=dict)
    collected_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metrics"] = {k: v.to_dict() for k, v in self.metrics.items()}
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> LongFormRecord:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in data.items() if k in known and k != "metrics"}
        obj = cls(**kwargs)
        obj.metrics = {
            k: PlatformMetrics.from_dict(m) if isinstance(m, dict) else m
            for k, m in (data.get("metrics") or {}).items()
        }
        return obj


@dataclass
class TrackingRecord:
    """
    Top-level tracking record for a project.

    One JSON file per project, stored in the project's tracking/ directory.
    """

    project: str
    location: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    shorts: List[ShortRecord] = field(default_factory=list)
    long_forms: List[LongFormRecord] = field(default_factory=list)

    # Computed helpers --------------------------------------------------------

    @property
    def total_shorts(self) -> int:
        return len(self.shorts)

    @property
    def shorts_with_metrics(self) -> List[ShortRecord]:
        """Shorts something actually MEASURED.

        Selecting on ``collected_at`` alone is wrong: that field means "a
        collection was attempted" and is stamped on a run every platform
        refused. A refusal carries no numbers, so such a short would
        satisfy a property whose name promises metrics and be counted under
        "Total shorts with metrics" -- the same claim
        ``test_tracking_collected_nothing_is_not_collected`` refuses for the
        log line and ``collect_all`` refuses for its count.

        Use ``shorts_collection_attempted`` for the "we reached it" question;
        the refusals themselves are in each short's ``observation_log``.
        """
        return [s for s in self.shorts if s.metrics]

    @property
    def shorts_collection_attempted(self) -> List[ShortRecord]:
        """Shorts a collection run reached, whether or not anything answered.

        The distinction ``shorts_with_metrics`` used to blur: a short every
        platform refused about was attempted and measured nothing.
        """
        return [s for s in self.shorts if s.collected_at is not None]

    # Serialisation -----------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project": self.project,
            "location": self.location,
            "created_at": self.created_at,
            "shorts": [s.to_dict() for s in self.shorts],
            "long_forms": [lf.to_dict() for lf in self.long_forms],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrackingRecord:
        shorts_raw = data.get("shorts", [])
        shorts = [ShortRecord.from_dict(s) for s in shorts_raw]
        return cls(
            project=data.get("project", ""),
            location=data.get("location", ""),
            created_at=data.get("created_at", ""),
            shorts=shorts,
            long_forms=[LongFormRecord.from_dict(x) for x in data.get("long_forms") or []],
        )

    @classmethod
    def from_json(cls, text: str) -> TrackingRecord:
        return cls.from_dict(json.loads(text))

    @classmethod
    def load(cls, path: Path) -> TrackingRecord:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
