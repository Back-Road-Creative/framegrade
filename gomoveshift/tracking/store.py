"""
Tracking Store — persists and retrieves tracking records as JSON files.

Each project gets a single JSON file: <root>/<project>/tracking/<project>.json
No database needed — simple file-based storage.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..paths import gms_output_dir
from .observation import MetricObservation
from .schema import (
    ContentMeta,
    LongFormRecord,
    ShortRecord,
    TrackingRecord,
)

logger = logging.getLogger(__name__)


def append_observation(
    record: TrackingRecord, short_index: int, observation: MetricObservation
) -> bool:
    """Append one :class:`MetricObservation` to a short's per-platform
    series (``ShortRecord.observation_log``) rather than overwriting it.

    Idempotent: replaying the identical fetch -- same platform, post id and
    ``fetched_at`` -- is a no-op, so a re-run collection can never
    double-count a reading it already recorded. Returns ``True`` when a new
    entry was appended, ``False`` when it was already present.

    Raises ``ValueError`` if no short with ``short_index`` exists -- the
    caller asked to record against a post this record doesn't have.
    """
    short = next((s for s in record.shorts if s.index == short_index), None)
    if short is None:
        raise ValueError(f"no short with index {short_index} in project {record.project!r}")
    series = short.observation_log.setdefault(observation.platform, [])
    key = (observation.post_id, observation.fetched_at)
    for existing in series:
        if (existing.get("post_id"), existing.get("fetched_at")) == key:
            return False
    series.append(observation.to_dict())
    return True


class TrackingStore:
    """
    File-based store for effectiveness tracking data.

    Directory layout:
        <root>/<project>/tracking/<project>.json

    Each file is a full TrackingRecord serialised as JSON.
    """

    def __init__(self, root: Optional[Path] = None):
        """
        Args:
            root: Base output directory. Each project gets a sub-directory.
                  Defaults to ``gomoveshift.paths.gms_output_dir()``, resolved
                  at construction time (not import time) so a test's
                  ``monkeypatch.setenv("GMS_OUTPUT_DIR", ...)`` — and a real
                  operator's env — always take effect. A hardcoded host path
                  here is a root nothing writes to.
        """
        self.root = Path(root) if root else gms_output_dir()

    # ── Path helpers ─────────────────────────────────────────────────────

    def _tracking_dir(self, project: str) -> Path:
        return self.root / project / "tracking"

    def _record_path(self, project: str) -> Path:
        return self._tracking_dir(project) / f"{project}.json"

    # ── CRUD ─────────────────────────────────────────────────────────────

    def exists(self, project: str) -> bool:
        """Check if a tracking record exists for a project."""
        return self._record_path(project).exists()

    def load(self, project: str) -> TrackingRecord:
        """
        Load a tracking record. Raises FileNotFoundError if it doesn't exist.
        """
        path = self._record_path(project)
        if not path.exists():
            raise FileNotFoundError(f"No tracking record at {path}")
        return TrackingRecord.load(path)

    def save(self, record: TrackingRecord) -> Path:
        """
        Save a tracking record. Creates directories as needed.
        Makes a backup of the previous version before overwriting.
        """
        path = self._record_path(record.project)

        # Backup existing file
        if path.exists():
            backup_dir = self._tracking_dir(record.project) / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / f"{record.project}_{ts}.json"
            # copyfile, not copy2: the output root commonly lives on a drvfs
            # mount under /mnt/<letter>, where copystat() (utime) raises
            # PermissionError and the refreshed record would then never be
            # written in a scheduled metrics run. A recovery backup needs
            # the bytes, not timestamps or mode bits (same reasoning as
            # pipeline/redrive_ost.py).
            shutil.copyfile(path, backup_path)
            logger.debug("Backed up previous record to %s", backup_path)

        record.save(path)
        logger.info("Saved tracking record to %s", path)
        return path

    def load_or_create(self, project: str, location: str = "") -> TrackingRecord:
        """Load existing record or create a fresh one."""
        if self.exists(project):
            return self.load(project)
        record = TrackingRecord(project=project, location=location)
        logger.info("Created new tracking record for %s", project)
        return record

    def delete(self, project: str) -> bool:
        """Delete a tracking record (moves to backups)."""
        path = self._record_path(project)
        if not path.exists():
            return False

        backup_dir = self._tracking_dir(project) / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"{project}_deleted_{ts}.json"
        shutil.move(str(path), str(backup_path))
        logger.info("Deleted (backed up to %s)", backup_path)
        return True

    def list_projects(self) -> List[str]:
        """List all projects that have tracking records."""
        projects = []
        if not self.root.exists():
            return projects
        for project_dir in sorted(self.root.iterdir()):
            if project_dir.is_dir():
                record_path = project_dir / "tracking" / f"{project_dir.name}.json"
                if record_path.exists():
                    projects.append(project_dir.name)
        return projects

    # ── Short Management ─────────────────────────────────────────────────

    def register_short(
        self,
        project: str,
        index: int,
        file: str = "",
        posted_at: Optional[str] = None,
        posted_at_source: Optional[str] = None,
        platform_ids: Optional[Dict[str, str]] = None,
        hook: str = "",
        cta: str = "",
        weather: str = "",
        duration_sec: float = 0.0,
        title_format: str = "",
        tags: Optional[List[str]] = None,
        location: str = "",
    ) -> ShortRecord:
        """
        Register a newly published short in the tracking store.

        If a short with the same index exists, it is updated (not duplicated).

        Args:
            project: Project name
            index: Short index (1-based)
            file: Filename of the short
            posted_at: ISO 8601 posting time
            posted_at_source: which moment ``posted_at`` is -- one of the
                queue's ``published_at_source`` values. Required whenever
                ``posted_at`` is given; see ``ShortRecord.__post_init__``.
            platform_ids: {"youtube": "video_id", "tiktok": "video_id", ...}
            hook: Opening text / location hook
            cta: Call to action text
            weather: Weather condition
            duration_sec: Short duration in seconds
            title_format: Title format used
            tags: List of tags/hashtags used
            location: Location string (used if creating new record)

        Returns:
            The created/updated ShortRecord
        """
        record = self.load_or_create(project, location=location)

        # Find existing short by index, or create new
        existing = None
        for s in record.shorts:
            if s.index == index:
                existing = s
                break

        content = ContentMeta(
            hook=hook,
            cta=cta,
            weather=weather,
            duration_sec=duration_sec,
            title_format=title_format,
            tags=tags or [],
        )

        if existing:
            # Update existing
            existing.file = file or existing.file
            # Moved together, always. Assigning through the instance skips
            # ``__post_init__``, so pairing them here is what keeps a time
            # from losing its provenance on an update.
            if posted_at:
                existing.posted_at = posted_at
                existing.posted_at_source = posted_at_source
            if platform_ids:
                existing.platform_ids.update(platform_ids)
            existing.content = content
            short = existing
            logger.info("Updated short #%d in project %s", index, project)
        else:
            short = ShortRecord(
                index=index,
                file=file,
                posted_at=posted_at,
                posted_at_source=posted_at_source,
                platform_ids=platform_ids or {},
                content=content,
            )
            record.shorts.append(short)
            # Keep sorted by index
            record.shorts.sort(key=lambda s: s.index)
            logger.info("Registered short #%d in project %s", index, project)

        self.save(record)
        return short

    def register_shorts_batch(
        self,
        project: str,
        shorts_data: List[Dict],
        location: str = "",
    ) -> int:
        """
        Register multiple shorts at once.

        Args:
            project: Project name
            shorts_data: List of dicts with keys matching register_short params
            location: Project location

        Returns:
            Number of shorts registered
        """
        record = self.load_or_create(project, location=location)
        count = 0

        for data in shorts_data:
            index = data.get("index", count + 1)

            content = ContentMeta(
                hook=data.get("hook", ""),
                cta=data.get("cta", ""),
                weather=data.get("weather", ""),
                duration_sec=data.get("duration_sec", 0.0),
                title_format=data.get("title_format", ""),
                tags=data.get("tags", []),
            )

            # Find or create
            existing = next((s for s in record.shorts if s.index == index), None)
            if existing:
                existing.file = data.get("file", existing.file)
                if data.get("posted_at"):
                    existing.posted_at = data["posted_at"]
                    existing.posted_at_source = data.get("posted_at_source")
                if data.get("platform_ids"):
                    existing.platform_ids.update(data["platform_ids"])
                existing.content = content
            else:
                short = ShortRecord(
                    index=index,
                    file=data.get("file", ""),
                    posted_at=data.get("posted_at"),
                    posted_at_source=data.get("posted_at_source"),
                    platform_ids=data.get("platform_ids", {}),
                    content=content,
                )
                record.shorts.append(short)

            count += 1

        record.shorts.sort(key=lambda s: s.index)
        self.save(record)
        logger.info("Registered %d shorts for project %s", count, project)
        return count

    def update_platform_id(
        self,
        project: str,
        short_index: int,
        platform: str,
        video_id: str,
        posted_at: Optional[str] = None,
        posted_at_source: Optional[str] = None,
    ) -> None:
        """Add or update a platform ID for a specific short, and the posting
        time it was published at.

        ``posted_at``/``posted_at_source`` move together or not at all: this is
        the one bridge from the delivery queue into tracking, and it is where
        the receipt's provenance either survives or is dropped. Passing a time
        with no source is refused rather than stored as an unqualified string,
        because an unqualified one is indistinguishable from a legacy record's
        honest unknown.
        """
        if posted_at and not posted_at_source:
            raise ValueError(
                f"refusing to record a posting time for short #{short_index} in "
                f"{project} with no provenance: pass posted_at_source, or neither"
            )
        record = self.load(project)
        for short in record.shorts:
            if short.index == short_index:
                short.platform_ids[platform] = video_id
                if posted_at:
                    short.posted_at = posted_at
                    short.posted_at_source = posted_at_source
                self.save(record)
                logger.info(
                    "Updated %s ID for short #%d in %s: %s",
                    platform,
                    short_index,
                    project,
                    video_id,
                )
                return
        raise ValueError(f"Short #{short_index} not found in project {project}")

    # ── Long-form Management ─────────────────────────────────────────────

    def register_long_form(
        self,
        project: str,
        video_id: str,
        title: str = "",
        published_date: Optional[str] = None,
        location: str = "",
        index: Optional[int] = None,
    ) -> LongFormRecord:
        """Register (or update) a long-form in a project, keyed on ``video_id``.

        ``video_id`` is the identity. An existing record with that id is
        updated in place (its index, duration and metrics are kept). A new id
        gets ``index`` when that slot is free, else ``max(existing)+1`` -- it
        never takes over another record's slot, so nothing already collected
        is overwritten. Duration and metrics are NOT set here: they come from
        the platform at collection time (``MetricsCollector.collect_long_forms``).
        """
        if not video_id:
            raise ValueError("register_long_form requires a video_id")
        record = self.load_or_create(project, location=location)
        lf = next((x for x in record.long_forms if x.video_id == video_id), None)
        if lf is None:
            taken = {x.index for x in record.long_forms}
            if index is None or index in taken:
                index = max(taken, default=0) + 1
            lf = LongFormRecord(index=index, video_id=video_id)
            record.long_forms.append(lf)
            record.long_forms.sort(key=lambda x: x.index)
        lf.title = title or lf.title
        lf.published_date = published_date or lf.published_date
        self.save(record)
        return lf

    def sync_long_forms_from_ledger(self, entries: Iterable[Any]) -> int:
        """Register every confirmed long-form in the content ledger.

        ``entries`` are ledger ``ContentEntry`` objects (duck-typed so this
        package does not import the pipeline). Only ``longform`` entries with
        a ``video_id`` and a ``project_name`` count -- a ledger entry exists
        only once an upload was confirmed. The LF index is the entry's 1-based
        position by date within its project (the ledger stores no index) and is
        only a preference: identity is the ``video_id``, a taken slot is never
        reused, so a backfilled earlier entry cannot displace a known one.
        Idempotent: returns the number of NEW video ids recorded.
        """
        by_project: Dict[str, List[Any]] = {}
        for e in entries:
            if e.content_type == "longform" and e.video_id and e.project_name:
                by_project.setdefault(e.project_name, []).append(e)
        added = 0
        for project, items in by_project.items():
            items.sort(key=lambda e: (e.date, e.time))
            known = set()
            if self.exists(project):
                known = {lf.video_id for lf in self.load(project).long_forms}
            for i, e in enumerate(items, start=1):
                if e.video_id in known:
                    continue
                self.register_long_form(
                    project, e.video_id, title=e.title, published_date=e.date, index=i
                )
                added += 1
        return added

    # ── Aggregate Queries ────────────────────────────────────────────────

    def load_all_records(self) -> List[TrackingRecord]:
        """Load all tracking records across all projects."""
        records = []
        for project in self.list_projects():
            try:
                records.append(self.load(project))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", project, exc)
        return records

    def get_all_shorts(self) -> List[ShortRecord]:
        """Get all shorts across all projects (for cross-project analysis)."""
        shorts = []
        for record in self.load_all_records():
            shorts.extend(record.shorts)
        return shorts
