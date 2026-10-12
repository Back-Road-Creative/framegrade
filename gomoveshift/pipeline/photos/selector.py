"""Photo selection with journey rotation.

Parses Hugo frontmatter from journey markdown files and selects photos
using weighted round-robin across journeys, sequential within each journey
by day order to maintain a narrative arc for followers.

At one photo a day the pool is consumed in journey-weighted order.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import yaml

from .._atomic_write import atomic_write_json
from .cull import CullFloors, CullScore, compute_cull_score, cull_decision, hamming_distance

logger = logging.getLogger(__name__)

# Optional face detection filter
try:
    from .face_filter import should_withhold_photo as _should_withhold

    _FACE_FILTER_AVAILABLE = True
except ImportError:
    _FACE_FILTER_AVAILABLE = False

# Hugo site paths
JOURNEYS_CONTENT = "content/journeys"
IMAGES_BASE = "static/images/journeys"


def _photos_config():
    """Live photo config. Resolved per call — never at import time, so an
    operator override (config.yaml / env) can't be shadowed by import order."""
    from ...config import get_config

    return get_config().photos


def default_site_root() -> Path:
    """Hugo site checkout the journey photos are read from."""
    return _photos_config().resolve_site_root()


def default_state_path() -> Path:
    """State file tracking which photos have already been posted."""
    return _photos_config().resolve_state_path()


def __getattr__(name: str):
    """PEP 562 — keep ``DEFAULT_SITE_ROOT`` / ``DEFAULT_STATE_PATH`` readable
    by existing importers while resolving them lazily from config."""
    if name == "DEFAULT_SITE_ROOT":
        return default_site_root()
    if name == "DEFAULT_STATE_PATH":
        return default_state_path()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class CullSkip:
    """A candidate the cull gates passed over, with why (never silently dropped)."""

    photo_key: str
    reason: str
    score: CullScore


@dataclass
class PhotoSelection:
    """A single selected photo with full context."""

    journey_slug: str
    journey_title: str
    day_number: int
    day_date: str
    location: str
    filename: str  # e.g., "photo-002.jpg"
    photo_path: str  # Full path to the photo file
    locations_visited: List[str] = field(default_factory=list)
    day_summary: str = ""
    statistics: Dict = field(default_factory=dict)


@dataclass
class JourneyInfo:
    """Parsed journey metadata."""

    slug: str
    title: str
    photo_count: int
    photos: List[dict]  # [{day_number, date, location, filename}, ...]


class PhotoSelectionStateError(RuntimeError):
    """A selection-state file exists but cannot be loaded.

    The file is the posted-photos consumption ledger — silently substituting
    an empty state would reset dedup history and re-offer the entire shipped
    back catalog as new, so a corrupt/incompatible file fails closed."""


@dataclass
class PhotoSelectionState:
    """Persistent state tracking posted photos and journey rotation."""

    posted_photos: List[str] = field(default_factory=list)  # ["journey/photo-NNN.jpg", ...]
    journey_rotation_index: int = 0
    last_updated: str = ""
    face_filtered_photos: List[str] = field(default_factory=list)
    # Selected but not yet confirmed shipped. Selection places a hold here (and
    # in posted_photos, so a concurrent batch can't pick the same photo);
    # consumption is only final once publish success clears the hold.
    pending_photos: List[str] = field(default_factory=list)

    def save(self, path: Optional[Path] = None):
        path = path or default_state_path()
        # Atomic: a crash mid-write must not corrupt the posted-photos ledger.
        atomic_write_json(path, asdict(self))
        logger.info(f"Photo selection state saved: {len(self.posted_photos)} posted")

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PhotoSelectionState":
        path = path or default_state_path()
        if not path.exists():
            # Genuinely absent (first run) — absence is not corruption.
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(**data)
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            # Fail closed. A corrupt or schema-incompatible ledger silently
            # replaced by an empty state re-offers every shipped photo as new.
            raise PhotoSelectionStateError(
                f"Cannot load photo selection state {path}: {type(e).__name__}: {e}. "
                f"Fix or restore the file, or explicitly accept the ledger wipe with "
                f"--reset-selection-state (the old file is archived aside, not deleted)."
            ) from e

    @classmethod
    def reset(cls, path: Optional[Path] = None) -> Optional[Path]:
        """Explicitly accept a ledger wipe (the ``--reset-selection-state`` escape).

        Archives any existing state file aside — never deletes it, so the old
        ledger stays recoverable — and lets the next load start fresh. Returns
        the archive path, or None when no file existed."""
        path = path or default_state_path()
        if not path.exists():
            return None
        archive = path.with_name(f"{path.name}.corrupt-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        path.rename(archive)
        logger.warning("Photo selection state archived to %s — dedup ledger reset", archive)
        return archive


def _parse_frontmatter(md_path: Path) -> Optional[dict]:
    """Extract YAML frontmatter from a markdown file."""
    text = md_path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None
    try:
        return yaml.safe_load(match.group(1))
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse frontmatter in {md_path}: {e}")
        return None


def _extract_photos_from_journey(fm: dict, slug: str) -> List[dict]:
    """Extract flat list of photos from journey frontmatter."""
    photos = []
    # Frontmatter file order is not guaranteed chronological; the narrative
    # arc depends on day order, so sort explicitly (day_number, then date).
    days = sorted(
        fm.get("days", []) or [],
        key=lambda d: (d.get("day_number") or 0, str(d.get("date") or "")),
    )
    for day in days:
        day_number = day.get("day_number", 0)
        day_date = day.get("date", "")
        day_summary = day.get("summary", "")
        locations_visited = day.get("locations", [])
        statistics = day.get("statistics", {})

        for photo_group in day.get("photos", []):
            location = photo_group.get("location", "Unknown Location")
            for filename in photo_group.get("photos", []):
                photos.append(
                    {
                        "day_number": day_number,
                        "date": day_date,
                        "location": location,
                        "filename": filename,
                        "day_summary": day_summary,
                        "locations_visited": locations_visited,
                        "statistics": statistics,
                    }
                )
    return photos


def release_pending_holds(
    state_path: Path, photo_keys: Optional[Iterable[str]] = None
) -> List[str]:
    """Return held photos in the state file at ``state_path`` to the pool.

    Needs only the selection-state file, not the site checkout, so recovery of a
    dropped or failed batch works with ``photos.site_root`` unset.
    ``photo_keys=None`` releases every outstanding hold.
    """
    state = PhotoSelectionState.load(state_path)
    held = set(state.pending_photos)
    released = held if photo_keys is None else held & set(photo_keys)
    if not released:
        return []
    state.pending_photos = sorted(held - released)
    state.posted_photos = sorted(set(state.posted_photos) - released)
    state.last_updated = datetime.now().isoformat()
    state.save(state_path)
    logger.info("Released %d held photo(s) back to the pool", len(released))
    return sorted(released)


class PhotoSelector:
    """Selects photos using weighted round-robin across journeys."""

    def __init__(
        self,
        site_root: Optional[Path] = None,
        state_path: Optional[Path] = None,
        cull_floors: Optional[CullFloors] = None,
    ):
        self.cull_floors = cull_floors or CullFloors()
        self.cull_skips: List[CullSkip] = []
        self._cull_scores: Dict[str, Optional[CullScore]] = {}
        self.site_root = site_root or default_site_root()
        self.state_path = state_path or default_state_path()
        self.journeys: List[JourneyInfo] = []
        self._load_journeys()

    def _load_journeys(self):
        """Parse all journey markdown files."""
        journeys_dir = self.site_root / JOURNEYS_CONTENT
        if not journeys_dir.exists():
            logger.warning(f"Journeys directory not found: {journeys_dir}")
            return

        for md_file in sorted(journeys_dir.glob("*.md")):
            slug = md_file.stem
            fm = _parse_frontmatter(md_file)
            if not fm:
                continue

            photos = _extract_photos_from_journey(fm, slug)
            if not photos:
                continue

            # Verify at least some photos exist on disk
            images_dir = self.site_root / IMAGES_BASE / slug
            if not images_dir.exists():
                logger.warning(f"Images directory missing for {slug}: {images_dir}")
                continue

            self.journeys.append(
                JourneyInfo(
                    slug=slug,
                    title=fm.get("title", slug),
                    photo_count=len(photos),
                    photos=photos,
                )
            )

        total = sum(j.photo_count for j in self.journeys)
        logger.info(f"Loaded {len(self.journeys)} journeys with {total} photos")

    def _cull_score(self, photo_path: Path) -> Optional[CullScore]:
        """Cull score for an on-disk photo; ``None`` (gate skipped) when the
        file cannot be decoded -- an unreadable file is not evidence of a bad photo."""
        key = str(photo_path)
        if key not in self._cull_scores:
            bgr = cv2.imread(key)
            self._cull_scores[key] = (
                None if bgr is None else compute_cull_score(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            )
        return self._cull_scores[key]

    def get_total_photos(self) -> int:
        return sum(j.photo_count for j in self.journeys)

    def commit_posted(self, photo_keys: Iterable[str]) -> None:
        """Confirm consumption of photos that actually shipped.

        Called only on publish success — this is the single point at which a
        photo is permanently spent.
        """
        keys = set(photo_keys)
        if not keys:
            return
        state = PhotoSelectionState.load(self.state_path)
        state.posted_photos = sorted(set(state.posted_photos) | keys)
        state.pending_photos = sorted(set(state.pending_photos) - keys)
        state.last_updated = datetime.now().isoformat()
        state.save(self.state_path)

    def release_pending(self, photo_keys: Optional[Iterable[str]] = None) -> List[str]:
        """Return held photos to the selectable pool (dropped/failed batch).

        ``photo_keys=None`` releases every outstanding hold.
        """
        return release_pending_holds(self.state_path, photo_keys)

    def select_batch(
        self,
        count: int = 7,
        filter_faces: bool = False,
        allow_shortfall: bool = False,
    ) -> List[PhotoSelection]:
        """Select a batch of photos using weighted round-robin.

        Weights are proportional to journey photo count. Within each journey,
        photos are selected sequentially by day order.

        Args:
            count: Number of photos to select (default 7 for a week)
            filter_faces: Skip photos with detected human faces (requires OpenCV)
            allow_shortfall: Accept fewer than ``count`` photos instead of
                raising (operator escape for a genuinely exhausted pool)

        Returns:
            List of PhotoSelection objects

        Raises:
            RuntimeError: Fewer than ``count`` photos are available and
                ``allow_shortfall`` is False.
        """
        self.cull_skips = []
        if count < 1:
            # Also enforced at argparse; guard programmatic callers too — an
            # empty selection from a bad count must never look like exhaustion.
            raise ValueError(f"count must be >= 1 (got {count})")
        if not self.journeys:
            raise ValueError("No journeys with photos found")

        state = PhotoSelectionState.load(self.state_path)
        if state.pending_photos:
            # Held by a batch that is generated but not yet published. Holds are
            # NOT auto-reclaimed here: an outstanding batch awaiting review is
            # indistinguishable from a crashed one, and reclaiming would re-offer
            # photos a pending week already ships. Recover with release_pending().
            logger.info("%d photo(s) held by an unpublished batch", len(state.pending_photos))

        selections = self._select_from_state(count, filter_faces, state)

        if not selections and self._pool_genuinely_exhausted(state):
            logger.warning("All photos have been posted! Resetting rotation.")
            state.posted_photos = []
            state.face_filtered_photos = []
            state.pending_photos = []
            state.journey_rotation_index = 0
            state.save(self.state_path)
            # Retry once with clean state (iterative, not recursive)
            selections = self._select_from_state(count, filter_faces, state)

        # Count-match gate: a near-exhausted pool used to return a short batch
        # silently, shipping a thinner week. Hard-fail upstream of captions,
        # Drive upload and the Metricool CSV instead.
        if len(selections) < count and not allow_shortfall:
            raise RuntimeError(
                f"PHOTO-CONTRACT: selected {len(selections)} < requested {count} photos "
                f"(pool of {self.get_total_photos()} across {len(self.journeys)} journeys "
                f"is exhausted, or too many were face-filtered) — the batch would ship "
                f"short; add photos or accept the shortfall explicitly"
            )

        return selections

    def _near_duplicate_of(self, score: CullScore, selections: List["PhotoSelection"]):
        """(index, key, score) of an already-picked near-duplicate, else None."""
        for i, sel in enumerate(selections):
            other = self._cull_scores.get(sel.photo_path)
            if other is None:
                continue
            if hamming_distance(score.phash, other.phash) <= self.cull_floors.duplicate_hamming:
                return i, f"{sel.journey_slug}/{sel.filename}", other
        return None

    def _pool_genuinely_exhausted(self, state: PhotoSelectionState) -> bool:
        """True only when every photo in the pool is consumed (posted or
        face-filtered) and nothing is held by an unpublished batch — the one
        case the rotation reset exists for.

        An empty selection from any other cause — files missing on disk (bad
        checkout), runtime face filtering, outstanding holds — must never wipe
        the posted-photos ledger: the reset would re-offer already-shipped
        photos as new.
        """
        if state.pending_photos:
            return False
        consumed = set(state.posted_photos) | set(state.face_filtered_photos)
        return all(
            f"{journey.slug}/{photo['filename']}" in consumed
            for journey in self.journeys
            for photo in journey.photos
        )

    def _select_from_state(
        self,
        count: int,
        filter_faces: bool,
        state: PhotoSelectionState,
    ) -> List[PhotoSelection]:
        """Core selection logic operating on the given state.

        Mutates state in-place (posted_photos, face_filtered_photos,
        journey_rotation_index) and saves it before returning.
        """
        posted_set = set(state.posted_photos)
        face_skipped = set(state.face_filtered_photos)
        pending = set(state.pending_photos)
        # Transient, never persisted: a file the checkout doesn't have yet must
        # not be recorded as consumed, only skipped for this pass.
        missing: set = set()
        # Likewise transient: cull-skipped photos stay in the pool for later.
        culled: set = set()

        # Build weighted rotation order with round-robin interleaving
        total_photos = self.get_total_photos()
        journey_weights = []
        for journey in self.journeys:
            weight = max(1, round(count * journey.photo_count / total_photos))
            journey_weights.append((journey, weight))

        # Interleave: cycle through journeys, one slot per pass
        rotation_order = []
        remaining = {i: w for i, (_, w) in enumerate(journey_weights)}
        while remaining:
            for i in sorted(remaining.keys()):
                rotation_order.append(journey_weights[i][0])
                remaining[i] -= 1
                if remaining[i] <= 0:
                    del remaining[i]

        selections = []
        rotation_idx = state.journey_rotation_index
        attempts = 0
        max_attempts = total_photos + count  # Safety bound

        while len(selections) < count and attempts < max_attempts:
            attempts += 1
            journey = rotation_order[rotation_idx % len(rotation_order)]
            rotation_idx += 1

            # Find next unposted photo in this journey (sequential by day)
            for photo_info in journey.photos:
                photo_key = f"{journey.slug}/{photo_info['filename']}"
                if (
                    photo_key in posted_set
                    or photo_key in face_skipped
                    or photo_key in missing
                    or photo_key in culled
                ):
                    continue

                photo_path = self.site_root / IMAGES_BASE / journey.slug / photo_info["filename"]
                if not photo_path.exists():
                    missing.add(photo_key)
                    continue

                # Face detection filter — skip but don't permanently blacklist
                if filter_faces and _FACE_FILTER_AVAILABLE:
                    if _should_withhold(str(photo_path)):
                        face_skipped.add(photo_key)
                        continue

                score = self._cull_score(photo_path)
                if score is not None:
                    verdict = cull_decision(score, floors=self.cull_floors)
                    if not verdict.keep:
                        culled.add(photo_key)
                        self.cull_skips.append(CullSkip(photo_key, verdict.reason, score))
                        continue
                    twin = self._near_duplicate_of(score, selections)
                    if twin is not None:
                        twin_idx, twin_key, twin_score = twin
                        if score.sharpness <= twin_score.sharpness:
                            culled.add(photo_key)
                            reason = f"duplicate of {twin_key} (keeps the sharper)"
                            self.cull_skips.append(CullSkip(photo_key, reason, score))
                            continue
                        # This one is the better of the group: displace the earlier pick.
                        culled.add(twin_key)
                        posted_set.discard(twin_key)
                        pending.discard(twin_key)
                        reason = f"duplicate of {photo_key} (keeps the sharper)"
                        self.cull_skips.append(CullSkip(twin_key, reason, twin_score))
                        del selections[twin_idx]

                selections.append(
                    PhotoSelection(
                        journey_slug=journey.slug,
                        journey_title=journey.title,
                        day_number=photo_info["day_number"],
                        day_date=photo_info["date"],
                        location=photo_info["location"],
                        filename=photo_info["filename"],
                        photo_path=str(photo_path),
                        locations_visited=photo_info.get("locations_visited", []),
                        day_summary=photo_info.get("day_summary", ""),
                        statistics=photo_info.get("statistics", {}),
                    )
                )
                posted_set.add(photo_key)
                pending.add(photo_key)
                break

        if face_skipped:
            logger.info(f"Face filter: skipped {len(face_skipped)} photo(s) with detected faces")

        if selections:
            # Update state
            state.posted_photos = list(posted_set)
            state.face_filtered_photos = list(face_skipped)
            state.pending_photos = list(pending)
            state.journey_rotation_index = rotation_idx
            state.last_updated = datetime.now().isoformat()
            state.save(self.state_path)

            logger.info(
                f"Selected {len(selections)} photos "
                f"from {len(set(s.journey_slug for s in selections))} journeys"
            )

        return selections
