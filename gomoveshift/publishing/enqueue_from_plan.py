"""Read a Phase D publish-plan sidecar into the delivery queue.

Bridges ``phase_d._emit_publish_plan_sidecar`` (writer,
``<project>_publish_plan.json``) and ``queue.enqueue`` (durable spool):
for every short in the sidecar, enqueue one ``deliveries`` row per platform
in that short's ``routes.selected``. See
``docs/social-publishing-queue.md``.

Idempotent by construction: re-running over the same sidecar enqueues zero
new rows (``queue.enqueue`` is idempotent on the exact
``(project_name, short_index, platform, plan_hash)`` 4-tuple). A DIFFERENT
``plan_hash`` already ACTIVE for the same ``(project, short_index,
platform)`` is a genuine conflict — the plan changed under an in-flight
delivery — and surfaces as :class:`DeliveryPlanConflictError`, never
swallowed.

The sidecar is untrusted JSON on disk: malformed JSON, a missing key, or a
wrong-typed field raises :class:`PublishPlanSidecarError` naming the file
path, never a bare ``KeyError``/``TypeError``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

from .dynamic_schedule import staggered_by_platform
from .first_comment import COMMENT_PLATFORMS, full_drive_comment_text
from .models import KNOWN_PLATFORMS
from .queue import enqueue, find_delivery

logger = logging.getLogger(__name__)

#: staggered_by_platform only ever returns facebook/instagram entries, so
#: the platforms it's asked about here never needs to track a record's
#: actual routes.selected -- an entry for a platform the record didn't
#: route to is simply never looked up.
_STAGGERABLE_PLATFORMS = ("facebook", "instagram")


class PublishPlanSidecarError(ValueError):
    """Sidecar JSON is missing, malformed, or shaped wrong. Message names the file path."""


class DeliveryPlanConflictError(RuntimeError):
    """A different plan_hash is already ACTIVE for (project, short_index, platform)."""


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    """Summary of one ``enqueue_from_plan_sidecar`` call."""

    enqueued: tuple[tuple[str, int, str, int], ...]  # (project, short_index, platform, id)
    already_present: tuple[tuple[str, int, str, int], ...]

    @property
    def enqueued_count(self) -> int:
        return len(self.enqueued)

    @property
    def already_present_count(self) -> int:
        return len(self.already_present)


def _load_sidecar(path: Path) -> dict:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PublishPlanSidecarError(f"publish-plan sidecar unreadable: {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PublishPlanSidecarError(
            f"publish-plan sidecar is not valid JSON: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("shorts"), list):
        raise PublishPlanSidecarError(f"publish-plan sidecar missing a 'shorts' list: {path}")
    return payload


def _field(record: dict, key: str, expected_type: type, path: Path, context: str) -> Any:
    if key not in record:
        raise PublishPlanSidecarError(
            f"publish-plan sidecar {path}: record missing {key!r} ({context})"
        )
    value = record[key]
    # bool is an int subclass -- reject it explicitly wherever int is expected.
    if expected_type is int and isinstance(value, bool):
        value_ok = False
    else:
        value_ok = isinstance(value, expected_type)
    if not value_ok:
        raise PublishPlanSidecarError(
            f"publish-plan sidecar {path}: {key!r} has the wrong type "
            f"(expected {expected_type.__name__}, got {type(value).__name__}) ({context})"
        )
    return value


def select_capped_shorts(shorts: list, max_shorts: Optional[int]) -> list:
    """The N lowest-``short_index`` records of ``shorts``, or all of them.

    Shared by the CLI so enqueue-time media verification covers exactly the
    shorts that will be spooled: a short past the cap is never published,
    so its media cannot be a reason to refuse the run. Records without a
    usable integer ``short_index`` sort last rather than raising -- the
    malformed-record error belongs to the caller that reads the field.
    """
    if max_shorts is None or max_shorts < 0 or len(shorts) <= max_shorts:
        return list(shorts)

    def rank(record: object) -> tuple[int, int]:
        if isinstance(record, dict) and isinstance(record.get("short_index"), int):
            return (0, record["short_index"])
        return (1, 0)

    return sorted(shorts, key=rank)[:max_shorts]


def _platform_scheduled_at(
    by_platform: dict, platform: str, fallback: str, path: Path, context: str
) -> str:
    when = by_platform.get(platform)
    if when is None:
        return fallback
    if not isinstance(when, str) or not when.strip():
        raise PublishPlanSidecarError(
            f"publish-plan sidecar {path}: schedule.scheduled_at_utc_by_platform"
            f"[{platform!r}] is not a timestamp ({context})"
        )
    return when


def enqueue_from_plan_sidecar(
    conn: sqlite3.Connection,
    sidecar_path: Union[str, Path],
    *,
    shadow: bool = False,
    max_shorts: Optional[int] = None,
    order: Optional[Callable[[list], list]] = None,
    stagger_hours: Optional[int] = None,
) -> EnqueueResult:
    """Enqueue every ``routes.selected`` platform of every short in the
    sidecar at ``sidecar_path``. Returns a summary of new vs. already-present
    rows. Raises :class:`PublishPlanSidecarError` on a malformed sidecar and
    :class:`DeliveryPlanConflictError` when a different plan is already
    in flight for the same (project, short_index, platform).

    ``shadow=True`` spools every row as a rehearsal no drain path can
    publish (see ``queue.enqueue``), so a shadow run proves the chain end to
    end without leaving live ammunition in the spool.

    Each row carries its plan's ``schedule.scheduled_at_utc``, which
    ``dispatch._select_due`` enforces; a record without one is refused rather
    than spooled as publish-immediately.
    ``max_shorts=N`` spools only the N lowest ``short_index`` records --
    planner rank order, so the cap keeps the best shorts. This is a VOLUME
    decision applied in memory: the sidecar on disk is never rewritten, so
    the Gate C/D content hash still describes the plan a human approved and
    every spooled short is one that review covered.

    ``order=``, applied AFTER the cap: a callable taking the capped
    list of raw sidecar records and returning them reordered (typically
    ``functools.partial(dynamic_schedule.order_shorts, ...)``). It changes
    only the order rows are inserted in, never which shorts are selected --
    that stays ``select_capped_shorts``'s job.

    ``stagger_hours=`` fills a record's ``facebook``/``instagram``
    entries of ``schedule.scheduled_at_utc_by_platform`` with
    ``dynamic_schedule.staggered_by_platform`` when the sidecar didn't
    already carry an explicit one for that platform -- the sidecar's own
    per-platform value always wins, this only fills a gap.
    """
    path = Path(sidecar_path)
    payload = _load_sidecar(path)

    enqueued: list[tuple[str, int, str, int]] = []
    already_present: list[tuple[str, int, str, int]] = []

    capped = select_capped_shorts(payload["shorts"], max_shorts)
    if order is not None:
        capped = order(capped)

    for record in capped:
        if not isinstance(record, dict):
            raise PublishPlanSidecarError(
                f"publish-plan sidecar {path}: a 'shorts' entry is not an object"
            )
        project_name = _field(record, "project_name", str, path, "record identity")
        short_index = _field(record, "short_index", int, path, "record identity")
        context = f"project={project_name!r} short_index={short_index}"
        plan_hash = _field(record, "reviewed_plan_hash", str, path, context)
        # The plan's publish time goes ONTO the row: `dispatch._select_due`
        # gates on it, so a short cannot be published before its slot by any
        # drain path. A sidecar without one is refused rather than spooled as
        # publish-immediately -- three of the four platforms post the moment
        # they are called.
        schedule = record.get("schedule")
        if not isinstance(schedule, dict) or not isinstance(schedule.get("scheduled_at_utc"), str):
            raise PublishPlanSidecarError(
                f"publish-plan sidecar {path}: record missing schedule.scheduled_at_utc "
                f"({context}); without it the delivery would publish immediately"
            )
        scheduled_at = schedule["scheduled_at_utc"]
        # A platform whose own best-time window moved it off the shared
        # slot carries its own instant here. Absent platform -> the shared
        # slot, never NULL: `dispatch._select_due` treats a NULL as "due
        # now", so a missing entry must not become publish-immediately.
        by_platform = schedule.get("scheduled_at_utc_by_platform") or {}
        if not isinstance(by_platform, dict):
            raise PublishPlanSidecarError(
                f"publish-plan sidecar {path}: schedule.scheduled_at_utc_by_platform "
                f"is not an object ({context})"
            )
        if stagger_hours:
            try:
                base_dt = datetime.fromisoformat(scheduled_at.replace("Z", "+00:00"))
            except ValueError as exc:
                raise PublishPlanSidecarError(
                    f"publish-plan sidecar {path}: schedule.scheduled_at_utc is not a "
                    f"parseable timestamp ({context}): {exc}"
                ) from exc
            computed = staggered_by_platform(
                base_dt, stagger_hours=stagger_hours, platforms=_STAGGERABLE_PLATFORMS
            )
            by_platform = {
                **{platform: dt.isoformat() for platform, dt in computed.items()},
                **by_platform,
            }
        routes = record.get("routes")
        if not isinstance(routes, dict) or not isinstance(routes.get("selected"), list):
            raise PublishPlanSidecarError(
                f"publish-plan sidecar {path}: record missing routes.selected list ({context})"
            )

        for platform in routes["selected"]:
            if not isinstance(platform, str) or platform not in KNOWN_PLATFORMS:
                raise PublishPlanSidecarError(
                    f"publish-plan sidecar {path}: unknown platform {platform!r} ({context}); "
                    f"known platforms: {sorted(KNOWN_PLATFORMS)}"
                )

            existing_row = conn.execute(
                "SELECT id FROM deliveries WHERE project_name=? AND short_index=? "
                "AND platform=? AND plan_hash=?",
                (project_name, short_index, platform, plan_hash),
            ).fetchone()

            try:
                delivery_id = enqueue(
                    conn,
                    project_name,
                    short_index,
                    platform,
                    plan_hash,
                    shadow=shadow,
                    sidecar_path=str(path),
                    scheduled_at=_platform_scheduled_at(
                        by_platform, platform, scheduled_at, path, context
                    ),
                    first_comment_text=_first_comment_for(record, platform, context),
                )
            except sqlite3.IntegrityError as exc:
                active = find_delivery(conn, project_name, short_index, platform)
                active_hash = active["plan_hash"] if active else "<unknown>"
                raise DeliveryPlanConflictError(
                    "publish-plan CONFLICT: an ACTIVE delivery already exists for "
                    f"project={project_name!r} short_index={short_index} "
                    f"platform={platform!r} with plan_hash={active_hash!r}, but this "
                    f"sidecar carries a DIFFERENT plan_hash={plan_hash!r} ({path}). The "
                    "plan changed under an in-flight delivery -- resolve before "
                    "re-enqueuing."
                ) from exc

            key = (project_name, short_index, platform, delivery_id)
            if existing_row is not None:
                already_present.append(key)
            else:
                enqueued.append(key)

    return EnqueueResult(enqueued=tuple(enqueued), already_present=tuple(already_present))


def _first_comment_for(record: dict, platform: str, context: str) -> Optional[str]:
    """The full-drive first comment for ``platform``, or ``None`` (TikTok has
    no comment API; a record with no parent id is warned about and left for
    ``backfill_first_comment_text`` -- a comment must never block an enqueue)."""
    if platform not in COMMENT_PLATFORMS:
        return None
    try:
        return full_drive_comment_text(record.get("parent_lf_video_id") or "")
    except ValueError:
        logger.warning("no parent_lf_video_id, first comment not scheduled (%s)", context)
        return None


@dataclass(frozen=True, slots=True)
class BackfillResult:
    filled: int = 0
    skipped: int = 0


def backfill_first_comment_text(conn: sqlite3.Connection) -> BackfillResult:
    """Fill ``first_comment_text`` on existing YouTube/Instagram/Facebook rows
    that lack it, from the row's own sidecar record (matched on project and
    ``short_index``). Idempotent; a row already posted is never touched; an
    unreadable sidecar or a missing parent id is skipped with a WARNING and
    never raises."""
    marks = ",".join("?" for _ in COMMENT_PLATFORMS)
    rows = conn.execute(
        f"SELECT id, project_name, short_index, sidecar_path FROM deliveries "  # noqa: S608
        f"WHERE platform IN ({marks}) AND first_comment_posted_at IS NULL "
        "AND (first_comment_text IS NULL OR first_comment_text = '') ORDER BY id",
        COMMENT_PLATFORMS,
    ).fetchall()
    cache: dict = {}
    filled = skipped = 0
    for row in rows:
        text = _backfill_text(row, cache)
        if text is None:
            skipped += 1
            continue
        with conn:
            conn.execute(
                "UPDATE deliveries SET first_comment_text=? WHERE id=? "
                "AND first_comment_posted_at IS NULL "
                "AND (first_comment_text IS NULL OR first_comment_text = '')",
                (text, row["id"]),
            )
        filled += 1
    return BackfillResult(filled=filled, skipped=skipped)


def _backfill_text(row, cache: dict) -> Optional[str]:
    label = f"delivery {row['id']} ({row['project_name']} #{row['short_index']})"
    sidecar = row["sidecar_path"]
    if not sidecar:
        logger.warning("first-comment backfill: %s has no sidecar_path; skipped", label)
        return None
    if sidecar not in cache:
        try:
            payload = json.loads(Path(sidecar).read_text(encoding="utf-8"))
            shorts = payload["shorts"]
            if not isinstance(shorts, list):
                raise TypeError("'shorts' is not a list")
            cache[sidecar] = [r for r in shorts if isinstance(r, dict)]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("first-comment backfill: sidecar %s unreadable (%s)", sidecar, exc)
            cache[sidecar] = None
    records = cache[sidecar]
    if records is None:
        return None
    for record in records:
        if (
            record.get("short_index") == row["short_index"]
            and record.get("project_name") == row["project_name"]
        ):
            try:
                return full_drive_comment_text(record.get("parent_lf_video_id") or "")
            except ValueError:
                break
    logger.warning("first-comment backfill: %s has no parent_lf_video_id; skipped", label)
    return None
