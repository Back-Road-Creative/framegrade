"""Sidecar for journey entries waiting on YouTube ``publish_at`` to fire.

The pipeline's :func:`journal_publisher.plan_merge` returns entries split
into ``inserts`` (publish_at in the past, ready to write into the journey
YAML) and ``deferrals`` (publish_at still in the future). The CLI handler
persists deferrals here and a later ``--replay-pending`` run flushes any
whose publish_at has since passed.

File location: the drive auto-hook writes it to the durable drive output dir
(``<output_base>/<project>/pending_journal.json``) so the replay cron's
``--scan-root`` finds it; the standalone CLI defaults to next to the state
file. The ``journey_path`` is stored absolute on each entry so replay can
write back to the correct site file without re-deriving from CLI args.

Schema is a flat JSON list. Idempotency key is ``(journey_path, youtube_id)``
so a re-run of the same pipeline updates rather than duplicates.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from ._file_lock import make_blocking_file_lock

PENDING_FILENAME = "pending_journal.json"

# Per-thread reentrancy depth per resolved sidecar path, mirroring
# content_ledger._ledger_lock: flock is tied to the open file description, so
# a second acquire in the same thread would open a fresh fd and self-deadlock.
_PENDING_LOCK_STATE = threading.local()


@dataclass
class PendingEntry:
    """One deferred journey entry. Mirrors ``VideoEntry`` plus the target
    site path so replay needs no extra args."""

    youtube_id: str
    publish_at: str  # ISO-8601 (Z form preferred)
    title: str
    duration: str
    day_date: str
    journey_path: str  # absolute path to ``<site>/content/journeys/<slug>.md``
    lf_index: int


def load_pending(path: Path) -> list[PendingEntry]:
    """Read the sidecar. Missing file → empty list (no error)."""
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list, got {type(raw).__name__}")
    return [PendingEntry(**item) for item in raw]


def save_pending(path: Path, entries: Iterable[PendingEntry]) -> None:
    """Write the sidecar. Empty input deletes the file so empty state
    is reflected on disk rather than as a stale ``[]`` artifact."""
    items = list(entries)
    if not items:
        if path.is_file():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a torn/partial sidecar is unparseable and would strand
    # every still-pending video. Write to a temp file then os.replace (atomic
    # on the same filesystem) so a concurrent reader sees old-or-new, never
    # half. (Does not serialize concurrent read-modify-write; see merge_pending.)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps([asdict(e) for e in items], indent=2) + "\n")
    os.replace(tmp, path)


# The sidecar is keyed by ``pending_base`` (the drive output dir), not by
# any one journey — two different journeys can share one sidecar (see the
# module docstring's idempotency key). ``journal_publisher.journey_lock``
# is therefore the wrong resource to extend here: it's keyed per
# ``journey_path`` and wouldn't exclude a second writer touching the same
# sidecar file for a different journey. This uses the same mechanism as
# ``content_ledger._ledger_lock`` (see ``._file_lock.make_blocking_file_lock``)
# — a blocking ``fcntl.flock(LOCK_EX)`` on a ``.lock`` sidecar beside the
# resource itself, reentrant per-thread so a caller that already holds it
# (none today, but future-proofed the same way the ledger lock is) doesn't
# self-deadlock.
#
# Every read-modify-write of this sidecar — ``merge_pending`` and the
# replay cron's post-flush consume in ``publish_journal.run_replay_pending``
# — must take this lock across its whole load -> mutate -> save window, or
# the exclusion is one-sided and the race persists.
pending_lock = make_blocking_file_lock(_PENDING_LOCK_STATE)


def merge_pending(path: Path, new_entries: Iterable[PendingEntry]) -> None:
    """Idempotent upsert by ``(journey_path, youtube_id)``.

    Re-running the same pipeline with --apply must update (not duplicate)
    existing pending entries. The two-key tuple matches the cross-journey
    invariant in ``journal_publisher._all_existing_ids``: a single
    youtube_id is unique across a journey, but two journeys may legitimately
    reference different IDs at the same lf_index.

    The whole load -> upsert -> save window runs under :func:`pending_lock`
    so a concurrent writer (the replay cron vs. a live drive run) can't
    interleave a stale read with this save — see that function's docstring.
    Serialized this way, ``save_pending``'s unlink-on-empty is safe: the last
    writer to observe a genuinely empty entry set is the true state, not a
    stale snapshot racing another writer's insert.
    """
    with pending_lock(path):
        existing = load_pending(path)
        by_key = {(e.journey_path, e.youtube_id): e for e in existing}
        for entry in new_entries:
            by_key[(entry.journey_path, entry.youtube_id)] = entry
        save_pending(path, by_key.values())


def split_by_readiness(
    entries: Iterable[PendingEntry],
    now: datetime,
) -> tuple[list[PendingEntry], list[PendingEntry]]:
    """Returns (ready, still_pending).

    Un-parseable ``publish_at`` stays in still_pending: replay must not
    write a video whose schedule we can't verify. An operator can manually
    edit the sidecar or re-run the pipeline if the value is corrupt.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    ready: list[PendingEntry] = []
    still: list[PendingEntry] = []
    for entry in entries:
        try:
            dt = datetime.fromisoformat(entry.publish_at.replace("Z", "+00:00"))
        except (AttributeError, ValueError):
            still.append(entry)
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if dt <= now:
            ready.append(entry)
        else:
            still.append(entry)
    return ready, still
