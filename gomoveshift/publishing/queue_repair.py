"""Repair a queued YouTube Short the final-text review would refuse, before its slot.

``publish`` runs ``short_review_refusal`` at dispatch; a refusal counts toward max
attempts and ends ``failed_permanent`` with the slot never refilled. Two mechanical
repairs already exist for the two causes seen in practice: the tag filter
(:func:`plan_title_sync.sync_plan_tags`, a sidecar written before plan build filtered
tags) and the route re-measure (:func:`route_remeasure.remeasure_short_route`, a route
record with no ``bins``). :func:`repair_queued_shorts` runs the review on every
queued, non-shadow YouTube row, and for a refused one applies those two repairs, in that
order, to a scratch copy of the sidecar. The sidecar is replaced (atomically) only when
the review passes on the repaired copy; otherwise it is left untouched and the reason is
reported. Nothing else changes: not the stamp prefix, the schedule, the recording date
or the experiment arms. The stored ``reviewed_plan_hash`` is never recomputed (see
``plan_title_sync``), so a repaired row keeps its queue ``plan_hash``.

Local file writes and, for a re-measure, the geocoder (cache first); never a YouTube call.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from .models import ShortPublishRecord
from .plan_io import load_publish_plan
from .plan_title_sync import _write_youtube_field
from .queue import log_response
from .route_remeasure import RemeasureRefusedError, remeasure_short_route

QUEUED_YOUTUBE_SQL = (
    "SELECT id, project_name, short_index, sidecar_path, scheduled_at FROM deliveries "
    "WHERE platform = 'youtube' AND status = 'queued' AND shadow = 0 ORDER BY id"
)


@dataclass(frozen=True)
class QueueRepair:
    delivery_id: int
    project_name: str
    outcome: str  # "ok" | "repaired" | "still_refused"
    reason: str = ""
    written: bool = False

    def line(self) -> str:
        tail = f":{self.reason}" if self.outcome == "still_refused" else ""
        return f"QUEUE_REPAIR {self.delivery_id} {self.project_name} {self.outcome}{tail}"


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def load_row_record(row, sidecar: Optional[Path] = None) -> ShortPublishRecord:
    """The row's record from its recorded sidecar (what dispatch's loader reads)."""
    path = Path(sidecar or row["sidecar_path"] or "")
    if not path.is_file():
        raise OSError(f"publish plan sidecar not readable: {path}")
    for short in load_publish_plan(path).shorts:
        if short.project_name == row["project_name"] and short.short_index == row["short_index"]:
            return short
    raise LookupError(f"{path}: no short {row['project_name']}#{row['short_index']}")


def row_refusal(row) -> Optional[str]:
    """Why the final-text review refuses this queued row now, ``None`` when it agrees.
    Raises ``OSError`` / ``LookupError`` / a plan-format error when the record cannot be read."""
    from .youtube_adapter import short_review_refusal

    return short_review_refusal(load_row_record(row))


def _repair_copy(
    row, scratch: Path, stages_dir: Path, geocoder_factory
) -> Tuple[Optional[str], bool]:
    """Apply tag filter then re-measure to ``scratch``. Returns ``(refusal, changed)``."""
    from gomoveshift.pipeline.final_text_record import reviewable_short_tags

    from .youtube_adapter import short_review_refusal

    scratch_row = {**dict(row), "sidecar_path": str(scratch)}
    changed = False
    record = load_row_record(scratch_row)
    old = tuple(record.payloads.youtube.tags)
    new = tuple(reviewable_short_tags(old, record.sampled_regions, record.sampled_roads))
    if new != old:
        _write_youtube_field(
            scratch, row["project_name"], row["short_index"], "tags", list(new), apply=True
        )
        changed = True
    refusal = short_review_refusal(load_row_record(scratch_row))
    if refusal and "bins" in refusal:
        geocoder = geocoder_factory()
        if geocoder is None:
            return "no geocoder available to re-measure with", changed
        try:
            remeasure_short_route(scratch_row, stages_dir, geocoder, apply=True)
        except RemeasureRefusedError as exc:
            return f"remeasure refused: {exc}", changed
        changed = True
        refusal = short_review_refusal(load_row_record(scratch_row))
    return refusal, changed


def config_stages_dir_for(config) -> Callable[[Path, str], Path]:
    """``stages_dir_for(sidecar, project)``: the ``_stages`` dir a row's re-measure reads (the
    run's intermediate dir next to its output), as ``dispatch-queue`` and ``health check`` see it."""
    from ..paths import gms_intermediate_dir

    folder = getattr(getattr(config, "paths", None), "intermediate_folder", "") or ""

    def stages_dir_for(sidecar: Path, project: str) -> Path:
        return gms_intermediate_dir(
            Path(sidecar).parent.parent, project, intermediate_folder=folder
        )

    return stages_dir_for


def repair_queued_shorts(
    conn,
    *,
    apply: bool,
    stages_dir_for: Callable[[Path, str], Path],
    geocoder_factory: Callable[[], Any],
) -> List[QueueRepair]:
    """One :class:`QueueRepair` per queued non-shadow YouTube row. Without ``apply`` the
    repairs run on the scratch copy only and no sidecar or queue row is written."""
    results: List[QueueRepair] = []
    for row in conn.execute(QUEUED_YOUTUBE_SQL).fetchall():
        rid, project = row["id"], row["project_name"]
        try:
            refusal = row_refusal(row)
        except Exception as exc:  # noqa: BLE001 -- an unreadable plan is reported, never fatal
            results.append(
                QueueRepair(rid, project, "still_refused", _one_line(f"unreadable:{exc}"))
            )
            continue
        if refusal is None:
            results.append(QueueRepair(rid, project, "ok"))
            continue
        sidecar = Path(row["sidecar_path"])
        before = sidecar.read_bytes()
        work = Path(tempfile.mkdtemp(prefix="queue-repair-"))
        try:
            scratch = work / sidecar.name
            scratch.write_bytes(before)
            try:
                after, changed = _repair_copy(
                    row, scratch, stages_dir_for(sidecar, project), geocoder_factory
                )
            except Exception as exc:  # noqa: BLE001
                after, changed = f"repair failed: {type(exc).__name__}: {exc}", False
            if after or not changed:
                results.append(
                    QueueRepair(rid, project, "still_refused", _one_line(after or refusal))
                )
                continue
            written = False
            if apply:
                if sidecar.read_bytes() != before:
                    results.append(
                        QueueRepair(rid, project, "still_refused", "sidecar changed during repair")
                    )
                    continue
                staged = sidecar.with_name(sidecar.name + ".repair-tmp")
                shutil.copyfile(scratch, staged)
                os.replace(staged, sidecar)
                written = True
                log_response(
                    conn,
                    "youtube",
                    "plan_sync",
                    0,
                    "queue_repair",
                    "live",
                    reason=json.dumps(
                        {"delivery_id": rid, "sidecar": str(sidecar), "was": _one_line(refusal)}
                    ),
                )
            results.append(QueueRepair(rid, project, "repaired", written=written))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return results
