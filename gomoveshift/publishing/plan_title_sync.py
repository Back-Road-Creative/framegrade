"""Keep a published short's publish-plan title in step with its live title.

``tools/retitle_apply.py`` changes a short's title on YouTube. The publish
plan sidecar that produced the upload still names the old title, so
``publishing-status --validate --live-catalog`` reads the gap as
``published_title_drift`` -- a defect made by our own tool. This module is
the one place that closes it:

* ``sync_plan_title`` rewrites ``shorts[i].payloads.youtube.title`` in every
  sidecar whose published YouTube delivery is that video. Local file write
  only; never a YouTube call. ``retitle_apply`` calls it in the same step
  that records APPLIED, and ``gomoveshift queue sync-plan-titles`` replays a
  retitle audit log through it for retitles made before it existed.
* ``applied_retitles`` reads those audit logs, so validate can tell "plan
  stale, retitled by us" apart from genuine drift.

* ``sync_plan_tags`` brings a queued Short's tags into line with the Short tag rule
  (Waypost tag templates plus the clip's own counties and roads,
  ``final_text_record.reviewable_short_tags``) when its sidecar was written
  before that rule, so the final-text review does not refuse it
  (``tag_unreviewable``). ``gomoveshift queue sync-plan-tags`` runs it.

The stored ``reviewed_plan_hash`` is a field of the record, never recomputed
from it, so editing the title or tags does not orphan any queued delivery.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .plan_io import load_publish_plan
from .queue import TERMINAL_STATUSES, log_response

SYNC_COMMAND = "gomoveshift queue sync-plan-titles --audit <retitle audit log> --apply"


@dataclass(frozen=True)
class Retitle:
    """The newest APPLIED retitle of one video."""

    title: str
    at: str
    audit_path: Path


@dataclass(frozen=True)
class PlanTitleChange:
    delivery_id: int
    sidecar_path: str
    old: str
    new: str
    written: bool


def applied_retitles(audit_paths: Iterable[Path]) -> Dict[str, Retitle]:
    """``{video_id: Retitle}`` from retitle_apply audit logs: the latest
    APPLIED row per video by timestamp (a revert's APPLIED row counts too --
    its ``new`` is the restored title, which is what is live)."""
    latest: Dict[str, Retitle] = {}
    for path in audit_paths:
        path = Path(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("result") != "APPLIED" or not row.get("video_id"):
                continue
            seen = latest.get(row["video_id"])
            if seen is None or row.get("timestamp", "") >= seen.at:
                latest[row["video_id"]] = Retitle(row["new"], row.get("timestamp", ""), path)
    return latest


@dataclass(frozen=True)
class PlanTagChange:
    delivery_id: int
    sidecar_path: str
    old: Tuple[str, ...]
    new: Tuple[str, ...]
    written: bool


def _write_title(
    sidecar: Path, project_name: str, short_index: int, title: str, *, apply: bool = True
) -> None:
    _write_youtube_field(sidecar, project_name, short_index, "title", title, apply=apply)


def _write_youtube_field(
    sidecar: Path, project_name: str, short_index: int, field: str, value, *, apply: bool
) -> None:
    """Rewrite one short's YouTube ``field``, keeping the sidecar's own format
    (sorted keys, indent 2, trailing newline). The new file is parsed by
    ``load_publish_plan`` -- every plan gate included -- before it replaces
    the old one, so a refused value never lands. Without ``apply`` the same
    parse runs on the temp copy and the sidecar is left untouched, so a dry
    run predicts the refusal an apply would hit."""
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    for short in raw["shorts"]:
        if short["project_name"] == project_name and short["short_index"] == short_index:
            short["payloads"]["youtube"][field] = value
            break
    else:
        raise ValueError(f"{sidecar}: no short {project_name}#{short_index}")
    tmp = sidecar.with_name(sidecar.name + ".sync-tmp")
    tmp.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        load_publish_plan(tmp)
    except Exception:
        tmp.unlink()
        raise
    if apply:
        os.replace(tmp, sidecar)
    else:
        tmp.unlink()


def sync_plan_title(
    conn, video_id: str, title: str, *, apply: bool, already: List[int] | None = None
) -> List[PlanTitleChange]:
    """Point every published YouTube delivery of ``video_id`` at ``title``.

    Returns one change per sidecar whose plan title differs (empty when the
    plan already agrees or no delivery has this video). Raises when the plan
    gate refuses the new title, apply or not. With ``apply`` the
    sidecar is rewritten and a ``sync_plan_title`` row is logged to
    ``response_log``; without it nothing is written."""
    rows = conn.execute(
        "SELECT id, project_name, short_index, sidecar_path FROM deliveries "
        "WHERE platform = 'youtube' AND status = 'published' AND platform_upload_id = ? "
        "AND sidecar_path IS NOT NULL ORDER BY id",
        (video_id,),
    ).fetchall()
    changes: List[PlanTitleChange] = []
    for row in rows:
        sidecar = Path(row["sidecar_path"])
        plan = load_publish_plan(sidecar)
        old = next(
            (
                s.payloads.youtube.title
                for s in plan.shorts
                if s.project_name == row["project_name"]
                and s.short_index == row["short_index"]
                and s.payloads.youtube
            ),
            None,
        )
        if old is not None and old == title and already is not None:
            already.append(row["id"])
        if old is None or old == title:
            continue
        _write_title(sidecar, row["project_name"], row["short_index"], title, apply=apply)
        if apply:
            log_response(
                conn,
                "youtube",
                "plan_sync",
                0,
                "sync_plan_title",
                "live",
                reason=json.dumps(
                    {
                        "delivery_id": row["id"],
                        "video_id": video_id,
                        "sidecar": str(sidecar),
                        "old": old,
                        "new": title,
                    }
                ),
            )
        changes.append(PlanTitleChange(row["id"], str(sidecar), old, title, apply))
    return changes


def sync_plan_tags(conn, *, apply: bool) -> List[PlanTagChange]:
    """Rewrite the sidecar tags of every non-terminal YouTube delivery whose tags are not
    all reviewable to ``reviewable_short_tags`` of its own clip. Published and other
    terminal rows keep the record of what was sent. Returns one change per row rewritten
    (or, without ``apply``, that would be); with ``apply`` each is logged to
    ``response_log`` as ``sync_plan_tags``."""
    from gomoveshift.pipeline.final_text_record import reviewable_short_tags

    terminal = ", ".join("?" for _ in TERMINAL_STATUSES)
    rows = conn.execute(
        "SELECT id, project_name, short_index, sidecar_path FROM deliveries "  # noqa: S608
        f"WHERE platform = 'youtube' AND status NOT IN ({terminal}) "
        "AND sidecar_path IS NOT NULL ORDER BY id",
        TERMINAL_STATUSES,
    ).fetchall()
    changes: List[PlanTagChange] = []
    for row in rows:
        sidecar = Path(row["sidecar_path"])
        record = next(
            (
                s
                for s in load_publish_plan(sidecar).shorts
                if s.project_name == row["project_name"]
                and s.short_index == row["short_index"]
                and s.payloads.youtube
            ),
            None,
        )
        if record is None:
            continue
        old = tuple(record.payloads.youtube.tags)
        new = tuple(reviewable_short_tags(old, record.sampled_regions, record.sampled_roads))
        if new == old:
            continue
        _write_youtube_field(
            sidecar, row["project_name"], row["short_index"], "tags", list(new), apply=apply
        )
        if apply:
            log_response(
                conn,
                "youtube",
                "plan_sync",
                0,
                "sync_plan_tags",
                "live",
                reason=json.dumps(
                    {
                        "delivery_id": row["id"],
                        "sidecar": str(sidecar),
                        "old": list(old),
                        "new": list(new),
                    }
                ),
            )
        changes.append(PlanTagChange(row["id"], str(sidecar), old, new, apply))
    return changes
