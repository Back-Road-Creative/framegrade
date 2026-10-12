"""Pre-import validator for Metricool shorts CSVs.

A shorts CSV is safe to import only when, for every row:

  * the first-comment YouTube link points at the CURRENT upload — not a stale
    id left over from a re-run (a batch that was re-uploaded, the old
    ids were deleted -> 404, but the originally-imported CSV kept them); and
  * the short is not scheduled before its parent LF goes public (the link is
    dead from post time until ``publish_at``).

Both are checked statically against the batch's ``youtube_uploads`` state. The
inline gate in :mod:`gomoveshift.pipeline.csv_generator` prevents the timing
case at write time; this is the upstream gate operators (or CI) run before the
irreversible Metricool import, and it additionally catches stale ids that a
re-run introduced after the CSV was first written.

CLI::

    python -m gomoveshift.pipeline.metricool_csv_audit SHORTS.csv [--state UPLOADS.json]

Exits non-zero if any STALE / EARLY / ORPHAN row is found.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schedule_constants import EST

_YT_ID = re.compile(r"(?:watch\?v=|youtu\.be/)([A-Za-z0-9_-]{11})")


@dataclass
class Findings:
    """Per-CSV audit result. ``failed`` is true if anything must be fixed."""

    rows_with_link: int = 0
    no_link: int = 0
    stale: list = field(default_factory=list)  # (row_no, old_id, new_id)
    early: list = field(default_factory=list)  # (row_no, id, slot_iso, pub_iso)
    orphan: list = field(default_factory=list)  # (row_no, id)

    @property
    def ok(self) -> int:
        return self.rows_with_link - len(self.stale) - len(self.orphan)

    @property
    def failed(self) -> bool:
        return bool(self.stale or self.early or self.orphan)


def load_state(path: Path) -> tuple[set, dict, dict]:
    """Return (current_video_ids, old_id->new_id, video_id->publish_at_iso)."""
    data = json.loads(Path(path).read_text() or "{}")
    current: set = set()
    old2new: dict = {}
    pub: dict = {}
    for rec in data.values():
        vid = rec.get("video_id")
        if not vid:
            continue
        current.add(vid)
        pub[vid] = rec.get("publish_at")
        if rec.get("old_video_id"):
            old2new[rec["old_video_id"]] = vid
    return current, old2new, pub


def _slot_instant(date_s: str, time_s: str) -> Optional[datetime]:
    date_s = (date_s or "").strip()
    time_s = (time_s or "").strip() or "00:00:00"
    for tfmt in ("%H:%M:%S", "%H:%M"):
        try:
            naive = datetime.strptime(f"{date_s} {time_s}", "%Y-%m-%d " + tfmt)
        except ValueError:
            continue
        return naive.replace(tzinfo=EST).astimezone(timezone.utc)
    return None


def _pub_instant(iso: Optional[str]) -> Optional[datetime]:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso).astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def audit_rows(rows: list, current_ids: set, old2new: dict, pub_by_id: dict) -> Findings:
    """Classify each CSV row. When ``current_ids`` is empty (no state file),
    only the timing check runs (id validity can't be judged)."""
    f = Findings()
    for i, row in enumerate(rows):
        m = _YT_ID.search(row.get("First Comment Text", "") or "")
        if not m:
            f.no_link += 1
            continue
        f.rows_with_link += 1
        vid = m.group(1)
        eff = vid
        if current_ids:
            if vid in old2new:
                eff = old2new[vid]
                f.stale.append((i + 2, vid, eff))
            elif vid not in current_ids:
                f.orphan.append((i + 2, vid))
        pub_dt = _pub_instant(pub_by_id.get(eff))
        slot_dt = _slot_instant(row.get("Date", ""), row.get("Time", ""))
        if pub_dt and slot_dt and slot_dt < pub_dt:
            f.early.append((i + 2, eff, slot_dt.isoformat(), pub_dt.isoformat()))
    return f


def _discover_state(csv_path: Path) -> Optional[Path]:
    """Prefer a *_RERUN.json (carries old->new remap) then youtube_uploads.json."""
    d = csv_path.parent
    rerun = sorted(d.glob("*RERUN*.json")) + sorted(d.glob("*rerun*.json"))
    for cand in (*rerun, d / "youtube_uploads.json"):
        if cand.exists() and json.loads(cand.read_text() or "{}"):
            return cand
    return None


def audit_file(csv_path: Path, state_path: Optional[Path]) -> Findings:
    state_path = state_path or _discover_state(csv_path)
    if state_path:
        current, old2new, pub = load_state(state_path)
    else:
        current, old2new, pub = set(), {}, {}
    with open(csv_path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return audit_rows(rows, current, old2new, pub)


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description="Validate Metricool shorts CSV(s) before import.")
    p.add_argument("csv", nargs="+", type=Path)
    p.add_argument(
        "--state", type=Path, default=None, help="youtube_uploads.json (auto-discovered if omitted)"
    )
    args = p.parse_args(argv)

    any_failed = False
    for csv_path in args.csv:
        f = audit_file(csv_path, args.state)
        status = "FAIL" if f.failed else "ok"
        print(
            f"[{status}] {csv_path.name}: ok={f.ok} stale={len(f.stale)} "
            f"early={len(f.early)} orphan={len(f.orphan)} no_link={f.no_link}"
        )
        for row_no, old, new in f.stale:
            print(f"    STALE row {row_no}: {old} -> {new}")
        for row_no, vid, slot_iso, pub_iso in f.early[:10]:
            print(f"    EARLY row {row_no}: {vid} posts {slot_iso} but publishes {pub_iso}")
        for row_no, vid in f.orphan[:10]:
            print(f"    ORPHAN row {row_no}: {vid} not in upload state")
        any_failed = any_failed or f.failed
    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
