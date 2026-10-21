"""Title-accuracy catalog inventory: join live channel videos to local evidence.

Every live video ID (from a ``live_catalog.json`` written by a read-only
``videos.list`` pass) is joined to what the pipeline recorded about it:
LF upload records, tracking files, the social queue's deliveries table, the
Waypost ``yt_sync_`` events and the content ledger. Read-only on every input;
the only writes are the slice CSVs under ``out_dir``.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

#: A video this long or shorter is a Short.
SHORT_MAX_SECONDS = 180

CSV_COLUMNS: List[str] = [
    "video_id",
    "kind",
    "privacy",
    "published_at",
    "publish_at",
    "duration_s",
    "recording_date",
    "title",
    "project",
    "lf_index",
    "short_index",
    "evidence_tier",
    "evidence_sources",
    "waypost_sync_events",
    "tracking_observations",
    "uploaded_title_matches_live",
]

DeliveryRow = Tuple[str, str, Any, Any]


def load_deliveries(db_path: Path) -> List[DeliveryRow]:
    """YouTube deliveries with an upload id, as
    ``(platform_upload_id, project_name, short_index, sidecar_path)``.
    Opens the DB read-only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [
            (pid, proj, idx, side)
            for pid, proj, idx, side in conn.execute(
                "select platform_upload_id, project_name, short_index, sidecar_path "
                "from deliveries where platform='youtube' and platform_upload_id is not null"
            )
        ]
    finally:
        conn.close()


def load_ledger_entries(path: Path) -> List[Dict[str, Any]]:
    return list(json.loads(Path(path).read_text())["entries"])


def build_inventory(
    live_catalog: Dict[str, Any],
    *,
    lf_upload_files: Iterable[Path],
    tracking_files: Iterable[Path],
    deliveries_rows: Iterable[DeliveryRow],
    waypost_event_lines: Iterable[str],
    ledger_entries: Iterable[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """One row per live video, in catalog order. Later sources of the same
    kind overwrite earlier ones (files are read in the order given)."""
    ev: Dict[Any, Dict[str, Any]] = {}
    for f in lf_upload_files:
        f = Path(f)
        for lf, r in json.loads(f.read_text()).items():
            ev.setdefault(r.get("video_id"), {})["lf_upload"] = {
                "project": f.parent.name,
                "lf": lf,
                "uploaded_title": r.get("title"),
                "src": str(f),
            }
    for f in tracking_files:
        f = Path(f)
        t = json.loads(f.read_text())
        for kind in ("shorts", "long_forms"):
            for s in t.get(kind, []):
                vid = (s.get("platform_ids") or {}).get("youtube")
                if vid:
                    obs = (s.get("observation_log") or {}).get("youtube") or []
                    ev.setdefault(vid, {})["tracking"] = {
                        "project": t.get("project"),
                        "kind": kind,
                        "index": s.get("index"),
                        "file": s.get("file"),
                        "observations": len(obs),
                        "src": str(f),
                    }
    for pid, proj, idx, side in deliveries_rows:
        ev.setdefault(pid, {})["delivery"] = {"project": proj, "short_index": idx, "plan": side}
    for line in waypost_event_lines:
        if '"yt_sync_' not in line:
            continue
        e = json.loads(line)
        d = ev.setdefault(e["data"].get("video_id"), {}).setdefault("waypost_events", {"n": 0})
        d["n"] += 1
        d["last"] = e["timestamp"]
    for e in ledger_entries:
        if e.get("video_id"):
            ev.setdefault(e["video_id"], {})["ledger"] = {
                "project": e.get("project_name"),
                "content_type": e.get("content_type"),
            }

    rows: List[Dict[str, Any]] = []
    for vid, v in live_catalog["videos"].items():
        e = ev.get(vid, {})
        proj = next(
            (
                e[k]["project"]
                for k in ("lf_upload", "delivery", "tracking", "ledger")
                if k in e and e[k].get("project")
            ),
            None,
        )
        footage = "lf_upload" in e or "delivery" in e or "tracking" in e
        rows.append(
            {
                "video_id": vid,
                "kind": "short" if (v.get("duration_s") or 0) <= SHORT_MAX_SECONDS else "long",
                "title": v.get("title"),
                "privacy": v.get("privacy"),
                "publish_at": v.get("publish_at"),
                "published_at": v.get("published_at"),
                "duration_s": v.get("duration_s"),
                "recording_date": v.get("recording_date"),
                "project": proj,
                "evidence_tier": "segment" if footage else ("project" if proj else "none"),
                "evidence": e,
            }
        )
    return rows


def _csv_row(r: Dict[str, Any]) -> Dict[str, Any]:
    e = r["evidence"]
    return {
        "video_id": r["video_id"],
        "kind": r["kind"],
        "privacy": r["privacy"],
        "published_at": r["published_at"],
        "publish_at": r["publish_at"] or "",
        "duration_s": r["duration_s"],
        "recording_date": r["recording_date"] or "",
        "title": r["title"],
        "project": r["project"] or "",
        "lf_index": e.get("lf_upload", {}).get("lf", ""),
        "short_index": e.get("delivery", {}).get(
            "short_index", e.get("tracking", {}).get("index", "")
        ),
        "evidence_tier": r["evidence_tier"],
        "evidence_sources": ";".join(sorted(k for k in e if k != "waypost_events")),
        "waypost_sync_events": e.get("waypost_events", {}).get("n", 0),
        "tracking_observations": e.get("tracking", {}).get("observations", ""),
        "uploaded_title_matches_live": ""
        if "lf_upload" not in e
        else str(e["lf_upload"]["uploaded_title"] == r["title"]).lower(),
    }


_SLICES = {
    "longform.csv": lambda r: r["kind"] == "long",
    "shorts-nonpublic.csv": lambda r: r["kind"] == "short" and r["privacy"] != "public",
    "shorts-public.csv": lambda r: r["kind"] == "short" and r["privacy"] == "public",
}


def write_slices(rows: Sequence[Dict[str, Any]], out_dir: Path) -> Dict[str, int]:
    """Write the three per-slice CSVs (oldest ``published_at`` first); return
    ``{filename: row count}``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for name, pred in _SLICES.items():
        chosen = sorted((r for r in rows if pred(r)), key=lambda r: r["published_at"] or "")
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, CSV_COLUMNS)
            w.writeheader()
            for r in chosen:
                w.writerow(_csv_row(r))
        counts[name] = len(chosen)
    return counts
