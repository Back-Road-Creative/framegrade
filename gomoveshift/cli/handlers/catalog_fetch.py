"""Handler for ``gomoveshift catalog-fetch``.

Read-only live pull of every own-channel video into the ``live_catalog.json``
that ``catalog-inventory --live-catalog`` reads.

With ``--known-ids-from-queue`` every queue id was checked by id, so the
result is also written to ``<storage_path>/live_catalog.latest.json``: the
snapshot the enqueue/dispatch gate reads (fresh = at most 24 h old). Without
it the snapshot is left alone -- a pass that skipped a queue id could make the
gate hold a pair for a video that still exists.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ...catalog.fetch import fetch_catalog
from ...pipeline.drive_phases.phase_d_scheduling import _build_readonly_youtube_service
from ...pipeline.site_sync.channel_manifest import enumerate_uploads
from ...publishing.queue import open_queue_db
from .pub_status import live_catalog_snapshot_path


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def cmd_catalog_fetch(args, config) -> int:
    known: list[str] = []
    if args.known_ids:
        known = json.loads(Path(args.known_ids).read_text())["known_ids"]
    if getattr(args, "known_ids_from_queue", False):
        known = sorted(set(known) | set(queue_youtube_upload_ids(config.seo.storage_path)))
    svc = _build_readonly_youtube_service()
    catalog = fetch_catalog(svc, enumerate_uploads(svc, known_ids=known), known_ids=known)
    catalog["fetched_at"] = _utcnow().isoformat()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalog, indent=1))
    print(f"enumerated {catalog['enumerated']}, listed {catalog['listed']} -> {out}")
    if getattr(args, "known_ids_from_queue", False):
        snapshot = live_catalog_snapshot_path(config.seo.storage_path)
        tmp = snapshot.with_suffix(".tmp")
        tmp.write_text(json.dumps(catalog, indent=1))
        tmp.replace(snapshot)
        print(f"gate snapshot -> {snapshot}")
    else:
        print("gate snapshot not updated: only a --known-ids-from-queue pass refreshes it")
    return 0


def queue_youtube_upload_ids(storage_path) -> list[str]:
    """Every upload id the delivery queue holds for a published or uploaded
    YouTube row, so the catalog checks each one by id (U-R1)."""
    conn = open_queue_db(storage_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT platform_upload_id FROM deliveries WHERE platform='youtube' "
            "AND status IN ('published', 'uploaded') AND platform_upload_id IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    return sorted(r[0] for r in rows)
