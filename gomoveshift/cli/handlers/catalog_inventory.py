"""Handler for ``gomoveshift catalog-inventory``.

Joins a live channel catalog (``live_catalog.json``) to local evidence and
writes the per-slice title-accuracy CSVs. Read-only on every source.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...catalog.inventory import (
    build_inventory,
    load_deliveries,
    load_ledger_entries,
    write_slices,
)
from ...paths import gms_output_dir


def _resolve(args, config):
    """Explicit flags win; the SEO-derived sources fall back to ``config.seo``."""
    vid = args.vid_output_dir or str(gms_output_dir())
    output = args.output_dir or str(gms_output_dir())
    queue_db, events, ledger = args.queue_db, args.events_jsonl, args.ledger
    if not (queue_db and events and ledger):
        seo = config.seo
        queue_db = queue_db or str(Path(seo.storage_path) / "social_queue.sqlite3")
        events = events or seo.events_jsonl
        ledger = ledger or str(Path(seo.storage_path) / ".content_ledger.json")
    return Path(vid), Path(output), Path(queue_db), Path(events), Path(ledger)


def cmd_catalog_inventory(args, config) -> int:
    vid, output, queue_db, events, ledger = _resolve(args, config)
    live = json.loads(Path(args.live_catalog).read_text())
    with open(events) as fh:
        event_lines = fh.readlines()
    rows = build_inventory(
        live,
        lf_upload_files=sorted(vid.glob("*/youtube_uploads.json")),
        tracking_files=sorted(output.glob("*/tracking/*.json")),
        deliveries_rows=load_deliveries(queue_db),
        waypost_event_lines=event_lines,
        ledger_entries=load_ledger_entries(ledger),
    )
    counts = write_slices(rows, Path(args.out_dir))
    print(f"inventory rows: {len(rows)}")
    for name, n in counts.items():
        print(f"  {name}: {n}")
    return 0
