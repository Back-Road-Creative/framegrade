"""``gomoveshift queue remeasure-short-route``: re-derive a Short's missing route ``bins``.

Dry run unless ``--apply``; writes the publish-plan sidecar's ``route_record`` only,
never the queue and never YouTube. See ``publishing.route_remeasure``.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ...publishing.queue import open_queue_db
from ...publishing.route_remeasure import RemeasureRefusedError, remeasure_short_route


def _geocoder():
    from ...pipeline.gps_enrichment import _build_shared_geocoder

    return _build_shared_geocoder()


def cmd_remeasure_short_route(args, config) -> int:
    conn = open_queue_db(config.seo.storage_path)
    try:
        row = conn.execute("SELECT * FROM deliveries WHERE id=?", (args.id,)).fetchone()
        if row is None:
            print(f"REFUSED: no delivery {args.id}", file=sys.stderr)
            return 1
        geocoder = _geocoder()
        if geocoder is None:
            print("REFUSED: no geocoder available to measure with", file=sys.stderr)
            return 1
        out = remeasure_short_route(row, Path(args.stages_dir), geocoder, apply=bool(args.apply))
    except RemeasureRefusedError as exc:
        print(f"REFUSED: delivery {args.id}: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()
    line = (
        f"delivery {args.id} short rank {out['rank']}: {out['bins']} bin(s), "
        f"{out['retained_seconds']} s retained, {out['unsettled_seconds']} s unsettled, "
        f"{out['gap_seconds']} s gap, windows_key {out['windows_key'][:12]}"
    )
    if out["applied"]:
        print(f"wrote route_record: {line}")
    else:
        print(f"dry run: would write route_record: {line}; re-run with --apply to write it")
    return 0
