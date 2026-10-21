"""Handler for ``gomoveshift catalog-classify``.

Builds each long-form video's GPS location-evidence record and classifies its
live title A-E; judges each Short's title against its clip's own audited
places. Read-only: it never calls a YouTube write.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ...catalog.classify import (
    classify_rows,
    load_lf_points,
    load_short_evidence,
    write_outputs,
)
from ...paths import gms_intermediate_dir, gms_output_dir


def _points_loader(output_dir):
    base = Path(output_dir) if output_dir else gms_output_dir()
    return load_lf_points(lambda project: gms_intermediate_dir(base, project))


def _shorts_loader(output_dir):
    base = Path(output_dir) if output_dir else gms_output_dir()
    return load_short_evidence(lambda project: gms_intermediate_dir(base, project))


def _geocoder():
    from ...pipeline.gps_enrichment import _build_shared_geocoder

    return _build_shared_geocoder()


def _forward():
    from ...pipeline.chapters.nominatim_source import _nominatim_search

    return _nominatim_search


def cmd_catalog_classify(args, config) -> int:
    geocoder = _geocoder()
    if geocoder is None:
        raise SystemExit("catalog-classify: no geocoder available (gps_journey not importable)")
    with open(args.inventory, newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if args.kind == "all" or r.get("kind") == args.kind]
    if args.limit:
        rows = rows[: args.limit]
    results = classify_rows(
        rows,
        points_for=_points_loader(args.output_dir),
        geocoder=geocoder,
        forward=_forward(),
        short_for=_shorts_loader(args.output_dir),
    )
    counts = write_outputs(results, Path(args.out_dir))
    print(f"classified rows: {len(results)}")
    for cls, n in counts.items():
        print(f"  {cls}: {n}")
    print(
        "wrote classification.csv, evidence.jsonl, worklist-AB.csv, "
        f"worklist-shorts-drift.csv under {args.out_dir}"
    )
    return 0
