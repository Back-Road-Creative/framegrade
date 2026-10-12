"""CLI adapter for preparing an offline Route Story map bundle from a GPX trip.

Real network access (a real ``requests.Session`` and a real MapLibre tarball
fetch) is wired up only here; ``pipeline.route_story.prepare`` itself never
imports ``requests`` and is fully testable with fakes.
"""

from __future__ import annotations

import json
import sys
from typing import Callable

import requests

from ...pipeline.route_story import prepare


def _fetch_maplibre(session: requests.Session) -> Callable[[str], bytes]:
    def fetch(url: str) -> bytes:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        return response.content

    return fetch


def cmd_route_story_prepare(args, config) -> int:
    """Build and verify a Route Story map bundle for one GPX trip."""
    try:
        session = requests.Session()
        report = prepare.prepare_bundle(
            args.gpx,
            session,
            _fetch_maplibre(session),
            recorded_on=args.recorded_on,
            margin_m=args.margin_m,
            overpass_endpoint=args.overpass_endpoint,
            terrain_exaggeration=args.exaggeration,
            trip_date_imagery=args.trip_date_imagery,
        )
    except (ValueError, OSError) as exc:
        print(f"Route Story bundle preparation failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str))
    if report.labels_degraded:
        print("LABELS DEGRADED:", file=sys.stderr)
        for problem in report.label_problems:
            print(f"  - {problem}", file=sys.stderr)
    if report.verify_problems:
        print("Bundle verification found problems:", file=sys.stderr)
        for problem in report.verify_problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    return 0
