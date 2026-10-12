"""CLI adapter for rendering a Route Story terrain proof clip.

Resolves the bundle a prior ``route-story-prepare`` run wrote for this GPX
through ``pipeline.route_story.prepare.find_prepared_bundle`` -- the exact
same day-selection/footprint/source-version key ``prepare_bundle`` writes
under -- and renders the proof clip against it. Real
``SceneCapture``/ffmpeg wiring lives in ``pipeline.route_story.proof``; this
handler only resolves paths and writes the receipt file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ...pipeline.route_story import prepare, proof


def cmd_route_story_proof(args, config) -> int:
    """Render a Route Story terrain proof clip and write its receipt alongside it."""
    try:
        margin_m = args.margin_m if args.margin_m is not None else prepare.DEFAULT_MARGIN_M
        bundle_root = prepare.find_prepared_bundle(
            args.gpx,
            recorded_on=args.recorded_on,
            margin_m=margin_m,
            terrain_exaggeration=args.exaggeration,
            regional_margin_km=args.regional_margin_km,
            trip_date_imagery=args.trip_date_imagery,
        )
        if bundle_root is None:
            raise ValueError(
                "no prepared Route Story bundle found for this GPX on "
                f"{args.recorded_on.isoformat()}; run route-story-prepare first"
            )
        receipt = proof.render_proof_clip(
            args.gpx,
            bundle_root,
            args.output,
            recorded_on=args.recorded_on,
            fps=args.fps,
            width=args.width,
            height=args.height,
            workers=args.workers,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Route Story proof render failed: {exc}", file=sys.stderr)
        return 1

    receipt_path = Path(str(args.output) + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0
