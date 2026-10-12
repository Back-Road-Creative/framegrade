"""``per_lf_gpx_path`` round-trip contract.

The Phase D
chapter-generation site indexes ``state["per_lf_gpx_path"][lf.index]``
(int key) to pick the correct GPX file for each LF on multi-day drives.
Without the field declared on ``PipelineState`` and registered in the
``from_dict`` int-key coercion list, the JSON round-trip silently drops
the map on resume and Phase D falls back to the legacy day-local-vs-
global GPX index lookup — the bug this field closes.
This test pins the contract at the load boundary.
"""

from __future__ import annotations

import json
from pathlib import Path

from gomoveshift.pipeline.drive_state_model import PipelineState


def test_per_lf_gpx_path_roundtrips_with_int_keys():
    state = PipelineState(
        source_dir=Path("/tmp/src"),
        output_base=Path("/tmp/out"),
        per_lf_gpx_path={
            1: "/abs/path/to/day_2026-02-20/gopro_01.gpx",
            2: "/abs/path/to/day_2026-02-21/gopro_02.gpx",
            3: "/abs/path/to/day_2026-02-21/gopro_03.gpx",
        },
    )
    on_disk = json.loads(json.dumps(state.to_dict()))
    restored = PipelineState.from_dict(on_disk)

    assert restored.per_lf_gpx_path is not None
    assert set(restored.per_lf_gpx_path.keys()) == {1, 2, 3}, (
        "Int-key coercion required for per_lf_gpx_path — Phase D's "
        "chapter site indexes by lf.index (int)."
    )
    assert restored.per_lf_gpx_path[2] == "/abs/path/to/day_2026-02-21/gopro_02.gpx"


def test_per_lf_gpx_path_none_roundtrips_cleanly():
    state = PipelineState(source_dir=Path("/tmp/src"), output_base=Path("/tmp/out"))
    restored = PipelineState.from_dict(state.to_dict())
    assert restored.per_lf_gpx_path is None
