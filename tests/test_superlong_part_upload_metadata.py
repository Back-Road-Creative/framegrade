"""Superlong part uploads must carry real tags and a recording date matching
the part's day. ``_upload_superlong_parts_to_youtube`` previously hardcoded
``tags=[]`` and never passed ``recording_date`` at all, so every part shipped
to YouTube with neither."""

from __future__ import annotations

from unittest.mock import MagicMock

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

from _state_helpers import make_state


def _part_record(tmp_path, location="Asheville, North Carolina"):
    path = tmp_path / "sl_01.mp4"
    path.write_bytes(b"x")
    return {
        "path": str(path),
        "part_index": 1,
        "title": "Superlong Part",
        "description": "d",
        "location": location,
    }


def test_superlong_part_upload_carries_tags_and_recording_date(tmp_path):
    """Load-bearing (fails on current code): tags is non-empty and
    recording_date matches the part's day (2026-03-01, from the
    ``state.superlong_parts`` key)."""
    state = make_state(
        region="usa",
        superlong_parts={"2026-03-01": {"part_count": 1, "records": [_part_record(tmp_path)]}},
    )
    config = Config()
    config.seo.storage_path = str(tmp_path)
    yt = MagicMock()
    yt.upload_if_needed.return_value = {"video_id": "PARTID"}

    esc._upload_superlong_parts_to_youtube(
        yt=yt,
        state=state,
        output_base=tmp_path,
        config=config,
        save_state=lambda s: None,
    )

    assert yt.upload_if_needed.call_count == 1
    _, kwargs = yt.upload_if_needed.call_args
    assert kwargs["tags"], "superlong part upload must carry non-empty tags"
    assert kwargs.get("recording_date") == "2026-03-01"
