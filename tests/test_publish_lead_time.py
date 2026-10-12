"""A superlong part's publish slot is resolved BEFORE its own multi-hour,
tens-of-GB upload begins. The upload can overrun that slot, so the
resolved ``publish_at`` must be re-checked AFTER the upload completes,
before the ledger entry is booked, and re-resolved (on YouTube too, via
``update_privacy``) if the original slot has been overrun."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

from gomoveshift.config import Config
from gomoveshift.pipeline.content_ledger import EST
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

from _state_helpers import make_state

_STALE_PUBLISH_AT = "2020-01-01T11:00:00-05:00"  # long past — any real "now" overruns it


def _part_record(tmp_path):
    path = tmp_path / "sl_01.mp4"
    path.write_bytes(b"x")
    return {"path": str(path), "part_index": 1, "title": "Superlong Part", "description": "d"}


def _state(tmp_path):
    return make_state(
        region="usa",
        superlong_parts={"2026-03-01": {"part_count": 1, "records": [_part_record(tmp_path)]}},
    )


def test_overrun_publish_at_is_rerechecked_and_rebooked(tmp_path):
    state = _state(tmp_path)
    config = Config()
    config.seo.storage_path = str(tmp_path)
    yt_mock = MagicMock()
    yt_mock.upload_if_needed.return_value = {
        "video_id": "PARTID",
        "publish_at": _STALE_PUBLISH_AT,
        "title": "Superlong Part",
    }

    sidecar_uploads = esc._upload_superlong_parts_to_youtube(
        yt=yt_mock,
        state=state,
        output_base=tmp_path,
        config=config,
        save_state=lambda s: None,
    )

    # YouTube itself must be told about the new slot — the stale
    # publishAt was already sent at upload time.
    yt_mock.update_privacy.assert_called_once()
    args, kwargs = yt_mock.update_privacy.call_args
    assert (args and args[0] == "PARTID") or kwargs.get("video_id") == "PARTID"
    new_publish_at = kwargs.get("publish_at")
    now = datetime.now(EST)
    assert new_publish_at is not None and new_publish_at > now

    saved_publish_at = sidecar_uploads["2026-03-01:01"]["publish_at"]
    assert datetime.fromisoformat(saved_publish_at).astimezone(EST) > now, (
        f"sidecar still carries the stale slot: {saved_publish_at}"
    )

    ledger_path = tmp_path / ".content_ledger.json"
    ledger_data = json.loads(ledger_path.read_text())
    (booked_entry,) = ledger_data["entries"]
    booked_dt = datetime.strptime(
        f"{booked_entry['date']} {booked_entry['time']}", "%Y-%m-%d %H:%M:%S"
    ).replace(tzinfo=EST)
    assert booked_dt > now, f"ledger booked the stale slot: {booked_entry}"
