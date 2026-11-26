"""A day reel published without a tracking record gets one at publish.

Day reels queued before the enqueue step wrote a record (docs/pipeline/186)
have nothing for ``update_platform_id`` to update. Dropping the id there
left them out of ``metrics collect`` and the growth review for good, so the
record is written at the one moment the id exists. Drive Shorts keep the
documented ``ValueError`` contract.
"""

from __future__ import annotations

import pytest

from gomoveshift.tracking.store import TrackingStore


def test_a_day_reel_with_no_record_is_registered_by_its_publish(tmp_path):
    store = TrackingStore(root=tmp_path)
    store.update_platform_id(
        "day-reel-abc123",
        0,
        "youtube",
        "vid42",
        posted_at="2026-11-26T11:17:00+00:00",
        posted_at_source="platform_confirmed",
    )
    short = store.load("day-reel-abc123").shorts[0]
    assert (short.index, short.platform_ids) == (0, {"youtube": "vid42"})
    assert short.posted_at == "2026-11-26T11:17:00+00:00"


def test_a_second_platform_joins_the_same_day_reel_record(tmp_path):
    store = TrackingStore(root=tmp_path)
    store.update_platform_id("day-reel-abc123", 0, "youtube", "vid42")
    store.update_platform_id("day-reel-abc123", 0, "facebook", "fb7")
    shorts = store.load("day-reel-abc123").shorts
    assert len(shorts) == 1
    assert shorts[0].platform_ids == {"youtube": "vid42", "facebook": "fb7"}


def test_a_drive_short_with_no_record_still_refuses(tmp_path):
    store = TrackingStore(root=tmp_path)
    store.register_short("nctopa-2026-03-05-rec0063", 1)
    with pytest.raises(ValueError):
        store.update_platform_id("nctopa-2026-03-05-rec0063", 9, "youtube", "v")
    with pytest.raises(FileNotFoundError):
        store.update_platform_id("some-other-project", 1, "youtube", "v")
