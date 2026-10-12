"""An earlier change taught ``resolve_superlong_publish_day`` rules 3/4 (previous
related-video weekday, else next open day) but NO production caller ever
supplied per-video identity — so the feature was inert. This wires
``project_name``/``title`` through the two real booking sites in
``_upload_superlong_parts_to_youtube`` (``next_free_superlong_slot`` at
upload time and the post-upload overrun re-check) and proves it here at
the production call, not the resolver."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.content_ledger import EST, ContentEntry, ContentLedger
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

from _state_helpers import make_state

ANCHOR = datetime(2026, 8, 10, 9, 0, tzinfo=EST)  # a Monday


class _FrozenDatetime(datetime):
    """Freezes ``datetime.now()`` inside ``esc`` to ANCHOR so booking
    outcomes are deterministic regardless of the real wall clock."""

    @classmethod
    def now(cls, tz=None):
        return ANCHOR


@pytest.fixture(autouse=True)
def _freeze_now(monkeypatch):
    monkeypatch.setattr(esc, "datetime", _FrozenDatetime)


def _part_record(tmp_path, title="Big Trip — Part 2 of 3"):
    path = tmp_path / "sl_02.mp4"
    path.write_bytes(b"x")
    return {"path": str(path), "part_index": 2, "title": title, "description": "d"}


def _state(tmp_path, title="Big Trip — Part 2 of 3"):
    return make_state(
        region="usa",
        superlong_parts={
            "2026-08-11": {"part_count": 1, "records": [_part_record(tmp_path, title)]}
        },
    )


def _related_entry(date: str, project_name: str, title: str) -> ContentEntry:
    return ContentEntry(
        date=date,
        time="11:00:00",
        content_type="longform",
        title=title,
        project_name=project_name,
        status="scheduled",
        source="pipeline",
        region="usa",
    )


def _run(tmp_path, config, project_base="roadtrip", title="Big Trip — Part 2 of 3"):
    state = _state(tmp_path, title)
    yt = MagicMock()
    yt.upload_if_needed.return_value = {"video_id": "PARTID"}
    esc._upload_superlong_parts_to_youtube(
        yt=yt,
        state=state,
        output_base=tmp_path,
        config=config,
        save_state=lambda s: None,
        project_base=project_base,
    )
    _, kwargs = yt.upload_if_needed.call_args
    return kwargs["publish_at"]


def test_related_history_weekday_wins_over_flat_default(tmp_path):
    """A RELATED prior superlong part (same project_name) in the ledger sets
    the day, not the flat ``superlong_publish_day`` default ("sat")."""
    config = Config()
    config.seo.storage_path = str(tmp_path)
    ledger = ContentLedger()
    ledger.entries.append(
        _related_entry("2026-08-04", "roadtrip", "Big Trip — Part 1 of 3")
    )  # a Tuesday
    ledger.save(tmp_path / ".content_ledger.json")

    publish_at = _run(tmp_path, config, project_base="roadtrip")

    assert publish_at.weekday() == 1  # Tuesday — the related part's weekday
    assert publish_at.weekday() != 5  # never the flat "sat" default


def test_no_related_history_lands_on_the_next_open_day(tmp_path):
    """No related history at all — with identity supplied the caller must
    search every weekday (rule 4), landing well before the next flat-default
    Saturday, not waiting for one."""
    config = Config()
    config.seo.storage_path = str(tmp_path)
    ledger = ContentLedger()  # nothing booked
    ledger.save(tmp_path / ".content_ledger.json")

    publish_at = _run(tmp_path, config, project_base="roadtrip")

    # ANCHOR is a Monday with nothing booked anywhere — any_day search lands
    # well before the following Saturday (2026-08-15).
    assert publish_at.date() < datetime(2026, 8, 15, tzinfo=EST).date()
    assert publish_at.weekday() != 5


def test_explicit_configured_day_still_wins_end_to_end(tmp_path):
    """Precedence rung 1 (explicit operator override) still beats both
    ledger history and the any-day search, even now that identity flows
    through the production booking call."""
    config = Config()
    config.seo.storage_path = str(tmp_path)
    config.drive.superlong_publish_day = "mon"
    ledger = ContentLedger()
    ledger.entries.append(
        _related_entry("2026-08-04", "roadtrip", "Big Trip — Part 1 of 3")
    )  # a Tuesday — contradicts the explicit "mon" override
    ledger.save(tmp_path / ".content_ledger.json")

    publish_at = _run(tmp_path, config, project_base="roadtrip")

    assert publish_at.weekday() == 0  # Monday — the explicit override


def test_unrelated_superlong_part_does_not_influence_the_day(tmp_path):
    """A superlong part booked under a DIFFERENT project must not be
    mistaken for related history."""
    config = Config()
    config.seo.storage_path = str(tmp_path)
    ledger = ContentLedger()
    ledger.entries.append(
        _related_entry("2026-08-05", "some_other_trip", "Other Trip — Part 1 of 2")
    )  # a Wednesday, but a DIFFERENT project
    ledger.save(tmp_path / ".content_ledger.json")

    publish_at = _run(tmp_path, config, project_base="roadtrip")

    assert publish_at.weekday() != 2  # not the unrelated part's Wednesday
    assert publish_at.date() < datetime(2026, 8, 15, tzinfo=EST).date()
