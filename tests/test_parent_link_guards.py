"""A Short links its full drive only when that drive is public by the time the
Short is: the plan's YouTube description and every first comment follow the
same rule as the reviewed description link (``parent_is_public_at``).

Tmp sqlite DBs only -- no network, never the live queue."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gomoveshift.publishing.enqueue_from_plan import (
    backfill_first_comment_text,
    enqueue_from_plan_sidecar,
)
from gomoveshift.publishing.first_comment import parent_is_public_at
from gomoveshift.publishing.plan_builder import build_short_payloads
from gomoveshift.publishing.queue import enqueue, open_queue_db

from test_enqueue_from_plan import _record, _write_sidecar

_SHORT_AT = "2026-09-07T16:00:00+00:00"  # _record's default slot
_BEFORE = "2026-09-07T12:00:00+00:00"
_AFTER = "2026-09-08T12:00:00+00:00"
_TEXT = "🎬 Watch the full drive: https://youtu.be/PARENT1"


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _utc(iso):
    return datetime.fromisoformat(iso)


# --- the rule ---------------------------------------------------------------


@pytest.mark.parametrize(
    "short_at, parent_at, live",
    [
        (_SHORT_AT, _BEFORE, True),
        (_SHORT_AT, _SHORT_AT, True),
        (_SHORT_AT, _AFTER, False),
        (_utc(_SHORT_AT), _utc(_BEFORE), True),
        (_utc(_SHORT_AT), _utc(_AFTER), False),
        ("2026-09-07T16:00:00Z", _BEFORE, True),
    ],
)
def test_the_parent_is_public_only_at_or_before_the_short(short_at, parent_at, live):
    assert parent_is_public_at(short_at, parent_at) is live


@pytest.mark.parametrize(
    "short_at, parent_at",
    [
        (None, _BEFORE),
        (_SHORT_AT, None),
        ("", _BEFORE),
        (_SHORT_AT, "not a time"),
        ("2026-09-07T16:00:00", _BEFORE),  # unzoned: cannot be ordered
        (_SHORT_AT, datetime(2026, 9, 7, 12)),
    ],
)
def test_an_unknown_or_unzoned_time_is_never_public(short_at, parent_at):
    assert parent_is_public_at(short_at, parent_at) is False


# --- first comment at enqueue -----------------------------------------------


def _comments(conn):
    return dict(conn.execute("SELECT platform, first_comment_text FROM deliveries").fetchall())


def test_a_short_public_before_its_drive_gets_no_first_comment(conn, tmp_path):
    record = _record(selected=("youtube", "instagram"))
    record.update(parent_lf_video_id="PARENT1", parent_lf_publish_at=_AFTER)
    enqueue_from_plan_sidecar(conn, _write_sidecar(tmp_path, [record]))
    assert _comments(conn) == {"youtube": None, "instagram": None}


def test_a_record_with_no_parent_time_gets_no_first_comment(conn, tmp_path):
    record = _record(selected=("youtube",))
    record["parent_lf_video_id"] = "PARENT1"
    enqueue_from_plan_sidecar(conn, _write_sidecar(tmp_path, [record]))
    assert _comments(conn) == {"youtube": None}


def test_each_platform_is_judged_at_its_own_time(conn, tmp_path):
    record = _record(selected=("youtube", "instagram"))
    record.update(parent_lf_video_id="PARENT1", parent_lf_publish_at="2026-09-07T18:00:00+00:00")
    record["schedule"]["scheduled_at_utc_by_platform"] = {"instagram": "2026-09-07T20:00:00+00:00"}
    enqueue_from_plan_sidecar(conn, _write_sidecar(tmp_path, [record]))
    assert _comments(conn) == {"youtube": None, "instagram": _TEXT}


# --- first comment backfill -------------------------------------------------


@pytest.mark.parametrize("parent_at, filled", [(_BEFORE, 1), (_AFTER, 0), (None, 0)])
def test_backfill_links_only_a_drive_public_by_the_slot(conn, tmp_path, parent_at, filled):
    record = _record(short_index=7, selected=("youtube",))
    record["parent_lf_video_id"] = "PARENT1"
    if parent_at:
        record["parent_lf_publish_at"] = parent_at
    path = _write_sidecar(tmp_path, [record])
    delivery_id = enqueue(
        conn, "p", 7, "youtube", "hash-a", sidecar_path=str(path), scheduled_at=_SHORT_AT
    )
    assert backfill_first_comment_text(conn).filled == filled
    text = conn.execute(
        "SELECT first_comment_text FROM deliveries WHERE id=?", (delivery_id,)
    ).fetchone()[0]
    assert text == (_TEXT if filled else None)


# --- the plan's YouTube description -----------------------------------------


def _post(description="Drive through Vilas County"):
    return SimpleNamespace(
        youtube_title="Vilas County",
        youtube_description=description,
        youtube_tags="",
        youtube_category="",
        captions={},
        caption="",
        tiktok_title="",
    )


def _description(post, parent_at):
    return build_short_payloads(
        post,
        recording_date="2026-09-01",
        parent_id="abcdefghijk",
        tiktok_privacy_level="SELF_ONLY",
        short_at=_SHORT_AT,
        parent_at=parent_at,
    ).youtube.description


def test_the_description_links_a_drive_already_public():
    assert _description(_post(), _BEFORE) == (
        "Drive through Vilas County\n\nhttps://youtu.be/abcdefghijk"
    )


@pytest.mark.parametrize("parent_at", [_AFTER, None])
def test_the_description_never_links_a_drive_not_yet_public(parent_at):
    assert _description(_post(), parent_at) == "Drive through Vilas County"


def test_an_empty_description_stays_empty_without_the_link():
    assert _description(_post(""), _AFTER) == ""
