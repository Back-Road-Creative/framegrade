# ruff: noqa: F811  (the conn fixture is imported from the missing_remote test module)
"""``queue recover-terminal`` on a YouTube ``failed_permanent`` row the final-text
review refused (``error_class`` ``FinalTextReview``): it never reached
``videos.insert``, so once the review agrees with the Short as sent today the one
audited way back is a new linked row. Every gate refuses and writes nothing.
All offline -- the YouTube reader, loaders and review check are fakes."""

from __future__ import annotations

from dataclasses import replace

import pytest
from test_youtube_missing_remote_recover import (
    NOW,
    FakeYouTube,
    _attested,
    _deps,
    _record,
    _refuses,
    _snapshot,
    conn,  # noqa: F401 -- fixture
)

from gomoveshift.cli.handlers.queue_recover import recover_terminal
from gomoveshift.publishing.queue import enqueue

DETAIL = "LF 2: final-text review refused (verdict:refused): missing_field:approval_hash"


def _refused_row(conn, *, error_class="FinalTextReviewError", upload_id=None):
    did = enqueue(conn, "proj", 4, "youtube", "hash-yt", sidecar_path="/x/plan.json")
    conn.execute(
        "UPDATE deliveries SET status='failed_permanent', error_class=?, error_detail=?, "
        "platform_upload_id=? WHERE id=?",
        (error_class, DETAIL, upload_id, did),
    )
    conn.commit()
    return did


def _agrees(record):
    return None


def _still_refused(record):
    return "name_unbacked:short_title:epic"


def _with_check(deps, check):
    return replace(deps, review_check=check)


def test_refused_row_recovers_once_the_review_agrees(conn, tmp_path):
    did = _refused_row(conn)
    reader = FakeYouTube()
    new_id = recover_terminal(
        conn, did, "operator 2026-11-12", _with_check(_deps(tmp_path, reader), _agrees), now=NOW
    )
    row = conn.execute("SELECT * FROM deliveries WHERE id=?", (new_id,)).fetchone()
    assert row["status"] == "queued" and row["recovers_delivery_id"] == did
    assert ("uploads", 200) in reader.calls  # a fresh duplicate scan, never a trusted id


def test_dry_run_runs_every_gate_and_writes_nothing(conn, tmp_path):
    did = _refused_row(conn)
    before = _snapshot(conn)
    assert (
        recover_terminal(
            conn, did, "operator", _with_check(_deps(tmp_path), _agrees), now=NOW, apply=False
        )
        is None
    )
    assert _snapshot(conn) == before


def test_a_review_that_still_refuses_blocks_recovery(conn, tmp_path):
    did = _refused_row(conn)
    _refuses(conn, did, _with_check(_deps(tmp_path), _still_refused), "name_unbacked")


def test_a_row_that_failed_for_another_reason_stays_refused(conn, tmp_path):
    did = _refused_row(conn, error_class="RuntimeError")
    _refuses(conn, did, _with_check(_deps(tmp_path), _agrees), "recovers missing_remote only")


def test_a_row_that_holds_an_upload_id_is_refused(conn, tmp_path):
    did = _refused_row(conn, upload_id="LIVEvid0001")
    _refuses(conn, did, _with_check(_deps(tmp_path), _agrees), "holds an upload id")


def test_a_title_already_on_the_channel_is_refused(conn, tmp_path):
    did = _refused_row(conn)
    title = _record().payloads.youtube.title
    reader = FakeYouTube(uploads=[{"id": "dup1", "title": title}])
    _refuses(conn, did, _with_check(_deps(tmp_path, reader), _agrees), "already carries")


def test_a_passed_slot_is_refused(conn, tmp_path):
    did = _refused_row(conn)
    deps = _with_check(_deps(tmp_path, record=_record(slot="2026-10-01T14:00:00Z")), _agrees)
    _refuses(conn, did, deps, "has passed")


def test_a_privacy_hold_is_refused(conn, tmp_path):
    did = _refused_row(conn)
    deps = _with_check(_deps(tmp_path, privacy=_attested("inconclusive")), _agrees)
    _refuses(conn, did, deps, "privacy hold")
