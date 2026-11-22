"""Re-render a drive and replace its still-scheduled uploads (``--replace-scheduled``).

The classifier fails closed; Phase D keeps public LFs, uploads a replaced LF with the old
``publishAt`` and title, and deletes the old video only after the new upload succeeded."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gomoveshift.pipeline import replace_scheduled as rs
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

from _state_helpers import make_state

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)


def _rec(vid, when, **kw):
    return {"video_id": vid, "url": f"https://youtu.be/{vid}", "publish_at": when.isoformat(),
            "title": f"T {vid}", "uploaded_at": "2026-11-01T00:00:00+00:00",
            "playlist_id": "PL1", **kw}  # fmt: skip


PUBLIC = _rec("pub", NOW - timedelta(days=2))
SOON = NOW + timedelta(days=5)
LATER = NOW + timedelta(days=12)


def _live(states):
    return lambda ids: {i: states[i] for i in ids if i in states}


def test_all_scheduled_is_accepted():
    uploads = {"1": _rec("a", SOON), "2": _rec("b", LATER)}
    plan, why = rs.classify_uploads(uploads, NOW, _live({"a": "scheduled", "b": "scheduled"}))
    assert why == "" and plan["kept"] == {} and set(plan["replace"]) == {"1", "2"}
    assert plan["replace"]["1"]["video_id"] == "a" and plan["replace"]["1"]["playlist_id"] == "PL1"


def test_public_and_scheduled_mix_keeps_the_public_lf():
    uploads = {"1": PUBLIC, "2": _rec("b", SOON), "3": _rec("c", LATER)}
    live = _live({"b": "scheduled", "c": "scheduled"})
    plan, why = rs.classify_uploads(uploads, NOW, live)
    assert why == "" and set(plan["kept"]) == {"1"} and set(plan["replace"]) == {"2", "3"}


def test_a_record_inside_the_margin_refuses():
    near = _rec("n", NOW + timedelta(hours=rs.REPLACE_MARGIN_H - 1))
    plan, why = rs.classify_uploads({"1": near}, NOW, _live({"n": "scheduled"}))
    assert plan is None and "margin" in why


def test_a_live_check_that_says_public_refuses():
    plan, why = rs.classify_uploads({"1": _rec("a", SOON)}, NOW, _live({"a": "public"}))
    assert plan is None and "public" in why


def test_a_video_missing_on_youtube_refuses():
    plan, why = rs.classify_uploads({"1": _rec("a", SOON)}, NOW, _live({}))
    assert plan is None and "not found" in why


def test_a_failing_live_lookup_refuses():
    def boom(ids):
        raise RuntimeError("quota")

    plan, why = rs.classify_uploads({"1": _rec("a", SOON)}, NOW, boom)
    assert plan is None and "live YouTube check failed" in why


def test_unreadable_records_refuse():
    for bad in ("abcDEF12345", {"video_id": "a"}, {"video_id": "a", "publish_at": "soon"}):
        plan, why = rs.classify_uploads({"1": bad}, NOW, _live({"a": "scheduled"}))
        assert plan is None and "not decidable" in why


def test_no_live_client_refuses():
    plan, why = rs.classify_uploads({"1": _rec("a", SOON)}, NOW, None)
    assert plan is None and "live" in why
