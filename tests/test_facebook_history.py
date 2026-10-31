"""Facebook history ledger (plan U7): Page Reels posted before first-party
publishing (2026-09-10) went out through Metricool, so the tracking store has
no record of them. The ledger lists them once, tags ``source``, and marks the
Reels whose captions were hand-edited on 2026-10-02. No network: fake session."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from gomoveshift.cli.handlers import metrics as metrics_handler
from gomoveshift.tracking import facebook_history as fh
from gomoveshift.tracking.collector import _FB_MAX_INSIGHT_PAGES, FacebookCollector
from gomoveshift.tracking.schema import PlatformMetrics

EDITED_ID = json.loads(fh.CAPTION_EDITS_PATH.read_text(encoding="utf-8"))["reel_ids"][0]


class _FakeResponse:
    def __init__(self, status_code: int, body: dict):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, *responses: _FakeResponse):
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


def _creds(scopes=("pages_read_engagement",)):
    return {"page_id": "page_1", "page_access_token": "tok", "scopes": list(scopes)}


def _reel(rid, created="2026-08-01T15:00:00+0000", desc="cap"):
    return {"id": rid, "created_time": created, "description": desc}


class _FakeCollector:
    """Stands in for FacebookCollector in ledger tests."""

    def __init__(self, reels):
        self._reels = reels
        self.fetched = []

    def list_page_reels(self, since, until):
        self.since_until = (since, until)
        return self._reels

    def collect_short_metrics(self, video_id):
        self.fetched.append(video_id)
        return PlatformMetrics(views=7)


def _ledger_file(root, brand="gomoveshift"):
    return root / brand / "tracking" / "facebook_history.json"


# -- collector listing ---------------------------------------------------------


def test_list_page_reels_follows_paging_and_sends_fields_and_bounds():
    session = _FakeSession(
        _FakeResponse(
            200, {"data": [_reel("1")], "paging": {"next": "https://graph.facebook.com/next2"}}
        ),
        _FakeResponse(200, {"data": [_reel("2")]}),
    )
    c = FacebookCollector(credentials=_creds(), session=session)

    reels = c.list_page_reels("2026-06-01", "2026-09-10")

    assert [r["id"] for r in reels] == ["1", "2"]
    url, params = session.calls[0]
    assert url.endswith("/page_1/video_reels")
    assert params["fields"] == "id,created_time,description"
    assert (params["since"], params["until"]) == ("2026-06-01", "2026-09-10")
    assert session.calls[1][0] == "https://graph.facebook.com/next2"


def test_list_page_reels_stops_at_the_page_cap():
    page = _FakeResponse(
        200, {"data": [_reel("x")], "paging": {"next": "https://graph.facebook.com/again"}}
    )
    session = _FakeSession(page)
    c = FacebookCollector(credentials=_creds(), session=session)

    assert len(c.list_page_reels("2026-06-01", "2026-09-10")) == _FB_MAX_INSIGHT_PAGES
    assert len(session.calls) == _FB_MAX_INSIGHT_PAGES


def test_list_page_reels_none_on_refusal_or_missing_creds_empty_on_empty():
    refused = _FakeSession(_FakeResponse(400, {"error": {"message": "nope"}}))
    assert (
        FacebookCollector(credentials=_creds(), session=refused).list_page_reels("a", "b") is None
    )
    later = _FakeSession(
        _FakeResponse(200, {"data": [_reel("1")], "paging": {"next": "https://g/n"}}),
        _FakeResponse(500, {"error": {"message": "late"}}),
    )
    assert FacebookCollector(credentials=_creds(), session=later).list_page_reels("a", "b") is None
    no_creds = FacebookCollector(credentials={}, session=_FakeSession(_FakeResponse(200, {})))
    assert no_creds.list_page_reels("a", "b") is None
    empty = _FakeSession(_FakeResponse(200, {"data": []}))
    assert FacebookCollector(credentials=_creds(), session=empty).list_page_reels("a", "b") == []


# -- ledger ----------------------------------------------------------------------


def test_source_splits_on_the_pacific_day_of_2026_09_10():
    # 2026-09-10T05:00Z is still Sept 9 in Pacific (UTC-7): Metricool era.
    assert fh.source_for("2026-09-10T05:00:00+0000") == "metricool"
    assert fh.source_for("2026-09-10T07:00:00+0000") == "first_party"
    assert fh.source_for("2026-09-09T20:00:00+0000") == "metricool"


def test_backfill_stores_entries_and_is_idempotent(tmp_path):
    c = _FakeCollector([_reel("a", desc="hello"), _reel("b", created="2026-09-11T20:00:00+0000")])

    assert fh.backfill_facebook_history(c, "gomoveshift", root=tmp_path) == (2, 2)
    first = _ledger_file(tmp_path).read_text()
    by_id = {e["id"]: e for e in json.loads(first)["entries"]}
    assert by_id["a"]["source"] == "metricool"
    assert by_id["a"]["pacific_day"] == "2026-08-01"
    assert by_id["a"]["caption"] == "hello"
    assert by_id["a"]["metrics"]["views"] == 7
    assert by_id["a"]["metrics"]["reach"] is None
    assert by_id["b"]["source"] == "first_party"
    assert c.since_until == ("2026-06-01", "2026-09-10")

    c.fetched.clear()
    assert fh.backfill_facebook_history(c, "gomoveshift", root=tmp_path) == (2, 2)
    assert c.fetched == []  # stored entries are never re-fetched
    assert json.loads(_ledger_file(tmp_path).read_text())["entries"] == json.loads(first)["entries"]


def test_rerun_keeps_entries_the_listing_no_longer_returns(tmp_path):
    fh.backfill_facebook_history(_FakeCollector([_reel("a"), _reel("b")]), "g", root=tmp_path)
    fh.backfill_facebook_history(_FakeCollector([_reel("a")]), "g", root=tmp_path)
    assert {e["id"] for e in fh.load_history("g", tmp_path)["entries"]} == {"a", "b"}


def test_refused_listing_writes_nothing_and_stays_incomplete(tmp_path):
    assert fh.backfill_facebook_history(_FakeCollector(None), "g", root=tmp_path) == (None, 0)
    assert not _ledger_file(tmp_path, "g").exists()
    assert not fh.is_complete("g", tmp_path)


def test_complete_once_a_listing_succeeded_even_when_empty(tmp_path):
    assert not fh.is_complete("g", tmp_path)
    fh.backfill_facebook_history(_FakeCollector([]), "g", root=tmp_path)
    assert fh.is_complete("g", tmp_path)


def test_corrupt_ledger_loads_empty(tmp_path):
    path = _ledger_file(tmp_path, "g")
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert fh.load_history("g", tmp_path)["entries"] == []
    assert not fh.is_complete("g", tmp_path)


def test_edited_ids_are_marked_and_leave_the_matched_cohort(tmp_path):
    c = _FakeCollector([_reel(EDITED_ID), _reel("other")])
    fh.backfill_facebook_history(c, "g", root=tmp_path)
    by_id = {e["id"]: e for e in fh.load_history("g", tmp_path)["entries"]}
    assert by_id[EDITED_ID]["caption_edited_on"] == "2026-10-02"
    assert "caption_edited_on" not in by_id["other"]
    assert fh.in_matched_cohort(by_id[EDITED_ID]) is False
    assert fh.in_matched_cohort(by_id["other"]) is True


def test_caption_edits_data_file_lists_the_eighteen_reels():
    data = json.loads(fh.CAPTION_EDITS_PATH.read_text(encoding="utf-8"))
    assert data["edited_on"] == "2026-10-02"
    assert len(data["reel_ids"]) == len(set(data["reel_ids"])) == 18


# -- nightly handler ----------------------------------------------------------------


def test_handler_prints_one_line_and_runs_only_while_incomplete(capsys):
    collector = SimpleNamespace(facebook=object())
    with (
        patch.object(metrics_handler, "facebook_history_complete", return_value=False),
        patch.object(metrics_handler, "backfill_facebook_history", return_value=(5, 5)) as backfill,
    ):
        metrics_handler._pull_facebook_history(collector, "gomoveshift")
    assert backfill.call_args.args[:2] == (collector.facebook, "gomoveshift")
    assert capsys.readouterr().out.strip() == "FB_HISTORY listed=5 stored=5"

    with (
        patch.object(metrics_handler, "facebook_history_complete", return_value=True),
        patch.object(metrics_handler, "backfill_facebook_history") as backfill,
    ):
        metrics_handler._pull_facebook_history(collector, "gomoveshift")
    backfill.assert_not_called()
    assert capsys.readouterr().out == ""


def test_handler_prints_unavailable_when_listing_refused(capsys):
    with (
        patch.object(metrics_handler, "facebook_history_complete", return_value=False),
        patch.object(metrics_handler, "backfill_facebook_history", return_value=(None, 0)),
    ):
        metrics_handler._pull_facebook_history(SimpleNamespace(facebook=object()), "g")
    assert capsys.readouterr().out.strip() == "FB_HISTORY listed=unavailable stored=0"


def test_handler_swallows_failures(capsys, caplog):
    with (
        patch.object(metrics_handler, "facebook_history_complete", return_value=False),
        patch.object(metrics_handler, "backfill_facebook_history", side_effect=RuntimeError("x")),
    ):
        metrics_handler._pull_facebook_history(SimpleNamespace(facebook=object()), "g")
    assert "FB_HISTORY" not in capsys.readouterr().out
    assert "facebook history backfill skipped" in caplog.text


def test_an_edited_id_leaves_the_cohort_even_outside_the_ledger():
    # A first-party tracking record (posted after the cutover, so never in the
    # ledger) carries only its id; the edit list still excludes it.
    edited = "1841588706835941"
    assert fh.caption_edited_on(edited) == "2026-10-02"
    assert fh.caption_edited_on("not-edited") is None
    assert fh.in_matched_cohort({"id": edited}) is False
    assert fh.in_matched_cohort({"video_id": edited}) is False
    assert fh.in_matched_cohort({"id": "not-edited"}) is True
