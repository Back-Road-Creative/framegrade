"""Rule ``seo_feedback_stale``: stale arm decisions and one grouped row of settled Shorts with no
core metrics. Detection only; every condition resolves when it clears."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.health import run_check, seo_feedback, short_text_sync
from gomoveshift.health.check import _line
from gomoveshift.health.store import load_all
from gomoveshift.publishing.experiments import ARM_DECISIONS_FILENAME
from gomoveshift.tracking.day7 import window_settled
from gomoveshift.tracking.schema import ContentMeta, PlatformMetrics, ShortRecord, TrackingRecord
from gomoveshift.tracking.store import TrackingStore

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
K_DEC = "seo_feedback_stale:decisions"
K_MET = "seo_feedback_stale:metrics"
K_SYNC = "seo_feedback_stale:sync"


class World:
    """tmp_path fakes: a seo storage dir, a tracking root and an incidents dir."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.storage = tmp_path / "seo"
        self.storage.mkdir()
        self.tracking = tmp_path / "track"
        self.tracking.mkdir()
        self.pushed: list = []
        self.now = NOW
        self.auto_stamp = True  # a fresh sync run record at every check, unless a test sets it

    def stamp_path(self) -> Path:
        return short_text_sync.run_record_path(self.tmp / "inc")

    def stamp(self, *, age_h=1.0):
        at = self.now - timedelta(hours=age_h)
        self.stamp_path().write_text(
            json.dumps(
                {
                    "schema": 1,
                    "ran_at": at.isoformat(),
                    "changed": 0,
                    "refused": 0,
                    "crashed": False,
                }
            )
        )

    def decisions(self, *, age_h=1.0, decisions=()):
        at = self.now - timedelta(hours=age_h)
        (self.storage / ARM_DECISIONS_FILENAME).write_text(
            json.dumps({"schema": 1, "decided_at": at.isoformat(), "decisions": list(decisions)})
        )

    def short(self, index=1, *, views=5, pct=50.0, age_days=20, ytid="yt1", metrics=True):
        posted = NOW - timedelta(days=age_days)
        pm = PlatformMetrics(views=views, avg_view_percentage=pct)
        rec = ShortRecord(
            index=index,
            posted_at=posted.isoformat(),
            posted_at_source="platform_confirmed",
            platform_ids={"youtube": ytid} if ytid else {},
            content=ContentMeta(hook="h"),
            metrics={"youtube": pm} if metrics else {},
        )
        store = TrackingStore(self.tracking)
        old = store.load("p") if store.exists("p") else TrackingRecord(project="p", shorts=[])
        old.shorts = [s for s in old.shorts if s.index != index] + [rec]
        store.save(old)

    def check(self):
        if self.auto_stamp:
            self.stamp()
        cfg = SimpleNamespace(seo=SimpleNamespace(storage_path=str(self.storage)))

        def notify(event, payload, text):
            self.pushed.append((event, payload["incident"] if "incident" in payload else None))
            return True

        return run_check(
            cfg,
            incidents_dir=self.tmp / "inc",
            jobs_dir=self.tmp / "jobs",
            code_root=self.tmp,
            staging_sha="abc",
            now=self.now,
            notify_fn=notify,
            tracking_root=self.tracking,
        )

    def rows(self):
        rows, _bad = load_all(self.tmp / "inc")
        return {r["key"]: r for r in rows if r["kind"] == "seo_feedback_stale"}

    def open_keys(self):
        return {k for k, r in self.rows().items() if not r.get("resolved_at")}


def _fresh_world(tmp_path) -> World:
    w = World(tmp_path)
    w.decisions()
    return w


# --- (a) stale decisions ----------------------------------------------------------


def test_decisions_older_than_the_constant_open_and_a_fresh_write_resolves(tmp_path):
    w = World(tmp_path)
    w.decisions(age_h=seo_feedback.DECISIONS_STALE_AFTER_H + 1)
    res = w.check()
    assert K_DEC in res["new"] and w.open_keys() == {K_DEC}
    assert ("health_seo_feedback_stale", K_DEC) in w.pushed
    w.check()
    assert w.pushed.count(("health_seo_feedback_stale", K_DEC)) == 1  # notified once
    w.decisions(age_h=1)
    res = w.check()
    assert K_DEC in res["resolved"] and w.open_keys() == set()


def test_decisions_exactly_at_the_constant_are_not_stale(tmp_path):
    w = World(tmp_path)
    w.decisions(age_h=seo_feedback.DECISIONS_STALE_AFTER_H)
    w.check()
    assert w.open_keys() == set()


def test_missing_and_unreadable_decisions_file_are_the_finding(tmp_path):
    w = World(tmp_path)
    w.check()
    assert w.open_keys() == {K_DEC}
    assert "missing" in w.rows()[K_DEC]["detail"]
    w.decisions()
    w.check()
    assert w.open_keys() == set()  # written: resolved
    (w.storage / ARM_DECISIONS_FILENAME).write_text("{not json")
    w.check()
    assert w.open_keys() == {K_DEC} and "unreadable" in _line(w.rows()[K_DEC])


def test_relative_storage_is_a_note_and_files_nothing(tmp_path):
    cfg = SimpleNamespace(seo=SimpleNamespace(storage_path="relative/seo"))
    res = run_check(
        cfg,
        incidents_dir=tmp_path / "inc",
        jobs_dir=tmp_path / "jobs",
        code_root=tmp_path,
        staging_sha="abc",
        now=NOW,
        notify_fn=lambda *a: True,
        tracking_root=tmp_path / "t",
    )
    assert not [k for k in res["new"] if k.startswith("seo_feedback_stale")]
    assert any("seo_feedback_stale" in n for n in res["notes"])


# --- (b) missing metrics ----------------------------------------------------------


def test_settled_short_with_null_core_metrics_opens_and_resolves_when_filled(tmp_path):
    w = _fresh_world(tmp_path)
    w.short(1, views=None)
    assert window_settled((NOW - timedelta(days=20)).date(), NOW.date())
    res = w.check()
    assert res["new"] == [K_MET] and w.open_keys() == {K_MET}
    assert w.rows()[K_MET]["count"] == 1 and "p#1 (views)" in w.rows()[K_MET]["detail"]
    w.short(1, views=5, pct=None)  # views back, retention still null: still open
    w.check()
    assert w.open_keys() == {K_MET}
    w.short(1, views=5, pct=40.0)
    res = w.check()
    assert res["resolved"] == [K_MET] and w.open_keys() == set()


def test_many_shorts_are_one_row_with_a_count_ten_names_and_the_rest_counted(tmp_path):
    w = _fresh_world(tmp_path)
    for i in range(1, 13):
        w.short(i, pct=None)
    res = w.check()
    assert res["new"] == [K_MET]
    row = w.rows()[K_MET]
    assert row["count"] == 12 and len(row["shorts"]) == 12
    assert row["detail"].startswith("12 settled")
    assert row["detail"].count("p#") == seo_feedback.LISTED
    assert "and 2 more" in row["detail"]
    assert w.pushed.count(("health_seo_feedback_stale", K_MET)) == 1


def test_a_short_aging_out_of_the_lookback_resolves_the_count_without_a_fix(tmp_path):
    w = _fresh_world(tmp_path)
    w.short(1, views=None, age_days=seo_feedback.METRICS_LOOKBACK_DAYS)
    w.check()
    assert w.open_keys() == {K_MET}
    w.now = NOW + timedelta(days=2)
    w.decisions(age_h=1)
    w.check()
    assert w.open_keys() == set()


def test_unsettled_old_and_idless_shorts_are_not_flagged(tmp_path):
    w = _fresh_world(tmp_path)
    w.short(1, views=None, age_days=3)  # window not settled yet
    w.short(2, views=None, age_days=seo_feedback.METRICS_LOOKBACK_DAYS + 1)  # beyond lookback
    w.short(3, views=None, ytid="")  # never on YouTube
    w.check()
    assert w.open_keys() == set()


def test_unreadable_tracking_record_keeps_metrics_rows_live(tmp_path):
    w = _fresh_world(tmp_path)
    w.short(1, views=None)
    w.check()
    assert w.open_keys() == {K_MET}
    for f in w.tracking.rglob("*.json"):
        f.write_text("{broken")
    res = w.check()
    assert w.open_keys() == {K_MET}
    assert any("metrics" in n and "not evaluated" in n for n in res["notes"])


def test_an_open_row_refreshes_its_detail_and_count_without_a_second_push(tmp_path):
    w = _fresh_world(tmp_path)
    w.short(1, views=None)
    w.check()
    assert w.rows()[K_MET]["count"] == 1
    w.short(2, views=None)
    w.check()
    row = w.rows()[K_MET]
    assert row["count"] == 2 and len(row["shorts"]) == 2 and "p#2" in row["detail"]
    assert row["pushed"] is True
    assert w.pushed.count(("health_seo_feedback_stale", K_MET)) == 1


def test_a_fixed_short_resolution_is_not_blamed_on_an_unrelated_old_short(tmp_path):
    w = _fresh_world(tmp_path)
    w.short(5, views=None)
    w.short(9, pct=None, age_days=90)  # months old, never filled: not in the row
    w.check()
    assert w.rows()[K_MET]["shorts"] == [{"project": "p", "short_index": 5, "missing": ["views"]}]
    w.short(5, views=5, pct=40.0)  # p#5 got real metrics
    res = w.check()
    assert w.open_keys() == set()
    assert not any("aged out" in n for n in res["notes"])


def test_a_short_aging_out_resolves_the_row_and_says_so_in_a_note(tmp_path):
    w = _fresh_world(tmp_path)
    w.short(1, views=None, age_days=seo_feedback.METRICS_LOOKBACK_DAYS)
    w.check()
    w.now = NOW + timedelta(days=2)
    w.decisions(age_h=1)
    res = w.check()
    assert w.open_keys() == set()
    assert any("aged out" in n and "metrics" in n for n in res["notes"])
    assert not any("aged out" in n for n in w.check()["notes"])  # said once, at the resolution


def test_a_listed_short_aging_out_while_the_row_stays_open_is_noted_once(tmp_path):
    w = _fresh_world(tmp_path)
    w.short(1, views=None, age_days=27)
    w.short(2, views=None, age_days=20)
    w.check()
    assert [e["short_index"] for e in w.rows()[K_MET]["shorts"]] == [1, 2]
    w.now = NOW + timedelta(days=2)  # p#1 is now past the lookback and still empty
    w.decisions(age_h=1)
    res = w.check()
    assert [e["short_index"] for e in w.rows()[K_MET]["shorts"]] == [2]  # refreshed, still open
    assert w.open_keys() == {K_MET}
    assert [n for n in res["notes"] if "aged out" in n and "not a fix" in n]
    assert not any("resolved" in n for n in res["notes"] if "aged out" in n)
    w.short(2, views=5, pct=40.0)
    res = w.check()
    assert w.open_keys() == set()
    assert not any("aged out" in n for n in res["notes"])  # said once, on the tick it happened
    assert w.pushed.count(("health_seo_feedback_stale", K_MET)) == 1


# --- (c) the nightly Short text sync has stopped running ------------------------------


def _sync_world(tmp_path) -> World:
    w = _fresh_world(tmp_path)
    w.auto_stamp = False
    return w


def test_a_missing_stamp_opens_the_sync_row(tmp_path):
    w = _sync_world(tmp_path)
    res = w.check()
    assert K_SYNC in res["new"] and w.open_keys() == {K_SYNC}
    assert "missing" in w.rows()[K_SYNC]["detail"]
    assert ("health_seo_feedback_stale", K_SYNC) in w.pushed


def test_a_stamp_over_the_limit_opens_the_row_and_names_the_age(tmp_path):
    w = _sync_world(tmp_path)
    w.stamp(age_h=37)
    w.check()
    assert w.open_keys() == {K_SYNC}
    assert "37" in w.rows()[K_SYNC]["detail"] and "stale" in w.rows()[K_SYNC]["detail"]


@pytest.mark.parametrize("age_h", [35, 36])
def test_a_stamp_inside_or_exactly_at_the_limit_does_not_open_the_row(tmp_path, age_h):
    assert seo_feedback.SYNC_STALE_AFTER_H == 36
    w = _sync_world(tmp_path)
    w.stamp(age_h=age_h)
    w.check()
    assert w.open_keys() == set()


def test_a_ran_at_more_than_an_hour_in_the_future_is_unreadable(tmp_path):
    w = _sync_world(tmp_path)
    w.stamp(age_h=-2)  # clock skew must not hide a dead job
    w.check()
    assert w.open_keys() == {K_SYNC} and "unreadable" in w.rows()[K_SYNC]["detail"]
    w.stamp(age_h=-0.5)  # within the skew allowance: fine
    w.check()
    assert w.open_keys() == set()


def test_a_naive_ran_at_is_read_as_utc(tmp_path):
    w = _sync_world(tmp_path)
    naive = (NOW - timedelta(hours=1)).replace(tzinfo=None).isoformat()
    w.stamp_path().write_text(json.dumps({"schema": 1, "ran_at": naive}))
    w.check()
    assert w.open_keys() == set()
    old = (NOW - timedelta(hours=40)).replace(tzinfo=None).isoformat()
    w.stamp_path().write_text(json.dumps({"schema": 1, "ran_at": old}))
    w.check()
    assert w.open_keys() == {K_SYNC} and "40" in w.rows()[K_SYNC]["detail"]


def test_a_crashed_run_stamp_does_not_open_the_sync_row(tmp_path):
    w = _sync_world(tmp_path)
    w.stamp_path().write_text(
        json.dumps({"schema": 1, "ran_at": (NOW - timedelta(hours=1)).isoformat(), "crashed": True})
    )
    w.check()
    assert w.open_keys() == set()  # short_text_sync_crash covers a crash, alone


@pytest.mark.parametrize("body", ["{not json", "[]", '{"ran_at": "nope"}', ""])
def test_an_unreadable_stamp_is_a_finding(tmp_path, body):
    w = _sync_world(tmp_path)
    w.stamp_path().write_text(body)
    w.check()
    assert w.open_keys() == {K_SYNC} and "unreadable" in w.rows()[K_SYNC]["detail"]


def test_a_fresh_stamp_resolves_the_row(tmp_path):
    w = _sync_world(tmp_path)
    w.check()
    assert w.open_keys() == {K_SYNC}
    w.stamp(age_h=1)
    res = w.check()
    assert K_SYNC in res["resolved"] and w.open_keys() == set()


def test_an_open_sync_row_refreshes_its_detail_without_a_second_push(tmp_path):
    w = _sync_world(tmp_path)
    w.stamp(age_h=37)
    w.check()
    first = w.rows()[K_SYNC]["detail"]
    w.now = NOW + timedelta(hours=10)
    w.decisions(age_h=1)
    w.check()
    row = w.rows()[K_SYNC]
    assert row["detail"] != first and "47" in row["detail"]
    assert row["pushed"] is True
    assert w.pushed.count(("health_seo_feedback_stale", K_SYNC)) == 1


def test_one_push_across_open_refresh_and_resolve(tmp_path):
    w = _sync_world(tmp_path)
    w.check()  # open: pushed
    w.check()  # still open: refresh only
    w.stamp(age_h=1)
    w.check()  # resolved
    assert w.open_keys() == set()
    assert w.pushed.count(("health_seo_feedback_stale", K_SYNC)) == 1
