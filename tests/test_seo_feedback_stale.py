"""Rule ``seo_feedback_stale``: stale arm decisions, a settled Short with no core metrics, and a
queued YouTube Short outside the sync lead time that C2's repeated-place check or a demoted
arm still marks. Detection only; every condition resolves when it clears."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.health import run_check, seo_feedback
from gomoveshift.health.check import _line
from gomoveshift.health.store import load_all
from gomoveshift.publishing.experiments import ARM_DECISIONS_FILENAME
from gomoveshift.publishing.queue import open_queue_db
from gomoveshift.publishing.queue_short_text import SYNC_LEAD_TIME
from gomoveshift.tracking.day7 import window_settled
from gomoveshift.tracking.schema import ContentMeta, PlatformMetrics, ShortRecord, TrackingRecord
from gomoveshift.tracking.store import TrackingStore

NOW = datetime(2026, 3, 1, 12, 0, tzinfo=timezone.utc)
K_DEC = "seo_feedback_stale:decisions"
K_MET = "seo_feedback_stale:metrics:p:1"
K_QUE = "seo_feedback_stale:queued:p:2"


class World:
    """tmp_path fakes: a seo storage dir, a tracking root and an incidents dir."""

    def __init__(self, tmp_path: Path):
        self.tmp = tmp_path
        self.storage = tmp_path / "seo"
        self.storage.mkdir()
        self.tracking = tmp_path / "track"
        self.tracking.mkdir()
        self.pushed: list = []

    def decisions(self, *, age_h=1.0, decisions=()):
        at = NOW - timedelta(hours=age_h)
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

    def queued(
        self,
        index,
        *,
        title_place,
        arms=None,
        hours=72,
        regions=("Cork, Ireland", "Kerry, Ireland"),
    ):
        """A queued YouTube row for short ``index`` of LF 1 in project ``p`` (its sidecar too)."""
        plan = self.tmp / "plans"
        plan.mkdir(exist_ok=True)
        (plan / "p_lf01_youtube.json").write_text(json.dumps({"location": "Kinsale, Cork"}))
        side = plan / "p_publish_plan.json"
        raw = json.loads(side.read_text()) if side.exists() else {"shorts": []}
        raw["shorts"] = [s for s in raw["shorts"] if s["short_index"] != index] + [
            {
                "project_name": "p",
                "short_index": index,
                "lf_index": 1,
                "sampled_regions": list(regions),
                "experiment_arms": arms or {},
                "payloads": {
                    "youtube": {
                        "title": f"Sep 14, {index}th stretch: {title_place}",
                        "description": "d",
                    }
                },
            }
        ]
        side.write_text(json.dumps(raw))
        conn = open_queue_db(self.storage)
        with conn:
            conn.execute("DELETE FROM deliveries WHERE short_index = ?", (index,))
            conn.execute(
                "INSERT INTO deliveries (project_name, short_index, platform, plan_hash, status,"
                " scheduled_at, sidecar_path, shadow, created_at, updated_at) VALUES"
                " ('p', ?, 'youtube', 'h', 'queued', ?, ?, 0, 'x', ?)",
                (
                    index,
                    (NOW + timedelta(hours=hours)).isoformat(),
                    str(side),
                    NOW.isoformat(),
                ),
            )
        conn.close()

    def check(self):
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
            now=NOW,
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
    w.short(1, views=5, pct=None)  # views back, retention still null: still open
    w.check()
    assert w.open_keys() == {K_MET}
    w.short(1, views=5, pct=40.0)
    res = w.check()
    assert res["resolved"] == [K_MET] and w.open_keys() == set()


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


# --- (c) unsynced queued row ------------------------------------------------------


def test_repeated_place_text_outside_lead_time_opens_and_resolves_when_rewritten(tmp_path):
    w = _fresh_world(tmp_path)
    w.queued(1, title_place="Cork")
    w.queued(2, title_place="Cork")  # repeats short 1's place while Kerry was measured
    res = w.check()
    assert res["new"] == [K_QUE] and w.open_keys() == {K_QUE}
    assert "repeated_place_text" in {c["code"] for c in w.rows()[K_QUE]["causes"]}
    w.queued(2, title_place="Kerry")  # what sync-short-text would write
    res = w.check()
    assert res["resolved"] == [K_QUE] and w.open_keys() == set()


def test_repeat_with_no_other_measured_place_is_not_flagged(tmp_path):
    w = _fresh_world(tmp_path)
    w.queued(1, title_place="Cork", regions=("Cork, Ireland",))
    w.queued(2, title_place="Cork", regions=("Cork, Ireland",))
    w.check()
    assert w.open_keys() == set()  # sync could not vary it either


def test_row_inside_the_lead_time_is_left_alone(tmp_path):
    w = _fresh_world(tmp_path)
    w.queued(1, title_place="Cork")
    w.queued(2, title_place="Cork", hours=SYNC_LEAD_TIME.total_seconds() / 3600 - 1)
    w.check()
    assert w.open_keys() == set()


def test_demoted_arm_in_the_persisted_record_opens_and_resolves_when_cleared(tmp_path):
    w = World(tmp_path)
    demoted = {"factor": "yt_title_body", "arm": "A", "decision": "demote"}
    w.decisions(decisions=[demoted])
    w.queued(2, title_place="Kerry", arms={"yt_title_body": "A", "yt_hashtag_set": "pov"})
    w.check()
    assert w.open_keys() == {K_QUE}
    assert "demoted_arm" in {c["code"] for c in w.rows()[K_QUE]["causes"]}
    w.decisions(decisions=[{**demoted, "decision": "hold"}])
    w.check()
    assert w.open_keys() == set()


def test_unreadable_queue_sidecar_keeps_queued_row_live(tmp_path):
    w = _fresh_world(tmp_path)
    w.queued(1, title_place="Cork")
    w.queued(2, title_place="Cork")
    w.check()
    assert w.open_keys() == {K_QUE}
    (w.tmp / "plans" / "p_publish_plan.json").write_text("{broken")
    res = w.check()
    assert w.open_keys() == {K_QUE}
    assert any("queued" in n and "not evaluated" in n for n in res["notes"])


def test_missing_queue_db_is_a_note_and_keeps_queued_rows_live(tmp_path):
    w = _fresh_world(tmp_path)
    w.queued(1, title_place="Cork")
    w.queued(2, title_place="Cork")
    w.check()
    (w.storage / "social_queue.sqlite3").unlink()
    res = w.check()
    assert w.open_keys() == {K_QUE}
    assert any("social_queue.sqlite3" in n for n in res["notes"])


def test_undecidable_demotion_holds_queued_rows_but_decisions_stays_the_finding(tmp_path):
    w = World(tmp_path)  # no decisions file
    w.queued(1, title_place="Cork")
    w.queued(2, title_place="Cork")
    w.check()
    assert w.open_keys() == {K_DEC, K_QUE}
    w.queued(2, title_place="Kerry")  # C2 half clears, but the demoted-arm half cannot be judged
    w.check()
    assert w.open_keys() == {K_DEC, K_QUE}
