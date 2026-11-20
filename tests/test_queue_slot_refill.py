"""A queued Short the review still refuses hands its slot to the next ready Short.

``refill_slots`` (called by ``dispatch-queue`` after the Q2 repair) swaps a refused,
queued YouTube row's slot (inside the 48 h lead) with the earliest later queued row that
passes; with none left it moves the refused row after the last queued slot on the daily
grid. It also re-reviews ``failed_permanent`` review refusals and requeues one that now
passes. Synthetic data, ``tmp_path`` only; the slot grid and sidecar writer are injected.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from gomoveshift.health import check as check_mod
from gomoveshift.publishing.queue import enqueue, open_queue_db
from gomoveshift.publishing.queue_slot_refill import (
    JP_KEPT_FAILED_PERMANENT_IDS,
    refill_slots,
)

NOW = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)
POOL = [f"{h:02d}:00" for h in range(13, 21)]  # hourly 13:00-20:00 UTC
FINAL_TEXT = "FinalTextReviewRefused"


def _at(hour, day=2):
    return datetime(2026, 8, day, hour, 0, tzinfo=timezone.utc)


def _grid(_row):
    return POOL, ZoneInfo("UTC")


class Env:
    def __init__(self, conn):
        self.conn = conn
        self.writes = []
        self._n = 0

    def add(self, slot, status="queued", error_class=None, shadow=0, row_id=None):
        self._n += 1
        rid = enqueue(
            self.conn,
            f"proj{self._n}",
            1,
            "youtube",
            "H",
            sidecar_path=f"/nowhere/{self._n}.json",
            scheduled_at=slot.isoformat(),
        )
        self.conn.execute(
            "UPDATE deliveries SET status=?, error_class=?, shadow=?, updated_at=? WHERE id=?",
            (status, error_class, shadow, NOW.isoformat(), rid),
        )
        if row_id is not None:
            self.conn.execute("UPDATE deliveries SET id=? WHERE id=?", (row_id, rid))
            rid = row_id
        self.conn.commit()
        return rid

    def slot(self, rid):
        row = self.conn.execute("SELECT scheduled_at FROM deliveries WHERE id=?", (rid,)).fetchone()
        return datetime.fromisoformat(row["scheduled_at"])

    def run(self, refused=(), *, apply=True, review=lambda row: None, now=NOW):
        rows = self.conn.execute(
            "SELECT id FROM deliveries WHERE status='queued' AND platform='youtube' AND shadow=0"
        ).fetchall()
        refusals = {r["id"]: ("tag_unreviewable: x" if r["id"] in refused else None) for r in rows}
        return refill_slots(
            self.conn,
            apply=apply,
            now=now,
            refusals=refusals,
            grid_for=_grid,
            review=review,
            write_slot=lambda row, slot: self.writes.append((row["id"], slot)),
        )

    def log(self, action="queue_slot"):
        return self.conn.execute("SELECT * FROM response_log WHERE action=?", (action,)).fetchall()

    def held_hours(self):
        rows = self.conn.execute(
            "SELECT scheduled_at FROM deliveries WHERE platform='youtube' "
            "AND status NOT IN ('failed_permanent','cancelled','missing_remote')"
        ).fetchall()
        return [datetime.fromisoformat(r["scheduled_at"]) for r in rows]


@pytest.fixture
def env(tmp_path):
    conn = open_queue_db(tmp_path / "q")
    yield Env(conn)
    conn.close()


def test_refused_row_swaps_with_the_next_passing_row(env):
    r = env.add(_at(13))
    p = env.add(_at(15))
    acts = env.run({r})
    assert env.slot(r) == _at(15) and env.slot(p) == _at(13)
    lines = [a.line() for a in acts]
    assert any(
        ln.startswith(f"QUEUE_SLOT {r} {_at(13).isoformat()} -> {_at(15).isoformat()} ")
        for ln in lines
    )
    assert any(
        ln.startswith(f"QUEUE_SLOT {p} {_at(15).isoformat()} -> {_at(13).isoformat()} ")
        for ln in lines
    )
    assert len(env.log()) == 2  # one response_log row per move
    assert {(i, s) for i, s in env.writes} == {(r, _at(15)), (p, _at(13))}


def test_only_scheduled_at_changes(env):
    r = env.add(_at(13))
    env.add(_at(15))
    before = dict(env.conn.execute("SELECT * FROM deliveries WHERE id=?", (r,)).fetchone())
    env.run({r})
    after = dict(env.conn.execute("SELECT * FROM deliveries WHERE id=?", (r,)).fetchone())
    changed = {k for k in before if before[k] != after[k]}
    assert changed <= {"scheduled_at", "updated_at"} and "scheduled_at" in changed
    assert after["status"] == "queued" and after["attempt_count"] == 0


def test_a_run_of_three_refused_rows_drifts_past_two_passing_rows(env):
    r1, r2, r3 = env.add(_at(13)), env.add(_at(14)), env.add(_at(15))
    p1, p2 = env.add(_at(16)), env.add(_at(17))
    env.run({r1, r2, r3})
    assert (env.slot(p1), env.slot(p2)) == (_at(13), _at(14))  # the ready rows moved up
    assert env.slot(r1) == _at(16) and env.slot(r2) == _at(17)
    assert env.slot(r3) == _at(18)  # no passing row left: after the last queued slot
    assert len(set(env.held_hours())) == 5


def test_no_passing_row_left_moves_after_the_last_queued_slot(env):
    p_earlier = env.add(_at(13))
    r = env.add(_at(14))
    later = env.add(_at(16))  # a later refused row still holds the last slot
    acts = env.run({r, later})
    assert env.slot(p_earlier) == _at(13)  # an EARLIER passing row is never a swap partner
    assert env.slot(r) == _at(17)
    assert any(f"QUEUE_SLOT {r} " in a.line() for a in acts)


def test_a_moved_row_never_lands_in_an_hour_that_already_has_a_short(env):
    r = env.add(_at(13))
    env.add(_at(14), status="published")  # a scheduled-private upload keeps its hour
    env.run({r})
    assert env.slot(r) == _at(15)
    hours = env.held_hours()
    assert len({h.replace(minute=0) for h in hours}) == len(hours)


def test_a_refused_row_outside_the_lead_is_left_alone(env):
    r = env.add(NOW + timedelta(hours=60))
    env.add(NOW + timedelta(hours=70))
    assert env.run({r}) == []
    assert env.writes == [] and env.log() == []


def test_a_slot_too_near_to_publish_into_is_not_swapped(env):
    r = env.add(NOW + timedelta(minutes=30))
    p = env.add(_at(15))
    env.run({r})
    assert env.slot(p) == _at(15)  # a past publishAt is never handed to a passing row
    assert env.slot(r) > _at(15)


def test_dry_run_reports_and_writes_nothing(env):
    r = env.add(_at(13))
    p = env.add(_at(15))
    failed = env.add(_at(10, day=1), status="failed_permanent", error_class=FINAL_TEXT)
    before = env.conn.execute("SELECT id, scheduled_at, status FROM deliveries").fetchall()
    acts = env.run({r}, apply=False)
    assert any(a.kind == "slot" and a.delivery_id == r for a in acts)
    assert any(a.kind == "recovered" and a.delivery_id == failed for a in acts)
    assert env.conn.execute("SELECT id, scheduled_at, status FROM deliveries").fetchall() == before
    assert env.writes == [] and env.log() == []
    assert env.conn.execute("SELECT count(*) FROM response_log").fetchone()[0] == 0
    assert env.slot(p) == _at(15)


def test_a_failed_permanent_row_that_now_passes_is_requeued_in_the_next_free_slot(env):
    failed = env.add(_at(10, day=1), status="failed_permanent", error_class=FINAL_TEXT)
    env.add(_at(14, day=1))  # a queued row holds 14:00 today, the first slot past the 1 h lead
    acts = env.run()
    [new] = env.conn.execute(
        "SELECT * FROM deliveries WHERE recovers_delivery_id=?", (failed,)
    ).fetchall()
    assert new["status"] == "queued" and new["scheduled_at"] == _at(15, day=1).isoformat()
    assert (
        env.conn.execute("SELECT status FROM deliveries WHERE id=?", (failed,)).fetchone()[0]
        == "failed_permanent"
    )
    assert any(a.kind == "recovered" and a.delivery_id == failed for a in acts)
    assert (failed, _at(15, day=1)) in env.writes  # the shared sidecar carries the publishAt


def test_a_failed_row_that_still_cannot_pass_is_reported_and_left(env):
    failed = env.add(_at(10, day=1), status="failed_permanent", error_class=FINAL_TEXT)
    acts = env.run(review=lambda row: "tag_unreviewable: Ireland road trip")
    [a] = [a for a in acts if a.delivery_id == failed]
    assert a.kind == "unrecoverable" and "tag_unreviewable" in a.line()
    assert env.conn.execute("SELECT count(*) FROM deliveries").fetchone()[0] == 1


def test_a_failed_row_that_is_not_a_review_refusal_is_not_recovered(env):
    env.add(_at(10, day=1), status="failed_permanent", error_class="HttpError")
    assert env.run() == []


def test_jps_beaufort_rows_stay_failed_permanent(env):
    assert JP_KEPT_FAILED_PERMANENT_IDS == frozenset({98, 102, 106, 110, 114})
    for i in sorted(JP_KEPT_FAILED_PERMANENT_IDS):
        env.add(_at(10, day=1), status="failed_permanent", error_class=FINAL_TEXT, row_id=i)
    assert env.run() == []  # the review would pass them; the exclusion is not data-dependent
    assert env.conn.execute("SELECT count(*) FROM deliveries").fetchone()[0] == 5
    assert env.writes == []


def test_a_published_row_is_never_touched(env):
    pub = env.add(_at(13), status="published")
    env.add(_at(15))
    acts = env.run({pub})  # even a published id the refusal map names
    assert acts == [] and env.slot(pub) == _at(13) and env.writes == []


def test_shadow_rows_are_never_moved(env):
    r = env.add(_at(13), shadow=1)
    assert env.run({r}) == []


# --- health --------------------------------------------------------------------------


def _incidents(env, now=NOW):
    return check_mod.failed_permanent_incidents(env.conn_path, now, now.isoformat(), [])


@pytest.fixture
def hp(env, tmp_path):
    env.conn_path = tmp_path / "q" / "social_queue.sqlite3"
    return env


def test_a_new_failed_permanent_youtube_row_raises_an_incident(hp):
    rid = hp.add(_at(10, day=1), status="failed_permanent", error_class=FINAL_TEXT)
    [row] = _incidents(hp)
    assert row["kind"] == "delivery_failed_permanent"
    assert row["key"] == f"delivery_failed_permanent:{rid}" and row["delivery_id"] == rid
    assert "failed permanently" in check_mod._line(row)
    assert check_mod._EVENTS["delivery_failed_permanent"] == "health_delivery_failed_permanent"


def test_the_incident_skips_jps_rows_old_rows_recovered_rows_and_other_platforms(hp):
    for i in JP_KEPT_FAILED_PERMANENT_IDS:
        hp.add(_at(10, day=1), status="failed_permanent", row_id=i)
    old = hp.add(_at(10, day=1), status="failed_permanent")
    hp.conn.execute(
        "UPDATE deliveries SET updated_at=? WHERE id=?",
        ((NOW - timedelta(days=30)).isoformat(), old),
    )
    done = hp.add(_at(10, day=1), status="failed_permanent")
    hp.conn.execute(
        "UPDATE deliveries SET recovers_delivery_id=? WHERE id=?",
        (done, hp.add(_at(15))),
    )
    other = hp.add(_at(10, day=1), status="failed_permanent")
    hp.conn.execute("UPDATE deliveries SET platform='instagram' WHERE id=?", (other,))
    hp.conn.commit()
    assert _incidents(hp) == []


def test_the_incident_check_never_writes(hp):
    hp.add(_at(10, day=1), status="failed_permanent")
    before = hp.conn_path.read_bytes()
    _incidents(hp)
    assert hp.conn_path.read_bytes() == before
    json.dumps(_incidents(hp))
