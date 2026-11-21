"""``queue sync-short-text``: refusal classes, unusable sidecars, failed writes, the location cap
(unit C5b, review fixes)."""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from types import SimpleNamespace

import pytest
from test_queue_sync_short_text import (
    NOW,
    PROJECT,
    PROJECT2,
    REGION,
    _other,
    _short,
    _two,
    _world,
)

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.publishing import anti_spam_gate
from gomoveshift.publishing import queue_short_text as qst
from gomoveshift.publishing.queue import enqueue
from gomoveshift.publishing.queue_short_text import sync_short_text


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _world2(tmp_path, day=6):
    plan, conn = _world(tmp_path, [_short(1)])
    return plan, _other(conn, tmp_path, day=day), conn


def _synced_text(scratch, project):
    """The title and description a sync writes for ``project`` in a scratch copy of the world."""
    _, other, probe = _world2(scratch)
    sync_short_text(probe, apply=True, now=NOW)
    path = other if project == PROJECT2 else scratch / f"{PROJECT}_publish_plan.json"
    youtube = json.loads(path.read_text())["shorts"][0]["payloads"]["youtube"]
    return {k: youtube[k] for k in ("title", "description")}


# -- M1: transient is named, everything else is actionable ---------------------------------


@pytest.mark.parametrize(
    "reason",
    [
        "generator: boom",
        "LF sidecar p_lf01_youtube.json unreadable: [Errno 2]",
        "LF sidecar p_lf01_youtube.json has no location",
        "stored arm(s) {'title': 'a'} can no longer be rendered by the generator",
        "no such Short in the plan",
        "youtube row 3 already holds the new plan hash",
        "youtube row 3 is a recovery row",
        "youtube row 3 is uploading",
        "youtube row 3 has no usable scheduled_at",
        "no clip place to compose a title from",
        "old title has no verifiable 'Nth stretch' stamp",
        "its title place repeats the previous Short's",
        "still refused by the Short review: x",
        "payload limit: x",
        "plan gate: x",
        "anti-spam gate: shared-prefix: x",
        "sidecar unusable: write failed: disk full",
        "a queue row's plan hash abcdef is unknown",
        "something nobody has classified yet",  # fails closed
    ],
)
def test_every_refusal_is_actionable_unless_named_transient(reason):
    change = qst.ShortTextChange(1, "p", 1, "refused", "a", "b", reason)
    assert qst.is_actionable_refusal(change) is True


TRANSIENT_TEXT = [
    "facebook row 3 holds a live lease",
    "youtube row 3 slot 2026-11-21T00:00:00+00:00 is inside the lead time",
    "an earlier sidecar's write failed; retried next run",
    "sidecar unusable: write failed: a row was leased during the write",
    "sidecar unusable: write failed: p.json changed while the sync ran",
]


@pytest.mark.parametrize("reason", TRANSIENT_TEXT)
def test_the_flagged_transient_refusals_clear_on_their_own(reason):
    change = qst.ShortTextChange(1, "p", 1, "refused", "a", "b", reason, transient=True)
    assert qst.is_actionable_refusal(change) is False
    assert not qst.is_actionable_refusal(qst.ShortTextChange(1, "p", 1, "changed", "a", "b", "x"))


@pytest.mark.parametrize("reason", TRANSIENT_TEXT)
def test_transience_is_the_flag_never_the_words_of_the_reason(reason):
    """A reason quoting a transient phrase (a Short's text can) is still actionable unflagged,
    and a flagged one is transient whatever it says."""
    assert qst.is_actionable_refusal(qst.ShortTextChange(1, "p", 1, "refused", "a", "b", reason))
    odd = qst.ShortTextChange(1, "p", 1, "refused", "a", "b", "anything at all", transient=True)
    assert not qst.is_actionable_refusal(odd)


def test_the_real_transient_refusals_carry_the_flag(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook"))
    conn.execute(
        "UPDATE deliveries SET lease_owner = 'd', lease_expires_at = ? WHERE platform = 'facebook'",
        ((NOW + timedelta(hours=1)).isoformat(),),
    )
    conn.commit()
    (change,) = sync_short_text(conn, apply=False, now=NOW)
    assert "holds a live lease" in change.reason
    assert change.transient and not qst.is_actionable_refusal(change)


def _facebook_sibling(tmp_path, **set_):
    """A world whose Short also has a facebook row, which ``set_`` (column -> value) edits."""
    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "facebook"))
    row = conn.execute("SELECT * FROM deliveries WHERE platform = 'facebook'").fetchone()
    now = {c: set_.get(c, row[c]) for c in ("scheduled_at", "recovers_delivery_id", "status")}
    conn.execute(
        "UPDATE deliveries SET scheduled_at = ?, recovers_delivery_id = ?, status = ? "
        "WHERE platform = 'facebook'",
        (now["scheduled_at"], now["recovers_delivery_id"], now["status"]),
    )
    conn.commit()
    return sync_short_text(conn, apply=False, now=NOW)[0]


@pytest.mark.parametrize(
    ("set_", "words", "transient"),
    [
        ({"scheduled_at": (NOW + timedelta(hours=1)).isoformat()}, "inside the lead time", True),
        (
            {"scheduled_at": (NOW - timedelta(hours=1)).isoformat()},
            "is past but still queued",
            False,
        ),
        ({"scheduled_at": NOW.isoformat()}, "is past but still queued", False),
        ({"recovers_delivery_id": 1}, "is a recovery row", False),
        ({"status": "uploading"}, "is uploading", False),
        ({"scheduled_at": None}, "has no usable scheduled_at", False),
        ({"scheduled_at": "not a time"}, "has no usable scheduled_at", False),
    ],
)
def test_each_sibling_reason_is_transient_only_when_it_clears_by_itself(
    tmp_path, set_, words, transient
):
    change = _facebook_sibling(tmp_path, **set_)
    assert change.status == "refused" and words in change.reason
    assert change.transient is transient and qst.is_actionable_refusal(change) is not transient


def test_the_cascade_after_a_failed_write_is_flagged_transient(tmp_path, monkeypatch):
    plan, other, conn = _world2(tmp_path)
    _fail_write_of(monkeypatch, plan, OSError("disk full"))
    by = {c.project_name: c for c in sync_short_text(conn, apply=True, now=NOW)}
    assert "retried next run" in by[PROJECT2].reason and by[PROJECT2].transient
    assert qst.is_actionable_refusal(by[PROJECT])


def test_a_generator_that_always_raises_is_an_actionable_refusal(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(qst, "_generate_lf", boom)
    _, conn = _two(tmp_path)
    changes = sync_short_text(conn, apply=False, now=NOW)
    assert {c.status for c in changes} == {"refused"}
    assert all(c.reason.startswith("generator: boom") for c in changes)
    assert all(qst.is_actionable_refusal(c) for c in changes)


def test_a_stale_row_whose_synced_hash_is_already_taken_is_a_no_op_not_a_refusal(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)])
    sync_short_text(conn, apply=True, now=NOW)  # the sidecar now holds the synced text
    old = json.loads(conn.execute("SELECT reason FROM response_log").fetchone()[0])
    new_hash = old["new_plan_hash"]
    row = dict(conn.execute("SELECT * FROM deliveries").fetchone())
    conn.execute("UPDATE deliveries SET plan_hash = ?", (old["old_plan_hash"],))
    row.update(id=None, plan_hash=new_hash, status="failed_permanent")
    conn.execute(
        f"INSERT INTO deliveries ({', '.join(row)}) VALUES ({', '.join('?' for _ in row)})",  # noqa: S608
        tuple(row.values()),
    )
    conn.commit()
    (change,) = sync_short_text(conn, apply=False, now=NOW)
    assert change.status == "unchanged" and "already holds the synced plan hash" in change.reason
    assert not qst.is_actionable_refusal(change)


def test_a_place_the_generator_could_not_vary_is_not_a_refusal(tmp_path):
    shorts = [_short(1), _short(2)]  # both stored texts name Ashby County
    sidecar = tmp_path / f"{PROJECT}_publish_plan.json"
    (tmp_path / f"{PROJECT}_lf01_youtube.json").write_text(
        json.dumps({"location": "Ashby County, Virginia"})
    )
    clash = qst._successor_clashes(sidecar, shorts, {1: shorts[1]})
    assert clash and "repeats" in clash[1]
    kept = {
        1: SimpleNamespace(
            place_variety={k: {"reason": "no_alternative"} for k in ("title", "description")}
        )
    }
    assert qst._successor_clashes(sidecar, shorts, {1: shorts[1]}, kept) == {}


# -- M2: an unusable sidecar's stored text still reaches the other sidecars' gates -----------


@pytest.mark.parametrize("a_first", [True, False])
def test_an_unusable_sidecars_stored_text_is_seen_by_the_gates(tmp_path, a_first):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    new_b = _synced_text(scratch, PROJECT2)
    plan, other, conn = _world2(tmp_path)
    raw = json.loads(plan.read_text())
    raw["shorts"][0]["payloads"]["youtube"].update(new_b)  # A's STORED text is B's new text
    plan.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n")
    (tmp_path / "social").mkdir()
    (tmp_path / "social" / "titles.yaml").write_text("- not a mapping\n")  # A is unusable
    if not a_first:
        conn.execute(
            "UPDATE deliveries SET scheduled_at = ? WHERE project_name = ?",
            ("2026-11-25T12:00:00+00:00", PROJECT),
        )
        conn.commit()
    changes = sync_short_text(conn, apply=False, now=NOW)
    a = next(c for c in changes if c.project_name == PROJECT)
    b = next(c for c in changes if c.project_name == PROJECT2)
    assert a.status == "refused" and a.reason.startswith("sidecar unusable")
    assert b.status == "refused" and "anti-spam gate" in b.reason


# -- m1: the lookback uses the injected now ----------------------------------------------


def test_the_recent_text_lookback_uses_the_injected_now(tmp_path, monkeypatch):
    seen = []
    real = anti_spam_gate.recent_text_entries
    monkeypatch.setattr(
        anti_spam_gate, "recent_text_entries", lambda c, **kw: seen.append(kw) or real(c, **kw)
    )
    _, conn = _two(tmp_path)
    sync_short_text(conn, apply=False, now=NOW)
    assert seen == [{"now": NOW}]


# -- m2: a failed write ---------------------------------------------------------------------


def _fail_write_of(monkeypatch, sidecar, exc):
    real = qst._apply_sidecar

    def apply(conn_, path, *a):
        if path == sidecar:
            raise exc
        return real(conn_, path, *a)

    monkeypatch.setattr(qst, "_apply_sidecar", apply)


def test_a_failed_write_writes_exactly_the_sidecars_before_it(tmp_path, monkeypatch):
    plan, other, conn = _world2(tmp_path)
    _fail_write_of(monkeypatch, other, OSError("disk full"))
    changes = sync_short_text(conn, apply=True, now=NOW)
    by = {c.project_name: c for c in changes}
    assert (by[PROJECT].status, by[PROJECT].written) == ("changed", True)
    assert (by[PROJECT2].status, by[PROJECT2].written) == ("refused", False)
    assert "write failed" in by[PROJECT2].reason
    assert json.loads(other.read_text())["shorts"][0]["payloads"]["youtube"]["title"].endswith(
        "zebra words #Shorts"
    )


def test_any_exception_in_a_write_is_a_refusal_not_a_crash_after_partial_writes(
    tmp_path, monkeypatch
):
    plan, other, conn = _world2(tmp_path)
    _fail_write_of(monkeypatch, plan, RuntimeError("weird"))
    changes = sync_short_text(conn, apply=True, now=NOW)
    by = {c.project_name: c for c in changes}
    assert by[PROJECT].status == "refused" and "RuntimeError: weird" in by[PROJECT].reason
    assert qst.is_actionable_refusal(by[PROJECT])
    assert "retried next run" in by[PROJECT2].reason
    assert not any(c.written for c in changes)


def _break_write(monkeypatch, conn, sidecar, how):
    """Make the real ``_apply_sidecar`` of ``sidecar`` fail the way ``how`` names."""
    real = qst._apply_sidecar

    def apply(conn_, path, *a):
        if path == sidecar:
            if how == "oserror":
                raise OSError("disk full")
            if how == "locked":
                raise sqlite3.OperationalError("database is locked")
            if how in ("busy_code", "locked_code", "io_code"):
                exc = sqlite3.OperationalError("x")
                exc.sqlite_errorcode = {
                    "busy_code": sqlite3.SQLITE_BUSY,
                    "locked_code": sqlite3.SQLITE_LOCKED,
                    "io_code": sqlite3.SQLITE_IOERR_WRITE,
                }[how]
                raise exc
            if how == "disk_io":
                raise sqlite3.OperationalError("disk I/O error")
            if how == "changed":  # an editor saves it between planning and writing
                path.write_text(path.read_text() + "\n")
            if how == "leased":  # the dispatcher leases the row between planning and writing
                conn_.execute(
                    "UPDATE deliveries SET lease_owner = 'd', lease_expires_at = ? "
                    "WHERE project_name = ?",
                    ((NOW + timedelta(hours=1)).isoformat(), PROJECT2),
                )
                conn_.commit()
        return real(conn_, path, *a)

    monkeypatch.setattr(qst, "_apply_sidecar", apply)


def _clashing_world(tmp_path):
    """B's STORED description is A's NEW one, so A (written) clashes with B (stays as stored)."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    new_a = _synced_text(scratch, PROJECT)
    plan, other, conn = _world2(tmp_path)
    raw = json.loads(other.read_text())
    raw["shorts"][0]["payloads"]["youtube"]["description"] = new_a["description"]
    other.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n")
    return plan, other, conn


@pytest.mark.parametrize("how", ["oserror", "changed", "leased"])
def test_a_failed_write_that_leaves_a_clash_is_always_actionable(tmp_path, monkeypatch, how):
    plan, other, conn = _clashing_world(tmp_path)
    _break_write(monkeypatch, conn, other, how)
    by = {c.project_name: c for c in sync_short_text(conn, apply=True, now=NOW)}
    assert (by[PROJECT].status, by[PROJECT].written) == ("changed", True)
    b = by[PROJECT2]
    assert b.status == "refused" and "write failed" in b.reason
    assert "clashes with this sidecar's stored text" in b.reason
    assert not b.transient and qst.is_actionable_refusal(b)


def _third(conn, tmp_path):
    """A third plan (own directory and project, ordered after the other two by path) whose
    STORED description is the first sidecar's NEW one."""
    scratch = tmp_path / "scratch3"
    scratch.mkdir()
    new_a = _synced_text(scratch, PROJECT)
    d = tmp_path / "third"
    d.mkdir()
    project = "nctopa-2026-03-07-rec0072"
    (d / f"{project}_lf01_youtube.json").write_text(json.dumps({"location": REGION}))
    short = _short(1, project_name=project, recording_date="2026-03-07")
    short["payloads"]["youtube"]["title"] = "Mar 7, 1st stretch: giraffe words #Shorts"
    short["payloads"]["youtube"]["description"] = new_a["description"]
    plan = d / f"{project}_publish_plan.json"
    plan.write_text(json.dumps({"shorts": [short]}, sort_keys=True, indent=2) + "\n")
    enqueue(
        conn,
        project,
        1,
        "youtube",
        short["reviewed_plan_hash"],
        sidecar_path=str(plan),
        scheduled_at=short["schedule"]["scheduled_at_utc"],
    )
    return plan, project


@pytest.mark.parametrize("how", ["changed", "oserror"])
def test_a_sidecar_skipped_by_the_cascade_is_clash_checked_against_what_was_written(
    tmp_path, monkeypatch, how
):
    """A is written, B's write fails, C is skipped and stays as STORED: C's stored text equals
    A's new text, so C is actionable and names the clash (not transient, not silent)."""
    plan, other, conn = _world2(tmp_path)
    third, project3 = _third(conn, tmp_path)
    _break_write(monkeypatch, conn, other, how)
    by = {c.project_name: c for c in sync_short_text(conn, apply=True, now=NOW)}
    assert (by[PROJECT].status, by[PROJECT].written) == ("changed", True)
    c = by[project3]
    assert c.status == "refused" and "retried next run" in c.reason
    assert "clashes with this sidecar's stored text" in c.reason
    assert not c.transient and qst.is_actionable_refusal(c)


@pytest.mark.parametrize(
    ("how", "transient"),
    [
        ("changed", True),
        ("leased", True),
        ("locked", True),
        ("busy_code", True),
        ("oserror", False),
        ("disk_io", False),
        ("io_code", False),
        ("locked_code", False),  # SQLITE_LOCKED is an intra-connection conflict: deterministic
    ],
)
def test_a_failed_write_without_a_clash_is_transient_only_for_a_race_or_a_busy_database(
    tmp_path, monkeypatch, how, transient
):
    plan, other, conn = _world2(tmp_path)
    _break_write(monkeypatch, conn, other, how)
    b = {c.project_name: c for c in sync_short_text(conn, apply=True, now=NOW)}[PROJECT2]
    assert b.status == "refused" and "write failed" in b.reason
    assert "clashes" not in b.reason
    assert b.transient is transient and qst.is_actionable_refusal(b) is not transient


# -- m3: the location cap matches enqueue ---------------------------------------------------


def test_the_location_cap_is_counted_inside_a_sidecar_never_across_sidecars(tmp_path):
    plan, other, conn = _world2(tmp_path)  # one Short per sidecar, one location
    changes = sync_short_text(conn, apply=False, now=NOW, per_location_cap=1)
    assert not any("location-cap" in c.reason for c in changes)
    own = tmp_path / "own"
    own.mkdir()
    _, conn2 = _two(own)  # two Shorts of one sidecar at one location
    changes = sync_short_text(conn2, apply=False, now=NOW, per_location_cap=1)
    assert any("location-cap" in c.reason for c in changes if c.status == "refused")
