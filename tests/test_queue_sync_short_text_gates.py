"""``queue sync-short-text``: refusal classes, unusable sidecars, failed writes, the location cap
(unit C5b, review fixes)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_queue_sync_short_text import (
    NOW,
    PROJECT,
    PROJECT2,
    _other,
    _short,
    _two,
    _world,
)

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.publishing import anti_spam_gate
from gomoveshift.publishing import queue_short_text as qst
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


@pytest.mark.parametrize(
    "reason",
    [
        "facebook row 3 holds a live lease",
        "youtube row 3 slot 2026-11-21T00:00:00+00:00 is inside the lead time",
        "an earlier sidecar's write failed; retried next run",
        "sidecar unusable: write failed: a row was leased during the write",
        "sidecar unusable: write failed: p.json changed while the sync ran",
    ],
)
def test_the_named_transient_refusals_clear_on_their_own(reason):
    change = qst.ShortTextChange(1, "p", 1, "refused", "a", "b", reason)
    assert qst.is_actionable_refusal(change) is False
    assert not qst.is_actionable_refusal(qst.ShortTextChange(1, "p", 1, "changed", "a", "b", "x"))


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


def test_after_a_failed_write_the_written_text_is_checked_against_the_failed_stored_text(
    tmp_path, monkeypatch
):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    new_a = _synced_text(scratch, PROJECT)
    plan, other, conn = _world2(tmp_path)
    raw = json.loads(other.read_text())  # B's STORED text is A's new text
    raw["shorts"][0]["payloads"]["youtube"].update(new_a)
    other.write_text(json.dumps(raw, sort_keys=True, indent=2) + "\n")
    _fail_write_of(monkeypatch, other, OSError("disk full"))
    changes = sync_short_text(conn, apply=True, now=NOW)
    first = next(c for c in changes if c.project_name == PROJECT)
    if first.written:  # planned against B's NEW text, which differs from what stays queued
        assert "stored text" in first.reason


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
