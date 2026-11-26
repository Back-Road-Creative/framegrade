"""``queue sync-short-text`` run unattended: ordered, idempotent across sidecars, gates that see
published text and later sidecars, one title snapshot per run (unit C5b)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_queue_sync_short_text import (
    NOW,
    PROJECT,
    PROJECT2,
    REGION,
    _named_first,
    _other,
    _short,
    _world,
)

from gomoveshift.pipeline import longform_metadata as lm
from gomoveshift.pipeline import title_decision
from gomoveshift.publishing import anti_spam_gate
from gomoveshift.publishing import queue_short_text as qst
from gomoveshift.publishing.queue import enqueue
from gomoveshift.publishing.queue_short_text import sync_short_text
from gomoveshift.tracking import title_evidence

_REAL_LEDGER = lm.load_published_titles


@pytest.fixture(autouse=True)
def _no_published_titles(monkeypatch):
    monkeypatch.setattr(lm, "load_published_titles", lambda: set())


def _rows(conn):
    return [tuple(r) for r in conn.execute("SELECT * FROM deliveries ORDER BY id")]


def _world2(tmp_path, day=6):
    plan, conn = _world(tmp_path, [_short(1)])
    other = _other(conn, tmp_path, day=day)
    return plan, other, conn


def _spy(monkeypatch):
    """Record ``(sidecar name, spooled)`` per ``_sidecar_changes`` call, in call order."""
    calls = []
    real = qst._sidecar_changes

    def spy(conn, sidecar, rows, now, spooled, *a):
        calls.append((sidecar.name, {k: list(v) for k, v in spooled.items()}))
        return real(conn, sidecar, rows, now, spooled, *a)

    monkeypatch.setattr(qst, "_sidecar_changes", spy)
    return calls


def test_sidecars_go_in_slot_order_ties_broken_by_path(tmp_path, monkeypatch):
    plan, other, conn = _world2(tmp_path)
    calls = _spy(monkeypatch)
    sync_short_text(conn, apply=False, now=NOW)
    assert [n for n, _ in calls] == [plan.name, other.name]  # equal slots: path order
    conn.execute(
        "UPDATE deliveries SET scheduled_at = ? WHERE project_name = ?",
        ("2026-11-25T12:00:00+00:00", PROJECT),
    )
    conn.commit()
    calls.clear()
    sync_short_text(conn, apply=False, now=NOW)
    assert [n for n, _ in calls] == [other.name, plan.name]  # the earlier slot first


def test_recent_text_is_outside_the_set_plus_earlier_final_text_only(tmp_path, monkeypatch):
    plan, other, conn = _world2(tmp_path)
    calls = _spy(monkeypatch)
    changes = sync_short_text(conn, apply=True, now=NOW)
    first_title = next(c.new_title for c in changes if c.project_name == PROJECT)
    (_, first_seen), (_, second_seen) = calls
    assert not any("zebra" in t for t in first_seen.get("youtube", []))
    bare = anti_spam_gate._bare_title(first_title).strip()
    assert bare in second_seen["youtube"]  # the earlier sidecar's FINAL text


@pytest.mark.parametrize("day", [5, 6, 7, 8, 9])
def test_a_second_apply_over_two_sidecars_is_a_no_op(tmp_path, day):
    plan, other, conn = _world2(tmp_path, day=day)
    first = sync_short_text(conn, apply=True, now=NOW)
    # day 5: both Shorts open "Mar 5, 1st stretch", so the second sidecar's gate refuses it
    assert any(c.written for c in first) or day == 5
    files = (plan.read_bytes(), other.read_bytes())
    rows = _rows(conn)
    log = conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0]
    second = sync_short_text(conn, apply=True, now=NOW)
    assert not any(c.written for c in second)
    assert {c.status for c in second} <= {"unchanged", "refused"}
    assert (plan.read_bytes(), other.read_bytes()) == files
    assert _rows(conn) == rows
    assert conn.execute("SELECT COUNT(*) FROM response_log").fetchone()[0] == log
    assert [c.status for c in second] == [
        c.status for c in sync_short_text(conn, apply=True, now=NOW)
    ]


def test_two_sidecars_that_share_a_lead_do_not_flip_it_on_every_apply(tmp_path, monkeypatch):
    # A stand-in generator with the real one's rule: skip a description lead that the
    # recent text already opens with. Both sidecars want LEAD A.
    seen = {}
    real = qst._generate_lf
    real_text = qst._youtube_text

    def spy(view, record, location, lf_index, *, spooled_texts=None, **kw):
        seen["spooled"] = spooled_texts or {}
        return real(view, record, location, lf_index, spooled_texts=spooled_texts, **kw)

    def text(short, post):
        taken = any(t.startswith("Lead A") for t in seen["spooled"].get("youtube", []))
        lead = "Lead B" if taken else "Lead A"
        made = real_text(short, post)
        return {**made, "description": f"{lead} drive. {made['description']}"}

    real_texts = anti_spam_gate._record_texts
    monkeypatch.setattr(
        anti_spam_gate,
        "_record_texts",
        lambda r: {
            **real_texts(r),
            "youtube": [r.payloads.youtube.description, *real_texts(r).get("youtube", [])],
        },
    )
    monkeypatch.setattr(qst, "_generate_lf", spy)
    monkeypatch.setattr(qst, "_youtube_text", text)
    plan, other, conn = _world2(tmp_path)
    sync_short_text(conn, apply=True, now=NOW)
    files = (plan.read_bytes(), other.read_bytes())
    rows = _rows(conn)
    again = sync_short_text(conn, apply=True, now=NOW)
    assert not any(c.written for c in again)
    assert (plan.read_bytes(), other.read_bytes()) == files and _rows(conn) == rows


def test_the_title_ledger_and_snapshot_are_read_once_per_run(tmp_path, monkeypatch):
    plan, other, conn = _world2(tmp_path)
    ledger, snap = [], []
    monkeypatch.setattr(lm, "load_published_titles", _REAL_LEDGER)
    monkeypatch.setattr(
        lm, "read_published_titles", lambda: (ledger.append(1) or set(), lm.LEDGER_OK)
    )
    monkeypatch.setattr(title_evidence, "load_title_evidence", lambda root: snap.append(1) or None)
    sync_short_text(conn, apply=False, now=NOW)
    assert (len(ledger), len(snap)) == (1, 1)
    sync_short_text(conn, apply=False, now=NOW)
    assert (len(ledger), len(snap)) == (2, 2)  # once per run, not cached across runs


def test_no_title_input_cache_outlives_a_run_and_no_function_is_swapped(tmp_path, monkeypatch):
    plan, other, conn = _world2(tmp_path)
    monkeypatch.setattr(lm, "load_published_titles", _REAL_LEDGER)
    real_s = title_decision.shipped_title_snapshot
    monkeypatch.setattr(lm, "read_published_titles", lambda: (set(), lm.LEDGER_OK))
    monkeypatch.setattr(title_evidence, "load_title_evidence", lambda root: None)
    inside = []
    real_gen = qst._generate
    monkeypatch.setattr(
        qst,
        "_generate",
        lambda *a: (
            inside.append(
                (
                    lm.load_published_titles is _REAL_LEDGER,
                    title_decision.shipped_title_snapshot is real_s,
                    title_decision.title_input_cache.get() is not None,
                )
            )
            or real_gen(*a)
        ),
    )
    sync_short_text(conn, apply=False, now=NOW)
    assert inside and set(inside) == {(True, True, True)}
    assert title_decision.title_input_cache.get() is None


def _other_two(conn, tmp_path):
    """A second plan (own directory) of two Shorts of one LF; Short 1's STORED text names
    the county the generator opens with, so how Short 2 is seeded decides a place clash."""
    d = tmp_path / "other"
    d.mkdir()
    (d / f"{PROJECT2}_lf01_youtube.json").write_text(json.dumps({"location": REGION}))
    shorts = []
    for n, base in ((1, _named_first(1)), (2, _short(2))):
        shorts.append({**base, "project_name": PROJECT2, "recording_date": "2026-03-06"})
        shorts[-1]["payloads"]["youtube"]["title"] = shorts[-1]["payloads"]["youtube"][
            "title"
        ].replace("Mar 5", "Mar 6")
    plan = d / f"{PROJECT2}_publish_plan.json"
    plan.write_text(json.dumps({"shorts": shorts}, sort_keys=True, indent=2) + "\n")
    for s in shorts:
        enqueue(
            conn,
            PROJECT2,
            s["short_index"],
            "youtube",
            s["reviewed_plan_hash"],
            sidecar_path=str(plan),
            scheduled_at=s["schedule"]["scheduled_at_utc"],
        )
    return plan


def test_a_cross_sidecar_refusal_rechecks_the_final_text(tmp_path, monkeypatch):
    # Refusing Short 1 of the later sidecar leaves its STORED text (it names Ashby County),
    # which is what Short 2 must now be seeded from: the pair is checked on the final text.
    plan, conn = _world(tmp_path, [_short(1)])
    _other_two(conn, tmp_path)
    clash_calls, generated = [], []
    real_clash = qst._successor_clashes
    monkeypatch.setattr(
        qst,
        "_successor_clashes",
        lambda sc, shorts, rewritten, *a: (
            clash_calls.append(sorted(rewritten)) or real_clash(sc, shorts, rewritten)
        ),
    )
    real_gen = qst._generate
    monkeypatch.setattr(
        qst,
        "_generate",
        lambda sc, shorts, active, spooled: (
            generated.append(sorted(active)) or real_gen(sc, shorts, active, spooled)
        ),
    )
    real = anti_spam_gate.check_batch
    fired = []

    def check(records, **kw):
        found = real(records, **kw)
        mine = [i for i, r in enumerate(records) if r.project_name == PROJECT2]
        if mine and len(records) > len(mine) and not fired:  # with the earlier sidecar's records
            fired.append(1)
            found = [*found, anti_spam_gate.Violation(mine[0], "youtube", "shared-prefix", "x")]
        return found

    monkeypatch.setattr(anti_spam_gate, "check_batch", check)
    changes = sync_short_text(conn, apply=False, now=NOW)
    first, second = (c for c in changes if c.project_name == PROJECT2)
    assert fired and first.status == "refused" and "shared-prefix" in first.reason
    assert second.status == "changed"
    title = second.new_title.split(" #")[0]
    assert "Harrow" in title and "Ashby County" not in title  # seeded from Short 1's stored text
    assert generated[-1] == [1] and len(generated) >= 3  # the later sidecar ran twice
    assert clash_calls[-1] == [1]  # the neighbour check ran again, on the final text


def _project2_world(conn, tmp_path):
    """Project 2: Short 1 PUBLISHED ("zebra words"), Short 2 queued, one sidecar."""
    d = tmp_path / "other"
    d.mkdir()
    (d / f"{PROJECT2}_lf01_youtube.json").write_text(json.dumps({"location": REGION}))
    shorts = []
    for n in (1, 2):
        short = _short(n, project_name=PROJECT2, recording_date="2026-03-06")
        short["payloads"]["youtube"]["title"] = f"Mar 6, {('1st', '2nd')[n - 1]} stretch: " + (
            "zebra words #Shorts" if n == 1 else "plain words #Shorts"
        )
        shorts.append(short)
    plan = d / f"{PROJECT2}_publish_plan.json"
    plan.write_text(json.dumps({"shorts": shorts}, sort_keys=True, indent=2) + "\n")
    for s in shorts:
        enqueue(
            conn,
            PROJECT2,
            s["short_index"],
            "youtube",
            s["reviewed_plan_hash"],
            sidecar_path=str(plan),
            scheduled_at=s["schedule"]["scheduled_at_utc"],
        )
    conn.execute(
        "UPDATE deliveries SET status = 'published' WHERE project_name = ? AND short_index = 1",
        (PROJECT2,),
    )
    conn.commit()
    return plan


def test_published_text_of_a_project_with_a_queued_short_reaches_the_gates(tmp_path, monkeypatch):
    plan, conn = _world(tmp_path, [_short(1)])
    _project2_world(conn, tmp_path)
    calls = _spy(monkeypatch)
    sync_short_text(conn, apply=False, now=NOW)
    (name_a, seen_a), (name_b, seen_b) = calls
    assert name_a == plan.name
    assert any("zebra words" in t for t in seen_a.get("youtube", []))  # another sidecar's gate
    assert not any("zebra words" in t for t in seen_b.get("youtube", []))  # its own project
    # ... and the queued Short 2 of project 2 is NOT read from disk: only `earlier` supplies it
    assert not any("plain words" in t for t in seen_a.get("youtube", []))


def test_recent_platform_texts_can_leave_out_single_shorts_of_a_project(tmp_path):
    plan, conn = _world(tmp_path, [_short(1)])
    _project2_world(conn, tmp_path)
    both = anti_spam_gate.recent_platform_texts(conn, exclude_projects={PROJECT})
    assert any("zebra words" in t for t in both["youtube"])
    assert any("plain words" in t for t in both["youtube"])
    only = anti_spam_gate.recent_platform_texts(
        conn, exclude_projects={PROJECT}, exclude_shorts={(PROJECT2, 2)}
    )
    assert any("zebra words" in t for t in only["youtube"])
    assert not any("plain words" in t for t in only["youtube"])
    entries = anti_spam_gate.recent_text_entries(conn, exclude_projects={PROJECT})
    assert {(p, i) for p, i, _pl, _t in entries} == {(PROJECT2, 1), (PROJECT2, 2)}


def test_a_queued_sibling_platform_row_of_a_synced_short_is_not_double_counted(
    tmp_path, monkeypatch
):
    plan, conn = _world(tmp_path, [_short(1)], platforms=("youtube", "tiktok"))
    calls = _spy(monkeypatch)
    sync_short_text(conn, apply=False, now=NOW)
    assert calls and not any(
        "stretch: old words" in t for t in calls[0][1].get("youtube", [])
    )  # own project: excluded


def _later_sidecar_that_stays_as_stored(tmp_path, conn, description):
    """A second plan whose Short is refused (a sibling row is uploading), so its stored
    text stays; its description is ``description``."""
    d = tmp_path / "other"
    d.mkdir()
    (d / f"{PROJECT2}_lf01_youtube.json").write_text(json.dumps({"location": REGION}))
    short = _short(1, project_name=PROJECT2, recording_date="2026-03-06")
    short["payloads"]["youtube"]["title"] = "Mar 6, 1st stretch: zebra words #Shorts"
    short["payloads"]["youtube"]["description"] = description
    plan = d / f"{PROJECT2}_publish_plan.json"
    plan.write_text(json.dumps({"shorts": [short]}, sort_keys=True, indent=2) + "\n")
    for platform in ("youtube", "facebook"):
        enqueue(
            conn,
            PROJECT2,
            1,
            platform,
            short["reviewed_plan_hash"],
            sidecar_path=str(plan),
            scheduled_at=short["schedule"]["scheduled_at_utc"],
        )
    conn.execute(
        "UPDATE deliveries SET status = 'uploading' WHERE platform = 'facebook' AND project_name = ?",
        (PROJECT2,),
    )
    conn.commit()
    return plan


def test_an_earlier_sidecars_new_text_is_checked_against_a_later_sidecar_that_stays(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    probe_plan, probe = _world(scratch, [_short(1)])
    new_description = None
    sync_short_text(probe, apply=True, now=NOW)
    from test_queue_sync_short_text import _youtube

    new_description = _youtube(probe_plan, 1).payloads.youtube.description
    assert new_description != _short(1)["payloads"]["youtube"]["description"]

    plan, conn = _world(tmp_path, [_short(1)])
    other = _later_sidecar_that_stays_as_stored(tmp_path, conn, new_description)
    before = (plan.read_bytes(), other.read_bytes())
    changes = sync_short_text(conn, apply=True, now=NOW)
    first = next(c for c in changes if c.project_name == PROJECT)
    assert first.status == "refused" and "cross-batch-repeat" in first.reason
    assert not first.written and (plan.read_bytes(), other.read_bytes()) == before


def test_a_failed_write_refuses_the_sidecars_after_it_and_they_converge_next_run(
    tmp_path, monkeypatch
):
    plan, other, conn = _world2(tmp_path)
    real = qst._apply_sidecar
    broken = [True]

    def apply(conn_, sidecar, *a):
        if sidecar == plan and broken[0]:
            raise OSError("disk full")
        return real(conn_, sidecar, *a)

    monkeypatch.setattr(qst, "_apply_sidecar", apply)
    before = (plan.read_bytes(), other.read_bytes())
    rows = _rows(conn)
    changes = sync_short_text(conn, apply=True, now=NOW)
    first = next(c for c in changes if c.project_name == PROJECT)
    second = next(c for c in changes if c.project_name == PROJECT2)
    assert first.status == "refused" and "write failed" in first.reason
    assert second.status == "refused" and "earlier sidecar's write failed" in second.reason
    assert (plan.read_bytes(), other.read_bytes()) == before and _rows(conn) == rows
    broken[0] = False
    again = sync_short_text(conn, apply=True, now=NOW)
    assert any(c.written for c in again)


def test_the_final_records_come_from_the_planned_sidecar_not_a_second_read(tmp_path):
    plan, other, conn = _world2(tmp_path)
    rows = conn.execute("SELECT * FROM deliveries WHERE project_name = ?", (PROJECT,)).fetchall()
    out = qst._sidecar_changes(conn, plan, rows, NOW, {}, (), {})
    verdict, rewritten, _sha, finals = out
    assert [r.short_index for r in finals] == [1]
    assert finals[0].payloads.youtube.title == verdict[rows[0]["id"]].new_title
    assert finals[0].payloads.youtube.title != _short(1)["payloads"]["youtube"]["title"]


def test_a_refused_hash_check_reseeds_the_neighbour_on_the_final_text(tmp_path):
    plan, conn = _world(tmp_path, [_named_first(), _short(2)])
    conn.execute("UPDATE deliveries SET plan_hash = ? WHERE short_index = 1", ("d" * 64,))
    conn.commit()
    changes = sync_short_text(conn, apply=False, now=NOW)
    first, second = changes
    assert first.status == "refused" and "is unknown" in first.reason
    title = second.new_title.split(" #")[0]
    assert "Harrow" in title and "Ashby County" not in title  # Short 1 keeps its stored place


def test_a_short_is_never_gated_against_its_own_stored_copy(tmp_path, monkeypatch):
    # One file under two sidecar_path spellings, the first unusable this run: its stored text
    # reaches the second's gates, which must never hold the second's own Short (unit C5b).
    second = _short(2)
    second["payloads"]["youtube"]["description"] = "Another view of Ashby County #Shorts"
    plan, conn = _world(tmp_path, [_short(1), second])
    raw = f"{plan.parent}/./{plan.name}"  # sorts first: planned first
    conn.execute("UPDATE deliveries SET sidecar_path = ? WHERE short_index = 2", (raw,))
    conn.commit()
    real_text, real_changes = qst._youtube_text, qst._sidecar_changes

    def keep_opening(short, post):  # the new text keeps the stored openings
        out = real_text(short, post)
        out["title"] = short["payloads"]["youtube"]["title"].split(":")[0] + ": " + out["title"]
        out["description"] = "Front-seat view driving through Ashby County, " + out["description"]
        return out

    def unusable(conn, sidecar, rows, *a):
        if rows[0]["short_index"] == 2:
            raise OSError("unreadable this run")
        return real_changes(conn, sidecar, rows, *a)

    monkeypatch.setattr(qst, "_youtube_text", keep_opening)
    monkeypatch.setattr(qst, "_sidecar_changes", unusable)
    first, second = sync_short_text(conn, apply=False, now=NOW)
    assert first.status == "changed", first.reason
    assert first.new_title.startswith("Mar 5, 1st stretch: ")
    assert second.status == "refused" and "unreadable this run" in second.reason
