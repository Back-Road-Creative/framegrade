"""Replace-scheduled finish: durable per-LF state, private insert, recheck, then delete.

The two failures that must never happen on a live channel: two scheduled videos in one slot, and
deleting a video that is already public. YouTube is a fake world here; no network."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from seo_manager.scheduling.store import SlotTaken

from gomoveshift.pipeline import replace_scheduled as rs
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc
from gomoveshift.pipeline.title_decision import require_title_decision

from _state_helpers import make_state

NOW = datetime(2026, 11, 22, 12, 0, tzinfo=timezone.utc)
AT2 = NOW + timedelta(days=5)
AT3 = NOW + timedelta(days=12)
PUBLIC = {"video_id": "pub", "publish_at": (NOW - timedelta(days=2)).isoformat(),
          "title": "T pub", "playlist_id": "PL1"}  # fmt: skip


class Crash(BaseException):
    """A process death: never caught by an ``except Exception``."""


def _old(vid, at, **kw):
    """An upload record as the pipeline wrote it: it carries the decision its title shipped under."""
    return {"video_id": vid, "url": f"https://youtu.be/{vid}", "publish_at": at.isoformat(),
            "title": f"Old {vid}", "playlist_id": "PL1",
            "title_decision": {"chosen_title": f"Old {vid}", "reason": "shipped"}, **kw}  # fmt: skip


class World:
    def __init__(self):
        self.videos = {
            "old2": {"privacy": "private", "publish_at": AT2, "playlists": ["PL1"]},
            "old3": {"privacy": "private", "publish_at": AT3, "playlists": ["PL1"]},
            "pub": {"privacy": "public", "publish_at": None, "playlists": ["PL1"]},
        }
        self.events: list = []
        self.crash_after: str | None = None
        self.fail: set = set()
        self.on_insert = None
        self.n = 0

    def log(self, *ev):
        self.events.append(ev)
        if self.crash_after == ev[0]:
            self.crash_after = None
            raise Crash(ev[0])

    def of(self, kind):
        return [e for e in self.events if e[0] == kind]

    def scheduled(self):
        return sorted(
            v for v, d in self.videos.items() if d["privacy"] == "private" and d["publish_at"]
        )


class FakeYT:
    def __init__(self, world):
        self.w = world

    def set_operator_checklist_path(self, path): ...

    def _emit_endscreen_checklist(self, **kw): ...

    def video_schedule(self, vid):
        if "schedule" in self.w.fail:
            raise RuntimeError("videos.list 500")
        d = self.w.videos.get(vid)
        return None if d is None else {"privacy": d["privacy"], "publish_at": d["publish_at"]}

    def upload_if_needed(
        self, *, state, lf_idx, title, publish_at, privacy, metadata, config, persist=None, **kw
    ):
        require_title_decision(title, metadata)  # the real gate
        self.w.n += 1
        vid = f"new{lf_idx}"
        self.w.videos[vid] = {"privacy": privacy, "publish_at": publish_at, "playlists": []}
        rec = {"video_id": vid, "url": f"https://youtu.be/{vid}", "title": title,
               "publish_at": publish_at.isoformat() if publish_at else None}  # fmt: skip
        if metadata.get("title_decision"):
            rec["title_decision"] = metadata["title_decision"]
        state.youtube_uploads[lf_idx] = rec
        self.w.log("insert", vid, privacy, publish_at, title, config)
        if persist:
            persist()
        if self.w.on_insert:
            self.w.on_insert(self.w)
        return rec

    def update_privacy(self, vid, *, privacy, publish_at=None):
        if "update_privacy" in self.w.fail:
            raise RuntimeError("videos.update 500")
        d = self.w.videos[vid]
        d["privacy"], d["publish_at"] = "private", publish_at
        self.w.log("update_privacy", vid, publish_at)

    def add_to_playlist(self, vid, playlist_id):
        if (
            playlist_id in self.w.videos[vid]["playlists"]
        ):  # the real method lists before it inserts
            return
        self.w.videos[vid]["playlists"].append(playlist_id)
        self.w.log("playlist", vid, playlist_id)

    def resolve_playlist_id(self, *, metadata, config):
        self.w.log("resolved", "PL-RESOLVED")
        return "PL-RESOLVED"

    def delete_video(self, vid):
        if "delete" in self.w.fail:
            raise RuntimeError("videos.delete 500")
        self.w.videos.pop(vid, None)
        self.w.log("delete", vid)


class Rig:
    def __init__(self, tmp_path, *, old2=None, old3=None, titles=None):
        self.dir = tmp_path
        self.world = World()
        self.state = make_state()
        self.booked: list = []
        self.reschedule = MagicMock(return_value=None)
        self.old2 = old2 or _old("old2", AT2)
        self.old3 = old3 or _old("old3", AT3)
        self.write_plan()
        self.titles = titles or {i: f"New {i}" for i in (1, 2, 3)}
        self.config = MagicMock()
        self.config.seo.storage_path = tmp_path / "seo"
        self.rebook = lambda now: None  # Waypost's next free slot; None = nothing free
        self.held: dict = {}  # slot date -> LF that holds it (the ledger + schedule.sqlite3, faked)
        self.book_error: Exception | None = None

    def reserve_factory(self, config, state, project_base, lf, old_id):
        def reserve(slot):
            if self.held.get(slot.date(), lf) != lf:
                raise SlotTaken(f"{slot.date()} is held by {self.held[slot.date()]}")
            self.held = {d: o for d, o in self.held.items() if o != lf}  # an LF holds one slot
            self.held[slot.date()] = lf
            self.world.log("reserve", lf, slot)

        return reserve

    def free_slot_after(self, start):
        """Waypost's allocator: the first day from ``start`` that nothing holds."""

        def pick(now):
            day = start
            while day.date() in self.held:
                day += timedelta(days=1)
            return day

        return pick

    def write_plan(self, **extra):
        rs.write_plan(self.dir, {"schema": rs.REPLACE_PLAN_SCHEMA, "kept": {"1": PUBLIC},
                                 "replace": {"2": self.old2, "3": self.old3}, **extra})  # fmt: skip

    def plan(self):
        return json.loads((self.dir / rs.REPLACE_PLAN_FILENAME).read_text())

    def prog(self, lf):
        return self.plan()["progress"][str(lf)]

    def meta(self, i):
        decision = {"chosen_title": self.titles[i], "reason": "fresh"}
        return {"title": self.titles[i], "description": "d", "tags": [], "title_decision": decision}

    def _raise_book(self):
        if self.book_error:
            raise self.book_error

    def run(self, *, crash_after=None, now=NOW):
        self.world.crash_after = crash_after
        sched = SimpleNamespace(lf_slots=[SimpleNamespace(
            datetime_est=(NOW + timedelta(days=d)).isoformat(), date="x") for d in (30, 31, 32)])  # fmt: skip
        with (
            patch.object(esc, "_resolve_youtube_uploader_class", return_value=lambda: FakeYT(self.world)),
            patch.object(esc, "_book_lf_upload_in_ledger",
                         side_effect=lambda **kw: (self._raise_book(), self.world.log("book", kw["record"]["video_id"]),
                                                   self.booked.append(kw["record"]))),
            patch.object(esc, "_reserve_lf_slot", create=True, side_effect=self.reserve_factory),
            patch.object(esc, "_next_free_lf_slot", side_effect=lambda config, state: lambda now: self.rebook(now)),  # fmt: skip
            patch.object(esc, "_upload_superlong_parts_to_youtube") as superlong,
        ):  # fmt: skip
            self.superlong = superlong
            esc._upload_lfs_to_youtube(
                drive_plan=SimpleNamespace(regular_lfs=[SimpleNamespace(index=i) for i in (1, 2, 3)]),
                regular_lf_paths=["/v/1", "/v/2", "/v/3"],
                metadata_per_lf={i: self.meta(i) for i in (1, 2, 3)},
                drive_schedule=sched, youtube_uploads={}, state=self.state, output_base=self.dir,
                config=self.config, save_state=lambda s: self.world.log("save"),
                skip_youtube_upload=False, youtube_upload_enabled=True, clock=lambda: now,
                reschedule=self.reschedule,
            )  # fmt: skip


@pytest.fixture(autouse=True)
def _swap(monkeypatch):
    monkeypatch.setattr(rs, "swap_ledger_video_id", lambda *a, **k: True)


def _assert_replaced(rig, lf=2):
    w = rig.world
    old, new = f"old{lf}", f"new{lf}"
    at = AT2 if lf == 2 else AT3
    assert old not in w.videos and w.videos[new]["publish_at"] == at
    assert w.videos[new]["playlists"] == ["PL1"]
    assert w.scheduled().count(new) == 1 and old not in w.scheduled()
    assert rig.state.youtube_uploads[lf]["video_id"] == new
    assert rig.prog(lf)["state"] == "done"


def test_happy_path_order_private_insert_recheck_delete_schedule_swap_book(tmp_path):
    rig = Rig(tmp_path)
    swaps = []
    with patch.object(
        rs, "swap_ledger_video_id", side_effect=lambda *a, **k: swaps.append(a) or True
    ):
        rig.run()
    w = rig.world
    ins = [e for e in w.of("insert") if e[1] == "new2"][0]
    assert ins[2] == "private" and ins[3] is None and ins[4] == "New 2"  # the NEW gated title
    assert ins[5] is None  # no playlist at insert: the old record's playlist is forced later
    seq = [e[:2] for e in w.events if e[1:2] in (("new2",), ("old2",)) and e[0] != "save"]
    assert seq == [("insert", "new2"), ("delete", "old2"), ("update_privacy", "new2"),
                   ("playlist", "new2"), ("book", "new2")]  # fmt: skip
    assert ("old2", "new2") in [a[-2:] for a in swaps]
    _assert_replaced(rig, 2)
    _assert_replaced(rig, 3)
    assert not w.of("resolved")  # playlist forced, never freshly resolved


@pytest.mark.parametrize("step", ["insert", "delete", "update_privacy", "playlist", "book"])
def test_crash_after_each_step_then_resume_finishes(tmp_path, step):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after=step)
    rig.run()
    w = rig.world
    _assert_replaced(rig, 2)
    _assert_replaced(rig, 3)
    assert len(w.of("delete")) == 2 and len(w.of("insert")) == 2
    assert len(w.of("playlist")) == 2  # filed once per video, never twice
    assert w.scheduled() == ["new2", "new3"]


def test_crash_in_the_ledger_swap_resumes_without_a_second_delete(tmp_path):
    rig = Rig(tmp_path)
    calls = []

    def swap(*a, **k):
        calls.append(a)
        if len(calls) == 1:
            raise Crash("swap")
        return True

    with patch.object(rs, "swap_ledger_video_id", side_effect=swap):
        with pytest.raises(Crash):
            rig.run()
        assert rig.prog(2)["state"] == "old_deleted"
        rig.run()
    assert len(rig.world.of("delete")) == 2
    _assert_replaced(rig, 2)


@pytest.mark.parametrize("state_name", ["uploaded", "old_deleted", "scheduled"])
def test_resume_from_each_recorded_state(tmp_path, state_name):
    rig = Rig(tmp_path)
    w = rig.world
    w.videos["new2"] = {"privacy": "private", "publish_at": None, "playlists": []}
    w.videos["new3"] = {"privacy": "private", "publish_at": None, "playlists": []}
    fields = {"old_id": "old2", "new_id": "new2", "state": state_name}
    if state_name in ("old_deleted", "scheduled"):
        w.videos.pop("old2")
    if state_name == "scheduled":
        w.videos["new2"].update(publish_at=AT2, playlists=["PL1"])
        fields.update(publish_set=True, playlist_set=True)
    rig.write_plan(progress={"2": fields, "3": {"old_id": "old3", "state": "pending"}})
    rig.state.youtube_uploads = {2: {"video_id": "new2", "title": "Old old2", "publish_at": None}}
    w.videos.pop("new3")
    rig.run()
    _assert_replaced(rig, 2)
    assert (
        len(w.of("delete")) == (1 if state_name == "uploaded" else 0) + 1
    )  # old2 only if uploaded; old3 always


def _went_public(w):
    w.videos["old2"].update(privacy="public", publish_at=None)


def _inside_margin(w):
    w.videos["old2"].update(publish_at=NOW + timedelta(hours=rs.REPLACE_MARGIN_H - 1))


@pytest.mark.parametrize("change", [_went_public, _inside_margin])
def test_a_refused_recheck_deletes_the_new_and_never_the_old(tmp_path, change):
    rig = Rig(tmp_path)
    rig.world.on_insert = change
    rig.run()
    w = rig.world
    assert "old2" in w.videos and "new2" not in w.videos
    assert ("delete", "old2") not in [e[:2] for e in w.events]
    assert ("delete", "new2") in [e[:2] for e in w.events]
    assert rig.prog(2)["state"] == "abandoned"
    assert rig.state.youtube_uploads[2]["video_id"] == "old2"
    assert not [e for e in w.of("update_privacy") if e[1] == "new2"]
    assert _incidents("abandoned")  # the replace did not happen: an open incident says so


def test_a_failed_delete_stays_uploaded_and_is_retried_next_entry(tmp_path):
    rig = Rig(tmp_path)
    rig.world.fail.add("delete")
    rig.run()
    w = rig.world
    assert rig.prog(2)["state"] == "uploaded" and "old2" in w.videos
    assert not w.of("update_privacy")  # never schedules the new while the old holds the slot
    assert _incidents("old_undeleted")
    w.fail.discard("delete")
    rig.run()
    _assert_replaced(rig, 2)
    assert len(w.of("insert")) == 2  # no second insert on the retry


def test_a_swap_failure_is_retried_next_entry_with_no_second_delete(tmp_path):
    rig = Rig(tmp_path)
    boom = {"on": True}

    def swap(*a, **k):
        if boom["on"]:
            raise RuntimeError("ledger locked")
        return True

    with patch.object(rs, "swap_ledger_video_id", side_effect=swap):
        rig.run()
        assert rig.prog(2)["state"] == "old_deleted"
        assert _incidents("schedule_failed")
        boom["on"] = False
        rig.run()
    assert [e[1] for e in rig.world.of("delete")].count("old2") == 1
    _assert_replaced(rig, 2)


def test_slot_passed_under_a_plan_is_refused_and_the_old_is_kept(tmp_path):
    rig = Rig(tmp_path, old2=_old("old2", NOW - timedelta(hours=1)))
    rig.run()
    w = rig.world
    assert not [e for e in w.of("insert") if e[1] == "new2"]
    assert "old2" in w.videos and ("delete", "old2") not in [e[:2] for e in w.events]
    rig.reschedule.assert_not_called()
    assert rig.state.youtube_uploads[2]["video_id"] == "old2"


def test_the_playlist_falls_back_to_resolution_only_without_one_on_the_old_record(tmp_path):
    old = _old("old2", AT2)
    del old["playlist_id"]
    rig = Rig(tmp_path, old2=old)
    rig.run()
    assert [e[1] for e in rig.world.of("resolved")] == ["PL-RESOLVED"]
    assert [e[2] for e in rig.world.of("playlist") if e[1] == "new2"] == ["PL-RESOLVED"]  # once


def _incidents(code=None):
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _bad = load_all(default_incidents_dir(default_data_root()))
    rows = [r for r in rows if r.get("kind") == "replace_scheduled_partial" and not r.get("resolved_at")]  # fmt: skip
    return [r for r in rows if code in (None, r.get("code"))]


def test_title_gate_refusal_fails_safe_pending_old_untouched(tmp_path, caplog):
    rig = Rig(tmp_path)
    good = rig.meta

    def meta(i):  # a decision that does not match its own title: the real gate refuses it
        out = good(i)
        if i == 2:
            out["title_decision"] = {"chosen_title": "Something else", "reason": "x"}
        return out

    rig.meta = meta
    with caplog.at_level("ERROR"):
        rig.run()
    w = rig.world
    assert not [e for e in w.of("insert") if e[1] == "new2"]
    assert "old2" in w.videos and ("delete", "old2") not in [e[:2] for e in w.events]
    assert rig.prog(2)["state"] == "pending"
    assert "REPLACE-TITLE-GATE" in caplog.text
    assert _incidents("title_gate")


@pytest.mark.parametrize("legacy", [False, True])
def test_the_replacement_goes_up_under_the_new_gated_title(tmp_path, legacy):
    old = _old("old2", AT2)
    if legacy:
        del old["title_decision"]  # a legacy record carries no decision at all
    rig = Rig(tmp_path, old2=old, titles={1: "New 1", 2: "A Different New Title", 3: "New 3"})
    rig.run()
    _assert_replaced(rig, 2)
    (ins,) = [e for e in rig.world.of("insert") if e[1] == "new2"]
    assert ins[4] == "A Different New Title"  # the title the NEW decision chose
    rec = rig.state.youtube_uploads[2]
    assert rec["title"] == "A Different New Title"
    assert rec["title_decision"]["chosen_title"] == "A Different New Title"
    assert (rig.prog(2)["old_title"], rig.prog(2)["new_title"]) == ("Old old2", "A Different New Title")  # fmt: skip


def test_a_kept_lf_gets_no_insert_book_or_superlong(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    w = rig.world
    assert not [e for e in w.events if "pub" in e[1:2] or "new1" in e[1:2]]
    assert rig.state.youtube_uploads[1]["video_id"] == "pub"
    rig.superlong.assert_not_called()


def test_a_naive_old_publish_at_is_read_as_utc(tmp_path):
    naive = _old("old2", AT2)
    naive["publish_at"] = AT2.replace(tzinfo=None).isoformat()
    rig = Rig(tmp_path, old2=naive)
    rig.run()
    assert rig.world.videos["new2"]["publish_at"] == AT2
    assert rig.world.videos["new2"]["publish_at"].tzinfo is not None


def test_a_transient_live_check_failure_before_the_upload_leaves_the_lf_pending(tmp_path):
    rig = Rig(tmp_path)
    rig.world.fail.add("schedule")
    rig.run()
    assert not rig.world.of("insert") and not rig.world.of("delete")
    assert rig.prog(2)["state"] == "pending"
    assert 2 not in rig.state.youtube_uploads  # not written back as the old record
    rig.world.fail.discard("schedule")
    rig.run()
    _assert_replaced(rig, 2)


def test_plan_state_is_on_disk_before_the_insert_and_written_atomically(tmp_path):
    rig = Rig(tmp_path)
    at_insert = []
    real_log = rig.world.log

    def log(*ev):  # the plan state as ``videos.insert`` is called, before the new id exists
        if ev[0] == "insert" and ev[1] == "new2":
            at_insert.append(rig.prog(2)["state"])
        real_log(*ev)

    rig.world.log = log
    rig.run()
    assert at_insert[0] == "pending"  # recorded BEFORE the external call
    assert rig.prog(2)["state"] == "done" and not list(tmp_path.glob("*.tmp"))
    with patch.object(rs.os, "replace", side_effect=Crash("mid-write")):
        with pytest.raises(Crash):
            rs.set_progress(tmp_path, 2, state="pending")
    assert rig.prog(2)["state"] == "done"  # the torn write never reached the plan


def test_keys_are_normalised_to_int_on_load(tmp_path):
    raw = {"schema": rs.REPLACE_PLAN_SCHEMA, "kept": {"1": PUBLIC},
           "replace": {"lf_2": _old("o", AT2), "3": _old("p", AT3)}}  # fmt: skip
    (tmp_path / rs.REPLACE_PLAN_FILENAME).write_text(json.dumps(raw))
    plan = rs.load_plan(tmp_path)
    assert sorted(rs.replace_map(plan)) == [2, 3] and sorted(rs.kept_map(plan)) == [1]


def _uploader(svc):
    from gomoveshift.pipeline.drive_phases.youtube_uploader import YouTubeUploader

    yt = YouTubeUploader.__new__(YouTubeUploader)
    yt._build_service = lambda: svc
    yt._add_video_to_playlist = MagicMock()
    return yt


def test_uploader_video_schedule_reads_live_privacy_and_publish_at():
    svc = MagicMock()
    items = [
        {"id": "a", "status": {"privacyStatus": "private", "publishAt": "2026-12-01T20:00:00Z"}}
    ]
    svc.videos().list().execute.return_value = {"items": items}
    got = _uploader(svc).video_schedule("a")
    assert got == {
        "privacy": "private",
        "publish_at": datetime(2026, 12, 1, 20, tzinfo=timezone.utc),
    }
    svc.videos().list().execute.return_value = {"items": []}
    assert _uploader(svc).video_schedule("a") is None


@pytest.mark.parametrize("already_in", [False, True])
def test_uploader_files_a_video_in_the_named_playlist_once(already_in):
    svc = MagicMock()
    svc.playlistItems().list().execute.return_value = {"items": [{"id": "x"}] if already_in else []}
    yt = _uploader(svc)
    yt.add_to_playlist("v", "PL9")
    assert yt._add_video_to_playlist.call_count == (0 if already_in else 1)


def test_a_lookup_error_is_not_read_as_no_longer_scheduled():
    class YT:
        def video_schedule(self, vid):
            raise RuntimeError("videos.list 500")

    verdict, why = rs.check_old(YT(), {"video_id": "o"}, NOW)
    assert verdict == "error" and why.startswith("lookup failed")


def _claim_rig(tmp_path, plan_state):
    """A claimed drive whose publish job ended, its output holding kept receipts and a plan."""
    from gomoveshift.health import publish

    out = tmp_path / "day-a"
    out.mkdir()
    (out / "youtube_uploads.json").write_text(json.dumps({"1": PUBLIC}))
    if plan_state:
        rs.write_plan(out, {"schema": rs.REPLACE_PLAN_SCHEMA, "kept": {"1": PUBLIC},
                            "replace": {"2": _old("old2", AT2)},
                            "progress": {"2": {"state": plan_state}}})  # fmt: skip
    drive = SimpleNamespace(project="day-a", status="claimed", claim_id=7, output_dir=str(out),
                            claimed_at=datetime.fromtimestamp(1, timezone.utc))  # fmt: skip
    store = SimpleNamespace(all_drives=lambda: [drive], finish_claim=lambda cid, ok: "complete" if ok else "failed")  # fmt: skip
    return publish, store


@pytest.mark.parametrize("plan_state", ["pending", "uploaded", "old_deleted", "slot_passed"])
def test_an_unfinished_replace_closes_the_claim_failed_so_the_publish_is_retried(tmp_path, monkeypatch, plan_state):  # fmt: skip
    # SPEC CHANGE (fixup 3): this used to assert ``complete`` on the kept receipts, which left no
    # automatic path back into Phase D. ``failed`` lets ``health backlog`` republish the drive.
    publish, store = _claim_rig(tmp_path, plan_state)
    monkeypatch.setattr(publish, "_job_records", lambda d: [{"name": "gms-publish-day-a", "created_at": 5, "result": {"state": "failed"}}])  # fmt: skip
    monkeypatch.setattr(publish, "retire_uploaded", lambda s: [])
    (row,) = publish.reconcile_publishes(store, tmp_path)
    assert row["status"] == "failed" and "partial" in row
    (inc,) = [i for i in _incidents("partial_finish") if i["drive"] == "day-a"]
    assert plan_state in inc["error"]
    assert not _incidents("partial_finish")[0].get("resolved_at")


def test_an_abandoned_replace_is_terminal_and_closes_complete(tmp_path, monkeypatch):
    # SPEC CHANGE (fixup 3): abandoned keeps the old video, nothing is left to retry.
    publish, store = _claim_rig(tmp_path, "abandoned")
    monkeypatch.setattr(publish, "_job_records", lambda d: [{"name": "gms-publish-day-a", "created_at": 5, "result": {"state": "failed"}}])  # fmt: skip
    monkeypatch.setattr(publish, "retire_uploaded", lambda s: [])
    (row,) = publish.reconcile_publishes(store, tmp_path)
    assert row["status"] == "complete" and "partial" not in row


def test_the_partial_incident_resolves_itself_once_the_replace_is_done(tmp_path, monkeypatch):
    publish, store = _claim_rig(tmp_path, "uploaded")
    jobs = [{"name": "gms-publish-day-a", "created_at": 5, "result": {"state": "failed"}}]
    monkeypatch.setattr(publish, "_job_records", lambda d: jobs)
    monkeypatch.setattr(publish, "retire_uploaded", lambda s: [])
    publish.reconcile_publishes(store, tmp_path)
    assert _incidents("partial_finish")
    out = tmp_path / "day-a"
    rs.set_progress(out, 2, state="done")
    store.all_drives()[0].status = "claimed"
    assert publish.reconcile_publishes(store, tmp_path)[0]["status"] == "complete"
    assert not _incidents()


@pytest.mark.parametrize("plan_state", [None, "done"])
def test_a_finished_or_plain_drive_closes_clean(tmp_path, monkeypatch, plan_state):
    publish, store = _claim_rig(tmp_path, plan_state)
    monkeypatch.setattr(publish, "_job_records", lambda d: [{"name": "gms-publish-day-a", "created_at": 5, "result": {"state": "succeeded"}}])  # fmt: skip
    monkeypatch.setattr(publish, "retire_uploaded", lambda s: [])
    assert publish.reconcile_publishes(store, tmp_path) == [{"project": "day-a", "status": "complete"}]  # fmt: skip
    assert not _incidents()


def _deleted_then_recheck_fails(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="delete")  # old2 deleted on YouTube, state still 'uploaded'
    assert "old2" not in rig.world.videos and rig.prog(2)["state"] == "uploaded"
    rig.world.fail.add("schedule")  # resume: videos.list errors
    return rig


def test_a_recheck_error_defers_and_never_deletes_the_only_copy(tmp_path):
    rig = _deleted_then_recheck_fails(tmp_path)
    rig.run()
    w = rig.world
    assert "new2" in w.videos and ("delete", "new2") not in [e[:2] for e in w.events]
    assert rig.prog(2)["state"] == "uploaded" and _incidents("recheck_failed")
    w.fail.discard("schedule")  # the lookup recovers: gone -> old_deleted -> scheduled -> done
    rig.run()
    _assert_replaced(rig, 2)


def test_a_delete_that_committed_but_errored_never_costs_the_new(tmp_path):
    from unittest.mock import patch as _p

    rig = Rig(tmp_path)
    w = rig.world
    n = {"n": 0}

    def flaky(self, vid):
        n["n"] += 1
        self.w.videos.pop(vid, None)
        if vid == "old2" and n["n"] == 1:
            raise RuntimeError("504 after commit")
        self.w.log("delete", vid)

    with _p.object(FakeYT, "delete_video", flaky):
        rig.run()
        assert rig.prog(2)["state"] == "uploaded" and "old2" not in w.videos
        w.fail.add("schedule")
        rig.run()
        assert "new2" in w.videos and rig.prog(2)["state"] == "uploaded"
        w.fail.discard("schedule")
        rig.run()
    _assert_replaced(rig, 2)
    assert ("delete", "new2") not in [e[:2] for e in w.events]


def test_a_404_on_deleting_the_old_is_success(tmp_path):
    rig = Rig(tmp_path)
    orig = FakeYT.delete_video

    def gone(self, vid):
        if vid == "old2":
            self.w.videos.pop(vid, None)
            raise RuntimeError("HttpError 404 notFound")
        orig(self, vid)

    with patch.object(FakeYT, "delete_video", gone):
        rig.run()
    _assert_replaced(rig, 2)


def test_a_quota_error_on_the_recheck_defers_never_abandons(tmp_path):
    from gomoveshift.pipeline.exceptions import YouTubeQuotaError

    rig = Rig(tmp_path)
    rig.world.on_insert = lambda w: w.fail.add("quota")
    orig = FakeYT.video_schedule

    def sched(self, vid):
        if "quota" in self.w.fail:
            raise YouTubeQuotaError("quotaExceeded")
        return orig(self, vid)

    with patch.object(FakeYT, "video_schedule", sched):
        rig.run()
    assert rig.prog(2)["state"] == "uploaded" and "new2" in rig.world.videos
    assert ("delete", "new2") not in [e[:2] for e in rig.world.events]


def test_the_new_id_reaches_the_plan_the_moment_the_insert_returns(tmp_path):
    rig = Rig(tmp_path)
    rig.world.on_insert = lambda w: (_ for _ in ()).throw(Crash("after persist"))
    with pytest.raises(Crash):
        rig.run()
    assert rig.prog(2)["new_id"] == "new2" and rig.prog(2)["state"] == "uploaded"
    rig.world.on_insert = None
    rig.run()
    _assert_replaced(rig, 2)
    assert len(rig.world.of("insert")) == 2


def test_a_slot_one_hour_ahead_after_the_old_is_deleted_is_scheduled_not_passed(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="delete")  # old deleted; the render took long
    rig.run(now=AT2 - timedelta(hours=1))  # still ahead of now: the slot is not lost
    _assert_replaced(rig, 2)
    assert not _incidents()


def test_a_slot_inside_the_small_margin_counts_as_passed(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="delete")
    rig.run(now=AT2 - timedelta(minutes=rs.REPLACE_SLOT_PASSED_MARGIN_MIN - 1))
    assert rig.prog(2)["state"] == "slot_passed" and rig.world.videos["new2"]["publish_at"] is None


def test_a_passed_slot_stays_private_and_is_retried_while_nothing_is_free(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="delete")
    rig.run(now=AT2 + timedelta(hours=1))  # truly passed and the allocator has nothing
    w = rig.world
    assert w.videos["new2"]["publish_at"] is None and w.videos["new2"]["privacy"] == "private"
    assert not [e for e in w.of("update_privacy") if e[1] == "new2"]
    assert rig.prog(2)["state"] == "slot_passed" and _incidents("slot_passed")
    assert "new2" in w.videos
    rig.rebook = lambda now: AT3 + timedelta(days=1)  # a slot frees: the next entry finishes it
    rig.run(now=AT2 + timedelta(hours=2))
    assert rig.prog(2)["state"] == "done" and not _incidents()


def test_a_truly_passed_slot_is_rebooked_at_the_next_free_slot_then_done(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="delete")
    nxt = AT2 + timedelta(days=9)
    asked = []
    rig.rebook = lambda now: (asked.append(now), nxt)[1]
    now = AT2 + timedelta(hours=1)
    rig.run(now=now)
    w = rig.world
    assert asked == [now] and w.videos["new2"]["publish_at"] == nxt
    assert nxt > now and [e[2] for e in w.of("update_privacy") if e[1] == "new2"] == [nxt]
    assert w.scheduled().count("new2") == 1 and "old2" not in w.videos
    assert rig.prog(2)["state"] == "done" and rig.prog(2)["slot"] == nxt.isoformat()
    assert [r for r in rig.booked if r["video_id"] == "new2"]  # booked through the normal path
    assert not _incidents()  # slot_rebooked was opened, then resolved at DONE
    from gomoveshift.health.store import default_data_root, default_incidents_dir, load_all

    rows, _ = load_all(default_incidents_dir(default_data_root()))
    assert [r for r in rows if r.get("code") == "slot_rebooked" and r.get("resolved_at")]


def test_a_rebook_that_raises_leaves_the_new_private_and_retried(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="delete")

    def boom(now):
        raise RuntimeError("schedule.sqlite3 locked")

    rig.rebook = boom
    rig.run(now=AT2 + timedelta(hours=1))
    assert rig.prog(2)["state"] == "slot_passed" and _incidents("slot_passed")
    assert rig.world.videos["new2"]["publish_at"] is None


def test_the_new_is_deleted_only_when_the_old_was_just_proven_to_exist(tmp_path):
    rig = Rig(tmp_path)
    rig.world.on_insert = _went_public
    rig.run()
    deletes = [e[1] for e in rig.world.of("delete")]
    assert "new2" in deletes and "old2" not in deletes  # proven public by the same call
    (tmp_path / "b").mkdir()
    rig2 = Rig(tmp_path / "b")  # abandon() called with nothing proven leaves the new alone
    rig2.write_plan(progress={"2": {"old_id": "old2", "new_id": "new2", "state": "uploaded"}})
    rig2.state.youtube_uploads = {}
    rig2.world.videos["new2"] = {"privacy": "private", "publish_at": None, "playlists": []}
    rs.abandon(FakeYT(rig2.world), lf=2, old=rig2.old2, plan_dir=rig2.dir, state=rig2.state,
               save_state=lambda s: None, why="x")  # fmt: skip
    assert "new2" in rig2.world.videos


def test_the_uploader_resolves_a_playlist_id_without_filing_anything():
    svc = MagicMock()
    yt = _uploader(svc)
    yt._get_or_create_playlist = MagicMock(return_value="PL7")
    with patch("gomoveshift.pipeline.drive_phases.youtube_uploader.resolve_playlist", return_value="Name"):  # fmt: skip
        assert yt.resolve_playlist_id(metadata={}, config=None) == "PL7"
    yt._add_video_to_playlist.assert_not_called()


# ---- fixup 3: automatic re-entry, incidents, not-found, persist, quota, thumbnails


def test_a_recheck_error_then_success_finishes_with_no_human_step(tmp_path):
    rig = _deleted_then_recheck_fails(tmp_path)
    rig.run()  # entry 1: the lookup errors, deferred (incident), nothing deleted
    assert rig.prog(2)["state"] == "uploaded" and _incidents("recheck_failed")
    rig.world.fail.discard("schedule")
    rig.run()  # entry 2: the same code path the publish retry re-enters; no operator input
    _assert_replaced(rig, 2)
    assert not _incidents()  # every replace incident cleared itself at DONE


def test_old_deleted_plus_a_schedule_failure_is_scheduled_by_the_retry(tmp_path):
    rig = Rig(tmp_path)
    orig = FakeYT.update_privacy

    def flaky(self, vid, *, privacy, publish_at=None):
        if "schedule_write" in self.w.fail:
            raise RuntimeError("videos.update 500")
        orig(self, vid, privacy=privacy, publish_at=publish_at)

    rig.world.fail.add("schedule_write")
    with patch.object(FakeYT, "update_privacy", flaky):
        rig.run()
        assert rig.prog(2)["state"] == "old_deleted" and "old2" not in rig.world.videos
        assert _incidents("schedule_failed") and rig.world.videos["new2"]["publish_at"] is None
        rig.world.fail.discard("schedule_write")
        rig.run()
    _assert_replaced(rig, 2)
    assert [e[1] for e in rig.world.of("delete")].count("old2") == 1
    assert not _incidents()


def test_the_unfinished_incident_carries_the_lf_and_clears_at_done(tmp_path):
    from gomoveshift.pipeline.drive_phases import phase_d

    rig = Rig(tmp_path)
    rig.world.fail.add("delete")
    rig.run()
    with pytest.raises(RuntimeError, match="REPLACE-UNFINISHED"):
        phase_d._assert_replace_finished(tmp_path)
    rows = _incidents("unfinished")
    assert sorted(r["lf"] for r in rows) == [2, 3]
    assert all(r["lf"] is not None for r in rows)
    rig.world.fail.discard("delete")
    rig.run()
    phase_d._assert_replace_finished(tmp_path)  # nothing left: no raise
    assert not _incidents()


def test_the_quality_gate_failure_is_not_hidden_behind_the_replace_check(tmp_path):
    from gomoveshift.pipeline.drive_phases import phase_d

    rig = Rig(tmp_path)
    rig.world.fail.add("delete")
    rig.run()
    with pytest.raises(RuntimeError) as exc:
        phase_d._assert_replace_finished(tmp_path, ValueError("title gate: duplicate"))
    assert "REPLACE-UNFINISHED" in str(exc.value) and "title gate: duplicate" in str(exc.value)


def test_a_deferred_pre_upload_check_opens_an_incident(tmp_path):
    rig = Rig(tmp_path)
    rig.world.fail.add("schedule")  # the old's live check errors before any insert
    rig.run()
    assert rig.prog(2)["state"] == "pending" and not rig.world.of("insert")
    assert [r for r in _incidents("deferred") if r["lf"] in (2, 3)]
    rig.world.fail.discard("schedule")
    rig.run()
    _assert_replaced(rig, 2)
    assert not _incidents()


def test_abandon_closes_every_incident_but_the_abandoned_one(tmp_path):
    rig = Rig(tmp_path)
    rs.record_incident(tmp_path.name, "recheck_failed", "x", lf=2)
    rs.record_incident(tmp_path.name, "unfinished", "y", lf=2)
    rs.record_incident(tmp_path.name, "deferred", "z", lf=3)
    rig.write_plan(progress={"2": {"old_id": "old2", "new_id": "new2", "state": "uploaded"}})
    rig.state.youtube_uploads = {}
    rig.world.videos["new2"] = {"privacy": "private", "publish_at": None, "playlists": []}
    rs.abandon(FakeYT(rig.world), lf=2, old=rig.old2, plan_dir=rig.dir, state=rig.state,
               save_state=lambda s: None, why="x", old_proven=True)  # fmt: skip
    codes = sorted((r["code"], r["lf"]) for r in _incidents())
    assert ("abandoned", 2) in codes and ("deferred", 3) in codes
    assert not [c for c in codes if c[1] == 2 and c[0] in ("recheck_failed", "unfinished")]


def test_a_403_whose_message_mentions_404_is_not_a_not_found():
    class Resp:
        status = 403

    class ForbiddenError(Exception):
        resp = Resp()

    assert rs._is_not_found(ForbiddenError("quota 404 forbidden")) is False
    Resp.status = 404
    assert rs._is_not_found(ForbiddenError("gone")) is True
    assert rs._is_not_found(RuntimeError("HttpError 404 notFound")) is True  # no status: regex


def test_the_real_upload_if_needed_persists_the_id_in_the_non_playlist_branch(tmp_path):
    from gomoveshift.pipeline.drive_phases.youtube_uploader import YouTubeUploader

    yt = YouTubeUploader.__new__(YouTubeUploader)
    state = make_state()
    state.youtube_uploads = {}
    seen = []
    meta = {"title": "T new", "title_decision": {"chosen_title": "T new", "reason": "fresh"}}
    with (
        patch.object(YouTubeUploader, "upload_video", return_value="vidNEW") as up,
        patch("gomoveshift.pipeline.final_text_record.final_text_context", return_value=None),
    ):
        rec = yt.upload_if_needed(
            state=state, lf_idx=2, video_path=tmp_path / "v.mp4", title="T new", description="d",
            tags=[], metadata=meta, config=None,
            persist=lambda: seen.append(dict(state.youtube_uploads)),
        )  # fmt: skip
    up.assert_called_once()
    assert rec["video_id"] == "vidNEW" and len(seen) == 1  # flushed, with the id already in
    assert seen[0][2]["video_id"] == "vidNEW"


def _budget_config(rig, *, units):
    rig.config.drive = SimpleNamespace(youtube_quota_daily_budget=units,
                                       youtube_video_insert_quota_cost=1600,
                                       youtube_playlist_insert_quota_cost=50,
                                       youtube_quota_enforce=True)  # fmt: skip


def test_the_quota_projection_counts_only_the_lfs_still_to_insert(tmp_path):
    rig = Rig(tmp_path)
    _budget_config(rig, units=1650)  # fits exactly one insert pair
    rig.write_plan(progress={"2": {"old_id": "old2", "new_id": "new2", "state": "done"}})
    rig.world.videos.pop("old2")
    rig.world.videos["new2"] = {"privacy": "private", "publish_at": AT2, "playlists": ["PL1"]}
    rig.state.youtube_uploads = {2: {"video_id": "new2", "title": "New 2"}}
    rig.run()  # LF2 done, LF3 pending: ONE insert is planned, so the one-insert budget holds
    assert rig.prog(3)["state"] == "done" and [e[1] for e in rig.world.of("insert")] == ["new3"]


def test_the_quota_projection_refuses_when_two_lfs_are_pending(tmp_path):
    from gomoveshift.pipeline.exceptions import YouTubeQuotaError

    rig = Rig(tmp_path)
    _budget_config(rig, units=1650)
    with pytest.raises(YouTubeQuotaError):
        rig.run()  # both pending: two inserts do not fit
    assert not rig.world.of("insert")


def test_an_in_flight_lf_is_finished_before_the_budget_is_checked(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="insert")  # LF2 inserted (uploaded), old still live
    _budget_config(rig, units=1650)
    rig.run()  # resume finishes LF2 (no insert), then exactly one insert (LF3) fits the budget
    _assert_replaced(rig, 2)
    assert rig.prog(3)["state"] == "done"
    assert [e[1] for e in rig.world.of("insert")] == ["new2", "new3"]  # LF2 never re-inserted


@pytest.mark.parametrize("state_name", ["uploaded", "old_deleted", "slot_passed", "pending", "abandoned"])  # fmt: skip
def test_an_lf_not_yet_scheduled_gets_no_thumbnail(state_name):
    plan = {"schema": rs.REPLACE_PLAN_SCHEMA, "kept": {},
            "replace": {"2": _old("old2", AT2)},
            "progress": {"2": {"state": state_name, "new_id": "new2", "old_id": "old2"}}}  # fmt: skip
    uploads = {2: {"video_id": "new2"}}
    assert rs.thumbnail_targets(plan, uploads) == {}
    plan["progress"]["2"]["state"] = "scheduled"
    assert list(rs.thumbnail_targets(plan, uploads)) == [2]


# --- a rebooked slot is RESERVED before any YouTube call: never two videos in one slot -----------


def _delete_both_then(rig):
    """Both old videos deleted, both news private and unscheduled: each LF now needs a rebook."""
    with pytest.raises(Crash):
        rig.run(crash_after="delete")  # LF 2's old is gone
    rig.world.fail.add("update_privacy")  # so LF 2 cannot finish before LF 3's old is deleted too
    with pytest.raises(Crash):
        rig.run(crash_after="delete")
    rig.world.fail.clear()


def test_two_slot_passed_lfs_whose_first_schedule_call_fails_end_on_different_dates(tmp_path):
    rig = Rig(tmp_path)
    _delete_both_then(rig)
    rig.rebook = rig.free_slot_after(AT3 + timedelta(days=3))
    late = AT3 + timedelta(days=1)
    rig.world.fail.add("update_privacy")
    rig.run(now=late)  # both rebook, both schedule calls fail
    assert rig.prog(2)["state"] == "old_deleted" and rig.prog(3)["state"] == "old_deleted"
    assert rig.prog(2)["slot"] != rig.prog(3)["slot"]  # reserved, so the second pick moved on
    rig.world.fail.clear()
    rig.run(now=late + timedelta(hours=1))
    at = {v: rig.world.videos[v]["publish_at"] for v in ("new2", "new3")}
    assert (
        at["new2"] is not None and at["new3"] is not None and at["new2"].date() != at["new3"].date()
    )
    assert rig.prog(2)["state"] == rig.prog(3)["state"] == "done"


def test_a_slot_another_drive_booked_during_the_backoff_forces_a_rebook(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="delete")
    rig.rebook = rig.free_slot_after(AT3 + timedelta(days=3))
    late = AT3 + timedelta(days=1)
    rig.world.fail.add("update_privacy")
    rig.run(now=late)
    d1 = rig.prog(2)["slot"]
    rig.world.fail.clear()
    rig.held[datetime.fromisoformat(d1).date()] = "other-drive"  # taken while we backed off
    rig.run(now=late + timedelta(hours=1))
    got = rig.world.videos["new2"]["publish_at"]
    assert got.isoformat() != d1 and got.date() not in {datetime.fromisoformat(d1).date()}
    assert rig.prog(2)["state"] == "done" and rig.prog(2)["slot"] == got.isoformat()


def test_a_taken_pick_is_searched_again_and_a_slot_is_never_sent_unreserved(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="delete")
    first = AT3 + timedelta(days=3)
    rig.held[first.date()] = "other-drive"
    picks = iter([first, first, first + timedelta(days=1)])  # a stale allocator keeps offering D
    rig.rebook = lambda now: next(picks)
    rig.run(now=AT3 + timedelta(days=1))
    assert rig.world.videos["new2"]["publish_at"] == first + timedelta(days=1)


def test_a_final_book_refused_is_an_incident_and_the_lf_is_not_done(tmp_path):
    rig = Rig(tmp_path)
    rig.book_error = SlotTaken("2026-12-01 is held")
    rig.run()
    assert rig.prog(2)["state"] != "done" and _incidents("book_failed")
    rig.book_error = None
    rig.run()  # retried on the next entry, finishes
    assert rig.prog(2)["state"] == "done" and not _incidents()


def test_a_store_that_refuses_the_final_strict_book_leaves_the_lf_unfinished(tmp_path):
    """The REAL store refuses the strict book (another drive holds the slot): the LF is not DONE
    and a ``book_failed`` incident stands (O7/O8)."""
    from seo_manager.scheduling.store import ScheduleStore

    from gomoveshift.config import Config
    from gomoveshift.pipeline import seo_slots
    from gomoveshift.pipeline.schedule_constants import EST

    config = Config()
    config.seo.storage_path = tmp_path / "seo"
    config.seo.storage_path.mkdir()
    ledger = config.seo.storage_path / ".content_ledger.json"
    store = ScheduleStore(ledger.parent / seo_slots.STORE_FILENAME, EST, seo_slots.cadence_for(config, "usa").regions)  # fmt: skip
    slot = AT3 + timedelta(days=5)
    store.book("theirs", 1, "usa", slot, "other")
    with pytest.raises(SlotTaken):
        seo_slots.book_in_store(ledger, config, project="mine", lf_index=2, region="usa", slot_at=slot, video_id="new2", strict=True)  # fmt: skip
    seo_slots.book_in_store(ledger, config, project="mine", lf_index=2, region="usa", slot_at=slot, video_id="new2")  # not strict: logged, never raised  # fmt: skip
    rig = Rig(tmp_path)
    try:
        seo_slots.book_in_store(ledger, config, project="mine", lf_index=2, region="usa", slot_at=slot, video_id="new2", strict=True)  # fmt: skip
    except SlotTaken as exc:
        rig.book_error = exc
    assert rig.book_error is not None
    rig.run()
    assert rig.prog(2)["state"] != "done" and _incidents("book_failed")


def test_a_rebook_inside_the_passed_margin_is_never_sent(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(Crash):
        rig.run(crash_after="delete")
    now = AT2 + timedelta(hours=1)
    rig.rebook = lambda n: n + timedelta(minutes=1)  # would publish at once
    rig.run(now=now)
    assert rig.prog(2)["state"] == "slot_passed"
    assert not [e for e in rig.world.of("update_privacy") if e[1] == "new2"]


def test_an_auditing_drive_with_an_unfinished_plan_and_receipts_is_not_retired(tmp_path):
    publish, _ = _claim_rig(tmp_path, "old_deleted")
    drive = SimpleNamespace(project="day-a", status="auditing", output_dir=str(tmp_path / "day-a"))
    retired = []
    store = SimpleNamespace(all_drives=lambda: [drive],
                            retire_published=lambda *a, **k: retired.append(a) or "complete")  # fmt: skip
    assert publish.retire_uploaded(store) == [] and not retired
    rs.set_progress(tmp_path / "day-a", 2, state="done")  # finished: now the receipts retire it
    assert publish.retire_uploaded(store) == [{"project": "day-a", "status": "complete"}]


def test_a_failed_claim_over_an_unfinished_replace_opens_the_incident_once(tmp_path, monkeypatch):
    publish, store = _claim_rig(tmp_path, "uploaded")
    store.all_drives()[0].project = "alias"  # the run dir (day-a) differs from the project name
    monkeypatch.setattr(publish, "_job_records", lambda d: [{"name": "gms-publish-alias", "created_at": 5, "result": {"state": "failed"}}])  # fmt: skip
    monkeypatch.setattr(publish, "retire_uploaded", lambda s: [])
    publish.reconcile_publishes(store, tmp_path)
    assert len(_incidents("partial_finish")) == 1
