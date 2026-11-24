"""Replace-scheduled finish: durable per-LF state, private insert, recheck, then delete.

The two failures that must never happen on a live channel: two scheduled videos in one slot, and
deleting a video that is already public. YouTube is a fake world here; no network."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

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

    def assign_video_to_resolved_playlist(self, vid, *, metadata, config):
        self.w.videos[vid]["playlists"].append("PL-RESOLVED")
        self.w.log("resolved", vid)
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

    def run(self, *, crash_after=None, now=NOW):
        self.world.crash_after = crash_after
        sched = SimpleNamespace(lf_slots=[SimpleNamespace(
            datetime_est=(NOW + timedelta(days=d)).isoformat(), date="x") for d in (30, 31, 32)])  # fmt: skip
        with (
            patch.object(esc, "_resolve_youtube_uploader_class", return_value=lambda: FakeYT(self.world)),
            patch.object(esc, "_book_lf_upload_in_ledger",
                         side_effect=lambda **kw: (self.world.log("book", kw["record"]["video_id"]),
                                                   self.booked.append(kw["record"]))),
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
    assert ins[2] == "private" and ins[3] is None and ins[4] == "Old old2"
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


def _live_check_errors(w):
    w.fail.add("schedule")


def _inside_margin(w):
    w.videos["old2"].update(publish_at=NOW + timedelta(hours=rs.REPLACE_MARGIN_H - 1))


@pytest.mark.parametrize("change", [_went_public, _live_check_errors, _inside_margin])
def test_a_failed_or_errored_recheck_deletes_the_new_and_never_the_old(tmp_path, change):
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


def test_a_failed_delete_stays_uploaded_and_is_retried_next_entry(tmp_path):
    rig = Rig(tmp_path)
    rig.world.fail.add("delete")
    rig.run()
    w = rig.world
    assert rig.prog(2)["state"] == "uploaded" and "old2" in w.videos
    assert not w.of("update_privacy")  # never schedules the new while the old holds the slot
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
    assert [e[1] for e in rig.world.of("resolved")] == ["new2"]
    assert not [e for e in rig.world.of("playlist") if e[1] == "new2"]


def test_title_gate_refusal_fails_safe_pending_old_untouched(tmp_path, caplog):
    bare = _old("old2", AT2)
    del bare["title_decision"]  # a legacy record: only the fresh decision ("New 2") is on hand
    rig = Rig(tmp_path, old2=bare, old3=_old("old3", AT3))
    with caplog.at_level("ERROR"):
        rig.run()
    w = rig.world
    assert not [e for e in w.of("insert") if e[1] == "new2"]
    assert "old2" in w.videos and ("delete", "old2") not in [e[:2] for e in w.events]
    assert rig.prog(2)["state"] == "pending"
    assert "REPLACE-TITLE-GATE" in caplog.text


def test_an_old_record_decision_is_passed_through_the_unmodified_gate(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    assert rig.state.youtube_uploads[2]["title_decision"]["chosen_title"] == "Old old2"
    _assert_replaced(rig, 2)


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
    rig.world.on_insert = lambda w: at_insert.append(rig.prog(2)["state"])
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
