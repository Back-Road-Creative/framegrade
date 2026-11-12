# ruff: noqa: F811  (the conn fixture is imported from the missing_remote test module)
"""Re-slotting overdue YouTube Shorts onto future Waypost slots.

``queue reslot-overdue-youtube`` moves a ``queued`` YouTube row whose slot has passed
(and that holds no upload id) to the region's next free Waypost shorts time, in the
queue row AND in the sidecar (the sidecar's ``scheduled_at_utc`` is what the upload
sends as ``publishAt``). ``queue recover-terminal`` does the same, through the same
allocator, for a FinalTextReview ``failed_permanent`` row whose slot has passed.
A passed slot still refuses every other row kind. Synthetic data, ``tmp_path`` only,
no network: the real queue shapes (rows 142, 158, 178) are rebuilt offline.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from freezegun import freeze_time
from test_cli_enqueue_plan import _record as _plan_record
from test_cli_enqueue_plan import _sidecar
from test_youtube_missing_remote_recover import (
    FakeYouTube,
    _attested,
    _gone_row,
    _snapshot,
)

from gomoveshift.cli.handlers import queue_reslot
from gomoveshift.cli.handlers.queue_recover import RecoveryDeps, cmd_queue, recover_terminal
from gomoveshift.publishing.models import PlatformPayloads, ScheduleWindow, YouTubePayload
from gomoveshift.publishing.plan_io import load_publish_plan
from gomoveshift.publishing.queue import RecoveryRefusedError, enqueue, open_queue_db

UTC = timezone.utc
NOW = datetime(2026, 11, 12, 16, 30, tzinfo=UTC)  # Thu 11:30 EST
NOW_ISO = NOW.isoformat()
POOL = ["10:00", "16:00"]  # America/New_York: 15:00Z and 21:00Z in November
ALL = ("youtube", "instagram", "facebook", "tiktok")
PAST = datetime(2026, 11, 12, 15, 0, tzinfo=UTC)  # row 142's shape: today's 10:00 slot
WAY_PAST = datetime(2026, 11, 11, 19, 0, tzinfo=UTC)  # row 178's shape
FUTURE = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(queue_reslot, "_shorts_pool", lambda config, region: list(POOL))
    monkeypatch.setattr(queue_reslot, "_cadence_tz", lambda config: "America/New_York")
    with freeze_time(NOW):
        yield


@pytest.fixture
def config(tmp_path):
    return SimpleNamespace(seo=SimpleNamespace(storage_path=str(tmp_path / "seo")))


@pytest.fixture
def conn(config):
    c = open_queue_db(config.seo.storage_path)
    try:
        yield c
    finally:
        c.close()


def _short(tmp_path, short_index, slot, regions=("Colleton County, South Carolina",)):
    out = tmp_path / f"out{short_index}"
    out.mkdir(exist_ok=True)
    rec = dataclasses.replace(
        _plan_record(short_index=short_index, selected=ALL),
        sampled_regions=regions,
        schedule=ScheduleWindow(slot, "America/New_York"),
        payloads=PlatformPayloads(youtube=YouTubePayload(f"Title {short_index}", "d")),
    )
    return _sidecar(out, rec)


def _row(conn, tmp_path, short_index, slot, *, status="queued", error_class=None, **kw):
    sidecar = kw.pop("sidecar", None) or _short(tmp_path, short_index, slot)
    did = enqueue(
        conn, "proj", short_index, "youtube", f"h{short_index}",
        scheduled_at=slot.isoformat(), sidecar_path=str(sidecar),
    )  # fmt: skip
    conn.execute(
        "UPDATE deliveries SET status=?, error_class=?, platform_upload_id=? WHERE id=?",
        (status, error_class, kw.get("upload_id"), did),
    )
    conn.commit()
    return did


def _get(conn, did):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone())


def _plan_slot(sidecar, short_index):
    plan = load_publish_plan(sidecar)
    return next(r for r in plan.shorts if r.short_index == short_index).schedule


def _run(config, **kw):
    args = SimpleNamespace(
        queue_action="reslot-overdue-youtube", apply=kw.get("apply", False), id=kw.get("id")
    )
    return cmd_queue(args, config)


# --- reslot-overdue-youtube -------------------------------------------------


def test_apply_moves_row_and_sidecar_to_the_next_free_slot(config, conn, tmp_path):
    did = _row(conn, tmp_path, 7, PAST)
    sidecar = tmp_path / "out7" / "proj_publish_plan.json"
    assert _run(config, apply=True) == 0
    new = datetime(2026, 11, 12, 21, 0, tzinfo=UTC)
    assert datetime.fromisoformat(_get(conn, did)["scheduled_at"]) == new
    assert _plan_slot(sidecar, 7).scheduled_at_utc == new
    assert new > NOW  # publishAt is a future time, never a past one


def test_other_platforms_keep_the_slot_they_had(config, conn, tmp_path):
    _row(conn, tmp_path, 7, PAST)
    sidecar = tmp_path / "out7" / "proj_publish_plan.json"
    _run(config, apply=True)
    by_platform = _plan_slot(sidecar, 7).scheduled_at_utc_by_platform
    assert {p: by_platform[p] for p in ("instagram", "facebook", "tiktok")} == {
        p: PAST for p in ("instagram", "facebook", "tiktok")
    }
    assert "youtube" not in by_platform


def test_dry_run_writes_nothing_and_names_the_slot(config, conn, tmp_path, capsys):
    did = _row(conn, tmp_path, 7, PAST)
    sidecar = tmp_path / "out7" / "proj_publish_plan.json"
    before_row, before_file = _get(conn, did), sidecar.read_bytes()
    assert _run(config) == 0
    assert _get(conn, did) == before_row and sidecar.read_bytes() == before_file
    out = capsys.readouterr().out
    assert f"delivery {did} youtube {PAST.isoformat()} -> 2026-11-12T21:00:00+00:00" in out
    assert "dry run" in out


def test_two_overdue_rows_get_distinct_slots_in_dry_run_and_apply(config, conn, tmp_path, capsys):
    a, b = _row(conn, tmp_path, 6, WAY_PAST), _row(conn, tmp_path, 7, PAST)
    assert _run(config) == 0
    out = capsys.readouterr().out
    assert f"delivery {a} youtube {WAY_PAST.isoformat()} -> 2026-11-12T21:00:00+00:00" in out
    assert f"delivery {b} youtube {PAST.isoformat()} -> 2026-11-13T15:00:00+00:00" in out
    assert _run(config, apply=True) == 0
    assert _get(conn, a)["scheduled_at"] == "2026-11-12T21:00:00+00:00"
    assert _get(conn, b)["scheduled_at"] == "2026-11-13T15:00:00+00:00"


def test_slot_held_by_another_short_is_skipped_whatever_its_status(config, conn, tmp_path):
    held = datetime(2026, 11, 12, 21, 0, tzinfo=UTC)
    _row(conn, tmp_path, 3, held, status="published")  # a scheduled-private upload holds it
    did = _row(conn, tmp_path, 7, PAST)
    assert _run(config, apply=True) == 0
    assert _get(conn, did)["scheduled_at"] == "2026-11-13T15:00:00+00:00"


def test_slots_of_dead_rows_are_free(config, conn, tmp_path):
    free = datetime(2026, 11, 12, 21, 0, tzinfo=UTC)
    _row(conn, tmp_path, 3, free, status="failed_permanent", error_class="x")
    _row(conn, tmp_path, 4, free, status="cancelled")
    did = _row(conn, tmp_path, 7, PAST)
    _run(config, apply=True)
    assert _get(conn, did)["scheduled_at"] == free.isoformat()


def test_a_slot_inside_the_upload_lead_is_not_offered(config, conn, tmp_path):
    with freeze_time(datetime(2026, 11, 12, 20, 30, tzinfo=UTC)):  # 21:00Z is 30 min away
        did = _row(conn, tmp_path, 7, PAST)
        _run(config, apply=True)
    assert _get(conn, did)["scheduled_at"] == "2026-11-13T15:00:00+00:00"


def test_only_overdue_queued_rows_without_an_upload_id_move(config, conn, tmp_path):
    future = _row(conn, tmp_path, 1, FUTURE)
    uploaded = _row(conn, tmp_path, 2, PAST, upload_id="LIVEvid0001")
    dead = _row(conn, tmp_path, 3, PAST, status="failed_permanent", error_class="x")
    before = [_get(conn, d) for d in (future, uploaded, dead)]
    assert _run(config, apply=True) == 0
    assert [_get(conn, d) for d in (future, uploaded, dead)] == before


def test_unreadable_sidecar_is_refused_and_row_untouched(config, conn, tmp_path, capsys):
    did = _row(conn, tmp_path, 7, PAST, sidecar=tmp_path / "gone" / "plan.json")
    before = _get(conn, did)
    assert _run(config, apply=True) == 1
    assert f"REFUSED: {did}: " in capsys.readouterr().err
    assert _get(conn, did) == before


def test_unresolvable_region_is_refused_and_nothing_is_written(config, conn, tmp_path, capsys):
    sidecar = _short(tmp_path, 7, PAST, regions=())
    did = _row(conn, tmp_path, 7, PAST, sidecar=sidecar)
    before, raw = _get(conn, did), sidecar.read_bytes()
    assert _run(config, apply=True) == 1
    assert "Waypost region" in capsys.readouterr().err
    assert _get(conn, did) == before and sidecar.read_bytes() == raw


def test_live_foreign_lease_refuses_before_writing(config, conn, tmp_path):
    did = _row(conn, tmp_path, 7, PAST)
    sidecar = tmp_path / "out7" / "proj_publish_plan.json"
    raw = sidecar.read_bytes()
    conn.execute(
        "UPDATE deliveries SET lease_owner='w', lease_expires_at='2099-01-01T00:00:00+00:00' "
        "WHERE id=?",
        (did,),
    )
    conn.commit()
    assert _run(config, apply=True) == 1
    assert sidecar.read_bytes() == raw


def test_nothing_overdue_says_so(config, conn, capsys):
    assert _run(config) == 0
    assert "no overdue" in capsys.readouterr().out


# --- recover-terminal -------------------------------------------------------

DETAIL = "LF 2: final-text review refused (verdict:refused): missing_field:approval_hash"


def _failed(
    conn, tmp_path, short_index, slot, *, error_class="MaxAttemptsExceeded(FinalTextReviewError)"
):
    did = _row(
        conn, tmp_path, short_index, slot, status="failed_permanent", error_class=error_class
    )
    return did


def _deps(config, conn, tmp_path, *, with_reslot=True, reader=None):
    media = tmp_path / "s.mp4"
    media.write_bytes(b"x")

    def loader(project, idx):
        side = tmp_path / f"out{idx}" / "proj_publish_plan.json"
        return next(r for r in load_publish_plan(side).shorts if r.short_index == idx)

    deps = RecoveryDeps(
        graph=reader or FakeYouTube(),
        record_loader=loader,
        privacy_loader=lambda row: _attested(),
        rendition_path=lambda rec: media,
        review_check=lambda rec: None,
    )
    if with_reslot:
        deps = dataclasses.replace(
            deps,
            slot_allocator=queue_reslot.youtube_slot_allocator(conn, config),
            slot_writer=queue_reslot.write_youtube_slot,
        )
    return deps


def test_review_refused_row_with_a_passed_slot_gets_the_next_free_slot(config, conn, tmp_path):
    did = _failed(conn, tmp_path, 4, WAY_PAST)  # row 178's shape
    new_id = recover_terminal(conn, did, "operator", _deps(config, conn, tmp_path), now=NOW_ISO)
    new = datetime(2026, 11, 12, 21, 0, tzinfo=UTC)
    row = _get(conn, new_id)
    assert row["recovers_delivery_id"] == did and row["status"] == "queued"
    assert datetime.fromisoformat(row["scheduled_at"]) == new
    sidecar = tmp_path / "out4" / "proj_publish_plan.json"
    assert _plan_slot(sidecar, 4).scheduled_at_utc == new
    evidence = json.loads(row["recovery_evidence"])
    assert evidence["reslot"] == {"from": WAY_PAST.isoformat(), "to": new.isoformat()}


def test_dry_run_reslots_nothing_on_disk(config, conn, tmp_path):
    did = _failed(conn, tmp_path, 4, WAY_PAST)
    sidecar = tmp_path / "out4" / "proj_publish_plan.json"
    before_file, before_db = sidecar.read_bytes(), _snapshot(conn)
    assert (
        recover_terminal(
            conn, did, "operator", _deps(config, conn, tmp_path), now=NOW_ISO, apply=False
        )
        is None
    )
    assert sidecar.read_bytes() == before_file and _snapshot(conn) == before_db


def test_recovered_slot_is_booked_so_the_next_recovery_takes_another(config, conn, tmp_path):
    a, b = _failed(conn, tmp_path, 4, WAY_PAST), _failed(conn, tmp_path, 5, WAY_PAST)
    deps = _deps(config, conn, tmp_path)
    ra = recover_terminal(conn, a, "operator", deps, now=NOW_ISO)
    rb = recover_terminal(conn, b, "operator", deps, now=NOW_ISO)
    assert _get(conn, ra)["scheduled_at"] == "2026-11-12T21:00:00+00:00"
    assert _get(conn, rb)["scheduled_at"] == "2026-11-13T15:00:00+00:00"


def test_review_refused_row_with_a_future_slot_keeps_it(config, conn, tmp_path):
    did = _failed(conn, tmp_path, 4, FUTURE)
    new_id = recover_terminal(conn, did, "operator", _deps(config, conn, tmp_path), now=NOW_ISO)
    assert _plan_slot(tmp_path / "out4" / "proj_publish_plan.json", 4).scheduled_at_utc == FUTURE
    assert "reslot" not in json.loads(_get(conn, new_id)["recovery_evidence"])


def test_without_an_allocator_a_passed_slot_still_refuses(config, conn, tmp_path):
    did = _failed(conn, tmp_path, 4, WAY_PAST)
    deps = _deps(config, conn, tmp_path, with_reslot=False)
    before = _snapshot(conn)
    with pytest.raises(RecoveryRefusedError, match="has passed"):
        recover_terminal(conn, did, "operator", deps, now=NOW_ISO)
    assert _snapshot(conn) == before


def test_a_missing_remote_row_with_a_passed_slot_still_refuses(config, conn, tmp_path):
    _short(tmp_path, 4, WAY_PAST)
    did = _gone_row(conn)
    conn.execute(
        "UPDATE deliveries SET sidecar_path=? WHERE id=?",
        (str(tmp_path / "out4" / "proj_publish_plan.json"), did),
    )
    conn.commit()
    sidecar = tmp_path / "out4" / "proj_publish_plan.json"
    raw, before = sidecar.read_bytes(), _snapshot(conn)
    with pytest.raises(RecoveryRefusedError, match="has passed"):
        recover_terminal(conn, did, "operator", _deps(config, conn, tmp_path), now=NOW_ISO)
    assert sidecar.read_bytes() == raw and _snapshot(conn) == before


@pytest.mark.parametrize(
    ("what", "match"),
    [("dupe", "already carries"), ("privacy", "privacy hold"), ("review", "still refuses")],
)
def test_other_gates_still_stop_a_reslotted_recovery(config, conn, tmp_path, what, match):
    did = _failed(conn, tmp_path, 4, WAY_PAST)
    deps = _deps(config, conn, tmp_path)
    if what == "dupe":
        deps = dataclasses.replace(
            deps, graph=FakeYouTube(uploads=[{"id": "d1", "title": "Title 4"}])
        )
    if what == "privacy":
        deps = dataclasses.replace(deps, privacy_loader=lambda row: _attested("inconclusive"))
    if what == "review":
        deps = dataclasses.replace(deps, review_check=lambda rec: "name_unbacked:x")
    sidecar = tmp_path / "out4" / "proj_publish_plan.json"
    raw, before = sidecar.read_bytes(), _snapshot(conn)
    with pytest.raises(RecoveryRefusedError, match=match):
        recover_terminal(conn, did, "operator", deps, now=NOW_ISO)
    assert sidecar.read_bytes() == raw and _snapshot(conn) == before


def test_a_row_holding_an_upload_id_is_never_reslotted(config, conn, tmp_path):
    did = _row(
        conn, tmp_path, 4, WAY_PAST, status="failed_permanent",
        error_class="FinalTextReviewError", upload_id="LIVEvid0001",
    )  # fmt: skip
    with pytest.raises(RecoveryRefusedError, match="holds an upload id"):
        recover_terminal(conn, did, "operator", _deps(config, conn, tmp_path), now=NOW_ISO)
