"""The replace path's slot helpers against a REAL ledger and store (the Rig in
test_replace_durable_finish stubs them): the rebook search, the reservation, the ledger swap."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from seo_manager.scheduling.store import SlotTaken

from gomoveshift.config import Config
from gomoveshift.pipeline import replace_scheduled as rs
from gomoveshift.pipeline import schedule_constants as sc
from gomoveshift.pipeline import seo_slots
from gomoveshift.pipeline.content_ledger import ContentEntry, ContentLedger
from gomoveshift.pipeline.drive_phases import external_systems_coordinator as esc

NOW = datetime(2026, 8, 10, 9, 0, tzinfo=sc.EST)  # a Monday


def _lf(date, region, vid, project="booked", time="16:00:00"):
    return ContentEntry(date=date, time=time, content_type="longform", title="Old title",
                        project_name=project, status="scheduled", source="pipeline",
                        region=region, video_id=vid)  # fmt: skip


@pytest.fixture
def env(tmp_path):
    config = Config()
    config.seo.storage_path = tmp_path
    path = tmp_path / ".content_ledger.json"
    ContentLedger().save(path)
    return SimpleNamespace(config=config, path=path)


def _put(env, *entries):
    ledger = ContentLedger.load(env.path)
    for e in entries:
        ledger.add(e)
    ledger.save(env.path)


def _pick(env, region):
    return esc._next_free_lf_slot(env.config, SimpleNamespace(region=region))(NOW)


def test_the_rebook_search_skips_a_day_its_own_region_booked(env):
    _put(env, _lf("2026-08-16", "ireland", "irl"))  # Ireland's next Sunday
    got = _pick(env, "ireland")
    assert got.date().isoformat() == "2026-08-23"  # region None/usa would answer 08-11


def test_the_rebook_search_asks_for_the_replace_margin_and_never_today(env):
    seen = {}
    real = seo_slots.next_free_slot

    def spy(ledger, config, region, **kw):
        seen.update(kw, region=region)
        return real(ledger, config, region, **kw)

    with patch.object(seo_slots, "next_free_slot", spy):
        got = _pick(env, "usa")
    assert seen["include_today"] is False and seen["region"] == "usa"
    assert seen["min_lead"] == timedelta(hours=rs.REPLACE_MARGIN_H)
    assert got - NOW >= timedelta(hours=rs.REPLACE_MARGIN_H)


def test_reserve_moves_the_old_entry_to_the_slot_and_refuses_a_held_day(env):
    _put(env, _lf("2026-08-11", "usa", "old2", project="mine"),
         _lf("2026-08-14", "usa", "other", project="theirs"))  # fmt: skip
    slot = datetime(2026, 8, 18, 20, 0, tzinfo=timezone.utc)  # 16:00 EST
    reserve = esc._reserve_lf_slot(env.config, SimpleNamespace(region="usa"), "mine", 2, "old2")
    reserve(slot)
    reserve(slot)  # idempotent
    mine = [e for e in ContentLedger.load(env.path).entries if e.video_id == "old2"][0]
    assert (mine.date, mine.time) == ("2026-08-18", "16:00:00")
    with pytest.raises(SlotTaken):
        reserve(datetime(2026, 8, 14, 20, 0, tzinfo=timezone.utc))  # another drive's day
    kept = [e for e in ContentLedger.load(env.path).entries if e.video_id == "old2"][0]
    assert kept.date == "2026-08-18"  # a refused reserve moves nothing


def test_reserve_books_the_store_atomically_and_refuses_a_taken_slot(env):
    from seo_manager.scheduling.store import ScheduleStore

    db = env.path.parent / seo_slots.STORE_FILENAME
    cad = seo_slots.cadence_for(env.config, "usa")
    store = ScheduleStore(db, sc.EST, cad.regions)
    slot = datetime(2026, 8, 18, 20, 0, tzinfo=timezone.utc)
    store.book("theirs", 1, "usa", slot, "other")  # another drive holds the slot
    _put(env, _lf("2026-08-11", "usa", "old2", project="mine"))
    reserve = esc._reserve_lf_slot(env.config, SimpleNamespace(region="usa"), "mine", 2, "old2")
    with pytest.raises(SlotTaken):
        reserve(slot)
    free = slot + timedelta(days=7)
    reserve(free)
    assert [b for b in store.bookings() if b.video_id == "old2"]


def test_swap_ledger_video_id_moves_slot_to_est_and_retitles(env):
    _put(env, _lf("2026-08-11", "usa", "old2"))
    when = datetime(2026, 8, 18, 20, 0, tzinfo=timezone.utc)  # 16:00 EST (EDT, UTC-4)
    assert rs.swap_ledger_video_id(env.path, "old2", "new2", title="New title", slot=when)
    e = ContentLedger.load(env.path).entries[0]
    assert (e.video_id, e.title, e.date, e.time) == ("new2", "New title", "2026-08-18", "16:00:00")
    assert not rs.swap_ledger_video_id(env.path, "gone", "x")  # no entry, no change


def test_swap_without_a_slot_keeps_the_booked_date(env):
    _put(env, _lf("2026-08-11", "usa", "old2"))
    rs.swap_ledger_video_id(env.path, "old2", "new2")
    e = ContentLedger.load(env.path).entries[0]
    assert (e.video_id, e.title, e.date) == ("new2", "Old title", "2026-08-11")


def test_the_live_catalog_path_lives_in_a_leaf_the_cli_re_exports(tmp_path):
    import ast
    import inspect

    from gomoveshift.cli.handlers import pub_status
    from gomoveshift.pipeline import longform_metadata
    from gomoveshift.publishing import live_catalog

    assert pub_status.live_catalog_snapshot_path is live_catalog.live_catalog_snapshot_path
    assert pub_status.LIVE_CATALOG_SNAPSHOT == live_catalog.LIVE_CATALOG_SNAPSHOT
    assert (
        live_catalog.live_catalog_snapshot_path(tmp_path) == tmp_path / "live_catalog.latest.json"
    )
    imported = [
        n.module or ""
        for n in ast.walk(ast.parse(inspect.getsource(longform_metadata)))
        if isinstance(n, ast.ImportFrom)
    ]
    assert not [m for m in imported if "cli" in m.split(".")]
