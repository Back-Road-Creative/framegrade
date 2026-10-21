"""``gomoveshift queue reslot-overdue-manual``: move overdue TikTok slots forward.

An operator-only recovery for rows still ``awaiting_manual_publish`` more than
the grace period after their slot. Dry run by default; never posts, never
renames a pack file, never runs from the drain. Synthetic data, ``tmp_path``
pack directory only.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from freezegun import freeze_time
from test_publishing_models import _record

from gomoveshift.cli.handlers import queue_reslot
from gomoveshift.cli.handlers.queue_recover import cmd_queue
from gomoveshift.publishing.queue import enqueue, open_queue_db, transition

UTC = timezone.utc
NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)  # Sat 08:00 EDT
OLD = datetime(2026, 10, 8, 14, 0, tzinfo=UTC)  # 46h behind NOW: overdue
RECENT = datetime(2026, 10, 10, 2, 0, tzinfo=UTC)  # 10h behind: inside the grace
POOL = ["10:00", "14:00", "18:00"]  # America/New_York
PACK_NAME = "2026-10-08-1000-JohnsonCity-{id}.mp4"


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    with freeze_time(NOW):
        monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "pack"))
        monkeypatch.delenv("GMS_TIKTOK_MODE", raising=False)
        (tmp_path / "pack").mkdir()
        monkeypatch.setattr(queue_reslot, "_shorts_pool", lambda config, region: list(POOL))
        monkeypatch.setattr(queue_reslot, "_cadence_tz", lambda config: "America/New_York")
        monkeypatch.setattr(queue_reslot, "_record_for", _fake_record)
        yield


def _fake_record(row):
    rec = _record()
    tiktok = dataclasses.replace(
        rec.payloads.tiktok, title="Johnson City — roads", caption="Cap #X"
    )
    schedule = dataclasses.replace(rec.schedule, display_timezone="America/New_York")
    regions = ("Unicoi, Tennessee",) if row["short_index"] != 99 else ()
    return dataclasses.replace(
        rec,
        short_index=row["short_index"],
        sampled_regions=regions,
        payloads=dataclasses.replace(rec.payloads, tiktok=tiktok),
        schedule=schedule,
    )


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


def _parked(conn, *, short_index=3, scheduled=OLD, platform="tiktok"):
    did = enqueue(conn, "proj", short_index, platform, "h", scheduled_at=scheduled.isoformat())
    assert transition(conn, did, "queued", "awaiting_manual_publish")
    return did


def _run(config, **kw):
    args = SimpleNamespace(
        queue_action="reslot-overdue-manual", apply=kw.get("apply", False), id=kw.get("id")
    )
    return cmd_queue(args, config)


def _row(conn, did):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (did,)).fetchone())


def _pack_file(did):
    path = Path(__import__("os").environ["GMS_TIKTOK_PACK_DIR"]) / PACK_NAME.format(id=did)
    path.write_bytes(b"mp4")
    return path


def test_pack_apply_appends_reslotted_block_and_moves_row(config, conn):
    did = _parked(conn)
    mp4 = _pack_file(did)
    assert _run(config, apply=True) == 0
    # Next pool time after NOW + 20min (08:20 EDT) is 10:00 EDT = 14:00 UTC.
    assert _row(conn, did)["scheduled_at"] == "2026-10-10T14:00:00+00:00"
    text = (mp4.parent / "captions.txt").read_bytes().decode("utf-8")
    header = f"{mp4.name} -- RESLOTTED for delivery {did} (2026-10-10 10:00 EDT)"
    assert text == "\r\n".join(
        [header, "Schedule: Sat Oct 10, 10:00 AM EDT", "", "Cap #X", "", "-" * 40, ""]
    )


def test_pack_file_is_not_renamed_or_added(config, conn):
    did = _parked(conn)
    mp4 = _pack_file(did)
    _run(config, apply=True)
    assert sorted(p.name for p in mp4.parent.iterdir()) == sorted([mp4.name, "captions.txt"])


def test_dry_run_writes_nothing(config, conn, capsys):
    did = _parked(conn)
    mp4 = _pack_file(did)
    before = _row(conn, did)
    assert _run(config) == 0
    assert _row(conn, did) == before
    assert not (mp4.parent / "captions.txt").exists()
    out = capsys.readouterr().out
    assert str(did) in out and "2026-10-10T14:00:00+00:00" in out and mp4.name in out


def test_dry_run_proposes_the_same_distinct_slots_apply_books(config, conn, capsys):
    first, second = _parked(conn), _parked(conn, short_index=4)
    _pack_file(first), _pack_file(second)
    assert _run(config) == 0
    out = capsys.readouterr().out
    assert f"delivery {first} tiktok {OLD.isoformat()} -> 2026-10-10T14:00:00+00:00" in out
    assert f"delivery {second} tiktok {OLD.isoformat()} -> 2026-10-10T18:00:00+00:00" in out
    assert _run(config, apply=True) == 0
    assert _row(conn, first)["scheduled_at"] == "2026-10-10T14:00:00+00:00"
    assert _row(conn, second)["scheduled_at"] == "2026-10-10T18:00:00+00:00"


def test_inbox_row_takes_only_the_db_path(config, conn, monkeypatch):
    monkeypatch.setenv("GMS_TIKTOK_MODE", "inbox")
    did = _parked(conn)
    assert _run(config, apply=True) == 0
    assert _row(conn, did)["scheduled_at"] == "2026-10-10T14:00:00+00:00"
    assert not (Path(__import__("os").environ["GMS_TIKTOK_PACK_DIR"]) / "captions.txt").exists()


def test_pack_row_without_a_pack_file_is_refused_and_untouched(config, conn, capsys):
    did = _parked(conn)
    before = _row(conn, did)
    assert _run(config, apply=True) == 1
    assert f"REFUSED: {did}: no pack file for delivery {did}" in capsys.readouterr().err
    assert _row(conn, did) == before


def test_proposed_slot_skips_a_time_another_live_row_holds(config, conn, capsys):
    other = _parked(conn, short_index=4, scheduled=datetime(2026, 10, 10, 14, 0, tzinfo=UTC))
    did = _parked(conn)
    _pack_file(did)
    assert other != did
    assert _run(config, apply=True) == 0
    assert _row(conn, did)["scheduled_at"] == "2026-10-10T18:00:00+00:00"


def test_unresolvable_region_refuses(config, conn, capsys):
    did = _parked(conn, short_index=99)
    before = _row(conn, did)
    assert _run(config, apply=True) == 1
    assert f"REFUSED: {did}: " in capsys.readouterr().err
    assert _row(conn, did) == before


def test_not_yet_overdue_row_is_untouched(config, conn):
    did = _parked(conn, scheduled=RECENT)
    _pack_file(did)
    before = _row(conn, did)
    assert _run(config, apply=True) == 0
    assert _row(conn, did) == before
    assert not (Path(__import__("os").environ["GMS_TIKTOK_PACK_DIR"]) / "captions.txt").exists()


def test_live_foreign_lease_refuses_before_writing(config, conn, capsys):
    did = _parked(conn)
    mp4 = _pack_file(did)
    conn.execute(
        "UPDATE deliveries SET lease_owner='w', lease_expires_at='2099-01-01T00:00:00+00:00' "
        "WHERE id=?",
        (did,),
    )
    conn.commit()
    assert _run(config, apply=True) == 1
    assert not (mp4.parent / "captions.txt").exists()


def test_the_drain_never_references_the_command():
    root = Path(__file__).resolve().parents[1] / "gomoveshift"
    for rel in ("cli/handlers/dispatch_queue.py", "publishing/dispatch.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
        }
        mods = {(n.module or "") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
            a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
        }
        assert "cmd_reslot_overdue_manual" not in names, rel
        assert not any("queue_reslot" in m for m in mods), rel
