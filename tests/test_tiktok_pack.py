"""Tests for ``gomoveshift.publishing.tiktok_pack``: the TikTok Studio pack.

The TikTok app is unaudited, so Direct Post is unavailable and inbox uploads
never reach the owner's inbox. The pipeline instead writes each TikTok short
(an H.264 file plus its caption and slot) to a folder on the PC, parks the
delivery ``awaiting_manual_publish``, and the owner schedules the week in
TikTok Studio. Nothing here may claim a post exists: the packer never sets
``platform_upload_id`` / ``platform_post_id`` and never reaches ``published``.

Tests use ``tmp_path`` and fakes only -- never the real pack directory.
"""

from __future__ import annotations

import dataclasses
import errno
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from freezegun import freeze_time
from test_publishing_models import _record

from gomoveshift.publishing import tiktok_pack
from gomoveshift.publishing.dispatch import PRIVACY_REQUIRED_CLASSES, TransientMediaError
from gomoveshift.publishing.queue import (
    acquire_lease,
    enqueue,
    log_response,
    open_queue_db,
    pause_platform,
    set_platform_daily_cap,
    transition,
)
from gomoveshift.publishing.tiktok_pack import (
    pack_and_park,
    pack_file_name,
    place_for,
    sweep_queued,
)

UTC = timezone.utc
NOW = datetime(2026, 10, 2, 12, 0, tzinfo=UTC)
SLOT = datetime(2026, 10, 3, 14, 0, tzinfo=UTC)  # Sat Oct 3, 10:00 AM EDT


@pytest.fixture(autouse=True)
def _clock_at_now():
    """Calls that pass no ``now`` read the wall clock, and the caption block
    says "Post now" once SLOT is under 20 minutes away. Pinned to NOW, so the
    fixed-date assertions don't start failing once SLOT is in the past
    (they once did in CI)."""
    with freeze_time(NOW):
        yield


@pytest.fixture
def conn(tmp_path):
    c = open_queue_db(tmp_path / "seo")
    try:
        yield c
    finally:
        c.close()


def _rec(
    *,
    title="Example Town — roads",
    caption="Full caption #POVDriving",
    scheduled=SLOT,
    by_platform=None,
    short_index=3,
    tz="America/New_York",
):
    rec = _record()
    tiktok = dataclasses.replace(rec.payloads.tiktok, title=title, caption=caption)
    return dataclasses.replace(
        rec,
        short_index=short_index,
        payloads=dataclasses.replace(rec.payloads, tiktok=tiktok),
        schedule=dataclasses.replace(
            rec.schedule,
            scheduled_at_utc=scheduled,
            display_timezone=tz,
            scheduled_at_utc_by_platform=by_platform or {},
        ),
    )


def _enqueue(conn, *, short_index=3, scheduled=SLOT, shadow=False, project="example-2026-08"):
    return enqueue(
        conn,
        project,
        short_index,
        "tiktok",
        "hash-a",
        shadow=shadow,
        scheduled_at=scheduled.isoformat() if scheduled else None,
    )


def _row(conn, delivery_id):
    return dict(conn.execute("SELECT * FROM deliveries WHERE id=?", (delivery_id,)).fetchone())


def _source(tmp_path, data=b"h264-bytes", name="short_003.mp4"):
    src = tmp_path / "src" / name
    src.parent.mkdir(exist_ok=True)
    src.write_bytes(data)
    return src


@pytest.fixture
def pack(tmp_path):
    return tmp_path / "TikTok"  # parent (tmp_path) exists, the pack dir itself does not


# --- the file name and the Place rule -------------------------------------


def test_file_name_uses_the_slot_in_the_display_timezone_and_the_delivery_id():
    name = pack_file_name(7, SLOT.isoformat(), _rec())
    assert name == "2026-10-03-1000-ExampleTown-7.mp4"


def test_file_name_date_follows_the_display_timezone_across_midnight():
    # 02:00Z on Oct 4 is still the evening of Oct 3 in New York.
    late = datetime(2026, 10, 4, 2, 0, tzinfo=UTC)
    name = pack_file_name(7, late.isoformat(), _rec(scheduled=late))
    assert name == "2026-10-03-2200-ExampleTown-7.mp4"


@pytest.mark.parametrize(
    ("title", "place"),
    [
        ("Example Town — roads", "ExampleTown"),
        ("Hot Springs - Arkansas", "HotSprings"),
        ("Moab: red rock", "Moab"),
        ("Zion, Utah", "Zion"),
        ("St. George — dusk", "StGeorge"),
        ("Curaçao — beach", "Curacao"),
        ("Route 66 — Oatman", "Route66"),
        ("Plain Title No Separator", "PlainTitleNoSeparator"),
    ],
)
def test_place_is_the_leading_phrase_camel_cased_to_ascii(title, place):
    assert place_for(_rec(title=title)) == place


@pytest.mark.parametrize("title", ["", " — only a tail", "!!!", ": x"])
def test_place_falls_back_to_the_short_index(title):
    assert place_for(_rec(title=title, short_index=9)) == "Short9"


def test_file_name_without_a_row_slot_uses_the_tiktok_platform_slot_first():
    tt_slot = datetime(2026, 10, 4, 23, 30, tzinfo=UTC)  # 7:30 PM EDT
    rec = _rec(by_platform={"tiktok": tt_slot})
    assert pack_file_name(4, None, rec) == "2026-10-04-1930-ExampleTown-4.mp4"


def test_file_name_without_any_platform_slot_uses_the_shared_slot():
    assert pack_file_name(4, None, _rec()) == "2026-10-03-1000-ExampleTown-4.mp4"


# --- packing one delivery -------------------------------------------------


def test_pack_copies_the_file_writes_a_crlf_caption_block_and_parks_the_row(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    final = pack_and_park(conn, delivery_id, _rec(), _source(tmp_path), pack_dir=pack)

    assert final == pack / f"2026-10-03-1000-ExampleTown-{delivery_id}.mp4"
    assert final.read_bytes() == b"h264-bytes"

    raw = (pack / "captions.txt").read_bytes()
    assert b"\n" not in raw.replace(b"\r\n", b"")  # Windows line endings throughout
    lines = raw.decode("utf-8").split("\r\n")
    assert lines[0] == final.name
    assert lines[1] == "Schedule: Sat Oct 3, 10:00 AM EDT"
    assert lines[2] == ""
    assert lines[3] == "Full caption #POVDriving"
    assert lines[4] == ""
    assert lines[5] and set(lines[5]) == {"-"}
    assert lines[6] == ""  # the block ends with a line break


def test_caption_falls_back_to_the_payload_title(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    pack_and_park(conn, delivery_id, _rec(caption=""), _source(tmp_path), pack_dir=pack)
    text = (pack / "captions.txt").read_text(encoding="utf-8")
    assert "Example Town — roads" in text


def test_a_second_block_is_appended_for_a_different_file(conn, tmp_path, pack):
    a = _enqueue(conn, short_index=3)
    b = _enqueue(conn, short_index=4)
    pack_and_park(conn, a, _rec(), _source(tmp_path), pack_dir=pack)
    other = _source(tmp_path, b"other-bytes", "short_004.mp4")
    pack_and_park(conn, b, _rec(title="Moab: red rock", short_index=4), other, pack_dir=pack)
    text = (pack / "captions.txt").read_text(encoding="utf-8")
    assert f"ExampleTown-{a}.mp4" in text
    assert f"Moab-{b}.mp4" in text


def test_a_second_run_copies_nothing_and_adds_no_second_block(conn, tmp_path, pack, monkeypatch):
    delivery_id = _enqueue(conn)
    src = _source(tmp_path)
    final = pack_and_park(conn, delivery_id, _rec(), src, pack_dir=pack)
    captions_before = (pack / "captions.txt").read_bytes()

    # The row comes back to queued (a lost race, a hand re-enqueue): same pass again.
    conn.execute("UPDATE deliveries SET status='queued' WHERE id=?", (delivery_id,))
    conn.commit()

    def no_copy(*a, **k):
        raise AssertionError("an identical file is already in the pack -- no second copy")

    monkeypatch.setattr(tiktok_pack.shutil, "copyfile", no_copy)
    again = pack_and_park(conn, delivery_id, _rec(), src, pack_dir=pack)

    assert again == final
    assert (pack / "captions.txt").read_bytes() == captions_before
    assert list(pack.glob("*.part*")) == []
    assert sorted(p.name for p in pack.iterdir()) == sorted([final.name, "captions.txt"])


def test_a_same_named_file_of_another_size_is_replaced_atomically(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    pack.mkdir()
    stale = pack / f"2026-10-03-1000-ExampleTown-{delivery_id}.mp4"
    stale.write_bytes(b"x")
    pack_and_park(conn, delivery_id, _rec(), _source(tmp_path), pack_dir=pack)
    assert stale.read_bytes() == b"h264-bytes"
    assert list(pack.glob("*.part*")) == []


def test_a_failed_copy_leaves_no_part_file_and_the_row_unparked(conn, tmp_path, pack, monkeypatch):
    delivery_id = _enqueue(conn)

    def boom(src, dst, *a, **k):
        open(dst, "wb").write(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(tiktok_pack.shutil, "copyfile", boom)
    with pytest.raises(OSError):
        pack_and_park(conn, delivery_id, _rec(), _source(tmp_path), pack_dir=pack)
    assert list(pack.glob("*.part*")) == []
    assert _row(conn, delivery_id)["status"] == "queued"


def test_a_missing_parent_is_transient_writes_nothing_and_leaves_the_row(conn, tmp_path):
    delivery_id = _enqueue(conn)
    offline = tmp_path / "no-such-drive" / "TikTok"
    before = _row(conn, delivery_id)

    with pytest.raises(TransientMediaError):
        pack_and_park(conn, delivery_id, _rec(), _source(tmp_path), pack_dir=offline)

    assert not offline.parent.exists()  # nothing was created, not even the parent
    after = _row(conn, delivery_id)
    assert after["status"] == "queued"
    assert after["updated_at"] == before["updated_at"]


def test_the_pack_dir_comes_from_the_environment(conn, tmp_path, monkeypatch):
    target = tmp_path / "from-env"
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(target))
    delivery_id = _enqueue(conn)
    final = pack_and_park(conn, delivery_id, _rec(), _source(tmp_path))
    assert final.parent == target


def test_the_default_pack_dir_is_under_the_data_home(monkeypatch, tmp_path):
    monkeypatch.delenv("GMS_TIKTOK_PACK_DIR", raising=False)
    monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "data"))
    assert tiktok_pack.resolve_pack_dir() == tmp_path / "data" / "tiktok-pack"
    monkeypatch.setenv("GMS_TIKTOK_PACK_DIR", str(tmp_path / "elsewhere"))
    assert tiktok_pack.resolve_pack_dir() == tmp_path / "elsewhere"


def test_the_default_pack_dir_is_created_with_its_parent(conn, tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_TIKTOK_PACK_DIR", raising=False)
    monkeypatch.setenv("GMS_DATA_HOME", str(tmp_path / "fresh" / "data"))
    delivery_id = _enqueue(conn)
    final = pack_and_park(conn, delivery_id, _rec(), _source(tmp_path))
    assert final.parent == tmp_path / "fresh" / "data" / "tiktok-pack"


def test_a_parked_row_claims_no_post_and_holds_no_lease(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    acquire_lease(conn, delivery_id, "me", (NOW + timedelta(minutes=15)).isoformat())
    pack_and_park(conn, delivery_id, _rec(), _source(tmp_path), pack_dir=pack)

    row = _row(conn, delivery_id)
    assert row["status"] == "awaiting_manual_publish"
    assert row["lease_owner"] is None and row["lease_expires_at"] is None
    assert row["platform_upload_id"] is None
    assert row["platform_post_id"] is None
    assert row["published_at"] is None
    assert row["next_attempt_at"] > "2100-01-01"  # the drain never reselects it
    assert row["next_attempt_reason"] == "awaiting_manual"


# --- the sweep ------------------------------------------------------------


def _loader_for(records, calls=None):
    def load(project, index):
        if calls is not None:
            calls.append((project, index))
        return records[index]

    return load


def _paths(tmp_path):
    """Short 3 is the usual ``b"h264-bytes"``; every other short gets bytes of
    its own, because two shorts with identical bytes are one short to the
    already-packed check."""

    def path_for(record):
        if record.short_index == 3:
            return _source(tmp_path)
        data = f"h264-bytes-{record.short_index}".encode()
        return _source(tmp_path, data, f"short_{record.short_index:03d}.mp4")

    return path_for


def test_the_sweep_packs_and_parks_every_due_queued_tiktok_row(conn, tmp_path, pack):
    a = _enqueue(conn, short_index=3)
    b = _enqueue(conn, short_index=4, scheduled=SLOT + timedelta(days=1))
    records = {3: _rec(), 4: _rec(title="Moab: red rock", short_index=4)}

    result = sweep_queued(
        conn, _loader_for(records), _paths(tmp_path), worker_id="w1", pack_dir=pack, now=NOW
    )

    assert (result.packed, result.skipped_leased, result.errors) == (2, 0, [])
    assert _row(conn, a)["status"] == "awaiting_manual_publish"
    assert _row(conn, b)["status"] == "awaiting_manual_publish"
    assert len(list(pack.glob("*.mp4"))) == 2


def test_rows_more_than_nine_days_out_are_untouched(conn, tmp_path, pack):
    near = _enqueue(conn, short_index=3, scheduled=NOW + timedelta(days=8, hours=23))
    far = _enqueue(conn, short_index=4, scheduled=NOW + timedelta(days=9, hours=1))
    records = {3: _rec(), 4: _rec(short_index=4)}
    calls = []

    result = sweep_queued(
        conn, _loader_for(records, calls), _paths(tmp_path), worker_id="w1", pack_dir=pack, now=NOW
    )

    assert result.packed == 1
    assert _row(conn, near)["status"] == "awaiting_manual_publish"
    assert _row(conn, far)["status"] == "queued"
    assert calls == [("example-2026-08", 3)]


def test_a_row_with_no_slot_is_due_now(conn, tmp_path, pack):
    delivery_id = _enqueue(conn, scheduled=None)
    result = sweep_queued(
        conn, _loader_for({3: _rec()}), _paths(tmp_path), worker_id="w1", pack_dir=pack, now=NOW
    )
    assert result.packed == 1
    assert _row(conn, delivery_id)["status"] == "awaiting_manual_publish"


def test_shadow_and_non_queued_rows_are_never_packed(conn, tmp_path, pack):
    shadow = _enqueue(conn, short_index=3, shadow=True)
    other = _enqueue(conn, short_index=4)
    conn.execute("UPDATE deliveries SET status='submitting' WHERE id=?", (other,))
    conn.commit()
    calls = []

    result = sweep_queued(
        conn,
        _loader_for({3: _rec(), 4: _rec(short_index=4)}, calls),
        _paths(tmp_path),
        worker_id="w1",
        pack_dir=pack,
        now=NOW,
    )

    assert result.packed == 0 and calls == []
    assert _row(conn, shadow)["status"] == "queued"
    assert _row(conn, other)["status"] == "submitting"


def test_a_row_leased_by_another_worker_is_skipped(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    # Live at the sweep's `now`; the sweep must judge it there, not by the real clock.
    assert acquire_lease(
        conn,
        delivery_id,
        "other-worker",
        (NOW + timedelta(hours=1)).isoformat(),
        now=NOW.isoformat(),
    )
    calls = []

    result = sweep_queued(
        conn,
        _loader_for({3: _rec()}, calls),
        _paths(tmp_path),
        worker_id="w1",
        pack_dir=pack,
        now=NOW,
    )

    assert (result.packed, result.skipped_leased) == (0, 1)
    assert calls == []
    row = _row(conn, delivery_id)
    assert row["status"] == "queued" and row["lease_owner"] == "other-worker"
    assert not pack.exists()


def test_nothing_is_packed_while_tiktok_is_paused(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    assert pause_platform(conn, "tiktok")
    calls = []

    result = sweep_queued(
        conn,
        _loader_for({3: _rec()}, calls),
        _paths(tmp_path),
        worker_id="w1",
        pack_dir=pack,
        now=NOW,
    )

    assert result.paused and result.packed == 0
    assert calls == []
    assert not pack.exists()
    assert _row(conn, delivery_id)["status"] == "queued"


def test_a_row_failure_is_recorded_not_raised_and_the_rest_still_pack(conn, tmp_path, pack):
    bad = _enqueue(conn, short_index=3)
    good = _enqueue(conn, short_index=4, scheduled=SLOT + timedelta(hours=1))
    records = {3: _rec(), 4: _rec(short_index=4)}
    src = _source(tmp_path)

    def paths(record):
        if record.short_index == 3:
            raise RuntimeError("ffmpeg exploded")
        return src

    result = sweep_queued(conn, _loader_for(records), paths, worker_id="w1", pack_dir=pack, now=NOW)

    assert result.packed == 1
    assert len(result.errors) == 1 and "ffmpeg exploded" in result.errors[0]
    assert _row(conn, good)["status"] == "awaiting_manual_publish"
    failed = _row(conn, bad)
    assert failed["status"] == "queued"
    assert failed["lease_owner"] is None  # released, so the drain can still take it


def test_a_row_without_a_tiktok_payload_is_an_error_and_writes_nothing(conn, tmp_path, pack):
    _enqueue(conn)
    rec = _rec()
    no_tiktok = dataclasses.replace(rec, payloads=dataclasses.replace(rec.payloads, tiktok=None))
    result = sweep_queued(
        conn, _loader_for({3: no_tiktok}), _paths(tmp_path), worker_id="w1", pack_dir=pack, now=NOW
    )
    assert result.packed == 0 and len(result.errors) == 1
    assert not pack.exists()


def test_an_offline_drive_is_one_error_and_no_row_is_touched(conn, tmp_path):
    ids = [_enqueue(conn, short_index=3), _enqueue(conn, short_index=4)]
    offline = tmp_path / "gone" / "TikTok"
    calls = []

    result = sweep_queued(
        conn,
        _loader_for({3: _rec(), 4: _rec(short_index=4)}, calls),
        _paths(tmp_path),
        worker_id="w1",
        pack_dir=offline,
        now=NOW,
    )

    assert result.packed == 0 and len(result.errors) == 1
    assert calls == []
    for delivery_id in ids:
        row = _row(conn, delivery_id)
        assert row["status"] == "queued" and row["lease_owner"] is None
    assert not offline.parent.exists()


def test_a_second_sweep_has_nothing_left_to_do(conn, tmp_path, pack):
    _enqueue(conn)
    args = (conn, _loader_for({3: _rec()}), _paths(tmp_path))
    first = sweep_queued(*args, worker_id="w1", pack_dir=pack, now=NOW)
    second = sweep_queued(*args, worker_id="w1", pack_dir=pack, now=NOW)
    assert (first.packed, second.packed) == (1, 0)
    assert (pack / "captions.txt").read_text(encoding="utf-8").count("Schedule:") == 1


# --- review fixes: one funnel with the drain ------------------------------


def _verified():
    return {"classes": {name: {"state": "verified"} for name in PRIVACY_REQUIRED_CLASSES}}


def _sweep(conn, tmp_path, pack, records=None, calls=None, **kwargs):
    return sweep_queued(
        conn,
        _loader_for(records or {3: _rec()}, calls),
        _paths(tmp_path),
        worker_id="w1",
        pack_dir=pack,
        now=kwargs.pop("now", NOW),
        **kwargs,
    )


def test_the_sweep_holds_a_row_whose_privacy_is_not_verified(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    seen = []

    def attest(row):
        seen.append(row)
        return None  # no attestation covers this delivery

    calls = []
    result = _sweep(conn, tmp_path, pack, calls=calls, privacy_attestations=attest)

    assert (result.packed, result.held_privacy) == (0, 1)
    assert [r["id"] for r in seen] == [delivery_id] and "sidecar_path" in seen[0]
    assert any("privacy hold" in h for h in result.holds)
    assert calls == [] and not pack.exists()
    row = _row(conn, delivery_id)
    assert row["status"] == "queued" and row["lease_owner"] is None and row["attempt_count"] == 0


def test_an_unreadable_privacy_attestation_holds_the_row_like_the_drain(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)

    def attest(row):
        raise ValueError("bad json")

    result = _sweep(conn, tmp_path, pack, privacy_attestations=attest)

    assert (result.packed, result.held_privacy) == (0, 1)
    assert "unreadable" in result.holds[0]
    assert _row(conn, delivery_id)["status"] == "queued" and not pack.exists()


def test_a_verified_privacy_attestation_lets_the_row_pack(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    result = _sweep(conn, tmp_path, pack, privacy_attestations=lambda row: _verified())
    assert (result.packed, result.held_privacy) == (1, 0)
    assert _row(conn, delivery_id)["status"] == "awaiting_manual_publish"


def test_the_sweep_skips_a_row_on_a_brand_blackout_date(conn, tmp_path, pack, monkeypatch):
    from gomoveshift.publishing import dispatch as dispatch_mod
    from gomoveshift.publishing.brand_policy import BrandPolicy

    monkeypatch.setattr(
        dispatch_mod, "load_brand_policy", lambda slug: BrandPolicy(blackout=(date(2026, 10, 3),))
    )
    blacked_out = _enqueue(conn, short_index=3)  # Oct 3
    fine = _enqueue(conn, short_index=4, scheduled=SLOT + timedelta(days=1))  # Oct 4
    records = {3: _rec(), 4: _rec(short_index=4)}

    result = _sweep(conn, tmp_path, pack, records=records)

    assert result.packed == 1
    assert _row(conn, blacked_out)["status"] == "queued"
    assert _row(conn, fine)["status"] == "awaiting_manual_publish"


def test_the_sweep_honours_the_quiet_window_after_a_suppression_alert(
    conn, tmp_path, pack, monkeypatch
):
    from gomoveshift.publishing import dispatch as dispatch_mod
    from gomoveshift.publishing.brand_policy import BrandPolicy

    monkeypatch.setattr(
        dispatch_mod, "load_brand_policy", lambda slug: BrandPolicy(quiet_after_alert_hours=6)
    )
    delivery_id = _enqueue(conn)
    log_response(
        conn,
        "tiktok",
        state="SUPPRESSED",
        cap=0,
        action="cap_set",
        mode="live",
        now=(NOW - timedelta(hours=1)).isoformat(),
    )

    held = _sweep(conn, tmp_path, pack)
    assert held.packed == 0 and _row(conn, delivery_id)["status"] == "queued"

    later = _sweep(conn, tmp_path, pack, now=NOW + timedelta(hours=6))
    assert later.packed == 1


def test_the_sweep_honours_a_live_daily_cap(conn, tmp_path, pack):
    first = _enqueue(conn, short_index=3)
    second = _enqueue(conn, short_index=4, scheduled=SLOT + timedelta(hours=1))
    set_platform_daily_cap(conn, "tiktok", cap=1, state="SUPPRESSED", mode="live")

    result = _sweep(conn, tmp_path, pack, records={3: _rec(), 4: _rec(short_index=4)})

    assert result.packed == 1
    assert _row(conn, first)["status"] == "awaiting_manual_publish"
    assert _row(conn, second)["status"] == "queued"


def test_the_sweep_leaves_a_row_waiting_out_its_retry_backoff(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    transition(
        conn,
        delivery_id,
        "queued",
        "queued",
        next_attempt_at=(NOW + timedelta(hours=3)).isoformat(),
        next_attempt_reason="retry_backoff",
    )
    assert _sweep(conn, tmp_path, pack).packed == 0
    assert _row(conn, delivery_id)["status"] == "queued"


# --- review fixes: leases and stale ids ------------------------------------


def test_a_failed_pack_releases_the_lease_whatever_the_row_became(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    src = _source(tmp_path)

    def paths(record):
        # The row moved on while this worker held the lease: not "queued" any more.
        assert transition(conn, delivery_id, "queued", "submitting")
        raise RuntimeError("boom")

    result = sweep_queued(
        conn, _loader_for({3: _rec()}), paths, worker_id="w1", pack_dir=pack, now=NOW
    )

    assert len(result.errors) == 1
    row = _row(conn, delivery_id)
    assert row["status"] == "submitting"
    assert row["lease_owner"] is None and row["lease_expires_at"] is None
    assert src.exists()


def test_parking_a_row_clears_the_stale_inbox_upload_id(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    assert transition(conn, delivery_id, "queued", "submitting", platform_upload_id="v_inbox~old")

    pack_and_park(conn, delivery_id, _rec(), _source(tmp_path), pack_dir=pack)

    row = _row(conn, delivery_id)
    assert row["status"] == "awaiting_manual_publish"
    assert row["platform_upload_id"] is None  # reconcile must never poll it


# --- review fixes: files the owner already placed --------------------------


def test_a_byte_identical_file_under_another_name_is_not_packed_again(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    pack.mkdir()
    by_hand = pack / "2026-10-03-Other.mp4"
    by_hand.write_bytes(b"h264-bytes")

    result = _sweep(conn, tmp_path, pack)

    assert (result.packed, result.already_packed, result.errors) == (0, 1, [])
    assert [p.name for p in pack.iterdir()] == [by_hand.name]  # no copy, no captions.txt
    row = _row(conn, delivery_id)
    assert row["status"] == "awaiting_manual_publish"
    assert row["platform_upload_id"] is None and row["lease_owner"] is None


def test_an_already_packed_short_adds_no_caption_block(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    pack.mkdir()
    (pack / "hand-copied.mp4").write_bytes(b"h264-bytes")
    (pack / "captions.txt").write_bytes(b"older block\r\n")

    pack_and_park(conn, delivery_id, _rec(), _source(tmp_path), pack_dir=pack)

    assert (pack / "captions.txt").read_bytes() == b"older block\r\n"
    assert sorted(p.name for p in pack.iterdir()) == ["captions.txt", "hand-copied.mp4"]


def test_a_same_size_file_with_other_bytes_does_not_count_as_packed(conn, tmp_path, pack):
    delivery_id = _enqueue(conn)
    pack.mkdir()
    other = pack / "someone-elses.mp4"
    other.write_bytes(b"h264-bYtes")  # same size as the source, different content

    result = _sweep(conn, tmp_path, pack)

    assert (result.packed, result.already_packed) == (1, 0)
    assert other.read_bytes() == b"h264-bYtes"
    assert (pack / f"2026-10-03-1000-ExampleTown-{delivery_id}.mp4").is_file()


def test_the_content_check_hashes_only_files_of_the_sources_size(conn, tmp_path, pack, monkeypatch):
    _enqueue(conn)
    pack.mkdir()
    (pack / "big.mp4").write_bytes(b"x" * 500)
    hashed = []
    real = tiktok_pack._sha256
    monkeypatch.setattr(tiktok_pack, "_sha256", lambda p: hashed.append(p.name) or real(p))

    _sweep(conn, tmp_path, pack)

    assert "big.mp4" not in hashed


# --- review fixes: file names, the schedule line, the drive -----------------


def test_each_copy_writes_its_own_part_file(tmp_path, monkeypatch):
    src = _source(tmp_path)
    final = tmp_path / "x.mp4"
    parts = []
    real = tiktok_pack.shutil.copyfile
    monkeypatch.setattr(
        tiktok_pack.shutil,
        "copyfile",
        lambda s, d, *a, **k: parts.append(Path(d).name) or real(s, d),
    )

    tiktok_pack._copy_into_pack(src, final)
    final.unlink()
    tiktok_pack._copy_into_pack(src, final)

    assert len(set(parts)) == 2
    assert all(re.fullmatch(r"x\.mp4\.part\.\d+-[0-9a-f]{32}", name) for name in parts)


@pytest.mark.parametrize(
    ("offset", "post_now"),
    [
        (timedelta(hours=-2), True),  # the slot has passed
        (timedelta(minutes=10), True),  # inside the lead-time margin
        (timedelta(minutes=25), False),
    ],
)
def test_a_slot_too_close_to_schedule_says_post_now(conn, tmp_path, pack, offset, post_now):
    slot = NOW + offset
    delivery_id = _enqueue(conn, scheduled=slot)
    pack_and_park(
        conn, delivery_id, _rec(scheduled=slot), _source(tmp_path), pack_dir=pack, now=NOW
    )
    text = (pack / "captions.txt").read_text(encoding="utf-8")
    assert ("Schedule: Post now" in text) is post_now
    assert ("Schedule: Fri Oct 2" in text) is (not post_now)


def test_a_mount_point_directory_with_nothing_mounted_is_offline(tmp_path, monkeypatch):
    (tmp_path / "d").mkdir()
    pack_dir = tmp_path / "d" / "TikTok"
    monkeypatch.setattr(tiktok_pack, "_DRIVE_MOUNT_ROOT", tmp_path)
    monkeypatch.setattr(tiktok_pack.os.path, "ismount", lambda p: False)
    with pytest.raises(TransientMediaError, match="not mounted"):
        tiktok_pack._ensure_pack_dir(pack_dir)
    assert not pack_dir.exists()  # nothing landed on the Linux root disk

    monkeypatch.setattr(tiktok_pack.os.path, "ismount", lambda p: True)
    tiktok_pack._ensure_pack_dir(pack_dir)
    assert pack_dir.is_dir()


@pytest.mark.parametrize("code", [errno.EIO, errno.ENODEV, errno.EACCES])
def test_a_dead_mount_error_is_transient_and_leaves_the_row(
    conn, tmp_path, pack, monkeypatch, code
):
    delivery_id = _enqueue(conn)

    def dead(src, dst, *a, **k):
        raise OSError(code, "stale mount")

    monkeypatch.setattr(tiktok_pack.shutil, "copyfile", dead)
    with pytest.raises(TransientMediaError):
        pack_and_park(conn, delivery_id, _rec(), _source(tmp_path), pack_dir=pack)
    assert _row(conn, delivery_id)["status"] == "queued"
    assert list(pack.glob("*.part*")) == []


def test_a_dead_mount_on_mkdir_is_transient(conn, tmp_path, pack, monkeypatch):
    delivery_id = _enqueue(conn)
    src = _source(tmp_path)  # made before mkdir is broken

    def dead(self, *a, **k):
        raise OSError(errno.ENODEV, "stale mount")

    monkeypatch.setattr(Path, "mkdir", dead)
    with pytest.raises(TransientMediaError):
        pack_and_park(conn, delivery_id, _rec(), src, pack_dir=pack)


def test_a_full_disk_is_still_an_ordinary_error(conn, tmp_path, pack, monkeypatch):
    delivery_id = _enqueue(conn)

    def full(src, dst, *a, **k):
        raise OSError(errno.ENOSPC, "no space")

    monkeypatch.setattr(tiktok_pack.shutil, "copyfile", full)
    with pytest.raises(OSError) as info:
        pack_and_park(conn, delivery_id, _rec(), _source(tmp_path), pack_dir=pack)
    assert not isinstance(info.value, TransientMediaError)


def test_the_suite_pins_the_pack_directory_inside_tmp(tmp_path):
    assert tiktok_pack.resolve_pack_dir().parent == tmp_path


# --- re-review fixes: a re-enqueued short whose file is already packed ------


def test_a_reenqueued_short_gets_an_updated_block_under_the_packed_file(conn, tmp_path, pack):
    """A plan change cancels the old row and enqueues a new one with the same
    bytes. The packed file is reused (a second copy could be scheduled twice),
    but the new slot and caption must reach captions.txt under that file."""
    old = _enqueue(conn)
    first = pack_and_park(conn, old, _rec(), _source(tmp_path), pack_dir=pack)
    later = SLOT + timedelta(days=1)
    new = _enqueue(conn, project="example-2026-08-replan", scheduled=later)

    final = pack_and_park(
        conn,
        new,
        _rec(scheduled=later, caption="New caption #Ireland"),
        _source(tmp_path),
        pack_dir=pack,
    )

    assert final == first
    assert sorted(p.name for p in pack.iterdir()) == sorted(["captions.txt", first.name])
    text = (pack / "captions.txt").read_text(encoding="utf-8")
    assert f"{first.name} -- UPDATED for delivery {new}" in text
    assert "Schedule: Sun Oct 4, 10:00 AM EDT" in text
    assert "New caption #Ireland" in text
    assert _row(conn, new)["status"] == "awaiting_manual_publish"


def test_the_updated_block_is_written_once(conn, tmp_path, pack):
    old = _enqueue(conn)
    pack_and_park(conn, old, _rec(), _source(tmp_path), pack_dir=pack)
    new = _enqueue(conn, project="example-2026-08-replan")
    tiktok_pack.pack_row(conn, new, _rec(), _source(tmp_path), pack_dir=pack)
    once = (pack / "captions.txt").read_bytes()

    tiktok_pack.pack_row(conn, new, _rec(), _source(tmp_path), pack_dir=pack)

    assert (pack / "captions.txt").read_bytes() == once
