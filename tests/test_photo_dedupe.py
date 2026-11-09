"""Duplicate detection (capture identity, sha256) and the cross-run ledger."""

import sqlite3
from pathlib import Path

import pytest

from gomoveshift.pipeline.photos import dedupe as dd
from gomoveshift.pipeline.photos.capture_meta import CaptureRecord


def rec(path, serial="S1", count=1, **kw):
    return CaptureRecord(
        path=Path(path), make=" NIKON ", model="Z50_2", serial=serial, shutter_count=count, **kw
    )


def write(tmp_path, name, data=b"x"):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def test_identity_shutter_count_form():
    assert dd.capture_identity(rec("a")) == "shot:nikon|z50_2|S1|sc:1"


def test_identity_time_form():
    r = CaptureRecord(
        path=Path("a"),
        make="N",
        model="M",
        serial="S",
        datetime_original="2024:01:02 03:04:05",
        subsec="07",
    )
    assert dd.capture_identity(r) == "shot:n|m|S|t:2024:01:02 03:04:05.07"


def test_identity_none_without_serial_or_time():
    assert dd.capture_identity(CaptureRecord(path=Path("a"))) is None
    assert (
        dd.capture_identity(CaptureRecord(path=Path("a"), serial="S", datetime_original="d"))
        is None
    )
    assert dd.capture_identity(CaptureRecord(path=Path("a"), shutter_count=3)) is None


def test_identity_missing_make_model_ok():
    r = CaptureRecord(path=Path("a"), serial="S", shutter_count=0)
    assert dd.capture_identity(r) == "shot:||S|sc:0"


def test_renamed_copy_is_same_shot(tmp_path):
    a, b = write(tmp_path, "d1/a.NEF"), write(tmp_path, "d2/copy.NEF")
    res = dd.dedupe([a, b], {a: rec(a), b: rec(b)})
    assert res.unique == (a,)
    (hit,) = res.duplicates
    assert (hit.duplicate, hit.kept, hit.tier) == (b, a, dd.DupTier.SAME_SHOT)
    assert res.keys == {a: "shot:nikon|z50_2|S1|sc:1"}


def test_nef_kept_over_jpg_across_folders(tmp_path):
    j = write(tmp_path, "a/x.jpg")
    n = write(tmp_path, "bbbbbbbb/longer_name.NEF")
    res = dd.dedupe([j, n], {j: rec(j), n: rec(n)})
    assert res.unique == (n,)
    assert res.duplicates[0].duplicate == j


def test_keeper_shorter_then_lexicographic(tmp_path):
    a, b, c = write(tmp_path, "b.jpg"), write(tmp_path, "a.jpg"), write(tmp_path, "aa.jpg")
    res = dd.dedupe([c, a, b], {p: rec(p) for p in (a, b, c)})
    assert res.unique == (b,)
    assert {h.duplicate for h in res.duplicates} == {a, c}


def test_unique_preserves_input_order(tmp_path):
    ps = [write(tmp_path, f"{n}.nef") for n in ("z", "m", "a")]
    recs = {p: rec(p, count=i) for i, p in enumerate(ps)}
    assert dd.dedupe(ps, recs).unique == tuple(ps)


def test_exact_copy_without_identity(tmp_path):
    a, b, c = (
        write(tmp_path, "a.png", b"same"),
        write(tmp_path, "b.png", b"same"),
        write(tmp_path, "c.png", b"diff"),
    )
    res = dd.dedupe([a, b, c], {})
    assert res.unique == (a, c)
    (hit,) = res.duplicates
    assert (hit.duplicate, hit.kept, hit.tier) == (b, a, dd.DupTier.EXACT_COPY)
    assert hit.key.startswith("sha256:")


def test_only_identityless_files_hashed(tmp_path):
    a, b = write(tmp_path, "a.nef"), write(tmp_path, "b.png")
    seen = []

    def spy(path):
        seen.append(path)
        return "h"

    dd.dedupe([a, b], {a: rec(a)}, hasher=spy)
    assert seen == [b]


def test_ledger_seen_before_and_reprocess(tmp_path):
    a = write(tmp_path, "a.nef")
    with dd.Ledger(tmp_path / "l.db") as led:
        led.record("run1", [("shot:nikon|z50_2|S1|sc:1", str(a), None)])
        res = dd.dedupe([a], {a: rec(a)}, ledger=led)
        assert res.unique == ()
        (hit,) = res.duplicates
        assert (hit.tier, hit.kept, hit.seen_run) == (dd.DupTier.SEEN_BEFORE, None, "run1")
        again = dd.dedupe([a], {a: rec(a)}, ledger=led, reprocess=True)
        assert again.unique == (a,) and again.duplicates == ()


def test_dedupe_does_not_write_ledger(tmp_path):
    a = write(tmp_path, "a.nef")
    with dd.Ledger(tmp_path / "l.db") as led:
        dd.dedupe([a], {a: rec(a)}, ledger=led)
        assert led.seen(["shot:nikon|z50_2|S1|sc:1"]) == {}


def test_record_first_sighting_wins_and_idempotent(tmp_path):
    with dd.Ledger(tmp_path / "l.db") as led:
        led.record("r1", [("k", "p1", "h")])
        led.record("r2", [("k", "p2", None), ("k2", "p3", None)])
        led.record("r2", [("k2", "p3", None)])
        assert led.seen(["k", "k2", "nope"]) == {"k": "r1", "k2": "r2"}
        assert led.seen([]) == {}


def test_seen_handles_many_keys(tmp_path):
    with dd.Ledger(tmp_path / "l.db") as led:
        led.record("r", [(f"k{i}", "p", None) for i in range(1200)])
        assert len(led.seen([f"k{i}" for i in range(1200)])) == 1200


def test_scores_round_trip(tmp_path):
    with dd.Ledger(tmp_path / "l.db") as led:
        assert led.get_score("h", "sharp") is None
        led.put_score("h", "sharp", 0.5)
        led.put_score("h", "sharp", 0.75)
        assert led.get_score("h", "sharp") == 0.75


def test_persists_across_open(tmp_path):
    db = tmp_path / "l.db"
    led = dd.Ledger(db)
    led.record("r", [("k", "p", None)])
    led.close()
    with dd.Ledger(db) as again:
        assert again.seen(["k"]) == {"k": "r"}


def test_missing_parent_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        dd.Ledger(tmp_path / "nope" / "l.db")


def test_newer_user_version_refused(tmp_path):
    db = tmp_path / "l.db"
    con = sqlite3.connect(db)
    con.execute("PRAGMA user_version = 2")
    con.commit()
    con.close()
    with pytest.raises(ValueError, match="newer"):
        dd.Ledger(db)


def test_schema_and_pragmas(tmp_path):
    db = tmp_path / "l.db"
    dd.Ledger(db).close()
    con = sqlite3.connect(db)
    assert con.execute("PRAGMA user_version").fetchone()[0] == 1
    assert con.execute("PRAGMA journal_mode").fetchone()[0] != "wal"
    cols = [r[1] for r in con.execute("PRAGMA table_info(shots)")]
    assert cols == ["key", "run_id", "source_path", "sha256", "first_seen_utc"]
    con.close()


def test_sha256_file(tmp_path):
    import hashlib

    p = write(tmp_path, "a", b"hello" * 10)
    assert dd.sha256_file(p, chunk=7) == hashlib.sha256(b"hello" * 10).hexdigest()


def test_sources_byte_identical_after_dedupe(tmp_path):
    a, b = write(tmp_path, "a.nef", b"AAA"), write(tmp_path, "b.png", b"BBB")
    before = (a.read_bytes(), b.read_bytes(), a.stat().st_mtime_ns, b.stat().st_mtime_ns)
    dd.dedupe([a, b], {a: rec(a)})
    assert before == (a.read_bytes(), b.read_bytes(), a.stat().st_mtime_ns, b.stat().st_mtime_ns)
