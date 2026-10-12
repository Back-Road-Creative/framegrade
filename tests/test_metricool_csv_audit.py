"""Pre-import audit of Metricool shorts CSVs: stale ids + early scheduling."""

from __future__ import annotations

from gomoveshift.pipeline.metricool_csv_audit import audit_rows

# Parent LF goes public 2026-07-21 20:00 UTC (= 16:00 EDT).
_PUB = {"NEW1234abcd": "2026-07-21T20:00:00+00:00"}
_CURRENT = {"NEW1234abcd"}
_OLD2NEW = {"OLD9999zzzz": "NEW1234abcd"}


def _row(vid, date, time="17:00:00"):
    return {
        "Date": date,
        "Time": time,
        "First Comment Text": f"Watch: https://youtube.com/watch?v={vid}",
    }


def test_current_id_after_publish_passes():
    # 17:00 EDT = 21:00 UTC, after the 20:00 UTC publish
    f = audit_rows([_row("NEW1234abcd", "2026-07-21", "17:00:00")], _CURRENT, _OLD2NEW, _PUB)
    assert not f.failed
    assert f.ok == 1


def test_stale_id_flagged_and_remapped():
    f = audit_rows([_row("OLD9999zzzz", "2026-07-25")], _CURRENT, _OLD2NEW, _PUB)
    assert f.failed
    assert f.stale == [(2, "OLD9999zzzz", "NEW1234abcd")]


def test_early_short_flagged():
    # 14:00 EDT = 18:00 UTC, before the 20:00 UTC publish
    f = audit_rows([_row("NEW1234abcd", "2026-07-21", "14:00:00")], _CURRENT, _OLD2NEW, _PUB)
    assert f.failed
    assert len(f.early) == 1


def test_orphan_id_flagged():
    # 11-char id not in current state and not a known old id
    f = audit_rows([_row("Orphan00000", "2026-07-25")], _CURRENT, _OLD2NEW, _PUB)
    assert f.failed
    assert f.orphan == [(2, "Orphan00000")]


def test_no_state_runs_timing_only():
    # empty current_ids -> id validity not judged, but a current-id timing check
    # still can't fire without pub data; a clean row passes.
    f = audit_rows([_row("NEW1234abcd", "2026-07-25")], set(), {}, {})
    assert not f.failed
    assert f.ok == 1


def test_row_without_link_counted_separately():
    f = audit_rows(
        [{"Date": "2026-07-25", "Time": "17:00:00", "First Comment Text": "no link here"}],
        _CURRENT,
        _OLD2NEW,
        _PUB,
    )
    assert not f.failed
    assert f.no_link == 1
