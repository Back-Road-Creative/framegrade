"""Tests for ``gomoveshift catalog-inventory`` (catalog/inventory.py). Synthetic data only."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.catalog.inventory import (
    CSV_COLUMNS,
    build_inventory,
    load_deliveries,
    load_ledger_entries,
    write_slices,
)
from gomoveshift.cli import _COMMAND_MAP
from gomoveshift.cli.parser import _build_parser


def _live(**videos):
    return {"videos": videos}


def _vid(**kw):
    base = {
        "title": "T",
        "privacy": "public",
        "publish_at": None,
        "published_at": "2025-01-01T00:00:00Z",
        "duration_s": 600,
        "recording_date": None,
    }
    base.update(kw)
    return base


def _empty(**over):
    kw = dict(
        lf_upload_files=[],
        tracking_files=[],
        deliveries_rows=[],
        waypost_event_lines=[],
        ledger_entries=[],
    )
    kw.update(over)
    return kw


def _by_id(rows):
    return {r["video_id"]: r for r in rows}


def test_tier_none_without_any_evidence():
    rows = build_inventory(_live(vidA=_vid()), **_empty())
    assert rows[0]["evidence_tier"] == "none"
    assert rows[0]["project"] is None
    assert rows[0]["evidence"] == {}


def test_tier_project_from_ledger_only():
    rows = build_inventory(
        _live(vidA=_vid()),
        **_empty(
            ledger_entries=[{"video_id": "vidA", "project_name": "projX", "content_type": "lf"}]
        ),
    )
    assert rows[0]["evidence_tier"] == "project"
    assert rows[0]["project"] == "projX"


def test_tier_segment_from_lf_upload_and_title_match(tmp_path):
    d = tmp_path / "projLF"
    d.mkdir()
    f = d / "youtube_uploads.json"
    f.write_text(
        json.dumps(
            {"1": {"video_id": "vidA", "title": "T"}, "2": {"video_id": "vidB", "title": "old"}}
        )
    )
    rows = _by_id(
        build_inventory(
            _live(vidA=_vid(), vidB=_vid()),
            **_empty(lf_upload_files=[f]),
        )
    )
    assert rows["vidA"]["evidence_tier"] == "segment"
    assert rows["vidA"]["project"] == "projLF"
    assert rows["vidA"]["evidence"]["lf_upload"]["lf"] == "1"
    assert rows["vidB"]["evidence"]["lf_upload"]["uploaded_title"] == "old"


def test_tier_segment_from_tracking_and_waypost(tmp_path):
    t = tmp_path / "tr.json"
    t.write_text(
        json.dumps(
            {
                "project": "projT",
                "shorts": [
                    {
                        "index": 3,
                        "file": "s.mp4",
                        "platform_ids": {"youtube": "vidA"},
                        "observation_log": {"youtube": [{}, {}]},
                    },
                    {"index": 4, "platform_ids": {}},
                ],
                "long_forms": [],
            }
        )
    )
    lines = [
        json.dumps({"timestamp": "t1", "data": {"video_id": "vidA"}}) + "\n",
        'unrelated "yt_sync_x" absent\n'.replace('"yt_sync_x"', "zzz"),
        '{"type": "yt_sync_ok", "timestamp": "t2", "data": {"video_id": "vidA"}}\n',
    ]
    lines[0] = '{"type": "yt_sync_a", "timestamp": "t1", "data": {"video_id": "vidA"}}\n'
    rows = build_inventory(
        _live(vidA=_vid(duration_s=30)), **_empty(tracking_files=[t], waypost_event_lines=lines)
    )
    r = rows[0]
    assert r["evidence_tier"] == "segment"
    assert r["evidence"]["tracking"]["observations"] == 2
    assert r["evidence"]["tracking"]["index"] == 3
    assert r["evidence"]["waypost_events"] == {"n": 2, "last": "t2"}


def test_delivery_rows_are_segment_evidence():
    rows = build_inventory(
        _live(vidA=_vid(duration_s=20)),
        **_empty(deliveries_rows=[("vidA", "projD", 2, "plan.json")]),
    )
    assert rows[0]["evidence_tier"] == "segment"
    assert rows[0]["evidence"]["delivery"]["short_index"] == 2


@pytest.mark.parametrize("dur,kind", [(180, "short"), (181, "long"), (None, "short"), (0, "short")])
def test_short_long_boundary(dur, kind):
    rows = build_inventory(_live(vidA=_vid(duration_s=dur)), **_empty())
    assert rows[0]["kind"] == kind


def test_slices_route_public_and_nonpublic_shorts(tmp_path):
    live = _live(
        lf=_vid(duration_s=900, published_at="2025-02-01"),
        pubS=_vid(duration_s=30),
        privS=_vid(duration_s=30, privacy="private"),
        unlS=_vid(duration_s=30, privacy="unlisted", published_at=None),
    )
    rows = build_inventory(live, **_empty())
    counts = write_slices(rows, tmp_path / "out")
    assert counts == {"longform.csv": 1, "shorts-nonpublic.csv": 2, "shorts-public.csv": 1}
    with open(tmp_path / "out" / "shorts-nonpublic.csv", newline="") as f:
        got = list(csv.DictReader(f))
    assert [r["video_id"] for r in got] == ["unlS", "privS"]  # null published_at sorts first
    with open(tmp_path / "out" / "longform.csv", newline="") as f:
        assert next(csv.reader(f)) == CSV_COLUMNS


def test_csv_cells_and_title_match_column(tmp_path):
    d = tmp_path / "projLF"
    d.mkdir()
    f = d / "youtube_uploads.json"
    f.write_text(
        json.dumps(
            {"1": {"video_id": "vidA", "title": "T"}, "2": {"video_id": "vidB", "title": "x"}}
        )
    )
    rows = build_inventory(
        _live(vidA=_vid(), vidB=_vid(), vidC=_vid()),
        **_empty(lf_upload_files=[f]),
    )
    write_slices(rows, tmp_path / "o")
    with open(tmp_path / "o" / "longform.csv", newline="") as fh:
        got = _by_id(list(csv.DictReader(fh)))
    assert got["vidA"]["uploaded_title_matches_live"] == "true"
    assert got["vidB"]["uploaded_title_matches_live"] == "false"
    assert got["vidC"]["uploaded_title_matches_live"] == ""
    assert got["vidA"]["lf_index"] == "1"
    assert got["vidA"]["evidence_sources"] == "lf_upload"
    assert got["vidC"]["waypost_sync_events"] == "0"


def _make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "create table deliveries (platform text, platform_upload_id text, project_name text,"
        " short_index integer, sidecar_path text)"
    )
    conn.execute("insert into deliveries values ('youtube','vidA','projD',1,'p.json')")
    conn.execute("insert into deliveries values ('youtube',NULL,'projD',2,'p.json')")
    conn.execute("insert into deliveries values ('tiktok','vidZ','projD',3,'p.json')")
    conn.commit()
    conn.close()


def test_load_deliveries_reads_youtube_ids_only_and_is_read_only(tmp_path):
    db = tmp_path / "q.sqlite3"
    _make_db(db)
    before = db.read_bytes()
    rows = load_deliveries(db)
    assert rows == [("vidA", "projD", 1, "p.json")]
    assert db.read_bytes() == before


def test_load_deliveries_connection_rejects_writes(tmp_path, monkeypatch):
    db = tmp_path / "q.sqlite3"
    _make_db(db)
    seen = []
    real = sqlite3.connect

    def spy(target, *a, **kw):
        seen.append((target, kw))
        return real(target, *a, **kw)

    monkeypatch.setattr("gomoveshift.catalog.inventory.sqlite3.connect", spy)
    load_deliveries(db)
    target, kw = seen[0]
    assert "mode=ro" in target and kw.get("uri") is True
    with pytest.raises(sqlite3.OperationalError):
        real(f"file:{db}?mode=ro", uri=True).execute("delete from deliveries")


def test_load_ledger_entries(tmp_path):
    p = tmp_path / "l.json"
    p.write_text(json.dumps({"entries": [{"video_id": "vidA"}]}))
    assert load_ledger_entries(p) == [{"video_id": "vidA"}]


def test_cli_registered():
    assert "catalog-inventory" in _COMMAND_MAP
    args = _build_parser().parse_args(
        ["catalog-inventory", "--live-catalog", "a", "--out-dir", "b"]
    )
    assert args.live_catalog == "a" and args.out_dir == "b"


def test_cli_end_to_end(tmp_path, capsys):
    from gomoveshift.cli.handlers.catalog_inventory import cmd_catalog_inventory

    vid_out = tmp_path / "vid"
    (vid_out / "projLF").mkdir(parents=True)
    (vid_out / "projLF" / "youtube_uploads.json").write_text(
        json.dumps({"1": {"video_id": "vidA", "title": "T"}})
    )
    out = tmp_path / "gms"
    (out / "projT" / "tracking").mkdir(parents=True)
    (out / "projT" / "tracking" / "t.json").write_text(
        json.dumps(
            {"project": "projT", "shorts": [{"index": 1, "platform_ids": {"youtube": "vidS"}}]}
        )
    )
    db = tmp_path / "q.sqlite3"
    _make_db(db)
    events = tmp_path / "events.jsonl"
    events.write_text('{"type": "yt_sync_ok", "timestamp": "t", "data": {"video_id": "vidS"}}\n')
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({"entries": []}))
    live = tmp_path / "live.json"
    live.write_text(
        json.dumps(
            {
                "videos": {
                    "vidA": _vid(),
                    "vidS": _vid(duration_s=30),
                    "vidN": _vid(duration_s=30, privacy="private"),
                }
            }
        )
    )
    args = SimpleNamespace(
        live_catalog=str(live),
        out_dir=str(tmp_path / "slices"),
        vid_output_dir=str(vid_out),
        output_dir=str(out),
        queue_db=str(db),
        events_jsonl=str(events),
        ledger=str(ledger),
    )
    assert cmd_catalog_inventory(args, SimpleNamespace()) == 0
    printed = capsys.readouterr().out
    assert "longform.csv: 1" in printed and "shorts-public.csv: 1" in printed
    assert "shorts-nonpublic.csv: 1" in printed
    assert (tmp_path / "slices" / "longform.csv").exists()
