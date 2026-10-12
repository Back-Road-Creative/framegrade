"""Consolidated shorts CSV tests.

Verifies the ``build_consolidated_shorts_csv`` helper in ``csv_generator``:
  - 4 LFs x 9 shorts = 36 rows.
  - Every row's First Comment Text begins with the YouTube URL prefix.
  - Header matches GOLDEN_COLUMNS exactly (length, order, contents).
"""

from __future__ import annotations

import csv as csvmod
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.csv_generator import (
    CSV_HEADERS,
    GOLDEN_COLUMNS,
    _assert_metricool_rows_have_empty_youtube_category,
    build_consolidated_shorts_csv,
)


def _make_post(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        caption=f"caption {short_index}",
        youtube_title=f"title {short_index}",
        youtube_category="19",
        youtube_tags="driving,ireland",
        tiktok_title=f"tt {short_index}",
        pinterest_board="drives",
        first_comment=f"#tag{short_index}",
    )


def _make_slot(short_index: int, day: int):
    return SimpleNamespace(
        short_index=short_index,
        date=f"2026-04-{day:02d}",
        time="16:00:00",
        datetime_est=f"2026-04-{day:02d}T16:00:00-04:00",
        day_of_week="Wednesday",
    )


def test_consolidated_csv_has_correct_header(tmp_path: Path):
    out = tmp_path / "out_shorts_metricool.csv"
    build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [_make_post(1)]},
        slots_by_short_index={1: _make_slot(1, 1)},
        drive_links_by_short_index={1: "https://drive.google.com/1"},
        parent_lf_video_id_by_short_index={1: "abc"},
        parent_recording_date_by_short_index={1: "2026-04-01"},
    )
    with open(out, encoding="utf-8-sig") as f:
        reader = csvmod.reader(f)
        header = next(reader)
    assert header == GOLDEN_COLUMNS
    assert header == CSV_HEADERS
    assert len(header) == 31


def test_consolidated_csv_row_count_4x9(tmp_path: Path):
    shorts_by_lf = {lf: [_make_post(lf * 100 + i) for i in range(9)] for lf in range(1, 5)}
    slots = {}
    drive_links = {}
    parent_ids = {}
    for lf, posts in shorts_by_lf.items():
        for i, p in enumerate(posts):
            slots[p.short_index] = _make_slot(p.short_index, lf)
            drive_links[p.short_index] = f"https://drive.google.com/s{p.short_index}"
            parent_ids[p.short_index] = f"vid_lf{lf:02d}"

    recording_dates = {
        p.short_index: "2026-04-01" for posts in shorts_by_lf.values() for p in posts
    }
    out = tmp_path / "proj_shorts_metricool.csv"
    build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf=shorts_by_lf,
        slots_by_short_index=slots,
        drive_links_by_short_index=drive_links,
        parent_lf_video_id_by_short_index=parent_ids,
        parent_recording_date_by_short_index=recording_dates,
    )
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csvmod.DictReader(f))
    assert len(rows) == 36


def test_consolidated_csv_first_comment_prefix(tmp_path: Path):
    shorts_by_lf = {1: [_make_post(1), _make_post(2)], 2: [_make_post(3)]}
    slots = {1: _make_slot(1, 1), 2: _make_slot(2, 1), 3: _make_slot(3, 2)}
    drive_links = {i: f"https://drive.google.com/s{i}" for i in (1, 2, 3)}
    parents = {1: "vidA", 2: "vidA", 3: "vidB"}

    out = tmp_path / "p_shorts_metricool.csv"
    build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf=shorts_by_lf,
        slots_by_short_index=slots,
        drive_links_by_short_index=drive_links,
        parent_lf_video_id_by_short_index=parents,
        parent_recording_date_by_short_index={1: "2026-04-01", 2: "2026-04-01", 3: "2026-04-01"},
    )
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csvmod.DictReader(f))

    for row in rows:
        assert row["First Comment Text"].startswith("▶ Full drive: https://youtu.be/")
    assert rows[0]["First Comment Text"].endswith("vidA")
    assert rows[2]["First Comment Text"].endswith("vidB")


# ── Metricool CSV YouTube-Category empty-string runtime gate ───────────


def test_metricool_gate_passes_on_empty_category():
    """Helper accepts rows whose YouTube Video Category column is empty."""
    rows = [{"YouTube Video Category": "", "YouTube Video Title": f"title {i}"} for i in range(3)]
    # Must not raise
    _assert_metricool_rows_have_empty_youtube_category(rows)


def test_metricool_gate_raises_on_populated_category():
    """Helper raises AssertionError when any row carries a non-empty category.

    Per docs/pipeline/81-resilience-resource-gates.md (Metricool CSV + YouTube
    category): populated numeric category IDs silently route Metricool
    posts to drafts. This runtime gate is the upstream-of-irreversible-
    artifact check (the irreversible artifact here is operator handoff to
    Metricool, which doesn't surface the silent drafting).
    """
    rows = [
        {"YouTube Video Category": "", "YouTube Video Title": "ok"},
        {"YouTube Video Category": "19", "YouTube Video Title": "bad"},
    ]
    with pytest.raises(AssertionError, match=r"YouTube Video Category"):
        _assert_metricool_rows_have_empty_youtube_category(rows)


def test_build_consolidated_shorts_csv_invokes_gate(tmp_path: Path, monkeypatch):
    """``build_consolidated_shorts_csv`` must call the runtime gate before
    handing rows to the writer — proves the upstream-of-CSV gate is wired,
    not just defined."""
    import gomoveshift.pipeline.csv_generator as csvgen

    called: dict[str, int] = {"count": 0}
    orig = csvgen._assert_metricool_rows_have_empty_youtube_category

    def _wrapped(rows):
        called["count"] += 1
        orig(rows)

    monkeypatch.setattr(
        csvgen,
        "_assert_metricool_rows_have_empty_youtube_category",
        _wrapped,
    )

    out = tmp_path / "wired_shorts_metricool.csv"
    build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [_make_post(1)]},
        slots_by_short_index={1: _make_slot(1, 1)},
        drive_links_by_short_index={1: "https://drive.google.com/1"},
        parent_lf_video_id_by_short_index={1: "abc"},
        parent_recording_date_by_short_index={1: "2026-04-01"},
    )
    assert called["count"] == 1
