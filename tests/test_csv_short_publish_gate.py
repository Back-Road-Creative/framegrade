"""Gate: a short must not post before its parent LF's ``publish_at``.

Each short's ``First Comment Text`` links to its parent LF's YouTube upload,
which is private until ``publish_at``. A short scheduled before that instant
ships a dead (404) first-comment link until the LF goes public. Slots are
America/New_York local; ``publish_at`` is UTC, so the gate compares instants
(17:00 EDT = 21:00 UTC clears a 20:00 UTC publish; 14:00 EDT = 18:00 UTC does not).
"""

from __future__ import annotations

import csv as csvmod
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.csv_generator import build_consolidated_shorts_csv


def _post(i: int):
    return SimpleNamespace(
        short_index=i,
        caption=f"c{i}",
        youtube_title=f"t{i}",
        youtube_category="19",
        youtube_tags="a,b",
        tiktok_title=f"tt{i}",
        pinterest_board="d",
        first_comment=f"#x{i}",
    )


def _slot(i: int, date: str, time: str = "16:00:00"):
    return SimpleNamespace(short_index=i, date=date, time=time)


def _build(tmp_path: Path, slot_date: str, slot_time: str, publish_at):
    return build_consolidated_shorts_csv(
        output_path=tmp_path / "p_shorts_metricool.csv",
        shorts_by_lf={1: [_post(1)]},
        slots_by_short_index={1: _slot(1, slot_date, slot_time)},
        drive_links_by_short_index={1: "https://drive.google.com/1"},
        parent_lf_video_id_by_short_index={1: "abc"},
        parent_recording_date_by_short_index={1: "2026-04-01"},
        parent_publish_at_by_short_index={1: publish_at},
    )


def test_short_same_day_before_publish_raises(tmp_path: Path):
    # publish 2026-07-21 20:00 UTC = 16:00 EDT; short posts 14:00 EDT (18:00 UTC) -> before
    with pytest.raises(AssertionError, match="SHORT-BEFORE-PUBLISH"):
        _build(tmp_path, "2026-07-21", "14:00:00", "2026-07-21T20:00:00+00:00")


def test_short_day_before_publish_raises(tmp_path: Path):
    with pytest.raises(AssertionError, match="SHORT-BEFORE-PUBLISH"):
        _build(tmp_path, "2026-07-20", "18:00:00", "2026-07-21T20:00:00+00:00")


def test_short_after_publish_passes(tmp_path: Path):
    # short posts 17:00 EDT (21:00 UTC), after the 20:00 UTC publish
    out = _build(tmp_path, "2026-07-21", "17:00:00", "2026-07-21T20:00:00+00:00")
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csvmod.DictReader(f))
    assert rows[0]["First Comment Text"].endswith("youtu.be/abc")


def test_missing_publish_at_is_noop(tmp_path: Path):
    # legacy / skip-upload caller threads None -> gate imposes no constraint
    out = _build(tmp_path, "2026-07-20", "14:00:00", None)
    assert Path(out).exists()


def test_no_map_is_noop(tmp_path: Path):
    # caller omits the map entirely -> gate is skipped
    out = build_consolidated_shorts_csv(
        output_path=tmp_path / "p_shorts_metricool.csv",
        shorts_by_lf={1: [_post(1)]},
        slots_by_short_index={1: _slot(1, "2026-07-20", "14:00:00")},
        drive_links_by_short_index={1: "https://drive.google.com/1"},
        parent_lf_video_id_by_short_index={1: "abc"},
        parent_recording_date_by_short_index={1: "2026-04-01"},
    )
    assert Path(out).exists()
