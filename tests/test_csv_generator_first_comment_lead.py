"""TDD tests: first-comment phrasing band-derived from trim_ratio.

* `_first_comment_lead(trim_ratio)` returns "▶ Highlights reel:"
  when the LF is a highlights cut (trim_ratio < HIGHLIGHTS_TRIM_THRESHOLD),
  otherwise "▶ Full drive:".
* The consolidated shorts CSV writer threads per-LF trim_ratio through to
  every row — the first-comment text matches the band of that short's
  parent LF.
* `HIGHLIGHTS_TRIM_THRESHOLD` is imported from `longform_metadata`
  (single source of truth).
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.pipeline.csv_generator import (
    _first_comment_lead,
    build_consolidated_shorts_csv,
)
from gomoveshift.pipeline.longform_metadata import HIGHLIGHTS_TRIM_THRESHOLD


class TestFirstCommentLead:
    def test_low_ratio_returns_highlights_phrasing(self):
        assert _first_comment_lead(0.3) == "▶ Highlights reel:"

    def test_high_ratio_returns_full_drive_phrasing(self):
        assert _first_comment_lead(0.7) == "▶ Full drive:"

    def test_none_returns_full_drive_phrasing_legacy_default(self):
        assert _first_comment_lead(None) == "▶ Full drive:"

    def test_at_threshold_returns_full_drive_phrasing(self):
        assert _first_comment_lead(HIGHLIGHTS_TRIM_THRESHOLD) == "▶ Full drive:"

    def test_threshold_imported_not_redefined(self):
        # csv_generator must not redefine the constant — single source of truth.
        from gomoveshift.pipeline import csv_generator as cg

        # If csv_generator declares its own constant, that's a regression;
        # accept it only when it's literally the same object/value.
        local = getattr(cg, "HIGHLIGHTS_TRIM_THRESHOLD", None)
        if local is not None:
            assert local is HIGHLIGHTS_TRIM_THRESHOLD or local == HIGHLIGHTS_TRIM_THRESHOLD


def _make_short_post(short_index: int, parent_lf_index: int):
    return SimpleNamespace(
        short_index=short_index,
        parent_lf_index=parent_lf_index,
        caption=f"Short {short_index} caption",
        first_comment="",
        youtube_title=f"Short {short_index}",
        youtube_tags="",
        tiktok_title=f"Short {short_index}",
        pinterest_board="",
    )


def _slot(date="2025-12-01", time="08:00:00", short_index=1):
    return SimpleNamespace(date=date, time=time, short_index=short_index)


class TestConsolidatedCSVTrimRatioWiring:
    def test_highlights_lf_short_uses_highlights_phrasing(self, tmp_path: Path):
        out = tmp_path / "shorts.csv"
        shorts_by_lf = {1: [_make_short_post(short_index=1, parent_lf_index=1)]}
        slots = {1: _slot(short_index=1)}
        drive_links = {1: "https://drive.example/1"}
        parent_ids = {1: "abc123"}
        # LF 1 is a heavy-trim cut.
        parent_trim = {1: 0.3}

        build_consolidated_shorts_csv(
            output_path=out,
            shorts_by_lf=shorts_by_lf,
            slots_by_short_index=slots,
            drive_links_by_short_index=drive_links,
            parent_lf_video_id_by_short_index=parent_ids,
            parent_recording_date_by_short_index={1: "2025-12-01"},
            parent_trim_ratio_by_short_index=parent_trim,
        )

        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        first_comment = rows[0]["First Comment Text"]
        assert first_comment.startswith("▶ Highlights reel:")
        assert "abc123" in first_comment

    def test_full_drive_lf_short_uses_full_drive_phrasing(self, tmp_path: Path):
        out = tmp_path / "shorts.csv"
        shorts_by_lf = {1: [_make_short_post(short_index=1, parent_lf_index=1)]}
        slots = {1: _slot(short_index=1)}
        drive_links = {1: "https://drive.example/1"}
        parent_ids = {1: "xyz789"}
        parent_trim = {1: 0.8}

        build_consolidated_shorts_csv(
            output_path=out,
            shorts_by_lf=shorts_by_lf,
            slots_by_short_index=slots,
            drive_links_by_short_index=drive_links,
            parent_lf_video_id_by_short_index=parent_ids,
            parent_recording_date_by_short_index={1: "2025-12-01"},
            parent_trim_ratio_by_short_index=parent_trim,
        )
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["First Comment Text"].startswith("▶ Full drive:")

    def test_missing_trim_ratio_falls_back_to_full_drive(self, tmp_path: Path):
        # Backward-compat: callers that don't pass the new arg keep the
        # legacy "full drive" phrasing (no behavior change for call sites
        # that predate the argument).
        out = tmp_path / "shorts.csv"
        shorts_by_lf = {1: [_make_short_post(short_index=1, parent_lf_index=1)]}
        slots = {1: _slot(short_index=1)}
        drive_links = {1: "https://drive.example/1"}
        parent_ids = {1: "noratio"}

        build_consolidated_shorts_csv(
            output_path=out,
            shorts_by_lf=shorts_by_lf,
            slots_by_short_index=slots,
            drive_links_by_short_index=drive_links,
            parent_lf_video_id_by_short_index=parent_ids,
            parent_recording_date_by_short_index={1: "2025-12-01"},
        )
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["First Comment Text"].startswith("▶ Full drive:")
