"""Shorts→LF funnel: First Comment carries the compact ▶ youtu.be full-drive link.

Metricool shorts rows multi-post (DEFAULT_PLATFORMS enables Facebook, Twitter,
Instagram, Pinterest, TikTok, YouTube, Threads), so the parent LF's YouTube
link must NOT enter the shared ``Text`` cell — it would ship inside TikTok/IG/
FB captions. The funnel line lives on the First Comment surface instead:

* ``parent_id`` present → ``▶ Full drive: https://youtu.be/<id>`` (or
  ``▶ Highlights reel: …`` when the parent LF is a heavy-trim cut — the
  trim-honesty band from ``_first_comment_lead`` still applies).
* ``parent_id`` missing → the post's own ``first_comment`` passes through
  unchanged, with no blank first line.
* The shared ``Text`` cell never contains a YouTube URL either way.
* The 31-column GOLDEN header is unchanged.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.csv_generator import (
    GOLDEN_COLUMNS,
    build_consolidated_shorts_csv,
)


def _make_post(short_index: int, parent_lf_index: int = 1, first_comment: str = ""):
    return SimpleNamespace(
        short_index=short_index,
        parent_lf_index=parent_lf_index,
        caption=f"Short {short_index} caption",
        first_comment=first_comment,
        youtube_title=f"Short {short_index}",
        youtube_tags="",
        tiktok_title=f"Short {short_index}",
        pinterest_board="",
    )


def _slot(short_index: int):
    return SimpleNamespace(date="2026-04-01", time="08:00:00", short_index=short_index)


def _build(tmp_path: Path, *, parent_ids: dict, parent_trim: dict | None = None, post=None):
    post = post or _make_post(1)
    out = tmp_path / "shorts.csv"
    build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [post]},
        slots_by_short_index={1: _slot(1)},
        drive_links_by_short_index={1: "https://drive.example/1"},
        parent_lf_video_id_by_short_index=parent_ids,
        parent_recording_date_by_short_index={1: "2026-04-01"},
        parent_trim_ratio_by_short_index=parent_trim,
    )
    with open(out, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames
    return rows, header


class TestFullDriveFunnelLink:
    def test_first_comment_is_compact_youtu_be_link(self, tmp_path: Path):
        rows, _ = _build(tmp_path, parent_ids={1: "abc123XYZ_-"})
        assert rows[0]["First Comment Text"] == "▶ Full drive: https://youtu.be/abc123XYZ_-"

    def test_heavy_trim_parent_uses_highlights_framing(self, tmp_path: Path):
        rows, _ = _build(tmp_path, parent_ids={1: "abc123XYZ_-"}, parent_trim={1: 0.3})
        assert rows[0]["First Comment Text"] == ("▶ Highlights reel: https://youtu.be/abc123XYZ_-")

    def test_text_cell_never_carries_youtube_url(self, tmp_path: Path):
        # Multi-network rows share Text across TikTok/IG/FB/Twitter — the
        # YouTube funnel link must stay off that surface.
        rows, _ = _build(tmp_path, parent_ids={1: "abc123XYZ_-"})
        text = rows[0]["Text"]
        assert "youtu" not in text
        assert text.startswith("Short 1 caption")

    def test_missing_parent_id_falls_back_unchanged(self, tmp_path: Path):
        post = _make_post(1, first_comment="own comment")
        rows, _ = _build(tmp_path, parent_ids={}, post=post)
        fc = rows[0]["First Comment Text"]
        assert fc == "own comment"
        assert not fc.startswith("\n")

    def test_golden_header_unchanged(self, tmp_path: Path):
        rows, header = _build(tmp_path, parent_ids={1: "abc123XYZ_-"})
        assert header == GOLDEN_COLUMNS
        assert len(header) == 31


class TestPartialParentIdGapRaises:
    """A caller that DOES thread parent LF video IDs but drops one short's
    entry is a bug, not the legacy no-participation case above (empty dict).
    A silently blank funnel link for one short out of a populated batch must
    raise naming that short_index rather than falling back unnoticed.
    """

    def test_one_short_missing_from_populated_map_raises(self, tmp_path: Path):
        out = tmp_path / "shorts.csv"
        with pytest.raises(ValueError, match="METRICOOL-CSV-NO-PARENT-ID") as exc:
            build_consolidated_shorts_csv(
                output_path=out,
                shorts_by_lf={1: [_make_post(1), _make_post(2)]},
                slots_by_short_index={1: _slot(1), 2: _slot(2)},
                drive_links_by_short_index={
                    1: "https://drive.example/1",
                    2: "https://drive.example/2",
                },
                parent_lf_video_id_by_short_index={1: "vid1"},  # short 2 missing
                parent_recording_date_by_short_index={
                    1: "2026-04-01",
                    2: "2026-04-01",
                },
            )
        assert "2" in str(exc.value)
        assert not out.exists()

    def test_fully_populated_map_still_passes(self, tmp_path: Path):
        rows, _ = _build(tmp_path, parent_ids={1: "abc123XYZ_-"})
        assert rows[0]["First Comment Text"].startswith("▶ Full drive:")
