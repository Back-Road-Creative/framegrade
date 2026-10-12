"""TikTok scheduling controlled by ``publish_tiktok``.

The Metricool shorts CSV publishes to TikTok by default. Metricool handles
publishing — no first-party TikTok auth is needed (the
``TikTokCollector.collect_short_metrics`` stub only affects analytics
*collection*, never publishing). These tests pin the toggle:

  - default (``publish_tiktok=True``) → ``TikTok=TRUE`` and populated
    TikTok-specific columns in every shorts row;
  - ``publish_tiktok=False`` → ``TikTok`` toggle forced to the literal
    "FALSE" and every TikTok-specific column empty (suppresses TikTok).

The 31-column GOLDEN header is unchanged in both modes (Metricool's importer
schema is fixed); only the *values* in the TikTok columns change.
"""

from __future__ import annotations

import csv as csvmod
from pathlib import Path
from types import SimpleNamespace

from gomoveshift.pipeline.csv_generator import (
    GOLDEN_COLUMNS,
    CSVGenerator,
    _drive_link_for_metricool,
    build_consolidated_shorts_csv,
)

# TikTok-specific value columns whose population would schedule a phantom
# TikTok post. The ``TikTok`` toggle is asserted separately (it should be
# the literal "FALSE", not empty, when gated off).
_TIKTOK_VALUE_COLUMNS = [
    "TikTok Title",
    "TikTok disable comments",
    "TikTok disable duet",
    "TikTok disable stitch",
    "TikTok Post Privacy",
    "TikTok Auto Add Music",
]


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


def _make_slot(short_index: int):
    return SimpleNamespace(short_index=short_index, date="2026-04-01", time="16:00:00")


def _write_consolidated(tmp_path: Path, **kwargs) -> list[dict]:
    out = tmp_path / "proj_shorts_metricool.csv"
    build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [_make_post(1)]},
        slots_by_short_index={1: _make_slot(1)},
        drive_links_by_short_index={1: "https://drive.google.com/1"},
        parent_lf_video_id_by_short_index={1: "abc"},
        parent_recording_date_by_short_index={1: "2026-04-01"},
        **kwargs,
    )
    with open(out, encoding="utf-8-sig") as f:
        return list(csvmod.DictReader(f))


def test_consolidated_default_enables_tiktok(tmp_path: Path):
    rows = _write_consolidated(tmp_path)
    assert rows, "expected at least one row"
    for row in rows:
        assert row["TikTok"] == "TRUE"
        assert row["TikTok Title"] == "tt 1"
        assert row["TikTok Post Privacy"] == "PUBLIC_TO_EVERYONE"
        assert row["TikTok disable comments"] == "no"


def test_consolidated_opt_out_disables_tiktok(tmp_path: Path):
    rows = _write_consolidated(tmp_path, publish_tiktok=False)
    assert rows, "expected at least one row"
    for row in rows:
        assert row["TikTok"] == "FALSE"
        for col in _TIKTOK_VALUE_COLUMNS:
            assert row[col] == "", f"{col} should be empty when publish_tiktok=False"


def test_consolidated_default_keeps_golden_header(tmp_path: Path):
    out = tmp_path / "proj_shorts_metricool.csv"
    build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [_make_post(1)]},
        slots_by_short_index={1: _make_slot(1)},
        drive_links_by_short_index={1: "https://drive.google.com/1"},
        parent_lf_video_id_by_short_index={1: "abc"},
        parent_recording_date_by_short_index={1: "2026-04-01"},
    )
    with open(out, encoding="utf-8-sig") as f:
        header = next(csvmod.reader(f))
    assert header == GOLDEN_COLUMNS


def test_consolidated_explicit_opt_in_enables_tiktok(tmp_path: Path):
    rows = _write_consolidated(tmp_path, publish_tiktok=True)
    assert rows
    for row in rows:
        assert row["TikTok"] == "TRUE"
        assert row["TikTok Title"] == "tt 1"
        assert row["TikTok Post Privacy"] == "PUBLIC_TO_EVERYONE"
        assert row["TikTok disable comments"] == "no"


def test_csvgenerator_build_row_default_enables_tiktok():
    gen = CSVGenerator()
    row = gen._build_row(
        post=_make_post(1),
        slot=_make_slot(1),
        drive_link="https://drive.google.com/1",
    )
    assert row["TikTok"] == "TRUE"
    assert row["TikTok Title"] == "tt 1"


def test_csvgenerator_build_row_opt_out_disables_tiktok():
    gen = CSVGenerator(publish_tiktok=False)
    row = gen._build_row(
        post=_make_post(1),
        slot=_make_slot(1),
        drive_link="https://drive.google.com/1",
    )
    assert row["TikTok"] == "FALSE"
    for col in _TIKTOK_VALUE_COLUMNS:
        assert row[col] == ""


# --- Drive media URL form: TikTok needs uc?export=download, not /view ---------

_VIEW = "https://drive.google.com/file/d/ABC123_xyz/view?usp=sharing"
_DOWNLOAD = "https://drive.google.com/uc?export=download&id=ABC123_xyz"


def test_drive_link_helper_converts_view_to_download():
    assert _drive_link_for_metricool(_VIEW) == _DOWNLOAD


def test_drive_link_helper_passes_through_non_view():
    # already-download, empty, and non-Drive links are untouched
    assert _drive_link_for_metricool(_DOWNLOAD) == _DOWNLOAD
    assert _drive_link_for_metricool("") == ""
    assert _drive_link_for_metricool("https://example.com/v.mp4") == "https://example.com/v.mp4"


def test_consolidated_shorts_emit_download_url(tmp_path: Path):
    out = tmp_path / "proj_shorts_metricool.csv"
    build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [_make_post(1)]},
        slots_by_short_index={1: _make_slot(1)},
        drive_links_by_short_index={1: _VIEW},
        parent_lf_video_id_by_short_index={1: "abc"},
        parent_recording_date_by_short_index={1: "2026-04-01"},
    )
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csvmod.DictReader(f))
    assert rows[0]["Picture Url 1"] == _DOWNLOAD


def test_lf_row_keeps_view_url(tmp_path: Path):
    # Long-form links can exceed Drive's ~100MB virus-scan threshold where
    # uc?export=download breaks — generate_lf_row must NOT convert.
    out = tmp_path / "lf_metricool.csv"
    CSVGenerator().generate_lf_row(
        title="t",
        tags="a,b",
        category="19",
        drive_link=_VIEW,
        schedule_date="2026-04-01",
        schedule_time="16:00:00",
        output_path=out,
    )
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csvmod.DictReader(f))
    assert rows[0]["Picture Url 1"] == _VIEW
