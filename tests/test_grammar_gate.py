"""Grammar gate: HARD-FAIL on preposition-collision artifacts.

Real shipped defects:

  * "POV drive through To Clonakilty, County Cork" -- "through To " splice
  * "Slow Drive Through To Clonakilty" -- "Through To " splice
  * "#ToClonakilty" -- a location hashtag that runs the preposition into
    the place name

The gate must fire on every generated text surface: the LF title/description
(``check_metadata_quality``) AND every row of the shorts/TikTok CSV
(``build_consolidated_shorts_csv``), upstream of the CSV write.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.csv_generator import build_consolidated_shorts_csv
from gomoveshift.pipeline.validation.content_gates import check_metadata_quality


def _base_metadata(**overrides):
    meta = {
        "title": "Kenmare to Sneem — POV Driving Kerry, Ireland 4K",
        "description": (
            "POV drive through Kenmare to Sneem on the wild Atlantic coast.\n\n"
            "\U0001f4cd Location: Kenmare to Sneem, Kerry\n"
            "\U0001f3ac Final cut: 32m 10s"
        ),
        "location": "Kenmare to Sneem, Kerry",
        "tags": ["Kerry", "Ireland", "POV drive"],
    }
    meta.update(overrides)
    return meta


class TestMetadataQualityGrammarGate:
    def test_clean_title_and_description_pass(self):
        result = check_metadata_quality(_base_metadata())
        assert result.passed, result.message

    def test_title_through_to_splice_hard_fails(self):
        meta = _base_metadata(
            title="POV drive through To Clonakilty, County Cork 4K",
        )
        result = check_metadata_quality(meta)
        assert not result.passed
        assert "title" in result.message.lower()
        assert "grammar" in result.message.lower() or "preposition" in result.message.lower()

    def test_description_through_to_splice_hard_fails(self):
        meta = _base_metadata(
            description=(
                "Slow Drive Through To Clonakilty on the wild Atlantic coast.\n\n"
                "\U0001f4cd Location: Kenmare to Sneem, Kerry\n"
                "\U0001f3ac Final cut: 32m 10s"
            ),
        )
        result = check_metadata_quality(meta)
        assert not result.passed
        assert "description" in result.message.lower()

    def test_hashtag_to_capital_hard_fails(self):
        meta = _base_metadata(
            description=(
                "POV drive through Clonakilty on the wild Atlantic coast. #ToClonakilty\n\n"
                "\U0001f4cd Location: Kenmare to Sneem, Kerry\n"
                "\U0001f3ac Final cut: 32m 10s"
            ),
        )
        result = check_metadata_quality(meta)
        assert not result.passed
        assert "description" in result.message.lower()

    def test_doubled_preposition_hard_fails(self):
        meta = _base_metadata(
            title="POV drive through through Clonakilty, County Cork 4K",
        )
        result = check_metadata_quality(meta)
        assert not result.passed

    def test_tag_starting_with_preposition_hard_fails(self):
        meta = _base_metadata(tags=["Kerry", "Ireland", "To Clonakilty"])
        result = check_metadata_quality(meta)
        assert not result.passed
        assert "tag" in result.message.lower()


def _make_post(short_index: int, **overrides):
    post = SimpleNamespace(
        short_index=short_index,
        caption=f"Short {short_index} caption",
        first_comment="",
        youtube_title=f"Short {short_index}",
        youtube_tags="",
        tiktok_title=f"Short {short_index}",
        pinterest_board="",
    )
    for k, v in overrides.items():
        setattr(post, k, v)
    return post


def _make_slot(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        date="2026-04-29",
        time="16:00:00",
        datetime_est="2026-04-29T16:00:00-04:00",
        day_of_week="Wednesday",
    )


def _build_csv(tmp_path: Path, post, name="shorts.csv"):
    out = tmp_path / name
    return build_consolidated_shorts_csv(
        output_path=out,
        shorts_by_lf={1: [post]},
        slots_by_short_index={1: _make_slot(1)},
        drive_links_by_short_index={1: "https://drive.example/1"},
        parent_lf_video_id_by_short_index={1: "vid1"},
        parent_recording_date_by_short_index={1: "2026-01-01"},
    )


class TestShortsCsvGrammarGate:
    def test_clean_row_passes(self, tmp_path: Path):
        post = _make_post(1, youtube_title="Kenmare to Sneem POV Drive")
        out = _build_csv(tmp_path, post)
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1

    def test_csv_title_through_to_splice_hard_fails(self, tmp_path: Path):
        post = _make_post(1, youtube_title="Slow Drive Through To Clonakilty")
        with pytest.raises(ValueError, match="(?i)grammar"):
            _build_csv(tmp_path, post)

    def test_csv_hashtag_to_capital_hard_fails(self, tmp_path: Path):
        post = _make_post(1, caption="Rolling through Clonakilty #ToClonakilty")
        with pytest.raises(ValueError, match="(?i)grammar"):
            _build_csv(tmp_path, post)

    def test_csv_error_names_the_bad_row(self, tmp_path: Path):
        post = _make_post(1, youtube_title="Slow Drive Through To Clonakilty")
        with pytest.raises(ValueError, match="row 0"):
            _build_csv(tmp_path, post)

    def test_csv_tag_starting_with_preposition_hard_fails_even_mid_cell(self, tmp_path: Path):
        # The offending tag was not the FIRST tag in the comma-joined cell, so
        # a whole-cell check never saw it as "starting with" a preposition.
        post = _make_post(1, youtube_tags="dashcam, To Goleen, driving")
        with pytest.raises(ValueError, match="(?i)grammar"):
            _build_csv(tmp_path, post)
