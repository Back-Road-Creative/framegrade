"""Metricool CSV recording-date enforcement.

The consolidated shorts CSV writer must append a ``Filmed <Month YYYY>.``
suffix to the ``Text`` column for every row, sourced from a *required*
``parent_recording_date_by_short_index`` parameter. Construction-correct:

* the parameter has NO default — callers cannot accidentally bypass it,
* missing entries raise ``ValueError`` naming the short index,
* malformed dates raise ``ValueError`` (no silent fallback),
* the pretty formatter is the single source of truth for the prose form.
"""

from __future__ import annotations

import csv
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.pipeline.csv_generator import (
    _format_recording_date_pretty,
    build_consolidated_shorts_csv,
)


def _make_post(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        caption=f"Short {short_index} caption",
        first_comment="",
        youtube_title=f"Short {short_index}",
        youtube_tags="",
        tiktok_title=f"Short {short_index}",
        pinterest_board="",
    )


def _make_slot(short_index: int):
    return SimpleNamespace(
        short_index=short_index,
        date="2026-04-29",
        time="16:00:00",
        datetime_est="2026-04-29T16:00:00-04:00",
        day_of_week="Wednesday",
    )


class TestFormatRecordingDatePretty:
    @pytest.mark.parametrize(
        "raw, pretty",
        [
            ("2025-09-14", "September 2025"),
            ("2026-01-01", "January 2026"),
            ("2025-12-31", "December 2025"),
        ],
    )
    def test_format_recording_date_pretty_known_dates(self, raw, pretty):
        assert _format_recording_date_pretty(raw) == pretty

    @pytest.mark.parametrize(
        "bad",
        ["", "not-a-date", "2025/09/14", "September 2025", "20250914", "2025-13-01"],
    )
    def test_format_recording_date_pretty_invalid_date_raises(self, bad):
        with pytest.raises(ValueError):
            _format_recording_date_pretty(bad)


class TestTextColumnFilmedSuffix:
    def test_text_column_contains_filmed_sentence(self, tmp_path: Path):
        out = tmp_path / "shorts.csv"
        build_consolidated_shorts_csv(
            output_path=out,
            shorts_by_lf={1: [_make_post(1)]},
            slots_by_short_index={1: _make_slot(1)},
            drive_links_by_short_index={1: "https://drive.example/1"},
            parent_lf_video_id_by_short_index={1: "vid1"},
            parent_recording_date_by_short_index={1: "2025-09-14"},
        )
        with open(out, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        text = rows[0]["Text"]
        assert text.endswith("Filmed September 2025."), text
        # original caption preserved before the suffix
        assert "Short 1 caption" in text
        # exactly one trailing period
        assert not text.endswith(".."), text

    def test_caption_without_terminator_gets_its_full_stop_then_one_space(self, tmp_path: Path):
        out = tmp_path / "shorts.csv"
        post = _make_post(1)
        post.caption = "no terminator here"
        build_consolidated_shorts_csv(
            output_path=out,
            shorts_by_lf={1: [post]},
            slots_by_short_index={1: _make_slot(1)},
            drive_links_by_short_index={1: "https://drive.example/1"},
            parent_lf_video_id_by_short_index={1: "vid1"},
            parent_recording_date_by_short_index={1: "2026-01-01"},
        )
        with open(out, encoding="utf-8") as f:
            row = next(csv.DictReader(f))
        # Never a run-on: the caption gets its own full stop before the Filmed sentence.
        assert row["Text"] == "no terminator here. Filmed January 2026."


class TestMissingRecordingDateRaises:
    def test_missing_recording_date_raises_at_csv_build(self, tmp_path: Path):
        out = tmp_path / "shorts.csv"
        with pytest.raises(ValueError, match="short_index 1"):
            build_consolidated_shorts_csv(
                output_path=out,
                shorts_by_lf={1: [_make_post(1)]},
                slots_by_short_index={1: _make_slot(1)},
                drive_links_by_short_index={1: "https://drive.example/1"},
                parent_lf_video_id_by_short_index={1: "vid1"},
                parent_recording_date_by_short_index={},
            )

    def test_empty_string_recording_date_raises(self, tmp_path: Path):
        out = tmp_path / "shorts.csv"
        with pytest.raises(ValueError, match="short_index 1"):
            build_consolidated_shorts_csv(
                output_path=out,
                shorts_by_lf={1: [_make_post(1)]},
                slots_by_short_index={1: _make_slot(1)},
                drive_links_by_short_index={1: "https://drive.example/1"},
                parent_lf_video_id_by_short_index={1: "vid1"},
                parent_recording_date_by_short_index={1: ""},
            )


class TestRequiredParameterNoDefault:
    def test_required_parameter_no_default(self):
        sig = inspect.signature(build_consolidated_shorts_csv)
        param = sig.parameters["parent_recording_date_by_short_index"]
        assert param.default is inspect.Parameter.empty, (
            "parent_recording_date_by_short_index must be a required parameter "
            "(no default) — otherwise callers can silently bypass the recording-"
            "date enforcement."
        )


# NOTE: caption-truncation-preserves-filmed-sentence test intentionally omitted.
# The current consolidated-CSV writer does NOT truncate per platform — the
# legacy ``truncate_for_platforms`` helper is only used by ``generate_lf_row``
# (long-form single-row writer), not by ``build_consolidated_shorts_csv``. The
# Twitter-280 cut path doesn't fire here, so there is no truncation step to
# preserve the suffix across. If a per-platform truncation is later added to
# the consolidated writer, add the suffix-preservation test alongside it.
