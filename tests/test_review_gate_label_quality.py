"""Gate C/D re-surfaces LABEL-QUALITY warnings at the human-review prompt.

``content_gates.check_label_quality`` correctly detects a county-only LF
label when the trace's place sequence carries a recognizable settlement, but
it is WARN-only and its message only ever reaches the log -- ~400 lines
above the Gate C/D prompt. This covers ``_label_quality_warning`` (the pure,
unit-testable mirror of that gate) and its inline surfacing in ``gate_cd``.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from gomoveshift.pipeline.review_gate import ReviewGate, _label_quality_warning
from gomoveshift.pipeline.validation.content_gates import check_label_quality

from _state_helpers import make_state


def _county_only_youtube_json() -> dict:
    return {
        "title": "County Cork in 4K",
        "location": "County Cork",
        "resolved_location": {
            "dominant_county": "Cork",
            "ordered_place_sequence": ["West Cork", "County Cork"],
        },
    }


def _title_prefixed_youtube_json() -> dict:
    return {
        "title": "To Mizen Head, County Cork",
        "location": "To Mizen Head, County Cork",
        "resolved_location": {
            "dominant_county": "Cork",
            "ordered_place_sequence": ["West Cork"],
        },
    }


def _settlement_anchored_youtube_json() -> dict:
    return {
        "title": "West Cork Coastal Drive",
        "location": "West Cork",
        "resolved_location": {
            "dominant_county": "Cork",
            "ordered_place_sequence": ["West Cork", "County Cork"],
        },
    }


class TestLabelQualityWarningHelper:
    """_label_quality_warning -- pure mirror of check_label_quality."""

    def test_county_only_label_with_settlement_warns(self):
        warning = _label_quality_warning(_county_only_youtube_json())
        assert warning is not None
        assert "West Cork" in warning

    def test_settlement_anchored_label_no_warning(self):
        assert _label_quality_warning(_settlement_anchored_youtube_json()) is None

    def test_no_resolved_location_silently_skipped(self):
        data = {"title": "County Cork in 4K", "location": "County Cork"}
        assert _label_quality_warning(data) is None

    def test_title_label_agrees_with_check_label_quality(self, caplog):
        """The two call sites share one helper -- they must never disagree on
        a title-vs-place verdict (the duplication that once let a fix reach one
        and not the other)."""
        meta = _title_prefixed_youtube_json()
        warning = _label_quality_warning(meta)
        assert warning is not None
        assert "title" in warning
        assert "county-only" not in warning

        with caplog.at_level("WARNING"):
            result = check_label_quality(meta)
        assert result.passed is True
        assert "title, not a place" in caplog.text
        assert "county-only" not in caplog.text


class TestGateCdSurfacesLabelQuality:
    """gate_cd prints the warning inline and a banner before the prompt."""

    def test_county_only_label_prints_inline(self, tmp_path, capsys):
        gate = ReviewGate(interactive=False)
        json_path = tmp_path / "example-2025-01-01-place-s1_lf01_youtube.json"
        json_path.write_text(json.dumps(_county_only_youtube_json()))

        state = make_state(regular_lf_paths=[], episode_dates=["2026-03-01"])
        gate.gate_cd(state=state, output_dir=tmp_path, date_key="2026-03-01")

        out = capsys.readouterr().out
        assert "LABEL-QUALITY" in out
        assert "West Cork" in out

    def test_county_only_label_banner_before_interactive_prompt(self, tmp_path, capsys):
        gate = ReviewGate(interactive=True)
        json_path = tmp_path / "example-2025-01-01-place-s1_lf01_youtube.json"
        json_path.write_text(json.dumps(_county_only_youtube_json()))

        state = make_state(regular_lf_paths=[], episode_dates=["2026-03-01"])
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            gate.gate_cd(state=state, output_dir=tmp_path, date_key="2026-03-01")

        out = capsys.readouterr().out
        assert "1 LF label" in out

    def test_settlement_anchored_label_no_warning_no_banner(self, tmp_path, capsys):
        gate = ReviewGate(interactive=False)
        json_path = tmp_path / "example-2025-01-01-place-s1_lf01_youtube.json"
        json_path.write_text(json.dumps(_settlement_anchored_youtube_json()))

        state = make_state(regular_lf_paths=[], episode_dates=["2026-03-01"])
        gate.gate_cd(state=state, output_dir=tmp_path, date_key="2026-03-01")

        out = capsys.readouterr().out
        assert "LABEL-QUALITY" not in out

    def test_operator_may_still_approve_with_warning_present(self, tmp_path, capsys):
        """Warn-only: a 'y' answer still passes the gate."""
        gate = ReviewGate(interactive=True)
        json_path = tmp_path / "example-2025-01-01-place-s1_lf01_youtube.json"
        json_path.write_text(json.dumps(_county_only_youtube_json()))

        state = make_state(regular_lf_paths=[], episode_dates=["2026-03-01"])
        with (
            patch("sys.stdin.isatty", return_value=True),
            patch("builtins.input", return_value="y"),
        ):
            gate.gate_cd(state=state, output_dir=tmp_path, date_key="2026-03-01")

        assert state.gate_cd_confirmed is True
        out = capsys.readouterr().out
        assert "LABEL-QUALITY" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
