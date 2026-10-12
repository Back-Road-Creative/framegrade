"""Tests for the recording-date content-fidelity DoD check.

A freshness-only check (mtime newer than run_start) cannot catch a wrong
recording date: the ``📹 Recorded: <date>`` line in the YouTube description
could default to a single date for every LF (only LF 1 correct) and still pass.

``recording_date_per_lf`` — youtube.json description's ``📹 Recorded:``
line must match the LF's day-batch date.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from _state_helpers import make_state

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import validate_pipeline_output as v  # noqa: E402

PROJECT = "tp"
LFS = [(1, Path("/tmp/unused_lf_01.mp4")), (2, Path("/tmp/unused_lf_02.mp4"))]


# ---------------------------------------------------------------------------
# Synthetic content helpers
# ---------------------------------------------------------------------------


SAMPLE_DESC_LONG = (
    "Beautiful drive through the Carolinas with stops at scenic overlooks, "
    "a quaint downtown, and a riverside park. We capture the back roads "
    "between mile markers and stop briefly at a roadside diner for a meal. "
    "The full route covers two counties and ~110 miles of mixed terrain.\n\n"
    "📹 Recorded: 2030-02-16\n\n"
    "🔔 Subscribe for more drives: https://example.test/subscribe\n"
    "📍 Map: https://example.test/map\n\n"
    "#scenicdrive #carolinas #roadtrip #travel"
)


def _write_youtube_json(path: Path, description: str, *, title: str = "Drive") -> None:
    import json as _json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps({"title": title, "description": description, "tags": []}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# metricool_text_fidelity — DELETED (dead-surface cleanup)
# ---------------------------------------------------------------------------
#
# The no-op stub was deleted entirely.
# A registration test below asserts the name is *absent* from the
# rendered report so accidental re-introduction is caught.


# ---------------------------------------------------------------------------
# recording_date_per_lf
# ---------------------------------------------------------------------------


def test_recording_date_wrong_fails(tmp_path: Path) -> None:
    """One youtube.json says 2030-02-15 but the day batch says 2030-02-16."""
    desc_wrong = SAMPLE_DESC_LONG.replace("2030-02-16", "2030-02-15")
    _write_youtube_json(tmp_path / f"{PROJECT}_lf01_youtube.json", desc_wrong)
    _write_youtube_json(tmp_path / f"{PROJECT}_lf02_youtube.json", SAMPLE_DESC_LONG)

    state = make_state(per_lf_recording_date={1: "2030-02-16", 2: "2030-02-16"})
    result = v.check_recording_date_per_lf(tmp_path, LFS, PROJECT, state)
    assert not result.passed
    assert "lf01" in result.detail or "lf_01" in result.detail
    assert "2030-02-15" in result.detail


def test_recording_date_correct_passes(tmp_path: Path) -> None:
    _write_youtube_json(tmp_path / f"{PROJECT}_lf01_youtube.json", SAMPLE_DESC_LONG)
    _write_youtube_json(tmp_path / f"{PROJECT}_lf02_youtube.json", SAMPLE_DESC_LONG)
    state = make_state(per_lf_recording_date={1: "2030-02-16", 2: "2030-02-16"})
    result = v.check_recording_date_per_lf(tmp_path, LFS, PROJECT, state)
    assert result.passed, result.detail


def test_recording_date_falls_back_to_episode_dates(tmp_path: Path) -> None:
    """Without per_lf_recording_date, derive via episode_dates + first segment."""
    _write_youtube_json(tmp_path / f"{PROJECT}_lf01_youtube.json", SAMPLE_DESC_LONG)
    _write_youtube_json(tmp_path / f"{PROJECT}_lf02_youtube.json", SAMPLE_DESC_LONG)
    # State field absent — fall back to episode_dates indexed by
    # first global segment.  Provide drive_plan-like structure too.
    state = {
        "episode_dates": ["2030-02-16", "2030-02-16"],
        "per_lf_first_segment_index": {1: 0, 2: 1},
    }
    result = v.check_recording_date_per_lf(tmp_path, LFS, PROJECT, state)
    assert result.passed, result.detail


def test_recording_date_no_state_skips_gracefully(tmp_path: Path) -> None:
    """No state, no episode_dates — check should PASS-as-skipped (cannot fail
    when there's nothing to compare against) rather than crash.
    """
    _write_youtube_json(tmp_path / f"{PROJECT}_lf01_youtube.json", SAMPLE_DESC_LONG)
    _write_youtube_json(tmp_path / f"{PROJECT}_lf02_youtube.json", SAMPLE_DESC_LONG)
    result = v.check_recording_date_per_lf(tmp_path, LFS, PROJECT, None)
    assert result.passed
    assert "skipped" in result.detail.lower() or "no expected" in result.detail.lower()


# ---------------------------------------------------------------------------
# Registration / wiring — both skip_upload_audit and post_upload
# ---------------------------------------------------------------------------


def test_new_checks_registered_in_post_upload_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validator's post_upload mode must include recording_date_per_lf.

    We don't care if it passes — only that the name appears in the
    rendered report.  This guards against regressions where someone
    deletes the registration line.
    """
    project_dir = tmp_path / PROJECT
    long_forms = project_dir / "long_forms"
    long_forms.mkdir(parents=True)
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    report = v.run_validation(
        project=PROJECT,
        output_base=tmp_path,
        snapshot_dir=snapshot,
        state_dir=state_dir,
        run_start=datetime.now(tz=timezone.utc),
        mode=v.Mode.POST_UPLOAD,
    )
    names = {c.name for c in report.checks}
    assert "metricool_text_fidelity" not in names  # retired
    assert "metricool_fresh" not in names  # retired
    assert "metricool_csv_format" not in names  # retired
    assert "recording_date_per_lf" in names


def test_new_checks_registered_in_skip_upload_mode(tmp_path: Path) -> None:
    project_dir = tmp_path / PROJECT
    project_dir.mkdir(parents=True)
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    report = v.run_validation(
        project=PROJECT,
        output_base=tmp_path,
        snapshot_dir=snapshot,
        state_dir=state_dir,
        run_start=datetime.now(tz=timezone.utc),
        mode=v.Mode.SKIP_UPLOAD_AUDIT,
    )
    names = {c.name for c in report.checks}
    assert "metricool_text_fidelity" not in names  # retired
    assert "metricool_fresh" not in names  # retired
    assert "metricool_csv_format" not in names  # retired
    assert "recording_date_per_lf" in names
