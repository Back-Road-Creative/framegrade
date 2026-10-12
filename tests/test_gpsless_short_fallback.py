"""Regression: a GPS-less short must NOT inherit the parent LF's
GPS-derived "A to B via Road" title.

When a short's clip GPS is empty on a GPS-less day, the burn used to keep the
parent LF's ``per_lf_ost_text`` — its "A to B" title, over footage that is
neither place. The refined logic suppresses that A→B title and anchors on (in
order): an operator override (verbatim), else the parent county, else the
drive-wide location (a short always carries a title). It never burns the
drive's destination POI: that is where the drive went, not where the clip was
filmed. On a day WITH GPS a clip with no fix of its own is never planned, and
fails before any burn —
see ``test_short_gps_gap_never_borrows_a_place``. GPS-bearing shorts are
untouched — see ``test_short_ost_per_lf_fallback``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from gomoveshift.models import ResolvedLocation
from gomoveshift.pipeline import day_pipeline
from gomoveshift.pipeline.ffmpeg_filters import shorten_ost_for_shorts
from _state_helpers import make_state

_PARENT_AB_TITLE = "Dingle to Gortmore via R559, County Kerry"


def _make_short_range(rank: int, start_sec: float, end_sec: float):
    return SimpleNamespace(rank=rank, start_sec=start_sec, end_sec=end_sec)


def _make_lf_segment(file_index: int, start: float, end: float):
    return SimpleNamespace(source_file_index=file_index, start_time=start, end_time=end)


def _make_lf_plan(index: int, segments):
    return SimpleNamespace(
        index=index,
        segments=list(segments),
        montage_peaks=[],
        total_duration=sum(s.end_time - s.start_time for s in segments),
    )


def _make_day_batch(date_str: str, episode_durations):
    episodes = [
        SimpleNamespace(index=i + 1, concat_path=None, total_duration=float(d))
        for i, d in enumerate(episode_durations)
    ]
    return SimpleNamespace(date=date_str, episodes=episodes)


@pytest.fixture
def captured_short_ost_text():
    return []


@pytest.fixture
def patched_render_pipeline(tmp_path, captured_short_ost_text):
    """Patch heavy I/O so run_day_pipeline runs end-to-end on synthetic
    inputs, capturing the short OST text passed to render. NO gps_points_by_date
    is supplied anywhere, so every short's clip is GPS-less."""

    def _capture_short(*, source_video, output_path, start_time, duration, location, **kwargs):
        captured_short_ost_text.append({"location": location, "rank": output_path.stem})
        Path(output_path).touch()
        return True

    patches = [
        patch.object(day_pipeline, "encode_day_master", return_value=tmp_path / "master.mp4"),
        patch.object(
            day_pipeline, "extract_and_concat_segments", return_value=tmp_path / "seg.mp4"
        ),
        patch.object(day_pipeline, "prepend_montage", return_value=tmp_path / "wm.mp4"),
        patch.object(day_pipeline, "apply_ost", side_effect=lambda **kw: kw["output_path"]),
        patch.object(day_pipeline, "_render_short_ffmpeg", side_effect=_capture_short),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def _run(tmp_path, **state_kwargs):
    lf_plan = _make_lf_plan(index=1, segments=[_make_lf_segment(0, 0.0, 1800.0)])
    day_batch = _make_day_batch("2025-09-14", episode_durations=[1800.0])
    short_ranges = [_make_short_range(rank=1, start_sec=30.0, end_sec=68.0)]
    per_lf_ost_text = state_kwargs.pop("per_lf_ost_text", None)
    per_lf_locations = state_kwargs.pop("per_lf_locations", None)
    with patch.object(day_pipeline, "plan_shorts", return_value=short_ranges):
        day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=np.array([[0.5] * 11] * 1800, dtype=np.float32),
            output_dir=tmp_path / "out",
            location="Dingle Peninsula, County Kerry",
            config=None,
            lf_plans=[lf_plan],
            per_lf_locations=per_lf_locations,
            per_lf_ost_text=per_lf_ost_text,
            state=make_state(ost_audit_confirmed=True, **state_kwargs),
        )


def test_gpsless_short_burns_county_never_destination(
    tmp_path, captured_short_ost_text, patched_render_pipeline
):
    """(a) parent A→B title + parent .destination → burns the county. The
    destination is where the drive went, not where this clip was filmed."""
    _run(
        tmp_path,
        per_lf_ost_text={1: _PARENT_AB_TITLE},
        per_lf_locations={1: "County Kerry"},
        per_lf_resolved_location={1: ResolvedLocation(destination="Slea Head")},
    )
    assert len(captured_short_ost_text) == 1
    burned = captured_short_ost_text[0]["location"]
    assert burned == shorten_ost_for_shorts("County Kerry")
    assert "Slea Head" not in burned  # the drive's destination is never borrowed
    assert "to Gortmore" not in burned  # A→B endpoints suppressed


def test_gpsless_short_burns_operator_override_verbatim(
    tmp_path, captured_short_ost_text, patched_render_pipeline
):
    """(b) operator --per-lf-location override → burns the override verbatim."""
    _run(
        tmp_path,
        per_lf_ost_text={1: _PARENT_AB_TITLE},
        per_lf_resolved_location={1: ResolvedLocation(destination="Slea Head")},
        per_lf_overrides={1: "Dingle Peninsula toward Gortmore, County Kerry"},
    )
    assert len(captured_short_ost_text) == 1
    assert captured_short_ost_text[0]["location"] == shorten_ost_for_shorts(
        "Dingle Peninsula toward Gortmore, County Kerry"
    )


def test_gpsless_short_burns_county_when_no_destination(
    tmp_path, captured_short_ost_text, patched_render_pipeline
):
    """(c) parent A→B title, no destination, has county → burns the county."""
    _run(
        tmp_path,
        per_lf_ost_text={1: _PARENT_AB_TITLE},
        per_lf_locations={1: "County Kerry"},
    )
    assert len(captured_short_ost_text) == 1
    burned = captured_short_ost_text[0]["location"]
    assert burned == shorten_ost_for_shorts("County Kerry")
    assert "to Gortmore" not in burned  # A→B endpoints suppressed


def test_gpsless_short_falls_back_to_drive_location_when_no_anchor(
    tmp_path, captured_short_ost_text, patched_render_pipeline
):
    """(d) parent A→B title, no destination/county/override → falls back to the
    drive-wide location. Policy: a short
    always carries a title — the broad region label is used rather than an
    empty overlay, and the A→B endpoints are still suppressed."""
    _run(tmp_path, per_lf_ost_text={1: _PARENT_AB_TITLE}, per_lf_locations=None)
    assert len(captured_short_ost_text) == 1
    burned = captured_short_ost_text[0]["location"]
    assert burned == shorten_ost_for_shorts("Dingle Peninsula, County Kerry")
    assert burned.strip()  # never a title-less omit
    assert "to Gortmore" not in burned  # A→B endpoints still suppressed
