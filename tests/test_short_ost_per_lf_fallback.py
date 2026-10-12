"""TDD test for short OST text drift fix.

Bug: short OST burn showed
"Dingle Peninsula, County Kerry" (drive-wide --location) while the
parent LF's OST correctly showed "Dingle Peninsula toward Gortmore,
County Kerry" (per-LF override). Three-surface divergence violates
the alignment rule: all surfaces (LF OST,
short OST, title, thumbnail) must derive from one source
(``state["per_lf_ost_text"]``).

This test exercises ``run_day_pipeline``'s short rendering loop: when
``per_lf_ost_text`` is supplied AND a short's parent LF is in the map,
the short's OST text MUST come from ``per_lf_ost_text[parent_lf_idx]``,
NOT the drive-wide ``location``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.pipeline import day_pipeline
from gomoveshift.pipeline.ffmpeg_filters import shorten_ost_for_shorts
from _state_helpers import make_state


def _make_short_range(rank: int, start_sec: float, end_sec: float):
    """Minimal ShortRange-like object."""
    return SimpleNamespace(
        rank=rank,
        start_sec=start_sec,
        end_sec=end_sec,
    )


def _make_lf_segment(file_index: int, start: float, end: float):
    return SimpleNamespace(
        source_file_index=file_index,
        start_time=start,
        end_time=end,
    )


def _make_lf_plan(index: int, segments):
    return SimpleNamespace(
        index=index,
        segments=list(segments),
        montage_peaks=[],
        total_duration=sum(s.end_time - s.start_time for s in segments),
    )


def _make_day_batch(date_str: str, episode_durations):
    episodes = [
        SimpleNamespace(
            index=i + 1,
            concat_path=None,
            total_duration=float(d),
        )
        for i, d in enumerate(episode_durations)
    ]
    return SimpleNamespace(date=date_str, episodes=episodes)


@pytest.fixture
def captured_short_ost_text():
    """Capture the ``location`` arg passed to render_short_ffmpeg."""
    return []


@pytest.fixture
def patched_render_pipeline(tmp_path, captured_short_ost_text):
    """Patch all heavy I/O so run_day_pipeline runs end-to-end on
    synthetic inputs, capturing the short OST text passed to render."""

    def _capture_short(*, source_video, output_path, start_time, duration, location, **kwargs):
        captured_short_ost_text.append({"location": location, "rank": output_path.stem})
        # Touch the output file so downstream logic sees it
        from pathlib import Path

        Path(output_path).touch()
        return True

    patches = [
        patch.object(day_pipeline, "encode_day_master", return_value=tmp_path / "master.mp4"),
        patch.object(
            day_pipeline,
            "extract_and_concat_segments",
            return_value=tmp_path / "seg.mp4",
        ),
        patch.object(day_pipeline, "prepend_montage", return_value=tmp_path / "wm.mp4"),
        patch.object(day_pipeline, "apply_ost", side_effect=lambda **kw: kw["output_path"]),
        patch.object(
            day_pipeline,
            "_render_short_ffmpeg",
            side_effect=_capture_short,
        ),
        # plan_shorts is the source of short_ranges; we override its return
        # in each test rather than patch globally.
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


def test_short_ost_uses_per_lf_text_when_supplied(
    tmp_path, captured_short_ost_text, patched_render_pipeline
):
    """short OST falls back to per_lf_ost_text[parent_lf_idx], NOT drive-wide location."""
    # 1 LF spanning 0..1800s; 2 shorts at 30s and 600s — both inside the LF.
    lf_plan = _make_lf_plan(index=1, segments=[_make_lf_segment(0, 0.0, 1800.0)])
    day_batch = _make_day_batch("2025-09-14", episode_durations=[1800.0])

    short_ranges = [
        _make_short_range(rank=1, start_sec=30.0, end_sec=68.0),
        _make_short_range(rank=2, start_sec=600.0, end_sec=638.0),
    ]

    with patch.object(day_pipeline, "plan_shorts", return_value=short_ranges):
        day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=np.array([[0.5] * 11] * 1800, dtype=np.float32),
            output_dir=tmp_path / "out",
            location="Dingle Peninsula, County Kerry",
            config=None,
            lf_plans=[lf_plan],
            per_lf_locations={1: "Dingle Peninsula, County Kerry"},
            per_lf_ost_text={1: "Dingle Peninsula toward Gortmore, County Kerry"},
            # These shorts are GPS-less (no gps_points_by_date). Model the
            # production operator ``--per-lf-location`` override that makes the
            # per-LF text AUTHORITATIVE — it propagates verbatim to the short
            # (the refined per-LF fallback logic suppresses only a GPS-DERIVED A→B title,
            # never an operator override). The expected burned value is unchanged.
            state=make_state(
                ost_audit_confirmed=True,
                per_lf_overrides={1: "Dingle Peninsula toward Gortmore, County Kerry"},
            ),
        )

    assert len(captured_short_ost_text) == 2, f"expected 2 shorts, got {captured_short_ost_text}"
    # The short OST is now shortened for the 9:16 mobile frame.
    # Intent unchanged — it must derive from per_lf_ost_text, never the
    # drive-wide location — but the burned text is the shortened form.
    expected = shorten_ost_for_shorts("Dingle Peninsula toward Gortmore, County Kerry")
    assert expected != "Dingle Peninsula, County Kerry"  # distinct from drive-wide
    for record in captured_short_ost_text:
        assert record["location"] == expected, (
            f"short {record['rank']} OST text {record['location']!r} did not use per_lf_ost_text — "
            "fell back to drive-wide --location"
        )


def test_short_ost_falls_back_to_per_lf_locations_when_no_ost_text(
    tmp_path, captured_short_ost_text, patched_render_pipeline
):
    """When per_lf_ost_text is absent, falls back to per_lf_locations[parent_lf_idx]."""
    lf_plan = _make_lf_plan(index=1, segments=[_make_lf_segment(0, 0.0, 1800.0)])
    day_batch = _make_day_batch("2025-09-14", episode_durations=[1800.0])

    short_ranges = [_make_short_range(rank=1, start_sec=30.0, end_sec=68.0)]

    with patch.object(day_pipeline, "plan_shorts", return_value=short_ranges):
        day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=np.array([[0.5] * 11] * 1800, dtype=np.float32),
            output_dir=tmp_path / "out",
            location="Drive Wide Default",
            config=None,
            lf_plans=[lf_plan],
            per_lf_locations={1: "Per LF Location Override"},
            per_lf_ost_text=None,
            state=make_state(ost_audit_confirmed=True),
        )

    assert len(captured_short_ost_text) == 1
    assert captured_short_ost_text[0]["location"] == "Per LF Location Override"


def test_short_ost_uses_drive_wide_when_no_per_lf_data(
    tmp_path, captured_short_ost_text, patched_render_pipeline
):
    """No per_lf_ost_text and no per_lf_locations → drive-wide location (preserved legacy)."""
    lf_plan = _make_lf_plan(index=1, segments=[_make_lf_segment(0, 0.0, 1800.0)])
    day_batch = _make_day_batch("2025-09-14", episode_durations=[1800.0])

    short_ranges = [_make_short_range(rank=1, start_sec=30.0, end_sec=68.0)]

    with patch.object(day_pipeline, "plan_shorts", return_value=short_ranges):
        day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=np.array([[0.5] * 11] * 1800, dtype=np.float32),
            output_dir=tmp_path / "out",
            location="Drive Wide",
            config=None,
            lf_plans=[lf_plan],
            per_lf_locations=None,
            per_lf_ost_text=None,
            state=make_state(ost_audit_confirmed=True),
        )

    assert len(captured_short_ost_text) == 1
    assert captured_short_ost_text[0]["location"] == "Drive Wide"
