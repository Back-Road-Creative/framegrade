"""day_pipeline routes short OST composition through
``trim_gps_to_clip_timeline``.

Even for single-segment shorts the code path goes through the helper
— no one-offs. The test spies on ``trim_gps_to_clip_timeline`` and
asserts it was invoked with the short's ``(start_sec, end_sec)`` range.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from gomoveshift.pipeline import day_pipeline as dp
from gomoveshift.pipeline.day_grouper import DayBatch
from gomoveshift.pipeline.short_planner import ShortRange


@dataclass
class _Ep:
    index: int
    concat_path: Path
    total_duration: float
    date: str
    creation_time: object = None


def _make_scores(n: int) -> np.ndarray:
    arr = np.zeros((n, 9), dtype=np.float64)
    arr[:, 0] = np.arange(n, dtype=np.float64)
    arr[:, 1] = 0.8
    return arr


def test_shorts_ost_respects_trim(tmp_path, monkeypatch):
    """When ``gps_points_by_date`` is supplied, the shorts loop calls
    ``trim_gps_to_clip_timeline`` with each short's (start, end) range."""
    date = "2026-04-22"
    ep = _Ep(
        index=1,
        concat_path=tmp_path / "ep0.mp4",
        total_duration=300.0,
        date=date,
    )
    ep.concat_path.write_bytes(b"x")
    day_batch = DayBatch(date=date, episodes=[ep])
    day_scores = _make_scores(300)
    output_dir = tmp_path / "out"

    # Stub heavy dependencies.
    monkeypatch.setattr(
        dp,
        "encode_day_master",
        lambda *a, **kw: _stub_master(tmp_path / "out" / f"day_{date}" / "day_master.mp4"),
    )
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda *a, **kw: [(0.0, 300.0)])
    monkeypatch.setattr(dp, "extract_and_concat_segments", lambda *a, **kw: None)
    monkeypatch.setattr(dp, "prepend_montage", lambda *a, **kw: None)
    monkeypatch.setattr(dp, "purge_day_intermediates", lambda *a, **kw: None)

    # Two shorts with distinct ranges (day-timeline seconds).
    short_ranges = [
        ShortRange(start_sec=10.0, end_sec=30.0, integrated_score=0.9, rank=1),
        ShortRange(start_sec=100.0, end_sec=120.0, integrated_score=0.85, rank=2),
    ]
    monkeypatch.setattr(dp, "plan_shorts", lambda *a, **kw: short_ranges)

    # Stub apply_ost to produce dummy output paths.
    def _apply_ost(input_path, output_path, location, layout, **kw):
        output_path.write_bytes(b"x")
        return output_path

    monkeypatch.setattr(dp, "apply_ost", _apply_ost)

    # Shorts now use render_short_ffmpeg (single-pass crop+scale+OST).
    def _render(source_video, output_path, start_time, duration, location, **kw):
        output_path.write_bytes(b"x")
        return True

    monkeypatch.setattr(dp, "_render_short_ffmpeg", _render)

    # Spy on trim_gps_to_clip_timeline.
    trim_calls: list[tuple] = []
    import gomoveshift.pipeline.gps_quality as gq

    real_trim = gq.trim_gps_to_clip_timeline

    def _spy_trim(points, included, **kw):
        trim_calls.append((list(points), list(included) if included else included))
        return real_trim(points, included, **kw)

    monkeypatch.setattr(gq, "trim_gps_to_clip_timeline", _spy_trim)
    monkeypatch.setattr(dp, "trim_gps_to_clip_timeline", _spy_trim, raising=False)

    # Moving trace (lat advances ~11 m/s): a real short is cut from moving
    # footage; a parked fixture would now trip the stop-free gate.
    gps_points = [(float(t), 40.0 + t * 0.0001, -122.0) for t in range(0, 300, 5)]
    dp.run_day_pipeline(
        day_batch=day_batch,
        day_scores=day_scores,
        output_dir=output_dir,
        location="Somewhere, TestCounty",
        config=None,
        lf_plans=[],
        gps_points_by_date={date: gps_points},
        # Fresh --ost-audit-confirmed bypasses the shorts pre-burn gate.
        state={"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
    )

    # trim helper should have been invoked once per short, with the
    # short's (start, end) range as the single included segment.
    included_args = [call[1] for call in trim_calls]
    assert [(10.0, 30.0)] in included_args
    assert [(100.0, 120.0)] in included_args


def _stub_master(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path
