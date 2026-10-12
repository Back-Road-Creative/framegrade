"""Shorts GPS trim is wired through ``run_all_days`` end-to-end.

``trim_gps_to_clip_timeline`` was added to ``run_day_pipeline``'s shorts OST loop,
but two gaps made it a no-op in production:

1.  ``run_all_days`` didn't forward ``gps_points_by_date`` to the
    per-day call, so ``day_gps_points`` was always ``None`` and the
    trim always returned ``[]``.
2.  Even when points survived the trim, the short OST text still
    defaulted to the drive-wide ``location`` — the trimmed points
    were computed but never consumed.

This test drives a multi-day batch through ``run_all_days`` with real
per-date GPS + a ``short_ost_text_fn`` callback and asserts:

* ``run_all_days`` forwards the per-date GPS to ``run_day_pipeline``;
* the trim helper receives non-empty points for each short;
* the short's OST text is derived from the trimmed points (NOT the
  drive-wide fallback string).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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


def _stub_master(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_run_all_days_forwards_gps_and_consumes_trimmed_points(tmp_path, monkeypatch):
    """``run_all_days`` forwards per-date GPS to each day, and the
    shorts OST text is derived from the trimmed points — NOT from the
    drive-wide ``location`` fallback."""
    date = "2026-04-22"
    ep = _Ep(
        index=1,
        concat_path=tmp_path / "ep0.mp4",
        total_duration=300.0,
        date=date,
    )
    ep.concat_path.write_bytes(b"x")
    day_batch = DayBatch(date=date, episodes=[ep])

    drive_scores = _make_scores(300)
    file_boundaries = [(0.0, 300.0)]
    output_base = tmp_path / "out"

    # Stub heavy dependencies inside day_pipeline.
    monkeypatch.setattr(
        dp,
        "encode_day_master",
        lambda *a, **kw: _stub_master(output_base / f"day_{date}" / "day_master.mp4"),
    )
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda *a, **kw: [(0.0, 300.0)])
    monkeypatch.setattr(dp, "extract_and_concat_segments", lambda *a, **kw: None)
    monkeypatch.setattr(dp, "prepend_montage", lambda *a, **kw: None)
    monkeypatch.setattr(dp, "purge_day_intermediates", lambda *a, **kw: None)

    # Two shorts with distinct non-overlapping day-timeline ranges.
    short_ranges = [
        ShortRange(start_sec=10.0, end_sec=30.0, integrated_score=0.9, rank=1),
        ShortRange(start_sec=100.0, end_sec=120.0, integrated_score=0.85, rank=2),
    ]
    monkeypatch.setattr(dp, "plan_shorts", lambda *a, **kw: short_ranges)

    # apply_ost just touches files and captures the OST text.
    captured_locations: list[str] = []

    def _apply_ost(input_path, output_path, location, layout, **kw):
        output_path.write_bytes(b"x")
        if layout == "short":
            captured_locations.append(location)
        return output_path

    monkeypatch.setattr(dp, "apply_ost", _apply_ost)

    # Shorts now go through render_short_ffmpeg (single-pass
    # crop+scale+OST). Capture location from this path now that
    # apply_ost(layout="short") is no longer in the shorts loop.
    def _render(source_video, output_path, start_time, duration, location, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"x")
        captured_locations.append(location)
        return True

    monkeypatch.setattr(dp, "_render_short_ffmpeg", _render, raising=False)

    # Per-date GPS in day-master timeline seconds: (t_sec, lat, lon).
    # Short #1 covers 10-30 → points 10,15,20,25,30 land inside.
    # Short #2 covers 100-120 → points 100,105,110,115,120 land inside.
    gps_points = [(float(t), 40.0 + t * 0.0001, -80.0) for t in range(0, 300, 5)]

    # Callback that derives per-short OST text from the trimmed points.
    # A production implementation would reverse-geocode the centroid;
    # here we return a marker string encoding the centroid so the test
    # can assert the callback was fed real trimmed data.
    def _short_ost_text(trimmed_points, short_range):
        if not trimmed_points:
            return None
        lats = [p[1] for p in trimmed_points]
        return f"trimmed:{short_range.rank}:{len(lats)}:{min(lats):.4f}"

    # Drive-wide drive location — this is what a broken wiring would
    # leak into every short's OST.
    drive_wide_location = "South Carolina to North Carolina"

    results = dp.run_all_days(
        day_batches=[day_batch],
        drive_scores=drive_scores,
        file_boundaries=file_boundaries,
        output_base=output_base,
        location=drive_wide_location,
        config=None,
        episode_paths=[ep.concat_path],
        lf_plans_by_date={date: []},
        gps_points_by_date={date: gps_points},
        short_ost_text_fn=_short_ost_text,
        # Fresh --ost-audit-confirmed bypasses the shorts pre-burn gate.
        state={"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
    )

    assert len(results) == 1
    # Two shorts, both should have OST text derived from the trimmed
    # points — NOT the drive-wide fallback. This is the gap the wiring closes.
    assert len(captured_locations) == 2
    for loc in captured_locations:
        assert loc != drive_wide_location, (
            f"short OST text leaked drive-wide location {drive_wide_location!r}; "
            f"the trim wiring is a no-op"
        )
        assert loc.startswith("trimmed:"), (
            f"short OST text {loc!r} does not reflect the trimmed points — "
            "short_ost_text_fn was either not called or called with empty input"
        )


def test_run_all_days_without_gps_falls_back_to_location(tmp_path, monkeypatch):
    """Backward-compat: when ``gps_points_by_date`` is None, the shorts
    OST text still falls back to the drive-wide ``location`` — no
    regression for call sites that haven't been updated yet."""
    date = "2026-04-22"
    ep = _Ep(
        index=1,
        concat_path=tmp_path / "ep0.mp4",
        total_duration=300.0,
        date=date,
    )
    ep.concat_path.write_bytes(b"x")
    day_batch = DayBatch(date=date, episodes=[ep])

    monkeypatch.setattr(
        dp,
        "encode_day_master",
        lambda *a, **kw: _stub_master(tmp_path / "out" / f"day_{date}" / "day_master.mp4"),
    )
    monkeypatch.setattr(dp, "compute_day_file_boundaries", lambda *a, **kw: [(0.0, 300.0)])
    monkeypatch.setattr(dp, "extract_and_concat_segments", lambda *a, **kw: None)
    monkeypatch.setattr(dp, "prepend_montage", lambda *a, **kw: None)
    monkeypatch.setattr(dp, "purge_day_intermediates", lambda *a, **kw: None)
    monkeypatch.setattr(
        dp,
        "plan_shorts",
        lambda *a, **kw: [ShortRange(start_sec=10.0, end_sec=30.0, integrated_score=0.9, rank=1)],
    )

    captured: list[str] = []

    def _apply_ost(input_path, output_path, location, layout, **kw):
        output_path.write_bytes(b"x")
        if layout == "short":
            captured.append(location)
        return output_path

    monkeypatch.setattr(dp, "apply_ost", _apply_ost)

    def _render(source_video, output_path, start_time, duration, location, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"x")
        captured.append(location)
        return True

    monkeypatch.setattr(dp, "_render_short_ffmpeg", _render, raising=False)

    dp.run_all_days(
        day_batches=[day_batch],
        drive_scores=_make_scores(300),
        file_boundaries=[(0.0, 300.0)],
        output_base=tmp_path / "out",
        location="Drive-Wide Location",
        config=None,
        episode_paths=[ep.concat_path],
        lf_plans_by_date={date: []},
        # Fresh --ost-audit-confirmed bypasses the shorts pre-burn gate.
        state={"ost_audit_confirmed": True, "ost_audit_confirmed_this_run": True},
        # No gps_points_by_date, no short_ost_text_fn.
    )

    assert captured == ["Drive-Wide Location"]
