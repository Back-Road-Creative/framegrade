"""Regression test: double remap of segment indices.

``_partition_lf_plans_by_date`` mutates ``seg.source_file_index`` on
the live LF segments from drive-global to day-local. Prior to this
fix, ``run_day_pipeline`` then called
``remap_segments_drive_to_day_local`` a second time on the SAME
objects, which for any day at non-zero drive offset produced
``IndexError`` → silent LF drop (logged as warning, no raise).

The failing shape was day 3 of a multi-day drive: a single episode at
drive-global index 5. After partition, seg.source_file_index = 0
(day-local). The second remap with ``day_episode_indices = [5]``
builds ``drive_to_day = {5: 0}``, looks up ``0`` (day-local value
treated as drive-global), doesn't find it, raises.

This test pins the contract: after ``_partition_lf_plans_by_date``,
segments and montage peaks carry day-local indices, and calling the
legacy second remap on them raises ``IndexError`` (proving the bug
existed). The fix removes the second remap call so day-local segments
flow straight into ``extract_and_concat_segments``.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from gomoveshift.models import LFSegment, RegularLFPlan
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _partition_lf_plans_by_date,
)
from keycut import remap_segment_indices

from gomoveshift.pipeline.montage_builder import to_cut_segment
from _state_helpers import make_state


def _seg(drive_global_idx: int, start: float = 0.0, end: float = 10.0) -> LFSegment:
    return LFSegment(
        source_file_index=drive_global_idx,
        source_path=Path(f"/tmp/ep{drive_global_idx}.mp4"),
        start_time=start,
        end_time=end,
        original_start_time=start,
        original_end_time=end,
        composite_score=0.8,
        gps_center_lat=0.0,
        gps_center_lon=0.0,
        terrain_type="coastal",
        contiguous_with_previous=False,
    )


def _four_day_fixture():
    """4-day layout: [0,1] [2,3,4] [5] [6,7]."""
    episode_files = [f"/tmp/ep{i}.mp4" for i in range(8)]
    episode_dates = [
        "2026-04-01",
        "2026-04-01",
        "2026-04-02",
        "2026-04-02",
        "2026-04-02",
        "2026-04-03",  # Day 3 — single episode at drive idx 5
        "2026-04-04",
        "2026-04-04",
    ]
    day_batches = []
    drive_idx_by_day = {
        "2026-04-01": [0, 1],
        "2026-04-02": [2, 3, 4],
        "2026-04-03": [5],
        "2026-04-04": [6, 7],
    }
    for date, idxs in drive_idx_by_day.items():
        episodes = [
            SimpleNamespace(
                index=drive_i + 1,  # 1-based per ingest convention
                concat_path=Path(episode_files[drive_i]),
                date=date,
                total_duration=60.0,
                gpx_path=None,
                creation_time=None,
            )
            for drive_i in idxs
        ]
        day_batches.append(SimpleNamespace(date=date, episodes=episodes))
    return episode_files, episode_dates, day_batches


def test_day3_single_episode_partition_produces_day_local():
    """Day 3 has one episode at drive-global idx 5.
    After partition, the LF's segment must carry source_file_index=0
    (day-local), not 5 (drive-global)."""
    episode_files, episode_dates, day_batches = _four_day_fixture()
    lf = RegularLFPlan(
        index=1,
        segments=[_seg(5)],
        montage_peaks=[_seg(5, start=2.0, end=4.0)],
        shorts_target=7,
        region="ireland",
        total_duration=10.0,
    )
    drive_plan = SimpleNamespace(regular_lfs=[lf])
    by_date = _partition_lf_plans_by_date(
        drive_plan,
        episode_files=episode_files,
        episode_dates=episode_dates,
        day_batches=day_batches,
    )
    assert "2026-04-03" in by_date
    day3_lf = by_date["2026-04-03"][0]
    assert day3_lf.segments[0].source_file_index == 0, (
        "partition must remap drive-global 5 → day-local 0 for Day 3's single episode"
    )
    assert day3_lf.montage_peaks[0].source_file_index == 0, (
        "partition must also remap montage peaks to day-local"
    )


def test_legacy_second_remap_would_have_raised_on_day3():
    """Document the exact shape of the double-remap bug.

    After partition, Day 3's seg.source_file_index = 0 (day-local).
    ``run_day_pipeline`` formerly built day_episode_indices = [5]
    (Episode.index 6 − 1) and called
    ``remap_segments_drive_to_day_local(segments, [5])``. That lookup
    treats 0 as drive-global; 0 is not in {5: 0}; IndexError.

    The fix removes that second call. This test pins the legacy bug
    so it cannot silently return by accident.
    """
    episode_files, episode_dates, day_batches = _four_day_fixture()
    lf = RegularLFPlan(
        index=1,
        segments=[_seg(5)],
        montage_peaks=[],
        shorts_target=7,
        region="ireland",
        total_duration=10.0,
    )
    drive_plan = SimpleNamespace(regular_lfs=[lf])
    by_date = _partition_lf_plans_by_date(
        drive_plan,
        episode_files=episode_files,
        episode_dates=episode_dates,
        day_batches=day_batches,
    )
    day3_segs = by_date["2026-04-03"][0].segments
    day3_episode_indices_drive_global = [5]  # Day 3's sole episode at drive idx 5

    with pytest.raises(IndexError, match="not among this master's parts"):
        remap_segment_indices(
            [to_cut_segment(s) for s in day3_segs],
            day3_episode_indices_drive_global,
        )


def test_day3_run_day_pipeline_does_not_drop_lf_after_partition(tmp_path, monkeypatch):
    """End-to-end: after partition, run_day_pipeline must NOT double-remap
    and must reach extract_and_concat_segments for Day 3's single-episode
    LF. On master, the second remap raised IndexError → LF silently
    skipped → extract never called. With the fix, extract is called
    exactly once with day-local segment indices."""
    from gomoveshift.pipeline import day_pipeline

    episode_files, episode_dates, day_batches = _four_day_fixture()
    day3_batch = next(b for b in day_batches if b.date == "2026-04-03")

    lf = RegularLFPlan(
        index=1,
        segments=[_seg(5)],
        montage_peaks=[_seg(5, start=2.0, end=4.0)],
        shorts_target=7,
        region="ireland",
        total_duration=10.0,
    )
    drive_plan = SimpleNamespace(regular_lfs=[lf])
    by_date = _partition_lf_plans_by_date(
        drive_plan,
        episode_files=episode_files,
        episode_dates=episode_dates,
        day_batches=day_batches,
    )
    day3_lfs = by_date["2026-04-03"]

    extract_calls: list[dict] = []

    def fake_extract(**kwargs):
        extract_calls.append(kwargs)
        out = kwargs.get("output_path")
        if out is not None:
            out.write_bytes(b"\x00")
        return out

    (tmp_path / "ep5.mp4").write_bytes(b"\x00")
    for ep in day3_batch.episodes:
        ep.concat_path = tmp_path / "ep5.mp4"

    monkeypatch.setattr(day_pipeline, "encode_day_master", lambda *a, **k: tmp_path / "master.mp4")
    monkeypatch.setattr(day_pipeline, "extract_and_concat_segments", fake_extract)
    monkeypatch.setattr(day_pipeline, "prepend_montage", lambda **k: tmp_path / "with_montage.mp4")
    monkeypatch.setattr(day_pipeline, "apply_ost", lambda **k: k.get("output_path"))
    monkeypatch.setattr(day_pipeline, "plan_shorts", lambda *a, **k: [])
    monkeypatch.setattr(day_pipeline, "compute_day_file_boundaries", lambda *a, **k: [(0.0, 60.0)])
    (tmp_path / "master.mp4").write_bytes(b"\x00")
    (tmp_path / "with_montage.mp4").write_bytes(b"\x00")

    day_pipeline.run_day_pipeline(
        day_batch=day3_batch,
        day_scores=None,
        output_dir=tmp_path / "out",
        location="Test",
        config=None,
        lf_plans=day3_lfs,
        state=make_state(ost_audit_confirmed=True),
    )

    assert len(extract_calls) == 1, (
        f"Expected extract_and_concat_segments called once for Day 3 single-episode "
        f"LF; got {len(extract_calls)}. Zero calls = the double-remap silent LF drop."
    )
    segs_passed = extract_calls[0]["segments"]
    assert segs_passed[0].source_file_index == 0, (
        f"Segment reaching extract must carry day-local index 0, got "
        f"{segs_passed[0].source_file_index}"
    )


def test_day1_accidentally_worked_under_legacy_double_remap():
    """Day 1 (episodes at drive-global [0, 1]) happens to have
    coincident drive-global and day-local indices — which is why
    the double-remap bug is invisible on Day-1-shaped data.

    After partition, Day 1's seg.source_file_index = 0 (still).
    Legacy second remap with day_episode_indices = [0, 1] builds
    drive_to_day = {0: 0, 1: 1}; lookup of 0 succeeds; no error.

    This test documents why Day-1-shaped data cannot catch the bug
    — the accident keeps the regression test honest about scope.
    """
    episode_files, episode_dates, day_batches = _four_day_fixture()
    lf = RegularLFPlan(
        index=1,
        segments=[_seg(0)],
        montage_peaks=[],
        shorts_target=7,
        region="ireland",
        total_duration=10.0,
    )
    drive_plan = SimpleNamespace(regular_lfs=[lf])
    by_date = _partition_lf_plans_by_date(
        drive_plan,
        episode_files=episode_files,
        episode_dates=episode_dates,
        day_batches=day_batches,
    )
    day1_segs = by_date["2026-04-01"][0].segments
    # No raise — Day 1 accident. keycut's remap returns a NEW list and leaves
    # the input untouched (it no longer mutates ``source_file_index``), so the
    # accident now shows up in the RETURN value rather than in place.
    remapped = remap_segment_indices([to_cut_segment(s) for s in day1_segs], [0, 1])
    assert remapped[0].source_index == 0
    assert day1_segs[0].source_file_index == 0
