"""Tests for day_pipeline, the per-day driver of the day-first pipeline.

The day pipeline is the glue that wires together all the day-first modules:
  day_grouper      →  split episodes into DayBatch objects
  day_master_encoder → one encode pass per day → day master
  regular_lf_planner → contiguous best ranges of the day's timeline
  short_planner      → rank-ordered short clips
  ost_applier        → per-output OST burn-in

Each day produces a DayArtifacts record with the master path + regular LF
paths + short paths (all OST-applied, ready for Phase D publish/upload).

Tests use mocked subprocess + module-level patches to avoid any real
ffmpeg / disk / encoding work. The goal is to verify the wiring: the
right modules get called in the right order with the right inputs.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.pipeline.day_grouper import DayBatch
from _state_helpers import make_state


# --- fixtures ---------------------------------------------------------

NUM_COLUMNS = 9
COL_TIMESTAMP = 0
COL_COMPOSITE = 1


COL_MOTION_ENERGY = 2


_DECIDED: dict = {}


@pytest.fixture(autouse=True)
def _decided_shorts(monkeypatch):
    """Stand in for Waypost's decision store: 14 per LF unless a test sets it."""
    from gomoveshift.pipeline import volume_source

    _DECIDED.clear()
    _DECIDED["value"] = 14

    def _fake(db=None):
        if _DECIDED.get("error"):
            raise volume_source.VolumeSourceUnreadableError("VOLUME_SOURCE_UNREADABLE")
        return volume_source.ShortsDecision(_DECIDED["value"], "policy", "TEST")

    monkeypatch.setattr(volume_source, "shorts_per_lf", _fake)


def _make_scores(duration_sec: int, score: float = 0.5, motion: float = 0.05) -> np.ndarray:
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, COL_TIMESTAMP] = np.arange(duration_sec, dtype=np.float64)
    arr[:, COL_COMPOSITE] = score
    # Real scorer output always carries per-second motion energy; keep it above
    # the stop-detection optical floor so the default-on per-day stop mask
    # doesn't read this synthetic footage as a full-day stop.
    arr[:, COL_MOTION_ENERGY] = motion
    return arr


@dataclass
class _FakeEpisode:
    index: int
    concat_path: Path
    total_duration: float
    date: str
    creation_time: object = None


def _make_day_batch(
    date: str,
    episode_count: int,
    per_ep_duration: float = 600.0,
    start_index: int = 0,
):
    """Build a DayBatch with globally-unique concat_paths.

    `start_index` is the drive-wide **0-based** position of the first
    episode. ``Episode.index`` is set **1-based** to match production
    (ingest sets ``enumerate(plan.episodes, 1)`` — see
    ``pipeline/ingest/__init__.py``). day_pipeline subtracts 1 to
    convert Episode.index → drive-wide 0-based position when remapping
    segment indices.
    """
    episodes = [
        _FakeEpisode(
            index=start_index + i + 1,
            concat_path=Path(f"/fake/ep_{start_index + i:03d}.mp4"),
            total_duration=per_ep_duration,
            date=date,
        )
        for i in range(episode_count)
    ]
    return DayBatch(date=date, episodes=episodes)


# --- slice_scores_for_day --------------------------------------------


class TestSliceScoresForDay:
    def test_slice_rebases_timestamps_to_zero(self):
        from gomoveshift.pipeline.day_pipeline import slice_scores_for_day

        # 2 hours of scores, slice out [3600, 7200) → expect 3600 rows starting at t=0
        scores = _make_scores(7200)
        day_slice = slice_scores_for_day(scores, 3600.0, 7200.0)
        assert day_slice.shape[0] == 3600
        assert day_slice[0, COL_TIMESTAMP] == 0.0
        assert day_slice[-1, COL_TIMESTAMP] == 3599.0

    def test_slice_composite_preserved(self):
        from gomoveshift.pipeline.day_pipeline import slice_scores_for_day

        scores = _make_scores(7200, score=0.77)
        day_slice = slice_scores_for_day(scores, 3600.0, 7200.0)
        assert np.all(day_slice[:, COL_COMPOSITE] == 0.77)

    def test_slice_empty_when_out_of_range(self):
        from gomoveshift.pipeline.day_pipeline import slice_scores_for_day

        scores = _make_scores(100)
        day_slice = slice_scores_for_day(scores, 500.0, 600.0)
        assert day_slice.shape[0] == 0

    def test_slice_half_open_interval(self):
        """[start, end) — the end-second is NOT included."""
        from gomoveshift.pipeline.day_pipeline import slice_scores_for_day

        scores = _make_scores(100)
        day_slice = slice_scores_for_day(scores, 10.0, 20.0)
        # t=10..19 inclusive → 10 rows
        assert day_slice.shape[0] == 10


# --- run_day_pipeline (single-day integration) ------------------------


class TestRunDayPipeline:
    def _patch_all_stages(self, tmp_path):
        """Helper: patch every heavy-duty call the day pipeline makes.

        Returns a dict of the mocks so individual tests can assert on them.
        """
        master_path = tmp_path / "out" / "day_2026-02-20" / "day_2026-02-20_master.mp4"
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_bytes(b"fake master")

        def _fake_ost(input_path, output_path, location, layout, **kwargs):
            output_path.write_bytes(b"fake ost")
            return output_path

        # The shorts loop calls render_short_ffmpeg (not apply_ost).
        def _fake_render(source_video, output_path, start_time, duration, location, **kwargs):
            from pathlib import Path

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"fake render short")
            return True

        mocks = {}
        mocks["encode_day_master"] = patch(
            "gomoveshift.pipeline.day_pipeline.encode_day_master",
            return_value=master_path,
        )
        mocks["apply_ost"] = patch(
            "gomoveshift.pipeline.day_pipeline.apply_ost",
            side_effect=_fake_ost,
        )
        mocks["_render_short_ffmpeg"] = patch(
            "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
            side_effect=_fake_render,
        )
        return master_path, mocks

    def test_single_day_produces_master_regulars_and_shorts(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-02-20", 3, per_ep_duration=1200.0)
        day_scores = _make_scores(3600, score=0.6)
        output_base = tmp_path / "out"

        master_path, mocks = TestRunDayPipeline()._patch_all_stages(tmp_path)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]

        with (
            mocks["encode_day_master"] as mock_encode,
            mocks["apply_ost"] as mock_ost,
            mocks["_render_short_ffmpeg"] as mock_render_short,
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
        ):
            artifacts = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test Location, SC",
                config=None,
                lf_plans=lf_plans,
                state=make_state(ost_audit_confirmed=True),
            )

        assert artifacts is not None
        assert artifacts.date == "2026-02-20"
        # Master OST is burned in the single-pass encode — master_path IS the final file
        assert artifacts.master_path.name == "day_2026-02-20_master.mp4"
        assert len(artifacts.regular_lf_paths) >= 1
        assert len(artifacts.short_paths) >= 1

        # Encode runs exactly once per day
        mock_encode.assert_called_once()
        # OST is applied to every regular LF (NOT master — burned in
        # single-pass; NOT shorts — they go through render_short_ffmpeg).
        assert mock_ost.call_count == len(artifacts.regular_lf_paths)
        # Shorts are produced by render_short_ffmpeg.
        assert mock_render_short.call_count == len(artifacts.short_paths)

    def test_ost_layout_per_output_type(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-02-20", 3, per_ep_duration=1200.0)
        day_scores = _make_scores(3600, score=0.5)
        output_base = tmp_path / "out"
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]

        _, mocks = self._patch_all_stages(tmp_path)
        with (
            mocks["encode_day_master"],
            mocks["apply_ost"] as mock_ost,
            mocks["_render_short_ffmpeg"],
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
        ):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test, SC",
                config=None,
                lf_plans=lf_plans,
                state=make_state(ost_audit_confirmed=True),
            )

        layouts_used = [call.kwargs.get("layout") for call in mock_ost.call_args_list]
        # The day master no longer burns OST (skip_ost=True); OST is
        # post-assembly only. Shorts are not OST'd via apply_ost —
        # render_short_ffmpeg composites the short's OST. So apply_ost only sees
        # layout="regular_lf".
        assert "regular_lf" in layouts_used
        assert "short" not in layouts_used

    def test_empty_day_batch_returns_none(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        day_batch = DayBatch(date="2026-02-20", episodes=[])
        result = day_pipeline.run_day_pipeline(
            day_batch=day_batch,
            day_scores=_make_scores(0),
            output_dir=tmp_path,
            location="",
            config=None,
            lf_plans=[],
            state=make_state(ost_audit_confirmed=True),
        )
        assert result is None

    def test_master_encode_failure_returns_none(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-02-20", 2, per_ep_duration=600.0)
        with patch(
            "gomoveshift.pipeline.day_pipeline.encode_day_master",
            return_value=None,
        ):
            result = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=_make_scores(1200),
                output_dir=tmp_path,
                location="",
                config=None,
                lf_plans=[],
                state=make_state(ost_audit_confirmed=True),
            )
        assert result is None


# --- run_all_days (multi-day loop) ------------------------------------


class TestRunAllDays:
    def test_multi_day_produces_one_artifact_per_day(self, tmp_path):
        """Two DayBatches → two DayArtifacts, day dirs isolated."""
        from gomoveshift.pipeline import day_pipeline

        batches = [
            _make_day_batch("2026-02-20", 3, per_ep_duration=1200.0, start_index=0),
            _make_day_batch("2026-02-21", 2, per_ep_duration=900.0, start_index=3),
        ]
        # Global scores span both days: day1=[0,3600), day2=[3600,5400)
        scores = _make_scores(5400, score=0.5)
        file_boundaries = [
            (0.0, 1200.0),
            (1200.0, 2400.0),
            (2400.0, 3600.0),  # day 1
            (3600.0, 4500.0),
            (4500.0, 5400.0),  # day 2
        ]

        call_dates = []

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kwargs):
            call_dates.append(day_batch.date)
            # Return a minimal artifact
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / f"day_{day_batch.date}_master.mp4",
                regular_lf_paths=[output_dir / f"lf_01_{day_batch.date}.mp4"],
                short_paths=[output_dir / f"short_01_{day_batch.date}.mp4"],
            )

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            artifacts_list = day_pipeline.run_all_days(
                day_batches=batches,
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="SC Trip",
                config=None,
            )

        assert len(artifacts_list) == 2
        assert call_dates == ["2026-02-20", "2026-02-21"]
        assert artifacts_list[0].date == "2026-02-20"
        assert artifacts_list[1].date == "2026-02-21"

    def test_each_day_gets_its_own_slice_of_scores(self, tmp_path):
        """run_all_days must pass the day-local slice, not the global scores."""
        from gomoveshift.pipeline import day_pipeline

        batches = [
            _make_day_batch("2026-02-20", 2, per_ep_duration=600.0, start_index=0),
            _make_day_batch("2026-02-21", 2, per_ep_duration=600.0, start_index=2),
        ]
        # Day 1: episodes 0,1 → global [0, 1200); Day 2: episodes 2,3 → [1200, 2400)
        scores = _make_scores(2400, score=0.5)
        # Mark day 2 rows distinctly so we can confirm the slice is rebased
        scores[1200:2400, COL_COMPOSITE] = 0.9
        file_boundaries = [
            (0.0, 600.0),
            (600.0, 1200.0),
            (1200.0, 1800.0),
            (1800.0, 2400.0),
        ]

        captured_slices = []

        def _capture(day_batch, day_scores, output_dir, location, config, **kwargs):
            captured_slices.append((day_batch.date, day_scores.copy()))
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / "master.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_capture,
        ):
            day_pipeline.run_all_days(
                day_batches=batches,
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
            )

        assert len(captured_slices) == 2
        day1_date, day1_scores = captured_slices[0]
        day2_date, day2_scores = captured_slices[1]
        assert day1_date == "2026-02-20"
        assert day2_date == "2026-02-21"
        # Day 1 slice: all 0.5 composite
        assert np.all(day1_scores[:, COL_COMPOSITE] == 0.5)
        # Day 2 slice: all 0.9 composite
        assert np.all(day2_scores[:, COL_COMPOSITE] == 0.9)
        # Both slices start at t=0 (rebased)
        assert day1_scores[0, COL_TIMESTAMP] == 0.0
        assert day2_scores[0, COL_TIMESTAMP] == 0.0

    def test_skips_day_if_run_day_pipeline_returns_none(self, tmp_path):
        """A failed day doesn't abort the whole run — subsequent days still run."""
        from gomoveshift.pipeline import day_pipeline

        batches = [
            _make_day_batch("2026-02-20", 1, per_ep_duration=600.0, start_index=0),
            _make_day_batch("2026-02-21", 1, per_ep_duration=600.0, start_index=1),
        ]
        scores = _make_scores(1200)
        file_boundaries = [(0.0, 600.0), (600.0, 1200.0)]

        def _side_effect(day_batch, day_scores, output_dir, location, config, **kwargs):
            if day_batch.date == "2026-02-20":
                return None
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / "master.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_side_effect,
        ):
            artifacts = day_pipeline.run_all_days(
                day_batches=batches,
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
            )

        assert len(artifacts) == 1
        assert artifacts[0].date == "2026-02-21"

    def test_empty_batches_returns_empty(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        result = day_pipeline.run_all_days(
            day_batches=[],
            drive_scores=_make_scores(0),
            file_boundaries=[],
            output_base=tmp_path,
            location="",
            config=None,
        )
        assert result == []


# --- run_all_days on_day_complete callback (mid-phase write-through) --


class TestEpisodeIndexConversion:
    """Regression: ``Episode.index`` is 1-based (ingest uses
    ``enumerate(plan.episodes, 1)``) but ``LongFormAssembler`` writes
    segments with 0-based enumerate positions. ``day_pipeline`` must
    convert when building ``day_episode_indices`` for the remap.

    This once surfaced as:
    ``segment source_file_index 0 not in this day's episode set [1]``.
    """

    def test_single_episode_at_first_drive_position(self, tmp_path):
        """1 episode with Episode.index=1 → day_episode_indices=[0].
        Segment source_file_index=0 resolves to day-local 0."""
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-02-20", 1, per_ep_duration=1620.0)
        day_scores = _make_scores(1620, score=0.6)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        master_path, mocks = TestRunDayPipeline()._patch_all_stages(tmp_path)
        with (
            mocks["encode_day_master"],
            mocks["apply_ost"],
            mocks["_render_short_ffmpeg"],
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ) as mock_asm,
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
        ):
            result = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=tmp_path / "out",
                location="Test, SC",
                config=None,
                lf_plans=lf_plans,
                state=make_state(ost_audit_confirmed=True),
            )
        assert result is not None
        assert len(result.regular_lf_paths) == 1
        assert mock_asm.call_count == 1
        # Verify the segment's index was remapped to day-local 0
        called_segments = mock_asm.call_args.kwargs["segments"]
        assert called_segments[0].source_file_index == 0

    def test_single_episode_at_nonzero_drive_position(self, tmp_path):
        """Day owns one episode at drive-wide position 5 (Episode.index=6).
        Under the current contract, segments reaching
        run_day_pipeline must already be day-local — the authoritative
        drive-global → day-local remap lives in
        phase_bc_day_first._partition_lf_plans_by_date. So the LF plan
        supplied here carries source_file_index=0 (day-local for a
        single-episode day), even though the episode sits at drive
        position 5. This pins the contract for a single-episode day whose
        episode sits at a later drive position."""
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-02-22", 1, per_ep_duration=1620.0, start_index=5)
        # Confirm fixture matches production convention.
        assert day_batch.episodes[0].index == 6  # 1-based, drive-wide position 5 + 1
        day_scores = _make_scores(1620, score=0.6)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        master_path, mocks = TestRunDayPipeline()._patch_all_stages(tmp_path)
        with (
            mocks["encode_day_master"],
            mocks["apply_ost"],
            mocks["_render_short_ffmpeg"],
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ) as mock_asm,
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
        ):
            result = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=tmp_path / "out",
                location="Test, SC",
                config=None,
                lf_plans=lf_plans,
                state=make_state(ost_audit_confirmed=True),
            )
        assert result is not None
        assert len(result.regular_lf_paths) == 1
        assert mock_asm.call_count == 1
        assert mock_asm.call_args.kwargs["segments"][0].source_file_index == 0


class TestRunAllDaysOnDayCompleteCallback:
    """`run_all_days` must invoke the ``on_day_complete`` callback after
    each successful day so the orchestrator can persist per-day progress
    (``per_day_completed`` + state-file save) BEFORE the next day starts.

    Without this, a mid-phase crash (e.g. day 22 master encode fails after
    days 20 + 21 land cleanly) leaves ``per_day_completed`` empty on disk,
    and the next resume can't short-circuit the already-done days —
    costing hours of wasted re-encoding.
    """

    def _two_batches(self):
        return [
            _make_day_batch("2026-02-20", 1, per_ep_duration=600.0, start_index=0),
            _make_day_batch("2026-02-21", 1, per_ep_duration=600.0, start_index=1),
        ]

    def _three_batches(self):
        return [
            _make_day_batch("2026-02-20", 1, per_ep_duration=600.0, start_index=0),
            _make_day_batch("2026-02-21", 1, per_ep_duration=600.0, start_index=1),
            _make_day_batch("2026-02-22", 1, per_ep_duration=600.0, start_index=2),
        ]

    def test_callback_invoked_after_each_successful_day(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        scores = _make_scores(1200)
        file_boundaries = [(0.0, 600.0), (600.0, 1200.0)]
        calls: list = []

        def _on_day_complete(artifacts, state):
            calls.append(artifacts.date)

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kw):
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / f"master_{day_batch.date}.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        fake_state = {}
        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            day_pipeline.run_all_days(
                day_batches=self._two_batches(),
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
                state=fake_state,
                on_day_complete=_on_day_complete,
            )

        assert calls == ["2026-02-20", "2026-02-21"], (
            f"callback must fire per-day in order; got {calls}"
        )

    def test_callback_not_invoked_for_failed_day(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        scores = _make_scores(1200)
        file_boundaries = [(0.0, 600.0), (600.0, 1200.0)]
        calls: list = []

        def _on_day_complete(artifacts, state):
            calls.append(artifacts.date)

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kw):
            if day_batch.date == "2026-02-20":
                return None  # first day fails
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / "master.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            day_pipeline.run_all_days(
                day_batches=self._two_batches(),
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
                state=make_state(),
                on_day_complete=_on_day_complete,
            )

        assert calls == ["2026-02-21"], (
            f"callback must NOT fire for a day that returned None; got {calls}"
        )

    def test_callback_exception_does_not_abort_remaining_days(self, tmp_path, caplog):
        """A callback failure is best-effort bookkeeping — log a warning but
        DO NOT stop the remaining days. Losing disk-side bookkeeping is
        recoverable; losing a day's encoded work is not."""
        from gomoveshift.pipeline import day_pipeline

        scores = _make_scores(1800)
        file_boundaries = [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1800.0)]

        def _on_day_complete(artifacts, state):
            if artifacts.date == "2026-02-20":
                raise OSError("simulated disk full during state save")

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kw):
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / f"master_{day_batch.date}.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            results = day_pipeline.run_all_days(
                day_batches=self._three_batches(),
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
                state=make_state(),
                on_day_complete=_on_day_complete,
            )

        assert [a.date for a in results] == [
            "2026-02-20",
            "2026-02-21",
            "2026-02-22",
        ], "all 3 days must complete even though day-20 callback failed"

    def test_callback_argument_is_optional(self, tmp_path):
        """Legacy callers that don't pass ``on_day_complete`` still work."""
        from gomoveshift.pipeline import day_pipeline

        scores = _make_scores(600)
        file_boundaries = [(0.0, 600.0)]

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kw):
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / "master.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            # No on_day_complete kwarg — must still work.
            results = day_pipeline.run_all_days(
                day_batches=[_make_day_batch("2026-02-20", 1, per_ep_duration=600.0)],
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
            )
        assert len(results) == 1


# -----------------------------------------------------------------------
# on_day_complete on the FAILURE path.
#
# Without recovery the callback fires only when run_day_pipeline returns non-None
# artifacts (i.e. success). When the day-master encode raises OR returns
# None, the day's per_day_completed entry is never written, and the next
# resume re-encodes from scratch — even if the day directory has finals
# or a raw concat that the finals-only reconstruction fast-path could pick
# up.
#
# Recovery: run_all_days wraps run_day_pipeline in try/except. On
# exception OR None return, if the day directory has finals or a raw
# concat on disk, invoke the callback with an "attempted" DayArtifacts
# carrying master_path=None so _record_day_and_save marks the day
# recorded. Then continue with remaining days (don't abort the whole
# loop).
# -----------------------------------------------------------------------


class TestRunAllDaysCrashRecovery:
    """When a day's run_day_pipeline raises, remaining days must still run
    AND the crashed day's on_day_complete must fire (with
    master_path=None) if the day dir has finals/raw on disk. Mirrors the
    finals-only reconstruction fast-path in phase_bc_day_first.py."""

    def _three_batches(self):
        return [
            _make_day_batch("2026-02-20", 1, per_ep_duration=600.0, start_index=0),
            _make_day_batch("2026-02-21", 1, per_ep_duration=600.0, start_index=1),
            _make_day_batch("2026-02-22", 1, per_ep_duration=600.0, start_index=2),
        ]

    def test_exception_on_day2_does_not_abort_day3(self, tmp_path):
        """Day 2 raises mid-encode. Day 1's artifacts are already emitted;
        Day 3 must still run (a failed day must not kill the whole loop)."""
        from gomoveshift.pipeline import day_pipeline

        scores = _make_scores(1800)
        file_boundaries = [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1800.0)]

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kw):
            if day_batch.date == "2026-02-21":
                raise RuntimeError("day 2 master encode exploded")
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / f"master_{day_batch.date}.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            results = day_pipeline.run_all_days(
                day_batches=self._three_batches(),
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
                state=make_state(),
            )

        dates = [a.date for a in results]
        assert "2026-02-20" in dates, "day 1 artifacts must survive"
        assert "2026-02-22" in dates, (
            "Regression: a day-2 exception must NOT abort the "
            "remaining days; day 3 should still be processed."
        )

    def test_exception_with_finals_on_disk_fires_callback(self, tmp_path):
        """When day-master raises AND the day dir already has finals on
        disk (e.g. a prior partial run landed LFs + shorts), the callback
        must fire with master_path=None so the finals-only
        reconstruction fast-path can pick up on resume."""
        from gomoveshift.pipeline import day_pipeline

        scores = _make_scores(1200)
        file_boundaries = [(0.0, 600.0), (600.0, 1200.0)]

        # Pre-seed day 2's finals on disk so the crash-recovery path sees
        # something worth recording.
        day2_dir = tmp_path / "day_2026-02-21"
        (day2_dir / "regular_lfs").mkdir(parents=True)
        (day2_dir / "shorts").mkdir(parents=True)
        (day2_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"lf-01")
        (day2_dir / "shorts" / "short_01.mp4").write_bytes(b"sh-01")

        callback_calls: list = []

        def _on_day_complete(artifacts, state):
            callback_calls.append(
                {
                    "date": artifacts.date,
                    "master_path": getattr(artifacts, "master_path", None),
                    "regular_lf_paths": list(getattr(artifacts, "regular_lf_paths", []) or []),
                    "short_paths": list(getattr(artifacts, "short_paths", []) or []),
                }
            )

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kw):
            if day_batch.date == "2026-02-21":
                raise RuntimeError("day-master encode crashed after finals landed")
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / f"master_{day_batch.date}.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        two_batches = [
            _make_day_batch("2026-02-20", 1, per_ep_duration=600.0, start_index=0),
            _make_day_batch("2026-02-21", 1, per_ep_duration=600.0, start_index=1),
        ]

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            day_pipeline.run_all_days(
                day_batches=two_batches,
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
                state=make_state(),
                on_day_complete=_on_day_complete,
            )

        dates_called = [c["date"] for c in callback_calls]
        # Day 1 must get the normal success callback.
        assert "2026-02-20" in dates_called, "day 1 callback must fire (existing success path)"
        # Day 2 must get an attempted-recovery callback with master_path=None.
        assert "2026-02-21" in dates_called, (
            "When day 2 crashes but finals are on disk, the "
            "callback must still fire so _record_day_and_save can mark "
            "the day so the next resume uses the finals-only fast-path."
        )
        day2_call = next(c for c in callback_calls if c["date"] == "2026-02-21")
        assert day2_call["master_path"] is None, (
            "The crash-recovery callback must pass master_path=None; "
            "the real master never landed on disk."
        )

    def test_exception_without_finals_does_not_record(self, tmp_path):
        """When the crash happens before ANY finals land (no regular_lfs/
        no shorts dir), the callback must NOT fire — nothing to record,
        and a bogus entry would prevent the next resume from re-encoding."""
        from gomoveshift.pipeline import day_pipeline

        scores = _make_scores(600)
        file_boundaries = [(0.0, 600.0)]

        callback_calls: list = []

        def _on_day_complete(artifacts, state):
            callback_calls.append(artifacts.date)

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kw):
            raise RuntimeError("crashed before finals")

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            day_pipeline.run_all_days(
                day_batches=[
                    _make_day_batch("2026-02-20", 1, per_ep_duration=600.0, start_index=0),
                ],
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
                state=make_state(),
                on_day_complete=_on_day_complete,
            )

        assert callback_calls == [], (
            "No finals on disk → no recovery callback. Without this guard, "
            "a pre-finals crash would leave a phantom per_day_completed "
            "entry that prevents the next resume from re-encoding."
        )

    def test_none_return_with_raw_concat_fires_callback(self, tmp_path):
        """When run_day_pipeline returns None (soft failure — reported
        and continued) but a raw concat file is on disk, fire the
        recovery callback so the operator can resume from the raw concat."""
        from gomoveshift.pipeline import day_pipeline

        scores = _make_scores(600)
        file_boundaries = [(0.0, 600.0)]

        # Seed a raw concat on disk — the day started encoding but the
        # inner loop returned None.
        day_dir = tmp_path / "day_2026-02-20"
        day_dir.mkdir()
        (day_dir / "day_2026-02-20_raw.mp4").write_bytes(b"raw-concat")

        callback_calls: list = []

        def _on_day_complete(artifacts, state):
            callback_calls.append(
                {
                    "date": artifacts.date,
                    "master_path": getattr(artifacts, "master_path", None),
                }
            )

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kw):
            return None  # soft failure

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            day_pipeline.run_all_days(
                day_batches=[
                    _make_day_batch("2026-02-20", 1, per_ep_duration=600.0, start_index=0),
                ],
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
                state=make_state(),
                on_day_complete=_on_day_complete,
            )

        assert len(callback_calls) == 1, (
            "A None return WITH raw concat on disk must fire "
            "the recovery callback so the raw concat can be picked up on "
            "resume."
        )
        assert callback_calls[0]["date"] == "2026-02-20"
        assert callback_calls[0]["master_path"] is None

    def test_crash_mid_shorts_with_master_on_disk_does_not_mark_thin_day(self, tmp_path):
        """A crash MID-shorts leaves the day master AND a *partial*
        shorts set on disk. The recovery callback must NOT write a
        finals-only "complete" marker for that day — that marker lets the
        next resume reconstruct-and-ship a THIN day (fewer shorts than the
        target) with no count-vs-target gate ever firing. Because the
        master survived, the day must instead RE-RUN on resume (reusing the
        cached master via encode_day_master's own resume shortcut) so the
        FULL shorts set is produced and the SHORTS-CONTRACT gate applies.
        So on_day_complete must NOT fire for the crashed day."""
        from gomoveshift.pipeline import day_pipeline

        scores = _make_scores(1200)
        file_boundaries = [(0.0, 600.0), (600.0, 1200.0)]

        # Pre-seed day 2: the day master survived on disk AND a partial
        # shorts set landed before the crash (a 1-of-N thin day).
        day2_dir = tmp_path / "day_2026-02-21"
        (day2_dir / "regular_lfs").mkdir(parents=True)
        (day2_dir / "shorts").mkdir(parents=True)
        (day2_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"lf-01")
        (day2_dir / "shorts" / "short_01.mp4").write_bytes(b"sh-01")
        (day2_dir / "day_2026-02-21_master.mp4").write_bytes(b"master-survived")

        callback_calls: list = []

        def _on_day_complete(artifacts, state):
            callback_calls.append(artifacts.date)

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kw):
            if day_batch.date == "2026-02-21":
                raise RuntimeError("crash mid-shorts (master already on disk)")
            return day_pipeline.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / f"master_{day_batch.date}.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        two_batches = [
            _make_day_batch("2026-02-20", 1, per_ep_duration=600.0, start_index=0),
            _make_day_batch("2026-02-21", 1, per_ep_duration=600.0, start_index=1),
        ]

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            day_pipeline.run_all_days(
                day_batches=two_batches,
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="",
                config=None,
                state=make_state(),
                on_day_complete=_on_day_complete,
            )

        assert "2026-02-20" in callback_calls, "day 1 success callback must still fire"
        assert "2026-02-21" not in callback_calls, (
            "Thin-day regression: a crash mid-shorts with the master still "
            "on disk must NOT write a finals-only recovery marker — that ships "
            "a thin day on resume (reconstruct fast-path, no count gate). The "
            "day must re-run (reusing the cached master) to produce the full "
            "shorts set."
        )


# --- segment-driven LF assembly ---------------------------


def _make_lf_plan(index, segments, montage_peaks):
    return SimpleNamespace(
        index=index,
        segments=segments,
        montage_peaks=montage_peaks,
    )


def _make_seg(ep_idx, start, end, lat=0.0, lon=0.0):
    return SimpleNamespace(
        source_file_index=ep_idx,
        start_time=float(start),
        end_time=float(end),
        duration=float(end - start),
        gps_center_lat=lat,
        gps_center_lon=lon,
    )


class TestRunDayPipelineSegmentDriven:
    """When an LF plan is supplied, LFs are assembled from scored
    segments concatenated via the day master, with a 30s branded montage
    prepended before OST burn-in."""

    def _patches(self, tmp_path, assembled_duration):
        master_path = tmp_path / "day_2026-02-20" / "day_2026-02-20_master.mp4"
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_bytes(b"fake")

        def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"concat")
            return output_path

        def _fake_prepend(raw_lf_path, peaks, output_path, work_dir, **kw):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"montage+lf")
            return output_path

        def _fake_ost(input_path, output_path, location, layout, **kwargs):
            output_path.write_bytes(b"ost")
            return output_path

        return master_path, {
            "encode": patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            "assemble": patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=_fake_assemble,
            ),
            "prepend": patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=_fake_prepend,
            ),
            "ost": patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=_fake_ost,
            ),
            # The shorts loop goes through render_short_ffmpeg.
            "render_short": patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=lambda source_video, output_path, start_time, duration, location, **kw: (
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    or Path(output_path).write_bytes(b"x")
                    or True
                ),
            ),
        }

    def test_lf_assembly_calls_segment_concat_then_montage_then_ost(self, tmp_path):
        """With an LF plan, each LF flows: concat segments → prepend
        30s montage → apply_ost(layout=regular_lf). OST t=31s is
        guaranteed by the layout definition."""
        from gomoveshift.pipeline import day_pipeline

        batch = _make_day_batch("2026-02-20", 2, per_ep_duration=3240.0)
        segs = [_make_seg(0, 100.0, 200.0), _make_seg(1, 50.0, 150.0)]
        peaks = [_make_seg(1, 50.0, 80.0)]
        lf_plans = [_make_lf_plan(1, segs, peaks)]

        _, patches = self._patches(tmp_path, assembled_duration=200.0)

        with (
            patches["encode"],
            patches["assemble"] as mock_asm,
            patches["prepend"] as mock_prep,
            patches["ost"] as mock_ost,
            patches["render_short"],
        ):
            result = day_pipeline.run_day_pipeline(
                day_batch=batch,
                day_scores=_make_scores(6480),
                output_dir=tmp_path / "out",
                location="NC, USA",
                config=None,
                lf_plans=lf_plans,
                state=make_state(ost_audit_confirmed=True),
            )

        assert result is not None
        assert mock_asm.call_count == 1
        assert mock_prep.call_count == 1
        # OST applied with regular_lf layout (t=31s start is part of the layout)
        ost_calls = [c for c in mock_ost.call_args_list if c.kwargs.get("layout") == "regular_lf"]
        assert len(ost_calls) == 1

    def test_segment_order_preserved_in_concat(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        batch = _make_day_batch("2026-02-20", 2, per_ep_duration=3240.0)
        segs = [
            _make_seg(0, 100.0, 200.0),
            _make_seg(1, 50.0, 150.0),
            _make_seg(1, 300.0, 400.0),
        ]
        lf_plans = [_make_lf_plan(1, segs, [])]

        captured = {}

        def _capture(master_path, segments, file_boundaries, output_path, work_dir, **kw):
            captured["segments"] = list(segments)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"x")
            return output_path

        _, patches = self._patches(tmp_path, 300.0)
        with (
            patches["encode"],
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=_capture,
            ),
            patches["prepend"],
            patches["ost"],
            patches["render_short"],
        ):
            day_pipeline.run_day_pipeline(
                day_batch=batch,
                day_scores=_make_scores(6480),
                output_dir=tmp_path / "out",
                location="L",
                config=None,
                lf_plans=lf_plans,
                state=make_state(ost_audit_confirmed=True),
            )

        assert [s.source_file_index for s in captured["segments"]] == [0, 1, 1]
        assert captured["segments"][0].start_time == 100.0

    def test_ost_layout_regular_lf_used_for_plan_driven_lfs(self, tmp_path):
        """The regular_lf layout is defined with location_start_sec=2.0
        in ost_applier.LAYOUTS — OST overlays the montage intro, landing
        the title in the YouTube retention window (first ~10s)."""
        from gomoveshift.pipeline.ost_applier import LAYOUTS

        assert LAYOUTS["regular_lf"]["location_start_sec"] == 2.0


# --- legacy path removed -----------------------------------------------


class TestLegacyPathRemoved:
    """`run_day_pipeline` no longer accepts a None/missing `lf_plans`.
    The legacy score-range fallback (no montage) shipped wrong content
    and is gone — callers MUST supply real plans."""

    def test_run_day_pipeline_requires_lf_plans_kwarg(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline

        batch = _make_day_batch("2026-02-20", 1, per_ep_duration=600.0)
        # Calling without lf_plans must raise TypeError (kw-only required)
        import pytest

        with pytest.raises(TypeError):
            day_pipeline.run_day_pipeline(
                day_batch=batch,
                day_scores=_make_scores(600),
                output_dir=tmp_path,
                location="",
                config=None,
            )

    def test_run_day_pipeline_assembles_two_lfs_via_segments_and_montage(
        self,
        tmp_path,
    ):
        """Given a day with 2 LF plans (3 segments each), verify the
        call order: extract_and_concat_segments → prepend_montage →
        apply_ost(layout=regular_lf). NO extract_slice path for LFs."""
        from gomoveshift.pipeline import day_pipeline

        batch = _make_day_batch("2026-02-20", 2, per_ep_duration=1200.0)
        lf1 = _make_lf_plan(
            1,
            [_make_seg(0, 100, 200), _make_seg(0, 300, 400), _make_seg(1, 50, 150)],
            [_make_seg(0, 110, 120)],
        )
        lf2 = _make_lf_plan(
            2,
            [_make_seg(1, 200, 300), _make_seg(1, 500, 600), _make_seg(1, 800, 900)],
            [_make_seg(1, 210, 220)],
        )

        master_path = tmp_path / "out" / "day_2026-02-20" / "master.mp4"
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_bytes(b"m")

        order: list = []

        def _asm(master_path, segments, file_boundaries, output_path, work_dir, **kw):
            order.append("assemble")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"x")
            return output_path

        def _prep(raw_lf_path, peaks, output_path, work_dir, **kw):
            order.append("prepend")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"x")
            return output_path

        def _ost(input_path, output_path, location, layout, **kw):
            order.append(f"ost:{layout}")
            output_path.write_bytes(b"x")
            return output_path

        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=_asm,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=_prep,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=_ost,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=lambda source_video, output_path, start_time, duration, location, **kw: (
                    order.append("render_short")
                    or Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    or Path(output_path).write_bytes(b"x")
                    or True
                ),
            ),
        ):
            artifacts = day_pipeline.run_day_pipeline(
                day_batch=batch,
                day_scores=_make_scores(2400),
                output_dir=tmp_path / "out",
                location="L",
                config=None,
                lf_plans=[lf1, lf2],
                state=make_state(ost_audit_confirmed=True),
            )

        assert artifacts is not None
        assert len(artifacts.regular_lf_paths) == 2
        # Per-LF ordering: assemble → prepend → ost(regular_lf), twice
        lf_pattern = ["assemble", "prepend", "ost:regular_lf"]
        assert order[:3] == lf_pattern
        assert order[3:6] == lf_pattern
        # Shorts use render_short_ffmpeg (single-pass crop+scale+OST),
        # not apply_ost(layout="short") + extract_slice. The order trace records
        # "render_short" entries for each short.
        assert "render_short" in order
        assert "ost:short" not in order


# --- per-day intermediate purge wiring ---------------------------------


class TestDayPipelinePurgeWiring:
    """``run_day_pipeline`` must trigger ``purge_day_intermediates`` at the
    end of a successful day unless ``state['keep_intermediates']`` is True.
    Guards against unbounded cache growth: the inline purge
    is what caps the working set at ~one day."""

    def _run(self, tmp_path, *, state=None):
        from gomoveshift.pipeline import day_pipeline

        batch = _make_day_batch("2026-02-20", 3, per_ep_duration=1200.0)
        day_scores = _make_scores(3600, score=0.6)
        output_base = tmp_path / "out"
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]

        master_path, mocks = TestRunDayPipeline()._patch_all_stages(tmp_path)
        with (
            mocks["encode_day_master"],
            mocks["apply_ost"],
            mocks["_render_short_ffmpeg"],
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch("gomoveshift.pipeline.day_pipeline.purge_day_intermediates") as mock_purge,
        ):
            day_pipeline.run_day_pipeline(
                day_batch=batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test",
                config=None,
                lf_plans=lf_plans,
                state=state,
            )
        return mock_purge

    def test_purge_called_by_default(self, tmp_path):
        mock_purge = self._run(tmp_path, state=make_state(ost_audit_confirmed=True))
        assert mock_purge.called, "purge_day_intermediates must run by default"

    def test_purge_skipped_when_keep_intermediates(self, tmp_path):
        mock_purge = self._run(
            tmp_path, state=make_state(keep_intermediates=True, ost_audit_confirmed=True)
        )
        assert not mock_purge.called, "keep_intermediates=True must suppress purge"

    def test_purge_exception_does_not_break_run(self, tmp_path):
        """Cleanup failure must not block artifact return — the purge is
        best-effort (disk full, permission error, race). The day's outputs
        are already on disk; failing the run would lose them."""
        from gomoveshift.pipeline import day_pipeline

        batch = _make_day_batch("2026-02-20", 3, per_ep_duration=1200.0)
        day_scores = _make_scores(3600, score=0.6)
        output_base = tmp_path / "out"
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]

        master_path, mocks = TestRunDayPipeline()._patch_all_stages(tmp_path)
        with (
            mocks["encode_day_master"],
            mocks["apply_ost"],
            mocks["_render_short_ffmpeg"],
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.purge_day_intermediates",
                side_effect=OSError("disk full during cleanup"),
            ),
        ):
            artifacts = day_pipeline.run_day_pipeline(
                day_batch=batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test",
                config=None,
                lf_plans=lf_plans,
                state=make_state(ost_audit_confirmed=True),
            )
        assert artifacts is not None, "Pipeline must return artifacts even if purge fails"


# --- shorts-per-LF target scaling -------------------------------------


class TestShortsPerLfTargetScaling:
    """Shorts `target_count` is planned at the day master level but is
    budgeted per-LF. With Waypost deciding ``N`` per LF (``volume_source``)
    and ``K`` LFs for the day, the planner must be invoked with
    ``target_count = N * K`` so each LF's publish window can be filled
    from the day master.

    The datetime-based partition in ``phase_d._partition_shorts_slots``
    then splits the resulting shorts across each LF's slot window.
    """

    def _run_with_config(self, tmp_path, drive_config, lf_plans, state=None):
        from types import SimpleNamespace

        from gomoveshift.pipeline import day_pipeline

        if state is None:
            # plan_shorts is stubbed to [] — escape the SHORTS-CONTRACT
            # gate by default; these tests assert the planner CALL.
            state = make_state(ost_audit_confirmed=True, allow_short_count=True)
        batch = _make_day_batch("2026-02-20", 3, per_ep_duration=1200.0)
        day_scores = _make_scores(3600, score=0.6)
        output_base = tmp_path / "out"
        # Pin target semantics, not the moving budget: disable the stop mask so
        # the moving-budget cap can't lower the exact target reaching the planner.
        drive_config.stop_mask_enabled = False
        config = SimpleNamespace(drive=drive_config)

        master_path, mocks = TestRunDayPipeline()._patch_all_stages(tmp_path)
        captured: dict = {}

        def _fake_plan_shorts(scores, **kwargs):
            captured["target_count"] = kwargs.get("target_count")
            captured["min_gap_sec"] = kwargs.get("min_gap_sec")
            return []  # no shorts rendered — we only care about the call

        with (
            # These tests pin the state/config/derived precedence in
            # isolation from the brand-policy cap (a brand with a policy
            # block can cap the target) — "examplebrand"
            # ships no policy block, so load_brand_policy returns no cap.
            patch.dict(os.environ, {"GMS_CONSOLE_BRAND": "examplebrand"}),
            mocks["encode_day_master"],
            mocks["apply_ost"],
            mocks["_render_short_ffmpeg"],
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.plan_shorts",
                side_effect=_fake_plan_shorts,
            ),
        ):
            day_pipeline.run_day_pipeline(
                day_batch=batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test",
                config=config,
                lf_plans=lf_plans,
                state=state,
            )
        return captured

    def test_single_lf_day_plans_decided_shorts(self, tmp_path):
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        captured = self._run_with_config(tmp_path, dc, lf_plans)
        assert captured["target_count"] == 14, (
            f"1 LF × decided 14 must request 14 shorts, got {captured['target_count']}"
        )

    def test_two_lf_day_plans_double_the_shorts(self, tmp_path):
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [
            _make_lf_plan(1, [_make_seg(0, 100, 200)], []),
            _make_lf_plan(2, [_make_seg(0, 300, 400)], []),
        ]
        captured = self._run_with_config(tmp_path, dc, lf_plans)
        assert captured["target_count"] == 28, (
            f"2 LFs × decided 14 must request 28 shorts, got {captured['target_count']}"
        )

    def test_per_lf_none_derives_from_publish_plan(self, tmp_path):
        """With no override the planner target is the decided per-LF count
        (the fixture decides 14, equal to 2 windows/day × 7-day LF interval)
        — never the old hardcoded-9 day cap."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [
            _make_lf_plan(1, [_make_seg(0, 100, 200)], []),
            _make_lf_plan(2, [_make_seg(0, 300, 400)], []),
        ]
        captured = self._run_with_config(tmp_path, dc, lf_plans)
        assert captured["target_count"] == 28, (
            f"Decided default must be 14 × 2 LFs = 28 "
            f"(never the legacy 9), got {captured['target_count']}"
        )

    def test_planned_shorts_equal_decision_times_num_lfs(self, tmp_path):
        """The per-LF count is Waypost's decision, whatever the config says."""
        from gomoveshift.config import DriveConfig

        _DECIDED["value"] = 9
        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [
            _make_lf_plan(1, [_make_seg(0, 100, 200)], []),
            _make_lf_plan(2, [_make_seg(0, 300, 400)], []),
        ]
        captured = self._run_with_config(tmp_path, dc, lf_plans)
        assert captured["target_count"] == 18

    def test_shorts_target_flag_only_lowers_and_logs_VOLUME_OVERRIDE_LOWER(self, tmp_path, caplog):
        import logging

        from gomoveshift.config import DriveConfig

        _DECIDED["value"] = 9
        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        with caplog.at_level(logging.INFO, logger="gomoveshift.pipeline.volume_source"):
            lowered = self._run_with_config(
                tmp_path,
                dc,
                lf_plans,
                state=make_state(shorts_target=4, ost_audit_confirmed=True, allow_short_count=True),
            )
            assert lowered["target_count"] == 4
            assert any("VOLUME_OVERRIDE_LOWER decided=9 flag=4" in m for m in caplog.messages)
            raised = self._run_with_config(
                tmp_path,
                dc,
                lf_plans,
                state=make_state(
                    shorts_target=20, ost_audit_confirmed=True, allow_short_count=True
                ),
            )
        assert raised["target_count"] == 9

    def test_unreadable_decision_store_raises_and_plans_nothing(self, tmp_path):
        import pytest

        from gomoveshift.config import DriveConfig
        from gomoveshift.pipeline import volume_source

        _DECIDED["error"] = True
        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        with pytest.raises(volume_source.VolumeSourceUnreadableError):
            self._run_with_config(tmp_path, dc, lf_plans)

    def test_reconstructed_shortfall_target_uses_same_function(self):
        """phase_bc's synthesized LF plan reads volume_source, not config."""
        import inspect

        from gomoveshift.pipeline.drive_phases import phase_bc_day_first

        src = inspect.getsource(phase_bc_day_first)
        assert "shorts_target=volume_source.shorts_per_lf().value" in src
        assert "shorts_target=self.config.drive.min_shorts_per_lf" not in src

    def test_derived_default_reaches_shorts_contract_gate(self, tmp_path):
        """The DERIVED requirement is what the SHORTS-CONTRACT gate enforces:
        no override, no --allow-short-count, planner yields 0 → hard-fail
        names the derived 14, not the old hardcoded 9."""
        import pytest

        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        with pytest.raises(RuntimeError, match=r"SHORTS-CONTRACT: planned 0 < required 14"):
            self._run_with_config(
                tmp_path, dc, lf_plans, state=make_state(ost_audit_confirmed=True)
            )


class TestShortsTargetStateIsPerLf:
    """``state["shorts_target"]`` (set by ``--shorts-target N``) is
    interpreted as a **per-LF** count, not a per-day total cap. The day
    planner must be invoked with ``target_count = N * num_lfs_in_day``
    so the per-LF intent (the shorts-per-LF count Waypost's cadence
    decision sets) holds across multi-LF days.

    Earlier behaviour: ``state["shorts_target"]`` short-circuited the
    per-LF multiplication and was used verbatim (per-day cap). That drift
    is fixed; the per-LF multiplication now applies.
    """

    def _run_with_state(self, tmp_path, drive_config, lf_plans, state):
        from types import SimpleNamespace

        from gomoveshift.pipeline import day_pipeline

        batch = _make_day_batch("2026-02-20", 3, per_ep_duration=1200.0)
        day_scores = _make_scores(3600, score=0.6)
        output_base = tmp_path / "out"
        # These tests pin per-LF TARGET semantics (N * num_lfs), not the moving
        # budget — disable the per-day stop mask so the moving-seconds cap is
        # a strict no-op and the exact requested target is what reaches the planner.
        drive_config.stop_mask_enabled = False
        config = SimpleNamespace(drive=drive_config)

        master_path, mocks = TestRunDayPipeline()._patch_all_stages(tmp_path)
        captured: dict = {}

        def _fake_plan_shorts(scores, **kwargs):
            captured["target_count"] = kwargs.get("target_count")
            captured["min_gap_sec"] = kwargs.get("min_gap_sec")
            return []

        with (
            # See TestShortsPerLfTargetScaling._run_with_config — isolate
            # from the default brand's policy cap.
            patch.dict(os.environ, {"GMS_CONSOLE_BRAND": "examplebrand"}),
            mocks["encode_day_master"],
            mocks["apply_ost"],
            mocks["_render_short_ffmpeg"],
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.plan_shorts",
                side_effect=_fake_plan_shorts,
            ),
        ):
            day_pipeline.run_day_pipeline(
                day_batch=batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test",
                config=config,
                lf_plans=lf_plans,
                state=state,
            )
        return captured

    def test_shorts_target_state_multiplied_by_num_lfs_three_lfs(self, tmp_path):
        """``--shorts-target=14`` with 3 LFs → planner gets 42 (14 * 3)."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [
            _make_lf_plan(1, [_make_seg(0, 100, 200)], []),
            _make_lf_plan(2, [_make_seg(0, 300, 400)], []),
            _make_lf_plan(3, [_make_seg(0, 500, 600)], []),
        ]
        captured = self._run_with_state(
            tmp_path,
            dc,
            lf_plans,
            state=make_state(shorts_target=14, ost_audit_confirmed=True, allow_short_count=True),
        )
        assert captured["target_count"] == 42, (
            f"--shorts-target=14 with 3 LFs must request 14 * 3 = 42 "
            f"shorts (per-LF semantics), got {captured['target_count']}"
        )

    def test_shorts_target_state_single_lf(self, tmp_path):
        """``--shorts-target=10`` with 1 LF → planner gets 10."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        captured = self._run_with_state(
            tmp_path,
            dc,
            lf_plans,
            state=make_state(shorts_target=10, ost_audit_confirmed=True, allow_short_count=True),
        )
        assert captured["target_count"] == 10, (
            f"--shorts-target=10 with 1 LF must request 10 shorts, got {captured['target_count']}"
        )

    def test_shorts_target_state_omitted_falls_back_to_per_lf_default(self, tmp_path):
        """No ``state["shorts_target"]`` → planner gets
        ``decided per-LF count * num_lfs`` (existing default, no regression)."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [
            _make_lf_plan(1, [_make_seg(0, 100, 200)], []),
            _make_lf_plan(2, [_make_seg(0, 300, 400)], []),
        ]
        captured = self._run_with_state(
            tmp_path,
            dc,
            lf_plans,
            state=make_state(ost_audit_confirmed=True, allow_short_count=True),
        )
        assert captured["target_count"] == 28, (
            f"Per-LF default path must request decided per-LF count * num_lfs "
            f"= 14 * 2 = 28, got {captured['target_count']}"
        )

    def test_shorts_target_state_zero_lfs_no_crash(self, tmp_path):
        """Edge case: 0 LFs with ``--shorts-target=14`` → target_count == 0,
        no crash. Defensive guard against malformed day plans."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans: list = []
        captured = self._run_with_state(
            tmp_path,
            dc,
            lf_plans,
            state=make_state(shorts_target=14, ost_audit_confirmed=True, allow_short_count=True),
        )
        assert captured["target_count"] == 0, (
            f"0 LFs must yield target_count == 0 (14 * 0), got {captured['target_count']}"
        )


class TestShortsTargetBrandPolicyCap:
    """A brand's ``policy.per_location_cap`` (``brand_policy.py``;
    the synthetic ``cappedbrand`` here sets 5) further trims the per-LF target
    computed by state/config/derived precedence -- never raises it, and
    never trips when an explicit ``--shorts-target`` (operator intent) is
    in force."""

    def _run(self, tmp_path, drive_config, lf_plans, state=None, brand="cappedbrand"):
        from types import SimpleNamespace

        import gomoveshift.publishing.brand_policy as brand_policy_mod
        from gomoveshift.pipeline import day_pipeline

        brands_dir = tmp_path / "brands"
        (brands_dir / "cappedbrand").mkdir(parents=True)
        (brands_dir / "cappedbrand" / "brand.yaml").write_text("policy:\n  per_location_cap: 5\n")
        (brands_dir / "examplebrand").mkdir()
        (brands_dir / "examplebrand" / "brand.yaml").write_text("disabled_tabs: []\n")

        if state is None:
            state = make_state(ost_audit_confirmed=True, allow_short_count=True)
        batch = _make_day_batch("2026-02-20", 3, per_ep_duration=1200.0)
        day_scores = _make_scores(3600, score=0.6)
        output_base = tmp_path / "out"
        drive_config.stop_mask_enabled = False
        config = SimpleNamespace(drive=drive_config)

        master_path, mocks = TestRunDayPipeline()._patch_all_stages(tmp_path)
        captured: dict = {}

        def _fake_plan_shorts(scores, **kwargs):
            captured["target_count"] = kwargs.get("target_count")
            return []

        env = {"GMS_CONSOLE_BRAND": brand} if brand else {}
        with (
            patch.dict(os.environ, env),
            patch.object(brand_policy_mod, "_SHIPPED_BRANDS_DIR", brands_dir),
            mocks["encode_day_master"],
            mocks["apply_ost"],
            mocks["_render_short_ffmpeg"],
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=lambda master_path, segments, file_boundaries, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=lambda raw_lf_path, peaks, output_path, work_dir, **kw: (
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    or output_path.write_bytes(b"x")
                    or output_path
                ),
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.plan_shorts",
                side_effect=_fake_plan_shorts,
            ),
        ):
            day_pipeline.run_day_pipeline(
                day_batch=batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test",
                config=config,
                lf_plans=lf_plans,
                state=state,
            )
        return captured

    def test_default_brand_cap_5_trims_config_target_14(self, tmp_path):
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        captured = self._run(tmp_path, dc, lf_plans)  # cappedbrand, cap 5
        assert captured["target_count"] == 5, (
            f"policy.per_location_cap=5 < config 14 must trim to 5, got {captured['target_count']}"
        )

    def test_cap_above_config_target_is_a_noop(self, tmp_path):
        from gomoveshift.config import DriveConfig

        _DECIDED["value"] = 3
        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        captured = self._run(tmp_path, dc, lf_plans)  # cappedbrand, cap 5
        assert captured["target_count"] == 3, (
            f"cap 5 >= config 3 must not raise the target, got {captured['target_count']}"
        )

    def test_no_policy_brand_is_unaffected(self, tmp_path):
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        captured = self._run(tmp_path, dc, lf_plans, brand="examplebrand")
        assert captured["target_count"] == 14

    def test_explicit_cli_shorts_target_beats_the_cap(self, tmp_path):
        """A per_location_cap below min_shorts_per_lf must not trip any
        floor -- and an explicit --shorts-target (operator intent) is
        never capped by brand policy at all."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        captured = self._run(
            tmp_path,
            dc,
            lf_plans,
            state=make_state(shorts_target=14, ost_audit_confirmed=True, allow_short_count=True),
        )
        assert captured["target_count"] == 14, (
            "explicit --shorts-target must win over the brand policy cap unchanged, "
            f"got {captured['target_count']}"
        )

    def test_cap_below_min_shorts_per_lf_sets_the_checked_floor(self, tmp_path):
        """A per_location_cap (5) below min_shorts_per_lf (7, the config
        default) becomes the number the SHORTS-CONTRACT gate checks --
        never the uncapped 14 nor the unrelated min_shorts_per_lf=7."""
        from gomoveshift.config import DriveConfig

        dc = DriveConfig(shorts_min_gap_sec=60.0)
        assert dc.min_shorts_per_lf == 7
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]
        captured = self._run(
            tmp_path,
            dc,
            lf_plans,
            state=make_state(ost_audit_confirmed=True, allow_short_count=True),
        )
        assert captured["target_count"] == 5


# ── OST burn uses per_lf_ost_text when provided ──────────────────────


class TestRunDayPipelineOSTTextFromCanonical:
    """``run_day_pipeline``'s OST burn-in for regular LFs uses
    ``per_lf_ost_text[lf.index]`` when provided, falling back to
    ``per_lf_locations[lf.index]`` otherwise.

    Contract: ``state["per_lf_ost_text"]`` is the canonical
    ``"{start} to {end} via {road}, {state}"`` text; OST burn-in
    must use it verbatim so the burned text matches the YouTube title
    and thumbnail text.
    """

    def _patches(self, tmp_path):
        master_path = tmp_path / "day_2026-02-20" / "day_2026-02-20_master.mp4"
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_bytes(b"fake")

        def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"concat")
            return output_path

        def _fake_prepend(raw_lf_path, peaks, output_path, work_dir, **kw):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"montage+lf")
            return output_path

        def _fake_ost(input_path, output_path, location, layout, **kwargs):
            output_path.write_bytes(b"ost")
            return output_path

        return master_path, {
            "encode": patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            "assemble": patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=_fake_assemble,
            ),
            "prepend": patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=_fake_prepend,
            ),
            "ost": patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=_fake_ost,
            ),
            "render_short": patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=lambda source_video, output_path, start_time, duration, location, **kw: (
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    or Path(output_path).write_bytes(b"x")
                    or True
                ),
            ),
        }

    def test_ost_burn_uses_canonical_text_when_available(self, tmp_path):
        """When per_lf_ost_text is passed, apply_ost's location= kwarg
        for the regular_lf layout uses the canonical text — NOT the
        county-level per_lf_locations fallback."""
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-02-20", 1, per_ep_duration=1200.0)
        day_scores = _make_scores(1200, score=0.5)
        output_base = tmp_path / "out"
        output_base.mkdir(parents=True, exist_ok=True)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]

        _, p = self._patches(tmp_path)
        with (
            p["encode"],
            p["assemble"],
            p["prepend"],
            p["ost"] as mock_ost,
            p["render_short"],
        ):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Drive-Wide Fallback",
                config=None,
                lf_plans=lf_plans,
                per_lf_locations={1: "Beaufort County, South Carolina"},
                per_lf_ost_text={
                    1: "Port Royal to Lady's Island via Sea Island Parkway, SC",
                },
                state=make_state(ost_audit_confirmed=True),
            )

        # Find the regular_lf OST call (layout="regular_lf")
        regular_lf_calls = [
            c for c in mock_ost.call_args_list if c.kwargs.get("layout") == "regular_lf"
        ]
        assert regular_lf_calls, "No regular_lf apply_ost call was made"
        burned = regular_lf_calls[0].kwargs.get("location")
        assert burned == "Port Royal to Lady's Island via Sea Island Parkway, SC", (
            f"OST burn must use canonical per_lf_ost_text for LF 1; got {burned!r}"
        )
        # Regression gate — must NOT fall back to county-level label
        assert "County" not in (burned or ""), (
            f"OST burn fell back to county label — canonical text ignored; got {burned!r}"
        )

    def test_ost_burn_falls_back_to_per_lf_locations_when_no_canonical(self, tmp_path):
        """When per_lf_ost_text is absent (or missing the LF's index),
        OST burn falls back to per_lf_locations — preserves the
        behaviour for older runs or degraded canonical paths.
        """
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-02-20", 1, per_ep_duration=1200.0)
        day_scores = _make_scores(1200, score=0.5)
        output_base = tmp_path / "out"
        output_base.mkdir(parents=True, exist_ok=True)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]

        _, p = self._patches(tmp_path)
        with (
            p["encode"],
            p["assemble"],
            p["prepend"],
            p["ost"] as mock_ost,
            p["render_short"],
        ):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Drive-Wide Fallback",
                config=None,
                lf_plans=lf_plans,
                per_lf_locations={1: "Beaufort County, South Carolina"},
                # per_lf_ost_text omitted — older resume path
                state=make_state(ost_audit_confirmed=True),
            )

        regular_lf_calls = [
            c for c in mock_ost.call_args_list if c.kwargs.get("layout") == "regular_lf"
        ]
        assert regular_lf_calls, "No regular_lf apply_ost call was made"
        burned = regular_lf_calls[0].kwargs.get("location")
        assert burned == "Beaufort County, South Carolina", (
            f"OST burn must fall back to per_lf_locations[1]; got {burned!r}"
        )


class TestMontageDurationThreadedToApplyOst:
    """``run_day_pipeline`` threads ``config.drive.montage_max_duration_sec``
    and ``config.drive.ost_title_crossfade_sec`` into ``apply_ost`` for the
    ``regular_lf`` call so the LOCATION title and the BRAND text span the montage with a
    tail crossfade."""

    def _patches(self, tmp_path):
        master_path = tmp_path / "out" / "day_2026-02-20" / "master.mp4"

        def _fake_encode(day_batch, day_dir, config, location, **kwargs):
            day_dir.mkdir(parents=True, exist_ok=True)
            master_path.parent.mkdir(parents=True, exist_ok=True)
            master_path.write_bytes(b"\x00" * 1024)
            return master_path

        def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"\x00" * 1024)
            return output_path

        def _fake_prepend(raw_lf_path, peaks, output_path, work_dir, master_path, file_boundaries):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"\x00" * 1024)
            return output_path

        def _fake_ost(*, input_path, output_path, location, layout, **kwargs):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"\x00" * 1024)
            return output_path

        return {
            "encode": patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                side_effect=_fake_encode,
            ),
            "assemble": patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=_fake_assemble,
            ),
            "prepend": patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=_fake_prepend,
            ),
            "ost": patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=_fake_ost,
            ),
            "render_short": patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=lambda source_video, output_path, start_time, duration, location, **kw: (
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    or Path(output_path).write_bytes(b"\x00" * 1024)
                    or True
                ),
            ),
        }

    def _run(self, tmp_path, config):
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-02-20", 1, per_ep_duration=1200.0)
        day_scores = _make_scores(1200, score=0.5)
        output_base = tmp_path / "out"
        output_base.mkdir(parents=True, exist_ok=True)
        lf_plans = [_make_lf_plan(1, [_make_seg(0, 100, 200)], [])]

        p = self._patches(tmp_path)
        with (
            p["encode"],
            p["assemble"],
            p["prepend"],
            p["ost"] as mock_ost,
            p["render_short"],
        ):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Somewhere",
                config=config,
                lf_plans=lf_plans,
                state=make_state(ost_audit_confirmed=True, allow_short_count=True),
            )
        return [c for c in mock_ost.call_args_list if c.kwargs.get("layout") == "regular_lf"]

    def test_montage_duration_read_from_config(self, tmp_path):
        """``montage_max_duration_sec`` from config propagates to apply_ost."""
        from gomoveshift.config import Config

        config = Config()
        config.drive.montage_max_duration_sec = 12.0
        config.drive.ost_title_crossfade_sec = 0.75
        regular_lf_calls = self._run(tmp_path, config)
        assert regular_lf_calls, "No regular_lf apply_ost call was made"
        kwargs = regular_lf_calls[0].kwargs
        assert kwargs.get("montage_duration") == 12.0, (
            f"montage_duration not threaded from config; got kwargs={kwargs}"
        )
        assert kwargs.get("crossfade_sec") == 0.75, (
            f"crossfade_sec not threaded from config; got kwargs={kwargs}"
        )

    def test_montage_duration_defaults_when_config_none(self, tmp_path):
        """When config=None, apply_ost still gets the 15s / 0.5s fallbacks."""
        regular_lf_calls = self._run(tmp_path, None)
        assert regular_lf_calls, "No regular_lf apply_ost call was made"
        kwargs = regular_lf_calls[0].kwargs
        assert kwargs.get("montage_duration") == 15.0
        assert kwargs.get("crossfade_sec") == 0.5


class TestMovingBudgetShortsCap:
    """The required shorts target is capped by the day's moving-seconds
    budget so a parked-heavy day doesn't force routine --allow-short-count."""

    def _cap(self, target, mask, min_gap=60.0):
        from gomoveshift.pipeline.day_pipeline import _cap_shorts_target_by_moving_budget

        return _cap_shorts_target_by_moving_budget(target, mask, min_gap)

    def test_parked_heavy_day_caps_target(self):
        # 10000s day, only 500s moving -> 500 / (38 + 60) = 5 shorts supportable.
        mask = np.ones(10000, dtype=bool)
        mask[:500] = False  # 500 moving seconds
        assert self._cap(14, mask) == 5

    def test_mostly_moving_day_keeps_target(self):
        # A long mostly-moving day supports far more than 14 -> target unchanged.
        mask = np.zeros(10000, dtype=bool)  # all moving
        assert self._cap(14, mask) == 14

    def test_no_mask_is_noop(self):
        assert self._cap(14, None) == 14

    def test_none_target_is_noop(self):
        assert self._cap(None, np.ones(100, dtype=bool)) is None
