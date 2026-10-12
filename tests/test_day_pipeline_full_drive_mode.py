"""Tests for `run_day_pipeline` full-drive LF selection mode.

When ``config.drive.lf_selection_mode == "full_drive"`` AND the caller
doesn't pre-supply ``lf_plans``, the day pipeline synthesizes a single
LF plan spanning the day's full post-stop-removal timeline by calling
``plan_regular_lfs(day_scores, mode="full_drive", ...)``. The single
synthetic LF is then handed off to the existing segment-driven
assembly path so every downstream surface (OST burn, shorts, Phase D
metadata) sees the same shape as a best-moments LF — just with one
member instead of N.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from gomoveshift.pipeline.day_grouper import DayBatch
from _state_helpers import make_state


NUM_COLUMNS = 9
COL_TIMESTAMP = 0
COL_COMPOSITE = 1
COL_MOTION_ENERGY = 2


def _make_scores(duration_sec: int, score: float = 0.7, motion: float = 0.02) -> np.ndarray:
    arr = np.zeros((duration_sec, NUM_COLUMNS), dtype=np.float64)
    arr[:, COL_TIMESTAMP] = np.arange(duration_sec, dtype=np.float64)
    arr[:, COL_COMPOSITE] = score
    # Moving footage: motion energy above the stop mask's optical floor (0.008)
    # so a GPS-less day isn't masked wholesale. A parked span is modeled by
    # zeroing this column for those seconds.
    arr[:, COL_MOTION_ENERGY] = motion
    return arr


@dataclass
class _FakeEpisode:
    index: int
    concat_path: Path
    total_duration: float
    date: str
    creation_time: object = None


def _make_day_batch(date: str, ep_count: int = 1, per_ep_duration: float = 2400.0) -> DayBatch:
    episodes = [
        _FakeEpisode(
            index=i + 1,
            concat_path=Path(f"/fake/ep_{i:03d}.mp4"),
            total_duration=per_ep_duration,
            date=date,
        )
        for i in range(ep_count)
    ]
    return DayBatch(date=date, episodes=episodes)


def _make_config(
    lf_selection_mode: str = "best_moments",
    full_drive_max_sec: int = 7200,
    part_min_sec=None,
    part_max_sec=None,
):
    """Build a tiny stand-in for ``Config`` that exposes only the fields
    ``run_day_pipeline`` reads off ``config.drive`` for this test."""
    return SimpleNamespace(
        drive=SimpleNamespace(
            brand_logo_path=None,
            montage_max_duration_sec=15.0,
            ost_title_crossfade_sec=0.5,
            lf_selection_mode=lf_selection_mode,
            full_drive_max_duration_sec=full_drive_max_sec,
            full_drive_part_min_sec=part_min_sec,
            full_drive_part_max_sec=part_max_sec,
            shorts_target_count=None,
            shorts_min_gap_sec=60.0,
        )
    )


def _patch_targets(tmp_path):
    """Common heavy-stage patches — identical pattern to test_day_pipeline.py."""
    master_path = tmp_path / "out" / "day_2026-05-01" / "day_2026-05-01_master.mp4"
    master_path.parent.mkdir(parents=True, exist_ok=True)
    master_path.write_bytes(b"fake master")

    def _fake_ost(input_path, output_path, location, layout, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake ost")
        return output_path

    def _fake_render(source_video, output_path, start_time, duration, location, **kwargs):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"fake short")
        return True

    def _fake_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"concat")
        return Path(output_path)

    def _fake_montage(raw_lf_path, peaks, output_path, work_dir, **kw):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"with_montage")
        return Path(output_path)

    return master_path, _fake_ost, _fake_render, _fake_assemble, _fake_montage


class TestFullDriveModeSynthesizesSingleLF:
    def test_full_drive_mode_calls_plan_regular_lfs_with_full_drive(self, tmp_path):
        """The wiring contract: when config requests full_drive AND no
        lf_plans are supplied, ``plan_regular_lfs`` is invoked with
        ``mode="full_drive"`` — not best_moments."""
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-05-01", ep_count=1, per_ep_duration=2400.0)
        day_scores = _make_scores(2400, score=0.7)
        output_base = tmp_path / "out"
        config = _make_config(lf_selection_mode="full_drive", full_drive_max_sec=7200)

        master_path, fake_ost, fake_render, fake_assemble, fake_montage = _patch_targets(tmp_path)
        # Forward the real plan_regular_lfs but spy on the call.
        real_plan = day_pipeline.plan_regular_lfs

        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.plan_regular_lfs",
                side_effect=real_plan,
            ) as mock_plan,
            patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=fake_ost,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=fake_render,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=fake_assemble,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=fake_montage,
            ),
        ):
            artifacts = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test Drive, Test County",
                config=config,
                lf_plans=None,
                state=make_state(ost_audit_confirmed=True),
            )

        assert artifacts is not None
        # plan_regular_lfs called exactly once with mode="full_drive"
        assert mock_plan.call_count == 1
        kwargs = mock_plan.call_args.kwargs
        assert kwargs.get("mode") == "full_drive"
        assert kwargs.get("full_drive_max_duration_sec") == 7200
        # Single LF produced, spanning the full day.
        assert len(artifacts.regular_lf_paths) == 1

    def test_full_drive_mode_raises_when_source_exceeds_ceiling(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline
        import pytest

        # 3 hours of scores, ceiling 2 hours → planner raises and the
        # day pipeline must propagate (no silent truncation).
        day_batch = _make_day_batch("2026-05-01", ep_count=1, per_ep_duration=3 * 3600)
        day_scores = _make_scores(3 * 3600, score=0.6)
        output_base = tmp_path / "out"
        config = _make_config(lf_selection_mode="full_drive", full_drive_max_sec=2 * 3600)

        master_path, *_ = _patch_targets(tmp_path)
        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            pytest.raises(ValueError, match="full_drive"),
        ):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test, County",
                config=config,
                lf_plans=None,
                state=make_state(ost_audit_confirmed=True),
            )

    def test_full_drive_mode_rejects_multi_lf_plans_without_parts(self, tmp_path):
        """With the part fields unset the contract is unchanged: a caller
        supplying several LF plans is refused (one LF per day)."""
        import pytest
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-05-01", ep_count=1, per_ep_duration=2400.0)
        config = _make_config(lf_selection_mode="full_drive")
        master_path, *_ = _patch_targets(tmp_path)
        plans = [
            SimpleNamespace(index=1, segments=[], montage_peaks=[], total_duration=1000.0),
            SimpleNamespace(index=2, segments=[], montage_peaks=[], total_duration=1000.0),
        ]
        with (
            patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master_path),
            pytest.raises(ValueError, match="exactly one"),
        ):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=_make_scores(2400),
                output_dir=tmp_path / "out",
                location="Test, County",
                config=config,
                lf_plans=plans,
                state=make_state(ost_audit_confirmed=True),
            )

    def _run_supplied(self, tmp_path, plans, config):
        from gomoveshift.pipeline import day_pipeline

        master_path, *_ = _patch_targets(tmp_path)
        with patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master_path):
            day_pipeline.run_day_pipeline(
                day_batch=_make_day_batch("2026-05-01", ep_count=1, per_ep_duration=2400.0),
                day_scores=_make_scores(2400),
                output_dir=tmp_path / "out",
                location="Test, County",
                config=config,
                lf_plans=plans,
                state=make_state(ost_audit_confirmed=True),
            )

    def test_full_drive_parts_reject_duplicate_indices(self, tmp_path):
        import pytest

        config = _make_config("full_drive", part_min_sec=1800, part_max_sec=3600)
        plans = [
            SimpleNamespace(index=1, segments=[], montage_peaks=[], total_duration=2000.0),
            SimpleNamespace(index=1, segments=[], montage_peaks=[], total_duration=2000.0),
        ]
        with pytest.raises(ValueError, match="unique"):
            self._run_supplied(tmp_path, plans, config)

    def test_full_drive_parts_reject_over_ceiling_part(self, tmp_path):
        import pytest

        config = _make_config(
            "full_drive", full_drive_max_sec=3600, part_min_sec=1800, part_max_sec=3600
        )
        plans = [
            SimpleNamespace(index=1, segments=[], montage_peaks=[], total_duration=2000.0),
            SimpleNamespace(index=2, segments=[], montage_peaks=[], total_duration=4000.0),
        ]
        with pytest.raises(ValueError, match="exceeds"):
            self._run_supplied(tmp_path, plans, config)

    def test_full_drive_parts_render_k_distinct_lfs(self, tmp_path):
        """K=2: a 7000 s day splits into two balanced LFs, rendered to
        lf_01.mp4 / lf_02.mp4 with distinct plans and non-overlapping spans."""
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-05-01", ep_count=1, per_ep_duration=7000.0)
        config = _make_config("full_drive", part_min_sec=1800, part_max_sec=3600)
        master_path, fake_ost, fake_render, fake_assemble, fake_montage = _patch_targets(tmp_path)
        captured = []

        def _assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
            captured.append((Path(output_path).name, list(segments)))
            return fake_assemble(
                master_path, segments, file_boundaries, output_path, work_dir, **kw
            )

        with (
            patch("gomoveshift.pipeline.day_pipeline.encode_day_master", return_value=master_path),
            patch("gomoveshift.pipeline.day_pipeline.apply_ost", side_effect=fake_ost),
            patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg", side_effect=fake_render
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=_assemble,
            ),
            patch("gomoveshift.pipeline.day_pipeline.prepend_montage", side_effect=fake_montage),
        ):
            artifacts = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=_make_scores(7000),
                output_dir=tmp_path / "out",
                location="Test Drive, Test County",
                config=config,
                lf_plans=None,
                state=make_state(ost_audit_confirmed=True),
            )

        names = sorted(Path(p).name for p in artifacts.regular_lf_paths)
        assert names == ["lf_01.mp4", "lf_02.mp4"]
        spans = sorted(
            (min(s.original_start_time for s in segs), max(s.original_end_time for s in segs))
            for _, segs in captured
        )
        assert len(spans) == 2
        assert spans[0][1] <= spans[1][0]
        assert abs((spans[0][1] - spans[0][0]) - 3500) < 1.0

    def test_best_moments_mode_does_not_synthesize(self, tmp_path):
        """Default mode must NOT call ``plan_regular_lfs`` from day_pipeline —
        the legacy contract is that LF plans arrive pre-built from
        ``LongFormAssembler``. Calling the planner here would be wasted
        work AND would mask a missing-lf_plans bug."""
        from gomoveshift.pipeline import day_pipeline

        day_batch = _make_day_batch("2026-05-01", ep_count=1, per_ep_duration=2400.0)
        day_scores = _make_scores(2400, score=0.7)
        output_base = tmp_path / "out"
        config = _make_config(lf_selection_mode="best_moments")

        master_path, fake_ost, fake_render, fake_assemble, fake_montage = _patch_targets(tmp_path)
        seg = SimpleNamespace(
            source_file_index=0,
            start_time=0.0,
            end_time=1200.0,
            duration=1200.0,
            gps_center_lat=0.0,
            gps_center_lon=0.0,
        )
        plan = SimpleNamespace(index=1, segments=[seg], montage_peaks=[])

        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.plan_regular_lfs",
            ) as mock_plan,
            patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=fake_ost,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=fake_render,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=fake_assemble,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=fake_montage,
            ),
        ):
            day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test, County",
                config=config,
                lf_plans=[plan],
                state=make_state(ost_audit_confirmed=True),
            )

        # best_moments path does not invoke the planner from day_pipeline.
        mock_plan.assert_not_called()


class TestFullDriveModeTrimRatioFriendly:
    """A full-drive LF strips parked spans via the per-day stop mask, so its
    trim_ratio drops BELOW 1.0 (stops removed) while staying above
    HIGHLIGHTS_TRIM_THRESHOLD (0.5) → ``_select_verb`` returns "Driving". This
    exercises the mask actually cutting the LF's segments, not a hardcoded
    ratio."""

    def test_full_drive_trim_ratio_selects_driving_verb(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline
        from gomoveshift.pipeline.longform_metadata import (
            HIGHLIGHTS_TRIM_THRESHOLD,
            _select_verb,
        )

        day_batch = _make_day_batch("2026-05-01", ep_count=1, per_ep_duration=2400.0)
        # 2400s scored day with a 400s PARKED middle span [1000, 1400): its
        # motion energy is zeroed, so the GPS-less stop mask (optical fallback)
        # masks it. The full-drive LF must physically cut it out.
        day_scores = _make_scores(2400, score=0.7)
        day_scores[1000:1400, COL_MOTION_ENERGY] = 0.0
        output_base = tmp_path / "out"
        config = _make_config(lf_selection_mode="full_drive", full_drive_max_sec=7200)

        master_path, fake_ost, fake_render, _fake_assemble, fake_montage = _patch_targets(tmp_path)

        # Capture the segments the extractor is handed — that IS the assembled
        # LF body, so its total duration is the LF's kept (post-stop) length.
        captured: dict = {}

        def _capture_assemble(master_path, segments, file_boundaries, output_path, work_dir, **kw):
            captured["segments"] = list(segments)
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_bytes(b"concat")
            return Path(output_path)

        with (
            patch(
                "gomoveshift.pipeline.day_pipeline.encode_day_master",
                return_value=master_path,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.apply_ost",
                side_effect=fake_ost,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline._render_short_ffmpeg",
                side_effect=fake_render,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.extract_and_concat_segments",
                side_effect=_capture_assemble,
            ),
            patch(
                "gomoveshift.pipeline.day_pipeline.prepend_montage",
                side_effect=fake_montage,
            ),
        ):
            artifacts = day_pipeline.run_day_pipeline(
                day_batch=day_batch,
                day_scores=day_scores,
                output_dir=output_base,
                location="Test Drive, Test County",
                config=config,
                lf_plans=None,
                state=make_state(ost_audit_confirmed=True),
            )

        assert artifacts is not None
        assert len(artifacts.regular_lf_paths) == 1
        segs = captured["segments"]
        kept = sum(s.end_time - s.start_time for s in segs)
        # The 400s parked middle is cut out — kept ≈ 2000s, not the full 2400.
        assert kept < 2400.0, f"masked stop span was NOT stripped (kept {kept})"
        assert 1900.0 <= kept <= 2100.0, f"unexpected kept duration {kept}"

        trim_ratio = kept / 2400.0
        assert trim_ratio < 1.0, "trim_ratio must reflect the stripped stops"
        assert trim_ratio >= HIGHLIGHTS_TRIM_THRESHOLD
        assert _select_verb(trim_ratio) == "Driving", (
            f"trim_ratio={trim_ratio} should pick 'Driving' (>= {HIGHLIGHTS_TRIM_THRESHOLD})"
        )
