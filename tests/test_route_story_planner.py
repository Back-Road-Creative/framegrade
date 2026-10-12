"""Tests for the Route Story drone shot planner (gomoveshift.pipeline.route_story.planner)."""

from __future__ import annotations

import math

import pytest

from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.opening import CameraPose
from gomoveshift.pipeline.route_story.planner import (
    DEFAULT_MOTION_LIMITS,
    ENLARGEMENT_MAX,
    Shot,
    ShotPlan,
    _haversine_m,
    arc,
    arrival,
    assert_smooth,
    descent,
    MotionReport,
    fit_zoom,
    hectic_windows,
    max_shot_zoom,
    motion_report,
    motion_report_from_poses,
    motion_summary,
    pullback,
    return_to,
    reveal,
    summarize_plan_motion,
)


def _still_shot(kind: str, start_frame: int, end_frame: int, pose: CameraPose) -> Shot:
    return Shot(kind, start_frame, end_frame, lambda frame: pose)


class TestShotAndPlan:
    def test_pose_at_rejects_outside_range(self) -> None:
        shot = _still_shot("reveal", 0, 5, CameraPose(0.0, 0.0, 10.0))
        with pytest.raises(ValueError):
            shot.pose_at(5)
        with pytest.raises(ValueError):
            shot.pose_at(-1)

    def test_shot_rejects_non_positive_range(self) -> None:
        with pytest.raises(ValueError):
            Shot("reveal", 5, 5, lambda frame: CameraPose(0.0, 0.0, 10.0))

    def test_plan_frame_count_and_pose_lookup(self) -> None:
        first = _still_shot("reveal", 0, 3, CameraPose(0.0, 0.0, 10.0))
        second = _still_shot("arc", 3, 6, CameraPose(0.0, 0.0, 10.0))
        plan = ShotPlan((first, second))
        assert plan.frame_count == 6
        assert plan.pose_at(0).zoom == 10.0
        assert plan.pose_at(5).zoom == 10.0
        with pytest.raises(ValueError):
            plan.pose_at(6)

    def test_validate_detects_gap(self) -> None:
        first = _still_shot("reveal", 0, 3, CameraPose(0.0, 0.0, 10.0))
        second = _still_shot("arc", 4, 6, CameraPose(0.0, 0.0, 10.0))
        with pytest.raises(ValueError, match="gap"):
            ShotPlan((first, second)).validate()

    def test_validate_detects_overlap(self) -> None:
        first = _still_shot("reveal", 0, 3, CameraPose(0.0, 0.0, 10.0))
        second = _still_shot("arc", 2, 6, CameraPose(0.0, 0.0, 10.0))
        with pytest.raises(ValueError, match="overlap"):
            ShotPlan((first, second)).validate()

    def test_validate_detects_pose_discontinuity(self) -> None:
        first = _still_shot("reveal", 0, 3, CameraPose(0.0, 0.0, 10.0))
        second = _still_shot("arc", 3, 6, CameraPose(1.0, 0.0, 10.0))
        with pytest.raises(ValueError):
            ShotPlan((first, second)).validate()

    def test_validate_passes_on_builder_join(self) -> None:
        start = CameraPose(-71.0, 42.0, 8.0)
        target = RoutePoint(0.0, 42.05, -71.05)
        shot_a = descent(start, target, fps=2.0, seconds=14.0)
        end_pose = shot_a._boundary_pose(shot_a.end_frame)
        shot_b = reveal(end_pose, fps=2.0, seconds=6.0, start_frame=shot_a.end_frame)
        ShotPlan((shot_a, shot_b)).validate()


class TestFitZoom:
    def test_smaller_bbox_needs_higher_zoom(self) -> None:
        small = fit_zoom((-71.01, 41.99, -70.99, 42.01), 1920, 1080, 80)
        large = fit_zoom((-72.0, 41.0, -70.0, 43.0), 1920, 1080, 80)
        assert small > large

    def test_more_padding_reduces_zoom(self) -> None:
        tight = fit_zoom((-71.01, 41.99, -70.99, 42.01), 1920, 1080, 10)
        padded = fit_zoom((-71.01, 41.99, -70.99, 42.01), 1920, 1080, 400)
        assert padded < tight

    def test_point_bbox_clamps_to_max_zoom(self) -> None:
        assert fit_zoom((-71.0, 42.0, -71.0, 42.0), 1920, 1080, 0) == pytest.approx(22.0)


class TestDescent:
    def test_rejects_out_of_range_duration(self) -> None:
        start = CameraPose(-71.0, 42.0, 3.0)
        target = RoutePoint(0.0, 42.0, -71.0)
        with pytest.raises(ValueError):
            descent(start, target, fps=2.0, seconds=10.0)

    def test_target_locks_the_end_pose(self) -> None:
        start = CameraPose(-71.0, 42.0, 3.0)
        target = RoutePoint(0.0, 42.05, -71.05)
        shot = descent(start, target, fps=2.0, seconds=14.0, final_zoom=15.0)
        end_pose = shot._boundary_pose(shot.end_frame)
        assert end_pose.longitude == pytest.approx(target.longitude, abs=1e-6)
        assert end_pose.latitude == pytest.approx(target.latitude, abs=1e-6)
        assert end_pose.zoom == pytest.approx(15.0)
        assert end_pose.pitch == 0.0

    def test_starts_at_the_given_pose(self) -> None:
        start = CameraPose(-71.0, 42.0, 3.0)
        target = RoutePoint(0.0, 42.05, -71.05)
        shot = descent(start, target, fps=2.0, seconds=14.0)
        begin_pose = shot.pose_at(0)
        assert begin_pose.longitude == pytest.approx(start.longitude, abs=1e-9)
        assert begin_pose.latitude == pytest.approx(start.latitude, abs=1e-9)
        assert begin_pose.zoom == pytest.approx(start.zoom)

    def test_zoom_is_monotonic(self) -> None:
        start = CameraPose(-71.0, 42.0, 3.0)
        target = RoutePoint(0.0, 42.05, -71.05)
        shot = descent(start, target, fps=2.0, seconds=14.0, final_zoom=15.0)
        zooms = [shot.pose_at(f).zoom for f in range(shot.start_frame, shot.end_frame)]
        assert zooms == sorted(zooms)


class TestReveal:
    def test_rejects_out_of_range(self) -> None:
        at = CameraPose(-71.0, 42.0, 15.0)
        with pytest.raises(ValueError):
            reveal(at, fps=2.0, seconds=1.0)
        with pytest.raises(ValueError):
            reveal(at, fps=2.0, seconds=7.0, final_pitch=10.0)

    def test_pitch_rises_smoothly_to_target(self) -> None:
        at = CameraPose(-71.0, 42.0, 15.0)
        shot = reveal(at, fps=2.0, seconds=6.0, final_pitch=40.0)
        assert shot.pose_at(shot.start_frame).pitch == pytest.approx(0.0)
        end_pose = shot._boundary_pose(shot.end_frame)
        assert end_pose.pitch == pytest.approx(40.0)
        pitches = [shot.pose_at(f).pitch for f in range(shot.start_frame, shot.end_frame)]
        assert pitches == sorted(pitches)


class TestArc:
    def test_rejects_out_of_range(self) -> None:
        center = RoutePoint(0.0, 42.0, -71.0)
        base = CameraPose(-71.01, 42.0, 14.0, pitch=55.0)
        with pytest.raises(ValueError):
            arc(center, base, fps=2.0, seconds=1.0)
        with pytest.raises(ValueError):
            arc(center, base, fps=2.0, seconds=7.5, degrees=5.0)

    def test_orbits_around_the_center_preserving_radius(self) -> None:
        center = RoutePoint(0.0, 42.0, -71.0)
        base = CameraPose(-71.01, 42.0, 14.0, pitch=55.0)
        shot = arc(center, base, fps=2.0, seconds=7.5, degrees=30.0)
        start_pose = shot.pose_at(shot.start_frame)
        end_pose = shot._boundary_pose(shot.end_frame)
        assert start_pose.longitude == pytest.approx(base.longitude, abs=1e-6)
        assert start_pose.latitude == pytest.approx(base.latitude, abs=1e-6)
        assert (start_pose.longitude, start_pose.latitude) != (
            end_pose.longitude,
            end_pose.latitude,
        )
        assert end_pose.zoom == base.zoom
        assert end_pose.pitch == base.pitch
        radius0 = _haversine_m(base.longitude, base.latitude, center.longitude, center.latitude)
        radius_end = _haversine_m(
            end_pose.longitude, end_pose.latitude, center.longitude, center.latitude
        )
        assert radius_end == pytest.approx(radius0, rel=1e-6)


class TestArrivalAndReturn:
    def test_return_ends_exactly_at_the_pose_it_left(self) -> None:
        base_after = CameraPose(-71.0, 42.0, 14.0, pitch=50.0, bearing=10.0)
        current = CameraPose(-71.02, 42.01, 15.5, pitch=55.0, bearing=340.0)
        shot = return_to(base_after, current, fps=10.0, seconds=0.7)
        end_pose = shot._boundary_pose(shot.end_frame)
        assert end_pose == base_after

    def test_arrival_moves_from_base_toward_target(self) -> None:
        base = CameraPose(-71.0, 42.0, 14.0)
        target = CameraPose(-71.0, 42.0, 15.5, pitch=45.0)
        shot = arrival(base, target, fps=10.0, seconds=2.0)
        end_pose = shot._boundary_pose(shot.end_frame)
        assert end_pose.zoom == pytest.approx(target.zoom)
        assert end_pose.pitch == pytest.approx(target.pitch)


class TestPullback:
    def test_rises_to_fit_the_route_bbox(self) -> None:
        end = CameraPose(-71.0, 42.0, 15.5, pitch=45.0)
        bbox = (-71.2, 41.8, -70.8, 42.2)
        shot = pullback(end, bbox, fps=10.0, seconds=6.0)
        end_pose = shot._boundary_pose(shot.end_frame)
        assert end_pose.zoom == pytest.approx(fit_zoom(bbox, 1920.0, 1080.0, 80.0))
        assert end_pose.pitch == pytest.approx(0.0, abs=1e-9)

    def test_min_zoom_floors_the_target_zoom(self) -> None:
        end = CameraPose(-71.0, 42.0, 15.5, pitch=45.0)
        bbox = (-71.2, 41.8, -70.8, 42.2)
        natural_zoom = fit_zoom(bbox, 1920.0, 1080.0, 80.0)
        floor = natural_zoom + 2.0

        shot = pullback(end, bbox, fps=10.0, seconds=6.0, min_zoom=floor)

        end_pose = shot._boundary_pose(shot.end_frame)
        assert end_pose.zoom == pytest.approx(floor)
        zooms = [shot.pose_at(f).zoom for f in range(shot.start_frame, shot.end_frame)]
        assert min(zooms) >= floor - 1e-9


class TestMotionBounds:
    def test_chained_builder_shots_pass_default_limits(self) -> None:
        start = CameraPose(-71.0, 42.0, 3.0)
        target = RoutePoint(0.0, 42.05, -71.05)
        descent_shot = descent(start, target, fps=30.0, seconds=16.0, final_zoom=12.0)
        end_pose = descent_shot._boundary_pose(descent_shot.end_frame)
        reveal_shot = reveal(end_pose, fps=30.0, seconds=7.0, start_frame=descent_shot.end_frame)
        plan = ShotPlan((descent_shot, reveal_shot))
        plan.validate()
        assert_smooth(plan, fps=30.0)

    @pytest.mark.parametrize(
        "shot_factory",
        [
            lambda: descent(
                CameraPose(-71.0, 42.0, 3.0),
                RoutePoint(0.0, 42.05, -71.05),
                fps=30.0,
                seconds=16.0,
                final_zoom=12.0,
            ),
            lambda: reveal(CameraPose(-71.0, 42.0, 15.0), fps=30.0, seconds=7.0, final_pitch=40.0),
            lambda: arc(
                RoutePoint(0.0, 42.0, -71.0),
                CameraPose(-71.01, 42.0, 14.0, pitch=55.0),
                fps=30.0,
                seconds=8.0,
                degrees=25.0,
            ),
        ],
    )
    def test_each_builder_passes_default_limits(self, shot_factory) -> None:
        shot = shot_factory()
        assert_smooth(ShotPlan((shot,)), fps=30.0)

    def test_assert_smooth_names_the_worst_frame(self) -> None:
        bad = Shot(
            "reveal",
            0,
            2,
            lambda frame: CameraPose(0.0, 0.0, 10.0, bearing=0.0 if frame == 0 else 300.0),
        )
        plan = ShotPlan((bad,))
        with pytest.raises(ValueError, match="frame 0"):
            assert_smooth(plan, fps=30.0)


class TestMotionSummary:
    def test_all_still_plan_is_zero(self) -> None:
        pose = CameraPose(-71.0, 42.0, 10.0)
        shot = _still_shot("reveal", 0, 5, pose)
        plan = ShotPlan((shot,))
        summary = summarize_plan_motion(plan, fps=30.0)
        assert summary.frame_count == 5
        assert summary.fps == 30.0
        assert summary.bearing_speed.mean_abs == 0.0
        assert summary.bearing_speed.p95_abs == 0.0
        assert summary.bearing_speed.max_abs == 0.0
        assert summary.bearing_speed.frames_over_limit == 0
        assert summary.bearing_accel.mean_abs == 0.0
        assert summary.bearing_accel.frames_over_limit == 0
        assert summary.pitch_speed_max_abs == 0.0
        assert summary.zoom_velocity_max_abs == 0.0

    def test_constant_bearing_rate(self) -> None:
        rate_per_frame = 0.1
        fps = 30.0
        expected_speed = rate_per_frame * fps  # 3.0 deg/s
        shot = Shot(
            "reveal",
            0,
            10,
            lambda frame: CameraPose(0.0, 0.0, 10.0, bearing=frame * rate_per_frame),
        )
        plan = ShotPlan((shot,))
        report = motion_report(plan, fps=fps)

        default_summary = motion_summary(report, fps=fps)
        assert default_summary.bearing_speed.mean_abs == pytest.approx(expected_speed)
        assert default_summary.bearing_speed.p95_abs == pytest.approx(expected_speed)
        assert default_summary.bearing_speed.max_abs == pytest.approx(expected_speed)
        assert default_summary.bearing_speed.frames_over_limit == 0
        assert default_summary.frame_count == 10

        tight_limits = dict(DEFAULT_MOTION_LIMITS)
        tight_limits["bearing_speed_deg_s"] = expected_speed / 2.0
        tight_summary = motion_summary(report, fps=fps, limits=tight_limits)
        assert tight_summary.bearing_speed.frames_over_limit == default_summary.frame_count - 1

    def test_single_frame_plan_has_no_division_by_zero(self) -> None:
        shot = _still_shot("reveal", 0, 1, CameraPose(-71.0, 42.0, 10.0))
        plan = ShotPlan((shot,))
        summary = summarize_plan_motion(plan, fps=30.0)
        assert summary.frame_count == 1
        assert summary.bearing_speed.mean_abs == 0.0
        assert summary.bearing_speed.max_abs == 0.0
        assert summary.bearing_speed.frames_over_limit == 0
        assert summary.pitch_speed_max_abs == 0.0
        assert summary.zoom_velocity_max_abs == 0.0

    def test_motion_report_from_poses_matches_motion_report_for_a_plan(self) -> None:
        fps = 24.0
        shot = Shot(
            "reveal",
            0,
            8,
            lambda frame: CameraPose(0.0, 0.0, 10.0 + frame * 0.05, bearing=frame * 0.3),
        )
        plan = ShotPlan((shot,))
        poses = [plan.pose_at(frame) for frame in range(plan.frame_count)]

        from_poses = motion_report_from_poses(poses, fps)
        from_plan = motion_report(plan, fps)

        assert from_poses == from_plan

    def test_motion_report_from_poses_accepts_camera_state_field_names(self) -> None:
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class _Pose:
            bearing_deg: float
            pitch_deg: float
            zoom: float

        fps = 10.0
        poses = [_Pose(bearing_deg=0.0, pitch_deg=60.0, zoom=15.0), _Pose(1.0, 60.0, 15.5)]

        report = motion_report_from_poses(poses, fps)

        assert report.bearing_speed_deg_s == (pytest.approx(10.0),)
        assert report.pitch_speed_deg_s == (pytest.approx(0.0),)
        assert report.zoom_velocity_per_s == (pytest.approx(5.0),)

    def test_as_receipt_round_trips_through_json(self) -> None:
        import json

        pose = CameraPose(-71.0, 42.0, 10.0)
        shot = _still_shot("reveal", 0, 5, pose)
        plan = ShotPlan((shot,))
        summary = summarize_plan_motion(plan, fps=30.0)
        receipt = summary.as_receipt()
        encoded = json.dumps(receipt)
        decoded = json.loads(encoded)
        assert decoded["motion"]["frame_count"] == 5
        assert decoded["motion"]["fps"] == 30.0
        assert decoded["motion"]["bearing_speed"]["mean_abs"] == 0.0
        assert decoded["motion"]["limits"] == {
            key: round(value, 3) for key, value in DEFAULT_MOTION_LIMITS.items()
        }


class TestEnlargementCap:
    def test_max_shot_zoom_applies_the_default_enlargement_cap(self) -> None:
        assert max_shot_zoom(15.0) == pytest.approx(15.3219, abs=1e-4)

    def test_max_shot_zoom_rejects_sub_unity_enlargement(self) -> None:
        with pytest.raises(ValueError):
            max_shot_zoom(15.0, enlargement=0.9)

    def test_max_shot_zoom_default_is_enlargement_max_constant(self) -> None:
        assert max_shot_zoom(15.0) == pytest.approx(15.0 + math.log2(ENLARGEMENT_MAX))

    def test_assert_within_zoom_raises_naming_the_offending_pose(self) -> None:
        base = CameraPose(-71.0, 42.0, 14.0)
        target = CameraPose(-71.0, 42.0, 15.6, pitch=45.0)
        shot = arrival(base, target, fps=10.0, seconds=2.0)
        plan = ShotPlan((shot,))
        max_zoom = max_shot_zoom(15.0)
        with pytest.raises(ValueError, match=r"shot 0 \('arrival'\).*zoom 15\.5\d+"):
            plan.assert_within_zoom(max_zoom)

    def test_assert_within_zoom_passes_inside_the_cap(self) -> None:
        base = CameraPose(-71.0, 42.0, 14.0)
        target = CameraPose(-71.0, 42.0, 15.2, pitch=45.0)
        shot = arrival(base, target, fps=10.0, seconds=2.0)
        plan = ShotPlan((shot,))
        max_zoom = max_shot_zoom(15.0)
        plan.assert_within_zoom(max_zoom)


def _bearing_speed_report(values: list[float]) -> MotionReport:
    return MotionReport(
        bearing_speed_deg_s=tuple(values),
        bearing_accel_deg_s2=(),
        bearing_jerk_deg_s3=(),
        pitch_speed_deg_s=(),
        zoom_velocity_per_s=(),
    )


class TestHecticWindows:
    def test_picks_highest_magnitude_peaks_spaced_apart_and_sorted_by_start(self) -> None:
        fps = 10.0
        series = [0.0] * 200
        series[100] = 95.0
        series[105] = -94.0  # near-duplicate peak, too close to the chosen one
        series[20] = 90.0
        series[180] = 70.0
        report = _bearing_speed_report(series)

        windows = hectic_windows(report, fps, count=3, window_s=4.0, min_gap_s=6.0)

        assert windows == [
            pytest.approx((0.05, 4.05)),
            pytest.approx((8.05, 12.05)),
            pytest.approx((16.05, 20.05)),
        ]

    def test_clips_a_window_to_the_film_bounds(self) -> None:
        fps = 10.0
        series = [0.0] * 20
        series[0] = 99.0
        report = _bearing_speed_report(series)

        windows = hectic_windows(report, fps, count=1, window_s=4.0, min_gap_s=6.0)

        assert windows == [pytest.approx((0.0, 2.05))]

    def test_returns_at_most_count_windows(self) -> None:
        fps = 10.0
        series = [0.0] * 200
        series[10] = 50.0
        series[80] = 60.0
        series[150] = 70.0
        report = _bearing_speed_report(series)

        windows = hectic_windows(report, fps, count=2, window_s=4.0, min_gap_s=6.0)

        assert len(windows) == 2

    def test_empty_for_fewer_than_two_frames(self) -> None:
        assert hectic_windows(_bearing_speed_report([]), 10.0) == []
        assert hectic_windows(_bearing_speed_report([1.0]), 10.0) == []
