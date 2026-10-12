"""Tests for terrain sampling, camera clearance, shot-plan serialization, and film relief overlays."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import terrain as terrain_mod
from gomoveshift.pipeline.route_story.bundle import (
    TILE_SIZE,
    Footprint,
    encode_terrain_rgb,
    lonlat_to_tile,
    tile_bounds,
    tiles_covering,
)
from gomoveshift.pipeline.route_story.opening import CameraPose
from gomoveshift.pipeline.route_story.planner import Shot, ShotPlan, assert_smooth
from gomoveshift.pipeline.route_story.terrain import TerrainGrid

ZOOM = 12
FOOTPRINT = Footprint(west=-122.45, south=37.76, east=-122.36, north=37.82)


def _write_terrain(root, heights_for_tile, zoom: int = ZOOM) -> None:
    """Write a terrain tile pyramid whose heights come from ``heights_for_tile(z, x, y)``."""
    for x, y in tiles_covering(FOOTPRINT, zoom):
        directory = root / "terrain" / str(zoom) / str(x)
        directory.mkdir(parents=True, exist_ok=True)
        heights = heights_for_tile(zoom, x, y)
        Image.fromarray(encode_terrain_rgb(heights), mode="RGB").save(directory / f"{y}.png")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "route-story-bundle/1",
                "footprint": {
                    "west": FOOTPRINT.west,
                    "south": FOOTPRINT.south,
                    "east": FOOTPRINT.east,
                    "north": FOOTPRINT.north,
                },
                "terrain": {"minzoom": zoom, "maxzoom": zoom, "encoding": "mapbox"},
            }
        ),
        encoding="utf-8",
    )


def _flat(height: float):
    return lambda z, x, y: np.full((TILE_SIZE, TILE_SIZE), height, dtype=np.float64)


def _ramp_west_to_east(low: float, high: float):
    """Heights rising linearly with tile column, identical down every row."""

    def build(z, x, y):
        row = np.linspace(low, high, TILE_SIZE, dtype=np.float64)
        return np.tile(row, (TILE_SIZE, 1))

    return build


class TestTerrainSampler:
    def test_reads_the_encoded_height_of_a_flat_tile(self, tmp_path):
        _write_terrain(tmp_path, _flat(845.0))
        sampler = terrain_mod.TerrainSampler.from_bundle(tmp_path)

        assert sampler.elevation_at(-122.40, 37.79) == pytest.approx(845.0, abs=0.05)

    def test_interpolates_between_neighbouring_samples(self, tmp_path):
        _write_terrain(tmp_path, _ramp_west_to_east(0.0, 1000.0))
        sampler = terrain_mod.TerrainSampler.from_bundle(tmp_path)

        x, y = lonlat_to_tile(-122.40, 37.79, ZOOM)
        bounds = tile_bounds(ZOOM, int(x), int(y))
        west_edge = sampler.elevation_at(bounds.west + 1e-6, 37.79)
        middle = sampler.elevation_at((bounds.west + bounds.east) / 2.0, 37.79)
        east_edge = sampler.elevation_at(bounds.east - 1e-6, 37.79)

        assert west_edge < middle < east_edge
        assert middle == pytest.approx((west_edge + east_edge) / 2.0, abs=10.0)

    def test_samples_exaggerated_height_when_the_scene_exaggerates(self, tmp_path):
        _write_terrain(tmp_path, _flat(500.0))
        sampler = terrain_mod.TerrainSampler.from_bundle(tmp_path, exaggeration=1.5)

        assert sampler.elevation_at(-122.40, 37.79) == pytest.approx(750.0, abs=0.1)

    def test_a_point_outside_the_tiles_raises(self, tmp_path):
        _write_terrain(tmp_path, _flat(100.0))
        sampler = terrain_mod.TerrainSampler.from_bundle(tmp_path)

        with pytest.raises(ValueError, match="no terrain tile"):
            sampler.elevation_at(2.35, 48.85)

    def test_a_bundle_with_no_terrain_raises(self, tmp_path):
        (tmp_path / "manifest.json").write_text(
            json.dumps({"schema": "route-story-bundle/1"}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="terrain"):
            terrain_mod.TerrainSampler.from_bundle(tmp_path)


class TestCameraAltitude:
    def test_matches_the_maplibre_projection_at_the_equator(self):
        pose = CameraPose(longitude=0.0, latitude=0.0, zoom=12.0, pitch=0.0)
        metres_per_pixel = 156543.03392804097 / (2.0**12)
        expected = (1080 / 2.0) * metres_per_pixel / math.tan(math.radians(36.87) / 2.0)

        altitude = terrain_mod.camera_altitude_m(pose, viewport_height_px=1080)

        assert altitude == pytest.approx(expected, rel=1e-9)

    def test_one_zoom_level_in_halves_the_altitude(self):
        low = CameraPose(longitude=0.0, latitude=37.79, zoom=11.0)
        high = CameraPose(longitude=0.0, latitude=37.79, zoom=12.0)

        assert terrain_mod.camera_altitude_m(high, viewport_height_px=1080) == pytest.approx(
            terrain_mod.camera_altitude_m(low, viewport_height_px=1080) / 2.0, rel=1e-9
        )

    def test_a_pitched_camera_sits_behind_its_centre_point(self):
        pose = CameraPose(longitude=-122.40, latitude=37.79, zoom=14.0, pitch=60.0, bearing=90.0)

        longitude, latitude = terrain_mod.camera_ground_point(pose, viewport_height_px=1080)

        # Bearing 90 looks east, so the camera itself stands to the west.
        assert longitude < pose.longitude
        assert latitude == pytest.approx(pose.latitude, abs=1e-4)

    def test_an_unpitched_camera_stands_over_its_centre_point(self):
        pose = CameraPose(longitude=-122.40, latitude=37.79, zoom=14.0, pitch=0.0)

        longitude, latitude = terrain_mod.camera_ground_point(pose, viewport_height_px=1080)

        assert (longitude, latitude) == pytest.approx((pose.longitude, pose.latitude))


def _single_shot_plan(poses: list[CameraPose]) -> ShotPlan:
    return ShotPlan((Shot("follow", 0, len(poses), lambda frame: poses[frame]),))


def _two_shot_plan(first: list[CameraPose], second: list[CameraPose]) -> ShotPlan:
    """Two shots meeting at a shared join pose, as ``ShotPlan.validate`` requires."""
    joined = [*first, second[0]]
    return ShotPlan(
        (
            Shot("descent", 0, len(first), lambda frame: joined[frame]),
            Shot(
                "follow",
                len(first),
                len(first) + len(second),
                lambda frame: second[frame - len(first)],
            ),
        )
    )


class TestClearTerrain:
    def test_raises_a_camera_that_would_fly_through_a_ridge(self, tmp_path):
        _write_terrain(tmp_path, _flat(2400.0))
        sampler = terrain_mod.TerrainSampler.from_bundle(tmp_path)
        poses = [CameraPose(-122.40, 37.79, 17.0, 0.0, 0.0) for _ in range(30)]
        plan = _single_shot_plan(poses)

        cleared = terrain_mod.clear_terrain(
            plan, sampler, fps=30.0, viewport_height_px=1080, margin_m=150.0
        )

        for frame in range(cleared.frame_count):
            pose = cleared.pose_at(frame)
            altitude = terrain_mod.camera_altitude_m(pose, viewport_height_px=1080)
            assert altitude >= 2400.0 + 150.0 - 1e-6
            assert pose.zoom < poses[frame].zoom
            assert (pose.longitude, pose.latitude) == (
                poses[frame].longitude,
                poses[frame].latitude,
            )
            assert (pose.pitch, pose.bearing) == (poses[frame].pitch, poses[frame].bearing)

    def test_leaves_a_plan_with_clearance_untouched(self, tmp_path):
        _write_terrain(tmp_path, _flat(20.0))
        sampler = terrain_mod.TerrainSampler.from_bundle(tmp_path)
        poses = [CameraPose(-122.40, 37.79, 13.0, 30.0, 10.0) for _ in range(30)]
        plan = _single_shot_plan(poses)

        cleared = terrain_mod.clear_terrain(
            plan, sampler, fps=30.0, viewport_height_px=1080, margin_m=150.0
        )

        for frame in range(plan.frame_count):
            assert cleared.pose_at(frame) == poses[frame]

    def test_the_cleared_plan_still_validates_and_stays_smooth(self, tmp_path):
        _write_terrain(tmp_path, _ramp_west_to_east(50.0, 3000.0))
        sampler = terrain_mod.TerrainSampler.from_bundle(tmp_path)
        longitudes = np.linspace(-122.44, -122.37, 60)
        first = [CameraPose(float(lon), 37.79, 15.0, 45.0, 90.0) for lon in longitudes[:30]]
        second = [CameraPose(float(lon), 37.79, 15.0, 45.0, 90.0) for lon in longitudes[30:]]
        plan = _two_shot_plan(first, second)

        cleared = terrain_mod.clear_terrain(
            plan, sampler, fps=30.0, viewport_height_px=1080, margin_m=150.0
        )

        cleared.validate(tolerance=1e-9)
        assert_smooth(cleared, fps=30.0)

    def test_a_terrain_hole_keeps_the_planned_pose_instead_of_failing(self, tmp_path):
        _write_terrain(tmp_path, _flat(100.0))
        sampler = terrain_mod.TerrainSampler.from_bundle(tmp_path)
        poses = [CameraPose(2.35, 48.85, 15.0, 0.0, 0.0) for _ in range(10)]
        plan = _single_shot_plan(poses)

        cleared = terrain_mod.clear_terrain(
            plan, sampler, fps=30.0, viewport_height_px=1080, margin_m=150.0
        )

        assert [cleared.pose_at(f) for f in range(10)] == poses

    def test_clearance_never_zooms_out_past_the_floor(self, tmp_path):
        _write_terrain(tmp_path, _flat(8000.0))
        sampler = terrain_mod.TerrainSampler.from_bundle(tmp_path)
        poses = [CameraPose(-122.40, 37.79, 16.0, 0.0, 0.0) for _ in range(20)]
        plan = _single_shot_plan(poses)

        cleared = terrain_mod.clear_terrain(
            plan,
            sampler,
            fps=30.0,
            viewport_height_px=1080,
            margin_m=150.0,
            min_zoom=15.0,
        )

        for frame in range(cleared.frame_count):
            assert cleared.pose_at(frame).zoom >= 15.0


class TestPlanToJson:
    def test_carries_every_frame_pose_and_the_timing(self, tmp_path):
        poses = [CameraPose(-122.40 + i * 1e-4, 37.79, 14.0, 30.0, 120.0) for i in range(5)]
        plan = _single_shot_plan(poses)

        payload = terrain_mod.plan_to_json(plan, fps=30.0)

        assert payload["schema"] == terrain_mod.PLAN_SCHEMA
        assert payload["fps"] == 30.0
        assert payload["frame_count"] == 5
        assert [p["frame"] for p in payload["poses"]] == [0, 1, 2, 3, 4]
        assert payload["poses"][3]["longitude"] == pytest.approx(poses[3].longitude)
        assert payload["poses"][3]["bearing"] == pytest.approx(120.0)
        assert payload["shots"] == [{"kind": "follow", "start_frame": 0, "end_frame": 5}]

    def test_round_trips_through_json_text(self, tmp_path):
        poses = [CameraPose(-122.40, 37.79, 14.0) for _ in range(3)]
        plan = _single_shot_plan(poses)

        destination = terrain_mod.write_plan_json(plan, fps=24.0, path=tmp_path / "plan.json")
        reloaded = json.loads(destination.read_text(encoding="utf-8"))

        assert reloaded == terrain_mod.plan_to_json(plan, fps=24.0)

    def test_refuses_a_non_positive_fps(self):
        plan = _single_shot_plan([CameraPose(0.0, 0.0, 10.0) for _ in range(2)])

        with pytest.raises(ValueError, match="fps"):
            terrain_mod.plan_to_json(plan, fps=0.0)


class TestTerrainGrid:
    def test_terrain_grid_builds_a_real_relief_overlay(self, tmp_path):
        samples = []
        for y, row in enumerate((100.0, 120.0)):
            for x, column in enumerate((-80.0, -79.9)):
                samples.append(
                    {"location": {"x": column, "y": row}, "value": str(100 + x * 20 + y * 10)}
                )
        source = tmp_path / "dem.json"
        source.write_text(json.dumps({"samples": samples}), encoding="utf-8")

        grid = TerrainGrid.from_json(source)
        overlay = grid.overlay(-80.0, 100.0, -79.9, 120.0, 64, 32)

        assert grid.elevations_m.shape == (2, 2)
        assert overlay.mode == "RGBA"
        assert overlay.size == (64, 32)
        assert np.asarray(overlay)[:, :, 3].max() > 0
