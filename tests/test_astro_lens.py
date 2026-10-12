"""``astro_lens``: distort/undistort round trip and remap composition.

Pure geometry -- no real card, no other ``astro_*`` module involved.
"""

import numpy as np
import pytest

from gomoveshift.pipeline.photos.astro_lens import (
    LensParams,
    build_remap,
    distort,
    undistort,
)

WIDTH = 5600
HEIGHT = 3728

# A realistic barrel-distortion rig: negative k1 pulls corners in. Magnitude
# chosen against this module's own normalization (coordinates divided by
# f_px, see astro_lens module docstring) -- NOT the raw-pixel-radius ~1e-8
# scale, since here xn/yn are already order-1 at the corner.
PARAMS = LensParams(k1=-0.08, k2=0.01, cx=(WIDTH - 1) / 2, cy=(HEIGHT - 1) / 2, f_px=4289.36)


def _frame_grid(step=200):
    xs = np.arange(0, WIDTH, step, dtype=np.float64)
    ys = np.arange(0, HEIGHT, step, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)


def test_distort_undistort_round_trip_sub_hundredth_px():
    grid = _frame_grid()
    ideal = undistort(grid, PARAMS)
    recovered = distort(ideal, PARAMS)
    error = np.hypot(recovered[..., 0] - grid[..., 0], recovered[..., 1] - grid[..., 1])
    assert float(error.max()) < 0.01


def test_undistort_moves_corner_but_not_center():
    center = np.array([PARAMS.cx, PARAMS.cy])
    corner = np.array([0.0, 0.0])
    ideal_center = undistort(center, PARAMS)
    ideal_corner = undistort(corner, PARAMS)
    assert np.hypot(*(ideal_center - center)) < 1e-9
    assert np.hypot(*(ideal_corner - corner)) > 1.0


def test_f_px_from_exif_matches_expected_value():
    params = LensParams.from_exif(focal_length_mm=18.0, sensor_width_mm=23.5, image_width_px=5600)
    assert params.f_px == pytest.approx(4289.36, abs=1.0)


def test_build_remap_identity_rotation_zero_distortion_is_identity():
    shape = (100, 150)
    zero_params = LensParams(k1=0.0, k2=0.0, cx=74.5, cy=49.5, f_px=200.0)
    identity_rotation = np.eye(3)
    map_x, map_y = build_remap(shape, zero_params, identity_rotation)
    xs, ys = np.meshgrid(np.arange(150, dtype=np.float32), np.arange(100, dtype=np.float32))
    assert np.abs(map_x - xs).max() < 1e-4
    assert np.abs(map_y - ys).max() < 1e-4


def test_build_remap_small_rotation_shifts_center_by_angle_times_f_px():
    shape = (100, 150)
    zero_params = LensParams(k1=0.0, k2=0.0, cx=74.5, cy=49.5, f_px=200.0)
    theta = 0.01  # radians, small pan about the y axis
    rotation_vector = np.array([0.0, theta, 0.0])
    map_x, map_y = build_remap(shape, zero_params, rotation_vector)
    cx_i, cy_i = int(round(zero_params.cx)), int(round(zero_params.cy))
    shift = map_x[cy_i, cx_i] - cx_i
    assert shift == pytest.approx(zero_params.f_px * theta, rel=0.05)


def test_build_remap_dtype_and_shape_match_cv2_remap_contract():
    shape = (100, 150)
    map_x, map_y = build_remap(shape, PARAMS, np.eye(3))
    assert map_x.dtype == np.float32
    assert map_y.dtype == np.float32
    assert map_x.shape == (100, 150)
    assert map_y.shape == (100, 150)
