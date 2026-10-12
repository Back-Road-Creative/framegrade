"""A shared lens model for astro frames: the geometry primitives a later stage
composes into per-frame de-rotation.

The Nikon Z DX 18-140 used for these night frames records
``DistortionCorrection: On (Required)``, but the camera only applies that
correction to JPEG — rawpy/LibRaw decode the uncorrected RAW, so every
stacked frame is barrel-distorted. Nothing here fits the model from real
frames or wires it into the pipeline; this module only provides the pure
array math a later stage will call: a shared lens model (fit once per rig)
plus a per-frame rotation on the celestial sphere.

Normalization: distortion coefficients ``k1``/``k2`` act on coordinates
normalized by ``f_px`` (focal length in pixels), i.e. ``xn = (x - cx) /
f_px`` — NOT raw pixel radius, and not radius normalized by the frame's
half-diagonal. This keeps the model self-contained (no frame shape needed to
evaluate ``undistort``/``distort``) since ``f_px`` already travels with the
other parameters. Because ``xn``/``yn`` are already order-1 at the frame
corner for a lens like this one, ``k1``/``k2`` here are order 0.01-0.1, not
the ~1e-8 px^-2 scale of a raw-pixel-radius parameterization.

Direction convention: ``undistort`` is the direct (closed-form) evaluation
of the Brown radial model -- distorted pixel in, ideal/pinhole pixel out.
``distort`` is its inverse, solved by fixed-point iteration, so a
``distort(undistort(xy))`` round trip returns (to solver tolerance) the
original distorted pixel.
"""

from dataclasses import dataclass

import numpy as np

# Fixed-point iterations for `distort`'s inversion of `undistort`. The model's
# distortion magnitude here (k1, k2 order 0.01-0.1 in normalized coordinates)
# converges to well under 1e-6 normalized units in under 10 iterations; 20 is
# a comfortable margin, not a tuned minimum.
DISTORT_ITERATIONS = 20


@dataclass(frozen=True)
class LensParams:
    """Shared lens parameters, fit once per rig (fitting itself is a later stage).

    ``k1``, ``k2``: Brown radial distortion coefficients, acting on
    coordinates normalized by ``f_px`` (see module docstring). ``cx``,
    ``cy``: principal point, pixels. ``f_px``: focal length in pixels.
    """

    k1: float
    k2: float
    cx: float
    cy: float
    f_px: float

    @classmethod
    def from_exif(
        cls, focal_length_mm, sensor_width_mm, image_width_px, k1=0.0, k2=0.0, cx=None, cy=None
    ):
        """Build params from EXIF focal length + sensor width + image width.

        ``f_px = focal_length_mm / sensor_width_mm * image_width_px`` — the
        standard pixels-per-mm conversion. ``cx``/``cy`` default to the image
        center given ``image_width_px`` (and, absent an explicit height, the
        same value for ``cy`` -- callers with a known height should pass
        ``cy`` explicitly).
        """
        f_px = focal_length_mm / sensor_width_mm * image_width_px
        if cx is None:
            cx = (image_width_px - 1) / 2
        if cy is None:
            cy = cx
        return cls(k1=k1, k2=k2, cx=cx, cy=cy, f_px=f_px)


def _split_xy(xy):
    xy = np.asarray(xy, dtype=np.float64)
    return xy[..., 0], xy[..., 1]


def undistort(xy, params: LensParams):
    """Brown radial model, direct evaluation: distorted pixel coords in,
    ideal/pinhole pixel coords out. ``xy`` is ``(..., 2)``; returns the same
    shape."""
    x, y = _split_xy(xy)
    xn = (x - params.cx) / params.f_px
    yn = (y - params.cy) / params.f_px
    r2 = xn * xn + yn * yn
    factor = 1.0 + params.k1 * r2 + params.k2 * r2 * r2
    xu = params.cx + params.f_px * xn * factor
    yu = params.cy + params.f_px * yn * factor
    return np.stack([xu, yu], axis=-1)


def distort(xy, params: LensParams, iterations: int = DISTORT_ITERATIONS):
    """Inverse of ``undistort``, by fixed-point iteration: ideal/pinhole
    pixel coords in, distorted pixel coords out, accurate to well under
    0.01 px for this module's realistic distortion magnitudes.

    Solves ``xdn * factor(xdn) = yn_target`` for the normalized distorted
    coordinate ``xdn`` by repeated substitution ``xdn <- yn_target /
    factor(xdn)``, starting from ``xdn = yn_target`` (the zero-distortion
    guess).
    """
    xu, yu = _split_xy(xy)
    xn_target = (xu - params.cx) / params.f_px
    yn_target = (yu - params.cy) / params.f_px
    xdn = xn_target.copy()
    ydn = yn_target.copy()
    for _ in range(iterations):
        r2 = xdn * xdn + ydn * ydn
        factor = 1.0 + params.k1 * r2 + params.k2 * r2 * r2
        xdn = xn_target / factor
        ydn = yn_target / factor
    xd = params.cx + params.f_px * xdn
    yd = params.cy + params.f_px * ydn
    return np.stack([xd, yd], axis=-1)


def _as_rotation_matrix(rotation):
    """Accept either a 3-vector (Rodrigues) or a 3x3 matrix; return 3x3."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape == (3, 3):
        return rotation
    if rotation.shape in ((3,), (3, 1), (1, 3)):
        rotation = rotation.reshape(3)
        theta = np.linalg.norm(rotation)
        if theta < 1e-12:
            return np.eye(3)
        axis = rotation / theta
        kx, ky, kz = axis
        k_cross = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
        return np.eye(3) + np.sin(theta) * k_cross + (1 - np.cos(theta)) * (k_cross @ k_cross)
    raise ValueError(f"rotation must be a 3-vector or 3x3 matrix, got shape {rotation.shape}")


def build_remap(shape, params: LensParams, rotation):
    """One ``(map_x, map_y)`` float32 pair, ready for a single ``cv2.remap``,
    composing undistort -> rotation on the sphere -> re-distortion.

    ``shape`` is ``(height, width)``. ``rotation`` is a 3-vector (Rodrigues)
    or a 3x3 matrix; either is converted to a 3x3 rotation applied to the
    pinhole ray of the OUTPUT pixel to find the corresponding ray in the
    SOURCE frame: ``v_source = R @ v_output``.

    The re-distortion step back into the input's distorted pixel grid is
    deliberate, not an oversight: the output canvas stays the reference
    frame's own raw (distorted) pixel grid, so a later stage's frozen ground
    band and horizon row -- defined in that same raw pixel grid -- keep their
    existing meaning. An identity rotation with zero distortion is therefore
    the identity map.
    """
    height, width = shape[:2]
    rotation_matrix = _as_rotation_matrix(rotation)

    xs, ys = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    out_grid = np.stack([xs, ys], axis=-1)

    ideal_out = undistort(out_grid, params)
    xu = (ideal_out[..., 0] - params.cx) / params.f_px
    yu = (ideal_out[..., 1] - params.cy) / params.f_px
    ones = np.ones_like(xu)
    rays = np.stack([xu, yu, ones], axis=-1)

    rays_source = rays @ rotation_matrix.T
    z = rays_source[..., 2]
    xu_source = rays_source[..., 0] / z
    yu_source = rays_source[..., 1] / z
    ideal_source = np.stack(
        [params.cx + params.f_px * xu_source, params.cy + params.f_px * yu_source], axis=-1
    )

    distorted_source = distort(ideal_source, params)
    map_x = distorted_source[..., 0].astype(np.float32)
    map_y = distorted_source[..., 1].astype(np.float32)
    return map_x, map_y
