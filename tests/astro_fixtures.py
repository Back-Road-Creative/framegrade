"""Synthetic scene-linear astro fixtures. Frames are float32 RGB, shaped
``(H, W, 3)`` -- ``astro_calibrate``/``astro_stack``'s own contract; lower-clipped at 0,
not upper-clipped, so bright stars/pedestals may exceed 1.0 by design. Pure
numpy, no file I/O, no RAW dependency. Every builder returns ``(array_or_list,
ground_truth: dict)`` so a metric helper
(``gomoveshift.pipeline.photos.astro_quality``) can compare a measurement to what
was actually injected. ``sky_frame`` components are each
independently switchable (falsy/``None`` disables) and always recorded in the truth dict.
``multi_frame_sequence`` shifts sky content frame to frame while sensor artifacts (hot
pixels, a sensor-fixed streak, warp borders) repaint at identical coordinates every frame.
"""

import numpy as np

DEFAULT_SHAPE = (128, 128)  # (H, W); small enough that the whole module runs well under 1s.


def gaussian_stack(n, shape=DEFAULT_SHAPE, sigma=0.02, background=0.30, seed=0):
    """Clean iid ``Normal(background, sigma)`` float32 stack -- no injected signal."""
    rng = np.random.default_rng(seed)
    frames = [
        np.clip(background + sigma * rng.standard_normal((*shape, 3)), 0.0, None).astype(np.float32)
        for _ in range(n)
    ]
    truth = {"n": n, "shape": shape, "sigma": sigma, "background": background, "seed": seed}
    return frames, truth


def _paint_streak(frame, endpoints, value):
    """Paint a straight ``value`` line between pixel ``(y, x)`` endpoints (coarse)."""
    (y0, x0), (y1, x1) = endpoints
    steps = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    ys = np.linspace(y0, y1, steps).round().astype(int)
    xs = np.linspace(x0, x1, steps).round().astype(int)
    h, w = frame.shape[:2]
    valid = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    frame[ys[valid], xs[valid]] = value


def _zero_border(frame, border_px):
    frame[:border_px, :] = 0.0
    frame[-border_px:, :] = 0.0
    frame[:, :border_px] = 0.0
    frame[:, -border_px:] = 0.0


def sky_frame(
    shape=DEFAULT_SHAPE,
    *,
    background=0.02,
    gradient_amplitude=0.0,
    gradient_angle_deg=30.0,
    nebulosity_contrast=None,
    nebulosity_center=(0.5, 0.5),
    nebulosity_radius_frac=0.35,
    stars=(),
    cloud_pedestal_level=None,
    cloud_pedestal_box=(0.3, 0.7, 0.3, 0.7),
    hot_pixel_sites=(),
    hot_pixel_value=1.0,
    streak=None,
    fixed_streak=None,
    streak_value=0.8,
    warp_border_px=0,
    ground_profile_frac=None,
    ground_level=0.01,
    noise_sigma=0.0,
    seed=0,
):
    """One scene-linear ``(H, W, 3)`` float32 sky frame; every component is independently
    switchable (falsy/``None`` disables it) and always recorded in the truth dict.
    ``stars`` entries are dicts ``{"x", "y", "flux", "psf_sigma", "rgb"}``; streak
    endpoints are ``((y0, x0), (y1, x1))``. Hot pixels/streaks/border paint last (after
    noise), so they are exact regardless of ``noise_sigma``."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frame = np.full((h, w, 3), background, dtype=np.float32)
    truth = {"shape": shape, "background": background, "gradient_amplitude": gradient_amplitude}
    theta = np.deg2rad(gradient_angle_deg)  # light-pollution ramp, normalised to [0, 1]
    ramp = (xx / w) * np.cos(theta) + (yy / h) * np.sin(theta)
    ramp = (ramp - ramp.min()) / max(float(ramp.max() - ramp.min()), 1e-6)
    frame += (gradient_amplitude * ramp).astype(np.float32)[..., None]
    sky = frame[..., 0].copy()  # realised local sky (bg + gradient); nebulosity scales off this
    if nebulosity_contrast:
        cy, cx = nebulosity_center[0] * h, nebulosity_center[1] * w
        radius = nebulosity_radius_frac * min(h, w)
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        profile = np.clip(1.0 - dist / radius, 0.0, 1.0)  # broad falloff, never a hard disk
        frame += (nebulosity_contrast * sky * profile).astype(np.float32)[..., None]
        truth["nebulosity_mask"] = profile > 0.0
    else:
        truth["nebulosity_mask"] = None
    truth["nebulosity_contrast"] = nebulosity_contrast
    star_truth = []
    for s in stars:
        rgb = s.get("rgb", (1.0, 1.0, 1.0))
        g = np.exp(-((xx - s["x"]) ** 2 + (yy - s["y"]) ** 2) / (2.0 * s["psf_sigma"] ** 2))
        g *= s["flux"] / (2.0 * np.pi * s["psf_sigma"] ** 2)  # PSF integrates to ``flux``
        frame += np.stack([g * ratio for ratio in rgb], axis=-1).astype(np.float32)
        star_truth.append({**s, "rgb": rgb})
    truth["stars"] = star_truth
    if cloud_pedestal_level:
        y0, y1, x0, x1 = cloud_pedestal_box
        box = (int(y0 * h), int(y1 * h), int(x0 * w), int(x1 * w))
        frame[box[0] : box[1], box[2] : box[3]] += cloud_pedestal_level
        truth["cloud_pedestal"] = {"level": cloud_pedestal_level, "box": box}
    else:
        truth["cloud_pedestal"] = None
    if ground_profile_frac is not None:
        boundary_row = np.asarray(ground_profile_frac, dtype=np.float32) * h
        gmask = np.arange(h, dtype=np.float32)[:, None] >= boundary_row[None, :]
        frame[gmask] = ground_level  # irregular silhouette: per-column boundary, never flat
        truth["ground_mask"] = gmask
    else:
        truth["ground_mask"] = None
    if noise_sigma:
        rng = np.random.default_rng(seed)
        frame = frame + noise_sigma * rng.standard_normal(frame.shape).astype(np.float32)
    for ry, rx in hot_pixel_sites:
        frame[ry, rx] = hot_pixel_value
    truth["hot_pixel_sites"] = tuple(hot_pixel_sites)
    for key, endpoints in (("streak", streak), ("fixed_streak", fixed_streak)):
        if endpoints:
            _paint_streak(frame, endpoints, streak_value)
        truth[key] = endpoints
    truth["warp_border_px"] = warp_border_px
    if warp_border_px > 0:
        _zero_border(frame, warp_border_px)
    return np.clip(frame, 0.0, None).astype(np.float32), truth


def multi_frame_sequence(
    n,
    shape=DEFAULT_SHAPE,
    shift_step=(1.0, 0.6),
    noise_sigma=0.01,
    seed=0,
    streak_frame_index=None,
    **sky_kwargs,
):
    """``n``-frame sequence from one ``sky_frame`` spec: sky content shifts by
    ``i * shift_step`` per frame via ``np.roll`` (registration/stacking tests); sensor
    artifacts re-paint at exact, unshifted coordinates every frame. ``streak`` (a
    single-frame trail) paints only once, at ``streak_frame_index`` (default: the middle
    frame). Each frame draws independent noise from ``seed + i``."""
    sensor = {
        k: sky_kwargs.pop(k)
        for k in ("hot_pixel_sites", "hot_pixel_value", "fixed_streak", "warp_border_px")
        if k in sky_kwargs
    }
    single_streak = sky_kwargs.pop("streak", None)
    streak_value = sky_kwargs.get("streak_value", 0.8)
    frame_index = n // 2 if streak_frame_index is None else streak_frame_index
    border = sensor.get("warp_border_px", 0)
    base, truth = sky_frame(shape, noise_sigma=0.0, **sky_kwargs)
    frames, shifts = [], []
    for i in range(n):
        dy, dx = round(shift_step[0] * i), round(shift_step[1] * i)
        moved = np.roll(base, shift=(dy, dx), axis=(0, 1))
        rng = np.random.default_rng(seed + i)
        moved = moved + noise_sigma * rng.standard_normal(moved.shape).astype(np.float32)
        if i == frame_index and single_streak:
            _paint_streak(moved, single_streak, streak_value)
        for ry, rx in sensor.get("hot_pixel_sites", ()):
            moved[ry, rx] = sensor.get("hot_pixel_value", 1.0)
        if sensor.get("fixed_streak"):
            _paint_streak(moved, sensor["fixed_streak"], streak_value)
        if border:
            _zero_border(moved, border)
        frames.append(np.clip(moved, 0.0, None).astype(np.float32))
        shifts.append((dy, dx))
    truth.update(
        shifts=shifts,
        hot_pixel_sites=sensor.get("hot_pixel_sites", ()),
        fixed_streak=sensor.get("fixed_streak"),
        streak=single_streak,
        streak_frame_index=frame_index if single_streak else None,
        warp_border_px=border,
    )
    return frames, truth
