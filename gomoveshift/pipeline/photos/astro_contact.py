"""Astro develop-variation contact sheet (``photos astro --contact-sheet``): the
SAME stack through develop knobs -- background WB on/off, darker/default/
lighter stretch target -- as one labeled grid. ``stack_panels`` renders every
variation from the ONE ``AstroReport`` a run's ``stack_run`` already produced
(~1 min/frame); ``master_panels`` (``--from-master``) skips stacking, recovering
the pre-stretch array from an existing ``.tif``+``.stack.json`` pair via
``recover_balanced``'s MTF ``m -> 1 - m`` self-inverse (<0.1% relative error vs a
real ``develop_stack`` call, tests).

Not shown, each for a stated reason: gradient removal (omitted for
brevity; both paths apply the module default, and ``remove_gradient`` is exported
for a manual A/B); star trails/meteor reinsertion (need a second decode pass
this ONE stack can't answer for); a timelapse (a video no still represents).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .astro_develop import (
    LUMA_WEIGHTS,
    STRETCH_STEPS,
    STRETCH_TARGET_BACKGROUND,
    background_wb,
    black_point,
    mtf,
    remove_gradient,
    solve_m,
)
from .astro_sidecar import STACK_SIDECAR_SUFFIX
from .astro_stack import horizon_mask
from .develop import _analysis_rgb

# Wider than astro_develop's tuned 0.10-0.15 window -- a comparison needs a spread.
STRETCH_TARGET_DARKER = 0.08
STRETCH_TARGET_LIGHTER = 0.18


class ContactSheetError(ValueError):
    """``--from-master`` could not load or reconstruct an existing stack."""


def _sky_median(rgb, sky_mask):
    luma = np.asarray(rgb, dtype=np.float64) @ LUMA_WEIGHTS
    return float(np.median(luma[sky_mask] if sky_mask is not None else luma))


def _uint8(rgb01):
    return (np.clip(rgb01, 0.0, 1.0) * 255.0).round().astype(np.uint8)


def stretch_to_target(rgb, target, *, steps=STRETCH_STEPS, sky_mask=None):
    rgb = np.asarray(rgb, dtype=np.float64)
    bp = black_point(rgb, sky_mask)
    shifted = np.clip(rgb - bp, 0.0, None)
    peak = float(shifted.max())
    working = shifted / peak if peak > 0 else shifted
    start = _sky_median(working, sky_mask)
    for step in range(1, steps + 1):
        t = step / steps
        step_target = max(start, 1e-6) ** (1 - t) * max(target, 1e-6) ** t
        working = mtf(working, solve_m(_sky_median(working, sky_mask), step_target))
    return np.clip(working, 0.0, 1.0)


def recover_balanced(rgb_uint16, black_point_value, m_values, peak) -> np.ndarray:
    recovered = np.asarray(rgb_uint16, dtype=np.float64) / 65535.0
    for m in reversed(m_values):
        recovered = mtf(recovered, 1.0 - m)
    return recovered * peak + black_point_value


def _wb_target_panels(balanced, pre_wb, gains_note, sky_mask):
    targets = ((STRETCH_TARGET_DARKER, "darker"), (STRETCH_TARGET_LIGHTER, "lighter"))
    return [
        (
            f"no background WB ({gains_note})",
            _uint8(stretch_to_target(pre_wb, STRETCH_TARGET_BACKGROUND, sky_mask=sky_mask)),
        ),
    ] + [
        (
            f"stretch target {t:g} ({name})",
            _uint8(stretch_to_target(balanced, t, sky_mask=sky_mask)),
        )
        for t, name in targets
    ]


def stack_panels(report) -> list[tuple[str, np.ndarray]]:
    stack_rgb, develop = report.stack.rgb, report.develop
    sky_mask = develop.sky_mask
    gains = "/".join(f"{g:.3f}" for g in develop.background_gains)
    degraded = remove_gradient(stack_rgb, sky_mask=sky_mask)
    balanced = background_wb(degraded)[0]
    t, m = STRETCH_TARGET_BACKGROUND, sky_mask is not None
    default = f"default (gradient removal, WB gains {gains}, target {t:g}, masked={m})"
    return [
        (default, _analysis_rgb(develop.rgb_uint16)),
        *_wb_target_panels(balanced, degraded, f"default gains {gains}", sky_mask),
    ]


# ``photos astro-inspect``: 1:1 pixel-peep crops at the four corners + centre of
# an already-stacked master, read-only (no develop/re-stretch). Side of each
# square crop, in real pixels -- NOT resized by compose_contact_sheet, which
# only shrinks a panel wider than CONTACT_PANEL_WIDTH.
INSPECT_CROP_SIZE_PX = 400


def _inspect_crop(rgb, cx: int, cy: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    half = INSPECT_CROP_SIZE_PX // 2
    x0, x1 = max(0, cx - half), min(w, cx + half)
    y0, y1 = max(0, cy - half), min(h, cy + half)
    return rgb[y0:y1, x0:x1]


def _inspect_positions(height: int, width: int):
    """(label, cx, cy) for the four corners + centre, inset so each crop stays
    fully in-frame on anything at least ``INSPECT_CROP_SIZE_PX`` on a side."""
    half = INSPECT_CROP_SIZE_PX // 2
    return (
        ("top-left", half, half),
        ("top-right", width - half, half),
        ("bottom-left", half, height - half),
        ("bottom-right", width - half, height - half),
        ("centre", width // 2, height // 2),
    )


def _region_shape_note(shapes, cx: int, cy: int) -> tuple[str, object]:
    from .astro_triage import _median_shape

    half = INSPECT_CROP_SIZE_PX // 2
    in_region = [shape for x, y, shape in shapes if abs(x - cx) <= half and abs(y - cy) <= half]
    region_shape = _median_shape(in_region)
    if region_shape is None:
        return "no stars detected in region", region_shape
    note = (
        f"FWHM {region_shape.fwhm_major_px:.1f}/{region_shape.fwhm_minor_px:.1f}px "
        f"ellip {region_shape.ellipticity:.2f}"
    )
    return note, region_shape


def inspect_master(path) -> tuple[list[tuple[str, np.ndarray]], str]:
    """READ-ONLY pixel-peep: 1:1 crops at the four corners + centre of a
    stacked master, each labeled with its own star FWHM/ellipticity summary --
    never modifies or re-develops the master (a bare ``tifffile.imread``, no
    ``astro_develop``/``recover_balanced`` call). Returns
    ``([(label, uint8_rgb_crop), ...], summary_text)`` for the caller to hand
    ``develop.compose_contact_sheet`` + ``develop.export_jpeg``.
    """
    import tifffile

    from .astro_triage import MAD_TO_SIGMA, _measure_star_shapes, detect_stars

    path = Path(path)
    try:
        rgb_uint16 = tifffile.imread(path)
    except (OSError, ValueError) as exc:
        raise ContactSheetError(f"astro-inspect could not read {path}: {exc}") from exc
    display = _analysis_rgb(rgb_uint16)
    luma = np.asarray(display, dtype=np.float32).mean(axis=2)
    height, width = luma.shape
    background = float(np.median(luma))
    noise_sigma = float(np.median(np.abs(luma - background))) * MAD_TO_SIGMA
    stars = detect_stars(luma)
    shapes = _measure_star_shapes(luma, stars, background, noise_sigma)

    panels, lines = [], []
    for label, cx, cy in _inspect_positions(height, width):
        note, _ = _region_shape_note(shapes, cx, cy)
        panels.append((f"{label} ({note})", _inspect_crop(display, cx, cy)))
        lines.append(f"{label}: {note}")
    return panels, "\n".join(lines)


def master_panels(path) -> list[tuple[str, np.ndarray]]:
    import tifffile

    path = Path(path)
    try:
        rgb_uint16 = tifffile.imread(path)
        sidecar = json.loads(path.with_suffix(STACK_SIDECAR_SUFFIX).read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContactSheetError(f"--from-master could not read {path} + sidecar: {exc}") from exc
    d = sidecar["develop"]
    balanced = recover_balanced(rgb_uint16, d["black_point"], d["m_values"], d["peak"])
    frac = (sidecar.get("mask") or {}).get("horizon_frac")
    mask = horizon_mask(balanced.shape[:2], horizon_frac=frac) >= 0.5 if frac is not None else None
    gains = np.asarray(d["background_gains"], dtype=np.float64)
    gains_note = "shipped gains " + "/".join(f"{g:.3f}" for g in gains)
    return [
        ("as shipped (from master)", _analysis_rgb(rgb_uint16)),
        *_wb_target_panels(balanced, balanced / gains, gains_note, mask),
    ]
