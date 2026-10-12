"""The named soft-mask registry: every mask a pure function of a graded frame.

Hermetic — every frame is synthesized from numpy at a REAL-RANGE tone (peak luma ≈ 0.66,
the only tone this grade chain produces), because a mask calibrated on a full-range
synthetic ships green and inert on a real develop.
"""

import ast
from pathlib import Path

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline.photos import develop as develop_mod
from gomoveshift.pipeline.photos import masks as masks_mod

NAMES = (
    "sky",
    "ground",
    "shadows",
    "midtones",
    "highlights",
    "radial",
    "linear",
    "glints",
    "texture",
    "hue",
)
TONAL = ("shadows", "midtones", "highlights")


#: Radians of ridgeline wiggle across the frame's WIDTH — the shipped 480-px phase,
#: re-expressed so it is a fraction of the frame rather than a pixel count. This scene
#: is rendered at two sizes to pin resolution anchoring, and a pixel-phased sine is a
#: DIFFERENT ridgeline at each of them: same envelope, more wiggles in the wider one.
RIDGE_WIGGLE = 480.0 / 23.0


def _ridge_scene(h=360, w=480):
    """Bright sky over a jagged ridgeline, darker land below; peak luma ≈ 0.66."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    horizon = 0.22 * h + 0.03 * h * np.sin(RIDGE_WIGGLE * xx / w)
    lum = np.where(yy < horizon, 0.70 - 0.20 * yy / h, 0.17 + 0.17 * yy / h).astype(np.float32)
    rgb = np.dstack([lum * 0.88, lum * 0.95, lum]) * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


SIGNS = ((0.55, 0.20), (0.62, 0.55), (0.78, 0.80), (0.50, 0.72), (0.86, 0.35))


def _sign_pixels(h, w):
    """Five SMALL lit-sign boxes, positioned and sized as FRACTIONS of the frame, so the
    scene below is one scene — and each sign one share of it — at every resolution."""
    yy, xx = np.meshgrid(np.arange(h) / h, np.arange(w) / w, indexing="ij")
    boxes = np.zeros((h, w), bool)
    for cy, cx in SIGNS:
        boxes |= (np.abs(yy - cy) < 0.035) & (np.abs(xx - cx) < 0.045)
    return boxes


def _signage_scene(h=360, w=480):
    """Bright sky over dark ground with five small lit signs standing in it: the frame
    that tells a SMALL-bright-region mask from an every-bright-pixel one. The sky is a
    third of the picture, each sign well under a hundredth of it; peak luma ≈ 0.66."""
    rows = np.arange(h, dtype=np.float32)[:, None] / h * np.ones((1, w), np.float32)
    lum = np.where(rows < 0.30, 0.66, 0.14 + 0.10 * rows).astype(np.float32)
    lum[_sign_pixels(h, w)] = 0.64
    rgb = np.dstack([lum, lum * 0.97, lum * 0.92]) * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _ramp(h=200, w=300):
    """A top-to-bottom tonal ramp spanning this chain's range (0.05 .. 0.66)."""
    lum = np.linspace(0.05, 0.66, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    return np.clip(np.rint(np.dstack([lum] * 3) * 255.0), 0, 255).astype(np.uint8)


def _hue_sweep(h=120, w=360, sat=200):
    """The whole colour wheel left to right at ONE saturation and ONE tone: the frame a
    hue mask is feathered across, as a luma mask is feathered across a ramp. The wheel is
    laid out as a fraction of the WIDTH, so this is one scene at every resolution."""
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = np.rint(179.0 * np.arange(w) / w).astype(np.uint8)  # OpenCV hue: 0..179
    hsv[..., 1] = sat
    hsv[..., 2] = 168  # peak luma ≈ 0.66, this chain's real range
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _col(w, degrees):
    """The column of ``_hue_sweep`` carrying ``degrees`` of hue."""
    return int(round(degrees / 360.0 * w)) % w


def _two_tone_colours(h=120, w=160):
    """One green at two very different tones beside one red at the same two tones, split
    left/right: only a mask that reads COLOUR can take both greens and leave both reds."""
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[: h // 2, : w // 2], rgb[h // 2 :, : w // 2] = (30, 150, 40), (10, 50, 14)
    rgb[: h // 2, w // 2 :], rgb[h // 2 :, w // 2 :] = (150, 40, 30), (50, 14, 10)
    return rgb


def _scene_for(name, h=360, w=480):
    """The scene a mask's SHAPE is measured on: ``glints`` needs a frame that holds both
    a large bright region and small ones and ``hue`` the colour wheel; every other mask
    reads the ridgeline."""
    if name == "glints":
        return _signage_scene(h, w)
    return _hue_sweep(h, w) if name == "hue" else _ridge_scene(h, w)


def test_every_registered_mask_is_shape_dtype_and_range_closed():
    """Black, white, flat and noise frames: (H, W) float32 in [0, 1], never NaN."""
    rng = np.random.default_rng(11)
    frames = {
        "black": np.zeros((40, 60, 3), np.uint8),
        "white": np.full((40, 60, 3), 255, np.uint8),
        "flat": np.full((40, 60, 3), 128, np.uint8),
        "noise": rng.integers(0, 256, (40, 60, 3), dtype=np.uint8),
        "ridge": _ridge_scene(40, 60),
    }
    for name in NAMES:
        for label, frame in frames.items():
            mask = masks_mod.mask_for(name, frame)
            assert mask.shape == (40, 60) and mask.dtype == np.float32, (name, label)
            assert np.isfinite(mask).all(), (name, label)
            assert mask.min() >= 0.0 and mask.max() <= 1.0, (name, label, mask.min(), mask.max())


def _split_scene(h=200, w=300):
    """One frame, one variable: both halves carry the SAME left-to-right tonal ramp,
    and only the right half carries detail on top of it. A mask that keyed on tone
    or on position could not tell the two apart."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    lum = 0.20 + 0.42 * xx / w
    detail = 0.11 * np.sign(np.sin(xx / 2.6) * np.sin(yy / 2.6))
    lum = lum + np.where(xx >= w // 2, detail, 0.0).astype(np.float32)
    return np.clip(np.rint(np.dstack([lum] * 3) * 255.0), 0, 255).astype(np.uint8)


def test_texture_selects_the_detailed_half_and_rejects_the_smooth_one():
    """The one question the registry could not ask: where does the picture hold
    detail? Both halves share a tonal ramp, so tone and position are ruled out."""
    w = 300
    mask = masks_mod.mask_for("texture", _split_scene(w=w))
    smooth, detailed = mask[:, : w // 2 - 20], mask[:, w // 2 + 20 :]
    assert detailed.mean() > 0.8, detailed.mean()
    assert smooth.mean() < 0.1, smooth.mean()


def test_hue_takes_one_colour_at_every_tone_and_leaves_the_others():
    """The question tone, position and detail cannot ask: WHICH COLOUR is this? Both
    greens sit at wildly different luma from each other, and each red sits at the luma
    of a green, so neither tone nor position can produce this split."""
    frame = _two_tone_colours()
    green = masks_mod.mask_for("hue", frame)  # the default band is foliage green
    half = frame.shape[1] // 2
    assert green[:, :half].min() > 0.9, green[:, :half].min()
    assert green[:, half:].max() < 0.01, green[:, half:].max()


def test_hue_wraps_across_the_seam_so_a_red_band_is_one_band():
    """Hue is ANGULAR: a band centred on red spans 350° through 10°, so the distance
    has to be measured the short way round the wheel. Subtracting the raw angles would
    keep 5° and drop 355° — the same red, cut in half at the seam — so BOTH ends are
    asserted, and the opposite side of the wheel pins that the band is still a band."""
    frame = _hue_sweep()
    w = frame.shape[1]
    red = masks_mod.mask_for("hue", frame, center=0.0, width=20.0)
    assert red[:, _col(w, 5)].mean() > 0.5, red[:, _col(w, 5)].mean()
    assert red[:, _col(w, 355)].mean() > 0.5, red[:, _col(w, 355)].mean()
    assert red[:, _col(w, 180)].max() < 0.01, red[:, _col(w, 180)].max()


def test_hue_selects_nothing_where_there_is_no_colour_to_name():
    """A grey pixel HAS no hue — the angle derived at zero chroma is arbitrary — so the
    mask selects exactly nothing rather than magnifying its own floor (``texture``'s
    precedent). The saturated twin stops that from passing vacuously, and dropping
    ``min_chroma`` reaches the near-grey the default refuses, so the floor IS the knob."""
    assert masks_mod.mask_for("hue", np.full((40, 60, 3), 128, np.uint8)).max() == 0.0
    faint = _hue_sweep(sat=8)
    assert masks_mod.mask_for("hue", faint, center=0.0).max() == 0.0
    assert masks_mod.mask_for("hue", faint, center=0.0, min_chroma=0.0).max() > 0.15
    assert masks_mod.mask_for("hue", _hue_sweep(), center=0.0).max() > 0.9


def test_the_registry_enumerates_its_legal_names_and_refuses_the_rest():
    assert set(masks_mod.MASKS) == set(NAMES)
    with pytest.raises(ValueError) as err:
        masks_mod.mask_for("depth", _ramp())
    assert "depth" in str(err.value) and "midtones" in str(err.value)


def test_each_mask_enumerates_the_shape_params_it_accepts_and_refuses_the_rest():
    """Derived from the signatures, not hand-kept: a mask grows a knob by declaring it.
    ``angle`` is real — for ``linear``; on any other mask it is a silently inert typo."""
    assert masks_mod.MASK_PARAMS["linear"] == frozenset({"angle", "span", "offset"})
    assert masks_mod.MASK_PARAMS["radial"] == frozenset({"cx", "cy", "size", "aspect"})
    assert masks_mod.MASK_PARAMS["glints"] == frozenset({"max_area"})
    assert masks_mod.MASK_PARAMS["hue"] == frozenset({"center", "width", "min_chroma"})
    assert masks_mod.MASK_PARAMS["sky"] == frozenset()
    with pytest.raises(ValueError) as err:
        masks_mod.mask_for("sky", _ramp(), angle=30.0)
    assert "angle" in str(err.value) and "sky" in str(err.value)


def test_masks_are_deterministic_and_hold_no_entropy():
    """A mask is a pure function of the frame: no RNG, no clock, no global state."""
    frame = _ridge_scene()
    for name in NAMES:
        np.random.seed(0)
        first = masks_mod.mask_for(name, frame)
        np.random.seed(1)
        assert np.array_equal(first, masks_mod.mask_for(name, frame)), name
    source = Path(masks_mod.__file__).read_text()
    assert "random" not in source and "time" not in source


def test_every_mask_is_feathered_rather_than_a_hard_step():
    """No 0/1 cliff: each mask's histogram carries intermediate values. A luma mask is
    feathered across TONE so it is measured on a ramp; a spatial one across POSITION."""
    for name in NAMES:
        frame = _ramp() if name in TONAL else _scene_for(name)
        mask = masks_mod.mask_for(name, frame)
        soft = int(((mask > 0.05) & (mask < 0.95)).sum())
        assert soft > mask.size // 100, (name, soft)


def test_the_luma_windows_partition_the_tonal_range():
    """The three sum to 1 everywhere, so a layer set covers the frame without seams,
    and each owns its own end of the ramp."""
    ramp = _ramp()
    windows = [masks_mod.mask_for(n, ramp) for n in TONAL]
    total = windows[0] + windows[1] + windows[2]
    assert np.allclose(total, 1.0, atol=1e-5), (total.min(), total.max())
    tops = [int(np.argmax(w[:, 0])) for w in windows]
    assert tops[0] < tops[1] < tops[2], tops  # dark at the ramp's top, bright at its bottom


def test_the_windows_are_frame_relative_so_they_never_ship_inert():
    """This chain peaks near 0.66 luma, so an ABSOLUTE highlight window selects nothing
    on a real develop. The window rides the frame's own span instead."""
    assert develop_mod._luma(_ramp()).max() / 255.0 < 0.67
    assert masks_mod.mask_for("highlights", _ramp()).max() > 0.9


def test_sky_and_ground_reuse_the_shipped_sky_mask_and_complement_it():
    frame = _ridge_scene()
    sky = masks_mod.mask_for("sky", frame)
    assert np.array_equal(sky, develop_mod._sky_mask(frame))
    assert np.allclose(sky + masks_mod.mask_for("ground", frame), 1.0, atol=1e-6)


def test_radial_peaks_at_the_centre_and_linear_ramps_along_its_angle():
    frame = _ramp(200, 200)
    radial = masks_mod.mask_for("radial", frame)
    assert radial[100, 100] > 0.99 and radial[0, 0] < 0.01
    bottom = masks_mod.mask_for("linear", frame)  # default angle points down
    assert bottom[-1, 100] > 0.9 and bottom[0, 100] < 0.05
    side = masks_mod.mask_for("linear", frame, angle=0.0)  # points right
    assert side[100, -1] > 0.9 and side[100, 0] < 0.05


def test_radial_sits_where_a_layer_puts_it_and_its_defaults_are_the_shipped_falloff():
    """The move a centre-anchored falloff cannot make: the bright spot ON the subject.
    Geometry is normalized — a centre in frame fractions, a ``size`` in multiples of the
    shipped reach — so the same values name the same oval at every export size, asserted
    here across a 4x pixel count. Naming none of it DELEGATES to develop's vignette gain,
    so every shipped recipe renders what it did structurally, not by matching arithmetic."""
    frame = _ramp(200, 200)
    assert np.array_equal(
        masks_mod.mask_for("radial", frame), develop_mod._radial_vignette_mask(200, 200, 1.0)
    )
    off = masks_mod.mask_for("radial", frame, cx=0.25, cy=0.75, size=0.4)
    assert off[150, 50] > 0.99, off[150, 50]  # peaks on the named spot
    assert off[100, 100] < 0.05, off[100, 100]  # and has left the middle behind
    shrunk = cv2.resize(off, (100, 100), interpolation=cv2.INTER_AREA)
    small = masks_mod.mask_for("radial", _ramp(100, 100), cx=0.25, cy=0.75, size=0.4)
    assert float(np.abs(shrunk - small).mean()) < 0.02
    wide = masks_mod.mask_for("radial", frame, aspect=3.0)
    assert wide[100, 10] > 0.9 and wide[10, 100] < 0.1, "aspect never stretched the oval"
    for bad in ({"size": 0.0}, {"aspect": 0.0}):
        with pytest.raises(ValueError):
            masks_mod.mask_for("radial", frame, **bad)


def test_linear_offset_pulls_the_ramp_in_off_the_frame_edge():
    """The ramp used to top out at the frame edge and nowhere else, so a gradient could
    run in from an edge and could not sit anywhere. ``offset`` says how far in it tops
    out — a plateau inside the picture, where today there is one row of it."""
    frame = _ramp(200, 200)
    plain = masks_mod.mask_for("linear", frame)
    pulled = masks_mod.mask_for("linear", frame, offset=0.25)
    assert float((pulled[:, 100] >= 1.0).mean()) > 0.2, "the ramp never topped out in-frame"
    assert float((plain[:, 100] >= 1.0).mean()) < 0.05, "the default already had a plateau"
    assert np.array_equal(masks_mod.mask_for("linear", frame, offset=0.0), plain)


def test_glints_takes_the_small_bright_regions_where_highlights_takes_the_whole_sky():
    """The case this mask exists for: on a landscape whose bright region IS
    the sky, ``highlights`` covers a third of the picture and a signage tint reads as a
    magenta sky. ``glints`` keeps the signs and drops the sky — the contrast IS the mask."""
    frame = _signage_scene()
    signs = _sign_pixels(*frame.shape[:2])
    sky = np.zeros(frame.shape[:2], bool)
    sky[: int(0.30 * frame.shape[0])] = True

    highlights = masks_mod.mask_for("highlights", frame)
    assert highlights[signs].mean() > 0.9 and highlights[sky].mean() > 0.9, "fixture is not lit"

    glints = masks_mod.mask_for("glints", frame)
    assert glints[signs].mean() > 0.7, glints[signs].mean()
    assert glints[sky].mean() < 0.05, glints[sky].mean()


def test_glints_max_area_sets_how_big_a_bright_region_may_be():
    """The shape param is the whole knob: raise the ceiling past the sky's own share of
    the frame and the mask takes the sky too, which is what ``highlights`` already does."""
    frame = _signage_scene()
    sky = np.zeros(frame.shape[:2], bool)
    sky[: int(0.30 * frame.shape[0])] = True
    assert masks_mod.mask_for("glints", frame, max_area=0.5)[sky].mean() > 0.7


def test_invert_complements_and_feather_softens_without_moving_the_extremes():
    frame = _ridge_scene()
    sky = masks_mod.mask_for("sky", frame)
    assert np.allclose(masks_mod.mask_for("sky", frame, invert=True), 1.0 - sky, atol=1e-6)
    soft = masks_mod.mask_for("radial", frame, feather=3.0)
    plain = masks_mod.mask_for("radial", frame)
    assert int(((soft > 0.05) & (soft < 0.95)).sum()) > int(((plain > 0.05) & (plain < 0.95)).sum())
    assert soft.min() >= 0.0 and soft.max() <= 1.0
    with pytest.raises(ValueError):
        masks_mod.mask_for("radial", frame, feather=0.0)


@pytest.mark.parametrize("name", NAMES)
def test_masks_are_resolution_anchored(name):
    """GRAIN_REFERENCE_DIM's lesson pre-applied: the SAME scene at 4x the pixels yields
    the same mask after resize, so a name means one look at every export size."""
    shrunk = cv2.resize(
        masks_mod.mask_for(name, _scene_for(name, 600, 800)),
        (200, 150),
        interpolation=cv2.INTER_AREA,
    )
    small = masks_mod.mask_for(name, _scene_for(name, 150, 200))
    if name == "glints":  # the one mask that can select NOTHING, and so match vacuously
        assert small.max() > 0.5, small.max()
    assert float(np.abs(shrunk - small).mean()) < 0.03


def test_masks_does_not_import_develop_at_module_level():
    """``develop`` imports ``masks``; a module-level import back into
    develop would make that a cycle. Function-level by design — do not 'tidy' it."""
    tree = ast.parse(Path(masks_mod.__file__).read_text())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any("develop" in ast.dump(node) for node in top)
