"""Matte fade presets — matte_shoulder / matte_knee (photo style variants).

Two opt-in styles whose tone move is a SHAPED fade rather than ``black_point``'s
single LINEAR output-floor lift: each sets ``curve_r``/``curve_g``/``curve_b`` to
ONE identical knot list, which the existing ``_apply_channel_curves`` runs
through its monotone-cubic interpolant — a master-shaped curve with no new
engine. Like the WB trio they are absent from ``_FAVORED``, so they are never
PREFERRED and automatic variant selection is bit-identical to before they
existed; they ride the ranked ladder's tail, where a backfill can reach them.
"""

import numpy as np
import pytest

from gomoveshift.pipeline.color_correction import CorrectionParams, _apply_channel_curves
from gomoveshift.pipeline.photos import styles
from gomoveshift.pipeline.photos.scene import SceneProfile

MATTE = ("matte_shoulder", "matte_knee")
RAMP = np.repeat(np.linspace(0.0, 1.0, 256, dtype=np.float64)[:, None], 3, axis=1)

BASE = CorrectionParams(
    shadow_lift=8.0,
    highlight_pull=5.0,
    saturation_mult=1.1,
    contrast_curve=0.03,
    gamma=1.05,
)


def make_profile(**overrides) -> SceneProfile:
    fields = {
        "exposure_class": "normal",
        "warmth": "neutral",
        "sat_profile": "normal",
        "haze": "clear",
        "dynamic_range": "normal",
        "is_near_monochrome": False,
    }
    return SceneProfile(**(fields | overrides))


def _curved(name: str, rgb: np.ndarray = RAMP, **strength) -> np.ndarray:
    """Run ``rgb`` through the preset's emitted per-channel curves."""
    params = styles.apply_style(name, BASE, make_profile(), **strength)
    return _apply_channel_curves(rgb, params)


# Literal pre-preset rankings of the six ranked styles, captured before the
# presets existed. The opt-in five tail the full walk `plan_variants` uses;
# ranking happens over _FAVORED, so these six must not move.
SELECTION_PINS = (
    (
        {"warmth": "warm", "sat_profile": "rich"},
        ["natural", "golden", "vivid", "soft", "moody", "mono"],
    ),
    ({"warmth": "cool"}, ["natural", "vivid", "moody", "golden", "mono", "soft"]),
    (
        {"haze": "foggy", "sat_profile": "muted", "dynamic_range": "flat"},
        ["natural", "moody", "mono", "soft", "golden", "vivid"],
    ),
)


@pytest.mark.parametrize(("overrides", "expected"), SELECTION_PINS)
def test_auto_selection_is_unchanged_by_the_presets(overrides, expected):
    scene = make_profile(**overrides)
    # The ranked six keep their exact positions; the pair can only sit in the
    # tail past them, where an exhausted backfill reaches it.
    assert styles.select_styles(scene, count=len(styles.STYLES))[: len(expected)] == expected
    assert styles.select_styles(scene) == expected[:3]
    assert not set(MATTE) & set(expected), "the presets are opt-in, never ranked"


def test_presets_are_registered_but_absent_from_the_ranking_tables():
    assert [n for n in styles.STYLES if n in MATTE] == list(MATTE)
    assert not set(MATTE) & set(styles._FAVORED)
    assert not set(MATTE) & set(styles._LEANING)


@pytest.mark.parametrize("name", MATTE)
def test_one_shared_curve_across_channels_passes_field_validation(name):
    out = styles.apply_style(name, BASE, make_profile())
    assert out.curve_r and out.curve_r == out.curve_g == out.curve_b
    # __post_init__ is the field's own gate: even-length, >= 2 points, x ascending
    CorrectionParams(curve_r=out.curve_r, curve_g=out.curve_g, curve_b=out.curve_b)
    xs = out.curve_r[0::2]
    ys = out.curve_r[1::2]
    assert all(0.0 <= x <= 1.0 for x in xs) and all(0.0 <= y <= 1.0 for y in ys)


@pytest.mark.parametrize("name", MATTE)
def test_curve_is_monotone_through_the_real_applier(name):
    curved = _curved(name)[:, 0]
    assert np.all(np.diff(curved) >= -1e-12), "a non-monotone knot set destroys tone"


def test_shoulder_lifts_the_toe_and_holds_the_highlights():
    curved = _curved("matte_shoulder")[:, 0]
    assert curved[0] >= 0.03, "the black end must lift onto a soft ramp"
    assert curved[-1] >= 0.99, "highlights held"
    # the lift must still bite at x=0.25 — inside the [0, 0.6] band a real frame
    # occupies where this curve runs, not only at the very bottom
    assert curved[64] - RAMP[64, 0] >= 0.03
    # a ROLLED toe, not black_point's linear lift: the lift decays up the ramp
    assert (curved[0] - RAMP[0, 0]) > (curved[128] - RAMP[128, 0]) > 0.0


def test_knee_rolls_the_highlights_off():
    curved = _curved("matte_knee")[:, 0]
    assert curved[-1] <= 0.95, "the top end must roll off early"
    assert curved[0] == pytest.approx(0.0, abs=1e-6)
    # the rolloff must bite by x=0.6, not in the top decile a real frame never reaches
    assert RAMP[153, 0] - curved[153] >= 0.03
    assert (RAMP[-1, 0] - curved[-1]) > (RAMP[128, 0] - curved[128]) > 0.0


def test_the_pair_renders_as_two_looks_not_one():
    """The shared distinctness gate never sees these presets: they are absent from
    ``_FAVORED``, so an explicit ``--styles`` request is the only way in, and
    explicit styles render verbatim (``test_explicit_styles_render_verbatim_even_when_identical`` in
    ``test_photo_variant_distinctness.py``).
    Pin the RENDERED separation here or nothing does — a first cut whose knee
    rolled off past x=0.6, above the tone a real frame reaches at this point in
    ``_grade``, rendered ~1.3 levels from the shoulder with the suite green."""
    from gomoveshift.pipeline.photos.develop import MIN_RENDERED_DISTINCTNESS, _rendered_distance

    scene = make_profile(sat_profile="muted", dynamic_range="flat", haze="hazy")
    ramp = np.linspace(20.0, 200.0, 48 * 64).reshape(48, 64)
    view = np.repeat(ramp[:, :, None], 3, axis=2).astype(np.uint8)
    graded = {n: styles.apply_style(n, BASE, scene) for n in ("natural", *MATTE)}
    from_natural = [_rendered_distance(view, graded[n], graded["natural"]) for n in MATTE]
    apart = _rendered_distance(view, graded["matte_shoulder"], graded["matte_knee"])
    assert min(from_natural) >= MIN_RENDERED_DISTINCTNESS, from_natural
    assert apart >= max(from_natural), (apart, from_natural)


@pytest.mark.parametrize("name", MATTE)
def test_presets_visibly_depart_from_natural(name):
    natural = _curved("natural")
    assert not np.allclose(_curved(name), natural, atol=1e-3)
    assert styles.apply_style(name, BASE, make_profile()) != styles.apply_style(
        "natural", BASE, make_profile()
    )


@pytest.mark.parametrize("name", MATTE)
def test_neutral_gray_stays_neutral(name):
    """One curve on all three channels ⇒ no colour cast, by construction."""
    gray = np.full((16, 3), 0.42, dtype=np.float64)
    out = _curved(name, gray)
    assert np.allclose(out[:, 0], out[:, 1]) and np.allclose(out[:, 1], out[:, 2])


@pytest.mark.parametrize("name", MATTE)
def test_strength_lever_carries_the_curve_shape_through_unscaled(name):
    """_LOOK_FIELDS holds only scalars, so the lever scales the scalar moves and
    leaves the curve shape as baked — it must never be dropped outright."""
    shipped = styles.apply_style(name, BASE, make_profile())
    strong = styles.apply_style(name, BASE, make_profile(), strength=2.0)
    assert strong.curve_r == shipped.curve_r
    assert strong.curve_g == shipped.curve_g and strong.curve_b == shipped.curve_b
