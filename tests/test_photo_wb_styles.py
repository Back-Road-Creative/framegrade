"""White-balance style trio — daylight / cloudy / shade (photo style variants).

Three opt-in styles that move ONLY the four WB fields over the measured base, so
the operator gets a colour-temperature bracket on the contact sheet. They are
deliberately absent from ``_FAVORED``, so they are never PREFERRED and automatic
variant selection is bit-identical to before they existed; they ride the ranked
ladder's tail, where an exhausted distinctness backfill can still reach them.
"""

import pytest

from gomoveshift.pipeline.color_correction import CorrectionParams
from gomoveshift.pipeline.photos import styles
from gomoveshift.pipeline.photos.scene import SceneProfile

WB_FIELDS = ("white_balance_r", "white_balance_b", "white_balance_r_high", "white_balance_b_high")
TRIO = ("daylight", "cloudy", "shade")

BASE = CorrectionParams(
    shadow_lift=8.0,
    highlight_pull=5.0,
    saturation_mult=1.1,
    contrast_curve=0.03,
    white_balance_r=-0.01,
    white_balance_b=0.01,
    gamma=1.05,
    black_point=2.0,
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


# Literal pre-trio rankings of the six ranked styles, captured before the trio
# existed. The opt-in five tail the full walk `plan_variants` uses; ranking
# happens over _FAVORED, so these six must not move.
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
    (
        {"exposure_class": "bright", "chroma_headroom": 0.9},
        ["natural", "vivid", "soft", "moody", "golden", "mono"],
    ),
)


@pytest.mark.parametrize(("overrides", "expected"), SELECTION_PINS)
def test_auto_selection_is_unchanged_by_the_trio(overrides, expected):
    scene = make_profile(**overrides)
    # The ranked six keep their exact positions; the trio can only sit in the
    # tail past them, where an exhausted backfill reaches it.
    assert styles.select_styles(scene, count=len(styles.STYLES))[: len(expected)] == expected
    assert styles.select_styles(scene) == expected[:3]
    assert not set(TRIO) & set(expected), "the trio is opt-in, never ranked"


def test_trio_is_registered_but_absent_from_the_ranking_tables():
    assert [n for n in styles.STYLES if n in TRIO] == list(TRIO)
    assert list(styles.STYLES)[-3:] == list(TRIO), "appended after soft — tie-break order held"
    assert not set(TRIO) & set(styles._FAVORED)
    assert not set(TRIO) & set(styles._LEANING)


def test_trio_brackets_the_base_from_cool_to_warm():
    scene = make_profile()
    natural = styles.apply_style("natural", BASE, scene)
    out = [styles.apply_style(name, BASE, scene) for name in TRIO]
    reds = [p.white_balance_r for p in out]
    blues = [p.white_balance_b for p in out]
    assert reds == sorted(reds) and len(set(reds)) == 3, reds
    assert blues == sorted(blues, reverse=True) and len(set(blues)) == 3, blues
    # a real bracket AROUND the measured base: daylight cools, shade warms
    assert reds[0] < natural.white_balance_r < reds[-1]
    assert blues[0] > natural.white_balance_b > blues[-1]
    assert all(p != natural for p in out)


def test_trio_moves_highlights_with_the_mids():
    scene = make_profile()
    for name in TRIO:
        out = styles.apply_style(name, BASE, scene)
        mid = out.white_balance_r - BASE.white_balance_r
        high = out.white_balance_r_high - BASE.white_balance_r_high
        assert high * mid > 0.0, f"{name}: highlights must follow the mid shift"
        assert abs(high) <= abs(mid)


@pytest.mark.parametrize("name", TRIO)
def test_trio_touches_white_balance_and_nothing_else(name):
    out = styles.apply_style(name, BASE, make_profile())
    for field in (
        "shadow_lift",
        "highlight_pull",
        "saturation_mult",
        "vibrance",
        "contrast_curve",
        "gamma",
        "black_point",
        "denoise_strength",
        "sharpen_amount",
    ):
        assert getattr(out, field) == getattr(BASE, field), field
    assert any(getattr(out, f) != getattr(BASE, f) for f in WB_FIELDS)


@pytest.mark.parametrize("name", TRIO)
def test_trio_is_visibly_distinct_from_natural(name):
    natural = styles.apply_style("natural", BASE, make_profile())
    out = styles.apply_style(name, BASE, make_profile())
    # the style-layer visibility floor used by the shared distinctness gate
    assert any(abs(getattr(out, f) - getattr(natural, f)) >= 0.015 for f in WB_FIELDS)


@pytest.mark.parametrize("name", TRIO)
def test_trio_deltas_sit_inside_the_style_wb_rail(name):
    out = styles.apply_style(name, CorrectionParams(), make_profile())
    for field in WB_FIELDS:
        assert abs(getattr(out, field)) <= styles.STYLE_WB_CAP
    assert abs(styles.WB_SHADE_SHIFT) < styles.STYLE_WB_CAP


@pytest.mark.parametrize("name", TRIO)
@pytest.mark.parametrize(
    "base",
    (
        BASE,
        CorrectionParams(**dict.fromkeys(WB_FIELDS, styles.STYLE_WB_CAP)),
        CorrectionParams(**dict.fromkeys(WB_FIELDS, -styles.STYLE_WB_CAP)),
    ),
)
def test_amplified_trio_stays_inside_the_wb_hard_cap(name, base):
    out = styles.apply_style(name, base, make_profile(), strength=3.0)
    for field in WB_FIELDS:
        assert abs(getattr(out, field)) <= styles.WB_HARD_CAP, field


def test_trio_is_scene_independent_and_deterministic():
    warm = make_profile(warmth="warm", sat_profile="rich")
    cool = make_profile(warmth="cool", haze="foggy")
    for name in TRIO:
        assert styles.apply_style(name, BASE, warm) == styles.apply_style(name, BASE, cool)
