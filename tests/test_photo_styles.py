"""Style library + variant selector (photo style variants, phase 2)."""

from dataclasses import fields as dc_fields
from dataclasses import replace

import pytest

from gomoveshift.pipeline.color_correction import (
    BLACK_POINT_MAX,
    BLACK_POINT_MIN,
    GAMMA_MAX,
    CorrectionParams,
)
from gomoveshift.pipeline.photos import styles
from gomoveshift.pipeline.photos.scene import SceneProfile

WB_FIELDS = ("white_balance_r", "white_balance_b", "white_balance_r_high", "white_balance_b_high")


_LIVE_FIELDS = {f.name for f in dc_fields(SceneProfile)}


def make_profile(**overrides) -> SceneProfile:
    fields = {
        "exposure_class": "normal",
        "warmth": "neutral",
        "sat_profile": "normal",
        "haze": "clear",
        "dynamic_range": "normal",
        "noise_level": 2.0,
        "sky_fraction": 0.3,
        "is_near_monochrome": False,
    }
    # Filtered by the live dataclass: a sibling change deletes several
    # SceneProfile fields, so a default is passed only while its field exists.
    return SceneProfile(**{k: v for k, v in (fields | overrides).items() if k in _LIVE_FIELDS})


MISTY_RIDGELINE = {"haze": "foggy", "sat_profile": "muted", "dynamic_range": "flat"}
GOLDEN_SUNSET = {"warmth": "warm", "sat_profile": "rich"}
BRIGHT_MEADOW = {"exposure_class": "bright"}

BASE = CorrectionParams(
    shadow_lift=8.0,
    highlight_pull=5.0,
    saturation_mult=1.1,
    contrast_curve=0.03,
    white_balance_r=-0.01,
    white_balance_b=0.01,
    gamma=1.05,
)
MAX_BASE = CorrectionParams(
    shadow_lift=styles.SHADOW_LIFT_MAX,
    highlight_pull=styles.HIGHLIGHT_PULL_MAX,
    saturation_mult=styles.STYLE_SATURATION_CAP,
    contrast_curve=styles.STYLE_CONTRAST_CAP,
    gamma=GAMMA_MAX,
    **dict.fromkeys(WB_FIELDS, styles.STYLE_WB_CAP),
)
MIN_BASE = CorrectionParams(
    saturation_mult=0.80,
    gamma=styles.GAMMA_MIN,
    **dict.fromkeys(WB_FIELDS, -styles.STYLE_WB_CAP),
)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (MISTY_RIDGELINE, ["natural", "moody", "mono"]),
        (GOLDEN_SUNSET, ["natural", "golden", "vivid"]),
        (BRIGHT_MEADOW, ["natural", "vivid", "soft"]),
    ],
)
def test_acceptance_selections(overrides, expected):
    assert styles.select_styles(make_profile(**overrides)) == expected


def test_natural_always_first_and_identity():
    for overrides in (MISTY_RIDGELINE, GOLDEN_SUNSET, BRIGHT_MEADOW, {}):
        assert styles.select_styles(make_profile(**overrides))[0] == "natural"
    assert styles.apply_style("natural", BASE, make_profile()) == BASE


def test_deterministic_same_profile_same_list():
    scene = make_profile(**MISTY_RIDGELINE)
    first = styles.select_styles(scene)
    assert first == styles.select_styles(scene)
    assert first == styles.select_styles(make_profile(**MISTY_RIDGELINE))


@pytest.mark.parametrize("overrides", [{"haze": "foggy"}, {"is_near_monochrome": True}])
def test_vivid_suppressed(overrides):
    scene = make_profile(**overrides)
    assert "vivid" not in styles.select_styles(scene)
    # last of the RANKING (the opt-in tail is appended past it, never scored)
    assert styles.select_styles(scene, count=len(styles._FAVORED) + 1)[-1] == "vivid"


@pytest.mark.parametrize("base", [BASE, MAX_BASE, MIN_BASE])
@pytest.mark.parametrize("name", list(styles.STYLES))
def test_outputs_stay_in_legal_ranges(name, base):
    out = styles.apply_style(name, base, make_profile())
    assert 0.0 <= out.shadow_lift <= styles.SHADOW_LIFT_MAX
    assert 0.0 <= out.highlight_pull <= styles.HIGHLIGHT_PULL_MAX
    assert 0.0 <= out.saturation_mult <= styles.STYLE_SATURATION_CAP
    assert 0.0 <= out.contrast_curve <= styles.STYLE_CONTRAST_CAP
    assert (
        styles.STYLE_BLACK_POINT_CRUSH_MAX <= out.black_point <= styles.STYLE_BLACK_POINT_FADE_MAX
    )
    for field in WB_FIELDS:
        assert -styles.STYLE_WB_CAP <= getattr(out, field) <= styles.STYLE_WB_CAP
    assert styles.GAMMA_MIN <= out.gamma <= GAMMA_MAX


def test_mono_zeroes_saturation_with_dr_tuned_contrast():
    flat = styles.apply_style("mono", BASE, make_profile(dynamic_range="flat"))
    wide = styles.apply_style("mono", BASE, make_profile(dynamic_range="wide"))
    assert flat.saturation_mult == 0.0 == wide.saturation_mult
    assert flat.contrast_curve > wide.contrast_curve


def test_unknown_style_raises():
    with pytest.raises(ValueError, match="Unknown style"):
        styles.apply_style("sepia", BASE, make_profile())


# Regression: a muted scene's base correction already ramps saturation to ~1.22,
# so rails at the corrective 1.25 cap left vivid +0.03 effective — measured on a
# real RAW frame, variants were nearly indistinguishable from natural.
NEAR_CAP_MUTED_BASE = CorrectionParams(
    shadow_lift=8.0,
    highlight_pull=5.0,
    saturation_mult=1.22,
    contrast_curve=0.0,
    gamma=1.05,
)


def test_vivid_moves_a_near_cap_muted_base():
    out = styles.apply_style("vivid", NEAR_CAP_MUTED_BASE, make_profile())
    assert out.saturation_mult >= 1.46  # the look-scale additive floor engaged in full
    assert out.saturation_mult <= styles.STYLE_SATURATION_CAP


def _visibly_differs(out: CorrectionParams, ref: CorrectionParams) -> bool:
    return (
        abs(out.saturation_mult - ref.saturation_mult) >= 0.08
        or abs(out.contrast_curve - ref.contrast_curve) >= 0.02
        or any(abs(getattr(out, f) - getattr(ref, f)) >= 0.015 for f in WB_FIELDS)
        or abs(out.shadow_lift - ref.shadow_lift) >= 2.0
        or abs(out.gamma - ref.gamma) >= 0.03
    )


def test_every_style_visibly_departs_from_natural():
    natural = styles.apply_style("natural", NEAR_CAP_MUTED_BASE, make_profile())
    for name in styles.STYLES:
        if name == "natural":
            continue
        out = styles.apply_style(name, NEAR_CAP_MUTED_BASE, make_profile())
        assert _visibly_differs(out, natural), name


def test_style_rails_bound_extreme_bases():
    extreme = replace(BASE, saturation_mult=1.4, white_balance_r=0.06, white_balance_b=0.06)
    vivid = styles.apply_style("vivid", extreme, make_profile())
    assert vivid.saturation_mult == styles.STYLE_SATURATION_CAP
    moody = styles.apply_style("moody", extreme, make_profile())
    assert abs(moody.white_balance_b) <= styles.STYLE_WB_CAP
    golden = styles.apply_style("golden", extreme, make_profile())
    for field in WB_FIELDS:
        assert abs(getattr(golden, field)) <= styles.STYLE_WB_CAP
    assert styles.apply_style("mono", extreme, make_profile()).saturation_mult == 0.0


# ── Scene-adaptive look identities ───────────────────────────────────


def test_vivid_identity_crushes_blacks():
    out = styles.apply_style("vivid", BASE, make_profile())
    assert out.black_point < 0.0


def test_moody_identity_matte_with_cool_shadows_not_highlights():
    out = styles.apply_style("moody", BASE, make_profile())
    assert out.black_point > 0.0  # matte fade
    shadow_shift = out.white_balance_b - BASE.white_balance_b
    high_shift = out.white_balance_b_high - BASE.white_balance_b_high
    assert shadow_shift > high_shift  # cool sits in the shadows, not globally


def test_golden_identity_concentrates_warmth_in_highlights():
    out = styles.apply_style("golden", BASE, make_profile())
    high_warm = out.white_balance_r_high - BASE.white_balance_r_high
    mid_warm = out.white_balance_r - BASE.white_balance_r
    assert high_warm > mid_warm > 0.0


def test_soft_identity_fades_and_lifts_gamma():
    out = styles.apply_style("soft", BASE, make_profile())
    assert out.black_point > 0.0
    assert out.gamma > BASE.gamma


def test_mono_crush_scales_with_dynamic_range():
    flat = styles.apply_style("mono", BASE, make_profile(dynamic_range="flat"))
    wide = styles.apply_style("mono", BASE, make_profile(dynamic_range="wide"))
    assert flat.black_point < wide.black_point < 0.0


def test_vivid_saturation_adapts_to_chroma_headroom():
    muted = make_profile(sat_profile="muted", chroma_headroom=0.9)
    rich = make_profile(sat_profile="rich", chroma_headroom=0.1)
    out_muted = styles.apply_style("vivid", BASE, muted)
    out_rich = styles.apply_style("vivid", BASE, rich)
    assert out_muted.saturation_mult > out_rich.saturation_mult
    assert out_rich.contrast_curve > out_muted.contrast_curve  # low headroom leans on contrast


def test_moody_adapts_muted_to_tone_rich_to_desat():
    muted = styles.apply_style("moody", BASE, make_profile(sat_profile="muted"))
    rich = styles.apply_style("moody", BASE, make_profile(sat_profile="rich", chroma_headroom=0.2))
    assert muted.black_point > rich.black_point  # muted scene leans fade/contrast
    assert rich.saturation_mult < muted.saturation_mult  # rich scene leans desat


def test_golden_backs_off_on_already_warm_scene():
    warm = styles.apply_style("golden", BASE, make_profile(warmth="warm"))
    neutral = styles.apply_style("golden", BASE, make_profile())
    warm_push = warm.white_balance_r_high - BASE.white_balance_r_high
    neutral_push = neutral.white_balance_r_high - BASE.white_balance_r_high
    assert 0.0 < warm_push < neutral_push


def test_black_point_rails_bound_extreme_bases():
    faded = styles.apply_style("soft", replace(BASE, black_point=12.0), make_profile())
    assert faded.black_point == styles.STYLE_BLACK_POINT_FADE_MAX
    crushed = styles.apply_style("vivid", replace(BASE, black_point=-8.0), make_profile())
    assert crushed.black_point == styles.STYLE_BLACK_POINT_CRUSH_MAX


def test_separation_regression_muted_overcast_near_cap_base():
    # The complaint behind the adaptive identities: on a real muted
    # overcast RAW frame the top variants rendered nearly identical JPEGs.
    from gomoveshift.pipeline.photos.develop import VARIANT_DISTINCTNESS_EPSILON, _params_distance

    scene = make_profile(
        sat_profile="muted", dynamic_range="flat", haze="hazy", chroma_headroom=0.85
    )
    outs = {
        n: styles.apply_style(n, NEAR_CAP_MUTED_BASE, scene) for n in ("natural", "vivid", "moody")
    }
    for a, b in (("natural", "vivid"), ("natural", "moody"), ("vivid", "moody")):
        distance = _params_distance(outs[a], outs[b])
        assert distance >= 2 * VARIANT_DISTINCTNESS_EPSILON, (a, b, distance)


def test_styles_deterministic_for_same_inputs():
    scene = make_profile(sat_profile="muted", chroma_headroom=0.7)
    for name in styles.STYLES:
        assert styles.apply_style(name, BASE, scene) == styles.apply_style(name, BASE, scene)


# ── Selector re-bias toward structural difference ────────────────────

# The failing sample: muted overcast, normal exposure/DR,
# clear air, plenty of chroma headroom. The old selector ranked vivid #2 here —
# the style LEAST able to differentiate a low-chroma frame — and shipped three
# same-theme variants (natural+vivid+moody).
MUTED_OVERCAST = {"sat_profile": "muted", "chroma_headroom": 0.85}


def test_muted_overcast_sample_drops_vivid_for_structural_styles():
    trio = styles.select_styles(make_profile(**MUTED_OVERCAST))
    assert "vivid" not in trio
    assert trio == ["natural", "soft", "moody"]


def test_muted_flat_scene_no_longer_pulls_mono_into_the_trio():
    """The muted+flat leaning fired on corpus-typical light — judged wrong on
    all nine offers, never best — so mono ranks by _FAVORED alone now: no bump,
    and mono sits below the styles the leaning let it displace."""
    scene = make_profile(sat_profile="muted", dynamic_range="flat", chroma_headroom=0.85)
    assert styles._score("mono", scene) == styles.SCORE_BASE
    assert "mono" not in styles.select_styles(scene)
    ranking = styles.select_styles(scene, count=len(styles.STYLES))
    assert ranking.index("mono") > ranking.index("moody")
    assert ranking.index("mono") > ranking.index("golden")


def test_muted_scene_soft_outranks_vivid():
    ranking = styles.select_styles(make_profile(**MUTED_OVERCAST), count=len(styles.STYLES))
    assert ranking.index("soft") < ranking.index("vivid")


def test_bright_clear_high_headroom_muted_scene_keeps_vivid_favored():
    scene = make_profile(exposure_class="bright", sat_profile="muted", chroma_headroom=0.9)
    assert styles.select_styles(scene) == ["natural", "vivid", "soft"]


def test_muted_penalty_sits_strictly_above_the_suppression_floor():
    penalized = styles._score("vivid", make_profile(**MUTED_OVERCAST))
    assert penalized < styles.SCORE_BASE  # below every unpenalized style
    assert styles.SCORE_SUPPRESSED < penalized  # but never down to suppression
    assert styles._score("vivid", make_profile(haze="foggy")) == styles.SCORE_SUPPRESSED


# ── Backfill ladder: the full walk reaches every legal style ─────────

# The opt-in five: absent from _FAVORED so they are never PREFERRED, but legal
# and structurally different, so an exhausted distinctness backfill must reach
# them rather than shipping near-identical variants.
OPT_IN = tuple(n for n in styles.STYLES if n not in ("natural", *styles._FAVORED))


def test_full_ladder_reaches_every_style():
    ladder = styles.select_styles(make_profile(**MUTED_OVERCAST), count=len(styles.STYLES))
    assert len(ladder) == len(styles.STYLES)
    assert set(ladder) == set(styles.STYLES)


def test_opt_in_styles_tail_the_ranking_in_declaration_order():
    # Appended, never scored in: every ranked style keeps its exact position,
    # so a candidate reaches the tail only once the ranking is exhausted.
    ranked = len(styles._FAVORED) + 1
    for overrides in (MISTY_RIDGELINE, GOLDEN_SUNSET, BRIGHT_MEADOW, MUTED_OVERCAST, {}):
        scene = make_profile(**overrides)
        ladder = styles.select_styles(scene, count=len(styles.STYLES))
        assert ladder[:ranked] == styles.select_styles(scene, count=ranked)
        assert ladder[ranked:] == list(OPT_IN)


def test_ordinary_selection_never_reaches_the_opt_in_tail():
    for haze in ("clear", "hazy", "foggy"):
        for sat in ("muted", "normal", "rich"):
            for exposure in ("low_key", "normal", "bright"):
                scene = make_profile(haze=haze, sat_profile=sat, exposure_class=exposure)
                assert not set(OPT_IN) & set(styles.select_styles(scene))


def test_selector_rebias_is_deterministic():
    for overrides in (MUTED_OVERCAST, {"sat_profile": "muted", "dynamic_range": "flat"}):
        first = styles.select_styles(make_profile(**overrides))
        assert first == styles.select_styles(make_profile(**overrides))


# ── Subject-routed selection ───────────────────────────────────


def _reading(category, confidence, source="inference"):
    from gomoveshift.pipeline.photos.subject import SubjectReading

    return SubjectReading(category, confidence, source, ())


def test_no_subject_is_bit_identical_to_before_routing_existed():
    scene = make_profile(**MISTY_RIDGELINE)
    assert styles.select_styles(scene, subject=None) == styles.select_styles(scene)


def test_unknown_subject_never_reorders_the_ladder():
    from gomoveshift.pipeline.photos.subject import UNKNOWN_SUBJECT

    scene = make_profile(**MISTY_RIDGELINE)
    assert styles.select_styles(scene, subject=UNKNOWN_SUBJECT) == styles.select_styles(scene)


def test_below_floor_reading_never_reorders_the_ladder():
    from gomoveshift.pipeline.photos.subject import CONFIDENCE_WEAK, SubjectCategory

    scene = make_profile()
    weak_astro = _reading(SubjectCategory.ASTRO, CONFIDENCE_WEAK)
    assert styles.select_styles(scene, subject=weak_astro) == styles.select_styles(scene)


def test_astro_reading_at_or_above_floor_routes_moody_after_natural():
    from gomoveshift.pipeline.photos.subject import CONFIDENCE_TRACKED_NIGHT, SubjectCategory

    scene = make_profile()
    reading = _reading(SubjectCategory.ASTRO, CONFIDENCE_TRACKED_NIGHT)
    routed = styles.select_styles(scene, count=len(styles.STYLES), subject=reading)
    assert routed[0] == "natural"
    assert routed[1] == "moody"


def test_exposure_bracket_reading_routes_matte_shoulder_after_natural():
    from gomoveshift.pipeline.photos.subject import CONFIDENCE_GROUPED, SubjectCategory

    scene = make_profile()
    reading = _reading(SubjectCategory.EXPOSURE_BRACKET, CONFIDENCE_GROUPED)
    routed = styles.select_styles(scene, count=len(styles.STYLES), subject=reading)
    assert routed[0] == "natural"
    assert routed[1] == "matte_shoulder"


def test_routing_never_displaces_natural_from_first():
    from gomoveshift.pipeline.photos.subject import CONFIDENCE_TRACKED_NIGHT, SubjectCategory

    for haze in ("clear", "hazy", "foggy"):
        scene = make_profile(haze=haze)
        reading = _reading(SubjectCategory.ASTRO, CONFIDENCE_TRACKED_NIGHT)
        assert styles.select_styles(scene, subject=reading)[0] == "natural"


def test_every_routed_style_is_a_registered_style():
    """A typo in ``SUBJECT_STYLE_ROUTING`` would otherwise insert a name that
    is not in ``STYLES``: ``select_styles`` would hand it back happily and
    ``apply_style`` would only reject it later, at render time. Catch it here,
    where the table is declared, rather than after an encode.
    """
    unregistered = {
        style for style in styles.SUBJECT_STYLE_ROUTING.values() if style not in styles.STYLES
    }
    assert unregistered == set()


def test_natural_is_never_a_routing_target():
    """Routing moves a style to right AFTER ``natural``. Routing TO ``natural``
    would be a no-op at best and, since the move rebuilds the ladder around
    whatever sits first, a way to lose the anchor at worst. The table must
    never name it."""
    assert "natural" not in styles.SUBJECT_STYLE_ROUTING.values()


def test_unrouted_category_never_reorders_the_ladder():
    from gomoveshift.pipeline.photos.subject import CONFIDENCE_MODERATE, SubjectCategory

    scene = make_profile(**MISTY_RIDGELINE)
    reading = _reading(SubjectCategory.PORTRAIT, CONFIDENCE_MODERATE)
    assert styles.select_styles(scene, subject=reading) == styles.select_styles(scene)


# ── Look-scale delta bake (strength sweep) ──────────────────

# Every style was rendered on a real sample at ×1/×2/×3 delta
# amplification (amplified = base + k×(styled − base) per field): ×1 read
# nearly identical to natural, ×2 was chosen, ×3 too loud. The constants bake
# the ×2 column. Expected values below are LITERALS — 2× the pre-bake deltas
# at these exact reference conditions — never derived from shipped constants.
BAKE_SCENE = {"sat_profile": "muted", "dynamic_range": "flat", "haze": "hazy"}


def bake_profile() -> SceneProfile:
    return make_profile(**BAKE_SCENE, chroma_headroom=0.85)


def test_bake_vivid_key_deltas_doubled_on_reference():
    out = styles.apply_style("vivid", NEAR_CAP_MUTED_BASE, bake_profile())
    # pre-bake sat delta at strength 0.8875 was +0.1299 -> x2 = +0.2599
    assert out.saturation_mult == pytest.approx(1.4799, abs=1e-3)
    # pre-bake contrast delta 0.056 -> x2 = 0.112
    assert out.contrast_curve == pytest.approx(0.112, abs=1e-6)
    # pre-bake shadow delta -4.0 (halved lift) -> x2 = -8.0 (lift removed)
    assert out.shadow_lift == 0.0
    # x2 crush target -12 exceeds the renderer's black-point authority: pinned at -10
    assert out.black_point == -10.0


def test_bake_moody_key_deltas_doubled_on_reference():
    out = styles.apply_style("moody", NEAR_CAP_MUTED_BASE, bake_profile())
    # pre-bake muted desat delta -0.183 -> x2 = -0.366
    assert out.saturation_mult == pytest.approx(0.854, abs=1e-6)
    assert out.white_balance_b == pytest.approx(0.06, abs=1e-6)
    assert out.white_balance_r == pytest.approx(-0.03, abs=1e-6)
    assert out.white_balance_b_high == pytest.approx(-0.024, abs=1e-6)
    assert out.white_balance_r_high == pytest.approx(0.024, abs=1e-6)
    # pre-bake muted contrast delta 0.05 -> x2 = 0.10
    assert out.contrast_curve == pytest.approx(0.10, abs=1e-6)
    # x2 muted fade target 26 exceeds the +20 black-point authority: lands ON it
    assert out.black_point == 20.0


def test_bake_golden_key_deltas_doubled_on_reference():
    out = styles.apply_style("golden", NEAR_CAP_MUTED_BASE, bake_profile())
    assert out.white_balance_r_high == pytest.approx(0.09, abs=1e-6)
    assert out.white_balance_b_high == pytest.approx(-0.06, abs=1e-6)
    assert out.white_balance_r == pytest.approx(0.04, abs=1e-6)
    assert out.white_balance_b == pytest.approx(-0.024, abs=1e-6)
    assert out.highlight_pull == pytest.approx(11.0, abs=1e-6)  # pre-bake +3 -> x2 = +6


def test_bake_soft_operator_set_identity():
    # Soft is operator-set past a blind double (a pure double still read weak):
    # fade +12, gamma +0.14, desat delta x2 (scale 0.70), S-curve flattened away.
    out = styles.apply_style("soft", NEAR_CAP_MUTED_BASE, bake_profile())
    assert out.black_point == pytest.approx(12.0)
    assert out.gamma == pytest.approx(1.19, abs=1e-6)
    assert out.saturation_mult == pytest.approx(0.854, abs=1e-6)
    assert out.contrast_curve == 0.0


def test_bake_mono_untouched():
    # Operator: mono is already structurally distinct — the bake leaves it alone.
    out = styles.apply_style("mono", NEAR_CAP_MUTED_BASE, bake_profile())
    assert out.saturation_mult == 0.0
    assert out.contrast_curve == pytest.approx(0.06, abs=1e-6)
    assert out.black_point == -8.0


def test_baked_rails_stay_inside_field_authority():
    assert styles.STYLE_BLACK_POINT_FADE_MAX <= BLACK_POINT_MAX
    assert styles.STYLE_BLACK_POINT_CRUSH_MAX >= BLACK_POINT_MIN
    faded = styles.apply_style(
        "moody", replace(NEAR_CAP_MUTED_BASE, black_point=10.0), bake_profile()
    )
    assert faded.black_point == styles.STYLE_BLACK_POINT_FADE_MAX


def test_bake_deterministic_on_reference():
    scene = bake_profile()
    for name in styles.STYLES:
        first = styles.apply_style(name, NEAR_CAP_MUTED_BASE, scene)
        assert first == styles.apply_style(name, NEAR_CAP_MUTED_BASE, scene)


# ── strength lever ───────────────────────────────────────────────────
# The lever amplifies each style's delta over the measured base and scales the
# style-layer rails with it. x1 must stay the shipped bake, bit for bit.


@pytest.mark.parametrize("name", list(styles.STYLES))
@pytest.mark.parametrize("scene_kwargs", ({}, MISTY_RIDGELINE, GOLDEN_SUNSET, BRIGHT_MEADOW))
def test_strength_one_is_the_shipped_bake(name, scene_kwargs):
    scene = make_profile(**scene_kwargs)
    assert styles.apply_style(name, BASE, scene, strength=1.0) == styles.apply_style(
        name, BASE, scene
    )


def test_strength_amplifies_the_delta_over_base():
    scene = make_profile(**MISTY_RIDGELINE)
    shipped = styles.apply_style("moody", BASE, scene)
    strong = styles.apply_style("moody", BASE, scene, strength=2.0)
    assert strong.saturation_mult < shipped.saturation_mult
    assert (strong.white_balance_b - BASE.white_balance_b) > (
        shipped.white_balance_b - BASE.white_balance_b
    ) * 1.5
    assert strong.contrast_curve > shipped.contrast_curve


def test_strength_below_one_softens_the_look():
    scene = make_profile(**GOLDEN_SUNSET)
    shipped = styles.apply_style("golden", BASE, scene)
    soft = styles.apply_style("golden", BASE, scene, strength=0.5)
    assert abs(soft.white_balance_r_high - BASE.white_balance_r_high) < abs(
        shipped.white_balance_r_high - BASE.white_balance_r_high
    )


def test_amplified_moody_never_desaturates_into_mono():
    """The desat floor is what keeps moody a LOOK and not a second mono."""
    scene = make_profile(**MISTY_RIDGELINE)
    strong = styles.apply_style("moody", BASE, scene, strength=5.0)
    assert strong.saturation_mult >= styles.STYLE_DESAT_FLOOR
    assert styles.apply_style("mono", BASE, scene, strength=5.0).saturation_mult == (
        styles.MONO_SATURATION
    )


@pytest.mark.parametrize("strength", (0.5, 2.0, 4.0))
@pytest.mark.parametrize("base", (BASE, MAX_BASE, MIN_BASE))
def test_amplified_params_stay_inside_renderer_authority(strength, base):
    scene = make_profile(**MISTY_RIDGELINE)
    for name in styles.STYLES:
        p = styles.apply_style(name, base, scene, strength=strength)
        assert 0.0 <= p.saturation_mult <= styles.SATURATION_HARD_CAP
        assert all(abs(getattr(p, f)) <= styles.WB_HARD_CAP for f in WB_FIELDS)
        assert BLACK_POINT_MIN <= p.black_point <= BLACK_POINT_MAX
        assert styles.GAMMA_MIN <= p.gamma <= GAMMA_MAX
        assert 0.0 <= p.shadow_lift <= styles.SHADOW_LIFT_MAX
        assert 0.0 <= p.highlight_pull <= styles.HIGHLIGHT_PULL_MAX


@pytest.mark.parametrize("strength", (2.0, 4.0, 8.0))
@pytest.mark.parametrize("base", (BASE, MAX_BASE, MIN_BASE))
def test_amplified_contrast_keeps_the_s_curve_monotone(strength, base):
    """Past the ceiling the curve applier's control points stop increasing and
    the monotone interpolant inverts — tone destroyed, not shaped."""
    for name in styles.STYLES:
        p = styles.apply_style(name, base, make_profile(**MISTY_RIDGELINE), strength=strength)
        bl_out = p.shadow_lift / 255.0
        wh_out = 1.0 - p.highlight_pull / 255.0
        wh_top = wh_out * (0.95 + p.highlight_pull / 500.0) if p.highlight_pull >= 5.0 else wh_out
        ys = [
            bl_out,
            0.25 + bl_out * 0.5 - p.contrast_curve,
            0.75 - (1.0 - wh_out) * 0.5 + p.contrast_curve,
            wh_top,
        ]
        assert ys == sorted(ys), f"{name} at x{strength} builds a non-monotone curve: {ys}"


@pytest.mark.parametrize("strength", (0.0, -1.0))
def test_non_positive_strength_raises(strength):
    with pytest.raises(ValueError, match="strength"):
        styles.apply_style("moody", BASE, make_profile(), strength=strength)


# ── scene-appropriate tuning: warm / rich scenes (landscape-calibration gaps) ──


def test_vivid_attenuates_the_chroma_push_on_a_rich_frame():
    # Below the rich knee the frame is already near-saturated; the saturation
    # push rolls off toward zero so a red close-up doesn't cartoon.
    at_knee = make_profile(chroma_headroom=styles.VIVID_RICH_KNEE)
    rich = make_profile(chroma_headroom=styles.VIVID_RICH_KNEE / 4)
    push_at_knee = styles.apply_style("vivid", BASE, at_knee).saturation_mult
    push_rich = styles.apply_style("vivid", BASE, rich).saturation_mult
    assert BASE.saturation_mult <= push_rich < push_at_knee


def test_vivid_rich_knee_sits_below_every_calibration_reference():
    # The bake/calibration profiles all sit at headroom >= 0.85; the knee must
    # stay below them so the rolloff never perturbs the locked sweep.
    assert styles.VIVID_RICH_KNEE < 0.85


def test_moody_restrains_the_teal_cast_on_a_warm_scene():
    warm = styles.apply_style("moody", BASE, make_profile(warmth="warm"))
    neutral = styles.apply_style("moody", BASE, make_profile(warmth="neutral"))
    # the cool/teal WB moves shrink; the matte identity (fade + desat) is untouched
    assert abs(warm.white_balance_b - BASE.white_balance_b) < abs(
        neutral.white_balance_b - BASE.white_balance_b
    )
    assert warm.black_point == neutral.black_point
    assert warm.saturation_mult == neutral.saturation_mult


def test_moody_neutral_landscape_teal_is_bit_identical():
    # Moody's home is the neutral hazy landscape; the warm restraint keys on
    # warmth == 'warm', so a neutral scene is untouched.
    scene = make_profile(haze="hazy", sat_profile="muted", dynamic_range="flat")
    out = styles.apply_style("moody", NEAR_CAP_MUTED_BASE, scene)
    assert out.white_balance_b == pytest.approx(
        NEAR_CAP_MUTED_BASE.white_balance_b + styles.MOODY_COOL_SHIFT
    )


def test_golden_keeps_highlight_glow_over_mid_on_a_warm_scene():
    # The warm-highlight glow IS the golden look; on a warm scene the highlight
    # moves keep more of their strength than the mid moves so golden stays
    # distinct from natural instead of collapsing and being dropped.
    warm = styles.apply_style("golden", BASE, make_profile(warmth="warm"))
    high_frac = (
        warm.white_balance_r_high - BASE.white_balance_r_high
    ) / styles.GOLDEN_HIGH_WARM_SHIFT
    mid_frac = (warm.white_balance_r - BASE.white_balance_r) / styles.GOLDEN_MID_WARM_SHIFT
    assert high_frac > mid_frac


def test_warm_scene_prefers_soft_over_moody_as_differentiator():
    # A warm close-up is not moody's scene (fog/low-key). Soft leans in so that
    # when golden (the warm favorite) is later distinctness-dropped for sitting
    # too close to natural, an airy soft backfills instead of a muddy moody.
    scene = make_profile(warmth="warm", sat_profile="normal", chroma_headroom=0.16)
    ranking = styles.select_styles(scene, count=len(styles.STYLES))
    assert ranking.index("soft") < ranking.index("moody")
    assert styles.select_styles(scene) == ["natural", "golden", "soft"]


def test_vivid_crush_and_contrast_soften_on_a_rich_frame():
    # A saturated close-up (headroom below the knee) must not get vivid's full
    # black crush + low-headroom contrast — together they crushed the frame at
    # extreme strength. Both attenuate with the same rich rolloff as the chroma
    # push, so a rich frame gets a clean pop, not a crush.
    rich = styles.apply_style(
        "vivid", BASE, make_profile(chroma_headroom=styles.VIVID_RICH_KNEE / 4)
    )
    at_knee = styles.apply_style(
        "vivid", BASE, make_profile(chroma_headroom=styles.VIVID_RICH_KNEE)
    )
    assert rich.black_point > at_knee.black_point  # gentler crush (closer to zero)
    assert rich.contrast_curve < at_knee.contrast_curve


def test_vivid_full_crush_survives_above_the_rich_knee():
    # A frame with chroma room keeps the full punchy crush — the rolloff is a
    # rich-frame guard, not a global softening.
    out = styles.apply_style("vivid", BASE, make_profile(chroma_headroom=1.0))
    assert out.black_point == BASE.black_point + styles.VIVID_CRUSH


# ── golden by real solar elevation, wall clock as the no-track fallback ──

# Pre-change wall-clock selections on the default (neutral) profile at every
# golden-relevant boundary hour, captured from the shipped selector before the
# solar path landed. Elevation None MUST reproduce these bit for bit.
WALL_CLOCK_PINS = (
    (None, ["natural", "vivid", "moody"]),
    (4, ["natural", "vivid", "moody"]),
    (5, ["natural", "golden", "vivid"]),
    (8, ["natural", "golden", "vivid"]),
    (9, ["natural", "vivid", "moody"]),
    (12, ["natural", "vivid", "moody"]),
    (16, ["natural", "vivid", "moody"]),
    (17, ["natural", "golden", "vivid"]),
    (20, ["natural", "golden", "vivid"]),
    (21, ["natural", "vivid", "moody"]),
)


def test_elevation_none_reproduces_the_wall_clock_selection_bit_identically():
    for hour, expected in WALL_CLOCK_PINS:
        scene = make_profile(capture_hour=hour, solar_elevation_deg=None)
        assert styles.select_styles(scene) == expected, hour
        assert styles.select_styles(scene, count=len(styles.STYLES)) == styles.select_styles(
            make_profile(capture_hour=hour), count=len(styles.STYLES)
        )


def test_elevation_none_keeps_every_pinned_fixture_ranking():
    for overrides, expected in (
        (MISTY_RIDGELINE, ["natural", "moody", "mono"]),
        (GOLDEN_SUNSET, ["natural", "golden", "vivid"]),
        (BRIGHT_MEADOW, ["natural", "vivid", "soft"]),
        (MUTED_OVERCAST, ["natural", "soft", "moody"]),
    ):
        scene = make_profile(**overrides, solar_elevation_deg=None)
        assert styles.select_styles(scene) == expected
        assert styles.select_styles(scene, count=len(styles.STYLES)) == styles.select_styles(
            make_profile(**overrides), count=len(styles.STYLES)
        )


def test_in_band_elevation_favors_golden_even_at_a_midday_clock_hour():
    # Solar truth beats the clock: 5.0 deg sits inside the golden band.
    scene = make_profile(capture_hour=12, solar_elevation_deg=5.0)
    assert styles._score("golden", scene) == styles.SCORE_BASE + styles.SCORE_FAVORED
    assert "golden" in styles.select_styles(scene)


def test_high_sun_at_a_golden_clock_hour_is_not_golden():
    # A frame shot at 08:xx wall clock (inside the fixed window) but the
    # sun already at 17.62 deg — the solar path answers, and says no.
    scene = make_profile(capture_hour=8, solar_elevation_deg=17.62)
    assert styles._score("golden", scene) == styles.SCORE_BASE
    assert "golden" not in styles.select_styles(scene)
    # ...while the same hour with no track (None) keeps today's clock behaviour.
    fallback = make_profile(capture_hour=8, solar_elevation_deg=None)
    assert styles._score("golden", fallback) == styles.SCORE_BASE + styles.SCORE_FAVORED


def test_solar_band_edges_are_inclusive_and_warmth_or_branch_survives():
    from sun_track import GOLDEN_ELEVATION_MAX, GOLDEN_ELEVATION_MIN

    for edge in (GOLDEN_ELEVATION_MIN, GOLDEN_ELEVATION_MAX):
        scene = make_profile(solar_elevation_deg=edge)
        assert styles._score("golden", scene) == styles.SCORE_BASE + styles.SCORE_FAVORED
    # a warm scene stays favored regardless of a high sun — the OR-branch holds
    warm = make_profile(warmth="warm", solar_elevation_deg=45.0)
    assert styles._score("golden", warm) == styles.SCORE_BASE + styles.SCORE_FAVORED
