"""Distinctness guard in auto-selected variant planning (photo style variants).

``select_styles`` scores the SceneProfile alone and cannot see that a picked
style collapsed onto an already-picked variant. ``plan_variants`` (auto-select
ONLY) must drop such a candidate loudly and backfill the next-ranked style
under a TWO-TIER keep test: a params-space pre-filter (params-identical implies
pixel-identical — dropped without rendering) and then RENDERED-pixel distance
on the bounded metering view — params distance alone is a broken proxy that
waves through a pastel variant rendering invisibly on a flat hazy frame. An
explicit ``styles`` list is operator authority — rendered verbatim, never
filtered, never warned about.
"""

import logging
from dataclasses import astuple, replace
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.color_correction import CorrectionParams
from gomoveshift.pipeline.photos import develop as develop_mod
from gomoveshift.pipeline.photos import styles
from gomoveshift.pipeline.photos.scene import compute_scene_profile

DEV_LOGGER = "gomoveshift.pipeline.photos.develop"
WB_FIELDS = ("white_balance_r", "white_balance_b", "white_balance_r_high", "white_balance_b_high")


def _decoded(seed=11):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    return develop_mod.DecodedPhoto(
        rgb=rgb, path=Path("IMG_TEST.jpg"), decoder="test",
        width=64, height=48, bit_depth=8, is_raw=False,
    )  # fmt: skip


def _hazy_decoded():
    """The operator's misty-mountain failure class: a flat, hazy, low-contrast
    bright frame — narrow-histogram gray gradient plus mild noise."""
    rng = np.random.default_rng(7)
    gradient = np.linspace(165.0, 195.0, 128)
    gray = np.clip(
        np.broadcast_to(gradient, (96, 128)) + rng.normal(0.0, 2.0, (96, 128)), 0, 255
    ).astype(np.uint8)
    return develop_mod.DecodedPhoto(
        rgb=np.stack([gray, gray, gray], axis=2), path=Path("IMG_HAZY.jpg"), decoder="test",
        width=128, height=96, bit_depth=8, is_raw=False,
    )  # fmt: skip


def _metering_view(decoded):
    return develop_mod._bounded_resize(
        develop_mod._analysis_rgb(decoded.rgb), develop_mod.DISTINCT_RENDER_DIM
    )


def _ranking(decoded):
    scene = compute_scene_profile(decoded.rgb)
    return styles.select_styles(scene, count=len(styles.STYLES))


def _pastel_like(base):
    """Params visibly far from natural (gamma lift, matte fade, halved contrast,
    desat) that render nearly identical on a flat hazy gray frame."""
    return replace(
        base,
        gamma=base.gamma + 0.1,
        black_point=base.black_point + 8.0,
        contrast_curve=base.contrast_curve * 0.5,
        saturation_mult=base.saturation_mult * 0.85,
    )


def _punchy(base):
    """A genuinely different look on the same hazy frame — deep contrast + crush."""
    return replace(
        base,
        contrast_curve=base.contrast_curve + 0.10,
        highlight_pull=base.highlight_pull + 15.0,
        black_point=base.black_point - 8.0,
    )


def _dim(base):
    return replace(base, gamma=base.gamma - 0.15, shadow_lift=base.shadow_lift + 6.0)


def _distinct_move(base, index):
    """Orthogonal look moves — every pair renders far over the distinctness
    floor on the colorful test frame, so structural tests never trip the
    rendered gate by accident (same-field offsets compress under clipping)."""
    moves = (
        dict(saturation_mult=base.saturation_mult + 0.5),
        dict(saturation_mult=0.0),
        dict(gamma=base.gamma - 0.18),
        dict(gamma=base.gamma + 0.18),
        dict(
            contrast_curve=base.contrast_curve + 0.12,
            black_point=base.black_point - 10.0,
            highlight_pull=base.highlight_pull + 12.0,
        ),
    )
    return replace(base, **moves[(index - 1) % len(moves)])


def _fake_apply(collapse_names, ranking):
    """Style stand-in: names in ``collapse_names`` return the base verbatim
    (identical to natural); every other style applies a distinct orthogonal
    look move — over the params epsilon AND far over the rendered floor.
    Monkeypatching keeps these tests independent of the real style deltas."""

    def fake(name, base, scene, strength=styles.DEFAULT_STRENGTH):
        if name == "natural" or name in collapse_names:
            return replace(base)
        return _distinct_move(base, ranking.index(name))

    return fake


def _warnings(caplog):
    return [r for r in caplog.records if r.name == DEV_LOGGER and r.levelno >= logging.WARNING]


def test_params_distance_identical_is_zero():
    a = CorrectionParams(shadow_lift=8.0, saturation_mult=1.1, contrast_curve=0.03, gamma=1.05)
    assert develop_mod._params_distance(a, replace(a)) == 0.0


def test_params_distance_full_authority_moves_are_comparable():
    a = CorrectionParams()
    d_sat = develop_mod._params_distance(
        a, replace(a, saturation_mult=a.saturation_mult + styles.STYLE_SATURATION_CAP)
    )
    d_shadow = develop_mod._params_distance(
        a, replace(a, shadow_lift=a.shadow_lift + styles.SHADOW_LIFT_MAX)
    )
    assert d_sat == pytest.approx(1.0)
    assert d_shadow == pytest.approx(1.0)


def test_params_distance_single_field_delta_scales_linearly():
    a = CorrectionParams()
    one = develop_mod._params_distance(a, replace(a, contrast_curve=0.02))
    two = develop_mod._params_distance(a, replace(a, contrast_curve=0.04))
    assert one > 0.0
    assert two == pytest.approx(2.0 * one)


def test_params_distance_counts_every_wb_field():
    a = CorrectionParams()
    for field in WB_FIELDS:
        assert develop_mod._params_distance(a, replace(a, **{field: 0.02})) > 0.0


def test_rendered_distance_identical_params_is_zero():
    view = _metering_view(_decoded())
    a = CorrectionParams(shadow_lift=8.0, saturation_mult=1.1, contrast_curve=0.03, gamma=1.05)
    assert develop_mod._rendered_distance(view, a, replace(a)) == 0.0


def test_rendered_distance_saturation_delta_scales_on_colorful_view():
    view = _metering_view(_decoded())
    base = CorrectionParams()
    small = develop_mod._rendered_distance(view, base, replace(base, saturation_mult=1.2))
    big = develop_mod._rendered_distance(view, base, replace(base, saturation_mult=1.4))
    assert small > 0.0
    assert big > small, "a bigger saturation-only delta must render further away"


# --- Colour-cast term of the rendered distance -------------------------------
#
# Mean |diff| alone is a TONE meter: a cast-type style (golden) shifts channels
# in opposite directions with a near-zero luma move, measures ~2.7 mean 8-bit
# levels on real captures, and was the only style the floor ever dropped — while
# a blind judge picks it as best on 7 of 23. The distance therefore adds a term
# for the CONSISTENT colour cast: the per-channel mean signed difference with
# its shared (luma) component removed. Scattered noise cancels in a signed
# mean, a uniform tone lift is removed with the shared component — only a
# directional colour move earns the boost.


def _mid_gray():
    return np.full((32, 32, 3), 128, dtype=np.uint8)


def _pinned_distance(monkeypatch, frame_a, frame_b):
    """``_rendered_distance`` between two PINNED synthetic renders: ``_grade``
    is stubbed to return the given frames verbatim, so these tests exercise the
    METRIC alone, independent of the grade math and the real style deltas."""
    a = CorrectionParams()
    b = replace(a, saturation_mult=a.saturation_mult + 1.0)
    frames = {astuple(a): frame_a, astuple(b): frame_b}
    monkeypatch.setattr(develop_mod, "_grade", lambda view, params: frames[astuple(params)])
    return develop_mod._rendered_distance(_mid_gray(), a, b)


def test_uniform_tone_shift_still_clears_the_floor(monkeypatch):
    lifted = (_mid_gray().astype(np.int16) + 25).clip(0, 255).astype(np.uint8)
    distance = _pinned_distance(monkeypatch, _mid_gray(), lifted)
    assert distance >= develop_mod.MIN_RENDERED_DISTINCTNESS


def test_consistent_colour_cast_clears_the_floor(monkeypatch):
    """THE BUG: a +8R/-2B whole-frame cast — a directional colour move a judge
    sees immediately — measures only ~3.3 mean 8-bit levels and was dropped."""
    cast = (_mid_gray().astype(np.int16) + np.array([8, 0, -2], dtype=np.int16)).astype(np.uint8)
    distance = _pinned_distance(monkeypatch, _mid_gray(), cast)
    assert distance >= develop_mod.MIN_RENDERED_DISTINCTNESS


def test_scattered_noise_with_zero_mean_cast_does_not_clear(monkeypatch):
    """Same mean |diff| as the cast pair above (8R/0G/2B) but sign-alternating
    per pixel: the signed per-channel means cancel, so no cast boost."""
    signs = ((np.indices((32, 32)).sum(axis=0) % 2) * 2 - 1).astype(np.int16)
    deltas = np.stack([8 * signs, 0 * signs, 2 * signs], axis=2)
    noisy = (_mid_gray().astype(np.int16) + deltas).astype(np.uint8)
    distance = _pinned_distance(monkeypatch, _mid_gray(), noisy)
    assert distance < develop_mod.MIN_RENDERED_DISTINCTNESS


def test_small_uniform_shift_gains_no_cast_boost(monkeypatch):
    """A +2-level whole-frame lift is the hazy-pastel failure class: its signed
    per-channel means are pure luma, and luma is already the mean-|diff| term's
    job — the cast term must not double-count it over the floor."""
    lifted = (_mid_gray().astype(np.int16) + 2).astype(np.uint8)
    distance = _pinned_distance(monkeypatch, _mid_gray(), lifted)
    assert distance < develop_mod.MIN_RENDERED_DISTINCTNESS


def _patch_hazy_styles(monkeypatch):
    ranking = ["natural", "soft", "vivid", "moody", "golden", "mono"]
    fakes = {"soft": _pastel_like, "vivid": _punchy, "moody": _dim}
    monkeypatch.setattr(
        styles, "select_styles", lambda scene, count=3, subject=None: ranking[:count]
    )
    monkeypatch.setattr(
        styles,
        "apply_style",
        lambda name, base, scene, strength=styles.DEFAULT_STRENGTH: fakes.get(name, replace)(base),
    )


def test_pastel_on_hazy_frame_dropped_by_rendered_check(monkeypatch, caplog):
    """THE BUG: on a flat hazy frame the pastel-like params pass the params
    pre-filter by a wide margin yet render within ~2 levels of natural — the
    rendered check must drop them and backfill the next-ranked style."""
    decoded = _hazy_decoded()
    _patch_hazy_styles(monkeypatch)
    with caplog.at_level(logging.WARNING, logger=DEV_LOGGER):
        plan = develop_mod.plan_variants(decoded)
    names = [n for n, _ in plan.styled]
    assert names == ["natural", "vivid", "moody"], "pastel drops, next-ranked backfills"

    natural = plan.styled[0][1]
    pastel = _pastel_like(natural)
    assert (
        develop_mod._params_distance(pastel, natural) >= develop_mod.VARIANT_DISTINCTNESS_EPSILON
    ), "the params proxy waves the pastel through — only the rendered check can catch it"
    measured = develop_mod._rendered_distance(_metering_view(decoded), pastel, natural)
    assert measured < 2.5, "pastel must render nearly identical to natural on this frame"
    assert measured < develop_mod.MIN_RENDERED_DISTINCTNESS

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "soft" in message and "natural" in message and "vivid" in message
    assert f"{measured:.2f}" in message, "the rendered-drop warning names the measured distance"


def test_genuinely_different_candidate_kept_on_hazy_frame(monkeypatch):
    decoded = _hazy_decoded()
    _patch_hazy_styles(monkeypatch)
    plan = develop_mod.plan_variants(decoded)
    assert "vivid" in [n for n, _ in plan.styled], "a visibly different look must be kept"
    natural = plan.styled[0][1]
    measured = develop_mod._rendered_distance(_metering_view(decoded), _punchy(natural), natural)
    assert measured >= develop_mod.MIN_RENDERED_DISTINCTNESS


def test_params_identical_candidate_dropped_without_rendering(monkeypatch):
    decoded = _decoded()
    ranking = _ranking(decoded)
    second = ranking[1]
    monkeypatch.setattr(styles, "apply_style", _fake_apply({second}, ranking))
    calls = []
    real = develop_mod._rendered_distance

    def spy(view, a, b, **kwargs):
        calls.append((a, b))
        return real(view, a, b, **kwargs)

    monkeypatch.setattr(develop_mod, "_rendered_distance", spy)
    plan = develop_mod.plan_variants(decoded)
    assert second not in [n for n, _ in plan.styled]
    assert calls, "params-distant candidates must be measured in rendered pixels"
    assert all(a != b for a, b in calls), "a params-identical candidate never pays a render"


def test_forced_collapse_backfills_next_ranked(monkeypatch, caplog):
    decoded = _decoded()
    ranking = _ranking(decoded)
    second = ranking[1]
    monkeypatch.setattr(styles, "apply_style", _fake_apply({second}, ranking))
    with caplog.at_level(logging.WARNING, logger=DEV_LOGGER):
        plan = develop_mod.plan_variants(decoded)
    names = [n for n, _ in plan.styled]
    assert len(names) == 3
    assert names[0] == "natural"
    assert second not in names, "a collapsed style must be dropped"
    assert names == [ranking[0], ranking[2], ranking[3]], "next-ranked styles must backfill"
    warnings = _warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert second in message and "natural" in message and ranking[2] in message


def test_explicit_styles_render_verbatim_even_when_identical(monkeypatch, caplog):
    decoded = _decoded()
    ranking = _ranking(decoded)
    monkeypatch.setattr(styles, "apply_style", _fake_apply({"vivid"}, ranking))
    with caplog.at_level(logging.WARNING, logger=DEV_LOGGER):
        plan = develop_mod.plan_variants(decoded, styles=["natural", "vivid"])
    assert [n for n, _ in plan.styled] == ["natural", "vivid"]
    assert plan.styled[0][1] == plan.styled[1][1], "operator list ships even identical params"
    assert _warnings(caplog) == []


def test_exhaustion_falls_back_to_top_ranked(monkeypatch, caplog):
    decoded = _decoded()
    ranking = _ranking(decoded)
    monkeypatch.setattr(styles, "apply_style", _fake_apply(set(styles.STYLES), ranking))
    with caplog.at_level(logging.WARNING, logger=DEV_LOGGER):
        plan = develop_mod.plan_variants(decoded)
    assert [n for n, _ in plan.styled] == ranking[:3], "never fewer variants than the target"
    assert any(
        "exhaust" in w.getMessage().lower() or "fall" in w.getMessage().lower()
        for w in _warnings(caplog)
    )


def test_backfill_reaches_the_opt_in_styles_when_every_ranked_style_collapses(monkeypatch, caplog):
    """THE BUG: ``_distinct_styled`` asks for ``count=len(STYLES)`` — "give me
    the whole ladder to back-fill from" — but ``select_styles`` ranked only the
    five ``_FAVORED`` predicates, so it got six of eleven. With those six
    collapsed onto natural the backfill exhausted and shipped near-identical
    variants while the matte pair and the WB trio — legal, calibrated,
    structurally different — were never scored at all."""
    decoded = _decoded()
    ranked = ("natural", *styles._FAVORED)
    opt_in = [n for n in styles.STYLES if n not in ranked]

    def fake(name, base, scene, strength=styles.DEFAULT_STRENGTH):
        if name in ranked:
            return replace(base)  # every ranked style collapses onto natural
        return _distinct_move(base, opt_in.index(name) + 1)

    monkeypatch.setattr(styles, "apply_style", fake)
    with caplog.at_level(logging.WARNING, logger=DEV_LOGGER):
        plan = develop_mod.plan_variants(decoded)
    assert [n for n, _ in plan.styled] == ["natural", *opt_in[:2]]
    assert not any("exhaust" in w.getMessage().lower() for w in _warnings(caplog))


def test_determinism_same_inputs_same_list():
    decoded = _decoded()
    first = develop_mod.plan_variants(decoded)
    second = develop_mod.plan_variants(decoded)
    assert [n for n, _ in first.styled] == [n for n, _ in second.styled]
    assert [p for _, p in first.styled] == [p for _, p in second.styled]
