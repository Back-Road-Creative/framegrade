"""Per-category quality score: orders shots inside a category, never excludes."""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline.photos import quality_order as qo


def _checker(size=128, cell=4):
    ys, xs = np.mgrid[0:size, 0:size]
    g = (((ys // cell) + (xs // cell)) % 2 * 200 + 30).astype(np.uint8)
    return np.dstack([g, g, g])


def _stars(n, size=192, seed=3):
    rng = np.random.default_rng(seed)
    img = rng.normal(12, 1.5, (size, size)).clip(0, 255).astype(np.float32)
    for _ in range(n):
        y, x = rng.integers(8, size - 8, 2)
        yy, xx = np.mgrid[-4:5, -4:5]
        img[y - 4 : y + 5, x - 4 : x + 5] += 220 * np.exp(-(yy**2 + xx**2) / 2.0)
    g = img.clip(0, 255).astype(np.uint8)
    return np.dstack([g, g, g])


def _in_range(x):
    return math.isfinite(x) and 0.0 <= x <= 1.0


def test_sharp_beats_blurred_technical():
    sharp = _checker()
    soft = cv2.GaussianBlur(sharp, (0, 0), 3)
    assert qo.technical_score(sharp)[0] > qo.technical_score(soft)[0]


def test_clipped_frame_penalised():
    base = _checker()
    clipped = base.copy()
    clipped[:, :80] = 255
    assert qo.technical_score(clipped)[0] < qo.technical_score(base)[0]
    assert qo.technical_score(clipped)[1]["clip_penalty"] > 0.0


@pytest.mark.parametrize("value", [0, 255])
@pytest.mark.parametrize("category", ["astro", "close_up", "landscape", "street"])
def test_flat_frames_finite_in_range(value, category):
    rgb = np.full((64, 64, 3), value, np.uint8)
    q = qo.score_developed(rgb, category)
    for x in (q.technical, q.category, q.total, *q.parts.values()):
        assert _in_range(x)


def test_night_more_stars_ranks_higher_and_starfree_low():
    many = qo.technical_score(_stars(40), night=True)[0]
    few = qo.technical_score(_stars(6), night=True)[0]
    none = qo.technical_score(_stars(0), night=True)[0]
    assert many > few > none
    assert none < 0.2


def test_night_parts_include_star_components():
    _, parts = qo.technical_score(_stars(30), night=True)
    assert {"stars", "star_sharpness", "trailing", "usable"} <= set(parts)


def test_night_trailing_penalised():
    round_stars = _stars(30)
    trailed = cv2.blur(round_stars, (15, 1))
    t_round = qo.technical_score(round_stars, night=True)[1]["trailing"]
    t_trail = qo.technical_score(trailed, night=True)[1]["trailing"]
    assert t_round >= t_trail


def test_close_up_isolation_beats_uniform_sharp():
    uniform = _checker(128, 2)
    iso = cv2.GaussianBlur(uniform, (0, 0), 4)
    h = uniform.shape[0]
    iso[h // 4 : 3 * h // 4, h // 4 : 3 * h // 4] = uniform[
        h // 4 : 3 * h // 4, h // 4 : 3 * h // 4
    ]
    assert qo.category_signals("close_up", iso)[0] > qo.category_signals("close_up", uniform)[0]


def test_landscape_warm_sky_beats_grey():
    h = w = 96
    grey = np.full((h, w, 3), 128, np.uint8)
    warm = np.zeros((h, w, 3), np.uint8)
    ramp = np.linspace(230, 120, h).astype(np.uint8)[:, None]
    warm[..., 0] = ramp  # R
    warm[..., 1] = (ramp * 0.7).astype(np.uint8)
    warm[..., 2] = (ramp * 0.35).astype(np.uint8)
    assert qo.category_signals("landscape", warm)[0] > qo.category_signals("landscape", grey)[0]


def test_astro_signals_prefer_dark_sky_with_stars():
    dark = _stars(30)
    bright = np.clip(dark.astype(int) + 90, 0, 255).astype(np.uint8)
    assert qo.category_signals("astro", dark)[0] > qo.category_signals("astro", bright)[0]


def test_astro_foreground_separation_in_parts():
    _, parts = qo.category_signals("astro", _stars(10))
    assert {"stars", "sky_dark", "foreground_sep"} <= set(parts)


def test_other_category_parts():
    score, parts = qo.category_signals("street", _checker())
    assert _in_range(score) and {"vibrancy", "midtone"} <= set(parts)


def test_combine_with_and_without_aesthetic():
    assert qo.combine(0.8, 0.5, None) == pytest.approx(0.4)
    w = qo.AESTHETIC_WEIGHT
    assert qo.combine(0.8, 0.5, 1.0) == pytest.approx(0.8 * ((1 - w) * 0.5 + w * 1.0))
    assert qo.combine(0.8, 0.5, 1.0, aesthetic_weight=0.0) == pytest.approx(0.4)
    assert qo.combine(2.0, 2.0, 2.0) == 1.0
    assert qo.combine(float("nan"), 0.5, None) == 0.0


def test_order_category_desc_and_tie_by_key():
    s = lambda t: qo.QualityScore(t, t, None, t, {})  # noqa: E731
    out = qo.order_category({"b": s(0.5), "a": s(0.5), "c": s(0.9), "d": s(0.1)})
    assert out == ["c", "a", "b", "d"]


def test_input_unchanged_and_parts_complete():
    rgb = _checker()
    before = rgb.copy()
    q = qo.score_developed(rgb, "landscape", aesthetic=0.7)
    assert np.array_equal(rgb, before)
    assert q.aesthetic == 0.7
    assert {"sharpness", "clip_penalty", "golden_hour", "sky_ratio"} <= set(q.parts)
    assert qo.score_developed(rgb, "landscape").aesthetic is None


def test_large_input_downscaled(monkeypatch):
    seen = []
    real = qo.compute_cull_score
    monkeypatch.setattr(qo, "compute_cull_score", lambda a: (seen.append(a.shape), real(a))[1])
    big = np.zeros((2000, 3000, 3), np.uint8)
    qo.technical_score(big)
    assert max(seen[0][:2]) <= qo.MAX_DIM


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        qo.technical_score(np.zeros((8, 8), np.uint8))
