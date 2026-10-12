"""Conventional rendering must preserve measured colors and source pixels."""

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from gomoveshift.pipeline.photos.astro_recipe import (
    derivative_receipt,
    recipe_for,
    render_linear,
)


def test_presets_preserve_neutrality_color_ratios_and_input():
    rgb = np.array([[[0, 0, 0], [0.1, 0.2, 0.3], [20, 20, 20]]], dtype=np.float32)
    original = rgb.copy()
    for name in ("natural", "accent", "bold"):
        output = render_linear(rgb, recipe_for(name))
        assert np.array_equal(rgb, original)
        assert np.isfinite(output).all()
        assert output.min() == 0 and output.max() <= 1
        np.testing.assert_allclose(output[0, 1] / output[0, 1, 0], [1, 2, 3])
        assert output[0, 2, 0] == output[0, 2, 1] == output[0, 2, 2]


def test_natural_has_no_black_clipping_or_spatial_processing():
    recipe = recipe_for("natural")
    assert recipe.black_point == 0 and recipe.contrast == 1
    rgb = np.full((2, 3, 3), 0.01)
    np.testing.assert_allclose(render_linear(rgb, recipe), rgb / (1 + rgb))
    assert render_linear(rgb, recipe_for("bold")).mean() > render_linear(rgb, recipe).mean()


def test_crop_is_explicit_and_equivalent_to_full_render_slice():
    rgb = np.arange(60).reshape(4, 5, 3) / 100
    recipe = recipe_for("natural")
    np.testing.assert_array_equal(
        render_linear(rgb, recipe, crop=(1, 1, 4, 3)),
        render_linear(rgb, recipe)[1:3, 1:4],
    )
    for crop in ((-1, 0, 2, 2), (0, 0, 6, 2), (2, 2, 1, 1), (0.5, 0, 2, 2)):
        with pytest.raises(ValueError, match="crop"):
            render_linear(rgb, recipe, crop=crop)


@pytest.mark.parametrize(
    "rgb",
    [
        np.zeros((2, 3)),
        np.zeros((0, 2, 3)),
        np.full((1, 1, 3), np.nan),
        np.full((1, 1, 3), np.inf),
        np.full((1, 1, 3), -1),
    ],
)
def test_rejects_invalid_scene_linear_input(rgb):
    with pytest.raises(ValueError):
        render_linear(rgb, recipe_for("natural"))


def test_recipe_rejects_invalid_parameters_and_unknown_versions():
    for updates in (
        {"exposure": float("inf")},
        {"contrast": 0},
        {"black_point": -1},
        {"version": "future"},
    ):
        with pytest.raises(ValueError):
            render_linear(np.ones((1, 1, 3)), replace(recipe_for("natural"), **updates))
    with pytest.raises(ValueError):
        recipe_for("invent-stars")


def test_receipt_binds_source_output_and_recipe(tmp_path):
    source, output = tmp_path / "manifest.json", tmp_path / "render.jpg"
    source.write_bytes(b"source identity")
    output.write_bytes(b"derivative bytes")
    recipe = recipe_for("natural")
    receipt = derivative_receipt(source, output, recipe, crop=(0, 0, 2, 2))
    assert receipt["source"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["crop"] == [0, 0, 2, 2]
    assert receipt["generative"] is False
    assert receipt == derivative_receipt(source, output, recipe, crop=(0, 0, 2, 2))
    changed = derivative_receipt(source, output, recipe_for("accent"), crop=(0, 0, 2, 2))
    assert changed["recipe_sha256"] != receipt["recipe_sha256"]


def test_extreme_highlights_remain_finite_and_neutral_ramp_is_monotonic():
    ramp = np.repeat(np.logspace(-12, 300, 100)[:, None, None], 3, axis=2)
    for name in ("natural", "accent", "bold"):
        with np.errstate(over="raise", invalid="raise"):
            output = render_linear(ramp, recipe_for(name))
        assert np.isfinite(output).all() and output.max() <= 1
        assert (np.diff(output[:, 0, 0]) >= 0).all()


def test_receipt_changes_when_crop_or_output_changes(tmp_path):
    source, output = tmp_path / "source", tmp_path / "output"
    source.write_bytes(b"source")
    output.write_bytes(b"first")
    recipe = recipe_for("natural")
    initial = derivative_receipt(source, output, recipe)
    cropped = derivative_receipt(source, output, recipe, crop=(0, 0, 1, 1))
    assert cropped["recipe_sha256"] != initial["recipe_sha256"]
    output.write_bytes(b"second")
    assert derivative_receipt(source, output, recipe)["output"] != initial["output"]
