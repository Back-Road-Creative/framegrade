"""``astro_contact``: ``stack_panels`` (fresh) and ``master_panels`` (from disk).
Synthetic arrays only, same discipline as ``astro_fixtures.py``."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from gomoveshift.pipeline.photos.astro_contact import (
    STRETCH_TARGET_DARKER,
    STRETCH_TARGET_LIGHTER,
    ContactSheetError,
    inspect_master,
    master_panels,
    recover_balanced,
    stack_panels,
    stretch_to_target,
)
from gomoveshift.pipeline.photos.astro_develop import (
    STRETCH_TARGET_BACKGROUND,
    background_wb,
    develop_stack,
    remove_gradient,
    stretch,
)


def _synthetic_sky(shape=(48, 48), seed=0):
    rng = np.random.default_rng(seed)
    rgb = np.clip(0.03 + 0.01 * rng.standard_normal((*shape, 3)), 0.0, None).astype(np.float32)
    for _ in range(15):
        y, x = rng.integers(0, shape[0]), rng.integers(0, shape[1])
        rgb[y, x] += rng.uniform(0.2, 1.0)
    return rgb.astype(np.float32)


def test_stretch_to_target_reproduces_stretch_at_the_default_target():
    rgb = _synthetic_sky()
    got = (np.clip(stretch_to_target(rgb, STRETCH_TARGET_BACKGROUND), 0, 1) * 65535.0).round()
    assert np.array_equal(got.astype(np.uint16), stretch(rgb))


def test_stretch_to_target_darker_and_lighter_move_the_background_the_right_way():
    rgb = _synthetic_sky()
    darker = stretch_to_target(rgb, STRETCH_TARGET_DARKER)
    lighter = stretch_to_target(rgb, STRETCH_TARGET_LIGHTER)
    assert np.median(darker) < np.median(lighter)


def test_recover_balanced_undoes_a_real_stretch_round_trip():
    rgb = _synthetic_sky()
    balanced, _gains = background_wb(remove_gradient(rgb))
    r = develop_stack(rgb)
    recovered = recover_balanced(r.rgb_uint16, r.black_point, r.m_values, r.peak)
    assert np.abs(recovered - balanced).max() < 1e-3


def test_stack_panels_labels_carry_the_parameters():
    stack_rgb = _synthetic_sky()
    report = SimpleNamespace(stack=SimpleNamespace(rgb=stack_rgb), develop=develop_stack(stack_rgb))
    labels = [label for label, _ in stack_panels(report)]
    assert any("default" in label and "target 0.12" in label for label in labels)
    assert any("no background WB" in label for label in labels)
    assert any(f"{STRETCH_TARGET_DARKER:g}" in label for label in labels)
    assert any(f"{STRETCH_TARGET_LIGHTER:g}" in label for label in labels)


def _write_master(tmp_path, r, *, horizon_frac=None):
    tif_path = tmp_path / "astro_stack_test.tif"
    tifffile.imwrite(tif_path, r.rgb_uint16)
    develop = dict(
        black_point=r.black_point,
        m_values=r.m_values,
        peak=r.peak,
        background_gains=list(r.background_gains),
    )
    sidecar = {"develop": develop, "mask": {"horizon_frac": horizon_frac}}
    tif_path.with_suffix(".stack.json").write_text(json.dumps(sidecar))
    return tif_path


def test_master_panels_reconstructs_and_omits_gradient_toggle(tmp_path):
    tif_path = _write_master(tmp_path, develop_stack(_synthetic_sky()))
    labels = [label for label, _ in master_panels(tif_path)]
    assert any("as shipped" in label for label in labels)
    assert any("no background WB" in label for label in labels)
    assert any(f"{STRETCH_TARGET_DARKER:g}" in label for label in labels)
    assert any(f"{STRETCH_TARGET_LIGHTER:g}" in label for label in labels)
    assert not any("gradient" in label for label in labels)


def test_master_panels_refuses_a_missing_sidecar(tmp_path):
    tif_path = tmp_path / "no_sidecar.tif"
    tifffile.imwrite(tif_path, _synthetic_sky().astype(np.uint16))
    with pytest.raises(ContactSheetError, match="stack.json"):
        master_panels(tif_path)


def test_inspect_master_returns_five_labeled_crops_and_never_touches_the_file(tmp_path):
    rng = np.random.default_rng(2)
    rgb16 = (rng.random((300, 300, 3)) * 30000).astype(np.uint16)
    tif_path = tmp_path / "master.tif"
    tifffile.imwrite(tif_path, rgb16)
    before = tif_path.read_bytes()

    panels, summary = inspect_master(tif_path)

    assert tif_path.read_bytes() == before, "astro-inspect must never modify the master"
    assert len(panels) == 5
    seen_positions = {label.split(" (")[0] for label, _ in panels}
    assert seen_positions == {"top-left", "top-right", "bottom-left", "bottom-right", "centre"}
    for _, crop in panels:
        assert crop.ndim == 3 and crop.shape[2] == 3
        assert crop.size > 0
    assert "centre" in summary


def test_inspect_master_refuses_an_unreadable_file(tmp_path):
    bogus = tmp_path / "not_a_tiff.tif"
    bogus.write_bytes(b"not a tiff")
    with pytest.raises(ContactSheetError, match="astro-inspect"):
        inspect_master(bogus)
