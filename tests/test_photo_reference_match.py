"""``--reference PATH``: grade a photo to a chosen hero frame's colour.

Distinct from lighting-cluster sharing, which NEUTRALIZES a shoot toward one
measured correction: this EMULATES a frame the operator picked. The transfer is
Reinhard's — convert to CIE Lab, shift and scale each channel so the frame's
mean and standard deviation equal the reference's, convert back.
"""

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.color_correction import (
    CorrectionParams,
    _apply_reference_match,
    lab_mean_std,
)
from gomoveshift.pipeline.photos import develop as dev


def _flat(seed, base, spread):
    """An (N, 3) float32 frame in [0, 1] scattered around ``base`` — the shape
    the ``_apply_*`` chain passes between its stages."""
    noise = np.random.default_rng(seed).normal(0.0, spread, (4096, 3)).astype(np.float32)
    return np.clip(np.asarray(base, np.float32) + noise, 0.0, 1.0)


def _write_photo(path, base=(0.26, 0.36, 0.52)):
    """A landscape-ish frame under a tonal ramp — cool blue-green by default,
    warm when ``base`` says so, so the pair reads as a real target/reference."""
    y = np.linspace(0.0, 1.0, 96, dtype=np.float32)[:, None, None]
    rgb = np.asarray(base, np.float32) + 0.25 * y + np.zeros((96, 128, 3), np.float32)
    rgb += np.random.default_rng(11).normal(0.0, 0.03, rgb.shape).astype(np.float32)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(cmd):
    return photos_handler.cmd_photos(_build_parser().parse_args(cmd.split()), None)


def _use_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


# ── the applier ──────────────────────────────────────────────────────


def test_an_unset_reference_is_a_bit_identical_no_op():
    """OFF by default and OFF by construction — the early ``rgb.copy()``."""
    frame = _flat(1, (0.40, 0.45, 0.60), 0.05)
    assert np.array_equal(_apply_reference_match(frame, CorrectionParams()), frame)


def test_matching_a_frame_to_its_own_stats_moves_nothing():
    """Its own mean and spread are already the target, so the transfer is the
    identity up to the Lab round trip."""
    frame = _flat(2, (0.35, 0.40, 0.55), 0.08)
    params = CorrectionParams(reference_lab=lab_mean_std(frame))
    assert np.allclose(_apply_reference_match(frame, params), frame, atol=2 / 255)


def test_a_matched_frame_lands_on_the_references_lab_mean_and_spread():
    """The whole point: a cool flat frame matched to a warm contrasty reference
    must ARRIVE at that reference's per-channel Lab mean and standard deviation,
    not merely drift toward it."""
    target = _flat(3, (0.30, 0.36, 0.52), 0.04)
    reference = _flat(4, (0.72, 0.45, 0.22), 0.10)
    ref = lab_mean_std(reference)
    before = np.abs(np.subtract(lab_mean_std(target), ref))
    matched = _apply_reference_match(target, CorrectionParams(reference_lab=ref))
    after = np.abs(np.subtract(lab_mean_std(matched), ref))
    assert after.max() < 1.0, f"did not converge: {after} (was {before})"
    assert after.max() < before.max()


def test_a_flat_channel_is_shifted_not_exploded():
    """A channel with no spread has an infinite scale ratio; it must be shifted
    onto the reference's mean and left alone, never blown up."""
    flat = np.full((256, 3), 0.5, np.float32)
    matched = _apply_reference_match(
        flat, CorrectionParams(reference_lab=(60.0, 20.0, 30.0, 9.0, 4.0, 6.0))
    )
    assert np.all(np.isfinite(matched)) and matched.min() >= 0.0 and matched.max() <= 1.0
    assert np.allclose(lab_mean_std(matched)[:3], (60.0, 20.0, 30.0), atol=1.0)


@pytest.mark.parametrize("bad", [(1.0, 2.0, 3.0), (50.0, 0.0, 0.0, 10.0, -1.0, 5.0)])
def test_a_malformed_reference_lab_is_refused_by_construction(bad):
    with pytest.raises(ValueError, match="reference_lab"):
        CorrectionParams(reference_lab=bad)


# ── the develop surface ──────────────────────────────────────────────


def test_the_match_rides_the_styled_variants_and_never_the_lut_grade(tmp_path):
    """A film emulation must stay the film's look, not the operator's hero frame."""
    photo = _write_photo(tmp_path / "IMG_0203.jpg")
    ref = (60.0, 20.0, 35.0, 10.0, 5.0, 7.0)
    options = dev.DevelopOptions(reference_lab=ref, lut_over="moody")
    plan = dev.plan_variants(photo, ["natural"], options=options)
    assert all(p.reference_lab == ref for _, p in plan.styled)
    assert plan.base.reference_lab == ()
    assert plan.lut_params is not None and plan.lut_params.reference_lab == ()


# ── the CLI door ─────────────────────────────────────────────────────


def test_a_missing_reference_fails_at_the_cli_door_naming_the_path(tmp_path, monkeypatch, capsys):
    """Upstream gate: the reference is decoded ONCE, before any source photo is."""
    photo = _write_photo(tmp_path / "IMG_0201.jpg")
    out = tmp_path / "out"
    _use_config(monkeypatch, developed_output_dir=str(out))
    missing = tmp_path / "hero.jpg"
    assert _run(f"photos develop {photo} --reference {missing}") == 1
    assert str(missing) in capsys.readouterr().err
    assert not out.exists(), "the gate must fire before a single photo is developed"


def test_the_reference_reaches_the_render_from_either_the_flag_or_the_config(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0202.jpg")
    hero = _write_photo(tmp_path / "hero.jpg", base=(0.66, 0.42, 0.16))
    run = f"photos develop {photo} --styles natural --out "

    def jpg(name):
        return np.asarray(Image.open(tmp_path / name / "IMG_0202_natural.jpg"), dtype=int)

    _use_config(monkeypatch)
    assert _run(run + str(tmp_path / "off")) == 0
    assert _run(f"{run}{tmp_path / 'flag'} --reference {hero}") == 0
    _use_config(monkeypatch, reference=str(hero))
    assert _run(run + str(tmp_path / "cfg")) == 0
    assert np.abs(jpg("off") - jpg("flag")).mean() >= dev.MIN_RENDERED_DISTINCTNESS
    assert np.array_equal(jpg("flag"), jpg("cfg")), "photos.reference is the flag's config layer"
