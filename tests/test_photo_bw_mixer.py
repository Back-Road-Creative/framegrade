"""Tests for the B&W channel mixer + duotone toning in the photo develop grade.

``_apply_monomix`` is a photo-develop-only stage riding ``_grade`` after
``_apply_hsl`` and before ``_apply_gamma``: custom R/G/B luma weights (the red
filter that darkens a blue sky) plus a two-colour luma ramp (sepia/selenium/
cyanotype). Both default-OFF and bit-identical, and deliberately NOT wired into
``params_to_lut3d_cube`` — video output and baked LUTs stay untouched.
"""

from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline import color_correction as cc
from gomoveshift.pipeline.photos import develop as dev

RED_MIX = (0.75, 0.25, 0.0)  # red filter: dark skies, bright red rock
SEPIA = (0.15, 0.08, 0.03, 0.98, 0.90, 0.72)
SELENIUM = (-0.06, 0.0, 0.10)  # cool split-tone in the shadows, over the B&W
TRITONE_MID = (0.45, 0.42, 0.30)  # the ramp's third anchor, at mid luma
SKY = np.array([[0.15, 0.35, 0.85]], dtype=np.float32)
BT709 = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
PATCHES = np.array([[0.15, 0.35, 0.85], [0.85, 0.20, 0.15], [0.5, 0.5, 0.5]], dtype=np.float32)


def test_default_off_is_bit_identical():
    """Both fields empty leaves every pixel untouched (early ``rgb.copy()``)."""
    rgb = np.random.default_rng(0).random((256, 3)).astype(np.float32)
    assert np.array_equal(cc._apply_monomix(rgb, cc.CorrectionParams()), rgb)


@pytest.mark.parametrize(
    "params",
    [
        cc.CorrectionParams(),
        # A shipped split-toned recipe (cyberpunk-shaped). No built-in look and no
        # style sets a mono control, so moving the mixer up the chain must not move
        # one pixel of one of them.
        cc.CorrectionParams(
            splittone_shadow=(-0.05, 0.06, 0.12),
            hue_sat=(0.90, 0.75, 0.80, 1.60, 1.60, 1.70),
            vibrance=0.7,
            saturation_mult=1.1,
            gamma=1.05,
        ),
    ],
)
def test_grade_default_off_matches_the_chain_without_monomix(params):
    """The photo grade with both mono controls OFF is byte-identical to the
    pre-mixer chain — the mixer's position in it cannot show."""
    frame = np.linspace(0, 255, 48 * 32 * 3, dtype=np.uint8).reshape(32, 48, 3)
    out = cc._apply_curves(frame.reshape(-1, 3).astype(np.float32) / 255.0, params)
    out = cc._apply_channel_curves(out, params)
    out = cc._apply_colorbalance(out, params)
    out = cc._apply_splittone(out, params)
    out = cc._apply_saturation(out, params.saturation_mult)
    out = cc._apply_vibrance(out, params)
    out = cc._apply_hsl(out, params)
    out = cc._apply_gamma(out, params.gamma)
    expected = np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(frame.shape)
    assert np.array_equal(dev._grade(frame, params), expected)


def test_red_weighted_mix_darkens_blue_and_lightens_red():
    """The mixer's whole point: a red-weighted mix renders a blue sky DARKER and
    red rock LIGHTER than BT.709 luma does, and the result is achromatic."""
    patches = np.array([[0.15, 0.35, 0.85], [0.85, 0.20, 0.15]], dtype=np.float32)
    out = cc._apply_monomix(patches, cc.CorrectionParams(mono_mix=RED_MIX))
    luma = patches @ BT709
    assert out[0, 0] < luma[0] and out[1, 0] > luma[1]
    assert np.array_equal(out[:, 0], out[:, 1]) and np.array_equal(out[:, 1], out[:, 2])


def test_duotone_alone_tones_the_frames_own_luma():
    """Without a mixer the ramp rides BT.709 luma: black lands on the shadow
    colour, white on the highlight colour, every channel monotone between."""
    ramp = np.repeat(np.linspace(0.0, 1.0, 32, dtype=np.float32)[:, None], 3, axis=1)
    out = cc._apply_monomix(ramp, cc.CorrectionParams(duotone=SEPIA))
    assert np.allclose(out[0], SEPIA[:3], atol=1e-6)
    assert np.allclose(out[-1], SEPIA[3:], atol=1e-6)
    assert np.all(np.diff(out, axis=0) > 0.0)


def test_duotone_rides_the_channel_mixer_when_both_are_set():
    """With a mixer the ramp keys on the MIXED luma, so the red filter pushes the
    sky further toward the shadow tone than BT.709 luma would."""
    mixed = cc._apply_monomix(SKY, cc.CorrectionParams(mono_mix=RED_MIX, duotone=SEPIA))
    assert mixed[0, 0] < cc._apply_monomix(SKY, cc.CorrectionParams(duotone=SEPIA))[0, 0]


def test_a_split_tone_lands_on_the_monochrome_frame():
    """Split-toned B&W — selenium shadows over a red-filter monochrome, the most
    standard darkroom move there is. The mixer runs BEFORE the split-tone, so the
    tone survives on the grey frame and stays keyed to luma: chroma in the
    shadows, none by the highlights."""
    ramp = np.linspace(0.0, 1.0, 16, dtype=np.float32)[:, None] * np.array(
        [0.9, 0.6, 0.4], dtype=np.float32
    )
    frame = np.rint(ramp * 255.0).astype(np.uint8)[None, :, :]
    toned = dev._grade(frame, cc.CorrectionParams(mono_mix=RED_MIX, splittone_shadow=SELENIUM))
    plain = dev._grade(frame, cc.CorrectionParams(mono_mix=RED_MIX))
    chroma = toned[0, :, 2].astype(int) - toned[0, :, 0].astype(int)
    assert chroma[0] > 8, "the shadows must carry the split-tone, not be wiped grey"
    assert chroma[0] > chroma[len(chroma) // 2] >= chroma[-1] == 0
    assert np.array_equal(toned[0, -1], plain[0, -1]), "past mid-luma the tone is untouched"
    assert len(set(plain[0, :, 0].tolist())) > 1 and np.array_equal(plain[..., 0], plain[..., 2])


def test_duotone_strength_lerps_between_the_tone_and_plain_monochrome():
    """The ramp used to REPLACE every pixel's colour, so a duotone was
    all-or-nothing (so ``synthwave`` had to be built out of split-tones). 1.0 is
    that hard ramp unchanged, 0.0 the plain monochrome, 0.5 exactly between."""
    hard = cc.CorrectionParams(mono_mix=RED_MIX, duotone=SEPIA)
    plain = cc._apply_monomix(PATCHES, cc.CorrectionParams(mono_mix=RED_MIX))
    full = cc._apply_monomix(PATCHES, hard)
    assert hard.duotone_strength == 1.0, "the shipped default is the hard ramp"
    assert np.array_equal(cc._apply_monomix(PATCHES, replace(hard, duotone_strength=1.0)), full)
    half = cc._apply_monomix(PATCHES, replace(hard, duotone_strength=0.5))
    assert np.allclose(cc._apply_monomix(PATCHES, replace(hard, duotone_strength=0.0)), plain)
    assert np.allclose(half, (full + plain) / 2.0, atol=1e-6)


def test_a_third_anchor_bends_the_ramp_through_the_midtones():
    """Tritone: the ramp gets a MIDTONE anchor, so the middle of the print can
    hold its own colour instead of whatever the straight line between the two
    ends happens to pass through. The ends are unmoved and the join is smooth."""
    ramp = np.repeat(np.linspace(0.0, 1.0, 33, dtype=np.float32)[:, None], 3, axis=1)
    out = cc._apply_monomix(ramp, cc.CorrectionParams(duotone=SEPIA, duotone_mid=TRITONE_MID))
    straight = cc._apply_monomix(ramp, cc.CorrectionParams(duotone=SEPIA))
    assert np.allclose(out[0], SEPIA[:3], atol=1e-6) and np.allclose(out[-1], SEPIA[3:], atol=1e-6)
    assert np.allclose(out[16], TRITONE_MID, atol=1e-6), "mid luma lands ON the third anchor"
    assert not np.allclose(out[16], straight[16]), "and that is not where the two-ender sits"
    assert np.all(np.diff(out[:, 0]) > 0.0), "no kink: the red ramp still climbs throughout"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mono_mix": (0.5, 0.5)},
        {"duotone": SEPIA, "duotone_strength": 1.5},
        {"duotone": SEPIA, "duotone_strength": -0.1},
        {"duotone": SEPIA, "duotone_mid": (0.4, 0.4)},
        {"duotone_mid": TRITONE_MID},  # a mid anchor with no ramp to anchor
        {"mono_mix": (0.0, 0.0, 0.0)},  # non-positive weight sum
        {"mono_mix": (0.4, -0.6, 0.0)},
        {"duotone": (0.1, 0.1, 0.1, 0.9, 0.9)},
        {"duotone": (0.1, 0.1, 0.1, 0.9, 0.9, 2.0)},
    ],
)
def test_malformed_controls_are_rejected_by_post_init(kwargs):
    with pytest.raises(ValueError):
        cc.CorrectionParams(**kwargs)


def test_video_lut_bake_is_byte_identical_with_the_fields_set(tmp_path):
    """Photo-only by construction: the baked .cube ignores both fields."""
    base = cc.CorrectionParams(saturation_mult=1.1, gamma=1.05)
    toned = replace(
        base, mono_mix=RED_MIX, duotone=SEPIA, duotone_strength=0.5, duotone_mid=TRITONE_MID
    )
    off = cc.params_to_lut3d_cube(base, cube_size=17, output_path=tmp_path / "off.cube")
    on = cc.params_to_lut3d_cube(toned, cube_size=17, output_path=tmp_path / "on.cube")
    assert off.read_bytes() == on.read_bytes()


def test_monomix_feeds_the_develop_cache_version():
    """A retune of the mixer — or a move of it along the chain — must invalidate
    the developed-photo cache, so both the applier and the chain are hashed."""
    assert cc._apply_monomix in dev._HASHED_SOURCES
    assert dev._grade in dev._HASHED_SOURCES


# ── CLI / config exposure (--bw-mix / --duotone) ──────────────────────


def _write_photo(path):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_controls_thread_onto_every_styled_variant_but_not_the_base(tmp_path):
    photo = _write_photo(tmp_path / "IMG_0080.jpg")
    options = dev.DevelopOptions(mono_mix=RED_MIX, duotone=SEPIA, duotone_strength=0.5)
    plan = dev.plan_variants(photo, ["natural", "moody"], options=options)
    for _, params in plan.styled:
        assert params.mono_mix == RED_MIX and params.duotone == SEPIA
        assert params.duotone_strength == 0.5
    assert plan.base.mono_mix == () and plan.base.duotone == ()  # LUT looks stay clean


@pytest.mark.parametrize(
    "flags,message",
    [
        ("--bw-mix 0 0 0", "--bw-mix weights must sum to more than 0"),
        ("--duotone 0 0 0 1 1 9", "--duotone values must each be in [0, 1]"),
        ("--duotone-strength 2", "--duotone-strength must be in [0, 1]"),
        ("--duotone-mid 0.4 0.4 0.3", "--duotone-mid needs --duotone"),
    ],
)
def test_malformed_flags_rejected_before_any_decode(tmp_path, monkeypatch, capsys, flags, message):
    photo = _write_photo(tmp_path / "IMG_0081.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail("decoded despite a bad control"))
    args = _build_parser().parse_args(f"photos develop {photo} --styles natural {flags}".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert message in capsys.readouterr().err


def test_config_accepts_the_controls_and_rejects_wrong_lengths():
    assert PhotosConfig(mono_mix=[0.75, 0.25, 0.0]).mono_mix == [0.75, 0.25, 0.0]
    assert PhotosConfig().duotone_strength == 1.0 and PhotosConfig(duotone_strength=0.4)
    with pytest.raises(ValidationError):
        PhotosConfig(duotone=[0.1, 0.2, 0.3])
