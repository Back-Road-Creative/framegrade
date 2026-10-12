"""Phase-3 spatial stage: red-biased halation (the film highlight bleed).

Bright areas scatter a warm glow into whatever surrounds them: the stage seeds off
bright luma, blurs that mask at a width-scaled radius, and SCREENS it back weighted
toward red. Invariants: amount 0 is a byte-identical no-op, the glow is LOCAL (a
far-away dark region is untouched), and the radius scales with frame width.

BOTH tone regimes are covered deliberately. A blown synthetic frame is the easy
case; the REAL-RANGE frame is the one that matters, because this pipeline's grade
tops out near 0.67 luma and an absolute threshold calibrated on a blown frame is
inert on every photograph while passing every blown-frame test.

``--halation-tint`` colours that bleed: red is the film physics and stays the
default (empty tint = OFF = ``HALATION_RED_BIAS``, byte-identical), while a cyan
or magenta tint buys the neon bloom the fixed red weights cannot express.
"""

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_mod


def _bar(width, bar, height=64, field=20, peak=250):
    """A full-height blown bar, centred: vertically uniform, so the glow is 1-D."""
    rgb = np.full((height, width, 3), field, dtype=np.uint8)
    rgb[:, (width - bar) // 2 : (width + bar) // 2] = peak
    return rgb


def _glow_past_the_bar(rgb, width, bar):
    """Mean red over a window a fixed FRACTION of the bar wide, past its edge."""
    return float(rgb[:, (width + bar) // 2 :][:, : bar // 4, 0].mean())


#: A neon bloom: the mirror image of the shipped red bias, blue-weighted.
CYAN_TINT = (0.3, 0.9, 1.0)


def _band(rgb, horizon):
    """Mean (r,g,b) of the hillside band the sky above it bleeds into."""
    return rgb[horizon : horizon + 20].reshape(-1, 3).mean(axis=0)


def _real_range_scene(size=400):
    """The tone this grade ACTUALLY lands in: a gradient sky peaking at luma 0.659
    over a dark hillside, nothing near clipping. Returns the frame + horizon row."""
    rgb = np.full((size, size, 3), 46, dtype=np.uint8)
    horizon = int(size * 0.45)
    rgb[:horizon] = np.linspace(150, 168, horizon, dtype=np.uint8)[:, None, None]
    return rgb, horizon


def test_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rgb = np.random.default_rng(5).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._halation(rgb, 0.0)
    assert out.dtype == np.uint8 and np.array_equal(out, rgb)


def test_a_real_range_frame_glows_warm_because_the_seed_follows_the_frame():
    """The regression this stage shipped with: an absolute 0.72 threshold sat above
    EVERY pixel a real develop produces (a real frame grades to max 0.671 luma), so it
    was inert on photographs while passing on blown synthetic frames. The glow it
    should have laid on the hillside under the sky is warm, local, and monotone."""
    rgb, horizon = _real_range_scene()
    assert float(develop_mod._luma(rgb).max()) / 255.0 < 0.67  # the real grade's ceiling
    assert develop_mod._halation_seed(rgb) < 0.67
    band = [
        develop_mod._halation(rgb, a)[horizon : horizon + 20].reshape(-1, 3).mean(axis=0)
        for a in (0.0, 0.5, 1.0)
    ]
    assert band[0][0] == 46.0  # amount 0 leaves the neutral hillside exactly as it was
    assert band[1][0] > 56.0, band[1]  # HALF strength, a 10+ level lift, not a nudge
    assert band[0][0] < band[1][0] < band[2][0], band  # monotone in the amount
    gain = band[1] - band[0]
    assert gain[0] > gain[1] > gain[2] > 0.0, gain  # red-biased, the way film halation is
    lit = develop_mod._halation(rgb, 1.0)
    assert np.array_equal(lit[-60:], rgb[-60:])  # local: far pixels keep their bytes


def test_a_dark_frame_does_not_bloom_its_own_noise():
    """The absolute floor: a frame whose whole range sits under it seeds nothing."""
    dark = np.random.default_rng(9).integers(0, 60, (120, 160, 3), dtype=np.uint8)
    assert develop_mod._halation_seed(dark) == develop_mod.HALATION_THRESHOLD_FLOOR
    assert np.array_equal(develop_mod._halation(dark, 1.0), dark)


def test_the_radius_scales_with_width_so_the_look_is_resolution_independent():
    """The same scene at 2x width glows the same distance out in FRACTIONS of the
    scene, not half as far -- the radius scales above SHARPEN_REFERENCE_WIDTH."""
    small = develop_mod._halation(_bar(1920, 120), 1.0)
    large = develop_mod._halation(_bar(3840, 240, height=128), 1.0)
    at_small = _glow_past_the_bar(small, 1920, 120)
    at_large = _glow_past_the_bar(large, 3840, 240)
    assert at_small > 45.0, at_small  # a 20-level field lifted past 45: plainly visible
    assert abs(at_small - at_large) < 3.0, (at_small, at_large)
    # ... and the RADIUS is what scaled: at the same ABSOLUTE offset the wider frame
    # still glows where the narrower one has fallen back to the field level.
    assert int(large[0, (3840 + 240) // 2 + 30, 0]) > int(small[0, (1920 + 120) // 2 + 30, 0]) + 10


def test_the_tunables_participate_in_the_algo_version(monkeypatch):
    """ALL_CAPS knobs, so a retune self-invalidates the reserved develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "HALATION_SEED_FRACTION", 0.9)
    assert develop_mod._develop_algo_version() != baseline


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_develop_variants_default_is_byte_identical_to_no_halation(tmp_path):
    photo = _write_photo(tmp_path / "IMG_H1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(halation=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)


def test_halation_runs_after_the_vignette_and_under_the_grain(tmp_path, monkeypatch):
    """Optical order: the vignette is glass in front of the film, the grain is in
    it, and halation happens in the emulsion between them."""
    order = []
    for stage in ("_vignette", "_halation", "_grain"):
        # ``*_t`` swallows the tint ``_halation`` takes as its third argument.
        monkeypatch.setattr(
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_H2.jpg")
    options = develop_mod.DevelopOptions(vignette=0.4, halation=0.6, grain=0.3)
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == ["_vignette", "_halation", "_grain"]


def test_develop_single_photo_path_is_off_by_default_and_glows_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_H3.jpg")
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, halation=0.0).rgb, base)
    lit = develop_mod.develop(photo, halation=0.9).rgb
    assert lit.astype(np.int16).mean() > base.astype(np.int16).mean()


def test_out_of_range_halation_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_H4.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on bad halation"))
    args = _build_parser().parse_args(f"photos develop {photo} --halation 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--halation must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_config_validator_rejects_an_out_of_range_halation(bad):
    with pytest.raises(ValueError):
        PhotosConfig(halation=bad)


# ── --halation-tint: the glow's colour ────────────────────────────────


def test_an_unset_tint_is_exactly_the_shipped_red_bias():
    """Empty = OFF = ``HALATION_RED_BIAS``, so the knob cannot re-grade a frame
    that never names it. The second assertion keeps the first non-vacuous."""
    rgb, _ = _real_range_scene()
    default = develop_mod._halation(rgb, 0.4)
    assert np.array_equal(default, develop_mod._halation(rgb, 0.4, develop_mod.HALATION_RED_BIAS))
    assert not np.array_equal(default, rgb)  # 0.4 really did bleed


def test_the_default_record_renders_the_red_bias_end_to_end(tmp_path):
    """The same identity through ``DevelopOptions`` and the whole render chain."""
    photo = _write_photo(tmp_path / "IMG_H5.jpg")

    def render(**knobs):
        options = develop_mod.DevelopOptions(halation=0.4, **knobs)
        return dict(develop_mod.develop_variants(photo, ["natural"], options=options))[
            "natural"
        ].rgb

    assert np.array_equal(render(), render(halation_tint=develop_mod.HALATION_RED_BIAS))
    assert not np.array_equal(render(), render(halation_tint=CYAN_TINT))


def test_zero_amount_stays_a_no_op_whatever_the_tint():
    rgb = np.random.default_rng(5).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    assert np.array_equal(develop_mod._halation(rgb, 0.0, CYAN_TINT), rgb)


def test_a_cyan_tint_bleeds_blue_where_the_red_default_bleeds_red():
    """The point of the knob: the CHANNEL ORDER of the bleed follows the tint."""
    rgb, horizon = _real_range_scene()
    base = _band(rgb, horizon)
    red = _band(develop_mod._halation(rgb, 1.0), horizon) - base
    cyan = _band(develop_mod._halation(rgb, 1.0, CYAN_TINT), horizon) - base
    assert red[0] > red[1] > red[2] > 0.0, red
    assert cyan[2] > cyan[1] > cyan[0] > 0.0, cyan


def test_a_full_tint_cannot_push_the_screen_blend_out_of_range():
    """Screen only brightens and cannot overshoot — a tint must not change that."""
    rgb, _ = _real_range_scene()
    out = develop_mod._halation(rgb, 1.0, (1.0, 1.0, 1.0))
    assert out.dtype == rgb.dtype
    assert (out >= rgb).all() and int(out.max()) <= 255


def test_two_renders_with_one_tint_are_byte_identical(tmp_path):
    photo = _write_photo(tmp_path / "IMG_H6.jpg")
    options = develop_mod.DevelopOptions(halation=0.5, halation_tint=CYAN_TINT)
    renders = [
        dict(develop_mod.develop_variants(photo, ["natural"], options=options))["natural"].rgb
        for _ in range(2)
    ]
    assert np.array_equal(*renders)


def test_a_malformed_halation_tint_is_refused_at_the_door(tmp_path, monkeypatch, capsys):
    """Wrong length dies in argparse, out-of-range in the handler — both before
    any decode, so a typo costs no render."""
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_H7.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on a bad tint"))
    with pytest.raises(SystemExit) as exited:
        _build_parser().parse_args(f"photos develop {photo} --halation-tint 0.3 0.9".split())
    assert exited.value.code != 0
    argv = f"photos develop {photo} --halation 0.5 --halation-tint -0.2 0.9 1.0".split()
    assert photos_handler.cmd_photos(_build_parser().parse_args(argv), None) == 1
    assert "--halation-tint values must each be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [[0.3, 0.9], [0.1] * 4])
def test_the_config_validator_rejects_a_wrong_length_halation_tint(bad):
    assert PhotosConfig().halation_tint == []  # empty = the red-bias default
    with pytest.raises(ValueError):
        PhotosConfig(halation_tint=bad)
