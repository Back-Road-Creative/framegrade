"""Tests for true split-toning in the photo develop grade chain.

``_apply_splittone`` is a photo-develop-only look control: two independent
signed (r,g,b) tints, one on the shadows and one on the highlights, blended by
BT.709 luma so midtones stay untouched. Default-OFF is bit-identical and it is
deliberately NOT wired into ``params_to_lut3d_cube``.
"""

import numpy as np
import pytest

from gomoveshift.pipeline import color_correction as cc
from gomoveshift.pipeline.photos import develop as dev


def _manual_chain(frame, params):
    """The develop grade chain WITHOUT split-toning — the pre-split-toning pixel path."""
    peak = 255.0
    flat = frame.reshape(-1, 3).astype(np.float32) / peak
    out = cc._apply_curves(flat, params)
    out = cc._apply_colorbalance(out, params)
    out = cc._apply_saturation(out, params.saturation_mult)
    out = cc._apply_hsl(out, params)
    out = cc._apply_gamma(out, params.gamma)
    h, w, _ = frame.shape
    return np.rint(np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(h, w, 3)


def test_default_off_is_a_bit_identical_no_op():
    """Empty tints leave every pixel byte-for-byte untouched."""
    rng = np.random.default_rng(0)
    rgb = rng.random((256, 3)).astype(np.float64)
    out = cc._apply_splittone(rgb, cc.CorrectionParams())
    assert out.tobytes() == rgb.tobytes()


def test_grade_default_off_matches_the_chain_without_splittone():
    """The default photo grade is byte-identical to the pre-split-tone chain."""
    frame = np.linspace(0, 255, 48 * 32 * 3, dtype=np.uint8).reshape(32, 48, 3)
    params = cc.CorrectionParams()
    assert np.array_equal(dev._grade(frame, params), _manual_chain(frame, params))


def test_shadow_and_highlight_tints_push_only_their_own_tones():
    """A teal shadow tint pushes a dark patch toward teal (red down, blue up); a
    warm highlight tint pushes a bright patch warm (red up, blue down); a mid-gray
    patch — where both luma weights vanish — is left effectively unchanged."""
    patches = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]], dtype=np.float64)
    params = cc.CorrectionParams(
        splittone_shadow=(-0.08, 0.05, 0.05),  # teal
        splittone_highlight=(0.08, 0.02, -0.06),  # warm
    )
    out = cc._apply_splittone(patches, params)
    d = out - patches
    # dark patch takes the shadow (teal) tint
    assert d[0, 0] < -1e-3 and d[0, 2] > 1e-3
    # bright patch takes the highlight (warm) tint
    assert d[1, 0] > 1e-3 and d[1, 2] < -1e-3
    # mid-gray untouched (both weights zero at luma 0.5)
    assert np.allclose(d[2], 0.0, atol=1e-6)


def test_grade_with_splittone_differs_from_the_neutral_chain():
    """A live tint measurably changes the graded pixels — proving _grade wires it."""
    frame = np.linspace(0, 255, 48 * 32 * 3, dtype=np.uint8).reshape(32, 48, 3)
    params = cc.CorrectionParams(splittone_shadow=(-0.1, 0.06, 0.06))
    assert not np.array_equal(dev._grade(frame, params), _manual_chain(frame, params))


def test_is_deterministic():
    rgb = np.random.default_rng(1).random((512, 3)).astype(np.float64)
    params = cc.CorrectionParams(
        splittone_shadow=(-0.05, 0.03, 0.04), splittone_highlight=(0.06, 0.0, -0.05)
    )
    assert np.array_equal(cc._apply_splittone(rgb, params), cc._apply_splittone(rgb, params))


def test_wrong_length_tint_is_rejected():
    with pytest.raises(ValueError, match="length 3"):
        cc.CorrectionParams(splittone_shadow=(0.1, 0.2))


def test_splittone_stage_feeds_the_develop_cache_version():
    """A retune of the split-tone applier must invalidate the developed-photo cache."""
    assert cc._apply_splittone in dev._HASHED_SOURCES


# ── CLI / config exposure (--split-tone-shadow / --split-tone-highlight) ──

from PIL import Image  # noqa: E402
from pydantic import ValidationError  # noqa: E402

import gomoveshift.config as config_module  # noqa: E402
from gomoveshift.cli.handlers import photos as photos_handler  # noqa: E402
from gomoveshift.cli.parser import _build_parser  # noqa: E402
from gomoveshift.config import Config, PhotosConfig  # noqa: E402


def _write_photo(path):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def _use_photos_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def test_splittone_threads_onto_every_styled_variant_not_the_base(tmp_path):
    """A non-empty tint lands on EVERY styled variant's params (natural too), but
    never the base — LUT looks ride a clean film-emulation base."""
    photo = _write_photo(tmp_path / "IMG_0090.jpg")
    options = dev.DevelopOptions(splittone_shadow=(-0.08, 0.05, 0.05))
    plan = dev.plan_variants(photo, ["natural", "moody"], options=options)
    for _, params in plan.styled:
        assert params.splittone_shadow == (-0.08, 0.05, 0.05)
        assert params.splittone_highlight == ()
    assert plan.base.splittone_shadow == ()  # LUT-look base stays clean


def test_splittone_flags_thread_through_the_handler(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0091.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    seen = []
    real = dev.develop_variants
    monkeypatch.setattr(
        dev,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    argv = f"photos develop {photo} --styles natural --split-tone-highlight 0.08 0.02 -0.06"
    assert _run(argv) == 0
    assert [(o.splittone_shadow, o.splittone_highlight) for o in seen] == [
        ((), (0.08, 0.02, -0.06))
    ]


def test_splittone_config_accepts_three_and_rejects_wrong_length():
    ok = PhotosConfig(splittone_shadow=[-0.08, 0.05, 0.05])
    assert ok.splittone_shadow == [-0.08, 0.05, 0.05]
    with pytest.raises(ValidationError):
        PhotosConfig(splittone_shadow=[0.1, 0.2])


def test_splittone_argparse_rejects_wrong_value_count(tmp_path):
    photo = _write_photo(tmp_path / "IMG_0092.jpg")
    with pytest.raises(SystemExit):
        _build_parser().parse_args(f"photos develop {photo} --split-tone-shadow 0.1 0.2".split())


def test_out_of_range_splittone_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0093.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail("decoded despite a bad split-tone"))
    assert _run(f"photos develop {photo} --styles natural --split-tone-shadow 0 0 9") == 1
    assert "--split-tone-shadow values must each be in [-0.5, 0.5]" in capsys.readouterr().err


# ── the third range: the midtones (--split-tone-midtone) ──────────────


def test_midtone_tint_moves_mid_luma_and_leaves_both_ends_alone():
    """A midtone tint rides the tent between the two existing ramps: full at
    mid-luma, nothing at black or white — where the shadow/highlight tints live."""
    patches = np.array([[0.0] * 3, [0.02] * 3, [0.5] * 3, [0.98] * 3, [1.0] * 3])
    out = cc._apply_splittone(patches, cc.CorrectionParams(splittone_midtone=(0.06, -0.03, 0.05)))
    d = out - patches
    assert np.allclose(d[0], 0.0, atol=1e-12), "black took a midtone tint"
    assert np.allclose(d[4], 0.0, atol=1e-12), "white took a midtone tint"
    assert d[2, 0] > 1e-3 and d[2, 1] < -1e-3 and d[2, 2] > 1e-3, "mid-luma took no tint"
    # The ramp is continuous, so the ends taper rather than switch off: a patch
    # near either end keeps at most a tenth of the mid-luma move.
    assert abs(d[1]).max() < 0.1 * abs(d[2]).max()
    assert abs(d[3]).max() < 0.1 * abs(d[2]).max()


def test_the_three_luma_weights_partition_the_range_with_no_seam():
    """Shadow + midtone + highlight weight must sum to exactly 1 at every luma —
    no gap where two ramps meet, no double-count where they overlap. Proven on a
    gray ramp: one identical tint on all three ranges is a UNIFORM push."""
    gray = np.repeat(np.linspace(0.1, 0.9, 129)[:, None], 3, axis=1)
    tint = (0.05, 0.05, 0.05)
    out = cc._apply_splittone(
        gray,
        cc.CorrectionParams(
            splittone_shadow=tint, splittone_midtone=tint, splittone_highlight=tint
        ),
    )
    assert np.allclose(out - gray, 0.05, atol=1e-9)


def test_an_empty_midtone_leaves_the_shipped_two_tint_math_bit_identical():
    """The two-tint pixel path as it was before the midtone tint, written out: an unset third tint must not perturb it."""
    rgb = np.random.default_rng(3).random((512, 3))
    sh, hi = (-0.05, 0.03, 0.04), (0.06, 0.0, -0.05)
    luma = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    want = rgb + np.asarray(sh) * np.clip(1.0 - 2.0 * luma, 0.0, 1.0)[:, None]
    want = want + np.asarray(hi) * np.clip(2.0 * luma - 1.0, 0.0, 1.0)[:, None]
    params = cc.CorrectionParams(splittone_shadow=sh, splittone_highlight=hi)
    assert cc._apply_splittone(rgb, params).tobytes() == np.clip(want, 0.0, 1.0).tobytes()


def test_an_all_zero_midtone_is_still_the_bit_identical_no_op():
    rgb = np.random.default_rng(4).random((64, 3))
    out = cc._apply_splittone(rgb, cc.CorrectionParams(splittone_midtone=(0.0, 0.0, 0.0)))
    assert out.tobytes() == rgb.tobytes()


def test_wrong_length_midtone_tint_is_rejected():
    with pytest.raises(ValueError, match="length 3"):
        cc.CorrectionParams(splittone_midtone=(0.1, 0.2))


def test_the_midtone_tint_never_reaches_the_baked_lut(tmp_path):
    """``_apply_splittone`` is deliberately outside ``params_to_lut3d_cube`` — the
    third tint must not change that, so video output and baked LUTs stay identical."""
    plain = cc.CorrectionParams(saturation_mult=1.1)
    toned = cc.CorrectionParams(saturation_mult=1.1, splittone_midtone=(0.4, -0.4, 0.4))
    a = cc.params_to_lut3d_cube(plain, cube_size=8, output_path=tmp_path / "a.cube")
    b = cc.params_to_lut3d_cube(toned, cube_size=8, output_path=tmp_path / "b.cube")
    assert a.read_bytes() == b.read_bytes()


def test_midtone_threads_onto_every_styled_variant_not_the_base(tmp_path):
    photo = _write_photo(tmp_path / "IMG_0094.jpg")
    options = dev.DevelopOptions(splittone_midtone=(0.06, -0.03, 0.05))
    plan = dev.plan_variants(photo, ["natural", "moody"], options=options)
    for _, params in plan.styled:
        assert params.splittone_midtone == (0.06, -0.03, 0.05)
    assert plan.base.splittone_midtone == ()  # LUT-look base stays clean


def test_midtone_is_reachable_from_a_look_and_from_a_layer():
    """A look names it by FLAG name; a layer may gate it by mask like its siblings."""
    look = {"split_tone_midtone": [0.06, -0.03, 0.05]}
    assert PhotosConfig(looks={"tinted": look}).looks["tinted"] == look
    resolved = dev.look_variant_options(look, dev.DevelopOptions())
    assert resolved.splittone_midtone == (0.06, -0.03, 0.05)
    config_module.check_layers("--layers", [{"mask": "sky", **look}])


def test_midtone_flag_threads_through_the_handler(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_0095.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    seen = []
    real = dev.develop_variants
    monkeypatch.setattr(
        dev,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    argv = f"photos develop {photo} --styles natural --split-tone-midtone 0.06 -0.03 0.05"
    assert _run(argv) == 0
    assert [o.splittone_midtone for o in seen] == [(0.06, -0.03, 0.05)]


def test_out_of_range_midtone_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_0096.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail("decoded despite a bad split-tone"))
    assert _run(f"photos develop {photo} --styles natural --split-tone-midtone 0 0 9") == 1
    assert "--split-tone-midtone values must each be in [-0.5, 0.5]" in capsys.readouterr().err
