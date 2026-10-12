"""The graduated ND: a darkening ramp the operator can angle, on the gain seam.

The classic landscape filter — smoked glass at the top of the holder, clear at the
bottom — held over the lens to hold a bright sky down while the foreground keeps its
exposure. Unlike ``_sky_darken`` it meters NOTHING: it rides ``masks._linear``, a pure
geometric ramp, so the same amount is the same filter on every frame.

Direction is the whole failure mode, so it is pinned twice: at the default angle the
sky darkens and the ground comes back byte-identical, and across the WHOLE legal
angle range the top of the frame is never darkened less than the bottom — the tilt is
bounded at +/-90 so a ground-darkening grad cannot be expressed at all.
"""

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_mod


def _tone(rgb):
    return develop_mod._luma(rgb) / 255.0


def _sky_scene(h=360, w=480):
    """Bright sky over darker land, peak luma ~= 0.66 — a real develop's range."""
    yy, _ = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    lum = np.where(yy < 0.4 * h, 0.66 - 0.2 * yy / h, 0.22).astype(np.float32)
    return np.clip(np.rint(np.dstack([lum * 0.88, lum * 0.95, lum]) * 255.0), 0, 255).astype(
        np.uint8
    )


def test_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes, at any angle."""
    rgb = np.random.default_rng(5).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    for angle in (0.0, 45.0, -90.0):
        out = develop_mod._graduated_nd(rgb, 0.0, angle)
        assert out.dtype == np.uint8 and np.array_equal(out, rgb)


def test_the_ramp_holds_the_sky_down_and_leaves_the_ground_byte_identical():
    """The whole point, and the direction check: a grad ND that darkens the ground
    is the failure mode, so the lower half must come back untouched code for code."""
    frame = _sky_scene()
    assert 0.6 < _tone(frame).max() < 0.67  # a real develop's peak, never blown
    out = develop_mod._graduated_nd(frame, 1.0, 0.0)
    band = int(frame.shape[0] * 0.1)
    assert (_tone(frame[:band]).mean() - _tone(out[:band]).mean()) * 255.0 > 20.0
    assert np.array_equal(out[frame.shape[0] // 2 :], frame[frame.shape[0] // 2 :])
    means = [
        float(_tone(develop_mod._graduated_nd(frame, a, 0.0))[:band].mean()) for a in (0, 0.5, 1)
    ]
    assert means == sorted(means, reverse=True), means


@pytest.mark.parametrize("tilt", [-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0])
def test_no_legal_angle_points_the_grad_at_the_ground(tilt):
    """Robust by construction: the tilt is the filter's rotation in its holder and is
    bounded at +/-90, so the ramp swings from the sky through either SIDE and can
    never reach the ground. Metered on flat grey, where the only asymmetry is the ramp."""
    flat = np.full((240, 320, 3), 160, np.uint8)
    out = develop_mod._graduated_nd(flat, 1.0, tilt)
    band = out.shape[0] // 6
    top, bottom = float(out[:band].mean()), float(out[-band:].mean())
    assert top <= bottom + 1e-6, (tilt, top, bottom)
    if abs(tilt) < 90.0:
        assert top < bottom - 1.0, (tilt, top, bottom)


def test_a_quarter_turn_darkens_one_side_and_leaves_the_other_alone():
    """The angle is real: +90 swings the smoked half to the right edge, -90 to the left."""
    flat = np.full((240, 320, 3), 160, np.uint8)
    right = develop_mod._graduated_nd(flat, 1.0, 90.0)
    left = develop_mod._graduated_nd(flat, 1.0, -90.0)
    assert np.array_equal(right[:, : flat.shape[1] // 2], flat[:, : flat.shape[1] // 2])
    assert np.array_equal(left[:, flat.shape[1] // 2 :], flat[:, flat.shape[1] // 2 :])
    assert float(right[:, -1].mean()) < 100.0 and float(left[:, 0].mean()) < 100.0


def test_the_ramp_is_a_ramp_and_never_a_seam():
    """A visible line across the picture is what a hard-edge grad buys and this one
    refuses: the span is fixed at the shared soft ``masks.LINEAR_SPAN``."""
    flat = np.full((360, 480, 3), 200, np.uint8)
    column = develop_mod._graduated_nd(flat, 1.0, 0.0)[:, 240, 0].astype(np.int16)
    assert int(np.abs(np.diff(column)).max()) <= 2, column
    assert int(((column > 110) & (column < 190)).sum()) > 40, column


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_both_develop_surfaces_default_off_and_bite_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_G1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(graduated_nd=0.0, graduated_nd_angle=30.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero), "the angle alone must be inert while the amount is 0"
    on = develop_mod.DevelopOptions(graduated_nd=0.8)
    lit = dict(develop_mod.develop_variants(photo, ["natural"], options=on))["natural"].rgb
    assert not np.array_equal(plain, lit)
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, graduated_nd=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, graduated_nd=0.8).rgb, base)


def test_the_grad_sits_with_the_glass_after_the_region_edit_and_before_the_vignette(
    tmp_path, monkeypatch
):
    """A filter in the holder is optics, not a region edit: it runs after the sky
    burn and immediately BEFORE the lens's own corner falloff."""
    order = []
    for stage in ("_sky_darken", "_graduated_nd", "_vignette", "_halation"):
        monkeypatch.setattr(  # ``*_t``: ``_halation`` takes a third (tint) argument
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_G2.jpg")
    options = develop_mod.DevelopOptions(
        sky_darken=0.5, graduated_nd=0.5, vignette=0.3, halation=0.2
    )
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == ["_sky_darken", "_graduated_nd", "_vignette", "_halation"]


def test_the_stage_and_its_tunable_reach_the_develop_fingerprint(monkeypatch):
    """A new pixel-moving function must be hashed or the provenance stamp lies."""
    assert develop_mod._graduated_nd in develop_mod._HASHED_SOURCES
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "GRAD_ND_MAX", 0.75)
    assert develop_mod._develop_algo_version() != baseline


@pytest.mark.parametrize(
    "flag, value, message",
    [
        ("--graduated-nd", "1.5", "--graduated-nd must be in [0, 1]"),
        ("--graduated-nd-angle", "120", "--graduated-nd-angle must be in [-90, 90]"),
    ],
)
def test_a_bad_value_is_refused_before_any_decode(
    tmp_path, monkeypatch, capsys, flag, value, message
):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_G3.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on a bad grad"))
    args = _build_parser().parse_args(f"photos develop {photo} {flag} {value}".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("knobs", [{"graduated_nd": 1.4}, {"graduated_nd_angle": -120.0}])
def test_a_look_carrying_an_out_of_range_grad_is_refused_at_config_load(knobs):
    """Refused as an out-of-RANGE value, naming the look — not as an unknown knob."""
    with pytest.raises(ValueError, match="must be a number in"):
        PhotosConfig(looks={"stormy": knobs})
