"""Phase-3 spatial stage: lateral chromatic aberration (the lens fringe).

Glass bends short wavelengths harder than long ones, so an off-axis point lands as
three slightly different-sized images: red magnified past green, blue under it.
Invariants: amount 0 is a byte-identical no-op, green never moves, the displacement
is ZERO on the optical axis and grows toward the corners, and the knob is anchored
to the FRAME (a fraction of the half-diagonal) rather than to pixels, so a 20 MP
master and a 2 K export carry one fringe -- the lesson ``GRAIN_REFERENCE_DIM`` records.
"""

import cv2
import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_mod


def _corner_block(size):
    """A bright block off-axis in the top-left quadrant, placed and sized in
    FRACTIONS of the frame -- the same scene at any resolution."""
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[size // 8 : size // 4, size // 8 : size // 4] = 240
    return rgb


def _mean_radius(plane):
    """Intensity-weighted mean distance from the frame centre, in HALF-DIAGONALS
    -- a scale-free measure of where one channel's content sits."""
    height, width = plane.shape
    ys = np.arange(height) - (height - 1) / 2.0
    xs = np.arange(width) - (width - 1) / 2.0
    radius = np.hypot(ys[:, None], xs[None, :]) / np.hypot(ys[0], xs[0])
    weight = plane.astype(np.float64)
    return float((radius * weight).sum() / max(weight.sum(), 1e-9))


def _fringe(rgb):
    """How far R sits OUTSIDE B, in half-diagonals: the fringe, measured scale-free."""
    return _mean_radius(rgb[:, :, 0]) - _mean_radius(rgb[:, :, 2])


def _checkers(size=401, cell=20):
    """An ACHROMATIC high-contrast grid: R==G==B everywhere, so any |R-B| the
    stage leaves behind is fringe it created, not colour that was already there."""
    axis = (np.arange(size) // cell) % 2
    tile = np.where(axis[:, None] ^ axis[None, :], 230, 30).astype(np.uint8)
    return np.repeat(tile[:, :, np.newaxis], 3, axis=2)


def _fringe_energy(rgb, window):
    """Mean |R-B| over a square window -- the visible colour fringe there."""
    view = rgb[window, window].astype(np.int16)
    return float(np.abs(view[:, :, 0] - view[:, :, 2]).mean())


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rgb = np.random.default_rng(5).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._chromatic_ab(rgb, 0.0)
    assert out.dtype == np.uint8 and np.array_equal(out, rgb)


def test_red_is_thrown_outward_and_blue_inward_while_green_holds():
    """The stage's whole claim: the two fringing channels move in OPPOSITE radial
    directions about the frame centre, and green -- the reference -- does not move."""
    rgb = _corner_block(400)
    out = develop_mod._chromatic_ab(rgb, 1.0)
    radii = [_mean_radius(out[:, :, c]) for c in (0, 1, 2)]
    assert radii[0] > radii[1] > radii[2], radii
    assert np.array_equal(out[:, :, 1], rgb[:, :, 1])  # green is the fixed reference
    assert _mean_radius(rgb[:, :, 1]) == radii[1]  # ... at exactly the source radius


def test_the_optical_axis_is_fixed_and_the_fringe_grows_toward_the_corner():
    """LATERAL chromatic aberration is zero ON AXIS by construction (the centre
    maps to itself) and grows with distance from it, which is why real fringing
    shows in the corners and never in the middle of the frame."""
    rgb = _checkers()
    out = develop_mod._chromatic_ab(rgb, 1.0)
    axis = slice(199, 202)  # 3x3 about the centre: displaced by well under 1/64 px
    assert np.array_equal(out[axis, axis], rgb[axis, axis])
    assert _fringe_energy(rgb, slice(0, 60)) == 0.0  # the source carries no fringe
    energy = [_fringe_energy(out, w) for w in (slice(170, 230), slice(80, 140), slice(0, 60))]
    assert energy[0] < energy[1] < energy[2], energy
    assert energy[2] > 10.0, energy  # and the corner fringe is plainly visible


def test_the_fringe_is_anchored_to_the_frame_not_to_the_pixel_grid():
    """The resolution-anchoring property, ASSERTED: one scene at two sizes carries
    one fringe, because the knob is a fraction of the half-diagonal rather than a
    pixel count. In pixels the 600 px render would fringe twice as far as the 300."""
    small = develop_mod._chromatic_ab(_corner_block(300), 1.0)
    large = develop_mod._chromatic_ab(_corner_block(600), 1.0)
    assert _fringe(small) > 0.002, _fringe(small)  # non-vacuous: there IS a fringe
    assert abs(_fringe(large) - _fringe(small)) < 0.05 * _fringe(small)
    shrunk = cv2.resize(large, (300, 300), interpolation=cv2.INTER_AREA)
    assert abs(_fringe(shrunk) - _fringe(small)) < 0.05 * _fringe(small)


@pytest.mark.parametrize("dtype, peak", [(np.uint8, 255), (np.uint16, 65535)])
def test_the_output_keeps_its_dtype_and_range_at_either_depth(dtype, peak):
    rgb = (np.random.default_rng(4).random((80, 120, 3)) * peak).astype(dtype)
    out = develop_mod._chromatic_ab(rgb, 1.0)
    assert out.dtype == dtype and out.shape == rgb.shape
    assert np.isfinite(out).all()
    assert 0 <= int(out.min()) and int(out.max()) <= peak


def test_the_tunables_participate_in_the_algo_version(monkeypatch):
    """ALL_CAPS knobs, so a retune self-invalidates the reserved develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "CHROMATIC_AB_CORNER_SHIFT", 0.02)
    assert develop_mod._develop_algo_version() != baseline


def test_develop_variants_default_is_byte_identical_to_no_chromatic_ab(tmp_path):
    photo = _write_photo(tmp_path / "IMG_C1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(chromatic_ab=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)


def test_chromatic_ab_runs_after_the_halation_and_under_the_grain(tmp_path, monkeypatch):
    """Optical order: the fringe is the LENS, but it is resolved by the sensor, so
    it sits under the noise floor the grain stands in for."""
    order = []
    for stage in ("_vignette", "_halation", "_chromatic_ab", "_grain"):
        # ``*_t`` swallows the tint ``_halation`` takes as its third argument.
        monkeypatch.setattr(
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_C2.jpg")
    options = develop_mod.DevelopOptions(vignette=0.4, halation=0.6, chromatic_ab=0.5, grain=0.3)
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == ["_vignette", "_halation", "_chromatic_ab", "_grain"]


def test_the_single_photo_path_is_off_by_default_and_fringes_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_C3.jpg")
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, chromatic_ab=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, chromatic_ab=1.0).rgb, base)


def test_two_runs_at_one_amount_are_byte_identical():
    """No RNG anywhere in the stage -- only a geometry read off the frame's own
    dimensions -- which the reserved develop cache and DEVELOP_ALGO_VERSION require."""
    rgb = _checkers()
    runs = [develop_mod._chromatic_ab(rgb, 0.8) for _ in range(2)]
    assert np.array_equal(*runs) and not np.array_equal(runs[0], rgb)


def test_out_of_range_chromatic_ab_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_C5.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on a bad knob"))
    args = _build_parser().parse_args(f"photos develop {photo} --chromatic-ab 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--chromatic-ab must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_config_validator_rejects_an_out_of_range_chromatic_ab(bad):
    with pytest.raises(ValueError):
        PhotosConfig(chromatic_ab=bad)
