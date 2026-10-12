"""Spatial stage: the light leak (a warm corner glow fogging the emulsion).

Stray light past the camera back exposes the film from one FIXED corner in one FIXED
warm hue -- no rng. Three invariants carry the design: amount 0 is a byte-identical
no-op, the blend is SCREEN so it cannot overshoot (``_halation``'s reason), and the
ramp is a fraction of the frame DIAGONAL, so one value is one look at every export
size -- the lesson ``GRAIN_REFERENCE_DIM`` records, measured.
"""

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_mod


def _flat(long_edge, value=160):
    """A perfectly flat 3:2 landscape frame — every corner starts identical."""
    return np.full((round(long_edge * 2 / 3), long_edge, 3), value, dtype=np.uint8)


def _corners(rgb):
    """The leak corner's patch and the OPPOSITE corner's, an eighth of the frame
    across, read off ``LIGHT_LEAK_CORNER`` so moving the constant moves the test."""
    height, width = rgb.shape[:2]
    box_h, box_w = max(2, height // 8), max(2, width // 8)
    corner_y, corner_x = develop_mod.LIGHT_LEAK_CORNER
    rows = slice(0, box_h) if corner_y < 0.5 else slice(height - box_h, height)
    cols = slice(0, box_w) if corner_x < 0.5 else slice(width - box_w, width)
    far_rows = slice(height - box_h, height) if corner_y < 0.5 else slice(0, box_h)
    far_cols = slice(width - box_w, width) if corner_x < 0.5 else slice(0, box_w)
    return rgb[rows, cols].astype(np.float64), rgb[far_rows, far_cols].astype(np.float64)


def test_light_leak_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._light_leak(rgb, 0.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_the_named_corner_brightens_and_warms_while_the_opposite_one_stays_put():
    """Direction, never magic values: the leak corner lifts AND its red-minus-blue
    grows (a warm glow, not a neutral bloom), and the far corner is outside the
    ramp's reach entirely — untouched bytes."""
    flat = _flat(600)
    out = develop_mod._light_leak(flat, 1.0)
    near, far = _corners(out)
    near_in, far_in = _corners(flat)
    assert near.mean() > near_in.mean() + 5.0, "the leak corner must actually glow"
    assert (
        near[..., 0].mean() - near[..., 2].mean() > near_in[..., 0].mean() - near_in[..., 2].mean()
    ), "the glow must be WARM — red above blue, not a neutral bloom"
    assert np.array_equal(far, far_in), "the opposite corner sits outside the ramp"


def test_the_ramp_falls_off_monotonically_away_from_the_leak_corner():
    """One radial ramp anchored at one corner: walking from it to the opposite
    corner, the lift never comes back up."""
    flat = _flat(600)
    lift = (develop_mod._light_leak(flat, 1.0).astype(np.float64) - flat).mean(axis=2)
    corner = np.asarray(develop_mod.LIGHT_LEAK_CORNER)
    steps = np.linspace(corner, 1.0 - corner, 40) * (np.asarray(lift.shape) - 1)
    walk = lift[tuple(np.rint(steps).astype(int).T)]
    assert np.all(np.diff(walk) <= 0.0), "the ramp must decay away from its corner"
    assert walk[0] > walk[-1], "and it must actually be a ramp, not a constant"


def test_screen_blend_cannot_overshoot_the_ceiling():
    """SCREEN, not add — the reason ``_halation`` documents. Screen closes a
    FRACTION of what headroom a pixel has left, so a frame already at full scale
    comes back byte-identical AND a frame just under the ceiling never reaches it.
    The second half is what an add + clip fails: it slams the whole bright corner
    flat against the ceiling instead of lifting it."""
    for dtype, peak in ((np.uint8, 255), (np.uint16, 65535)):
        full = np.full((60, 90, 3), peak, dtype=dtype)
        out = develop_mod._light_leak(full, 1.0)
        assert out.dtype == dtype
        assert np.array_equal(out, full), "screen overshot its own ceiling"
        nearly = np.full((60, 90, 3), int(peak * 0.94), dtype=dtype)
        lifted = develop_mod._light_leak(nearly, 1.0)
        assert lifted.max() > nearly.max(), "the near-max frame must still glow"
        assert lifted.max() < peak, "a clipped ADD, not a screen — the corner is flat"


def test_the_leak_is_frame_relative_and_survives_a_larger_export():
    """The scale-invariance property, pre-applied: the ramp is a fraction of the DIAGONAL, so
    doubling the export carries the same glow over the same FRACTION of the frame.
    A pixel-sized ramp would shrink to a quarter of the frame at twice the size."""
    reach, lift = [], []
    for long_edge in (600, 1200):
        flat = _flat(long_edge)
        out = develop_mod._light_leak(flat, 1.0)
        reach.append(np.count_nonzero(out != flat) / out.size)
        lift.append(out.astype(np.float64).mean() - flat.mean())
    assert reach[0] > 0.2, "the small export must carry a real glow to compare against"
    assert reach[1] == pytest.approx(reach[0], rel=0.02), f"reach moved: {reach}"
    assert lift[1] == pytest.approx(lift[0], rel=0.02), f"strength moved: {lift}"


def test_light_leak_is_deterministic_and_stays_in_range():
    """No rng at all — a pure function of the frame's own geometry — so two renders
    match and nothing can leave the dtype's range or land on a NaN."""
    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    first = develop_mod._light_leak(rgb, 0.7)
    assert np.array_equal(first, develop_mod._light_leak(rgb, 0.7))
    assert first.dtype == np.uint8
    assert first.min() >= 0 and first.max() <= 255
    assert not np.isnan(first).any()


def test_light_leak_tunables_participate_in_the_algo_version(monkeypatch):
    """Strength and reach are ALL_CAPS tunables, so a retune self-invalidates the
    reserved develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "LIGHT_LEAK_STRENGTH", develop_mod.LIGHT_LEAK_STRENGTH / 2)
    assert develop_mod._develop_algo_version() != baseline


# ── end to end through the develop surface ───────────────────────────


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_develop_variants_default_is_byte_identical_to_no_light_leak(tmp_path):
    photo = _write_photo(tmp_path / "IMG_L1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(light_leak=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)
    on = develop_mod.DevelopOptions(light_leak=0.9)
    leaked = dict(develop_mod.develop_variants(photo, ["natural"], options=on))["natural"].rgb
    assert not np.array_equal(plain, leaked)


def test_light_leak_runs_over_the_halation_and_under_the_fringe_and_grain(tmp_path, monkeypatch):
    """Position: stray light fogs the EMULSION, so it lands with halation's bleed —
    over it, under the lens fringe, and under the grain like every emulsion effect."""
    order = []
    for stage in ("_halation", "_light_leak", "_chromatic_ab", "_grain"):
        # ``*_t`` swallows the tint ``_halation`` takes as its third argument.
        monkeypatch.setattr(
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_L2.jpg")
    options = develop_mod.DevelopOptions(halation=0.6, light_leak=0.5, chromatic_ab=0.4, grain=0.3)
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == ["_halation", "_light_leak", "_chromatic_ab", "_grain"]


def test_develop_single_photo_path_is_off_by_default_and_leaks_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_L3.jpg")
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, light_leak=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, light_leak=0.9).rgb, base)


def test_out_of_range_light_leak_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_L4.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on bad leak"))
    args = _build_parser().parse_args(f"photos develop {photo} --light-leak 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--light-leak must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_config_validator_rejects_an_out_of_range_light_leak(bad):
    with pytest.raises(ValueError):
        PhotosConfig(light_leak=bad)
