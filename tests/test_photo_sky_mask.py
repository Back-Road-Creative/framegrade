"""Spatial stage: a soft sky mask and ``--sky-darken``.

Until now the sky was metered only to be EXCLUDED (``analyze_frame``'s
bright-upper-band ROI) and never rendered. ``_sky_mask`` renders it — soft-edged,
frame-relative and landscape-only, the same rules the meter already plays by —
and ``_sky_darken`` rides it onto the shared ``_apply_gain_mask`` seam. Every
quantitative test runs on a REAL-RANGE frame (max luma ≈ 0.66), the only tone
this grade chain produces: four features here shipped green and inert after
being calibrated on a full-range synthetic.
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


def _ridge_scene(h=360, w=480):
    """Bright sky over a JAGGED ridgeline, darker land below; peak luma ≈ 0.66."""
    rng = np.random.default_rng(3)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    horizon = 0.22 * h + 0.03 * h * np.sin(xx / 23.0)  # never axis-aligned
    sky = 0.70 - 0.20 * yy / h
    land = np.clip(0.17 + 0.17 * yy / h + 0.02 * rng.standard_normal(yy.shape), 0.02, 0.7)
    lum = np.where(yy < horizon, sky, land).astype(np.float32)
    rgb = np.dstack([lum * 0.88, lum * 0.95, lum])
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def test_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rgb = np.random.default_rng(5).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._sky_darken(rgb, 0.0)
    assert out.dtype == np.uint8 and np.array_equal(out, rgb)


def test_the_sky_of_a_real_range_frame_darkens_and_the_foreground_does_not():
    """The whole point, measured on the tone this chain actually produces."""
    frame = _ridge_scene()
    assert 0.6 < _tone(frame).max() < 0.67  # a real develop's peak, never blown
    out = develop_mod._sky_darken(frame, 1.0)
    band = int(frame.shape[0] * 0.15)
    sky_before, sky_after = _tone(frame[:band]).mean(), _tone(out[:band]).mean()
    assert (sky_before - sky_after) * 255.0 > 12.0, (sky_before, sky_after)
    # The foreground is untouched, byte for byte -- not merely "close".
    assert np.array_equal(out[frame.shape[0] // 2 :], frame[frame.shape[0] // 2 :])
    # ... and it is monotone in the amount.
    means = [float(_tone(develop_mod._sky_darken(frame, a))[:band].mean()) for a in (0, 0.5, 1.0)]
    assert means == sorted(means, reverse=True), means


def test_the_horizon_edge_is_soft():
    """A hard-edged sky mask is the classic tell of a bad landscape edit: the
    ridgeline must pass through intermediate values in BOTH axes, never a step."""
    mask = develop_mod._sky_mask(_ridge_scene())
    assert mask.shape == (360, 480) and 0.0 <= mask.min() and mask.max() > 0.9
    assert float(np.abs(np.diff(mask, axis=0)).max()) < 0.15  # down the ridge
    assert float(np.abs(np.diff(mask, axis=1)).max()) < 0.15  # along it
    column = mask[:, mask.shape[1] // 2]
    assert int(((column > 0.15) & (column < 0.85)).sum()) > 10, column


def test_a_portrait_frame_is_a_no_op():
    """The sky ROI is LANDSCAPE-only — a portrait still's bright
    top is usually the subject, so darkening it would darken a head."""
    tall = np.transpose(_ridge_scene(), (1, 0, 2)).copy()
    assert tall.shape[0] > tall.shape[1]
    assert not develop_mod._sky_mask(tall).any()
    assert np.array_equal(develop_mod._sky_darken(tall, 1.0), tall)


def test_a_frame_with_no_sky_gets_a_near_zero_mask():
    """No tonal separation between median and highlight end ⇒ no sky to render,
    so the confidence term takes the mask to zero instead of masking noise."""
    rng = np.random.default_rng(9)
    flat = (128 + rng.integers(-1, 2, (300, 400, 3))).astype(np.uint8)
    assert develop_mod._sky_mask(flat).max() < 0.05
    assert np.array_equal(develop_mod._sky_darken(flat, 1.0), flat)


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_both_develop_surfaces_default_off_and_bite_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_S1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(sky_darken=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, sky_darken=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, sky_darken=1.0).rgb, base)


def test_sky_darken_runs_before_the_vignette(tmp_path, monkeypatch):
    """A subject-region edit sits UNDER the lens effect, not over it."""
    order = []
    stages = ("_clarity", "_sky_darken", "_sky_saturate", "_vignette", "_halation", "_grain")
    for stage in stages:
        monkeypatch.setattr(  # ``*_t``: ``_halation`` takes a third (tint) argument
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_S2.jpg")
    options = develop_mod.DevelopOptions(
        clarity=0.4, sky_darken=0.5, sky_saturate=0.5, vignette=0.3, halation=0.2
    )
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == list(stages)


def test_the_tunables_participate_in_the_algo_version(monkeypatch):
    """ALL_CAPS knobs, so a retune self-invalidates the reserved develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "SKY_DARKEN_MAX", 0.5)
    assert develop_mod._develop_algo_version() != baseline


def test_out_of_range_sky_darken_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_S3.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on bad sky-darken"))
    args = _build_parser().parse_args(f"photos develop {photo} --sky-darken 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--sky-darken must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_config_validator_rejects_an_out_of_range_sky_darken(bad):
    with pytest.raises(ValueError):
        PhotosConfig(sky_darken=bad)


# ── ``--sky-saturate``: the SECOND lever on the one mask ─────────────────────
# The mask is shared; the seam is not. ``_apply_gain_mask`` takes a per-pixel
# SCALAR gain, and scaling all three channels of a pixel by one factor leaves
# ``(max-min)/max`` exactly where it was — a gain provably cannot move
# saturation. So this stage rides the same mask onto the masked-LERP shape
# ``_dehaze``/``_halation`` use instead.


def _chroma(rgb):
    """Mean per-pixel distance from that pixel's own luma — the axis saturation
    moves, and the one a gain cannot touch."""
    src = rgb.astype(np.float64)
    return np.abs(src - develop_mod._luma(rgb)[:, :, np.newaxis]).mean(axis=2)


def test_sky_saturate_zero_is_byte_identical():
    """The load-bearing no-op, exactly as ``_sky_darken``'s is."""
    rgb = np.random.default_rng(5).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._sky_saturate(rgb, 0.0)
    assert out.dtype == np.uint8 and np.array_equal(out, rgb)


def test_the_boost_lands_inside_the_sky_and_the_ground_keeps_its_bytes():
    frame = _ridge_scene()
    band = int(frame.shape[0] * 0.15)
    out = develop_mod._sky_saturate(frame, 1.0)
    assert _chroma(out[:band]).mean() > 1.25 * _chroma(frame[:band]).mean()
    # Outside the mask the move is ``0 * (rgb - luma)`` — exactly zero, so the
    # ground is untouched BY CONSTRUCTION, not merely "close".
    assert np.array_equal(out[frame.shape[0] // 2 :], frame[frame.shape[0] // 2 :])
    means = [float(_chroma(develop_mod._sky_saturate(frame, a))[:band].mean()) for a in (0, 0.5, 1)]
    assert means == sorted(means), means


def test_the_boost_moves_the_skys_colour_and_not_its_tone():
    """What makes it a DIFFERENT lever from ``--sky-darken`` rather than a
    second dial on the same one: the lerp is around each pixel's own luma, so
    the sky's tone survives it."""
    frame = _ridge_scene()
    band = int(frame.shape[0] * 0.15)
    out = develop_mod._sky_saturate(frame, 1.0)
    assert abs(_tone(out[:band]).mean() - _tone(frame[:band]).mean()) * 255.0 < 1.0


def test_sky_saturate_is_a_no_op_wherever_the_mask_is(tmp_path):
    """The mask's own guards are inherited, not re-derived: a portrait frame and
    a frame too flat to hold a sky both come back byte-identical."""
    tall = np.transpose(_ridge_scene(), (1, 0, 2)).copy()
    assert np.array_equal(develop_mod._sky_saturate(tall, 1.0), tall)
    flat = (128 + np.random.default_rng(9).integers(-1, 2, (300, 400, 3))).astype(np.uint8)
    assert np.array_equal(develop_mod._sky_saturate(flat, 1.0), flat)


def test_both_develop_surfaces_default_the_saturation_off_and_bite_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_S4.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(sky_saturate=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, sky_saturate=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, sky_saturate=1.0).rgb, base)


def test_the_saturation_tunable_participates_in_the_algo_version(monkeypatch):
    """A new pixel-moving stage that is not hashed makes the provenance stamp lie."""
    assert develop_mod._sky_saturate in develop_mod._HASHED_SOURCES
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "SKY_SATURATE_MAX", 0.9)
    assert develop_mod._develop_algo_version() != baseline


def test_out_of_range_sky_saturate_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_S5.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on bad sky-saturate"))
    args = _build_parser().parse_args(f"photos develop {photo} --sky-saturate 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--sky-saturate must be in [0, 1]" in capsys.readouterr().err
