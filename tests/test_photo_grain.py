"""Phase-3 spatial stage: luminance film grain (the ADD counterpart to denoise).

Denoise REMOVES sensor grain from the source so grading can't amplify it; grain
ADDS a controlled, filmic texture back onto the finished frame so film-emulation
LUTs don't render sterile-clean. The grain is a single MONOCHROME (luminance)
noise field added equally to R/G/B (colour grain reads as sensor noise), luma-
WEIGHTED so it's strongest in the midtones and rolls off in deep shadows and near-
clipped highlights. Two hard invariants: amount 0 is a byte-identical no-op, and
the grain is DETERMINISTIC (a fixed-seed ``default_rng``) so two develops of one
frame at one amount are byte-identical -- the develop cache/algo-version rely on it.
"""

import cv2
import numpy as np
import pytest

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_mod


def _flat(value, size=(120, 160)):
    """A perfectly flat 3-channel frame at ``value`` (no texture at all)."""
    return np.full((*size, 3), value, dtype=np.uint8)


def _hf_std(rgb, radius=2.0):
    """Std of the high-pass (luma - blur): a flat frame reads ~0, grain lifts it."""
    luma = develop_mod._luma(rgb)
    return float(np.std(luma - cv2.GaussianBlur(luma, (0, 0), radius)))


# ── the grain stage ──────────────────────────────────────────────────


def test_grain_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._grain(rgb, 0.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_grain_is_deterministic_across_two_runs():
    """Determinism is a hard invariant: two develops of one frame at one amount
    must be byte-identical. Perturbing the GLOBAL np.random state between the two
    runs proves the field is drawn from a seeded LOCAL rng, not global entropy."""
    rgb = _flat(128)
    np.random.seed(1)
    first = develop_mod._grain(rgb, 0.5)
    np.random.seed(999)
    second = develop_mod._grain(rgb, 0.5)
    assert np.array_equal(first, second)


def test_grain_adds_high_frequency_variance():
    """amount>0 lays high-frequency noise onto an otherwise flat frame."""
    rgb = _flat(128)
    assert _hf_std(rgb) < 0.1  # the input carries no texture
    grained = develop_mod._grain(rgb, 0.6)
    assert _hf_std(grained) > 2.0, "grain must raise high-frequency variance"


# ── resolution independence: one --grain value, one look, at any export size ──


def _field(long_edge, amount=1.0):
    """The isolated grain field on a flat frame of the given long edge, as float."""
    rgb = _flat(128, size=(round(long_edge * 2 / 3), long_edge))
    return develop_mod._grain(rgb, amount).astype(np.float64).mean(axis=2) - 128.0


def _lag1(field):
    """Lag-1 autocorrelation along a row: ~0 = an independent draw per pixel (a
    particle one pixel wide), rising toward 1 as the particles clump."""
    a, b = field[:, :-1].ravel(), field[:, 1:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def _viewed_sigma(field, view=300):
    """Grain strength once the frame is normalised to a common viewing size — what
    the eye actually gets from a print or a screen, rather than per stored pixel."""
    scale = view / max(field.shape)
    size = (max(1, round(field.shape[1] * scale)), max(1, round(field.shape[0] * scale)))
    return float(cv2.resize(field, size, interpolation=cv2.INTER_AREA).std())


def test_grain_particle_size_scales_with_the_export():
    """A grain particle is a physical speck of silver: on a bigger scan of the same
    negative it covers MORE pixels. Drawn one independent sample per output pixel it
    covers exactly one at every size, which is the signature of sensor noise, not
    film. Past the reference dim the field must be visibly clumped."""
    big = _field(develop_mod.GRAIN_REFERENCE_DIM * 2)
    assert _lag1(big) > 0.25, "grain past the reference dim must clump, not read per-pixel"


def test_grain_strength_survives_a_larger_export():
    """The load-bearing property: ``--grain 1.0`` must mean ONE look, not a look per
    export size. An uncorrelated per-pixel field averages away as the frame grows —
    the same photo at 4x the long edge came back with a quarter of the grain — so
    the flag silently meant something different for every output size."""
    small = _viewed_sigma(_field(develop_mod.GRAIN_REFERENCE_DIM))
    large = _viewed_sigma(_field(develop_mod.GRAIN_REFERENCE_DIM * 2))
    assert small > 1.0  # the small export carries real grain to compare against
    assert 0.75 < large / small < 1.25, (
        f"grain strength must survive the export size: {large:.2f} vs {small:.2f} at a "
        "common viewing size"
    )


def test_grain_at_or_under_the_reference_dim_keeps_the_per_pixel_draw():
    """Below the reference dim there is nothing to scale — the field is drawn at the
    frame's own size exactly as before, so small frames (every other test here, and
    the contact-sheet panels) render the bytes they always did."""
    field = _field(develop_mod.GRAIN_REFERENCE_DIM // 2)
    assert abs(_lag1(field)) < 0.1, "small frames must keep the cheap per-pixel field"


def test_grain_stays_deterministic_on_a_scaled_export():
    """Determinism is a hard invariant and the resize must not break it."""
    first = _field(develop_mod.GRAIN_REFERENCE_DIM * 2)
    np.random.seed(7)
    second = _field(develop_mod.GRAIN_REFERENCE_DIM * 2)
    assert np.array_equal(first, second)


def test_grain_is_monochrome_luminance_noise():
    """One noise field added equally to R/G/B -- the per-pixel delta is identical
    across channels (luminance grain), not independent per channel (sensor noise)."""
    rgb = _flat(128)
    delta = develop_mod._grain(rgb, 0.6).astype(np.int16) - rgb.astype(np.int16)
    assert np.array_equal(delta[:, :, 0], delta[:, :, 1])
    assert np.array_equal(delta[:, :, 1], delta[:, :, 2])
    assert np.any(delta != 0)  # and it actually moved pixels


def test_grain_weight_favours_the_midtones():
    """The luma weight peaks in the midtones and falls off toward both clipping ends."""
    mid = float(np.mean(develop_mod._grain_weight(_flat(128))))
    shadow = float(np.mean(develop_mod._grain_weight(_flat(8))))
    highlight = float(np.mean(develop_mod._grain_weight(_flat(248))))
    assert mid > shadow * 3.0, "midtones must carry more grain than deep shadow"
    assert mid > highlight * 3.0, "midtones must carry more grain than near-clip highlight"


def test_grain_midtones_get_more_grain_than_near_black_or_near_white():
    """End to end: the same amount lays visibly more grain on a midtone patch than
    on a near-black or near-white one (the luma weighting, not just clipping)."""

    def spread(value):
        rgb = _flat(value)
        out = develop_mod._grain(rgb, 1.0)
        return float(np.std(out.astype(np.int16) - rgb.astype(np.int16)))

    mid = spread(128)
    assert mid > spread(5) * 3.0
    assert mid > spread(250) * 3.0


def test_grain_tunables_participate_in_the_algo_version(monkeypatch):
    """The sigma scale and the fixed seed are ALL_CAPS tunables, so a retune (or a
    reseed) self-invalidates the reserved develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "GRAIN_SIGMA_SCALE", develop_mod.GRAIN_SIGMA_SCALE + 4.0)
    assert develop_mod._develop_algo_version() != baseline
    monkeypatch.undo()
    monkeypatch.setattr(develop_mod, "GRAIN_SEED", develop_mod.GRAIN_SEED + 1)
    assert develop_mod._develop_algo_version() != baseline


# ── end to end through the develop stage ─────────────────────────────


def _write_photo(path, size=(80, 100)):
    from PIL import Image

    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_develop_variants_default_is_byte_identical_to_no_grain(tmp_path):
    photo = _write_photo(tmp_path / "IMG_A.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(grain=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)


def test_develop_variants_grain_changes_pixels_and_raises_texture(tmp_path):
    photo = _write_photo(tmp_path / "IMG_B.jpg")
    base = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    on = develop_mod.DevelopOptions(grain=0.8)
    grn = dict(develop_mod.develop_variants(photo, ["natural"], options=on))["natural"].rgb
    assert not np.array_equal(base, grn)
    assert _hf_std(grn) > _hf_std(base)


def test_develop_single_photo_grain_is_deterministic(tmp_path):
    photo = _write_photo(tmp_path / "IMG_C.jpg")
    first = develop_mod.develop(photo, grain=0.7).rgb
    second = develop_mod.develop(photo, grain=0.7).rgb
    assert np.array_equal(first, second)


# ── particle SIZE: how COARSE a speck is, independent of how MUCH of it ──


def _sized(long_edge, size, amount=1.0):
    """The isolated grain field at one particle size, as float."""
    rgb = _flat(128, size=(round(long_edge * 2 / 3), long_edge))
    return develop_mod._grain(rgb, amount, size).astype(np.float64).mean(axis=2) - 128.0


def test_grain_size_one_is_the_shipped_field_on_both_draw_paths():
    """1.0 must be byte-for-byte what shipped: the cheap per-pixel draw under the
    reference dim (pinned against the raw seeded field itself, not merely against
    the default call) and the reference-grid resample above it."""
    small, large = (80, 120), (800, 2400)
    seeded = np.random.default_rng(develop_mod.GRAIN_SEED).standard_normal(small)
    assert np.array_equal(develop_mod._grain_field(small, 1.0), seeded.astype(np.float32))
    for shape in (small, large):
        assert np.array_equal(develop_mod._grain_field(shape, 1.0), develop_mod._grain_field(shape))


def test_a_larger_grain_size_draws_coarser_particles():
    """Measured, not merely "different bytes": lag-1 autocorrelation rises as the
    speck widens. At this export 0.5 is finer than the shipped grid (back to one
    sample per pixel) and 2.0 is twice as coarse."""
    fine, shipped, coarse = (_lag1(_sized(1800, s)) for s in (0.5, 1.0, 2.0))
    assert fine < shipped < coarse, f"size must set coarseness: {fine=} {shipped=} {coarse=}"
    assert fine < 0.1, "a fine size draws a per-pixel speck"
    assert coarse > 0.5, "a coarse size must clump visibly"


def test_grain_size_stays_deterministic():
    """The hard invariant survives the knob: perturbing the GLOBAL numpy state
    between two renders at one size changes nothing."""
    first = _sized(1800, 2.0)
    np.random.seed(5)
    assert np.array_equal(first, _sized(1800, 2.0))


@pytest.mark.parametrize("size", [0.25, 1.0, 4.0])
def test_zero_grain_is_a_byte_identical_no_op_at_every_size(size):
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    assert np.array_equal(develop_mod._grain(rgb, 0.0, size), rgb)


def test_grain_size_reaches_both_develop_surfaces(tmp_path):
    photo = _write_photo(tmp_path / "IMG_S.jpg", size=(400, 600))

    def variant(**kw):
        options = develop_mod.DevelopOptions(grain=0.8, **kw)
        return dict(develop_mod.develop_variants(photo, ["natural"], options=options))[
            "natural"
        ].rgb

    assert np.array_equal(variant(), variant(grain_size=1.0))
    assert not np.array_equal(variant(), variant(grain_size=4.0))
    base = develop_mod.develop(photo, grain=0.8).rgb
    assert np.array_equal(base, develop_mod.develop(photo, grain=0.8, grain_size=1.0).rgb)
    assert not np.array_equal(base, develop_mod.develop(photo, grain=0.8, grain_size=4.0).rgb)


def test_out_of_range_grain_size_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_R.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on a bad size"))
    args = _build_parser().parse_args(f"photos develop {photo} --grain-size 8".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--grain-size must be in [0.25, 4]" in capsys.readouterr().err
