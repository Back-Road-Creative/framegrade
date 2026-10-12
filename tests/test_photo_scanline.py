"""Spatial stage: CRT scanlines (a row-periodic luma modulation).

The frame is darkened along a SOFT SINE profile that repeats down the rows -- a
line structure, not a comb of hard on/off rows, because hard rows alias into moire
the moment the export is resized. Two invariants carry the design: amount 0 is a
byte-identical no-op, and the line PERIOD is anchored to ``SCANLINE_REFERENCE_DIM``
so one ``--scanline`` value names one line DENSITY at every export size -- the
lesson ``GRAIN_REFERENCE_DIM`` learned the hard way (a 5600 px export carried a
sixth of the grain a 900 px one did from the same flag).
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
    """A perfectly flat 3:2 landscape frame — every row mean starts identical."""
    return np.full((round(long_edge * 2 / 3), long_edge, 3), value, dtype=np.uint8)


def _row_means(rgb):
    return rgb.astype(np.float64).mean(axis=(1, 2))


def _cycles(means):
    """Line cycles across the frame: zero crossings of the mean-centred row signal
    over two. Measured off PIXELS, never off the period the stage computed."""
    centred = means - means.mean()
    return float(np.count_nonzero(np.diff(np.signbit(centred))) / 2.0)


def _expected_period(long_edge):
    return develop_mod.SCANLINE_PERIOD * long_edge / develop_mod.SCANLINE_REFERENCE_DIM


def test_scanline_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._scanline(rgb, 0.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_scanline_row_means_oscillate_at_the_reference_anchored_period():
    """On a flat frame the rows carry the line structure and nothing else: the
    signal repeats every ``period`` rows and troughs half a period in."""
    long_edge = develop_mod.SCANLINE_REFERENCE_DIM * 2
    period = int(_expected_period(long_edge))
    assert period == _expected_period(long_edge) >= 4, "pick constants this test can index"
    means = _row_means(develop_mod._scanline(_flat(long_edge), 1.0))
    assert np.allclose(means[: -2 * period], means[period:-period]), "must repeat every period"
    assert means[period // 2] < means[0], "the trough sits half a period from the crest"
    assert _cycles(means) == pytest.approx(len(means) / period, rel=0.05)


def test_scanline_profile_is_soft_not_two_hard_levels():
    """The whole design: a sine, so most rows land BETWEEN the crest and the
    trough. A hard on/off comb would put every row at one of two levels — and
    alias into moire on the site's downscale."""
    means = _row_means(develop_mod._scanline(_flat(develop_mod.SCANLINE_REFERENCE_DIM * 2), 1.0))
    lo, hi = means.min(), means.max()
    assert hi - lo > 5.0, "the lines must actually be visible"
    span = hi - lo
    intermediate = np.count_nonzero((means > lo + 0.1 * span) & (means < hi - 0.1 * span))
    assert intermediate > 0.5 * len(means), "a soft profile leaves most rows mid-ramp"
    assert len(np.unique(np.round(means, 3))) > 2, "two levels is the hard comb, not a sine"


def test_scanline_density_survives_a_larger_export():
    """The scale-invariance property, pre-applied: one flag value is one LOOK at any export
    size. A fixed pixel period would double the line count when the master doubles,
    so the flag would name a different look for every export."""
    small = _cycles(
        _row_means(develop_mod._scanline(_flat(develop_mod.SCANLINE_REFERENCE_DIM), 1.0))
    )
    large = _cycles(
        _row_means(develop_mod._scanline(_flat(develop_mod.SCANLINE_REFERENCE_DIM * 2), 1.0))
    )
    assert small > 10, "the small export must carry real lines to compare against"
    assert large == pytest.approx(small, rel=0.05), (
        f"line density must survive the export size: {large} vs {small} cycles"
    )


def test_scanline_leaves_the_column_structure_untouched():
    """Row-only by construction: every pixel in a row takes the same gain, so a
    flat frame's columns stay identical to each other."""
    out = develop_mod._scanline(_flat(develop_mod.SCANLINE_REFERENCE_DIM), 0.8)
    assert np.array_equal(out, np.broadcast_to(out[:, :1], out.shape))


def test_scanline_is_deterministic_and_stays_in_range():
    """Pure function of the row index — no rng at all, so two renders match and
    nothing can leave the dtype's range or land on a NaN."""
    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    first = develop_mod._scanline(rgb, 0.7)
    second = develop_mod._scanline(rgb, 0.7)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert first.min() >= 0 and first.max() <= 255
    assert not np.isnan(first).any()


def test_scanline_tunables_participate_in_the_algo_version(monkeypatch):
    """The depth and the period anchor are ALL_CAPS tunables, so a retune
    self-invalidates the reserved develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "SCANLINE_DEPTH", develop_mod.SCANLINE_DEPTH / 2)
    assert develop_mod._develop_algo_version() != baseline


# ── end to end through the develop surface ───────────────────────────


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_develop_variants_default_is_byte_identical_to_no_scanline(tmp_path):
    photo = _write_photo(tmp_path / "IMG_S1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(scanline=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)
    on = develop_mod.DevelopOptions(scanline=0.9)
    lined = dict(develop_mod.develop_variants(photo, ["natural"], options=on))["natural"].rgb
    assert not np.array_equal(plain, lined)


def test_scanline_runs_under_the_grain_and_over_the_halation(tmp_path, monkeypatch):
    """Position: the tube's line structure sits on the finished emulsion — after
    the halation bleed (and after any chromatic-aberration stage), under the grain."""
    order = []
    for stage in ("_vignette", "_halation", "_scanline", "_grain"):
        # ``*_t`` swallows the tint ``_halation`` takes as its third argument.
        monkeypatch.setattr(
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_S2.jpg")
    options = develop_mod.DevelopOptions(vignette=0.4, halation=0.6, scanline=0.5, grain=0.3)
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == ["_vignette", "_halation", "_scanline", "_grain"]


def test_develop_single_photo_path_is_off_by_default_and_lines_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_S3.jpg")
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, scanline=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, scanline=0.9).rgb, base)


def test_out_of_range_scanline_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_S4.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on bad scanline"))
    args = _build_parser().parse_args(f"photos develop {photo} --scanline 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--scanline must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_config_validator_rejects_an_out_of_range_scanline(bad):
    with pytest.raises(ValueError):
        PhotosConfig(scanline=bad)
