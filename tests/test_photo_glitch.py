"""Spatial stage: glitch (seeded horizontal band tearing).

The frame is cut into a grid of horizontal BANDS, a subset of them is displaced
sideways, and each torn band splits its channels -- the digital-corruption read, a
fault in the SIGNAL rather than in the optics. Three invariants carry the design:
amount 0 is a byte-identical no-op; the pattern comes from a FIXED seed (never
entropy), so two develops of one frame match byte for byte; and the band grid is
anchored to ``GLITCH_REFERENCE_DIM``, so one ``--glitch`` value names one band COUNT
and one frame-relative tear at every export size -- ``GRAIN_REFERENCE_DIM``'s lesson.
"""

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_mod


def _marker(long_edge):
    """A black 3:2 frame, ONE white column dead centre: a row's shift is where it lands."""
    frame = np.zeros((round(long_edge * 2 / 3), long_edge, 3), dtype=np.uint8)
    frame[:, long_edge // 2, :] = 255
    return frame


def _row_shifts(out, plane=1):
    """Per-row shift in px off the marker column; plane 1 (green) is the band's own."""
    return out[:, :, plane].argmax(axis=1) - out.shape[1] // 2


def _expected_band_count(long_edge):
    """The band grid the constants describe — never read off the stage's own output."""
    height = round(long_edge * 2 / 3)
    band = max(1.0, develop_mod.GLITCH_BAND_HEIGHT * long_edge / develop_mod.GLITCH_REFERENCE_DIM)
    return max(1, round(height / band))


def _band_shifts(out, count):
    """The shift at each band's centre row, so two export sizes compare band for band."""
    return _row_shifts(out)[(((np.arange(count) + 0.5) * out.shape[0]) / count).astype(int)]


def test_glitch_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._glitch(rgb, 0.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_glitch_moves_columns_and_never_rows():
    """Horizontal-only by construction: a frame constant along x cannot change under a
    sideways gather, so any vertical motion or row reorder (the row codes are unique)
    shows up here — while a frame that VARIES along x must move."""
    height, width = 240, 360
    codes = np.arange(height, dtype=np.uint8)[:, np.newaxis, np.newaxis]
    flat_rows = np.broadcast_to(codes, (height, width, 3)).copy()
    assert np.array_equal(develop_mod._glitch(flat_rows, 1.0), flat_rows)
    ramp = np.arange(width, dtype=np.uint8)[np.newaxis, :, np.newaxis]
    columns = np.broadcast_to(ramp, (height, width, 3)).copy()
    torn = develop_mod._glitch(columns, 1.0)
    assert torn.shape == columns.shape
    assert not np.array_equal(torn, columns)


def test_glitch_displaces_a_few_discrete_bands_not_every_row_on_its_own():
    """Bands, not per-row noise: at most one distinct shift per band of the grid."""
    shifts = _row_shifts(develop_mod._glitch(_marker(develop_mod.GLITCH_REFERENCE_DIM), 1.0))
    runs = shifts[np.concatenate(([True], np.diff(shifts) != 0))]
    assert len(runs) <= _expected_band_count(develop_mod.GLITCH_REFERENCE_DIM)
    assert len(runs) < len(shifts) / 10, "one shift per row is noise, not a torn band"
    assert np.count_nonzero(shifts) > len(shifts) / 10, "the frame must actually tear"


def test_a_torn_band_separates_the_three_channels():
    """The per-band channel offset: R, G and B land at three different places."""
    out = develop_mod._glitch(_marker(develop_mod.GLITCH_REFERENCE_DIM), 1.0)
    red, green, blue = (_row_shifts(out, plane) for plane in (0, 1, 2))
    torn = np.flatnonzero(green)
    assert torn.size, "no band tore, so this proves nothing"
    row = int(torn[0])
    assert len({red[row], green[row], blue[row]}) == 3


def test_band_count_and_relative_tear_survive_a_larger_export():
    """A band height (or shift) in raw px halves the tear when the master doubles."""
    small_edge = develop_mod.GLITCH_REFERENCE_DIM
    large_edge = small_edge * 2
    count = _expected_band_count(small_edge)
    assert count == _expected_band_count(large_edge) > 8, "the grid must be size-invariant"
    small = _band_shifts(develop_mod._glitch(_marker(small_edge), 1.0), count) / small_edge
    large = _band_shifts(develop_mod._glitch(_marker(large_edge), 1.0), count) / large_edge
    assert np.count_nonzero(small) > 2, "the small export must tear, else this proves nothing"
    assert np.count_nonzero(small) == np.count_nonzero(large), "same torn-band COUNT"
    assert np.allclose(small, large, atol=1.0 / small_edge), "same frame-relative tear"


def test_glitch_is_deterministic_and_stays_in_range():
    """Seeded rng, never entropy; and a pure GATHER cannot leave the dtype or go NaN."""
    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, (240, 360, 3), dtype=np.uint8)
    first = develop_mod._glitch(rgb, 0.7)
    assert np.array_equal(first, develop_mod._glitch(rgb, 0.7))
    assert first.dtype == np.uint8
    assert first.min() >= 0 and first.max() <= 255
    assert not np.isnan(first).any()


def test_glitch_tunables_participate_in_the_algo_version(monkeypatch):
    """ALL_CAPS tunables, so a retune or a reseed self-invalidates the develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "GLITCH_BAND_HEIGHT", develop_mod.GLITCH_BAND_HEIGHT / 2)
    assert develop_mod._develop_algo_version() != baseline


# ── end to end through the develop surface ───────────────────────────


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_develop_variants_default_is_byte_identical_to_no_glitch(tmp_path):
    photo = _write_photo(tmp_path / "IMG_G1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(glitch=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)
    on = develop_mod.DevelopOptions(glitch=0.9)
    torn = dict(develop_mod.develop_variants(photo, ["natural"], options=on))["natural"].rgb
    assert not np.array_equal(plain, torn)


def test_glitch_runs_over_the_scanlines_and_under_the_grain(tmp_path, monkeypatch):
    """The tear is a fault in the SIGNAL, so it sits over the tube's line structure."""
    order = []
    for stage in ("_halation", "_scanline", "_glitch", "_grain"):
        # ``*_t`` swallows the tint ``_halation`` takes as its third argument.
        monkeypatch.setattr(
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_G2.jpg")
    options = develop_mod.DevelopOptions(halation=0.6, scanline=0.5, glitch=0.7, grain=0.3)
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == ["_halation", "_scanline", "_glitch", "_grain"]


def test_develop_single_photo_path_is_off_by_default_and_tears_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_G3.jpg")
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, glitch=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, glitch=0.9).rgb, base)


def test_out_of_range_glitch_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_G4.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on bad glitch"))
    args = _build_parser().parse_args(f"photos develop {photo} --glitch 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--glitch must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_config_validator_rejects_an_out_of_range_glitch(bad):
    with pytest.raises(ValueError):
        PhotosConfig(glitch=bad)
