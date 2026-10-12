"""Phase-3 spatial stage: the holographic sheen (an iridescent, tone-keyed tint).

A low-amplitude cast whose HUE rotates smoothly across the frame's tonal range, so
shadows, midtones and highlights each pick up a different colour and the print shimmers
rather than shifting as a whole. Three invariants carry the design: amount 0 is a
byte-identical no-op; the cast is a pure function of a pixel's LUMA, so two pixels of
equal tone match wherever they sit; and that tone is measured against the frame's OWN
span, so it bites on a develop peaking near 0.66 luma instead of shipping green and
inert the way an absolute ramp does (the lesson ``_halation_seed`` records)."""

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_mod

#: Ceiling on the per-channel tint, in 8-bit codes, at amount 1.0. A sheen is a WASH:
#: ``--duotone`` and ``--bw-mix`` already exist for wholesale recolouring and rewrite
#: channels by a hundred codes or more. Twelve is under 5% of full scale — an oil-slick
#: shimmer over the print, never a second look competing with the grade.
SHEEN_CEILING = 12.0


def _ramp(lo=16, hi=240, width=256, height=64):
    """A grey column ramp: luma rises left to right, every row identical."""
    row = np.linspace(lo, hi, width).astype(np.uint8)
    return np.repeat(np.tile(row, (height, 1))[:, :, np.newaxis], 3, axis=2)


def _tint(rgb, amount=1.0):
    return develop_mod._holo_sheen(rgb, amount).astype(np.float64) - rgb


def _sweep(rgb, bins=12):
    """Unwrapped hue angle of that tint, sampled across the ramp's tonal range.

    The tint is zero-sum across channels, so it lives in the chroma plane spanned by
    ``a = dr - (dg + db) / 2`` and ``b = (sqrt(3)/2)(dg - db)``; ``atan2(b, a)`` is the
    cast's HUE. Asserting on ANGLES rather than pixel values keeps these tests about the
    rotation, never about one calibration of the amplitude. The outer 5% is skipped —
    ``_holo_tone`` clips there, so those columns carry no rotation."""
    delta = _tint(rgb)
    cols = (np.linspace(0.05, 0.95, bins) * (rgb.shape[1] - 1)).astype(int)
    d = delta[:, cols].mean(axis=0)
    a = d[:, 0] - 0.5 * (d[:, 1] + d[:, 2])
    b = (np.sqrt(3.0) / 2.0) * (d[:, 1] - d[:, 2])
    return np.unwrap(np.arctan2(b, a))


def test_holo_sheen_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._holo_sheen(rgb, 0.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_the_cast_rotates_monotonically_from_the_shadows_to_the_highlights():
    """The whole stage: shadows, midtones and highlights each take a DIFFERENT hue,
    and the hue turns one way the whole distance — no fold back, no plateau."""
    angles = _sweep(_ramp())
    assert np.all(np.diff(angles) > 0.0), f"the rotation must be monotonic, got {angles}"
    full = 2.0 * np.pi * develop_mod.HOLO_TURNS
    assert 0.7 * full < angles[-1] - angles[0] <= full, "the sampled span must sweep the wheel"


def test_the_sweep_is_frame_relative_so_it_bites_on_this_chains_tonal_range():
    """``_halation_seed``'s lesson, pre-applied. This chain peaks near 0.66 luma and
    never clips; an ABSOLUTE ramp would sweep ~two thirds of the wheel on such a frame
    (less on a flatter one), so one flag value would mean a different look on every
    photograph. Frame-relative, the narrow real-range frame sweeps what the wide one does."""
    wide = _sweep(_ramp(16, 240))
    narrow = _sweep(_ramp(20, 168))  # 0.08..0.66 luma — what a real develop lands on
    assert np.all(np.diff(narrow) > 0.0), "the real-range frame must rotate, else this is vacuous"
    assert narrow[-1] - narrow[0] == pytest.approx(wide[-1] - wide[0], rel=0.1)


def test_two_pixels_of_equal_luma_take_the_same_cast_wherever_they_sit():
    """Keyed to TONE, not to position: the top half's ramp and the bottom half's
    mirrored ramp hold the same luma at mirrored columns, so their tints must match."""
    top = _ramp(16, 240, height=32)
    frame = np.ascontiguousarray(np.vstack([top, top[:, ::-1]]))
    delta = _tint(frame)
    assert np.abs(delta).max() > 1.0, "a cast must be present, else matching proves nothing"
    assert np.array_equal(delta[:32], delta[32:][:, ::-1])


def test_the_shift_is_a_bounded_chroma_wash_not_a_duotone_or_an_exposure_move():
    """Amplitude is the difference between a sheen and a recolour. Two bounds: no channel
    moves further than ``SHEEN_CEILING``, and the three offsets SUM to zero (the tint is
    pure chroma), so the frame cannot brighten or darken under it."""
    delta = _tint(_ramp())
    assert np.abs(delta).max() > 1.0, "the sheen must actually be visible"
    assert np.abs(delta).max() <= SHEEN_CEILING
    assert np.abs(delta.sum(axis=2)).max() <= 1.5, "rounding only — no exposure shift"


def test_holo_sheen_is_deterministic_and_stays_in_range():
    """Pure function of the frame's own luma — no rng, no seed — so two renders
    match and nothing can leave the dtype's range or land on a NaN."""
    rng = np.random.default_rng(9)
    rgb = rng.integers(0, 256, (200, 300, 3), dtype=np.uint8)
    first = develop_mod._holo_sheen(rgb, 0.7)
    assert np.array_equal(first, develop_mod._holo_sheen(rgb, 0.7))
    assert first.dtype == np.uint8
    assert first.min() >= 0 and first.max() <= 255
    assert not np.isnan(first).any()


def test_holo_sheen_tunables_participate_in_the_algo_version(monkeypatch):
    """Amplitude and turn count are ALL_CAPS tunables, so a retune self-invalidates
    the reserved develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "HOLO_AMPLITUDE", develop_mod.HOLO_AMPLITUDE / 2)
    assert develop_mod._develop_algo_version() != baseline


# ── end to end through the develop surface ───────────────────────────


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_develop_variants_default_is_byte_identical_to_no_sheen(tmp_path):
    photo = _write_photo(tmp_path / "IMG_H1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(holo_sheen=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)
    on = develop_mod.DevelopOptions(holo_sheen=0.9)
    sheened = dict(develop_mod.develop_variants(photo, ["natural"], options=on))["natural"].rgb
    assert not np.array_equal(plain, sheened)


def test_holo_sheen_runs_over_the_chromatic_ab_and_under_the_scanline(tmp_path, monkeypatch):
    """Position: a surface property of the PRINT — over the lens fringe, under the tube."""
    order = []
    for stage in ("_halation", "_chromatic_ab", "_holo_sheen", "_scanline", "_grain"):
        # ``*_t`` swallows the tint ``_halation`` takes as its third argument.
        monkeypatch.setattr(
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_H2.jpg")
    options = develop_mod.DevelopOptions(
        halation=0.6, chromatic_ab=0.4, holo_sheen=0.5, scanline=0.3, grain=0.2
    )
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == ["_halation", "_chromatic_ab", "_holo_sheen", "_scanline", "_grain"]


def test_develop_single_photo_path_is_off_by_default_and_shimmers_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_H3.jpg")
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, holo_sheen=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, holo_sheen=0.9).rgb, base)


def test_out_of_range_holo_sheen_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_H4.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on bad holo-sheen"))
    args = _build_parser().parse_args(f"photos develop {photo} --holo-sheen 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--holo-sheen must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_config_validator_rejects_an_out_of_range_holo_sheen(bad):
    with pytest.raises(ValueError):
        PhotosConfig(holo_sheen=bad)
