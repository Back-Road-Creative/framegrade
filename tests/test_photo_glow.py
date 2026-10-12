"""Spatial stage: the Orton glow (a wide, screen-blended soft diffusion).

The frame's own brighter tones are blurred at a wide, width-anchored radius and
SCREENED back over the sharp original -- the darkroom sandwich print. One of the two
ways this chain can soften a frame: this ADDS diffused light, where a negative
``_clarity`` flattens the frame's own broad structure instead.

Four invariants carry the design: amount 0 is a byte-identical no-op, screen cannot
overshoot (an already-maxed pixel stays put), the bloom radius is anchored to
``SHARPEN_REFERENCE_WIDTH`` so one flag value is one look at every export size, and
the tone the diffusion starts at is MEASURED off the frame rather than an absolute
constant. That last one is the ``_halation_seed`` lesson -- this chain's output tops
out near 0.66 luma and never clips, so an absolute threshold ships GREEN AND INERT --
which is why the fixtures here deliberately stay well under the clip point.
"""

import numpy as np
import pytest
from PIL import Image

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as develop_mod


def _textured(height=200, width=300, low=36, high=168):
    """High-frequency detail over a ramp peaking at 168/255 = 0.66 luma -- the
    range a real develop off this chain actually lands in, nowhere near clipping."""
    rng = np.random.default_rng(3)
    ramp = np.linspace(low, high, width, dtype=np.float64)[np.newaxis, :]
    plane = np.clip(ramp + rng.normal(0.0, 14.0, (height, width)), 0.0, float(high))
    return np.repeat(plane[:, :, np.newaxis], 3, axis=2).astype(np.uint8)


def _detail(rgb):
    """High-frequency energy: mean absolute horizontal neighbour difference."""
    return float(np.abs(np.diff(rgb.astype(np.float64).mean(axis=2), axis=1)).mean())


def _lift_profile(before, after, bins=12):
    """Where the bloom landed, in FRACTIONAL-width buckets -- scale-free."""
    lift = (after.astype(np.float64) - before.astype(np.float64)).mean(axis=(0, 2))
    return np.array([bucket.mean() for bucket in np.array_split(lift, bins)])


def test_glow_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._glow(rgb, 0.0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, rgb)


def test_glow_softens_the_frame_and_lifts_it_slightly():
    """The whole point: high-frequency energy falls, brightness rises a little."""
    frame = _textured()
    out = develop_mod._glow(frame, 1.0)
    assert _detail(out) < _detail(frame), "the frame must soften, not sharpen"
    assert out.astype(np.float64).mean() > frame.astype(np.float64).mean()


def test_screen_cannot_overshoot_so_a_maxed_pixel_stays_maxed():
    """SCREEN, not add: ``1-(1-a)(1-b)`` cannot exceed full scale."""
    frame = np.full((80, 120, 3), 60, dtype=np.uint8)
    frame[30:50, 45:75] = 255
    out = develop_mod._glow(frame, 1.0)
    assert out.max() <= 255
    assert out[30:50, 45:75].min() == 255, "a maxed pixel must stay exactly maxed"
    assert out[:10, :10].mean() > 60.0, "its surroundings must take the bloom"


def test_glow_bites_on_this_chains_sub_clipping_tone():
    """The ``_halation_seed`` lesson pre-applied, and the single most likely way this
    stage ships broken-but-green: an ABSOLUTE seed touches nothing here."""
    frame = _textured()
    assert frame.max() < 0.7 * 255, "the fixture must stay well under the clip point"
    out = develop_mod._glow(frame, 1.0)
    assert np.mean(out != frame) > 0.5, "most of the frame must take the bloom"
    assert out.astype(np.float64).mean() > frame.astype(np.float64).mean() + 2.0


def test_glow_carries_the_same_bloom_at_twice_the_export_size(monkeypatch):
    """The scale-invariance property: the radius is a fraction of the frame, so one value is one
    look at every export size. Shrunk here only to keep a 4K-wide blur cheap -- the
    SCALING LAW is under test, not the constant."""
    monkeypatch.setattr(develop_mod, "GLOW_RADIUS", 12.0)
    small = _textured(height=48, width=int(develop_mod.SHARPEN_REFERENCE_WIDTH))
    large = np.repeat(np.repeat(small, 2, axis=0), 2, axis=1)
    at_1x = _lift_profile(small, develop_mod._glow(small, 1.0))
    at_2x = _lift_profile(large, develop_mod._glow(large, 1.0))
    assert at_1x.max() - at_1x.min() > 2.0, "the bloom must vary, else this proves nothing"
    assert np.allclose(at_1x, at_2x, atol=1.5), f"bloom moved with the export: {at_1x} vs {at_2x}"


def test_glow_is_deterministic_and_stays_in_range():
    """No rng -- a pure function of the frame -- so two renders match and nothing
    can leave the dtype's range or land on a NaN."""
    frame = _textured(height=120, width=160)
    first = develop_mod._glow(frame, 0.7)
    second = develop_mod._glow(frame, 0.7)
    assert np.array_equal(first, second)
    assert first.dtype == np.uint8
    assert first.min() >= 0 and first.max() <= 255
    assert not np.isnan(first).any()


def test_glow_tunables_participate_in_the_algo_version(monkeypatch):
    """ALL_CAPS tunables, so a retune self-invalidates the reserved develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "GLOW_STRENGTH", develop_mod.GLOW_STRENGTH / 2)
    assert develop_mod._develop_algo_version() != baseline


# ── end to end through the develop surface ───────────────────────────


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_both_develop_surfaces_default_off_and_bloom_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_G1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(glow=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)
    on = develop_mod.DevelopOptions(glow=0.9)
    bloomed = dict(develop_mod.develop_variants(photo, ["natural"], options=on))["natural"].rgb
    assert not np.array_equal(plain, bloomed)
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, glow=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, glow=0.9).rgb, base)


def test_glow_runs_after_dehaze_and_before_clarity(tmp_path, monkeypatch):
    """Glow then re-sharpen reads as film; the reverse reads as haze. Dehaze stays
    FIRST -- it subtracts the atmospheric veil, so running it after a deliberate
    glow would have it fight the glow."""
    order = []
    for stage in ("_dehaze", "_glow", "_clarity", "_vignette"):
        # ``*_t`` swallows the tint a neighbouring stage takes as a third argument.
        monkeypatch.setattr(
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_G2.jpg")
    options = develop_mod.DevelopOptions(dehaze=0.5, glow=0.6, clarity=0.4, vignette=0.3)
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == ["_dehaze", "_glow", "_clarity", "_vignette"]


def test_out_of_range_glow_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_G3.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on bad glow"))
    args = _build_parser().parse_args(f"photos develop {photo} --glow 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--glow must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_config_validator_rejects_an_out_of_range_glow(bad):
    with pytest.raises(ValueError):
        PhotosConfig(glow=bad)
