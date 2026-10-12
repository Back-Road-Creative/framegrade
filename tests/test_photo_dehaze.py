"""Phase-3 spatial stage: real dehaze via the dark-channel prior.

The pre-existing ``color_correction`` "dehaze" is a bounded S-curve increment — it adds
contrast, it never lifts a veil. This does the physics: estimate the atmospheric light,
estimate transmission, invert ``I = J·t + A·(1−t)``. Every quantitative test runs on a
REAL-RANGE frame (max luma ≈ 0.66) — the only tone this grade chain produces, and three
features here shipped green and inert after calibration on full-range synthetics.
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


def _clear_scene(size=360):
    """Sky, slope, a shaded mid-ground ridge, a deep-shadow NEAR foreground; peak 0.66."""
    rng = np.random.default_rng(11)
    horizon = int(size * 0.45)
    rgb = np.zeros((size, size, 3), np.float32)
    sky = np.linspace(0.66, 0.42, horizon, dtype=np.float32)[:, None]
    rgb[:horizon] = np.dstack([sky * 0.82, sky * 0.90, sky])
    yy, xx = np.meshgrid(np.arange(size - horizon), np.arange(size), indexing="ij")
    tex = 0.06 * np.sin(xx / 9.0) * np.cos(yy / 7.0) + 0.03 * rng.standard_normal(yy.shape)
    g = np.clip(0.20 + 0.22 * yy / yy.max() + tex, 0.02, 0.66).astype(np.float32)
    rgb[horizon:] = np.dstack([g, g * 0.88, g * 0.66])
    rgb[horizon : horizon + 30] *= 0.55  # a shaded ridge, out in the veil
    rgb[int(size * 0.86) :] *= 0.13  # the near foreground, in deep shadow
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _veiled(clear, far=0.30, air=(0.60, 0.63, 0.66)):
    """A KNOWN veil, ``I = J·t + A·(1−t)``, ``t`` ramping ``far`` (top) to 1.0 (near):
    haze is depth-dependent; a uniform slab would be a black-lift."""
    src = clear.astype(np.float32) / 255.0
    t = np.linspace(far, 1.0, clear.shape[0], dtype=np.float32)[:, None, None]
    out = src * t + np.asarray(air, np.float32) * (1.0 - t)
    return np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)


def _tone(rgb):
    return develop_mod._luma(rgb) / 255.0


def _local_contrast(rgb, radius=6.0):  # std of the high-pass: what a veil scales by t
    a = _tone(rgb)
    return float(np.std(a - cv2.GaussianBlur(a, (0, 0), radius)))


def _chroma(rgb):
    a = rgb.astype(np.float64)
    return float((a.max(axis=2) - a.min(axis=2)).mean())


def _near_black(rgb):
    return float((_tone(rgb) < 16.0 / 255.0).mean())


def test_zero_amount_is_byte_identical():
    """The load-bearing no-op: amount 0 returns the exact input bytes."""
    rgb = np.random.default_rng(5).integers(0, 256, (40, 50, 3), dtype=np.uint8)
    out = develop_mod._dehaze(rgb, 0.0)
    assert out.dtype == np.uint8 and np.array_equal(out, rgb)


@pytest.fixture(scope="module")
def scene():
    """The truth, a known veil composited over it, and the full-amount recovery."""
    clear = _clear_scene()
    hazy = _veiled(clear)
    return clear, hazy, develop_mod._dehaze(hazy, 1.0)


def test_a_known_veil_is_subtracted_off_a_real_range_frame(scene):
    """Recover TOWARD the truth, not merely add contrast: error down, structure and
    chroma back up (a veil scales both by ``t``) — and never past the truth."""
    clear, hazy, out = scene
    assert _tone(hazy).max() < 0.67  # the tone a real develop of this pipeline lands in
    before = float(np.abs(hazy.astype(np.float64) - clear).mean())
    assert before > 15.0, before  # the veil really is heavy
    assert float(np.abs(out.astype(np.float64) - clear).mean()) < before * 0.6
    assert _local_contrast(hazy) * 1.2 < _local_contrast(out) <= _local_contrast(clear) * 1.1
    assert _chroma(hazy) * 1.25 < _chroma(out) <= _chroma(clear) * 1.1
    # ... and it is monotone in the amount, 20+ levels end to end -- never a nudge.
    means = [float(_tone(develop_mod._dehaze(hazy, a)).mean()) for a in (0.0, 0.5, 0.75, 1.0)]
    assert means == sorted(means, reverse=True), means  # more veil off => darker
    assert (means[0] - means[-1]) * 255.0 > 20.0, means


def test_shadows_are_not_crushed_past_the_truth(scene):
    """The bound is the truth: never manufacture black that was never in the scene."""
    clear, hazy, out = scene
    assert _near_black(clear) > 0.10  # the truth genuinely HAS deep shadows
    assert _near_black(hazy) < _near_black(clear) / 2  # the veil lifted them off black
    assert _near_black(out) <= _near_black(clear), (_near_black(out), _near_black(clear))
    assert int(out.min()) > 0  # the non-negativity guard held through the upsample


def test_the_transmission_estimate_is_resolution_independent():
    """Metered on a bounded view with a FRACTIONAL patch, so 2× size looks the same."""
    hazy = _veiled(_clear_scene())
    big = cv2.resize(hazy, (720, 720), interpolation=cv2.INTER_LINEAR)
    small_out = develop_mod._dehaze(hazy, 1.0)
    big_out = cv2.resize(develop_mod._dehaze(big, 1.0), (360, 360), interpolation=cv2.INTER_AREA)
    assert abs(_tone(small_out).mean() - _tone(big_out).mean()) * 255.0 < 4.0
    assert float(np.abs(small_out.astype(np.float64) - big_out).mean()) < 6.0


def test_the_tunables_participate_in_the_algo_version(monkeypatch):
    """ALL_CAPS knobs, so a retune self-invalidates the reserved develop cache."""
    baseline = develop_mod._develop_algo_version()
    monkeypatch.setattr(develop_mod, "DEHAZE_OMEGA_MAX", 0.5)
    assert develop_mod._develop_algo_version() != baseline


def _write_photo(path, size=(80, 100)):
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def test_both_develop_surfaces_default_off_and_bite_when_asked(tmp_path):
    photo = _write_photo(tmp_path / "IMG_D1.jpg")
    plain = dict(develop_mod.develop_variants(photo, styles=["natural"]))["natural"].rgb
    off = develop_mod.DevelopOptions(dehaze=0.0)
    zero = dict(develop_mod.develop_variants(photo, ["natural"], options=off))["natural"].rgb
    assert np.array_equal(plain, zero)
    base = develop_mod.develop(photo).rgb
    assert np.array_equal(develop_mod.develop(photo, dehaze=0.0).rgb, base)
    assert not np.array_equal(develop_mod.develop(photo, dehaze=1.0).rgb, base)


def test_dehaze_runs_first_of_the_spatial_stages(tmp_path, monkeypatch):
    """Veil removal is a CORRECTION: the later stages work on the corrected frame."""
    order = []
    for stage in ("_dehaze", "_clarity", "_vignette", "_halation", "_grain"):
        monkeypatch.setattr(  # ``*_t``: ``_halation`` takes a third (tint) argument
            develop_mod, stage, lambda rgb, _a, *_t, s=stage: (order.append(s), rgb)[1]
        )
    photo = _write_photo(tmp_path / "IMG_D2.jpg")
    options = develop_mod.DevelopOptions(dehaze=0.5, clarity=0.4, vignette=0.3, halation=0.2)
    list(develop_mod.develop_variants(photo, ["natural"], options=options))
    assert order == ["_dehaze", "_clarity", "_vignette", "_halation", "_grain"]


def test_out_of_range_dehaze_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    photo = _write_photo(tmp_path / "IMG_D4.jpg")
    photos = PhotosConfig(developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))
    monkeypatch.setattr(develop_mod, "decode", lambda p: pytest.fail("decoded on bad dehaze"))
    args = _build_parser().parse_args(f"photos develop {photo} --dehaze 1.5".split())
    assert photos_handler.cmd_photos(args, None) == 1
    assert "--dehaze must be in [0, 1]" in capsys.readouterr().err


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_the_config_validator_rejects_an_out_of_range_dehaze(bad):
    with pytest.raises(ValueError):
        PhotosConfig(dehaze=bad)
