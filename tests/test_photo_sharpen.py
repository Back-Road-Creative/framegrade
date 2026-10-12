"""Operator sharpen scale: ``--sharpen`` / ``photos.sharpen`` on ``photos develop``.

The unsharp amount stays MEASURED (the detail ramp); the knob only SCALES it,
so 1.0 — the default — is bit-identical to the automatic-only chain, 0.0 turns
sharpening off outright, and 2.0 doubles it under a hard halo ceiling.
"""

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.photos import develop as dev


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)


def _use_photos_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def _write_photo(path):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _soft(amp=24, size=(200, 260)):
    """A softly-textured frame whose measured detail lands mid-ramp (amount > 0)."""
    h, w = size
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    frame = np.clip(128 + amp * np.sin(xx / 4.0) * np.cos(yy / 4.0), 0, 255).astype(np.uint8)
    return np.repeat(frame[:, :, np.newaxis], 3, axis=2)


def _high_frequency(rgb):
    """Std of the fine high-pass — how much detail energy a frame carries."""
    a = dev._luma(rgb)
    return float(np.std(a - cv2.GaussianBlur(a, (0, 0), 1.2)))


# ── the scale is a multiplier on the MEASURED amount ─────────────────


def test_scale_one_is_the_measured_amount_untouched():
    """The load-bearing guarantee: 1.0 renders exactly the pre-knob output."""
    rgb = _soft()
    measured = dev._sharpen_amount(
        dev._detail_score(dev._bounded_view(rgb, dev.SHARPEN_DETAIL_MAX_DIM))
    )
    assert measured > 0.0, "fixture must land mid-ramp or the test proves nothing"
    default_out, default_amount = dev._sharpen(rgb)
    scaled_out, scaled_amount = dev._sharpen(rgb, 1.0)
    assert scaled_amount == default_amount == measured
    assert np.array_equal(scaled_out, default_out)


def test_scale_zero_disables_sharpening_entirely():
    rgb = _soft()
    out, amount = dev._sharpen(rgb, 0.0)
    assert amount == 0.0
    assert np.array_equal(out, rgb), "0.0 must return the input bytes untouched"


def test_scale_two_doubles_the_measured_amount_and_the_detail_response():
    rgb = _soft()
    once, amount_1 = dev._sharpen(rgb, 1.0)
    twice, amount_2 = dev._sharpen(rgb, 2.0)
    assert amount_2 == pytest.approx(min(amount_1 * 2.0, dev.SHARPEN_MAX_SCALED_AMOUNT))
    assert _high_frequency(twice) > _high_frequency(once) * 1.05


def test_an_amplified_soft_frame_is_clamped_to_the_halo_ceiling():
    """A big scale can't run the unsharp amount away from the calibrated range."""
    _, amount = dev._sharpen(_soft(), 2.0)
    assert amount <= dev.SHARPEN_MAX_SCALED_AMOUNT
    assert dev.SHARPEN_MAX_SCALED_AMOUNT == pytest.approx(dev.SHARPEN_MAX_AMOUNT * 2.0)


def test_the_ceiling_is_a_tunable_that_self_invalidates_the_cache(monkeypatch):
    baseline = dev._develop_algo_version()
    monkeypatch.setattr(dev, "SHARPEN_MAX_SCALED_AMOUNT", dev.SHARPEN_MAX_SCALED_AMOUNT + 0.5)
    assert dev._develop_algo_version() != baseline


# ── end to end through develop / develop_variants ────────────────────


def test_develop_variants_default_is_byte_identical_to_the_pre_knob_chain(tmp_path):
    photo = _write_photo(tmp_path / "IMG_S1.jpg")
    plain = dict(dev.develop_variants(photo, styles=["natural"]))["natural"].rgb
    unity = dev.DevelopOptions(sharpen=1.0)
    one = dict(dev.develop_variants(photo, ["natural"], options=unity))["natural"].rgb
    assert np.array_equal(plain, one)


def _record_scales(monkeypatch):
    """Capture the scale every ``_sharpen`` call site is handed."""
    from pycube_lut import Cube

    seen = []
    real = dev._sharpen
    monkeypatch.setattr(
        dev, "_sharpen", lambda rgb, scale=1.0: (seen.append(scale), real(rgb, scale))[1]
    )
    black = Cube(
        2, np.zeros((2, 2, 2, 3), np.float32), np.zeros(3, np.float32), np.ones(3, np.float32)
    )
    return seen, black


def test_the_scale_reaches_both_the_styled_and_the_lut_look_surfaces(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_S2.jpg")
    seen, cube = _record_scales(monkeypatch)
    options = dev.DevelopOptions(sharpen=0.4)
    list(dev.develop_variants(photo, ["natural"], luts=[("look", cube)], options=options))
    assert seen == [0.4, 0.4], "styled AND LUT-look surfaces must both take the knob"


def test_the_scale_reaches_the_single_photo_develop_path(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_S3.jpg")
    seen, _ = _record_scales(monkeypatch)
    dev.develop(photo, sharpen=0.4)
    assert seen == [0.4]


# ── the upstream gates ───────────────────────────────────────────────


def test_sharpen_flag_threads_through_the_handler(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_S4.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    seen = []
    real = dev.develop_variants
    monkeypatch.setattr(
        dev,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural --sharpen 0.5") == 0
    assert [o.sharpen for o in seen] == [0.5]


def test_sharpen_config_default_applies_when_the_flag_is_absent(tmp_path, monkeypatch):
    photo = _write_photo(tmp_path / "IMG_S5.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"), sharpen=1.4)
    seen = []
    real = dev.develop_variants
    monkeypatch.setattr(
        dev,
        "develop_variants",
        lambda *a, **k: (seen.append(a[4]), real(*a, **k))[1],  # a[4] = the options record
    )
    assert _run(f"photos develop {photo} --styles natural") == 0
    assert [o.sharpen for o in seen] == [1.4]


def test_out_of_range_sharpen_is_rejected_before_any_decode(tmp_path, monkeypatch, capsys):
    photo = _write_photo(tmp_path / "IMG_S6.jpg")
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(dev, "decode", lambda p: pytest.fail("decoded despite a bad sharpen"))
    assert _run(f"photos develop {photo} --sharpen 2.5") == 1
    assert "--sharpen must be in [0, 2]" in capsys.readouterr().err


def test_config_default_is_one_and_out_of_range_values_are_rejected():
    assert PhotosConfig().sharpen == 1.0
    assert PhotosConfig(sharpen=0.0).sharpen == 0.0
    with pytest.raises(ValidationError):
        PhotosConfig(sharpen=2.5)
    with pytest.raises(ValidationError):
        PhotosConfig(sharpen=-0.1)
