"""Archival 16-bit master export for ``photos develop`` (``--format tiff16``).

The chain has always graded in float32 and quantized to uint8 at the last line
of ``_grade``, so every sub-8-bit code it computed — and every sensor code a
16-bit RAW decode carried in — was discarded at write. ``--format tiff16``
renders the SAME picture with the whole chain held at 16 bits into a 48-bit
sRGB-tagged TIFF; these tests pin that the depth is REAL, never an 8-bit render
poured into a wider container. ``--format`` unset is the untouched JPEG path."""

import numpy as np
import pytest
import tifffile
from PIL import Image
from pydantic import ValidationError

import gomoveshift.config as config_module
from gomoveshift.cli.handlers import photos as photos_handler
from gomoveshift.cli.parser import _build_parser
from gomoveshift.config import Config, PhotosConfig
from gomoveshift.pipeline.color_correction import CorrectionParams
from gomoveshift.pipeline.photos import develop as dev


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)


def _use_photos_config(monkeypatch, **kwargs):
    photos = PhotosConfig(**kwargs)
    monkeypatch.setattr(config_module, "get_config", lambda *a, **k: Config(photos=photos))


def _write_photo(path, size=(48, 64)):
    rng = np.random.default_rng(11)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8), "RGB").save(path, "JPEG")
    return path


def _run(argv):
    return photos_handler.cmd_photos(_build_parser().parse_args(argv.split()), None)


def test_sensor_codes_that_share_a_uint8_level_stay_distinct_at_16_bits():
    """THE proof. Four adjacent 16-bit sensor codes all quantize to ONE uint8
    level; graded at 16 bits they stay four. No 8-bit render, upsampled by any
    factor, can separate codes it has already merged."""
    src = np.zeros((1, 4, 3), np.uint16)
    src[0, :, :] = np.asarray([1000, 1001, 1002, 1003], np.uint16)[:, None]
    params = CorrectionParams()
    assert np.unique(dev._grade(src, params)).size == 1, "the uint8 grade collapses them"
    out = dev._grade(src, params, depth=16)
    assert out.dtype == np.uint16
    assert np.unique(out).size == 4, f"16-bit grade must keep them apart, got {np.unique(out)}"


def test_the_whole_spatial_chain_holds_16_bits(tmp_path):
    """Every look stage runs in the frame's OWN depth, so a master carrying
    clarity, dehaze, vignette, halation and grain is still genuinely 16-bit: an
    8-bit frame scaled by 257 has as many distinct levels as its uint8 reduction,
    a real 16-bit render strictly more."""
    photo = _write_photo(tmp_path / "IMG_0002.jpg", size=(96, 128))
    options = dev.DevelopOptions(vignette=0.4, grain=0.5, halation=0.4, clarity=0.3, dehaze=0.2)
    rendered = list(dev.develop_variants(photo, ["natural"], options=options, depth=16))
    master = rendered[0][1].rgb
    assert master.dtype == np.uint16
    assert np.unique(master).size > np.unique(master >> 8).size


def test_export_tiff16_refuses_anything_but_a_16_bit_frame(tmp_path):
    """Robust by construction: the writer cannot be handed an 8-bit render to
    upsample, so a master can never claim headroom the pixels do not carry."""
    with pytest.raises(ValueError):
        dev.export_tiff16(np.zeros((4, 4, 3), np.uint8), tmp_path / "fake.tif")
    assert not list(tmp_path.iterdir()), "a refused write leaves nothing behind"


def test_cli_default_writes_a_jpeg_and_no_tiff(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    photo = _write_photo(tmp_path / "IMG_0004.jpg")
    assert _run(f"photos develop {photo} --styles natural") == 0
    out = tmp_path / "out"
    assert (out / "IMG_0004_natural.jpg").exists()
    assert not list(out.glob("*.tif")), "--format unset must not change what is written"


def test_cli_format_tiff16_writes_a_48_bit_srgb_master_of_the_same_picture(tmp_path, monkeypatch):
    _use_photos_config(monkeypatch, developed_output_dir=str(tmp_path / "out"))
    photo = _write_photo(tmp_path / "IMG_0005.jpg")
    assert _run(f"photos develop {photo} --styles natural --format tiff16") == 0
    out = tmp_path / "out"
    assert not list(out.glob("*.jpg")), "tiff16 replaces the master, it does not add one"
    master = out / "IMG_0005_natural.tif"
    data = tifffile.imread(master)
    assert data.dtype == np.uint16 and data.shape[2] == 3
    with tifffile.TiffFile(master) as handle:
        icc = bytes(handle.pages[0].tags["InterColorProfile"].value)
    assert icc == dev.srgb_icc_profile(), "the master carries the same sRGB tag the JPEGs do"
    # ...of the SAME picture: reduced to 8 bits it lands within a level or two of
    # the JPEG render, the residue being the intermediate rounding it no longer pays.
    developed = list(dev.develop_variants(photo, ["natural"]))[0][1]
    assert np.abs((data >> 8).astype(int) - developed.rgb.astype(int)).max() <= 2


def test_config_refuses_an_unknown_develop_format():
    assert PhotosConfig().develop_format == "jpeg"
    with pytest.raises(ValidationError):
        PhotosConfig(develop_format="tiff8")
