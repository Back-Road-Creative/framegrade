"""Optional social aspect centre-crop in the photo resizer.

Unset (the default) must stay byte-identical to the aspect-preserving scale;
set, it trims the largest centred rectangle of the target ratio AFTER the EXIF
orientation and BEFORE the width scale, one output per photo.
"""

import pytest
from PIL import Image

from gomoveshift.config import PhotosConfig
from gomoveshift.pipeline.photos import resizer as resizer_mod
from gomoveshift.pipeline.photos.resizer import PhotoResizer

ORIENTATION = 0x0112
GPS_IFD = 0x8825


def _save(path, size, color=(90, 120, 160), exif=None):
    img = Image.new("RGB", size, color)
    img.save(path, "JPEG", quality=95, **({"exif": exif} if exif is not None else {}))
    return str(path)


def _gps_exif():
    exif = Image.Exif()
    exif[GPS_IFD] = {1: "N", 2: (40.0, 0.0, 0.0), 3: "W", 4: (74.0, 0.0, 0.0)}
    return exif


def _exif_orientation(value):
    exif = Image.Exif()
    exif[ORIENTATION] = value
    return exif


def test_unset_is_byte_identical_and_never_crops(tmp_path, monkeypatch):
    """An empty aspect leaves the shipped path untouched — same bytes, no crop call.
    No ICC pinning needed: ``srgb_icc_profile`` zeroes the ``dateTime`` littleCMS
    stamps, so two encodes a second apart already agree byte for byte."""
    src = _save(tmp_path / "wide.jpg", (1920, 1440))
    shipped = PhotoResizer(output_base=tmp_path / "default").resize_batch([src], "2026-07-24")[0]

    def _unreachable(*_a, **_k):
        raise AssertionError("crop code ran with social_aspect unset")

    monkeypatch.setattr(resizer_mod, "_centre_crop", _unreachable)
    off = PhotoResizer(social_aspect="", output_base=tmp_path / "off").resize_batch(
        [src], "2026-07-24"
    )[0]

    assert off.read_bytes() == shipped.read_bytes()
    with Image.open(off) as img:
        assert img.size == (1080, 810)  # native 4:3, uncropped


@pytest.mark.parametrize("size", [(1920, 1440), (1440, 1920)])
@pytest.mark.parametrize("aspect,ratio", [("square", 1.0), ("4:5", 0.8), ("9:16", 0.5625)])
def test_each_target_ratio_from_landscape_and_portrait(tmp_path, size, aspect, ratio):
    src = _save(tmp_path / f"{size[0]}x{size[1]}.jpg", size)
    out = PhotoResizer(social_aspect=aspect, output_base=tmp_path / "out").resize_batch(
        [src], "2026-07-24"
    )[0]
    with Image.open(out) as img:
        w, h = img.size
    assert abs(w / h - ratio) <= 0.01, f"{aspect} from {size} landed on {w}x{h}"


def test_centre_crop_keeps_the_centred_rectangle():
    """Equal margins trimmed — the off-centre marker bands are the ones dropped."""
    img = Image.new("RGB", (1200, 600), (255, 0, 0))
    img.paste(Image.new("RGB", (600, 600), (0, 255, 0)), (300, 0))
    out = resizer_mod._centre_crop(img, 1.0)
    assert out.size == (600, 600)
    assert out.getcolors() == [(600 * 600, (0, 255, 0))]


def test_orientation_is_applied_before_the_crop(tmp_path):
    """A rotated portrait crops along the upright axis, not the stored one."""
    src = _save(tmp_path / "rotated.jpg", (2000, 1000), exif=_exif_orientation(6))
    out = PhotoResizer(social_aspect="9:16", output_base=tmp_path / "out").resize_batch(
        [src], "2026-07-24"
    )[0]
    with Image.open(out) as img:
        # Upright is 1000x2000: keep the full width, trim 111px top and bottom.
        # Cropping the stored 2000x1000 first would ship a landscape 1000x562.
        assert img.size == (1000, 1778)
        assert img.getexif().get(ORIENTATION) is None


def test_gps_gate_still_fires_with_the_crop_enabled(tmp_path, monkeypatch):
    """The crop must not route around _assert_no_gps — no output, hard fail."""
    src = _save(tmp_path / "geo.jpg", (1920, 1440), exif=_gps_exif())
    monkeypatch.setattr(resizer_mod, "_gps_in_xmp_or_comment", lambda _img: True)
    out_base = tmp_path / "out"

    with pytest.raises(ValueError, match="GPS XMP/comment survived"):
        PhotoResizer(social_aspect="4:5", output_base=out_base).resize_batch([src], "2026-07-24")

    assert not list((out_base / "2026-07-24").glob("*.jpg"))


def test_gps_exif_still_stripped_from_a_cropped_output(tmp_path):
    src = _save(tmp_path / "geo2.jpg", (1920, 1440), exif=_gps_exif())
    out = PhotoResizer(social_aspect="square", output_base=tmp_path / "out").resize_batch(
        [src], "2026-07-24"
    )[0]
    with Image.open(out) as img:
        assert GPS_IFD not in img.getexif()


def test_unknown_aspect_rejected_by_the_resizer(tmp_path):
    with pytest.raises(ValueError, match="square, 4:5, 9:16"):
        PhotoResizer(social_aspect="16:9", output_base=tmp_path / "out")


def test_publish_and_batch_expose_the_flag():
    from gomoveshift.cli.parser import _build_parser

    parse = _build_parser().parse_args
    pub = ["photos", "publish", "--captions", "c.yaml"]
    assert parse(pub).social_aspect is None  # None -> photos.social_aspect
    assert parse([*pub, "--social-aspect", "4:5"]).social_aspect == "4:5"
    batch = ["photos", "batch", "--start", "2026-07-24", "--end", "2026-07-30"]
    assert parse([*batch, "--social-aspect", "9:16"]).social_aspect == "9:16"
    with pytest.raises(SystemExit):
        parse([*pub, "--social-aspect", "16:9"])


def test_orchestrator_forwards_the_aspect_to_its_resizer(tmp_path):
    from gomoveshift.pipeline.photos.orchestrator import PhotoOrchestrator

    orch = PhotoOrchestrator(site_root=tmp_path, output_dir=tmp_path, social_aspect="square")
    assert orch.resizer.social_aspect == "square"


def test_unknown_aspect_rejected_by_the_config():
    with pytest.raises(ValueError, match="square, 4:5, 9:16"):
        PhotosConfig(social_aspect="portrait")

    assert PhotosConfig().social_aspect == ""
