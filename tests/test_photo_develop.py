"""Tests for the photo decode front-end (gomoveshift.pipeline.photos.develop).

Fixtures are synthesized at test time rather than committed as binaries.
"""

import io
import logging
import struct
import types

import numpy as np
import pillow_heif
import pytest
import rawpy
from PIL import Image

from gomoveshift.pipeline.color_analysis import analyze_frame
from gomoveshift.pipeline.photos.develop import DecodedPhoto, PhotoDecodeError, decode


def _write_jpeg(path, size=(8, 6), orientation=None):
    kwargs = {}
    if orientation is not None:
        exif = Image.Exif()
        exif[0x0112] = orientation
        kwargs["exif"] = exif
    Image.new("RGB", size, (200, 100, 50)).save(path, **kwargs)
    return path


def _icc_xyz(x, y, z):
    return b"XYZ \x00\x00\x00\x00" + b"".join(
        struct.pack(">i", round(v * 65536)) for v in (x, y, z)
    )


def _display_p3_profile():
    """Minimal ICC v2 matrix-shaper profile carrying Display P3 primaries.

    ``ImageCms.createProfile`` can only emit sRGB/LAB/XYZ, so the wide-gamut tag
    an iPhone HEIC carries is synthesized by hand: header + white point +
    D50-adapted primary matrix + gamma-2.2 curves is everything lcms needs to
    build a P3 -> sRGB transform.
    """
    curv = b"curv\x00\x00\x00\x00" + struct.pack(">IH", 1, round(2.2 * 256))
    desc = b"desc\x00\x00\x00\x00" + struct.pack(">I", 3) + b"P3\x00" + b"\x00" * 79
    tags = [
        (b"desc", desc),
        (b"wtpt", _icc_xyz(0.9642, 1.0, 0.8249)),
        (b"rXYZ", _icc_xyz(0.51512, 0.2412, -0.00105)),
        (b"gXYZ", _icc_xyz(0.29198, 0.69225, 0.04189)),
        (b"bXYZ", _icc_xyz(0.1571, 0.06657, 0.78407)),
        (b"rTRC", curv),
        (b"gTRC", curv),
        (b"bTRC", curv),
    ]
    table, blob, offset = b"", b"", 128 + 4 + 12 * len(tags)
    for sig, body in tags:
        table += sig + struct.pack(">II", offset, len(body))
        offset += len(body) + (-len(body) % 4)
        blob += body + b"\x00" * (-len(body) % 4)
    header = struct.pack(
        ">I4sI4s4s4s12x4s28x3i48x",
        128 + 4 + len(table) + len(blob),
        b"lcms",
        0x02400000,
        b"mntr",
        b"RGB ",
        b"XYZ ",
        b"acsp",
        *(round(v * 65536) for v in (0.9642, 1.0, 0.8249)),
    )
    return header + struct.pack(">I", len(tags)) + table + blob


def test_pil_jpeg_decodes_to_uint8_srgb_rgb(tmp_path):
    out = decode(_write_jpeg(tmp_path / "a.jpg"))

    assert isinstance(out, DecodedPhoto)
    assert out.rgb.shape == (6, 8, 3)
    assert out.rgb.dtype == np.uint8
    assert out.decoder == "pil"
    assert out.is_raw is False
    assert (out.width, out.height) == (8, 6)
    assert out.bit_depth == 8


def test_png_with_alpha_flattens_to_three_channels(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGBA", (5, 4), (10, 20, 30, 128)).save(path)

    assert decode(path).rgb.shape == (4, 5, 3)


def test_exif_orientation_applied_on_decode(tmp_path):
    """Orientation 6 stores an 8x4 frame that must come back upright as 4x8."""
    out = decode(_write_jpeg(tmp_path / "rot.jpg", size=(8, 4), orientation=6))

    assert out.rgb.shape == (8, 4, 3)
    # Metadata reports the upright dimensions — always consistent with the array.
    assert (out.width, out.height) == (4, 8)
    assert (out.width, out.height) == (out.rgb.shape[1], out.rgb.shape[0])


def test_icc_tagged_source_is_converted_to_srgb(tmp_path):
    """A wide-gamut (Display P3) tag must change the numbers, not just ride along."""
    tagged, untagged = tmp_path / "p3.jpg", tmp_path / "plain.jpg"
    img = Image.new("RGB", (8, 6), (200, 60, 60))
    img.save(tagged, "JPEG", icc_profile=_display_p3_profile(), quality=95)
    img.save(untagged, "JPEG", quality=95)

    plain = decode(untagged)
    converted = decode(tagged)

    # Untagged output stays the raw PIL numbers — no profile means no conversion.
    with Image.open(untagged) as check:
        assert np.array_equal(plain.rgb, np.asarray(check.convert("RGB")))
    # A saturated P3 red maps to a more extreme sRGB red: R rises, G falls.
    assert int(converted.rgb[3, 3, 0]) > int(plain.rgb[3, 3, 0]) + 5
    assert int(converted.rgb[3, 3, 1]) < int(plain.rgb[3, 3, 1]) - 5


def test_unusable_icc_profile_is_refused_not_decoded_untagged(tmp_path):
    """A profile that cannot be converted refuses the decode (superseding the
    old untagged fallback, which let unconverted pixels be exported as sRGB)."""
    path = tmp_path / "bad-icc.jpg"
    Image.new("RGB", (8, 6), (200, 100, 50)).save(path, "JPEG", icc_profile=b"not-an-icc-profile")

    with pytest.raises(ValueError, match="unusable ICC colour profile"):
        decode(path)


def test_i16_tiff_decodes_at_full_16_bit_depth(tmp_path):
    """A 16-bit TIFF rides the grade chain at full depth, not clipped through uint8."""
    values = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 5000 + 300).astype(np.uint16)
    path = tmp_path / "deep.tif"
    Image.frombytes("I;16", (4, 3), values.astype("<u2").tobytes()).save(path, "TIFF")

    out = decode(path)

    assert out.rgb.dtype == np.uint16
    assert out.bit_depth == 16
    assert out.rgb.shape == (3, 4, 3)
    assert (out.width, out.height) == (4, 3)
    for channel in range(3):
        assert np.array_equal(out.rgb[:, :, channel], values)


def test_heic_decodes_via_pillow_heif(tmp_path):
    pillow_heif.register_heif_opener()
    path = tmp_path / "a.heic"
    Image.new("RGB", (8, 6), (12, 34, 56)).save(path, format="HEIF")

    out = decode(path)

    assert out.rgb.shape == (6, 8, 3)
    assert out.rgb.dtype == np.uint8
    assert out.decoder == "pillow-heif"
    assert out.is_raw is False


def test_raw_decodes_via_rawpy(tmp_path, monkeypatch):
    """RAW path, mocked at the ``rawpy.imread`` boundary.

    LibRaw rejects every synthetic DNG we could build, and a real camera RAW is
    a multi-MB binary we will not commit, so the seam is exercised at imread.
    """
    pixels = np.full((6, 8, 3), 23130, dtype=np.uint16)

    class _FakeRaw:
        raw_image = np.zeros((6, 8), dtype=np.uint16)
        white_level = 16383  # a 14-bit sensor saturates here, not at the container's 16

        def postprocess(self, **kwargs):
            self.kwargs = kwargs
            return pixels

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = _FakeRaw()
    monkeypatch.setattr(rawpy, "imread", lambda _path: fake)
    path = tmp_path / "a.dng"
    path.write_bytes(b"placeholder")

    out = decode(path)

    assert out.decoder == "rawpy"
    assert out.is_raw is True
    assert out.rgb.shape == (6, 8, 3)
    # 16-bit output survives decode; the grade quantizes to uint8 only at its end.
    assert out.rgb.dtype == np.uint16
    assert out.bit_depth == 14  # honest sensor depth from white_level, not the container's 16
    # 16-bit, no content-dependent auto-bright gain, true sRGB transfer (not BT.709).
    assert fake.kwargs["output_bps"] == 16
    assert fake.kwargs["no_auto_bright"] is True
    assert fake.kwargs["gamma"] == (2.4, 12.92)
    assert fake.kwargs["use_camera_wb"] is True
    # Orientation is LibRaw's job here; disabling its flip would ship sideways frames.
    assert "user_flip" not in fake.kwargs


def _jpeg_bytes(size, color=(120, 80, 40)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _fake_unpackable_raw(preview=None, flip=0, sensor=(8, 6)):
    """Factory for raws whose stream LibRaw cannot unpack — the HE NEF shape.

    Real HE/HE* NEFs fail exactly like this: ``raw_image``/``postprocess``
    raise LibRawDataError ("data corrupted") while the file header, sizes and
    embedded preview all read fine — but only on a *fresh* handle. A failed
    unpack poisons the handle, so the fake refuses out-of-order thumbnail
    calls exactly like LibRaw does; monkeypatch ``imread`` to this factory so
    every open gets a new instance.
    """

    class _FakeHERaw:
        _poisoned = False
        sizes = types.SimpleNamespace(width=sensor[0], height=sensor[1], flip=flip)

        @property
        def raw_image(self):
            self._poisoned = True
            raise rawpy.LibRawDataError("data corrupted at 1541636")

        def postprocess(self, **kwargs):
            self._poisoned = True
            raise rawpy.LibRawDataError("data corrupted at 1541636")

        def extract_thumb(self):
            if self._poisoned:
                raise rawpy.LibRawOutOfOrderCallError("out-of-order call")
            if preview is None:
                raise rawpy.LibRawNoThumbnailError("no thumbnail")
            return types.SimpleNamespace(format=rawpy.ThumbFormat.JPEG, data=preview)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    return _FakeHERaw


def test_he_nef_falls_back_to_full_res_embedded_preview(tmp_path, monkeypatch, caplog):
    """An unpackable raw with a full-size preview develops from that preview."""
    raw_cls = _fake_unpackable_raw(preview=_jpeg_bytes((8, 6)), sensor=(8, 6))
    monkeypatch.setattr(rawpy, "imread", lambda _path: raw_cls())
    path = tmp_path / "he.nef"
    path.write_bytes(b"placeholder")

    with caplog.at_level(logging.WARNING):
        out = decode(path)

    assert out.decoder == "rawpy-preview"
    assert out.is_raw is False
    assert out.bit_depth == 8
    assert out.rgb.shape == (6, 8, 3)
    assert out.rgb.dtype == np.uint8
    # The fallback is loud, names the likely cause, and names the remedy.
    assert "High Efficiency" in caplog.text
    assert "Lossless Compressed" in caplog.text


def test_he_nef_preview_honors_camera_flip(tmp_path, monkeypatch):
    """flip=6 (90° CW) brings a landscape-stored preview upright."""
    raw_cls = _fake_unpackable_raw(preview=_jpeg_bytes((8, 4)), flip=6, sensor=(8, 4))
    monkeypatch.setattr(rawpy, "imread", lambda _path: raw_cls())
    path = tmp_path / "portrait.nef"
    path.write_bytes(b"placeholder")

    out = decode(path)

    assert out.rgb.shape == (8, 4, 3)
    # Preview-path metadata still reports the *stored* dimensions (only the
    # PIL path derives width/height from the upright array).
    assert (out.width, out.height) == (8, 4)


def test_he_nef_without_preview_fails_closed_with_remediation(tmp_path, monkeypatch):
    raw_cls = _fake_unpackable_raw(preview=None)
    monkeypatch.setattr(rawpy, "imread", lambda _path: raw_cls())
    path = tmp_path / "he.nef"
    path.write_bytes(b"placeholder")

    with pytest.raises(PhotoDecodeError, match="High Efficiency"):
        decode(path)


def test_he_nef_with_tiny_thumbnail_fails_closed(tmp_path, monkeypatch):
    """A 160x120-class thumbnail is no substitute for the frame — fail closed."""
    raw_cls = _fake_unpackable_raw(preview=_jpeg_bytes((2, 2)), sensor=(8, 6))
    monkeypatch.setattr(rawpy, "imread", lambda _path: raw_cls())
    path = tmp_path / "he.nef"
    path.write_bytes(b"placeholder")

    with pytest.raises(PhotoDecodeError, match="High Efficiency"):
        decode(path)


def test_corrupt_file_fails_closed_naming_the_file(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 definitely not a jpeg")

    with pytest.raises(PhotoDecodeError, match="broken.jpg"):
        decode(path)


def test_unsupported_extension_fails_closed(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 16)

    with pytest.raises(PhotoDecodeError, match="clip.mp4"):
        decode(path)


def test_missing_file_fails_closed(tmp_path):
    with pytest.raises(PhotoDecodeError, match="nope.jpg"):
        decode(tmp_path / "nope.jpg")


def test_output_is_accepted_by_analyze_frame(tmp_path):
    """The decode contract is exactly what the existing grade engine consumes."""
    stats = analyze_frame(decode(_write_jpeg(tmp_path / "a.jpg", size=(16, 12))).rgb)

    assert 0.0 <= stats.saturation <= 1.0
    assert 0.0 <= stats.white_point <= 255.0
