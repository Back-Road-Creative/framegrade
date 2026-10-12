"""Tests for Canon CR3 RAW support in the photo decode front-end.

Mirrors ``test_raw_decodes_via_rawpy`` in ``test_photo_develop.py``: LibRaw
rejects synthetic CR3 fixtures too, so the seam is exercised at the
``rawpy.imread`` boundary.
"""

import numpy as np
import pytest
import rawpy

from gomoveshift.pipeline.photos.develop import RAW_EXTENSIONS, SUPPORTED_EXTENSIONS, decode


def test_cr3_is_a_raw_extension():
    assert ".cr3" in RAW_EXTENSIONS
    assert ".cr3" in SUPPORTED_EXTENSIONS


def test_cr3_decodes_via_rawpy(tmp_path, monkeypatch):
    pixels = np.full((6, 8, 3), 23130, dtype=np.uint16)

    class _FakeRaw:
        raw_image = np.zeros((6, 8), dtype=np.uint16)
        white_level = 16383

        def postprocess(self, **kwargs):
            self.kwargs = kwargs
            return pixels

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fake = _FakeRaw()
    monkeypatch.setattr(rawpy, "imread", lambda _path: fake)
    path = tmp_path / "a.cr3"
    path.write_bytes(b"placeholder")

    out = decode(path)

    assert out.decoder == "rawpy"
    assert out.is_raw is True
    assert out.rgb.shape == (6, 8, 3)
    assert out.rgb.dtype == np.uint16
    assert out.bit_depth == 14


def test_cr3_still_fails_closed_on_unsupported_things(tmp_path):
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"\x00" * 16)

    with pytest.raises(Exception):
        decode(path)
