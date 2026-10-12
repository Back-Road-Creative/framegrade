"""Gallery-export output correctness: atomic write, sRGB tag, chroma, quality knob.

The developed gallery master is the archival artifact the social resizer later
re-encodes from, so its encode must not silently inherit social-upload
compromises (4:2:0 chroma, q85, untagged colour) or leave partial files on a
failed write.
"""

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageCms, JpegImagePlugin

import gomoveshift.pipeline.photos.develop as dev

SUBSAMPLING_444 = 0
SUBSAMPLING_420 = 2


def _frame(width=64, height=48):
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _capture_save_kwargs(monkeypatch):
    captured = {}
    original = Image.Image.save

    def spy(self, fp, format=None, **kwargs):
        captured.update(kwargs)
        return original(self, fp, format, **kwargs)

    monkeypatch.setattr(Image.Image, "save", spy)
    return captured


def test_export_output_carries_srgb_icc_tag(tmp_path):
    out = dev.export_jpeg(_frame(), tmp_path / "out.jpg")

    with Image.open(out) as img:
        icc = img.info.get("icc_profile")
        assert icc, "gallery export shipped with no ICC tag"
        desc = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc)))
        assert "sRGB" in desc


def test_export_keeps_full_chroma_resolution(tmp_path):
    out = dev.export_jpeg(_frame(), tmp_path / "out.jpg")

    with Image.open(out) as img:
        sampling = JpegImagePlugin.get_sampling(img)
    assert sampling == SUBSAMPLING_444, (
        f"gallery master encoded with subsampling mode {sampling}, expected 4:4:4 "
        "(PIL's default 4:2:0 halves chroma before the resizer re-encodes)"
    )


def test_failed_export_leaves_no_partial_or_target_file(tmp_path, monkeypatch):
    def exploding_save(self, fp, format=None, **kwargs):
        Path(fp).write_bytes(b"partial garbage")
        raise OSError("disk full mid-write")

    monkeypatch.setattr(Image.Image, "save", exploding_save)
    target = tmp_path / "nested" / "out.jpg"

    with pytest.raises(OSError, match="disk full"):
        dev.export_jpeg(_frame(), target)

    assert not target.exists(), "a partial gallery master became the target artifact"
    assert list(target.parent.iterdir()) == [], "failed export left an orphan temp file"


def test_successful_export_leaves_only_the_final_file(tmp_path):
    dev.export_jpeg(_frame(), tmp_path / "out.jpg")

    assert [p.name for p in tmp_path.iterdir()] == ["out.jpg"]


def test_export_defaults_to_the_gallery_quality_knob(tmp_path, monkeypatch):
    import gomoveshift.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(photos=SimpleNamespace(jpeg_quality=57, gallery_jpeg_quality=91)),
    )
    captured = _capture_save_kwargs(monkeypatch)

    dev.export_jpeg(_frame(), tmp_path / "default.jpg")

    assert captured["quality"] == 91


def test_gallery_quality_config_default_exceeds_social_quality():
    from gomoveshift.config import PhotosConfig

    cfg = PhotosConfig()
    assert cfg.gallery_jpeg_quality == 92
    assert cfg.gallery_jpeg_quality > cfg.jpeg_quality
