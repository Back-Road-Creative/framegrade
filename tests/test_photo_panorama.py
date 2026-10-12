"""Panorama stitching: ordered image paths -> one stitched frame, gated
by a seam-quality check so a degenerate stitch never reaches disk.
"""

import cv2
import numpy as np
import pytest

from gomoveshift.pipeline.photos import panorama


def _textured_frame(size=(480, 720)):
    """A deterministic, feature-rich frame -- OpenCV's Stitcher needs real
    corners/edges to match, a flat or pure-noise frame won't align."""
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # checkerboard gives the detector plenty of corners across the whole frame
    tile = 40
    for gy in range(0, h, tile):
        for gx in range(0, w, tile):
            if ((gx // tile) + (gy // tile)) % 2 == 0:
                img[gy : gy + tile, gx : gx + tile] = (200, 200, 200)
            else:
                img[gy : gy + tile, gx : gx + tile] = (60, 60, 60)
    cv2.circle(img, (150, 150), 55, (30, 30, 220), -1)
    cv2.rectangle(img, (380, 90), (520, 260), (30, 200, 30), -1)
    cv2.circle(img, (600, 360), 70, (220, 30, 30), -1)
    cv2.putText(img, "PANO", (60, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 0), 3)
    return img


def _write_overlapping_tiles(tmp_path, count=3, tile_frac=0.55):
    full = _textured_frame()
    h, w = full.shape[:2]
    tile_w = int(w * tile_frac)
    step = (w - tile_w) // (count - 1)
    paths = []
    for i in range(count):
        x = i * step
        tile = full[:, x : x + tile_w]
        p = tmp_path / f"tile_{i}.png"
        cv2.imwrite(str(p), tile)
        paths.append(p)
    return paths


def _write_noise(tmp_path, name, seed, size=(200, 200)):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)
    p = tmp_path / name
    cv2.imwrite(str(p), img)
    return p


class _FakeStitcher:
    def __init__(self, status, image):
        self._status = status
        self._image = image

    def stitch(self, images):
        return self._status, self._image


def test_stitch_panorama_succeeds_on_overlapping_tiles(tmp_path):
    tiles = _write_overlapping_tiles(tmp_path)
    out = tmp_path / "out.png"
    result = panorama.stitch_panorama(tiles, out)
    assert result.status == "ok"
    assert out.exists()
    assert result.width > 0 and result.height > 0
    assert result.valid_area_fraction > 0.5


def test_stitch_panorama_refuses_fewer_than_two_images(tmp_path):
    tiles = _write_overlapping_tiles(tmp_path, count=3)
    out = tmp_path / "out.png"
    with pytest.raises(panorama.PanoramaError, match=">=2"):
        panorama.stitch_panorama(tiles[:1], out)
    assert not out.exists()


def test_stitch_panorama_non_overlapping_images_raise_a_clear_error(tmp_path):
    a = _write_noise(tmp_path, "a.png", seed=1)
    b = _write_noise(tmp_path, "b.png", seed=2)
    out = tmp_path / "out.png"
    with pytest.raises(panorama.PanoramaError):
        panorama.stitch_panorama([a, b], out)
    assert not out.exists()


def test_stitch_panorama_rejects_low_valid_area(tmp_path, monkeypatch):
    tiles = _write_overlapping_tiles(tmp_path)
    bad = np.zeros((100, 500, 3), dtype=np.uint8)
    bad[:, :50] = 255  # only 10% of the frame is non-black
    monkeypatch.setattr(
        panorama.cv2,
        "Stitcher_create",
        lambda *a, **k: _FakeStitcher(cv2.Stitcher_OK, bad),
    )
    out = tmp_path / "out.png"
    with pytest.raises(panorama.PanoramaError, match="seam QA"):
        panorama.stitch_panorama(tiles, out)
    assert not out.exists()


def test_stitch_panorama_rejects_output_narrower_than_widest_input(tmp_path, monkeypatch):
    tiles = _write_overlapping_tiles(tmp_path)
    widest = max(cv2.imread(str(p)).shape[1] for p in tiles)
    narrow = np.full((100, widest - 10, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(
        panorama.cv2,
        "Stitcher_create",
        lambda *a, **k: _FakeStitcher(cv2.Stitcher_OK, narrow),
    )
    out = tmp_path / "out.png"
    with pytest.raises(panorama.PanoramaError, match="seam QA"):
        panorama.stitch_panorama(tiles, out)
    assert not out.exists()
