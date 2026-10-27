"""Panorama seam QA: ghosting and exposure steps are measured per seam and
a result past the named thresholds is refused before anything is written.
"""

import cv2
import pytest

from gomoveshift.pipeline.photos import panorama
from tests.test_photo_panorama import _textured_frame


def _tiles(tmp_path, edit_middle=None, count=3, tile_frac=0.55):
    full = _textured_frame()
    h, w = full.shape[:2]
    tile_w = int(w * tile_frac)
    step = (w - tile_w) // (count - 1)
    paths = []
    for i in range(count):
        tile = full[:, i * step : i * step + tile_w].copy()
        if edit_middle is not None and i == 1:
            edit_middle(tile)
        p = tmp_path / f"t{i}.png"
        cv2.imwrite(str(p), tile)
        paths.append(p)
    return paths


def _ghost(tile):
    # a large object present in the second frame only, inside the overlap
    cv2.rectangle(tile, (20, 280), (140, 440), (250, 250, 250), -1)
    cv2.circle(tile, (80, 360), 40, (0, 0, 0), -1)


def _brighten(tile):
    tile[:] = cv2.convertScaleAbs(tile, alpha=1.0, beta=45)


def test_clean_set_passes_with_low_measurements(tmp_path):
    out = tmp_path / "out.png"
    result = panorama.stitch_panorama(_tiles(tmp_path), out)
    assert out.exists()
    assert len(result.seam_ghost_fractions) == 2
    assert len(result.seam_luma_steps) == 2
    assert max(result.seam_ghost_fractions) < panorama.MAX_SEAM_GHOST_FRACTION
    assert max(result.seam_luma_steps) < panorama.MAX_SEAM_LUMA_STEP


def test_ghost_in_overlap_is_refused(tmp_path):
    out = tmp_path / "out.png"
    with pytest.raises(panorama.PanoramaError, match="ghost"):
        panorama.stitch_panorama(_tiles(tmp_path, _ghost), out)
    assert not out.exists()


def test_exposure_step_is_refused(tmp_path):
    out = tmp_path / "out.png"
    with pytest.raises(panorama.PanoramaError, match="exposure"):
        panorama.stitch_panorama(_tiles(tmp_path, _brighten), out)
    assert not out.exists()


def test_measure_seam_reports_both_numbers_directly(tmp_path):
    clean = _tiles(tmp_path)
    a, b = (cv2.imread(str(p)) for p in clean[:2])
    ghost, step = panorama.measure_seam(a, b)
    assert ghost < panorama.MAX_SEAM_GHOST_FRACTION
    assert step < 1.0
    _, step_bright = panorama.measure_seam(a, cv2.convertScaleAbs(b, alpha=1.0, beta=45))
    assert step_bright > panorama.MAX_SEAM_LUMA_STEP
