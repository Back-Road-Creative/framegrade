"""Local satellite imagery for Route Story openings."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from gomoveshift.pipeline.route_story.opening import CameraPose
from gomoveshift.pipeline.route_story.satellite import (
    SatelliteRaster,
    bundle_opening_rasters,
)


def test_local_satellite_raster_crops_a_camera_view_without_network(tmp_path: Path):
    source = tmp_path / "usa.png"
    image = Image.new("RGB", (360, 180), "#123456")
    image.putpixel((180, 90), (255, 255, 0))
    image.save(source)
    raster = SatelliteRaster(source, west=-180, south=-90, east=180, north=90, credit="Test")

    view = raster.view(CameraPose(0.0, 0.0, 2.0), 320, 180)

    assert view.size == (320, 180)
    assert view.getbbox() is not None
    assert raster.credit == "Test"


def test_satellite_raster_requires_a_readable_local_image(tmp_path: Path):
    missing = tmp_path / "missing.png"

    try:
        SatelliteRaster(missing, west=-180, south=-90, east=180, north=90, credit="Test")
    except ValueError as exc:
        assert "regular image" in str(exc)
    else:
        raise AssertionError("missing image was accepted")


def test_satellite_raster_receipt_fingerprints_local_source_and_geography(tmp_path: Path):
    source = tmp_path / "usa.png"
    Image.new("RGB", (360, 180), "#123456").save(source)
    raster = SatelliteRaster(source, west=-180, south=-90, east=180, north=90, credit="NASA Earth")

    receipt = raster.receipt()

    assert receipt["sha256"]
    assert receipt["dimensions"] == [360, 180]
    assert receipt["bounds"] == [-180, -90, 180, 90]
    assert receipt["credit"] == "NASA Earth"


def test_satellite_raster_prepares_a_content_addressed_local_cache(tmp_path: Path):
    source = tmp_path / "usa.png"
    Image.new("RGB", (360, 180), "#123456").save(source)
    raster = SatelliteRaster(source, west=-180, south=-90, east=180, north=90, credit="NASA Earth")

    prepared = raster.prepare(tmp_path / "cache")

    assert prepared.source.is_file()
    assert prepared.source.parent == tmp_path / "cache"
    assert prepared.receipt() == raster.receipt()
    assert raster.prepare(tmp_path / "cache").source == prepared.source


def test_bundle_mosaic_equalises_tone_seam_between_naip_tiles(tmp_path: Path):
    """Two adjoining NAIP acquisition tiles at very different exposures mosaic into
    one tone-matched raster: the seam step at their shared boundary all but
    disappears while each tile's own texture, away from that boundary, survives."""
    root = tmp_path / "bundle"
    rng = np.random.default_rng(0)
    noise_left = rng.integers(-15, 16, size=(256, 256, 3))
    noise_right = rng.integers(-15, 16, size=(256, 256, 3))

    def _tile(x: int, base: int, noise: np.ndarray) -> None:
        path = root / "imagery" / "2" / str(x) / "0.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.clip(base + noise, 0, 255).astype(np.uint8)
        Image.fromarray(pixels, "RGB").save(path, quality=100)

    _tile(0, 60, noise_left)
    _tile(1, 180, noise_right)
    manifest = {
        "imagery": {
            "bands": [
                {
                    "source": "test_band",
                    "provenance": {"attribution": "Test Imagery Agency"},
                    "written_zoom_range": {"minzoom": 2, "maxzoom": 2},
                }
            ]
        }
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    (raster,) = bundle_opening_rasters(root, tmp_path / "work")
    with Image.open(raster.source) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.float64)

    seam_step = abs(pixels[:, 254:256, :].mean() - pixels[:, 256:258, :].mean())
    assert seam_step < 15  # a bare mosaic of these tiles steps by ~120

    interior_std = pixels[:, 20:200, :].std()
    assert 5 < interior_std < 15  # in-tile texture is preserved, not smoothed away
