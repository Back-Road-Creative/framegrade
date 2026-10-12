"""The film's opening descends through the scene bundle's own tile pyramid."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import film as film_module
from gomoveshift.pipeline.route_story.bundle import tile_bounds
from gomoveshift.pipeline.route_story.layers import finest_last, opening_credit
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import SatelliteRaster, bundle_opening_rasters

_COLOURS = {(0, 2): (200, 0, 0), (1, 2): (0, 200, 0), (0, 3): (0, 0, 200), (1, 3): (200, 200, 0)}


def _tile(root: Path, z: int, x: int, y: int, colour) -> None:
    path = root / "imagery" / str(z) / str(x) / f"{y}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (256, 256), colour).save(path, quality=95)


def _bundle(tmp_path: Path, *, skip: tuple[int, int] | None = None) -> Path:
    root = tmp_path / "bundle"
    _tile(root, 2, 0, 1, (90, 90, 90))
    for (x, y), colour in _COLOURS.items():
        if (x, y) != skip:
            _tile(root, 3, x, y, colour)
    manifest = {
        "imagery": {
            "bands": [
                {
                    "source": "test_band",
                    "provenance": {"attribution": "Test Imagery Agency"},
                    "written_zoom_range": {"minzoom": 2, "maxzoom": 3},
                }
            ]
        }
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_a_band_contributes_one_raster_per_complete_zoom_coarse_to_fine(tmp_path):
    # test_band's manifest range is z2-3 and both are complete on disk -- the
    # camera needs a same-band fallback between a band's own zooms (not just
    # between bands), so every complete zoom becomes its own raster.
    rasters = bundle_opening_rasters(_bundle(tmp_path), tmp_path / "work")

    assert len(rasters) == 2
    with Image.open(rasters[0].source) as image:
        assert image.size == (256, 256)  # z2, coarser, listed first
    raster = rasters[-1]  # z3, finest
    north_west, south_east = tile_bounds(3, 0, 2), tile_bounds(3, 1, 3)
    assert raster.credit == "Test Imagery Agency"
    assert (raster.west, raster.north) == pytest.approx((north_west.west, north_west.north))
    assert (raster.east, raster.south) == pytest.approx((south_east.east, south_east.south))
    with Image.open(raster.source) as image:
        assert image.size == (512, 512)


def test_rows_are_respaced_from_mercator_to_even_latitude(tmp_path):
    # The renderer reads a raster as evenly spaced in latitude. In Mercator the
    # y=2/y=3 tile boundary (40.98 N) is row 256 of 512; evenly spaced between
    # 66.51 N and the equator it belongs at row 196.
    rasters = bundle_opening_rasters(_bundle(tmp_path), tmp_path / "work")
    raster = rasters[-1]  # z3, the one with this y=2/y=3 boundary content

    with Image.open(raster.source) as image:
        upper = image.getpixel((10, 180))
        lower = image.getpixel((10, 215))
    assert upper[0] > 150 and upper[2] < 80
    assert lower[2] > 150 and lower[0] < 80


def test_a_level_too_wide_to_mosaic_steps_down_a_zoom(tmp_path):
    (raster,) = bundle_opening_rasters(_bundle(tmp_path), tmp_path / "work", max_width_px=300)

    with Image.open(raster.source) as image:
        assert image.width == 256


def test_a_zoom_with_a_hole_steps_down_rather_than_showing_a_gap(tmp_path):
    (raster,) = bundle_opening_rasters(_bundle(tmp_path, skip=(1, 3)), tmp_path / "work")

    with Image.open(raster.source) as image:
        assert image.width == 256


def test_a_band_with_no_complete_zoom_is_left_out_and_said_so(tmp_path, caplog):
    bundle = _bundle(tmp_path, skip=(1, 3))
    (bundle / "imagery" / "2" / "0" / "1.jpg").unlink()

    with caplog.at_level("WARNING"):
        rasters = bundle_opening_rasters(bundle, tmp_path / "work")

    assert rasters == ()
    assert "test_band" in caplog.text


def _flat(tmp_path: Path, name: str, bounds, credit: str, width: int) -> SatelliteRaster:
    path = tmp_path / f"{name}.jpg"
    Image.new("RGB", (width, width // 2), "#3c6f83").save(path)
    west, south, east, north = bounds
    return SatelliteRaster(path, west, south, east, north, credit)


def test_the_opening_shows_the_finest_raster_that_covers_the_frame(tmp_path):
    national = _flat(tmp_path, "national", (-180, -90, 180, 90), "National", 512)
    local = _flat(tmp_path, "local", (-84.0, 35.0, -83.0, 35.5), "Local", 512)
    sharper = _flat(tmp_path, "sharper", (-84.0, 35.0, -83.0, 35.5), "Sharper", 2048)
    start = RoutePoint(time_s=0.0, latitude=35.25, longitude=-83.5)

    ordered = finest_last((national, sharper, local))

    assert [r.credit for r in ordered] == ["National", "Local", "Sharper"]
    assert opening_credit(ordered, start, 1.0, 1920, 1080) == "Sharper"
    assert opening_credit(ordered, start, 0.0, 1920, 1080) == "National"


def test_the_film_command_hands_the_bundle_pyramid_to_the_opening(tmp_path, monkeypatch):
    gpx = tmp_path / "day.gpx"
    gpx.write_text(
        """<gpx><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T12:00:10Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )
    photos = tmp_path / "photos"
    photos.mkdir()
    national, local = tmp_path / "national.jpg", tmp_path / "local.jpg"
    Image.new("RGB", (64, 32), "#3c6f83").save(national)
    Image.new("RGB", (64, 32), "#4e7540").save(local)
    receipt = tmp_path / "receipt.json"
    rendered: dict = {}

    def fake_render(**kwargs):
        rendered["credits"] = [r.credit for r in kwargs["opening_rasters"]]
        Path(kwargs["output"]).write_bytes(b"fake film bytes")

    monkeypatch.setattr(film_module, "render_route_story_film", fake_render)
    monkeypatch.setattr(
        film_module, "encode_final", lambda s, d: Path(d).write_bytes(Path(s).read_bytes())
    )
    argv = [
        "--gpx", str(gpx), "--recorded-on", "2026-09-16", "--photos-dir", str(photos),
        "--national-raster", str(national), "--national-credit", "Test overview",
        "--local-raster", str(local), "--local-credit", "Test local",
        "--local-bounds", "-80.1", "34.9", "-79.9", "35.1",
        "--output", str(tmp_path / "film.mp4"), "--receipt", str(receipt),
        "--bundle", str(_bundle(tmp_path)), "--preflight-frames", "0",
    ]  # fmt: skip

    assert film_module.main(argv) == 0

    # test_band's z2 and z3 are both complete -- one raster per zoom now.
    assert rendered["credits"] == ["Test Imagery Agency", "Test Imagery Agency"]
    recorded = json.loads(receipt.read_text(encoding="utf-8"))["opening_pyramid"]
    assert [level["credit"] for level in recorded] == ["Test Imagery Agency", "Test Imagery Agency"]
    assert [level["zoom"] for level in recorded] == [2, 3]


def test_the_opening_credit_stays_readable_over_white_cloud():
    from gomoveshift.pipeline.route_story.hud import _draw_opening

    image = Image.new("RGB", (1280, 720), (255, 255, 255))

    _draw_opening(image, 1.0, "Test Imagery Agency")

    band = image.crop((0, 720 - 60, 640, 720)).convert("L")
    assert min(band.getdata()) < 60
