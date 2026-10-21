"""Offline map asset bundle: footprint, tile math, terrain-rgb codec,
tile pyramid writer, and content-addressed manifest.

This is the pure, offline bundle core. Nothing here makes a network call —
fetching imagery/terrain is done elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift import paths as gms_paths
from gomoveshift.pipeline.route_story import bundle
from gomoveshift.pipeline.route_story.metrics import RoutePoint


# ---------------------------------------------------------------------------
# gms_route_story_cache_dir
# ---------------------------------------------------------------------------


def test_route_story_cache_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GMS_ROUTE_STORY_CACHE_DIR", raising=False)
    default = gms_paths.gms_route_story_cache_dir()
    assert default == Path(gms_paths._DEFAULT_ROUTE_STORY_CACHE_DIR)
    assert str(default).endswith("/.cache/gomoveshift/route-story")


def test_route_story_cache_dir_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "rs-cache"))
    assert gms_paths.gms_route_story_cache_dir() == tmp_path / "rs-cache"


# ---------------------------------------------------------------------------
# Footprint
# ---------------------------------------------------------------------------


def test_footprint_valid() -> None:
    fp = bundle.Footprint(west=-84.0, south=35.0, east=-83.0, north=36.0)
    assert fp.west == -84.0 and fp.north == 36.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"west": -83.0, "south": 35.0, "east": -84.0, "north": 36.0},  # west >= east
        {"west": -84.0, "south": 36.0, "east": -83.0, "north": 35.0},  # south >= north
        {"west": -190.0, "south": 35.0, "east": -83.0, "north": 36.0},  # bad lon
        {"west": -84.0, "south": -95.0, "east": -83.0, "north": 36.0},  # bad lat
    ],
)
def test_footprint_rejects_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        bundle.Footprint(**kwargs)


def test_footprint_for_route_expands_bbox_by_margin() -> None:
    points = [
        RoutePoint(time_s=0.0, latitude=35.0, longitude=-84.0),
        RoutePoint(time_s=1.0, latitude=35.1, longitude=-83.9),
    ]
    fp_no_margin = bundle.footprint_for_route(points, margin_m=0.0)
    fp_margin = bundle.footprint_for_route(points, margin_m=1000.0)
    assert fp_no_margin.west == pytest.approx(-84.0)
    assert fp_no_margin.north == pytest.approx(35.1)
    assert fp_margin.west < fp_no_margin.west
    assert fp_margin.east > fp_no_margin.east
    assert fp_margin.south < fp_no_margin.south
    assert fp_margin.north > fp_no_margin.north


def test_footprint_for_route_rejects_empty_points() -> None:
    with pytest.raises(ValueError):
        bundle.footprint_for_route([], margin_m=100.0)


def test_footprint_for_route_rejects_negative_margin() -> None:
    points = [RoutePoint(time_s=0.0, latitude=35.0, longitude=-84.0)]
    with pytest.raises(ValueError):
        bundle.footprint_for_route(points, margin_m=-1.0)


# ---------------------------------------------------------------------------
# Tile math
# ---------------------------------------------------------------------------


def test_lonlat_to_tile_origin_z0() -> None:
    x, y = bundle.lonlat_to_tile(0.0, 0.0, 0)
    assert x == pytest.approx(0.5)
    assert y == pytest.approx(0.5)


def test_lonlat_to_tile_known_value_z12() -> None:
    x, y = bundle.lonlat_to_tile(-83.5, 35.25, 12)
    assert int(x) == 1097
    assert int(y) == 1618


@pytest.mark.parametrize(
    "lon,lat,z",
    [(-200.0, 0.0, 5), (0.0, -90.0, 5), (0.0, 0.0, -1)],
)
def test_lonlat_to_tile_rejects_invalid(lon: float, lat: float, z: int) -> None:
    with pytest.raises(ValueError):
        bundle.lonlat_to_tile(lon, lat, z)


def test_tile_bounds_contains_source_point() -> None:
    x, y = bundle.lonlat_to_tile(-83.5, 35.25, 12)
    bounds = bundle.tile_bounds(12, int(x), int(y))
    assert bounds.west <= -83.5 <= bounds.east
    assert bounds.south <= 35.25 <= bounds.north


def test_tiles_covering_small_footprint_z12() -> None:
    fp = bundle.Footprint(west=-83.51, south=35.24, east=-83.49, north=35.26)
    tiles = bundle.tiles_covering(fp, 12)
    assert (1097, 1618) in tiles
    assert tiles == sorted(tiles)


def test_tiles_covering_empty_when_out_of_range() -> None:
    # A footprint entirely inside a single tile at low zoom still returns it.
    fp = bundle.Footprint(west=-1.0, south=-1.0, east=1.0, north=1.0)
    tiles = bundle.tiles_covering(fp, 0)
    assert tiles == [(0, 0)]


def test_tiles_covering_symmetric_epsilon_on_west_and_north_edges() -> None:
    # A footprint whose west/north edge is a float-rounding hair past a tile
    # boundary must not pull in a whole extra column/row of tiles: the
    # min-side epsilon must match the max-side one already applied to
    # x_max/y_max.
    z = 12
    x, y = 1097, 1618
    bounds = bundle.tile_bounds(z, x, y)
    on_boundary = bundle.Footprint(
        west=bounds.west,
        south=bounds.south + 1e-6,
        east=bounds.east - 1e-6,
        north=bounds.north,
    )
    noisy = bundle.Footprint(
        west=bounds.west - 1e-12,
        south=bounds.south + 1e-6,
        east=bounds.east - 1e-6,
        north=bounds.north + 1e-12,
    )
    assert bundle.tiles_covering(noisy, z) == bundle.tiles_covering(on_boundary, z)
    assert bundle.tiles_covering(on_boundary, z) == [(x, y)]


# ---------------------------------------------------------------------------
# Terrain-RGB codec
# ---------------------------------------------------------------------------


def test_terrain_rgb_roundtrip() -> None:
    heights = np.array([[0.0, 1000.0], [-500.5, 8848.0]], dtype=np.float32)
    encoded = bundle.encode_terrain_rgb(heights)
    assert encoded.dtype == np.uint8
    assert encoded.shape == (2, 2, 3)
    decoded = bundle.decode_terrain_rgb(encoded)
    assert np.max(np.abs(decoded - heights)) < 0.1


def test_terrain_rgb_bytes_are_exact_at_every_channel_boundary() -> None:
    """Pins the byte layout (R = unit >> 16, G = unit >> 8, B = unit, each & 255)
    at the carries between channels, so a faster encoder cannot shift a byte."""
    units = np.array([0, 1, 255, 256, 65535, 65536, 65537, 1_000_000, 256**3 - 1], dtype=np.int64)
    heights = -10000.0 + units.astype(np.float64) * 0.1
    encoded = bundle.encode_terrain_rgb(heights[None, :])[0]
    expected = np.stack([(units >> 16) & 255, (units >> 8) & 255, units & 255], axis=-1)
    assert encoded.tolist() == expected.tolist()


def test_terrain_rgb_rejects_nan() -> None:
    heights = np.array([[0.0, float("nan")]], dtype=np.float32)
    with pytest.raises(ValueError):
        bundle.encode_terrain_rgb(heights)


def test_terrain_rgb_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        bundle.encode_terrain_rgb(np.array([[-20000.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        bundle.encode_terrain_rgb(np.array([[2_000_000.0]], dtype=np.float32))


def test_decode_terrain_rgb_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        bundle.decode_terrain_rgb(np.zeros((2, 2), dtype=np.uint8))


# ---------------------------------------------------------------------------
# write_tile_pyramid
# ---------------------------------------------------------------------------


def _source_footprint() -> bundle.Footprint:
    return bundle.Footprint(west=-84.0, south=34.5, east=-83.0, north=35.5)


def test_write_tile_pyramid_terrain(tmp_path: Path) -> None:
    footprint = _source_footprint()
    source = np.full((64, 64), 500.0, dtype=np.float32)
    source[0, 0] = 600.0
    report = bundle.write_tile_pyramid(tmp_path, "terrain", source, footprint, zooms=[10])
    assert report.written
    assert all(z == 10 for z, _, _ in report.skipped_partial)
    z, x, y = report.written[0]
    tile_path = tmp_path / "terrain" / str(z) / str(x) / f"{y}.png"
    assert tile_path.is_file()
    with Image.open(tile_path) as img:
        assert img.size == (256, 256)
        arr = np.asarray(img.convert("RGB"))
    decoded = bundle.decode_terrain_rgb(arr)
    assert np.all(np.abs(decoded - 500.0) < 50.0)


def test_write_tile_pyramid_imagery_bilinear(tmp_path: Path) -> None:
    footprint = _source_footprint()
    source = np.zeros((64, 64, 3), dtype=np.uint8)
    source[:, :, 0] = 200
    report = bundle.write_tile_pyramid(
        tmp_path, "imagery", source, footprint, zooms=[10], resample="bilinear"
    )
    assert report.written
    z, x, y = report.written[0]
    tile_path = tmp_path / "imagery" / str(z) / str(x) / f"{y}.jpg"
    assert tile_path.is_file()
    with Image.open(tile_path) as img:
        assert img.size == (256, 256)
        assert img.format == "JPEG"


def test_write_tile_pyramid_imagery_lanczos(tmp_path: Path) -> None:
    footprint = _source_footprint()
    source = np.zeros((64, 64, 3), dtype=np.uint8)
    source[:, :, 1] = 150
    report = bundle.write_tile_pyramid(
        tmp_path, "imagery", source, footprint, zooms=[10], resample="lanczos"
    )
    assert report.written


def test_write_tile_pyramid_skips_partial_tiles(tmp_path: Path) -> None:
    # z=0 covers the whole world in one tile; the tiny source footprint
    # cannot fully cover it, so it must be skipped rather than faked.
    footprint = _source_footprint()
    source = np.full((8, 8), 100.0, dtype=np.float32)
    report = bundle.write_tile_pyramid(tmp_path, "terrain", source, footprint, zooms=[0])
    assert report.written == []
    assert report.skipped_partial == [(0, 0, 0)]
    assert not (tmp_path / "terrain").exists()


def test_write_tile_pyramid_rejects_bad_kind(tmp_path: Path) -> None:
    footprint = _source_footprint()
    source = np.zeros((8, 8), dtype=np.float32)
    with pytest.raises(ValueError):
        bundle.write_tile_pyramid(tmp_path, "bogus", source, footprint, zooms=[6])


def test_write_tile_pyramid_rejects_bad_resample(tmp_path: Path) -> None:
    footprint = _source_footprint()
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        bundle.write_tile_pyramid(
            tmp_path, "imagery", source, footprint, zooms=[6], resample="bogus"
        )


def test_write_tile_pyramid_extent_restricts_written_tiles(tmp_path: Path) -> None:
    """A narrower ``extent`` writes only the tiles inside it, even when
    the source raster fully covers more of its own footprint."""
    footprint = bundle.Footprint(west=-84.0, south=34.0, east=-82.0, north=36.0)
    source = np.full((64, 64, 3), 10, dtype=np.uint8)
    full_report = bundle.write_tile_pyramid(tmp_path / "full", "imagery", source, footprint, [10])
    narrow_extent = bundle.Footprint(west=-83.6, south=34.4, east=-82.4, north=35.6)
    narrow_report = bundle.write_tile_pyramid(
        tmp_path / "narrow", "imagery", source, footprint, [10], extent=narrow_extent
    )
    assert len(narrow_report.written) < len(full_report.written)
    assert set(narrow_report.written).issubset(set(full_report.written))


class _AstypeCountingArray(np.ndarray):
    """Counts ``astype`` calls made on an array of the full source shape."""

    full_shape: tuple[int, ...] = ()
    full_copies = 0

    def astype(self, *args, **kwargs):  # type: ignore[override]
        if self.shape == _AstypeCountingArray.full_shape:
            _AstypeCountingArray.full_copies += 1
        return super().astype(*args, **kwargs)


@pytest.mark.parametrize(
    ("kind", "source"),
    [
        ("terrain", np.full((64, 64), 500.0, dtype=np.float32)),
        ("imagery", np.full((64, 64, 3), 10, dtype=np.uint8)),
    ],
)
def test_write_tile_pyramid_never_copies_the_whole_source_per_tile(
    tmp_path: Path, kind: str, source: np.ndarray
) -> None:
    """Sampling one 256x256 tile must only convert the pixels it reads. A
    whole-raster float64 copy per tile made a regional Copernicus DEM
    (3600 px/degree in production) cost minutes per band -- enough to push a
    CI leg past its timeout."""
    _AstypeCountingArray.full_shape = source.shape
    _AstypeCountingArray.full_copies = 0
    report = bundle.write_tile_pyramid(
        tmp_path, kind, source.view(_AstypeCountingArray), _source_footprint(), zooms=[10]
    )
    assert len(report.written) > 1
    assert _AstypeCountingArray.full_copies == 0


@pytest.mark.parametrize(
    "source",
    [
        np.random.default_rng(3).uniform(-50.0, 2500.0, (97, 131)).astype(np.float32),
        np.random.default_rng(4).integers(0, 256, (97, 131, 3), dtype=np.uint8),
    ],
)
def test_tile_sampling_on_separable_axes_is_bit_identical(source: np.ndarray) -> None:
    """A tile's lon depends only on its column and its lat only on its row, so
    sampling gathers on the two 1-D axes instead of two full 256x256 index
    grids (~40 s of a full-terrain prepare). Every written tile must be the
    same float64 array, bit for bit, as sampling the full meshgrid."""
    footprint = _source_footprint()
    for z in (9, 10, 12):
        for x, y in bundle.tiles_covering(footprint, z):
            if not bundle._fully_covered(bundle.tile_bounds(z, x, y), footprint):
                continue
            lon, lat = bundle._tile_lonlat_grid(z, x, y)
            row, col = bundle._to_source_rowcol(lon, lat, footprint, source.shape[:2])
            expected = bundle._bilinear(source, row, col)
            got = bundle._sample_tile_bilinear(source, footprint, z, x, y)
            assert got.dtype == expected.dtype
            assert np.array_equal(got, expected), (z, x, y)


def test_written_tiles_footprint_unions_the_written_tiles(tmp_path: Path) -> None:
    footprint = _source_footprint()
    source = np.full((64, 64, 3), 10, dtype=np.uint8)
    report = bundle.write_tile_pyramid(tmp_path, "imagery", source, footprint, [10])
    extent = bundle.written_tiles_footprint(report, 10)
    assert extent is not None
    for _z, x, y in report.written:
        bounds = bundle.tile_bounds(10, x, y)
        assert extent.west <= bounds.west and extent.east >= bounds.east
        assert extent.south <= bounds.south and extent.north >= bounds.north


def test_written_tiles_footprint_none_when_nothing_written(tmp_path: Path) -> None:
    footprint = _source_footprint()
    source = np.full((8, 8, 3), 10, dtype=np.uint8)
    report = bundle.write_tile_pyramid(tmp_path, "imagery", source, footprint, [0])
    assert bundle.written_tiles_footprint(report, 0) is None


def test_write_tile_pyramid_is_deterministic(tmp_path: Path) -> None:
    footprint = _source_footprint()
    source = np.full((64, 64), 500.0, dtype=np.float32)
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    bundle.write_tile_pyramid(out_a, "terrain", source, footprint, zooms=[10])
    bundle.write_tile_pyramid(out_b, "terrain", source, footprint, zooms=[10])
    z, x, y = bundle.write_tile_pyramid(out_a, "terrain", source, footprint, zooms=[10]).written[0]
    a_bytes = (out_a / "terrain" / str(z) / str(x) / f"{y}.png").read_bytes()
    b_bytes = (out_b / "terrain" / str(z) / str(x) / f"{y}.png").read_bytes()
    assert a_bytes == b_bytes


# ---------------------------------------------------------------------------
# manifest / verify_bundle
# ---------------------------------------------------------------------------


def _write_minimal_bundle(root: Path, *, zoom: int = 6) -> None:
    (root / "labels.geojson").write_text('{"type": "FeatureCollection", "features": []}\n')
    (root / "route.geojson").write_text('{"type": "FeatureCollection", "features": []}\n')
    # A manifest may only declare zooms the bundle actually holds, so a
    # minimal bundle carries one real tile per kind at the declared zoom.
    for kind, suffix in (("imagery", "jpg"), ("terrain", "png")):
        tile_dir = root / kind / str(zoom) / "0"
        tile_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (bundle.TILE_SIZE, bundle.TILE_SIZE), (12, 34, 56)).save(
            tile_dir / f"0.{suffix}"
        )


def test_write_manifest_and_verify_roundtrip(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    manifest_path = bundle.write_manifest(
        tmp_path,
        {
            "footprint": {"west": -84.0, "south": 34.5, "east": -83.0, "north": 35.5},
            "imagery": {
                "minzoom": 6,
                "maxzoom": 6,
                "tile_size": 256,
                "format": "jpg",
                "sources": [],
            },
            "terrain": {
                "minzoom": 6,
                "maxzoom": 6,
                "tile_size": 256,
                "encoding": "mapbox",
                "sources": [],
                "vertical_datum": "EGM96",
            },
        },
    )
    assert manifest_path.name == "manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    manifest = json.loads(text)
    assert manifest["schema"] == "route-story-bundle/1"
    assert "labels.geojson" in manifest["files"]
    assert bundle.verify_bundle(tmp_path) == []


def test_write_manifest_is_deterministic(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    _write_minimal_bundle(root_a)
    _write_minimal_bundle(root_b)
    fields = {"footprint": {"west": -84.0, "south": 34.5, "east": -83.0, "north": 35.5}}
    bundle.write_manifest(root_a, fields)
    bundle.write_manifest(root_b, fields)
    assert (root_a / "manifest.json").read_bytes() == (root_b / "manifest.json").read_bytes()


def test_verify_bundle_missing_manifest(tmp_path: Path) -> None:
    problems = bundle.verify_bundle(tmp_path)
    assert any("manifest.json" in p for p in problems)


def test_verify_bundle_missing_file(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    bundle.write_manifest(tmp_path, {})
    (tmp_path / "labels.geojson").unlink()
    problems = bundle.verify_bundle(tmp_path)
    assert any("missing file" in p and "labels.geojson" in p for p in problems)


def test_verify_bundle_checksum_mismatch(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    bundle.write_manifest(tmp_path, {})
    (tmp_path / "labels.geojson").write_text("tampered")
    problems = bundle.verify_bundle(tmp_path)
    assert any("checksum mismatch" in p for p in problems)


def test_verify_bundle_extra_file(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    bundle.write_manifest(tmp_path, {})
    (tmp_path / "route.geojson").unlink()
    (tmp_path / "extra.txt").write_text("surprise")
    problems = bundle.verify_bundle(tmp_path)
    assert any("extra file" in p and "extra.txt" in p for p in problems)


def test_verify_bundle_flags_imagery_extent_wider_than_terrain(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    bundle.write_manifest(
        tmp_path,
        {
            "imagery": {
                "minzoom": 6,
                "maxzoom": 6,
                "extent": {"west": -84.0, "south": 34.0, "east": -83.0, "north": 35.0},
            },
            "terrain": {
                "minzoom": 6,
                "maxzoom": 6,
                "extent": {"west": -83.9, "south": 34.1, "east": -83.1, "north": 34.9},
            },
        },
    )
    problems = bundle.verify_bundle(tmp_path)
    assert any("imagery extent" in p for p in problems)


def test_verify_bundle_flags_a_band_whose_written_zooms_exceed_its_declared_ones(
    tmp_path: Path,
) -> None:
    """A band's two zoom fields are read by two different surfaces.

    ``satellite`` builds the opening raster from ``written_zoom_range``; the
    scene style is built from the band's planned ``zooms``. Nothing compared
    them, so a bundle patched by hand could ship a band claiming z9-11 while
    recording that it wrote z10-12, and verify clean.
    """
    _write_minimal_bundle(tmp_path)
    bundle.write_manifest(
        tmp_path,
        {
            "imagery": {
                "minzoom": 6,
                "maxzoom": 6,
                "bands": [
                    {
                        "source": "naip_coarse",
                        "zooms": [9, 11],
                        "written_zoom_range": {"minzoom": 10, "maxzoom": 12},
                    }
                ],
            },
        },
    )
    problems = bundle.verify_bundle(tmp_path)
    assert any("naip_coarse" in p and "10-12" in p for p in problems)


def test_verify_bundle_accepts_a_band_that_wrote_fewer_zooms_than_planned(
    tmp_path: Path,
) -> None:
    """Writing a SUBSET is normal: a partly covered zoom is skipped by design."""
    _write_minimal_bundle(tmp_path)
    bundle.write_manifest(
        tmp_path,
        {
            "imagery": {
                "minzoom": 6,
                "maxzoom": 6,
                "bands": [
                    {
                        "source": "gibs",
                        "zooms": [5, 8],
                        "written_zoom_range": {"minzoom": 6, "maxzoom": 8},
                    }
                ],
            },
        },
    )
    assert bundle.verify_bundle(tmp_path) == []


def test_verify_bundle_accepts_imagery_extent_within_terrain(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    bundle.write_manifest(
        tmp_path,
        {
            "imagery": {
                "minzoom": 6,
                "maxzoom": 6,
                "extent": {"west": -83.9, "south": 34.1, "east": -83.1, "north": 34.9},
            },
            "terrain": {
                "minzoom": 6,
                "maxzoom": 6,
                "extent": {"west": -84.0, "south": 34.0, "east": -83.0, "north": 35.0},
            },
        },
    )
    assert bundle.verify_bundle(tmp_path) == []


def test_verify_bundle_flags_duplicate_route_features(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    line_feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
    }
    (tmp_path / "route.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": [line_feature, line_feature]})
    )
    bundle.write_manifest(tmp_path, {})
    problems = bundle.verify_bundle(tmp_path)
    assert any("duplicate feature" in p for p in problems)


def test_verify_bundle_schema_mismatch(tmp_path: Path) -> None:
    _write_minimal_bundle(tmp_path)
    bundle.write_manifest(tmp_path, {})
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema"] = "route-story-bundle/999"
    manifest_path.write_text(json.dumps(manifest))
    problems = bundle.verify_bundle(tmp_path)
    assert any("schema" in p for p in problems)


# ---------------------------------------------------------------------------
# bundle_dir_for
# ---------------------------------------------------------------------------


def test_bundle_dir_for_deterministic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache"))
    fp = bundle.Footprint(west=-84.0, south=34.5, east=-83.0, north=35.5)
    dir_a = bundle.bundle_dir_for(fp, {"osm": "v1", "srtm": "v2"})
    dir_b = bundle.bundle_dir_for(fp, {"srtm": "v2", "osm": "v1"})
    assert dir_a == dir_b
    assert dir_a.parent == tmp_path / "cache"
    assert len(dir_a.name) == 16


def test_bundle_dir_for_differs_by_source_versions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache"))
    fp = bundle.Footprint(west=-84.0, south=34.5, east=-83.0, north=35.5)
    dir_a = bundle.bundle_dir_for(fp, {"osm": "v1"})
    dir_b = bundle.bundle_dir_for(fp, {"osm": "v2"})
    assert dir_a != dir_b


def test_bundle_dir_for_default_exaggeration_matches_omitted_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache"))
    fp = bundle.Footprint(west=-84.0, south=34.5, east=-83.0, north=35.5)
    dir_default = bundle.bundle_dir_for(fp, {"osm": "v1"})
    dir_explicit_default = bundle.bundle_dir_for(fp, {"osm": "v1"}, terrain_exaggeration=1.0)
    assert dir_default == dir_explicit_default


def test_bundle_dir_for_differs_by_terrain_exaggeration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("GMS_ROUTE_STORY_CACHE_DIR", str(tmp_path / "cache"))
    fp = bundle.Footprint(west=-84.0, south=34.5, east=-83.0, north=35.5)
    dir_a = bundle.bundle_dir_for(fp, {"osm": "v1"}, terrain_exaggeration=1.0)
    dir_b = bundle.bundle_dir_for(fp, {"osm": "v1"}, terrain_exaggeration=1.25)
    assert dir_a != dir_b


def _bundle_with_declared_zooms(root: Path, *, declared: tuple[int, int], present: int) -> Path:
    """A bundle whose manifest declares an imagery zoom range wider than its tiles."""
    tile_dir = root / "imagery" / str(present) / "0"
    tile_dir.mkdir(parents=True)
    Image.new("RGB", (bundle.TILE_SIZE, bundle.TILE_SIZE), (10, 20, 30)).save(
        tile_dir / "0.jpg", quality=92
    )
    bundle.write_manifest(
        root,
        {
            "imagery": {
                "minzoom": declared[0],
                "maxzoom": declared[1],
                "tile_size": bundle.TILE_SIZE,
                "format": "jpg",
            }
        },
    )
    return root


def test_verify_bundle_flags_a_declared_zoom_with_no_tiles(tmp_path: Path) -> None:
    root = _bundle_with_declared_zooms(tmp_path / "b", declared=(11, 13), present=13)

    problems = bundle.verify_bundle(root)

    assert any("imagery" in problem and "zoom 11" in problem for problem in problems)
    assert any("zoom 12" in problem for problem in problems)


def test_verify_bundle_accepts_a_manifest_that_declares_only_what_it_holds(
    tmp_path: Path,
) -> None:
    root = _bundle_with_declared_zooms(tmp_path / "b", declared=(13, 13), present=13)

    assert bundle.verify_bundle(root) == []
