"""Offline satellite-raster access for Route Story openings."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from .bundle import lonlat_to_tile, tile_bounds
from .opening import CameraPose

logger = logging.getLogger(__name__)

# Widest mosaic built for one pyramid level; a 32-tile row.
OPENING_MOSAIC_MAX_WIDTH_PX = 8192

# The ladder's coarsest band (`bands.plan_imagery_bands`, z2-4). It exists to
# sit BEHIND the map at the very top of the zoom, where nothing else covers
# the frame -- it is a backdrop, never an establishing shot of anywhere. Named
# here because `layers._country_framing` has to be able to tell it apart: its
# z4 mosaic is ~4096 px wide, so any test of "is this raster sharp enough to
# fill the frame" passes it, and it then brings its whole-world span along.
BACKDROP_BAND = "blue_marble"


@dataclass(frozen=True)
class SatelliteRaster:
    """A credited local geographic raster; rendering never fetches tiles."""

    source: Path
    west: float
    south: float
    east: float
    north: float
    credit: str
    # Which imagery band this came from, when it came from a bundle at all.
    # `None` is the flat --national-raster/--local-raster path, which has no
    # ladder behind it -- so an unlabelled raster is always a candidate, and
    # only a raster KNOWN to be the backdrop is held back.
    band: str | None = None

    def __post_init__(self) -> None:
        source = Path(self.source)
        if source.is_symlink() or not source.is_file():
            raise ValueError("satellite source must be a regular image file")
        if not self.west < self.east or not self.south < self.north:
            raise ValueError("satellite geographic bounds must have positive area")
        if not self.credit.strip():
            raise ValueError("satellite imagery needs a source credit")
        try:
            with Image.open(source) as image:
                image.verify()
        except (OSError, SyntaxError) as exc:
            raise ValueError("satellite source must be a readable image") from exc

    def view(self, pose: CameraPose, width: int, height: int) -> Image.Image:
        """Render a local raster crop at a deterministic camera pose."""
        if width <= 0 or height <= 0:
            raise ValueError("satellite output dimensions must be positive")
        with Image.open(self.source) as opened:
            image = opened.convert("RGB")
        degrees_wide = 360.0 * width / (512.0 * 2**pose.zoom)
        degrees_high = degrees_wide * height / width
        half_w, half_h = degrees_wide / 2, degrees_high / 2
        left, right = pose.longitude - half_w, pose.longitude + half_w
        top, bottom = pose.latitude + half_h, pose.latitude - half_h
        left, right = max(left, self.west), min(right, self.east)
        top, bottom = min(top, self.north), max(bottom, self.south)
        if left >= right or bottom >= top:
            raise ValueError("satellite camera view does not overlap local imagery")
        x0 = round((left - self.west) / (self.east - self.west) * image.width)
        x1 = round((right - self.west) / (self.east - self.west) * image.width)
        y0 = round((self.north - top) / (self.north - self.south) * image.height)
        y1 = round((self.north - bottom) / (self.north - self.south) * image.height)
        return image.crop((x0, y0, x1, y1)).resize((width, height), Image.Resampling.LANCZOS)

    def receipt(self) -> dict[str, object]:
        """Return provenance facts needed to reproduce a private render."""
        sha256 = self._sha256()
        with Image.open(self.source) as image:
            dimensions = [image.width, image.height]
        return {
            "sha256": sha256,
            "dimensions": dimensions,
            "bounds": [self.west, self.south, self.east, self.north],
            "credit": self.credit,
        }

    def prepare(self, cache_root: Path) -> SatelliteRaster:
        """Copy this approved image once to an immutable content-addressed cache."""
        sha256 = self._sha256()
        cache_root = Path(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        destination = cache_root / f"{sha256}{self.source.suffix.lower()}"
        if not destination.exists():
            temporary = cache_root / f".{sha256}.tmp"
            shutil.copyfile(self.source, temporary)
            temporary.replace(destination)
        return replace(self, source=destination)

    def _sha256(self) -> str:
        with self.source.open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()


def bundle_opening_rasters(
    bundle: Path, workdir: Path, *, max_width_px: int = OPENING_MOSAIC_MAX_WIDTH_PX
) -> tuple[SatelliteRaster, ...]:
    """One credited raster per complete zoom level of a scene bundle, coarse to fine.

    Every zoom in a band shares that band's bbox by construction
    (``bands.plan_imagery_bands``), so a band with two complete zoom levels on
    disk is two same-bbox rasters at different resolutions, not one -- the
    opening camera (``layers._viewport_raster``) picks the sharpest raster
    that still fully covers its viewport, so it can fall back to a coarser
    zoom of the SAME band instead of dropping to the next, much coarser band
    (the "dark and fuzzy when zooming in" defect: with only the finest zoom
    per band emitted, a span too narrow for a band's finest zoom but wider
    than the next band's bbox had no covering raster to hand off to at all).
    Tiles are Web Mercator; the rows are respaced to even latitude, which is
    how the opening camera reads a raster. A zoom missing even one tile is
    left out of its band: the opening never shows a hole.
    """
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    workdir.mkdir(parents=True, exist_ok=True)
    rasters = []
    for band in manifest.get("imagery", {}).get("bands", ()):
        zooms = band["written_zoom_range"]
        band_had_level = False
        for zoom in range(int(zooms["minzoom"]), int(zooms["maxzoom"]) + 1):
            level = _complete_level(bundle / "imagery" / str(zoom), max_width_px)
            if level is None:
                continue
            band_had_level = True
            columns, rows = level
            north_west = tile_bounds(zoom, columns[0], rows[0])
            south_east = tile_bounds(zoom, columns[-1], rows[-1])
            mosaic = _even_latitude(
                _mosaic(bundle / "imagery" / str(zoom), columns, rows),
                zoom,
                rows[0],
                north_west.north,
                south_east.south,
            )
            source = workdir / f"opening-z{zoom}.jpg"
            Image.fromarray(mosaic, "RGB").save(source, quality=92)
            rasters.append(
                SatelliteRaster(
                    source,
                    north_west.west,
                    south_east.south,
                    south_east.east,
                    north_west.north,
                    str(band["provenance"]["attribution"]),
                    band=str(band.get("source")) if band.get("source") else None,
                )
            )
        if not band_had_level:
            logger.warning(
                "opening pyramid: band %s has no complete zoom level of at most %d px; left out",
                band.get("source", "?"),
                max_width_px,
            )
    return tuple(rasters)


def _complete_level(level: Path, max_width_px: int) -> tuple[list[int], list[int]] | None:
    """The contiguous tile columns and rows of a zoom directory, or None."""
    if not level.is_dir():
        return None
    columns = sorted(int(entry.name) for entry in level.iterdir() if entry.name.isdigit())
    if not columns or len(columns) * 256 > max_width_px:
        return None
    rows = sorted(
        {int(tile.stem) for column in columns for tile in (level / str(column)).iterdir()}
    )
    if columns != list(range(columns[0], columns[-1] + 1)):
        return None
    if not rows or rows != list(range(rows[0], rows[-1] + 1)):
        return None
    if any(not (level / str(x) / f"{y}.jpg").is_file() for x in columns for y in rows):
        return None
    return columns, rows


def _mosaic(level: Path, columns: list[int], rows: list[int]) -> np.ndarray:
    canvas = np.zeros((len(rows) * 256, len(columns) * 256, 3), dtype=np.uint8)
    for column_index, x in enumerate(columns):
        for row_index, y in enumerate(rows):
            with Image.open(level / str(x) / f"{y}.jpg") as tile:
                pixels = np.asarray(tile.convert("RGB").resize((256, 256)))
            canvas[
                row_index * 256 : (row_index + 1) * 256,
                column_index * 256 : (column_index + 1) * 256,
            ] = pixels
    return _equalize_seams(canvas, 256, len(columns), len(rows))


# Width, on each side of a tile seam, over which a tone correction is blended in.
# A NAIP mosaic tile is 256px; this is deliberately a small fraction of that, so
# the correction never reaches into a tile's own interior.
_SEAM_FEATHER_PX = 16


def _equalize_seams(canvas: np.ndarray, tile_px: int, columns: int, rows: int) -> np.ndarray:
    """Tone-match adjoining source tiles across each interior mosaic seam.

    Every NAIP acquisition tile carries its own exposure/white-balance, which
    without correction shows up as a hard rectangular brightness/colour step at
    every tile boundary (see the mosaic comment in ``layers.py``). For each pair
    of tiles that share a seam, this solves the per-channel offset that would
    equalise their two (whole-tile, pre-correction) means, then blends that
    offset in -- confined to that tile pair's own footprint -- from full
    strength at the seam to zero ``_SEAM_FEATHER_PX`` away from it. A tile's own
    interior, more than that band from any seam, is left bit-for-bit alone, so
    real in-tile texture and detail survive.
    """
    if columns <= 0 or rows <= 0:
        return canvas
    equalized = canvas.astype(np.float64)
    feather = max(1, min(_SEAM_FEATHER_PX, tile_px // 2))
    # 1/feather .. 1.0: full correction lands exactly on the seam-adjacent pixel.
    ramp = (np.arange(feather, dtype=np.float64) + 1) / feather

    def _tile(row_index: int, column_index: int) -> np.ndarray:
        return canvas[
            row_index * tile_px : (row_index + 1) * tile_px,
            column_index * tile_px : (column_index + 1) * tile_px,
        ].astype(np.float64)

    tile_mean = np.array(
        [[_tile(r, c).mean(axis=(0, 1)) for c in range(columns)] for r in range(rows)]
    )

    for row_index in range(rows):
        rows_slice = slice(row_index * tile_px, (row_index + 1) * tile_px)
        for column_index in range(1, columns):
            x = column_index * tile_px
            left_mean, right_mean = tile_mean[row_index, column_index - 1 : column_index + 1]
            target = (left_mean + right_mean) / 2.0
            equalized[rows_slice, x - feather : x, :] += (target - left_mean) * ramp[None, :, None]
            equalized[rows_slice, x : x + feather, :] += (target - right_mean) * ramp[::-1][
                None, :, None
            ]

    for column_index in range(columns):
        columns_slice = slice(column_index * tile_px, (column_index + 1) * tile_px)
        for row_index in range(1, rows):
            y = row_index * tile_px
            top_mean, bottom_mean = tile_mean[row_index - 1 : row_index + 1, column_index]
            target = (top_mean + bottom_mean) / 2.0
            equalized[y - feather : y, columns_slice, :] += (target - top_mean) * ramp[
                :, None, None
            ]
            equalized[y : y + feather, columns_slice, :] += (target - bottom_mean) * ramp[::-1][
                :, None, None
            ]

    return np.clip(equalized, 0, 255).astype(np.uint8)


def _even_latitude(
    mosaic: np.ndarray, zoom: int, first_row: int, north: float, south: float
) -> np.ndarray:
    """Respace Mercator rows so latitude runs evenly from north to south."""
    height = mosaic.shape[0]
    latitudes = north + (south - north) * (np.arange(height) + 0.5) / height
    source_rows = [
        (lonlat_to_tile(0.0, float(latitude), zoom)[1] - first_row) * 256 for latitude in latitudes
    ]
    index = np.clip(np.floor(source_rows).astype(int), 0, height - 1)
    return mosaic[index]


def opening_pyramid_receipt(rasters: tuple[SatelliteRaster, ...]) -> list[dict[str, object]]:
    """What the opening drew from the bundle: zoom, credit and bounds per level."""
    return [
        {
            "zoom": int(raster.source.stem.removeprefix("opening-z")),
            "credit": raster.credit,
            "bounds": [raster.west, raster.south, raster.east, raster.north],
        }
        for raster in rasters
    ]
