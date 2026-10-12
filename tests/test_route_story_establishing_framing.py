"""The establishing shot is a country, never the whole globe.

The imagery ladder's coarsest band (``blue_marble``, z2-4) exists to be a
BACKDROP: something to sit behind the map at the very top of the zoom, where
no real imagery covers the frame. ``_country_framing`` did not know that. It
picked the widest raster whose own pixel width cleared the framing ratio, and
the z4 Blue Marble mosaic is ~4096 px wide -- so it cleared the bar easily and
brought its ~336 degree span with it.

The result is the defect the operator reported as "the starting shot of the
usa doesnt look like the usa": the film opened on a whole-world map with
Africa, Europe and Australia in frame, travelled across the Atlantic, and only
reached the United States three-quarters of the way down. The sharpness gate
eventually refused the render, but on a technicality (0.50 source px per
output px against a 0.5 floor) rather than on the real problem, which is that
a 336 degree span cannot read as any one country at all.

The fix is by construction rather than by threshold: the bundle's manifest
already names each band's source, so the backdrop band is excluded from the
framing candidates outright. No degree ceiling to tune, and no way for a
future band that happens to be wide and high-resolution to sneak back in.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.route_story import layers
from gomoveshift.pipeline.route_story.metrics import RoutePoint
from gomoveshift.pipeline.route_story.satellite import BACKDROP_BAND, SatelliteRaster

FRAME_WIDTH = 1920


def _raster(
    tmp_path: Path,
    name: str,
    *,
    west: float,
    south: float,
    east: float,
    north: float,
    width_px: int,
    band: str | None = None,
) -> SatelliteRaster:
    source = tmp_path / f"{name}.jpg"
    height_px = max(2, width_px // 2)
    Image.fromarray(np.full((height_px, width_px, 3), 96, dtype=np.uint8), "RGB").save(
        source, quality=90
    )
    return SatelliteRaster(source, west, south, east, north, f"{name} credit", band=band)


def _globe(tmp_path: Path) -> SatelliteRaster:
    """The z4 Blue Marble mosaic: wide, high-resolution, and the whole world."""
    return _raster(
        tmp_path,
        "blue-marble-z4",
        west=-168.0,
        south=-80.0,
        east=168.0,
        north=80.0,
        width_px=4096,
        band=BACKDROP_BAND,
    )


def _country(tmp_path: Path) -> SatelliteRaster:
    """The gibs z8 band: the recognisable slice of the country."""
    return _raster(
        tmp_path,
        "gibs-z8",
        west=-86.0,
        south=33.0,
        east=-80.0,
        north=37.5,
        width_px=3072,
        band="gibs",
    )


class TestTheBackdropBandIsNotAFraming:
    def test_the_globe_is_not_chosen_over_the_country(self, tmp_path: Path) -> None:
        """The whole bug in one assertion."""
        _lat, _lon, span = layers._country_framing(
            (_globe(tmp_path), _country(tmp_path)), FRAME_WIDTH
        )
        assert span == pytest.approx(6.0)

    def test_the_chosen_framing_could_read_as_one_country(self, tmp_path: Path) -> None:
        """A span wider than a hemisphere is not an establishing shot of
        anywhere -- whatever its pixel count."""
        _lat, _lon, span = layers._country_framing(
            (_globe(tmp_path), _country(tmp_path)), FRAME_WIDTH
        )
        assert span < 180.0

    def test_the_backdrop_is_still_used_when_it_is_all_there_is(self, tmp_path: Path) -> None:
        """Excluding the backdrop must not leave the opening with nothing to
        frame on. A bundle whose only complete level is the backdrop still
        gets an establishing shot, because the alternative is raising on a
        film that could have been made."""
        _lat, _lon, span = layers._country_framing((_globe(tmp_path),), FRAME_WIDTH)
        assert span == pytest.approx(336.0)

    def test_a_raster_with_no_band_is_still_a_candidate(self, tmp_path: Path) -> None:
        """The flat ``--national-raster``/``--local-raster`` path carries no
        band at all. Excluding everything unlabelled would silently break
        every film that does not use a bundle."""
        flat = _raster(
            tmp_path,
            "flat-national",
            west=-125.0,
            south=24.0,
            east=-66.0,
            north=50.0,
            width_px=3072,
        )
        _lat, _lon, span = layers._country_framing((flat,), FRAME_WIDTH)
        assert span == pytest.approx(59.0)


class TestTheDescentStillStartsWhereTheFramingIs:
    def test_the_camera_opens_on_the_country_framing_it_was_given(self, tmp_path: Path) -> None:
        """`_opening_camera` holds the country framing exactly at fraction 0,
        so if the framing is the country the first frame is too."""
        rasters = (_globe(tmp_path), _country(tmp_path))
        country_lat, country_lon, country_span = layers._country_framing(rasters, FRAME_WIDTH)
        target = RoutePoint(longitude=-83.5, latitude=35.25, time_s=0.0, elevation_m=600.0)
        lat, lon, span = layers._opening_camera(rasters, target, 0.0, FRAME_WIDTH)
        assert (lat, lon, span) == pytest.approx((country_lat, country_lon, country_span))

    def test_the_descent_never_widens_beyond_the_establishing_span(self, tmp_path: Path) -> None:
        rasters = (_globe(tmp_path), _country(tmp_path))
        target = RoutePoint(longitude=-83.5, latitude=35.25, time_s=0.0, elevation_m=600.0)
        _lat, _lon, opening_span = layers._opening_camera(rasters, target, 0.0, FRAME_WIDTH)
        for step in range(21):
            _lat, _lon, span = layers._opening_camera(rasters, target, step / 20, FRAME_WIDTH)
            assert span <= opening_span + 1e-9


# --- The unlabelled whole-world raster the band exclusion above cannot see ---
#
# The band exclusion shipped and the film still opened on the globe. Excluding
# by NAME only covers rasters that carry a name: the real render command also
# passes ``--national-raster``, which ``film.py`` builds with hardcoded
# ``-180, -90, 180, 90`` bounds and no band at all. ``None != "blue_marble"``,
# so it was never a backdrop as far as the filter was concerned; it cleared the
# sharpness floor on raw pixel count and won "widest" at a full 360 degrees.
# Measured against the real inputs before this fix, both the film and its own
# review pack computed a framing of ``(0.0, 0.0, 360)``.
#
# So the mechanism is wrong, not the threshold. A name is a property of where a
# raster came from; what disqualifies it is how much of the planet it covers.
# A span test needs no source to label itself honestly and closes the hole for
# every future imagery source at once.


def _unlabelled_world(tmp_path: Path) -> SatelliteRaster:
    """`--national-raster`: whole-world bounds, high resolution, no band."""
    return _raster(
        tmp_path,
        "cli-national",
        west=-180.0,
        south=-90.0,
        east=180.0,
        north=90.0,
        width_px=4096,
        band=None,
    )


def _country_band(tmp_path: Path) -> SatelliteRaster:
    """A gibs band sized to show a recognisable slice of the country."""
    return _raster(
        tmp_path,
        "gibs-z5",
        west=-101.25,
        south=18.98,
        east=-67.5,
        north=51.89,
        width_px=4096,
        band="gibs",
    )


class TestAWholeWorldRasterIsNeverTheEstablishingShot:
    def test_an_unlabelled_world_raster_does_not_win_on_pixel_count(self, tmp_path: Path) -> None:
        """The exact defect: 4096 px wide, sharp by every measure, and useless."""
        _, _, span = layers._country_framing(
            [_unlabelled_world(tmp_path), _country_band(tmp_path)], FRAME_WIDTH
        )
        assert span == pytest.approx(33.75)

    def test_it_loses_even_when_it_is_the_sharpest_thing_available(self, tmp_path: Path) -> None:
        world = _raster(
            tmp_path,
            "cli-national-huge",
            west=-180.0,
            south=-90.0,
            east=180.0,
            north=90.0,
            width_px=8192,
            band=None,
        )
        _, centre_lon, span = layers._country_framing([world, _country_band(tmp_path)], FRAME_WIDTH)
        assert span == pytest.approx(33.75)
        assert centre_lon == pytest.approx(-84.375)

    def test_a_continent_sized_raster_still_qualifies(self, tmp_path: Path) -> None:
        """The cap disqualifies the planet, not a legitimately wide country."""
        wide = _raster(
            tmp_path,
            "conus",
            west=-125.0,
            south=24.0,
            east=-66.0,
            north=49.0,
            width_px=4096,
            band="gibs",
        )
        _, _, span = layers._country_framing([wide, _country_band(tmp_path)], FRAME_WIDTH)
        assert span == pytest.approx(59.0)

    def test_with_nothing_but_world_rasters_it_takes_the_narrowest(self, tmp_path: Path) -> None:
        """Falling back to the WIDEST would reinstate the bug it just refused."""
        narrower = _raster(
            tmp_path,
            "hemisphere",
            west=-180.0,
            south=-90.0,
            east=0.0,
            north=90.0,
            width_px=4096,
            band=None,
        )
        _, _, span = layers._country_framing([_unlabelled_world(tmp_path), narrower], FRAME_WIDTH)
        assert span == pytest.approx(180.0)


# --- Asking a raster how wide it is must not decode it ---
#
# The opening asked every candidate raster for its pixel width on EVERY frame,
# and got it from ``_raster_image(source).width`` -- a full decode. That cache
# holds 8 images; a bundle film passes 18 sources (four from the command line
# plus one per pyramid level, the largest a 41 MB JPEG), so the cache thrashed
# and each of the opening's 360 frames re-decoded about ten mosaics to read ten
# integers. Measured on a real render: ~5 s per frame on one core, thirty
# minutes for a twelve-second opening, with the GPU and eleven cores idle.
#
# A width is in the file header. Reading it there is the same number, and no
# cache size can make it slow again.


class TestAWidthIsReadFromTheHeaderNotTheDecodedPixels:
    def _many(self, tmp_path: Path) -> list[SatelliteRaster]:
        """More sources than the decoded-image cache can hold."""
        return [
            _raster(
                tmp_path,
                f"level-{index}",
                west=-101.25 + index,
                south=18.98,
                east=-67.5 - index,
                north=51.89,
                width_px=2048 + 64 * index,
                band="gibs",
            )
            for index in range(12)
        ]

    def test_the_header_width_is_the_decoded_width(self, tmp_path: Path) -> None:
        raster = _country_band(tmp_path)
        assert layers._raster_width(raster.source) == layers._raster_image(raster.source).width

    def test_framing_the_camera_decodes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        rasters = self._many(tmp_path)
        target = RoutePoint(longitude=-83.5, latitude=35.25, time_s=0.0, elevation_m=600.0)
        decoded: list[Path] = []

        def _counting(source: Path) -> Image.Image:
            decoded.append(source)
            raise AssertionError(f"decoded {source.name} just to measure it")

        monkeypatch.setattr(layers, "_raster_image", _counting)
        for step in range(11):
            layers._opening_camera(rasters, target, step / 10, FRAME_WIDTH)
        # Synthetic rasters may well fail the floor; measuring them is the point.
        with contextlib.suppress(ValueError):
            layers.check_opening_sharpness(rasters, target, FRAME_WIDTH, 1080)
        layers.finest_last(rasters)
        assert decoded == []
