"""Tests for chapter terrain-band decoration."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

try:
    import gps_journey  # noqa: F401

    HAS_GPS = True
except ImportError:
    HAS_GPS = False

if not HAS_GPS:
    pytest.skip("gps_journey package not installed", allow_module_level=True)

from gps_journey.core.models import Journey, Location, Waypoint  # noqa: E402

from gomoveshift.config import Config  # noqa: E402
from gomoveshift.pipeline.chapters import ChapterGenerator  # noqa: E402
from gomoveshift.pipeline.chapters.generator import _classify_chapter_band  # noqa: E402
from gomoveshift.pipeline.chapters.marker_merger import MarkerMerger  # noqa: E402

T0 = datetime(2026, 1, 1, 12, 0, 0)


def _wp(lat, lon, t, elev=None):
    return Waypoint(latitude=lat, longitude=lon, timestamp=t, elevation=elev)


def _loc(city, lat=52.0, lon=-9.0):
    return Location(latitude=lat, longitude=lon, city=city, county="X")


class TestClassifyChapterBand:
    def test_mountain(self):
        # 200m climb over ~0.5 km → 400 m/km ≫ 30 m/km threshold
        wps = [
            _wp(52.0 + i * 0.001, -9.0, T0 + timedelta(seconds=i * 20), elev=100.0 + 50 * i)
            for i in range(5)
        ]
        vts = [i * 20.0 for i in range(5)]
        assert _classify_chapter_band(wps, vts, 0.0, 80.0, []) == "mountain"

    def test_flat_coastal_noise_not_mountain(self):
        # A FLAT drive whose GPS elevation oscillates ±3 m around a constant.
        # Over many closely-spaced fixes the summed positive deltas accumulate
        # into a large bogus gain_per_km (≫ 30 m/km) while the NET range stays
        # tiny (~6 m). The net-range guard must keep this out of "mountain".
        noise = [0.0, 3.0, -3.0, 2.5, -2.5, 3.0, -3.0, 2.0, -2.0, 3.0, -3.0, 2.5]
        wps = [
            _wp(
                52.0 + i * 0.00002,  # ~2.2 m/step → ~24 m total, gain_per_km blows up
                -9.0,
                T0 + timedelta(seconds=i * 5),
                elev=10.0 + noise[i],
            )
            for i in range(len(noise))
        ]
        vts = [i * 5.0 for i in range(len(noise))]
        # Sanity: the OLD summed-positive metric would exceed 30 m/km here.
        band = _classify_chapter_band(wps, vts, 0.0, 60.0, [])
        assert band != "mountain"

    def test_sustained_climb_still_mountain(self):
        # A genuine monotonic climb of >100 m net over the slice still
        # classifies as "mountain" (net range ~250 m, gain_per_km ≫ 30).
        wps = [
            _wp(52.0 + i * 0.0008, -9.0, T0 + timedelta(seconds=i * 20), elev=100.0 + 25.0 * i)
            for i in range(11)
        ]
        vts = [i * 20.0 for i in range(11)]
        assert _classify_chapter_band(wps, vts, 0.0, 220.0, []) == "mountain"

    def test_motorway(self):
        # ~33 m/s over flat terrain → motorway
        wps = [
            _wp(52.0 + i * 0.0003, -9.0, T0 + timedelta(seconds=i), elev=10.0) for i in range(10)
        ]
        vts = [float(i) for i in range(10)]
        assert _classify_chapter_band(wps, vts, 0.0, 9.0, []) == "motorway"

    def test_coastal(self):
        wps = [
            _wp(52.0 + i * 0.0001, -9.0, T0 + timedelta(seconds=i * 20), elev=10.0)
            for i in range(3)
        ]
        vts = [i * 20.0 for i in range(3)]
        assert _classify_chapter_band(wps, vts, 0.0, 40.0, [(52.0001, -9.0)]) == "coastal"

    def test_scenic_default_and_missing_elevation(self):
        # Moderate speed, near-zero gain, no coastline → None
        wps = [
            _wp(52.0 + i * 0.002, -9.0, T0 + timedelta(seconds=i * 30), elev=10.0 + i * 0.1)
            for i in range(6)
        ]
        vts = [i * 30.0 for i in range(6)]
        assert _classify_chapter_band(wps, vts, 0.0, 150.0, []) is None
        # No elevation data → no raise, band still None for slow flat drive
        bare = [_wp(52.0, -9.0, T0), _wp(52.001, -9.0, T0 + timedelta(seconds=60))]
        assert _classify_chapter_band(bare, [0.0, 60.0], 0.0, 60.0, []) is None
        # Empty input → None (no raise)
        assert _classify_chapter_band([], [], 0.0, 60.0, []) is None


class TestConsecutiveBandSuppression:
    def _three(self):
        return [
            (120.0, _loc("A")),
            (240.0, _loc("B", 52.1, -9.1)),
            (360.0, _loc("C", 52.2, -9.2)),
        ]

    def _merge(self, bands=None, **kw):
        return MarkerMerger().merge_markers_with_chapters(
            waypoints_with_towns=self._three(),
            markers=None,
            video_duration=600.0,
            start_location="Start",
            end_location="End",
            terrain_bands=bands,
            **kw,
        )

    def test_three_same_band_only_first_decorated(self):
        chapters = self._merge(bands=["mountain", "mountain", "mountain"])
        mid = [c for c in chapters if "Start" not in c["title"] and "Finish" not in c["title"]]
        assert len(mid) == 3
        assert "Mountain Climb" in mid[0]["title"]
        assert "Mountain Climb" not in mid[1]["title"]
        assert "Mountain Climb" not in mid[2]["title"]

    def test_no_bands_kwarg_keeps_legacy(self):
        merger = MarkerMerger()
        chapters = merger.merge_markers_with_chapters(
            waypoints_with_towns=[(120.0, _loc("A"))],
            markers=None,
            video_duration=600.0,
            start_location="Start",
            end_location="End",
        )
        mid = [c for c in chapters if "Start" not in c["title"] and "Finish" not in c["title"]]
        assert len(mid) == 1
        assert "—" not in mid[0]["title"]


class TestCoastlineQueriedOncePerDrive:
    def test_called_once_for_three_chapters(self, tmp_path):
        cfg = Config(
            project_name="test",
            base_input_path=Path("/tmp/test_input"),
            base_output_path=Path("/tmp/test_output"),
        )
        towns = [("A", 52.0, -9.0), ("B", 52.01, -9.01), ("C", 52.02, -9.02)]
        wps = []
        for i, (town, lat, lon) in enumerate(towns):
            loc = Location(latitude=lat, longitude=lon, city=town, county="X")
            wps.append(
                Waypoint(
                    latitude=lat,
                    longitude=lon,
                    timestamp=T0 + timedelta(minutes=i * 20),
                    elevation=10.0,
                    location=loc,
                )
            )
        journey = Journey(
            start_date=T0,
            end_date=T0 + timedelta(minutes=60),
            waypoints=tuple(wps),
            statistics=None,
        )

        with (
            patch("gomoveshift.pipeline.chapters._query_overpass", return_value={"elements": []}),
            patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls,
            patch(
                "gomoveshift.pipeline.chapters.overpass_source.post_overpass_query",
                return_value={"elements": []},
            ) as mock_coastline,
        ):
            mock_geo_cls.return_value.reverse_geocode.return_value = None
            mock_geo_cls.return_value.reverse_geocode_detailed.return_value = {}

            ChapterGenerator(cfg).generate(
                gpx_path=Path("/fake.gpx"),
                video_duration=3600,
                start_location="Start, X",
                end_location="End, X",
                _mock_journey=journey,
                stages_dir=tmp_path,
            )
            # Counts the Overpass POST helper, not urlopen: since the coastal
            # band gained mirror failover, one logical query can span several
            # urlopen calls while still being one coastline query per drive.
            assert mock_coastline.call_count == 1, (
                f"Expected 1 coastline call, got {mock_coastline.call_count}"
            )
