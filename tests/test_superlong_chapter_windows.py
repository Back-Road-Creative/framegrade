"""Superlong chapter windows: each part must chart its OWN slice of the
whole day's GPS, not one linear scale of the first GPS-bearing episode's
GPX smeared across every part.

(a) ``_produce_superlong_parts`` must build a day-master-time journey
    (from ``gps_points_by_date``, hoisted out of ``run_day_pipeline``'s
    stop-mask branch) spanning every episode of the day and thread it to
    the generator -- not just the first GPS-bearing episode's GPX.

(b) ``_build_superlong_part_metadata`` must window that journey onto
    THIS part's own ``part.ranges`` (day-master seconds) via
    ``included_segments``, so part 2 of a multi-part day names towns from
    the middle of the day and shares zero chapters with part 1 -- today
    the two parts get byte-identical chapter lists, each opening at the
    day's first town.
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

try:
    import gps_journey  # noqa: F401

    HAS_GPS = True
except ImportError:
    HAS_GPS = False

if not HAS_GPS:
    pytest.skip("gps_journey package not installed", allow_module_level=True)

from gps_journey.core.models import Journey, Location, Waypoint

from _state_helpers import make_state
from gomoveshift.config import Config
from gomoveshift.models import SuperlongPartPlan
from gomoveshift.pipeline import day_pipeline
from gomoveshift.pipeline.chapters.generator import ChapterGenerator

_EPOCH = datetime(1970, 1, 1)


@pytest.fixture(autouse=True)
def mock_network():
    with patch("gomoveshift.pipeline.chapters._query_overpass", return_value={"elements": []}):
        with patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls:
            mock_geo_cls.return_value.reverse_geocode.return_value = None
            mock_geo_cls.return_value.reverse_geocode_detailed.return_value = {}
            yield


def _town_waypoint(t_master: float) -> Waypoint:
    """A distinct named town every 300s of day-master time, 0..9000."""
    lat = 35.0 + t_master * 1e-4
    lon = -82.0 - t_master * 1e-4
    return Waypoint(
        latitude=lat,
        longitude=lon,
        timestamp=_EPOCH + timedelta(seconds=t_master),
        location=Location(
            latitude=lat,
            longitude=lon,
            city=f"Town{int(t_master):04d}",
            county="Test County",
            country="USA",
        ),
    )


@pytest.fixture
def full_day_journey() -> Journey:
    waypoints = tuple(_town_waypoint(t) for t in range(0, 9001, 300))
    return Journey(
        start_date=waypoints[0].timestamp,
        end_date=waypoints[-1].timestamp,
        waypoints=waypoints,
    )


def _config(tmp_path: Path) -> Config:
    return Config(project_name="test", base_input_path=tmp_path, base_output_path=tmp_path)


class TestDayJourneyThreadedThroughProduceSuperlongParts:
    def test_journey_covers_the_whole_day_not_one_episode(self, tmp_path, monkeypatch):
        """(a) The journey handed to the generator must span every episode
        of the day -- today it's the FIRST GPS-bearing episode's GPX only."""
        captured = {}

        def _fake_generate(self, **kwargs):
            captured.update(kwargs)
            return []

        monkeypatch.setattr(ChapterGenerator, "generate", _fake_generate)
        monkeypatch.setattr(day_pipeline, "check_superlong_output_space", lambda *a, **kw: None)

        # Three episodes' worth of GPS, ~2.5h of day-master time -- far
        # beyond what one ~50min episode's own GPX would cover.
        gps_points = [(0.0, 35.0, -82.0), (4500.0, 35.5, -82.5), (8900.0, 36.0, -83.0)]
        stop_mask = np.zeros(9000, dtype=bool)

        day_pipeline._produce_superlong_parts(
            date="2026-06-01",
            stop_mask=stop_mask,
            episodes=[],
            file_boundaries=[],  # no overlap -> no segs -> no ffmpeg needed
            master_path=tmp_path / "master.mp4",
            day_dir=tmp_path / "day",
            config=_config(tmp_path),
            location="Somewhere, Test County",
            ost_crossfade_sec=0.5,
            day_gps_points=gps_points,
            # OST audit gate: drives the burn directly, so it stands in for a reviewer.
            state=make_state(ost_audit_confirmed_this_run=True),
            # D8b: this test is about the chapter journey, not the route claim
            # (test_superlong_route_claims.py), so the check keeps the text.
            superlong_route_fn=lambda idx, ranges, pts, text: (text, "", None),
        )

        journey = captured.get("day_journey")
        assert journey is not None, "generator never received a day-master journey"
        span = (journey.waypoints[-1].timestamp - journey.waypoints[0].timestamp).total_seconds()
        assert span > 5000, f"journey spans only {span}s -- looks like one episode, not the day"


class TestPartWindowsOwnSliceOfTheDay:
    def test_part_two_shares_no_chapters_with_part_one(self, tmp_path, full_day_journey):
        """(b) Part 2 must name towns from the middle of the day and share
        ZERO chapters with part 1 -- today the lists are byte-identical."""
        part1 = SuperlongPartPlan(
            part_index=1, part_count=2, ranges=[(0.0, 3000.0)], kept_duration_sec=3000.0
        )
        part2 = SuperlongPartPlan(
            part_index=2, part_count=2, ranges=[(6000.0, 9000.0)], kept_duration_sec=3000.0
        )
        config = _config(tmp_path)

        meta1 = day_pipeline._build_superlong_part_metadata(
            part=part1,
            date="2026-06-01",
            location="Test Drive, Test County",
            gpx_path=None,
            resolved_location=None,
            config=config,
            stages_dir=tmp_path,
            day_journey=full_day_journey,
        )
        meta2 = day_pipeline._build_superlong_part_metadata(
            part=part2,
            date="2026-06-01",
            location="Test Drive, Test County",
            gpx_path=None,
            resolved_location=None,
            config=config,
            stages_dir=tmp_path,
            day_journey=full_day_journey,
        )

        # Start/Finish bookends legitimately repeat the same burned OST
        # label on every part -- the town-named chapters in between are
        # the real, GPS-derived signal under test here.
        def _intermediate(chapters):
            return {c["title"] for c in chapters if not c["title"].startswith(("Start", "Finish"))}

        titles1 = _intermediate(meta1["chapters"])
        titles2 = _intermediate(meta2["chapters"])
        assert titles1 and titles2, (meta1["chapters"], meta2["chapters"])
        shared = titles1 & titles2
        assert not shared, f"part 1/part 2 share town chapters: {shared}"
