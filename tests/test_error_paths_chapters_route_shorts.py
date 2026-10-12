"""Error path coverage for chapters, route_context, ShortsValidator.

Three areas:

* chapters error paths: empty/malformed GPX, gps_date mismatch,
  Nominatim 429/timeout
* route_context error paths: empty location, empty chapters,
  malformed Entering, missing country
* ShortsValidator ffprobe edge cases: malformed JSON, missing
  dimensions, ffprobe crash

These tests fill gaps where the production paths
already degrade gracefully but were untested — so a future regression
that turns "warn-and-continue" into "raise" or vice versa would slip
through CI silently.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import gps_journey  # noqa: F401

    HAS_GPS = True
except ImportError:
    HAS_GPS = False


# =========================================================================
# Chapters error paths
# =========================================================================


@pytest.mark.skipif(not HAS_GPS, reason="gps_journey package not installed")
class TestChaptersErrorPaths:
    """ChapterGenerator + load_and_filter_journey degrade gracefully on
    every input class that can land in the wild: missing files,
    malformed GPX, no waypoints, gps_date matching nothing, gps_range
    matching nothing, and reverse-geocode failures.
    """

    @pytest.fixture
    def config(self):
        from gomoveshift.config import Config

        return Config(
            project_name="test",
            base_input_path=Path("/tmp/test_input"),
            base_output_path=Path("/tmp/test_output"),
        )

    @pytest.fixture
    def generator(self, config):
        from gomoveshift.pipeline.chapters import ChapterGenerator

        with patch(
            "gomoveshift.pipeline.chapters._query_overpass",
            return_value={"elements": []},
        ):
            with patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls:
                mock_geo_cls.return_value.reverse_geocode.return_value = None
                mock_geo_cls.return_value.reverse_geocode_detailed.return_value = {}
                yield ChapterGenerator(config)

    def test_malformed_gpx_returns_none(self, tmp_path, caplog):
        """A GPX file the parser can't read returns None (caller falls back)."""
        from gomoveshift.pipeline.chapters._journey_filter import load_and_filter_journey

        bad_gpx = tmp_path / "bad.gpx"
        bad_gpx.write_text("<<<not-valid-gpx>>>")

        with caplog.at_level(logging.ERROR):
            result = load_and_filter_journey(
                gpx_path=bad_gpx,
                gps_date=None,
                gps_range=None,
                _mock_journey=None,
            )

        assert result is None
        assert any("Failed to parse GPS file" in r.message for r in caplog.records)

    def test_missing_gpx_returns_none_with_warning(self, tmp_path, caplog):
        """Absent GPX file warns and returns None (no exception)."""
        from gomoveshift.pipeline.chapters._journey_filter import load_and_filter_journey

        with caplog.at_level(logging.WARNING):
            result = load_and_filter_journey(
                gpx_path=tmp_path / "absent.gpx",
                gps_date=None,
                gps_range=None,
                _mock_journey=None,
            )

        assert result is None
        assert any("GPS file not found" in r.message for r in caplog.records)

    def test_gps_date_no_match_hard_fails(self):
        """A gps_date that matches no waypoints hard-fails rather than
        silently shipping the WHOLE (possibly multi-day) journey unfiltered.
        On camera-clock skew the unfiltered fallback samples OTHER days'
        waypoints, corrupting chapters and the end location."""
        from datetime import datetime, timedelta

        from gomoveshift.pipeline.chapters._journey_filter import _filter_by_date
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        wps = [
            Waypoint(
                latitude=52.0,
                longitude=-9.0,
                timestamp=start + timedelta(minutes=i),
                location=Location(latitude=52.0, longitude=-9.0),
            )
            for i in range(3)
        ]
        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=2),
            waypoints=tuple(wps),
            statistics=None,
        )

        with pytest.raises(ValueError, match="matched no waypoints"):
            _filter_by_date(journey, "1999-01-01")

    def test_gps_date_invalid_format_hard_fails(self):
        """A malformed gps_date string hard-fails rather than silently
        falling back to the unfiltered journey."""
        from datetime import datetime, timedelta

        from gomoveshift.pipeline.chapters._journey_filter import _filter_by_date
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=1),
            waypoints=(
                Waypoint(
                    latitude=0.0,
                    longitude=0.0,
                    timestamp=start,
                    location=Location(latitude=0.0, longitude=0.0),
                ),
            ),
            statistics=None,
        )

        with pytest.raises(ValueError, match="Invalid gps_date format"):
            _filter_by_date(journey, "not-a-date")

    def test_gps_date_local_day_retains_post_utc_midnight(self):
        """A continuous local-evening drive crossing UTC midnight groups by
        LOCAL day, so the post-midnight-UTC half is retained instead of
        being truncated at UTC midnight. North Carolina waypoints
        (lon ~-80, ~UTC-5) recorded 23:30-00:30 UTC are all one local day."""
        from datetime import datetime, timedelta

        from gomoveshift.pipeline.chapters._journey_filter import _filter_by_date
        from gps_journey.core.models import Journey, Location, Waypoint

        # 23:30 UTC Sep 10 through 00:30 UTC Sep 11 — crosses UTC midnight.
        start = datetime(2025, 9, 10, 23, 30, 0)
        wps = [
            Waypoint(
                latitude=35.6,
                longitude=-80.0,
                timestamp=start + timedelta(minutes=i * 10),
                location=Location(latitude=35.6, longitude=-80.0),
            )
            for i in range(7)
        ]
        journey = Journey(
            start_date=wps[0].timestamp,
            end_date=wps[-1].timestamp,
            waypoints=tuple(wps),
            statistics=None,
        )

        # lon -80 -> round(-80/15) = -5h; local day for all waypoints is Sep 10.
        result = _filter_by_date(journey, "2025-09-10")

        # All 7 retained — none dropped at UTC midnight.
        assert len(result.waypoints) == len(journey.waypoints)

    def test_detect_gps_date_local_day_single_day_no_split(self):
        """A continuous local-evening drive crossing UTC midnight is ONE
        local day, so date detection returns None (no spurious multi-day
        split whose majority-date filter would drop the post-midnight
        half)."""
        from datetime import datetime, timedelta

        from gomoveshift.pipeline.chapters.nominatim_source import (
            detect_gps_date_for_location,
        )
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 23, 30, 0)
        wps = [
            Waypoint(
                latitude=35.6,
                longitude=-80.0,
                timestamp=start + timedelta(minutes=i * 10),
                location=Location(latitude=35.6, longitude=-80.0),
            )
            for i in range(7)
        ]
        journey = Journey(
            start_date=wps[0].timestamp,
            end_date=wps[-1].timestamp,
            waypoints=tuple(wps),
            statistics=None,
        )
        mock_parser = MagicMock()
        mock_parser.parse.return_value = journey

        with (
            patch(
                "gomoveshift.pipeline.chapters.nominatim_source._nominatim_search",
                return_value=(35.6, -80.0),
            ),
            patch(
                "gomoveshift.pipeline.chapters.nominatim_source.GPXParser",
                return_value=mock_parser,
            ),
        ):
            result = detect_gps_date_for_location(Path("/unused.gpx"), "Charlotte, NC")

        assert result is None

    def test_gps_range_invalid_format_hard_fails(self):
        """A gps_range that doesn't split into exactly START-END hard-fails
        rather than silently shipping the WHOLE journey unfiltered — mirrors
        _filter_by_date's malformed-input contract."""
        from datetime import datetime, timedelta

        from gomoveshift.pipeline.chapters._journey_filter import apply_gps_range
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=5),
            waypoints=(
                Waypoint(
                    latitude=0.0,
                    longitude=0.0,
                    timestamp=start,
                    location=Location(latitude=0.0, longitude=0.0),
                ),
            ),
            statistics=None,
        )

        with pytest.raises(ValueError, match="Invalid gps_range format"):
            apply_gps_range(journey, "60")

    def test_gps_range_invalid_values_hard_fails(self):
        """A gps_range with non-numeric START/END hard-fails rather than
        silently shipping the WHOLE journey unfiltered."""
        from datetime import datetime, timedelta

        from gomoveshift.pipeline.chapters._journey_filter import apply_gps_range
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=5),
            waypoints=(
                Waypoint(
                    latitude=0.0,
                    longitude=0.0,
                    timestamp=start,
                    location=Location(latitude=0.0, longitude=0.0),
                ),
            ),
            statistics=None,
        )

        with pytest.raises(ValueError, match="Invalid gps_range values"):
            apply_gps_range(journey, "a-b")

    def test_gps_range_no_match_returns_none(self, tmp_path, caplog):
        """gps_range that matches no waypoints returns None from
        load_and_filter_journey (caller falls back to basic chapters)."""
        from datetime import datetime, timedelta

        from gomoveshift.pipeline.chapters._journey_filter import load_and_filter_journey
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=5),
            waypoints=(
                Waypoint(
                    latitude=0.0,
                    longitude=0.0,
                    timestamp=start,
                    location=Location(latitude=0.0, longitude=0.0),
                ),
            ),
            statistics=None,
        )

        with caplog.at_level(logging.WARNING):
            result = load_and_filter_journey(
                gpx_path=Path("/unused"),
                gps_date=None,
                gps_range="60-120",  # waypoint at minute 0 — no match
                _mock_journey=journey,
            )

        assert result is None
        assert any("yielded no waypoints" in r.message for r in caplog.records)

    def test_generate_empty_waypoints_falls_back_to_basic(self, generator):
        """Journey with zero waypoints triggers basic-chapter fallback."""
        from datetime import datetime, timedelta

        from gps_journey.core.models import Journey

        start = datetime(2025, 9, 10, 10, 0, 0)
        empty_journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=10),
            waypoints=(),
            statistics=None,
        )

        chapters = generator.generate(
            gpx_path=Path("/unused"),
            video_duration=600,
            start_location="Ennis, County Clare",
            _mock_journey=empty_journey,
        )

        # Basic fallback produces Start + Finish, padded to YouTube's minimum
        # renderable shape (>=3 chapters) by the chapter gate.
        from gomoveshift.pipeline.chapters.generator import YOUTUBE_MIN_CHAPTERS

        assert len(chapters) == YOUTUBE_MIN_CHAPTERS
        assert "Start" in chapters[0]["title"]
        assert "Finish" in chapters[-1]["title"]

    def test_generate_zero_gps_duration_falls_back_to_basic(self, generator):
        """Journey whose start_date == end_date has zero duration — fall back."""
        from datetime import datetime

        from gps_journey.core.models import Journey, Location, Waypoint

        t = datetime(2025, 9, 10, 10, 0, 0)
        wp = Waypoint(
            latitude=0.0,
            longitude=0.0,
            timestamp=t,
            location=Location(latitude=0.0, longitude=0.0),
        )
        zero_span = Journey(
            start_date=t,
            end_date=t,
            waypoints=(wp,),
            statistics=None,
        )

        chapters = generator.generate(
            gpx_path=Path("/unused"),
            video_duration=600,
            start_location="Cork, Ireland",
            _mock_journey=zero_span,
        )

        # Basic fallback padded to YouTube's minimum renderable shape.
        from gomoveshift.pipeline.chapters.generator import YOUTUBE_MIN_CHAPTERS

        assert len(chapters) == YOUTUBE_MIN_CHAPTERS
        assert "Start" in chapters[0]["title"]

    def test_geocode_timeout_does_not_propagate(self, config):
        """A timeout (or any exception) from the reverse geocoder must be
        absorbed — Nominatim 429s and timeouts must not bubble out and
        crash chapter generation. The end-location detection branch is
        the typical spot for this leak."""
        from datetime import datetime, timedelta

        from gomoveshift.pipeline.chapters import ChapterGenerator
        from gps_journey.core.models import Journey, Location, Waypoint

        start = datetime(2025, 9, 10, 10, 0, 0)
        wps = [
            Waypoint(
                latitude=52.0 + i * 0.01,
                longitude=-9.0,
                timestamp=start + timedelta(minutes=i * 5),
                location=Location(latitude=52.0 + i * 0.01, longitude=-9.0),
            )
            for i in range(4)
        ]
        journey = Journey(
            start_date=start,
            end_date=start + timedelta(minutes=15),
            waypoints=tuple(wps),
            statistics=None,
        )

        with patch(
            "gomoveshift.pipeline.chapters._query_overpass",
            return_value={"elements": []},
        ):
            with patch("gomoveshift.pipeline.chapters.NominatimRoadsClient") as mock_geo_cls:
                geocoder = mock_geo_cls.return_value
                # reverse_geocode raises like a Nominatim timeout would
                geocoder.reverse_geocode.side_effect = subprocess.TimeoutExpired("ffprobe", 5)
                geocoder.reverse_geocode_detailed.return_value = {}

                gen = ChapterGenerator(config)
                chapters = gen.generate(
                    gpx_path=Path("/unused"),
                    video_duration=900,
                    start_location="Ennis, County Clare",
                    _mock_journey=journey,
                )

        # Generator survives the geocoder failure and still produces ≥2 chapters
        assert len(chapters) >= 2


# =========================================================================
# route_context error paths
# =========================================================================


class TestRouteContextErrorPaths:
    """extract_route_context handles every degenerate input shape without
    raising and produces a usable RouteContext."""

    def test_empty_location_and_chapters(self):
        from gomoveshift.pipeline.route_context import extract_route_context

        ctx = extract_route_context("", [])
        assert ctx.gps_towns == []
        assert ctx.gps_start_town is None
        assert ctx.gps_end_town is None
        assert ctx.country is None
        # Title must still be a non-empty string for the SEO floor.
        title = ctx.make_default_title()
        assert isinstance(title, str)
        assert len(title) >= 10

    def test_empty_chapters_with_location_only(self):
        """No GPS chapters → no gps_towns; title falls back to location."""
        from gomoveshift.pipeline.route_context import extract_route_context

        ctx = extract_route_context("Beara Peninsula, County Cork, Ireland", [])
        assert ctx.gps_towns == []
        assert ctx.country == "Ireland"
        title = ctx.make_default_title()
        assert "Beara" in title or "Cork" in title

    def test_chapters_without_entering_prefix_yield_no_towns(self):
        """Only `Entering` lines feed gps_towns. Start/Finish lines do not."""
        from gomoveshift.pipeline.route_context import extract_route_context

        chapters = [
            {"timestamp": "0:00", "title": "Start - Cork"},
            {"timestamp": "5:00", "title": "Random milestone"},
            {"timestamp": "10:00", "title": "Finish - Cork"},
        ]
        ctx = extract_route_context("Cork, Ireland", chapters)
        assert ctx.gps_towns == []
        assert ctx.smart_start is None
        assert ctx.smart_end is None

    def test_malformed_entering_line_skipped(self):
        """Entering line missing the county part doesn't crash the regex."""
        from gomoveshift.pipeline.route_context import extract_route_context

        chapters = [
            {"timestamp": "0:00", "title": "Entering Lonely-Place"},  # no county
            {"timestamp": "10:00", "title": "Entering Kilrush, County Clare"},
        ]
        ctx = extract_route_context("County Clare, Ireland", chapters)
        # Only the well-formed one parses.
        assert ctx.gps_towns == ["Kilrush"]

    def test_location_without_country_segment(self):
        """Single-segment location: country falls back to None."""
        from gomoveshift.pipeline.route_context import extract_route_context

        ctx = extract_route_context("Ennis", [])
        assert ctx.country is None
        # No raise; title is still valid.
        title = ctx.make_default_title()
        assert len(title) >= 10

    def test_zero_span_route_single_town(self):
        """Same town for start and end (zero-span route) collapses to a
        single-town smart_start/smart_end — make_default_title still produces a
        valid, non-empty title. The old 40-char SEO floor was padded with a
        "| Scenic Drive" genre word that is now banned (not all footage is
        scenic drives), so a degenerate single-town title may fall below 40; it
        carries only the factual "POV Driving" pad and no genre claim."""
        from gomoveshift.pipeline.route_context import extract_route_context

        chapters = [
            {"timestamp": "0:00", "title": "Start - Cork"},
            {"timestamp": "5:00", "title": "Entering Cobh, County Cork"},
            {"timestamp": "10:00", "title": "Finish - Cork"},
        ]
        ctx = extract_route_context("Cork, Ireland", chapters)
        assert ctx.smart_start == "Cobh"
        assert ctx.smart_end == "Cobh"
        title = ctx.make_default_title()
        assert title == "POV Driving Cork | Ireland"
        assert "Scenic Drive" not in title


# =========================================================================
# ShortsValidator ffprobe edge cases
# =========================================================================


class TestShortsValidatorFfprobeEdgeCases:
    """ShortsValidator's per-short pipeline must absorb every realistic
    ffprobe failure mode without crashing the run: invocation failure,
    malformed JSON output, missing required fields. The gate fails;
    nothing raises."""

    def _make_short(self, tmp_path: Path, name: str = "short_01.mp4", size_kb: int = 600) -> Path:
        p = tmp_path / name
        p.write_bytes(b"x" * (size_kb * 1024))
        return p

    def _validator(self, subprocess_mock):
        from gomoveshift.pipeline.validation.ffprobe_client import FFprobeClient
        from gomoveshift.pipeline.validation.shorts_validator import ShortsValidator

        return ShortsValidator(FFprobeClient(subprocess_mock))

    def test_ffprobe_nonzero_returncode_fails_integrity(self, tmp_path):
        """ffprobe rc != 0 (mid-encode corruption, truncated file) → integrity fail."""
        sub = MagicMock()
        bad = MagicMock(returncode=1, stdout="", stderr="moov atom not found")
        sub.run.return_value = bad

        validator = self._validator(sub)
        short = self._make_short(tmp_path)
        gates = validator._validate_single_short(short)

        assert gates, "expected at least one gate"
        integrity = next(g for g in gates if g.gate.startswith("short_integrity:"))
        assert not integrity.passed
        assert "corrupt or unreadable" in integrity.message

    def test_ffprobe_malformed_json_fails_via_probe_or_none(self, tmp_path):
        """ffprobe rc=0 but stdout is non-JSON → probe_or_none returns None →
        downstream resolution / duration / bitrate gates skipped, probe gate
        fails. Verifies the JSON parse failure does not bubble out."""
        sub = MagicMock()
        # First call: integrity probe — succeed.
        # Second call: full probe — return non-JSON stdout.
        integrity_ok = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "30.0"}}),
            stderr="",
        )
        garbage = MagicMock(returncode=0, stdout="not json {{{", stderr="")
        sub.run.side_effect = [integrity_ok, garbage]

        validator = self._validator(sub)
        short = self._make_short(tmp_path)
        gates = validator._validate_single_short(short)

        # Integrity passes; probe gate is the second one and fails.
        assert gates[0].gate.startswith("short_integrity:")
        assert gates[0].passed
        assert gates[1].gate.startswith("short_probe:")
        assert not gates[1].passed

    def test_ffprobe_missing_dimensions_fails_resolution(self, tmp_path):
        """ffprobe returns valid JSON but the video stream has width=0 /
        no height key — resolution gate must FAIL (1080×1920 contract)."""
        sub = MagicMock()
        integrity_ok = MagicMock(
            returncode=0,
            stdout=json.dumps({"format": {"duration": "30.0"}}),
            stderr="",
        )
        broken_probe = MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "format": {"duration": "30.0", "bit_rate": "4000000"},
                    # codec_type present but width/height keys missing
                    "streams": [{"codec_type": "video", "codec_name": "hevc"}],
                }
            ),
            stderr="",
        )
        # Two probe calls (integrity + full), then loudnorm ffmpeg call.
        sub.run.side_effect = [integrity_ok, broken_probe, MagicMock(returncode=1, stderr="")]

        validator = self._validator(sub)
        short = self._make_short(tmp_path)
        gates = validator._validate_single_short(short)

        resolution = next(g for g in gates if g.gate.startswith("short_resolution:"))
        assert not resolution.passed
        assert "0x0" in resolution.message

    def test_ffprobe_crash_fails_integrity(self, tmp_path):
        """subprocess.run itself raises (binary missing, signal'd) → gate
        fails with the exception message — nothing propagates out."""
        sub = MagicMock()
        sub.run.side_effect = FileNotFoundError("ffprobe: command not found")

        validator = self._validator(sub)
        short = self._make_short(tmp_path)
        gates = validator._validate_single_short(short)

        integrity = next(g for g in gates if g.gate.startswith("short_integrity:"))
        assert not integrity.passed
        assert "ffprobe" in integrity.message

    def test_empty_short_file_list_returns_skipped(self, caplog):
        """validate_short_files with empty list returns a passing-skipped
        gate and logs a warning — defensive against upstream silent drops."""
        sub = MagicMock()
        validator = self._validator(sub)

        with caplog.at_level(logging.WARNING):
            result = validator.validate_short_files([])

        assert result.passed
        assert len(result.gates) == 1
        assert "skipped" in result.gates[0].message.lower()
        assert any("empty list" in r.message.lower() for r in caplog.records)
