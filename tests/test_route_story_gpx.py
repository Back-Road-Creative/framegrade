"""GPX import preserves the evidence needed by Route Story telemetry."""

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from gomoveshift.pipeline.route_story.gpx import load_gpx_segments, select_gpx_day


def test_gpx_import_preserves_segment_boundaries_and_source_clock(tmp_path: Path):
    source = tmp_path / "trip.gpx"
    source.write_text(
        """<gpx version="1.1" creator="test"><trk><trkseg>
        <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
        <trkpt lat="35.01" lon="-79.99"><time>2026-09-16T12:00:10Z</time></trkpt>
        </trkseg><trkseg>
        <trkpt lat="35.02" lon="-79.98"><ele>120</ele><time>2026-09-16T12:01:40Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )

    segments = load_gpx_segments(source)

    assert [[point.time_s for point in segment] for segment in segments] == [[0.0, 10.0], [100.0]]
    assert segments[0][0].elevation_m == 100.0
    assert segments[0][1].elevation_m is None
    assert segments[1][0].elevation_m == 120.0


def test_gpx_import_refuses_timeless_points_instead_of_inventing_a_clock(tmp_path: Path):
    source = tmp_path / "timeless.gpx"
    source.write_text('<gpx><trk><trkseg><trkpt lat="35" lon="-80" /></trkseg></trk></gpx>')

    with pytest.raises(ValueError, match="timestamp"):
        load_gpx_segments(source)


def test_gpx_import_refuses_timezone_free_timestamps(tmp_path: Path):
    source = tmp_path / "ambiguous-clock.gpx"
    source.write_text(
        '<gpx><trk><trkseg><trkpt lat="35" lon="-80">'
        "<time>2026-09-16T12:00:00</time></trkpt></trkseg></trk></gpx>"
    )

    with pytest.raises(ValueError, match="timezone"):
        load_gpx_segments(source)


def test_day_selection_excludes_other_dates_and_an_exact_duplicate_with_a_receipt(tmp_path: Path):
    source = tmp_path / "mixed-history.gpx"
    source.write_text(
        """<gpx version="1.1" creator="test"><trk>
        <trkseg><trkpt lat="35" lon="-80"><time>2025-12-12T12:00:00Z</time></trkpt></trkseg>
        <trkseg>
          <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
          <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T12:00:30Z</time></trkpt>
        </trkseg>
        <trkseg>
          <trkpt lat="35" lon="-80"><ele>100</ele><time>2026-09-16T12:00:00Z</time></trkpt>
          <trkpt lat="35.01" lon="-79.99"><ele>110</ele><time>2026-09-16T12:00:30Z</time></trkpt>
        </trkseg>
        </trk></gpx>""",
        encoding="utf-8",
    )

    selection = select_gpx_day(source, date(2026, 9, 16))

    assert len(selection.segments) == 1
    assert [point.time_s for point in selection.segments[0]] == [0.0, 30.0]
    assert selection.source_segment_count == 3
    assert selection.excluded_segment_count == 2
    assert selection.excluded_other_date_count == 1
    assert selection.excluded_duplicate_count == 1
    assert selection.source_started_at == datetime(2026, 9, 16, 12, 0, tzinfo=timezone.utc)


def test_gpx_import_parses_garmin_trackpoint_extension_body_data(tmp_path: Path):
    source = tmp_path / "body.gpx"
    source.write_text(
        """<gpx version="1.1" creator="test"
             xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">
        <trk><trkseg>
        <trkpt lat="35" lon="-80"><time>2026-09-16T12:00:00Z</time>
          <extensions><gpxtpx:TrackPointExtension>
            <gpxtpx:hr>142</gpxtpx:hr>
            <gpxtpx:cad>88</gpxtpx:cad>
            <gpxtpx:atemp>21.5</gpxtpx:atemp>
          </gpxtpx:TrackPointExtension></extensions>
        </trkpt>
        <trkpt lat="35.01" lon="-79.99"><time>2026-09-16T12:00:10Z</time></trkpt>
        </trkseg></trk></gpx>""",
        encoding="utf-8",
    )

    segments = load_gpx_segments(source)

    assert segments[0][0].heart_rate_bpm == 142.0
    assert segments[0][0].cadence_rpm == 88.0
    assert segments[0][0].temperature_c == 21.5
    assert segments[0][1].heart_rate_bpm is None
    assert segments[0][1].cadence_rpm is None
    assert segments[0][1].temperature_c is None


def test_day_selection_refuses_when_the_requested_day_has_no_track_points(tmp_path: Path):
    source = tmp_path / "old-track.gpx"
    source.write_text(
        """<gpx><trk><trkseg><trkpt lat="35" lon="-80">
        <time>2025-12-12T12:00:00Z</time></trkpt></trkseg></trk></gpx>"""
    )

    with pytest.raises(ValueError, match="2026-09-16"):
        select_gpx_day(source, date(2026, 9, 16))
