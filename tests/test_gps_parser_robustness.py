"""Tests for GPX parser robustness — documents handling of malformed input."""

from pathlib import Path

import pytest

from gomoveshift.analysis.gps_parser import parse_gpx_file
from gomoveshift.analysis.shorts_models import GPSError


@pytest.fixture
def gpx_dir(tmp_path):
    """Return a temp directory for GPX test files."""
    return tmp_path


def _write_gpx(path: Path, body: str) -> Path:
    """Write a minimal GPX file with the given body inside <trkseg>."""
    content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1">\n'
        "  <trk><trkseg>\n"
        f"    {body}\n"
        "  </trkseg></trk>\n"
        "</gpx>\n"
    )
    path.write_text(content)
    return path


class TestValidGPX:
    """Tests for well-formed GPX files."""

    def test_single_trackpoint(self, gpx_dir):
        """Parse a GPX with one trackpoint including elevation and time."""
        gpx = _write_gpx(
            gpx_dir / "one.gpx",
            '<trkpt lat="53.3498" lon="-6.2603">'
            "  <ele>10.5</ele>"
            "  <time>2026-04-01T12:00:00Z</time>"
            "</trkpt>",
        )
        data = parse_gpx_file(gpx)
        assert len(data.trackpoints) == 1
        tp = data.trackpoints[0]
        assert tp.lat == pytest.approx(53.3498)
        assert tp.lon == pytest.approx(-6.2603)
        assert tp.elevation == pytest.approx(10.5)
        assert tp.timestamp > 0

    def test_multiple_trackpoints_sorted(self, gpx_dir):
        """Multiple trackpoints are sorted by timestamp."""
        gpx = _write_gpx(
            gpx_dir / "multi.gpx",
            '<trkpt lat="53.0" lon="-6.0">'
            "  <time>2026-04-01T12:01:00Z</time>"
            "</trkpt>\n"
            '<trkpt lat="54.0" lon="-7.0">'
            "  <time>2026-04-01T12:00:00Z</time>"
            "</trkpt>",
        )
        data = parse_gpx_file(gpx)
        assert len(data.trackpoints) == 2
        # Second point has earlier time, should come first after sort
        assert data.trackpoints[0].lat == pytest.approx(54.0)
        assert data.trackpoints[1].lat == pytest.approx(53.0)

    def test_trackpoint_without_elevation(self, gpx_dir):
        """Elevation is optional — None when missing."""
        gpx = _write_gpx(
            gpx_dir / "no_ele.gpx",
            '<trkpt lat="53.0" lon="-6.0">  <time>2026-04-01T12:00:00Z</time></trkpt>',
        )
        data = parse_gpx_file(gpx)
        assert data.trackpoints[0].elevation is None

    def test_trackpoint_without_time(self, gpx_dir):
        """Timeless trackpoint is dropped, not stamped 0.0."""
        gpx = _write_gpx(
            gpx_dir / "no_time.gpx",
            '<trkpt lat="53.0" lon="-6.0"></trkpt>',
        )
        data = parse_gpx_file(gpx)
        assert data.trackpoints == []


class TestMalformedGPX:
    """Tests for graceful handling of malformed GPX input."""

    def test_missing_lat_attribute_skipped(self, gpx_dir):
        """Malformed trkpt with no lat is gracefully skipped."""
        gpx = _write_gpx(
            gpx_dir / "no_lat.gpx",
            '<trkpt lon="-6.0"><time>2026-04-01T12:00:00Z</time></trkpt>',
        )
        result = parse_gpx_file(gpx)
        assert len(result.trackpoints) == 0  # malformed point skipped

    def test_missing_lon_attribute_skipped(self, gpx_dir):
        """Malformed trkpt with no lon is gracefully skipped."""
        gpx = _write_gpx(
            gpx_dir / "no_lon.gpx",
            '<trkpt lat="53.0"><time>2026-04-01T12:00:00Z</time></trkpt>',
        )
        result = parse_gpx_file(gpx)
        assert len(result.trackpoints) == 0  # malformed point skipped

    def test_non_numeric_lat_skipped(self, gpx_dir):
        """Non-numeric lat value is gracefully skipped."""
        gpx = _write_gpx(
            gpx_dir / "bad_lat.gpx",
            '<trkpt lat="abc" lon="-6.0"></trkpt>',
        )
        result = parse_gpx_file(gpx)
        assert len(result.trackpoints) == 0  # malformed point skipped


class TestEdgeCases:
    """Edge cases for GPX parsing."""

    def test_file_not_found(self, gpx_dir):
        """Non-existent file raises GPSError."""
        with pytest.raises(GPSError, match="not found"):
            parse_gpx_file(gpx_dir / "nope.gpx")

    def test_invalid_xml(self, gpx_dir):
        """Broken XML raises GPSError."""
        bad = gpx_dir / "broken.gpx"
        bad.write_text("<gpx><not-closed")
        with pytest.raises(GPSError, match="Invalid GPX XML"):
            parse_gpx_file(bad)

    def test_empty_trackpoints(self, gpx_dir):
        """GPX with no trkpt elements returns empty trackpoints list."""
        gpx = _write_gpx(gpx_dir / "empty.gpx", "")
        data = parse_gpx_file(gpx)
        assert data.trackpoints == []

    def test_gpx_without_namespace(self, gpx_dir):
        """GPX without namespace is still parsed via fallback xpath."""
        content = (
            '<?xml version="1.0"?>\n'
            "<gpx>\n"
            "  <trk><trkseg>\n"
            '    <trkpt lat="53.0" lon="-6.0"><time>2026-04-01T12:00:00Z</time></trkpt>\n'
            "  </trkseg></trk>\n"
            "</gpx>\n"
        )
        gpx = gpx_dir / "no_ns.gpx"
        gpx.write_text(content)
        data = parse_gpx_file(gpx)
        assert len(data.trackpoints) == 1


class TestMaliciousGPX:
    """GPX is operator/GoPro-supplied — guard against XML entity-expansion DoS."""

    def test_billion_laughs_raises_gpserror(self, gpx_dir):
        """A billion-laughs payload raises GPSError, never expands/OOMs/hangs."""
        # Nested entity definitions: a single &lol9; reference expands to 10^9
        # entities under a naive parser. defusedxml forbids entity expansion and
        # the parser must surface it as a clean GPSError rather than crashing.
        bomb = (
            '<?xml version="1.0"?>\n'
            "<!DOCTYPE gpx [\n"
            '  <!ENTITY lol "lol">\n'
            '  <!ENTITY lol1 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">\n'
            '  <!ENTITY lol2 "&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;&lol1;">\n'
            '  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">\n'
            '  <!ENTITY lol4 "&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;&lol3;">\n'
            "]>\n"
            '<gpx xmlns="http://www.topografix.com/GPX/1/1" version="1.1">\n'
            "  <trk><trkseg>\n"
            '    <trkpt lat="53.0" lon="-6.0"><name>&lol4;</name></trkpt>\n'
            "  </trkseg></trk>\n"
            "</gpx>\n"
        )
        gpx = gpx_dir / "bomb.gpx"
        gpx.write_text(bomb)
        with pytest.raises(GPSError):
            parse_gpx_file(gpx)


class TestTimestampParity:
    """Timestamp handling in ``gomoveshift.analysis.gps_parser.parse_gpx_file``.

    Timeless trackpoints must be DROPPED (a 0.0 default sorts to the front and
    scrambles chronological order plus the GPS↔video offset), and a <time>
    without a Z/offset must be read as UTC (a naive ``.timestamp()`` skews the
    epoch by whole hours in a non-UTC local tz).
    """

    def test_timeless_point_dropped_not_sorted_to_front(self, gpx_dir):
        """A timeless trackpoint is dropped, never a spurious 0.0 front point."""
        gpx = _write_gpx(
            gpx_dir / "mixed.gpx",
            '<trkpt lat="53.0" lon="-6.0"><time>2025-09-01T10:00:00Z</time></trkpt>\n'
            '    <trkpt lat="54.0" lon="-7.0"></trkpt>\n'
            '    <trkpt lat="53.5" lon="-6.5"><time>2025-09-01T11:00:00Z</time></trkpt>',
        )
        data = parse_gpx_file(gpx)
        assert len(data.trackpoints) == 2
        # No 0.0 front point; chronological order preserved.
        assert all(tp.timestamp > 0 for tp in data.trackpoints)
        assert data.trackpoints[0].lat == pytest.approx(53.0)
        assert data.trackpoints[1].lat == pytest.approx(53.5)

    def test_naive_iso_time_parsed_as_utc(self, gpx_dir):
        """A <time> with no Z/offset is interpreted as UTC, not the local tz."""
        import os
        import time
        from datetime import datetime, timezone

        original_tz = os.environ.get("TZ")
        os.environ["TZ"] = "XXX5"  # local = UTC-5, no DST, no tzdata files needed
        time.tzset()
        try:
            gpx = _write_gpx(
                gpx_dir / "naive.gpx",
                '<trkpt lat="53.0" lon="-6.0"><time>2025-09-01T10:00:00</time></trkpt>',
            )
            data = parse_gpx_file(gpx)
            expected = datetime(2025, 9, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp()
            assert data.trackpoints[0].timestamp == pytest.approx(expected)
        finally:
            if original_tz is None:
                os.environ.pop("TZ", None)
            else:
                os.environ["TZ"] = original_tz
            time.tzset()
