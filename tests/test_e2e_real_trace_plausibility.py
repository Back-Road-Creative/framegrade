"""End-to-end plausibility on a tiny REAL GPX trace.

Every numeric/terrain defect of this kind was "unit-tested in isolation, broke on real
data, caught by a late gate." This test closes that gap: it writes ONE realistic
flat-coastal GPX file (Beaufort, SC — the drive that published "315 mi", inflated
elevation, and a fake "Mountain Climb"), opens it with a leading null-island
cold-start fix AND a stale-lock teleport spike carrying a garbage 5,000 m
altitude (the exact shapes that fired the defects), then parses it through the
REAL ``GPXParser`` and drives MULTIPLE numeric surfaces — distance, elevation
gain, and chapter terrain classification — asserting every one produces a
PLAUSIBLE value. A single real-trace integration assertion like this would have
caught the elevation, distance and terrain (Mountain-Climb-on-coast) defects
BEFORE publish, where the isolated unit tests did not.

Hermetic: no network (no reverse-geocode); the three asserted defects are all
GPS-numeric / terrain, derivable from the trace alone.
"""

from datetime import datetime, timedelta, timezone

import pytest

# Beaufort, SC coastal area — flat, near sea level.
_LAT0, _LON0 = 32.43, -80.67
_T0 = datetime(2026, 5, 14, 14, 0, tzinfo=timezone.utc)


def _write_real_gpx(path) -> None:
    """A realistic flat-coastal drive with the two defect-triggering lead fixes.

    Point 0: null-island (0,0) cold-start fix (pre-GPS-lock).
    Point 1: a stale inland lock ~80 km away, 0.5 s later — a sub-second teleport
             (the 315-mi shape), carrying a garbage 5,000 m altitude (the
             inflated-elevation shape).
    Points 2..N: ~1 km steps along a flat coastal road, elevations 2-9 m with
             small ±2 m GPS noise (real coastal terrain — NOT a mountain).
    """
    trkpts = []

    def pt(lat, lon, when, ele):
        return (
            f'<trkpt lat="{lat:.6f}" lon="{lon:.6f}">'
            f"<ele>{ele:.1f}</ele>"
            f"<time>{when.strftime('%Y-%m-%dT%H:%M:%SZ')}</time>"
            f"</trkpt>"
        )

    # Defect-shaped lead fixes.
    trkpts.append(pt(0.0, 0.0, _T0, 0.0))  # null-island cold start
    trkpts.append(pt(33.20, -81.40, _T0 + timedelta(seconds=0.5), 5000.0))  # teleport + garbage alt

    # 14 km of flat coastal driving: ~1 km steps (~0.009° lat) at ~60 km/h,
    # elevations hovering 2-9 m with ±2 m noise — net range stays tiny.
    noise = [0.0, 2.0, -1.0, 1.0, 0.0, 2.0, -2.0, 1.0, 0.0, 1.0, -1.0, 2.0, 0.0, 1.0]
    for i in range(14):
        lat = _LAT0 + i * 0.009
        ele = 5.0 + noise[i]
        when = _T0 + timedelta(seconds=60 * (i + 1))
        trkpts.append(pt(lat, _LON0, when, ele))

    gpx = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<gpx version="1.1" creator="test"><trk><trkseg>'
        + "".join(trkpts)
        + "</trkseg></trk></gpx>"
    )
    path.write_text(gpx)


@pytest.fixture
def real_gpx(tmp_path):
    p = tmp_path / "beaufort_coastal.gpx"
    _write_real_gpx(p)
    return p


def _cleaned_journey_waypoints(gpx_path):
    """Parse with the REAL parser, then clean exactly as the numeric surfaces do."""
    from gps_journey.core.gpx_parser import GPXParser

    from gomoveshift.pipeline.post_encoding_invariants import _cleaned_waypoints

    journey = GPXParser().parse(str(gpx_path))
    return journey, _cleaned_waypoints(journey)


def test_real_parse_yields_waypoints(real_gpx):
    """Sanity: the fixture is a real GPX the production parser actually reads."""
    journey, cleaned = _cleaned_journey_waypoints(real_gpx)
    assert len(journey.waypoints) == 16  # 2 lead fixes + 14 body
    # Cleaning strips the null-island + teleport lead; the coastal body survives.
    assert 10 <= len(cleaned) <= 14


def test_distance_is_plausible_not_315_miles(real_gpx):
    """Distance routes through the cleaned trace + plausibility gate, so the
    ~80 km teleport jump is never summed into a 315-mi figure."""
    from gomoveshift.pipeline.post_encoding_invariants import _compute_per_lf_distance

    result = _compute_per_lf_distance({0: str(real_gpx)})  # real GPXParser
    assert 0 in result, "a clean flat drive must yield a distance"
    km = result[0]
    # ~13 km of real coastal driving — well under any impossible figure.
    assert 8.0 < km < 30.0, f"distance {km:.1f} km implausible — teleport not stripped?"
    assert km < 80.0  # the stale-lock jump alone was ~80 km


def test_elevation_gain_is_plausible_not_thousands_of_feet(real_gpx):
    """The 5,000 m garbage cold-start altitude is stripped with the teleport
    head, so a flat coastal drive does not report a fabricated climb."""
    from gomoveshift.pipeline.post_encoding_invariants import _compute_per_lf_elevation_gain

    result = _compute_per_lf_elevation_gain({0: str(real_gpx)})  # real GPXParser
    assert 0 in result
    gain_m = result[0]
    # Flat coast: only the small ±2 m noise contributes; nowhere near 5,000 m
    # (the garbage spike) or even a real hill.
    assert gain_m < 50.0, f"elevation gain {gain_m:.0f} m implausible for a flat coast"


def test_terrain_not_classified_mountain_on_flat_coast(real_gpx):
    """The chapter terrain classifier must NOT label a flat coastal drive a
    'mountain' (net-range guard) once the trace is cleaned."""
    from gomoveshift.pipeline.chapters.generator import _classify_chapter_band

    _journey, cleaned = _cleaned_journey_waypoints(real_gpx)
    assert len(cleaned) >= 2
    base = cleaned[0].timestamp
    video_times = [(wp.timestamp - base).total_seconds() for wp in cleaned]
    band = _classify_chapter_band(
        cleaned,
        video_times,
        chapter_start_video_t=0.0,
        chapter_end_video_t=video_times[-1],
        coastline_points=[],
    )
    assert band != "mountain", "flat coastal drive misclassified as mountain"
