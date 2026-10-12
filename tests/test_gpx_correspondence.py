"""GPX↔video correspondence gate.

GoPro GPMF telemetry is frame-aligned by construction (extracted from the
same MP4), so the gate skips it. The risk surface is the resume +
generic-external-GPX path, where an operator (or a stale resume state)
can pair a video with a GPX from a *different* drive. The gate asserts the
GPX temporally corresponds to the video it labels, and — when the video
carries an embedded ISO-6709 location tag — that the GPX first fix is near
it.
"""

from datetime import datetime, timezone

import pytest

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.pipeline.gopro_gps import (
    GpxCorrespondenceError,
    assert_gpx_video_correspondence,
)
from gomoveshift.pipeline.ingest.gopro_parser import Episode


def _gps(start_unix: float, span_sec: float, lat: float = 32.42, lon: float = -80.46):
    """Two-point GPSData spanning ``span_sec`` from ``start_unix``."""
    return GPSData(
        trackpoints=[
            GPSTrackpoint(lat, lon, None, start_unix),
            GPSTrackpoint(lat + 0.01, lon + 0.01, None, start_unix + span_sec),
        ]
    )


# ---------------------------------------------------------------------------
# Episode provenance flag
# ---------------------------------------------------------------------------


def test_episode_gpx_source_defaults_to_gpmf():
    """Frame-aligned GoPro GPMF is the default — gate skips it."""
    ep = Episode(index=1)
    assert ep.gpx_source == "gpmf"


def test_episode_gpx_source_settable_external():
    ep = Episode(index=1, gpx_source="external")
    assert ep.gpx_source == "external"


# ---------------------------------------------------------------------------
# Temporal correspondence (start-time + span) — gated even with no coord anchor
# ---------------------------------------------------------------------------


def test_corresponding_gpx_passes():
    creation = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    gps = _gps(creation.timestamp() + 5, span_sec=600)  # +5s start, 600s span
    # Should not raise.
    assert_gpx_video_correspondence(gps, creation, video_duration=600.0)


def test_swapped_gpx_wrong_start_time_raises():
    creation = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    # GPX from a drive two hours later — start-time delta > 1h.
    gps = _gps(creation.timestamp() + 2 * 3600, span_sec=600)
    with pytest.raises(GpxCorrespondenceError, match="start-time"):
        assert_gpx_video_correspondence(gps, creation, video_duration=600.0)


def test_swapped_gpx_wrong_span_raises():
    creation = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    # Start lines up, but GPX spans 30 min for a 10 min video — swapped track.
    gps = _gps(creation.timestamp(), span_sec=1800)
    with pytest.raises(GpxCorrespondenceError, match="span"):
        assert_gpx_video_correspondence(gps, creation, video_duration=600.0)


def test_span_within_tolerance_passes():
    creation = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    # 600s video, 615s GPX — within tolerance (GPS sampling tail).
    gps = _gps(creation.timestamp(), span_sec=615)
    assert_gpx_video_correspondence(gps, creation, video_duration=600.0)


# ---------------------------------------------------------------------------
# Embedded ISO-6709 coord anchor — first fix within a sane radius
# ---------------------------------------------------------------------------


def test_first_fix_near_embedded_coord_passes():
    creation = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    gps = _gps(creation.timestamp(), span_sec=600, lat=32.42, lon=-80.46)
    assert_gpx_video_correspondence(
        gps, creation, video_duration=600.0, embedded_coord=(32.43, -80.47)
    )


def test_first_fix_far_from_embedded_coord_raises():
    creation = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    # GPX first fix in NC mountains, embedded tag in coastal SC — swap.
    gps = _gps(creation.timestamp(), span_sec=600, lat=35.6, lon=-82.5)
    with pytest.raises(GpxCorrespondenceError, match="embedded"):
        assert_gpx_video_correspondence(
            gps, creation, video_duration=600.0, embedded_coord=(32.42, -80.46)
        )


# ---------------------------------------------------------------------------
# Degrade-not-fail: no embedded tag — STILL gate on span/start-time
# ---------------------------------------------------------------------------


def test_no_embedded_coord_still_gates_temporal():
    creation = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    gps = _gps(creation.timestamp() + 2 * 3600, span_sec=600)
    # No coord anchor, but temporal gate still fires on a swapped GPX.
    with pytest.raises(GpxCorrespondenceError, match="start-time"):
        assert_gpx_video_correspondence(gps, creation, video_duration=600.0, embedded_coord=None)


def test_no_embedded_coord_corresponding_gpx_passes():
    creation = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    gps = _gps(creation.timestamp(), span_sec=600)
    assert_gpx_video_correspondence(gps, creation, video_duration=600.0, embedded_coord=None)


# ---------------------------------------------------------------------------
# Insufficient data degrades, never raises
# ---------------------------------------------------------------------------


def test_empty_gps_degrades_no_raise():
    creation = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)
    # No trackpoints / no creation_time — can't gate, must not raise.
    assert_gpx_video_correspondence(GPSData(trackpoints=[]), creation, video_duration=600.0)
    assert_gpx_video_correspondence(_gps(0, 600), None, video_duration=600.0)
