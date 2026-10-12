"""Tests for GPS-speed corroboration of optical stops.

An optical "stop" the GPS trace shows was still moving (false dark/tunnel stop)
is dropped; a GPS-absent LF keeps the optical-only behavior exactly; a span
where GPS speed is ~0 is preserved.
"""

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.config import Config
from gomoveshift.models import DeadSpot
from gomoveshift.pipeline.stop_processor import StopProcessor


def _config(enabled: bool, stop_speed_kmh: float = 10.0) -> Config:
    config = Config()
    config.stop_detection.gps_speed_corroboration.enabled = enabled
    config.stop_detection.gps_speed_corroboration.stop_speed_kmh = stop_speed_kmh
    return config


def _moving_trace(start_unix: float = 0.0) -> GPSData:
    """Trace covering 0-30s of source time at a steady highway pace (~90 km/h).

    Trackpoints are spaced 0.00025 deg of longitude (~25 m near the equator)
    every 1s → ~90 km/h, well above the default 10 km/h stop threshold.
    """
    tps = []
    for i in range(31):
        tps.append(
            GPSTrackpoint(
                lat=10.0, lon=10.0 + i * 0.00025, elevation=None, timestamp=start_unix + i
            )
        )
    return GPSData(trackpoints=tps)


def _parked_trace(start_unix: float = 0.0) -> GPSData:
    """Trace covering 0-30s of source time with the vehicle parked (~0 km/h)."""
    tps = [
        GPSTrackpoint(lat=10.0, lon=10.0, elevation=None, timestamp=start_unix + i)
        for i in range(31)
    ]
    return GPSData(trackpoints=tps)


def _one_stop() -> list[DeadSpot]:
    return [DeadSpot(start_time=5.0, end_time=20.0, motion_energy=0.001)]


class TestGPSCorroboration:
    def test_moving_stop_is_dropped(self):
        """Optical stop + GPS showing nonzero (highway) speed → stop dropped."""
        proc = StopProcessor(_config(enabled=True), gps_data=_moving_trace())
        kept = proc._corroborate_stops_with_gps(_one_stop())
        assert kept == []

    def test_gps_absent_identical_to_optical_only(self):
        """No GPS trace → corroboration is a strict no-op (optical-only)."""
        stops = _one_stop()
        proc = StopProcessor(_config(enabled=True), gps_data=None)
        assert proc._corroborate_stops_with_gps(stops) == stops

    def test_disabled_identical_to_optical_only(self):
        """Disabled config → optical stops pass through unchanged even with a trace."""
        stops = _one_stop()
        proc = StopProcessor(_config(enabled=False), gps_data=_moving_trace())
        assert proc._corroborate_stops_with_gps(stops) == stops

    def test_parked_stop_is_preserved(self):
        """GPS speed ~0 over the span → optical stop is kept (real stop)."""
        stops = _one_stop()
        proc = StopProcessor(_config(enabled=True), gps_data=_parked_trace())
        assert proc._corroborate_stops_with_gps(stops) == stops

    def test_degraded_span_leaves_stop_untouched(self):
        """Span with < 2 trackpoints (partial trace) → stop left as-is."""
        # Trace only covers 0-3s; the stop at 5-20s has no points in range.
        sparse = GPSData(
            trackpoints=[
                GPSTrackpoint(lat=10.0, lon=10.0, elevation=None, timestamp=0.0),
                GPSTrackpoint(lat=10.0, lon=10.001, elevation=None, timestamp=3.0),
            ]
        )
        stops = _one_stop()
        proc = StopProcessor(_config(enabled=True), gps_data=sparse)
        assert proc._corroborate_stops_with_gps(stops) == stops

    def test_empty_trackpoints_no_op(self):
        """A trace object with no trackpoints behaves like GPS-absent."""
        stops = _one_stop()
        proc = StopProcessor(_config(enabled=True), gps_data=GPSData(trackpoints=[]))
        assert proc._corroborate_stops_with_gps(stops) == stops


def _creep_trace(start_unix: float = 0.0) -> GPSData:
    """Trace covering 0-30s crawling at ~6 km/h — sub-``stop_speed_kmh`` but
    clearly moving (median well above ``parked_speed_kmh``)."""
    tps = [
        GPSTrackpoint(lat=10.0, lon=10.0 + i * 0.0000152, elevation=None, timestamp=start_unix + i)
        for i in range(31)
    ]
    return GPSData(trackpoints=tps)


class TestGPSDerivedStops:
    """ADDITIVE half: emit stops the optical detector missed (parked with
    passing traffic — high optical energy, GPS ~0)."""

    def test_parked_span_emits_additive_stop(self):
        """A sustained parked (~0 km/h) span becomes a stop with NO optical input."""
        proc = StopProcessor(_config(enabled=True), gps_data=_parked_trace())
        derived = proc._gps_derived_stops()
        assert len(derived) == 1
        assert derived[0].start_time == 0.0
        assert derived[0].end_time >= 29.0

    def test_moving_span_emits_nothing(self):
        """A highway-speed trace yields no additive stops."""
        proc = StopProcessor(_config(enabled=True), gps_data=_moving_trace())
        assert proc._gps_derived_stops() == []

    def test_creep_above_parked_speed_emits_nothing(self):
        """Sub-``stop_speed_kmh`` but median > ``parked_speed_kmh`` (a slow crawl,
        not parked) does NOT emit an additive stop."""
        proc = StopProcessor(_config(enabled=True), gps_data=_creep_trace())
        assert proc._gps_derived_stops() == []

    def test_disabled_emits_nothing(self):
        proc = StopProcessor(_config(enabled=False), gps_data=_parked_trace())
        assert proc._gps_derived_stops() == []

    def test_gps_absent_emits_nothing(self):
        proc = StopProcessor(_config(enabled=True), gps_data=None)
        assert proc._gps_derived_stops() == []

    def test_merge_coalesces_overlapping_spans(self):
        """Optical + GPS-derived spans that overlap coalesce into one."""
        merged = StopProcessor._merge_stop_spans(
            [
                DeadSpot(start_time=0.0, end_time=10.0, motion_energy=0.001),
                DeadSpot(start_time=5.0, end_time=20.0, motion_energy=0.0),
                DeadSpot(start_time=40.0, end_time=45.0, motion_energy=0.0),
            ]
        )
        assert [(s.start_time, s.end_time) for s in merged] == [(0.0, 20.0), (40.0, 45.0)]
