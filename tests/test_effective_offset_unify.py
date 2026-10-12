"""Footage-to-GPS anchoring through the shared effective offset.

Every footage→GPS anchor must route through the shared ``effective_offset``
helper (``gomoveshift/analysis/shorts_models.py``) so a leading GoPro
cold-start anomaly — a null-island ``(0,0)`` run or a stale clock lock —
can never shift a sampling window into the anomalous lead:

* ``post_encoding_invariants._window_journey_to_included_segments``
  anchored its datetime→source-second projection at the first RAW waypoint,
  so a cold-start lead pulled every per-LF description-fact window onto the
  garbage prefix (and excluded the real footage).
* the legacy per-short mapper ``shorts_extractor._derive_short_location_map``
  seeded ``VideoToGpsTimeMapper`` with the raw ``gps_data.start_time``, so a
  short's GPS window opened on the cold-start lead instead of the first real
  fix.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
from gomoveshift.config import Config
from gomoveshift.pipeline.post_encoding_invariants import _window_journey_to_included_segments
from gomoveshift.pipeline.shorts_extractor import _derive_short_location_map

_BASE = datetime(2026, 5, 1, 12, 0, 0, tzinfo=timezone.utc)


def _wp(lat, lon, t_sec):
    return SimpleNamespace(latitude=lat, longitude=lon, timestamp=_BASE + timedelta(seconds=t_sec))


class TestWindowJourneyEffectiveOffset:
    """The per-LF fact window must anchor at the first VALID fix."""

    def test_null_island_lead_shifts_window_to_first_valid_fix(self):
        # Cold-start lead: (0,0) fixes at t=0..40, real moving fixes t=50..150.
        lead = [_wp(0.0, 0.0, t) for t in range(0, 50, 10)]
        valid = [_wp(32.30 + 0.001 * i, -80.63, 50 + 10 * i) for i in range(11)]
        journey = SimpleNamespace(waypoints=lead + valid)

        out = _window_journey_to_included_segments(journey, [(0.0, 30.0)])

        kept = list(out.waypoints)
        assert kept, "window must land on the real fixes, not come back empty"
        assert all((wp.latitude, wp.longitude) != (0.0, 0.0) for wp in kept), (
            "window kept null-island cold-start fixes — anchor is the RAW "
            "first waypoint, not the effective offset"
        )
        # Effective anchor = first VALID fix ⇒ source-second 0 is valid[0].
        assert kept[0].timestamp == valid[0].timestamp
        assert kept[-1].timestamp == valid[3].timestamp  # 30 s window

    def test_clean_trace_window_unchanged(self):
        # No cold-start anomaly: prior behaviour byte-for-byte.
        valid = [_wp(32.30 + 0.001 * i, -80.63, 10 * i) for i in range(11)]
        journey = SimpleNamespace(waypoints=valid)

        out = _window_journey_to_included_segments(journey, [(0.0, 30.0)])

        assert list(out.waypoints) == valid[:4]


class TestLegacyShortMapperEffectiveOffset:
    """The legacy per-short mapper must seed the effective GPS offset."""

    def _derive(self, gps, tmp_path, monkeypatch):
        import gomoveshift.analysis.gps_parser as gps_parser
        import gomoveshift.pipeline.gps_enrichment as gps_enrichment

        monkeypatch.setattr(gps_parser, "parse_gpx_file", lambda path: gps)
        monkeypatch.setattr(gps_enrichment, "_build_shared_geocoder", lambda cache_dir: object())
        seen: list = []

        def _make_fn(geocoder, **_kw):
            def _fn(pts, short):
                seen.append(pts)
                return "Beaufort County, South Carolina" if pts else None

            return _fn

        monkeypatch.setattr(gps_enrichment, "_make_short_ost_text_fn", _make_fn)
        gpx = tmp_path / "drive.gpx"
        gpx.write_text("<gpx/>")
        short = SimpleNamespace(index=1, start_time=0.0, end_time=10.0)
        out = _derive_short_location_map([short], gpx, None, 0.0, 0.0, Config())
        return out, seen

    def test_null_island_lead_maps_short_to_valid_fixes(self, tmp_path, monkeypatch):
        # Cold-start lead at ts 1000..1004, real moving fixes at ts 1050..1080.
        lead = [GPSTrackpoint(0.0, 0.0, None, 1000.0 + i) for i in range(5)]
        valid = [GPSTrackpoint(32.30 + 0.001 * i, -80.63, None, 1050.0 + i) for i in range(31)]
        gps = GPSData(trackpoints=lead + valid)

        out, seen = self._derive(gps, tmp_path, monkeypatch)

        assert out == {1: "Beaufort County, South Carolina"}
        (pts,) = seen
        assert pts, "short window must land on real fixes, not the cold-start lead"
        assert all((lat, lon) != (0.0, 0.0) for _ts, lat, lon in pts), (
            "short window sampled null-island cold-start fixes — mapper was "
            "seeded with the raw start_time, not the effective offset"
        )

    def test_clean_trace_keeps_stored_offset(self, tmp_path, monkeypatch):
        # No anomaly: the stored offset (first fix) is the anchor, as before.
        valid = [GPSTrackpoint(32.30 + 0.001 * i, -80.63, None, 1000.0 + i) for i in range(31)]
        gps = GPSData(trackpoints=valid)

        out, seen = self._derive(gps, tmp_path, monkeypatch)

        assert out == {1: "Beaufort County, South Carolina"}
        (pts,) = seen
        assert pts and pts[0][0] == 1000.0
