"""Head/tail GPS teleport-run guard on the chapter path.

A stale-nonzero cold-start fix survives the null-island strip: a recorded drive had
~15 leading waypoints locked ~400 km north (Boone NC) before ~2650 continuous
coastal points. Those 15 points fabricate a bogus chapter. ``_strip_teleport_runs``
removes such a contiguous head/tail run when it is separated from the rest of the
trace by a single physically-impossible spatial jump.

The discriminator is DISCONTINUITY, not distance-from-centroid: a
genuine long traverse (a real ~250 mi coast-to-mountains drive)
spreads its points continuously with no single jump, and a distance-from-mode
rule would amputate its true endpoints. This guard fires only on the seam.
"""

from datetime import datetime, timedelta

from gps_journey.core.models import Journey, Waypoint

from gomoveshift.pipeline.chapters.generator import _strip_teleport_runs

_START = datetime(2025, 1, 1, 0, 0, 0)


def _wp(lat: float, lon: float, secs: float) -> Waypoint:
    return Waypoint(
        latitude=lat,
        longitude=lon,
        timestamp=_START + timedelta(seconds=secs),
        location=None,
    )


def _journey(waypoints: tuple) -> Journey:
    return Journey(
        start_date=_START,
        end_date=_START + timedelta(seconds=max((len(waypoints) - 1, 1))),
        waypoints=waypoints,
        statistics=None,
    )


def _continuous_run(lat0, lon0, lat1, lon1, n, t0=0.0):
    """n waypoints marching evenly from (lat0,lon0) to (lat1,lon1)."""
    out = []
    for i in range(n):
        frac = i / (n - 1) if n > 1 else 0.0
        out.append(_wp(lat0 + (lat1 - lat0) * frac, lon0 + (lon1 - lon0) * frac, t0 + i))
    return out


class TestStripTeleportRuns:
    def test_leading_teleport_stripped(self):
        """15 leading mountain-town fixes, then a contiguous coastal body."""
        # 15 leading near-stationary fixes at a mountain town (~36.2, -81.7).
        boone = _continuous_run(36.20, -81.70, 36.21, -81.69, 15, t0=0.0)
        # ~400 contiguous coastal points near (32.4, -80.4), small even steps —
        # large enough that the 15-point run stays under the 5% strip bound
        # (mirroring a realistic ratio of ~15 / ~2665).
        coast = _continuous_run(32.40, -80.40, 32.55, -80.55, 400, t0=15.0)
        j = _journey(tuple(boone + coast))

        out = _strip_teleport_runs(j)

        assert len(out.waypoints) == 400
        first = out.waypoints[0]
        assert 32.0 < first.latitude < 33.0  # coastal, not Boone
        assert abs(len(j.waypoints) - len(out.waypoints)) == 15

    def test_continuous_traverse_not_stripped(self):
        """Guard: a real continuous 250-mile traverse is left intact."""
        body = _continuous_run(32.40, -80.40, 35.30, -82.90, 200, t0=0.0)
        j = _journey(tuple(body))

        out = _strip_teleport_runs(j)

        assert len(out.waypoints) == len(j.waypoints)
        assert out.waypoints[0].latitude == j.waypoints[0].latitude
        assert out.waypoints[-1].latitude == j.waypoints[-1].latitude

    def test_trailing_teleport_stripped(self):
        """Continuous coastal body, then a trailing run after one big jump."""
        coast = _continuous_run(32.40, -80.40, 32.55, -80.55, 400, t0=0.0)
        boone = _continuous_run(36.20, -81.70, 36.21, -81.69, 15, t0=400.0)
        j = _journey(tuple(coast + boone))

        out = _strip_teleport_runs(j)

        assert len(out.waypoints) == 400
        last = out.waypoints[-1]
        assert 32.0 < last.latitude < 33.0  # coastal, not Boone

    def test_tiny_journey_safety(self):
        """Stripping that would leave <2 points returns the original."""
        # Two points far apart — a "seam" exists but stripping empties the body.
        wps = (_wp(36.20, -81.70, 0), _wp(32.40, -80.40, 1))
        j = _journey(wps)
        out = _strip_teleport_runs(j)
        assert out.waypoints == wps

    def test_empty_journey_safe(self):
        j = _journey(())
        out = _strip_teleport_runs(j)
        assert out.waypoints == ()
