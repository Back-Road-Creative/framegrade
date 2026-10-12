"""Caller wiring: PhaseBCDayFirst builds per-date GPS + geocoder
closure and passes both through ``run_all_days``.

The plumbing (``gps_points_by_date``, ``short_ost_text_fn`` kwargs
on ``run_all_days`` / ``run_day_pipeline``) is useless without a
caller that actually populates those kwargs from state. This test
asserts ``PhaseBCDayFirst.run`` does so: it projects
``state["gps_per_episode"]`` into each day's master timeline and
installs a reverse-geocoding closure that derives per-short OST text
from the trimmed-clip centroid.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from _state_helpers import make_state


class _Loc:
    def __init__(self, city, state="SC", country="USA"):
        self.city = city
        self.locality = city
        self.name = city
        self.suburb = None
        self.county = None
        self.state = state
        self.country = country


def _tp(ts, lat, lon):
    from gomoveshift.analysis.shorts_models import GPSTrackpoint

    return GPSTrackpoint(lat=lat, lon=lon, elevation=0.0, timestamp=float(ts))


def test_phase_bc_passes_gps_and_ost_fn_to_run_all_days(tmp_path):
    """``PhaseBCDayFirst.run`` must thread per-date GPS + a
    short-OST-text callback into ``run_all_days``. Without these the
    shorts trim inside run_day_pipeline is dead."""
    from gomoveshift.analysis.shorts_models import GPSData
    from gomoveshift.config import Config
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        PhaseBCDayFirst,
    )

    output_base = tmp_path / "out"
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True)

    # Single day with one episode, 300s long, GPS covering it.
    ep = SimpleNamespace(
        index=0,
        concat_path=Path("/fake/ep_000.mp4"),
        date="2026-02-20",
        total_duration=300.0,
        gpx_path=None,
        creation_time=None,
    )
    ingest_plan = SimpleNamespace(episodes=[ep])

    trackpoints = [_tp(t, 32.30 + t * 0.00001, -80.60) for t in range(0, 300, 10)]

    state = make_state(
        output_base=output_base,
        ingest_plan=ingest_plan,
        episode_files=["/fake/ep_000.mp4"],
        episode_dates=["2026-02-20"],
        episode_locations=[""],
        file_durations=[300.0],
        region="usa",
        location="SC Trip",
        drive_scores=None,
        file_boundaries=[(0.0, 300.0)],
        gps_per_episode=[(GPSData(trackpoints=trackpoints), 0.0)],
    )

    geocoder = MagicMock()
    geocoder.reverse_geocode_detailed.return_value = {
        "city": "Port Royal",
        "county": "Beaufort County",
        "state": "SC",
    }

    captured: dict = {}

    def _fake_run_all_days(**kwargs):
        captured.update(kwargs)
        artifacts = []
        for batch in kwargs["day_batches"]:
            artifacts.append(
                SimpleNamespace(
                    date=batch.date,
                    master_path=Path(f"/fake/master_{batch.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
            )
        return artifacts

    cfg = Config()
    phase = PhaseBCDayFirst(cfg)

    with (
        patch(
            "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
            return_value=geocoder,
        ),
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
            side_effect=_fake_run_all_days,
        ),
        # Offline settlement lookup for the closure's trace check:
        # Port Royal sits ~0.7 km from the trimmed fixes below, so the voted
        # town verifies without calling the live geocoder.
        patch(
            "gomoveshift.pipeline.drive_phases.phase_bc_day_first._forward_geocode_places",
            return_value=[("Port Royal", 32.305, -80.605)],
        ),
    ):
        phase.run(state, probe_duration_fn=lambda p: 100.0)

    # run_all_days was called with per-date GPS in day-master timeline.
    gps_by_date = captured.get("gps_points_by_date")
    assert gps_by_date, "gps_points_by_date not threaded into run_all_days"
    assert "2026-02-20" in gps_by_date, "per-date GPS missing the day's key"
    day_pts = gps_by_date["2026-02-20"]
    assert day_pts, "per-date GPS list is empty — projection dropped all trackpoints"
    # Tuple shape: (t_master, lat, lon). Compatible with trim's default
    # time accessor (reads point[0]).
    assert len(day_pts[0]) == 3
    t_master, lat, lon = day_pts[0]
    assert 0.0 <= t_master <= 300.0
    assert 32.29 < lat < 32.35
    assert -80.62 < lon < -80.58

    # short_ost_text_fn is wired and, given the trimmed tuples the
    # trim helper produces, returns the geocoded label (not None). The
    # label now derives from the modal county voted across the trimmed
    # points (dominant_location), with the place name from that county.
    fn = captured.get("short_ost_text_fn")
    assert fn is not None, "short_ost_text_fn not threaded into run_all_days"
    trimmed = [(50.0, 32.30, -80.60), (60.0, 32.31, -80.61)]
    short_range = SimpleNamespace(rank=1, start_sec=40.0, end_sec=70.0)
    # No destination POI for this clip — keeps the resolver offline and asserts
    # the dominant_location fallback (behaviour is unchanged when
    # no destination resolves). The closure looks the helper up as a module
    # global at call time, so patch around the invocations.
    with patch(
        "gomoveshift.pipeline.gps_enrichment._resolve_short_destination",
        return_value=None,
    ):
        result = fn(trimmed, short_range)
        assert result == "Port Royal, SC", (
            f"short_ost_text_fn returned {result!r}; expected the place + state "
            "label derived from the voted dominant county"
        )

        # Empty trimmed → None (so the shorts loop falls back to location).
        assert fn([], short_range) is None


def test_gps_projection_respects_multi_episode_day_cursor(tmp_path):
    """Two episodes concatenated on the same day → the second episode's
    trackpoints are offset by the first episode's duration in the day-
    master timeline."""
    from gomoveshift.analysis.shorts_models import GPSData
    from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
        _build_gps_points_by_date,
    )
    from gomoveshift.pipeline.day_grouper import DayBatch

    ep0 = SimpleNamespace(
        index=0, concat_path=Path("/fake/e0.mp4"), date="2026-02-20", total_duration=100.0
    )
    ep1 = SimpleNamespace(
        index=1, concat_path=Path("/fake/e1.mp4"), date="2026-02-20", total_duration=200.0
    )
    batch = DayBatch(date="2026-02-20", episodes=[ep0, ep1])

    gps_per_episode = [
        (GPSData(trackpoints=[_tp(50, 32.0, -80.0)]), 0.0),
        (GPSData(trackpoints=[_tp(30, 33.0, -81.0)]), 0.0),
    ]

    result = _build_gps_points_by_date(
        [batch],
        episode_files=["/fake/e0.mp4", "/fake/e1.mp4"],
        gps_per_episode=gps_per_episode,
    )

    assert "2026-02-20" in result
    points = sorted(result["2026-02-20"], key=lambda p: p[0])
    assert len(points) == 2
    # ep0's point at t=50 (local) → t=50 in master.
    assert points[0] == (50.0, 32.0, -80.0)
    # ep1's point at t=30 (local) → t=100+30=130 in master.
    assert points[1] == (130.0, 33.0, -81.0)
