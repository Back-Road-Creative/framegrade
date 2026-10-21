"""Tests for the GPS-quality filter wiring at the derive_per_lf_locations
stage of ``phase_bc_day_first``.

``derive_per_lf_locations`` takes a ``gps_points_by_lf`` kwarg that,
when supplied, runs the raw per-LF GPS points through
``filter_high_quality_points`` before computing the centroid that feeds
the reverse-geocoder. The filter keeps noisy /
stationary fixes from dragging centroids into neighboring towns.

The two production call sites in ``PhaseBCDayFirst.run`` (the OST-label
derivation before ``run_all_days`` and the defensive recomputation after
it) must pass ``gps_points_by_lf``; otherwise the quality-filter branch is
dead in production and the derive-stage call sites fall through to
``_centroid_for_segments(segments)`` unfiltered.

These tests assert that when ``state["gps_per_episode"]`` carries raw
GPS trackpoints, ``PhaseBCDayFirst.run`` threads them per-LF into
``derive_per_lf_locations`` and ``filter_high_quality_points`` is
actually exercised in production.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from _state_helpers import make_state
from gomoveshift.pipeline.drive_phases.phase_bc_day_first import PHASE_BC_CODE_VERSION


class _Loc:
    def __init__(self, name, county=None, state=None, country=None):
        self.city = name
        self.locality = name
        self.name = name
        self.suburb = None
        self.county = county
        self.state = state
        self.country = country


def _make_real_lf_dict(index, segments):
    return {
        "index": index,
        "segments": segments,
        "montage_peaks": [segments[0]] if segments else [],
        "shorts_target": 1,
        "region": "usa",
        "total_duration": sum(s["end_time"] - s["start_time"] for s in segments),
        "undersized": False,
    }


def _seg_dict(ep_idx, start, end, lat, lon):
    return {
        "source_file_index": ep_idx,
        "source_path": f"/fake/ep_{ep_idx:03d}.mp4",
        "start_time": float(start),
        "end_time": float(end),
        "original_start_time": float(start),
        "original_end_time": float(end),
        "composite_score": 0.5,
        "gps_center_lat": lat,
        "gps_center_lon": lon,
        "terrain_type": "coastal",
        "contiguous_with_previous": False,
        "avg_motion_energy": 0.0,
    }


def _write_drive_plan(stages_dir: Path, lfs: list) -> Path:
    payload = {
        "source_files": [],
        "processed_files": [],
        "regular_lfs": lfs,
        "score_threshold": 0.0,
        "total_source_duration": 0.0,
        "total_highlight_duration": 0.0,
        "highlight_ratio": 0.0,
        "best_moments_filtered": False,
        "plan_code_version": PHASE_BC_CODE_VERSION,
    }
    path = stages_dir / "drive_plan.json"
    path.write_text(json.dumps(payload))
    return path


def _trackpoint(ts, lat, lon):
    from gomoveshift.analysis.shorts_models import GPSTrackpoint

    return GPSTrackpoint(lat=lat, lon=lon, elevation=0.0, timestamp=float(ts))


class TestGPSQualityFilterAtDeriveStage:
    def test_gps_points_by_lf_threaded_and_filter_exercised(self, tmp_path):
        """``PhaseBCDayFirst.run`` must pass ``gps_points_by_lf`` through
        to ``derive_per_lf_locations`` so ``filter_high_quality_points``
        is actually invoked in production.

        Given a state with ``gps_per_episode`` holding raw trackpoints
        that cover each LF's segment time-range, the phase must slice
        the points per-LF and feed them through the quality filter
        before the centroid is computed.
        """
        from gomoveshift.analysis.shorts_models import GPSData
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        output_base = tmp_path / "out"
        stages_dir = output_base / "_stages"
        stages_dir.mkdir(parents=True)

        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)]),
            _make_real_lf_dict(2, [_seg_dict(0, 500, 600, 34.0, -81.0)]),
        ]
        _write_drive_plan(stages_dir, lfs)

        ep = SimpleNamespace(
            index=0,
            concat_path=Path("/fake/ep_000.mp4"),
            date="2026-02-20",
            total_duration=1500.0,
            gpx_path=None,
            creation_time=None,
        )
        ingest_plan = SimpleNamespace(episodes=[ep])

        # Build a GPSData trace spanning both LFs' segments. The phase
        # should slice by (segment.start_time + gps_offset, end_time +
        # gps_offset) per LF — LF1 gets points around t=100-200, LF2
        # gets points around t=500-600.
        trackpoints = [
            _trackpoint(100, 32.30, -80.60),
            _trackpoint(150, 32.31, -80.61),
            _trackpoint(200, 32.32, -80.62),
            _trackpoint(500, 31.10, -81.40),
            _trackpoint(550, 31.11, -81.41),
            _trackpoint(600, 31.12, -81.42),
        ]
        gps_data = GPSData(trackpoints=trackpoints)

        state = make_state(
            output_base=output_base,
            ingest_plan=ingest_plan,
            episode_files=["/fake/ep_000.mp4"],
            episode_dates=["2026-02-20"],
            episode_locations=[""],
            file_durations=[1500.0],
            region="usa",
            location="SC Trip",
            drive_scores=None,
            file_boundaries=[(0.0, 1500.0)],
            gps_per_episode=[(gps_data, 0.0)],
        )

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = [
            _Loc("Port Royal", state="SC", country="USA"),
            _Loc("Beaufort", state="SC", country="USA"),
        ]

        def _fake_run_all_days(**kwargs):
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

        # Spy on filter_high_quality_points at its module of origin so
        # we see BOTH imports (phase_bc_day_first's lazy import and
        # title_builder's lazy import both resolve to this symbol).
        from gomoveshift.pipeline import gps_quality as gps_quality_mod

        real_filter = gps_quality_mod.filter_high_quality_points
        calls: list[list] = []

        def _spy_filter(points, **kwargs):
            calls.append(list(points or []))
            return real_filter(points, **kwargs)

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
            # Fixes 50 s apart are all GPS gap to the retained-route check;
            # that check has its own tests (test_retained_gps_claims.py) and
            # this one covers the quality-filter wiring only.
            patch(
                "gomoveshift.pipeline.route_claims.enforce_route_claims",
                return_value={},
            ),
            # This test deliberately bypasses route measurement, so title
            # approval has no route record to consume either.
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first"
                "._approve_waypost_titles_before_encode",
            ),
            patch.object(
                gps_quality_mod,
                "filter_high_quality_points",
                side_effect=_spy_filter,
            ),
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)

        # Filter must have been called with each LF's sliced points.
        # Before the fix, phase_bc_day_first never passed
        # ``gps_points_by_lf`` so the filter branch was dead → zero
        # calls from the derive-stage code path.
        non_empty = [c for c in calls if c]
        assert non_empty, (
            "filter_high_quality_points was never called with GPS "
            "points from the derive-stage path — the "
            "wiring gap is still open"
        )
        # At least one call must correspond to LF1's time-range
        # (timestamps 100-200) and one to LF2's (500-600). The spy
        # receives GPSTrackpoint objects, not dicts.
        lf1_seen = any(all(100.0 <= p.timestamp <= 200.0 for p in c) for c in non_empty)
        lf2_seen = any(all(500.0 <= p.timestamp <= 600.0 for p in c) for c in non_empty)
        assert lf1_seen, "no filter call matched LF1's segment timerange"
        assert lf2_seen, "no filter call matched LF2's segment timerange"

    def test_centroid_uses_gps_points_not_segment_decoy(self, tmp_path):
        """End-to-end observation: the reverse_geocoder must receive a
        centroid computed from the GPS trackpoints, not from the
        segment's own ``gps_center_lat`` / ``gps_center_lon``.

        Synthetic trace: LF1's segment window contains clean points
        clustered near (32.30, -80.60), while the segment carries a
        decoy gps_center at (10.0, 10.0). The fallback
        ``_centroid_for_segments(segments)`` path would use the decoy,
        so a centroid near the clean cluster proves the GPS-points
        branch ran.
        """
        from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        output_base = tmp_path / "out"
        stages_dir = output_base / "_stages"
        stages_dir.mkdir(parents=True)

        # Segment has gps_center = (10.0, 10.0) — a decoy. If the fix
        # is wired correctly, the GPS-points branch runs first and
        # derives the centroid from filtered trackpoints near 32.30.
        # If the fix is missing, the fallback segment-centroid at
        # (10.0, 10.0) is used.
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 10.0, 10.0)]),
        ]
        _write_drive_plan(stages_dir, lfs)

        ep = SimpleNamespace(
            index=0,
            concat_path=Path("/fake/ep_000.mp4"),
            date="2026-02-20",
            total_duration=1500.0,
            gpx_path=None,
            creation_time=None,
        )
        ingest_plan = SimpleNamespace(episodes=[ep])

        # Clean cluster near (32.30, -80.60); no point near the decoy.
        clean = [
            _trackpoint(100, 32.30, -80.60),
            _trackpoint(150, 32.31, -80.61),
            _trackpoint(200, 32.29, -80.59),
        ]
        gps_data = GPSData(trackpoints=clean)

        state = make_state(
            output_base=output_base,
            ingest_plan=ingest_plan,
            episode_files=["/fake/ep_000.mp4"],
            episode_dates=["2026-02-20"],
            episode_locations=[""],
            file_durations=[1500.0],
            region="usa",
            location="",
            drive_scores=None,
            file_boundaries=[(0.0, 1500.0)],
            gps_per_episode=[(gps_data, 0.0)],
        )

        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = _Loc("Port Royal", state="SC", country="USA")

        def _fake_run_all_days(**kwargs):
            return [
                SimpleNamespace(
                    date=batch.date,
                    master_path=Path(f"/fake/master_{batch.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for batch in kwargs["day_batches"]
            ]

        cfg = Config()
        phase = PhaseBCDayFirst(cfg)

        # Silence unused GPSTrackpoint import
        _ = GPSTrackpoint

        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=_fake_run_all_days,
            ),
            # A MagicMock geocoder resolves no retained place; the route
            # check has its own tests (test_retained_gps_claims.py).
            patch(
                "gomoveshift.pipeline.route_claims.enforce_route_claims",
                return_value={},
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first"
                "._approve_waypost_titles_before_encode",
            ),
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)

        # The geocoder must have been called with the GPS-derived
        # centroid (~32.30, -80.60), NOT the segment-centroid decoy
        # at (10.0, 10.0). This proves the GPS-points branch ran.
        called_lats = [c.args[0] for c in geocoder.reverse_geocode.call_args_list]
        called_lons = [c.args[1] for c in geocoder.reverse_geocode.call_args_list]
        assert called_lats, "reverse_geocode was never called"
        assert any(abs(lat - 32.30) < 0.5 for lat in called_lats), (
            f"reverse_geocode received lats {called_lats} — none near "
            "32.30, suggesting the GPS-points branch did not run and "
            "the derive-stage fell through to the segment-centroid decoy"
        )
        assert any(abs(lon - (-80.60)) < 0.5 for lon in called_lons), (
            f"reverse_geocode received lons {called_lons} — none near -80.60"
        )
