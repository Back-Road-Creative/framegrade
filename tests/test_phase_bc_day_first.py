"""Tests for PhaseBCDayFirst per-LF GPS-derived location assignment
and real-plan threading into run_all_days.

The original phase_bc_day_first copied a day-wide location to every LF
in that day → duplicate YouTube titles across LFs on the same day.
The fix derives each LF's location from the GPS centroid of its
segments and reverse-geocodes. Where two LFs resolve to the same
place, disambiguation tries a road-name qualifier first, then falls
back to a numeric ``(Part N)`` suffix when the geocoder is unavailable.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
    _assert_drive_plan_built,
    _assert_skeleton_lf_has_location,
)

from gomoveshift.pipeline.drive_phases import phase_bc_day_first as phase_bc_day_first_mod
from _state_helpers import make_state


class _Loc:
    def __init__(self, name, county=None, state=None, country=None):
        self.city = name
        self.locality = name
        self.name = name
        self.suburb = None
        self.county = county
        self.state = state
        self.country = country


def _segment(lat, lon):
    return SimpleNamespace(
        gps_center_lat=lat,
        gps_center_lon=lon,
        source_file_index=0,
        start_time=0.0,
        end_time=10.0,
    )


class TestPerLFLocationUniqueness:
    def test_three_distinct_centroids_produce_three_unique_labels(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            derive_per_lf_locations,
        )

        # Two segment centroids per LF: geocode quorum (the bare-centroid
        # fallback is gone — labels come from the county vote).
        lfs = [
            SimpleNamespace(segments=[_segment(35.0, -80.0), _segment(35.1, -80.1)]),
            SimpleNamespace(segments=[_segment(34.0, -81.0), _segment(34.1, -81.1)]),
            SimpleNamespace(segments=[_segment(33.0, -82.0), _segment(33.1, -82.1)]),
        ]

        geocoder = MagicMock()
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        result = derive_per_lf_locations(
            lfs,
            geocoder=geocoder,
            gps_fallback=None,
        )

        assert len(set(result.values())) == 3
        assert all(v for v in result.values())

    def test_identical_centroids_disambiguate_with_road_name(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            derive_per_lf_locations,
        )

        # Two segment centroids per LF: geocode quorum.
        lfs = [
            SimpleNamespace(segments=[_segment(35.0, -80.0), _segment(35.0, -80.0)]),
            SimpleNamespace(segments=[_segment(35.0, -80.0), _segment(35.0, -80.0)]),
        ]

        # No ``gps_points_by_lf`` (resume) VOTES the modal county over
        # the segment centroids via ``reverse_geocode_detailed`` (county/state),
        # then disambiguates duplicates with the winning-county road carried by
        # that same vote (dom.road) — NEVER a later detailed call on the bare
        # centroid, which on a boundary-crossing LF sits in the wrong county.
        geocoder = MagicMock()
        geocoder.reverse_geocode_detailed.side_effect = [
            *[{"county": "Buncombe County", "state": "NC"}] * 2,  # LF1 votes
            *[{"county": "Buncombe County", "state": "NC", "road": "Interstate 40"}] * 2,  # LF2
        ]

        result = derive_per_lf_locations(
            lfs,
            geocoder=geocoder,
            gps_fallback=None,
        )

        values = list(result.values())
        assert len(set(values)) == 2
        # Road names surface — no numeric fallback needed
        assert any("Parkway" in v or "Interstate" in v for v in values)
        assert not any("Part " in v for v in values)

    def test_per_lf_locations_uses_filtered_gps(self):
        """When ``gps_points_by_lf`` is supplied, noisy/
        stationary points must be filtered via ``filter_high_quality_points``
        before geocoding, so the reverse-geocoder only sees CLEAN points —
        not coords dragged by GPS noise. The label now derives from the
        modal county voted across those clean points (``dominant_location``),
        so we assert every geocode call lands on the clean cluster.
        """
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            derive_per_lf_locations,
        )

        lf = SimpleNamespace(
            index=1,
            segments=[_segment(35.0, -80.0)],  # fallback if no gps_points
        )
        clean = [
            {"lat": 32.30, "lon": -80.60, "hdop": 1.0, "speed_mps": 12.0, "time": 0.0},
            {"lat": 32.31, "lon": -80.61, "hdop": 1.0, "speed_mps": 12.0, "time": 10.0},
        ]
        noise = [
            {"lat": 33.0, "lon": -79.0, "hdop": 9.0, "speed_mps": 12.0, "time": 5.0},
            {"lat": 33.0, "lon": -79.0, "hdop": 1.0, "speed_mps": 0.0, "time": 7.0},
        ]
        seen_coords = []

        def _detailed(lat, lon):
            seen_coords.append((lat, lon))
            return {"county": "Beaufort County", "state": "South Carolina"}

        geocoder = MagicMock()
        geocoder.reverse_geocode_detailed.side_effect = _detailed

        result = derive_per_lf_locations(
            [lf],
            geocoder=geocoder,
            gps_fallback=None,
            gps_points_by_lf={1: clean + noise},
        )

        assert "Beaufort County" in result[1]
        assert seen_coords, "geocoder was never called"
        # Every geocode call must be on the clean cluster — noise filtered out.
        for lat, lon in seen_coords:
            assert abs(lat - 32.30) < 0.2
            assert abs(lon - (-80.60)) < 0.2

    # ── derive is total; ambiguous input raises ──────
    def test_no_gps_no_override_raises(self):
        """Zero-centroid LFs without an override raise ValueError
        (no drive-wide fallback). Replaces the legacy
        ``test_no_geocoder_duplicates_get_numeric_suffix`` /
        ``test_zero_centroid_falls_back_to_day_location`` cases that
        exercised the deleted ``day_fallback`` path."""
        import pytest

        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            derive_per_lf_locations,
        )

        lfs = [
            SimpleNamespace(index=1, segments=[_segment(0.0, 0.0)]),
            SimpleNamespace(index=2, segments=[_segment(0.0, 0.0)]),
        ]
        with pytest.raises(ValueError) as exc:
            derive_per_lf_locations(lfs, geocoder=None, gps_fallback=None)
        msg = str(exc.value)
        assert "1" in msg and "2" in msg
        assert "--per-lf-location" in msg

    def test_overrides_supply_label_for_no_gps_lfs(self):
        """Operator overrides cover GPS-less LFs (sole legitimate
        path now that the drive-wide fallback is gone)."""
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            derive_per_lf_locations,
        )

        lfs = [
            SimpleNamespace(index=1, segments=[_segment(0.0, 0.0)]),
            SimpleNamespace(index=2, segments=[_segment(0.0, 0.0)]),
        ]
        result = derive_per_lf_locations(
            lfs,
            geocoder=None,
            gps_fallback=None,
            overrides={
                1: "Killarney to Dingle via N86, Kerry",
                2: "Dingle to Tralee via R559, Kerry",
            },
        )
        assert result[1] == "Killarney to Dingle via N86, Kerry"
        assert result[2] == "Dingle to Tralee via R559, Kerry"

    def test_mixed_gps_and_override_inputs(self):
        """Some LFs supply GPS, some supply overrides — both work."""
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            derive_per_lf_locations,
        )

        lfs = [
            # Two segment centroids so the geocode quorum (≥2 located
            # samples) is met — both vote Buncombe via the mock geocoder.
            SimpleNamespace(
                index=1, segments=[_segment(35.0, -80.0), _segment(35.1, -80.1)]
            ),  # GPS
            SimpleNamespace(index=2, segments=[_segment(0.0, 0.0)]),  # override
        ]
        # The GPS LF (no ``gps_points_by_lf``) votes the modal county over
        # its segment centroids via ``reverse_geocode_detailed``.
        geocoder = MagicMock()
        geocoder.reverse_geocode_detailed.return_value = {
            "county": "Buncombe County",
            "state": "NC",
        }
        result = derive_per_lf_locations(
            lfs,
            geocoder=geocoder,
            gps_fallback=None,
            overrides={2: "Custom route, NC"},
        )
        assert result[1] == "Buncombe County, NC"
        assert result[2] == "Custom route, NC"


class TestPostDiscoveryValidationGate:
    """The post-discovery, pre-encode validation gate
    refuses to enter encoding when an LF lacks both GPS and an override.
    """

    def test_gate_raises_when_lf_lacks_gps_and_override(self):
        import pytest

        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_per_lf_locations_resolvable,
        )

        plans = {
            "2026-02-20": [
                SimpleNamespace(index=1, segments=[_segment(0.0, 0.0)]),
                SimpleNamespace(index=2, segments=[_segment(35.0, -80.0)]),
            ]
        }
        with pytest.raises(ValueError) as exc:
            _assert_per_lf_locations_resolvable(plans, gps_points_by_lf={}, overrides={})
        msg = str(exc.value)
        assert "[1]" in msg or "1" in msg
        assert "--per-lf-location" in msg
        assert "--list-lfs" in msg

    def test_gate_passes_when_override_covers_gps_less_lf(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_per_lf_locations_resolvable,
        )

        plans = {
            "2026-02-20": [
                SimpleNamespace(index=1, segments=[_segment(0.0, 0.0)]),
                SimpleNamespace(index=2, segments=[_segment(35.0, -80.0)]),
            ]
        }
        # Override covers LF 1; GPS covers LF 2 → no raise.
        _assert_per_lf_locations_resolvable(
            plans,
            gps_points_by_lf={},
            overrides={1: "Some Route, Kerry"},
        )

    def test_gate_passes_when_gps_points_present_even_without_segment_centroid(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_per_lf_locations_resolvable,
        )

        plans = {
            "2026-02-20": [
                SimpleNamespace(index=1, segments=[_segment(0.0, 0.0)]),
            ]
        }
        # Raw points present → counts as GPS-bearing.
        _assert_per_lf_locations_resolvable(
            plans,
            gps_points_by_lf={1: [{"lat": 1.0, "lon": 2.0}]},
            overrides={},
        )


class TestDrivePlanBuiltGate:
    """A failed plan build (real_plan None → empty ``lf_plans_by_date``)
    makes the per-LF resolvable gate iterate nothing, so ``run_all_days`` would
    encode on the drive-wide fallback cascade. ``_assert_drive_plan_built``
    hard-fails that — but ONLY when the plan should have built (footage queued +
    complete inputs), so no-footage / no-scores paths are unaffected.
    """

    def _state(self, *, drive_scores=True):
        return SimpleNamespace(
            drive_scores=(object() if drive_scores else None),
            file_boundaries=[(0.0, 100.0)],
            episode_files=["/fake/ep0.mp4"],
        )

    def _batches(self, *, with_episodes=True, n=2):
        eps = [SimpleNamespace(index=0)] if with_episodes else []
        return [SimpleNamespace(date=f"2026-02-2{i}", episodes=eps) for i in range(n)]

    def test_raises_when_plan_none_with_footage_and_inputs(self):
        import pytest

        with pytest.raises(ValueError) as exc:
            _assert_drive_plan_built(None, self._state(), self._batches())
        assert "--per-lf-location" in str(exc.value)
        assert "2" in str(exc.value)  # day-batch count surfaced

    def test_passes_when_plan_present(self):
        _assert_drive_plan_built(object(), self._state(), self._batches())

    def test_passes_when_no_footage_even_if_inputs_present(self):
        # Empty-episode batches (nothing to encode) → no raise.
        _assert_drive_plan_built(None, self._state(), self._batches(with_episodes=False))

    def test_passes_when_scores_missing(self):
        # drive_scores=None → plan legitimately can't build (bookkeeping fixture).
        _assert_drive_plan_built(None, self._state(drive_scores=False), self._batches())


class TestSkeletonLFResolvableGate:
    """The post-``run_all_days`` fallback OST loop synthesizes a
    ``RegularLFPlan`` (empty segments) for a render-skip gap. A skeleton absent
    from ``per_lf_locations`` used to silently take the drive-wide label via
    ``per_lf_locations.get(idx, drive_wide_location)`` — a bypass; raise instead.
    """

    def test_skeleton_lf_without_location_or_override_raises(self):
        import pytest

        with pytest.raises(ValueError) as exc:
            _assert_skeleton_lf_has_location(7, {1: "Kerry"}, {}, "County Kerry")
        assert "7" in str(exc.value)
        assert "--per-lf-location" in str(exc.value)

    def test_skeleton_lf_with_resolved_location_passes(self):
        _assert_skeleton_lf_has_location(7, {7: "Cork, Ireland"}, {}, "County Kerry")

    def test_skeleton_lf_with_operator_override_passes(self):
        # Operator override is authoritative — covers the gap (non-empty map, LF 7 missing).
        _assert_skeleton_lf_has_location(
            7, {1: "Cork, Ireland"}, {7: "Kenmare to Sneem"}, "County Kerry"
        )

    def test_empty_per_lf_locations_is_the_plan_built_gates_domain(self):
        # Wholly-empty map (no plan resolved) is the domain of the
        # drive-plan-built gate — no raise here.
        _assert_skeleton_lf_has_location(7, {}, {}, "County Kerry")


# --- real LF plans threaded into run_all_days ---------------------


def _make_real_lf_dict(index, segments):
    """Shape a dict compatible with RegularLFPlan.from_dict.

    Each segment is split into two contiguous halves so the geocode
    quorum (≥2 located samples) is met — the bare-centroid
    geocode fallback is deleted; per-LF labels only come from the
    dominant-county vote over the per-segment centroids.
    """
    split = []
    for seg in segments:
        mid = (seg["start_time"] + seg["end_time"]) / 2
        split.append(dict(seg, end_time=mid))
        split.append(dict(seg, start_time=mid, contiguous_with_previous=True))
    segments = split
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


def _detailed_by_lat(lat, lon):
    """Vote-path geocoder stub: county/state keyed by int(lat)."""
    return {
        35: {"county": "Buncombe County", "state": "NC"},
        34: {"county": "Richland County", "state": "SC"},
        33: {"county": "Charleston County", "state": "SC"},
        32: {"county": "Beaufort County", "state": "SC"},
    }[int(lat)].copy()


class TestRealLFPlansThreaded:
    def _write_drive_plan(self, stages_dir: Path, lfs: list) -> Path:
        payload = {
            "source_files": [],
            "processed_files": [],
            "regular_lfs": lfs,
            "score_threshold": 0.0,
            "total_source_duration": 0.0,
            "total_highlight_duration": 0.0,
            "highlight_ratio": 0.0,
            "best_moments_filtered": False,
        }
        path = stages_dir / "drive_plan.json"
        path.write_text(json.dumps(payload))
        return path

    def test_real_lf_plans_produce_unique_per_lf_locations(self, tmp_path):
        """With a real drive_plan.json on disk, run() loads the real
        RegularLFPlans and derives per-LF labels from their segment
        centroids — not skeleton empty-segment fallbacks."""
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
            _make_real_lf_dict(3, [_seg_dict(0, 1000, 1100, 33.0, -82.0)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        ep = SimpleNamespace(
            index=0,
            concat_path=Path("/fake/ep_000.mp4"),
            date="2026-02-20",
            total_duration=1500.0,
            gpx_path=None,
            creation_time=None,
        )
        ingest_plan = SimpleNamespace(episodes=[ep])

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
        )

        places = iter(["Asheville", "Columbia", "Charleston"])

        def _rg(lat, lon):
            return _Loc(next(places), state="SC", country="USA")

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        captured_kwargs = {}

        def _fake_run_all_days(**kwargs):
            captured_kwargs.update(kwargs)
            artifacts = []
            # Produce one artifact per day batch — no actual LF paths
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
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)

        # run_all_days was called with lf_plans_by_date keyed by date
        assert "lf_plans_by_date" in captured_kwargs
        lf_by_date = captured_kwargs["lf_plans_by_date"]
        assert "2026-02-20" in lf_by_date
        assert len(lf_by_date["2026-02-20"]) == 3
        # And plan objects have real segments, not empty skeletons
        assert all(len(getattr(p, "segments", [])) > 0 for p in lf_by_date["2026-02-20"])

        # per_lf_locations derived from real centroids — three unique labels
        labels = state.get("per_lf_locations") or {}
        assert len(labels) == 3
        assert len(set(labels.values())) == 3

    def test_audit_visible_state_preseeded_before_run_all_days(self, tmp_path):
        """per_lf_titles + per_lf_recording_date must be populated
        BEFORE run_all_days runs, because the OST audit gate inside
        run_day_pipeline writes ost_audit_<date>.json from those state
        fields (no kwarg fallback) before the apply_ost loop. Capturing
        them at run_all_days call-time proves the audit sees real
        verb-bearing titles + dates, not the empty placeholders that the
        post-run population left on a fresh run."""
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
        self._write_drive_plan(stages_dir, lfs)

        ep = SimpleNamespace(
            index=0,
            concat_path=Path("/fake/ep_000.mp4"),
            date="2026-02-20",
            total_duration=1500.0,
            gpx_path=None,
            creation_time=None,
        )
        ingest_plan = SimpleNamespace(episodes=[ep])

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
        )

        places = iter(["Asheville", "Columbia"])

        def _rg(lat, lon):
            return _Loc(next(places), state="SC", country="USA")

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        seen_at_call_time: dict = {}
        approval_order = []
        real_approve = phase_bc_day_first_mod._approve_waypost_titles_before_encode

        def _approve_before_encode(*args, **kwargs):
            approval_order.append("approve")
            return real_approve(*args, **kwargs)

        def _fake_run_all_days(**kwargs):
            approval_order.append("encode")
            # Snapshot the audit-visible state at the moment the day loop
            # (which fires the OST audit gate) is entered.
            seen_at_call_time["titles"] = dict(state.get("per_lf_titles") or {})
            seen_at_call_time["dates"] = dict(state.get("per_lf_recording_date") or {})
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
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first"
                "._approve_waypost_titles_before_encode",
                side_effect=_approve_before_encode,
            ),
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)

        # Titles + recording dates were already populated when the audit
        # gate would have fired — not left empty until after run_all_days.
        assert set(seen_at_call_time["titles"].keys()) == {1, 2}
        assert all(v for v in seen_at_call_time["titles"].values())
        assert seen_at_call_time["dates"] == {1: "2026-02-20", 2: "2026-02-20"}
        assert approval_order == ["approve", "encode"]

    def test_lf_plans_partitioned_by_date_across_days(self, tmp_path):
        """4 LFs where LF 1-2 belong to day 1 (ep 0) and LF 3-4 belong
        to day 2 (ep 1). lf_plans_by_date must reflect that."""
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
            _make_real_lf_dict(3, [_seg_dict(1, 100, 200, 33.0, -82.0)]),
            _make_real_lf_dict(4, [_seg_dict(1, 500, 600, 32.0, -83.0)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        eps = [
            SimpleNamespace(
                index=0,
                concat_path=Path("/fake/ep_000.mp4"),
                date="2026-02-20",
                total_duration=1200.0,
                gpx_path=None,
                creation_time=None,
            ),
            SimpleNamespace(
                index=1,
                concat_path=Path("/fake/ep_001.mp4"),
                date="2026-02-21",
                total_duration=1200.0,
                gpx_path=None,
                creation_time=None,
            ),
        ]
        ingest_plan = SimpleNamespace(episodes=eps)

        state = make_state(
            output_base=output_base,
            ingest_plan=ingest_plan,
            episode_files=["/fake/ep_000.mp4", "/fake/ep_001.mp4"],
            episode_dates=["2026-02-20", "2026-02-21"],
            episode_locations=["", ""],
            file_durations=[1200.0, 1200.0],
            region="usa",
            location="",
            drive_scores=None,
            file_boundaries=[(0.0, 1200.0), (1200.0, 2400.0)],
        )

        captured_kwargs = {}

        def _fake_run_all_days(**kwargs):
            captured_kwargs.update(kwargs)
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        # The geocoder must return a usable label; the previous
        # ``return_value=None`` relied on the deleted day_fallback path.
        places = iter(["A", "B", "C", "D"])

        def _rg(lat, lon):
            return _Loc(next(places), state="SC", country="USA")

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

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
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)

        lf_by_date = captured_kwargs["lf_plans_by_date"]
        assert set(lf_by_date.keys()) == {"2026-02-20", "2026-02-21"}
        assert [p.index for p in lf_by_date["2026-02-20"]] == [1, 2]
        assert [p.index for p in lf_by_date["2026-02-21"]] == [3, 4]


class TestPerDayCompletedBookkeeping:
    """`per_day_completed` state bookkeeping.

    After each day batch's master lands successfully, phase_bc_day_first
    writes `{date: {"master_path": ..., "master_sha": ...}}` to
    `state.per_day_completed`. On resume, days with a recorded master
    that still exists on disk skip `run_all_days` entirely — no
    ffprobe, no re-encode. Missing or legacy state falls through to
    the existing file-existence + duration-probe gate.
    """

    def _base_state(self, tmp_path, dates=("2026-02-20", "2026-02-21")):
        output_base = tmp_path / "out"
        stages_dir = output_base / "_stages"
        stages_dir.mkdir(parents=True)

        episodes = []
        episode_files = []
        episode_dates_list = []
        file_durations = []
        file_boundaries = []
        cursor = 0.0
        for i, d in enumerate(dates):
            ep_path = f"/fake/ep_{i:03d}.mp4"
            episode_files.append(ep_path)
            episode_dates_list.append(d)
            file_durations.append(1500.0)
            file_boundaries.append((cursor, cursor + 1500.0))
            cursor += 1500.0
            episodes.append(
                SimpleNamespace(
                    index=i,
                    concat_path=Path(ep_path),
                    date=d,
                    total_duration=1500.0,
                    gpx_path=None,
                    creation_time=None,
                )
            )
        ingest_plan = SimpleNamespace(episodes=episodes)

        state = make_state(
            output_base=output_base,
            ingest_plan=ingest_plan,
            episode_files=episode_files,
            episode_dates=episode_dates_list,
            episode_locations=[""] * len(dates),
            file_durations=file_durations,
            region="usa",
            location="Test",
            drive_scores=None,
            file_boundaries=file_boundaries,
        )
        return state, output_base, stages_dir

    def _run_phase(self, state, run_all_days_impl, geocoder=None):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        if geocoder is None:
            geocoder = MagicMock()
            geocoder.reverse_geocode.return_value = None

        # Fast-path #1 now ffprobes recorded masters. These tests
        # write placeholder bytes to the master files; a real ffprobe call
        # would return 0.0 duration and reject them. Stub the client to
        # return a duration that always passes the ``expected * 0.99``
        # threshold so the existing per_day_completed contract is exercised
        # without being coupled to ffmpeg-readable test fixtures.
        class _PassingFFprobeClient:
            def __init__(self, _subprocess_module, *, timeout=30):
                pass

            def probe_duration(self, _path):
                return 1e9  # arbitrarily large — always >= expected * 0.99

        cfg = Config()
        phase = PhaseBCDayFirst(cfg)
        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=run_all_days_impl,
            ) as m,
            patch(
                "gomoveshift.pipeline.validation.ffprobe_client.FFprobeClient",
                _PassingFFprobeClient,
            ),
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)
        return m

    def test_fresh_run_records_per_day_completed_entries(self, tmp_path):
        """After a successful fresh run, state.per_day_completed has one
        entry per successful day keyed on master_path — and never streams
        the ~60 GB master for a whole-file sha (integrity is covered by the
        master_record size:mtime fingerprint, not a hash)."""
        state, output_base, stages_dir = self._base_state(tmp_path)

        # Simulate run_all_days producing real master files on disk
        def _fake_run_all_days(**kwargs):
            artifacts = []
            for batch in kwargs["day_batches"]:
                master = stages_dir / f"day_{batch.date}_master.mp4"
                master.write_bytes(b"fake-master-" + batch.date.encode())
                artifacts.append(
                    SimpleNamespace(
                        date=batch.date,
                        master_path=master,
                        regular_lf_paths=[],
                        short_paths=[],
                    )
                )
            return artifacts

        self._run_phase(state, _fake_run_all_days)

        recorded = state.get("per_day_completed")
        assert isinstance(recorded, dict)
        assert set(recorded.keys()) == {"2026-02-20", "2026-02-21"}
        for date, entry in recorded.items():
            assert "master_path" in entry
            assert "master_sha" not in entry  # no whole-file hash of the 60 GB master
            assert Path(entry["master_path"]).exists()

    def test_resume_skips_days_already_in_per_day_completed(self, tmp_path):
        """State marks 2 days as done with masters present on disk.
        On resume, run_all_days is called with NO day_batches (or not at
        all) — no re-probe, no re-encode."""
        state, output_base, stages_dir = self._base_state(tmp_path)

        # Pre-populate masters + per_day_completed records with real hashes
        import hashlib

        recorded = {}
        for d in ("2026-02-20", "2026-02-21"):
            master = stages_dir / f"day_{d}_master.mp4"
            master.write_bytes(b"pre-existing-" + d.encode())
            recorded[d] = {
                "master_path": str(master),
                "master_sha": hashlib.sha256(master.read_bytes()).hexdigest(),
            }
        state["per_day_completed"] = recorded

        call_count = {"n": 0, "batches_seen": []}

        def _fake_run_all_days(**kwargs):
            call_count["n"] += 1
            call_count["batches_seen"].extend([b.date for b in kwargs.get("day_batches") or []])
            return []

        self._run_phase(state, _fake_run_all_days)

        # Both days are in per_day_completed with masters on disk:
        # run_all_days must receive ZERO day_batches (or not be called).
        assert call_count["batches_seen"] == [], (
            "Days already in per_day_completed with existing masters must be "
            "skipped before reaching run_all_days"
        )
        # per_day_completed entries preserved
        assert set(state.get("per_day_completed") or {}) == {"2026-02-20", "2026-02-21"}

    def test_backward_compat_state_missing_per_day_completed_field(self, tmp_path):
        """Legacy state has no `per_day_completed` key — phase runs every
        day through the existing file-exists + duration-probe gate.
        Regression guard: nothing KeyErrors, fresh run executes normally.
        """
        state, output_base, stages_dir = self._base_state(tmp_path)
        # Explicitly remove / don't set per_day_completed — simulate legacy state
        assert "per_day_completed" not in state or state.get("per_day_completed") is None

        captured_batches = []

        def _fake_run_all_days(**kwargs):
            captured_batches.extend(kwargs["day_batches"])
            artifacts = []
            for batch in kwargs["day_batches"]:
                master = stages_dir / f"day_{batch.date}_master.mp4"
                master.write_bytes(b"m-" + batch.date.encode())
                artifacts.append(
                    SimpleNamespace(
                        date=batch.date,
                        master_path=master,
                        regular_lf_paths=[],
                        short_paths=[],
                    )
                )
            return artifacts

        self._run_phase(state, _fake_run_all_days)

        # Every day was dispatched to run_all_days (fall-through path)
        assert sorted(b.date for b in captured_batches) == ["2026-02-20", "2026-02-21"]
        # And the new bookkeeping landed as a side-effect of the fresh run
        recorded = state.get("per_day_completed") or {}
        assert set(recorded) == {"2026-02-20", "2026-02-21"}

    def test_resume_falls_through_when_recorded_master_missing(self, tmp_path):
        """per_day_completed lists a day as done, but the recorded
        master_path no longer exists on disk — phase must NOT skip,
        must fall through to run_all_days for that day (the existing
        gate is the backstop)."""
        state, output_base, stages_dir = self._base_state(tmp_path)

        # Record a stale entry pointing at a file that never existed
        state["per_day_completed"] = {
            "2026-02-20": {
                "master_path": str(stages_dir / "ghost_master.mp4"),
                "master_sha": "deadbeef",
            }
            # 2026-02-21 has no record at all — also falls through
        }

        seen = []

        def _fake_run_all_days(**kwargs):
            seen.extend(b.date for b in kwargs["day_batches"])
            artifacts = []
            for batch in kwargs["day_batches"]:
                master = stages_dir / f"day_{batch.date}_master.mp4"
                master.write_bytes(b"m-" + batch.date.encode())
                artifacts.append(
                    SimpleNamespace(
                        date=batch.date,
                        master_path=master,
                        regular_lf_paths=[],
                        short_paths=[],
                    )
                )
            return artifacts

        self._run_phase(state, _fake_run_all_days)

        # Both days fell through because stale record is invalid AND no record for day 2
        assert sorted(seen) == ["2026-02-20", "2026-02-21"]


class TestTrailFilmDriveWiring:
    """The drive hands the trail film a span vote built on ITS shared geocoder.

    ``attach_trail_film`` (fresh day) and ``retry_trail_skip`` (resume) fail
    closed without a working ``span_location``, so a broken wiring here turns
    every drive film into a recorded skip. The trail functions are replaced by
    recorders that call the ``span_location`` they were handed on a real GPX.
    """

    DAY = "2026-02-20"
    VOTE = {"Swain County, NC": 0.6, "Haywood County, NC": 0.4}

    def _gpx(self, tmp_path):
        gpx = tmp_path / "day.gpx"
        pts = "".join(
            f'<trkpt lat="35.{i}" lon="-83.{i}"><time>{self.DAY}T{h}:00:00Z</time></trkpt>'
            for i, h in enumerate(("14", "15", "20"))
        )
        gpx.write_text(
            '<gpx version="1.1" creator="t" xmlns="http://www.topografix.com/GPX/1/1">'
            "<trk><trkseg>" + pts + "</trkseg></trk></gpx>"
        )
        return gpx

    def _drive(self, tmp_path, monkeypatch, *, resume):
        from datetime import date

        from gomoveshift.models import ResolvedLocation
        from gomoveshift.pipeline import gps_enrichment
        from gomoveshift.pipeline.media_api import Deliverables

        harness = TestPerDayCompletedBookkeeping()
        state, _out, stages_dir = harness._base_state(tmp_path, dates=(self.DAY,))
        gpx = self._gpx(tmp_path)
        geocoder = MagicMock(name="drive_geocoder")
        geocoder.reverse_geocode.return_value = None
        votes: list = []
        monkeypatch.setattr(
            gps_enrichment,
            "build_resolved_location",
            lambda points, *, geocoder, **kw: (
                votes.append((points, geocoder)) or ResolvedLocation(vote=dict(self.VOTE))
            ),
        )
        seen: dict = {}

        def _record(name):
            def _fn(*args, span_location=None, **kwargs):
                seen[name] = span_location
                if span_location is not None:
                    seen[name + "_vote"] = span_location(gpx, date.fromisoformat(self.DAY))
                return False

            return _fn

        monkeypatch.setattr(phase_bc_day_first_mod, "attach_trail_film", _record("attach"))
        monkeypatch.setattr(phase_bc_day_first_mod, "retry_trail_skip", _record("retry"))

        master = stages_dir / f"day_{self.DAY}_master.mp4"
        master.write_bytes(b"m")
        if resume:
            state["per_day_completed"] = {self.DAY: {"master_path": str(master)}}

        def _fake_run_all_days(**kwargs):
            day = Deliverables(date=self.DAY, master_path=master, regular_lf_paths=[])
            kwargs["on_day_complete"](day, state)
            return [day]

        harness._run_phase(state, _fake_run_all_days, geocoder=geocoder)
        return seen, votes, geocoder

    def test_fresh_day_gets_a_working_span_vote_on_the_drive_geocoder(self, tmp_path, monkeypatch):
        seen, votes, geocoder = self._drive(tmp_path, monkeypatch, resume=False)

        assert callable(seen["attach"])
        assert seen["attach_vote"].vote == self.VOTE
        # The vote was built over the GPX fixes with the drive's own geocoder.
        points, used = votes[-1]
        assert used is geocoder
        assert [round(p["lat"], 1) for p in points] == [35.0, 35.1, 35.2]
        assert "retry" not in seen  # no resumed day on a fresh run

    def test_resumed_day_retry_gets_a_working_span_vote_on_the_drive_geocoder(
        self, tmp_path, monkeypatch
    ):
        seen, votes, geocoder = self._drive(tmp_path, monkeypatch, resume=True)

        assert callable(seen["retry"])
        assert seen["retry_vote"].vote == self.VOTE
        assert votes[-1][1] is geocoder


class TestPerDayCompletedStateRoundTrip:
    """Serialization guards for the new field.

    `from_dict` must not KeyError on legacy state files that lack the
    new `per_day_completed` field, and must migrate legacy list-shaped
    values (the prior type) to None without raising.
    """

    def test_from_dict_accepts_legacy_state_without_per_day_completed(self):
        """Legacy state file predates the new field — from_dict loads it
        cleanly and per_day_completed defaults to None."""
        from gomoveshift.pipeline.drive_state_model import PipelineState

        legacy = {
            "source_dir": "/src",
            "output_base": "/out",
            "region": "usa",
            "phases_completed": ["phase_a"],
            # NOTE: no per_day_completed key
        }
        state = PipelineState.from_dict(legacy)
        assert state.per_day_completed is None

    def test_from_dict_migrates_legacy_list_per_day_completed(self):
        """Pre-fix, the field was declared as Optional[list] (unused).
        A state file with a list value (even empty) must not break
        new code expecting a dict."""
        from gomoveshift.pipeline.drive_state_model import PipelineState

        legacy = {
            "source_dir": "/src",
            "output_base": "/out",
            "per_day_completed": [],  # legacy list shape
        }
        state = PipelineState.from_dict(legacy)
        # Coerced to None so new skip logic falls through safely
        assert state.per_day_completed is None

    def test_from_dict_preserves_well_formed_dict_per_day_completed(self):
        """New-shape state file round-trips verbatim."""
        from gomoveshift.pipeline.drive_state_model import PipelineState

        payload = {
            "source_dir": "/src",
            "output_base": "/out",
            "per_day_completed": {
                "2026-02-20": {
                    "master_path": "/out/_stages/day_2026-02-20_master.mp4",
                    "master_sha": "abc123",
                },
            },
        }
        state = PipelineState.from_dict(payload)
        assert isinstance(state.per_day_completed, dict)
        assert state.per_day_completed["2026-02-20"]["master_sha"] == "abc123"

    def test_to_dict_from_dict_roundtrip_preserves_per_day_completed(self):
        """Serialize a populated state, round-trip through JSON, and
        assert the dict structure survives intact."""
        import json

        from gomoveshift.pipeline.drive_state_model import PipelineState

        original = PipelineState(
            source_dir=Path("/src"),
            output_base=Path("/out"),
            per_day_completed={
                "2026-02-20": {
                    "master_path": "/tmp/a.mp4",
                    "master_sha": "11" * 32,
                },
                "2026-02-21": {
                    "master_path": "/tmp/b.mp4",
                    "master_sha": "22" * 32,
                },
            },
        )
        payload = json.loads(json.dumps(original.to_dict()))
        restored = PipelineState.from_dict(payload)
        assert restored.per_day_completed == original.per_day_completed


class TestMasterDropAndFinalsResume:
    """`day_*_master.mp4` is dropped after bookkeeping lands, and
    resume reconstructs from ``lf_XX.mp4`` + ``short_XX.mp4`` finals alone.

    Gates that matter for safety:
      * Master is dropped only when both LF and short finals exist with
        non-zero size (otherwise the day is considered incomplete).
      * `keep_masters=True` preserves masters (debug / forensic).
      * On resume, a per_day_completed entry with master-missing-but-
        finals-present takes a dedicated fast-path — NOT re-encode.
      * A day with no finals AND no master falls through to re-encode.
    """

    def _base_state(self, tmp_path, dates=("2026-02-20",)):
        output_base = tmp_path / "out"
        stages_dir = output_base / "_stages"
        stages_dir.mkdir(parents=True)

        episodes = []
        episode_files = []
        episode_dates_list = []
        file_durations = []
        file_boundaries = []
        cursor = 0.0
        for i, d in enumerate(dates):
            ep_path = f"/fake/ep_{i:03d}.mp4"
            episode_files.append(ep_path)
            episode_dates_list.append(d)
            file_durations.append(1500.0)
            file_boundaries.append((cursor, cursor + 1500.0))
            cursor += 1500.0
            episodes.append(
                SimpleNamespace(
                    index=i,
                    concat_path=Path(ep_path),
                    date=d,
                    total_duration=1500.0,
                    gpx_path=None,
                    creation_time=None,
                )
            )
        ingest_plan = SimpleNamespace(episodes=episodes)
        state = make_state(
            output_base=output_base,
            ingest_plan=ingest_plan,
            episode_files=episode_files,
            episode_dates=episode_dates_list,
            episode_locations=[""] * len(dates),
            file_durations=file_durations,
            region="usa",
            location="Test",
            drive_scores=None,
            file_boundaries=file_boundaries,
        )
        return state, output_base, stages_dir

    def _run_phase(self, state, run_all_days_impl):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = None
        cfg = Config()
        phase = PhaseBCDayFirst(cfg)
        # drop_day_master now ffprobes shorts before dropping. The
        # synthetic byte fixtures here are not real mp4s, so patch the
        # integrity check to pass — these tests focus on the master-drop +
        # resume contract, not on ffprobe behaviour (covered in
        # test_pipeline_cleanup.py::TestVerifyShortsIntegrity).
        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=run_all_days_impl,
            ) as m,
            patch(
                "gomoveshift.pipeline.cleanup.verify_shorts_integrity",
                return_value={
                    "ok": True,
                    "reason": "",
                    "missing": [],
                    "bad": [],
                    "count_actual": 1,
                    "count_expected": 1,
                },
            ),
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)
        return m

    def _day_with_finals(self, stages_dir: Path, date: str, *, write_master=True):
        day_dir = stages_dir / f"day_{date}"
        (day_dir / "regular_lfs").mkdir(parents=True, exist_ok=True)
        (day_dir / "shorts").mkdir(parents=True, exist_ok=True)
        master = day_dir / f"day_{date}_master.mp4"
        if write_master:
            master.write_bytes(b"master-" + date.encode() + b"-" + b"X" * 1000)
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"lf" * 100)
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"short" * 50)
        return day_dir, master

    def test_fresh_run_drops_master_when_finals_present(self, tmp_path):
        state, output_base, stages_dir = self._base_state(tmp_path)

        def _fake_run_all_days(**kwargs):
            artifacts = []
            for batch in kwargs["day_batches"]:
                _, master = self._day_with_finals(stages_dir, batch.date)
                artifacts.append(
                    SimpleNamespace(
                        date=batch.date,
                        master_path=master,
                        regular_lf_paths=[master.parent / "regular_lfs" / "lf_01.mp4"],
                        short_paths=[master.parent / "shorts" / "short_01.mp4"],
                    )
                )
                kwargs["on_day_complete"](artifacts[-1], kwargs["state"])
            return artifacts

        self._run_phase(state, _fake_run_all_days)

        # Master file must be gone; per_day_completed records its path
        master = stages_dir / "day_2026-02-20" / "day_2026-02-20_master.mp4"
        assert not master.exists(), "Master must be dropped after bookkeeping"
        recorded = state.get("per_day_completed") or {}
        assert "2026-02-20" in recorded
        assert recorded["2026-02-20"]["master_path"], "path recorded before drop"
        assert "master_sha" not in recorded["2026-02-20"]  # no whole-file hash

    def test_keep_masters_preserves_master(self, tmp_path):
        state, output_base, stages_dir = self._base_state(tmp_path)
        state["keep_masters"] = True

        def _fake_run_all_days(**kwargs):
            artifacts = []
            for batch in kwargs["day_batches"]:
                _, master = self._day_with_finals(stages_dir, batch.date)
                artifacts.append(
                    SimpleNamespace(
                        date=batch.date,
                        master_path=master,
                        regular_lf_paths=[master.parent / "regular_lfs" / "lf_01.mp4"],
                        short_paths=[master.parent / "shorts" / "short_01.mp4"],
                    )
                )
                kwargs["on_day_complete"](artifacts[-1], kwargs["state"])
            return artifacts

        self._run_phase(state, _fake_run_all_days)

        master = stages_dir / "day_2026-02-20" / "day_2026-02-20_master.mp4"
        assert master.exists(), "keep_masters=True must preserve the master"

    def test_resume_uses_finals_when_master_dropped(self, tmp_path):
        """Master path recorded but file missing. Finals on disk. Must skip
        re-encode via the new finals-only fast-path, NOT call run_all_days."""
        state, output_base, stages_dir = self._base_state(tmp_path)

        # Simulate prior run: finals on disk, master dropped
        self._day_with_finals(stages_dir, "2026-02-20", write_master=False)
        state["per_day_completed"] = {
            "2026-02-20": {
                "master_path": str(stages_dir / "day_2026-02-20" / "day_2026-02-20_master.mp4"),
                "master_sha": "stale-sha-from-dropped-master",
            }
        }

        run_all_days_call_count = [0]

        def _fake_run_all_days(**kwargs):
            run_all_days_call_count[0] += 1
            # Must be called with zero batches — all days resumed from finals
            assert len(kwargs["day_batches"]) == 0
            return []

        self._run_phase(state, _fake_run_all_days)

        # run_all_days may or may not be called with empty batches depending
        # on the phase's short-circuit. The invariant is: NOT called with a
        # batch for a day that has finals on disk.
        # (Day 2026-02-20 must not be in any batch handed to run_all_days.)

    def test_resume_falls_through_when_no_master_and_no_finals(self, tmp_path):
        """Stale per_day_completed entry with no master and no finals on
        disk — must fall through to re-encode (the existing backstop)."""
        state, output_base, stages_dir = self._base_state(tmp_path)

        # Record entry but create no files on disk
        state["per_day_completed"] = {
            "2026-02-20": {
                "master_path": str(stages_dir / "day_2026-02-20" / "ghost.mp4"),
                "master_sha": "ghost",
            }
        }

        seen = []

        def _fake_run_all_days(**kwargs):
            seen.extend(b.date for b in kwargs["day_batches"])
            artifacts = []
            for batch in kwargs["day_batches"]:
                _, master = self._day_with_finals(stages_dir, batch.date)
                artifacts.append(
                    SimpleNamespace(
                        date=batch.date,
                        master_path=master,
                        regular_lf_paths=[master.parent / "regular_lfs" / "lf_01.mp4"],
                        short_paths=[master.parent / "shorts" / "short_01.mp4"],
                    )
                )
            return artifacts

        self._run_phase(state, _fake_run_all_days)

        # Day fell through because neither master nor finals existed
        assert seen == ["2026-02-20"]


class TestShortsPerLFPopulation:
    """``state["shorts_per_lf"]`` must be populated from each artifact's
    ``short_paths``, partitioned across the day's LFs.

    Regression: projects used to ship Metricool CSVs with 1 row per LF
    (fallback from ``max(num_shorts, 1)`` when ``shorts_per_lf[lf_idx]``
    was empty), so a day rendering 42 shorts across 3 LFs produced a CSV
    with 3 rows, not 42.
    """

    def _run_phase(self, state, run_all_days_impl):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = None

        cfg = Config()
        phase = PhaseBCDayFirst(cfg)
        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=run_all_days_impl,
            ),
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)

    def _base_state(self, tmp_path, dates):
        output_base = tmp_path / "out"
        stages_dir = output_base / "_stages"
        stages_dir.mkdir(parents=True)

        episodes = []
        episode_files = []
        episode_dates_list = []
        file_durations = []
        file_boundaries = []
        cursor = 0.0
        for i, d in enumerate(dates):
            ep_path = f"/fake/ep_{i:03d}.mp4"
            episode_files.append(ep_path)
            episode_dates_list.append(d)
            file_durations.append(1500.0)
            file_boundaries.append((cursor, cursor + 1500.0))
            cursor += 1500.0
            episodes.append(
                SimpleNamespace(
                    index=i,
                    concat_path=Path(ep_path),
                    date=d,
                    total_duration=1500.0,
                    gpx_path=None,
                    creation_time=None,
                )
            )
        state = make_state(
            output_base=output_base,
            ingest_plan=SimpleNamespace(episodes=episodes),
            episode_files=episode_files,
            episode_dates=episode_dates_list,
            episode_locations=[""] * len(dates),
            file_durations=file_durations,
            region="usa",
            location="Test",
            drive_scores=None,
            file_boundaries=file_boundaries,
        )
        return state, output_base, stages_dir

    def test_single_lf_day_gets_all_shorts(self, tmp_path):
        """Day with 1 LF + 14 shorts → shorts_per_lf[1] has 14 entries."""
        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))

        def _fake(**kwargs):
            master = stages_dir / "day_2026-02-20_master.mp4"
            master.write_bytes(b"m")
            lf_path = stages_dir / "day_2026-02-20" / "regular_lfs" / "lf_01.mp4"
            lf_path.parent.mkdir(parents=True, exist_ok=True)
            lf_path.write_bytes(b"lf")
            shorts = []
            for i in range(1, 15):
                sp = stages_dir / "day_2026-02-20" / "shorts" / f"short_{i:02d}.mp4"
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_bytes(b"s")
                shorts.append(sp)
            return [
                SimpleNamespace(
                    date="2026-02-20",
                    master_path=master,
                    regular_lf_paths=[lf_path],
                    short_paths=shorts,
                )
            ]

        self._run_phase(state, _fake)

        spl = state.get("shorts_per_lf") or {}
        assert 1 in spl, f"Expected key 1 in shorts_per_lf; got {list(spl.keys())}"
        assert len(spl[1]) == 14, f"Expected 14 shorts for LF 1, got {len(spl[1])}"

    def test_multi_lf_day_without_short_ranges_hard_fails(self, tmp_path):
        """A multi-LF day whose
        artifact lacks ``short_ranges`` cannot be attributed — the legacy
        equal-split (42 shorts → 14/14/14 by rank position) linked shorts
        to arbitrary parent LFs, so ambiguity now hard-fails instead."""
        import pytest

        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-24",))

        def _fake(**kwargs):
            master = stages_dir / "day_2026-02-24_master.mp4"
            master.write_bytes(b"m")
            lf_paths = []
            for i in range(1, 4):
                lp = stages_dir / "day_2026-02-24" / "regular_lfs" / f"lf_{i:02d}.mp4"
                lp.parent.mkdir(parents=True, exist_ok=True)
                lp.write_bytes(b"lf")
                lf_paths.append(lp)
            shorts = []
            for i in range(1, 43):
                sp = stages_dir / "day_2026-02-24" / "shorts" / f"short_{i:02d}.mp4"
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_bytes(b"s")
                shorts.append(sp)
            return [
                SimpleNamespace(
                    date="2026-02-24",
                    master_path=master,
                    regular_lf_paths=lf_paths,
                    short_paths=shorts,
                )
            ]

        with pytest.raises(RuntimeError, match="cannot attribute"):
            self._run_phase(state, _fake)

    def test_uneven_multi_lf_day_without_short_ranges_hard_fails(self, tmp_path):
        """2 LFs + 11 shorts with no source-window data used to split 6+5
        by rank position — arbitrary attribution (wrong first-comment link,
        recording date, trim ratio). Now a hard-fail."""
        import pytest

        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-22",))

        def _fake(**kwargs):
            master = stages_dir / "day_2026-02-22_master.mp4"
            master.write_bytes(b"m")
            lf_paths = []
            for i in range(1, 3):
                lp = stages_dir / "day_2026-02-22" / "regular_lfs" / f"lf_{i:02d}.mp4"
                lp.parent.mkdir(parents=True, exist_ok=True)
                lp.write_bytes(b"lf")
                lf_paths.append(lp)
            shorts = []
            for i in range(1, 12):
                sp = stages_dir / "day_2026-02-22" / "shorts" / f"short_{i:02d}.mp4"
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_bytes(b"s")
                shorts.append(sp)
            return [
                SimpleNamespace(
                    date="2026-02-22",
                    master_path=master,
                    regular_lf_paths=lf_paths,
                    short_paths=shorts,
                )
            ]

        with pytest.raises(RuntimeError, match="cannot attribute"):
            self._run_phase(state, _fake)

    def test_zero_shorts_still_initializes_empty_buckets(self, tmp_path):
        """Day with 0 shorts (renderer failure or no content) → each LF
        gets empty list. Phase D handles empty downstream."""
        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))

        def _fake(**kwargs):
            master = stages_dir / "day_2026-02-20_master.mp4"
            master.write_bytes(b"m")
            lf_path = stages_dir / "day_2026-02-20" / "regular_lfs" / "lf_01.mp4"
            lf_path.parent.mkdir(parents=True, exist_ok=True)
            lf_path.write_bytes(b"lf")
            return [
                SimpleNamespace(
                    date="2026-02-20",
                    master_path=master,
                    regular_lf_paths=[lf_path],
                    short_paths=[],
                )
            ]

        self._run_phase(state, _fake)

        spl = state.get("shorts_per_lf") or {}
        assert spl.get(1) == [], f"Expected empty list for LF 1; got {spl.get(1)}"


class TestPerLFLocationsThreadedToRunAllDays:
    """Per-LF OST locations must be computed BEFORE ``run_all_days``
    and threaded through via the ``per_lf_locations_by_date`` kwarg.

    **Bug before the fix.** ``derive_per_lf_locations`` was called AFTER
    ``run_all_days`` returned — so the OST burn-in inside
    ``run_day_pipeline`` (``day_pipeline.py``) always fell back
    to the drive-wide ``location`` string. Multi-LF days got identical
    burned OST text for every LF.

    **Fix.** Compute ``per_lf_locations_by_date:
    dict[str, dict[int, str]]`` up front and pass it into
    ``run_all_days`` via the existing kwarg. ``run_day_pipeline``
    already consumes ``per_lf_locations`` to pick each LF's OST text
    (``day_pipeline.py``); the only thing missing was the
    end-to-end thread.

    Regression gates:
      * Ordering — derive must precede run_all_days.
      * Multi-LF day — each LF receives its own OST text.
      * Single-LF day — unchanged.
    """

    def _base_state(self, tmp_path, dates):
        output_base = tmp_path / "out"
        stages_dir = output_base / "_stages"
        stages_dir.mkdir(parents=True)

        episodes = []
        episode_files = []
        episode_dates_list = []
        file_durations = []
        file_boundaries = []
        cursor = 0.0
        for i, d in enumerate(dates):
            ep_path = f"/fake/ep_{i:03d}.mp4"
            episode_files.append(ep_path)
            episode_dates_list.append(d)
            file_durations.append(1500.0)
            file_boundaries.append((cursor, cursor + 1500.0))
            cursor += 1500.0
            episodes.append(
                SimpleNamespace(
                    index=i,
                    concat_path=Path(ep_path),
                    date=d,
                    total_duration=1500.0,
                    gpx_path=None,
                    creation_time=None,
                )
            )
        state = make_state(
            output_base=output_base,
            ingest_plan=SimpleNamespace(episodes=episodes),
            episode_files=episode_files,
            episode_dates=episode_dates_list,
            episode_locations=[""] * len(dates),
            file_durations=file_durations,
            region="usa",
            location="Drive-Wide Fallback",
            drive_scores=None,
            file_boundaries=file_boundaries,
        )
        return state, output_base, stages_dir

    def _write_drive_plan(self, stages_dir: Path, lfs: list) -> Path:
        payload = {
            "source_files": [],
            "processed_files": [],
            "regular_lfs": lfs,
            "score_threshold": 0.0,
            "total_source_duration": 0.0,
            "total_highlight_duration": 0.0,
            "highlight_ratio": 0.0,
            "best_moments_filtered": False,
        }
        path = stages_dir / "drive_plan.json"
        path.write_text(json.dumps(payload))
        return path

    def _run_phase(self, state, run_all_days_impl, geocoder=None):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        if geocoder is None:
            geocoder = MagicMock()
            geocoder.reverse_geocode.return_value = None

        cfg = Config()
        phase = PhaseBCDayFirst(cfg)
        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=run_all_days_impl,
            ) as m,
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)
        return m

    def test_per_lf_locations_computed_before_ost(self, tmp_path):
        """``run_all_days`` must be called with a populated
        ``per_lf_locations_by_date`` kwarg — proving derive ran BEFORE
        ``run_all_days``, not after. Regression pin for the ordering fix.
        """
        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)]),
            _make_real_lf_dict(2, [_seg_dict(0, 500, 600, 34.0, -81.0)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        places = iter(["Asheville, NC", "Charlotte, NC"])

        def _rg(lat, lon):
            return _Loc(next(places), state="NC", country="USA")

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        captured = {}

        def _fake(**kwargs):
            captured.update(kwargs)
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        self._run_phase(state, _fake, geocoder=geocoder)

        # Regression pin: per_lf_locations_by_date MUST be populated at
        # run_all_days call time (proves derive ran BEFORE, not AFTER).
        assert "per_lf_locations_by_date" in captured, (
            "run_all_days was not called with per_lf_locations_by_date kwarg — "
            "derive_per_lf_locations must run BEFORE run_all_days so OST burn-in "
            "inside run_day_pipeline sees the right per-LF location."
        )
        by_date = captured["per_lf_locations_by_date"]
        assert by_date, (
            "per_lf_locations_by_date is empty at run_all_days call time — "
            "derive must produce labels BEFORE run_all_days."
        )
        assert "2026-02-20" in by_date, (
            f"Expected date key '2026-02-20' in per_lf_locations_by_date; got {list(by_date.keys())}"
        )
        day_labels = by_date["2026-02-20"]
        # One entry per LF index
        assert 1 in day_labels and 2 in day_labels, (
            f"Expected LF indices 1 and 2; got {list(day_labels.keys())}"
        )
        # And the labels come from reverse-geocoded centroids, not from
        # the drive-wide fallback.
        assert day_labels[1] != day_labels[2], (
            f"Multi-LF day produced identical OST labels — {day_labels}"
        )

    def test_multi_lf_day_produces_distinct_ost_per_lf(self, tmp_path):
        """Each LF on a multi-LF day must receive its OWN ``location``
        when run_day_pipeline routes OST burn-in via ``per_lf_locations``.

        Mirrors the OST-location choice in ``run_day_pipeline``:
            lf_location = location
            if per_lf_locations and idx in per_lf_locations:
                lf_location = per_lf_locations[idx]

        If per_lf_locations_by_date isn't wired, all LFs on the day fall
        back to the shared ``location`` string — the bug this ordering fix closes.
        """
        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)]),
            _make_real_lf_dict(2, [_seg_dict(0, 500, 600, 34.0, -81.0)]),
            _make_real_lf_dict(3, [_seg_dict(0, 900, 1000, 33.0, -82.0)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        places = iter(["Asheville, NC", "Charlotte, NC", "Wilmington, NC"])

        def _rg(lat, lon):
            return _Loc(next(places), state="NC", country="USA")

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        captured = {}

        def _fake(**kwargs):
            captured.update(kwargs)
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        self._run_phase(state, _fake, geocoder=geocoder)

        # Each LF in run_day_pipeline.per_lf_locations receives its own
        # label — assert on the dict that would be passed through.
        assert "per_lf_locations_by_date" in captured
        day_labels = captured["per_lf_locations_by_date"].get("2026-02-20", {})
        assert len(day_labels) == 3, (
            f"Expected 3 per-LF labels, got {len(day_labels)}: {day_labels}"
        )
        labels_set = set(day_labels.values())
        assert len(labels_set) == 3, (
            f"Multi-LF day must produce 3 DISTINCT OST labels (not "
            f"identical drive-wide fallback); got {day_labels}"
        )
        # Every label must be non-empty (no silent drop to '')
        assert all(v for v in day_labels.values()), f"Empty OST label(s) in {day_labels}"

    def test_single_lf_day_still_works(self, tmp_path):
        """Single-LF-day regression. The fix must not change behaviour
        when there's only one LF on a day — the single LF still gets its
        own centroid-derived OST label, not an identity-collision artifact.
        """
        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = _Loc("Asheville", state="NC", country="USA")
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        captured = {}

        def _fake(**kwargs):
            captured.update(kwargs)
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        self._run_phase(state, _fake, geocoder=geocoder)

        assert "per_lf_locations_by_date" in captured
        day_labels = captured["per_lf_locations_by_date"].get("2026-02-20", {})
        assert list(day_labels.keys()) == [1], (
            f"Single-LF day must produce one entry keyed by LF index; got {day_labels}"
        )
        assert day_labels[1], "Single-LF label must be non-empty"
        # Resulting state.per_lf_locations mirrors the per-LF map
        labels_after = state.get("per_lf_locations") or {}
        assert labels_after.get(1) == day_labels[1], (
            f"post-run state.per_lf_locations[1]={labels_after.get(1)!r} "
            f"must match value threaded to run_all_days={day_labels[1]!r}"
        )

    def test_multi_day_each_day_gets_its_own_per_lf_map(self, tmp_path):
        """Multi-day drive: ``per_lf_locations_by_date`` is keyed by date
        and each day's entry contains only that day's LFs. Prevents
        bleed across days.
        """
        state, output_base, stages_dir = self._base_state(
            tmp_path, dates=("2026-02-20", "2026-02-21")
        )
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)]),
            _make_real_lf_dict(2, [_seg_dict(0, 500, 600, 34.0, -81.0)]),
            _make_real_lf_dict(3, [_seg_dict(1, 100, 200, 33.0, -82.0)]),
            _make_real_lf_dict(4, [_seg_dict(1, 500, 600, 32.0, -83.0)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        places = iter(["A-NC", "B-NC", "C-SC", "D-SC"])

        def _rg(lat, lon):
            return _Loc(next(places), state="XX", country="USA")

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        captured = {}

        def _fake(**kwargs):
            captured.update(kwargs)
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        self._run_phase(state, _fake, geocoder=geocoder)

        by_date = captured["per_lf_locations_by_date"]
        # Keys are the two day strings; values are per-day LF maps
        assert set(by_date.keys()) == {"2026-02-20", "2026-02-21"}, (
            f"Expected date keys for both days; got {list(by_date.keys())}"
        )
        day1 = by_date["2026-02-20"]
        day2 = by_date["2026-02-21"]
        # LFs 1,2 belong to day 1; 3,4 belong to day 2 (no cross-day bleed)
        assert set(day1.keys()) == {1, 2}, f"Day 1 should have LFs 1,2; got {list(day1.keys())}"
        assert set(day2.keys()) == {3, 4}, f"Day 2 should have LFs 3,4; got {list(day2.keys())}"


# ── canonical per-LF OST text (state["per_lf_ost_text"]) ──────


class TestPerLFOSTCanonicalText:
    """``state["per_lf_ost_text"]`` is the single source of truth
    for OST burn-in text, YouTube title text, and thumbnail text.

    OST, title, and thumbnail must all derive from the same per-LF
    string. Previously ``derive_per_lf_locations`` produced county-level labels
    (e.g. ``"Beaufort County, South Carolina (road)"``) that OST burn
    then used verbatim — coarser than the canonical
    ``"{start} to {end} via {road}, {state}"`` format already produced
    by ``title_builder.build_lf_title`` for YouTube titles.

    This class pins:
      * canonical route format used (not county fallback) when
        geocoder returns distinct start + end towns and the per-LF
        location carries a road suffix;
      * graceful fallback to county-level text when the canonical path
        can't be derived (no road, or geocoder empty);
      * one entry per LF index in ``state["per_lf_ost_text"]``;
      * OST burn-in in ``run_day_pipeline`` uses the canonical text,
        not the dedup-suffixed title;
      * JSON round-trip preserves int keys on resume.
    """

    def _base_state(self, tmp_path, dates):
        output_base = tmp_path / "out"
        stages_dir = output_base / "_stages"
        stages_dir.mkdir(parents=True)

        episodes = []
        episode_files = []
        episode_dates_list = []
        file_durations = []
        file_boundaries = []
        cursor = 0.0
        for i, d in enumerate(dates):
            ep_path = f"/fake/ep_{i:03d}.mp4"
            episode_files.append(ep_path)
            episode_dates_list.append(d)
            file_durations.append(1500.0)
            file_boundaries.append((cursor, cursor + 1500.0))
            cursor += 1500.0
            episodes.append(
                SimpleNamespace(
                    index=i,
                    concat_path=Path(ep_path),
                    date=d,
                    total_duration=1500.0,
                    gpx_path=None,
                    creation_time=None,
                )
            )
        state = make_state(
            output_base=output_base,
            ingest_plan=SimpleNamespace(episodes=episodes),
            episode_files=episode_files,
            episode_dates=episode_dates_list,
            episode_locations=[""] * len(dates),
            file_durations=file_durations,
            region="usa",
            location="Drive-Wide Fallback",
            drive_scores=None,
            file_boundaries=file_boundaries,
        )
        return state, output_base, stages_dir

    def _write_drive_plan(self, stages_dir: Path, lfs: list) -> Path:
        payload = {
            "source_files": [],
            "processed_files": [],
            "regular_lfs": lfs,
            "score_threshold": 0.0,
            "total_source_duration": 0.0,
            "total_highlight_duration": 0.0,
            "highlight_ratio": 0.0,
            "best_moments_filtered": False,
        }
        path = stages_dir / "drive_plan.json"
        path.write_text(json.dumps(payload))
        return path

    def _run_phase(self, state, run_all_days_impl, geocoder=None):
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        if geocoder is None:
            geocoder = MagicMock()
            geocoder.reverse_geocode.return_value = None

        cfg = Config()
        phase = PhaseBCDayFirst(cfg)
        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=run_all_days_impl,
            ) as m,
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)
        return m

    def test_per_lf_ost_uses_route_not_county(self, tmp_path):
        """When start and end towns resolve, per_lf_ost_text uses the
        canonical ``"{start} to {end}, {state}"`` — NOT the county-level
        label that ``derive_per_lf_locations`` produces. A geocoded road is
        never burned: no caller measures its retained-time coverage.
        """
        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))
        # Two LFs at identical centroids → road suffix kicks in on
        # derive_per_lf_locations, giving per_lf_location strings with
        # a real road in parentheses.
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 32.30, -80.60)]),
            _make_real_lf_dict(2, [_seg_dict(0, 500, 600, 32.30, -80.60)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        # derive never calls reverse_geocode (labels come from the
        # reverse_geocode_detailed county vote), so the only calls are the
        # canonical OST pre-computation via build_lf_title — start
        # then end town per LF: LF 1 → calls 1-2, LF 2 → calls 3-4.
        towns = iter(["Port Royal", "Lady's Island", "Frogmore", "Saint Helena"])

        def _rg(lat, lon):
            return _Loc(next(towns), state="South Carolina", country="USA")

        def _rgd(lat, lon):
            # County/state feed the vote (both LFs land on the same label,
            # so LF 2 collides); the road feeds derive's disambiguator,
            # adding the "(road)" suffix to LF 2 only.
            return {
                "county": "Beaufort County",
                "state": "South Carolina",
                "road": "Sea Island Parkway",
            }

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _rgd

        captured = {}

        def _fake(**kwargs):
            captured.update(kwargs)
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        self._run_phase(state, _fake, geocoder=geocoder)

        ost_text = state.get("per_lf_ost_text") or {}
        per_lf_locations = state.get("per_lf_locations") or {}
        # LF 1's per_lf_location is "Beaufort, South Carolina" (no road
        # suffix — was the first to claim the label). LF 2's is
        # "Beaufort, South Carolina (Sea Island Parkway)" (road added
        # by disambig). title_builder's build_lf_title extracts the
        # road only when present.
        # LF 1: start + end towns resolved, no road → Level 2 fallback
        assert ost_text.get(1) == "Port Royal to Lady's Island, SC", (
            f"LF 1 OST text must be canonical endpoint format; "
            f"got {ost_text.get(1)!r}. per_lf_locations={per_lf_locations!r}"
        )
        # LF 2: start + end + an unmeasured road → the road is omitted
        assert ost_text.get(2) == "Frogmore to Saint Helena, SC", (
            f"LF 2 OST text must be canonical endpoint format with no "
            f"unmeasured road; "
            f"got {ost_text.get(2)!r}. per_lf_locations={per_lf_locations!r}"
        )
        # Regression gate — must NOT be the coarser county-level label
        # produced by derive_per_lf_locations.
        assert "County" not in ost_text.get(1, ""), (
            f"LF 1 OST text must NOT fall back to county label; got {ost_text.get(1)!r}"
        )

    def test_per_lf_ost_falls_back_to_county_when_route_unknown(self, tmp_path):
        """When the geocoder can't resolve start/end towns (returns None),
        per_lf_ost_text falls back to the county-level label from
        derive_per_lf_locations — graceful degradation preserves the
        county-level behaviour.
        """
        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        # First reverse_geocode call (title builder's start-town lookup)
        # returns a concrete Location; subsequent calls for the remaining
        # town lookups → return None (no route resolution).
        call_log = []

        def _rg(lat, lon):
            call_log.append((lat, lon))
            if len(call_log) == 1:
                return _Loc(
                    "Asheville",
                    county="Buncombe County",
                    state="North Carolina",
                    country="USA",
                )
            return None  # title builder _geocode_town returns None

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        # Deterministic detailed lookup so derive_per_lf_locations builds a
        # real "Buncombe County, North Carolina" label rather than MagicMock
        # repr junk — the title-builder fallback now uses the trailing
        # region verbatim, so per_lf_location must be a clean string.
        geocoder.reverse_geocode_detailed.return_value = {
            "county": "Buncombe County",
            "state": "North Carolina",
            "country": "USA",
        }

        def _fake(**kwargs):
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        self._run_phase(state, _fake, geocoder=geocoder)

        ost_text = state.get("per_lf_ost_text") or {}
        per_lf_locations = state.get("per_lf_locations") or {}
        # When canonical route can't be derived, OST text must match the
        # per_lf_location fallback (not be empty, not KeyError).
        assert 1 in ost_text, (
            f"Expected LF 1 in per_lf_ost_text even in fallback; got {list(ost_text)}"
        )
        assert ost_text[1], f"LF 1 OST text must be non-empty on fallback; got {ost_text[1]!r}"
        # The fallback is either the canonical county form OR the raw
        # per_lf_location — both are acceptable as long as the text is
        # populated and not empty.
        assert (
            ost_text[1] == per_lf_locations.get(1)
            or "Asheville" in ost_text[1]
            or "Buncombe" in ost_text[1]
        ), f"Fallback OST text must reflect location label; got {ost_text[1]!r}"

    def test_per_lf_ost_populated_per_lf_index(self, tmp_path):
        """Every LF in the drive plan gets exactly one entry in
        ``state["per_lf_ost_text"]``, keyed by int LF index."""
        state, output_base, stages_dir = self._base_state(
            tmp_path, dates=("2026-02-20", "2026-02-21")
        )
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)]),
            _make_real_lf_dict(2, [_seg_dict(0, 500, 600, 34.0, -81.0)]),
            _make_real_lf_dict(3, [_seg_dict(1, 100, 200, 33.0, -82.0)]),
            _make_real_lf_dict(4, [_seg_dict(1, 500, 600, 32.0, -83.0)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        # Enough unique reverse_geocode responses for the title builder's
        # first/last town lookups (2 per LF; derive uses only the
        # reverse_geocode_detailed county vote).
        seq = [
            # title first/last for LF 1
            _Loc("StartA", state="NC"),
            _Loc("EndA", state="NC"),
            # LF 2
            _Loc("StartB", state="NC"),
            _Loc("EndB", state="NC"),
            # LF 3
            _Loc("StartC", state="SC"),
            _Loc("EndC", state="SC"),
            # LF 4
            _Loc("StartD", state="SC"),
            _Loc("EndD", state="SC"),
        ]
        iseq = iter(seq)

        def _rg(lat, lon):
            return next(iseq, None)

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        def _fake(**kwargs):
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        self._run_phase(state, _fake, geocoder=geocoder)

        ost_text = state.get("per_lf_ost_text") or {}
        assert set(ost_text.keys()) == {1, 2, 3, 4}, (
            f"per_lf_ost_text must have one entry per LF index; got {list(ost_text)}"
        )
        assert all(isinstance(k, int) for k in ost_text.keys()), (
            f"per_lf_ost_text keys must be int; got {[type(k).__name__ for k in ost_text]}"
        )
        assert all(v for v in ost_text.values()), (
            f"Every per_lf_ost_text value must be non-empty; got {ost_text}"
        )

    def test_per_lf_ost_drives_ost_burn_not_dedup_suffixed_title(self, tmp_path):
        """Two LFs with IDENTICAL canonical text → per_lf_titles after
        ensure_unique_titles gets "(Part N of M)" dedup suffixes, but
        per_lf_ost_text must NOT carry those suffixes. OST burn-in is
        the raw canonical text; title dedup is a title-only policy.

        Pins the rule that per_lf_ost_text is captured BEFORE
        ensure_unique_titles runs.
        """
        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))
        # Two LFs at the same centroid — same town pair, same road.
        # derive_per_lf_locations' road-disambiguation path adds a
        # parenthetical suffix when two LFs collide.
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 32.30, -80.60)]),
            _make_real_lf_dict(2, [_seg_dict(0, 500, 600, 32.30, -80.60)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        # Simulate: the county vote gives both LFs the same base label
        # (collision → per_lf_location disambiguation), and the title
        # builder returns the same start/end towns for both LFs (making
        # the canonical titles identical, triggering ensure_unique_titles'
        # (Part N of M) suffix). derive never calls reverse_geocode.
        towns = iter(["Port Royal", "Lady's Island", "Port Royal", "Lady's Island"])

        def _rg(lat, lon):
            return _Loc(next(towns), state="South Carolina")

        def _rgd(lat, lon):
            # County/state feed the vote; the road feeds the disambiguator.
            return {
                "county": "Beaufort County",
                "state": "South Carolina",
                "road": "Sea Island Parkway",
            }

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _rgd

        def _fake(**kwargs):
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        self._run_phase(state, _fake, geocoder=geocoder)

        ost_text = state.get("per_lf_ost_text") or {}
        titles = state.get("per_lf_titles") or {}

        # per_lf_ost_text MUST NOT contain the "(Part N of M)" dedup suffix
        # (that's a title-only deduplication). OST burn-in shows the
        # canonical route text verbatim on every LF.
        for idx, text in ost_text.items():
            assert "Part " not in text or "of " not in text, (
                f"per_lf_ost_text[{idx}] contains dedup suffix — OST text "
                f"must be canonical only; got {text!r}"
            )

    def test_per_lf_ost_threaded_to_run_all_days_for_burn(self, tmp_path):
        """``run_all_days`` must receive ``per_lf_ost_text_by_date`` so
        ``run_day_pipeline``'s OST burn-in step uses canonical text per
        LF, not the coarser ``per_lf_locations`` label.

        Single-source-of-truth contract: OST, title, thumbnail
        all read from state["per_lf_ost_text"].
        """
        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))
        lfs = [
            _make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)]),
            _make_real_lf_dict(2, [_seg_dict(0, 500, 600, 34.0, -81.0)]),
        ]
        self._write_drive_plan(stages_dir, lfs)

        # reverse_geocode feeds only the title-builder town lookups.
        seq = iter(
            [
                _Loc("Hendersonville", state="North Carolina"),
                _Loc("Brevard", state="North Carolina"),
                _Loc("Davidson", state="North Carolina"),
                _Loc("Concord", state="North Carolina"),
            ]
        )

        def _rg(lat, lon):
            return next(seq, None)

        geocoder = MagicMock()
        geocoder.reverse_geocode.side_effect = _rg
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        captured = {}

        def _fake(**kwargs):
            captured.update(kwargs)
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        self._run_phase(state, _fake, geocoder=geocoder)

        # run_all_days must be called with per_lf_ost_text_by_date kwarg
        assert "per_lf_ost_text_by_date" in captured, (
            "run_all_days was not called with per_lf_ost_text_by_date kwarg — "
            "the canonical-text thread into OST burn-in is missing."
        )
        by_date = captured["per_lf_ost_text_by_date"]
        assert "2026-02-20" in by_date, f"Expected date key '2026-02-20'; got {list(by_date)}"
        day_ost = by_date["2026-02-20"]
        assert set(day_ost.keys()) == {1, 2}, f"Expected LF indices 1,2; got {list(day_ost)}"
        assert all(day_ost.values()), f"Every OST text must be non-empty; got {day_ost}"

    def test_phase_bc_refresh_hard_fails_on_exact_channel_duplicate(self, tmp_path):
        """Phase BC calls ``refresh_channel_title_ledger`` right
        before ``load_published_titles``; an exact collision it surfaces is
        a hard failure at BC, not a silent "(Part N)" rename."""
        from gomoveshift.pipeline.exceptions import PhaseBCExitContractError

        state, output_base, stages_dir = self._base_state(tmp_path, dates=("2026-02-20",))
        lfs = [_make_real_lf_dict(1, [_seg_dict(0, 100, 200, 35.0, -80.0)])]
        self._write_drive_plan(stages_dir, lfs)

        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = _Loc("Hendersonville", state="North Carolina")
        geocoder.reverse_geocode_detailed.side_effect = _detailed_by_lat

        def _fake(**kwargs):
            return [
                SimpleNamespace(
                    date=b.date,
                    master_path=Path(f"/fake/master_{b.date}.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
                for b in kwargs["day_batches"]
            ]

        with (
            patch(
                "gomoveshift.pipeline.longform_metadata.assemble_final_title",
                return_value="Pennsylvania Scenic Drive in 4K | Gettysburg, PA",
            ),
            patch(
                "gomoveshift.pipeline.longform_metadata.refresh_channel_title_ledger",
                return_value={"pennsylvania scenic drive in 4k | gettysburg, pa": "yt_LIVE1"},
            ) as mock_refresh,
            pytest.raises(PhaseBCExitContractError, match="yt_LIVE1"),
        ):
            self._run_phase(state, _fake, geocoder=geocoder)
        mock_refresh.assert_called()


class TestPhaseBCMidPhaseStateSave:
    """Mid-phase state write-through: a per-day ``on_day_complete``
    callback must fire from ``run_all_days`` so ``per_day_completed``
    lands on disk BEFORE the next day starts.

    The regression this pins: a run that crashed mid-phase with earlier
    days done in memory but never persisted to the state file forced the
    next resume to re-encode those days from scratch.
    """

    def _base_state(self, tmp_path, dates):
        output_base = tmp_path / "out"
        stages_dir = output_base / "_stages"
        stages_dir.mkdir(parents=True)

        episodes = []
        episode_files = []
        episode_dates_list = []
        file_durations = []
        file_boundaries = []
        cursor = 0.0
        for i, d in enumerate(dates):
            ep_path = f"/fake/ep_{i:03d}.mp4"
            episode_files.append(ep_path)
            episode_dates_list.append(d)
            file_durations.append(1500.0)
            file_boundaries.append((cursor, cursor + 1500.0))
            cursor += 1500.0
            episodes.append(
                SimpleNamespace(
                    index=i,
                    concat_path=Path(ep_path),
                    date=d,
                    total_duration=1500.0,
                    gpx_path=None,
                    creation_time=None,
                )
            )
        ingest_plan = SimpleNamespace(episodes=episodes)

        state = make_state(
            output_base=output_base,
            ingest_plan=ingest_plan,
            episode_files=episode_files,
            episode_dates=episode_dates_list,
            episode_locations=[""] * len(dates),
            file_durations=file_durations,
            region="usa",
            location="Test",
            drive_scores=None,
            file_boundaries=file_boundaries,
        )
        return state, output_base, stages_dir

    def test_state_save_fn_fires_after_each_day_not_just_at_end(self, tmp_path):
        """``state_save_fn`` must be invoked per-day from inside
        run_all_days, BEFORE the whole phase completes. Saves are
        observable by recording the per_day_completed snapshot at each
        save call time."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        dates = ("2026-02-20", "2026-02-21", "2026-02-22")
        state, output_base, stages_dir = self._base_state(tmp_path, dates)

        # Capture a snapshot of per_day_completed at each save.
        save_snapshots: list = []

        def _save_fn(st):
            per_day = st.get("per_day_completed") or {}
            save_snapshots.append(sorted(per_day.keys()))

        def _real_run_all_days(**kwargs):
            """Simulate run_all_days: for each batch, write a master and
            invoke the on_day_complete callback the orchestrator passed in.
            This reproduces the production wiring in day_pipeline.run_all_days."""
            artifacts = []
            cb = kwargs.get("on_day_complete")
            for batch in kwargs["day_batches"]:
                master = stages_dir / f"day_{batch.date}" / f"day_{batch.date}_master.mp4"
                master.parent.mkdir(parents=True, exist_ok=True)
                master.write_bytes(b"m-" + batch.date.encode())
                art = SimpleNamespace(
                    date=batch.date,
                    master_path=master,
                    regular_lf_paths=[],
                    short_paths=[],
                )
                artifacts.append(art)
                if cb is not None:
                    cb(art, kwargs.get("state"))
            return artifacts

        cfg = Config()
        phase = PhaseBCDayFirst(cfg)
        phase._save_callback = _save_fn

        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = None
        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=_real_run_all_days,
            ),
        ):
            phase.run(
                state,
                probe_duration_fn=lambda p: 100.0,
                state_save_fn=phase._save_callback,
            )

        # Save must have fired at least 3 times (one per day).
        assert len(save_snapshots) >= 3, (
            f"Expected ≥3 saves (one per day); got {len(save_snapshots)} snapshots"
        )
        # Snapshots must be monotonically growing — each day added, none lost.
        assert save_snapshots[0] == ["2026-02-20"], (
            f"First save must record day 20; got {save_snapshots[0]}"
        )
        assert save_snapshots[1] == ["2026-02-20", "2026-02-21"], (
            f"Second save must have both day 20 and 21; got {save_snapshots[1]}"
        )
        assert save_snapshots[2] == ["2026-02-20", "2026-02-21", "2026-02-22"], (
            f"Third save must include all 3 days; got {save_snapshots[2]}"
        )

    def test_mid_phase_crash_leaves_completed_days_recorded(self, tmp_path):
        """Simulate a crash on day 3: verify days 1 + 2 are already in
        the persisted state via the per-day save_fn callback."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        dates = ("2026-02-20", "2026-02-21", "2026-02-22")
        state, output_base, stages_dir = self._base_state(tmp_path, dates)

        # Last-observed state snapshot — what would be on disk after crash.
        on_disk: dict = {}

        def _save_fn(st):
            # Mirror state.per_day_completed into on_disk (simulating a
            # JSON write).
            per_day = st.get("per_day_completed") or {}
            on_disk.clear()
            on_disk.update(per_day)

        def _run_all_days_with_crash_on_day_3(**kwargs):
            """Run days 1 and 2 normally, crash before yielding day 3's artifact."""
            artifacts = []
            cb = kwargs.get("on_day_complete")
            for batch in kwargs["day_batches"]:
                if batch.date == "2026-02-22":
                    raise RuntimeError("simulated master encode crash on day 22")
                master = stages_dir / f"day_{batch.date}" / f"day_{batch.date}_master.mp4"
                master.parent.mkdir(parents=True, exist_ok=True)
                master.write_bytes(b"m")
                art = SimpleNamespace(
                    date=batch.date,
                    master_path=master,
                    regular_lf_paths=[],
                    short_paths=[],
                )
                artifacts.append(art)
                if cb is not None:
                    cb(art, kwargs.get("state"))
            return artifacts

        import pytest

        cfg = Config()
        phase = PhaseBCDayFirst(cfg)
        phase._save_callback = _save_fn

        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = None
        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=_run_all_days_with_crash_on_day_3,
            ),
            pytest.raises(RuntimeError, match="simulated master encode crash"),
        ):
            phase.run(
                state,
                probe_duration_fn=lambda p: 100.0,
                state_save_fn=phase._save_callback,
            )

        # The crash happened on day 3. Days 1 and 2 must be persisted to
        # the "on disk" snapshot via the per-day save_fn callback.
        assert sorted(on_disk.keys()) == ["2026-02-20", "2026-02-21"], (
            f"After mid-phase crash, days 20 + 21 must be recorded in the "
            f"last-saved state; got {sorted(on_disk.keys())}"
        )


class TestDayCompletionGate:
    """Completion gate — Phase BC must raise RuntimeError when any planned
    day produces no artifacts (e.g. NVENC probe failure in run_all_days
    swallowed the exception and returned a partial list).

    Regression guard: when a day raises an NVENC probe failure and the
    exception is caught in day_pipeline, the pipeline must not silently
    proceed to Phase D with fewer LFs than planned.
    """

    def _base_state(self, tmp_path, dates=("2026-03-04", "2026-03-05")):
        output_base = tmp_path / "out"
        stages_dir = output_base / "_stages"
        stages_dir.mkdir(parents=True)

        episodes = []
        episode_files = []
        episode_dates_list = []
        file_durations = []
        file_boundaries = []
        cursor = 0.0
        for i, d in enumerate(dates):
            ep_path = f"/fake/ep_{i:03d}.mp4"
            episode_files.append(ep_path)
            episode_dates_list.append(d)
            file_durations.append(1500.0)
            file_boundaries.append((cursor, cursor + 1500.0))
            cursor += 1500.0
            episodes.append(
                SimpleNamespace(
                    index=i,
                    concat_path=Path(ep_path),
                    date=d,
                    total_duration=1500.0,
                    gpx_path=None,
                    creation_time=None,
                )
            )
        ingest_plan = SimpleNamespace(episodes=episodes)
        state = make_state(
            output_base=output_base,
            ingest_plan=ingest_plan,
            episode_files=episode_files,
            episode_dates=episode_dates_list,
            episode_locations=[""] * len(dates),
            file_durations=file_durations,
            region="usa",
            location="NC to PA",
            drive_scores=None,
            file_boundaries=file_boundaries,
        )
        return state, output_base, stages_dir

    def test_raises_when_day_produces_no_artifacts(self, tmp_path):
        """run_all_days returns only 1 artifact for day 2026-03-04; day
        2026-03-05 produced nothing (NVENC failure swallowed).  Phase BC
        must raise RuntimeError naming the missing date."""
        from gomoveshift.config import Config
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            PhaseBCDayFirst,
        )

        state, _output_base, _stages_dir = self._base_state(tmp_path)

        def _fake_run_all_days(**kwargs):
            # Only return artifact for first day — day 2026-03-05 silently missing
            return [
                SimpleNamespace(
                    date="2026-03-04",
                    master_path=Path("/fake/master_2026-03-04.mp4"),
                    regular_lf_paths=[],
                    short_paths=[],
                )
            ]

        geocoder = MagicMock()
        geocoder.reverse_geocode.return_value = None
        cfg = Config()
        phase = PhaseBCDayFirst(cfg)

        import pytest

        with (
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_shared_geocoder",
                return_value=geocoder,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=_fake_run_all_days,
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            phase.run(state, probe_duration_fn=lambda p: 100.0)

        assert "2026-03-05" in str(exc_info.value)


class TestPreEncodeLocationInvariant:
    """The location desync check runs BEFORE run_all_days, so a
    far-road label refuses the encode instead of costing it."""

    def _resolved(self):
        from gomoveshift.models import ResolvedLocation

        rl = ResolvedLocation()
        rl.dominant_county = "Charleston County"
        rl.dominant_state = "South Carolina"
        rl.regions = ["Charleston County, South Carolina"]
        rl.vote = {"Charleston County, South Carolina": 1.0}
        return rl

    def test_desynced_ost_text_raises_before_encode(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_pre_encode_location_invariant,
        )

        import pytest

        with pytest.raises(RuntimeError) as exc:
            _assert_pre_encode_location_invariant(
                {3: "Drive in Colleton County"},
                {3: self._resolved()},
                evidence_by_lf={3: {"corroborations": 1}},
            )
        msg = str(exc.value)
        assert "Pre-encode location invariant HARD FAIL" in msg
        assert "LF 3" in msg
        assert "Refusing to start the encode" in msg

    def test_consistent_ost_text_passes(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_pre_encode_location_invariant,
        )

        _assert_pre_encode_location_invariant(
            {3: "Charleston County, South Carolina Scenic Drive"},
            {3: self._resolved()},
            evidence_by_lf={3: {"corroborations": 1}},
        )

    def test_lf_without_resolved_model_is_skipped(self):
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _assert_pre_encode_location_invariant,
        )

        # GPS-less / quorum-degraded LF has no model → no-op (Phase D parity).
        _assert_pre_encode_location_invariant(
            {3: "Drive in Colleton County"}, {}, evidence_by_lf={}
        )
