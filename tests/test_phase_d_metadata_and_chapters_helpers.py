"""Unit tests for Phase D metadata + chapters helpers.

The helpers were extracted from ``PhaseD.run`` to drive the run-body
line count down. Behaviour matches the previous inline blocks; these
tests pin the per-helper contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from gomoveshift.pipeline.drive_phases.phase_d import (
    PhaseD,
    _build_metadata_per_lf,
    _generate_chapters_all_lfs,
)
from gomoveshift.pipeline.route_profile import RouteProfile

from _state_helpers import make_state
from test_phase_d import _install_fake_youtube_uploader, _phase_d_patches, _setup_mocks
from test_phase_d import phase_d_state as phase_d_state  # noqa: F401 -- fixture re-export


def _pass_redundancy():
    return patch(
        "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
        return_value=SimpleNamespace(passed=True, message=""),
    )


def _lf(idx, *, total_duration=1800.0, segments=None):
    return SimpleNamespace(
        index=idx,
        total_duration=total_duration,
        segments=segments or [],
    )


# ─── _build_metadata_per_lf ─────────────────────────────────────────────


class TestBuildMetadataPerLf:
    def _mk_meta_gen(self, *, returns=None):
        gen = MagicMock()
        gen.generate.side_effect = lambda **kw: dict(
            returns or {"title": f"T-{kw['project_name']}", "description": "d"}
        )
        return gen

    def _mk_drive_plan(self, lf_indices):
        return SimpleNamespace(regular_lfs=[_lf(i) for i in lf_indices])

    def test_writes_youtube_json_per_lf(self, tmp_path):
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1, 2])
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ):
            out = _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
                state=make_state(),
                location="Kerry, Ireland",
                publish_day_full="Wednesday",
                project_base="proj",
                output_base=tmp_path,
            )
        assert set(out.keys()) == {1, 2}
        assert (tmp_path / "proj_lf01_youtube.json").exists()
        assert (tmp_path / "proj_lf02_youtube.json").exists()
        # Defaults injected for the gate downstream
        assert out[1]["location"] == "Kerry, Ireland"
        assert out[1]["video_duration"] == 1800.0

    def test_per_lf_location_override_wins(self, tmp_path):
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1])
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf.mp4"],
                state=make_state(per_lf_locations={1: "Dingle, Kerry"}),
                location="Ireland",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        assert gen.generate.call_args.kwargs["location"] == "Dingle, Kerry"

    def test_populated_map_missing_lf_index_raises(self, tmp_path):
        """Gate-bypass hardening: a populated
        per_lf_locations missing a rendered LF's index is state drift —
        raise, never silently fall back to the drive-wide label."""
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1, 2])
        with _pass_redundancy(), pytest.raises(RuntimeError, match="missing from"):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
                state=make_state(per_lf_locations={1: "Dingle, Kerry"}),
                location="Ireland",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )

    def test_populated_ost_text_map_missing_lf_index_raises(self, tmp_path):
        """A populated per_lf_ost_text map missing a rendered LF's
        index is state drift — raise, mirroring the per_lf_locations
        hard-fail. Silently dropping to no title_override lets the SEO
        template win, diverging the YouTube title from the burned OST
        pixels. An entirely empty map (GPS-less / legacy) still degrades."""
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1, 2])
        with _pass_redundancy(), pytest.raises(RuntimeError, match="per_lf_ost_text"):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
                state=make_state(per_lf_ost_text={1: "Killarney to Dingle"}),
                location="Ireland",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )

    def test_ost_text_drives_title_override_with_dedup(self, tmp_path):
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1, 2])
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
                state=make_state(
                    per_lf_ost_text={1: "Killarney to Dingle", 2: "Killarney to Dingle"}
                ),
                location="Ireland",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        # ensure_unique_titles should dedup the two identical titles
        titles = [c.kwargs["title_override"] for c in gen.generate.call_args_list]
        assert len(set(titles)) == 2
        assert all(t and "Killarney to Dingle" in t for t in titles)

    def test_trim_ratio_threads_source_duration(self, tmp_path):
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1])
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf.mp4"],
                state=make_state(per_lf_trim_ratio={1: 0.5}),
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        kwargs = gen.generate.call_args.kwargs
        assert kwargs["trim_ratio"] == 0.5
        assert kwargs["source_duration"] == 3600.0  # 1800 / 0.5

    def test_has_gps_threaded_from_state_per_lf(self, tmp_path):
        """Per-LF has_gps derived from state["per_lf_gpx_path"].

        For a mixed GPS/GPS-less drive, the call-site must pass has_gps=True
        for LFs that have a gpx path in state, and has_gps=False otherwise.
        This activates the robust-by-construction chapter-density skip.
        """
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1, 2])
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
                state=make_state(per_lf_gpx_path={1: "/g/lf1.gpx"}),
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        calls = {c.kwargs["project_name"]: c.kwargs for c in gen.generate.call_args_list}
        assert calls["p_lf01"]["has_gps"] is True
        assert calls["p_lf02"]["has_gps"] is False

    def test_has_gps_false_when_state_missing_gpx_map(self, tmp_path):
        """No per_lf_gpx_path in state at all → has_gps=False for every LF."""
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1])
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ):
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf.mp4"],
                state=make_state(),
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        assert gen.generate.call_args.kwargs["has_gps"] is False

    # Title source == units source, and the playlist region is stamped.
    def _run_with_resolved(self, tmp_path, *, location, resolved):
        from gomoveshift.models import ResolvedLocation

        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1])
        rl = ResolvedLocation(**resolved)
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=True, message=""),
        ):
            out = _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf.mp4"],
                state=make_state(per_lf_resolved_location={1: rl}),
                location=location,
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        return gen, rl, out

    def test_resolved_threaded_and_region_stamped(self, tmp_path):
        # The model reaches generate; its region is stamped for the playlist
        # — both even though the display label says Ireland.
        gen, rl, out = self._run_with_resolved(
            tmp_path, location="Somewhere, Ireland", resolved={"is_us": True}
        )
        assert gen.generate.call_args.kwargs["resolved_location"] is rl
        assert out[1]["region"] == "usa"

    def test_units_divergence_warns_only_on_disagree(self, tmp_path, caplog):
        import logging

        # Label parses non-US (Ireland) but model says US → WARN; agree → none.
        with caplog.at_level(logging.WARNING):
            self._run_with_resolved(
                tmp_path, location="Somewhere, Ireland", resolved={"is_us": True}
            )
        assert "UNITS-DIVERGE" in caplog.text
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            self._run_with_resolved(
                tmp_path, location="Beaufort County, South Carolina", resolved={"is_us": True}
            )
        assert "UNITS-DIVERGE" not in caplog.text

    def test_full_drive_fallback_distance_flagged_and_passes_gate(self, tmp_path):
        """An LF with no resolvable segment window (empty
        ``segments``, no ``per_lf_gpx_episode_paths`` entry) had its
        persisted distance/speed computed from the whole unwindowed GPX
        trace (see docs/pipeline-internals.md). The call site
        must flag both stats via ``full_drive_stats`` so the real
        ``check_description_redundancy`` gate (not a stub) requires — and
        gets — a "Full drive:" prefix rather than hard-failing.
        """
        from gomoveshift.pipeline.validation.content_gates import (
            check_description_redundancy,
        )

        gen = MagicMock()

        def _generate(**kw):
            full_drive = kw.get("full_drive_stats") or []
            lines = ["Hook.", ""]
            if "distance" in full_drive:
                lines.append("\U0001f6e3️ Full drive: 22 km (14 mi)")
            if "speed" in full_drive:
                lines.append("\U0001f697 Full drive: 60 km/h avg")
            lines.append("\U0001f4cd Location: x")
            return {
                "title": kw["project_name"],
                "description": "\n".join(lines),
                "location": kw["location"],
                "full_drive_stats": full_drive or None,
            }

        gen.generate.side_effect = _generate
        drive_plan = self._mk_drive_plan([1])
        state = make_state(
            per_lf_distance_km={1: 22.0},
            per_lf_speed={1: {"avg_kmh": 60.0, "std_kmh": 5.0}},
        )
        # Real gate (no stub) — the point of this test is that it passes.
        out = _build_metadata_per_lf(
            meta_gen=gen,
            drive_plan=drive_plan,
            regular_lf_paths=["/v/lf.mp4"],
            state=state,
            location="x",
            publish_day_full="Wednesday",
            project_base="p",
            output_base=tmp_path,
        )
        kwargs = gen.generate.call_args.kwargs
        assert sorted(kwargs["full_drive_stats"]) == ["distance", "speed"]
        assert check_description_redundancy(out[1]).passed

    def test_kept_segment_lf_does_not_flag_full_drive(self, tmp_path):
        """A kept-segment LF (resolvable segment window) must NOT set
        ``full_drive_stats`` — output stays byte-identical to the pre-flag behaviour."""
        gen = self._mk_meta_gen()
        drive_plan = SimpleNamespace(
            regular_lfs=[
                _lf(
                    1,
                    segments=[SimpleNamespace(source_file_index=0, start_time=0.0, end_time=30.0)],
                )
            ]
        )
        state = make_state(
            per_lf_distance_km={1: 22.0},
            per_lf_speed={1: {"avg_kmh": 60.0, "std_kmh": 5.0}},
        )
        with _pass_redundancy():
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf.mp4"],
                state=state,
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        kwargs = gen.generate.call_args.kwargs
        assert kwargs.get("full_drive_stats") is None

    def test_episode_span_map_present_does_not_flag_full_drive(self, tmp_path):
        """A ``per_lf_gpx_episode_paths`` entry means distance/speed were
        summed per-episode (windowed) — never flag as full-drive even when
        ``segments`` is empty (legacy/multi-episode LFs may have no
        single-representative window)."""
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1])
        state = make_state(
            per_lf_distance_km={1: 22.0},
            per_lf_gpx_episode_paths={1: {"/g/ep0.gpx": [(0.0, 30.0)]}},
        )
        with _pass_redundancy():
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf.mp4"],
                state=state,
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        kwargs = gen.generate.call_args.kwargs
        assert kwargs.get("full_drive_stats") is None

    def test_redundancy_gate_failure_raises(self, tmp_path):
        gen = self._mk_meta_gen()
        drive_plan = self._mk_drive_plan([1])
        with patch(
            "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
            return_value=SimpleNamespace(passed=False, message="dup"),
        ):
            with pytest.raises(RuntimeError, match="redundancy gate failed for LF 1"):
                _build_metadata_per_lf(
                    meta_gen=gen,
                    drive_plan=drive_plan,
                    regular_lf_paths=["/v/lf.mp4"],
                    state=make_state(),
                    location="x",
                    publish_day_full="Wednesday",
                    project_base="p",
                    output_base=tmp_path,
                )


class TestBuildMetadataRouteProfileWiring:
    """``state.route_profiles`` was READ (route_profiles.get(lf.index)) but
    never WRITTEN, so the GPS-verified per-LF enrichment was dead. The metadata
    loop must aggregate the persisted per-EPISODE profiles into a per-LF profile
    and thread it to generate(), mapping episodes→LF by recording date (NOT the
    DAY-LOCAL segment source_file_index, which aliases another day's episodes).
    """

    def _mk_meta_gen(self):
        gen = MagicMock()
        gen.generate.side_effect = lambda **kw: {"title": kw["project_name"], "description": "d"}
        return gen

    def test_route_profile_wired_from_episode_profiles(self, tmp_path):
        gen = self._mk_meta_gen()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
        ep_profile = RouteProfile(
            start_town="Springfield",
            end_town="Decatur",
            towns_traversed=["Springfield", "Decatur"],
            roads_traversed=["US-150"],
            states_traversed=["Illinois"],
        ).to_dict()
        state = make_state(
            episode_profiles=[ep_profile],
            episode_dates=["2025-09-01"],
            per_lf_recording_date={1: "2025-09-01"},
        )
        with _pass_redundancy():
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4"],
                state=state,
                location="Springfield, IL",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        passed = gen.generate.call_args.kwargs["route_profile"]
        assert passed is not None
        assert passed["towns_traversed"] == ["Springfield", "Decatur"]
        assert passed["start_town"] == "Springfield"
        assert passed["end_town"] == "Decatur"
        # Persisted for resume.
        assert state.route_profiles.get(1) is not None

    def test_multi_day_lf_gets_only_its_own_days_episodes(self, tmp_path):
        """The episode→LF map keys on recording date, never the DAY-LOCAL
        segment source_file_index — LF 2 (day 2) must not borrow day-1 towns."""
        gen = self._mk_meta_gen()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1), _lf(2)])
        day1 = RouteProfile(
            start_town="Springfield",
            end_town="Decatur",
            towns_traversed=["Springfield", "Decatur"],
        ).to_dict()
        day2 = RouteProfile(
            start_town="Beaufort",
            end_town="Charleston",
            towns_traversed=["Beaufort", "Charleston"],
        ).to_dict()
        state = make_state(
            episode_profiles=[day1, day2],
            episode_dates=["2025-09-01", "2025-09-02"],
            per_lf_recording_date={1: "2025-09-01", 2: "2025-09-02"},
        )
        with _pass_redundancy():
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
                state=state,
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        by_proj = {
            c.kwargs["project_name"]: c.kwargs["route_profile"] for c in gen.generate.call_args_list
        }
        assert by_proj["p_lf01"]["towns_traversed"] == ["Springfield", "Decatur"]
        assert by_proj["p_lf02"]["towns_traversed"] == ["Beaufort", "Charleston"]

    def test_sibling_lfs_same_day_arc_windowed_to_own_resolved(self, tmp_path):
        """Two LFs recorded the SAME calendar day must NOT each inherit the
        whole day's journey arc + towns. Each LF's arc/town tags are windowed to
        its OWN canonical ``ResolvedLocation.ordered_place_sequence`` (the single
        source of truth the title/OST already used), never the day-wide aggregate
        that spans every sibling's segments.
        """
        from gomoveshift.models import ResolvedLocation
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

        gen = self._mk_meta_gen()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1), _lf(2)])
        # One calendar day, two episodes → the day-wide aggregate spans all four
        # towns; the buggy path hands that same aggregate to BOTH sibling LFs.
        ep_a = RouteProfile(
            start_town="Alpha",
            end_town="Beta",
            towns_traversed=["Alpha", "Beta"],
        ).to_dict()
        ep_b = RouteProfile(
            start_town="Gamma",
            end_town="Delta",
            towns_traversed=["Gamma", "Delta"],
        ).to_dict()
        state = make_state(
            episode_profiles=[ep_a, ep_b],
            episode_dates=["2025-09-01", "2025-09-01"],
            per_lf_recording_date={1: "2025-09-01", 2: "2025-09-01"},
            per_lf_resolved_location={
                1: ResolvedLocation(ordered_place_sequence=["Alpha", "Beta"]),
                2: ResolvedLocation(ordered_place_sequence=["Gamma", "Delta"]),
            },
        )
        with _pass_redundancy():
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
                state=state,
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        by_proj = {
            c.kwargs["project_name"]: c.kwargs["route_profile"] for c in gen.generate.call_args_list
        }
        # Each LF's arc/towns is its OWN sub-arc — no sibling leak.
        assert by_proj["p_lf01"]["towns_traversed"] == ["Alpha", "Beta"]
        assert by_proj["p_lf01"]["start_town"] == "Alpha"
        assert by_proj["p_lf01"]["end_town"] == "Beta"
        assert by_proj["p_lf02"]["towns_traversed"] == ["Gamma", "Delta"]
        assert by_proj["p_lf02"]["start_town"] == "Gamma"
        assert by_proj["p_lf02"]["end_town"] == "Delta"
        # Symptom level: LF1's rendered description/tags never name LF2's towns.
        enriched = LongFormMetadataGenerator._enrich_with_route_profile(
            {"description": "", "tags": []}, by_proj["p_lf01"]
        )
        assert "Started in Alpha, ended in Beta" in enriched["description"]
        assert "Gamma" not in enriched["description"]
        assert "Delta" not in enriched["description"]
        assert not any("Gamma" in t or "Delta" in t for t in enriched["tags"])

    def test_sibling_lfs_same_day_roads_windowed_to_own_episodes(self, tmp_path):
        """Two LFs recorded the SAME calendar day must NOT each
        ship the whole DAY's road list. The road tags + "Roads traveled:" line
        are windowed to the LF's OWN episode span
        (``state.per_lf_gpx_episode_paths``) so LF 2 never names roads only
        LF 1 drove.
        """
        from gomoveshift.pipeline.longform_metadata import LongFormMetadataGenerator

        gen = self._mk_meta_gen()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1), _lf(2)])
        # One calendar day, two episodes → the day aggregate merges both road
        # lists; the buggy path hands that same merge to BOTH sibling LFs.
        ep_a = RouteProfile(roads_traversed=["US-17"]).to_dict()
        ep_b = RouteProfile(roads_traversed=["SC-174"]).to_dict()
        state = make_state(
            episode_profiles=[ep_a, ep_b],
            episode_dates=["2025-09-01", "2025-09-01"],
            per_lf_recording_date={1: "2025-09-01", 2: "2025-09-01"},
            gpx_files=["/g/ep0.gpx", "/g/ep1.gpx"],
            per_lf_gpx_episode_paths={
                1: {"/g/ep0.gpx": [(0.0, 60.0)]},
                2: {"/g/ep1.gpx": [(0.0, 60.0)]},
            },
        )
        with _pass_redundancy():
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
                state=state,
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        by_proj = {
            c.kwargs["project_name"]: c.kwargs["route_profile"] for c in gen.generate.call_args_list
        }
        assert by_proj["p_lf01"]["roads_traversed"] == ["US-17"]
        assert by_proj["p_lf02"]["roads_traversed"] == ["SC-174"]
        # Symptom level: LF2's rendered description/tags never name LF1's road.
        enriched = LongFormMetadataGenerator._enrich_with_route_profile(
            {"description": "", "tags": []}, by_proj["p_lf02"]
        )
        assert "Roads traveled: SC-174." in enriched["description"]
        assert "US-17" not in enriched["description"]
        assert not any("US-17" in t for t in enriched["tags"])

    def test_roads_keep_day_aggregate_without_episode_span_map(self, tmp_path):
        """Legacy / GPS-less state with NO ``per_lf_gpx_episode_paths`` keeps
        the day-aggregate road list — the same degrade contract as the arc/towns
        rescope (never invent a window from missing evidence)."""
        gen = self._mk_meta_gen()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1), _lf(2)])
        ep_a = RouteProfile(roads_traversed=["US-17"]).to_dict()
        ep_b = RouteProfile(roads_traversed=["SC-174"]).to_dict()
        state = make_state(
            episode_profiles=[ep_a, ep_b],
            episode_dates=["2025-09-01", "2025-09-01"],
            per_lf_recording_date={1: "2025-09-01", 2: "2025-09-01"},
        )
        with _pass_redundancy():
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4", "/v/lf2.mp4"],
                state=state,
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        by_proj = {
            c.kwargs["project_name"]: c.kwargs["route_profile"] for c in gen.generate.call_args_list
        }
        assert by_proj["p_lf01"]["roads_traversed"] == ["US-17", "SC-174"]
        assert by_proj["p_lf02"]["roads_traversed"] == ["US-17", "SC-174"]

    def test_single_lf_day_roads_not_windowed(self, tmp_path):
        """A single-LF day keeps the day-aggregate road list even when the
        episode-span map names only a subset of the day's episodes — there is
        no sibling to leak into (same no-op contract as the sibling-LF windowing)."""
        gen = self._mk_meta_gen()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
        ep_a = RouteProfile(roads_traversed=["US-17"]).to_dict()
        ep_b = RouteProfile(roads_traversed=["SC-174"]).to_dict()
        state = make_state(
            episode_profiles=[ep_a, ep_b],
            episode_dates=["2025-09-01", "2025-09-01"],
            per_lf_recording_date={1: "2025-09-01"},
            gpx_files=["/g/ep0.gpx", "/g/ep1.gpx"],
            per_lf_gpx_episode_paths={1: {"/g/ep0.gpx": [(0.0, 60.0)]}},
        )
        with _pass_redundancy():
            _build_metadata_per_lf(
                meta_gen=gen,
                drive_plan=drive_plan,
                regular_lf_paths=["/v/lf1.mp4"],
                state=state,
                location="x",
                publish_day_full="Wednesday",
                project_base="p",
                output_base=tmp_path,
            )
        passed = gen.generate.call_args.kwargs["route_profile"]
        assert passed["roads_traversed"] == ["US-17", "SC-174"]


# ─── _generate_chapters_all_lfs ─────────────────────────────────────────


class TestGenerateChaptersAllLfs:
    def test_no_gpx_state_returns_early(self, tmp_path, caplog):
        config = MagicMock()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
        meta_per_lf = {1: {"description": "d"}}
        with caplog.at_level("INFO"):
            _generate_chapters_all_lfs(
                config=config,
                drive_plan=drive_plan,
                state=make_state(),
                metadata_per_lf=meta_per_lf,
                location="x",
                project_base="p",
                output_base=tmp_path,
                stages_dir=tmp_path,
            )
        assert meta_per_lf == {1: {"description": "d"}}  # unchanged
        assert any("No GPX files" in r.getMessage() for r in caplog.records)

    def test_missing_gpx_file_skips_lf(self, tmp_path):
        config = MagicMock()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
        meta_per_lf = {1: {"description": "d"}}
        _generate_chapters_all_lfs(
            config=config,
            drive_plan=drive_plan,
            state=make_state(per_lf_gpx_path={1: str(tmp_path / "missing.gpx")}),
            metadata_per_lf=meta_per_lf,
            location="x",
            project_base="p",
            output_base=tmp_path,
            stages_dir=tmp_path,
        )
        # No chapters written because gpx doesn't exist on disk
        assert "chapters" not in meta_per_lf[1]

    def test_embeds_chapters_and_rewrites_json(self, tmp_path):
        config = MagicMock()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
        gpx = tmp_path / "lf1.gpx"
        gpx.write_text("<gpx></gpx>")
        meta_per_lf = {1: {"description": "Hook line", "title": "T"}}
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf",
                # The embed only fires for a renderable block
                # (>= YOUTUBE_MIN_CHAPTERS). Use a realistic post-pad 3-chapter
                # return so the embed path is exercised.
                return_value=(
                    [
                        {"start": 0, "text": "Intro"},
                        {"start": 60, "text": "Midpoint"},
                        {"start": 120, "text": "Finish"},
                    ],
                    "0:00 Intro",
                ),
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._LFMeta._embed_chapters",
                return_value="Hook line\n\n0:00 Intro",
            ),
            patch(
                "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
                return_value=SimpleNamespace(passed=True, message=""),
            ),
        ):
            _generate_chapters_all_lfs(
                config=config,
                drive_plan=drive_plan,
                state=make_state(per_lf_gpx_path={1: str(gpx)}),
                metadata_per_lf=meta_per_lf,
                location="x",
                project_base="proj",
                output_base=tmp_path,
                stages_dir=tmp_path,
            )
        assert meta_per_lf[1]["chapters"] == [
            {"start": 0, "text": "Intro"},
            {"start": 60, "text": "Midpoint"},
            {"start": 120, "text": "Finish"},
        ]
        assert meta_per_lf[1]["chapter_text"] == "0:00 Intro"
        assert "0:00 Intro" in meta_per_lf[1]["description"]
        # Re-written youtube.json carries the embedded description
        written = json.loads((tmp_path / "proj_lf01_youtube.json").read_text())
        assert "0:00 Intro" in written["description"]

    def test_chapter_generation_exception_is_logged_not_raised(self, tmp_path, caplog):
        config = MagicMock()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
        gpx = tmp_path / "lf1.gpx"
        gpx.write_text("<gpx></gpx>")
        meta_per_lf = {1: {"description": "d"}}
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf",
                side_effect=RuntimeError("gpx parse oops"),
            ),
            patch(
                "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
                return_value=SimpleNamespace(passed=True, message=""),
            ),
            caplog.at_level("WARNING"),
        ):
            _generate_chapters_all_lfs(
                config=config,
                drive_plan=drive_plan,
                state=make_state(per_lf_gpx_path={1: str(gpx)}),
                metadata_per_lf=meta_per_lf,
                location="x",
                project_base="p",
                output_base=tmp_path,
                stages_dir=tmp_path,
            )
        assert meta_per_lf[1]["chapters"] == []
        assert meta_per_lf[1]["chapter_text"] == ""
        assert any("chapter generation failed" in r.getMessage() for r in caplog.records)

    def test_redundancy_gate_failure_after_embed_raises(self, tmp_path):
        config = MagicMock()
        drive_plan = SimpleNamespace(regular_lfs=[_lf(1)])
        gpx = tmp_path / "lf1.gpx"
        gpx.write_text("<gpx></gpx>")
        meta_per_lf = {1: {"description": "Hook"}}
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._generate_chapters_for_lf",
                return_value=([], ""),
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_d._LFMeta._embed_chapters",
                return_value="Hook",
            ),
            patch(
                "gomoveshift.pipeline.validation.content_gates.check_description_redundancy",
                return_value=SimpleNamespace(passed=False, message="dup-after-embed"),
            ),
        ):
            with pytest.raises(RuntimeError, match="after chapter embed"):
                _generate_chapters_all_lfs(
                    config=config,
                    drive_plan=drive_plan,
                    state=make_state(per_lf_gpx_path={1: str(gpx)}),
                    metadata_per_lf=meta_per_lf,
                    location="x",
                    project_base="p",
                    output_base=tmp_path,
                    stages_dir=tmp_path,
                )


def test_publish_run_no_titles_yaml_composes_for_every_short(tmp_path):
    """titles.yaml is an optional override: a publish run with verified sidecars
    and NO titles.yaml composes a stamped title for every short."""
    from gomoveshift.pipeline.drive_phases.phase_d_scheduling import _build_social_posts_per_lf
    from gomoveshift.pipeline.social_generator import SocialGenerator
    from _state_helpers import verified_shorts_per_lf

    stages = tmp_path / "stages"
    shorts = verified_shorts_per_lf(stages, {1: 2, 2: 1})
    titles_file = tmp_path / "social" / "titles.yaml"
    assert not titles_file.exists()
    posts = _build_social_posts_per_lf(
        social_gen_cls=SocialGenerator,
        drive_plan=SimpleNamespace(regular_lfs=[_lf(1), _lf(2)]),
        shorts_per_lf=shorts,
        per_lf_locations_override={1: "Cork, Ireland", 2: "Cork, Ireland"},
        project_base="p",
        output_base=tmp_path / "out",
        titles_file=titles_file,
        require_verified_places=True,
    )
    titles = [p.youtube_title for lf_posts in posts.values() for p in lf_posts]
    assert len(titles) == 3
    assert all(t.strip() and "stretch:" in t for t in titles)
    assert len(set(titles)) == 3


def _config():
    from gomoveshift.config import get_config

    return get_config()
