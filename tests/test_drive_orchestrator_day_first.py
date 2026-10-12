"""Tests for the day-first wiring in DrivePipelineOrchestrator.

The day-first path adds a feature-flagged branch in
`run()` that swaps the old per-episode Phase B + Phase C for a single
`_phase_bc_day_first` method which delegates to
`day_pipeline.run_all_days`. Phase D still runs afterwards, driven by a
minimal synthetic `DrivePlan` this new method populates on state.

These tests mock `run_all_days` and `group_episodes_by_day` to avoid
any real encoding, and verify only the wiring: the right calls happen
with the right inputs, and `state` ends up with everything Phase D
expects (regular_lf_paths, drive_plan skeleton, etc.).
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.config import Config
from gomoveshift.models import RegularLFPlan
from gomoveshift.pipeline.day_grouper import DayBatch
from gomoveshift.pipeline.day_pipeline import DayArtifacts
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

from _state_helpers import make_state


@pytest.fixture(autouse=True)
def _bypass_boundary_contracts(monkeypatch):
    """Bypass BC→D boundary contracts for orchestrator wiring tests.

    These tests mock ``run_all_days`` / ``group_episodes_by_day`` to avoid
    real encoding and verify only orchestrator wiring (the right calls fire
    with the right inputs). The aggregator contracts
    (``assert_phase_bc_per_lf_complete`` / ``assert_phase_d_per_lf_inputs_present``)
    require BC to populate ``per_lf_locations`` / ``per_lf_trim_ratio`` /
    ``per_lf_recording_date`` for every rendered LF — which the mocked path
    cannot satisfy because the fake ``DayArtifacts`` carry no real LF plans
    for the BC computation to consume. We no-op the contracts here so the
    wiring assertions can still be made; the contracts themselves are
    covered by ``tests/test_phase_boundary_contracts.py``.
    """
    monkeypatch.setattr(
        "gomoveshift.pipeline.drive_phases.phase_bc_day_first.assert_phase_bc_per_lf_complete",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "gomoveshift.pipeline.drive_phases.phase_d.assert_phase_d_per_lf_inputs_present",
        lambda *a, **k: None,
    )


class _FakeEpisode:
    def __init__(self, index, concat_path, total_duration, date):
        self.index = index
        self.concat_path = concat_path
        self.total_duration = total_duration
        self.date = date
        self.creation_time = None
        self.gpx_path = None


def _make_state_for_day_first(
    tmp_path,
    num_days=2,
    eps_per_day=2,
    per_day_locations=None,
):
    """Build a state dict resembling what Phase A hands off to Phase B/C.

    `per_day_locations` is an optional list of OST strings, one per day,
    applied to every episode within that day. Use it to simulate realistic
    multi-day drives where Day1 has a different location than Day2.
    """
    output_base = tmp_path / "out"
    output_base.mkdir(parents=True, exist_ok=True)
    stages_dir = output_base / "_stages"
    stages_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    episode_files = []
    episode_locations = []
    file_durations = []
    file_boundaries = []
    episode_dates = []
    offset = 0.0
    drive_ep_index = 0
    dates = [f"2026-02-{20 + d}" for d in range(num_days)]
    for day_idx, date in enumerate(dates):
        day_loc = (
            per_day_locations[day_idx]
            if per_day_locations and day_idx < len(per_day_locations)
            else "Test, SC"
        )
        for _ in range(eps_per_day):
            ep_path = tmp_path / f"ep_{drive_ep_index:03d}.mp4"
            ep_path.write_bytes(b"fake")
            episodes.append(
                _FakeEpisode(
                    index=drive_ep_index,
                    concat_path=ep_path,
                    total_duration=600.0,
                    date=date,
                )
            )
            episode_files.append(ep_path)
            episode_locations.append(day_loc)
            file_durations.append(600.0)
            file_boundaries.append((offset, offset + 600.0))
            offset += 600.0
            episode_dates.append(date)
            drive_ep_index += 1

    # Fake ingest plan whose .episodes attribute is our list
    class _FakeIngestPlan:
        pass

    plan = _FakeIngestPlan()
    plan.episodes = episodes

    # Drive-wide score array spanning all episodes
    total_sec = int(offset)
    drive_scores = np.zeros((total_sec, 9), dtype=np.float64)
    drive_scores[:, 0] = np.arange(total_sec, dtype=np.float64)
    drive_scores[:, 1] = 0.5
    # Real scorer output for moving footage carries motion energy; give every
    # second a realistic driving value so the default-on min_motion_energy floor
    # + per-day stop mask don't read this composite-only fixture as parked (which
    # would assemble 0 LF segments and break short attribution).
    drive_scores[:, 2] = 0.05

    state = make_state(
        source_dir=tmp_path,
        output_base=output_base,
        region="usa",
        publish_day="wed",
        location="Test Drive, SC",
        ingest_plan=plan,
        episode_files=episode_files,
        file_durations=file_durations,
        file_boundaries=file_boundaries,
        episode_dates=episode_dates,
        drive_scores=drive_scores,
        gps_per_episode=[(None, 0.0)] * len(episodes),
        episode_locations=episode_locations,
        gpx_files=[None] * len(episodes),
    )
    # Synthetic drive_scores produce LFs without
    # GPS centroids; the post-discovery validation gate would
    # otherwise refuse to enter encoding. Provide blanket per-LF
    # overrides distinct from "Test Drive, SC" so the gate's
    # ambiguous-input branch is satisfied without changing test
    # intent (these tests pin orchestrator wiring, not OST text).
    state.per_lf_overrides = {i: f"Synthetic Route {i}, SC" for i in range(0, 21)}
    return state, dates


def _fake_day_artifacts(tmp_path, date, num_regular=2, num_shorts=4):
    day_dir = tmp_path / "out" / "_stages" / f"day_{date}"
    day_dir.mkdir(parents=True, exist_ok=True)
    master = day_dir / f"day_{date}_master.mp4"
    master.write_bytes(b"master")
    regular_lfs = []
    for i in range(num_regular):
        lf_path = day_dir / "regular_lfs" / f"lf_{i + 1:02d}.mp4"
        lf_path.parent.mkdir(parents=True, exist_ok=True)
        lf_path.write_bytes(b"lf")
        regular_lfs.append(lf_path)
    shorts = []
    short_ranges = []
    for i in range(num_shorts):
        sp = day_dir / "shorts" / f"short_{i + 1:02d}.mp4"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_bytes(b"short")
        shorts.append(sp)
        # Real ``run_all_days`` artifacts always carry ``short_ranges``
        # parallel to ``short_paths`` (day-master timeline). The legacy
        # equal-split fallback is a hard-fail, so the fakes
        # must honour the same contract.
        start = 100.0 + i * 300.0
        short_ranges.append(SimpleNamespace(rank=i + 1, start_sec=start, end_sec=start + 40.0))
    return DayArtifacts(
        date=date,
        master_path=master,
        regular_lf_paths=regular_lfs,
        short_paths=shorts,
        short_ranges=short_ranges,
    )


# TestDayFirstConfigFlag removed — the day_first flag was deleted.
# day-first is no longer a configurable mode; it is the only path.


class TestPhaseBcDayFirst:
    """Tests for the new _phase_bc_day_first method."""

    def test_method_exists(self):
        orch = DrivePipelineOrchestrator()
        assert hasattr(orch, "_phase_bc_day_first")

    def test_calls_group_episodes_by_day(self, tmp_path):
        state, dates = _make_state_for_day_first(tmp_path)
        orch = DrivePipelineOrchestrator()

        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[
                    DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes[:2]),
                    DayBatch(date=dates[1], episodes=state["ingest_plan"].episodes[2:]),
                ],
            ) as mock_group,
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=[
                    _fake_day_artifacts(tmp_path, dates[0]),
                    _fake_day_artifacts(tmp_path, dates[1]),
                ],
            ),
        ):
            orch._phase_bc_day_first(state)

        mock_group.assert_called_once()

    def test_calls_run_all_days_with_correct_inputs(self, tmp_path):
        state, dates = _make_state_for_day_first(tmp_path)
        orch = DrivePipelineOrchestrator()

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return [
                _fake_day_artifacts(tmp_path, dates[0]),
                _fake_day_artifacts(tmp_path, dates[1]),
            ]

        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[
                    DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes[:2]),
                    DayBatch(date=dates[1], episodes=state["ingest_plan"].episodes[2:]),
                ],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=_capture,
            ),
        ):
            orch._phase_bc_day_first(state)

        assert "day_batches" in captured
        assert len(captured["day_batches"]) == 2
        assert "drive_scores" in captured
        assert captured["drive_scores"] is state["drive_scores"]
        assert "file_boundaries" in captured
        assert captured["file_boundaries"] == state["file_boundaries"]
        assert captured["location"] == "Test Drive, SC"
        assert captured["episode_paths"] == state["episode_files"]

    def test_populates_regular_lf_paths_across_days(self, tmp_path):
        state, dates = _make_state_for_day_first(tmp_path)
        orch = DrivePipelineOrchestrator()

        artifacts = [
            _fake_day_artifacts(tmp_path, dates[0], num_regular=2),
            _fake_day_artifacts(tmp_path, dates[1], num_regular=3),
        ]
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[
                    DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes[:2]),
                    DayBatch(date=dates[1], episodes=state["ingest_plan"].episodes[2:]),
                ],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=artifacts,
            ),
        ):
            orch._phase_bc_day_first(state)

        regular_paths = state["regular_lf_paths"]
        assert len(regular_paths) == 5  # 2 + 3

    def test_populates_skeleton_drive_plan_for_phase_d(self, tmp_path):
        """Phase D depends on drive_plan.regular_lfs — synthesize a skeleton."""
        state, dates = _make_state_for_day_first(tmp_path)
        orch = DrivePipelineOrchestrator()

        artifacts = [
            _fake_day_artifacts(tmp_path, dates[0], num_regular=2),
            _fake_day_artifacts(tmp_path, dates[1], num_regular=3),
        ]
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[
                    DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes[:2]),
                    DayBatch(date=dates[1], episodes=state["ingest_plan"].episodes[2:]),
                ],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=artifacts,
            ),
            patch.object(
                DrivePipelineOrchestrator,
                "_probe_duration",
                return_value=900.0,
            ),
        ):
            orch._phase_bc_day_first(state)

        drive_plan = state["drive_plan"]
        assert drive_plan is not None
        assert hasattr(drive_plan, "regular_lfs")
        assert len(drive_plan.regular_lfs) == 5  # 2 + 3
        # Each LF (real or fallback skeleton) must have a unique index
        indices = [lf.index for lf in drive_plan.regular_lfs]
        assert len(set(indices)) == 5
        # Each needs a region for Phase D
        for lf in drive_plan.regular_lfs:
            assert lf.region == "usa"

    def test_empty_artifacts_list_raises_runtime_error(self, tmp_path):
        """When run_all_days returns no artifacts (all days failed),
        Phase BC must raise RuntimeError so Phase D never runs on empty results."""
        import pytest

        state, dates = _make_state_for_day_first(tmp_path)
        orch = DrivePipelineOrchestrator()

        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[DayBatch(date=dates[0], episodes=[])],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=[],
            ),
            pytest.raises(RuntimeError, match="Phase BC incomplete"),
        ):
            orch._phase_bc_day_first(state)

    def test_resumes_from_phase_a_without_ingest_plan_object(self, tmp_path):
        """When resuming from a Phase A checkpoint, `ingest_plan` is NOT
        in state (Python objects don't survive JSON serialization). The
        day-first phase must synthesize episodes from the persisted
        scalar fields (episode_files, episode_dates, file_durations,
        gpx_files). Without this fix, re-runs under day_first
        abort immediately with 'no ingest plan'.
        """
        output_base = tmp_path / "out"
        output_base.mkdir(parents=True)
        (output_base / "_stages").mkdir()

        # Build what a JSON-loaded state looks like — Paths as strings,
        # no ingest_plan object
        ep_paths = [tmp_path / f"ep_{i:02d}.mp4" for i in range(4)]
        for p in ep_paths:
            p.write_bytes(b"fake")
        state = make_state(
            source_dir=tmp_path,
            output_base=output_base,
            region="usa",
            publish_day="sat",
            location="Beaufort to Highlands, SC",
            episode_files=[str(p) for p in ep_paths],
            episode_dates=[
                "2026-02-20",
                "2026-02-20",
                "2026-02-21",
                "2026-02-21",
            ],
            file_durations=[600.0, 600.0, 900.0, 900.0],
            file_boundaries=[
                (0.0, 600.0),
                (600.0, 1200.0),
                (1200.0, 2100.0),
                (2100.0, 3000.0),
            ],
            episode_locations=[
                "Beaufort, SC",
                "Beaufort, SC",
                "Cashiers, NC",
                "Cashiers, NC",
            ],
            gpx_files=[None, None, None, None],
            drive_scores=np.zeros((3000, 9), dtype=np.float64),
            # NO ingest_plan — simulates a JSON resume
        )
        # Blanket per-LF overrides so the post-discovery
        # validation gate doesn't reject the synthetic GPS-less plan.
        state.per_lf_overrides = {i: f"Synthetic Route {i}, SC" for i in range(0, 21)}
        # Seed timestamps
        state.drive_scores[:, 0] = np.arange(3000, dtype=np.float64)

        orch = DrivePipelineOrchestrator()

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return [
                _fake_day_artifacts(tmp_path, "2026-02-20", num_regular=1),
                _fake_day_artifacts(tmp_path, "2026-02-21", num_regular=1),
            ]

        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=_capture,
            ),
            patch.object(
                DrivePipelineOrchestrator,
                "_probe_duration",
                return_value=900.0,
            ),
        ):
            orch._phase_bc_day_first(state)

        # The phase should have successfully built day batches and called run_all_days
        assert "day_batches" in captured
        assert len(captured["day_batches"]) == 2
        assert captured["day_batches"][0].date == "2026-02-20"
        assert captured["day_batches"][1].date == "2026-02-21"

        # Per-day location derivation should still work from episode_locations
        assert state["per_day_locations"]["2026-02-20"] == "Beaufort, SC"
        assert state["per_day_locations"]["2026-02-21"] == "Cashiers, NC"

        # And state keys that Phase D needs are populated
        assert "drive_plan" in state
        assert state["drive_plan"] is not None
        assert len(state["regular_lf_paths"]) == 2


class TestRunDispatchWithDayFirst:
    """When config.drive.day_first=True, run() must dispatch to the new path."""

    @patch("gomoveshift.pipeline.drive_orchestrator.ReviewGate")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator."
        "DrivePipelineOrchestrator._phase_d_metadata_and_publish"
    )
    @patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator._phase_bc_day_first")
    @patch(
        "gomoveshift.pipeline.drive_orchestrator."
        "DrivePipelineOrchestrator._phase_a_ingest_and_analyze"
    )
    def test_day_first_dispatches_to_bc_day_first(
        self,
        mock_a,
        mock_bc,
        mock_d,
        mock_rg,
        tmp_path,
    ):
        # Phase methods mutate `state` in place; the coherence guard
        # refuses to mark a phase complete if its required keys are
        # missing/empty (phase_a needs episode_files + drive_scores;
        # phase_bc_day_first needs day_artifacts). Populate via side_effect.
        def _phase_a_side_effect(state):
            state["episode_files"] = ["fake_ep.mp4"]
            state["drive_scores"] = "non-empty-marker"
            # Real phase_a always writes episode_locations parallel to
            # episode_files; empty-with-episodes now hard-fails the OST
            # gate, so the mock must be honest.
            state["episode_locations"] = ["Fake Town, SC"]

        def _phase_bc_side_effect(state):
            state["day_artifacts"] = [object()]
            # Cross-field invariants
            state["regular_lf_paths"] = []
            state["shorts_per_lf"] = {}

        def _phase_d_side_effect(state, drive_plan=None):
            return None

        mock_a.side_effect = _phase_a_side_effect
        mock_bc.side_effect = _phase_bc_side_effect
        mock_d.side_effect = _phase_d_side_effect

        source = tmp_path / "source"
        source.mkdir()

        orch = DrivePipelineOrchestrator()
        result = orch.run(
            source_dir=source,
            output_base=tmp_path / "output",
            region="usa",
            publish_day="wed",
        )

        mock_a.assert_called_once()
        mock_bc.assert_called_once()
        mock_d.assert_called_once()
        assert "phase_bc_day_first" in result.stages_completed


class TestPerDayLocationDerivation:
    """Multi-day drives must not leak Day 1's
    location onto Day 2's LFs. Phase_bc_day_first must derive a per-day
    location from episode_locations and expose it via state so Phase D
    can use the right location per LF.
    """

    def test_derives_distinct_locations_for_two_days(self, tmp_path):
        """Day 1 = Beaufort; Day 2 = Cashiers. Each day's LFs get their
        own location."""
        state, dates = _make_state_for_day_first(
            tmp_path,
            num_days=2,
            eps_per_day=2,
            per_day_locations=["Beaufort, SC", "Cashiers, NC"],
        )
        orch = DrivePipelineOrchestrator()

        artifacts = [
            _fake_day_artifacts(tmp_path, dates[0], num_regular=2),
            _fake_day_artifacts(tmp_path, dates[1], num_regular=3),
        ]
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[
                    DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes[:2]),
                    DayBatch(date=dates[1], episodes=state["ingest_plan"].episodes[2:]),
                ],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=artifacts,
            ),
            patch.object(
                DrivePipelineOrchestrator,
                "_probe_duration",
                return_value=900.0,
            ),
        ):
            orch._phase_bc_day_first(state)

        per_day_locations = state["per_day_locations"]
        assert per_day_locations[dates[0]] == "Beaufort, SC"
        assert per_day_locations[dates[1]] == "Cashiers, NC"

    def test_two_town_day_uses_majority_vote_not_first_episode(self, tmp_path):
        """A two-town day takes
        the majority label across its episodes, not the morning town's."""
        state, dates = _make_state_for_day_first(tmp_path, num_days=1, eps_per_day=3)
        state["episode_locations"] = ["Town A, SC", "Town B, SC", "Town B, SC"]
        orch = DrivePipelineOrchestrator()

        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[
                    DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes),
                ],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=[_fake_day_artifacts(tmp_path, dates[0], num_regular=2)],
            ),
            patch.object(
                DrivePipelineOrchestrator,
                "_probe_duration",
                return_value=900.0,
            ),
        ):
            orch._phase_bc_day_first(state)

        assert state["per_day_locations"][dates[0]] == "Town B, SC"

    def test_per_lf_location_map_spans_all_days(self, tmp_path):
        """state['per_lf_locations'] maps 1..N → a non-empty, distinct
        label per LF.

        The legacy form of this test asserted that the
        per-LF map fell back to ``per_day_locations`` (Beaufort/Cashiers)
        — that fallback path was deleted. The test now pins the
        surviving postcondition: the map covers every LF and has
        distinct labels (the override path the helper installs supplies
        ``"Synthetic Route N, SC"``)."""
        state, dates = _make_state_for_day_first(
            tmp_path,
            num_days=2,
            eps_per_day=2,
            per_day_locations=["Beaufort, SC", "Cashiers, NC"],
        )
        orch = DrivePipelineOrchestrator()

        artifacts = [
            _fake_day_artifacts(tmp_path, dates[0], num_regular=2),
            _fake_day_artifacts(tmp_path, dates[1], num_regular=3),
        ]
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[
                    DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes[:2]),
                    DayBatch(date=dates[1], episodes=state["ingest_plan"].episodes[2:]),
                ],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=artifacts,
            ),
            patch.object(
                DrivePipelineOrchestrator,
                "_probe_duration",
                return_value=900.0,
            ),
        ):
            orch._phase_bc_day_first(state)

        per_lf_locations = state["per_lf_locations"]
        labels = list(per_lf_locations.values())
        assert all(lbl for lbl in labels), "every LF must have a non-empty label"
        assert len(set(labels)) == len(labels), "labels must be distinct per LF"

    def test_passes_per_day_locations_to_run_all_days(self, tmp_path):
        """run_all_days must receive the per-day locations so the OST
        burns on each day's master + LFs + shorts get the right text."""
        state, dates = _make_state_for_day_first(
            tmp_path,
            num_days=2,
            eps_per_day=2,
            per_day_locations=["Beaufort, SC", "Cashiers, NC"],
        )
        orch = DrivePipelineOrchestrator()

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return [
                _fake_day_artifacts(tmp_path, dates[0]),
                _fake_day_artifacts(tmp_path, dates[1]),
            ]

        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[
                    DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes[:2]),
                    DayBatch(date=dates[1], episodes=state["ingest_plan"].episodes[2:]),
                ],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                side_effect=_capture,
            ),
            patch.object(
                DrivePipelineOrchestrator,
                "_probe_duration",
                return_value=900.0,
            ),
        ):
            orch._phase_bc_day_first(state)

        # per_day_locations kwarg must be present and populated
        assert "per_day_locations" in captured
        assert captured["per_day_locations"][dates[0]] == "Beaufort, SC"
        assert captured["per_day_locations"][dates[1]] == "Cashiers, NC"

    def test_falls_back_to_global_location_when_episode_locations_empty(
        self,
        tmp_path,
    ):
        """When a day has only empty episode_locations, the per-day map
        falls back to the drive-wide state['location']."""
        state, dates = _make_state_for_day_first(tmp_path, num_days=1, eps_per_day=2)
        # Wipe out episode locations
        state["episode_locations"] = ["", ""]
        orch = DrivePipelineOrchestrator()

        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[
                    DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes),
                ],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=[_fake_day_artifacts(tmp_path, dates[0])],
            ),
            patch.object(
                DrivePipelineOrchestrator,
                "_probe_duration",
                return_value=900.0,
            ),
        ):
            orch._phase_bc_day_first(state)

        # Falls back to drive-wide location
        assert state["per_day_locations"][dates[0]] == "Test Drive, SC"


class TestBuildLfTitleGpsPointsWiring:
    """build_lf_title must receive gps_points so GPS-filtered centroids
    are used instead of segment centroids, which may be contaminated by
    cold-start anomalies (null-island or wrong-satellite-cluster coords)."""

    def test_gps_points_passed_to_build_lf_title(self, tmp_path):
        """Phase BC wires gps_points_by_lf into build_lf_title so the
        spatial outlier filter can reject anomalous GPS before geocoding."""
        from unittest.mock import call as mcall

        state, dates = _make_state_for_day_first(tmp_path, num_days=1, eps_per_day=2)
        # The helper seeds a synthetic --per-lf-location override for every
        # LF (so the resolvability gate passes on this no-real-GPS fixture). An
        # operator override is now burned VERBATIM and skips build_lf_title, so
        # clear them here — this test exercises the GPS-cascade wiring, and the
        # patched fake_gps below satisfies the gate via GPS instead.
        state.per_lf_overrides = {}
        orch = DrivePipelineOrchestrator()

        fake_gps = [object()]  # sentinel — any truthy value
        captured_calls = []

        def _fake_build_lf_title(segments, geocoder, per_lf_location, gps_points=None, **kwargs):
            captured_calls.append(gps_points)
            return per_lf_location

        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=[DayBatch(date=dates[0], episodes=state["ingest_plan"].episodes)],
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=[_fake_day_artifacts(tmp_path, dates[0])],
            ),
            patch(
                "gomoveshift.pipeline.gps_enrichment._build_gps_points_by_lf",
                return_value={1: fake_gps, 2: fake_gps},
            ),
            # The sentinel fake_gps isn't real coordinates, so stub the label
            # derivation (which would otherwise raise on the missing centroid) —
            # this test only cares that gps_points reaches build_lf_title.
            # Cover LF 3 too: the synthetic drive_scores assemble a 3rd LF that
            # run_all_days (mocked to 2 finals) leaves as an empty-segment
            # skeleton; the resolvable gate refuses a skeleton absent from
            # per_lf_locations, so give it a label like production's complete
            # derive would.
            patch(
                "gomoveshift.pipeline.gps_enrichment.derive_per_lf_locations",
                return_value={1: "Label 1, SC", 2: "Label 2, SC", 3: "Label 3, SC"},
            ),
            patch(
                "gomoveshift.pipeline.gps_enrichment.build_lf_title",
                side_effect=_fake_build_lf_title,
            ),
            patch.object(DrivePipelineOrchestrator, "_probe_duration", return_value=900.0),
        ):
            orch._phase_bc_day_first(state)

        assert captured_calls, "build_lf_title was never called"
        assert any(c is not None for c in captured_calls), (
            "gps_points was None in every build_lf_title call — "
            "GPS filter is dead and segment centroids (possibly contaminated) are used instead"
        )


class TestBuildGpsPointsByLf:
    """Unit tests for _build_gps_points_by_lf — GPS slicing helper."""

    def _make_drive_plan(self, lf_segs):
        """Build a minimal fake DrivePlan with one LF whose segments are
        ``lf_segs`` — list of (start_time, end_time, file_idx) tuples."""
        from gomoveshift.analysis.shorts_models import GPSData, GPSTrackpoint

        class _Seg:
            def __init__(self, start, end, fi):
                self.start_time = start
                self.end_time = end
                self.source_file_index = fi

        class _LF:
            index = 1
            segments = [_Seg(s, e, fi) for s, e, fi in lf_segs]

        class _Plan:
            regular_lfs = [_LF()]

        return _Plan()

    def _make_trackpoints(self, coords_ts):
        """Return a raw GPS episode dict: {trackpoints: [...]} with one
        point per (lat, lon, timestamp) triple."""
        return {
            "trackpoints": [
                {"lat": lat, "lon": lon, "elevation": 0.0, "timestamp": ts}
                for lat, lon, ts in coords_ts
            ]
        }

    def test_null_island_offset_corrected_to_first_valid_gps(self):
        """When the stored GPS offset is from null-island pre-lock data the
        effective offset must shift to the first non-null-island timestamp."""
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _build_gps_points_by_lf

        # Stored offset = 1000.0 (from null-island era).
        # Valid GPS starts at 9000.0 (post-lock epoch).
        # Segment relative [5, 15] + corrected offset 9000 → abs [9005, 9015].
        null_pts = [
            {"lat": 0.0, "lon": 0.0, "elevation": 0.0, "timestamp": 1000.0 + i} for i in range(5)
        ]
        valid_pts = [
            {"lat": 35.0 + i * 0.001, "lon": -82.0, "elevation": 0.0, "timestamp": 9000.0 + i}
            for i in range(30)
        ]
        ep0 = [{"trackpoints": null_pts + valid_pts}, 1000.0]

        plan = self._make_drive_plan([(5, 15, 0)])
        result = _build_gps_points_by_lf(plan, [ep0])

        assert 1 in result, "LF 1 must have GPS points after offset correction"
        lats = [p.lat for p in result[1]]
        assert all(lat > 34.0 for lat in lats), (
            f"All returned points must be valid NC-area, got: {lats[:5]}"
        )

    def test_stuck_coordinates_deduplicated(self):
        """Consecutive identical (lat, lon) pairs must be collapsed to one
        representative point so the spatial outlier filter is not defeated."""
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import _build_gps_points_by_lf
        from gomoveshift.pipeline.gps_quality import filter_high_quality_points

        # Correct offset (no null-island epoch issue).
        # Segment [0, 199] captures 200 points:
        #   5 valid NC points at offset 0–4 → lat 36.x
        #   195 stuck Caribbean points at offset 5–199 → lat 20.4 (all identical)
        # Without deduplication the stuck points dominate and the NC points
        # are evicted by the 500 km spatial outlier filter.
        valid_nc = [
            {
                "lat": 36.0 + i * 0.001,
                "lon": -81.0 + i * 0.001,
                "elevation": 0.0,
                "timestamp": float(i),
            }
            for i in range(5)
        ]
        stuck = [
            {"lat": 20.4, "lon": -67.4, "elevation": 0.0, "timestamp": float(5 + i)}
            for i in range(195)
        ]
        ep0 = [{"trackpoints": valid_nc + stuck}, 0.0]

        plan = self._make_drive_plan([(0, 199, 0)])
        result = _build_gps_points_by_lf(plan, [ep0])

        assert 1 in result, "LF 1 must have GPS points"
        pts = result[1]
        filtered = filter_high_quality_points(pts)
        assert filtered, "filter_high_quality_points must not evict all points"
        lats = [p.lat for p in filtered]
        assert all(lat > 34.0 for lat in lats), (
            f"After dedup+filter only NC coords should survive, got: {lats[:5]}"
        )


class TestDayPipelineRunAllDaysPerDayLocations:
    """day_pipeline.run_all_days must route per-day locations through
    to run_day_pipeline so each day's OST burn-in uses its own text."""

    def test_run_all_days_passes_per_day_location(self, tmp_path):
        from gomoveshift.pipeline import day_pipeline as dp

        class _Ep:
            def __init__(self, idx):
                self.index = idx
                self.concat_path = tmp_path / f"ep_{idx}.mp4"
                self.concat_path.write_bytes(b"x")
                self.total_duration = 600.0
                self.date = None
                self.creation_time = None

        batches = [
            DayBatch(date="2026-02-20", episodes=[_Ep(0), _Ep(1)]),
            DayBatch(date="2026-02-21", episodes=[_Ep(2), _Ep(3)]),
        ]
        scores = np.zeros((2400, 9), dtype=np.float64)
        scores[:, 0] = np.arange(2400, dtype=np.float64)
        scores[:, 1] = 0.5
        scores[:, 2] = 0.05  # realistic moving footage (see _make_state_for_day_first)
        file_boundaries = [
            (0.0, 600.0),
            (600.0, 1200.0),
            (1200.0, 1800.0),
            (1800.0, 2400.0),
        ]

        captured_locations = []

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kwargs):
            captured_locations.append((day_batch.date, location))
            return dp.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / "master.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            dp.run_all_days(
                day_batches=batches,
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="GLOBAL FALLBACK",
                config=None,
                per_day_locations={
                    "2026-02-20": "Beaufort, SC",
                    "2026-02-21": "Cashiers, NC",
                },
            )

        assert ("2026-02-20", "Beaufort, SC") in captured_locations
        assert ("2026-02-21", "Cashiers, NC") in captured_locations

    def test_run_all_days_falls_back_to_global_when_per_day_missing(
        self,
        tmp_path,
    ):
        from gomoveshift.pipeline import day_pipeline as dp

        class _Ep:
            def __init__(self, idx):
                self.index = idx
                self.concat_path = tmp_path / f"ep_{idx}.mp4"
                self.concat_path.write_bytes(b"x")
                self.total_duration = 600.0
                self.date = None
                self.creation_time = None

        batches = [DayBatch(date="2026-02-20", episodes=[_Ep(0)])]
        scores = np.zeros((600, 9), dtype=np.float64)
        scores[:, 0] = np.arange(600, dtype=np.float64)
        scores[:, 2] = 0.05  # realistic moving footage (see _make_state_for_day_first)
        file_boundaries = [(0.0, 600.0)]

        captured = []

        def _fake_run_day(day_batch, day_scores, output_dir, location, config, **kwargs):
            captured.append(location)
            return dp.DayArtifacts(
                date=day_batch.date,
                master_path=output_dir / "master.mp4",
                regular_lf_paths=[],
                short_paths=[],
            )

        with patch(
            "gomoveshift.pipeline.day_pipeline.run_day_pipeline",
            side_effect=_fake_run_day,
        ):
            dp.run_all_days(
                day_batches=batches,
                drive_scores=scores,
                file_boundaries=file_boundaries,
                output_base=tmp_path,
                location="GLOBAL LOCATION",
                config=None,
                # No per_day_locations passed
            )

        assert captured == ["GLOBAL LOCATION"]


class TestPhaseDReadsPerLFLocationOverride:
    """Phase D must consult state['per_lf_locations'] per-LF so day-first
    metadata gets the right location."""

    def test_phase_d_uses_per_lf_override_when_present(self, tmp_path):
        """With per_lf_locations set, metadata generator receives the
        per-LF location, not the drive-wide state['location']."""
        from gomoveshift.models import DrivePlan, RegularLFPlan

        output_base = tmp_path / "out"
        output_base.mkdir(parents=True)
        (output_base / "_stages").mkdir()

        lf1 = RegularLFPlan(
            index=1,
            segments=[],
            montage_peaks=[],
            shorts_target=7,
            region="usa",
            total_duration=600.0,
        )
        lf2 = RegularLFPlan(
            index=2,
            segments=[],
            montage_peaks=[],
            shorts_target=7,
            region="usa",
            total_duration=600.0,
        )
        plan = DrivePlan(
            source_files=[],
            processed_files=[],
            regular_lfs=[lf1, lf2],
            score_threshold=0.0,
            total_source_duration=1200.0,
            total_highlight_duration=1200.0,
        )
        # ≥1 MiB placeholders so the pre-upload file_integrity gate
        # doesn't HARD-fail (ffprobe stubbed via the PipelineValidator patch).
        lf1_path = output_base / "lf_01.mp4"
        lf1_path.write_bytes(b"\0" * (1024 * 1024 + 1))
        lf2_path = output_base / "lf_02.mp4"
        lf2_path.write_bytes(b"\0" * (1024 * 1024 + 1))

        state = make_state(
            drive_plan=plan,
            regular_lf_paths=[lf1_path, lf2_path],
            shorts_per_lf={1: [], 2: []},
            region="usa",
            # Invariant: publish_day must land on
            # LONG_FORM_DAYS=[2] (Wed). The quality check enforces the
            # same in assert_phase_d_quality.
            publish_day="wed",
            location="DRIVE-WIDE",
            output_base=output_base,
            skip_upload=True,
            skip_csv=True,
            source_dir=tmp_path,
            per_lf_locations={1: "Beaufort, SC", 2: "Cashiers, NC"},
            # BC→D boundary contract — every rendered LF must appear.
            per_lf_recording_date={1: "2026-02-20", 2: "2026-02-21"},
            per_lf_trim_ratio={1: 0.7, 2: 0.7},
            file_durations=[600.0, 600.0],
            episode_files=[lf1_path, lf2_path],
        )

        captured_locations = []

        def _fake_generate(**kwargs):
            captured_locations.append(kwargs.get("location"))
            # Titles must be distinct per LF.
            return {
                "title": f"stub for {kwargs.get('project_name', 'lf')}",
                "description": "",
                "tags": "",
            }

        orch = DrivePipelineOrchestrator()
        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d.LongFormMetadataGenerator"
        ) as mock_cls:
            instance = MagicMock()
            instance.generate = MagicMock(side_effect=_fake_generate)
            mock_cls.return_value = instance
            with patch("gomoveshift.pipeline.drive_phases.phase_d.PipelineScheduler") as mock_sched:
                # drive_schedule must have one
                # lf_slot per regular_lf and dates must land on
                # LONG_FORM_DAYS (Wed). 2026-05-06 is a Wed.
                from datetime import datetime as _dt
                from datetime import timedelta as _td

                _wed_dates = [
                    (_dt(2026, 5, 6) + _td(days=7 * i)).strftime("%Y-%m-%d")
                    for i in range(len(plan.regular_lfs))
                ]
                _lf_slots = [
                    MagicMock(
                        date=d,
                        time="11:00:00",
                        datetime_est=f"{d}T11:00:00-04:00",
                        day_of_week="Wednesday",
                    )
                    for d in _wed_dates
                ]
                sched_instance = MagicMock()
                sched_instance.generate_drive_schedule.return_value = MagicMock(
                    lf_slots=_lf_slots,
                    shorts_slots=[],
                    to_dict=lambda: {},
                )
                mock_sched.return_value = sched_instance
                mock_sched.from_config.return_value = sched_instance
                with patch(
                    "gomoveshift.pipeline.drive_phases.phase_d.SocialGenerator"
                ) as mock_social:
                    soc = MagicMock()
                    soc.generate_all.return_value = []
                    mock_social.return_value = soc
                    with patch(
                        "gomoveshift.pipeline.drive_phases.phase_d.PipelineValidator"
                    ) as mock_val:
                        # 1-byte placeholder LF files now HARD-fail the
                        # pre-upload file_integrity gate (was a silent skip). This
                        # test asserts the per-LF location wiring, not the gate —
                        # stub the validator to pass.
                        mock_val.return_value.validate.return_value = MagicMock(
                            passed=True, summary=lambda: "ok"
                        )
                        orch._phase_d_metadata_and_publish(state)

        assert "Beaufort, SC" in captured_locations
        assert "Cashiers, NC" in captured_locations
        assert "DRIVE-WIDE" not in captured_locations

    def test_phase_d_raises_when_per_lf_locations_missing(
        self,
        tmp_path,
    ):
        """BC→D boundary contract: a missing per_lf_locations entry is
        unrepresentable at Phase D entry. The legacy drive-wide fallback
        is gone — BC populates an entry per LF in production, and a
        missing entry signals state corruption. Phase D raises
        ``PhaseDEntryContractError`` rather than silently falling back.
        """
        from gomoveshift.models import DrivePlan, RegularLFPlan
        from gomoveshift.pipeline.exceptions import PhaseDEntryContractError

        output_base = tmp_path / "out"
        output_base.mkdir(parents=True)
        (output_base / "_stages").mkdir()

        lf1 = RegularLFPlan(
            index=1,
            segments=[],
            montage_peaks=[],
            shorts_target=7,
            region="usa",
            total_duration=600.0,
        )
        plan = DrivePlan(
            source_files=[],
            processed_files=[],
            regular_lfs=[lf1],
            score_threshold=0.0,
            total_source_duration=600.0,
            total_highlight_duration=600.0,
        )
        lf1_path = output_base / "lf_01.mp4"
        lf1_path.write_bytes(b"x")

        state = make_state(
            drive_plan=plan,
            regular_lf_paths=[lf1_path],
            shorts_per_lf={1: []},
            region="usa",
            # Invariant: publish_day on Wed.
            publish_day="wed",
            location="DRIVE-WIDE",
            output_base=output_base,
            skip_upload=True,
            skip_csv=True,
            source_dir=tmp_path,
            # Deliberately no per_lf_locations — the boundary contract should raise.
            file_durations=[600.0],
            episode_files=[lf1_path],
        )

        orch = DrivePipelineOrchestrator()
        # The autouse fixture at module level patches the boundary
        # assertion to no-op; restore the real assertion for THIS test
        # because the test's whole point is that the contract fires.
        from gomoveshift.pipeline import state_invariants

        with patch(
            "gomoveshift.pipeline.drive_phases.phase_d.assert_phase_d_per_lf_inputs_present",
            state_invariants.assert_phase_d_per_lf_inputs_present,
        ):
            with pytest.raises(PhaseDEntryContractError, match="per-LF boundary contract failed"):
                orch._phase_d_metadata_and_publish(state)


class TestRegularLfPathsRecordingDateOrder:
    """regular_lf_paths must be sorted by LF index after phase_bc_day_first.

    Finals-only days process before fresh-encode days, so artifacts may arrive
    out of recording order. Without the sort, Phase D assigns the earliest
    publish slots to later-day LFs.
    """

    def test_regular_lf_paths_sorted_by_lf_index_when_artifacts_out_of_order(self, tmp_path):
        """Artifacts arriving [day_B, day_A] produce paths sorted [day_A, day_B]."""
        # day_A (2026-03-04) → LF indices 0,1  (first trip day, should be first)
        # day_B (2026-03-05) → LF indices 2,3  (second trip day, processed first)
        date_a, date_b = "2026-03-04", "2026-03-05"

        lf_a0 = RegularLFPlan(
            index=0,
            segments=[],
            montage_peaks=[],
            shorts_target=2,
            region="usa",
            total_duration=900.0,
        )
        lf_a1 = RegularLFPlan(
            index=1,
            segments=[],
            montage_peaks=[],
            shorts_target=2,
            region="usa",
            total_duration=900.0,
        )
        lf_b0 = RegularLFPlan(
            index=2,
            segments=[],
            montage_peaks=[],
            shorts_target=2,
            region="usa",
            total_duration=900.0,
        )
        lf_b1 = RegularLFPlan(
            index=3,
            segments=[],
            montage_peaks=[],
            shorts_target=2,
            region="usa",
            total_duration=900.0,
        )

        fake_lf_plans_by_date = {
            date_a: [lf_a0, lf_a1],
            date_b: [lf_b0, lf_b1],
        }

        # day_B artifacts arrive first in the list (simulates finals-only fast path)
        art_b = _fake_day_artifacts(tmp_path, date_b, num_regular=2)
        art_a = _fake_day_artifacts(tmp_path, date_a, num_regular=2)
        artifacts_out_of_order = [art_b, art_a]

        state, _ = _make_state_for_day_first(tmp_path, num_days=2)
        state["episode_dates"] = [date_a, date_a, date_b, date_b]
        # Provide a non-None drive_plan so _partition_lf_plans_by_date is called
        state["drive_plan"] = MagicMock()

        orch = DrivePipelineOrchestrator()

        day_batches = [
            DayBatch(date=date_a, episodes=state["ingest_plan"].episodes[:2]),
            DayBatch(date=date_b, episodes=state["ingest_plan"].episodes[2:]),
        ]

        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=day_batches,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=artifacts_out_of_order,
            ),
            patch(
                "gomoveshift.pipeline.gps_enrichment._partition_lf_plans_by_date",
                return_value=fake_lf_plans_by_date,
            ),
            patch.object(DrivePipelineOrchestrator, "_probe_duration", return_value=900.0),
        ):
            orch._phase_bc_day_first(state)

        paths = state["regular_lf_paths"]
        plan_lfs = state["drive_plan"].regular_lfs

        assert len(paths) == 4, f"expected 4 paths, got {len(paths)}"

        # Paths must be in LF-index order: day_A first, day_B second
        assert date_a in str(paths[0]), f"paths[0] should be day_A: {paths[0]}"
        assert date_a in str(paths[1]), f"paths[1] should be day_A: {paths[1]}"
        assert date_b in str(paths[2]), f"paths[2] should be day_B: {paths[2]}"
        assert date_b in str(paths[3]), f"paths[3] should be day_B: {paths[3]}"

        # drive_plan.regular_lfs indices must also be monotone
        indices = [lf.index for lf in plan_lfs]
        assert indices == sorted(indices), f"drive_plan LF indices not sorted: {indices}"


class TestMidDayRenderSkipMisattribution:
    """A mid-day LF render skip must not shift a later LF's file onto an
    earlier LF's plan/label/bucket (regression guard).

    ``run_day_pipeline`` appends each rendered LF path to a list; when a
    middle LF's render fails it warns + continues, leaving a positional gap.
    The authoritative LF index is carried alongside every rendered path so
    ``phase_bc_day_first`` pairs a path to its plan by identity, never by
    list position — a skipped LF then leaves an explicit gap keyed by index
    instead of collapsing every later LF onto the previous plan/label/bucket.
    """

    def _plan(self, index):
        return RegularLFPlan(
            index=index,
            segments=[],
            montage_peaks=[],
            shorts_target=2,
            region="usa",
            total_duration=900.0,
        )

    def _artifact(self, tmp_path, date, rendered_indices):
        """DayArtifacts for one day whose rendered LFs are ``rendered_indices``.

        A middle index omitted simulates a mid-day render skip: the file is
        absent and the authoritative index travels with each rendered path.
        """
        lf_dir = tmp_path / "out" / "_stages" / f"day_{date}" / "regular_lfs"
        lf_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for idx in rendered_indices:
            p = lf_dir / f"lf_{idx:02d}.mp4"
            p.write_bytes(b"lf")
            paths.append(p)
        art = DayArtifacts(date=date, master_path=None, regular_lf_paths=paths)
        # Carry each rendered path's authoritative LF index (day_pipeline
        # populates this from ``lf_plan.index``). Set post-construction so the
        # test drives the same contract on both the pre- and post-fix code.
        art.regular_lf_indices = list(rendered_indices)
        return art

    def _run_phase_bc(self, tmp_path, plans_by_date, artifact, date):
        state, _ = _make_state_for_day_first(tmp_path, num_days=1)
        state["drive_plan"] = MagicMock()
        orch = DrivePipelineOrchestrator()
        day_batches = [DayBatch(date=date, episodes=state["ingest_plan"].episodes)]
        with (
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.group_episodes_by_day",
                return_value=day_batches,
            ),
            patch(
                "gomoveshift.pipeline.drive_phases.phase_bc_day_first.run_all_days",
                return_value=[artifact],
            ),
            patch(
                "gomoveshift.pipeline.gps_enrichment._partition_lf_plans_by_date",
                return_value=plans_by_date,
            ),
            patch.object(DrivePipelineOrchestrator, "_probe_duration", return_value=900.0),
        ):
            orch._phase_bc_day_first(state)
        return state

    def test_skipped_middle_lf_does_not_shift_later_lf_onto_earlier_plan(self, tmp_path):
        date = "2026-02-20"
        plans = [self._plan(1), self._plan(2), self._plan(3)]
        # LF2's render failed → only LF1 & LF3 landed on disk.
        art = self._artifact(tmp_path, date, rendered_indices=[1, 3])
        state = self._run_phase_bc(tmp_path, {date: plans}, art, date)

        plan_lfs = state["drive_plan"].regular_lfs
        indices = [lf.index for lf in plan_lfs]
        # LF3's file must pair with plan index 3; the unrendered LF2 leaves an
        # explicit gap. Under the positional bug LF3's file collapsed onto
        # LF2's plan, so indices came back [1, 2].
        assert indices == [1, 3], f"expected [1, 3], got {indices} (mis-attribution)"
        assert 2 not in indices

        paths = state["regular_lf_paths"]
        # Phase D count/pairing gate: plan count == path count, aligned 1:1.
        assert len(plan_lfs) == len(paths)
        by_index = {lf.index: paths[pos] for pos, lf in enumerate(plan_lfs)}
        assert "lf_01" in str(by_index[1])
        assert "lf_03" in str(by_index[3]), (
            f"LF3's plan (index 3) must pair with lf_03.mp4, got {by_index[3]}"
        )
        # Bucket: no shorts attributed to the dropped LF2.
        assert set(state["shorts_per_lf"].keys()) == {1, 3}

    def test_all_lfs_rendered_still_pairs_in_order(self, tmp_path):
        date = "2026-02-20"
        plans = [self._plan(1), self._plan(2), self._plan(3)]
        art = self._artifact(tmp_path, date, rendered_indices=[1, 2, 3])
        state = self._run_phase_bc(tmp_path, {date: plans}, art, date)

        plan_lfs = state["drive_plan"].regular_lfs
        assert [lf.index for lf in plan_lfs] == [1, 2, 3]
        paths = state["regular_lf_paths"]
        for pos, lf in enumerate(plan_lfs):
            assert f"lf_{lf.index:02d}" in str(paths[pos])
        assert set(state["shorts_per_lf"].keys()) == {1, 2, 3}

    def test_disk_reconstruction_carries_indices_so_resume_pairs_by_index(self, tmp_path):
        """``_reconstruct_day_artifacts_from_disk`` must populate
        ``regular_lf_indices`` from the ``lf_NN`` filename stems (drive-global
        ``lf_plan.index`` by construction). A resume after a mid-day render
        skip otherwise falls back to POSITIONAL pairing in the consumer and
        pairs lf_05.mp4 with plan 4 — another LF's title/OST/schedule.
        """
        from gomoveshift.pipeline.drive_phases.phase_bc_day_first import (
            _reconstruct_day_artifacts_from_disk,
        )

        date = "2026-02-20"
        day_dir = tmp_path / "out" / "_stages" / f"day_{date}"
        lf_dir = day_dir / "regular_lfs"
        lf_dir.mkdir(parents=True)
        for idx in (3, 5):  # lf_04's render was skipped mid-day
            (lf_dir / f"lf_{idx:02d}.mp4").write_bytes(b"lf")

        art = _reconstruct_day_artifacts_from_disk(date, None, day_dir)
        assert list(art.regular_lf_indices) == [3, 5]

        # Consumer: the carried-index (use_carried) pairing path must
        # activate. Positional fallback would report indices [3, 4].
        plans = [self._plan(3), self._plan(4), self._plan(5)]
        state = self._run_phase_bc(tmp_path, {date: plans}, art, date)
        plan_lfs = state["drive_plan"].regular_lfs
        indices = [lf.index for lf in plan_lfs]
        assert indices == [3, 5], f"expected carried-index pairing [3, 5], got {indices}"
        by_index = {lf.index: state["regular_lf_paths"][pos] for pos, lf in enumerate(plan_lfs)}
        assert "lf_03" in str(by_index[3])
        assert "lf_05" in str(by_index[5])
        assert set(state["shorts_per_lf"].keys()) == {3, 5}

    def test_recovery_callback_reconstruction_carries_indices(self, tmp_path):
        """Second reconstruction site: the crash-recovery checkpoint in
        ``_maybe_fire_recovery_callback`` must carry ``regular_lf_indices``
        parsed from the same ``lf_NN`` stems, so the artifact it records is
        index-pairable on the next resume.
        """
        from gomoveshift.pipeline.day_pipeline import _maybe_fire_recovery_callback

        date = "2026-02-20"
        day_dir = tmp_path / f"day_{date}"
        lf_dir = day_dir / "regular_lfs"
        sh_dir = day_dir / "shorts"
        lf_dir.mkdir(parents=True)
        sh_dir.mkdir(parents=True)
        for idx in (3, 5):  # lf_04's render was skipped mid-day
            (lf_dir / f"lf_{idx:02d}.mp4").write_bytes(b"lf")
            # A current-code burn writes the render-fingerprint sidecar; the
            # recovery pairing requires it (a bare glob would record stale
            # pre-fix renders).
            (lf_dir / f"lf_{idx:02d}.ost.sha256").write_text("deadbeef")
        (sh_dir / "short_01.mp4").write_bytes(b"short")
        # No day_<date>_master.mp4 on disk — the master-survived branch
        # would skip the recovery marker entirely.

        captured = []
        _maybe_fire_recovery_callback(
            date,
            day_dir,
            state={},
            on_day_complete=lambda art, st: captured.append(art),
        )

        assert len(captured) == 1
        art = captured[0]
        assert [p.name for p in art.regular_lf_paths] == ["lf_03.mp4", "lf_05.mp4"]
        assert list(art.regular_lf_indices) == [3, 5]
