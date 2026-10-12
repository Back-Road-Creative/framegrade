"""Phase D resume fingerprint invalidates stale checkpoint state.

Background: ``DrivePipelineOrchestrator._try_resume`` skips ``phase_d``
whenever ``"phase_d"`` appears in ``state["phases_completed"]``. Previously
that branch ignored per-invocation CLI flags affecting Phase D
output (``--lf-day``, ``--publish``, ``--start-date``, ``--shorts-target``,
``--shorts-per-day``, ``--per-lf-location``, ``--location``) and ignored
changes to the Phase D output generators themselves. Result: silent
"Drive pipeline completed in 0.0s" with stale metadata, and the operator
forced to use ``--phase D --force`` to bypass.

These tests pin the fix:

* Same flags + same code version → no spurious re-run (existing skip
  behaviour preserved on a no-op resume).
* Different ``lf_day`` / ``per_lf_location`` / any other Phase D field →
  Phase D re-runs, invalidation log line names which fields changed.
* Bumped ``PHASE_D_CODE_VERSION`` → Phase D re-runs.
* Successful Phase D completion → fingerprint persisted on state.
* Missing fingerprint (state from pre-fingerprint code version) →
  Phase D re-runs with the explicit pre-fingerprint log message.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline._phase_d_fingerprint import (
    compute_phase_d_fingerprint,
    extract_cli_inputs_from_args_and_state,
)
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator
from gomoveshift.pipeline.drive_phases import phase_d as phase_d_mod
from gomoveshift.pipeline.orchestrator_fingerprint import (
    build_phase_d_fingerprint_inputs,
    invalidate_phase_d_if_fingerprint_stale,
    persist_phase_d_fingerprint,
)
from gomoveshift.pipeline.drive_state_model import PipelineState


ORCH_LOGGER = "gomoveshift.pipeline.drive_orchestrator"


def _make_state(
    *,
    source_dir: Path,
    publish_day: str = "wed",
    location: str = "Ireland",
    start_date: str | None = None,
    shorts_target: int | None = None,
    shorts_per_day: int | None = None,
    per_lf_overrides: dict | None = None,
) -> PipelineState:
    state = PipelineState(
        source_dir=source_dir,
        output_base=source_dir.parent,
        publish_day=publish_day,
        location=location,
    )
    if start_date is not None:
        state["start_date"] = start_date
    if shorts_target is not None:
        state["shorts_target"] = shorts_target
    if shorts_per_day is not None:
        state["shorts_per_day"] = shorts_per_day
    state["per_lf_overrides"] = dict(per_lf_overrides or {})
    return state


def _orch_with_lf_day(lf_day: str | None) -> DrivePipelineOrchestrator:
    """Build an orchestrator whose config encodes the given ``--lf-day``."""
    cfg = Config()
    dow = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
    if lf_day is not None:
        cfg.drive.long_form_publish_days = [dow[lf_day]]
    return DrivePipelineOrchestrator(config=cfg)


# ────────────────────────────────────────────────────────────────────────
# Pure fingerprint tests
# ────────────────────────────────────────────────────────────────────────


class TestPhaseDFingerprintPureFunctions:
    """Pin the canonical inputs dict shape + sensitivity to every field."""

    def test_phase_d_fingerprint_includes_all_required_fields(self):
        """Each of ``lf_day``, ``publish``, ``start_date``, ``shorts_target``,
        ``shorts_per_day``, ``per_lf_locations``, ``location``, ``code_version``
        independently changes the fingerprint hash."""
        baseline = {
            "lf_day": [2],
            "publish": "wed",
            "start_date": "2026-05-09",
            "shorts_target": 7,
            "shorts_per_day": 14,
            "per_lf_locations": {1: "Glenfable"},
            "location": "Ireland",
            "lf_time": "14:00",
            "shorts_times": ["15:00", "18:00"],
            "lf_publish_date": "2026-05-09",
            "project_name": "glenfable",
        }
        baseline_hash = compute_phase_d_fingerprint(baseline, code_version=1)

        # Each mutation must produce a different hash than baseline.
        mutations = [
            ("lf_day", [3]),
            ("publish", "thu"),
            ("start_date", "2026-05-10"),
            ("shorts_target", 8),
            ("shorts_per_day", 15),
            ("per_lf_locations", {1: "Ballymock"}),
            ("location", "Wales"),
            ("lf_time", "15:00"),
            ("shorts_times", ["10:00", "16:00"]),
            ("lf_publish_date", "2026-05-16"),
            ("project_name", "kerry"),
        ]
        for field, mutated in mutations:
            mutated_inputs = dict(baseline)
            mutated_inputs[field] = mutated
            mutated_hash = compute_phase_d_fingerprint(mutated_inputs, code_version=1)
            assert mutated_hash != baseline_hash, (
                f"Mutating {field!r} from {baseline[field]!r} to {mutated!r} "
                f"did NOT change the fingerprint — field is missing from canonicalisation."
            )

        # And bumping code_version with everything else equal must also change.
        bumped = compute_phase_d_fingerprint(baseline, code_version=2)
        assert bumped != baseline_hash, "Bumping code_version did NOT change the fingerprint."

    def test_extract_cli_inputs_normalises_per_lf_locations(self):
        """JSON round-trip turns int dict-keys into str; the extractor
        must coerce them back so a no-change resume doesn't spuriously
        invalidate."""
        args = {
            "lf_day": [2],
            "publish": "wed",
            "start_date": None,
            "shorts_target": None,
            "shorts_per_day": None,
            # Mixed key types — both must canonicalise identically.
            "per_lf_locations": {"1": "A", 2: "B"},
            "location": "Ireland",
        }
        inputs = extract_cli_inputs_from_args_and_state(args, project_name="p")
        assert inputs["per_lf_locations"] == {1: "A", 2: "B"}


# ────────────────────────────────────────────────────────────────────────
# Resume-path behaviour
# ────────────────────────────────────────────────────────────────────────


class TestResumeFingerprintBehaviour:
    """Exercise the module-level
    ``invalidate_phase_d_if_fingerprint_stale`` directly. Mirrors the
    pattern in ``test_phase_d_split_skip_upload.py::TestPhaseDReRunTrigger``
    — small, focused predicate test rather than full orchestrator graph.
    """

    def _populate_fingerprint(self, orch: DrivePipelineOrchestrator, state) -> None:
        """Helper: persist a fingerprint matching the current invocation."""
        persist_phase_d_fingerprint(state, config=orch.config)

    def test_resume_with_same_flags_skips_phase_d(self, tmp_path):
        """Same CLI flags + same code version → fingerprint matches →
        phase_d stays in resumed_phases (existing skip preserved)."""
        source = tmp_path / "src"
        source.mkdir()
        orch = _orch_with_lf_day("wed")
        state = _make_state(source_dir=source, publish_day="wed", location="Ireland")
        self._populate_fingerprint(orch, state)
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]

        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        result = invalidate_phase_d_if_fingerprint_stale(state, resumed, stages, config=orch.config)
        assert "phase_d" in result, (
            f"Same-flags resume must NOT invalidate phase_d — got resumed={result}"
        )
        assert "phase_d" in stages
        assert "phase_d" in state["phases_completed"]

    def test_resume_with_different_lf_day_reruns_phase_d(self, tmp_path, caplog):
        """Persisted fingerprint encodes ``--lf-day mon``; current
        invocation passes ``--lf-day tue`` → phase_d re-runs."""
        source = tmp_path / "src"
        source.mkdir()
        # Persisted state was captured under --lf-day mon
        orch_mon = _orch_with_lf_day("mon")
        state = _make_state(source_dir=source, publish_day="wed", location="Ireland")
        self._populate_fingerprint(orch_mon, state)
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]

        # Current invocation: --lf-day tue
        orch_tue = _orch_with_lf_day("tue")
        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        with caplog.at_level(logging.INFO, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_fingerprint_stale(
                state, resumed, stages, config=orch_tue.config
            )

        assert "phase_d" not in result, (
            f"--lf-day change must invalidate phase_d; got resumed={result}"
        )
        assert "phase_d" not in stages
        assert "phase_d" not in state["phases_completed"]
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "lf_day" in joined, (
            f"Invalidation log line must name 'lf_day' as a changed field; got:\n{joined}"
        )

    def test_resume_with_different_per_lf_location_reruns_phase_d(self, tmp_path, caplog):
        """Persisted fingerprint has ``per_lf_overrides={1: 'A'}``; current
        invocation passes ``per_lf_overrides={1: 'B'}`` → phase_d re-runs."""
        source = tmp_path / "src"
        source.mkdir()
        orch = _orch_with_lf_day("wed")
        state_a = _make_state(
            source_dir=source,
            publish_day="wed",
            location="Ireland",
            per_lf_overrides={1: "Glenfable Peninsula, County Kerry"},
        )
        self._populate_fingerprint(orch, state_a)
        # Operator updates the per-LF label and resumes.
        state_a["per_lf_overrides"] = {1: "Ballymock to Glenfable via N86, Kerry"}
        state_a["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]

        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        with caplog.at_level(logging.INFO, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_fingerprint_stale(
                state_a, resumed, stages, config=orch.config
            )

        assert "phase_d" not in result
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "per_lf_locations" in joined, (
            f"Invalidation log must name 'per_lf_locations'; got:\n{joined}"
        )

    def test_resume_with_bumped_code_version_reruns_phase_d(self, tmp_path, monkeypatch, caplog):
        """Monkeypatching ``PHASE_D_CODE_VERSION`` higher than the
        persisted state's value invalidates phase_d on resume."""
        source = tmp_path / "src"
        source.mkdir()
        orch = _orch_with_lf_day("wed")
        state = _make_state(source_dir=source, publish_day="wed", location="Ireland")
        # Persist under code version 1.
        monkeypatch.setattr(phase_d_mod, "PHASE_D_CODE_VERSION", 1)
        self._populate_fingerprint(orch, state)
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]

        # Bump constant — simulates a code merge that changed Phase D outputs.
        monkeypatch.setattr(phase_d_mod, "PHASE_D_CODE_VERSION", 2)
        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        with caplog.at_level(logging.INFO, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_fingerprint_stale(
                state, resumed, stages, config=orch.config
            )

        assert "phase_d" not in result, (
            "Bumping PHASE_D_CODE_VERSION must invalidate the phase_d checkpoint."
        )
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "code_version" in joined or "code version" in joined, (
            f"Log line must mention code version; got:\n{joined}"
        )

    def test_resume_logs_diff_on_invalidation(self, tmp_path, caplog):
        """Log line lists which specific fields changed (caplog assertion)."""
        source = tmp_path / "src"
        source.mkdir()
        orch_a = _orch_with_lf_day("wed")
        state = _make_state(
            source_dir=source,
            publish_day="wed",
            location="Ireland",
            start_date="2026-05-09",
        )
        persist_phase_d_fingerprint(state, config=orch_a.config)
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]

        # Change start_date AND location.
        state["start_date"] = "2026-05-16"
        # The orchestrator reads location off state.location, so we rebuild
        # via a new orchestrator to mutate config-driven inputs cleanly. The
        # ``location`` field on the state is what compute reads.
        object.__setattr__(state, "location", "Wales")

        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        with caplog.at_level(logging.INFO, logger=ORCH_LOGGER):
            invalidate_phase_d_if_fingerprint_stale(state, resumed, stages, config=orch_a.config)
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "start_date" in joined, joined
        assert "location" in joined, joined
        # Old + new hashes must both appear so the log entry self-documents.
        assert "Old hash" in joined and "New hash" in joined, joined

    def test_resume_with_missing_fingerprint_reruns_phase_d(self, tmp_path, caplog):
        """Backward-compat: state lacks ``phase_d_fingerprint`` (a state
        file written before fingerprints existed) → phase_d invalidated with
        the explicit pre-fingerprint log message."""
        source = tmp_path / "src"
        source.mkdir()
        orch = _orch_with_lf_day("wed")
        state = _make_state(source_dir=source)
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]
        # No call to _persist_phase_d_fingerprint — simulates a pre-fingerprint state.
        assert state.get("phase_d_fingerprint") is None

        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        with caplog.at_level(logging.INFO, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_fingerprint_stale(
                state, resumed, stages, config=orch.config
            )
        assert "phase_d" not in result
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "pre-fingerprint" in joined, (
            f"Missing-fingerprint case must use the explicit log message; got:\n{joined}"
        )


# ────────────────────────────────────────────────────────────────────────
# Persistence behaviour
# ────────────────────────────────────────────────────────────────────────


class TestPhaseDFingerprintPersistence:
    """A successful Phase D completion must persist the fingerprint
    so the next resume can detect staleness."""

    def test_phase_d_fingerprint_persisted_after_completion(self, tmp_path):
        source = tmp_path / "src"
        source.mkdir()
        orch = _orch_with_lf_day("wed")
        state = _make_state(
            source_dir=source,
            publish_day="wed",
            location="Ireland",
            start_date="2026-05-09",
            shorts_target=7,
            per_lf_overrides={1: "Glenfable"},
        )
        persist_phase_d_fingerprint(state, config=orch.config)
        assert state.get("phase_d_fingerprint"), (
            "phase_d_fingerprint must be set on state after a successful run"
        )
        inputs = state.get("phase_d_fingerprint_inputs")
        assert isinstance(inputs, dict)
        # Sanity: inputs reflect the current invocation's flags.
        assert inputs["publish"] == "wed"
        assert inputs["location"] == "Ireland"
        assert inputs["start_date"] == "2026-05-09"
        assert inputs["shorts_target"] == 7
        assert inputs["per_lf_locations"] == {1: "Glenfable"}
        assert inputs["lf_day"] == [2]
        assert inputs["code_version"] == phase_d_mod.PHASE_D_CODE_VERSION


# ────────────────────────────────────────────────────────────────────────
# Publish-cadence fields: --lf-time / --shorts-times / --lf-publish-date
# ────────────────────────────────────────────────────────────────────────


class TestResumeFingerprintCadenceFields:
    """A completed-``phase_d`` resume with CORRECTED publish times must
    re-run Phase D. Previously the fingerprint omitted ``--lf-time`` /
    ``--shorts-times`` / ``--lf-publish-date``, so a corrected-time resume
    fingerprint-matched, skipped Phase D, and shipped the OLD schedule —
    the staleness class the fingerprint exists to prevent."""

    def test_resume_with_different_lf_time_reruns_phase_d(self, tmp_path, caplog):
        source = tmp_path / "src"
        source.mkdir()
        orch = _orch_with_lf_day("wed")
        orch.config.drive.long_form_publish_time = "14:00"
        state = _make_state(source_dir=source, publish_day="wed", location="Ireland")
        persist_phase_d_fingerprint(state, config=orch.config)
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]

        # Operator corrects the long-form time and resumes.
        orch.config.drive.long_form_publish_time = "15:00"
        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        with caplog.at_level(logging.INFO, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_fingerprint_stale(
                state, resumed, stages, config=orch.config
            )
        assert "phase_d" not in result, (
            f"--lf-time change must invalidate phase_d; got resumed={result}"
        )
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "lf_time" in joined, joined

    def test_resume_with_different_shorts_times_reruns_phase_d(self, tmp_path, caplog):
        source = tmp_path / "src"
        source.mkdir()
        orch = _orch_with_lf_day("wed")
        orch.config.drive.shorts_publish_times = ["15:00", "18:00"]
        state = _make_state(source_dir=source, publish_day="wed", location="Ireland")
        persist_phase_d_fingerprint(state, config=orch.config)
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]

        # Operator corrects the shorts window and resumes.
        orch.config.drive.shorts_publish_times = ["10:00", "16:00"]
        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        with caplog.at_level(logging.INFO, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_fingerprint_stale(
                state, resumed, stages, config=orch.config
            )
        assert "phase_d" not in result, (
            f"--shorts-times change must invalidate phase_d; got resumed={result}"
        )
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "shorts_times" in joined, joined

    def test_resume_with_different_lf_publish_date_reruns_phase_d(self, tmp_path, caplog):
        source = tmp_path / "src"
        source.mkdir()
        orch = _orch_with_lf_day("wed")
        state = _make_state(source_dir=source, publish_day="wed", location="Ireland")
        persist_phase_d_fingerprint(state, config=orch.config)
        state["phases_completed"] = ["phase_a", "phase_bc_day_first", "phase_d"]

        # Operator anchors the LF to an explicit publish date and resumes.
        state["lf_publish_date"] = "2026-06-01"
        resumed = ["phase_a", "phase_bc_day_first", "phase_d"]
        stages = ["phase_a", "phase_bc_day_first", "phase_d"]
        with caplog.at_level(logging.INFO, logger=ORCH_LOGGER):
            result = invalidate_phase_d_if_fingerprint_stale(
                state, resumed, stages, config=orch.config
            )
        assert "phase_d" not in result, (
            f"--lf-publish-date change must invalidate phase_d; got resumed={result}"
        )
        joined = "\n".join(r.getMessage() for r in caplog.records)
        assert "lf_publish_date" in joined, joined


# ────────────────────────────────────────────────────────────────────────
# Project name is an explicit argument, never sniffed off a state object
# ────────────────────────────────────────────────────────────────────────


class TestFingerprintProjectNameIsExplicit:
    """``PipelineState`` is a dataclass with no ``project_name`` field and no
    ``__getattr__``, so an extractor that sniffs the attribute off its state
    argument silently fingerprints an empty project — a wrong fingerprint,
    i.e. a silently skipped or spurious Phase D re-run. The extractor takes
    the name as a required argument instead, so that shape is unrepresentable.
    """

    ARGS = {
        "lf_day": [2],
        "publish": "wed",
        "start_date": None,
        "shorts_target": None,
        "shorts_per_day": None,
        "per_lf_locations": {1: "Glenfable"},
        "location": "Ireland",
        "lf_time": "14:00",
        "shorts_times": ["15:00", "18:00"],
        "lf_publish_date": None,
    }

    def test_state_object_cannot_supply_the_project_name(self, tmp_path):
        """Handing the extractor a real ``PipelineState`` must fail loudly
        rather than return an empty ``project_name``."""
        source = tmp_path / "Glenfable Day 1"
        source.mkdir()
        state = _make_state(source_dir=source)
        assert not hasattr(state, "project_name"), (
            "PipelineState grew a project_name attribute — revisit this test."
        )
        with pytest.raises(TypeError):
            extract_cli_inputs_from_args_and_state(dict(self.ARGS), state)

    def test_project_name_comes_from_the_explicit_argument(self):
        inputs = extract_cli_inputs_from_args_and_state(
            dict(self.ARGS), project_name="glenfable_day_1"
        )
        assert inputs["project_name"] == "glenfable_day_1"

    def test_production_caller_fingerprint_is_unchanged(self, tmp_path):
        """Pinned pre-refactor hash of the production caller's inputs — proves
        the explicit-argument version fingerprints identically."""
        source = tmp_path / "Dingle Day 1"
        source.mkdir()
        cfg = Config()
        cfg.drive.long_form_publish_days = [2]
        cfg.drive.long_form_publish_time = "14:00"
        cfg.drive.shorts_publish_times = ["15:00", "18:00"]
        state = _make_state(source_dir=source, per_lf_overrides={1: "Dingle"})
        inputs = build_phase_d_fingerprint_inputs(state, config=cfg)
        assert inputs["project_name"] == "dingle_day_1"
        # Pinned production fingerprint: proves this refactor is behaviour-preserving.
        # Not a credential — it is a SHA-256 of the canonical inputs dict, and the
        # pragma keeps CI's detect-secrets high-entropy heuristic from reading it as one.
        expected_fingerprint = "f522fe7445c82d7c31054d8d1172262efe860751648b825a3e2c45bc160b9698"  # pragma: allowlist secret
        assert compute_phase_d_fingerprint(inputs, code_version=1) == expected_fingerprint
