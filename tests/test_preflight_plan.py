"""Tests for ``pipeline/preflight_plan.py``.

Context
-------
Operators commit to multi-hour pipeline runs with no preview of what is
going to happen. Without one, an overnight run can quietly re-encode days
(hours wasted) because the orchestrator gave no signal that master files had
been purged and ``per_day_completed`` was empty.

The plan is therefore surfaced before committing resources.

``preflight_plan.build_plan(state, stages_dir)`` returns a
``PreflightPlan`` dataclass enumerating rows (Phase A, per-day,
Phase D) with status + reason + wall-clock estimate.
``preflight_plan.print_plan(state, stages_dir, out=...)`` renders it to
stdout (or a passed file-like) as a fixed table of rows.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pytest

from gomoveshift.pipeline import preflight_plan
from gomoveshift.pipeline.drive_state_model import PipelineState


# ---------------------------------------------------------------------------
# Helpers — minimal state fixtures
# ---------------------------------------------------------------------------


@dataclass
class _FakeEpisode:
    index: int
    concat_path: Path
    total_duration: float
    date: str
    creation_time: None = None
    gpx_path: None = None


class _FakeIngestPlan:
    def __init__(self, episodes):
        self.episodes = list(episodes)


def _make_state_with_days(
    tmp_path: Path,
    *,
    dates: list[str],
    per_day_completed: dict[str, dict] | None = None,
    drive_scores_exists: bool = False,
    phases_completed: list[str] | None = None,
) -> tuple[PipelineState, Path]:
    """Return a (state, stages_dir) fixture with one episode per date."""
    output_base = tmp_path / "out"
    output_base.mkdir(parents=True, exist_ok=True)
    stages_dir = tmp_path / "_stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    if drive_scores_exists:
        (stages_dir / "drive_scores.npz").write_bytes(b"fake-npz")
    eps = []
    for i, date in enumerate(dates):
        ep_path = tmp_path / f"ep_{i:03d}.mp4"
        ep_path.write_bytes(b"fake")
        eps.append(
            _FakeEpisode(
                index=i,
                concat_path=ep_path,
                total_duration=3600.0,
                date=date,
            )
        )
    state = PipelineState(
        source_dir=tmp_path,
        output_base=output_base,
        region="usa",
        publish_day="wed",
        location="South Carolina",
        fresh=False,
        skip_upload=False,
        skip_csv=False,
        stages_base=stages_dir,
    )
    state["ingest_plan"] = _FakeIngestPlan(eps)
    state["episode_files"] = [ep.concat_path for ep in eps]
    state["episode_dates"] = [ep.date for ep in eps]
    if per_day_completed is not None:
        state["per_day_completed"] = per_day_completed
    if phases_completed is not None:
        state["phases_completed"] = phases_completed
    return state, stages_dir


# ---------------------------------------------------------------------------
# Plan generator
# ---------------------------------------------------------------------------


class TestBuildPlanFreshRun:
    """Without any cached state, every phase + day is RUN."""

    def test_fresh_run_all_rows_run(self, tmp_path):
        state, stages_dir = _make_state_with_days(tmp_path, dates=["2026-02-22", "2026-02-24"])
        plan = preflight_plan.build_plan(state, stages_dir)
        # Phase A + 2 days + Phase D = 4 rows
        assert len(plan.rows) == 4
        # All RUN.
        statuses = [row.status for row in plan.rows]
        assert statuses == ["RUN", "RUN", "RUN", "RUN"], statuses
        labels = [row.label for row in plan.rows]
        assert labels[0] == "Phase A"
        assert labels[1] == "Day 2026-02-22"
        assert labels[2] == "Day 2026-02-24"
        assert labels[3] == "Phase D"


class TestSuperlongWallClock:
    """``wall_clock_breakdown["superlong"]``.

    A ``--superlong`` run produces per-day ``sl_NN.mp4`` parts (opt-in,
    additive; see ``day_pipeline._produce_superlong_parts``) on top of the
    regular LF/shorts encodes, but the preflight ETA breakdown never
    accounted for that GPU time.
    """

    def test_superlong_enabled_contributes_positive_and_totals(self, tmp_path):
        state, stages_dir = _make_state_with_days(tmp_path, dates=["2026-02-22"])
        state.superlong_enabled = True
        plan = preflight_plan.build_plan(state, stages_dir)
        assert plan.wall_clock_breakdown["superlong"] > 0
        assert plan.wall_clock_estimate_seconds == sum(plan.wall_clock_breakdown.values())

    def test_superlong_disabled_contributes_zero(self, tmp_path):
        state, stages_dir = _make_state_with_days(tmp_path, dates=["2026-02-22"])
        state.superlong_enabled = False
        plan = preflight_plan.build_plan(state, stages_dir)
        assert plan.wall_clock_breakdown["superlong"] == 0


class TestBuildPlanSkipReasons:
    """Cached artifacts + recorded completions translate to SKIP rows."""

    def test_plan_prints_skip_reasons_for_cached_phases(self, tmp_path):
        """Phase A SKIP when drive_scores.npz is on disk; day SKIP when
        per_day_completed names it AND its master is still on disk."""
        master_path = tmp_path / "_stages" / "day_2026-02-22" / "day_2026-02-22_master.mp4"
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_bytes(b"fake-master")
        state, stages_dir = _make_state_with_days(
            tmp_path,
            dates=["2026-02-22", "2026-02-24"],
            per_day_completed={
                "2026-02-22": {"master_path": str(master_path), "master_sha": "abc"},
            },
            drive_scores_exists=True,
            phases_completed=["phase_a"],
        )
        plan = preflight_plan.build_plan(state, stages_dir)
        # Row 0: Phase A → SKIP (cache hit + completion recorded)
        phase_a = plan.rows[0]
        assert phase_a.label == "Phase A"
        assert phase_a.status == "SKIP"
        assert "cached" in phase_a.reason.lower() or "drive_scores" in phase_a.reason.lower()
        # Row 1: Day 2026-02-22 → SKIP (per_day_completed + master on disk)
        day22 = plan.rows[1]
        assert day22.label == "Day 2026-02-22"
        assert day22.status == "SKIP"
        assert "per_day_completed" in day22.reason
        assert "master on disk" in day22.reason
        # Row 2: Day 2026-02-24 → RUN (no completion record)
        day24 = plan.rows[2]
        assert day24.status == "RUN"
        assert "fresh" in day24.reason.lower()

    def test_plan_flags_finals_only_reconstruction(self, tmp_path):
        """A day with per_day_completed recorded but master purged AND
        finals + shorts on disk → SKIP (finals-only reconstruction)."""
        day_dir = tmp_path / "_stages" / "day_2026-02-21"
        (day_dir / "regular_lfs").mkdir(parents=True)
        (day_dir / "shorts").mkdir(parents=True)
        (day_dir / "regular_lfs" / "lf_01.mp4").write_bytes(b"lf")
        (day_dir / "shorts" / "short_01.mp4").write_bytes(b"short")
        master_purged = tmp_path / "_stages" / "day_2026-02-21" / "day_2026-02-21_master.mp4"
        # DO NOT create master_purged — master is gone.
        state, stages_dir = _make_state_with_days(
            tmp_path,
            dates=["2026-02-21"],
            per_day_completed={
                "2026-02-21": {"master_path": str(master_purged), "master_sha": "abc"},
            },
            drive_scores_exists=True,
        )
        plan = preflight_plan.build_plan(state, stages_dir)
        day21 = next(r for r in plan.rows if r.label == "Day 2026-02-21")
        assert day21.status == "SKIP"
        assert "finals-only reconstruction" in day21.reason

    def test_plan_flags_partial_master_as_reencode(self, tmp_path):
        """Raw concat on disk but too short (<99% expected) → RUN with
        'partial' / 're-encode' reason so the operator sees why a full
        encode is pending despite prior work."""
        # Build a raw concat under the expected day layout; no master present.
        day_dir = tmp_path / "_stages" / "day_2026-02-22"
        day_dir.mkdir(parents=True)
        raw_path = day_dir / "day_2026-02-22_raw.mp4"
        raw_path.write_bytes(b"partial-raw")
        state, stages_dir = _make_state_with_days(
            tmp_path,
            dates=["2026-02-22"],
            drive_scores_exists=True,
        )

        # Probe returns a partial duration — under 99% of expected.
        # Expected day duration = 3600s (one episode @ 3600s).
        # We return 0.4 * 3600 = 1440s → 40% of expected → below the 99%
        # "re-encode" threshold.
        def fake_probe(path):
            return 0.4 * 3600

        plan = preflight_plan.build_plan(state, stages_dir, probe_duration_fn=fake_probe)
        day22 = next(r for r in plan.rows if r.label == "Day 2026-02-22")
        assert day22.status == "RUN"
        assert "partial" in day22.reason.lower() or "re-encode" in day22.reason.lower()


class TestPlanRenderedOutput:
    """print_plan renders the header, the Phase A / per-day / Phase D rows and the run hints."""

    def test_header_line_present(self, tmp_path):
        state, stages_dir = _make_state_with_days(tmp_path, dates=["2026-02-22"])
        out = io.StringIO()
        preflight_plan.print_plan(state, stages_dir, out=out)
        rendered = out.getvalue()
        assert "EXECUTION PLAN" in rendered
        assert "Phase A" in rendered
        assert "Phase D" in rendered
        assert "Day 2026-02-22" in rendered

    def test_proceed_prompt_rendered(self, tmp_path):
        state, stages_dir = _make_state_with_days(tmp_path, dates=["2026-02-22"])
        out = io.StringIO()
        preflight_plan.print_plan(state, stages_dir, out=out)
        rendered = out.getvalue()
        # Hint text about --dry-run / --force is rendered so operators
        # know how to proceed or exit non-interactively.
        assert "--dry-run" in rendered
        assert "--force" in rendered

    def test_wall_clock_estimate_rendered(self, tmp_path):
        state, stages_dir = _make_state_with_days(tmp_path, dates=["2026-02-22", "2026-02-24"])
        out = io.StringIO()
        preflight_plan.print_plan(state, stages_dir, out=out)
        rendered = out.getvalue()
        assert "Estimated wall-clock" in rendered
        # The breakdown names the master-encode category
        assert "master" in rendered.lower()


class TestPlanEstimatesWallClock:
    """Sanity check on the estimate — not exact minutes, reasonable range."""

    def test_plan_estimates_wall_clock(self, tmp_path):
        state, stages_dir = _make_state_with_days(tmp_path, dates=["2026-02-22", "2026-02-24"])
        plan = preflight_plan.build_plan(state, stages_dir)
        # Two days RUN + phase_a RUN + phase_d RUN.
        # Sanity bounds:
        #   * at least ~2 days of master encode time (conservative 1.5h/day)
        #   * at most some absurdly high bound to catch regressions
        est_sec = plan.wall_clock_estimate_seconds
        assert est_sec >= 2 * 1.5 * 3600, est_sec  # >= 3 hours
        assert est_sec <= 20 * 3600, est_sec  # <= 20 hours (sanity upper bound)
        # Breakdown structure is a dict with known keys.
        bd = plan.wall_clock_breakdown
        assert "master" in bd
        assert "phase_d" in bd

    def test_all_skip_plan_has_near_zero_estimate(self, tmp_path):
        """When nothing needs to run, the estimate should be near zero."""
        master_path = tmp_path / "_stages" / "day_2026-02-22" / "day_2026-02-22_master.mp4"
        master_path.parent.mkdir(parents=True, exist_ok=True)
        master_path.write_bytes(b"fake-master")
        state, stages_dir = _make_state_with_days(
            tmp_path,
            dates=["2026-02-22"],
            per_day_completed={
                "2026-02-22": {"master_path": str(master_path), "master_sha": "abc"},
            },
            drive_scores_exists=True,
            phases_completed=["phase_a", "phase_bc_day_first"],
        )
        plan = preflight_plan.build_plan(state, stages_dir)
        # Only phase_d is left (its drive_schedule.json isn't on disk yet).
        est = plan.wall_clock_estimate_seconds
        assert est < 30 * 60, est  # under 30 minutes


# ---------------------------------------------------------------------------
# Orchestrator integration — dry-run + force + tty behaviour
# ---------------------------------------------------------------------------


class TestOrchestratorDryRun:
    """run(dry_run=True) prints the plan and returns success without doing work."""

    def test_dry_run_exits_zero_after_plan(self, tmp_path, monkeypatch):
        """orchestrator.run(dry_run=True) returns success=True,
        stages_completed=[], and does NOT invoke any phase method."""
        from unittest.mock import MagicMock, patch

        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        source = tmp_path / "src"
        source.mkdir()

        orch = DrivePipelineOrchestrator()
        mock_a = MagicMock()
        mock_bc = MagicMock()
        mock_d = MagicMock()

        captured_plan = io.StringIO()

        # Redirect print_plan output to a buffer so we can assert on it.
        real_print_plan = preflight_plan.print_plan

        def spy_print(state, stages_dir, out=None):  # noqa: ARG001
            real_print_plan(state, stages_dir, out=captured_plan)

        with patch.object(orch, "_phase_a_ingest_and_analyze", mock_a):
            with patch.object(orch, "_phase_bc_day_first", mock_bc):
                with patch.object(orch, "_phase_d_metadata_and_publish", mock_d):
                    with patch(
                        "gomoveshift.pipeline.orchestrator_preflight.print_plan",
                        spy_print,
                    ):
                        result = orch.run(
                            source_dir=source,
                            output_base=tmp_path / "out",
                            region="usa",
                            publish_day="wed",
                            dry_run=True,
                            force=True,
                        )
        assert result.success is True
        assert result.stages_completed == []
        mock_a.assert_not_called()
        mock_bc.assert_not_called()
        mock_d.assert_not_called()
        # Plan was printed.
        assert "EXECUTION PLAN" in captured_plan.getvalue()


def _install_passthrough_gates(monkeypatch):
    """Replace ReviewGate on the orchestrator with a stub whose gate_* methods
    are all no-ops. Lets the orchestrator run past Gate A / B / CD without
    asserting state shape we don't care about in preflight-prompt tests."""

    class _NoopReviewGate:
        def __init__(self, *_, **__):
            pass

        def gate_a(self, *_, **__):
            pass

        def gate_b(self, *_, **__):
            pass

        def gate_cd(self, *_, **__):
            pass

        def gate_preview(self, *_, **__):
            pass

    monkeypatch.setattr("gomoveshift.pipeline.drive_orchestrator.ReviewGate", _NoopReviewGate)


class TestOrchestratorForceFlag:
    """force=True skips the Proceed? prompt."""

    def test_force_skips_prompt(self, tmp_path, monkeypatch):
        """When force=True the orchestrator MUST NOT call input()/readline()."""
        from unittest.mock import MagicMock, patch

        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        source = tmp_path / "src"
        source.mkdir()

        orch = DrivePipelineOrchestrator()

        def _should_not_prompt(*args, **kwargs):
            raise AssertionError("input() called under force=True — the prompt was not skipped")

        monkeypatch.setattr("builtins.input", _should_not_prompt)

        # Mock isatty to return True so the non-tty path doesn't skip the
        # prompt for the wrong reason — we want to prove force itself wins.
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        _install_passthrough_gates(monkeypatch)

        def _bc_side_effect(state):
            state["regular_lf_paths"] = ["/mock/lf_01.mp4"]
            state["shorts_per_lf"] = {1: []}

        with patch.object(orch, "_phase_a_ingest_and_analyze", MagicMock()):
            with patch.object(
                orch,
                "_phase_bc_day_first",
                MagicMock(side_effect=_bc_side_effect),
            ):
                with patch.object(orch, "_phase_d_metadata_and_publish", MagicMock()):
                    # Skip the terminal DoD gate; isolation test.
                    orch.config.drive.skip_validation = True
                    result = orch.run(
                        source_dir=source,
                        output_base=tmp_path / "out",
                        region="usa",
                        publish_day="wed",
                        force=True,
                    )
        assert result.success is True


class TestOrchestratorNonTtyBehavior:
    """When stdin is not a tty, we never prompt — skip + proceed."""

    def test_non_tty_does_not_prompt(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch

        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        source = tmp_path / "src"
        source.mkdir()
        orch = DrivePipelineOrchestrator()

        def _should_not_prompt(*args, **kwargs):
            raise AssertionError("input() called under non-tty — the prompt was not skipped")

        monkeypatch.setattr("builtins.input", _should_not_prompt)
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        _install_passthrough_gates(monkeypatch)

        def _bc_side_effect(state):
            state["regular_lf_paths"] = ["/mock/lf_01.mp4"]
            state["shorts_per_lf"] = {1: []}

        with patch.object(orch, "_phase_a_ingest_and_analyze", MagicMock()):
            with patch.object(
                orch,
                "_phase_bc_day_first",
                MagicMock(side_effect=_bc_side_effect),
            ):
                with patch.object(orch, "_phase_d_metadata_and_publish", MagicMock()):
                    orch.config.drive.skip_validation = True
                    result = orch.run(
                        source_dir=source,
                        output_base=tmp_path / "out",
                        region="usa",
                        publish_day="wed",
                        force=False,  # explicit: even without force, non-tty skips
                    )
        assert result.success is True


class TestOrchestratorTtyPromptDeclined:
    """On a tty without force, the prompt runs; a non-yes answer aborts."""

    def test_prompt_no_answer_aborts(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch

        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        source = tmp_path / "src"
        source.mkdir()
        orch = DrivePipelineOrchestrator()
        mock_a = MagicMock()
        mock_bc = MagicMock()

        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        # Answer "n" → abort, phase methods never called.
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "n")

        with patch.object(orch, "_phase_a_ingest_and_analyze", mock_a):
            with patch.object(orch, "_phase_bc_day_first", mock_bc):
                result = orch.run(
                    source_dir=source,
                    output_base=tmp_path / "out",
                    region="usa",
                    publish_day="wed",
                    force=False,
                )
        assert result.success is True  # graceful exit, not a failure
        assert result.stages_completed == []
        mock_a.assert_not_called()
        mock_bc.assert_not_called()


# ---------------------------------------------------------------------------
# Preflight ordering
#
# The sidecar merge in drive_orchestrator.run may force
# skip_upload=True when youtube_uploads.json already has entries and the
# caller has not explicitly opted into uploads. print_plan reads
# state["skip_upload"] to pick the Phase D wall-clock bucket. If print_plan
# runs BEFORE the merge, the operator sees a plan that claims "upload"
# when the actual run will skip-upload. Moving print_plan AFTER the merge
# restores truth: --dry-run can audit the real decision; --force sees the
# real plan.
# ---------------------------------------------------------------------------


class TestPreflightPlanFiresAfterSidecarMerge:
    """print_plan must be invoked AFTER the sidecar-forced skip_upload
    adjustment, not before. Otherwise --dry-run lies about the run it
    would actually execute."""

    def test_dry_run_plan_reflects_sidecar_forced_skip_upload(self, tmp_path, monkeypatch):
        """When youtube_uploads.json has entries and the caller passes
        neither --skip-upload nor --youtube-upload, the orchestrator
        defensively flips skip_upload to True. print_plan must see the
        flipped value so the estimate + the log line match reality."""
        from unittest.mock import patch

        from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator

        source = tmp_path / "src"
        source.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        project_dir = out_dir / "src"
        project_dir.mkdir()

        # Simulate a sidecar with a prior upload entry so load_sidecar
        # returns something and the merge path fires (the merge flips
        # state["skip_upload"] to True).
        sidecar_path = project_dir / "youtube_uploads.json"
        sidecar_path.write_text('{"1": {"video_id": "abc123", "uploaded_at": "2026-04-21"}}')

        # Capture the value of state.get("skip_upload") as seen BY
        # print_plan at the moment it runs. This must be
        # True (the sidecar merge forced it), not the caller default False
        # that print_plan would see if it ran first.
        captured = {"skip_upload_at_print": "SENTINEL"}

        def spy_print(state, stages_dir, out=None):  # noqa: ARG001
            captured["skip_upload_at_print"] = state.get("skip_upload")

        orch = DrivePipelineOrchestrator()

        with patch(
            "gomoveshift.pipeline.orchestrator_preflight.print_plan",
            spy_print,
        ):
            result = orch.run(
                source_dir=source,
                output_base=out_dir,
                region="usa",
                publish_day="wed",
                # Argparse sentinel: ``None`` means the operator did
                # NOT pass the flag — exactly the "no flag passed" case
                # this test models. Passing explicit ``False`` would now
                # (correctly: CLI overrides state) suppress
                # the sidecar auto-default.
                skip_upload=None,
                youtube_upload=None,
                dry_run=True,
                force=True,
            )

        assert result.success is True
        assert captured["skip_upload_at_print"] is True, (
            "print_plan saw skip_upload=%r, but the sidecar merge should "
            "have forced it to True before the plan rendered. Ordering "
            "regression: print_plan is still firing BEFORE the sidecar "
            "merge." % (captured["skip_upload_at_print"],)
        )


# ---------------------------------------------------------------------------
# Calibrated ETA — a measurement beats the fixed model
# ---------------------------------------------------------------------------


def _calibration_record(**over):
    from gomoveshift.pipeline.calibration import CalibrationRecord
    from gomoveshift.pipeline.day_master_encoder import MASTER_ENCODE_ALGO_VERSION

    base = dict(
        measured_at="2026-10-07T12:00:00+00:00",
        chain_version=MASTER_ENCODE_ALGO_VERSION,
        encoder="hevc_nvenc",
        width=3840,
        height=2160,
        source_fps=30.0,
        slice_seconds=15.0,
        wall_seconds=22.5,
        blur=True,
        ffmpeg_version="ffmpeg version 6.1",
        cpu_count=16,
    )
    base.update(over)
    return CalibrationRecord(**base)


@pytest.fixture(autouse=False)
def _isolated_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "gms-cache"))


class TestCalibratedEstimate:
    def test_measurement_replaces_fixed_factors(self):
        rec = _calibration_record(slice_seconds=15.0, wall_seconds=22.5)
        assert rec.realtime_factor == 1.5
        got = preflight_plan.estimate_master_encode_seconds(3600, "4k", "p7", 2, calibration=rec)
        assert got == 5400.0

    def test_without_calibration_model_unchanged(self):
        got = preflight_plan.estimate_master_encode_seconds(3600, "4k", "p7", 2)
        assert got == pytest.approx(3600 * 1.0 * 1.10 * 1.30)

    def test_build_plan_uses_record(self, tmp_path):
        state, stages_dir = _make_state_with_days(tmp_path, dates=["2026-02-22"])
        modelled = preflight_plan.build_plan(state, stages_dir, calibration=None)
        rec = _calibration_record(wall_seconds=150.0)  # 10x realtime
        measured = preflight_plan.build_plan(state, stages_dir, calibration=rec)
        assert measured.calibration is rec
        assert modelled.calibration is None
        assert measured.wall_clock_breakdown["master"] > modelled.wall_clock_breakdown["master"]

    def test_build_plan_loads_record_when_kwarg_omitted(
        self, tmp_path, monkeypatch, _isolated_calibration
    ):
        from gomoveshift.pipeline.calibration import save_calibration

        save_calibration(_calibration_record())
        state, stages_dir = _make_state_with_days(tmp_path, dates=["2026-02-22"])
        assert preflight_plan.build_plan(state, stages_dir).calibration is not None


class TestEtaBasisLine:
    def test_measured_line(self):
        plan = preflight_plan.PreflightPlan(calibration=_calibration_record())
        lines = preflight_plan._render_plan_lines(plan)
        basis = [ln for ln in lines if "ETA basis" in ln]
        assert basis == [
            "  ETA basis: measured on this machine — 20.0 fps "
            "(hevc_nvenc, 3840x2160, blur on, 2026-10-07)"
        ]
        assert lines.index(basis[0]) + 1 == next(
            i for i, ln in enumerate(lines) if "Estimated wall-clock" in ln
        )

    def test_reference_model_line(self):
        lines = preflight_plan._render_plan_lines(preflight_plan.PreflightPlan())
        basis = [ln for ln in lines if "ETA basis" in ln]
        assert len(basis) == 1
        assert "built-in reference model" in basis[0]
        assert "gomoveshift doctor --calibrate <clip>" in basis[0]


class TestOrchestratorCalibrationWiring:
    def _config(self, exts=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            drive=SimpleNamespace(
                video_extensions=exts or [".mp4"], blur_enabled=True, skip_blur=False
            )
        )

    def test_dry_run_only_loads_never_measures(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import calibration, orchestrator_preflight

        monkeypatch.setattr(calibration, "load_calibration", lambda *a: None)
        monkeypatch.setattr(
            calibration,
            "calibrate_master_chain",
            lambda *a, **k: pytest.fail("dry run must not encode"),
        )
        (tmp_path / "a.mp4").write_bytes(b"x")
        got = orchestrator_preflight._calibration_for_plan(
            dry_run=True, source_dir=tmp_path, stages_dir=tmp_path / "st", config=self._config()
        )
        assert got is None

    def test_real_run_measures_first_sorted_video_with_blur(self, tmp_path, monkeypatch):
        from gomoveshift.pipeline import calibration, orchestrator_preflight

        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", str(tmp_path / "cache"))
        for name in ("b.MP4", "a.mp4", "notes.txt"):
            (tmp_path / name).write_bytes(b"x")
        rec = _calibration_record()
        seen = {}

        def fake(config, source, **kw):
            seen.update(source=source, **kw)
            return rec

        monkeypatch.setattr(calibration, "calibrate_master_chain", fake)
        got = orchestrator_preflight._calibration_for_plan(
            dry_run=False, source_dir=tmp_path, stages_dir=tmp_path / "st", config=self._config()
        )
        assert got is rec
        assert seen["source"].name == "a.mp4" and seen["blur"] is True
        assert seen["workdir"] == tmp_path / "st" / "_calibration"
        assert calibration.load_calibration() == rec
