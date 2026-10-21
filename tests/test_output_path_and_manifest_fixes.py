"""Tests for three orchestrator and CLI fixes: double-nested output path, CLI
dict plan, stale shorts manifest reconciliation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.drive_orchestrator import DrivePipelineOrchestrator


# ---------------------------------------------------------------------------
# Double-nested output path
# ---------------------------------------------------------------------------


class TestDoubleNestedOutputPath:
    """When --output-dir already ends with the project name, the orchestrator
    must NOT append it again (e.g. /out/src01/src01)."""

    def _run_orchestrator(self, tmp_path, source_name, output_dir, monkeypatch):
        """Run the orchestrator with stubbed phases and return the state's output_base."""
        from gomoveshift.pipeline import drive_orchestrator
        from gomoveshift.pipeline.review_gate import ReviewGate

        monkeypatch.setattr(ReviewGate, "gate_a", lambda *a, **kw: None)
        monkeypatch.setattr(ReviewGate, "gate_b", lambda *a, **kw: None)
        monkeypatch.setattr(ReviewGate, "gate_cd", lambda *a, **kw: None)

        source = tmp_path / source_name
        source.mkdir()
        output = tmp_path / output_dir
        output.mkdir(parents=True, exist_ok=True)

        orch = DrivePipelineOrchestrator()
        orch.config.drive.no_review = True
        # Isolation test: stub a passing DoD validator. --skip-validation is
        # refused on a run that uploads (GMS-17c).
        monkeypatch.setattr(
            "gomoveshift.pipeline.drive_orchestrator.run_validation",
            lambda **_kw: MagicMock(passed_count=13, failed_count=0, checks=[]),
        )

        # Stub all phases
        captured_state = {}

        def capture_phase_a(state):
            captured_state.update(state)
            state["phases_completed"] = ["phase_a"]

        def noop(state, **kw):
            pass

        orch._phase_a_ingest_and_analyze = capture_phase_a
        orch._phase_b_process_files = noop
        orch._phase_c_assemble = noop
        orch._phase_d_metadata_and_publish = noop

        result = orch.run(
            source_dir=source,
            output_base=output,
            region="usa",
            publish_day="wed",
        )
        return result, captured_state

    def test_no_double_nesting_when_output_matches_project(self, tmp_path, monkeypatch):
        """output_dir=.../src01 + source=SRC01 → project_dir is output_dir itself."""
        result, state = self._run_orchestrator(tmp_path, "SRC01", "output/src01", monkeypatch)
        # output_dir should be .../output/src01, NOT .../output/src01/src01
        assert result.output_dir == tmp_path / "output" / "src01"
        assert not (tmp_path / "output" / "src01" / "src01").exists()

    def test_normal_nesting_when_output_differs(self, tmp_path, monkeypatch):
        """output_dir=.../output + source=SRC01 → project_dir is output/src01."""
        result, state = self._run_orchestrator(tmp_path, "SRC01", "output", monkeypatch)
        assert result.output_dir == tmp_path / "output" / "src01"


# ---------------------------------------------------------------------------
# CLI AttributeError when drive_plan is a dict
# ---------------------------------------------------------------------------


class TestCLIDrivePlanDict:
    """``cmd_drive`` crashes with AttributeError when result.drive_plan is a dict
    (from checkpoint state) instead of a DrivePlan object."""

    def test_cli_handles_dict_plan(self, capsys, tmp_path):
        """cmd_drive must not crash when drive_plan is a dict."""
        from gomoveshift.cli.handlers.drive import cmd_drive

        dict_plan = {
            "regular_lfs": [{"segments": []}],
            "highlight_ratio": 0.42,
            "source_files": [],
            "processed_files": [],
        }
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.drive_plan = dict_plan
        mock_result.errors = []

        src = tmp_path / "fake"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        mock_args = MagicMock()
        mock_args.no_tmux = True
        mock_args.directory = src
        mock_args.region = "usa"
        mock_args.publish = "tue"
        mock_args.location = None
        mock_args.start_date = None
        mock_args.shorts_per_day = None
        mock_args.shorts_target = None
        mock_args.shorts_gap = None
        mock_args.per_day = False
        mock_args.day_first = False
        mock_args.preview = False
        mock_args.no_review = True
        mock_args.no_audio = False
        mock_args.skip_validation = True
        mock_args.phase = None
        mock_args.output_dir = out
        mock_args.fresh = False
        mock_args.skip_upload = True
        mock_args.skip_csv = True
        mock_args.youtube_upload = False

        config = Config()

        with patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator") as MockOrch:
            MockOrch.return_value.run.return_value = mock_result
            ret = cmd_drive(mock_args, config)

        assert ret == 0
        captured = capsys.readouterr()
        assert "Regular LFs:" in captured.out

    def test_cli_handles_dataclass_plan(self, capsys, tmp_path):
        """cmd_drive works with a proper DrivePlan object too."""
        from gomoveshift.cli.handlers.drive import cmd_drive
        from gomoveshift.models import DrivePlan

        plan = DrivePlan(
            source_files=[],
            processed_files=[],
            regular_lfs=[MagicMock()],
            score_threshold=0.5,
            total_source_duration=1000.0,
            total_highlight_duration=420.0,
        )
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.drive_plan = plan
        mock_result.errors = []

        src = tmp_path / "fake"
        src.mkdir()
        out = tmp_path / "out"
        out.mkdir()
        mock_args = MagicMock()
        mock_args.no_tmux = True
        mock_args.directory = src
        mock_args.region = "usa"
        mock_args.publish = "tue"
        mock_args.location = None
        mock_args.start_date = None
        mock_args.shorts_per_day = None
        mock_args.shorts_target = None
        mock_args.shorts_gap = None
        mock_args.per_day = False
        mock_args.day_first = False
        mock_args.preview = False
        mock_args.no_review = True
        mock_args.no_audio = False
        mock_args.skip_validation = True
        mock_args.phase = None
        mock_args.output_dir = out
        mock_args.fresh = False
        mock_args.skip_upload = True
        mock_args.skip_csv = True
        mock_args.youtube_upload = False

        config = Config()

        with patch("gomoveshift.pipeline.drive_orchestrator.DrivePipelineOrchestrator") as MockOrch:
            MockOrch.return_value.run.return_value = mock_result
            ret = cmd_drive(mock_args, config)

        assert ret == 0
        captured = capsys.readouterr()
        assert "Regular LFs: 1" in captured.out
        assert "Highlight ratio: 42%" in captured.out


# ---------------------------------------------------------------------------
# Stale shorts_rendered.json manifest
# ---------------------------------------------------------------------------


class TestShortsManifestReconciliation:
    """After rendering, the manifest must agree with actual mp4 files on disk."""

    def test_manifest_reconciles_with_extra_files(self, tmp_path):
        """If mp4 files exist from a prior run beyond what the current analysis
        found, the saved manifest's num_rendered must still match actual count."""
        from gomoveshift.pipeline.shorts_extractor import ShortsExtractionResult

        # Simulate: manifest says 2 rendered, but 5 mp4 files exist
        output_dir = tmp_path / "shorts" / "lf_01"
        output_dir.mkdir(parents=True)
        for i in range(1, 6):
            (output_dir / f"short_{i:02d}_rank{i}.mp4").write_bytes(b"\x00" * 200_000)

        result = ShortsExtractionResult(
            source_video="/fake/lf_01.mp4",
            output_dir=str(output_dir),
            num_requested=14,
            num_rendered=2,
            shorts=[MagicMock(to_dict=lambda: {}) for _ in range(2)],
            errors=[],
            processing_time=10.0,
        )
        result_path = output_dir / "shorts_rendered.json"
        result.save(result_path)

        saved = json.loads(result_path.read_text())
        assert saved["num_rendered"] == 5, (
            "num_rendered must match actual short_*.mp4 count on disk"
        )

    def test_manifest_matches_when_correct(self, tmp_path):
        """When num_rendered already matches file count, no change needed."""
        from gomoveshift.pipeline.shorts_extractor import ShortsExtractionResult

        output_dir = tmp_path / "shorts" / "lf_02"
        output_dir.mkdir(parents=True)
        for i in range(1, 4):
            (output_dir / f"short_{i:02d}_rank{i}.mp4").write_bytes(b"\x00" * 200_000)

        result = ShortsExtractionResult(
            source_video="/fake/lf_02.mp4",
            output_dir=str(output_dir),
            num_requested=14,
            num_rendered=3,
            shorts=[MagicMock(to_dict=lambda: {}) for _ in range(3)],
            errors=[],
            processing_time=5.0,
        )
        result_path = output_dir / "shorts_rendered.json"
        result.save(result_path)

        saved = json.loads(result_path.read_text())
        assert saved["num_rendered"] == 3
