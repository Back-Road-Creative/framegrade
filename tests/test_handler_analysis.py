"""Handler-level tests for cmd_analyze, cmd_process, cmd_scan, cmd_find_stops.

These handlers all live in ``gomoveshift/cli/handlers/analysis.py``.

Each handler is exercised against three failure modes:
1. Happy path — orchestrator mocked to return success, exit 0.
2. Invalid args — missing input file (or directory) — exit 1.
3. Underlying-pipeline exception — handler's ``except Exception`` branch maps
   to exit 1 with logged error.

Handlers are pure call-graph dispatchers: every external dependency
(``AnalysisPipeline``, ``VideoAssembler``, ``MotionAnalyzer``, resume
helpers) is patched at the handler-module namespace where it's imported.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.cli.handlers import analysis as analysis_handler


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    p = tmp_path / "drive.mp4"
    p.write_bytes(b"\x00")
    return p


@pytest.fixture
def fake_config() -> MagicMock:
    cfg = MagicMock()
    cfg.dead_spots.motion_threshold = 0.5
    cfg.dead_spots.min_duration_seconds = 3.0
    return cfg


# ---------------------------------------------------------------------------
# cmd_analyze
# ---------------------------------------------------------------------------


class TestCmdAnalyze:
    def _args(self, video: Path, output: Path | None = None) -> SimpleNamespace:
        return SimpleNamespace(video=video, output=output, no_frames=True)

    def test_happy_path_returns_zero(self, video_file, fake_config, tmp_path):
        out = tmp_path / "out.analysis.json"
        args = self._args(video_file, out)

        fake_analysis = MagicMock()
        fake_analysis.processing_stats = {"frames_processed": 100}
        fake_analysis.locations = []
        fake_analysis.dead_spots = []
        fake_analysis.suggested_cuts = []

        with patch.object(analysis_handler, "AnalysisPipeline") as mock_pipe:
            mock_pipe.return_value.analyze.return_value = fake_analysis
            rc = analysis_handler.cmd_analyze(args, fake_config)

        assert rc == 0
        fake_analysis.save.assert_called_once_with(out)

    def test_missing_video_returns_one(self, tmp_path, fake_config, capsys):
        missing = tmp_path / "nope.mp4"
        args = self._args(missing, None)
        rc = analysis_handler.cmd_analyze(args, fake_config)
        assert rc == 1
        err = capsys.readouterr().err
        assert "Video file not found" in err

    def test_pipeline_exception_returns_one(self, video_file, fake_config, capsys):
        args = self._args(video_file, None)
        with patch.object(analysis_handler, "AnalysisPipeline") as mock_pipe:
            mock_pipe.return_value.analyze.side_effect = RuntimeError("boom")
            rc = analysis_handler.cmd_analyze(args, fake_config)
        assert rc == 1
        err = capsys.readouterr().err
        assert "Error during analysis" in err
        assert "boom" in err


# ---------------------------------------------------------------------------
# cmd_process
# ---------------------------------------------------------------------------


class TestCmdProcess:
    def _args(self, video: Path, analysis: Path, output_dir: Path | None = None) -> SimpleNamespace:
        return SimpleNamespace(video=video, analysis=analysis, output_dir=output_dir)

    def test_happy_path_returns_zero(self, video_file, fake_config, tmp_path):
        analysis_path = tmp_path / "a.json"
        analysis_path.write_text("{}")
        out_dir = tmp_path / "outdir"
        args = self._args(video_file, analysis_path, out_dir)

        fake_plan = MagicMock()
        fake_plan.segments_to_keep = []
        fake_plan.cut_points = []
        fake_plan.output_paths = []

        with (
            patch("gomoveshift.models.VideoAnalysis.load") as mock_load,
            patch.object(analysis_handler, "VideoAssembler") as mock_asm,
        ):
            mock_load.return_value = MagicMock()
            mock_asm.return_value.create_assembly_plan.return_value = fake_plan
            mock_asm.return_value.execute.return_value = []
            rc = analysis_handler.cmd_process(args, fake_config)

        assert rc == 0

    def test_happy_path_marks_outputs_ungated(self, video_file, fake_config, tmp_path, capsys):
        """The manual `process` path has no stop/audio/blur gate, so it must
        print an EXPLICIT ungated marker."""
        analysis_path = tmp_path / "a.json"
        analysis_path.write_text("{}")
        args = self._args(video_file, analysis_path, tmp_path / "outdir")

        fake_plan = MagicMock()
        fake_plan.segments_to_keep = []
        fake_plan.cut_points = []
        fake_plan.output_paths = []

        with (
            patch("gomoveshift.models.VideoAnalysis.load") as mock_load,
            patch.object(analysis_handler, "VideoAssembler") as mock_asm,
        ):
            mock_load.return_value = MagicMock()
            mock_asm.return_value.create_assembly_plan.return_value = fake_plan
            mock_asm.return_value.execute.return_value = []
            rc = analysis_handler.cmd_process(args, fake_config)

        assert rc == 0
        assert "UNGATED" in capsys.readouterr().err

    def test_missing_video_returns_one(self, tmp_path, fake_config, capsys):
        analysis_path = tmp_path / "a.json"
        analysis_path.write_text("{}")
        args = self._args(tmp_path / "nope.mp4", analysis_path)
        rc = analysis_handler.cmd_process(args, fake_config)
        assert rc == 1
        assert "Video file not found" in capsys.readouterr().err

    def test_missing_analysis_returns_one(self, video_file, fake_config, capsys):
        args = self._args(video_file, video_file.parent / "missing.json")
        rc = analysis_handler.cmd_process(args, fake_config)
        assert rc == 1
        assert "Analysis file not found" in capsys.readouterr().err

    def test_assembler_exception_returns_one(self, video_file, fake_config, tmp_path, capsys):
        analysis_path = tmp_path / "a.json"
        analysis_path.write_text("{}")
        args = self._args(video_file, analysis_path, tmp_path / "outdir")

        fake_plan = MagicMock()
        fake_plan.segments_to_keep = []
        fake_plan.cut_points = []
        fake_plan.output_paths = []

        with (
            patch("gomoveshift.models.VideoAnalysis.load") as mock_load,
            patch.object(analysis_handler, "VideoAssembler") as mock_asm,
        ):
            mock_load.return_value = MagicMock()
            mock_asm.return_value.create_assembly_plan.return_value = fake_plan
            mock_asm.return_value.execute.side_effect = RuntimeError("encode-fail")
            rc = analysis_handler.cmd_process(args, fake_config)

        assert rc == 1
        err = capsys.readouterr().err
        assert "Error during processing" in err
        assert "encode-fail" in err


# ---------------------------------------------------------------------------
# cmd_scan
# ---------------------------------------------------------------------------


class TestCmdScan:
    def _args(self, video: Path) -> SimpleNamespace:
        return SimpleNamespace(video=video)

    def test_happy_path_returns_zero(self, video_file, fake_config, capsys):
        args = self._args(video_file)
        with patch.object(analysis_handler, "AnalysisPipeline") as mock_pipe:
            mock_pipe.return_value.quick_scan.return_value = {
                "duration": 120.0,
                "width": 1920,
                "height": 1080,
                "fps": 30.0,
                "total_frames": 3600,
            }
            rc = analysis_handler.cmd_scan(args, fake_config)
        assert rc == 0
        out = capsys.readouterr().out
        assert "1920x1080" in out
        assert "Duration" in out

    def test_missing_video_returns_one(self, tmp_path, fake_config, capsys):
        args = self._args(tmp_path / "missing.mp4")
        rc = analysis_handler.cmd_scan(args, fake_config)
        assert rc == 1
        assert "Video file not found" in capsys.readouterr().err

    def test_quick_scan_exception_propagates(self, video_file, fake_config):
        """``cmd_scan`` does NOT wrap ``quick_scan`` in try/except — exceptions
        propagate up to the CLI's top-level handler. Pin that contract so a
        future refactor that adds a swallow-and-return-1 path doesn't go
        unnoticed."""
        args = self._args(video_file)
        with patch.object(analysis_handler, "AnalysisPipeline") as mock_pipe:
            mock_pipe.return_value.quick_scan.side_effect = RuntimeError("probe-fail")
            with pytest.raises(RuntimeError, match="probe-fail"):
                analysis_handler.cmd_scan(args, fake_config)


# ---------------------------------------------------------------------------
# cmd_find_stops
# ---------------------------------------------------------------------------


class TestCmdFindStops:
    def _args(
        self, input_dir: Path, output: Path | None = None, fresh: bool = False
    ) -> SimpleNamespace:
        return SimpleNamespace(input_dir=input_dir, output=output, fresh=fresh)

    def test_missing_directory_returns_one(self, tmp_path, fake_config, capsys):
        args = self._args(tmp_path / "absent")
        rc = analysis_handler.cmd_find_stops(args, fake_config)
        assert rc == 1
        assert "Directory not found" in capsys.readouterr().err

    def test_empty_directory_returns_zero(self, tmp_path, fake_config, capsys):
        # Empty directory: short-circuit, no analyzer needed.
        args = self._args(tmp_path)
        rc = analysis_handler.cmd_find_stops(args, fake_config)
        assert rc == 0
        assert "No video files found" in capsys.readouterr().out

    def test_happy_path_writes_report(self, tmp_path, fake_config):
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 1024)
        args = self._args(tmp_path)

        with (
            patch("gomoveshift.resume.get_videos_to_process") as mock_to_proc,
            patch("gomoveshift.resume.print_resume_status"),
            patch("gomoveshift.resume.merge_results") as mock_merge,
            patch("gomoveshift.resume.get_checkpoint_state") as mock_state,
            patch("gomoveshift.analysis.motion.MotionAnalyzer") as mock_analyzer,
        ):
            mock_to_proc.return_value = [video]
            mock_state.return_value = {"completed_videos": []}
            mock_analyzer.return_value.analyze.return_value = []
            mock_merge.return_value = {
                "videos": {"clip.mp4": {"path": str(video), "name": "clip.mp4", "stops": []}}
            }
            rc = analysis_handler.cmd_find_stops(args, fake_config)

        assert rc == 0
        report = tmp_path / "stops_report.json"
        assert report.exists()

    def test_analyzer_exception_captured_per_video(self, tmp_path, fake_config, capsys):
        """``cmd_find_stops`` catches per-video exceptions and records them in
        the results dict instead of aborting the whole run — the function
        still returns 0."""
        video = tmp_path / "clip.mp4"
        video.write_bytes(b"\x00" * 1024)
        args = self._args(tmp_path)

        with (
            patch("gomoveshift.resume.get_videos_to_process") as mock_to_proc,
            patch("gomoveshift.resume.print_resume_status"),
            patch("gomoveshift.resume.merge_results") as mock_merge,
            patch("gomoveshift.resume.get_checkpoint_state") as mock_state,
            patch("gomoveshift.analysis.motion.MotionAnalyzer") as mock_analyzer,
        ):
            mock_to_proc.return_value = [video]
            mock_state.return_value = {"completed_videos": []}
            mock_analyzer.return_value.analyze.side_effect = RuntimeError("probe-bust")
            mock_merge.return_value = {"videos": {"clip.mp4": {"error": "probe-bust"}}}
            rc = analysis_handler.cmd_find_stops(args, fake_config)

        assert rc == 0
        out = capsys.readouterr().out
        assert "ERROR: probe-bust" in out
