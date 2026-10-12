"""
TDD tests for pipeline resilience features:
1. Motion analysis caching in stop processor
2. Pipeline state persistence (stage resume)
3. tmux launcher
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gomoveshift.config import get_config
from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, RunOptions
from gomoveshift.pipeline.stop_processor import StopProcessor

# Resume tests drive the full orchestrator on a dummy video; blur is on by
# default, so opt out of the real detector (see conftest ``disable_blur``).
pytestmark = pytest.mark.usefixtures("disable_blur")


class TestEnsureCacheDir:
    """Tests for _ensure_cache_dir permission behaviour."""

    def test_ensure_cache_dir_uses_mode_775(self, tmp_path):
        """_ensure_cache_dir should create the directory with mode 0o775 (group-writable)."""
        from gomoveshift.pipeline.stop_processor import _ensure_cache_dir

        target = tmp_path / "motion_cache"
        result = _ensure_cache_dir(target)

        assert result == target
        # stat returns the full mode including file-type bits; mask with 0o777 for perms only
        actual_mode = result.stat().st_mode & 0o777
        assert actual_mode == 0o775, (
            f"Expected mode 0o775 but got {oct(actual_mode)}. "
            "motion_cache must be group-writable so every user in the group can write."
        )

    def test_ensure_cache_dir_does_not_use_world_writable(self, tmp_path):
        """_ensure_cache_dir must not create a world-writable directory (no 0o777)."""
        from gomoveshift.pipeline.stop_processor import _ensure_cache_dir

        target = tmp_path / "motion_cache_sec"
        result = _ensure_cache_dir(target)

        actual_mode = result.stat().st_mode & 0o777
        # World-writable bit must NOT be set
        assert not (actual_mode & 0o002), (
            f"Directory mode {oct(actual_mode)} is world-writable — security risk."
        )

    def test_ensure_cache_dir_creates_parents(self, tmp_path):
        """_ensure_cache_dir should create intermediate parent directories."""
        from gomoveshift.pipeline.stop_processor import _ensure_cache_dir

        deep = tmp_path / "level1" / "level2" / "motion_cache"
        result = _ensure_cache_dir(deep)

        assert result.exists()
        assert result.is_dir()


class TestMotionCaching:
    """Tests for motion analysis caching in StopProcessor."""

    @pytest.fixture
    def temp_video(self, tmp_path):
        """Create a small test video."""
        video_path = tmp_path / "test_video.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=640x360:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            pytest.skip("FFmpeg not available")
        return video_path

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache = tmp_path / "motion_cache"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def test_compute_file_hash_for_cache_key(self, temp_video):
        """Should compute stable file hash from first+last 64KB + file size."""
        from gomoveshift.pipeline.stop_processor import compute_file_hash

        hash1 = compute_file_hash(temp_video)
        hash2 = compute_file_hash(temp_video)

        # Same file = same hash
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex string

    def test_different_files_have_different_hashes(self, tmp_path):
        """Different video files should have different hashes."""
        from gomoveshift.pipeline.stop_processor import compute_file_hash

        # Create two different videos (different durations to ensure different content)
        video1 = tmp_path / "video1.mp4"
        video2 = tmp_path / "video2.mp4"

        cmd1 = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=1:size=640x360:rate=30",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video1),
        ]
        subprocess.run(cmd1, capture_output=True, check=True)

        cmd2 = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=640x360:rate=30",  # Different duration
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video2),
        ]
        subprocess.run(cmd2, capture_output=True, check=True)

        hash1 = compute_file_hash(video1)
        hash2 = compute_file_hash(video2)

        # Different files = different hashes
        assert hash1 != hash2

    def test_get_cache_path_uses_file_hash(self, temp_video, cache_dir):
        """Cache path should use file hash as key."""
        from gomoveshift.pipeline.stop_processor import compute_file_hash, get_motion_cache_path

        file_hash = compute_file_hash(temp_video)
        cache_path = get_motion_cache_path(temp_video, cache_dir)

        # Cache path should include the hash
        assert file_hash in str(cache_path)
        assert cache_path.suffix == ".npz"
        assert cache_path.parent == cache_dir

    def test_detect_stops_saves_cache_on_first_run(self, temp_video, cache_dir):
        """First run should analyze and save cache."""
        config = get_config()
        processor = StopProcessor(config, cache_dir=cache_dir)

        # Cache should not exist yet
        from gomoveshift.pipeline.stop_processor import get_motion_cache_path

        cache_path = get_motion_cache_path(temp_video, cache_dir)
        assert not cache_path.exists()

        # Run detection
        _stops = processor.detect_stops(temp_video)

        # Cache should now exist
        assert cache_path.exists()

        # Should be a valid npz file with the expected arrays
        data = np.load(cache_path)
        assert "timestamps" in data
        assert "motion_energies" in data
        assert len(data["timestamps"]) > 0
        assert len(data["motion_energies"]) > 0

    def test_detect_stops_loads_from_cache_on_second_run(self, temp_video, cache_dir):
        """Second run should load from cache without analyzing."""
        config = get_config()
        processor = StopProcessor(config, cache_dir=cache_dir)

        # First run - creates cache
        stops1 = processor.detect_stops(temp_video)

        # Patch the motion analyzer to verify it's NOT called on second run
        with patch.object(processor, "_analyze_motion_from_video") as mock_analyze:
            # Second run - should use cache
            stops2 = processor.detect_stops(temp_video)

            # Motion analyzer should not have been called
            mock_analyze.assert_not_called()

        # Results should be identical
        assert len(stops1) == len(stops2)

    def test_cache_miss_when_file_changes(self, temp_video, cache_dir):
        """If video file changes, cache should be invalidated."""
        config = get_config()
        processor = StopProcessor(config, cache_dir=cache_dir)

        # First run - creates cache
        _stops1 = processor.detect_stops(temp_video)

        # Modify the video file (append a byte)
        with open(temp_video, "ab") as f:
            f.write(b"\x00")

        # Patch analyzer to verify it IS called after file change
        with patch.object(
            processor, "_analyze_motion_from_video", return_value=([], [])
        ) as mock_analyze:
            _stops2 = processor.detect_stops(temp_video)

            # Analyzer should have been called (cache miss)
            mock_analyze.assert_called_once()

    def test_corrupted_cache_triggers_reanalysis(self, temp_video, cache_dir):
        """Corrupted cache file should trigger fresh analysis."""
        config = get_config()
        processor = StopProcessor(config, cache_dir=cache_dir)

        # First run - creates cache
        _stops1 = processor.detect_stops(temp_video)

        # Corrupt the cache file
        from gomoveshift.pipeline.stop_processor import get_motion_cache_path

        cache_path = get_motion_cache_path(temp_video, cache_dir)
        cache_path.write_text("corrupted data")

        # Should handle corruption gracefully and re-analyze
        stops2 = processor.detect_stops(temp_video)

        # Should succeed (not crash)
        assert isinstance(stops2, list)

        # Cache should be recreated with valid data
        assert cache_path.exists()
        data = np.load(cache_path)
        assert "timestamps" in data


class TestPipelineStateResume:
    """Tests for pipeline state persistence and resume capability."""

    @pytest.fixture
    def temp_video(self, tmp_path):
        """Create a small test video."""
        video_path = tmp_path / "test_video.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=640x360:rate=30",
            "-an",  # No audio
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            pytest.skip("FFmpeg not available")
        return video_path

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create temporary output directory."""
        out = tmp_path / "output"
        out.mkdir()
        return out

    def test_save_pipeline_state_after_each_stage(self, temp_video, output_dir):
        """Pipeline should save state after each completed stage."""
        config = get_config()
        orchestrator = PipelineOrchestrator(config)

        # Run pipeline with all stages skipped except property detection
        result = orchestrator.run(
            RunOptions.from_legacy(
                video_path=temp_video,
                location="Test Location",
                publish_day="wed",
                output_base=output_dir,
                skip_stops=True,
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
            )
        )

        # State file should exist
        state_file = output_dir / result.project_name / "_stages" / "pipeline_state.json"
        assert state_file.exists()

        # State should contain required fields
        state = json.loads(state_file.read_text())
        assert "project_name" in state
        assert "source_video" in state
        assert "source_hash" in state
        assert "stages_completed" in state
        assert "current_video" in state
        assert "params" in state
        assert "video_properties" in state

        # Should have at least detect_properties stage
        assert "detect_properties" in state["stages_completed"]

    def test_resume_from_saved_state(self, temp_video, output_dir):
        """Pipeline should resume from saved state if source matches."""
        config = get_config()
        project_name = temp_video.stem
        project_dir = output_dir / project_name
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        # First run: complete detect_properties only
        orchestrator = PipelineOrchestrator(config)
        _result1 = orchestrator.run(
            RunOptions.from_legacy(
                video_path=temp_video,
                location="Test Location",
                publish_day="wed",
                output_base=output_dir,
                skip_stops=True,
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
            )
        )

        state_file = stages_dir / "pipeline_state.json"
        assert state_file.exists()

        # Verify initial state
        state = json.loads(state_file.read_text())
        assert "detect_properties" in state["stages_completed"]
        initial_stage_count = len(state["stages_completed"])

        # Second run: should resume (detect_properties already done)
        # Run same command again
        result2 = orchestrator.run(
            RunOptions.from_legacy(
                video_path=temp_video,
                location="Test Location",
                publish_day="wed",
                output_base=output_dir,
                skip_stops=True,
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
            )
        )

        # Should have same stages (nothing new to do)
        assert len(result2.stages_completed) == initial_stage_count
        assert "detect_properties" in result2.stages_completed

    def test_fresh_flag_ignores_saved_state(self, temp_video, output_dir):
        """--fresh flag should ignore saved state and start from scratch."""
        config = get_config()
        orchestrator = PipelineOrchestrator(config)
        project_name = temp_video.stem

        # First run
        _result1 = orchestrator.run(
            RunOptions.from_legacy(
                video_path=temp_video,
                location="Test Location",
                publish_day="wed",
                output_base=output_dir,
                skip_stops=True,
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
            )
        )

        # Verify state was saved
        state_file = output_dir / project_name / "_stages" / "pipeline_state.json"
        assert state_file.exists()

        # Modify state to add fake completed stages
        state = json.loads(state_file.read_text())
        state["stages_completed"].append("fake_stage")
        state_file.write_text(json.dumps(state, indent=2))

        # Second run with fresh=True should ignore saved state
        result2 = orchestrator.run(
            RunOptions.from_legacy(
                video_path=temp_video,
                location="Test Location",
                publish_day="wed",
                output_base=output_dir,
                fresh=True,
                skip_stops=True,
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
            )
        )

        # Should NOT have the fake stage
        assert "fake_stage" not in result2.stages_completed

    def test_reject_resume_if_source_hash_differs(self, temp_video, output_dir):
        """Should reject resume if source video has changed."""
        config = get_config()
        orchestrator = PipelineOrchestrator(config)
        project_name = temp_video.stem
        project_dir = output_dir / project_name
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        # Create state with WRONG hash
        state = {
            "project_name": project_name,
            "source_video": str(temp_video),
            "source_hash": "wrong_hash_12345678901234567890123456789012",
            "stages_completed": ["detect_properties"],
            "current_video": str(temp_video),
            "video_properties": {"width": 640, "height": 360, "fps": 30.0, "duration": 2.0},
            "timings": {},
            "params": {
                "location": "Test",
                "publish_day": "wed",
                "skip_color": True,
                "skip_stops": True,
            },
        }
        state_file = stages_dir / "pipeline_state.json"
        state_file.write_text(json.dumps(state))

        # Should detect hash mismatch and restart
        result = orchestrator.run(
            RunOptions.from_legacy(
                video_path=temp_video,
                location="Test",
                publish_day="wed",
                output_base=output_dir,
                skip_stops=True,
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
            )
        )

        # Should have run detect_properties again (restart)
        assert "detect_properties" in result.stages_completed
        # State should have correct hash now
        new_state = json.loads(state_file.read_text())
        assert new_state["source_hash"] != "wrong_hash_12345678901234567890123456789012"

    def test_reject_resume_if_params_differ(self, temp_video, output_dir):
        """Should reject resume if pipeline params have changed."""
        config = get_config()
        orchestrator = PipelineOrchestrator(config)
        project_name = temp_video.stem
        project_dir = output_dir / project_name
        stages_dir = project_dir / "_stages"
        stages_dir.mkdir(parents=True)

        from gomoveshift.pipeline.stop_processor import compute_file_hash

        source_hash = compute_file_hash(temp_video)

        # Create state with different params
        state = {
            "project_name": project_name,
            "source_video": str(temp_video),
            "source_hash": source_hash,
            "stages_completed": ["detect_properties"],
            "current_video": str(temp_video),
            "video_properties": {"width": 640, "height": 360, "fps": 30.0, "duration": 2.0},
            "timings": {},
            "params": {
                "location": "Old Location",
                "publish_day": "mon",
                "skip_color": False,
                "skip_stops": False,
            },
        }
        state_file = stages_dir / "pipeline_state.json"
        state_file.write_text(json.dumps(state))

        # Run with DIFFERENT params
        result = orchestrator.run(
            RunOptions.from_legacy(
                video_path=temp_video,
                location="New Location",  # DIFFERENT
                publish_day="wed",  # DIFFERENT
                output_base=output_dir,
                skip_stops=True,
                skip_color=True,
                skip_ost=True,
                skip_audio=True,
                skip_metadata=True,
                skip_social=True,
                skip_upload=True,
                skip_csv=True,
            )
        )

        # Should restart from scratch
        assert "detect_properties" in result.stages_completed
        # State should have new params
        new_state = json.loads(state_file.read_text())
        assert new_state["params"]["location"] == "New Location"
        assert new_state["params"]["publish_day"] == "wed"


class TestTmuxLauncher:
    """Tests for automatic tmux session launcher."""

    def test_detects_already_in_tmux(self):
        """Should detect if already running inside tmux."""
        from gomoveshift.cli import is_in_tmux

        # Test with TMUX env var set
        with patch.dict(os.environ, {"TMUX": "/tmp/tmux-1000/default,12345,0"}):
            assert is_in_tmux() is True

        # Test without TMUX env var
        with patch.dict(os.environ, {}, clear=True):
            assert is_in_tmux() is False

    def test_pipeline_launches_tmux_by_default(self):
        """Pipeline command should launch tmux session when not already in one."""
        from gomoveshift.cli import cmd_pipeline

        # Mock: not in tmux, tmux available
        with (
            patch("gomoveshift.cli.handlers.pipeline.is_in_tmux", return_value=False),
            patch("gomoveshift.cli.handlers.pipeline.tmux_available", return_value=True),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)

            args = MagicMock(
                video=Path("/fake/video.mp4"),
                location="Test",
                publish="wed",
                no_tmux=False,
            )

            # Should re-exec in tmux
            cmd_pipeline(args, get_config())

            # Verify tmux was called
            assert mock_run.called
            call_args = mock_run.call_args[0][0]
            assert "tmux" in call_args
            assert "gms-pipeline" in call_args

    def test_no_tmux_flag_skips_launcher(self, tmp_path):
        """--no-tmux flag should skip tmux launcher."""
        from gomoveshift.cli import cmd_pipeline

        # Create a real temp video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = Path(f.name)

        try:
            with (
                patch("gomoveshift.cli.handlers.pipeline.is_in_tmux", return_value=False),
                patch("gomoveshift.cli.handlers.pipeline.tmux_available", return_value=True),
                patch("gomoveshift.cli.launch_in_tmux") as mock_launch,
                patch("gomoveshift.pipeline.PipelineOrchestrator") as _mock_orch,
            ):
                args = MagicMock(
                    video=video_path,
                    location="Test",
                    publish="wed",
                    no_tmux=True,  # Explicitly disable
                    output_dir=tmp_path,
                    lut=None,
                    crossfade=0.5,
                    skip_color=True,
                    skip_stops=True,
                    fresh=False,
                )

                # Should run directly without tmux
                try:
                    cmd_pipeline(args, get_config())
                except Exception:
                    pass  # Ignore pipeline errors

                # launch_in_tmux should NOT be called
                mock_launch.assert_not_called()
        finally:
            video_path.unlink(missing_ok=True)

    def test_already_in_tmux_runs_directly(self, tmp_path):
        """If already in tmux, should run directly without re-launching."""
        from gomoveshift.cli import cmd_pipeline

        # Create a real temp video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = Path(f.name)

        try:
            with (
                patch("gomoveshift.cli.handlers.pipeline.is_in_tmux", return_value=True),
                patch("gomoveshift.cli.launch_in_tmux") as mock_launch,
                patch("gomoveshift.pipeline.PipelineOrchestrator") as _mock_orch,
            ):
                args = MagicMock(
                    video=video_path,
                    location="Test",
                    publish="wed",
                    no_tmux=False,
                    output_dir=tmp_path,
                    lut=None,
                    crossfade=0.5,
                    skip_color=True,
                    skip_stops=True,
                    fresh=False,
                )

                # Should run directly (already in tmux)
                try:
                    cmd_pipeline(args, get_config())
                except Exception:
                    pass  # Ignore pipeline errors

                # launch_in_tmux should NOT be called
                mock_launch.assert_not_called()
        finally:
            video_path.unlink(missing_ok=True)

    def test_tmux_session_name_is_gms_pipeline(self):
        """tmux session should be named 'gms-pipeline'."""
        from gomoveshift.cli import launch_in_tmux

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            launch_in_tmux(["python", "script.py"])

            # Find the new-session call (should be second call after kill)
            calls = mock_run.call_args_list
            new_session_call = None
            for call in calls:
                args = call[0][0]
                if "new-session" in args:
                    new_session_call = args
                    break

            assert new_session_call is not None, "Should have called tmux new-session"
            assert "-s" in new_session_call
            idx = new_session_call.index("-s")
            assert new_session_call[idx + 1] == "gms-pipeline"

    def test_warning_printed_before_tmux_launch(self, capsys):
        """Should print warning before launching tmux."""
        from gomoveshift.cli import launch_in_tmux

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            launch_in_tmux(["python", "script.py"])

            captured = capsys.readouterr()
            assert "tmux session 'gms-pipeline'" in captured.out.lower()


def _session_name(stem: str) -> str:
    """A tmux session name no CONCURRENT run of this suite can also pick.

    These two tests drive the real tmux server, which is per-user and shared by
    every process on the host — so a fixed name is not a name, it is a lock two
    runs fight over. ``launch_in_tmux`` refuses to create a duplicate session on
    purpose (a second launch must never kill an in-flight multi-hour drive), so
    the loser's ``new-session`` is REFUSED and its ``result == 0`` fails; worse,
    the loser's ``finally`` then kills the winner's session out from under it and
    ``capture-pane`` returns nothing — the ``assert 'hello world' in ''`` this
    suite kept showing on CI, where two runners share one host.

    The pid makes the collision unrepresentable rather than unlikely: concurrent
    runs are distinct processes (xdist workers included), so they cannot select
    the same name. Nothing here is timing-dependent, so there is no flake left to
    retry.
    """
    return f"pytest-{stem}-{os.getpid()}"


class TestTmuxLauncherIntegration:
    """Integration tests that use real tmux."""

    def test_tmux_launcher_persists_session(self):
        """Session should persist after launch_in_tmux returns."""
        import subprocess
        import time

        from gomoveshift.cli import launch_in_tmux, tmux_available

        if not tmux_available():
            pytest.skip("tmux not available")

        session_name = _session_name("pipeline")

        # Cleanup any existing session
        subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)

        try:
            # Launch a command that sleeps
            result = launch_in_tmux(["sleep", "10"], session_name=session_name)
            assert result == 0

            # Give it a moment to start
            time.sleep(0.5)

            # Session should exist
            check = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True)
            assert check.returncode == 0, "tmux session should persist"
        finally:
            # Cleanup
            subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)

    def test_tmux_launcher_handles_quoted_args(self):
        """Arguments with spaces should be properly quoted."""
        import subprocess
        import time

        from gomoveshift.cli import launch_in_tmux, tmux_available

        if not tmux_available():
            pytest.skip("tmux not available")

        session_name = _session_name("quote")

        # Cleanup any existing session
        subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)

        try:
            # Launch with an argument containing spaces - use bash -c to echo and sleep
            # so the session stays alive long enough to capture output
            result = launch_in_tmux(
                ["bash", "-c", 'echo "hello world" "test arg"; sleep 5'], session_name=session_name
            )
            assert result == 0

            # Give it a moment
            time.sleep(0.5)

            # Capture output
            output = subprocess.run(
                ["tmux", "capture-pane", "-t", session_name, "-p"], capture_output=True, text=True
            )
            # Should have both arguments properly passed
            assert "hello world" in output.stdout
            assert "test arg" in output.stdout
        finally:
            # Cleanup
            subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)


class TestTmuxEnforcement:
    """Tests for mandatory tmux enforcement."""

    def test_fails_when_tmux_unavailable_and_no_opt_out(self):
        """Should error if tmux unavailable and --no-tmux not specified."""
        from unittest.mock import MagicMock, patch

        from gomoveshift.cli import cmd_pipeline

        args = MagicMock()
        args.no_tmux = False
        args.video = MagicMock()
        args.video.exists.return_value = True

        # cmd_pipeline runs `config.seo.validate_paths_exist()` (the fail-fast
        # gate) before the tmux check. A MagicMock config
        # makes that a no-op so the test reaches the tmux branch.
        config = MagicMock()

        # Mock: not in tmux, tmux NOT available
        with (
            patch("gomoveshift.cli.handlers.pipeline.is_in_tmux", return_value=False),
            patch("gomoveshift.cli.handlers.pipeline.tmux_available", return_value=False),
        ):
            result = cmd_pipeline(args, config)
            assert result == 1  # Should fail

    def test_refuses_unsupervised_no_tmux(self, capsys, monkeypatch):
        """--no-tmux without a supervisor is REFUSED, not warned.

        A printed WARNING that continues is not enough: an unsupervised
        long encode can die with its parent process unnoticed. A warning ahead of irreversible
        work is a late gate, so the flag asserts supervision and is checked.
        """
        import os
        from unittest.mock import MagicMock, patch

        from gomoveshift.cli import cmd_pipeline

        args = MagicMock()
        args.no_tmux = True
        args.video = MagicMock()
        args.video.exists.return_value = False
        args.location = "Test"
        args.publish = "sat"
        config = MagicMock()

        # Defeat the conftest supervision declaration: this case IS the
        # unsupervised operator.
        monkeypatch.delenv("GMS_SUPERVISED", raising=False)
        monkeypatch.setattr(os, "getsid", lambda _pid: os.getpid() + 1)

        with (
            patch("gomoveshift.cli.handlers.pipeline.is_in_tmux", return_value=False),
            patch("gomoveshift.cli.handlers.pipeline.tmux_available", return_value=True),
        ):
            result = cmd_pipeline(args, config)

        assert result == 2, "an unsupervised --no-tmux run must not start"
        captured = capsys.readouterr()
        assert "GMS_SUPERVISED" in captured.err, "the error must name the escape hatch"

    def test_allows_supervised_no_tmux(self, monkeypatch):
        """A declared supervisor keeps --no-tmux usable for CI and tests.

        Passage is proved by reaching the video-exists check below the gate
        (returns 1), which a refusal (2) would never reach.
        """
        from unittest.mock import MagicMock, patch

        from gomoveshift.cli import cmd_pipeline

        args = MagicMock()
        args.no_tmux = True
        args.video = MagicMock()
        args.video.exists.return_value = False
        args.location = "Test"
        args.publish = "sat"
        config = MagicMock()

        monkeypatch.setenv("GMS_SUPERVISED", "1")

        with (
            patch("gomoveshift.cli.handlers.pipeline.is_in_tmux", return_value=False),
            patch("gomoveshift.cli.handlers.pipeline.tmux_available", return_value=True),
        ):
            result = cmd_pipeline(args, config)

        assert result == 1, "supervised --no-tmux must pass the gate and reach the file check"

    def test_launch_in_tmux_verifies_session_exists(self):
        """launch_in_tmux should verify the session persists after creation."""
        from unittest.mock import MagicMock, patch

        from gomoveshift.cli import launch_in_tmux

        # Mock successful tmux create but failed verify
        with patch("subprocess.run") as mock_run:
            # No kill-session: new-session succeeds, then the
            # has-session verify FAILS -> the persistence gate fires.
            mock_run.side_effect = [
                MagicMock(returncode=0),  # new-session
                MagicMock(returncode=1),  # has-session FAILS
            ]

            result = launch_in_tmux(["echo", "test"], session_name="test-session")

            # Should fail because verification failed
            assert result == 1
