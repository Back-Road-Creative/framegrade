"""Tests for motion cache location — StopProcessor defaults to ``<cache home>/motion_cache``."""

import os
from pathlib import Path
from unittest.mock import MagicMock

from gomoveshift.pipeline.stop_processor import DEFAULT_CACHE_DIR, StopProcessor


class TestMotionCacheLocation:
    """Verify StopProcessor respects custom cache_dir."""

    def test_default_cache_dir_is_the_module_default(self):
        """Without explicit cache_dir, StopProcessor uses ``<cache home>/motion_cache``."""
        config = MagicMock()
        config.stop_detection.motion_threshold = 0.5
        config.stop_detection.min_stop_duration = 2.0

        proc = StopProcessor(config)
        assert proc.cache_dir == DEFAULT_CACHE_DIR

    def test_default_cache_dir_lives_under_the_cache_home(self, tmp_path):
        """The default follows GMS_CACHE_HOME (bound at import), not a fixed system path."""
        import subprocess
        import sys

        probe = "from gomoveshift.pipeline.stop_processor import DEFAULT_CACHE_DIR; print(DEFAULT_CACHE_DIR)"
        env = {**os.environ, "GMS_CACHE_HOME": str(tmp_path / "cachehome")}
        out = subprocess.run(
            [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
        ).stdout.strip()
        assert Path(out) == tmp_path / "cachehome" / "motion_cache"

    def test_custom_cache_dir_stored(self, tmp_path):
        """StopProcessor stores and uses a custom cache_dir when provided."""
        config = MagicMock()
        config.stop_detection.motion_threshold = 0.5
        config.stop_detection.min_stop_duration = 2.0

        custom_dir = tmp_path / "motion_cache"
        proc = StopProcessor(config, cache_dir=custom_dir)
        assert proc.cache_dir == custom_dir

    def test_stop_removal_stage_passes_project_local_cache(self):
        """StopRemovalStage should pass a project-local cache_dir to StopProcessor."""
        from gomoveshift.pipeline.stages.stop_removal import StopRemovalStage

        # Build a mock context with the paths the stage needs
        ctx = MagicMock()
        ctx.paths.stages_dir = Path("/fake/stages")
        ctx.crossfade_duration = 0.5
        ctx.video_properties.height = 1080
        ctx.config.stop_detection.motion_threshold = 0.5
        ctx.config.stop_detection.min_stop_duration = 2.0
        ctx.current_video = Path("/fake/input.mp4")
        ctx.project_name = "test"

        # Capture the StopProcessor constructor call
        captured = {}

        class FakeStopProcessor:
            def __init__(self, config, **kwargs):
                captured.update(kwargs)

            def process(self, video_path, output_path):
                return {"stops_found": 0, "stops": [], "total_removed_duration": 0.0}

        # Patch StopProcessor import in orchestrator (where stop_removal.py imports it)
        import gomoveshift.pipeline.orchestrator as orch

        original = orch.StopProcessor
        orch.StopProcessor = FakeStopProcessor
        try:
            stage = StopRemovalStage()
            stage.run(ctx)
        finally:
            orch.StopProcessor = original

        expected_cache = Path("/fake/stages/motion_cache")
        assert captured.get("cache_dir") == expected_cache, (
            f"Expected cache_dir={expected_cache}, got {captured.get('cache_dir')}"
        )
