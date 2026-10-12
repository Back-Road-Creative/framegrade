"""Tests for orchestrator intermediate path wiring."""

from pathlib import Path

from gomoveshift.pipeline.orchestrator import _PipelinePaths


class TestPipelinePathsResolve:
    """_PipelinePaths.resolve uses gms_intermediate_dir for stages_dir."""

    def test_drvfs_output_redirects_stages(self):
        paths = _PipelinePaths.resolve(Path("/mnt/x/Output"), "proj1")
        assert not str(paths.stages_dir).startswith("/mnt/")
        assert "proj1" in str(paths.stages_dir)
        assert "_stages" in str(paths.stages_dir)

    def test_ext4_output_keeps_stages_local(self):
        paths = _PipelinePaths.resolve(Path("/srv/output"), "proj1")
        assert paths.stages_dir == Path("/srv/output/proj1/_stages")

    def test_state_file_stays_on_drvfs(self):
        """pipeline_state.json must stay on drvfs for persistence."""
        paths = _PipelinePaths.resolve(Path("/mnt/x/Output"), "proj1")
        assert str(paths.state_file).startswith("/mnt/")

    def test_intermediate_folder_config(self):
        paths = _PipelinePaths.resolve(
            Path("/mnt/x/Output"),
            "proj1",
            intermediate_folder="/data/fast",
        )
        assert paths.stages_dir == Path("/data/fast/proj1/_stages")
