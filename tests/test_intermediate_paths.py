"""Tests for gms_intermediate_dir() and PipelineState.stages_base."""

import os
from pathlib import Path

import pytest

from gomoveshift.paths import gms_intermediate_dir
from gomoveshift.pipeline.drive_state_model import PipelineState


class TestIntermediateDirDrvfs:
    """When output is on /mnt/ (drvfs), redirect intermediates to ext4."""

    def test_returns_ext4_when_drvfs(self):
        result = gms_intermediate_dir(Path("/mnt/d/output"), "proj1")
        assert not str(result).startswith("/mnt/")
        # Must be on a persistent filesystem — /tmp (tmpfs) is wiped on reboot
        # and kills long-running pipelines mid-flight.
        assert not str(result).startswith("/tmp/")
        assert result.name == "_stages"
        assert result.parent.name == "proj1"

    def test_drvfs_default_is_under_the_cache_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GMS_INTERMEDIATE_DIR", raising=False)
        monkeypatch.setenv("GMS_CACHE_HOME", str(tmp_path / "cache"))
        result = gms_intermediate_dir(Path("/mnt/d/output"), "proj1")
        assert result == tmp_path / "cache" / "proj1" / "_stages"

    def test_returns_output_base_when_ext4(self):
        result = gms_intermediate_dir(Path("/srv/gms/output"), "proj1")
        assert result == Path("/srv/gms/output/proj1/_stages")

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", "/fast/ssd")
        result = gms_intermediate_dir(Path("/mnt/d/output"), "proj1")
        assert result == Path("/fast/ssd/proj1/_stages")

    def test_env_var_overrides_ext4_too(self, monkeypatch):
        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", "/fast/ssd")
        result = gms_intermediate_dir(Path("/srv/gms/output"), "proj1")
        assert result == Path("/fast/ssd/proj1/_stages")

    def test_config_override_on_drvfs(self):
        result = gms_intermediate_dir(
            Path("/mnt/d/output"),
            "proj1",
            intermediate_folder="/data/fast",
        )
        assert result == Path("/data/fast/proj1/_stages")

    def test_config_ignored_on_ext4(self):
        """Config intermediate_folder is only used for drvfs redirect."""
        result = gms_intermediate_dir(
            Path("/srv/gms/output"),
            "proj1",
            intermediate_folder="/data/fast",
        )
        assert result == Path("/srv/gms/output/proj1/_stages")

    def test_env_var_beats_config(self, monkeypatch):
        monkeypatch.setenv("GMS_INTERMEDIATE_DIR", "/fast/ssd")
        result = gms_intermediate_dir(
            Path("/mnt/d/output"),
            "proj1",
            intermediate_folder="/data/fast",
        )
        assert result == Path("/fast/ssd/proj1/_stages")


class TestPipelineStateStagesBase:
    """PipelineState.stages_base field serialization."""

    def test_roundtrip_with_stages_base(self):
        state = PipelineState(
            output_base=Path("/mnt/d/output/proj1"),
            stages_base=Path("/tmp/gomoveshift/proj1/_stages"),
        )
        d = state.to_dict()
        restored = PipelineState.from_dict(d)
        assert restored.stages_base == Path("/tmp/gomoveshift/proj1/_stages")

    def test_backward_compat_no_stages_base(self):
        """Old state dicts without stages_base load without error."""
        d = {"output_base": "/mnt/d/output/proj1", "region": "us"}
        state = PipelineState.from_dict(d)
        assert state.stages_base is None
        assert state.output_base == Path("/mnt/d/output/proj1")

    def test_none_stages_base_not_serialized(self):
        state = PipelineState(output_base=Path("/mnt/d/output/proj1"))
        d = state.to_dict()
        assert "stages_base" not in d
