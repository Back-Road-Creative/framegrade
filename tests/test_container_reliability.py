"""
Tests for container reliability — state persistence, permissions, credentials.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFailPersistsState:
    """_fail() must save state to disk so hours of progress are not lost."""

    def test_fail_saves_state_file(self, tmp_path):
        from gomoveshift.pipeline.orchestrator import (
            PipelineOrchestrator,
            PipelineResult,
            VideoProperties,
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        stages_dir = output_dir / "_stages"
        stages_dir.mkdir()

        result = PipelineResult(
            success=True,
            project_name="test_project",
            output_dir=output_dir,
            final_video=output_dir / "test_FINAL.mp4",
            stages_completed=["detect_properties", "stop_removal"],
            errors=[],
            video_properties=VideoProperties(
                width=3840,
                height=2160,
                fps=59.94,
                duration=3600.0,
                codec="hevc",
                audio_codec="aac",
                file_size_bytes=15_000_000_000,
            ),
        )

        orch = PipelineOrchestrator()
        orch._fail("Test failure message", result)

        state_file = stages_dir / "pipeline_state.json"
        assert state_file.exists(), "_fail() did not persist state file"

        state = json.loads(state_file.read_text())
        assert state["project_name"] == "test_project"
        assert "detect_properties" in state["stages_completed"]
        assert state["params"]["failed"] is True

    def test_fail_without_video_properties_does_not_crash(self, tmp_path):
        """_fail() should gracefully handle missing video properties."""
        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator, PipelineResult

        result = PipelineResult(
            success=True,
            project_name="test",
            output_dir=tmp_path,
        )

        orch = PipelineOrchestrator()
        failed_result = orch._fail("Error", result)
        assert not failed_result.success
        assert "Error" in failed_result.errors


class TestAnalyticsNotFatal:
    """Analytics FileNotFoundError must not abort the pipeline."""

    def test_analytics_file_not_found_continues(self):
        """FileNotFoundError in analytics sync should warn, not abort."""
        import inspect

        from gomoveshift.pipeline.orchestrator import PipelineOrchestrator

        from gomoveshift.pipeline.stages.analytics_sync import AnalyticsSyncStage

        source = inspect.getsource(AnalyticsSyncStage.run)
        # After our change, FileNotFoundError should NOT call self._fail
        # It should log a warning instead
        # Check that the FileNotFoundError handler does not contain _fail
        import re

        # Find the FileNotFoundError block
        match = re.search(
            r"except FileNotFoundError.*?(?=except |\n        #|\n    def )",
            source,
            re.DOTALL,
        )
        assert match, "Could not find FileNotFoundError handler in orchestrator.run"
        handler = match.group()
        assert "_fail" not in handler, (
            "FileNotFoundError in analytics sync still calls _fail — should warn and continue"
        )


class TestSOPSDirectFirst:
    """SOPS should try direct sops before any opt-in sudo attempt."""

    @staticmethod
    def _run_loader(tmp_path, monkeypatch, sudo_user):
        from unittest.mock import MagicMock, patch

        import pytest

        from gomoveshift.pipeline import _yt_creds

        if sudo_user is None:
            monkeypatch.delenv("GMS_SOPS_SUDO_USER", raising=False)
        else:
            monkeypatch.setenv("GMS_SOPS_SUDO_USER", sudo_user)
        # The SOPS file lives next to (not inside) the credentials directory.
        (tmp_path / "youtube_upload_token.json.enc").write_text("encrypted")
        creds_dir = tmp_path / "credentials"
        creds_dir.mkdir()

        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(list(cmd))
            result = MagicMock()
            result.returncode = 1
            result.stderr = "decrypt failed"
            result.stdout = ""
            return result

        with patch.object(_yt_creds.subprocess, "run", side_effect=fake_run):
            with pytest.raises(FileNotFoundError):
                _yt_creds.load_youtube_credentials(credentials_dir=creds_dir)
        return calls

    def test_sops_tries_direct_before_sudo(self, tmp_path, monkeypatch):
        """The credential loader tries 'sops' directly before 'sudo -u <user> sops'."""
        calls = self._run_loader(tmp_path, monkeypatch, "keyowner")

        assert calls[0][0] == "sops", "Direct sops should be tried first"
        sudo_idx = [i for i, c in enumerate(calls) if c[:4] == ["sudo", "-u", "keyowner", "sops"]]
        assert sudo_idx, "Sudo sops fallback not attempted with GMS_SOPS_SUDO_USER set"
        assert sudo_idx[0] > 0, "Direct sops should be tried BEFORE sudo sops fallback"

    def test_no_sudo_fallback_when_unset(self, tmp_path, monkeypatch):
        """Without GMS_SOPS_SUDO_USER the loader never invokes sudo."""
        calls = self._run_loader(tmp_path, monkeypatch, None)

        assert calls, "Direct sops should still be attempted"
        assert all(c[0] != "sudo" for c in calls)


class TestMotionCacheDirPermissions:
    """Motion cache dir must be created with permissive mode and handle PermissionError."""

    def test_ensure_cache_dir_creates_with_mode(self, tmp_path):
        from gomoveshift.pipeline.stop_processor import _ensure_cache_dir

        cache_dir = tmp_path / "test_cache"
        result = _ensure_cache_dir(cache_dir)
        assert result.exists()
        assert result == cache_dir

    def test_ensure_cache_dir_falls_back_on_permission_error(self, tmp_path):
        from gomoveshift.pipeline.stop_processor import _ensure_cache_dir

        # Use a path that will fail (e.g., under a read-only parent)
        with patch("gomoveshift.pipeline.stop_processor.Path.mkdir", side_effect=PermissionError):
            result = _ensure_cache_dir(Path("/nonexistent/restricted/cache"))
            assert result.exists(), "Fallback temp dir should exist"
            assert "gomoveshift_motion_cache_" in str(result)


class TestDoctorOverlayMounts:
    """The doctor warns when a compose-overlay variable is set but unmounted."""

    def test_warns_for_each_set_variable_without_a_mount(self):
        from gomoveshift.cli.handlers.doctor import WARN, check_overlay_mounts

        env = {
            "GMS_SITE_HOST_PATH": "/host/site",
            "GMS_NIGHT_HOST_ROOT": "/host/cards",
            "GMS_VID_OUTPUT_DIR": "/extra/out",
        }
        results = check_overlay_mounts(env, in_container=True, is_mount=lambda p: False)
        assert [r.status for r in results] == [WARN, WARN, WARN]
        assert "/site" in results[0].detail
        assert "/night-source" in results[1].detail
        assert "/extra/out" in results[2].detail

    def test_silent_when_mounted_unset_or_on_the_host(self):
        from gomoveshift.cli.handlers.doctor import check_overlay_mounts

        env = {"GMS_SITE_HOST_PATH": "/host/site", "GMS_VID_OUTPUT_DIR": "/extra/out"}
        assert check_overlay_mounts(env, in_container=True, is_mount=lambda p: True) == []
        assert check_overlay_mounts({}, in_container=True, is_mount=lambda p: False) == []
        assert check_overlay_mounts(env, in_container=False, is_mount=lambda p: False) == []
