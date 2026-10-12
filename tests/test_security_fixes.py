"""Tests for the security and correctness fixes.

Covers:
- caption_generator uses --permission-mode default (not bypassPermissions)
- phase_b creates per-thread StopProcessor instances
- trailing stop end_time uses video_duration when provided
"""

from unittest.mock import MagicMock, patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline.exceptions import ProbeFailedError
from gomoveshift.pipeline.stop_processor import StopProcessor


class TestCaptionGeneratorPermissionMode:
    """The subprocess call must NOT contain bypassPermissions."""

    @patch("subprocess.Popen")
    def test_call_llm_uses_default_permission_mode(self, mock_run, tmp_path):
        """_call_llm must pass --permission-mode default, never bypassPermissions."""
        from gomoveshift.pipeline.photos.caption_generator import CaptionGenerator

        mock_run.return_value = MagicMock(
            returncode=0, communicate=MagicMock(return_value=("test output", ""))
        )

        gen = CaptionGenerator(caption_dir=tmp_path)
        gen._call_llm("test prompt")

        cmd_list = mock_run.call_args[0][0]
        # Must contain --permission-mode followed by "default"
        assert "--permission-mode" in cmd_list
        pm_idx = cmd_list.index("--permission-mode")
        assert cmd_list[pm_idx + 1] == "default"
        # Must NOT contain bypassPermissions anywhere
        assert "bypassPermissions" not in cmd_list

    @patch("subprocess.Popen")
    def test_call_llm_passes_prompt_via_stdin_not_argv(self, mock_run, tmp_path):
        """Prompt must go through STDIN (input=), never argv (visible in ps/proc)."""
        from gomoveshift.pipeline.photos.caption_generator import CaptionGenerator

        mock_run.return_value = MagicMock(
            returncode=0, communicate=MagicMock(return_value=("test output", ""))
        )

        gen = CaptionGenerator(caption_dir=tmp_path)
        prompt_text = "do-not-leak-this-prompt-on-argv"
        gen._call_llm(prompt_text)

        cmd_list = mock_run.call_args[0][0]
        # Prompt must NOT appear in the argv list (no /proc/<pid>/cmdline exposure)
        assert prompt_text not in cmd_list
        # Prompt must be supplied via stdin (piped to communicate)
        assert mock_run.call_args[1].get("stdin") is not None
        assert mock_run.return_value.communicate.call_args[1].get("input") == prompt_text
        # The -p flag is retained so claude reads the prompt from stdin
        assert "-p" in cmd_list


# TestPhaseBThreadSafety removed — phase_b.py was deleted
# along with the entire non-day-first path. The per-thread StopProcessor
# invariant that remains in the surviving day-first path is covered by
# tests under tests/test_day_pipeline.py and the StopProcessor unit tests
# below (TestTrailingStopEndTime).


class TestTrailingStopEndTime:
    """A trailing stop end_time uses video_duration, not last sample."""

    def _make_processor(self) -> StopProcessor:
        """Create a StopProcessor with known thresholds."""
        config = Config()
        proc = StopProcessor(config)
        proc.motion_threshold = 0.02
        proc.min_stop_duration = 5.0
        return proc

    def test_trailing_stop_uses_video_duration(self):
        """When video_duration is provided, trailing stop end_time == video_duration."""
        proc = self._make_processor()

        # 20 timestamps at 1-second intervals; last 8 seconds are below threshold
        timestamps = list(range(20))
        energies = [0.1] * 12 + [0.001] * 8  # stop starts at t=12

        stops = proc._find_stops_in_motion_data(timestamps, energies, video_duration=25.0)

        assert len(stops) == 1
        assert stops[0].start_time == 12.0
        assert stops[0].end_time == 25.0  # video duration, not 19.0

    def test_trailing_stop_hard_fails_when_duration_unavailable(self):
        """No silent fallback: a real trailing stop without a probed duration
        raises ProbeFailedError instead of truncating to timestamps[-1]."""
        proc = self._make_processor()

        timestamps = list(range(20))
        energies = [0.1] * 12 + [0.001] * 8

        with pytest.raises(ProbeFailedError):
            proc._find_stops_in_motion_data(timestamps, energies, video_duration=None)

    def test_no_trailing_stop_tolerates_missing_duration(self):
        """When there is NO trailing stop, a missing duration is harmless —
        the probe failure must not block an otherwise clean video."""
        proc = self._make_processor()

        # Motion is above threshold at the final samples -> no trailing stop.
        timestamps = list(range(20))
        energies = [0.1] * 12 + [0.001] * 6 + [0.1] * 2

        stops = proc._find_stops_in_motion_data(timestamps, energies, video_duration=None)

        # The interior stop (t=12..18) is still found; no exception raised.
        assert len(stops) == 1
        assert stops[0].start_time == 12.0
        assert stops[0].end_time == 18.0

    def test_no_trailing_stop_when_duration_too_short(self):
        """Trailing stop shorter than min_stop_duration is ignored."""
        proc = self._make_processor()
        proc.min_stop_duration = 10.0

        timestamps = list(range(20))
        energies = [0.1] * 17 + [0.001] * 3  # only 3s of stop

        stops = proc._find_stops_in_motion_data(timestamps, energies, video_duration=20.0)

        assert len(stops) == 0

    def test_empty_timestamps_returns_no_stops(self):
        """Edge case: empty input returns empty list."""
        proc = self._make_processor()
        stops = proc._find_stops_in_motion_data([], [], video_duration=10.0)
        assert stops == []
