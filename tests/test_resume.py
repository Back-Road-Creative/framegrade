"""TDD tests for video processing resume capability.

Tests the ability to resume stop detection from where it left off
using stops_report.json as an implicit checkpoint.
"""

import json

import pytest


class TestResumeDetection:
    """Tests for detecting existing checkpoint."""

    def test_detects_existing_output_file(self, tmp_path):
        """Should detect stops_report.json as checkpoint."""
        from gomoveshift.resume import get_checkpoint_state

        # Create checkpoint file
        checkpoint = {
            "videos": {
                "video1.mp4": {"stops": [10.5, 25.3], "duration": 120.0},
                "video2.mp4": {"stops": [5.0], "duration": 90.0},
            }
        }
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        state = get_checkpoint_state(tmp_path)

        assert state is not None
        assert state["has_checkpoint"] is True
        assert "video1.mp4" in state["completed_videos"]
        assert "video2.mp4" in state["completed_videos"]

    def test_no_checkpoint_when_no_output_file(self, tmp_path):
        """Fresh start when no stops_report.json exists."""
        from gomoveshift.resume import get_checkpoint_state

        state = get_checkpoint_state(tmp_path)

        assert state["has_checkpoint"] is False
        assert state["completed_videos"] == []

    def test_extracts_completed_video_names(self, tmp_path):
        """Should parse completed video names from JSON."""
        from gomoveshift.resume import get_checkpoint_state

        checkpoint = {
            "videos": {
                "VID_001.mp4": {"stops": [10.5], "duration": 120.0},
                "VID_002.mp4": {"stops": [], "duration": 90.0},
                "VID_003.mp4": {"stops": [5.0, 15.0, 25.0], "duration": 180.0},
            }
        }
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        state = get_checkpoint_state(tmp_path)

        assert len(state["completed_videos"]) == 3
        assert "VID_001.mp4" in state["completed_videos"]
        assert "VID_002.mp4" in state["completed_videos"]
        assert "VID_003.mp4" in state["completed_videos"]


class TestVideoFiltering:
    """Tests for filtering out completed videos."""

    def test_filters_completed_videos(self, tmp_path):
        """Should exclude already-processed videos."""
        from gomoveshift.resume import get_videos_to_process

        # Create video files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mp4").touch()
        (tmp_path / "video3.mp4").touch()

        # Create checkpoint with video1 and video2 complete
        checkpoint = {
            "videos": {
                "video1.mp4": {"stops": [10.5], "duration": 120.0},
                "video2.mp4": {"stops": [], "duration": 90.0},
            }
        }
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        videos = get_videos_to_process(tmp_path)

        assert len(videos) == 1
        assert videos[0].name == "video3.mp4"

    def test_includes_new_videos_added_after_checkpoint(self, tmp_path):
        """New videos added after checkpoint should be processed."""
        from gomoveshift.resume import get_videos_to_process

        # Create video files including a new one
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mp4").touch()
        (tmp_path / "video_new.mp4").touch()

        # Checkpoint only has video1
        checkpoint = {"videos": {"video1.mp4": {"stops": [10.5], "duration": 120.0}}}
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        videos = get_videos_to_process(tmp_path)

        names = [v.name for v in videos]
        assert len(videos) == 2
        assert "video2.mp4" in names
        assert "video_new.mp4" in names

    def test_preserves_original_sort_order(self, tmp_path):
        """Should maintain alphabetical order of videos."""
        from gomoveshift.resume import get_videos_to_process

        # Create video files
        (tmp_path / "c_video.mp4").touch()
        (tmp_path / "a_video.mp4").touch()
        (tmp_path / "b_video.mp4").touch()

        videos = get_videos_to_process(tmp_path)

        names = [v.name for v in videos]
        assert names == ["a_video.mp4", "b_video.mp4", "c_video.mp4"]


class TestCheckpointIntegrity:
    """Tests for handling corrupted/incomplete checkpoints."""

    def test_handles_corrupted_json(self, tmp_path):
        """Corrupted JSON should trigger fresh start."""
        from gomoveshift.resume import get_checkpoint_state

        (tmp_path / "stops_report.json").write_text("{ invalid json }")

        state = get_checkpoint_state(tmp_path)

        assert state["has_checkpoint"] is False
        assert state["completed_videos"] == []

    def test_handles_empty_output_file(self, tmp_path):
        """Empty file should be treated as no checkpoint."""
        from gomoveshift.resume import get_checkpoint_state

        (tmp_path / "stops_report.json").write_text("")

        state = get_checkpoint_state(tmp_path)

        assert state["has_checkpoint"] is False

    def test_handles_partial_video_entry(self, tmp_path):
        """Video entry missing 'stops' key is incomplete, should retry."""
        from gomoveshift.resume import get_checkpoint_state

        checkpoint = {
            "videos": {
                "video1.mp4": {"stops": [10.5], "duration": 120.0},
                "video2.mp4": {"duration": 90.0},  # Missing 'stops'
            }
        }
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        state = get_checkpoint_state(tmp_path)

        # video2 should NOT be in completed since it's incomplete
        assert "video1.mp4" in state["completed_videos"]
        assert "video2.mp4" not in state["completed_videos"]

    def test_video_with_error_not_counted_as_complete(self, tmp_path):
        """Videos with 'error' key should be retried."""
        from gomoveshift.resume import get_checkpoint_state

        checkpoint = {
            "videos": {
                "video1.mp4": {"stops": [10.5], "duration": 120.0},
                "video2.mp4": {"error": "Memory allocation failed"},
            }
        }
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        state = get_checkpoint_state(tmp_path)

        assert "video1.mp4" in state["completed_videos"]
        assert "video2.mp4" not in state["completed_videos"]


class TestOutputMerging:
    """Tests for merging new results with checkpoint."""

    def test_merges_new_results_with_existing(self, tmp_path):
        """Should combine checkpoint data with new results."""
        from gomoveshift.resume import merge_results

        # Existing checkpoint
        checkpoint = {"videos": {"video1.mp4": {"stops": [10.5], "duration": 120.0}}}
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        # New results
        new_results = {
            "video2.mp4": {"stops": [5.0, 15.0], "duration": 90.0},
            "video3.mp4": {"stops": [], "duration": 60.0},
        }

        merged = merge_results(tmp_path, new_results)

        assert len(merged["videos"]) == 3
        assert "video1.mp4" in merged["videos"]
        assert "video2.mp4" in merged["videos"]
        assert "video3.mp4" in merged["videos"]

    def test_new_result_overwrites_error_entry(self, tmp_path):
        """Retry result should replace error entry."""
        from gomoveshift.resume import merge_results

        # Checkpoint with error
        checkpoint = {"videos": {"video1.mp4": {"error": "Failed previously"}}}
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        # Successful retry
        new_results = {"video1.mp4": {"stops": [10.5], "duration": 120.0}}

        merged = merge_results(tmp_path, new_results)

        assert "error" not in merged["videos"]["video1.mp4"]
        assert merged["videos"]["video1.mp4"]["stops"] == [10.5]

    def test_maintains_original_order_in_output(self, tmp_path):
        """Output should be sorted alphabetically by video name."""
        from gomoveshift.resume import merge_results

        # Checkpoint
        checkpoint = {"videos": {"z_video.mp4": {"stops": [], "duration": 60.0}}}
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        # New results
        new_results = {
            "a_video.mp4": {"stops": [5.0], "duration": 90.0},
            "m_video.mp4": {"stops": [10.0], "duration": 120.0},
        }

        merged = merge_results(tmp_path, new_results)

        keys = list(merged["videos"].keys())
        assert keys == sorted(keys)


class TestCLIIntegration:
    """Tests for CLI resume messaging."""

    def test_prints_resume_message_when_resuming(self, tmp_path, capsys):
        """Should print 'Resuming: X/Y already complete' when resuming."""
        from gomoveshift.resume import print_resume_status

        # Create video files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mp4").touch()
        (tmp_path / "video3.mp4").touch()

        # Checkpoint with 1 complete
        checkpoint = {"videos": {"video1.mp4": {"stops": [10.5], "duration": 120.0}}}
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        print_resume_status(tmp_path, total_videos=3)

        captured = capsys.readouterr()
        assert "Resuming" in captured.out
        assert "1" in captured.out  # 1 already complete
        assert "3" in captured.out  # out of 3 total

    def test_no_resume_message_on_fresh_start(self, tmp_path, capsys):
        """Should not print resume message when starting fresh."""
        from gomoveshift.resume import print_resume_status

        # No checkpoint file
        print_resume_status(tmp_path, total_videos=3)

        captured = capsys.readouterr()
        assert "Resuming" not in captured.out

    def test_force_fresh_flag_ignores_checkpoint(self, tmp_path):
        """--fresh flag should process all videos regardless of checkpoint."""
        from gomoveshift.resume import get_videos_to_process

        # Create video files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mp4").touch()
        (tmp_path / "video3.mp4").touch()

        # Checkpoint with all complete
        checkpoint = {
            "videos": {
                "video1.mp4": {"stops": [10.5], "duration": 120.0},
                "video2.mp4": {"stops": [], "duration": 90.0},
                "video3.mp4": {"stops": [5.0], "duration": 60.0},
            }
        }
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        # With force_fresh=True, should return all videos
        videos = get_videos_to_process(tmp_path, force_fresh=True)

        assert len(videos) == 3


class TestVideoIndexTracking:
    """Tests for correct video index display during resume."""

    def test_returns_start_index_for_display(self, tmp_path):
        """Should return correct starting index for [X/total] display."""
        from gomoveshift.resume import get_resume_start_index

        # Create video files
        (tmp_path / "video1.mp4").touch()
        (tmp_path / "video2.mp4").touch()
        (tmp_path / "video3.mp4").touch()
        (tmp_path / "video4.mp4").touch()
        (tmp_path / "video5.mp4").touch()

        # Checkpoint with first 2 complete
        checkpoint = {
            "videos": {
                "video1.mp4": {"stops": [10.5], "duration": 120.0},
                "video2.mp4": {"stops": [], "duration": 90.0},
            }
        }
        (tmp_path / "stops_report.json").write_text(json.dumps(checkpoint))

        start_index = get_resume_start_index(tmp_path)

        # Should start at index 3 (1-based) since 2 are complete
        assert start_index == 3

    def test_start_index_is_one_on_fresh_start(self, tmp_path):
        """Should return 1 when no checkpoint exists."""
        from gomoveshift.resume import get_resume_start_index

        start_index = get_resume_start_index(tmp_path)

        assert start_index == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
