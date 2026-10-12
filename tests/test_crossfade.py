"""Tests for crossfade transitions at stop-removal cut points.

crossfade_duration>0 → a SINGLE xfade+acrossfade encode over all segments (the
old per-pair loop kept only the last transition and collapsed the output);
crossfade_duration=0 or a too-short segment → hard-cut concat. FFmpeg mocked.
"""

from unittest.mock import patch

import pytest

from gomoveshift.config import Config
from gomoveshift.pipeline import encoding_constants as ec
from gomoveshift.pipeline.stop_processor import StopProcessor, build_stop_crossfade_filter


@pytest.fixture
def config():
    return Config()


def test_build_filter_preserves_every_body_and_total_duration():
    """Σ durations − (N−1)·crossfade — every segment body survives the stitch."""
    fc, total = build_stop_crossfade_filter([10.0, 10.0, 10.0], 0.5)
    assert total == pytest.approx(29.0)  # 30 − 2·0.5
    assert fc.count("xfade=transition=fade") == 2
    assert "offset=9.500000" in fc and "offset=19.000000" in fc
    assert fc.count("acrossfade=d=0.5") == 2
    assert "[v]" in fc and "[aud]" in fc


def test_build_filter_rejects_single_segment():
    with pytest.raises(ValueError):
        build_stop_crossfade_filter([10.0], 0.5)


class TestCrossfadeDisabled:
    """When crossfade_duration=0, hard cuts are used (no xfade calls)."""

    @patch("gomoveshift.pipeline.stop_processor.detect_nvenc", return_value=False)
    def test_hard_cut_no_xfade_calls(self, mock_nvenc, config, tmp_path):
        """crossfade_duration=0 should use concat demuxer, not xfade."""
        proc = StopProcessor(config, crossfade_duration=0)
        output = tmp_path / "output.mp4"
        segments = [(0, 10), (20, 30), (40, 50)]

        with patch.object(proc, "_run_ffmpeg") as mock_run:
            proc._cut_and_stitch(tmp_path / "input.mp4", output, segments)

        # Should have segment extractions + 1 concat call, no xfade
        descriptions = [c.args[1] for c in mock_run.call_args_list]
        assert any("concat" in d for d in descriptions)
        assert not any("crossfade" in d for d in descriptions)


class TestSingleSegment:
    """Single segment bypasses both crossfade and concat."""

    @patch("gomoveshift.pipeline.stop_processor.detect_nvenc", return_value=False)
    def test_single_segment_trim_only(self, mock_nvenc, config, tmp_path):
        """Single segment should do a simple trim, no concat or xfade."""
        proc = StopProcessor(config, crossfade_duration=0.5)
        output = tmp_path / "output.mp4"
        segments = [(5, 15)]

        with patch.object(proc, "_run_ffmpeg") as mock_run:
            proc._cut_and_stitch(tmp_path / "input.mp4", output, segments)

        # Only one call: trim single segment
        assert mock_run.call_count == 1
        desc = mock_run.call_args_list[0].args[1]
        assert "trim" in desc or "single" in desc.lower()


class TestCrossfadeEnabled:
    """crossfade_duration > 0 with multiple segments → ONE xfade-stitch encode."""

    @patch("gomoveshift.pipeline.stop_processor.detect_nvenc", return_value=False)
    def test_three_segments_single_stitch_encode(self, mock_nvenc, config, tmp_path):
        proc = StopProcessor(config, crossfade_duration=0.5)
        with (
            patch.object(proc, "_run_ffmpeg") as mock_run,
            patch.object(proc, "_get_duration", return_value=10.0),
        ):
            proc._cut_and_stitch(
                tmp_path / "in.mp4", tmp_path / "o.mp4", [(0, 10), (20, 30), (40, 50)]
            )
        desc = [c.args[1] for c in mock_run.call_args_list]
        assert len([d for d in desc if "stream-copy" in d]) == 3
        assert len([d for d in desc if "crossfade stitch" in d]) == 1

    @patch("gomoveshift.pipeline.stop_processor.detect_nvenc", return_value=False)
    def test_stitch_is_single_pass_filter_complex(self, mock_nvenc, config, tmp_path):
        proc = StopProcessor(config, crossfade_duration=0.5)
        cmds = []
        with (
            patch.object(
                proc,
                "_run_ffmpeg",
                side_effect=lambda c, d, timeout=0: (
                    cmds.append(c) if "crossfade stitch" in d else None
                ),
            ),
            patch.object(proc, "_get_duration", return_value=10.0),
        ):
            proc._cut_and_stitch(tmp_path / "in.mp4", tmp_path / "o.mp4", [(0, 10), (20, 30)])
        assert len(cmds) == 1 and cmds[0].count("-i") == 2 and "-sseof" not in cmds[0]
        assert "hevc_nvenc" in cmds[0]
        fc = cmds[0][cmds[0].index("-filter_complex") + 1]
        assert "xfade=transition=fade" in fc and "acrossfade" in fc

    @patch("gomoveshift.pipeline.stop_processor.detect_nvenc", return_value=False)
    def test_stitch_encodes_at_the_frozen_nvenc_quality(
        self, mock_nvenc, config, tmp_path, monkeypatch
    ):
        """The stitch is a full extra generation on the way to the master, so it
        must read the frozen NVENC preset/CQ/profile — not a hardcoded preset and
        not the x265 CRF constant (which only matches the NVENC CQ by numeric
        coincidence today). Re-pointing the constants must move the emitted args.
        """
        monkeypatch.setattr(ec, "LONGFORM_NVENC_PRESET", "p6")
        monkeypatch.setattr(ec, "LONGFORM_NVENC_CQ", 21)
        monkeypatch.setattr(ec, "LONGFORM_CRF", 99)  # x265 knob — must NOT be used here
        proc = StopProcessor(config, crossfade_duration=0.5)
        cmds = []
        with (
            patch.object(
                proc,
                "_run_ffmpeg",
                side_effect=lambda c, d, timeout=0: (
                    cmds.append(c) if "crossfade stitch" in d else None
                ),
            ),
            patch.object(proc, "_get_duration", return_value=10.0),
        ):
            proc._cut_and_stitch(tmp_path / "in.mp4", tmp_path / "o.mp4", [(0, 10), (20, 30)])
        cmd = cmds[0]
        assert cmd[cmd.index("-preset") + 1] == "p6"
        assert cmd[cmd.index("-cq") + 1] == "21"
        assert cmd[cmd.index("-profile:v") + 1] == ec.LONGFORM_PROFILE
        assert "99" not in cmd

    @patch("gomoveshift.pipeline.stop_processor.detect_nvenc", return_value=False)
    def test_short_segment_falls_back_to_hard_cut(self, mock_nvenc, config, tmp_path):
        """A kept segment shorter than the crossfade falls back to hard cuts
        (a negative xfade offset would otherwise be emitted)."""
        proc = StopProcessor(config, crossfade_duration=0.5)
        with (
            patch.object(proc, "_run_ffmpeg") as mock_run,
            patch.object(proc, "_get_duration", return_value=0.3),
        ):
            proc._cut_and_stitch(tmp_path / "in.mp4", tmp_path / "o.mp4", [(0, 10), (20, 30)])
        desc = [c.args[1] for c in mock_run.call_args_list]
        assert any("concat" in d for d in desc) and not any("crossfade stitch" in d for d in desc)
