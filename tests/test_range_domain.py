"""Range-domain correctness across the grade, analysis and selection paths.

Untagged full-range GoPro input is expanded as if 16-235 were legal, clipping
true shadows and highlights on every RGB hop. All three consumers must carry
the same probed tag. Rationale: docs/pipeline-internals.md.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

GRADE = "curves=m='0/0.03 1/0.98',eq=saturation=1.1"
RAW = np.zeros((90, 160, 3), dtype=np.uint8).tobytes() * 3


def _sp():
    from gomoveshift.pipeline.single_pass import SinglePassProcessor

    config = MagicMock()
    config.output.audio_bitrate = "192k"
    return SinglePassProcessor(config)


BASE = dict(
    color_filter=GRADE, location="", font_path="", duration=60.0, fade_video=False, skip_ost=True
)


@pytest.mark.parametrize("rng", ["full", "limited"])
@pytest.mark.parametrize("builder", ["_build_video_filter", "_build_blur_filter_complex"])
def test_grade_chain_tags_probed_range_before_the_grade(builder, rng):
    """Both emitters must tag ahead of the grade, and must never disagree."""
    extra = {"blur_filter": "boxblur=4:2"} if "blur" in builder else {}
    out = getattr(_sp(), builder)(**BASE, source_range=rng, **extra)
    # Ahead of the first grading filter, or the auto-inserted YUV->RGB
    # converter has already applied the wrong assumption.
    assert out.index(f"setrange={rng}") < out.index("curves=")
    assert f"setrange={'limited' if rng == 'full' else 'full'}" not in out


@pytest.mark.parametrize("rng", ["full", "limited"])
@patch("gomoveshift.pipeline.color_analysis.subprocess.Popen")
def test_batch_sampling_tags_range_before_the_rgb_hop(mock_popen, rng):
    from gomoveshift.pipeline.color import sample_frames

    mock_popen.return_value = MagicMock(returncode=0, **{"communicate.return_value": (RAW, b"")})
    sample_frames(Path("/fake/v.mp4"), 30.0, source_range=rng)

    cmd = " ".join(mock_popen.call_args[0][0])
    assert cmd.index(f"setrange={rng}") < cmd.index("rgb24"), cmd


@patch("gomoveshift.pipeline.color_analysis.subprocess.run")
def test_sequential_fallback_tags_range(mock_run):
    from gomoveshift.pipeline.color import _sample_frames_sequential

    mock_run.return_value = MagicMock(returncode=0, stdout=RAW[: 160 * 90 * 3])
    _sample_frames_sequential(Path("/fake/v.mp4"), 30.0, source_range="full")

    cmd = " ".join(mock_run.call_args[0][0])
    assert cmd.index("setrange=full") < cmd.index("rgb24"), cmd


@patch("gomoveshift.pipeline.color_analysis._sample_frames_sequential")
@patch("gomoveshift.pipeline.color_analysis.subprocess.Popen")
def test_batch_failure_forwards_range_to_the_fallback(mock_popen, mock_seq):
    """A dropped range on the fallback path is the same bug, resurrected."""
    from gomoveshift.pipeline.color import sample_frames

    mock_popen.side_effect = Exception("batch failed")
    mock_seq.return_value = []
    sample_frames(Path("/fake/v.mp4"), 30.0, source_range="limited")

    assert mock_seq.call_args.kwargs.get("source_range") == "limited"


@pytest.mark.parametrize(
    "probe_rc,probe_out,expected",
    [
        (0, "pc\n60.0\n", "full"),
        (0, "tv\n60.0\n", "limited"),
        (0, "\n60.0\n", "full"),  # untagged -> documented GoPro default
        (1, "", "full"),  # unreadable probe must not abort scoring
    ],
)
@patch("gomoveshift.analysis.unified_scorer.subprocess.run")
def test_selection_extraction_tags_probed_range(mock_run, probe_rc, probe_out, expected):
    """drive_scores.npz decides which footage ships — same domain, same fix."""
    from gomoveshift.analysis.unified_scorer import UnifiedSegmentScorer

    frame = np.zeros((8, 8, 3), dtype=np.uint8).tobytes()
    mock_run.side_effect = lambda cmd, *a, **kw: (
        MagicMock(returncode=probe_rc, stdout=probe_out)
        if cmd[0] == "ffprobe"
        else MagicMock(returncode=0, stdout=frame * 2)
    )

    scorer = UnifiedSegmentScorer({"motion_energy": 1.0})
    scorer._extract_frames_bgr(Path("/fake/v.mp4"), 1.0, width=8, height=8)

    cmds = [" ".join(c.args[0]) for c in mock_run.call_args_list if c.args[0][0] == "ffmpeg"]
    assert cmds, "no ffmpeg extraction command was issued"
    # Both the GPU command and its CPU fallback must carry the tag.
    for cmd in cmds:
        assert cmd.index(f"setrange={expected}") < cmd.index("bgr24"), cmd


def test_one_shared_range_token_mapper():
    """Two definitions of "full" is how the three sites drift apart again."""
    from gomoveshift.pipeline.color_analysis import scale_range_token
    from gomoveshift.pipeline.single_pass import _scale_range_token

    assert _scale_range_token is scale_range_token


@patch("gomoveshift.pipeline.single_pass.sample_frames")
def test_analyze_color_threads_probed_range_into_sampling(mock_sample):
    """The measurement must reuse the probe the encode already ran."""
    mock_sample.return_value = []
    p = _sp()
    with patch.object(p, "_get_video_info", return_value={"duration": 30.0, "color_range": "tv"}):
        with patch("gomoveshift.pipeline.single_pass.is_analysis_cached", return_value=False):
            p._analyze_color(Path("/fake/v.mp4"))

    assert mock_sample.call_args.kwargs.get("source_range") == "limited"
