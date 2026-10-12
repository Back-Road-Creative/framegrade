"""Color processing pipeline — facade re-exporting sub-module functions.

Split from a 1,671-line god file into:
  - color_analysis.py — frame sampling, histogram, segment stats
  - color_correction.py — parameter computation, LUT generation
  - color_filters.py — ffmpeg filter strings, encoding args
"""

from .color_analysis import (  # noqa: F401
    SAMPLE_INTERVAL_SEC,
    SEGMENT_DURATION_SEC,
    FrameStats,
    SegmentStats,
    _sample_frames_sequential,
    aggregate_segment_stats,
    analyze_frame,
    analyze_segments,
    cluster_files_by_lighting,
    compute_variance,
    get_video_properties,
    merge_adjacent_segments,
    sample_frames,
    scale_range_token,
    should_use_adaptive,
)
from .color_correction import (  # noqa: F401
    CorrectionParams,
    _apply_colorbalance,
    _apply_curves,
    _apply_gamma,
    _apply_saturation,
    _average_params,
    _monotone_cubic_interp,
    compute_cluster_correction,
    compute_correction,
    params_to_lut3d_cube,
)
from .color_filters import (  # noqa: F401
    _build_single_filter,
    _curves_point,
    _nvenc_args,
    _x265_args,
    build_filter_chain,
    detect_nvenc,
    encode_video,
    get_encoding_args,
)

__all__ = [
    # color_analysis
    "SAMPLE_INTERVAL_SEC",
    "SEGMENT_DURATION_SEC",
    "FrameStats",
    "SegmentStats",
    "aggregate_segment_stats",
    "analyze_frame",
    "analyze_segments",
    "cluster_files_by_lighting",
    "compute_variance",
    "get_video_properties",
    "merge_adjacent_segments",
    "sample_frames",
    "scale_range_token",
    "should_use_adaptive",
    # color_correction
    "CorrectionParams",
    "compute_cluster_correction",
    "compute_correction",
    "params_to_lut3d_cube",
    # color_filters
    "build_filter_chain",
    "detect_nvenc",
    "encode_video",
    "get_encoding_args",
]
