"""LF sharpen / denoise are measured per day master, not hard-set to 0.0."""

import cv2
import numpy as np

from gomoveshift.pipeline.color_analysis import SegmentStats, analyze_frame
from gomoveshift.pipeline.color_correction import CorrectionParams, compute_correction
from gomoveshift.pipeline.color_filters import build_filter_chain
from gomoveshift.pipeline.quality import (
    denoise_strength_for,
    sharpen_amount_for,
)

# Detail scores (variance of Laplacian, 8-bit levels, bounded view):
# log10 0.5 = 3.2 (floor), log10 3.0 = 1000 (crisp).
SOFT, MID, CRISP, DEAD = 10.0, 100.0, 5000.0, 0.0


def _chart(h=270, w=480) -> np.ndarray:
    rng = np.random.default_rng(7)
    base = rng.integers(0, 256, (h // 6, w // 6), dtype=np.uint8)
    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.dstack([img, img, img])


def _stats(**kw) -> SegmentStats:
    base = dict(black_point=60, white_point=230, mid_tone=128, saturation=0.3)
    base.update(kw)
    return SegmentStats(**base)


def test_lf_sharpen_amount_follows_measured_detail():
    soft = compute_correction(_stats(detail_score=MID))
    crisp = compute_correction(_stats(detail_score=CRISP))
    flagged = compute_correction(_stats(detail_score=MID), no_sharpen=True)
    assert soft.sharpen_amount > 0.0
    assert soft.sharpen_radius > 0.0
    assert crisp.sharpen_amount == 0.0
    assert flagged.sharpen_amount == 0.0
    assert flagged.sharpen_disabled is True
    assert soft.sharpen_disabled is False


def test_sharpen_ramp_zero_at_both_ends_and_monotone():
    assert sharpen_amount_for(CRISP, 1920, disabled=False)[0] == 0.0
    assert sharpen_amount_for(DEAD, 1920, disabled=False)[0] == 0.0
    a_soft = sharpen_amount_for(SOFT, 1920, disabled=False)[0]
    a_mid = sharpen_amount_for(MID, 1920, disabled=False)[0]
    assert a_soft > a_mid > 0.0


def test_sharpen_radius_scales_with_width():
    narrow = sharpen_amount_for(MID, 1920, disabled=False)[1]
    wide = sharpen_amount_for(MID, 3840, disabled=False)[1]
    assert wide == narrow * 2


def test_denoise_follows_measured_noise():
    noisy = compute_correction(_stats(noise_score=6.0))
    clean = compute_correction(_stats(noise_score=0.5))
    flagged = compute_correction(_stats(noise_score=6.0), no_denoise=True)
    assert noisy.denoise_strength > 0.0
    assert clean.denoise_strength == 0.0
    assert flagged.denoise_strength == 0.0
    assert flagged.denoise_disabled is True
    assert denoise_strength_for(6.0, disabled=False) > denoise_strength_for(3.0, disabled=False)


def test_no_denoise_also_overrides_black_point_ramp():
    dark = compute_correction(_stats(black_point=2), no_denoise=True)
    assert dark.denoise_strength == 0.0


def test_unmeasured_stats_leave_both_off():
    p = compute_correction(_stats(black_point=60))
    assert p.sharpen_amount == 0.0
    assert p.denoise_strength == 0.0


def test_photo_mode_does_not_double_sharpen():
    p = compute_correction(_stats(detail_score=MID), mode="photo")
    assert p.sharpen_amount == 0.0


def test_measurement_orders_blur_and_noise():
    sharp = _chart()
    blurred = cv2.GaussianBlur(sharp, (0, 0), 2.5)
    rng = np.random.default_rng(1)
    flat = np.full((270, 480, 3), 120, dtype=np.uint8)
    noisy = np.clip(flat + rng.normal(0, 8, flat.shape), 0, 255).astype(np.uint8)
    assert analyze_frame(sharp).detail_score > analyze_frame(blurred).detail_score
    assert analyze_frame(noisy).noise_score > analyze_frame(flat).noise_score
    assert analyze_frame(flat).noise_score < 0.5


def test_unsharp_radius_reaches_filter_chain():
    chain = build_filter_chain([CorrectionParams(sharpen_amount=0.5, sharpen_radius=2.4)])
    assert "unsharp=11:11:0.50" in chain
    legacy = build_filter_chain([CorrectionParams(sharpen_amount=0.5)])
    assert "unsharp=5:5:0.50" in legacy


# ── drive-level wiring ───────────────────────────────────────────────


def _analyze_with_fakes(tmp_path, *, width, no_sharpen, no_denoise, adaptive=False):
    from unittest.mock import patch

    from gomoveshift.config import Config
    from gomoveshift.pipeline import single_pass as sp

    cfg = Config()
    cfg.drive.no_sharpen = no_sharpen
    cfg.drive.no_denoise = no_denoise
    proc = sp.SinglePassProcessor(cfg)
    info = {"duration": 60, "height": 1080, "fps": 30.0}
    if width is not None:
        info["width"] = width
    seg = _stats(detail_score=MID, noise_score=6.0)
    with (
        patch.object(sp, "is_analysis_cached", return_value=False),
        patch.object(proc, "_get_video_info", return_value=info),
        patch.object(sp, "sample_frames", return_value=[np.zeros((4, 4, 3), np.uint8)]),
        patch.object(sp, "analyze_segments", return_value=([seg], adaptive)),
        patch.object(sp, "merge_adjacent_segments", return_value=([seg], [60.0])),
        patch.object(sp, "compute_correction", return_value=CorrectionParams()) as cc,
        patch.object(proc, "_corrections_to_filter", return_value=("null", "global", None)),
        patch.object(sp, "save_cache_json"),
    ):
        proc._analyze_color(tmp_path / "v.mp4")
    return cc


def test_analyze_color_passes_width_and_opt_outs(tmp_path):
    cc = _analyze_with_fakes(tmp_path, width=3840, no_sharpen=True, no_denoise=True)
    kw = cc.call_args.kwargs
    assert (kw["frame_width"], kw["no_sharpen"], kw["no_denoise"]) == (3840, True, True)


def test_analyze_color_adaptive_branch_passes_kwargs(tmp_path):
    cc = _analyze_with_fakes(tmp_path, width=3840, no_sharpen=True, no_denoise=False, adaptive=True)
    kw = cc.call_args.kwargs
    assert (kw["frame_width"], kw["no_sharpen"], kw["no_denoise"]) == (3840, True, False)


def test_analyze_color_global_branch_carries_measured_scores(tmp_path):
    cc = _analyze_with_fakes(tmp_path, width=1920, no_sharpen=False, no_denoise=False)
    stats = cc.call_args.args[0]
    assert stats.detail_score == MID and stats.noise_score == 6.0


def test_analyze_color_missing_width_falls_back_to_1920(tmp_path):
    cc = _analyze_with_fakes(tmp_path, width=None, no_sharpen=False, no_denoise=False)
    assert cc.call_args.kwargs["frame_width"] == 1920


def test_drive_parser_opt_out_flags_land_on_config():
    import argparse
    from unittest.mock import patch

    from gomoveshift.config import Config

    with patch("importlib.metadata.version", return_value="0"):
        from gomoveshift.cli.parser import _build_parser

        parser = _build_parser()
    args = parser.parse_args(["drive", "/tmp/src", "--no-sharpen", "--no-denoise"])
    assert isinstance(args, argparse.Namespace)
    assert args.no_sharpen is True and args.no_denoise is True
    default = parser.parse_args(["drive", "/tmp/src"])
    assert default.no_sharpen is False and default.no_denoise is False
    cfg = Config()
    assert cfg.drive.no_sharpen is False and cfg.drive.no_denoise is False


def test_cache_with_different_opt_out_is_not_reused(tmp_path):
    from unittest.mock import patch

    from gomoveshift.config import Config
    from gomoveshift.pipeline import single_pass as sp

    cfg = Config()
    cfg.drive.no_sharpen = True
    proc = sp.SinglePassProcessor(cfg)
    cached = {
        "version": sp.TRANSIENT_COLOR_CACHE_SCHEMA_VERSION,
        "corrections": [{"sharpen_amount": 0.4}],
    }
    with (
        patch.object(sp, "is_analysis_cached", return_value=True),
        patch.object(sp, "load_cached_json", return_value=cached),
        patch.object(proc, "_get_video_info", side_effect=RuntimeError("reanalyze")),
    ):
        try:
            proc._analyze_color(tmp_path / "v.mp4")
        except RuntimeError as e:
            assert str(e) == "reanalyze"
        else:
            raise AssertionError("stale cache reused despite --no-sharpen")
