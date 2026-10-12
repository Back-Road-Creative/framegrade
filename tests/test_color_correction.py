"""
Tests for adaptive color correction with histogram analysis.

TDD: Write tests first, verify they fail, then implement.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Condition-fixture grading pins (night / tunnel / backlit) ─────────
#
# Synthetic frames driven through the WHOLE analyze_segments -> compute_correction
# pipeline (not hand-built SegmentStats), pinning that each named lighting
# condition grades inside its documented caps: shadows lifted but never past the
# cap, no S-curve fighting an underexposed lift, a strong cast corrected toward
# neutral without being amplified past the ±clamp, and a blown sky metered as
# foreground rather than pulled to the highlight rail. Regression pins on
# already-correct behaviour — they pass on today's source.

_FIXTURE_H, _FIXTURE_W = 90, 160


def _night_frames(n=12):
    """Deep-night footage: near-black, neutral, narrow low spread."""
    rng = np.random.default_rng(1)
    frames = []
    for _ in range(n):
        base = rng.integers(2, 30, size=(_FIXTURE_H, _FIXTURE_W, 1))
        rgb = np.repeat(base, 3, axis=2) + rng.integers(-3, 4, size=(_FIXTURE_H, _FIXTURE_W, 3))
        frames.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return frames


def _tunnel_frames(n=12):
    """Sodium-lit tunnel: dark, with a strong warm cast (R > G > B)."""
    rng = np.random.default_rng(2)
    frames = []
    for _ in range(n):
        r = rng.integers(35, 60, size=(_FIXTURE_H, _FIXTURE_W))
        g = rng.integers(18, 34, size=(_FIXTURE_H, _FIXTURE_W))
        b = rng.integers(6, 16, size=(_FIXTURE_H, _FIXTURE_W))
        frames.append(np.stack([r, g, b], axis=-1).astype(np.uint8))
    return frames


def _backlit_frames(n=12):
    """Backlit: blown sky filling the upper third, dark foreground below."""
    rng = np.random.default_rng(3)
    top = int(_FIXTURE_H * 0.33)
    frames = []
    for _ in range(n):
        frame = np.empty((_FIXTURE_H, _FIXTURE_W, 3), dtype=np.int64)
        frame[:top] = rng.integers(245, 256, size=(top, _FIXTURE_W, 3))
        frame[top:] = rng.integers(40, 95, size=(_FIXTURE_H - top, _FIXTURE_W, 3))
        frames.append(frame.astype(np.uint8))
    return frames


class TestConditionFixtureGrading:
    """Night / tunnel / backlit footage graded end-to-end within documented caps."""

    def _segments(self, frames):
        from gomoveshift.pipeline.color import analyze_segments

        segments, _ = analyze_segments(frames)
        assert segments, "fixture must yield at least one segment"
        return segments

    def _assert_within_documented_caps(self, p):
        from gomoveshift.pipeline.color_correction import (
            CAST_CORRECTABLE_MAX,
            CONTRAST_CURVE_MAX,
            DEHAZE_CONTRAST_MAX,
            DENOISE_MAX_STRENGTH,
            GAMMA_MAX,
        )

        assert 2.0 <= p.shadow_lift <= 20.0
        assert 2.0 <= p.highlight_pull <= 15.0
        assert 0.80 <= p.saturation_mult <= 1.45
        assert 0.0 <= p.contrast_curve <= CONTRAST_CURVE_MAX + DEHAZE_CONTRAST_MAX
        assert abs(p.white_balance_r) <= CAST_CORRECTABLE_MAX
        assert abs(p.white_balance_b) <= CAST_CORRECTABLE_MAX
        assert abs(p.white_balance_r_high) <= CAST_CORRECTABLE_MAX
        assert abs(p.white_balance_b_high) <= CAST_CORRECTABLE_MAX
        assert 0.9 <= p.gamma <= GAMMA_MAX
        assert 0.0 <= p.denoise_strength <= DENOISE_MAX_STRENGTH

    def test_night_lifts_shadows_and_gamma_within_caps(self):
        from gomoveshift.pipeline.color_correction import (
            DENOISE_MAX_STRENGTH,
            GAMMA_MAX,
            compute_correction,
        )

        for seg in self._segments(_night_frames()):
            p = compute_correction(seg)
            self._assert_within_documented_caps(p)
            # Deep night lifts shadows hard (no crushed blacks), capped at 20.
            assert p.shadow_lift > 15.0
            # Dark midtones lifted, bounded by GAMMA_MAX.
            assert 1.0 < p.gamma <= GAMMA_MAX
            # Full temporal denoise for the near-black frame.
            assert p.denoise_strength == DENOISE_MAX_STRENGTH
            # No S-curve: underexposure is not flatness, so it must not fight the lift.
            assert p.contrast_curve == 0.0

    def test_tunnel_corrects_warm_cast_without_amplifying_it(self):
        from gomoveshift.pipeline.color_correction import (
            CAST_CORRECTABLE_MAX,
            compute_correction,
        )

        for seg in self._segments(_tunnel_frames()):
            p = compute_correction(seg)
            self._assert_within_documented_caps(p)
            # Warm sodium cast pulled toward neutral (negative red shift) ...
            assert p.white_balance_r < 0.0
            # ... but never past the ±clamp colorbalance can actually honour.
            assert abs(p.white_balance_r) <= CAST_CORRECTABLE_MAX
            assert abs(p.white_balance_b) <= CAST_CORRECTABLE_MAX
            # The dark tunnel still lifts shadows; no S-curve on an underexposed frame.
            assert p.shadow_lift > 10.0
            assert p.contrast_curve == 0.0

    def test_backlit_meters_foreground_not_the_blown_sky(self):
        from gomoveshift.pipeline.color_correction import compute_correction

        for seg in self._segments(_backlit_frames()):
            # The blown upper-band sky is detected and down-weighted, not graded.
            assert seg.sky_fraction > 0.0
            p = compute_correction(seg)
            self._assert_within_documented_caps(p)
            # Highlight pull stays moderate — the sky is excluded, so the metered
            # white point never reaches the rail that would clamp pull to 15.
            assert p.highlight_pull < 12.0
            # The dark foreground is lifted, not crushed.
            assert p.shadow_lift > 5.0
            assert p.gamma > 1.0


# ── Histogram Analysis Tests ─────────────────────────────────────────


class TestHistogramAnalysis:
    """Test luminance and chroma histogram analysis from sampled frames."""

    def test_analyze_frame_returns_stats(self):
        """A single frame analysis should return black_point, white_point, mid_tone, saturation."""
        from gomoveshift.pipeline.color import FrameStats, analyze_frame

        # Simulate a well-exposed frame (values spread 20-235)
        frame = np.random.randint(20, 235, size=(240, 320, 3), dtype=np.uint8)
        stats = analyze_frame(frame)

        assert isinstance(stats, FrameStats)
        assert 0 <= stats.black_point <= 255
        assert 0 <= stats.white_point <= 255
        assert stats.black_point < stats.white_point
        assert 0 <= stats.mid_tone <= 255
        assert stats.saturation >= 0

    def test_crushed_blacks_detected(self):
        """Frame with crushed blacks should have very low black_point (near 0)."""
        from gomoveshift.pipeline.color import analyze_frame

        # Dark frame with values clustered 0-30
        frame = np.random.randint(0, 30, size=(240, 320, 3), dtype=np.uint8)
        stats = analyze_frame(frame)

        assert stats.black_point < 10, "Should detect crushed blacks near 0"

    def test_clipped_highlights_detected(self):
        """Frame with clipped highlights should have white_point near 255."""
        from gomoveshift.pipeline.color import analyze_frame

        # Bright frame with values clustered 220-255
        frame = np.random.randint(220, 256, size=(240, 320, 3), dtype=np.uint8)
        stats = analyze_frame(frame)

        assert stats.white_point > 245, "Should detect clipped highlights near 255"

    def test_well_exposed_frame_moderate_points(self):
        """Well-exposed frame should have moderate black/white points."""
        from gomoveshift.pipeline.color import analyze_frame

        # Normal exposure: bell curve centered around 128
        lum = np.clip(np.random.normal(128, 40, size=(240, 320)), 0, 255).astype(np.uint8)
        frame = np.stack([lum, lum, lum], axis=-1)
        stats = analyze_frame(frame)

        assert stats.black_point > 20, "Well-exposed should not have crushed blacks"
        assert stats.white_point < 240, "Well-exposed should not have clipped highlights"
        assert 80 < stats.mid_tone < 180, "Mid-tone should be around 128"

    def test_saturation_measurement(self):
        """High saturation frame should report higher saturation than grayscale."""
        from gomoveshift.pipeline.color import analyze_frame

        # Grayscale frame
        gray = np.full((240, 320, 3), 128, dtype=np.uint8)
        gray_stats = analyze_frame(gray)

        # Saturated frame (strong red)
        saturated = np.zeros((240, 320, 3), dtype=np.uint8)
        saturated[:, :, 0] = 200  # R
        saturated[:, :, 1] = 50  # G
        saturated[:, :, 2] = 50  # B
        sat_stats = analyze_frame(saturated)

        assert sat_stats.saturation > gray_stats.saturation, (
            "Colored frame should have higher saturation than grayscale"
        )


# ── Segment Analysis Tests ───────────────────────────────────────────


class TestSegmentAnalysis:
    """Test per-segment histogram analysis and variance detection."""

    def test_segment_stats_aggregation(self):
        """Multiple frame stats should aggregate into a segment summary."""
        from gomoveshift.pipeline.color import FrameStats, aggregate_segment_stats

        frames = [
            FrameStats(black_point=10, white_point=240, mid_tone=125, saturation=0.3),
            FrameStats(black_point=15, white_point=235, mid_tone=130, saturation=0.35),
            FrameStats(black_point=12, white_point=238, mid_tone=128, saturation=0.32),
        ]
        segment = aggregate_segment_stats(frames)

        # Should be median/mean of frame values
        assert 10 <= segment.black_point <= 15
        assert 235 <= segment.white_point <= 240
        assert 125 <= segment.mid_tone <= 130
        assert 0.3 <= segment.saturation <= 0.35

    def test_variance_detection_stable(self):
        """Stable lighting across segments should report low variance."""
        from gomoveshift.pipeline.color import SegmentStats, compute_variance

        segments = [
            SegmentStats(black_point=12, white_point=238, mid_tone=128, saturation=0.3),
            SegmentStats(black_point=14, white_point=236, mid_tone=130, saturation=0.32),
            SegmentStats(black_point=11, white_point=237, mid_tone=127, saturation=0.31),
        ]
        variance = compute_variance(segments)

        assert variance < 0.3, "Stable lighting should have low variance"

    def test_variance_detection_variable(self):
        """Highly variable lighting should report high variance."""
        from gomoveshift.pipeline.color import SegmentStats, compute_variance

        segments = [
            SegmentStats(
                black_point=5, white_point=250, mid_tone=80, saturation=0.2
            ),  # dark tunnel
            SegmentStats(
                black_point=40, white_point=200, mid_tone=160, saturation=0.5
            ),  # bright sun
            SegmentStats(black_point=10, white_point=245, mid_tone=100, saturation=0.25),  # shadow
        ]
        variance = compute_variance(segments)

        assert variance > 0.3, "Variable lighting should have high variance"

    def test_global_vs_adaptive_decision(self):
        """Low variance → global correction, high variance → adaptive."""
        from gomoveshift.pipeline.color import should_use_adaptive

        assert should_use_adaptive(0.1) is False, "Low variance → global"
        assert should_use_adaptive(0.5) is True, "High variance → adaptive"


class TestChromaVariance:
    """Chroma (colour-cast) swings must trip adaptive mode, not just luma swings.

    A day that runs overcast → golden hour, or in and out of sodium-lit
    tunnels, holds steady luminance while the white balance swings hard. A
    luma-only variance score leaves such a day in single-correction mode and
    ships ONE averaged white balance across visibly different lighting.
    """

    LUMA = {"black_point": 12.0, "white_point": 238.0, "mid_tone": 128.0, "saturation": 0.3}

    def test_chroma_swing_at_stable_luma_trips_adaptive(self):
        """Overcast ⇄ golden-hour cast at identical luma must reach the threshold."""
        from gomoveshift.pipeline.color import SegmentStats, compute_variance, should_use_adaptive

        # Identical luminance in every segment — the ONLY thing moving is cast.
        cool = SegmentStats(**self.LUMA, r_mean=0.47, g_mean=0.50, b_mean=0.54)
        warm = SegmentStats(**self.LUMA, r_mean=0.58, g_mean=0.50, b_mean=0.40)
        segments = [cool, cool, warm, warm]

        variance = compute_variance(segments)

        assert should_use_adaptive(variance), (
            f"colour-temperature swing at stable luma must trip adaptive (got {variance:.3f})"
        )

    def test_luma_only_day_still_trips(self):
        """Regression: a day that trips on luma today must keep tripping."""
        from gomoveshift.pipeline.color import SegmentStats, compute_variance

        segments = [
            SegmentStats(black_point=5, white_point=250, mid_tone=80, saturation=0.2),
            SegmentStats(black_point=40, white_point=200, mid_tone=160, saturation=0.5),
            SegmentStats(black_point=10, white_point=245, mid_tone=100, saturation=0.25),
        ]

        assert compute_variance(segments) > 0.3

    def test_chroma_term_never_lowers_the_score(self):
        """The chroma term is additive — it can only raise an existing luma score."""
        from gomoveshift.pipeline.color import SegmentStats, compute_variance

        luma_only = [
            SegmentStats(black_point=5, white_point=250, mid_tone=80, saturation=0.2),
            SegmentStats(black_point=40, white_point=200, mid_tone=160, saturation=0.5),
        ]
        with_cast = [
            SegmentStats(
                black_point=s.black_point,
                white_point=s.white_point,
                mid_tone=s.mid_tone,
                saturation=s.saturation,
                r_mean=r,
                g_mean=0.50,
                b_mean=b,
            )
            for s, r, b in zip(luma_only, (0.47, 0.58), (0.54, 0.40))
        ]

        assert compute_variance(with_cast) >= compute_variance(luma_only)

    def test_uniform_day_does_not_trip(self):
        """False-positive guard: steady light + ordinary chroma noise stays global."""
        from gomoveshift.pipeline.color import SegmentStats, compute_variance, should_use_adaptive

        rng = np.random.default_rng(7)
        segments = []
        for _ in range(40):
            g = 0.50 + float(rng.normal(0, 0.03))
            segments.append(
                SegmentStats(
                    black_point=12.0 + float(rng.normal(0, 1.0)),
                    white_point=238.0 + float(rng.normal(0, 1.5)),
                    mid_tone=128.0 + float(rng.normal(0, 2.0)),
                    saturation=0.3,
                    r_mean=g + float(rng.normal(0, 0.008)),
                    g_mean=g,
                    b_mean=g + float(rng.normal(0, 0.010)),
                )
            )

        variance = compute_variance(segments)

        assert not should_use_adaptive(variance), (
            f"uniform day must stay in global mode (got {variance:.3f})"
        )

    def test_scene_content_variation_alone_does_not_trip(self):
        """Foliage-vs-road content swing under one illuminant must not flip modes."""
        from gomoveshift.pipeline.color import SegmentStats, compute_variance, should_use_adaptive

        road = SegmentStats(**self.LUMA, r_mean=0.44, g_mean=0.47, b_mean=0.50)
        green = SegmentStats(**self.LUMA, r_mean=0.40, g_mean=0.45, b_mean=0.44)

        variance = compute_variance([road, green] * 20)

        assert not should_use_adaptive(variance), (
            f"scene-content chroma drift alone must not trip adaptive (got {variance:.3f})"
        )


# ── Correction Parameter Tests ───────────────────────────────────────


class TestCorrectionParams:
    """Test generation of correction parameters from histogram stats."""

    def test_shadow_lift_from_crushed_blacks(self):
        """Crushed blacks should produce a shadow lift."""
        from gomoveshift.pipeline.color import CorrectionParams, SegmentStats, compute_correction

        stats = SegmentStats(black_point=3, white_point=238, mid_tone=128, saturation=0.3)
        params = compute_correction(stats)

        assert isinstance(params, CorrectionParams)
        assert params.shadow_lift > 0, "Crushed blacks should lift shadows"
        assert params.shadow_lift <= 20, "Shadow lift should be gentle (max 20 levels)"

    def test_highlight_pull_from_clipped_whites(self):
        """Clipped highlights should produce a highlight pull."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(black_point=20, white_point=252, mid_tone=128, saturation=0.3)
        params = compute_correction(stats)

        assert params.highlight_pull > 0, "Clipped whites should pull highlights"
        assert params.highlight_pull <= 20, "Highlight pull should be gentle (max 20 levels)"

    def test_well_exposed_moderate_correction(self):
        """Well-exposed footage gets adaptive range-conversion compensation."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(black_point=25, white_point=230, mid_tone=128, saturation=0.35)
        params = compute_correction(stats)

        # Shadow lift scales with black_point: 15 - 25*0.15 = 11.25 (slope 0.15)
        assert 8 <= params.shadow_lift <= 12, f"Expected ~11.25, got {params.shadow_lift}"
        # Highlight pull scales with white_point: (230-170)*0.2 = 12
        assert 10 <= params.highlight_pull <= 14, f"Expected ~12, got {params.highlight_pull}"

    def test_saturation_adjustment_bounds(self):
        """Saturation adjustment should stay within 1.0-1.25 range."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        # Low saturation
        low_sat = SegmentStats(black_point=20, white_point=230, mid_tone=128, saturation=0.15)
        params_low = compute_correction(low_sat)
        assert 1.0 <= params_low.saturation_mult <= 1.25

        # Normal saturation
        normal = SegmentStats(black_point=20, white_point=230, mid_tone=128, saturation=0.35)
        params_normal = compute_correction(normal)
        assert 1.0 <= params_normal.saturation_mult <= 1.25

        # High saturation — should NOT boost further
        high_sat = SegmentStats(black_point=20, white_point=230, mid_tone=128, saturation=0.6)
        params_high = compute_correction(high_sat)
        assert params_high.saturation_mult <= 1.05, "Already saturated content shouldn't boost more"

    def test_correction_never_exceeds_bounds(self):
        """Even extreme inputs should produce bounded corrections."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        # Worst case: pitch black
        extreme_dark = SegmentStats(black_point=0, white_point=50, mid_tone=25, saturation=0.0)
        params = compute_correction(extreme_dark)
        assert params.shadow_lift <= 20
        assert params.highlight_pull <= 20
        assert 1.0 <= params.saturation_mult <= 1.25

        # Worst case: blown out
        extreme_bright = SegmentStats(
            black_point=200, white_point=255, mid_tone=230, saturation=0.8
        )
        params = compute_correction(extreme_bright)
        assert params.shadow_lift <= 20
        assert params.highlight_pull <= 20
        assert 1.0 <= params.saturation_mult <= 1.25


# ── FFmpeg Filter Generation Tests ───────────────────────────────────


class TestFilterGeneration:
    """Test ffmpeg filter string generation from correction params."""

    def test_global_curves_filter(self):
        """Global correction should produce a valid curves filter string."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = CorrectionParams(shadow_lift=10, highlight_pull=8, saturation_mult=1.1)
        filter_str = build_filter_chain([params])

        assert "curves=" in filter_str, "Should contain curves filter"
        assert "eq=saturation=" in filter_str, "Should contain saturation filter"

    # NOTE: the former ``test_adaptive_filter_uses_sendcmd`` was removed here.
    # It was named for a ``sendcmd`` mechanism that never shipped and accepted
    # a ``colorlevels=`` filter that no longer exists in the source. Its actual
    # scenario (differing params, no baked LUTs) is covered exactly by
    # test_color_filters_adaptive.py::TestDispatch, which asserts equality with
    # the averaged single filter rather than a substring match.

    def test_no_correction_passthrough(self):
        """Zero correction should produce minimal/identity filter."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = CorrectionParams(shadow_lift=0, highlight_pull=0, saturation_mult=1.0)
        filter_str = build_filter_chain([params])

        # Should be an identity/null filter or empty
        assert filter_str is not None

    def test_filter_string_is_valid_ffmpeg(self):
        """Filter string should be parseable by ffmpeg (no syntax errors)."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = CorrectionParams(shadow_lift=10, highlight_pull=8, saturation_mult=1.1)
        filter_str = build_filter_chain([params])

        # Should not contain invalid chars
        assert "\n" not in filter_str, "Filter should be single-line"
        # Balanced quotes/brackets
        assert filter_str.count("'") % 2 == 0 or "'" not in filter_str

    def test_curves_points_are_space_separated(self):
        """curves filter points MUST be space-separated, not colon-separated.

        Regression guard for a silent quality bug: ffmpeg's `curves` filter
        expects control points separated by SPACES (e.g. ``m='0/0 0.5/0.5 1/1'``).
        If colons are used, ffmpeg silently accepts the string but parses it as
        a degenerate single-point curve, producing a near-uniform gray output.
        NVENC then encodes that low-entropy frame at ~300 kbps instead of the
        expected 60-120 Mbps at 4K — destroying the entire day's LF content.

        The post-encode quality gate catches this failure.
        """
        import re

        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = CorrectionParams(
            shadow_lift=10,
            highlight_pull=8,
            saturation_mult=1.1,
            contrast_curve=0.02,
        )
        filter_str = build_filter_chain([params])

        # Extract the curves=m='...' expression
        match = re.search(r"curves=m='([^']*)'", filter_str)
        assert match is not None, f"expected curves=m='...' in filter string, got: {filter_str}"
        curve_expr = match.group(1)

        # Must contain multiple X/Y pairs separated by SPACES, never colons
        # between pairs. A colon inside a coordinate (e.g. "0.5/0.5:0.75/0.75")
        # is the bug — ffmpeg silently accepts it but produces crushed output.
        assert ":" not in curve_expr, (
            f"curves points joined with ':' — ffmpeg requires spaces between "
            f"points or it produces crushed dark-gray output. Got: '{curve_expr}'"
        )
        # Sanity: there should be multiple points (at least 2 X/Y pairs)
        points = curve_expr.split()
        assert len(points) >= 2, (
            f"expected >=2 curve points after split on space, got {len(points)}: '{curve_expr}'"
        )
        # Each point must be an X/Y pair in [0,1]
        for pt in points:
            assert "/" in pt, f"malformed curve point '{pt}' (expected X/Y)"
            x_str, y_str = pt.split("/")
            x, y = float(x_str), float(y_str)
            assert 0.0 <= x <= 1.0, f"curve point X out of [0,1]: {pt}"
            assert 0.0 <= y <= 1.0, f"curve point Y out of [0,1]: {pt}"


# ── Encoding Spec Tests ──────────────────────────────────────────────


class TestEncodingSpecs:
    """Test optimal encoding specification generation."""

    def test_4k_hevc_specs(self):
        """4K input should produce HEVC Main 10 CRF 18 specs (YouTube-optimized)."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(width=3840, height=2160, fps=59.94, has_nvenc=False)

        assert "-c:v" in args
        assert "libx265" in args
        assert "-crf" in args
        assert "18" in args
        assert "-profile:v" in args
        assert "main10" in args
        assert "-preset" in args
        assert "slow" in args
        assert "-pix_fmt" in args
        assert "yuv420p10le" in args

    def test_nvenc_when_available(self):
        """Should use NVENC when GPU is available."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(width=3840, height=2160, fps=59.94, has_nvenc=True)

        assert "hevc_nvenc" in args, "Should use NVENC encoder"
        # NVENC uses -cq not -crf
        assert "-cq" in args or "-crf" in args

    def test_faststart_flag(self):
        """Output should have faststart for streaming."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(width=1920, height=1080, fps=30.0, has_nvenc=False)

        assert "-movflags" in args
        assert "+faststart" in args

    def test_gop_2_seconds(self):
        """Keyframe interval should be 2 seconds at the output fps (29.97).

        All encodes are normalized to 29.97fps output, so GOP is always 60
        regardless of source fps. A 60fps source is downsampled to 29.97fps
        before encoding; 24fps source is upsampled; GOP stays 60 either way.
        """
        from gomoveshift.pipeline.color import get_encoding_args

        # 59.94fps source → output is 29.97fps → GOP = 60
        args = get_encoding_args(width=3840, height=2160, fps=59.94, has_nvenc=False)
        gop_idx = args.index("-g") if "-g" in args else -1
        assert gop_idx >= 0, "Should set GOP size"
        assert args[gop_idx + 1] == "60", "GOP should be 60 (2s at 29.97fps output)"

        # 30fps source → output is 29.97fps → GOP = 60
        args_30 = get_encoding_args(width=3840, height=2160, fps=30.0, has_nvenc=False)
        gop_idx = args_30.index("-g")
        assert args_30[gop_idx + 1] == "60", "GOP should be 60 (2s at 29.97fps output)"

        # 24fps source → output is 29.97fps → GOP = 60
        args_24 = get_encoding_args(width=3840, height=2160, fps=24.0, has_nvenc=False)
        gop_idx = args_24.index("-g")
        assert args_24[gop_idx + 1] == "60", "GOP should be 60 (2s at 29.97fps output)"

    def test_audio_copy(self):
        """Audio should be stream-copied (normalization is a separate stage)."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(width=3840, height=2160, fps=59.94, has_nvenc=False)

        assert "-c:a" in args
        assert "copy" in args

    def test_1080p_still_hevc_main10(self):
        """Even 1080p should use HEVC Main 10 for consistency."""
        from gomoveshift.pipeline.color import get_encoding_args

        args = get_encoding_args(width=1920, height=1080, fps=30.0, has_nvenc=False)

        assert "libx265" in args
        assert "main10" in args


# ── Integration Test (mocked ffmpeg) ─────────────────────────────────


class TestColorProcessorIntegration:
    """Test encoder capability detection."""

    @pytest.mark.real_nvenc
    def test_nvenc_detection(self):
        """Should detect NVENC availability."""
        from gomoveshift.pipeline.color import detect_nvenc

        # This actually tests on the current system
        result = detect_nvenc()
        assert isinstance(result, bool)


# ── Enhanced Video Quality Tests ─────────────────────────────────────


class TestEnhancedCorrection:
    """Tests for S-curve, white balance, gamma, denoise, sharpen, raised sat cap."""

    def test_per_channel_means_in_frame_stats(self):
        """analyze_frame should return per-channel means (r_mean, g_mean, b_mean)."""
        from gomoveshift.pipeline.color import analyze_frame

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # R
        frame[:, :, 1] = 100  # G
        frame[:, :, 2] = 50  # B
        stats = analyze_frame(frame)

        assert hasattr(stats, "r_mean"), "FrameStats must have r_mean"
        assert hasattr(stats, "g_mean"), "FrameStats must have g_mean"
        assert hasattr(stats, "b_mean"), "FrameStats must have b_mean"
        assert abs(stats.r_mean - 200 / 255.0) < 0.02
        assert abs(stats.g_mean - 100 / 255.0) < 0.02
        assert abs(stats.b_mean - 50 / 255.0) < 0.02

    def test_scurve_zero_for_every_contrasty_source(self):
        """compute_correction originates no contrast_curve for contrasty sources.

        On a high-contrast source (mid_tone=117) any non-zero curve strength
        closes the shadows; only c=0 keeps them open. The curve is gated on the
        measured white-black spread, and every fixture below spreads 210 — far
        above the gate — so it stays at zero. Flat-source behaviour lives
        in TestContrastCurveIsConditioned.
        """
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        # Covers the former peak (mid=128), the soft band (mid=117), and the
        # former-extreme branch outside the 80-180 window.
        for mid_tone in (128, 117, 50, 230):
            stats = SegmentStats(
                black_point=20,
                white_point=230,
                mid_tone=mid_tone,
                saturation=0.3,
                r_mean=0.5,
                g_mean=0.5,
                b_mean=0.5,
            )
            params = compute_correction(stats)
            assert params.contrast_curve == 0.0, (
                f"mid_tone={mid_tone}: expected 0.0, got {params.contrast_curve}"
            )

    def test_white_balance_corrects_blue_cast(self):
        """High b_mean (blue cast) → negative wb_b."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=20,
            white_point=230,
            mid_tone=128,
            saturation=0.3,
            r_mean=0.5,
            g_mean=0.5,
            b_mean=0.6,
        )
        params = compute_correction(stats)
        assert params.white_balance_b < 0, "Blue cast should produce negative wb_b"

    def test_white_balance_corrects_warm_cast(self):
        """High r_mean (warm cast) → negative wb_r."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=20,
            white_point=230,
            mid_tone=128,
            saturation=0.3,
            r_mean=0.6,
            g_mean=0.5,
            b_mean=0.5,
        )
        params = compute_correction(stats)
        assert params.white_balance_r < 0, "Warm cast should produce negative wb_r"

    def test_white_balance_capped(self):
        """Extreme cast should be capped at ±0.05."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=20,
            white_point=230,
            mid_tone=128,
            saturation=0.3,
            r_mean=0.9,
            g_mean=0.5,
            b_mean=0.1,
        )
        params = compute_correction(stats)
        assert abs(params.white_balance_r) <= 0.05
        assert abs(params.white_balance_b) <= 0.05

    def test_gamma_lifts_dark_midtones(self):
        """Mid_tone=90 → gamma > 1.0."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=10,
            white_point=230,
            mid_tone=90,
            saturation=0.3,
            r_mean=0.35,
            g_mean=0.35,
            b_mean=0.35,
        )
        params = compute_correction(stats)
        assert params.gamma > 1.0, "Dark midtones should lift gamma"

    def test_gamma_pulls_bright_midtones(self):
        """Mid_tone=160 → gamma < 1.0."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=40,
            white_point=250,
            mid_tone=160,
            saturation=0.3,
            r_mean=0.6,
            g_mean=0.6,
            b_mean=0.6,
        )
        params = compute_correction(stats)
        assert params.gamma < 1.0, "Bright midtones should pull gamma"

    def test_gamma_neutral_for_centered(self):
        """Mid_tone=128 → gamma = 1.0."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=20,
            white_point=230,
            mid_tone=128,
            saturation=0.3,
            r_mean=0.5,
            g_mean=0.5,
            b_mean=0.5,
        )
        params = compute_correction(stats)
        assert abs(params.gamma - 1.0) < 0.01, "Centered midtones should have neutral gamma"

    def test_denoise_enabled_for_dark_scenes(self):
        """Very dark scenes (black_point < 10) still get strength=5.0.

        Dark driving footage looks worse than it should without temporal
        smoothing, so this branch stays on while the well-lit and moderate
        branches are zero.
        Rollback for the dark branch is `git revert`, not a config flag.
        """
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=5,
            white_point=230,
            mid_tone=128,
            saturation=0.3,
            r_mean=0.5,
            g_mean=0.5,
            b_mean=0.5,
        )
        params = compute_correction(stats)
        assert params.denoise_strength == 5.0

    def test_denoise_disabled_for_moderate_black_point(self):
        """black_point=20 (the old 3.0 branch) now returns 0.0.

        Denoise is not visible on common outdoor driving footage at the
        moderate-noise tier, so this branch is off.
        """
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=20,
            white_point=230,
            mid_tone=128,
            saturation=0.3,
            r_mean=0.5,
            g_mean=0.5,
            b_mean=0.5,
        )
        params = compute_correction(stats)
        assert params.denoise_strength == 0.0

    def test_denoise_disabled_for_well_lit(self):
        """black_point=30 (the old 2.0 branch) now returns 0.0.

        Well-lit footage gains nothing visible from denoise, so the old
        2.0 branch is off.
        """
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=30,
            white_point=230,
            mid_tone=128,
            saturation=0.3,
            r_mean=0.5,
            g_mean=0.5,
            b_mean=0.5,
        )
        params = compute_correction(stats)
        assert params.denoise_strength == 0.0

    def test_shadow_lift_slope_deep_blacks(self):
        """black_point=5 → shadow_lift=14.25 under tightened 0.15 slope.

        Pins the slope change. If this drops below 14.0, the slope
        regressed back to 0.2 (or steeper).
        """
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=5,
            white_point=230,
            mid_tone=128,
            saturation=0.3,
            r_mean=0.5,
            g_mean=0.5,
            b_mean=0.5,
        )
        params = compute_correction(stats)
        assert abs(params.shadow_lift - 14.25) < 0.01, (
            f"Expected 14.25 (slope=0.15), got {params.shadow_lift}"
        )

    def test_sharpen_disabled_by_default(self):
        """Sharpen defaults to off — preserves GoPro native detail."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        stats = SegmentStats(
            black_point=20,
            white_point=230,
            mid_tone=128,
            saturation=0.3,
            r_mean=0.5,
            g_mean=0.5,
            b_mean=0.5,
        )
        params = compute_correction(stats)
        assert params.sharpen_amount == 0.0  # no measured detail_score -> unmeasured, off

    def test_saturation_cap_raised(self):
        """Saturation cap should be raised to 1.25 for driving content."""
        from gomoveshift.pipeline.color import SegmentStats, compute_correction

        # Very low saturation should produce higher mult now
        stats = SegmentStats(
            black_point=20,
            white_point=230,
            mid_tone=128,
            saturation=0.05,
            r_mean=0.5,
            g_mean=0.5,
            b_mean=0.5,
        )
        params = compute_correction(stats)
        assert params.saturation_mult <= 1.25, "Cap should be 1.25"
        assert params.saturation_mult > 1.15, "Low sat should exceed old 1.15 cap"

    def test_filter_chain_order(self):
        """Filters should appear in correct sequence: curves → colorbalance → eq → hqdn3d → unsharp."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = CorrectionParams(
            shadow_lift=10,
            highlight_pull=8,
            saturation_mult=1.1,
            contrast_curve=0.10,
            white_balance_r=-0.02,
            white_balance_b=0.03,
            gamma=1.05,
            denoise_strength=3.0,
            sharpen_amount=1.0,
        )
        filter_str = build_filter_chain([params])

        # Check order: curves before colorbalance before eq before hqdn3d before unsharp
        idx_curves = filter_str.find("curves=")
        idx_colorbal = filter_str.find("colorbalance=")
        idx_eq = filter_str.find("eq=")
        idx_hqdn3d = filter_str.find("hqdn3d=")
        idx_unsharp = filter_str.find("unsharp=")

        assert idx_curves < idx_colorbal < idx_eq < idx_hqdn3d < idx_unsharp, (
            f"Wrong filter order: curves={idx_curves}, colorbalance={idx_colorbal}, "
            f"eq={idx_eq}, hqdn3d={idx_hqdn3d}, unsharp={idx_unsharp}"
        )

    def test_eq_merges_saturation_and_gamma(self):
        """Single eq= filter should contain both saturation and gamma."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = CorrectionParams(
            shadow_lift=0,
            highlight_pull=0,
            saturation_mult=1.1,
            gamma=1.05,
        )
        filter_str = build_filter_chain([params])

        # Should be one eq= filter with both params
        assert "eq=" in filter_str
        assert "saturation=" in filter_str
        assert "gamma=" in filter_str
        # Should NOT have two separate eq= filters
        assert filter_str.count("eq=") == 1, "Must merge saturation+gamma into one eq filter"


# ── Anti-Flicker Tests ───────────────────────────────────────────────


class TestAdaptiveFilterNoFlicker:
    def test_adaptive_filter_no_enable_expressions(self):
        """Adaptive filter must not contain enable= expressions (causes flicker)."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = [
            CorrectionParams(shadow_lift=10, highlight_pull=5),
            CorrectionParams(shadow_lift=20, highlight_pull=8),
            CorrectionParams(shadow_lift=15, highlight_pull=3),
        ]
        result = build_filter_chain(params, segment_duration=60.0)
        assert "enable=" not in result

    def test_adaptive_filter_produces_valid_filter(self):
        """Adaptive filter with multiple params returns a non-null filter string."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = [
            CorrectionParams(shadow_lift=10, highlight_pull=5),
            CorrectionParams(shadow_lift=20, highlight_pull=8),
        ]
        result = build_filter_chain(params, segment_duration=60.0)
        assert result != "null"
        assert len(result) > 0

    def test_adaptive_filter_averages_params(self):
        """Adaptive filter correction is the average of all segment params."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = [
            CorrectionParams(shadow_lift=10, highlight_pull=0),
            CorrectionParams(shadow_lift=20, highlight_pull=0),
        ]
        result = build_filter_chain(params, segment_duration=60.0)
        # Average shadow_lift = 15 → rimin = 15/255 ≈ 0.0588
        # Single-segment result with shadow_lift=15 should match
        single = build_filter_chain(
            [CorrectionParams(shadow_lift=15, highlight_pull=0)], segment_duration=0.0
        )
        assert result == single


# ── Fixed Frame Dimensions ──────────────────────────────────


class TestSampleFramesFixedDimensions:
    """Frame height must not be guessed from a hardcoded list."""

    def test_sample_frames_ffmpeg_command_uses_fixed_size(self):
        """The ffmpeg sampling command must use '-s' with fixed 160x90 dimensions."""
        import inspect

        from gomoveshift.pipeline.color import sample_frames

        source = inspect.getsource(sample_frames)

        # Must set w=160, h=90 as fixed dimensions
        assert "w, h = 160, 90" in source, "sample_frames must set fixed dimensions w=160, h=90"
        # Must pin those dimensions in the emitted scale (not zscale with h=-1).
        assert "scale={w}:{h}" in source, "sample_frames must pin fixed output dimensions"
        # Must NOT contain the old height-guessing loop
        assert "for test_h in" not in source, (
            "sample_frames still contains the height-guessing loop"
        )

    def test_sequential_fallback_uses_fixed_size(self):
        """The sequential fallback must also use fixed 160x90 dimensions."""
        import inspect

        from gomoveshift.pipeline.color import _sample_frames_sequential

        source = inspect.getsource(_sample_frames_sequential)

        # Must set w=160, h=90 as fixed dimensions
        assert "w, h = 160, 90" in source, (
            "_sample_frames_sequential must set fixed dimensions w=160, h=90"
        )
        # Must pin those dimensions in the emitted scale.
        assert "scale={w}:{h}" in source, (
            "_sample_frames_sequential must pin fixed output dimensions"
        )
        # Must NOT use zscale with h=-1 (aspect-ratio-dependent height)
        assert "h=-1" not in source, "_sample_frames_sequential still uses h=-1 (variable height)"

    def test_batch_mode_no_height_guessing(self):
        """Batch mode must NOT iterate common heights to guess frame size."""
        import inspect

        from gomoveshift.pipeline.color import sample_frames

        source = inspect.getsource(sample_frames)

        for bad_height in [180, 240, 320, 135, 200]:
            assert str(bad_height) not in source or "160" in source, (
                f"sample_frames still references guessed height {bad_height}"
            )


# ── _build_adaptive_filter Removed ──────────────────────


class TestBuildAdaptiveFilterRemoved:
    """_build_adaptive_filter should be deleted; multi-param always averages."""

    def test_build_adaptive_filter_does_not_exist(self):
        """_build_adaptive_filter should no longer exist in the module."""
        import gomoveshift.pipeline.color as color_mod

        assert not hasattr(color_mod, "_build_adaptive_filter"), (
            "_build_adaptive_filter still exists — it should be deleted"
        )

    def test_build_filter_chain_multiple_params_averages(self):
        """Multiple params must produce the same result as averaging then building single filter."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = [
            CorrectionParams(shadow_lift=10, highlight_pull=6, saturation_mult=1.1),
            CorrectionParams(shadow_lift=20, highlight_pull=12, saturation_mult=1.2),
        ]
        result = build_filter_chain(params, segment_duration=45.0)

        # Manually compute expected: average of params → single filter
        expected = build_filter_chain(
            [CorrectionParams(shadow_lift=15, highlight_pull=9, saturation_mult=1.15)],
        )
        assert result == expected, "Multiple params should produce averaged single filter"

    def test_build_filter_chain_multiple_params_no_enable(self):
        """Multiple params must not produce enable= expressions (flicker)."""
        from gomoveshift.pipeline.color import CorrectionParams, build_filter_chain

        params = [
            CorrectionParams(shadow_lift=10, highlight_pull=5),
            CorrectionParams(shadow_lift=20, highlight_pull=8),
            CorrectionParams(shadow_lift=15, highlight_pull=3),
        ]
        result = build_filter_chain(params, segment_duration=60.0)

        assert "enable=" not in result, "Adaptive filter must not use enable= expressions"


# ── Module-level functions (ColorProcessor removed) ────────


class TestModuleLevelFunctions:
    """Module-level functions are accessible via facade."""

    def test_module_level_sample_frames_exists(self):
        """sample_frames must exist as a module-level function."""
        from gomoveshift.pipeline.color import sample_frames

        assert callable(sample_frames)

    def test_module_level_analyze_segments_exists(self):
        """analyze_segments must exist as a module-level function."""
        from gomoveshift.pipeline.color import analyze_segments

        assert callable(analyze_segments)

    def test_module_level_encode_video_exists(self):
        """encode_video must exist as a module-level function."""
        from gomoveshift.pipeline.color import encode_video

        assert callable(encode_video)

    def test_sample_frames_callable(self):
        """sample_frames should be directly callable as a module-level function."""
        from gomoveshift.pipeline.color import sample_frames

        with patch("gomoveshift.pipeline.color_analysis.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = (b"", b"")
            mock_proc.returncode = 1
            mock_popen.return_value = mock_proc
            with patch("gomoveshift.pipeline.color_analysis.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=1, stdout=b"")
                result = sample_frames(Path("/fake/video.mp4"), 100.0)
                assert result == []

    def test_analyze_segments_callable(self):
        """analyze_segments should be directly callable as a module-level function."""
        from gomoveshift.pipeline.color import analyze_segments

        result = analyze_segments([])
        assert result == ([], False)


# ── Deficit-Scaled Exposure + Ramped Denoise ─────────────────────────


def _stats(**overrides):
    """SegmentStats with neutral, well-exposed defaults."""
    from gomoveshift.pipeline.color import SegmentStats

    base = dict(
        black_point=25,
        white_point=230,
        mid_tone=128,
        saturation=0.3,
        r_mean=0.5,
        g_mean=0.5,
        b_mean=0.5,
    )
    base.update(overrides)
    return SegmentStats(**base)


class TestExposureScalesWithMeasuredDeficit:
    """The fixed gamma/shadow-lift clamps saturated on dusk footage.

    Below ``DEEP_UNDEREXPOSURE_MID`` both clamps used to pin, so a nearly
    black frame received exactly the same correction as a merely dim one —
    no headroom precisely where the most is needed.
    """

    def test_gamma_keeps_rising_below_the_old_clamp(self):
        from gomoveshift.pipeline.color_correction import compute_correction

        deep = compute_correction(_stats(mid_tone=10, black_point=2))
        mild = compute_correction(_stats(mid_tone=30, black_point=2))

        assert deep.gamma > mild.gamma, (
            f"deeper underexposure must get more gamma: {deep.gamma} vs {mild.gamma}"
        )
        assert deep.gamma > 1.15, "old fixed ceiling must no longer bind"

    def test_shadow_lift_keeps_rising_below_the_old_clamp(self):
        from gomoveshift.pipeline.color_correction import compute_correction

        deep = compute_correction(_stats(mid_tone=5, black_point=0))
        mild = compute_correction(_stats(mid_tone=30, black_point=0))

        assert deep.shadow_lift > mild.shadow_lift, (
            f"deeper underexposure must get more lift: {deep.shadow_lift} vs {mild.shadow_lift}"
        )
        assert deep.shadow_lift > 15.0, "old fixed ceiling must no longer bind"

    def test_correction_stays_inside_the_documented_field_ranges(self):
        from gomoveshift.pipeline.color_correction import compute_correction

        pitch_black = compute_correction(_stats(black_point=0, white_point=40, mid_tone=0))

        assert pitch_black.shadow_lift <= 20.0
        assert pitch_black.gamma <= 1.30

    def test_ordinary_footage_grades_exactly_as_before(self):
        """No-regression guard: the ramp must bite only at the extremes.

        These are the historical values for common-case outdoor driving
        footage (shadow lift 15 - 25*0.15, highlight pull (230-170)*0.2,
        neutral gamma, denoise off). If any of them moves, ordinary
        published footage regrades — that is the regression this pins.
        """
        from gomoveshift.pipeline.color_correction import compute_correction

        params = compute_correction(_stats())

        assert params.shadow_lift == 11.25
        assert params.highlight_pull == 12.0
        assert params.gamma == 1.0
        assert params.denoise_strength == 0.0

    def test_mid_tone_above_the_deep_band_is_untouched(self):
        """The deep-underexposure term is exactly zero at and above the knee."""
        from gomoveshift.pipeline.color_correction import (
            DEEP_UNDEREXPOSURE_MID,
            compute_correction,
        )

        knee = compute_correction(_stats(mid_tone=DEEP_UNDEREXPOSURE_MID, black_point=5))

        # 1.0 + (110 - 35) / 500 and 15 - 5*0.15, i.e. the pre-existing math.
        assert knee.gamma == 1.15
        assert knee.shadow_lift == 14.25


class TestDenoiseRampsWithMeasurement:
    """Denoise used to be a binary cliff at black_point 10."""

    def test_mid_dark_footage_gets_non_zero_denoise(self):
        """black_point 15 sat squarely in the dead band and got exactly 0.0."""
        from gomoveshift.pipeline.color_correction import compute_correction

        params = compute_correction(_stats(black_point=15))

        assert 0.0 < params.denoise_strength < 5.0

    def test_strength_decreases_monotonically_with_black_point(self):
        from gomoveshift.pipeline.color_correction import compute_correction

        strengths = [
            compute_correction(_stats(black_point=bp)).denoise_strength for bp in (8, 12, 16)
        ]

        assert strengths[0] > strengths[1] > strengths[2] > 0.0, strengths

    def test_dusk_photo_gets_a_budget_for_the_photo_noise_ramp_to_spend(self):
        """Photo consequence: develop() gates NL-means on this budget being > 0.

        A dusk/high-ISO frame used to arrive with a zero ceiling, so the
        photo path's own measured-noise ramp had nothing to spend and the
        frame shipped un-denoised however noisy it actually was.
        """
        from gomoveshift.pipeline.color_correction import compute_correction

        assert compute_correction(_stats(black_point=12)).denoise_strength > 0.0

    def test_a_well_lit_daylight_source_stays_denoise_off(self):
        """A well-lit 4K daylight source (black_point ~26) stays denoise-off."""
        from gomoveshift.pipeline.color_correction import compute_correction

        assert compute_correction(_stats(black_point=26)).denoise_strength == 0.0
        assert compute_correction(_stats(black_point=30)).denoise_strength == 0.0


class TestNoDeadDenoiseKnob:
    """``denoise_enabled`` had no production caller, only a test."""

    def test_compute_correction_exposes_no_dead_denoise_flag(self):
        import inspect

        from gomoveshift.pipeline.color_correction import compute_correction

        assert "denoise_enabled" not in inspect.signature(compute_correction).parameters


# Reference stats of a well-lit, high-contrast source
# (action-camera 4K daylight): black_point ~25.7, white_point ~214.9 -> spread ~189.
WELL_LIT_REF = dict(black_point=25.7, white_point=214.9, mid_tone=117, saturation=0.3)


class TestContrastCurveIsConditioned:
    """The S-curve is gated on measured contrast, not amputated globally."""

    def test_flat_source_receives_an_s_curve(self):
        """A compressed histogram is exactly what the S-curve exists for."""
        from gomoveshift.pipeline.color_correction import compute_correction

        flat = _stats(black_point=48, white_point=148, mid_tone=104, saturation=0.18)
        assert compute_correction(flat).contrast_curve > 0.0

    def test_curve_strength_scales_with_measured_flatness(self):
        """Measured-first: the flatter the source, the stronger the curve."""
        from gomoveshift.pipeline.color_correction import compute_correction

        flatter = compute_correction(_stats(black_point=55, white_point=140, mid_tone=100))
        less_flat = compute_correction(_stats(black_point=45, white_point=165, mid_tone=100))
        assert flatter.contrast_curve > less_flat.contrast_curve > 0.0

    def test_no_curve_when_the_cast_exceeds_white_balance_authority(self):
        """curves steepens R/G/B independently, so it amplifies a cast — and it
        runs before colorbalance, whose shift is clamped to ±0.05. Measured on a
        flat, strongly-cast frame, c=0.03 cancels the whole WB correction."""
        from gomoveshift.pipeline.color_correction import compute_correction

        flat = dict(black_point=48, white_point=148, mid_tone=104, saturation=0.18)
        correctable = compute_correction(_stats(**flat, r_mean=0.53, g_mean=0.5, b_mean=0.5))
        beyond = compute_correction(_stats(**flat, r_mean=0.66, g_mean=0.5, b_mean=0.5))

        assert correctable.contrast_curve > 0.0
        assert beyond.contrast_curve == 0.0

    def test_a_well_lit_reference_source_regrades_identically(self):
        """No-regression guard: on a high-contrast source the curve strength stays at zero.

        Its spread (~189) sits far above the gate, so the curve stays exactly
        0.0 and every other parameter keeps today's value.
        """
        from gomoveshift.pipeline.color_correction import compute_correction

        params = compute_correction(_stats(**WELL_LIT_REF))
        assert params.contrast_curve == 0.0
        assert params.shadow_lift == pytest.approx(11.145, abs=1e-3)
        assert params.highlight_pull == pytest.approx(8.98, abs=1e-3)
        assert params.saturation_mult == pytest.approx(1.086, abs=1e-3)
        assert params.gamma == 1.0
        assert params.denoise_strength == 0.0

    def test_ordinary_contrasty_footage_keeps_a_flat_zero_curve(self):
        """Everything with a normal histogram spread grades as it does today."""
        from gomoveshift.pipeline.color_correction import compute_correction

        for mid_tone in (128, 117, 50, 230):
            params = compute_correction(_stats(black_point=20, white_point=230, mid_tone=mid_tone))
            assert params.contrast_curve == 0.0, f"mid_tone={mid_tone}"

    def test_underexposed_source_gets_lift_not_a_curve(self):
        """Low spread from darkness is not flatness — an S-curve would fight
        the shadow lift and gamma that the deep-underexposure ramp applies."""
        from gomoveshift.pipeline.color_correction import compute_correction

        params = compute_correction(_stats(black_point=0, white_point=50, mid_tone=25))
        assert params.contrast_curve == 0.0
        assert params.gamma > 1.0

    def test_curve_actually_opens_contrast_in_the_baked_pixels(self):
        """Measured behaviour, not the presence of a field: the graded 0.25/0.75
        band must spread further apart than the same grade with c forced off."""
        import dataclasses

        from gomoveshift.pipeline.color_correction import _apply_curves, compute_correction

        params = compute_correction(
            _stats(black_point=48, white_point=148, mid_tone=104, saturation=0.18)
        )
        band = np.tile(np.array([[0.25], [0.75]], dtype=np.float64), (1, 3))
        with_curve = _apply_curves(band, params)
        without = _apply_curves(band, dataclasses.replace(params, contrast_curve=0.0))

        spread_with = with_curve[1, 0] - with_curve[0, 0]
        spread_without = without[1, 0] - without[0, 0]
        assert spread_with > spread_without


class TestSourceProfileDetection:
    """The slopes are Natural-tuned; a Flat/Log source must be detected."""

    def test_flat_log_source_is_detected(self):
        from gomoveshift.pipeline.color_correction import (
            SOURCE_PROFILE_FLAT,
            detect_source_profile,
        )

        flat = _stats(black_point=48, white_point=148, mid_tone=104, saturation=0.15)
        assert detect_source_profile(flat) == SOURCE_PROFILE_FLAT

    def test_natural_profile_source_takes_the_existing_path(self):
        from gomoveshift.pipeline.color_correction import (
            SOURCE_PROFILE_NATURAL,
            detect_source_profile,
        )

        assert detect_source_profile(_stats(**WELL_LIT_REF)) == SOURCE_PROFILE_NATURAL
        assert detect_source_profile(_stats()) == SOURCE_PROFILE_NATURAL

    def test_low_contrast_but_saturated_source_is_not_flat(self):
        """Haze compresses the histogram without desaturating like Log does."""
        from gomoveshift.pipeline.color_correction import (
            SOURCE_PROFILE_NATURAL,
            detect_source_profile,
        )

        hazy = _stats(black_point=48, white_point=148, mid_tone=104, saturation=0.45)
        assert detect_source_profile(hazy) == SOURCE_PROFILE_NATURAL

    def test_underexposed_source_is_not_a_flat_profile(self):
        from gomoveshift.pipeline.color_correction import (
            SOURCE_PROFILE_NATURAL,
            detect_source_profile,
        )

        dark = _stats(black_point=0, white_point=50, mid_tone=25, saturation=0.0)
        assert detect_source_profile(dark) == SOURCE_PROFILE_NATURAL

    def test_flat_profile_branches_past_the_natural_saturation_cap(self):
        """A Log source ships washed out when clamped to the Natural 1.25 cap."""
        from gomoveshift.pipeline.color_correction import compute_correction

        flat = compute_correction(
            _stats(black_point=48, white_point=148, mid_tone=104, saturation=0.15)
        )
        natural = compute_correction(_stats(saturation=0.15))
        assert flat.saturation_mult > 1.25
        assert natural.saturation_mult <= 1.25

    def test_natural_profile_saturation_is_untouched(self):
        """The no-regression guard for the saturation branch."""
        from gomoveshift.pipeline.color_correction import compute_correction

        assert compute_correction(_stats(saturation=0.05)).saturation_mult == pytest.approx(
            1.229, abs=1e-3
        )
        assert compute_correction(_stats(saturation=0.6)).saturation_mult == 1.0


class TestSaturationCanBePulledDown:
    """The saturation term was floored at 1.0, so the grader could only
    ever ADD saturation. Harsh midday sun or an already-punchy camera profile
    got no correction at all and could only be pushed further."""

    def test_oversaturated_source_is_pulled_below_unity(self):
        from gomoveshift.pipeline.color_correction import compute_correction

        assert compute_correction(_stats(saturation=0.85)).saturation_mult < 1.0

    def test_pulldown_scales_with_the_measured_excess(self):
        """Measured-first: the more oversaturated, the harder the pull."""
        from gomoveshift.pipeline.color_correction import compute_correction

        more = compute_correction(_stats(saturation=0.95)).saturation_mult
        less = compute_correction(_stats(saturation=0.75)).saturation_mult
        assert more < less < 1.0

    def test_pulldown_is_bounded_at_the_mirror_of_the_boost_cap(self):
        from gomoveshift.pipeline.color_correction import (
            SATURATION_PULLDOWN_MIN,
            compute_correction,
        )

        for sat in (0.9, 1.0):
            mult = compute_correction(_stats(saturation=sat)).saturation_mult
            assert mult >= SATURATION_PULLDOWN_MIN

    def test_correctly_saturated_source_is_left_exactly_alone(self):
        """No-false-positive guard, and the pinned 0.6 case stays 1.0."""
        from gomoveshift.pipeline.color_correction import compute_correction

        for sat in (0.45, 0.5, 0.55, 0.6):
            assert compute_correction(_stats(saturation=sat)).saturation_mult == 1.0, sat

    def test_the_boost_ramp_regrades_identically(self):
        """Everything under-saturated keeps today's multiplier to the digit."""
        from gomoveshift.pipeline.color_correction import compute_correction

        assert compute_correction(_stats(**WELL_LIT_REF)).saturation_mult == pytest.approx(
            1.086, abs=1e-3
        )
        assert compute_correction(_stats(saturation=0.05)).saturation_mult == pytest.approx(
            1.229, abs=1e-3
        )

    def test_a_blown_out_frame_is_not_evidence_of_oversaturation(self):
        """(max-min)/max collapses as channels clip together and inflates on
        near-black pixels, so it is only evidence inside the sane-exposure
        window the module already defines. Outside it, no pull-down."""
        from gomoveshift.pipeline.color_correction import compute_correction

        blown = _stats(black_point=200, white_point=255, mid_tone=230, saturation=0.8)
        crushed = _stats(black_point=0, white_point=50, mid_tone=25, saturation=0.8)
        assert compute_correction(blown).saturation_mult == 1.0
        assert compute_correction(crushed).saturation_mult == 1.0

    def test_the_pulldown_actually_desaturates_the_baked_pixels(self):
        """Measured behaviour, not the presence of a field."""
        from gomoveshift.pipeline.color_correction import _apply_saturation, compute_correction

        params = compute_correction(_stats(saturation=0.9))
        band = np.array([[0.8, 0.5, 0.2]], dtype=np.float64)
        out = _apply_saturation(band, params.saturation_mult)

        assert (out[0, 0] - out[0, 2]) < (band[0, 0] - band[0, 2])


class TestDehazeStretchesCompressedHistogram:
    """Hazy footage (compressed histogram + lifted black point) shipped flat.

    Haze deposits a milky veil that lifts the black point AND compresses the
    spread. The base S-curve already fires on the compression; the
    dehaze term adds contrast for the ADDITIONAL veil signal and rides the SAME
    contrast_curve, so contrast is applied exactly once and cannot crush blacks.
    """

    def test_veiled_source_beats_an_equally_flat_but_clear_source(self):
        """Isolates the dehaze term: at an identical (flat) spread the base
        S-curve grades both the same, so any difference is the veil response."""
        from gomoveshift.pipeline.color_correction import compute_correction

        veiled = compute_correction(_stats(black_point=60, white_point=130, mid_tone=95))
        clearer = compute_correction(_stats(black_point=35, white_point=105, mid_tone=95))

        assert veiled.contrast_curve > clearer.contrast_curve > 0.0

    def test_clear_full_spread_footage_gets_no_dehaze(self):
        """Self-limiting guard: a clear, full-spread frame gets exactly 0.0 —
        no S-curve and no dehaze."""
        from gomoveshift.pipeline.color_correction import compute_correction

        clear = compute_correction(_stats(black_point=25, white_point=230, mid_tone=128))
        assert clear.contrast_curve == 0.0

    def test_non_veiled_flat_source_regrades_to_the_base_s_curve(self):
        """No-regression pin: a flat source whose blacks sit below the veil floor
        earns the base S-curve alone — byte-identical to the pre-dehaze grade."""
        from gomoveshift.pipeline.color_correction import (
            CONTRAST_CURVE_MAX,
            compute_correction,
        )

        # spread 100 -> flatness 0.75; black_point 20 is below the veil floor.
        flat = compute_correction(_stats(black_point=20, white_point=120, mid_tone=100))
        assert flat.contrast_curve == round(CONTRAST_CURVE_MAX * 0.75, 4)

    def test_dehaze_composes_without_crushing_blacks(self):
        """The crux: on maximally hazy footage the combined curve keeps the
        shadow knot well above black and stays monotonic — no crushed blacks."""
        from gomoveshift.pipeline.color_correction import _apply_curves, compute_correction

        params = compute_correction(_stats(black_point=75, white_point=110, mid_tone=95))
        ramp = np.tile(
            np.array([[0.0], [0.1], [0.25], [0.5], [0.75], [1.0]], dtype=np.float64), (1, 3)
        )
        graded = _apply_curves(ramp, params)

        assert np.all(np.diff(graded[:, 0]) >= -1e-9), "curve must stay monotonic"
        # 0.25 input is the darkest S-curve knot; it must remain open, not crushed.
        assert graded[2, 0] > 0.12

    def test_dehaze_opens_local_contrast_in_the_baked_pixels(self):
        """Measured behaviour: the veiled grade spreads the 0.25/0.75 band wider
        than the same grade with only the base S-curve."""
        import dataclasses

        from gomoveshift.pipeline.color_correction import (
            CONTRAST_CURVE_MAX,
            _apply_curves,
            compute_correction,
        )

        params = compute_correction(_stats(black_point=60, white_point=130, mid_tone=95))
        base_only = dataclasses.replace(params, contrast_curve=CONTRAST_CURVE_MAX)
        band = np.tile(np.array([[0.25], [0.75]], dtype=np.float64), (1, 3))

        dehazed = _apply_curves(band, params)
        base = _apply_curves(band, base_only)

        assert (dehazed[1, 0] - dehazed[0, 0]) > (base[1, 0] - base[0, 0])

    def test_dehaze_contrast_is_bounded(self):
        """Even a fully flat, fully veiled source stays under the combined cap."""
        from gomoveshift.pipeline.color_correction import (
            CONTRAST_CURVE_MAX,
            DEHAZE_CONTRAST_MAX,
            compute_correction,
        )

        params = compute_correction(_stats(black_point=80, white_point=110, mid_tone=95))
        assert params.contrast_curve <= CONTRAST_CURVE_MAX + DEHAZE_CONTRAST_MAX
        assert params.contrast_curve == pytest.approx(CONTRAST_CURVE_MAX + DEHAZE_CONTRAST_MAX)

    def test_underexposed_flatness_is_not_read_as_haze(self):
        """Darkness compresses the spread too, but its blacks are pinned LOW —
        the veil factor is zero, so an underexposed frame gets lift, not dehaze."""
        from gomoveshift.pipeline.color_correction import compute_correction

        dark = compute_correction(_stats(black_point=0, white_point=50, mid_tone=25))
        assert dark.contrast_curve == 0.0
        assert dark.gamma > 1.0


class TestOvercastLightIsGradedForContrast:
    """The overcast/rainy weather signal never reached the grade.

    Overcast diffuse light flattens contrast the same way haze does, but its
    blacks are not necessarily lifted, so the veil term (which keys on lifted
    blacks) misses it. The moderate-brightness window the weather detector
    calls OVERCAST is reconstructed locally from ``mid_tone`` and rides the
    SAME ``contrast_curve`` under the SAME combined cap — no second knob.
    """

    def test_overcast_band_footage_beats_an_equally_flat_out_of_band_source(self):
        """Isolates the overcast term. Both sources are equally flat (spread 115)
        with blacks below the veil floor, so the base S-curve and the dehaze term
        grade them identically; the ONLY difference is that one's median brightness
        sits inside the overcast band and the other's sits below it."""
        from gomoveshift.pipeline.color_correction import compute_correction

        overcast = compute_correction(_stats(black_point=25, white_point=140, mid_tone=125))
        out_of_band = compute_correction(_stats(black_point=25, white_point=140, mid_tone=95))

        assert overcast.contrast_curve > out_of_band.contrast_curve > 0.0

    def test_out_of_band_flat_source_regrades_to_the_base_s_curve(self):
        """No-regression pin: below the overcast floor the increment is exactly 0,
        so a flat, unlifted, dim source keeps the byte-identical base S-curve."""
        from gomoveshift.pipeline.color_correction import (
            CONTRAST_CURVE_MAX,
            compute_correction,
        )

        # spread 115 -> flatness (130 - 115) / 40 = 0.375; blacks below the veil
        # floor and mid_tone below the overcast floor -> both increments are 0.
        out = compute_correction(_stats(black_point=25, white_point=140, mid_tone=95))
        assert out.contrast_curve == round(CONTRAST_CURVE_MAX * 0.375, 4)

    def test_sunny_bright_flat_source_gets_no_overcast_increment(self):
        """Self-limiting guard: above the overcast band the increment is 0 — a
        washed-out bright source is not read as overcast."""
        from gomoveshift.pipeline.color_correction import (
            CONTRAST_CURVE_MAX,
            compute_correction,
        )

        # spread 124 -> flatness 0.15; blacks below the veil floor and mid_tone
        # above the overcast ceiling -> both increments are exactly 0.
        bright = compute_correction(_stats(black_point=28, white_point=152, mid_tone=150))
        assert bright.contrast_curve == round(CONTRAST_CURVE_MAX * 0.15, 4)

    def test_overcast_increment_shares_the_one_combined_cap(self):
        """No stacking: a maximally flat, fully veiled, in-band source stays at
        the SAME combined ceiling as haze alone — the overcast term draws from
        the shared increment headroom, it does not add a second one."""
        from gomoveshift.pipeline.color_correction import (
            CONTRAST_CURVE_MAX,
            DEHAZE_CONTRAST_MAX,
            compute_correction,
        )

        params = compute_correction(_stats(black_point=80, white_point=140, mid_tone=125))
        assert params.contrast_curve == pytest.approx(CONTRAST_CURVE_MAX + DEHAZE_CONTRAST_MAX)


# ── Photo-Mode Grade Profile ──────────────────────────────────────────


class TestPhotoModeGrade:
    """Photo mode drops the limited-range floors and preserves warm light.

    The 2-level shadow/highlight floors exist only to protect the
    full->limited squeeze video gets for YouTube; a full-range sRGB still
    must not inherit them. Likewise a golden-hour cast on a still is the
    subject, not an error — photo mode neutralizes only the warm excess
    beyond a preserving allowance. Video is the default and grades
    byte-identically to before.
    """

    def test_photo_mode_has_no_shadow_or_highlight_floors(self):
        from gomoveshift.pipeline.color_correction import GRADE_MODE_PHOTO, compute_correction

        # Bright still: both raw ramps land below the video 2.0 floor.
        stats = _stats(black_point=90, white_point=175)
        video = compute_correction(stats)
        photo = compute_correction(stats, mode=GRADE_MODE_PHOTO)

        assert video.shadow_lift == 2.0 and video.highlight_pull == 2.0
        assert photo.shadow_lift == pytest.approx(15.0 - 90 * 0.15)  # raw ramp, 1.5
        assert photo.highlight_pull == pytest.approx((175 - 170) * 0.2)  # raw ramp, 1.0

    def test_photo_mode_clamps_negative_ramps_to_zero(self):
        from gomoveshift.pipeline.color_correction import GRADE_MODE_PHOTO, compute_correction

        # Both raw ramps negative: a still earns exactly no lift/pull.
        photo = compute_correction(_stats(black_point=110, white_point=150), mode=GRADE_MODE_PHOTO)
        assert photo.shadow_lift == 0.0
        assert photo.highlight_pull == 0.0

    def test_photo_mode_preserves_a_warm_cast_video_would_neutralize(self):
        from gomoveshift.pipeline.color_correction import GRADE_MODE_PHOTO, compute_correction

        # Golden-hour cast entirely inside the warmth allowance.
        warm = _stats(r_mean=0.53, g_mean=0.50, b_mean=0.48)
        video = compute_correction(warm)
        photo = compute_correction(warm, mode=GRADE_MODE_PHOTO)

        assert video.white_balance_r == pytest.approx(-0.03)
        assert video.white_balance_b == pytest.approx(0.02)
        assert photo.white_balance_r == 0.0
        assert photo.white_balance_b == 0.0

    def test_photo_mode_corrects_only_the_warm_excess_beyond_the_allowance(self):
        from gomoveshift.pipeline.color_correction import (
            GRADE_MODE_PHOTO,
            PHOTO_WARM_CAST_ALLOWANCE,
            compute_correction,
        )

        strong = _stats(r_mean=0.55, g_mean=0.50, b_mean=0.50)  # r_cast = 0.05
        photo = compute_correction(strong, mode=GRADE_MODE_PHOTO)
        assert photo.white_balance_r == pytest.approx(-(0.05 - PHOTO_WARM_CAST_ALLOWANCE))

    def test_photo_mode_corrects_a_cool_cast_with_full_authority(self):
        from gomoveshift.pipeline.color_correction import GRADE_MODE_PHOTO, compute_correction

        cool = _stats(r_mean=0.46, g_mean=0.50, b_mean=0.54)
        video = compute_correction(cool)
        photo = compute_correction(cool, mode=GRADE_MODE_PHOTO)

        assert photo.white_balance_r == video.white_balance_r
        assert photo.white_balance_b == video.white_balance_b

    def test_default_mode_is_video_and_matches_explicit_video(self):
        """Regression pin: omitting the mode is exactly today's video grade."""
        from gomoveshift.pipeline.color_correction import GRADE_MODE_VIDEO, compute_correction

        for stats in (
            _stats(black_point=90, white_point=175),
            _stats(r_mean=0.53, g_mean=0.50, b_mean=0.48),
            _stats(black_point=3, white_point=250, mid_tone=40),
        ):
            assert compute_correction(stats) == compute_correction(stats, mode=GRADE_MODE_VIDEO)

    def test_unknown_mode_raises(self):
        from gomoveshift.pipeline.color_correction import compute_correction

        with pytest.raises(ValueError, match="mode"):
            compute_correction(_stats(), mode="hdr")

    def test_cluster_correction_threads_the_mode(self):
        from gomoveshift.pipeline.color_correction import (
            GRADE_MODE_PHOTO,
            compute_cluster_correction,
        )

        stats = _stats(black_point=90, white_point=175)
        photo = compute_cluster_correction([stats], mode=GRADE_MODE_PHOTO)
        assert photo.shadow_lift < 2.0
        assert photo.highlight_pull < 2.0


class TestPhotoModeSceneGuards:
    """Photo mode protects deliberate looks the video ramps read as defects.

    A video frame is one sample of a stream graded for YouTube, so its ramps
    err toward correction; a still IS the scene, so a silhouette must stay
    dark, fog must stay muted, mist must keep its veil and a sunset must keep
    its saturation. Every guard is photo-only — video (the default) grades
    byte-identically to before.
    """

    # Low-key / silhouette guard ────────────────────────────────────────────────────

    def test_low_key_still_keeps_its_darkness(self):
        """Blacks at true black + low midtones + INTACT highlights is the
        deliberate-low-key signature; the lift/gamma response is damped so a
        silhouette is not pulled to gray."""
        from gomoveshift.pipeline.color_correction import (
            GRADE_MODE_PHOTO,
            LOW_KEY_RESPONSE_SCALE,
            compute_correction,
        )

        silhouette = _stats(black_point=2, white_point=240, mid_tone=20)
        video = compute_correction(silhouette)
        photo = compute_correction(silhouette, mode=GRADE_MODE_PHOTO)

        assert photo.shadow_lift == pytest.approx(video.shadow_lift * LOW_KEY_RESPONSE_SCALE)
        assert photo.gamma == pytest.approx(1.0 + (video.gamma - 1.0) * LOW_KEY_RESPONSE_SCALE)

    def test_genuinely_underexposed_still_keeps_the_full_response(self):
        """Underexposure drags the WHOLE histogram down, highlights included —
        without the intact top end a dark still earns the full correction."""
        from gomoveshift.pipeline.color_correction import GRADE_MODE_PHOTO, compute_correction

        dim = _stats(black_point=2, white_point=140, mid_tone=20)
        video = compute_correction(dim)
        photo = compute_correction(dim, mode=GRADE_MODE_PHOTO)

        assert photo.shadow_lift == pytest.approx(video.shadow_lift)
        assert photo.gamma == pytest.approx(video.gamma)

    def test_video_mode_low_key_regrades_unchanged(self):
        """No-regression pin: the exact pre-guard lift/gamma math."""
        from gomoveshift.pipeline.color_correction import compute_correction

        video = compute_correction(_stats(black_point=2, white_point=240, mid_tone=20))
        # deficit (35-20)/35 continues both ramps past the knee, as before.
        assert video.shadow_lift == pytest.approx(15.0 - 2 * 0.15 + (15.0 / 35.0) * 5.0)
        assert video.gamma == pytest.approx(1.21)

    # Fog guard ───────────────────────────────────────────────────────────

    def test_foggy_still_gets_no_flat_profile_saturation_boost(self):
        """A still is never Protune/Flat-Log footage: fog measures exactly like
        a log profile (compressed spread, muted sat) and the profile boost
        would saturate the weather."""
        from gomoveshift.pipeline.color_correction import GRADE_MODE_PHOTO, compute_correction

        fog = _stats(black_point=40, white_point=140, mid_tone=110, saturation=0.18)
        photo = compute_correction(fog, mode=GRADE_MODE_PHOTO)

        # Base Natural ramp only: 1 + (0.45-0.18)/0.35 * 0.20, no ×1.15 gain.
        assert photo.saturation_mult == pytest.approx(1.154)

    def test_video_mode_fog_still_gets_the_profile_boost(self):
        """No-regression pin: video keeps the Flat/Log inference and its gain."""
        from gomoveshift.pipeline.color_correction import compute_correction

        fog = _stats(black_point=40, white_point=140, mid_tone=110, saturation=0.18)
        assert compute_correction(fog).saturation_mult == pytest.approx(1.327)

    # Mist (stacked contrast) guard ────────────────────────────────────────────────────────

    def test_misty_still_stacked_contrast_is_capped(self):
        """The base S-curve plus the atmospheric increment can stack to 0.10 on
        a deliberate mist scene; the low-spread/low-sat class caps at the base
        ceiling so the veil survives."""
        from gomoveshift.pipeline.color_correction import (
            GRADE_MODE_PHOTO,
            PHOTO_MIST_CONTRAST_MAX,
            compute_correction,
        )

        mist = _stats(black_point=70, white_point=160, mid_tone=115, saturation=0.18)
        photo = compute_correction(mist, mode=GRADE_MODE_PHOTO)
        assert photo.contrast_curve == pytest.approx(PHOTO_MIST_CONTRAST_MAX)

    def test_colorful_flat_still_is_not_read_as_mist(self):
        """The cap is scoped to the low-sat class: an equally flat but colorful
        still keeps the full stacked response."""
        from gomoveshift.pipeline.color_correction import (
            CONTRAST_CURVE_MAX,
            DEHAZE_CONTRAST_MAX,
            GRADE_MODE_PHOTO,
            compute_correction,
        )

        colorful = _stats(black_point=70, white_point=160, mid_tone=115, saturation=0.3)
        photo = compute_correction(colorful, mode=GRADE_MODE_PHOTO)
        assert photo.contrast_curve == pytest.approx(CONTRAST_CURVE_MAX + DEHAZE_CONTRAST_MAX)

    def test_video_mode_mist_keeps_the_full_stacked_response(self):
        """No-regression pin: video stacks base + atmospheric to the combined cap."""
        from gomoveshift.pipeline.color_correction import (
            CONTRAST_CURVE_MAX,
            DEHAZE_CONTRAST_MAX,
            compute_correction,
        )

        mist = _stats(black_point=70, white_point=160, mid_tone=115, saturation=0.18)
        assert compute_correction(mist).contrast_curve == pytest.approx(
            CONTRAST_CURVE_MAX + DEHAZE_CONTRAST_MAX
        )

    # Warm-sunset saturation guard ───────────────────────────────────────────────────

    def test_saturated_warm_sunset_still_keeps_its_saturation(self):
        """High mean sat TOGETHER WITH a warm cast reads as golden-hour light
        filling the frame — the scene, not an oversaturation artifact."""
        from gomoveshift.pipeline.color_correction import GRADE_MODE_PHOTO, compute_correction

        sunset = _stats(saturation=0.80, mid_tone=120, r_mean=0.55, g_mean=0.50, b_mean=0.47)
        photo = compute_correction(sunset, mode=GRADE_MODE_PHOTO)
        assert photo.saturation_mult == 1.0

    def test_neutral_cast_oversaturation_is_still_pulled_down_on_a_still(self):
        """The guard is scoped to warm scenes: a neutral-cast oversaturated
        still keeps the pulldown — that IS the artifact class."""
        from gomoveshift.pipeline.color_correction import GRADE_MODE_PHOTO, compute_correction

        neon = _stats(saturation=0.80, mid_tone=120)
        photo = compute_correction(neon, mode=GRADE_MODE_PHOTO)
        assert photo.saturation_mult == pytest.approx(0.914)

    def test_video_mode_sunset_keeps_the_pulldown(self):
        """No-regression pin: video pulls the same warm saturated frame down."""
        from gomoveshift.pipeline.color_correction import compute_correction

        sunset = _stats(saturation=0.80, mid_tone=120, r_mean=0.55, g_mean=0.50, b_mean=0.47)
        assert compute_correction(sunset).saturation_mult == pytest.approx(0.914)
