"""Sky-ROI metering — exclude a bright sky when grading exposure and white balance.

The exposure/white-balance grade metered the whole frame, so a bright
sky dragged the meter and the road/foreground was graded too dark (highlight pull
tuned for the sky's clipping) or its white balance pulled toward sky blue. The
grade now down-weights the sky region of the frame by the *measured* sky fraction,
so:
  * measured, not invented — the sky region and its area share come from the frame
    itself (bright pixels in the upper band), not a tuned on/off switch;
  * self-limiting — the correction scales with the measured sky fraction, so a
    frame with little sky is barely affected and a no-sky frame is untouched;
  * shared-authority — the white-balance shift still rides the same ±0.05 clamp;
  * the foreground drives the meter — highlight pull reflects the road's highlights
    (much lower than a blown sky), and the cast reflects the foreground, not the sky.
"""

import numpy as np
import pytest

from gomoveshift.pipeline.color import (
    SegmentStats,
    aggregate_segment_stats,
    analyze_frame,
    compute_correction,
)


def _sky_frame(sky_rgb, road_rgb, sky_frac=0.33, size=(90, 160)):
    """A frame whose top ``sky_frac`` rows are the sky colour, the rest road."""
    h, w = size
    frame = np.empty((h, w, 3), dtype=np.uint8)
    split = int(h * sky_frac)
    frame[:split] = sky_rgb
    frame[split:] = road_rgb
    return frame


def _whole_frame_meter(stats: SegmentStats) -> SegmentStats:
    """Same segment metered over the whole frame (sky NOT excluded)."""
    return SegmentStats(
        black_point=stats.black_point,
        white_point=stats.white_point,
        mid_tone=stats.mid_tone,
        saturation=stats.saturation,
        r_mean=stats.r_mean,
        g_mean=stats.g_mean,
        b_mean=stats.b_mean,
    )


class TestSkyRegionMeasurement:
    def test_bright_sky_is_measured_and_foreground_metered_separately(self):
        frame = _sky_frame(sky_rgb=(200, 205, 215), road_rgb=(60, 60, 60))
        stats = analyze_frame(frame)
        # The bright sky is detected as a non-zero area share ...
        assert stats.sky_fraction > 0.0
        # ... and the foreground meter sees the darker road, not the blown sky.
        assert stats.fg_white_point < stats.white_point
        assert stats.fg_mid_tone <= stats.mid_tone

    def test_no_bright_sky_leaves_the_measurement_untouched(self):
        # Upper band DARKER than the rest (tunnel mouth, dense canopy): no sky.
        frame = _sky_frame(sky_rgb=(30, 30, 30), road_rgb=(120, 120, 120))
        stats = analyze_frame(frame)
        assert stats.sky_fraction == 0.0
        assert stats.fg_white_point == stats.white_point
        assert stats.fg_mid_tone == stats.mid_tone


class TestSkyExcludedExposure:
    def test_bright_sky_no_longer_drives_the_highlight_pull(self):
        frame = _sky_frame(sky_rgb=(205, 210, 220), road_rgb=(70, 70, 70))
        stats = aggregate_segment_stats([analyze_frame(frame)])

        graded = compute_correction(stats)
        whole = compute_correction(_whole_frame_meter(stats))

        # Sky excluded → highlights metered on the road → gentler pull, and the
        # foreground is never darkened by a pull tuned for the sky's clipping.
        assert graded.highlight_pull < whole.highlight_pull
        assert graded.shadow_lift >= whole.shadow_lift
        assert graded.gamma >= whole.gamma


class TestSkyExcludedWhiteBalance:
    def test_blue_sky_does_not_pull_the_foreground_warm(self):
        # Blue sky over a neutral road: the whole-frame cast is blue (but within
        # the ±0.05 clamp, so the softening is visible), while the road is neutral.
        frame = _sky_frame(sky_rgb=(140, 150, 180), road_rgb=(110, 110, 110))
        stats = aggregate_segment_stats([analyze_frame(frame)])

        graded = compute_correction(stats)
        whole = compute_correction(_whole_frame_meter(stats))

        # The main white-balance shift shares the ±0.05 clamp; excluding the sky
        # feeds the neutral foreground cast into that same clamp → gentler shift.
        assert abs(graded.white_balance_b) < abs(whole.white_balance_b)
        assert abs(graded.white_balance_b) <= 0.05


class TestNoSkyGuard:
    def test_a_frame_with_no_sky_grades_like_the_whole_frame_meter(self):
        # Uniform frame: nothing in the upper band is brighter than the median,
        # so sky_fraction is 0 and the grade must be identical to today.
        frame = np.full((90, 160, 3), 120, dtype=np.uint8)
        stats = aggregate_segment_stats([analyze_frame(frame)])

        graded = compute_correction(stats)
        whole = compute_correction(_whole_frame_meter(stats))

        assert graded.highlight_pull == pytest.approx(whole.highlight_pull)
        assert graded.shadow_lift == pytest.approx(whole.shadow_lift)
        assert graded.gamma == pytest.approx(whole.gamma)
        assert graded.white_balance_r == pytest.approx(whole.white_balance_r)
        assert graded.white_balance_b == pytest.approx(whole.white_balance_b)
