"""Stills are metered at the video-calibrated analysis scale.

The grade engine's saturation/mid-tone thresholds were calibrated on the
video sampler's fixed 160x90 analysis frames (``color_analysis.sample_frames``).
Metering a 20 MP still per-pixel reads sensor noise as chroma, so its measured
saturation is inflated relative to the calibration scale and the thresholds
systematically misfire. These tests pin the calibrated-scale metering seam
(one scale for the per-photo and batch-measure paths) and the landscape-only
sky ROI.
"""

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.color_analysis import analyze_frame
from gomoveshift.pipeline.photos import develop as dev

# Long edge of the video sampler's fixed 160x90 analysis frames — the scale
# the grade thresholds were calibrated on and stills must be metered at.
CALIBRATED_LONG_EDGE = 160


def _spy_analyze_frame(monkeypatch):
    """Record every frame handed to analyze_frame while running the real one."""
    frames = []
    real = dev.analyze_frame

    def spy(frame):
        frames.append(frame)
        return real(frame)

    monkeypatch.setattr(dev, "analyze_frame", spy)
    return frames


class TestCalibratedMeteringScale:
    def test_large_still_is_metered_at_the_calibrated_scale(self, monkeypatch):
        """The analysis view shrinks to the video-calibrated long edge, aspect kept."""
        frames = _spy_analyze_frame(monkeypatch)

        dev._frame_stats(np.full((1500, 2000, 3), 90, dtype=np.uint8))

        assert len(frames) == 1
        assert frames[0].shape == (120, CALIBRATED_LONG_EDGE, 3)

    def test_portrait_still_keeps_its_orientation_when_metered(self, monkeypatch):
        """The meter must see true orientation, or the sky ROI guard misjudges."""
        frames = _spy_analyze_frame(monkeypatch)

        dev._frame_stats(np.full((2000, 1500, 3), 90, dtype=np.uint8))

        assert frames[0].shape == (CALIBRATED_LONG_EDGE, 120, 3)

    def test_small_frame_is_metered_from_its_exact_pixels(self, monkeypatch):
        """Under the bound there is nothing to downsample — exact-stats behavior holds."""
        frames = _spy_analyze_frame(monkeypatch)
        small = np.full((24, 32, 3), 90, dtype=np.uint8)

        dev._frame_stats(small)

        assert frames[0] is small

    def test_noise_inflated_saturation_collapses_to_the_clean_truth(self):
        """The inflation the calibrated scale removes: pure per-pixel sensor noise
        on a neutral gray frame reads as heavy chroma at full resolution, and as
        (nearly) the clean frame's zero saturation once area-averaged down to the
        calibrated scale."""
        rng = np.random.default_rng(42)
        noise = rng.integers(-40, 41, size=(1080, 1620, 3))
        noisy = np.clip(128 + noise, 0, 255).astype(np.uint8)

        inflated = analyze_frame(noisy).saturation
        calibrated = dev._frame_stats(noisy).saturation

        assert inflated > 0.15
        assert calibrated == pytest.approx(0.0, abs=0.05)


class TestMeasurePathAverages:
    def test_measure_bound_averages_rather_than_strides(self, tmp_path):
        """The batch measure view must meter like the calibrated full decode: a
        stride keeps exact full-contrast source pixels (re-injecting the noise
        the calibrated scale cancels); area averaging blends them toward the
        values the calibrated meter reads."""
        frame = np.zeros((64, 3200, 3), dtype=np.uint8)
        frame[:, ::2] = 255  # single-pixel-wide full-contrast stripes
        path = tmp_path / "stripes.png"
        Image.fromarray(frame, mode="RGB").save(path)

        measured = dev._measure_rgb(path)

        assert max(measured.shape[:2]) <= dev.MEASURE_MAX_DIM
        assert measured.min() > 20
        assert measured.max() < 235


class TestSkyRoiLandscapeOnly:
    @staticmethod
    def _framed(height, width):
        """Dark frame with a bright upper band (30% of rows)."""
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        frame[: int(height * 0.3)] = 230
        return frame

    def test_portrait_still_skips_sky_exclusion(self):
        """The bright-upper-band ROI assumes landscape (dashcam) framing; a
        portrait still's bright top is usually the subject, not sky, so a
        taller-than-wide frame meters whole-frame."""
        stats = analyze_frame(self._framed(160, 90))

        assert stats.sky_fraction == 0.0
        assert stats.fg_white_point == stats.white_point
        assert stats.fg_mid_tone == stats.mid_tone
        assert (stats.fg_r_mean, stats.fg_g_mean, stats.fg_b_mean) == (
            stats.r_mean,
            stats.g_mean,
            stats.b_mean,
        )

    def test_landscape_sky_metering_is_unchanged(self):
        """Same content in landscape keeps the sky exclusion exactly as before."""
        stats = analyze_frame(self._framed(90, 160))

        assert stats.sky_fraction > 0.2
        assert stats.fg_white_point < stats.white_point
