"""Flat + bias frame calibration: pure-math tests on synthetic Bayer
mosaics, mirroring ``test_astro_calibrate.py``'s darks tests.

Standard calibration chain covered here: master bias = per-pixel median of
bias frames; master dark is bias-subtracted when a master bias is supplied
(``build_master_dark``'s existing bias-blind default is unchanged); master
flat = median of (flat - bias), normalised to its own median; a light is
divided by that normalised flat, with values below a floor clipped up to it
before dividing (guards divide-by-small) and the clipped count reported.
"""

import numpy as np
import pytest
import rawpy

from gomoveshift.pipeline.photos.astro_calibrate import (
    AstroCalibrationError,
    FLAT_FIELD_FLOOR,
    apply_flat_field,
    build_master_bias,
    build_master_dark,
    build_master_flat,
    decode_linear,
)


class _FakeRawBase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestBuildMasterBias:
    def test_it_is_the_per_pixel_median(self):
        bias = [
            np.array([[10, 20], [30, 40]], dtype=np.uint16),
            np.array([[12, 18], [29, 42]], dtype=np.uint16),
            np.array([[11, 22], [31, 39]], dtype=np.uint16),
        ]
        master = build_master_bias(bias)
        assert master.dtype == np.float32
        np.testing.assert_array_equal(master, np.median(np.stack(bias), axis=0))

    def test_it_refuses_no_bias_frames(self):
        with pytest.raises(AstroCalibrationError):
            build_master_bias([])

    def test_it_refuses_mixed_shapes(self):
        with pytest.raises(AstroCalibrationError):
            build_master_bias(
                [np.zeros((4, 4), dtype=np.uint16), np.zeros((4, 6), dtype=np.uint16)]
            )


class TestBuildMasterDarkBiasSubtraction:
    def test_no_master_bias_is_unchanged_from_todays_behavior(self):
        darks = [np.array([[10, 20]], dtype=np.uint16), np.array([[12, 18]], dtype=np.uint16)]
        master = build_master_dark(darks)
        np.testing.assert_array_equal(master, np.median(np.stack(darks), axis=0))

    def test_master_bias_is_subtracted_before_the_median(self):
        darks = [
            np.array([[110.0, 220.0]], dtype=np.float32).astype(np.uint16),
            np.array([[112.0, 218.0]], dtype=np.float32).astype(np.uint16),
        ]
        bias = np.array([[100.0, 200.0]], dtype=np.float32)

        master = build_master_dark(darks, master_bias=bias)

        expected = np.median(np.stack([d.astype(np.float32) - bias for d in darks]), axis=0)
        np.testing.assert_allclose(master, expected)

    def test_master_bias_shape_mismatch_refuses(self):
        with pytest.raises(AstroCalibrationError):
            build_master_dark(
                [np.zeros((4, 4), dtype=np.uint16)],
                master_bias=np.zeros((4, 6), dtype=np.float32),
            )


class TestBuildMasterFlat:
    def test_it_is_the_median_normalised_to_its_own_median(self):
        flats = [
            np.array([[100, 200], [300, 400]], dtype=np.uint16),
            np.array([[110, 190], [290, 410]], dtype=np.uint16),
        ]
        master = build_master_flat(flats)

        combined = np.median(np.stack([f.astype(np.float32) for f in flats]), axis=0)
        expected = combined / float(np.median(combined))
        assert master.dtype == np.float32
        np.testing.assert_allclose(master, expected)
        assert np.median(master) == pytest.approx(1.0)

    def test_master_bias_is_subtracted_before_combining(self):
        flats = [
            np.array([[1100.0, 2200.0]], dtype=np.float32).astype(np.uint16),
            np.array([[1080.0, 2180.0]], dtype=np.float32).astype(np.uint16),
        ]
        bias = np.array([[1000.0, 2000.0]], dtype=np.float32)

        master = build_master_flat(flats, master_bias=bias)

        combined = np.median(np.stack([f.astype(np.float32) - bias for f in flats]), axis=0)
        expected = combined / float(np.median(combined))
        np.testing.assert_allclose(master, expected)

    def test_it_refuses_no_flat_frames(self):
        with pytest.raises(AstroCalibrationError):
            build_master_flat([])

    def test_it_refuses_mixed_shapes(self):
        with pytest.raises(AstroCalibrationError):
            build_master_flat(
                [np.zeros((4, 4), dtype=np.uint16), np.zeros((4, 6), dtype=np.uint16)]
            )

    def test_master_bias_shape_mismatch_refuses(self):
        with pytest.raises(AstroCalibrationError):
            build_master_flat(
                [np.zeros((4, 4), dtype=np.uint16)],
                master_bias=np.zeros((4, 6), dtype=np.float32),
            )

    def test_a_zero_median_refuses_rather_than_dividing_by_zero(self):
        with pytest.raises(AstroCalibrationError):
            build_master_flat([np.zeros((4, 4), dtype=np.uint16)])


class TestApplyFlatField:
    def test_known_vignette_is_removed_within_tolerance(self):
        # A radial multiplicative vignette: the flat carries the SAME falloff
        # as the light, so dividing the light by the (normalised) flat should
        # flatten the light back to its pre-vignette (uniform) level.
        shape = (32, 32)
        rows, cols = np.indices(shape)
        center = (shape[0] / 2, shape[1] / 2)
        radius = np.sqrt((rows - center[0]) ** 2 + (cols - center[1]) ** 2)
        vignette = 1.0 - 0.6 * (radius / radius.max()) ** 2  # 1.0 at center, 0.4 at the corner

        flat_level = 20000.0
        flat = (flat_level * vignette).astype(np.float32)
        master_flat = flat / float(np.median(flat))

        true_signal = 1000.0  # uniform scene-linear light, pre-vignette
        light = (true_signal * vignette).astype(np.float32)

        corrected = apply_flat_field(light, master_flat).mosaic

        # Flattened back to a single level, well within the vignette's own spread.
        assert corrected.std() < light.std() * 0.05
        np.testing.assert_allclose(corrected, true_signal * float(np.median(vignette)), rtol=0.05)

    def test_shape_mismatch_refuses(self):
        with pytest.raises(AstroCalibrationError):
            apply_flat_field(np.zeros((4, 4), dtype=np.float32), np.ones((4, 6), dtype=np.float32))

    def test_values_below_the_floor_are_clipped_and_the_count_is_reported(self):
        mosaic = np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32)
        flat = np.array([[1.0, 0.01], [0.0, 1.0]], dtype=np.float32)

        result = apply_flat_field(mosaic, flat)

        assert result.clipped_pixel_count == 2  # the 0.01 and 0.0 sites
        assert result.mosaic[0, 0] == pytest.approx(100.0)
        assert result.mosaic[0, 1] == pytest.approx(100.0 / FLAT_FIELD_FLOOR)
        assert result.mosaic[1, 0] == pytest.approx(100.0 / FLAT_FIELD_FLOOR)

    def test_no_clipping_when_every_flat_value_clears_the_floor(self):
        mosaic = np.full((3, 3), 100.0, dtype=np.float32)
        flat = np.full((3, 3), 1.0, dtype=np.float32)

        result = apply_flat_field(mosaic, flat)

        assert result.clipped_pixel_count == 0


class TestDecodeLinearFlatCalibration:
    def test_with_neither_darks_nor_flats_output_is_unchanged(self, monkeypatch, tmp_path):
        """No --darks/--flats: byte-identical to the pre-flats-and-bias behavior."""
        seen = {}

        class _FakeRaw(_FakeRawBase):
            black_level_per_channel = [0.0] * 4

            def __init__(self):
                self.raw_image = np.full((4, 4), 500, dtype=np.uint16)

            def postprocess(self, **kwargs):
                seen["mosaic"] = self.raw_image.copy()
                return np.zeros((4, 4, 3), dtype=np.uint16)

        monkeypatch.setattr(rawpy, "imread", lambda p: _FakeRaw())

        result = decode_linear(tmp_path / "light.nef")

        np.testing.assert_array_equal(seen["mosaic"], np.full((4, 4), 500, dtype=np.uint16))
        assert result.flat_field_clipped_pixels == 0

    def test_master_flat_divides_the_calibrated_mosaic_before_postprocess(
        self, monkeypatch, tmp_path
    ):
        seen = {}

        class _FakeRaw(_FakeRawBase):
            black_level_per_channel = [0.0] * 4

            def __init__(self):
                self.raw_image = np.full((4, 4), 200.0, dtype=np.uint16)

            def postprocess(self, **kwargs):
                seen["mosaic"] = self.raw_image.copy()
                return np.zeros((4, 4, 3), dtype=np.uint16)

        monkeypatch.setattr(rawpy, "imread", lambda p: _FakeRaw())
        master_flat = np.full((4, 4), 0.5, dtype=np.float32)

        result = decode_linear(tmp_path / "light.nef", master_flat=master_flat)

        np.testing.assert_allclose(seen["mosaic"], np.full((4, 4), 400.0), atol=1.0)
        assert result.flat_field_clipped_pixels == 0

    def test_master_flat_reports_clipped_pixels_below_the_floor(self, monkeypatch, tmp_path):
        seen = {}

        class _FakeRaw(_FakeRawBase):
            black_level_per_channel = [0.0] * 4

            def __init__(self):
                self.raw_image = np.full((2, 2), 200.0, dtype=np.uint16)

            def postprocess(self, **kwargs):
                seen["mosaic"] = self.raw_image.copy()
                return np.zeros((2, 2, 3), dtype=np.uint16)

        monkeypatch.setattr(rawpy, "imread", lambda p: _FakeRaw())
        master_flat = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)

        result = decode_linear(tmp_path / "light.nef", master_flat=master_flat)

        assert result.flat_field_clipped_pixels == 1
