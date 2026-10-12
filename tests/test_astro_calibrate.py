"""Pure-math calibration tests on synthetic Bayer mosaics (no RAW fixtures),
plus ``read_raw_mosaic``'s HE/typed-error boundary with fake rawpy (the
``test_astro_orchestrator.py`` pattern).

The rawpy postprocess/write-back path in ``decode_linear`` is deliberately
NOT exercised here — the real-NEF
``raw.raw_image`` write-back spike is pending a reference night card.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rawpy

import gomoveshift.pipeline.photos.develop as dm
from gomoveshift.pipeline.photos.astro_calibrate import (
    AstroCalibrationError,
    AstroDarkIdentityMismatchError,
    AstroHighEfficiencyError,
    CaptureIdentity,
    NO_DARK_LOCAL_SIGMA_K,
    NO_DARK_MIN_FRAMES,
    NO_DARK_PERSISTENCE_SHARE,
    _local_cfa_outliers,
    build_master_dark,
    decode_linear,
    cosmetic_correct,
    hot_pixel_map,
    no_dark_hot_pixel_map,
    read_capture_identity,
    read_raw_mosaic,
    subtract_dark,
    validate_dark_identity,
)

#: RGGB base level per (row%2, col%2) parity — a Bayer mosaic repeats every
#: 2x2, so parity alone determines the CFA channel.
_BASE_BY_PARITY = {(0, 0): 1000.0, (0, 1): 2000.0, (1, 0): 3000.0, (1, 1): 4000.0}

#: One hot photosite per parity class, including a true corner (0, 0) to
#: cover the in-bounds-neighbor-subset edge case.
_HOT_SITES = [(0, 0), (0, 3), (5, 0), (3, 3)]


def _mosaic(shape, base_by_parity=_BASE_BY_PARITY, rng=None, noise_std=0.0):
    out = np.zeros(shape, dtype=np.uint16)
    for (pr, pc), base in base_by_parity.items():
        mask = np.zeros(shape, dtype=bool)
        mask[pr::2, pc::2] = True
        n = int(mask.sum())
        noise = rng.normal(0.0, noise_std, size=n) if rng is not None and noise_std else 0.0
        out[mask] = np.clip(base + noise, 0, 65535).astype(np.uint16)
    return out


class TestBuildMasterDark:
    def test_it_is_the_per_pixel_median(self):
        darks = [
            np.array([[10, 20], [30, 40]], dtype=np.uint16),
            np.array([[12, 18], [29, 42]], dtype=np.uint16),
            np.array([[11, 22], [31, 39]], dtype=np.uint16),
        ]
        master = build_master_dark(darks)
        assert master.dtype == np.float32
        np.testing.assert_array_equal(master, np.median(np.stack(darks), axis=0))

    def test_it_refuses_mixed_shapes(self):
        with pytest.raises(AstroCalibrationError):
            build_master_dark(
                [np.zeros((4, 4), dtype=np.uint16), np.zeros((4, 6), dtype=np.uint16)]
            )

    def test_it_refuses_no_darks(self):
        with pytest.raises(AstroCalibrationError):
            build_master_dark([])


class TestHotPixelMap:
    def test_it_flags_only_the_injected_hot_sites(self):
        rng = np.random.default_rng(20260813)
        darks = []
        for _ in range(3):
            dark = _mosaic((8, 8), rng=rng, noise_std=2.0)
            for r, c in _HOT_SITES:
                dark[r, c] = 60000  # stuck hot, consistent across every dark
            darks.append(dark)
        master = build_master_dark(darks)

        hotmap = hot_pixel_map(master)

        assert hotmap.dtype == bool
        flagged = set(zip(*np.nonzero(hotmap)))
        assert flagged == {(r, c) for r, c in _HOT_SITES}


class TestNoDarkHotPixelMap:
    """No darks -> local same-CFA residual + cross-frame persistence,
    never a global threshold or a temporal median (would preserve stars as "hot")."""

    def test_a_site_hot_in_every_frame_is_detected_and_corrected(self):
        rng = np.random.default_rng(20260813)
        mosaics = [_mosaic((16, 16), rng=rng, noise_std=8.0) for _ in range(NO_DARK_MIN_FRAMES)]
        site = (6, 6)
        for m in mosaics:
            m[site] = 60000

        hotmap = no_dark_hot_pixel_map(mosaics)
        assert hotmap[site]
        expected = _BASE_BY_PARITY[(site[0] % 2, site[1] % 2)]
        for m in mosaics:
            assert cosmetic_correct(m, hotmap)[site] == pytest.approx(expected, abs=50)

    def test_a_moving_bright_point_is_never_corrected(self):
        # A star drifts sites every frame -- persistence never accumulates.
        rng = np.random.default_rng(20260817)
        sites = [(4, 4), (4, 6), (6, 4), (6, 6), (8, 8)][:NO_DARK_MIN_FRAMES]
        mosaics = [_mosaic((16, 16), rng=rng, noise_std=8.0) for _ in range(NO_DARK_MIN_FRAMES)]
        for m, site in zip(mosaics, sites):
            m[site] = 60000
        assert _local_cfa_outliers(mosaics[0])[sites[0]]  # flagged in its own frame

        hotmap = no_dark_hot_pixel_map(mosaics)
        assert not hotmap.any()

    def test_a_site_hot_in_only_one_of_many_frames_is_left_alone(self):
        rng = np.random.default_rng(20260818)
        mosaics = [_mosaic((16, 16), rng=rng, noise_std=8.0) for _ in range(NO_DARK_MIN_FRAMES)]
        site = (6, 6)
        mosaics[0][site] = 60000  # only the first frame

        hotmap = no_dark_hot_pixel_map(mosaics)
        assert not hotmap[site]

    def test_too_few_frames_warns_and_corrects_nothing(self, caplog):
        mosaics = [_mosaic((16, 16)) for _ in range(NO_DARK_MIN_FRAMES - 1)]
        for m in mosaics:
            m[6, 6] = 60000  # persistent in every frame supplied -- still refused

        with caplog.at_level(logging.WARNING):
            hotmap = no_dark_hot_pixel_map(mosaics)
        assert not hotmap.any()
        assert any("confidence floor" in r.message for r in caplog.records)

    def test_correction_preserves_cfa_colour_structure(self):
        # Corner site (0, 3): in-bounds-neighbor-subset edge case, same-parity.
        rng = np.random.default_rng(20260814)
        mosaics = [_mosaic((16, 16), rng=rng, noise_std=8.0) for _ in range(NO_DARK_MIN_FRAMES)]
        site = (0, 3)
        for m in mosaics:
            m[site] = 60000

        hotmap = no_dark_hot_pixel_map(mosaics)
        corrected = cosmetic_correct(mosaics[0], hotmap)
        assert corrected[site] == pytest.approx(_BASE_BY_PARITY[(site[0] % 2, site[1] % 2)], abs=50)

    def test_majority_clipped_scale_collapse_does_not_flag_real_signal(self):
        # A clipped sky over >=50% of the frame collapses sigma to 0.
        shape = (20, 20)
        mosaics = []
        for _ in range(NO_DARK_MIN_FRAMES):
            m = np.zeros(shape, dtype=np.uint16)  # sky: clipped to 0
            m[8:11, 8:11] = 500  # fixed foreground texture, same every frame
            mosaics.append(m)

        hotmap = no_dark_hot_pixel_map(mosaics)
        assert not hotmap.any()

    @pytest.mark.parametrize("shape,n", [((24, 24), 7), ((40, 40), 10)])
    def test_a_solid_same_cfa_cluster_flags_only_its_boundary_corners(self, shape, n, caplog):
        # An INTERIOR site's neighbors are hot too, so only corners clear the
        # test; n=10 sits above a prior boundary-gate's ~79-site failure.
        rng = np.random.default_rng(20260815)
        mosaics = [_mosaic(shape, rng=rng, noise_std=8.0) for _ in range(NO_DARK_MIN_FRAMES)]
        cluster = [(r, c) for r in range(4, 4 + 2 * n, 2) for c in range(4, 4 + 2 * n, 2)]
        for m in mosaics:
            for site in cluster:
                m[site] = 60000
        rows, cols = sorted({r for r, _ in cluster}), sorted({c for _, c in cluster})
        r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        with caplog.at_level(logging.WARNING):
            hotmap = no_dark_hot_pixel_map(mosaics)

        assert {(int(r), int(c)) for r, c in zip(*np.nonzero(hotmap))} == corners
        assert any("NOT verified safe" in r.message for r in caplog.records)

    def test_a_cluster_on_a_gradient_sky_is_flagged_with_no_false_positives_off_cluster(
        self, caplog
    ):
        # Must ignore the gradient, not just the cluster.
        rng = np.random.default_rng(20260819)
        rows, cols = np.indices((96, 96))
        base = (600.0 + 40.0 * cols + 15.0 * rows).astype(np.float32)
        mosaics = [
            np.clip(base + rng.normal(0, 8.0, base.shape), 0, 65535).astype(np.uint16)
            for _ in range(NO_DARK_MIN_FRAMES)
        ]
        cluster = [(r, c) for r in range(20, 25, 2) for c in range(20, 25, 2)]  # 3x3 solid
        for m in mosaics:
            for site in cluster:
                m[site] = 3000

        with caplog.at_level(logging.WARNING):
            hotmap = no_dark_hot_pixel_map(mosaics)

        flagged = set(zip(*np.nonzero(hotmap)))
        assert flagged  # the cluster is detected, not swallowed by the gradient
        assert flagged <= set(cluster)  # nothing off-cluster is a gradient false positive
        assert any("NOT verified safe" in r.message for r in caplog.records)

    def test_saturated_content_beside_a_sharp_edge_would_be_gutted_by_blind_correction(self):
        # ZERO injected defects: a saturated lamp crossed by a dark wire.
        # Off-wire sites flag as real-signal FPs: a detector cost, never a correction.
        rng = np.random.default_rng(20260820)
        shape = (96, 96)
        rows, cols = np.indices(shape)
        lamp = (rows >= 30) & (rows < 60) & (cols >= 30) & (cols < 60)
        wire = lamp & (rows == cols)
        truth = np.where(lamp & ~wire, 66000.0, 1000.0).astype(np.float32)  # sky everywhere else
        mosaics = [
            np.clip(truth + rng.normal(0, 8.0, shape), 0, 65535).astype(np.uint16)
            for _ in range(NO_DARK_MIN_FRAMES)
        ]

        hotmap = no_dark_hot_pixel_map(mosaics)
        flagged_lamp_sites = list(zip(*np.nonzero(hotmap & ~wire)))
        assert flagged_lamp_sites  # real saturated content is flagged as a candidate

        corrected = cosmetic_correct(mosaics[0], hotmap)
        assert any(corrected[r, c] < 50000 for r, c in flagged_lamp_sites)

    def test_persistence_share_constant_is_pinned(self):
        # The no-darks contract rests on this exact share -- pin it.
        assert NO_DARK_PERSISTENCE_SHARE == 0.8

    def test_stars_at_documented_drift_are_not_filtered_by_persistence(self, caplog):
        # At the drift floor a PSF still wins the SAME site most frames.
        rng = np.random.default_rng(20260821)
        shape, hot_site = (64, 64), (32, 34)
        rows, cols = np.indices(shape)
        stars = [(rng.uniform(12, 52), rng.uniform(12, 52)) for _ in range(15)]
        mosaics = []
        for i in range(NO_DARK_MIN_FRAMES):
            frame = np.full(shape, 900.0)
            for r0, c0 in stars:  # each star drifts 1px/frame, in-spec
                frame += 8000 * np.exp(-(((rows - r0 - i) ** 2 + (cols - c0) ** 2) / (2 * 1.2**2)))
            frame += rng.normal(0, 8.0, shape)
            frame[hot_site] = 61000  # one genuine glued-to-the-sensor defect
            mosaics.append(np.clip(frame, 0, 65535).astype(np.uint16))
        with caplog.at_level(logging.WARNING):
            hotmap = no_dark_hot_pixel_map(mosaics)

        assert hotmap[hot_site]  # the real defect is still found
        assert int(hotmap.sum()) > len(stars)  # persistence doesn't filter the stars out
        assert any("star" in r.message.lower() for r in caplog.records)

    def test_a_blurred_static_light_flags_with_no_injected_defects(self):
        # Suppresses smooth GRADIENTS, but not curvature at the 2px scale.
        rows, cols = np.indices((96, 96))
        lamp = 1000.0 + 20000.0 * np.exp(-(((rows - 48) ** 2 + (cols - 48) ** 2) / 12.5))
        rng = np.random.default_rng(20260824)
        mosaics = [
            np.clip(lamp + rng.normal(0, 8.0, (96, 96)), 0, 65535).astype(np.uint16)
            for _ in range(NO_DARK_MIN_FRAMES)
        ]
        assert no_dark_hot_pixel_map(mosaics).sum() > 50

    def test_purity_inputs_untouched_output_shares_no_memory(self):
        # Mutation-battery: in-place correction sneaking back must go red.
        rng = np.random.default_rng(20260826)
        for dtype in (np.uint16, np.float32):
            m = rng.normal(1000, 8, (16, 16)).astype(dtype)
            m[6, 6] = 1200
            before = m.copy()
            hotmap = no_dark_hot_pixel_map([m] * NO_DARK_MIN_FRAMES)
            assert np.array_equal(m, before)
            assert not np.shares_memory(hotmap, m)

    def test_a_mild_non_saturated_stuck_site_is_flagged(self):
        assert NO_DARK_LOCAL_SIGMA_K == 6.0  # the sensitivity below is measured at 6.0
        rng = np.random.default_rng(20260823)
        mosaics = [_mosaic((16, 16), rng=rng, noise_std=8.0) for _ in range(NO_DARK_MIN_FRAMES)]
        for m in mosaics:
            m[6, 6] = 1200  # sky ~1000, mild non-saturated elevation
        assert no_dark_hot_pixel_map(mosaics)[6, 6]


class TestCosmeticCorrect:
    def test_it_replaces_a_hot_site_on_every_cfa_channel_with_its_neighbor_median(self):
        mosaic = _mosaic((8, 8))  # noise-free: neighbor median == exact base
        for r, c in _HOT_SITES:
            mosaic[r, c] = 60000
        hotmap = np.zeros(mosaic.shape, dtype=bool)
        for r, c in _HOT_SITES:
            hotmap[r, c] = True

        corrected = cosmetic_correct(mosaic, hotmap)

        for r, c in _HOT_SITES:
            expected = _BASE_BY_PARITY[(r % 2, c % 2)]
            assert corrected[r, c] == expected
        # untouched sites are unchanged, and the input mosaic was never mutated.
        untouched = ~hotmap
        np.testing.assert_array_equal(corrected[untouched], mosaic[untouched])
        assert mosaic[0, 0] == 60000  # `mosaic` itself: purity, not in-place

    def test_it_refuses_a_shape_mismatch(self):
        with pytest.raises(AstroCalibrationError):
            cosmetic_correct(np.zeros((4, 4), dtype=np.uint16), np.zeros((4, 6), dtype=bool))

    def test_original_was_untouched(self):
        mosaic = _mosaic((8, 8))
        before = mosaic.copy()
        hotmap = np.zeros(mosaic.shape, dtype=bool)
        hotmap[0, 0] = True
        cosmetic_correct(mosaic, hotmap)
        np.testing.assert_array_equal(mosaic, before)


class TestSubtractDark:
    def test_it_subtracts_the_dark_excess_and_clips_negative_results_at_zero(self):
        # black_level 0 => dark_excess IS the master dark, the pre-pedestal-fix arithmetic.
        mosaic = np.array([[100, 50], [10, 200]], dtype=np.uint16)
        master_dark = np.array([[20.0, 60.0], [5.0, 40.0]], dtype=np.float32)

        result = subtract_dark(mosaic, master_dark, black_level=0.0)

        assert result.dtype == np.float32
        np.testing.assert_array_equal(
            result, np.array([[80.0, 0.0], [5.0, 160.0]], dtype=np.float32)
        )

    def test_it_leaves_the_black_pedestal_in_so_postprocess_removes_it_once(self):
        """Light and master dark BOTH carry +black_level. Subtracting the whole
        dark cancels it here, then rawpy's postprocess subtracts the camera black level
        again, which removes almost all of the signal. Only the dark's EXCESS over the
        pedestal may be removed here."""
        black = 1008.0
        signal = np.array([[200.0, 40.0], [0.0, 900.0]], dtype=np.float32)
        dark_current = np.array([[5.0, 7.0], [4.0, 6.0]], dtype=np.float32)
        mosaic = (black + signal + dark_current).astype(np.uint16)
        master_dark = (black + dark_current).astype(np.float32)

        result = subtract_dark(mosaic, master_dark, black_level=black)

        # the pedestal survives, and exactly the dark current is gone
        np.testing.assert_allclose(result, black + signal, atol=1e-4)
        assert float(result.min()) >= black - 1e-4  # nothing clipped: clip needs signal < 0

    def test_black_level_is_required_so_the_bug_cannot_be_reintroduced(self):
        """No default: a caller that omits it would silently restore the double
        subtraction above, which is a render-destroying defect rather than a nit."""
        with pytest.raises(TypeError):
            subtract_dark(  # type: ignore[call-arg]
                np.zeros((2, 2), dtype=np.uint16), np.zeros((2, 2), dtype=np.float32)
            )

    def test_it_refuses_a_shape_mismatch(self):
        with pytest.raises(AstroCalibrationError, match="!="):
            subtract_dark(
                np.zeros((4, 4), dtype=np.uint16),
                np.zeros((4, 6), dtype=np.float32),
                black_level=0.0,
            )


class _FakeRawBase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestDecodeLinearPreservesThePedestal:
    def test_the_mosaic_handed_to_postprocess_still_carries_the_black_level(
        self, monkeypatch, tmp_path
    ):
        """End to end: whatever decode_linear writes back into raw_image is what
        postprocess subtracts the camera black level from. If dark subtraction has
        already taken the pedestal out, postprocess takes it out twice and the sky --
        only a few hundred counts above the pedestal to begin with -- goes to zero."""
        black, signal, dark_current = 1008.0, 300.0, 6.0
        seen = {}

        class _FakeRaw(_FakeRawBase):
            black_level_per_channel = [black] * 4

            def __init__(self):
                self.raw_image = np.full((4, 4), black + signal + dark_current, dtype=np.uint16)

            def postprocess(self, **kwargs):
                seen["mosaic"] = self.raw_image.copy()
                return np.zeros((4, 4, 3), dtype=np.uint16)

        monkeypatch.setattr(rawpy, "imread", lambda p: _FakeRaw())
        master_dark = np.full((4, 4), black + dark_current, dtype=np.float32)

        decode_linear(tmp_path / "light.nef", master_dark=master_dark)

        # the dark current is gone; the pedestal postprocess needs is still there
        np.testing.assert_allclose(seen["mosaic"], black + signal, atol=1.0)
        assert float(seen["mosaic"].min()) > black  # NOT signal-only, which would be 300


class _FakeHERaw(_FakeRawBase):
    """HE/HE* failure shape: opening succeeds, unpacking the mosaic does not."""

    @property
    def raw_image(self):
        raise rawpy.LibRawDataError("data corrupted at 1541636")


class TestReadRawMosaicHEFallback:
    def test_he_dark_reaches_the_dng_path_and_returns_that_mosaic(self, monkeypatch, tmp_path):
        nef_path = tmp_path / "dark.nef"
        dng_path = tmp_path / "dark.dng"
        dng_mosaic = np.arange(16, dtype=np.uint16).reshape(4, 4)

        class _FakeDngRaw(_FakeRawBase):
            raw_image = dng_mosaic

        routes = {str(nef_path): _FakeHERaw(), str(dng_path): _FakeDngRaw()}
        monkeypatch.setattr(rawpy, "imread", lambda p: routes[str(p)])
        monkeypatch.setattr(dm, "_convert_he_to_dng", lambda p: dng_path)

        result = read_raw_mosaic(nef_path)

        np.testing.assert_array_equal(result, dng_mosaic)

    def test_conversion_failure_raises_the_typed_error_not_librawerror(self, monkeypatch, tmp_path):
        nef_path = tmp_path / "dark.nef"
        convert_calls = []

        monkeypatch.setattr(rawpy, "imread", lambda p: _FakeHERaw())
        monkeypatch.setattr(dm, "_convert_he_to_dng", lambda p: convert_calls.append(p) or None)

        with pytest.raises(AstroHighEfficiencyError) as excinfo:
            read_raw_mosaic(nef_path)

        assert convert_calls == [nef_path]  # the DNG route was actually attempted
        assert "dark.nef" in str(excinfo.value)
        assert not isinstance(excinfo.value, rawpy.LibRawError)

    def test_imread_failure_on_a_missing_file_raises_the_typed_open_error(
        self, monkeypatch, tmp_path
    ):
        missing = tmp_path / "missing.nef"

        def _boom(p):
            raise rawpy.LibRawIOError("Input/output error")

        monkeypatch.setattr(rawpy, "imread", _boom)

        with pytest.raises(AstroCalibrationError):
            read_raw_mosaic(missing)

    def test_he_cache_skips_the_failed_direct_probe_on_a_known_he_path(self, monkeypatch, tmp_path):
        nef_path = tmp_path / "dark.nef"
        dng_path = tmp_path / "dark.dng"
        dng_mosaic = np.arange(16, dtype=np.uint16).reshape(4, 4)

        class _FakeDngRaw(_FakeRawBase):
            raw_image = dng_mosaic

        def _imread(p):
            if str(p) == str(nef_path):
                raise AssertionError("a cached HE path must skip the direct probe")
            return _FakeDngRaw()

        monkeypatch.setattr(rawpy, "imread", _imread)
        monkeypatch.setattr(dm, "_convert_he_to_dng", lambda p: dng_path)

        result = read_raw_mosaic(nef_path, he_cache={nef_path})

        np.testing.assert_array_equal(result, dng_mosaic)


class TestDecodeLinearHEErrorMessage:
    def test_the_he_error_names_the_real_remedy_not_a_stale_roadmap_claim(
        self, monkeypatch, tmp_path
    ):
        """DNG-conversion HE routing shipped in astro_orchestrator's fallback;
        the message must send the operator toward the converter config, not
        toward wiring that already exists."""
        monkeypatch.setattr(rawpy, "imread", lambda p: _FakeHERaw())

        with pytest.raises(AstroHighEfficiencyError) as excinfo:
            decode_linear(tmp_path / "light.nef")

        message = str(excinfo.value)
        assert "he_dng_converter_exe" in message
        assert "wave" not in message.replace(str(tmp_path), "").lower()
        assert "arrives with orchestrator wiring" not in message


class _FakeIdentityRaw(_FakeRawBase):
    """Metadata-only fake: the rawpy attributes ``read_capture_identity`` reads
    without ever touching pixels."""

    def __init__(self, *, shape=(8, 8), pattern=((0, 1), (3, 2)), iso=800.0, shutter=30.0):
        self.sizes = SimpleNamespace(raw_height=shape[0], raw_width=shape[1])
        self.raw_pattern = np.array(pattern, dtype=np.uint8)
        self.other = SimpleNamespace(iso_speed=iso, shutter_speed=shutter)


def _write_camera_tags(path, make, model):
    """A minimal real TIFF carrying Make/Model in IFD0, the same
    container-level tags a NEF/CR2/ARW/DNG carries -- no Exif sub-IFD, so
    also the "body never wrote AmbientTemperature" case."""
    from PIL import Image, TiffImagePlugin

    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[0x010F] = make
    ifd[0x0110] = model
    Image.new("L", (2, 2)).save(path, format="TIFF", tiffinfo=ifd)


class TestReadCaptureIdentity:
    def test_it_reads_shape_cfa_iso_exposure_and_camera(self, monkeypatch, tmp_path):
        path = tmp_path / "light.nef"
        _write_camera_tags(path, "NIKON CORPORATION", "NIKON Z 6")
        monkeypatch.setattr(rawpy, "imread", lambda p: _FakeIdentityRaw())

        identity = read_capture_identity(path)

        assert identity.raw_shape == (8, 8)
        assert identity.cfa_pattern == (0, 1, 3, 2)
        assert identity.iso == 800.0
        assert identity.exposure_s == 30.0
        assert identity.camera == "NIKON CORPORATION NIKON Z 6"
        assert identity.temperature_c is None  # no Exif sub-IFD in this fixture

    def test_it_reads_sensor_temperature_from_the_exif_sub_ifd(self, monkeypatch, tmp_path):
        # PIL's tiffinfo= save can't write a nested sub-IFD, so this hand-builds
        # the bytes with test_photo_nef_exif's proven `_ifd` helper.
        import struct

        from test_photo_nef_exif import _ifd

        path = tmp_path / "light.nef"
        sub_at = 8 + 2 + 12 + 4
        ifd0 = _ifd([(0x8769, 4, 1, struct.pack("<I", sub_at))], 8)
        sub = _ifd([(0x9400, 11, 1, struct.pack("<f", 12.5))], sub_at)
        path.write_bytes(b"II\x2a\x00" + struct.pack("<I", 8) + ifd0 + sub)
        monkeypatch.setattr(rawpy, "imread", lambda p: _FakeIdentityRaw())

        assert read_capture_identity(path).temperature_c == pytest.approx(12.5)


def _identity(**overrides):
    base = dict(
        path=Path("x.nef"),
        camera="NIKON Z 6",
        raw_shape=(8, 8),
        cfa_pattern=(0, 1, 3, 2),
        iso=800.0,
        exposure_s=30.0,
        temperature_c=None,
    )
    base.update(overrides)
    return CaptureIdentity(**base)


class TestValidateDarkIdentity:
    def test_matching_identity_passes(self):
        validate_dark_identity(_identity(), _identity())  # no raise

    @pytest.mark.parametrize(
        "field,dark_value",
        [
            ("camera", "SONY A7S"),
            ("raw_shape", (6, 6)),
            ("cfa_pattern", (1, 0, 2, 3)),
            ("iso", 1600.0),
            ("exposure_s", 15.0),
        ],
    )
    def test_each_mismatched_field_refuses_naming_the_field_and_both_values(
        self, field, dark_value
    ):
        light = _identity()
        dark = _identity(**{field: dark_value})

        with pytest.raises(AstroDarkIdentityMismatchError) as excinfo:
            validate_dark_identity(light, dark)

        message = str(excinfo.value)
        assert field in message
        assert str(getattr(light, field)) in message
        assert str(dark_value) in message

    def test_missing_temperature_on_either_side_does_not_refuse(self):
        validate_dark_identity(_identity(temperature_c=None), _identity(temperature_c=-5.0))
        validate_dark_identity(_identity(temperature_c=-5.0), _identity(temperature_c=None))

    def test_large_temperature_difference_warns_not_refuses(self, caplog):
        with caplog.at_level(logging.WARNING):
            validate_dark_identity(_identity(temperature_c=20.0), _identity(temperature_c=5.0))

        assert any("temperature" in r.message for r in caplog.records)


class TestBuildMasterDarkIdentityOptIn:
    def test_no_metadata_supplied_degrades_to_todays_behavior(self):
        darks = [np.zeros((4, 4), dtype=np.uint16), np.ones((4, 4), dtype=np.uint16)]

        master = build_master_dark(darks)  # no identity kwargs at all -- old call shape

        np.testing.assert_array_equal(master, np.median(np.stack(darks), axis=0))

    def test_supplying_only_one_side_of_identity_refuses_rather_than_silently_skipping(self):
        with pytest.raises(AstroCalibrationError):
            build_master_dark([np.zeros((4, 4), dtype=np.uint16)], light_identity=_identity())

    def test_mismatched_dark_identity_refuses_before_building_the_master(self):
        with pytest.raises(AstroDarkIdentityMismatchError):
            build_master_dark(
                [np.zeros((4, 4), dtype=np.uint16)],
                light_identity=_identity(),
                dark_identities=[_identity(camera="SONY A7S")],
            )

    def test_matching_dark_identity_still_builds_the_master(self):
        darks = [np.zeros((4, 4), dtype=np.uint16), np.full((4, 4), 2, dtype=np.uint16)]

        master = build_master_dark(
            darks, light_identity=_identity(), dark_identities=[_identity(), _identity()]
        )

        np.testing.assert_array_equal(master, np.median(np.stack(darks), axis=0))
