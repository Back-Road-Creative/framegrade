"""Tests for the photo develop stage (gomoveshift.pipeline.photos.develop).

The develop stage layers grading on top of the decode front-end by reusing the
video grade engine, so these tests assert *measured* correction rather than
fixed pixel values.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.color_analysis import SegmentStats, analyze_frame
from gomoveshift.pipeline.color_correction import compute_correction
from gomoveshift.pipeline.photos import develop as dev


def _cast_frame(width=48, height=32, *, base=120, r_shift=40, b_shift=-20):
    """Smooth mid-range ramp carrying a deliberate warm cast."""
    ramp = np.linspace(base - 25, base + 25, width, dtype=np.float32)
    plane = np.repeat(ramp[None, :], height, axis=0)
    frame = np.stack([plane + r_shift, plane, plane + b_shift], axis=2)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _write(tmp_path, name, frame):
    path = tmp_path / name
    Image.fromarray(frame, mode="RGB").save(path)
    return path


def test_color_cast_is_measurably_corrected(tmp_path):
    frame = _cast_frame()
    out = dev.develop(_write(tmp_path, "warm.jpg", frame))

    before, after = analyze_frame(frame), analyze_frame(out.rgb)

    assert abs(after.r_mean - after.g_mean) < abs(before.r_mean - before.g_mean)
    assert abs(after.b_mean - after.g_mean) < abs(before.b_mean - before.g_mean)
    assert out.rgb.dtype == np.uint8
    assert out.rgb.shape == frame.shape


def test_batch_grades_a_related_set_consistently(tmp_path):
    """The clustering payoff: one cluster, one correction, applied to every member."""
    paths = [_write(tmp_path, f"s{i}.jpg", _cast_frame(base=120 + i)) for i in range(3)]

    results = dev.develop_batch(paths)

    assert [r.cluster_index for r in results] == [0, 0, 0]
    assert len({dataclasses.astuple(r.params) for r in results}) == 1


def test_batch_splits_clusters_on_different_lighting(tmp_path):
    paths = [
        _write(tmp_path, "bright.jpg", _cast_frame(base=170)),
        _write(tmp_path, "dark.jpg", _cast_frame(base=35)),
    ]

    bright, dark = dev.develop_batch(paths)

    assert bright.cluster_index != dark.cluster_index
    assert dataclasses.astuple(bright.params) != dataclasses.astuple(dark.params)


def test_export_writes_srgb_jpeg_with_no_exif(tmp_path):
    out_path = dev.export_jpeg(_cast_frame(), tmp_path / "nested" / "out.jpg")

    with Image.open(out_path) as img:
        assert img.format == "JPEG"
        assert img.mode == "RGB"
        assert dict(img.getexif()) == {}
        assert np.asarray(img).dtype == np.uint8


def _cache_key(path, params, **varied):
    return dev.develop_cache_key(
        path, params, decoder="pil", **{"options": dev.DevelopOptions(), "depth": 8, **varied}
    )


def test_cache_key_changes_when_grading_params_change(tmp_path):
    path = _write(tmp_path, "a.jpg", _cast_frame())
    params = compute_correction(analyze_frame(_cast_frame()))

    baseline = _cache_key(path, params)

    assert baseline == _cache_key(path, params)
    assert baseline != _cache_key(path, dataclasses.replace(params, gamma=params.gamma + 0.05))


def test_cache_key_changes_when_the_algorithm_version_changes(tmp_path, monkeypatch):
    path = _write(tmp_path, "a.jpg", _cast_frame())
    params = compute_correction(analyze_frame(_cast_frame()))
    baseline = _cache_key(path, params)

    monkeypatch.setattr(dev, "DEVELOP_ALGO_VERSION", "deadbeefcafe")

    assert _cache_key(path, params) != baseline


def test_algo_version_tracks_the_shared_grade_engine():
    """Self-invalidating like GRADE_ALGO_VERSION — never a hand-bumped integer."""
    from gomoveshift.pipeline.color_correction import GRADE_ALGO_VERSION

    assert GRADE_ALGO_VERSION in dev._develop_algo_inputs()


def test_a_dark_frame_measures_denoise_and_the_spatial_stage_consumes_it(tmp_path):
    """The dark-scene measurement is now wired through, not merely carried."""
    dark = dev.develop(_write(tmp_path, "dark.jpg", _cast_frame(base=6, r_shift=4, b_shift=-2)))

    assert dark.params.denoise_strength > 0.0
    assert dark.rgb.dtype == np.uint8
    # a clean synthetic ramp carries no actual noise, so the authorised budget
    # still ramps to nothing — measurement decides, not the dark-scene flag
    assert dark.denoise_applied is False


# ── 16-bit raw source through the float grade ─────────────────────────


def test_grade_from_a_16bit_source_avoids_8bit_banding():
    """An 8-bit decode quantizes BEFORE the grade re-floats it — banding."""
    from gomoveshift.pipeline.color_correction import CorrectionParams

    ramp = np.linspace(2000, 6000, 64 * 64, dtype=np.uint16).reshape(64, 64)
    frame16 = np.repeat(ramp[:, :, None], 3, axis=2)
    frame8 = (frame16 >> 8).astype(np.uint8)
    params = CorrectionParams(gamma=1.15)  # shadow stretch — where banding shows

    out16 = dev._grade(frame16, params)
    out8 = dev._grade(frame8, params)

    assert out16.dtype == np.uint8
    # The 16-bit source fills every intermediate output code; the 8-bit source
    # cannot mint codes its pre-quantized input already collapsed.
    assert len(np.unique(out16)) > len(np.unique(out8))
    assert np.max(np.diff(np.unique(out16.astype(int)))) == 1


def test_grade_uses_float32_internals(monkeypatch):
    """float64 flattens a 20 MP frame to ~500 MB and every apply step copies."""
    seen = {}
    real_curves, real_gamma = dev._apply_curves, dev._apply_gamma

    def spy_curves(flat, params):
        seen["first"] = flat.dtype
        return real_curves(flat, params)

    def spy_gamma(flat, gamma):
        seen["last"] = flat.dtype
        return real_gamma(flat, gamma)

    monkeypatch.setattr(dev, "_apply_curves", spy_curves)
    monkeypatch.setattr(dev, "_apply_gamma", spy_gamma)

    frame = _cast_frame()
    dev._grade(frame, _params_for(frame))

    assert seen["first"] == np.float32
    assert seen["last"] == np.float32  # the chain must not upcast midway


def test_develop_accepts_a_16bit_decode_and_meters_it_like_its_8bit_view(tmp_path):
    """``analyze_frame`` is 8-bit-calibrated: a 16-bit frame is metered through
    a uint8 view but graded from the full-depth original."""
    frame8 = _cast_frame()
    frame16 = frame8.astype(np.uint16) * 257  # exact 8-to-16 code expansion

    def _decoded(rgb, name, bits):
        return dev.DecodedPhoto(
            rgb=rgb,
            path=tmp_path / name,
            decoder="rawpy",
            width=48,
            height=32,
            bit_depth=bits,
            is_raw=True,
        )

    out16 = dev.develop(_decoded(frame16, "a.nef", 14))
    out8 = dev.develop(_decoded(frame8, "b.nef", 8))

    assert out16.rgb.dtype == np.uint8
    assert np.allclose(out16.rgb.astype(int), out8.rgb.astype(int), atol=1)


# ── Spatial stages: measured denoise + unsharp sharpen ────────────────


def _hf_energy(rgb):
    """High-frequency energy, measured independently of the engine's estimators."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return float(np.std(gray - cv2.GaussianBlur(gray, (0, 0), 1.5)))


def _flat_scene(size=160, base=110):
    """Smooth ramp: no noise and no real texture."""
    ramp = np.linspace(base - 30, base + 30, size, dtype=np.float32)
    plane = np.repeat(ramp[None, :], size, axis=0)
    return np.clip(np.stack([plane] * 3, axis=2), 0, 255).astype(np.uint8)


def _detailed_scene(size=160):
    """Crisp edges — genuine texture, deliberately noise-free."""
    frame = np.full((size, size, 3), 60, np.uint8)
    frame[::8, :] = 200
    frame[:, ::8] = 200
    frame[20:60, 20:60] = 240
    return frame


def _add_noise(frame, sigma, seed=7):
    rng = np.random.default_rng(seed)
    noised = frame.astype(np.float32) + rng.normal(0, sigma, frame.shape)
    return np.clip(noised, 0, 255).astype(np.uint8)


def _params_for(frame, **overrides):
    return dataclasses.replace(compute_correction(analyze_frame(frame)), **overrides)


def test_a_noisy_photo_is_denoised_while_a_clean_one_is_left_alone(tmp_path):
    """Conditionality, not just that the call happens: same params, both frames."""
    clean = _flat_scene()
    noisy = _add_noise(clean, 18)
    params = _params_for(clean, denoise_strength=5.0)

    clean_out = dev.develop(_write(tmp_path, "clean.png", clean), params)
    noisy_out = dev.develop(_write(tmp_path, "noisy.png", noisy), params)

    # the clean frame is not merely "denoised less" — the spatial stage never ran
    assert clean_out.denoise_applied is False
    assert np.array_equal(clean_out.rgb, dev._grade(clean, params))

    assert noisy_out.denoise_applied is True
    assert _hf_energy(noisy_out.rgb) < _hf_energy(dev._grade(noisy, params)) * 0.5


def test_denoise_ramps_with_measured_noise_instead_of_cliffing(tmp_path):
    """Denoise must not switch on and off at a threshold, as the video path once did."""
    params = _params_for(_flat_scene(), denoise_strength=5.0)

    mild = dev.develop(_write(tmp_path, "mild.png", _add_noise(_flat_scene(), 5, seed=3)), params)
    heavy = dev.develop(
        _write(tmp_path, "heavy.png", _add_noise(_flat_scene(), 20, seed=3)), params
    )

    assert 0.0 < mild.denoise_h < heavy.denoise_h


def test_denoise_budget_scales_with_the_already_measured_strength(tmp_path):
    """``params.denoise_strength`` is consumed, never re-derived from the pixels."""
    noisy = _add_noise(_flat_scene(), 18)
    path = _write(tmp_path, "n.png", noisy)

    weak = dev.develop(path, _params_for(noisy, denoise_strength=2.0))
    strong = dev.develop(path, _params_for(noisy, denoise_strength=6.0))
    off = dev.develop(path, _params_for(noisy, denoise_strength=0.0))

    assert 0.0 < weak.denoise_h < strong.denoise_h
    assert off.denoise_applied is False and off.denoise_h == 0.0


def test_denoise_applies_its_measured_correction_to_a_16bit_frame():
    """OpenCV's NL-means is 8-bit-only: the correction it measures on the uint8
    view is applied to the 16-bit frame as a residual, so pixels it leaves
    alone keep their sub-8-bit gradations instead of being requantized."""
    noisy8 = _add_noise(_flat_scene(), 18)
    noisy16 = noisy8.astype(np.uint16) * 257
    params = _params_for(_flat_scene(), denoise_strength=5.0)

    out8, h8 = dev._denoise(noisy8, params)
    out16, h16 = dev._denoise(noisy16, params)

    assert h16 == h8 > 0.0
    assert out16.dtype == np.uint16
    assert np.array_equal(out16, out8.astype(np.uint16) * 257)


def test_denoise_never_runs_nl_means_on_an_oversized_frame(monkeypatch):
    """NL-means wall clock grows linearly with pixel count (minutes at 20MP), so
    an over-bound frame is denoised on a downscale and the measured correction
    returns as an upsampled residual — never the full frame in one call."""
    seen = []
    real = dev.cv2.fastNlMeansDenoisingColored

    def spy(src, *args, **kwargs):
        seen.append(src.shape)
        return real(src, *args, **kwargs)

    monkeypatch.setattr(dev.cv2, "fastNlMeansDenoisingColored", spy)
    monkeypatch.setattr(dev, "DENOISE_MAX_DIM", 128.0)
    noisy = _add_noise(_flat_scene(size=320), 18)

    out, h = dev._denoise(noisy, _params_for(_flat_scene(), denoise_strength=5.0))

    assert h > 0.0
    assert seen and all(max(shape[:2]) <= 128 for shape in seen)
    assert out.shape == noisy.shape and out.dtype == np.uint8
    assert not np.array_equal(out, noisy)  # the upsampled residual actually landed


def test_denoise_runs_full_frame_below_the_runtime_bound(monkeypatch):
    """At or under the bound the path stays byte-identical to direct full-frame
    NL-means — the runtime bound must be a strict no-op for small frames."""
    seen = []
    real = dev.cv2.fastNlMeansDenoisingColored

    def spy(src, *args, **kwargs):
        seen.append(src.shape)
        return real(src, *args, **kwargs)

    monkeypatch.setattr(dev.cv2, "fastNlMeansDenoisingColored", spy)
    noisy = _add_noise(_flat_scene(), 18)

    out, h = dev._denoise(noisy, _params_for(_flat_scene(), denoise_strength=5.0))

    assert seen == [noisy.shape]
    assert np.array_equal(out, real(noisy, None, h, h, 7, 21))


def test_a_soft_photo_is_sharpened_and_a_crisp_one_is_not(tmp_path):
    import cv2

    crisp = _detailed_scene()
    soft = cv2.GaussianBlur(crisp, (9, 9), 3.0)
    params = _params_for(crisp, denoise_strength=0.0)

    soft_out = dev.develop(_write(tmp_path, "soft.png", soft), params)
    crisp_out = dev.develop(_write(tmp_path, "crisp.png", crisp), params)

    assert soft_out.sharpen_applied is True
    assert soft_out.sharpen_amount > 0.0
    assert _hf_energy(soft_out.rgb) > _hf_energy(dev._grade(soft, params))

    # already-sharp input is passed through untouched, never over-sharpened
    assert crisp_out.sharpen_applied is False and crisp_out.sharpen_amount == 0.0
    assert np.array_equal(crisp_out.rgb, dev._grade(crisp, params))


def test_sharpen_radius_scales_with_width_above_the_reference(monkeypatch):
    """A fixed 1.2px radius at 5568 wide is ~0.2px after the site's ~5x
    downscale — invisible. Radius scales with width; small frames unchanged."""
    sigmas = []
    real_blur = dev.cv2.GaussianBlur

    def spy(src, ksize, sigma, **kwargs):
        sigmas.append(sigma)
        return real_blur(src, ksize, sigma, **kwargs)

    monkeypatch.setattr(dev.cv2, "GaussianBlur", spy)
    monkeypatch.setattr(dev, "_detail_score", lambda frame: 100.0)  # mid-ramp

    _, small_amount = dev._sharpen(_flat_scene())  # 160 wide: under the reference
    _, wide_amount = dev._sharpen(np.zeros((32, 5568, 3), dtype=np.uint8))

    assert small_amount > 0.0 and wide_amount == small_amount
    assert sigmas[0] == dev.SHARPEN_RADIUS
    assert sigmas[1] == pytest.approx(dev.SHARPEN_RADIUS * 5568 / dev.SHARPEN_REFERENCE_WIDTH)


def test_sharpen_ramp_meters_detail_on_a_resolution_normalized_view(monkeypatch):
    """Variance-of-Laplacian is resolution- and noise-sensitive — at sensor
    scale high-ISO grain reads as crisp detail — so the ramp meters a bounded
    view; a small frame is metered exactly as before."""
    seen = []
    real = dev._detail_score

    def spy(frame):
        seen.append(frame)
        return real(frame)

    monkeypatch.setattr(dev, "_detail_score", spy)

    small = _flat_scene()
    dev._sharpen(small)
    assert seen[-1] is small  # at/under the bound the exact frame is metered

    dev._sharpen(np.tile(_flat_scene(size=64), (1, 40, 1)))  # 64 x 2560
    assert max(seen[-1].shape[:2]) <= dev.SHARPEN_DETAIL_MAX_DIM
    assert seen[-1].shape[1] < 2560


def test_develop_of_a_small_frame_matches_the_unbounded_chain(tmp_path):
    """Resolution-aware scaling must no-op at/below the reference size: a small
    frame's develop output stays byte-identical to the direct spatial chain."""
    import cv2

    noisy_soft = _add_noise(cv2.GaussianBlur(_detailed_scene(), (9, 9), 3.0), 10)
    params = _params_for(noisy_soft, denoise_strength=4.0)

    out = dev.develop(_write(tmp_path, "small.png", noisy_soft), params)

    assert out.denoise_applied and out.sharpen_applied
    h, amount = out.denoise_h, out.sharpen_amount
    graded = dev._grade(cv2.fastNlMeansDenoisingColored(noisy_soft, None, h, h, 7, 21), params)
    blurred = cv2.GaussianBlur(graded.astype(np.float32), (0, 0), dev.SHARPEN_RADIUS)
    expected = graded.astype(np.float32) * (1.0 + amount) - blurred * amount
    assert np.array_equal(out.rgb, np.clip(expected, 0.0, 255.0).astype(np.uint8))


def test_detail_score_ranks_crisp_above_soft_above_featureless():
    import cv2

    crisp = _detailed_scene()
    soft = cv2.GaussianBlur(crisp, (9, 9), 3.0)

    assert dev._detail_score(crisp) > dev._detail_score(soft) > dev._detail_score(_flat_scene())


def test_noise_estimate_does_not_mistake_real_texture_for_noise():
    """varLaplacian alone would rank clean texture as noisier than actual noise."""
    assert dev._noise_sigma(_detailed_scene()) < 1.0
    assert dev._noise_sigma(_add_noise(_flat_scene(), 18)) > 5.0


def test_cache_key_covers_the_spatial_tuning(monkeypatch):
    """The new knobs must sit INSIDE the algo hash, not beside it."""
    baseline = dev._develop_algo_version()

    monkeypatch.setattr(dev, "DENOISE_NOISE_FULL", dev.DENOISE_NOISE_FULL + 1.0)
    assert dev._develop_algo_version() != baseline

    monkeypatch.undo()
    monkeypatch.setattr(dev, "SHARPEN_MAX_AMOUNT", dev.SHARPEN_MAX_AMOUNT + 0.1)
    assert dev._develop_algo_version() != baseline

    monkeypatch.undo()
    monkeypatch.setattr(dev, "DENOISE_MAX_DIM", dev.DENOISE_MAX_DIM + 256.0)
    assert dev._develop_algo_version() != baseline


# ── Batch edges, frame stats, export quality ─────────────────────────


def test_batch_aborts_on_an_undecodable_file_instead_of_skipping(tmp_path):
    """One bad file fails the whole batch closed, naming the offender — no silent skip."""
    good = _write(tmp_path, "good.jpg", _cast_frame())
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"this is not a jpeg")

    with pytest.raises(dev.PhotoDecodeError, match=r"bad\.jpg"):
        dev.develop_batch([good, bad])


def test_batch_with_no_files_returns_empty():
    assert dev.develop_batch([]) == []


def test_frame_stats_of_a_uniform_frame_are_exact():
    """Known array in, exactly-computable segment stats out."""
    frame = np.full((24, 32, 3), (200, 100, 50), dtype=np.uint8)

    stats = dev._frame_stats(frame)

    luma = 0.2126 * 200 + 0.7152 * 100 + 0.0722 * 50  # BT.709
    assert isinstance(stats, SegmentStats)
    assert stats.black_point == pytest.approx(luma, abs=0.01)
    assert stats.white_point == pytest.approx(luma, abs=0.01)
    assert stats.mid_tone == pytest.approx(luma, abs=0.01)
    assert stats.saturation == pytest.approx((200 - 50) / 200)
    assert (stats.r_mean, stats.g_mean, stats.b_mean) == pytest.approx(
        (200 / 255, 100 / 255, 50 / 255)
    )
    # no pixel clears the highlight knee and none outshines the median, so both
    # region meters report zero area and the fg stats mirror the whole frame
    assert stats.highlight_frac == 0.0
    assert stats.sky_fraction == 0.0
    assert stats.fg_mid_tone == pytest.approx(stats.mid_tone)
    assert (stats.fg_r_mean, stats.fg_g_mean, stats.fg_b_mean) == pytest.approx(
        (stats.r_mean, stats.g_mean, stats.b_mean)
    )


def _capture_save_quality(monkeypatch):
    """Record the kwargs PIL's save receives while still writing the real file."""
    captured = {}
    original = Image.Image.save

    def spy(self, fp, format=None, **kwargs):
        captured.update(kwargs)
        return original(self, fp, format, **kwargs)

    monkeypatch.setattr(Image.Image, "save", spy)
    return captured


def test_export_honors_an_explicit_jpeg_quality(tmp_path, monkeypatch):
    captured = _capture_save_quality(monkeypatch)

    out = dev.export_jpeg(_cast_frame(), tmp_path / "explicit.jpg", quality=42)

    assert captured["quality"] == 42
    assert out.is_file()


def test_export_defaults_to_the_configured_jpeg_quality(tmp_path, monkeypatch):
    import gomoveshift.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "get_config",
        lambda: SimpleNamespace(photos=SimpleNamespace(jpeg_quality=57)),
    )
    captured = _capture_save_quality(monkeypatch)

    dev.export_jpeg(_cast_frame(), tmp_path / "default.jpg")

    assert captured["quality"] == 57
