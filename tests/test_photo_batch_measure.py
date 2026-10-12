"""develop_batch measure-pass tests: bounded measure decode, unchanged grade.

The measure pass exists only to feed clustering, so it must never pay the
full-resolution demosaic — that is the grade pass's job (and its cost).
"""

import numpy as np
import pytest
import rawpy
from PIL import Image

from gomoveshift.pipeline.color_correction import GRADE_MODE_PHOTO, compute_cluster_correction
from gomoveshift.pipeline.photos.develop import (
    _frame_stats,
    decode,
    develop,
    develop_batch,
)


def _spy_rawpy(monkeypatch, pixels):
    """Fake rawpy.imread whose postprocess records every call's kwargs."""
    calls = []

    class _FakeRaw:
        raw_image = np.zeros(pixels.shape[:2], dtype=np.uint16)

        def postprocess(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("half_size"):
                return np.ascontiguousarray(pixels[::2, ::2])
            return pixels

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(rawpy, "imread", lambda _path: _FakeRaw())
    return calls


def test_measure_pass_decodes_raw_at_half_size(tmp_path, monkeypatch):
    """Only the grade pass may run the full demosaic — the measure pass is bounded."""
    pixels = np.full((6, 8, 3), 90, dtype=np.uint8)
    calls = _spy_rawpy(monkeypatch, pixels)
    path = tmp_path / "a.dng"
    path.write_bytes(b"placeholder")

    results = develop_batch([path])

    assert len(results) == 1
    assert [bool(k.get("half_size")) for k in calls] == [True, False], (
        "expected one half-size measure decode then one full grade decode"
    )


def test_measure_stats_track_full_decode_stats(tmp_path):
    """Calibration: the bounded measure decode stays representative of the full frame.

    Both paths end at the same calibrated metering resize, so their percentile
    stats differ only by two-step vs one-step area averaging (plus the RAW
    half-size decode's 2x2 binning) — far inside the 0.3 cluster threshold on
    the 0-1 feature scale.
    """
    from gomoveshift.pipeline.photos.develop import MEASURE_MAX_DIM, _measure_rgb

    rng = np.random.default_rng(7)
    frame = rng.integers(0, 256, size=(64, 3200, 3), dtype=np.uint8)
    path = tmp_path / "big.png"
    Image.fromarray(frame, mode="RGB").save(path)

    measured_rgb = _measure_rgb(path)
    assert max(measured_rgb.shape[:2]) <= MEASURE_MAX_DIM

    measured = _frame_stats(measured_rgb)
    full = _frame_stats(decode(path).rgb)

    assert measured.black_point == pytest.approx(full.black_point, abs=8)
    assert measured.white_point == pytest.approx(full.white_point, abs=8)
    assert measured.mid_tone == pytest.approx(full.mid_tone, abs=8)
    assert measured.saturation == pytest.approx(full.saturation, abs=0.05)


def test_small_batch_grade_matches_per_photo_develop(tmp_path):
    """A frame under the measure bound is measured from its exact pixels, so the
    batch output is byte-identical to developing with that cluster correction."""
    ramp = np.linspace(60, 200, 32, dtype=np.float32)
    frame = np.stack([np.tile(ramp, (24, 1)) + d for d in (30, 0, -20)], axis=2)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    path = tmp_path / "a.png"
    Image.fromarray(frame, mode="RGB").save(path)

    (batch_result,) = develop_batch([path])

    params = compute_cluster_correction([_frame_stats(frame)], mode=GRADE_MODE_PHOTO)
    direct = develop(path, params, cluster_index=0)
    assert batch_result.params == params
    assert batch_result.cluster_index == 0
    assert np.array_equal(batch_result.rgb, direct.rgb)


def test_batch_splits_opposite_casts_end_to_end(tmp_path):
    """Warm and cool frames at near-equal luma get separate cluster corrections."""
    warm = tmp_path / "warm.png"
    cool = tmp_path / "cool.png"
    Image.new("RGB", (32, 24), (200, 150, 100)).save(warm)
    Image.new("RGB", (32, 24), (100, 150, 200)).save(cool)

    results = develop_batch([warm, cool])

    assert results[0].cluster_index != results[1].cluster_index


def test_clustering_miss_fails_loudly_instead_of_dropping(tmp_path, monkeypatch):
    """An index the clustering fails to assign must raise, naming the dropped
    file — matching decode's fail-closed contract, never a silent filter."""
    from gomoveshift.pipeline.photos import develop as dev

    paths = [tmp_path / "a.png", tmp_path / "b.png"]
    for path in paths:
        Image.new("RGB", (16, 12), (120, 120, 120)).save(path)
    monkeypatch.setattr(dev, "cluster_files_by_lighting", lambda stats, threshold: [[0]])

    with pytest.raises(RuntimeError, match=r"b\.png"):
        develop_batch(paths)


def test_batch_threshold_references_the_span_constant():
    """The batch threshold IS the span-splitting constant — one knob (the None
    sentinel resolves ``photos.cluster_variance_threshold``, then the span
    constant; see test_photo_develop_cli), never a drifting 0.3 copy of it."""
    import inspect

    from gomoveshift.pipeline.photos import develop as dev

    sig = inspect.signature(develop_batch)
    assert sig.parameters["variance_threshold"].default is None
    assert "0.3" not in inspect.getsource(develop_batch)
    helper = dev.compute_batch_corrections
    assert inspect.signature(helper).parameters["variance_threshold"].default is None
    assert "0.3" not in inspect.getsource(helper)
