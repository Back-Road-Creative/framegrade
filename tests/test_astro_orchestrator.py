"""``prepare_frames`` tests. No RAW fixtures: rawpy is faked (test_photo_he_dng.py's
pattern) only where HE routing is under test; other tests mock decode/measure."""

from dataclasses import replace
from pathlib import Path

import time

import numpy as np
import pytest
import rawpy

import gomoveshift.pipeline.photos.astro_orchestrator as om
import gomoveshift.pipeline.photos.develop as dm
from gomoveshift.pipeline.photos.astro_calibrate import (
    AstroCalibrationError,
    AstroDarkIdentityMismatchError,
    AstroHighEfficiencyError,
    DecodedAstroFrame,
)
from gomoveshift.pipeline.photos.astro_develop import AstroDevelopResult, develop_stack
from gomoveshift.pipeline.photos.astro_orchestrator import (
    AstroOrchestratorError,
    MeteorPolicy,
    MeteorPolicyError,
    PreparedFrames,
    prepare_frames,
    stack_run,
)
from gomoveshift.pipeline.photos.astro_register import RegistrationError, RegistrationResult
from gomoveshift.pipeline.photos.astro_stack import MIN_AUTO_HORIZON_STARS, StackResult
from gomoveshift.pipeline.photos.astro_triage import (
    MIN_USABLE_STAR_COUNT,
    FrameMetrics,
    StreakClassification,
    TriageReport,
)

from test_astro_calibrate import _identity  # the CaptureIdentity builder, defined once

_STUB_RGB = np.zeros((4, 4, 3), dtype=np.float32)


def _stub_decode(path, master_dark, hotmap):
    return DecodedAstroFrame(rgb=_STUB_RGB, path=Path(path), decoder="stub-linear", bit_depth=16)


def _measure_from(table):
    """``measure_frame`` stand-in, keyed by path -> (star_count, fwhm_px)."""

    def _measure(path, luma, mask=None, exposure_s=None, focal_mm=None, **kwargs):
        star_count, fwhm_px = table[Path(path)]
        usable = star_count >= MIN_USABLE_STAR_COUNT
        return FrameMetrics(Path(path), star_count, 0.0, fwhm_px, [], usable)

    return _measure


def _paths(n, tmp_path):
    return [tmp_path / f"frame{i}.nef" for i in range(n)]


class _FakeRawBase:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_reference_is_the_best_frame_not_the_first(monkeypatch, tmp_path):
    paths = _paths(5, tmp_path)
    table = {p: (6, 2.0) for p in paths}
    table[paths[1]] = (9, 2.5)
    table[paths[2]] = (9, 2.0)  # ties paths[1] on star_count, wins on lower fwhm
    monkeypatch.setattr(om, "decode_linear", _stub_decode)
    monkeypatch.setattr(om, "measure_frame", _measure_from(table))

    result = prepare_frames(paths)

    assert isinstance(result, PreparedFrames)
    assert result.reference_index == 2


def test_the_reference_is_centred_among_frames_tied_on_star_count(monkeypatch, tmp_path):
    """Star count climbs through a real night -- the sky darkens and the band rises -- so
    the frame with the MOST stars is nearly always a chunk's LAST frame, by a margin far
    too small to mean anything. Picking it asks registration to bridge the full span of
    sky rotation from frame 1 instead of half of it, and costs frames to ``too few
    outer-third inliers``. Inside the tie band the reference belongs at the chunk's
    MIDDLE, where the worst-case rotation is halved."""
    paths = _paths(10, tmp_path)
    # A rising count, last frame best -- every frame within REFERENCE_STAR_COUNT_TIE_FRACTION.
    table = {p: (400 + 2 * i, 2.0) for i, p in enumerate(paths)}
    monkeypatch.setattr(om, "decode_linear", _stub_decode)
    monkeypatch.setattr(om, "measure_frame", _measure_from(table))

    result = prepare_frames(paths)

    # Centre of 0..9 is 4.5, so 4 and 5 are equidistant; the richer of the two wins.
    assert result.reference_index == 5


def test_a_frame_genuinely_richer_in_stars_still_beats_the_centre(monkeypatch, tmp_path):
    """Centring only breaks near-ties. A frame outside the tie band is a better reference
    on the merit the old rule measured, and must still win however far from the middle."""
    paths = _paths(10, tmp_path)
    table = {p: (400, 2.0) for p in paths}
    table[paths[9]] = (600, 2.0)  # 50% richer: far outside the band, not a tie
    monkeypatch.setattr(om, "decode_linear", _stub_decode)
    monkeypatch.setattr(om, "measure_frame", _measure_from(table))

    result = prepare_frames(paths)

    assert result.reference_index == 9


@pytest.mark.parametrize("n", [3, 4])
def test_it_accepts_min_stack_frames_not_min_usable_star_count(monkeypatch, tmp_path, n):
    paths = _paths(n, tmp_path)
    monkeypatch.setattr(om, "decode_linear", _stub_decode)
    monkeypatch.setattr(om, "measure_frame", _measure_from({p: (6, 2.0) for p in paths}))

    result = prepare_frames(paths)

    assert len(result.frames) == n
    assert 0 <= result.reference_index < n


def test_parallel_decode_keeps_frames_in_source_order(monkeypatch, tmp_path):
    """Concurrency must not reorder anything. ``frames`` and ``metrics`` are positionally
    paired, ``reference_index`` indexes into both, and the CLI names chunks by first/last
    source frame -- a decode that returns out of order silently pairs each frame with a
    DIFFERENT frame's triage and mislabels every output.

    The stub finishes the LAST path first and the first path last, so a naive
    completion-ordered collection cannot accidentally pass.
    """
    paths = _paths(6, tmp_path)
    order = []

    def _staggered_decode(path, master_dark, hotmap):
        # Reverse-index sleep: path[0] sleeps longest and returns LAST, path[5] returns
        # first, so completion order is the reverse of submission for the first pool-full.
        time.sleep(0.02 * (len(paths) - paths.index(Path(path))))
        order.append(Path(path))
        return _stub_decode(path, master_dark, hotmap)

    table = {p: (6, 2.0) for p in paths}
    table[paths[4]] = (9, 2.0)  # the reference, so its index is checkable too
    monkeypatch.setattr(om, "decode_linear", _staggered_decode)
    monkeypatch.setattr(om, "measure_frame", _measure_from(table))

    result = prepare_frames(paths, decode_workers=4)

    assert [f.path for f in result.frames] == paths
    assert [m.path for m in result.metrics] == paths
    assert result.reference_index == 4
    # Proves the stub really did complete out of order -- otherwise this test would pass
    # against a serial implementation and prove nothing about concurrency.
    assert order != paths


def test_parallel_decode_reports_failures_against_the_right_paths(monkeypatch, tmp_path):
    """A frame that fails to decode is dropped from ``frames`` but named in ``failed``.
    Under concurrency the surviving frames must still line up with their own metrics --
    a failure in the middle must not shift every later frame onto the wrong triage."""
    paths = _paths(6, tmp_path)
    bad = {paths[1], paths[3]}

    def _decode(path, master_dark, hotmap):
        if Path(path) in bad:
            raise om.AstroHighEfficiencyError(f"stub: {Path(path).name} unreadable")
        return _stub_decode(path, master_dark, hotmap)

    monkeypatch.setattr(om, "decode_linear", _decode)
    monkeypatch.setattr(om, "_convert_he_to_dng_or_none", lambda p: None, raising=False)
    monkeypatch.setattr(om, "measure_frame", _measure_from({p: (6, 2.0) for p in paths}))

    result = prepare_frames(paths, decode_workers=4)

    survivors = [p for p in paths if p not in bad]
    assert [f.path for f in result.frames] == survivors
    assert [m.path for m in result.metrics] == survivors
    assert sorted(p for p, _ in result.failed) == sorted(bad)


def test_decode_is_called_with_none_dark_and_hotmap_when_darks_absent(monkeypatch, tmp_path):
    paths = _paths(5, tmp_path)
    calls = []

    def _recording_decode(path, master_dark, hotmap):
        calls.append((master_dark, hotmap))
        return _stub_decode(path, master_dark, hotmap)

    def _boom(*a, **k):
        raise AssertionError("build_master_dark must not run with no dark_paths")

    monkeypatch.setattr(om, "decode_linear", _recording_decode)
    monkeypatch.setattr(om, "measure_frame", _measure_from({p: (6, 2.0) for p in paths}))
    monkeypatch.setattr(om, "build_master_dark", _boom)

    result = prepare_frames(paths)

    assert len(result.frames) == 5
    assert calls == [(None, None)] * 5


def _recording_triage(captured):
    def _triage(metrics, *, run_baseline=None):
        captured["run_baseline"] = run_baseline
        return TriageReport(frames=tuple(metrics), cloud_frames=(), dew_frames=())

    return _triage


def test_cloud_baseline_threads_into_triage(monkeypatch, tmp_path):
    paths = _paths(3, tmp_path)
    monkeypatch.setattr(om, "decode_linear", _stub_decode)
    monkeypatch.setattr(om, "measure_frame", _measure_from({p: (6, 2.0) for p in paths}))
    captured = {}
    monkeypatch.setattr(om, "triage", _recording_triage(captured))

    sentinel = (100.0, 0.01)
    prepare_frames(paths, cloud_baseline=sentinel)

    assert captured["run_baseline"] == sentinel


def test_cloud_baseline_defaults_to_none(monkeypatch, tmp_path):
    paths = _paths(3, tmp_path)
    monkeypatch.setattr(om, "decode_linear", _stub_decode)
    monkeypatch.setattr(om, "measure_frame", _measure_from({p: (6, 2.0) for p in paths}))
    captured = {}
    monkeypatch.setattr(om, "triage", _recording_triage(captured))

    prepare_frames(paths)

    assert captured["run_baseline"] is None


def test_a_failed_he_conversion_lands_in_failed_without_killing_the_run(monkeypatch, tmp_path):
    good = _paths(5, tmp_path)
    bad = tmp_path / "he.nef"

    def _decode(path, master_dark, hotmap):
        if Path(path) == bad:
            raise AstroHighEfficiencyError(f"{path}: raw stream undecodable")
        return _stub_decode(path, master_dark, hotmap)

    monkeypatch.setattr(om, "decode_linear", _decode)
    monkeypatch.setattr(om, "measure_frame", _measure_from({p: (6, 2.0) for p in good}))
    monkeypatch.setattr(dm, "_convert_he_to_dng", lambda p: None)

    result = prepare_frames([*good, bad])

    assert len(result.frames) == 5
    assert len(result.failed) == 1
    failed_path, reason = result.failed[0]
    assert failed_path == bad
    assert "undecodable" in reason


def test_he_routes_through_convert_then_decode_linear_never_display_referred(monkeypatch, tmp_path):
    he_nef, dng_path = tmp_path / "he.nef", tmp_path / "he.dng"
    rgb16 = np.full((4, 4, 3), 5000, dtype=np.uint16)

    class _FakeHERaw(_FakeRawBase):
        @property
        def raw_image(self):
            raise rawpy.LibRawDataError("data corrupted at 1541636")

    class _FakeDngRaw(_FakeRawBase):
        raw_image = rgb16[:, :, 0]

        def postprocess(self, **kwargs):
            assert kwargs.get("gamma") == (1, 1)  # scene-linear, never display sRGB
            return rgb16

    routes = {str(he_nef): _FakeHERaw(), str(dng_path): _FakeDngRaw()}
    monkeypatch.setattr(rawpy, "imread", lambda p: routes[str(p)])
    monkeypatch.setattr(dm, "_convert_he_to_dng", lambda p: dng_path)

    def _boom(*a, **k):
        raise AssertionError("_decode_he_via_dng must never be called")

    monkeypatch.setattr(dm, "_decode_he_via_dng", _boom)

    decoded, reason = om.decode_with_he_fallback(he_nef, None, None)

    assert reason is None
    assert decoded.path == he_nef
    assert decoded.decoder == "he-dng-linear"
    np.testing.assert_array_equal(decoded.rgb, (rgb16 / 65535.0).astype(np.float32))


def test_it_raises_when_every_light_frame_fails_to_decode(monkeypatch, tmp_path):
    paths = _paths(3, tmp_path)

    def _always_fails(path, master_dark, hotmap):
        raise AstroHighEfficiencyError(f"{path}: raw stream undecodable")

    monkeypatch.setattr(om, "decode_linear", _always_fails)
    monkeypatch.setattr(dm, "_convert_he_to_dng", lambda p: None)

    with pytest.raises(AstroOrchestratorError):
        prepare_frames(paths)


def _stack_prepared(n, tmp_path, reference_index=0, shape=(4, 4, 3)):
    """A ``PreparedFrames`` of ``n`` frames; rgb uniformly filled with the frame's own
    index, so a stubbed ``detect_stars``/``_luma`` can identify which frame is which."""
    paths = _paths(n, tmp_path)
    frames = tuple(
        DecodedAstroFrame(
            rgb=np.full(shape, float(i), dtype=np.float32),
            path=paths[i],
            decoder="rawpy-linear",
            bit_depth=16,
        )
        for i in range(n)
    )
    metrics = tuple(FrameMetrics(paths[i], 6, 0.0, 2.0, [], True) for i in range(n))
    report = TriageReport(frames=metrics, cloud_frames=(), dew_frames=())
    return PreparedFrames(frames, metrics, report, reference_index, ())


def _stack_prepared_with_verdicts(
    n, tmp_path, *, reference_index=0, cloud=(), dew=(), unusable=(), star_counts=None, fwhm=None
):
    """``_stack_prepared``, but with cloud/dew/usable verdicts an operator can control
    per frame index, for triage-gating tests."""
    paths = _paths(n, tmp_path)
    frames = tuple(
        DecodedAstroFrame(
            rgb=np.full((4, 4, 3), float(i), dtype=np.float32),
            path=paths[i],
            decoder="rawpy-linear",
            bit_depth=16,
        )
        for i in range(n)
    )
    star_counts = star_counts or dict.fromkeys(range(n), 6)
    fwhm = fwhm or dict.fromkeys(range(n), 2.0)
    metrics = tuple(
        FrameMetrics(paths[i], star_counts[i], 0.0, fwhm[i], [], i not in unusable)
        for i in range(n)
    )
    report = TriageReport(
        frames=metrics,
        cloud_frames=tuple(paths[i] for i in cloud),
        dew_frames=tuple(paths[i] for i in dew),
    )
    return PreparedFrames(frames, metrics, report, reference_index, ())


def _stub_detect_stars(luma, sigma=None, mask=None):
    """Encodes the frame's fill value into every star's flux, so a stubbed ``register``
    can tell frames apart without touching real pixel data. Returns
    ``MIN_AUTO_HORIZON_STARS`` detections so a test below exercises its own subject
    rather than the automatic-horizon evidence floor; they sit at ONE pixel, so the
    density map -- and every mask derived from it -- is identical to the
    single-detection stub this replaced."""
    return np.repeat(
        np.array([[1.0, 1.0, float(luma[0, 0])]], dtype=np.float32),
        MIN_AUTO_HORIZON_STARS,
        axis=0,
    )


def _detect_n_stars(n):
    """``detect_stars`` stand-in returning exactly ``n`` detections over the frame's top
    60% -- ``stack_run``'s floor counts detections, so the count IS the fixture."""

    def _detect(luma, sigma=None, mask=None):
        rng = np.random.default_rng(3)
        height, width = luma.shape
        return np.stack(
            [
                rng.uniform(0, width, n).astype(np.float32),
                rng.uniform(0, height * 0.6, n).astype(np.float32),
                np.full(n, 5.0, dtype=np.float32),
            ],
            axis=1,
        )

    return _detect


_IDENTITY_REGISTRATION = RegistrationResult(
    transform=np.eye(2, 3, dtype=np.float64), n_matched=10, inlier_fraction=0.9
)

_STUB_DEVELOP_RESULT = AstroDevelopResult(
    rgb_uint16=np.zeros((4, 4, 3), dtype=np.uint16),
    black_point=0.0,
    m_values=[1.0],
    background_before=0.0,
    background_after=0.0,
    background_gains=(1.0, 1.0, 1.0),
)


def _make_fake_register(bad_fill):
    def _fake_register(luma_ref, stars_ref, luma_frame, stars_frame, **kwargs):
        # The orchestrator must hand registration the same sky mask it detects stars
        # under -- without it a landscape frame's coarse prior is dragged toward zero
        # by stationary foreground and every star pair is lost (astro_register).
        assert kwargs.get("sky_mask") is not None
        if float(stars_frame[0, 2]) == bad_fill:
            raise RegistrationError("stub: no match")
        return _IDENTITY_REGISTRATION

    return _fake_register


def _stub_stack_capturing(captured):
    def _fake_stack(aligned_frames, unaligned_frames, mask, *, sky_validity=None, **_kwargs):
        captured["aligned"] = list(aligned_frames)
        captured["unaligned"] = list(unaligned_frames)
        captured["mask"] = mask
        # A copy, like aligned/unaligned above: stack_run releases its own reference
        # to this list once stack() returns, so a bare alias would read back empty.
        captured["sky_validity"] = list(sky_validity) if sky_validity is not None else None
        return StackResult(
            rgb=np.zeros((4, 4, 3), dtype=np.float32),
            n_frames=len(aligned_frames),
            mean_rejected_fraction=0.0,
        )

    return _fake_stack


def _apply_stack_run_stubs(monkeypatch, bad_fill=-1.0, stub_horizon_mask=True):
    # No photometry stub: the real pedestal-only _frame_photometry is a no-op on these
    # flat fixtures, so every test below exercises it rather than an identity stand-in.
    monkeypatch.setattr(om, "detect_stars", _stub_detect_stars)
    if stub_horizon_mask:
        monkeypatch.setattr(
            om, "horizon_mask", lambda shape_hw, **kw: np.ones(shape_hw, dtype=np.float32)
        )
    monkeypatch.setattr(om, "warp_to_reference", lambda rgb, transform, out_shape: rgb)
    monkeypatch.setattr(om, "register", _make_fake_register(bad_fill))
    monkeypatch.setattr(om, "develop_stack", lambda rgb, sky_mask=None: _STUB_DEVELOP_RESULT)
    monkeypatch.setattr(om, "classify_streaks", lambda metrics: ())
    monkeypatch.setattr(dm, "export_tiff16", lambda rgb, out_path: Path(out_path))


def test_reference_carries_no_registration_and_decoder_tag_reflects_survivors(
    monkeypatch, tmp_path
):
    prepared = _stack_prepared(4, tmp_path)  # frame 2's registration is made to fail below
    _apply_stack_run_stubs(monkeypatch, bad_fill=2.0)
    captured_stack: dict = {}
    captured_payload: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing(captured_stack))
    monkeypatch.setattr(
        om,
        "stack_sidecar_payload",
        lambda **kwargs: captured_payload.update(kwargs) or {"stub": True},
    )
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    used = captured_payload["used_frames"]
    assert used[0].registration is None
    assert used[0].metrics.path == prepared.frames[0].path
    assert [u.metrics.path for u in used[1:]] == [prepared.frames[1].path, prepared.frames[3].path]
    assert all(u.registration is _IDENTITY_REGISTRATION for u in used[1:])
    # 3 frames survive stacking (ref + frame1 + frame3) -- the tag reflects that, not
    # the 4 frames prepare_frames originally handed over.
    assert captured_payload["decoder"] == "rawpy-linear+astrostack3"
    assert report.rejected_frames == ((prepared.frames[2].path, "stub: no match"),)


def test_the_sidecar_records_the_darks_the_master_was_built_with(monkeypatch, tmp_path):
    """The sidecar is the audit record for how a master was built, so it must not deny
    provenance that exists: ``stack_sidecar_payload`` takes ``dark_paths``, so the call
    site must pass them. A run that DID subtract a validated master dark must not record
    ``inputs.dark: []`` while its own artifact-name hash encodes the darks."""
    darks = tuple(tmp_path / f"dark{i}.NEF" for i in range(2))
    prepared = replace(_stack_prepared(3, tmp_path), dark_paths=darks)
    _apply_stack_run_stubs(monkeypatch)
    captured_payload: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing({}))
    monkeypatch.setattr(
        om, "stack_sidecar_payload", lambda **kwargs: captured_payload.update(kwargs) or {}
    )
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    stack_run(prepared, tmp_path / "stack.tiff")

    assert tuple(captured_payload["dark_paths"]) == darks


def test_a_no_darks_run_claims_no_darks_in_its_sidecar(monkeypatch, tmp_path):
    prepared = _stack_prepared(3, tmp_path)
    _apply_stack_run_stubs(monkeypatch)
    captured_payload: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing({}))
    monkeypatch.setattr(
        om, "stack_sidecar_payload", lambda **kwargs: captured_payload.update(kwargs) or {}
    )
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    stack_run(prepared, tmp_path / "stack.tiff")

    assert tuple(captured_payload["dark_paths"]) == ()


def test_a_registration_failure_is_dropped_without_killing_the_run(monkeypatch, tmp_path):
    prepared = _stack_prepared(4, tmp_path)
    _apply_stack_run_stubs(monkeypatch, bad_fill=2.0)
    captured_stack: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing(captured_stack))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    assert len(captured_stack["aligned"]) == len(captured_stack["unaligned"]) == 3
    assert not any(f[0, 0, 0] == 2.0 for f in captured_stack["aligned"])
    assert not any(f[0, 0, 0] == 2.0 for f in captured_stack["unaligned"])
    assert report.rejected_frames == ((prepared.frames[2].path, "stub: no match"),)
    # sky_validity is paired with aligned/unaligned -- the rejected frame's own
    # validity never reaches stack(); the reference's is unconditional full coverage.
    assert len(captured_stack["sky_validity"]) == 3
    assert captured_stack["sky_validity"][0].all()


def test_too_few_registered_frames_raises_instead_of_stacking(monkeypatch, tmp_path):
    prepared = _stack_prepared(4, tmp_path)

    def _always_fails(luma_ref, stars_ref, luma_frame, stars_frame, **kwargs):
        raise RegistrationError("stub: no match")

    monkeypatch.setattr(om, "detect_stars", _stub_detect_stars)
    monkeypatch.setattr(om, "register", _always_fails)

    with pytest.raises(AstroOrchestratorError):
        stack_run(prepared, tmp_path / "stack.tiff")


def test_triage_rejected_frame_is_excluded_from_stack_and_listed_with_reason(monkeypatch, tmp_path):
    prepared = _stack_prepared_with_verdicts(4, tmp_path, cloud=(2,))
    _apply_stack_run_stubs(monkeypatch)
    captured_stack: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing(captured_stack))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    # 3 of 4 frames stack (ref 0 + frame1 + frame3); the cloud-flagged frame 2 never
    # reaches registration.
    assert len(captured_stack["aligned"]) == 3
    assert not any(f[0, 0, 0] == 2.0 for f in captured_stack["aligned"])
    reasons = dict(report.rejected_frames)
    assert prepared.frames[2].path in reasons
    assert "cloud" in reasons[prepared.frames[2].path]


def test_gating_below_min_stack_frames_raises_with_per_reason_counts(monkeypatch, tmp_path):
    prepared = _stack_prepared_with_verdicts(4, tmp_path, cloud=(1, 2))
    _apply_stack_run_stubs(monkeypatch)

    with pytest.raises(AstroOrchestratorError) as exc_info:
        stack_run(prepared, tmp_path / "stack.tiff")

    message = str(exc_info.value)
    assert "cloud=2" in message
    assert "dew=0" in message
    assert "unusable=0" in message


def test_reference_lands_on_an_accepted_frame_even_when_the_best_frame_was_rejected(
    monkeypatch, tmp_path
):
    # frame1 has the highest star_count -- prepare_frames would have picked it as the
    # reference -- but it's cloud-flagged, so stack_run must fall back to the best
    # ACCEPTED frame under the SAME rule prepare_frames uses: frames 0/2/3 are tied on
    # star_count, so the one nearest the chunk centre (1.5) wins, which is frame2.
    prepared = _stack_prepared_with_verdicts(
        4, tmp_path, reference_index=1, cloud=(1,), star_counts={0: 6, 1: 9, 2: 6, 3: 6}
    )
    _apply_stack_run_stubs(monkeypatch)
    captured_payload: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing({}))
    monkeypatch.setattr(
        om, "stack_sidecar_payload", lambda **kwargs: captured_payload.update(kwargs) or {}
    )
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    used = captured_payload["used_frames"]
    assert used[0].registration is None
    assert used[0].metrics.path == prepared.frames[2].path
    assert any(p == prepared.frames[1].path for p, _reason in report.rejected_frames)


def test_gate_triage_false_restores_the_old_all_frames_behavior(monkeypatch, tmp_path):
    prepared = _stack_prepared_with_verdicts(4, tmp_path, cloud=(2,))
    _apply_stack_run_stubs(monkeypatch)
    captured_stack: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing(captured_stack))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    report = stack_run(prepared, tmp_path / "stack.tiff", gate_triage=False)

    assert len(captured_stack["aligned"]) == 4  # cloud verdict ignored, old behavior
    assert report.rejected_frames == ()


def test_a_none_sidecar_still_yields_a_written_image_and_a_non_raising_report(
    monkeypatch, tmp_path
):
    prepared = _stack_prepared(3, tmp_path)
    _apply_stack_run_stubs(monkeypatch)
    monkeypatch.setattr(om, "stack", _stub_stack_capturing({}))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {"stub": True})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    out_path = tmp_path / "stack.tiff"
    report = stack_run(prepared, out_path)

    assert report.sidecar_path is None
    assert report.image_path == out_path


def test_a_viewable_preview_lands_beside_every_master(monkeypatch, tmp_path):
    """A run must hand the operator something they can open, not only a 16-bit .tif.

    Robust by construction rather than a flag: a master nobody can look at without a
    second tool is how an ad-hoc render ends up being the deliverable.
    """
    prepared = _stack_prepared(3, tmp_path)
    _apply_stack_run_stubs(monkeypatch)
    monkeypatch.setattr(om, "stack", _stub_stack_capturing({}))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {"stub": True})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)
    written: dict = {}
    monkeypatch.setattr(
        dm,
        "export_jpeg",
        lambda rgb, out_path, quality=None: (
            written.update(path=Path(out_path), rgb=rgb) or Path(out_path)
        ),
    )

    report = stack_run(prepared, tmp_path / "stack.tiff")

    assert report.preview_path == tmp_path / "stack.preview.jpg"
    assert written["path"] == report.preview_path
    assert written["rgb"].dtype == np.uint8, "a preview is a display render, not archival data"


def test_a_failed_preview_never_costs_the_master(monkeypatch, tmp_path):
    """The .tif is the artifact hours of stacking bought; a cosmetic JPEG is not."""
    prepared = _stack_prepared(3, tmp_path)
    _apply_stack_run_stubs(monkeypatch)
    monkeypatch.setattr(om, "stack", _stub_stack_capturing({}))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {"stub": True})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    def _boom(rgb, out_path, quality=None):
        raise OSError("no space left on device")

    monkeypatch.setattr(dm, "export_jpeg", _boom)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    assert report.image_path == tmp_path / "stack.tiff"
    assert report.preview_path is None


def test_linear_only_writes_complete_artifact_and_skips_develop_and_tiff(monkeypatch, tmp_path):
    prepared = _stack_prepared(3, tmp_path)
    _apply_stack_run_stubs(monkeypatch)
    shape = prepared.frames[0].rgb.shape
    component_result = StackResult(
        rgb=np.zeros(shape, dtype=np.float32),
        n_frames=3,
        mean_rejected_fraction=0.0,
        sky_sum=np.zeros(shape, dtype=np.float32),
        sky_count=np.full(shape[:2], 3, dtype=np.uint16),
        ground=np.zeros(shape, dtype=np.float32),
    )
    monkeypatch.setattr(om, "stack", lambda *a, **k: component_result)
    monkeypatch.setattr(om, "read_capture_identity", lambda path: _identity(path=Path(path)))
    monkeypatch.setattr(
        om, "develop_stack", lambda *a, **k: pytest.fail("linear-only must not develop")
    )
    captured = {}
    monkeypatch.setattr(
        om,
        "write_linear_artifact",
        lambda destination, result, **kwargs: captured.update(
            destination=Path(destination), result=result, **kwargs
        ),
    )
    destination = tmp_path / "chunk.linear"

    report = stack_run(
        prepared,
        None,
        horizon_frac=0.5,
        linear_artifact_path=destination,
        linear_only=True,
    )

    assert report.image_path is None and report.develop is None
    assert report.linear_artifact_path == destination
    assert captured["destination"] == destination
    assert captured["result"] is component_result
    assert captured["source_frames"] == [str(frame.path.resolve()) for frame in prepared.frames]
    assert captured["mask_provenance"]["mode"] == "explicit"
    assert captured["capture_identity"]["camera"] == "NIKON Z 6"
    assert captured["registration_provenance"][0]["transform"] == "identity"
    assert len(captured["photometry_provenance"]) == 3


def test_linear_artifact_registration_provenance_carries_per_frame_quality(monkeypatch, tmp_path):
    prepared = _stack_prepared(3, tmp_path)
    _apply_stack_run_stubs(monkeypatch)
    shape = prepared.frames[0].rgb.shape
    component_result = StackResult(
        rgb=np.zeros(shape, dtype=np.float32),
        n_frames=3,
        mean_rejected_fraction=0.0,
        sky_sum=np.zeros(shape, dtype=np.float32),
        sky_count=np.full(shape[:2], 3, dtype=np.uint16),
        ground=np.zeros(shape, dtype=np.float32),
    )
    monkeypatch.setattr(om, "stack", lambda *a, **k: component_result)
    monkeypatch.setattr(om, "read_capture_identity", lambda path: _identity(path=Path(path)))
    radial = {
        "center": {"n": 4, "median_px": 0.1, "p95_px": 0.2},
        "middle": {"n": 6, "median_px": 0.2, "p95_px": 0.3},
        "outer": {"n": 2, "median_px": 0.4, "p95_px": 0.6},
    }
    non_reference_registration = RegistrationResult(
        transform=np.eye(2, 3, dtype=np.float64),
        n_matched=10,
        inlier_fraction=0.9,
        radial_inlier_residuals_px=radial,
    )
    monkeypatch.setattr(om, "register", lambda *a, **k: non_reference_registration)
    captured = {}
    monkeypatch.setattr(
        om,
        "write_linear_artifact",
        lambda destination, result, **kwargs: captured.update(
            destination=Path(destination), result=result, **kwargs
        ),
    )

    stack_run(
        prepared,
        None,
        horizon_frac=0.5,
        linear_artifact_path=tmp_path / "chunk.linear",
        linear_only=True,
    )

    provenance = captured["registration_provenance"]
    reference_entry = provenance[0]  # frame 0 is the reference: registration is None
    assert reference_entry["radial_inlier_residuals_px_in_sample"] is None
    assert reference_entry["triage"]["star_count"] == 6
    assert reference_entry["triage"]["fwhm_px"] == 2.0
    assert reference_entry["triage"]["background_median"] == 0.0
    non_reference_entry = provenance[1]
    assert non_reference_entry["radial_inlier_residuals_px_in_sample"] == radial
    assert non_reference_entry["triage"]["star_count"] == 6
    assert "exposure_s" in non_reference_entry
    assert "captured_at" in non_reference_entry


def test_stack_run_releases_per_frame_alignment_buffers_after_last_use(monkeypatch, tmp_path):
    """``aligned_frames``/``unaligned_frames``/``aligned_by_index`` must not outlive
    their last consumer -- otherwise peak RSS ratchets upward across consecutive
    chunks in one long-lived process (``preflight.py`` sizes each chunk against
    CURRENT free memory, so a held buffer starves a later, unrelated chunk)."""
    prepared = _stack_prepared(3, tmp_path)
    _apply_stack_run_stubs(monkeypatch)
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)
    captured: dict = {}
    real_stack = om.stack

    def _capturing_stack(aligned_frames, unaligned_frames, mask, *, sky_validity=None, **kwargs):
        # Save the SAME list objects stack_run holds (not a copy) -- clearing them
        # in place after this call is exactly what the test asserts on below.
        captured["aligned_frames"] = aligned_frames
        captured["unaligned_frames"] = unaligned_frames
        return real_stack(
            aligned_frames, unaligned_frames, mask, sky_validity=sky_validity, **kwargs
        )

    monkeypatch.setattr(om, "stack", _capturing_stack)
    real_resolve = om._resolve_meteor_selection

    def _capturing_resolve(streaks, policy, aligned_by_index):
        captured["aligned_by_index"] = aligned_by_index
        return real_resolve(streaks, policy, aligned_by_index)

    monkeypatch.setattr(om, "_resolve_meteor_selection", _capturing_resolve)

    stack_run(prepared, tmp_path / "stack.tiff")

    assert captured["aligned_frames"] == []
    assert captured["unaligned_frames"] == []
    assert captured["aligned_by_index"] == {}


def test_star_density_mask_reads_ground_below_the_star_band(monkeypatch, tmp_path):
    """No ``om.horizon_mask`` stub -- the REAL one, fed a sparse star field."""
    h, w = 1000, 1500
    rng = np.random.default_rng(0)
    xs = rng.integers(0, w, size=300).astype(np.float32)
    ys = rng.integers(0, int(h * 0.6), size=300).astype(np.float32)
    flux = rng.uniform(1.0, 100.0, size=300).astype(np.float32)
    stars = np.stack([xs, ys, flux], axis=1)

    prepared = _stack_prepared(3, tmp_path, shape=(h, w, 3))
    _apply_stack_run_stubs(monkeypatch, stub_horizon_mask=False)
    monkeypatch.setattr(om, "detect_stars", lambda luma, sigma=None, mask=None: stars)
    captured: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing(captured))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    stack_run(prepared, tmp_path / "stack.tiff")

    mask = captured["mask"]
    assert mask[:50].mean() > 0.9  # inside the star band: sky
    assert mask[-50:].mean() < 0.1  # below the star band: ground


def _auto_horizon_case(monkeypatch, tmp_path, n_detected):
    """``stack_run`` fixture whose ONLY variable is the reference frame's detection
    count: real ``horizon_mask``, everything else stubbed."""
    prepared = _stack_prepared(3, tmp_path, shape=(200, 200, 3))
    _apply_stack_run_stubs(monkeypatch, stub_horizon_mask=False)
    monkeypatch.setattr(om, "detect_stars", _detect_n_stars(n_detected))
    monkeypatch.setattr(om, "stack", _stub_stack_capturing({}))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)
    return prepared


@pytest.mark.parametrize("n_detected", [0, 6, 17, 23, 27, 30, 35, 39])
def test_auto_horizon_refuses_below_the_detected_star_floor(monkeypatch, tmp_path, n_detected):
    """Low detection counts such as 6/17/23/27 must be refused: an automatic boundary
    estimated from so few stars is unreliable and would hand a large share of the real
    sky to the ground path with no refusal. 39 is one below the floor and 40 (the next
    test) is at it, so this sweep is carried ACROSS its own cutover rather than stopped
    short of it."""
    prepared = _auto_horizon_case(monkeypatch, tmp_path, n_detected)

    with pytest.raises(AstroOrchestratorError, match=r"automatic horizon needs >= 40 detected"):
        stack_run(prepared, tmp_path / "stack.tiff")


def test_the_refusal_offers_horizon_without_promising_the_chunk_will_then_stack(
    monkeypatch, tmp_path
):
    """The remedy is real but conditional, and the message must not overstate it.

    Re-running a refused chunk with ``--horizon`` clears this refusal but can still fail
    later, for example with ``only 1 of 10 frames registered against the reference (3
    required to stack)``. Registration is a separate gate, so ``--horizon`` resolves THIS
    refusal and guarantees nothing beyond it.
    """
    prepared = _auto_horizon_case(monkeypatch, tmp_path, 31)

    with pytest.raises(AstroOrchestratorError) as excinfo:
        stack_run(prepared, tmp_path / "stack.tiff")

    message = str(excinfo.value)
    assert "--horizon" in message, "the operator still needs to be told the remedy"
    assert "stack normally" not in message, (
        "must not promise a successful stack -- registration can still fail independently"
    )


@pytest.mark.parametrize("n_detected", [40, 41, 60, 300])
def test_auto_horizon_proceeds_at_and_above_the_floor(monkeypatch, tmp_path, n_detected):
    """The green side of the same sweep: 40 must NOT refuse, and no count above it does."""
    prepared = _auto_horizon_case(monkeypatch, tmp_path, n_detected)

    assert stack_run(prepared, tmp_path / "stack.tiff").stack.n_frames == 3


def test_an_operator_horizon_stacks_the_very_frame_the_floor_refuses(monkeypatch, tmp_path):
    """The refusal names ``--horizon FRACTION`` as the remedy, so the remedy must work
    on the refused frame: the floor gates the ESTIMATE, never the frame's stackability
    (which is ``astro_triage``'s separate, far lower usable-frame floor)."""
    prepared = _auto_horizon_case(monkeypatch, tmp_path, 6)

    report = stack_run(prepared, tmp_path / "stack.tiff", horizon_frac=0.8)

    assert report.stack.n_frames == 3


def test_the_floor_skips_the_chunk_at_the_cli_instead_of_writing_a_master(
    monkeypatch, tmp_path, capsys
):
    """Not merely "the exception type is raised": the refusal must reach the shipped
    OUTCOME. ``photos astro`` already catches ``AstroOrchestratorError`` per chunk and
    continues, so a single-chunk run whose reference frame is under the floor writes no
    master, exits 1, and tells the operator the remedy -- no second refusal path."""
    from gomoveshift.cli.handlers import photos as photos_handler
    from gomoveshift.cli.parser import _build_parser
    from gomoveshift.pipeline.photos import astro_sequence as seq_mod
    from gomoveshift.pipeline.photos import preflight as preflight_mod

    light = tmp_path / "IMG_0001.NEF"
    light.write_bytes(b"x")
    out_dir = tmp_path / "out"
    prepared = _auto_horizon_case(monkeypatch, tmp_path, 6)
    monkeypatch.setattr(photos_handler, "configure_logging", lambda *a, **k: None)
    monkeypatch.setattr(dm, "resolve_he_dng_converter", lambda: "/opt/dngc")
    monkeypatch.setattr(seq_mod, "detect_sequences", lambda ps, **kw: [(light,)])
    monkeypatch.setattr(seq_mod, "chunk_sequence", lambda run, **kw: [run])
    monkeypatch.setattr(photos_handler, "_astro_frame_megapixels", lambda p: 24.0)
    monkeypatch.setattr(preflight_mod, "preflight_astro_stack", lambda *a, **k: None)
    monkeypatch.setattr(om, "prepare_frames", lambda *a, **k: prepared)

    args = _build_parser().parse_args(f"photos astro {light} --output-dir {out_dir}".split())
    assert photos_handler.cmd_photos(args, None) == 1

    err = capsys.readouterr().err
    assert "automatic horizon needs >= 40 detected" in err
    assert "--horizon FRACTION" in err
    assert "No chunk produced a stacked master" in err
    assert not list(out_dir.glob("*.tif"))


def test_operator_horizon_registers_using_only_sky_stars(monkeypatch, tmp_path):
    """An operator ``horizon_frac`` must exclude a ground light from the
    star set fed to ``register``, not just the final sky/ground blend."""
    h, w = 200, 200
    sky_y, ground_y = 40, 160  # horizon_frac=0.6 -> boundary_row=120

    def _rgb(fill):
        luma = np.full((h, w), 0.05, dtype=np.float32)
        luma[sky_y, 50] = fill  # a real star, above the horizon
        luma[ground_y, 50] = fill  # a ground light, below the horizon
        return np.repeat(luma[:, :, None], 3, axis=2).astype(np.float32)

    paths = _paths(3, tmp_path)
    frames = tuple(
        DecodedAstroFrame(rgb=_rgb(1.0), path=paths[i], decoder="rawpy-linear", bit_depth=16)
        for i in range(3)
    )
    metrics = tuple(FrameMetrics(paths[i], 6, 0.0, 2.0, [], True) for i in range(3))
    report = TriageReport(frames=metrics, cloud_frames=(), dew_frames=())
    prepared = PreparedFrames(frames, metrics, report, 0, ())

    calls = []

    def _capturing_register(luma_ref, stars_ref, luma_frame, stars_frame, **kwargs):
        calls.append((stars_ref, stars_frame))
        return _IDENTITY_REGISTRATION

    monkeypatch.setattr(om, "register", _capturing_register)
    monkeypatch.setattr(om, "warp_to_reference", lambda rgb, transform, out_shape: rgb)
    monkeypatch.setattr(om, "_frame_photometry", lambda *a, **k: (0.0, 1.0))
    monkeypatch.setattr(
        om,
        "stack",
        lambda aligned, unaligned, mask, **kwargs: StackResult(
            rgb=np.zeros((h, w, 3), dtype=np.float32),
            n_frames=len(aligned),
            mean_rejected_fraction=0.0,
        ),
    )
    monkeypatch.setattr(
        om,
        "develop_stack",
        lambda rgb, sky_mask=None: AstroDevelopResult(
            rgb_uint16=np.zeros_like(rgb, dtype=np.uint16),
            black_point=0.0,
            m_values=[1.0],
            background_before=0.0,
            background_after=0.0,
            background_gains=(1.0, 1.0, 1.0),
        ),
    )
    monkeypatch.setattr(om, "classify_streaks", lambda metrics: ())
    monkeypatch.setattr(dm, "export_tiff16", lambda rgb, out_path: Path(out_path))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    stack_run(prepared, tmp_path / "stack.tiff", horizon_frac=0.6)

    assert len(calls) == 2  # frame 1 and frame 2, each registered against the reference
    for stars_ref, stars_frame in calls:
        assert stars_ref.shape[0] == 1
        assert stars_ref[0, 1] == pytest.approx(sky_y, abs=2)
        assert stars_frame.shape[0] == 1
        assert stars_frame[0, 1] == pytest.approx(sky_y, abs=2)


def test_auto_mode_still_registers_after_the_extra_reference_pass(monkeypatch, tmp_path):
    """Auto mode's mask comes FROM star density, so an unmasked first pass on
    the reference is unavoidable -- but the reference must be re-detected under its
    own mask before registration, like every other frame."""
    prepared = _stack_prepared(3, tmp_path)
    _apply_stack_run_stubs(monkeypatch, stub_horizon_mask=False)
    calls = []

    def _counting_detect_stars(luma, sigma=None, mask=None):
        calls.append(mask is not None)
        return _stub_detect_stars(luma, sigma=sigma, mask=mask)

    monkeypatch.setattr(om, "detect_stars", _counting_detect_stars)
    captured: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing(captured))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    # reference: one unmasked pass (seeds density), then a masked re-detect; every
    # other frame is detected once, already under the mask.
    assert calls == [False, True, True, True]
    assert report.image_path == tmp_path / "stack.tiff"


def _translated_registration(tx, ty=0.0):
    """A pure-translation ``RegistrationResult`` -- ``warp_validity`` on this
    transform must show the frame's own border vacated by exactly that shift."""
    transform = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float64)
    return RegistrationResult(transform=transform, n_matched=10, inlier_fraction=0.9)


def test_stack_receives_real_per_frame_validity_and_surfaces_coverage(monkeypatch, tmp_path):
    """``warp_validity``/``contributor_count`` existed with zero callers, so the
    thin-coverage refusal could never fire. A frame registered with a fractional
    (sub-pixel) 15.5px translation on a 20px-wide frame must vacate its own left
    border in reference space -- REAL ``stack()`` (not stubbed) must then measure a
    below-floor contributor count exactly there, while the identity-registered frame
    and the untouched reference keep it fully covered -- proving the value reaching
    ``stack`` is derived from each frame's own transform, not a constant/all-ones
    stand-in, and that the result reaches the report through the same ``StackResult``
    channel every other stack output already uses. The fractional shift (vs. a whole-
    pixel one) exercises ``warp_coverage``'s bilinear blend rather than only its
    binary 0/1 edges."""
    h, w = 20, 20
    prepared = _stack_prepared(3, tmp_path, shape=(h, w, 3))
    _apply_stack_run_stubs(monkeypatch)  # leaves the real astro_stack.stack in place
    registrations = [_IDENTITY_REGISTRATION, _translated_registration(tx=15.5)]
    monkeypatch.setattr(om, "register", lambda *a, **k: registrations.pop(0))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    coverage, thin = report.stack.coverage, report.stack.thin_coverage
    assert coverage is not None and thin is not None
    # ref + identity frame always cover the left border; the 15.5px-shifted frame never
    # does -- only 2 of 3 frames back it, below MIN_STACK_CONTRIBUTORS (3).
    assert thin[:, :3].all()
    # The right edge sits well inside the shifted frame's overlap -- all 3 contribute.
    assert not thin[:, -3:].any()
    assert coverage.below_floor_fraction > 0.0


def test_stack_run_develops_the_master_from_the_sky_mask_not_the_whole_frame(monkeypatch, tmp_path):
    """Wiring: the sky mask ``stack_run`` already feeds ``detect_stars`` and the
    photometry pedestal must ALSO reach ``develop_stack``, or the capability is unreachable
    from production. On this fixture only (32x32, rows 20+ at 0.30 against a 0.02 sky, mask
    = rows 0-19) the sky-only black point is 0.01593 and the whole-frame one 0.0."""
    prepared = _stack_prepared(3, tmp_path, shape=(32, 32, 3))
    _apply_stack_run_stubs(monkeypatch)
    monkeypatch.setattr(om, "develop_stack", develop_stack)  # the real develop chain
    sky = np.zeros((32, 32), dtype=bool)
    sky[:20] = True
    monkeypatch.setattr(om, "horizon_mask", lambda shape_hw, **kw: sky.astype(np.float32))
    stacked = np.full((32, 32, 3), 0.02, dtype=np.float32)
    stacked += (0.002 * np.random.default_rng(11).standard_normal((32, 32, 3))).astype(np.float32)
    stacked[20:] = 0.30  # a lit horizon band the whole-frame stats would swallow
    monkeypatch.setattr(
        om, "stack", lambda *a, **kw: StackResult(rgb=stacked, n_frames=3, mean_rejected_fraction=0)
    )

    report = stack_run(prepared, tmp_path / "stack.tiff")

    assert report.develop.black_point == pytest.approx(
        develop_stack(stacked, sky_mask=sky).black_point, rel=1e-9
    )
    assert report.develop.black_point != pytest.approx(develop_stack(stacked).black_point, rel=0.05)


def _textured_sky(height=10, width=10, sky=0.02, star_level=0.06, n_bright=3):
    """A small, non-degenerate 2-D 'sky' luma: overwhelmingly flat (a real dark-sky
    field) with a handful of brighter 'star' pixels for realism -- irrelevant to the
    median pedestal, which tolerates up to 50% contamination by construction (module
    comment above ``FramePhotometry``)."""
    luma = np.full(height * width, sky, dtype=np.float64)
    luma[:n_bright] = star_level
    return luma.reshape(height, width)


def test_frame_photometry_is_a_noop_for_an_identical_frame():
    luma = _textured_sky()
    mask = np.ones(luma.shape, dtype=bool)
    ref_pedestal = om._sky_pedestal(luma, mask)

    offset, scale = om._frame_photometry(luma, ref_pedestal, mask)

    assert offset == 0.0
    assert scale == 1.0


def test_frame_photometry_recovers_a_pedestal_shift_exactly_and_never_corrects_gain():
    """PEDESTAL-ONLY: ``scale`` is pinned at 1.0 -- a gain
    distortion is NOT corrected, only tested here as "stays 1.0", not
    "recovered"."""
    base = _textured_sky()
    mask = np.ones(base.shape, dtype=bool)
    ref_pedestal = om._sky_pedestal(base, mask)
    pedestal_shift = 0.01
    frame = base + pedestal_shift

    offset, scale = om._frame_photometry(frame, ref_pedestal, mask)

    assert scale == 1.0
    assert offset == pytest.approx(pedestal_shift, rel=1e-9, abs=1e-12)
    np.testing.assert_allclose(frame - offset, base, rtol=1e-9, atol=1e-12)


def test_sky_pedestal_ignores_ground_and_foreground_pixels():
    # SKEWED on purpose: 10 of the 50 sky pixels are star-lit. A symmetric fixture would
    # give mean == median and let a mean-based pedestal pass, which is the whole claim.
    sky_values = np.concatenate([np.linspace(0.01, 0.03, 40), np.full(10, 0.30)])
    ground_values = np.full(50, 5.0)  # bright foreground -- must never set the pedestal
    luma = np.concatenate([sky_values, ground_values]).reshape(10, 10)
    mask = np.zeros((10, 10), dtype=bool)
    mask[:5, :] = True  # sky rows only

    pedestal = om._sky_pedestal(luma, mask)

    assert pedestal == pytest.approx(float(np.median(sky_values)))
    assert pedestal != pytest.approx(float(np.mean(sky_values)))  # a mean would be dragged up


def test_stack_run_normalises_frames_photometrically_before_averaging(monkeypatch, tmp_path):
    """PEDESTAL-ONLY: only pedestal differs frame to frame (no gain) -- gain
    correction was cut, so a fixture that varied gain could no longer recover ``base``
    from ``arr`` after correction."""
    n, h, w, c = 3, 6, 6, 3
    base = _textured_sky(h, w).astype(np.float32)
    pedestals = [0.0, 0.004, -0.002]
    paths = _paths(n, tmp_path)
    frames = tuple(
        DecodedAstroFrame(
            rgb=np.repeat((base + pedestals[i])[:, :, None], c, axis=2).astype(np.float32),
            path=paths[i],
            decoder="rawpy-linear",
            bit_depth=16,
        )
        for i in range(n)
    )
    metrics = tuple(FrameMetrics(paths[i], 6, 0.0, 2.0, [], True) for i in range(n))
    report_triage = TriageReport(frames=metrics, cloud_frames=(), dew_frames=())
    prepared = PreparedFrames(frames, metrics, report_triage, 0, ())
    _apply_stack_run_stubs(monkeypatch)
    captured_stack: dict = {}
    monkeypatch.setattr(om, "stack", _stub_stack_capturing(captured_stack))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    assert len(report.photometry) == 3
    assert report.photometry[0] == om.FramePhotometry(paths[0], 0.0, 1.0, 1.0)
    for i in (1, 2):
        entry = next(p for p in report.photometry if p.path == paths[i])
        assert entry.scale == 1.0  # pinned, never estimated (module comment above)
        assert entry.offset == pytest.approx(pedestals[i], rel=1e-5, abs=1e-8)
        assert entry.weight == 1.0
    for arr in captured_stack["aligned"]:
        np.testing.assert_allclose(arr[:, :, 0], base, atol=2e-4)


def test_stack_run_photometry_survives_the_warp_zero_fill_border(monkeypatch, tmp_path):
    """The warp's zero-fill border used to fabricate scale/offset on
    identical frames under the (now-cut) percentile estimator. Real warp_to_reference +
    _frame_photometry run; only register is faked. Bound sits near the MEASURED offset
    (7.2e-7 at 17px drift, 5.7e-6 at 68px -- deterministic, fixed seed) with ~7x headroom.
    Measuring POST-warp instead gives 3.8e-4, so the border regression turns this red."""
    height, width = 40, 300
    rng = np.random.default_rng(1)
    drifts = [17, 68]
    big = rng.normal(0.02, 0.001, size=(height, width + max(drifts))).astype(np.float64)
    n_stars = int(0.012 * height * big.shape[1])
    big[rng.integers(0, height, n_stars), rng.integers(0, big.shape[1], n_stars)] = 0.06
    paths = _paths(3, tmp_path)

    def _frame(path, x0):
        rgb = np.repeat(big[:, x0 : x0 + width, None], 3, axis=2).astype(np.float32)
        return DecodedAstroFrame(rgb, path, "rawpy-linear", 16)

    frames, queue = [_frame(paths[0], 0)], []
    for i, drift in enumerate(drifts, start=1):
        frames.append(_frame(paths[i], drift))
        queue.append(RegistrationResult(np.array([[1.0, 0, drift], [0, 1.0, 0]]), 10, 0.9))
    metrics = tuple(FrameMetrics(paths[i], 6, 0.0, 2.0, [], True) for i in range(3))
    tr = TriageReport(frames=metrics, cloud_frames=(), dew_frames=())
    prepared = PreparedFrames(tuple(frames), metrics, tr, 0, ())

    real_warp = om.warp_to_reference
    _apply_stack_run_stubs(monkeypatch)
    monkeypatch.setattr(om, "warp_to_reference", real_warp)
    monkeypatch.setattr(om, "register", lambda *a, **k: queue.pop(0))
    monkeypatch.setattr(om, "stack", _stub_stack_capturing({}))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    assert report.rejected_frames == ()
    for e in report.photometry[1:]:
        assert e.scale == 1.0  # pinned, never estimated
        assert abs(e.offset) < 4e-5  # ~7x the largest measured offset (5.7e-6)


def test_photometry_is_a_noop_under_genuine_star_field_turnover(monkeypatch, tmp_path):
    """A real multi-hour drift turns the star FIELD over -- different
    stars enter/leave the frame -- unlike a fixture that only repositioned a
    FIXED in-frame star list (same count every frame, so a percentile statistic over
    VALUES couldn't tell the difference). Stars live on a canvas WIDER than the frame;
    each frame's window slides along it, so the star COUNT genuinely varies. The sky
    gradient stays fixed in FRAME pixel coordinates (module comment above ``FramePhotometry``)
    -- a static rig's own vignette/glow, never the sky's own coordinates."""
    height, width = 40, 900
    rng = np.random.default_rng(9)
    gradient = 0.02 + np.linspace(0.0, 0.02, width)
    canvas_width = width + 900
    star_canvas = np.zeros((height, canvas_width))
    n_stars = int(0.01 * height * canvas_width)  # ~1%: the density that broke the
    # (now-cut) percentile estimator; a masked median stays correct at any density.
    star_rows = rng.integers(0, height, n_stars)
    star_cols = rng.integers(0, canvas_width, n_stars)
    star_canvas[star_rows, star_cols] = 0.09
    drifts, paths = [140, 300], _paths(3, tmp_path)  # a plausible multi-hour drift for a static rig
    frames = []
    for i, x0 in enumerate([0, *drifts]):
        stars_window = star_canvas[:, x0 : x0 + width]
        img = np.maximum(gradient[None, :] + rng.normal(0.0, 0.001, (height, width)), stars_window)
        rgb = np.repeat(img[:, :, None], 3, axis=2).astype(np.float32)
        frames.append(DecodedAstroFrame(rgb, paths[i], "rawpy-linear", 16))
    metrics = tuple(FrameMetrics(paths[i], 6, 0.0, 2.0, [], True) for i in range(3))
    prepared = PreparedFrames(tuple(frames), metrics, TriageReport(metrics, (), ()), 0, ())
    queue = [RegistrationResult(np.array([[1.0, 0, -d], [0, 1.0, 0]]), 10, 0.9) for d in drifts]
    _apply_stack_run_stubs(monkeypatch)
    monkeypatch.setattr(om, "register", lambda *a, **k: queue.pop(0))
    monkeypatch.setattr(om, "stack", _stub_stack_capturing({}))
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)
    report = stack_run(prepared, tmp_path / "stack.tiff")
    assert report.rejected_frames == ()
    for e in report.photometry[1:]:
        assert e.scale == 1.0  # pinned, never estimated
        assert abs(e.offset) < 1.5e-4  # measured ~2e-5; ~6x headroom for RNG jitter


def _develop_result_stub(shape, fill=100):
    return AstroDevelopResult(
        rgb_uint16=np.full(shape, fill, dtype=np.uint16),
        black_point=0.0,
        m_values=[0.5],
        background_before=0.0,
        background_after=0.0,
        background_gains=(1.0, 1.0, 1.0),
        peak=1.0,
    )


def _bright_layer_stub(fill):
    """``develop_stack_with_params`` stand-in: ignores the layer's actual pixels and
    the base's params, returning a uniform BRIGHT frame -- isolates the meteor
    wiring (layer selection, streak warp, the ``reinsert`` call) from
    astro_develop's own tone math, which test_astro_develop.py covers directly."""

    def _stub(layer, base):
        return np.full(base.rgb_uint16.shape, fill, dtype=base.rgb_uint16.dtype)

    return _stub


class TestMeteorReinsertion:
    """``reinsert``/``streak_mask`` run for real here -- only the develop replay and
    the frames/registration around them are stubbed."""

    SHAPE = (60, 60, 3)

    def _prepared(self, tmp_path, n=3):
        paths = _paths(n, tmp_path)
        frames = tuple(
            DecodedAstroFrame(
                rgb=np.full(self.SHAPE, float(i), dtype=np.float32),
                path=paths[i],
                decoder="rawpy-linear",
                bit_depth=16,
            )
            for i in range(n)
        )
        metrics = tuple(FrameMetrics(paths[i], 6, 0.0, 2.0, [], True) for i in range(n))
        report = TriageReport(frames=metrics, cloud_frames=(), dew_frames=())
        return PreparedFrames(frames, metrics, report, 0, ())

    def _apply_common_stubs(self, monkeypatch, register=None):
        monkeypatch.setattr(om, "detect_stars", _stub_detect_stars)
        monkeypatch.setattr(
            om, "horizon_mask", lambda shape_hw, **kw: np.ones(shape_hw, dtype=np.float32)
        )
        monkeypatch.setattr(om, "warp_to_reference", lambda rgb, transform, out_shape: rgb)
        monkeypatch.setattr(om, "_frame_photometry", lambda *a, **k: (0.0, 1.0))
        monkeypatch.setattr(om, "register", register or (lambda *a, **k: _IDENTITY_REGISTRATION))
        monkeypatch.setattr(
            om,
            "stack",
            lambda aligned, unaligned, mask, **kwargs: StackResult(
                rgb=np.zeros(self.SHAPE, dtype=np.float32),
                n_frames=len(aligned),
                mean_rejected_fraction=0.0,
            ),
        )
        monkeypatch.setattr(
            om, "develop_stack", lambda rgb, sky_mask=None: _develop_result_stub(self.SHAPE)
        )
        monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
        monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)
        exported: dict = {}

        def _capture_export(rgb, out_path):
            exported["rgb"] = rgb
            return Path(out_path)

        monkeypatch.setattr(dm, "export_tiff16", _capture_export)
        return exported

    def test_only_meteor_streaks_are_reinserted_at_their_warped_position(
        self, monkeypatch, tmp_path
    ):
        prepared = self._prepared(tmp_path)
        transform = np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 0.0]])  # frame1 -> reference: x += 20
        exported = self._apply_common_stubs(
            monkeypatch, register=lambda *a, **k: RegistrationResult(transform, 10, 0.9)
        )
        monkeypatch.setattr(om, "develop_stack_with_params", _bright_layer_stub(60000))
        meteor = StreakClassification(
            1, prepared.frames[1].path, (10.0, 10.0, 10.0, 30.0), "meteor"
        )
        satellite = StreakClassification(
            2, prepared.frames[2].path, (10.0, 40.0, 10.0, 55.0), "satellite"
        )
        monkeypatch.setattr(om, "classify_streaks", lambda metrics: (meteor, satellite))
        policy = MeteorPolicy(mode="reviewed", allowlist=frozenset({(1, 0)}))  # meteor only

        report = stack_run(prepared, tmp_path / "stack.tiff", meteors=policy)

        rgb = exported["rgb"]
        assert rgb[20, 30, 0] > 50000  # meteor landed at the WARPED x=30, not frame-space x=10
        assert rgb[20, 10, 0] < 200  # frame-space x=10 (pre-warp) stayed background
        assert rgb[47, 10, 0] < 200  # satellite's own line never appears (its own coordinates)
        assert rgb[47, 30, 0] < 200  # ... nor warped -- a satellite is never reinserted at all
        assert report.meteors_reinserted == 1

    def test_a_meteor_in_an_unregistered_frame_is_rejected_not_reinserted(
        self, monkeypatch, tmp_path
    ):
        prepared = self._prepared(tmp_path, n=4)  # frame1 fails; 3 of the remaining still stack
        exported = self._apply_common_stubs(monkeypatch, register=_make_fake_register(bad_fill=1.0))
        monkeypatch.setattr(om, "develop_stack_with_params", _bright_layer_stub(60000))
        meteor = StreakClassification(1, prepared.frames[1].path, (5.0, 5.0, 5.0, 15.0), "meteor")
        monkeypatch.setattr(om, "classify_streaks", lambda metrics: (meteor,))
        policy = MeteorPolicy(mode="reviewed", allowlist=frozenset({(1, 0)}))

        report = stack_run(prepared, tmp_path / "stack.tiff", meteors=policy)

        assert report.meteors_reinserted == 0
        assert any(p == prepared.frames[1].path for p, _reason in report.rejected_frames)
        np.testing.assert_array_equal(exported["rgb"], _develop_result_stub(self.SHAPE).rgb_uint16)

    def test_zero_meteors_leaves_the_export_byte_identical_to_the_develop_output(
        self, monkeypatch, tmp_path
    ):
        prepared = self._prepared(tmp_path)
        exported = self._apply_common_stubs(monkeypatch)
        monkeypatch.setattr(om, "classify_streaks", lambda metrics: ())

        report = stack_run(prepared, tmp_path / "stack.tiff")

        assert report.meteors_reinserted == 0
        np.testing.assert_array_equal(exported["rgb"], _develop_result_stub(self.SHAPE).rgb_uint16)

    def test_default_posture_is_off_even_with_real_meteor_candidates(self, monkeypatch, tmp_path):
        """No ``meteors=`` argument reinserts nothing, still exports."""
        prepared = self._prepared(tmp_path)
        exported = self._apply_common_stubs(monkeypatch)
        monkeypatch.setattr(om, "develop_stack_with_params", _bright_layer_stub(60000))
        meteor = StreakClassification(
            1, prepared.frames[1].path, (10.0, 10.0, 10.0, 30.0), "meteor"
        )
        monkeypatch.setattr(om, "classify_streaks", lambda metrics: (meteor,))
        captured_payload: dict = {}
        monkeypatch.setattr(
            om, "stack_sidecar_payload", lambda **kwargs: captured_payload.update(kwargs) or {}
        )

        report = stack_run(prepared, tmp_path / "stack.tiff")

        assert report.meteors_reinserted == 0
        np.testing.assert_array_equal(exported["rgb"], _develop_result_stub(self.SHAPE).rgb_uint16)
        assert captured_payload["meteor_mode"] == "off"
        assert captured_payload["meteor_candidate_count"] == 1
        assert captured_payload["meteor_skip_reason"]

    def test_develop_replay_runs_once_per_unique_frame_not_per_streak(self, monkeypatch, tmp_path):
        prepared = self._prepared(tmp_path)  # 3 streaks/frame must not replay 3x
        exported = self._apply_common_stubs(monkeypatch)
        calls = []

        def _counting_develop(layer, base):
            calls.append(1)
            return np.full(base.rgb_uint16.shape, 60000, dtype=base.rgb_uint16.dtype)

        monkeypatch.setattr(om, "develop_stack_with_params", _counting_develop)
        streaks = tuple(
            StreakClassification(
                1, prepared.frames[1].path, (5.0 + k, 5.0, 5.0 + k, 15.0), "meteor"
            )
            for k in range(3)
        )
        monkeypatch.setattr(om, "classify_streaks", lambda metrics: streaks)
        policy = MeteorPolicy(mode="reviewed", allowlist=frozenset({(1, 0), (1, 1), (1, 2)}))

        report = stack_run(prepared, tmp_path / "stack.tiff", meteors=policy)

        assert len(calls) == 1
        assert report.meteors_reinserted == 3
        assert exported["rgb"][10, 6, 0] > 50000


class TestResolveMeteorSelection:
    """``_resolve_meteor_selection`` is pure -- no stack_run harness needed."""

    _METEOR = StreakClassification(0, Path("f.nef"), (0.0, 0.0, 1.0, 1.0), "meteor")

    def test_off_mode_selects_nothing_but_still_counts_and_records_a_reason(self):
        sel, n, reason = om._resolve_meteor_selection((self._METEOR,), MeteorPolicy(), {0: None})
        assert sel == []
        assert n == 1
        assert reason

    def test_reviewed_picks_only_the_allowlisted_identity(self):
        other = StreakClassification(0, Path("f.nef"), (2.0, 0.0, 3.0, 1.0), "meteor")
        policy = MeteorPolicy(mode="reviewed", allowlist=frozenset({(0, 0)}))
        sel, n, reason = om._resolve_meteor_selection((self._METEOR, other), policy, {0: None})
        assert sel == [self._METEOR]
        assert n == 2
        assert reason is None

    def test_excess_candidates_skips_everything_and_records_the_reason(self):
        candidates = tuple(self._METEOR for _ in range(om.MAX_METEOR_CANDIDATES + 1))
        policy = MeteorPolicy(mode="reviewed", allowlist=frozenset())
        selected, count, reason = om._resolve_meteor_selection(candidates, policy, {0: None})
        assert selected == []
        assert count == om.MAX_METEOR_CANDIDATES + 1
        assert str(om.MAX_METEOR_CANDIDATES) in reason

    def test_auto_mode_is_refused_with_a_typed_error(self):
        with pytest.raises(MeteorPolicyError):
            om._resolve_meteor_selection((), MeteorPolicy(mode="auto"), {})


class TestDarkIdentityWiring:
    """The dark-identity refusal reaches the production call site: ``prepare_frames``
    must judge every ``--darks`` frame against the lights before any master is built.
    Numbers below are one typical capture setting (15 s exposures at ISO 6400)."""

    def _identities(self, paths, **overrides):
        """One matching ``CaptureIdentity`` per path at the shared capture setting."""
        return {p: _identity(path=p, iso=6400.0, exposure_s=15.0, **overrides) for p in paths}

    def _wire(self, monkeypatch, identities, lights, mosaic_reads):
        monkeypatch.setattr(om, "decode_linear", _stub_decode)
        monkeypatch.setattr(om, "measure_frame", _measure_from({p: (6, 2.0) for p in lights}))
        # raising=False so the RED run fails on the missing refusal, not on the seam
        monkeypatch.setattr(
            om, "read_capture_identity", lambda p: identities[Path(p)], raising=False
        )
        monkeypatch.setattr(om, "hot_pixel_map", lambda md: np.zeros((4, 4), bool))

        def _read_mosaic(path):
            mosaic_reads.append(Path(path))
            return np.zeros((4, 4), dtype=np.uint16)

        monkeypatch.setattr(om, "read_raw_mosaic", _read_mosaic)

    def test_a_ten_second_dark_among_fifteen_second_darks_refuses_naming_file_and_field(
        self, monkeypatch, tmp_path
    ):
        """A 10 s frame sitting among 15 s darks. The refusal names the file AND the field, so
        one frame can be dropped from ``--darks`` without guessing which."""
        lights = _paths(5, tmp_path)
        darks = [tmp_path / f"DSC_000{n}.NEF" for n in (1, 2, 3)]
        identities = self._identities([*lights, *darks])
        identities[darks[-1]] = _identity(path=darks[-1], iso=6400.0, exposure_s=10.0)
        self._wire(monkeypatch, identities, lights, [])

        def _boom(*a, **k):
            raise AssertionError("refuse before decoding any light (~1 min/frame)")

        monkeypatch.setattr(om, "decode_linear", _boom)

        with pytest.raises(
            AstroDarkIdentityMismatchError,
            match=r"DSC_0003\.NEF: dark exposure_s does not match light .*15\.0.*10\.0",
        ):
            prepare_frames(lights, dark_paths=darks)

    def test_the_first_light_is_the_identity_every_dark_is_judged_against(
        self, monkeypatch, tmp_path
    ):
        lights = _paths(5, tmp_path)
        darks = [tmp_path / f"dark{i}.NEF" for i in range(2)]
        self._wire(monkeypatch, self._identities([*lights, *darks]), lights, [])
        seen = {}

        def _spy(mosaics, *, light_identity=None, dark_identities=None):
            seen["light"] = light_identity
            seen["darks"] = dark_identities
            return np.zeros((4, 4), dtype=np.float32)

        monkeypatch.setattr(om, "build_master_dark", _spy)

        prepare_frames(lights, dark_paths=darks)

        assert seen["light"].path == lights[0]
        assert [d.path for d in seen["darks"]] == darks

    def test_a_fully_matching_dark_set_calibrates_every_light_unchanged(
        self, monkeypatch, tmp_path
    ):
        lights = _paths(5, tmp_path)
        darks = [tmp_path / f"dark{i}.NEF" for i in range(3)]
        reads = []
        self._wire(monkeypatch, self._identities([*lights, *darks]), lights, reads)
        calls = []

        def _recording_decode(path, master_dark, hotmap):
            calls.append((master_dark, hotmap))
            return _stub_decode(path, master_dark, hotmap)

        monkeypatch.setattr(om, "decode_linear", _recording_decode)

        result = prepare_frames(lights, dark_paths=darks)

        assert len(result.frames) == 5
        assert reads == darks
        assert all(md is not None and hm is not None for md, hm in calls)

    def test_the_prepared_frames_carry_the_darks_the_run_calibrated_with(
        self, monkeypatch, tmp_path
    ):
        """``stack_run`` writes the provenance sidecar and sees only a ``PreparedFrames``,
        so the darks have to travel on it -- without that the sidecar's ``inputs.dark``
        was ``[]`` on every run that subtracted a master dark."""
        lights = _paths(5, tmp_path)
        darks = [tmp_path / f"dark{i}.NEF" for i in range(3)]
        self._wire(monkeypatch, self._identities([*lights, *darks]), lights, [])

        assert list(prepare_frames(lights, dark_paths=darks).dark_paths) == darks
        # A no-darks run makes the honest empty claim, never a guess.
        assert prepare_frames(lights).dark_paths == ()

    @pytest.mark.parametrize("side", ["light", "dark"])
    def test_an_unreadable_field_refuses_instead_of_passing_as_a_match(
        self, monkeypatch, tmp_path, side
    ):
        """``validate_dark_identity`` SKIPS a field that is ``None`` on either side, so
        an unreadable exposure would sail through as if it matched. Refused here, before
        a single dark mosaic is read, and never reported as a mismatch."""
        lights = _paths(5, tmp_path)
        darks = [tmp_path / "DSC_0001.NEF"]
        identities = self._identities([*lights, *darks])
        odd = lights[0] if side == "light" else darks[0]
        identities[odd] = _identity(path=odd, iso=6400.0, exposure_s=None)
        reads = []
        self._wire(monkeypatch, identities, lights, reads)

        with pytest.raises(
            AstroCalibrationError, match=rf"{odd.name}: this {side}'s exposure_s could not be read"
        ) as excinfo:
            prepare_frames(lights, dark_paths=darks)

        assert not isinstance(excinfo.value, AstroDarkIdentityMismatchError)
        assert reads == []

    def test_a_run_without_darks_reads_no_capture_identity_at_all(self, monkeypatch, tmp_path):
        lights = _paths(5, tmp_path)

        def _boom(*a, **k):
            raise AssertionError("the no-darks path must read no capture identity")

        monkeypatch.setattr(om, "decode_linear", _stub_decode)
        monkeypatch.setattr(om, "measure_frame", _measure_from({p: (6, 2.0) for p in lights}))
        monkeypatch.setattr(om, "read_capture_identity", _boom, raising=False)

        assert len(prepare_frames(lights).frames) == 5


# --- the stacking path warps through the lens+rotation remap, not the 2x3 ---------

_LENS_FRAME_SHAPE = (64, 64, 3)
_LENS_STAR_STEP = 9  # 7x7 = 49 detections, clear of the automatic-horizon evidence floor
_LENS_BARREL_CORNER_PX = 6.0


def _lens_grid_stars(shape_hw, step=_LENS_STAR_STEP):
    """A regular star grid, far enough apart that mutual-best matching can't
    confuse two of them under a few px of distortion."""
    height, width = shape_hw[:2]
    xs = np.arange(step // 2, width, step, dtype=np.float64)
    ys = np.arange(step // 2, height, step, dtype=np.float64)
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    return np.concatenate([grid, np.full((len(grid), 1), 5.0)], axis=1).astype(np.float32)


def _lens_barrel_stars(stars, shape_hw, corner_px=_LENS_BARREL_CORNER_PX):
    """The same stars, displaced radially by ``corner_px * r_norm**2`` -- barrel
    distortion no 2x3 similarity can undo, biggest in the outer third."""
    height, width = shape_hw[:2]
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    offset = stars[:, :2].astype(np.float64) - center
    radius = np.linalg.norm(offset, axis=1, keepdims=True)
    r_norm = radius / np.linalg.norm(center)
    moved = stars.copy()
    moved[:, :2] = (
        center + offset * (1.0 + corner_px * r_norm**2 / np.maximum(radius, 1e-9))
    ).astype(np.float32)
    return moved


def _lens_case(monkeypatch, tmp_path, n=3):
    """``stack_run`` stubs plus a distorted star field: the reference frame carries
    ``_lens_grid_stars``, every other frame the barrel-displaced copy."""
    prepared = _stack_prepared(n, tmp_path, shape=_LENS_FRAME_SHAPE)
    _apply_stack_run_stubs(monkeypatch)
    ref_stars = _lens_grid_stars(_LENS_FRAME_SHAPE)
    frame_stars = _lens_barrel_stars(ref_stars, _LENS_FRAME_SHAPE)
    monkeypatch.setattr(
        om,
        "detect_stars",
        lambda luma, sigma=None, mask=None: ref_stars if float(luma[0, 0]) == 0.0 else frame_stars,
    )
    warps = []
    monkeypatch.setattr(
        om,
        "warp_to_reference",
        lambda rgb, transform, out_shape: warps.append(transform) or rgb,
    )
    height, width = _LENS_FRAME_SHAPE[:2]
    xs, ys = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    monkeypatch.setattr(om, "build_remap", lambda shape, params, rotation: (xs, ys))
    monkeypatch.setattr(om, "stack", lambda *a, **k: _lens_stack_result())
    monkeypatch.setattr(om, "stack_sidecar_payload", lambda **kwargs: {})
    monkeypatch.setattr(om, "write_stack_sidecar", lambda payload, image_path: None)
    return prepared, warps


def _lens_stack_result():
    return StackResult(
        rgb=np.zeros(_LENS_FRAME_SHAPE, dtype=np.float32),
        n_frames=3,
        mean_rejected_fraction=0.0,
        sky_sum=np.zeros(_LENS_FRAME_SHAPE, dtype=np.float32),
        sky_count=np.full(_LENS_FRAME_SHAPE[:2], 3, dtype=np.uint16),
        ground=np.zeros(_LENS_FRAME_SHAPE, dtype=np.float32),
    )


def _stub_solve_night(monkeypatch, *, outer_p95_px, method="rotation", raises=None):
    from gomoveshift.pipeline.photos.astro_geometry import FrameSolution, SolveResult
    from gomoveshift.pipeline.photos.astro_lens import LensParams

    fitted = LensParams(k1=-0.05, k2=0.0, cx=31.5, cy=31.5, f_px=50.0)

    def _solve(matches, frame_shape, initial_lens, **kwargs):
        if raises is not None:
            raise raises
        return SolveResult(
            lens=fitted,
            frames=[
                FrameSolution(
                    frame_id=m.frame_id,
                    method=method,
                    rotation=np.zeros(3) if method == "rotation" else None,
                    homography=None if method == "rotation" else np.eye(3),
                    train_residual_p95_px=0.1,
                    held_out_residuals_px={
                        "center": {"n": 6, "median_px": 0.1, "p95_px": 0.1},
                        "middle": {"n": 6, "median_px": 0.1, "p95_px": 0.1},
                        "outer": {"n": 8, "median_px": outer_p95_px / 2, "p95_px": outer_p95_px},
                    },
                )
                for m in matches
            ],
            converged=True,
            cost=1.0,
        )

    monkeypatch.setattr(om, "solve_night", _solve)
    return fitted


def _is_remap(transform):
    return isinstance(transform, tuple) and len(transform) == 2


def test_stacking_warps_through_the_lens_remap_when_the_fit_beats_the_affine(monkeypatch, tmp_path):
    prepared, warps = _lens_case(monkeypatch, tmp_path)
    fitted = _stub_solve_night(monkeypatch, outer_p95_px=0.05)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    # Every non-reference frame goes through cv2.remap's (map_x, map_y) pair, never
    # the 2x3 an affine can't bend into a barrel-distorted corner.
    assert len(warps) == 2
    assert all(_is_remap(t) for t in warps)
    assert report.geometry["frames_lens_warped"] == 2
    assert report.geometry["frames_affine_warped"] == 0
    assert report.geometry["lens"]["k1"] == fitted.k1
    assert report.geometry["lens"]["f_px"] == fitted.f_px


def test_a_geometry_fit_that_does_not_converge_keeps_the_2x3_warp(monkeypatch, tmp_path):
    from gomoveshift.pipeline.photos.astro_geometry import GeometrySolveError

    prepared, warps = _lens_case(monkeypatch, tmp_path)
    _stub_solve_night(
        monkeypatch, outer_p95_px=0.05, raises=GeometrySolveError("stub: no convergence")
    )

    report = stack_run(prepared, tmp_path / "stack.tiff")

    assert len(warps) == 2
    assert not any(_is_remap(t) for t in warps)
    assert report.geometry["frames_lens_warped"] == 0
    assert report.geometry["frames_affine_warped"] == 2
    assert "stub: no convergence" in report.geometry["reason"]


def test_a_lens_fit_worse_than_the_2x3_on_held_out_stars_is_refused(monkeypatch, tmp_path):
    prepared, warps = _lens_case(monkeypatch, tmp_path)
    _stub_solve_night(monkeypatch, outer_p95_px=50.0)

    report = stack_run(prepared, tmp_path / "stack.tiff")

    assert not any(_is_remap(t) for t in warps)
    assert report.geometry["frames_lens_warped"] == 0


def test_a_frame_the_rotation_model_cannot_fit_keeps_the_2x3_warp(monkeypatch, tmp_path):
    prepared, warps = _lens_case(monkeypatch, tmp_path)
    _stub_solve_night(monkeypatch, outer_p95_px=0.05, method="homography")

    report = stack_run(prepared, tmp_path / "stack.tiff")

    assert not any(_is_remap(t) for t in warps)
    assert report.geometry["frames_lens_warped"] == 0


def test_linear_artifact_records_which_warp_each_frame_took(monkeypatch, tmp_path):
    prepared, _warps = _lens_case(monkeypatch, tmp_path)
    _stub_solve_night(monkeypatch, outer_p95_px=0.05)
    monkeypatch.setattr(om, "read_capture_identity", lambda path: _identity(path=Path(path)))
    captured = {}
    monkeypatch.setattr(
        om,
        "write_linear_artifact",
        lambda destination, result, **kwargs: captured.update(kwargs),
    )

    stack_run(
        prepared,
        None,
        linear_artifact_path=tmp_path / "chunk.linear",
        linear_only=True,
    )

    provenance = captured["registration_provenance"]
    assert provenance[0]["warp"]["model"] == "reference"
    assert provenance[1]["warp"]["model"] == "lens_rotation"
    assert provenance[1]["warp"]["held_out_outer_p95_px"]["lens"] == 0.05
    assert provenance[1]["warp"]["held_out_outer_p95_px"]["similarity"] > 0.05
