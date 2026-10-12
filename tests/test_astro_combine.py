"""Validation and reference-selection contract for deep linear combines."""

import json

import numpy as np
import pytest

import gomoveshift.pipeline.photos.astro_combine as combine_mod
from gomoveshift.pipeline.photos.astro_combine import (
    ArtifactTransform,
    AstroCombineError,
    combine_ground_two_pass,
    combine_sky_two_pass,
    compose_affine,
    deep_combine,
    estimate_chunk_photometry,
    fit_artifact_pair,
    plan_artifact_transforms,
    plan_direct_transforms,
    validate_artifact_set,
)
from gomoveshift.pipeline.photos.astro_geometry import GeometrySolveError
from gomoveshift.pipeline.photos.astro_lens import LensParams, build_remap, distort, undistort
from gomoveshift.pipeline.photos.astro_linear import load_linear_artifact, write_linear_artifact
from gomoveshift.pipeline.photos.astro_register import (
    MIN_OUTER_THIRD_INLIERS,
    RANSAC_REPROJ_THRESHOLD_PX,
    RegistrationError,
    RegistrationResult,
    fit_similarity,
    radial_inlier_residuals_px,
    warp_to_reference,
)
from gomoveshift.pipeline.photos.astro_stack import StackResult, blend, horizon_mask

from fixtures.astro_synthetic import TRUE_LENS, build_synthetic_night


def _write_artifact(root, index, *, mode="automatic"):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"chunk-{index}.linear"
    shape = (3, 4, 3)
    write_linear_artifact(
        path,
        StackResult(
            rgb=np.zeros(shape, dtype=np.float32),
            n_frames=10,
            mean_rejected_fraction=0.0,
            sky_sum=np.full(shape, index + 1, dtype=np.float32),
            sky_count=np.full(shape[:2], 10, dtype=np.uint16),
            ground=np.full(shape, 0.1, dtype=np.float32),
        ),
        reference_raw=f"RAW-{index}-mid.NEF",
        capture_identity={
            "camera": "NIKON Z 6",
            "raw_shape": [3, 4],
            "cfa_pattern": [0, 1, 3, 2],
            "iso": 6400.0,
            "exposure_s": 15.0,
            "temperature_c": index,
        },
        mask_provenance={
            "mode": mode,
            "requested_horizon_frac": 0.6 if mode == "explicit" else None,
            "resolved_boundary_row": 2 + index * 0.01,
        },
        source_frames=[f"RAW-{index}-{frame}.NEF" for frame in range(10)],
        registration_provenance=[{"frame": f"RAW-{index}-0.NEF", "transform": "identity"}],
        photometry_provenance=[{"frame": f"RAW-{index}-0.NEF", "offset": 0.0, "scale": 1.0}],
        algorithm_version="linear-hash-a",
    )
    return path


def _mutate_manifest(path, mutate):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest))


def test_validate_artifact_set_selects_temporal_midpoint_and_preserves_order(tmp_path):
    paths = [_write_artifact(tmp_path, index) for index in range(5)]

    result = validate_artifact_set(paths)

    assert [artifact.path for artifact in result.artifacts] == paths
    assert result.reference_index == 2
    assert result.reference.path == paths[2]


def test_validate_artifact_set_excludes_low_quality_chunks_from_the_midpoint_search(tmp_path):
    """The reference is the temporal midpoint AMONG the top quality quartile only.

    Artifacts 3 and 4 sit closest to the midpoint of an 8-chunk night but are the two
    lowest-quality (``sky_weight``) chunks, so they must never anchor the whole stack.
    Artifacts 1 and 6 are the two highest-quality chunks (the top quartile of 8); between
    those, artifact 1 wins the existing quality/frame-count/index tie-break.
    """
    paths = [_write_artifact(tmp_path, index) for index in range(8)]
    weights = {0: 0.5, 1: 0.95, 2: 0.5, 3: 0.1, 4: 0.1, 5: 0.5, 6: 0.95, 7: 0.5}
    for index, weight in weights.items():
        _mutate_manifest(paths[index], lambda manifest, w=weight: manifest.update(sky_weight=w))

    result = validate_artifact_set(paths)

    assert result.reference_index == 1


def test_validate_artifact_set_accepts_an_explicit_reference(tmp_path):
    paths = [_write_artifact(tmp_path, index) for index in range(4)]

    result = validate_artifact_set(paths, reference=paths[0])

    assert result.reference_index == 0


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda manifest: manifest.update(algorithm_version="different"), "algorithm"),
        (lambda manifest: manifest["capture_identity"].update(camera="OTHER"), "capture"),
        (lambda manifest: manifest["payloads"]["ground.npy"].update(shape=[99, 99, 3]), "payload"),
    ],
)
def test_validate_artifact_set_refuses_incompatible_or_corrupt_members(tmp_path, mutation, message):
    paths = [_write_artifact(tmp_path, index) for index in range(3)]
    _mutate_manifest(paths[-1], mutation)

    with pytest.raises(AstroCombineError, match=message):
        validate_artifact_set(paths)


def test_validate_artifact_set_refuses_overlapping_source_frames(tmp_path):
    paths = [_write_artifact(tmp_path, index) for index in range(3)]
    first_source = json.loads((paths[0] / "manifest.json").read_text())["source_frames"][0]
    _mutate_manifest(paths[1], lambda manifest: manifest["source_frames"].append(first_source))

    with pytest.raises(AstroCombineError, match="overlap"):
        validate_artifact_set(paths)


def test_validate_artifact_set_allows_auto_boundaries_to_differ_but_not_mask_modes(tmp_path):
    automatic = [_write_artifact(tmp_path, index) for index in range(3)]
    validate_artifact_set(automatic)  # resolved rows differ by design

    mixed = [_write_artifact(tmp_path / "mixed", 0, mode="automatic")]
    mixed.extend(_write_artifact(tmp_path / "mixed", index, mode="explicit") for index in (1, 2))
    with pytest.raises(AstroCombineError, match="mask mode"):
        validate_artifact_set(mixed)


def test_validate_artifact_set_refuses_tiffs_and_too_few_chunks(tmp_path):
    artifact = _write_artifact(tmp_path, 0)
    tiff = tmp_path / "developed.tif"
    tiff.write_bytes(b"not linear")

    with pytest.raises(AstroCombineError, match="at least 2"):
        validate_artifact_set([artifact])
    with pytest.raises(AstroCombineError, match="linear artifact"):
        validate_artifact_set([artifact, tiff])


def _translation(dx):
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0]], dtype=np.float64)


def test_compose_affine_applies_the_first_transform_then_the_second():
    composed = compose_affine(_translation(2.0), _translation(3.0))

    np.testing.assert_allclose(composed, _translation(5.0))


def test_transform_plan_uses_direct_fit_then_records_adjacent_chain_fallback(tmp_path):
    paths = [_write_artifact(tmp_path, index) for index in range(5)]
    artifacts = validate_artifact_set(paths)
    index_by_path = {path: index for index, path in enumerate(paths)}
    calls = []

    def fit(moving, reference):
        pair = index_by_path[moving.path], index_by_path[reference.path]
        calls.append(pair)
        if pair == (0, 2):
            raise RegistrationError("coarse prior found too few matches")
        return _translation(pair[1] - pair[0])

    plans = plan_artifact_transforms(artifacts, fit)

    assert [plan.method for plan in plans] == [
        "adjacent_chain",
        "direct",
        "identity",
        "direct",
        "direct",
    ]
    assert plans[0].links == ((0, 1), (1, 2))
    np.testing.assert_allclose(plans[0].transform, _translation(2.0))
    assert calls.index((0, 2)) < calls.index((0, 1)), (
        "artifact 0 must try its direct fit before falling back to a chained link"
    )


def test_transform_plan_refuses_when_the_required_adjacent_chain_is_broken(tmp_path):
    paths = [_write_artifact(tmp_path, index) for index in range(3)]
    artifacts = validate_artifact_set(paths)
    index_by_path = {path: index for index, path in enumerate(paths)}

    def fit(moving, reference):
        pair = index_by_path[moving.path], index_by_path[reference.path]
        if pair in {(0, 1), (0, artifacts.reference_index)}:
            raise RegistrationError("no adjacent consensus")
        return _translation(pair[1] - pair[0])

    with pytest.raises(AstroCombineError, match="adjacent transform 0->1"):
        plan_artifact_transforms(artifacts, fit)


def test_transform_plan_retains_registration_diagnostics(tmp_path):
    paths = [_write_artifact(tmp_path, index) for index in range(2)]
    artifacts = validate_artifact_set(paths, reference=paths[1])

    def fit(moving, reference):
        return RegistrationResult(
            transform=_translation(1.0),
            n_matched=42,
            inlier_fraction=0.9,
            residual_median_px=0.4,
            residual_p95_px=1.2,
            match_yield=0.8,
        )

    plan = plan_artifact_transforms(artifacts, fit)[0]

    assert plan.diagnostics[0]["n_matched"] == 42
    assert plan.diagnostics[0]["residual_p95_px"] == pytest.approx(1.2)


def test_plan_direct_transforms_fits_every_artifact_straight_to_the_reference(tmp_path):
    paths = [_write_artifact(tmp_path, index) for index in range(5)]
    artifacts = validate_artifact_set(paths)
    index_by_path = {path: index for index, path in enumerate(paths)}
    calls = []

    def fit(moving, reference):
        pair = index_by_path[moving.path], index_by_path[reference.path]
        calls.append(pair)
        return _translation(pair[1] - pair[0])

    plans = plan_direct_transforms(artifacts, fit)

    assert [plan.method for plan in plans] == ["direct", "direct", "identity", "direct", "direct"]
    assert all(pair[1] == artifacts.reference_index for pair in calls), (
        "every fit must target the global reference directly, never a neighbour"
    )
    np.testing.assert_allclose(plans[0].transform, _translation(2.0))
    assert plans[0].links == ((0, 2),)
    assert plans[0].depth == 1


def test_plan_direct_transforms_refuses_a_failed_direct_fit_without_chaining(tmp_path):
    """Direct fits retire the adjacent-chain fallback for the deep path: a chunk that cannot register
    straight to the single global reference is a hard failure, never silently composed through
    a neighbour that DID register cleanly."""
    paths = [_write_artifact(tmp_path, index) for index in range(5)]
    artifacts = validate_artifact_set(paths)
    index_by_path = {path: index for index, path in enumerate(paths)}
    calls = []

    def fit(moving, reference):
        pair = index_by_path[moving.path], index_by_path[reference.path]
        calls.append(pair)
        if pair == (0, 2):
            raise RegistrationError("coarse prior found too few matches")
        return _translation(pair[1] - pair[0])

    with pytest.raises(AstroCombineError, match=r"direct transform 0->2"):
        plan_direct_transforms(artifacts, fit)

    assert (0, 1) not in calls, "a failed direct fit must never fall back to an adjacent chain"


def test_plan_direct_transforms_reports_every_unregisterable_chunk_not_just_the_first(tmp_path):
    """One refused pair must not throw away the answer for all the others.

    Aborting on the first failure means a night pays the full per-pair fit cost and learns
    about exactly one chunk, so the operator re-runs to discover the next one. Every pair is
    attempted and the refusal names them all, plus the widest run of chunks around the
    reference that DID register -- the subset of paths a re-run can actually ship.
    """
    paths = [_write_artifact(tmp_path, index) for index in range(5)]
    artifacts = validate_artifact_set(paths)
    index_by_path = {path: index for index, path in enumerate(paths)}
    calls = []

    def fit(moving, reference):
        pair = index_by_path[moving.path], index_by_path[reference.path]
        calls.append(pair)
        if pair[0] in (0, 4):
            raise RegistrationError("coarse prior found too few matches")
        return _translation(pair[1] - pair[0])

    with pytest.raises(AstroCombineError) as excinfo:
        plan_direct_transforms(artifacts, fit)

    message = str(excinfo.value)
    assert "0->2" in message and "4->2" in message, (
        f"both refused pairs must be named in one refusal, got: {message}"
    )
    assert sorted(calls) == [(0, 2), (1, 2), (3, 2), (4, 2)], (
        "every non-reference chunk is attempted before the refusal is raised"
    )
    assert "1-3" in message, f"the registrable span around the reference must be named: {message}"


def test_plan_direct_transforms_span_covers_only_chunks_contiguous_with_the_reference(tmp_path):
    """The reported span is the run CONTAINING the reference, not the longest run anywhere.

    A combine's geometry is planned against one global reference, so a stretch of chunks that
    register to it but sits behind a gap is not a shippable span -- the gap is the part that
    cannot be warped into the same frame.
    """
    paths = [_write_artifact(tmp_path, index) for index in range(7)]
    artifacts = validate_artifact_set(paths)
    index_by_path = {path: index for index, path in enumerate(paths)}

    def fit(moving, reference):
        if index_by_path[moving.path] == 1:
            raise RegistrationError("coarse prior found too few matches")
        return _translation(1.0)

    assert 2 <= artifacts.reference_index <= 6, "the gap at chunk 1 must sit outside the span"

    with pytest.raises(AstroCombineError) as excinfo:
        plan_direct_transforms(artifacts, fit)

    assert "2-6" in str(excinfo.value), (
        f"span must start after the gap at chunk 1, got: {excinfo.value}"
    )


def test_artifact_pair_fit_uses_sky_means_and_persisted_boundary(tmp_path, monkeypatch):
    reference = np.full((12, 12, 3), 0.2, dtype=np.float32)
    moving = np.full((12, 12, 3), 0.3, dtype=np.float32)
    paths = [
        _write_sky_artifact(tmp_path, 0, reference),
        _write_sky_artifact(tmp_path, 1, moving),
    ]
    for path in paths:
        _mutate_manifest(
            path, lambda manifest: manifest["mask_provenance"].update(resolved_boundary_row=5)
        )
    artifacts = validate_artifact_set(paths, reference=paths[0])
    seen = {}
    stars = np.array([[i, i, 1.0] for i in range(6)], dtype=np.float32)
    monkeypatch.setattr(combine_mod, "detect_stars", lambda luma, **kwargs: stars)

    def fake_fit(ref_luma, ref_stars, moving_luma, moving_stars, *, sky_mask):
        seen.update(ref_luma=ref_luma, moving_luma=moving_luma, mask=sky_mask)
        return RegistrationResult(np.eye(2, 3), 6, 1.0, 0.1, 0.2, 1.0)

    monkeypatch.setattr(combine_mod, "fit_lens_aware", fake_fit)

    result = fit_artifact_pair(artifacts.artifacts[1], artifacts.artifacts[0])

    assert result.n_matched == 6
    assert np.allclose(seen["ref_luma"], 0.2)
    assert np.allclose(seen["moving_luma"], 0.3)
    assert np.all(seen["mask"][:5]) and not np.any(seen["mask"][5:])


def test_chunk_photometry_recovers_known_linear_offset_and_scale():
    reference = np.full((20, 20, 3), 0.2, dtype=np.float32)
    reference[2:8, 2:8] += 0.4
    moving = reference / 1.25 + 0.07

    offset, scale = estimate_chunk_photometry(
        moving, reference, np.ones(reference.shape[:2], dtype=bool)
    )

    assert offset == pytest.approx(0.07, abs=1e-5)
    assert scale == pytest.approx(1.25, abs=1e-5)


def _write_sky_artifact(root, index, sky, *, ground=None):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"chunk-{index}.linear"
    count = np.full(sky.shape[:2], 10, dtype=np.uint16)
    write_linear_artifact(
        path,
        StackResult(
            rgb=sky.astype(np.float32),
            n_frames=10,
            mean_rejected_fraction=0.0,
            sky_sum=sky.astype(np.float32) * count[..., None],
            sky_count=count,
            ground=(
                np.full(sky.shape, 0.1, dtype=np.float32)
                if ground is None
                else np.asarray(ground, dtype=np.float32)
            ),
        ),
        reference_raw=f"RAW-{index}-mid.NEF",
        capture_identity={
            "camera": "NIKON Z 6",
            "raw_shape": list(sky.shape[:2]),
            "cfa_pattern": [0, 1, 3, 2],
            "iso": 6400.0,
            "exposure_s": 15.0,
        },
        mask_provenance={"mode": "automatic", "resolved_boundary_row": sky.shape[0]},
        source_frames=[f"RAW-{index}-{frame}.NEF" for frame in range(10)],
        registration_provenance=[{"frame": f"RAW-{index}-0.NEF", "transform": "identity"}],
        photometry_provenance=[{"frame": f"RAW-{index}-0.NEF", "offset": 0.0, "scale": 1.0}],
        algorithm_version="linear-hash-a",
    )
    return path


def test_two_pass_sky_rejects_a_bright_chunk_pixel_and_keeps_frame_weight(tmp_path):
    base = np.full((12, 12, 3), 0.2, dtype=np.float32)
    paths = []
    for index in range(6):
        sky = base.copy()
        if index == 5:
            sky[4, 5] = 4.0
        paths.append(_write_sky_artifact(tmp_path, index, sky))
    artifacts = validate_artifact_set(paths)
    transforms = tuple(ArtifactTransform(np.eye(2, 3), "identity", ()) for _ in artifacts.artifacts)

    result = combine_sky_two_pass(artifacts, transforms)

    sky_mean = result.sky_sum / result.sky_count[..., None]
    assert sky_mean[4, 5, 0] == pytest.approx(0.2, abs=1e-5)
    assert result.sky_count[4, 5] == pytest.approx(50.0)
    assert result.sky_count[0, 0] == pytest.approx(60.0)
    assert result.rejected_pixel_fraction > 0.0


def test_two_pass_sky_photometry_maps_each_chunk_to_the_reference(tmp_path):
    reference = np.full((20, 20, 3), 0.2, dtype=np.float32)
    reference[2:8, 2:8] += 0.4
    paths = [
        _write_sky_artifact(tmp_path, 0, reference / 1.25 + 0.07),
        _write_sky_artifact(tmp_path, 1, reference),
    ]
    artifacts = validate_artifact_set(paths, reference=paths[1])
    transforms = tuple(ArtifactTransform(np.eye(2, 3), "identity", ()) for _ in artifacts.artifacts)

    result = combine_sky_two_pass(artifacts, transforms)

    np.testing.assert_allclose(result.sky_sum / result.sky_count[..., None], reference, atol=1e-5)
    assert result.photometry[0].offset == pytest.approx(0.07, abs=1e-5)
    assert result.photometry[0].scale == pytest.approx(1.25, abs=1e-5)


def test_two_pass_sky_never_keeps_more_than_one_chunk_payload_live(tmp_path):
    paths = [
        _write_sky_artifact(tmp_path, index, np.full((8, 8, 3), 0.2, dtype=np.float32))
        for index in range(8)
    ]
    artifacts = validate_artifact_set(paths)
    transforms = tuple(ArtifactTransform(np.eye(2, 3), "identity", ()) for _ in artifacts.artifacts)

    class Tracked:
        active = 0
        maximum = 0

        def __init__(self, path):
            loaded = load_linear_artifact(path)
            self.sky_sum, self.sky_count = loaded.sky_sum, loaded.sky_count
            type(self).active += 1
            type(self).maximum = max(type(self).maximum, type(self).active)

        def __del__(self):
            type(self).active -= 1

    combine_sky_two_pass(artifacts, transforms, loader=Tracked)

    assert Tracked.maximum == 1
    assert Tracked.active == 0


def test_ground_combine_stays_fixed_coordinate_and_rejects_one_surviving_light(tmp_path):
    sky = np.full((12, 12, 3), 0.2, dtype=np.float32)
    paths = []
    for index in range(6):
        ground = np.full(sky.shape, 0.1, dtype=np.float32)
        if index == 5:
            ground[8, 3] = 4.0
        paths.append(_write_sky_artifact(tmp_path, index, sky, ground=ground))
    artifacts = validate_artifact_set(paths)

    result = combine_ground_two_pass(artifacts)

    assert result.ground[8, 3, 0] == pytest.approx(0.1, abs=1e-5)
    assert result.weight[8, 3] == pytest.approx(50.0)
    assert result.weight[0, 0] == pytest.approx(60.0)
    assert result.rejected_pixel_fraction > 0.0


def test_ground_combine_maps_chunk_illumination_to_global_reference(tmp_path):
    sky = np.full((20, 20, 3), 0.2, dtype=np.float32)
    reference = np.full(sky.shape, 0.15, dtype=np.float32)
    reference[2:8, 2:8] += 0.3
    paths = [
        _write_sky_artifact(tmp_path, 0, sky, ground=reference / 1.2 + 0.04),
        _write_sky_artifact(tmp_path, 1, sky, ground=reference),
    ]
    artifacts = validate_artifact_set(paths, reference=paths[1])

    result = combine_ground_two_pass(artifacts)

    np.testing.assert_allclose(result.ground, reference, atol=1e-5)
    assert result.photometry[0].offset == pytest.approx(0.04, abs=1e-5)
    assert result.photometry[0].scale == pytest.approx(1.2, abs=1e-5)


def test_ground_combine_never_keeps_more_than_one_chunk_payload_live(tmp_path):
    sky = np.full((8, 8, 3), 0.2, dtype=np.float32)
    paths = [_write_sky_artifact(tmp_path, index, sky) for index in range(8)]
    artifacts = validate_artifact_set(paths)

    class Tracked:
        active = 0
        maximum = 0

        def __init__(self, path):
            loaded = load_linear_artifact(path)
            self.ground = loaded.ground
            type(self).active += 1
            type(self).maximum = max(type(self).maximum, type(self).active)

        def __del__(self):
            type(self).active -= 1

    combine_ground_two_pass(artifacts, loader=Tracked)

    assert Tracked.maximum == 1
    assert Tracked.active == 0


def test_deep_combine_blends_once_under_reference_mask_and_reports_effective_coverage(tmp_path):
    sky = np.full((40, 20, 3), 0.2, dtype=np.float32)
    ground = np.full(sky.shape, 0.1, dtype=np.float32)
    paths = [_write_sky_artifact(tmp_path, index, sky, ground=ground) for index in range(2)]
    for path in paths:
        _mutate_manifest(
            path, lambda manifest: manifest["mask_provenance"].update(resolved_boundary_row=20)
        )

    result = deep_combine(
        paths,
        reference=paths[0],
        fit=lambda moving, reference: np.eye(2, 3, dtype=np.float64),
    )

    expected_mask = horizon_mask(sky.shape[:2], horizon_frac=0.5)
    # Default output is the scoped frame: 40 -> 26 rows (margin 14, even), 20 -> 12 cols
    # (margin 8 after the parity trim).  Indices are hardcoded rather than derived from
    # scoped_frame_slice so this test cannot agree with a broken implementation.
    assert result.scope == "scoped"
    assert result.frame_origin == (7, 4)
    assert result.mask.shape == (26, 12)
    np.testing.assert_allclose(result.mask, expected_mask[7:33, 4:16])
    # The blend still happens at FULL frame -- the horizon row is defined against the full
    # height -- and only then is the scoped region shipped.  Same hardcoded window.
    np.testing.assert_allclose(result.rgb, blend(sky, ground, expected_mask)[7:33, 4:16], atol=1e-5)
    assert result.coverage.nominal_contributors == 20
    assert result.coverage.median_contributors == pytest.approx(20.0)
    assert result.coverage.below_floor_fraction == 0.0
    assert [plan.method for plan in result.transforms] == ["identity", "direct"]


def test_deep_combine_refuses_a_failed_direct_fit_rather_than_chaining(tmp_path):
    """The deep path no longer plans via ``plan_artifact_transforms`` -- a chunk whose direct
    fit to the single global reference fails must hard-fail even though an adjacent chain
    through its neighbour would have succeeded under the retired fallback."""
    sky = np.full((20, 20, 3), 0.2, dtype=np.float32)
    paths = [_write_sky_artifact(tmp_path, index, sky) for index in range(3)]
    index_by_path = {path: index for index, path in enumerate(paths)}

    def fit(moving, reference):
        pair = index_by_path[moving.path], index_by_path[reference.path]
        if pair == (0, 1):
            raise RegistrationError("coarse prior found too few matches")
        return np.eye(2, 3, dtype=np.float64)

    with pytest.raises(AstroCombineError, match=r"direct transform 0->1"):
        deep_combine(paths, reference=paths[1], fit=fit)


def test_deep_combine_hashes_during_validation_then_uses_trusted_reloads(tmp_path):
    sky = np.full((20, 20, 3), 0.2, dtype=np.float32)
    paths = [_write_sky_artifact(tmp_path, index, sky) for index in range(2)]
    reloads = []

    def loader(path, *, verify_hashes):
        reloads.append(verify_hashes)
        return load_linear_artifact(path, verify_hashes=verify_hashes)

    deep_combine(
        paths,
        reference=paths[0],
        fit=lambda moving, reference: np.eye(2, 3, dtype=np.float64),
        loader=loader,
    )

    assert reloads and not any(reloads)


def test_transform_plan_chains_to_the_nearest_resolved_artifact_not_the_reference(tmp_path):
    """An isolated direct-fit failure must cost ONE adjacent link, not a walk to the reference.

    Registration error compounds with every composed link (an audit found chunks needing 18
    composed links were destroyed while adjacent pairs were fine). Artifacts between a failed
    one and the reference are already resolved by the time it is planned, so chaining past them
    re-derives geometry that is already known and multiplies the error for nothing."""
    paths = [_write_artifact(tmp_path, index) for index in range(9)]
    artifacts = validate_artifact_set(paths)
    index_by_path = {path: index for index, path in enumerate(paths)}
    assert artifacts.reference_index == 4

    calls = []

    def fit(moving, reference):
        pair = index_by_path[moving.path], index_by_path[reference.path]
        calls.append(pair)
        if pair == (0, 4):  # only the far artifact fails its direct fit
            raise RegistrationError("coarse prior found too few matches")
        return _translation(pair[1] - pair[0])

    plans = plan_artifact_transforms(artifacts, fit)

    # One NEW fit (0->1); the rest of the path is artifact 1's already-verified direct link.
    assert plans[0].links == ((0, 1), (1, 4))
    assert plans[0].depth == 2
    assert (0, 2) not in calls and (0, 3) not in calls, (
        "artifact 0 must not re-derive geometry the artifacts between it and the reference hold"
    )
    np.testing.assert_allclose(plans[0].transform, _translation(4.0))


def test_transform_plan_reports_its_worst_composed_depth(tmp_path):
    """The combine must be able to refuse a run whose worst artifact is composed too deeply.

    A CONTIGUOUS run of direct-fit failures still chains -- artifact 0 can only reach the
    reference through artifact 1, which is itself chained. Depth makes that visible instead of
    leaving it buried in a links tuple."""
    paths = [_write_artifact(tmp_path, index) for index in range(7)]
    artifacts = validate_artifact_set(paths)
    index_by_path = {path: index for index, path in enumerate(paths)}
    assert artifacts.reference_index == 3

    def fit(moving, reference):
        pair = index_by_path[moving.path], index_by_path[reference.path]
        if pair in {(0, 3), (1, 3)}:  # a contiguous run of direct failures
            raise RegistrationError("coarse prior found too few matches")
        return _translation(pair[1] - pair[0])

    plans = plan_artifact_transforms(artifacts, fit)

    assert plans[3].depth == 0, "the reference composes nothing"
    assert plans[2].depth == 1, "a direct fit is one link deep"
    assert plans[1].depth == 2, "one hop onto a direct neighbour"
    assert plans[0].depth == 3, "one hop onto an already-chained neighbour"
    assert max(plan.depth for plan in plans) == 3
    np.testing.assert_allclose(plans[0].transform, _translation(3.0))


# --- the frozen ground band comes from the calmest (sharpest-ground) chunks only ---


def _flat_ground(shape):
    return np.full(shape, 0.2, dtype=np.float32)


def _poked_ground(shape, *, row, col, value):
    """A single-pixel spike on an otherwise flat field -- high local Laplacian variance
    (sharp), while staying robust-statistics-invisible (one pixel out of hundreds never moves
    a median/90th-percentile-based photometric estimate), matching the isolated-outlier shape
    ``test_ground_combine_stays_fixed_coordinate_and_rejects_one_surviving_light`` already
    exercises safely."""
    ground = _flat_ground(shape)
    ground[row, col] = value
    return ground


def test_ground_band_keep_mask_keeps_the_calmest_scores_not_the_first_n():
    # best two scores are at index 2 and 4 -- neither first, nor contiguous, nor last-N.
    scores = [1.0, 2.0, 9.0, 3.0, 8.0]

    assert combine_mod._ground_band_keep_mask(scores) == [False, False, True, False, True]


def test_ground_band_keep_mask_keeps_everything_when_nothing_distinguishes_the_chunks():
    assert combine_mod._ground_band_keep_mask([5.0] * 6) == [True] * 6


def test_ground_region_sharpness_scores_a_textured_chunk_above_a_flat_one():
    shape = (12, 12, 3)
    flat = _flat_ground(shape)
    textured = _poked_ground(shape, row=6, col=6, value=5.0)

    assert combine_mod._ground_region_sharpness(
        textured, 0.0
    ) > combine_mod._ground_region_sharpness(flat, 0.0)


def test_ground_region_sharpness_ignores_rows_above_the_horizon_boundary():
    # noisy "sky" rows above the boundary; perfectly flat "ground" rows below it -- the score
    # must come from the ground rows only, or the sky noise would swamp it.
    shape = (10, 6, 3)
    ground = np.full(shape, 0.2, dtype=np.float32)
    ground[:5] = np.random.default_rng(0).normal(0.2, 5.0, size=(5, 6, 3)).astype(np.float32)

    score = combine_mod._ground_region_sharpness(ground, boundary_row=5.0)

    assert score == pytest.approx(0.0, abs=1e-9)


def test_ground_combine_keeps_only_the_calmest_ground_band_by_weight_count(tmp_path):
    shape = (16, 16, 3)
    sky = np.full(shape, 0.2, dtype=np.float32)
    grounds = [
        _flat_ground(shape),  # index 0: blurriest (flat) -- also the reference
        _flat_ground(shape),  # index 1: blurry, tied with index 0
        _poked_ground(shape, row=8, col=8, value=6.0),  # index 2: sharpest, MIDDLE of the list
        _flat_ground(shape),  # index 3: blurry, tied with index 0/1
        _poked_ground(shape, row=3, col=12, value=4.0),  # index 4: 2nd-sharpest, LAST
    ]
    paths = [_write_sky_artifact(tmp_path, index, sky, ground=g) for index, g in enumerate(grounds)]
    for path in paths:
        _mutate_manifest(
            path, lambda manifest: manifest["mask_provenance"].update(resolved_boundary_row=0)
        )
    artifacts = validate_artifact_set(paths, reference=paths[0])

    result = combine_ground_two_pass(artifacts)

    # 2 of 5 chunks (the calmest 30%, rounded up) feed the ground -- 20 frames, not 50, and
    # they are chunks 2 and 4 -- neither the first two nor a contiguous run.
    assert result.weight[0, 0] == pytest.approx(20.0)


def test_ground_combine_excludes_a_bad_ground_chunk_even_when_its_sky_is_fine(tmp_path):
    shape = (12, 12, 3)
    sky = np.full(shape, 0.2, dtype=np.float32)  # identical, healthy sky for both chunks
    blurry_ground = _flat_ground(shape)  # perfectly flat: sharpness score 0
    sharp_ground = _poked_ground(shape, row=6, col=6, value=5.0)
    paths = [
        _write_sky_artifact(tmp_path, 0, sky, ground=blurry_ground),
        _write_sky_artifact(tmp_path, 1, sky, ground=sharp_ground),
    ]
    for path in paths:
        _mutate_manifest(
            path, lambda manifest: manifest["mask_provenance"].update(resolved_boundary_row=0)
        )
    artifacts = validate_artifact_set(paths, reference=paths[0])
    transforms = tuple(ArtifactTransform(np.eye(2, 3), "identity", ()) for _ in artifacts.artifacts)

    ground_result = combine_ground_two_pass(artifacts)
    sky_result = combine_sky_two_pass(artifacts, transforms)

    # the blurry chunk contributes nothing to the ground band...
    assert ground_result.weight[0, 0] == pytest.approx(10.0)
    # ...but its sky -- a wholly separate combine the ground selection never touches -- is fine.
    assert sky_result.sky_count[0, 0] == pytest.approx(20.0)


def test_deep_combine_horizon_seam_blends_across_rows_not_at_one_hard_row(tmp_path):
    shape = (40, 20, 3)
    sky = np.full(shape, 0.2, dtype=np.float32)
    ground = np.full(shape, 0.1, dtype=np.float32)
    paths = [_write_sky_artifact(tmp_path, index, sky, ground=ground) for index in range(2)]
    for path in paths:
        _mutate_manifest(
            path, lambda manifest: manifest["mask_provenance"].update(resolved_boundary_row=20)
        )

    result = deep_combine(
        paths, reference=paths[0], fit=lambda moving, reference: np.eye(2, 3, dtype=np.float64)
    )

    column = np.round(result.mask[:, 0], 3)
    # a hard-switch seam would show ONLY {0.0, 1.0}; a soft one shows intermediate values too.
    assert not set(column.tolist()).issubset({0.0, 1.0})


# --- lens-aware chunk-to-chunk registration -----------------------------------------


def _radius_fraction(xy, frame_shape):
    """Radius from frame centre over centre-to-corner distance -- the same banding
    ``astro_register.radial_inlier_residuals_px`` bins on."""
    height, width = frame_shape[:2]
    center = np.array([(width - 1) / 2, (height - 1) / 2])
    return np.linalg.norm(np.asarray(xy) - center, axis=1) / np.linalg.norm(center)


def _synthetic_chunk_pair(thin_outer_third=True):
    """One chunk pair with a KNOWN Brown radial distortion plus a known rotation, shaped
    like the real 249-frame night's nearest pair.

    ``tests/fixtures/astro_synthetic`` renders both sides through one shared lens, so
    every pair here is a true match -- the only thing a fit can get wrong is the MODEL.
    The outer radial third is thinned to a quarter of its stars because that is the
    supply a real pair has there (the measured 11->12 pair matched 251 centre / 353
    middle / 43 outer): with the outer third as populous as uniform sampling makes it,
    RANSAC's consensus is no longer the centre-only one the real fits collapse to.
    """
    night = build_synthetic_night(n_frames=1, n_stars=600, rotation_scale_rad=0.016, seed=7)
    match = night.matches[0]
    frame_shape = night.frame_shape
    height, width = frame_shape
    on_sensor = (
        (match.ref_xy[:, 0] >= 0)
        & (match.ref_xy[:, 0] < width)
        & (match.ref_xy[:, 1] >= 0)
        & (match.ref_xy[:, 1] < height)
    )
    ref_xy, frame_xy = match.ref_xy[on_sensor], match.frame_xy[on_sensor]
    if not thin_outer_third:
        return ref_xy, frame_xy, frame_shape
    rng = np.random.default_rng(11)
    thinned = (_radius_fraction(ref_xy, frame_shape) < 2 / 3) | (rng.random(len(ref_xy)) < 0.25)
    return ref_xy[thinned], frame_xy[thinned], frame_shape


def test_chunk_pair_fit_keeps_the_outer_third_a_similarity_throws_away():
    """The 4-DOF similarity has no term that can absorb radial distortion, so RANSAC
    labels every corner star an outlier and ``MIN_OUTER_THIRD_INLIERS`` correctly refuses
    the fit. The lens+rotation model fits the whole sensor instead -- same gates, new
    model's residuals, and residuals that do not grow with radius."""
    ref_xy, frame_xy, frame_shape = _synthetic_chunk_pair()

    similarity = fit_similarity(ref_xy, frame_xy)
    predicted = frame_xy @ similarity[:, :2].T + similarity[:, 2]
    similarity_inliers = np.linalg.norm(predicted - ref_xy, axis=1) <= RANSAC_REPROJ_THRESHOLD_PX
    similarity_radial = radial_inlier_residuals_px(
        similarity, ref_xy, frame_xy, similarity_inliers, frame_shape
    )
    assert similarity_radial["center"]["n"] >= MIN_OUTER_THIRD_INLIERS
    assert similarity_radial["outer"]["n"] == 0, (
        "fixture must reproduce the measured defect: a centre-only similarity fit"
    )
    assert _radius_fraction(ref_xy, frame_shape).max() >= 2 / 3

    fit = combine_mod.fit_matched_pairs(ref_xy, frame_xy, frame_shape)

    assert fit.geometry_method == "lens_rotation"
    assert fit.fallback_reason is None
    radial = fit.radial_inlier_residuals_px
    assert radial["outer"]["n"] >= MIN_OUTER_THIRD_INLIERS
    assert radial["outer"]["median_px"] <= 2.0 * radial["center"]["median_px"], (
        "residual must not grow with radius under the lens model"
    )
    assert fit.residual_p95_px < RANSAC_REPROJ_THRESHOLD_PX
    map_x, map_y = fit.transform
    assert map_x.shape == frame_shape and map_y.shape == frame_shape


def test_chunk_pair_fit_reports_a_similarity_fallback_when_the_lens_solve_refuses(monkeypatch):
    """A non-convergent lens solve falls back to the old similarity -- explicitly named on
    the result, never a silent downgrade, and still subject to ``register``'s own gates."""
    ref_xy, frame_xy, frame_shape = _synthetic_chunk_pair()
    stars_ref = np.column_stack([ref_xy, np.ones(len(ref_xy))]).astype(np.float64)
    stars_moving = np.column_stack([frame_xy, np.ones(len(frame_xy))]).astype(np.float64)
    luma = np.zeros(frame_shape, dtype=np.float32)
    monkeypatch.setattr(combine_mod, "coarse_shift", lambda *args, **kwargs: (0.0, 0.0))

    def refuse(*args, **kwargs):
        raise GeometrySolveError("least_squares did not converge from any k1 start offset")

    monkeypatch.setattr(combine_mod, "solve_night", refuse)
    sentinel = RegistrationResult(np.eye(2, 3), 12, 0.9, 0.3, 0.8, 0.7)
    monkeypatch.setattr(combine_mod, "register", lambda *args, **kwargs: sentinel)

    fit = combine_mod.fit_lens_aware(luma, stars_ref, luma, stars_moving)

    assert fit.geometry_method == "similarity_fallback"
    assert "did not converge" in fit.fallback_reason
    assert fit.n_matched == 12
    np.testing.assert_allclose(fit.transform, np.eye(2, 3))


def test_direct_plan_and_warp_accept_a_lens_remap(tmp_path):
    """``plan_direct_transforms``/``_warped_components`` must carry and apply a
    ``(map_x, map_y)`` remap, not only a 2x3 affine."""
    sky = np.zeros((24, 32, 3), dtype=np.float32)
    sky[8:16, 10:22] = 0.6
    paths = [_write_sky_artifact(tmp_path, index, sky) for index in range(2)]
    artifacts = validate_artifact_set(paths, reference=paths[0])
    shape = sky.shape
    lens = LensParams(k1=0.08, k2=0.0, cx=(shape[1] - 1) / 2, cy=(shape[0] - 1) / 2, f_px=30.0)
    maps = build_remap(shape[:2], lens, np.array([0.0, 0.0, 0.05]))

    plans = plan_direct_transforms(artifacts, lambda moving, reference: maps)

    assert plans[0].method == "identity"
    assert plans[1].transform is None, "a remap is not a matrix and must not pose as one"
    np.testing.assert_allclose(plans[1].remap[0], maps[0])
    warped_sum, warped_count = combine_mod._warped_components(
        artifacts.artifacts[1], plans[1].warp, shape, load_linear_artifact
    )
    loaded = load_linear_artifact(paths[1])
    np.testing.assert_allclose(
        warped_sum, warp_to_reference(loaded.sky_sum, maps, shape[:2]), atol=1e-5
    )
    assert warped_count.shape == shape[:2]
    assert not np.allclose(warped_sum, np.asarray(loaded.sky_sum)), (
        "a remap that rotates the sky must actually move pixels"
    )


def _synthetic_homography_pair():
    """The same synthetic pair plus a small NON-rotational component -- a 0.8% anisotropic
    scale and shear applied in UNDISTORTED space, i.e. a tripod that shifted slightly as
    well as turning. No Rodrigues rotation describes that, so ``solve_night``'s own
    rotation-residual gate hands the pair to its 8-DOF homography branch. Unlike the
    similarity, that homography is fitted AFTER the Brown model is removed, so it is a
    genuine lens-aware result rather than a model that has to absorb distortion."""
    ref_xy, frame_xy, frame_shape = _synthetic_chunk_pair(thin_outer_third=False)
    centre = np.array([TRUE_LENS.cx, TRUE_LENS.cy])
    skew = 0.008
    non_rotation = np.array([[1 + skew, 0.75 * skew], [-0.75 * skew, 1 - skew]])
    ideal = (undistort(frame_xy, TRUE_LENS) - centre) @ non_rotation.T + centre
    return ref_xy, distort(ideal, TRUE_LENS), frame_shape


def test_chunk_pair_fit_accepts_a_homography_solution_as_lens_aware():
    """A pair `solve_night` describes with a homography is a first-class lens-aware fit,
    distinguishable in diagnostics -- not a refusal, and not a downgrade to the similarity."""
    ref_xy, frame_xy, frame_shape = _synthetic_homography_pair()

    fit = combine_mod.fit_matched_pairs(ref_xy, frame_xy, frame_shape)

    assert fit.geometry_method == "lens_homography"
    assert fit.fallback_reason is None
    radial = fit.radial_inlier_residuals_px
    assert radial["outer"]["n"] >= MIN_OUTER_THIRD_INLIERS
    assert fit.residual_p95_px < RANSAC_REPROJ_THRESHOLD_PX
    map_x, map_y = fit.transform
    assert map_x.shape == frame_shape and map_y.shape == frame_shape


def test_a_lens_fit_refused_by_its_gates_names_the_lens_model():
    """A refusal carries WHICH model was refused, so a caller never has to guess whether the
    gates measured the lens model or a downgraded similarity."""
    ref_xy, frame_xy, frame_shape = _synthetic_chunk_pair()
    inner = _radius_fraction(ref_xy, frame_shape) < 0.6

    with pytest.raises(RegistrationError) as excinfo:
        combine_mod.fit_matched_pairs(ref_xy[inner], frame_xy[inner], frame_shape)

    assert "outer-third inliers" in str(excinfo.value)
    assert excinfo.value.geometry_method == "lens_rotation"
    assert excinfo.value.fallback_reason is None


def test_a_failed_direct_pair_names_its_geometry_model_and_downgrade(tmp_path, monkeypatch):
    """A pair that fell back to the similarity and then died in ``register``'s gates must say
    so where the combine actually aborts -- an invisible downgrade is the failure mode the
    fallback exists to avoid."""
    ref_xy, frame_xy, frame_shape = _synthetic_chunk_pair()
    stars_ref = np.column_stack([ref_xy, np.ones(len(ref_xy))]).astype(np.float64)
    stars_moving = np.column_stack([frame_xy, np.ones(len(frame_xy))]).astype(np.float64)
    luma = np.zeros(frame_shape, dtype=np.float32)
    monkeypatch.setattr(combine_mod, "coarse_shift", lambda *args, **kwargs: (0.0, 0.0))

    def refuse_solve(*args, **kwargs):
        raise GeometrySolveError("least_squares did not converge from any k1 start offset")

    def refuse_register(*args, **kwargs):
        raise RegistrationError("fit kept only 3 inliers, need >= 6")

    monkeypatch.setattr(combine_mod, "solve_night", refuse_solve)
    monkeypatch.setattr(combine_mod, "register", refuse_register)
    paths = [_write_artifact(tmp_path, index) for index in range(2)]
    artifacts = validate_artifact_set(paths, reference=paths[0])

    with pytest.raises(AstroCombineError) as excinfo:
        plan_direct_transforms(
            artifacts,
            lambda moving, reference: combine_mod.fit_lens_aware(
                luma, stars_ref, luma, stars_moving
            ),
        )

    message = str(excinfo.value)
    assert "direct transform 1->0" in message
    assert "fit kept only 3 inliers" in message
    assert "similarity_fallback" in message
    assert "least_squares did not converge" in message
