"""Persistence contract for reusable deep-linear stack components."""

import hashlib
import json

import numpy as np
import pytest

from gomoveshift.pipeline.photos.astro_linear import (
    LINEAR_ARTIFACT_SCHEMA,
    LinearArtifactError,
    chunk_sky_weight,
    load_linear_artifact,
    write_linear_artifact,
)
from gomoveshift.pipeline.photos.astro_stack import StackResult


def _result() -> StackResult:
    sky_sum = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    sky_count = np.array([[4, 3, 4, 2], [4, 4, 3, 4]], dtype=np.uint16)
    ground = np.full((2, 4, 3), 0.125, dtype=np.float32)
    return StackResult(
        rgb=np.zeros((2, 4, 3), dtype=np.float32),
        n_frames=4,
        mean_rejected_fraction=0.1,
        sky_sum=sky_sum,
        sky_count=sky_count,
        ground=ground,
    )


def test_linear_artifact_round_trips_memmap_components_and_provenance(tmp_path):
    destination = tmp_path / "chunk.linear"
    write_linear_artifact(
        destination,
        _result(),
        reference_raw="DSC_0003.NEF",
        capture_identity={"camera": "Nikon Z 6", "raw_shape": [3728, 5600], "iso": 6400},
        mask_provenance={"mode": "explicit", "horizon_frac": 0.62},
        source_frames=["DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF", "DSC_0004.NEF"],
        registration_provenance=[{"frame": "DSC_0001.NEF", "transform": "identity"}],
        photometry_provenance=[{"frame": "DSC_0001.NEF", "offset": 0.0, "scale": 1.0}],
        algorithm_version="astro-linear-v1",
    )

    artifact = load_linear_artifact(destination)

    assert isinstance(artifact.sky_sum, np.memmap)
    assert artifact.sky_sum.dtype == np.float32
    # sky_count.npy persists a per-pixel quality-weighted contribution sum (float32).
    # A single registration entry carries no triage, so `chunk_sky_weight` falls back to 1.0
    # and the weighted values are numerically identical to the original uint16 counts.
    assert artifact.sky_count.dtype == np.float32
    assert np.array_equal(artifact.sky_count, _result().sky_count.astype(np.float32))
    assert np.array_equal(artifact.sky_sum, _result().sky_sum)
    assert artifact.manifest["reference_raw"] == "DSC_0003.NEF"
    assert artifact.manifest["capture_identity"]["camera"] == "Nikon Z 6"
    assert artifact.manifest["mask_provenance"] == {"horizon_frac": 0.62, "mode": "explicit"}
    assert artifact.manifest["source_frames"] == [
        "DSC_0001.NEF",
        "DSC_0002.NEF",
        "DSC_0003.NEF",
        "DSC_0004.NEF",
    ]
    assert artifact.manifest["registration"][0]["transform"] == "identity"
    assert artifact.manifest["photometry"][0]["scale"] == 1.0
    assert artifact.manifest["payloads"]["ground.npy"]["sha256"]


def test_linear_artifact_manifest_carries_per_frame_quality_and_bumped_schema(tmp_path):
    destination = tmp_path / "chunk.linear"
    registration_provenance = [
        {
            "frame": "DSC_0001.NEF",
            "transform": "identity",
            "n_matched": 42,
            "inlier_fraction": 0.95,
            "residual_median_px": 0.12,
            "residual_p95_px": 0.30,
            # In-sample: RANSAC-inlier residuals under the fitted transform, not
            # held-out geometry evidence.
            "radial_inlier_residuals_px_in_sample": {
                "center": {"n": 10, "median_px": 0.1, "p95_px": 0.2},
                "middle": {"n": 20, "median_px": 0.15, "p95_px": 0.25},
                "outer": {"n": 5, "median_px": 0.3, "p95_px": 0.5},
            },
            "triage": {
                "star_count": 120,
                "background_median": 512.0,
                "fwhm_px": 2.4,
                "shape": {"fwhm_major_px": 2.6, "fwhm_minor_px": 2.2, "ellipticity": 0.15},
            },
            "exposure_s": 20.0,
            "captured_at": "2026-08-01T03:14:00+00:00",
        }
    ]
    write_linear_artifact(
        destination,
        _result(),
        reference_raw="DSC_0003.NEF",
        capture_identity={"camera": "Nikon Z 6", "raw_shape": [3728, 5600], "iso": 6400},
        mask_provenance={"mode": "explicit", "horizon_frac": 0.62},
        source_frames=["DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF", "DSC_0004.NEF"],
        registration_provenance=registration_provenance,
        photometry_provenance=[{"frame": "DSC_0001.NEF", "offset": 0.0, "scale": 1.0}],
        algorithm_version="astro-linear-v1",
    )

    artifact = load_linear_artifact(destination)

    assert LINEAR_ARTIFACT_SCHEMA >= 2
    assert artifact.manifest["schema"] == LINEAR_ARTIFACT_SCHEMA
    entry = artifact.manifest["registration"][0]
    assert entry["triage"]["fwhm_px"] == 2.4
    assert entry["triage"]["shape"]["ellipticity"] == 0.15
    assert entry["radial_inlier_residuals_px_in_sample"]["outer"]["median_px"] == 0.3
    assert entry["exposure_s"] == 20.0
    assert entry["captured_at"] == "2026-08-01T03:14:00+00:00"


def test_linear_artifact_manifest_carries_match_yield_by_stage_and_bumped_schema(tmp_path):
    destination = tmp_path / "chunk.linear"
    registration_provenance = [
        {
            "frame": "DSC_0001.NEF",
            "transform": "identity",
            "n_matched": 42,
            "inlier_fraction": 0.95,
            "residual_median_px": 0.12,
            "residual_p95_px": 0.30,
            "radial_inlier_residuals_px_in_sample": {
                "center": {"n": 10, "median_px": 0.1, "p95_px": 0.2},
                "middle": {"n": 20, "median_px": 0.15, "p95_px": 0.25},
                "outer": {"n": 8, "median_px": 0.3, "p95_px": 0.5},
            },
            # The match_progressive per-stage match counts -- where this fit's
            # matched pairs actually came from, in-sample like the residuals above.
            "match_yield_by_stage": {
                "seed": 20,
                "wide": 38,
                "tighten_10px": 40,
                "tighten_4px": 41,
                "final": 42,
            },
        }
    ]
    write_linear_artifact(
        destination,
        _result(),
        reference_raw="DSC_0003.NEF",
        capture_identity={"camera": "Nikon Z 6", "raw_shape": [3728, 5600], "iso": 6400},
        mask_provenance={"mode": "explicit", "horizon_frac": 0.62},
        source_frames=["DSC_0001.NEF", "DSC_0002.NEF", "DSC_0003.NEF", "DSC_0004.NEF"],
        registration_provenance=registration_provenance,
        photometry_provenance=[{"frame": "DSC_0001.NEF", "offset": 0.0, "scale": 1.0}],
        algorithm_version="astro-linear-v1",
    )

    artifact = load_linear_artifact(destination)

    assert LINEAR_ARTIFACT_SCHEMA >= 4
    assert artifact.manifest["schema"] == LINEAR_ARTIFACT_SCHEMA
    entry = artifact.manifest["registration"][0]
    assert entry["match_yield_by_stage"]["final"] == 42
    assert entry["match_yield_by_stage"]["seed"] == 20


def test_linear_artifact_reader_tolerates_an_older_schema_written_before_this_change(tmp_path):
    destination = tmp_path / "chunk.linear"
    write_linear_artifact(
        destination,
        _result(),
        reference_raw="DSC_0003.NEF",
        capture_identity={"camera": "Nikon Z 6", "raw_shape": [3728, 5600], "iso": 6400},
        mask_provenance={"mode": "explicit"},
        source_frames=["DSC_0003.NEF"],
        registration_provenance=[{"frame": "DSC_0003.NEF", "transform": "identity"}],
        photometry_provenance=[{"frame": "DSC_0003.NEF", "offset": 0.0, "scale": 1.0}],
        algorithm_version="astro-linear-v1",
    )
    with (destination / "manifest.json").open() as stream:
        manifest = json.load(stream)
    # A schema-1 artifact predates the sky_count weighting: sky_count.npy is a literal
    # uint16 frame count on disk, not the float32 weighted sum write_linear_artifact now
    # produces. Rewrite the payload itself, not just the schema number, to genuinely simulate
    # a pre-existing artifact rather than one this version wrote with the number changed.
    legacy_count = _result().sky_count
    np.save(destination / "sky_count.npy", legacy_count, allow_pickle=False)
    with (destination / "sky_count.npy").open("rb") as stream:
        digest_source = stream.read()
    manifest["payloads"]["sky_count.npy"] = {
        "sha256": hashlib.sha256(digest_source).hexdigest(),
        "shape": list(legacy_count.shape),
        "dtype": str(legacy_count.dtype),
    }
    manifest["schema"] = 1  # simulates an artifact written before the weighted format
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    artifact = load_linear_artifact(destination)

    assert artifact.manifest["schema"] == 1
    assert artifact.sky_count.dtype == np.uint16


def test_chunk_sky_weight_is_neutral_without_enough_triage():
    """Fewer than two triage-bearing frames -- legacy manifests, or triage-free entries --
    contribute exactly as before: weight 1.0, no invented signal."""
    assert chunk_sky_weight([{"frame": "a"}]) == 1.0
    assert chunk_sky_weight([]) == 1.0
    assert (
        chunk_sky_weight(
            [
                {"triage": {"fwhm_px": 2.4, "background_median": 0.03}},
                {"frame": "b"},
            ]
        )
        == 1.0
    )


def test_chunk_sky_weight_falls_below_one_when_a_chunks_own_frames_disagree():
    """Frames whose focus and background level agree keep the chunk near full weight; frames
    that vary wildly against each other pull the chunk's own weight down below 1.0."""
    steady = chunk_sky_weight(
        [
            {"triage": {"fwhm_px": 2.4, "background_median": 0.030}},
            {"triage": {"fwhm_px": 2.41, "background_median": 0.0301}},
            {"triage": {"fwhm_px": 2.39, "background_median": 0.0299}},
        ]
    )
    erratic = chunk_sky_weight(
        [
            {"triage": {"fwhm_px": 2.0, "background_median": 0.02}},
            {"triage": {"fwhm_px": 6.0, "background_median": 0.08}},
            {"triage": {"fwhm_px": 3.0, "background_median": 0.05}},
        ]
    )
    assert steady > 0.9
    assert 0.0 < erratic < steady


def test_write_linear_artifact_scales_sky_count_by_the_chunk_weight(tmp_path):
    destination = tmp_path / "chunk.linear"
    registration_provenance = [
        {"triage": {"fwhm_px": 2.0, "background_median": 0.02}},
        {"triage": {"fwhm_px": 6.0, "background_median": 0.08}},
        {"triage": {"fwhm_px": 3.0, "background_median": 0.05}},
    ]
    write_linear_artifact(
        destination,
        _result(),
        reference_raw="DSC_0003.NEF",
        capture_identity={"camera": "Nikon Z 6", "raw_shape": [3728, 5600], "iso": 6400},
        mask_provenance={"mode": "explicit"},
        source_frames=["DSC_0003.NEF"],
        registration_provenance=registration_provenance,
        photometry_provenance=[{"frame": "DSC_0003.NEF", "offset": 0.0, "scale": 1.0}],
        algorithm_version="astro-linear-v1",
    )

    artifact = load_linear_artifact(destination)
    weight = chunk_sky_weight(registration_provenance)

    assert 0.0 < weight < 1.0
    assert artifact.manifest["sky_weight"] == weight
    assert np.allclose(artifact.sky_count, _result().sky_count.astype(np.float32) * weight)


def test_linear_artifact_refuses_a_payload_whose_hash_no_longer_matches(tmp_path):
    destination = tmp_path / "chunk.linear"
    write_linear_artifact(
        destination,
        _result(),
        reference_raw="DSC_0003.NEF",
        capture_identity={"camera": "Nikon Z 6", "raw_shape": [3728, 5600], "iso": 6400},
        mask_provenance={"mode": "explicit"},
        source_frames=["DSC_0003.NEF"],
        registration_provenance=[{"frame": "DSC_0003.NEF", "transform": "identity"}],
        photometry_provenance=[{"frame": "DSC_0003.NEF", "offset": 0.0, "scale": 1.0}],
        algorithm_version="astro-linear-v1",
    )
    with (destination / "manifest.json").open() as stream:
        manifest = json.load(stream)
    manifest["payloads"]["ground.npy"]["sha256"] = "0" * 64
    (destination / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(LinearArtifactError, match="sha256"):
        load_linear_artifact(destination)

    trusted = load_linear_artifact(destination, verify_hashes=False)

    assert np.array_equal(trusted.ground, _result().ground)


def test_linear_artifact_refuses_partial_stack_results_without_leaving_a_destination(tmp_path):
    destination = tmp_path / "chunk.linear"
    partial = StackResult(
        rgb=np.zeros((2, 4, 3), dtype=np.float32), n_frames=4, mean_rejected_fraction=0.0
    )

    with pytest.raises(LinearArtifactError, match="sky_sum"):
        write_linear_artifact(
            destination,
            partial,
            reference_raw="DSC_0003.NEF",
            capture_identity={"camera": "Nikon Z 6", "raw_shape": [3728, 5600], "iso": 6400},
            mask_provenance={"mode": "explicit"},
            source_frames=["DSC_0003.NEF"],
            registration_provenance=[{"frame": "DSC_0003.NEF", "transform": "identity"}],
            photometry_provenance=[{"frame": "DSC_0003.NEF", "offset": 0.0, "scale": 1.0}],
            algorithm_version="astro-linear-v1",
        )

    assert not destination.exists()
