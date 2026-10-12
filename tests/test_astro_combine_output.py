import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.photos.astro_combine import (
    ArtifactTransform,
    ChunkPhotometry,
    DeepCombineResult,
    DeepCoverage,
    DeepGroundResult,
    DeepSkyResult,
    ValidatedArtifact,
    ValidatedArtifactSet,
)
from gomoveshift.pipeline.photos.astro_combine_output import (
    DeepOutputError,
    deep_output_paths,
    deep_sidecar_payload,
    publish_deep_product,
)
from gomoveshift.pipeline.photos.astro_develop import AstroDevelopResult, GradientResidual


def _result(tmp_path):
    shape = (4, 5, 3)
    artifacts = []
    for index in range(2):
        manifest = {
            "n_frames": 10,
            "reference_raw": f"RAW-{index}-mid.NEF",
            "source_frames": [f"RAW-{index}-{frame}.NEF" for frame in range(10)],
            "capture_identity": {"camera": "NIKON Z 6", "iso": 6400},
            "mask_provenance": {"mode": "automatic", "resolved_boundary_row": 2},
            "payloads": {"sky_sum.npy": {"sha256": str(index) * 64}},
        }
        artifacts.append(ValidatedArtifact(tmp_path / f"chunk-{index}.linear", manifest))
    sky_count = np.full(shape[:2], 20, dtype=np.float32)
    return DeepCombineResult(
        scope="scoped",
        frame_origin=(0, 0),
        rgb=np.full(shape, 0.2, dtype=np.float32),
        mask=np.vstack([np.ones((2, 5)), np.zeros((2, 5))]).astype(np.float32),
        sky=DeepSkyResult(
            sky_sum=np.full(shape, 4.0, dtype=np.float32),
            sky_count=sky_count,
            photometry=(ChunkPhotometry(0.0, 1.0), ChunkPhotometry(0.1, 1.2)),
            rejected_pixel_fraction=0.01,
        ),
        ground=DeepGroundResult(
            ground=np.full(shape, 0.1, dtype=np.float32),
            weight=sky_count,
            photometry=(ChunkPhotometry(0.0, 1.0), ChunkPhotometry(0.2, 0.9)),
            rejected_pixel_fraction=0.02,
        ),
        transforms=(
            ArtifactTransform(np.eye(2, 3), "identity", ()),
            ArtifactTransform(
                np.array([[1.0, 0.0, 2.0], [0.0, 1.0, -1.0]]),
                "direct",
                ((1, 0),),
                ({"residual_p95_px": 1.2, "radial": {"outer_px": 2.3}},),
            ),
        ),
        coverage=DeepCoverage(20, 10.0, 20.0, 20.0, 0.0),
        artifacts=ValidatedArtifactSet(tuple(artifacts), 0),
    )


def _developed(result, gradient_residual=None):
    return AstroDevelopResult(
        np.full(result.rgb.shape, 32768, dtype=np.uint16),
        0.01,
        [0.2, 0.3],
        0.1,
        0.2,
        (1.0, 0.9, 1.1),
        0.8,
        result.mask > 0.5,
        gradient_residual,
    )


def test_publish_deep_product_develops_and_exports_once_with_coverage_and_sidecar(tmp_path):
    result = _result(tmp_path)
    calls = []

    def develop(rgb, *, sky_mask):
        calls.append(("develop", sky_mask.copy()))
        return _developed(result)

    def export(rgb, path):
        calls.append(("export", rgb.copy()))
        Path(path).write_bytes(b"tiff")
        return Path(path)

    paths = publish_deep_product(
        result,
        tmp_path / "night.tif",
        develop=develop,
        export=export,
        disk_check=lambda *args, **kwargs: None,
    )

    assert [call[0] for call in calls] == ["develop", "export"]
    np.testing.assert_array_equal(np.load(paths.coverage), result.sky.sky_count)
    sidecar = json.loads(paths.sidecar.read_text())
    assert sidecar["reference"]["artifact"] == str(result.artifacts.reference.path)
    assert sidecar["chunks"][1]["sky_photometry"]["scale"] == pytest.approx(1.2)
    assert sidecar["chunks"][1]["transform"]["diagnostics"][0]["radial"]["outer_px"] == 2.3
    # existing develop keys, unchanged by this addition
    assert sidecar["develop"]["black_point"] == pytest.approx(0.01)
    assert sidecar["develop"]["m_values"] == [0.2, 0.3]
    assert sidecar["develop"]["background_before"] == pytest.approx(0.1)
    assert sidecar["develop"]["background_after"] == pytest.approx(0.2)
    assert sidecar["develop"]["background_gains"] == pytest.approx([1.0, 0.9, 1.1])
    assert sidecar["develop"]["peak"] == pytest.approx(0.8)
    # no residual measured (develop stub didn't return one) -> honest absence, not 0.0
    assert sidecar["develop"]["gradient_residual"] == {"center": None, "corners": None}

    with pytest.raises(DeepOutputError, match="already exists"):
        publish_deep_product(
            result,
            paths.image,
            develop=lambda *args, **kwargs: pytest.fail("develop ran after collision"),
            disk_check=lambda *args, **kwargs: None,
        )


def test_deep_sidecar_payload_records_the_gradient_corner_residual(tmp_path):
    result = _result(tmp_path)
    paths = deep_output_paths(tmp_path / "night.tif")
    developed = _developed(result, gradient_residual=GradientResidual(center=0.01, corners=0.42))
    Path(paths.image).write_bytes(b"tiff")

    payload = deep_sidecar_payload(result, developed, paths)

    assert payload["develop"]["gradient_residual"] == {"center": 0.01, "corners": 0.42}


def test_deep_sidecar_payload_records_the_auto_scope_decision(tmp_path):
    result = dataclasses.replace(
        _result(tmp_path),
        scope="full",
        requested_scope="auto",
        scope_decision={"resolved_scope": "full", "outer_passed": True},
    )
    paths = deep_output_paths(tmp_path / "night.tif")
    Path(paths.image).write_bytes(b"tiff")

    payload = deep_sidecar_payload(result, _developed(result), paths)

    assert payload["scope"] == {
        "requested": "auto",
        "resolved": "full",
        "decision": {"resolved_scope": "full", "outer_passed": True},
    }


def test_deep_sidecar_payload_falls_back_to_resolved_scope_when_requested_is_unset(tmp_path):
    """A result built directly (pre-auto-scope construction, no `requested_scope`) still serialises a
    non-null "requested" -- never a missing/None key for an explicit, non-auto request."""
    result = _result(tmp_path)
    paths = deep_output_paths(tmp_path / "night.tif")
    Path(paths.image).write_bytes(b"tiff")

    payload = deep_sidecar_payload(result, _developed(result), paths)

    assert payload["scope"] == {"requested": "scoped", "resolved": "scoped", "decision": None}
