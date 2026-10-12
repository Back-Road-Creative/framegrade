"""Reusable renders require verified inputs and complete, bound outputs."""

import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.photos import night_render as render
from gomoveshift.pipeline.photos.astro_linear import write_linear_artifact
from gomoveshift.pipeline.photos.astro_stack import StackResult


@pytest.fixture
def source(tmp_path):
    rgb = np.full((12, 16, 3), 0.1, dtype=np.float32)
    path = tmp_path / "source.linear"
    write_linear_artifact(
        path,
        StackResult(
            rgb=rgb,
            n_frames=2,
            mean_rejected_fraction=0,
            sky_sum=rgb * 2,
            sky_count=np.full((12, 16), 2, np.uint16),
            ground=rgb,
        ),
        reference_raw="original.NEF",
        capture_identity={},
        mask_provenance={"resolved_boundary_row": 10},
        source_frames=["original.NEF"],
        registration_provenance=[{"frame": "original.NEF"}],
        photometry_provenance=[{"frame": "original.NEF"}],
        algorithm_version="test",
    )
    return path


def test_render_receipt_and_output_integrity(source, tmp_path):
    import tifffile

    destination = tmp_path / "attempt"
    before = (source / "manifest.json").read_bytes()
    receipt = render.render_artifact(source, destination)
    assert render.validate_render(source, destination)
    assert receipt["source_manifest_sha256"] == hashlib.sha256(before).hexdigest()
    assert receipt["source_payloads"]["sky_sum.npy"]["sha256"]
    assert receipt["recipe"]["version"]
    assert (source / "manifest.json").read_bytes() == before
    with Image.open(destination / "preview.jpg") as image:
        assert image.size == (16, 12)
        assert image.info["icc_profile"]
    with tifffile.TiffFile(destination / "master.tif") as master:
        assert master.asarray().dtype == np.uint16
        assert master.pages[0].tags[34675].value
    assert not render.validate_render(source, destination, "bold")
    (destination / "preview.jpg").unlink()
    assert not render.validate_render(source, destination)


@pytest.mark.parametrize("target", ["manifest.json", "sky_sum.npy", "preview.jpg", "receipt.json"])
def test_changed_source_or_outputs_not_reusable(source, tmp_path, target):
    destination = tmp_path / "attempt"
    render.render_artifact(source, destination)
    path = (source if target in {"manifest.json", "sky_sum.npy"} else destination) / target
    path.write_bytes(path.read_bytes() + b" ")
    # JSON whitespace alone is meaningful for the source digest but not receipt semantics.
    if target == "receipt.json":
        value = json.loads(path.read_text())
        value["outputs"]["preview.jpg"]["sha256"] = "tampered"
        path.write_text(json.dumps(value))
    assert not render.validate_render(source, destination)


def test_preview_failure_keeps_master_but_never_marks_complete(source, tmp_path, monkeypatch):
    destination = tmp_path / "attempt"

    def fail(*args, **kwargs):
        raise OSError("preview disk failure")

    monkeypatch.setattr(render, "export_jpeg", fail)
    with pytest.raises(OSError, match="preview disk failure"):
        render.render_artifact(source, destination)
    assert (destination / "master.tif").exists()
    assert not render.validate_render(source, destination)
    with pytest.raises(FileExistsError):
        render.render_artifact(source, destination)


def test_recomposition_undoes_quality_weight_and_rejects_missing_boundary(source):
    from gomoveshift.pipeline.photos.astro_linear import load_linear_artifact

    artifact = load_linear_artifact(source)
    artifact.manifest["sky_weight"] = 0.5
    from dataclasses import replace

    artifact = replace(artifact, sky_count=artifact.sky_count * 0.5)
    assert np.allclose(render.recompose_chunk(artifact), 0.1)
    del artifact.manifest["mask_provenance"]["resolved_boundary_row"]
    with pytest.raises(ValueError, match="boundary"):
        render.recompose_chunk(artifact)


def test_uncovered_sky_keeps_real_ground_samples(source):
    from dataclasses import replace

    from gomoveshift.pipeline.photos.astro_linear import load_linear_artifact

    artifact = load_linear_artifact(source)
    artifact = replace(artifact, sky_count=np.zeros((12, 16), np.float32))
    assert np.allclose(render.recompose_chunk(artifact), artifact.ground)


@pytest.mark.parametrize("boundary", [float("nan"), -1, 13])
def test_invalid_horizon_refuses_render(source, boundary):
    from gomoveshift.pipeline.photos.astro_linear import load_linear_artifact

    artifact = load_linear_artifact(source)
    artifact.manifest["mask_provenance"]["resolved_boundary_row"] = boundary
    with pytest.raises(ValueError):
        render.recompose_chunk(artifact)


def test_tampered_payload_refuses_before_export(source, tmp_path):
    path = source / "ground.npy"
    path.write_bytes(path.read_bytes() + b"tampered")
    destination = tmp_path / "attempt"
    with pytest.raises(ValueError, match="sha256"):
        render.render_artifact(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("filename", ["master.tif", "preview.jpg"])
def test_matching_hash_does_not_make_broken_image_reusable(source, tmp_path, filename):
    destination = tmp_path / "attempt"
    receipt = render.render_artifact(source, destination)
    (destination / filename).write_bytes(b"not an image")
    receipt["outputs"][filename]["sha256"] = hashlib.sha256(b"not an image").hexdigest()
    (destination / "receipt.json").write_text(json.dumps(receipt))
    assert not render.validate_render(source, destination)


def test_changing_build_during_render_refuses_receipt(source, tmp_path, monkeypatch):
    builds = iter([{"version": "old"}, {"version": "new"}])
    monkeypatch.setattr(render, "_build", lambda: next(builds))
    destination = tmp_path / "attempt"
    with pytest.raises(ValueError, match="build changed"):
        render.render_artifact(source, destination)
    assert not (destination / "receipt.json").exists()


def test_receipt_fsync_failure_never_publishes_success(source, tmp_path, monkeypatch):
    destination = tmp_path / "attempt"

    def fail(_fd):
        raise OSError("receipt flush failed")

    monkeypatch.setattr(render.os, "fsync", fail)
    with pytest.raises(OSError, match="receipt flush failed"):
        render.render_artifact(source, destination)
    assert not (destination / "receipt.json").exists()
    assert not render.validate_render(source, destination)


def test_matching_hash_wrong_image_dimensions_not_reusable(source, tmp_path):
    destination = tmp_path / "attempt"
    receipt = render.render_artifact(source, destination)
    path = destination / "preview.jpg"
    with Image.open(path) as image:
        profile = image.info["icc_profile"]
    Image.new("RGB", (2, 2)).save(path, icc_profile=profile)
    receipt["outputs"]["preview.jpg"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (destination / "receipt.json").write_text(json.dumps(receipt))
    assert not render.validate_render(source, destination)


def test_failed_preview_recovers_existing_master_without_redevelopment(
    source, tmp_path, monkeypatch
):
    destination = tmp_path / "attempt"
    original_export = render.export_jpeg
    monkeypatch.setattr(
        render, "export_jpeg", lambda *a, **k: (_ for _ in ()).throw(OSError("full"))
    )
    with pytest.raises(OSError, match="full"):
        render.render_artifact(source, destination)
    pending = json.loads((destination / "preview-pending.json").read_text())
    assert pending["state"] == "preview_pending"
    before = (destination / "master.tif").read_bytes()
    monkeypatch.setattr(render, "export_jpeg", original_export)

    def forbidden(*args, **kwargs):
        raise AssertionError("must reuse developed master")

    monkeypatch.setattr(render, "render_linear", forbidden)
    monkeypatch.setattr(render, "recompose_chunk", forbidden)
    assert render.recover_preview(source, destination)
    assert (destination / "master.tif").read_bytes() == before
    assert render.validate_render(source, destination)


@pytest.mark.parametrize("corrupt", ["master", "source"])
def test_preview_recovery_refuses_corrupt_inputs_without_overwrite(
    source, tmp_path, monkeypatch, corrupt
):
    destination = tmp_path / "attempt"
    monkeypatch.setattr(
        render, "export_jpeg", lambda *a, **k: (_ for _ in ()).throw(OSError("full"))
    )
    with pytest.raises(OSError):
        render.render_artifact(source, destination)
    target = destination / "master.tif" if corrupt == "master" else source / "ground.npy"
    target.write_bytes(target.read_bytes() + b"corrupt")
    before = target.read_bytes()
    assert not render.recover_preview(source, destination)
    assert target.read_bytes() == before
    assert not (destination / "receipt.json").exists()


def test_corrupt_master_export_cannot_publish_completion(source, tmp_path, monkeypatch):
    destination = tmp_path / "attempt"

    def corrupt_export(rgb, path):
        path.write_bytes(b"truncated TIFF")

    monkeypatch.setattr(render, "export_tiff16", corrupt_export)
    with pytest.raises(ValueError):
        render.render_artifact(source, destination)
    assert not (destination / "receipt.json").exists()
    assert not (destination / "preview-pending.json").exists()
