"""Selected night stills pass existing privacy gates without mutating their master."""

import hashlib
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.day_master_encoder import REQUIRED_PRIVACY_CLASSES
from gomoveshift.pipeline.photos.develop import export_jpeg
from gomoveshift.pipeline.photos import night_private_export as module


def _row(tmp_path):
    project = "a" * 64
    attempt = tmp_path / project / "0-best_stills-attempt"
    attempt.mkdir(parents=True)
    preview = export_jpeg(np.full((8, 10, 3), 80, dtype=np.uint8), attempt / "preview.jpg")
    return (
        {
            "schema": 2,
            "kind": "night_products",
            "id": project,
            "brand": "brand",
            "build": {"night_products.py": "build-sha"},
            "runtime": {"python": "3.12"},
            "tasks": [
                {
                    "product": "best_stills",
                    "status": "complete",
                    "output": str(attempt),
                }
            ],
        },
        attempt,
        preview,
    )


def _receipt(preview):
    return {
        "outputs": [
            {
                "path": preview.name,
                "role": "preview",
                "size": preview.stat().st_size,
                "sha256": hashlib.sha256(preview.read_bytes()).hexdigest(),
            }
        ]
    }


def _privacy(state="verified"):
    return {name: {"state": state, "reason": "test"} for name in REQUIRED_PRIVACY_CLASSES}


def test_private_export_blurs_copy_strips_exif_and_reuses_receipt(tmp_path):
    row, attempt, preview = _row(tmp_path)
    calls = []

    def blur(pixels, label, blur):
        calls.append((pixels.shape, label, blur))
        return np.clip(pixels + 1, 0, 255).astype(np.uint8), 2

    kwargs = {
        "receipt_validator": lambda *_args: _receipt(preview),
        "privacy_reporter": lambda **_kwargs: _privacy(),
        "blur_filter": blur,
    }
    first = module.prepare_private_export(row, 0, tmp_path, **kwargs)
    second = module.prepare_private_export(row, 0, tmp_path, **kwargs)

    assert first == second
    assert len(calls) == 1
    assert first["privacy_regions"] == 2
    assert first["gps_exif_absent"] is True
    assert first["publishable"] is False
    output = tmp_path / row["id"] / first["path"]
    with Image.open(output) as image:
        image.load()
        assert image.format == "JPEG"
        assert image.info.get("icc_profile")
        assert 34853 not in image.getexif()
    assert preview.parent == attempt
    assert preview.exists()
    assert module.validate_private_export(row, first, tmp_path) == output
    output.write_bytes(b"tampered")
    assert module.validate_private_export(row, first, tmp_path) is None


def test_private_export_refuses_unattested_class_before_pixel_detection(tmp_path):
    row, _attempt, preview = _row(tmp_path)
    blur = Mock()
    privacy = _privacy()
    privacy["people"] = {"state": "inconclusive", "reason": "model disabled"}

    with pytest.raises(ValueError, match="people"):
        module.prepare_private_export(
            row,
            0,
            tmp_path,
            receipt_validator=lambda *_args: _receipt(preview),
            privacy_reporter=lambda **_kwargs: privacy,
            blur_filter=blur,
        )

    blur.assert_not_called()


def test_private_export_refuses_video_without_a_still_preview(tmp_path):
    row, _attempt, _preview = _row(tmp_path)
    row["tasks"][0]["product"] = "timelapse"

    with pytest.raises(ValueError, match="still preview"):
        module.prepare_private_export(
            row,
            0,
            tmp_path,
            receipt_validator=lambda *_args: {"outputs": [{"path": "night.mp4", "role": "video"}]},
            privacy_reporter=lambda **_kwargs: _privacy(),
        )
