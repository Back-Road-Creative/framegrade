"""Every still the film composites or exports is blurred + EXIF-stripped
into a working copy BEFORE it is used -- the film never reads an original
photo path past the gate. See ``docs/pipeline/125-route-story-privacy-gate.md``.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gomoveshift.pipeline.photos import face_filter
from gomoveshift.pipeline.photos.face_filter import FaceBlurError
from gomoveshift.pipeline.route_story.anchors import MediaAnchor
from gomoveshift.pipeline.route_story.film import PhotoMoment
from gomoveshift.pipeline.route_story.privacy import apply_privacy_gate
from gomoveshift.publishing.dispatch import PRIVACY_REQUIRED_CLASSES, privacy_hold_reason

PATCH = (10, 10, 20, 20)
GPS_IFD_TAG = 0x8825


def _gps_exif() -> Image.Exif:
    """A minimal GPS EXIF block, mirroring ``tests/test_photo_resizer.py``'s
    own fixture -- no piexif dependency needed."""
    exif = Image.Exif()
    exif[GPS_IFD_TAG] = {1: "N", 2: (41.0, 0.0, 0.0), 3: "W", 4: (71.0, 0.0, 0.0)}
    return exif


class _PatchFinder:
    """Fake detector: reports a fixed box for every frame (no model dep)."""

    def detect(self, frame, conf):
        return [PATCH]


class _EmptyFinder:
    def detect(self, frame, conf):
        return []


class _Broken:
    def detect(self, frame, conf):
        raise RuntimeError("onnxruntime session died")


def _photo_with_gps(tmp_path: Path, name: str = "a.jpg") -> PhotoMoment:
    """A real JPEG carrying GPS EXIF, so the gate's strip is a real assertion."""
    path = tmp_path / name
    Image.fromarray(np.full((60, 60, 3), 90, dtype=np.uint8)).save(path, "JPEG", exif=_gps_exif())

    anchor = MediaAnchor(name, "mapped", 12.0, 41.0, -71.0)
    return PhotoMoment(path, anchor, captured_at_exif_s=12.0)


def test_default_run_attestation_clears_the_publisher_hold(tmp_path, monkeypatch):
    """Blur on (default): every required class resolves verified, and the
    publisher's own hold gate reads it as clear -- no translation needed."""
    monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_EmptyFinder(), _EmptyFinder()))
    monkeypatch.setattr(
        face_filter, "resolve_face_model", lambda d: tmp_path / "face.onnx", raising=False
    )
    monkeypatch.setattr(
        face_filter, "resolve_plate_model", lambda d: tmp_path / "plate.onnx", raising=False
    )
    monkeypatch.setattr(
        face_filter, "resolve_vehicle_model", lambda d: tmp_path / "vehicle.onnx", raising=False
    )
    monkeypatch.setattr(
        face_filter, "resolve_person_model", lambda d: tmp_path / "person.onnx", raising=False
    )
    for fname in ("face.onnx", "plate.onnx", "vehicle.onnx", "person.onnx"):
        (tmp_path / fname).write_bytes(b"x")

    photo = _photo_with_gps(tmp_path)
    result = apply_privacy_gate([photo], blur=None, work_dir=tmp_path / "work")

    assert privacy_hold_reason(result.attestation) is None
    assert set(result.attestation["classes"]) == set(PRIVACY_REQUIRED_CLASSES)


def test_skip_blur_every_class_inconclusive_and_held(tmp_path, monkeypatch):
    """``--skip-blur`` means nothing was asked -- every class inconclusive,
    and the publisher hold gate reads a non-None reason."""
    monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_EmptyFinder(), _EmptyFinder()))

    photo = _photo_with_gps(tmp_path)
    result = apply_privacy_gate([photo], blur=False, work_dir=tmp_path / "work")

    for cls in PRIVACY_REQUIRED_CLASSES:
        assert result.attestation["classes"][cls]["state"] == "inconclusive"
    assert privacy_hold_reason(result.attestation) is not None


def test_working_copy_has_no_gps_while_source_still_does(tmp_path, monkeypatch):
    monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_EmptyFinder(), _EmptyFinder()))

    photo = _photo_with_gps(tmp_path)
    result = apply_privacy_gate([photo], blur=None, work_dir=tmp_path / "work")

    gated_path = Path(result.photos[0].source)
    assert gated_path != Path(photo.source)
    with Image.open(photo.source) as src_img:
        assert GPS_IFD_TAG in src_img.getexif()  # source untouched, still carries GPS

    with Image.open(gated_path) as out_img:
        assert GPS_IFD_TAG not in out_img.getexif()


def test_gated_photo_path_is_under_work_dir_never_the_original(tmp_path, monkeypatch):
    """The film must composite the WORKING copy -- never re-read the
    caller's original photo path."""
    monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_EmptyFinder(), _EmptyFinder()))
    work_dir = tmp_path / "work"

    photo = _photo_with_gps(tmp_path)
    result = apply_privacy_gate([photo], blur=None, work_dir=work_dir)

    gated_path = Path(result.photos[0].source)
    assert work_dir in gated_path.parents
    assert not gated_path.samefile(photo.source)


def test_detector_failure_without_skip_blur_raises_before_any_render(tmp_path, monkeypatch):
    """A default run whose detector cannot run must hard-fail -- never a
    silent unblurred film."""
    monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_Broken(), None))

    photo = _photo_with_gps(tmp_path)
    with pytest.raises(FaceBlurError, match="--skip-blur"):
        apply_privacy_gate([photo], blur=None, work_dir=tmp_path / "work")
