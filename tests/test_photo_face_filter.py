"""Tests for face_filter — single-image detection, blur-at-resize, selector integration."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest

import redactcam
from redactcam import YoloDetector
from redactcam import detect as redactcam_detect
from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.photos import face_filter
from gomoveshift.pipeline.photos.face_filter import FaceBlurError, should_withhold_photo
from gomoveshift.pipeline.photos.resizer import PhotoResizer

PATCH = (60, 40, 50, 50)  # x, y, w, h of the synthetic "face"


class _PatchFinder:
    """Fake detector (no model dep): reports the known patch when it is present."""

    def detect(self, frame, conf):
        return [PATCH] if (frame[:, :, 0] > 200).any() else []


def _make_solid_image(tmp_path: Path, name: str = "landscape.jpg", size=(200, 300)) -> Path:
    """Create a solid-color test image (no faces)."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    img[:] = (120, 180, 80)  # Green-ish landscape color
    path = tmp_path / name
    cv2.imwrite(str(path), img)
    return path


def _face_photo(tmp_path, name="photo.png"):
    """Photo with a STRIPED bright patch — stripes so a real blur must change pixels."""
    img = np.full((200, 200, 3), 90, dtype=np.uint8)
    x, y, w, h = PATCH
    img[y : y + h, x : x + w], img[y : y + h : 2, x : x + w] = 255, 205
    cv2.imwrite(str(path := tmp_path / name), img)
    return path


def test_detect_image_finds_face_and_reuses_sessions(tmp_path, monkeypatch):
    """Single-image entry point: no VideoCapture, one handle reused across a batch."""
    monkeypatch.delattr(cv2, "VideoCapture")  # any video decode would AttributeError
    builds = []

    def _build(*_a, **_k):
        builds.append(1)
        return (_PatchFinder(), None)

    monkeypatch.setattr(redactcam_detect, "build_detectors", _build)
    frame = cv2.imread(str(_face_photo(tmp_path)))

    detectors = redactcam.build_image_detectors()
    for _ in range(3):
        regions = redactcam.detect_image(frame, detectors=detectors)
    assert regions.faces == [PATCH] and regions.boxes == regions.faces
    assert len(builds) == 1  # reused, not rebuilt per image
    redactcam.detect_image(frame)
    assert len(builds) == 2  # a call with no handle builds its own


class _Broken:
    """Detector whose session has died mid-batch."""

    def detect(self, frame, conf):
        raise RuntimeError("onnxruntime session died")


def _patch_std(path: Path) -> float:
    """Stripe contrast in the CORE of the detected patch — collapses towards 0 once
    blurred. Sampled inside the mask feather, where alpha is fully 1."""
    x, y, w, h = PATCH
    core = cv2.imread(str(path))[y + h // 3 : y + 2 * h // 3, x + w // 3 : x + 2 * w // 3]
    return float(core.std())


def _resize(tmp_path, src: Path):
    out = PhotoResizer(target_width=1080, output_base=tmp_path / "resized").resize_batch(
        [str(src)], "2026-08-01"
    )
    return out[0]


def test_source_photo_is_never_modified(tmp_path, monkeypatch):
    """REGRESSION GUARD: blur happens on the resized copy — the source photo in
    the site checkout must come out of a run byte-for-byte identical."""
    src = _face_photo(tmp_path)
    before = src.read_bytes()
    monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_PatchFinder(), None))

    _resize(tmp_path, src)

    assert src.read_bytes() == before


def test_resized_output_is_blurred(tmp_path, monkeypatch):
    """The artifact that actually gets uploaded carries the blur."""
    src = _face_photo(tmp_path)
    monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_PatchFinder(), None))

    out = _resize(tmp_path, src)

    assert _patch_std(out) < _patch_std(src) / 2


def test_skip_blur_leaves_output_unblurred(tmp_path, monkeypatch):
    """Operator opt-out reaches the blur — pixels ship as shot."""
    src = _face_photo(tmp_path)
    monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_PatchFinder(), None))
    monkeypatch.setattr(face_filter, "_blur_enabled", False)

    out = _resize(tmp_path, src)

    assert _patch_std(out) > _patch_std(src) / 2


def test_detector_error_raises_instead_of_shipping(tmp_path, monkeypatch):
    """A detector failure must never read as 'no faces, ship it'."""
    monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_Broken(), None))
    with pytest.raises(FaceBlurError):
        _resize(tmp_path, _face_photo(tmp_path))


class TestShouldWithholdPhoto:
    """Contract of the renamed selection gate: True means WITHHOLD."""

    def test_face_is_no_reason_to_withhold(self, tmp_path, monkeypatch):
        """Blur on (default): the photo ships and its pixels are untouched here."""
        path = _face_photo(tmp_path)
        before = path.read_bytes()
        monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_PatchFinder(), None))

        assert should_withhold_photo(str(path)) is False
        assert path.read_bytes() == before

    def test_skip_blur_falls_back_to_withholding(self, tmp_path, monkeypatch):
        """With the blur opted out, a detected face is withheld rather than shipped."""
        path = _face_photo(tmp_path)
        monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_PatchFinder(), None))
        assert should_withhold_photo(str(path), blur=False) is True

    def test_detector_error_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_Broken(), None))
        with pytest.raises(FaceBlurError):
            should_withhold_photo(str(_face_photo(tmp_path)), blur=False)

    def test_no_faces_in_landscape(self, tmp_path):
        """Solid-color image should have no detected faces."""
        img_path = _make_solid_image(tmp_path)
        assert should_withhold_photo(str(img_path)) is False

    def test_missing_file(self, tmp_path):
        """Missing file should return False with warning."""
        result = should_withhold_photo(str(tmp_path / "does_not_exist.jpg"))
        assert result is False

    def test_unreadable_image(self, tmp_path):
        """Corrupt/unreadable file cannot be blurred, so it is WITHHELD."""
        bad_file = tmp_path / "corrupt.jpg"
        bad_file.write_bytes(b"not an image")
        assert should_withhold_photo(str(bad_file)) is True


class TestSelectorFaceFiltering:
    """Integration tests for face filtering in PhotoSelector."""

    def _make_journey_fixtures(self, tmp_path):
        """Create minimal journey fixtures for selector testing."""
        site_root = tmp_path / "site"
        content_dir = site_root / "content" / "journeys"
        content_dir.mkdir(parents=True)
        images_dir = site_root / "static" / "images" / "journeys" / "test-journey"
        images_dir.mkdir(parents=True)

        # Create journey markdown with frontmatter
        md_content = """---
title: Test Journey
days:
  - day_number: 1
    date: "2024-01-01"
    summary: "Day one"
    locations: ["Place A"]
    photos:
      - location: "Place A"
        photos:
          - "photo-001.jpg"
          - "photo-002.jpg"
          - "photo-003.jpg"
---
Content here.
"""
        (content_dir / "test-journey.md").write_text(md_content)

        # Create actual image files
        for i, name in enumerate(["photo-001.jpg", "photo-002.jpg", "photo-003.jpg"]):
            # Sharp, distinct frames: a flat black frame would be cull-skipped.
            img = np.random.default_rng(i).integers(0, 256, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(str(images_dir / name), img)

        return site_root

    def test_selector_filter_faces_skips(self, tmp_path):
        """Photos with faces should be skipped during selection."""
        from gomoveshift.pipeline.photos.selector import PhotoSelector

        site_root = self._make_journey_fixtures(tmp_path)
        state_path = tmp_path / "state.json"

        selector = PhotoSelector(site_root=site_root, state_path=state_path)

        # Mock the gate: photo-002 is withheld, others are not
        def mock_withhold(path, **kwargs):
            return "photo-002" in path

        with patch("gomoveshift.pipeline.photos.selector._should_withhold", mock_withhold):
            with patch("gomoveshift.pipeline.photos.selector._FACE_FILTER_AVAILABLE", True):
                # count=2: photo-002 is filtered out, so 001 + 003 fill the batch
                selections = selector.select_batch(count=2, filter_faces=True)

        filenames = [s.filename for s in selections]
        assert "photo-002.jpg" not in filenames
        assert "photo-001.jpg" in filenames
        assert "photo-003.jpg" in filenames

    def test_selector_face_skip_not_persisted(self, tmp_path):
        """Face-skipped photos should NOT be saved as 'posted' in state."""
        from gomoveshift.pipeline.photos.selector import PhotoSelector

        site_root = self._make_journey_fixtures(tmp_path)
        state_path = tmp_path / "state.json"

        selector = PhotoSelector(site_root=site_root, state_path=state_path)

        # Mock: photo-002 is withheld
        def mock_withhold(path, **kwargs):
            return "photo-002" in path

        with patch("gomoveshift.pipeline.photos.selector._should_withhold", mock_withhold):
            with patch("gomoveshift.pipeline.photos.selector._FACE_FILTER_AVAILABLE", True):
                selector.select_batch(count=2, filter_faces=True)

        # Read persisted state
        state_data = json.loads(state_path.read_text())
        posted = state_data["posted_photos"]

        # photo-002 was face-skipped — it should NOT appear in posted_photos
        assert "test-journey/photo-002.jpg" not in posted
        # photo-001 and photo-003 were actually selected — they should be posted
        assert "test-journey/photo-001.jpg" in posted
        assert "test-journey/photo-003.jpg" in posted


PLATE = (30, 120, 60, 20)  # 3:1 — a real plate's aspect
VEHICLE = (10, 60, 110, 100)  # contains PLATE's center → plate-corroborated
BILLBOARD = (200, 20, 60, 60)  # COCO "vehicle" misfire with no plate inside
CABIN = (10, 60, 110, 60)  # top blur_vehicle_cabin_frac (default) of VEHICLE


def _still(box=None):
    """400×400 RGB frame, optionally striped inside ``box`` so a blur must change pixels."""
    img = np.full((400, 400, 3), 90, dtype=np.uint8)
    if box is not None:
        x, y, w, h = box
        img[y : y + h, x : x + w], img[y : y + h : 2, x : x + w] = 255, 205
    return img


def _core_std(frame, box):
    """Stripe contrast in the CORE of ``box`` — collapses towards 0 once blurred."""
    x, y, w, h = box
    return float(frame[y + h // 3 : y + 2 * h // 3, x + w // 3 : x + 2 * w // 3].std())


class _Boxes:
    """Fake detector returning fixed boxes for any frame."""

    def __init__(self, boxes):
        self._boxes = list(boxes)

    def detect(self, frame, conf):
        return list(self._boxes)


def _yolo_plate(anchors, min_aspect=0.0, max_aspect=0.0):
    """REAL ``YoloDetector`` decode + aspect band over a crafted output tensor
    (no model file) — the same construction trick redactcam's own detect suite uses."""
    arr = np.zeros((1, 5, 8), dtype=np.float32)
    arr[0, 4, :] = 0.01  # every anchor below conf...
    for i, (cx, cy, w, h) in enumerate(anchors):
        arr[0, :, i] = [cx, cy, w, h, 0.9]  # ...except the crafted ones
    det = YoloDetector.__new__(YoloDetector)  # bypass __init__ (no real model)
    det._sess = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images")], run=lambda _n, _f: [arr]
    )
    det._input_name = "images"
    det._infer_size = 640
    det._iou = 0.45
    det._min_aspect = min_aspect
    det._max_aspect = max_aspect
    det._class_filter = None
    return det


class TestStillPlateAndOccupantBlur:
    """Stills carry the video blur contract: faces ∪ plates ∪ plate-corroborated cabins."""

    def test_plate_reaches_the_still_blur(self, monkeypatch):
        """A detected plate is blurred on the published copy — faces-only consumption
        (the still path reading only ``.faces``) shipped every plate readable."""
        monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_Boxes([]), _Boxes([PLATE])))
        frame = _still(PLATE)
        out, blurred = face_filter.blur_faces_for_publish(frame, "still.jpg")
        assert blurred == 1
        assert _core_std(out, PLATE) < _core_std(frame, PLATE) / 2

    def test_sign_shaped_box_dropped_by_the_aspect_band(self, monkeypatch):
        """The video plate aspect band rides along: a squarish road sign stays readable
        while a wide real plate blurs — aspect, not confidence, separates them."""
        drive = DriveConfig()
        sign, plate = (100, 100, 40, 25), (100, 200, 60, 20)  # 1.6:1 sign vs 3:1 plate
        det = _yolo_plate(
            [(192.0, 180.0, 64.0, 40.0), (208.0, 336.0, 96.0, 32.0)],  # 400→640 letterbox coords
            min_aspect=drive.blur_plate_min_aspect,
            max_aspect=drive.blur_plate_max_aspect,
        )
        monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_Boxes([]), det))
        regions = face_filter._detect_regions(_still(), "still.jpg")
        assert plate in regions and sign not in regions

    def test_cabin_of_plate_corroborated_vehicle_included(self, monkeypatch):
        """Occupants can't be face-detected through a windshield: the top cabin band of
        a plate-corroborated vehicle joins the blur; a plate-less 'vehicle' (billboard/
        sign the COCO model misreads) contributes nothing."""
        monkeypatch.setattr(face_filter, "_drive_cfg", DriveConfig(), raising=False)
        monkeypatch.setattr(
            face_filter,
            "_get_detectors",
            lambda: (_Boxes([]), _Boxes([PLATE]), _Boxes([VEHICLE, BILLBOARD])),
        )
        regions = face_filter._detect_regions(_still(PLATE), "still.jpg")
        assert CABIN in regions
        assert regions.count(PLATE) == 1 and len(regions) == 2  # nothing from the billboard

    def test_no_plates_skips_the_vehicle_pass(self, monkeypatch):
        """No plate → nothing can corroborate a cabin; the vehicle detector never runs."""

        class _MustNotRun:
            def detect(self, frame, conf):
                raise AssertionError("vehicle detector consulted with no plates")

        monkeypatch.setattr(face_filter, "_drive_cfg", DriveConfig(), raising=False)
        monkeypatch.setattr(
            face_filter, "_get_detectors", lambda: (_Boxes([]), _Boxes([]), _MustNotRun())
        )
        assert face_filter._detect_regions(_still(), "still.jpg") == []

    def test_person_reaches_the_still_blur(self, monkeypatch):
        """A whole-person box (back turned, no detectable face) is blurred on stills
        too — the still path previously carried NO whole-person detector at all, so a
        photographed person who never faced the camera shipped completely unblurred."""
        person = (250, 250, 60, 120)
        monkeypatch.setattr(face_filter, "_drive_cfg", DriveConfig(), raising=False)
        monkeypatch.setattr(
            face_filter,
            "_get_detectors",
            lambda: (_Boxes([]), _Boxes([]), _Boxes([]), _Boxes([person])),
        )
        frame = _still(person)
        out, blurred = face_filter.blur_faces_for_publish(frame, "still.jpg")
        assert blurred == 1
        assert _core_std(out, person) < _core_std(frame, person) / 2

    def test_detectors_wired_from_config_and_model_resolvers(self, monkeypatch):
        """The still builder resolves the video contract's models (download-cached)
        and plate gates from config instead of hardcoding plate_model_path=None.

        Extended for the whole-person detector: person resolution/build is
        mocked exactly like vehicle's, and the tuple now carries a 4th slot."""
        drive = DriveConfig()
        built, vehicle_built, person_built = {}, {}, {}
        monkeypatch.setattr(face_filter, "_detectors", None)
        monkeypatch.setattr(face_filter, "_drive_cfg", drive, raising=False)
        monkeypatch.setattr(
            face_filter, "resolve_plate_model", lambda d: Path("/m/plate.onnx"), raising=False
        )
        monkeypatch.setattr(
            face_filter, "resolve_face_model", lambda d: Path("/m/face.onnx"), raising=False
        )
        monkeypatch.setattr(
            face_filter, "resolve_vehicle_model", lambda d: Path("/m/vehicle.onnx"), raising=False
        )
        monkeypatch.setattr(
            face_filter, "resolve_person_model", lambda d: Path("/m/person.onnx"), raising=False
        )

        def _fake_build(**kwargs):
            built.update(kwargs)
            return (_Boxes([]), _Boxes([]))

        def _fake_vehicle(path, *args, **kwargs):
            vehicle_built["path"] = path
            return _Boxes([])

        def _fake_person(path, *args, **kwargs):
            person_built["path"] = path
            return _Boxes([])

        monkeypatch.setattr(face_filter, "build_image_detectors", _fake_build)
        monkeypatch.setattr(face_filter, "build_vehicle_detector", _fake_vehicle)
        monkeypatch.setattr(face_filter, "build_person_detector", _fake_person)

        detectors = face_filter._get_detectors()

        assert built.get("plate_model_path") == Path("/m/plate.onnx")
        assert built.get("face_model_path") == Path("/m/face.onnx")
        assert built.get("plate_min_aspect") == drive.blur_plate_min_aspect
        assert built.get("plate_max_aspect") == drive.blur_plate_max_aspect
        assert vehicle_built.get("path") == Path("/m/vehicle.onnx")
        assert person_built.get("path") == Path("/m/person.onnx")
        assert len(detectors) == 4


# CI trigger
