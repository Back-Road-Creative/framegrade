"""Stills record WHICH privacy classes a run could actually attest, mirroring
the video attestation vocabulary (day_master_encoder) instead of collapsing
four separate facts into one blur-state string plus silence.

The question every case here asks is the same one ``_detector_class`` asks on
the video side: did the model this run would actually use resolve? "Blur was
switched on" is not an answer to it.
"""

import pytest

from gomoveshift.config import DriveConfig
from gomoveshift.pipeline.day_master_encoder import (
    PRIVACY_INCONCLUSIVE,
    PRIVACY_VERIFIED,
    REQUIRED_PRIVACY_CLASSES,
)
from gomoveshift.pipeline.photos import face_filter

_RESOLVERS = {
    "face": "resolve_face_model",
    "plate": "resolve_plate_model",
    "vehicle": "resolve_vehicle_model",
    "person": "resolve_person_model",
}


@pytest.fixture
def models(monkeypatch, tmp_path):
    """Point every detector resolver at a real file by default; a test sets
    one to ``None`` to mean "this model did not resolve"."""
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"not a real model, only a resolvable path")
    resolved = {name: model_file for name in _RESOLVERS}

    for name, attr in _RESOLVERS.items():
        monkeypatch.setattr(face_filter, attr, lambda _drive, _n=name: resolved[_n], raising=True)
    return resolved


def _use(monkeypatch, drive: DriveConfig) -> None:
    monkeypatch.setattr(face_filter, "_drive_cfg", drive, raising=False)


def test_fully_resolved_all_verified(monkeypatch, models):
    _use(monkeypatch, DriveConfig())

    result = face_filter.attestable_privacy_classes()

    assert set(result) == set(REQUIRED_PRIVACY_CLASSES)
    for cls in REQUIRED_PRIVACY_CLASSES:
        assert result[cls]["state"] == PRIVACY_VERIFIED


def test_vehicle_cabin_disabled_by_config_is_inconclusive_with_reason(monkeypatch, models):
    drive = DriveConfig()
    drive.blur_vehicle_cabin_frac = 0.0
    _use(monkeypatch, drive)

    result = face_filter.attestable_privacy_classes()

    assert result["vehicle_occupants"]["state"] == PRIVACY_INCONCLUSIVE
    assert result["vehicle_occupants"]["reason"]  # non-empty, names the cause
    # Config-off is NOT "not required" — and it says nothing about the others.
    assert result["faces"]["state"] == PRIVACY_VERIFIED
    assert result["plates"]["state"] == PRIVACY_VERIFIED


def test_person_blur_disabled_by_config_is_inconclusive_with_reason(monkeypatch, models):
    drive = DriveConfig()
    drive.blur_person_enabled = False
    _use(monkeypatch, drive)

    result = face_filter.attestable_privacy_classes()

    assert result["people"]["state"] == PRIVACY_INCONCLUSIVE
    assert result["people"]["reason"]


def test_person_model_unresolved_is_inconclusive_with_reason(monkeypatch, models):
    """Enabled-but-unresolvable is not a clean detection: nothing ran."""
    _use(monkeypatch, DriveConfig())
    models["person"] = None

    result = face_filter.attestable_privacy_classes()

    assert result["people"]["state"] == PRIVACY_INCONCLUSIVE
    assert result["people"]["reason"]


def test_vehicle_model_unresolved_is_inconclusive_with_reason(monkeypatch, models):
    _use(monkeypatch, DriveConfig())
    models["vehicle"] = None

    result = face_filter.attestable_privacy_classes()

    assert result["vehicle_occupants"]["state"] == PRIVACY_INCONCLUSIVE
    assert result["vehicle_occupants"]["reason"]


@pytest.mark.parametrize(
    ("missing", "privacy_class"),
    [("face", "faces"), ("plate", "plates")],
)
def test_face_or_plate_model_unresolved_is_never_verified(
    monkeypatch, models, missing, privacy_class
):
    """The regression this guards: faces and plates are the two classes it is
    most tempting to call ``verified`` the moment blur is switched on, because
    they need no config flag. Blur being on says nothing about whether their
    model resolved, and a class nothing ran for must never read as covered."""
    _use(monkeypatch, DriveConfig())
    models[missing] = None

    result = face_filter.attestable_privacy_classes()

    assert result[privacy_class]["state"] == PRIVACY_INCONCLUSIVE
    assert result[privacy_class]["reason"]


def test_skip_blur_all_four_inconclusive(monkeypatch):
    """Nothing was asked, so nothing is known — and no detector is consulted."""
    _use(monkeypatch, DriveConfig())
    monkeypatch.setattr(
        face_filter,
        "_get_detectors",
        lambda: pytest.fail("--skip-blur must never consult a detector"),
    )

    result = face_filter.attestable_privacy_classes(blur=False)

    assert set(result) == set(REQUIRED_PRIVACY_CLASSES)
    for cls in REQUIRED_PRIVACY_CLASSES:
        assert result[cls]["state"] == PRIVACY_INCONCLUSIVE
        assert result[cls]["reason"]


def test_attestation_never_builds_a_detector(monkeypatch, models):
    """Report, never gate. Building detector sessions here would move a
    model-load failure earlier than ``_detect_regions``, where ingest wraps it
    as ``PhotoIngestError`` — letting an attestation decide an outcome."""
    _use(monkeypatch, DriveConfig())
    monkeypatch.setattr(
        face_filter,
        "_get_detectors",
        lambda: pytest.fail("attestation must not build detector sessions"),
    )

    face_filter.attestable_privacy_classes()
