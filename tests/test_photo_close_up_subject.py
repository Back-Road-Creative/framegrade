"""CLOSE_UP subject: near focus or macro optics, ranked after astro/bracket, before wildlife."""

from pathlib import Path

import pytest

from gomoveshift.pipeline.photos import subject as subject_mod
from gomoveshift.pipeline.photos.scene import PhotoExif
from gomoveshift.pipeline.photos.subject import (
    CLOSE_UP_FOCUS_M_MAX,
    CONFIDENCE_INFERENCE_MAX,
    CONFIDENCE_MODERATE,
    CONFIDENCE_SEQUENCE_BONUS,
    CONFIDENCE_WEAK,
    MACRO_MAGNIFICATION_MIN,
    CaptureOptics,
    SubjectCategory,
    read_subject_correction,
    recognise_photos,
    recognise_subject,
    write_subject_correction,
)

DAY = PhotoExif(iso=100, exposure_s=1 / 500, capture_hour=12, solar_elevation_deg=42.0)
NIGHT = PhotoExif(iso=3200, exposure_s=20.0, capture_hour=23, solar_elevation_deg=-31.0)
BOOST = min(CONFIDENCE_MODERATE + CONFIDENCE_SEQUENCE_BONUS, CONFIDENCE_INFERENCE_MAX)


def _read(focus, lens=None, focal=50.0, exif=DAY, **kw):
    optics = CaptureOptics(focal_length_mm=focal, focus_distance_m=focus, lens_model=lens)
    return recognise_subject(exif=exif, optics=optics, **kw)


def test_enum_member_value():
    assert SubjectCategory("close_up") is SubjectCategory.CLOSE_UP


def test_focus_1_2m_is_close_up():
    r = _read(1.2)
    assert r.category is SubjectCategory.CLOSE_UP
    assert r.confidence == CONFIDENCE_MODERATE
    assert r.evidence == ("focus distance 1.2 m <= 3.0 m",)


def test_boundary_3m_is_close_up():
    assert _read(CLOSE_UP_FOCUS_M_MAX).category is SubjectCategory.CLOSE_UP


@pytest.mark.parametrize("focus", [3.01, float("inf"), float("nan"), 0.0, -1.0, None])
def test_not_close_up_via_focus(focus):
    assert _read(focus).category is not SubjectCategory.CLOSE_UP


def test_macro_lens_near_focus_is_higher_confidence():
    r = _read(0.5, lens="AF-S VR Micro-NIKKOR 105mm f/2.8G IF-ED (Macro)", focal=105.0)
    assert r.category is SubjectCategory.CLOSE_UP
    assert r.confidence == BOOST > CONFIDENCE_MODERATE
    assert any("macro" in e.lower() for e in r.evidence)


def test_macro_name_far_focus_is_not_close_up():
    assert _read(20.0, lens="Some Macro 90mm", focal=90.0).category is not SubjectCategory.CLOSE_UP


def test_macro_name_without_focus_is_weak_close_up():
    r = _read(None, lens="Some Macro 90mm", focal=90.0)
    assert r.category is SubjectCategory.CLOSE_UP
    assert r.confidence == CONFIDENCE_WEAK


def test_magnification_without_macro_name_is_close_up_boosted():
    # f/(d-f) = 0.1/(0.5-0.1) = 0.25
    assert MACRO_MAGNIFICATION_MIN == 0.25
    r = _read(0.5, lens="Plain 100mm", focal=100.0)
    assert r.category is SubjectCategory.CLOSE_UP
    assert r.confidence == BOOST


def test_focus_inside_focal_length_has_no_magnification():
    # d <= f: magnification undefined, near-focus rule still answers
    r = _read(0.05, focal=100.0)
    assert r.category is SubjectCategory.CLOSE_UP
    assert r.confidence == CONFIDENCE_MODERATE


def test_night_astro_near_focus_stays_astro():
    assert _read(1.0, exif=NIGHT).category is SubjectCategory.ASTRO


def test_bracket_member_near_focus_stays_bracket():
    r = _read(1.0, in_exposure_bracket=True)
    assert r.category is SubjectCategory.EXPOSURE_BRACKET


def test_operator_correction_wins():
    r = _read(1.0, correction=SubjectCategory.LANDSCAPE)
    assert r.category is SubjectCategory.LANDSCAPE


def test_250mm_at_2m_is_close_up_not_wildlife():
    assert _read(2.0, focal=250.0).category is SubjectCategory.CLOSE_UP


def test_250mm_far_is_still_wildlife():
    assert _read(50.0, focal=250.0).category is SubjectCategory.WILDLIFE


def test_recognise_photos_uses_optics_mapping(tmp_path, monkeypatch):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"")
    other = tmp_path / "b.jpg"
    other.write_bytes(b"")
    monkeypatch.setattr(subject_mod, "extract_exif_capture_metadata", lambda p: DAY)
    monkeypatch.setattr(subject_mod, "extract_exif_capture_timestamp", lambda p: None)
    monkeypatch.setattr(subject_mod, "detect_brackets", lambda paths: [])
    monkeypatch.setattr(subject_mod, "read_capture_optics", lambda p: CaptureOptics())
    readings = recognise_photos(
        [photo, other], optics={Path(photo): CaptureOptics(focus_distance_m=1.0)}
    )
    assert readings[photo].category is SubjectCategory.CLOSE_UP
    assert readings[other].category is SubjectCategory.UNKNOWN


def test_sidecar_round_trip(tmp_path):
    photo = tmp_path / "a.jpg"
    photo.write_bytes(b"")
    sidecar = write_subject_correction(photo, SubjectCategory.CLOSE_UP)
    assert '"close_up"' in sidecar.read_text()
    assert read_subject_correction(photo) is SubjectCategory.CLOSE_UP
