"""Lens-profile correction. A clear, named no-op with the ``lens``
extra absent, with no camera EXIF, or with no matching lensfun profile —
never a guess, never a crash. The lensfun DB lookup and the remap are
isolated behind ``_find_profile``/``_remap`` so every test here runs without
the real package installed.
"""

from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.photos import lens_correct
from gomoveshift.pipeline.photos.develop import DecodedPhoto, DevelopOptions, develop_variants


def _rgb(h=8, w=10):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, : w // 2] = (200, 60, 60)
    return frame


def _decoded(h=8, w=10):
    return DecodedPhoto(
        rgb=_rgb(h, w),
        path=Path("fixture.jpg"),
        decoder="test",
        width=w,
        height=h,
        bit_depth=8,
        is_raw=False,
    )


def test_no_op_when_the_lens_extra_is_absent(monkeypatch):
    monkeypatch.setattr(lens_correct, "LENSFUN_AVAILABLE", False)
    rgb = _rgb()
    result = lens_correct.correct(rgb, Path("photo.jpg"))
    assert result.applied is False
    assert "not installed" in result.reason
    assert np.array_equal(result.rgb, rgb)


def test_no_op_when_the_photo_carries_no_camera_exif(monkeypatch):
    monkeypatch.setattr(lens_correct, "LENSFUN_AVAILABLE", True)
    monkeypatch.setattr(lens_correct, "read_lens_exif", lambda path: lens_correct.LensExif())
    rgb = _rgb()
    result = lens_correct.correct(rgb, Path("photo.jpg"))
    assert result.applied is False
    assert "no camera EXIF" in result.reason
    assert np.array_equal(result.rgb, rgb)


def test_no_op_when_no_lensfun_profile_matches(monkeypatch):
    monkeypatch.setattr(lens_correct, "LENSFUN_AVAILABLE", True)
    exif = lens_correct.LensExif(camera_make="NIKON CORPORATION", camera_model="NIKON Z 50")
    monkeypatch.setattr(lens_correct, "read_lens_exif", lambda path: exif)
    monkeypatch.setattr(lens_correct, "_find_profile", lambda e: None)
    rgb = _rgb()
    result = lens_correct.correct(rgb, Path("photo.jpg"))
    assert result.applied is False
    assert "no lensfun profile" in result.reason
    assert "NIKON" in result.reason
    assert np.array_equal(result.rgb, rgb)


def test_correct_applies_the_matched_profiles_remap(monkeypatch):
    monkeypatch.setattr(lens_correct, "LENSFUN_AVAILABLE", True)
    exif = lens_correct.LensExif(
        camera_make="NIKON CORPORATION",
        camera_model="NIKON Z 50",
        lens_model="18-140mm f/3.5-5.6",
        focal_length_mm=18.0,
    )
    monkeypatch.setattr(lens_correct, "read_lens_exif", lambda path: exif)

    class _Lens:
        model = "18-140mm f/3.5-5.6"

    camera, lens = object(), _Lens()
    monkeypatch.setattr(lens_correct, "_find_profile", lambda e: (camera, lens))

    rgb = _rgb()
    corrected_frame = np.full_like(rgb, 42)
    captured = {}

    def _fake_remap(frame, cam, ln, focal_length_mm):
        captured["frame"] = frame
        captured["camera"] = cam
        captured["lens"] = ln
        captured["focal_length_mm"] = focal_length_mm
        return corrected_frame

    monkeypatch.setattr(lens_correct, "_remap", _fake_remap)

    result = lens_correct.correct(rgb, Path("photo.jpg"))
    assert result.applied is True
    assert np.array_equal(result.rgb, corrected_frame)
    assert "18-140mm" in result.reason
    assert captured["camera"] is camera
    assert captured["lens"] is lens
    assert captured["focal_length_mm"] == 18.0
    assert np.array_equal(captured["frame"], rgb)


def test_read_lens_exif_on_a_missing_file_is_the_all_empty_default(tmp_path):
    exif = lens_correct.read_lens_exif(tmp_path / "does-not-exist.jpg")
    assert exif == lens_correct.LensExif()


def test_read_lens_exif_reads_camera_and_lens_tags(tmp_path):
    from PIL import Image

    photo = tmp_path / "photo.jpg"
    img = Image.new("RGB", (4, 4), color=(10, 20, 30))
    exif = img.getexif()
    exif[lens_correct._EXIF_MAKE_TAG] = "NIKON CORPORATION"
    exif[lens_correct._EXIF_MODEL_TAG] = "NIKON Z 50"
    sub_ifd = exif.get_ifd(lens_correct._EXIF_SUBIFD_TAG)
    sub_ifd[lens_correct._EXIF_LENS_MODEL_TAG] = "18-140mm f/3.5-5.6"
    sub_ifd[lens_correct._EXIF_FOCAL_LENGTH_TAG] = 24.0
    img.save(photo, exif=exif)

    read = lens_correct.read_lens_exif(photo)
    assert read.camera_make == "NIKON CORPORATION"
    assert read.camera_model == "NIKON Z 50"
    assert read.lens_model == "18-140mm f/3.5-5.6"
    assert read.focal_length_mm == pytest.approx(24.0)


def test_develop_options_lens_correct_defaults_off():
    assert DevelopOptions().lens_correct is False


def test_lens_correct_off_by_default_never_calls_correct(monkeypatch):
    """The default develop path must stay byte-identical: with the field
    unset, ``lens_correct.correct`` is never even called."""

    def _boom(rgb, path):
        raise AssertionError("lens_correct.correct must not run when disabled")

    monkeypatch.setattr(lens_correct, "correct", _boom)
    decoded = _decoded()
    variants = list(develop_variants(decoded, styles=["natural"]))
    assert variants
    for _, developed in variants:
        assert developed.lens_correct_applied is False
        assert developed.lens_correct_reason == ""


def test_lens_correct_enabled_reaches_the_stage_before_straighten_and_crop(monkeypatch):
    decoded = _decoded()
    corrected_frame = np.full_like(decoded.rgb, 99)
    captured = {}

    def _fake_correct(rgb, path):
        captured["rgb"] = rgb
        captured["path"] = path
        return lens_correct.LensCorrectionResult(
            rgb=corrected_frame, applied=True, reason="corrected: test rig"
        )

    monkeypatch.setattr(lens_correct, "correct", _fake_correct)
    variants = list(
        develop_variants(decoded, styles=["natural"], options=DevelopOptions(lens_correct=True))
    )
    assert variants
    for _, developed in variants:
        assert developed.lens_correct_applied is True
        assert developed.lens_correct_reason == "corrected: test rig"
    assert np.array_equal(captured["rgb"], decoded.rgb)
    assert captured["path"] == decoded.path


def test_grade_sidecar_payload_records_the_lens_correct_decision(tmp_path):
    from gomoveshift.pipeline.color_correction import CorrectionParams
    from gomoveshift.pipeline.photos.develop import DevelopedPhoto, grade_sidecar_payload

    photo = tmp_path / "fixture.jpg"
    photo.write_bytes(b"\x00")
    developed = DevelopedPhoto(
        rgb=_rgb(),
        path=photo,
        params=CorrectionParams(),
        cluster_index=0,
        lens_correct_applied=False,
        lens_correct_reason="no camera EXIF to match a lens profile",
        decoder="test",
    )
    payload = grade_sidecar_payload(developed, "natural", 1.0, DevelopOptions(), depth=8)
    assert payload["measured"]["lens_correct_applied"] is False
    assert payload["measured"]["lens_correct_reason"] == "no camera EXIF to match a lens profile"
