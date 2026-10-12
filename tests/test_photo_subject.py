"""Subject recognition: what the photo is OF, with confidence and an honest unknown.

``scene.py`` measures the LIGHT — exposure, warmth, haze, chroma. Nothing in the
pipeline said whether a frame was a Milky Way sequence, a long lens on an animal,
a portrait, a tripod water frame or one rung of an exposure bracket. These tests
pin the recognition layer that answers that from EXIF, source grouping and the
existing measurements — and, above all, pin that ``unknown`` is a REAL answer
that is never coerced into a best guess.
"""

import datetime
import socket

import pytest

from gomoveshift.pipeline.photos.bracket import detect_brackets
from gomoveshift.pipeline.photos.scene import PhotoExif, SceneProfile
from gomoveshift.pipeline.photos.styles import STYLES, is_artistic_style, select_styles
from gomoveshift.pipeline.photos.subject import (
    CONFIDENCE_OPERATOR,
    CaptureOptics,
    SubjectCategory,
    SubjectReading,
    read_subject_correction,
    read_capture_optics,
    recognise_photos,
    recognise_subject,
    write_subject_correction,
)
from test_photo_nef_exif import _raw


def _scene(**kwargs):
    fields = dict(
        exposure_class="normal",
        warmth="neutral",
        sat_profile="normal",
        haze="clear",
        dynamic_range="normal",
        is_near_monochrome=False,
    )
    fields.update(kwargs)
    return SceneProfile(**fields)


NIGHT = PhotoExif(iso=3200, exposure_s=20.0, capture_hour=23, solar_elevation_deg=-31.0)
DAY = PhotoExif(iso=100, exposure_s=1 / 500, capture_hour=12, solar_elevation_deg=42.0)


class TestUnknownIsARealAnswer:
    def test_no_evidence_at_all_reads_unknown_not_a_guess(self):
        reading = recognise_subject()
        assert reading.category is SubjectCategory.UNKNOWN
        assert reading.confidence == 0.0
        assert reading.source == "inference"

    def test_evidence_that_matches_nothing_stays_unknown(self):
        """A daylight frame with no optics matches no category — and is not
        rounded up to 'landscape' because landscape is the common case."""
        reading = recognise_subject(exif=DAY, scene=_scene(exposure_class="bright"))
        assert reading.category is SubjectCategory.UNKNOWN

    def test_unknown_may_never_carry_confidence(self):
        with pytest.raises(ValueError):
            SubjectReading(SubjectCategory.UNKNOWN, 0.6, "inference")

    def test_a_named_category_may_never_carry_zero_confidence(self):
        with pytest.raises(ValueError):
            SubjectReading(SubjectCategory.ASTRO, 0.0, "inference")

    def test_confidence_is_a_probability(self):
        with pytest.raises(ValueError):
            SubjectReading(SubjectCategory.ASTRO, 1.4, "inference")

    def test_the_taxonomy_carries_an_unknown_member(self):
        assert SubjectCategory("unknown") is SubjectCategory.UNKNOWN


class TestExifIsTheStrongestEvidence:
    def test_long_exposure_high_iso_under_a_tracked_night_sky_reads_astro(self):
        reading = recognise_subject(exif=NIGHT, scene=_scene(exposure_class="low_key"))
        assert reading.category is SubjectCategory.ASTRO
        assert reading.confidence >= 0.8
        assert any("solar" in e for e in reading.evidence)

    def test_without_a_solar_fix_astro_is_the_same_answer_with_less_confidence(self):
        tracked = recognise_subject(exif=NIGHT)
        untracked = recognise_subject(
            exif=PhotoExif(iso=3200, exposure_s=20.0, capture_hour=23),
            scene=_scene(exposure_class="low_key"),
        )
        assert untracked.category is SubjectCategory.ASTRO
        assert 0.0 < untracked.confidence < tracked.confidence

    def test_a_long_exposure_in_daylight_is_never_astro(self):
        reading = recognise_subject(
            exif=PhotoExif(iso=3200, exposure_s=20.0, capture_hour=12, solar_elevation_deg=40.0)
        )
        assert reading.category is not SubjectCategory.ASTRO

    def test_a_long_lens_at_a_fast_shutter_reads_wildlife(self):
        reading = recognise_subject(exif=DAY, optics=CaptureOptics(400.0, 5.6))
        assert reading.category is SubjectCategory.WILDLIFE

    def test_a_short_fast_lens_wide_open_reads_portrait(self):
        reading = recognise_subject(
            exif=PhotoExif(iso=200, exposure_s=1 / 250, capture_hour=16, solar_elevation_deg=20.0),
            optics=CaptureOptics(85.0, 1.8),
        )
        assert reading.category is SubjectCategory.PORTRAIT

    def test_a_stopped_down_tripod_frame_in_daylight_reads_long_exposure(self):
        reading = recognise_subject(
            exif=PhotoExif(iso=64, exposure_s=2.0, capture_hour=10, solar_elevation_deg=25.0),
            optics=CaptureOptics(24.0, 16.0),
        )
        assert reading.category is SubjectCategory.LONG_EXPOSURE

    def test_a_wide_stopped_down_daylight_frame_reads_landscape_weakly(self):
        reading = recognise_subject(exif=DAY, optics=CaptureOptics(24.0, 11.0))
        assert reading.category is SubjectCategory.LANDSCAPE
        assert reading.confidence < 0.6  # honest: the pixels were never asked

    def test_every_reading_names_the_evidence_it_used(self):
        assert recognise_subject(exif=NIGHT).evidence


class TestSourceGrouping:
    def test_a_frame_the_bracket_detector_claims_reads_exposure_bracket(self):
        reading = recognise_subject(exif=DAY, in_exposure_bracket=True)
        assert reading.category is SubjectCategory.EXPOSURE_BRACKET
        assert any("bracket" in e for e in reading.evidence)

    def test_a_run_of_astro_frames_from_one_session_is_a_sequence(self, tmp_path):
        """The Milky Way case: one 20 s frame is astro, eight consecutive ones
        are a SEQUENCE, and the sequence is the stronger claim."""
        single = recognise_subject(
            exif=PhotoExif(iso=3200, exposure_s=20.0, capture_hour=23), session_astro_frames=1
        )
        seq = recognise_subject(
            exif=PhotoExif(iso=3200, exposure_s=20.0, capture_hour=23), session_astro_frames=12
        )
        assert seq.category is SubjectCategory.ASTRO
        assert seq.confidence > single.confidence
        assert any("sequence" in e for e in seq.evidence)

    def test_a_card_reads_as_a_batch_and_agrees_with_the_bracket_detector(self, tmp_path):
        card = [
            _raw(tmp_path / "DSC_0001.NEF", when="2024:01:15 10:15:00", exposure=(1, 320)),
            _raw(tmp_path / "DSC_0002.NEF", when="2024:01:15 10:15:30", exposure=(1, 400)),
            _raw(tmp_path / "DSC_0003.NEF", when="2024:01:15 10:15:37", exposure=(1, 500)),
        ]
        assert len(detect_brackets(card)) == 1
        readings = recognise_photos(card)
        assert set(readings) == set(card)
        assert all(r.category is SubjectCategory.EXPOSURE_BRACKET for r in readings.values())

    def test_a_batch_recognises_a_night_sequence_off_the_card(self, tmp_path):
        card = [
            _raw(
                tmp_path / f"DSC_{i:04d}.NEF",
                when=f"2024:01:15 23:1{i}:00",
                iso=3200,
                exposure=(20, 1),
            )
            for i in range(8)
        ]
        readings = recognise_photos(card)
        assert all(r.category is SubjectCategory.ASTRO for r in readings.values())
        assert all(any("sequence" in e for e in r.evidence) for r in readings.values())

    def test_a_file_with_no_exif_at_all_reads_unknown_never_raises(self, tmp_path):
        blank = tmp_path / "not-a-photo.jpg"
        blank.write_bytes(b"nonsense")
        readings = recognise_photos([blank])
        assert readings[blank].category is SubjectCategory.UNKNOWN


class TestOperatorCorrectionsWin:
    def test_a_correction_beats_every_inference(self):
        reading = recognise_subject(exif=NIGHT, correction=SubjectCategory.WILDLIFE)
        assert reading.category is SubjectCategory.WILDLIFE
        assert reading.source == "operator"
        assert reading.confidence == CONFIDENCE_OPERATOR

    def test_an_operator_may_correct_a_frame_to_unknown(self):
        reading = recognise_subject(exif=NIGHT, correction=SubjectCategory.UNKNOWN)
        assert reading.category is SubjectCategory.UNKNOWN
        assert reading.confidence == 0.0
        assert reading.source == "operator"

    def test_a_correction_persists_beside_the_photo_and_survives_a_reread(self, tmp_path):
        photo = tmp_path / "DSC_0009.NEF"
        photo.write_bytes(b"x")
        sidecar = write_subject_correction(photo, SubjectCategory.WILDLIFE)
        assert sidecar.exists()
        assert read_subject_correction(photo) is SubjectCategory.WILDLIFE

    def test_no_sidecar_reads_none_not_unknown(self, tmp_path):
        photo = tmp_path / "DSC_0010.NEF"
        photo.write_bytes(b"x")
        assert read_subject_correction(photo) is None

    def test_a_corrupt_or_unrecognised_sidecar_reads_none_and_never_raises(self, tmp_path):
        photo = tmp_path / "DSC_0011.NEF"
        photo.write_bytes(b"x")
        write_subject_correction(photo, SubjectCategory.PORTRAIT)
        sidecar = photo.with_suffix(photo.suffix + ".subject.json")
        sidecar.write_text("{not json")
        assert read_subject_correction(photo) is None
        sidecar.write_text('{"category": "unicorn"}')
        assert read_subject_correction(photo) is None

    def test_a_batch_prefers_the_sidecar_over_what_the_exif_says(self, tmp_path):
        photo = _raw(tmp_path / "DSC_0012.NEF", when="2024:01:15 23:10:00", exposure=(20, 1))
        write_subject_correction(photo, SubjectCategory.PORTRAIT)
        reading = recognise_photos([photo])[photo]
        assert reading.category is SubjectCategory.PORTRAIT
        assert reading.source == "operator"


class TestNaturalIsTheDefaultAndArtisticIsLabelled:
    def test_natural_is_not_artistic_and_anchors_every_selection(self):
        assert is_artistic_style("natural") is False
        assert select_styles(_scene())[0] == "natural"

    def test_the_white_balance_trio_is_correction_not_restyling(self):
        assert [is_artistic_style(n) for n in ("daylight", "cloudy", "shade")] == [False] * 3

    def test_a_look_move_is_artistic(self):
        assert is_artistic_style("moody") is True
        assert is_artistic_style("mono") is True

    def test_every_registered_style_is_classified_one_way_or_the_other(self):
        assert all(isinstance(is_artistic_style(name), bool) for name in STYLES)

    def test_an_unregistered_name_raises_rather_than_guessing(self):
        with pytest.raises(ValueError):
            is_artistic_style("not_a_style")

    def test_selection_did_not_regress(self):
        """No subject reading passed in: selection stays exactly what it was
        before subject routing was wired in (`styles.select_styles`'s ``subject``
        parameter defaults to ``None``, which never reorders anything)."""
        assert select_styles(_scene(haze="foggy", sat_profile="muted")) == [
            "natural",
            "moody",
            "mono",
        ]
        assert select_styles(_scene(exposure_class="bright", chroma_headroom=0.8)) == [
            "natural",
            "vivid",
            "soft",
        ]


class TestNothingReachesTheNetwork:
    def test_recognition_runs_with_sockets_disabled(self, monkeypatch):
        def _no_network(*args, **kwargs):
            raise AssertionError("subject recognition must never open a socket")

        monkeypatch.setattr(socket, "socket", _no_network)
        assert recognise_subject(exif=NIGHT).category is SubjectCategory.ASTRO

    def test_optics_read_off_a_card_never_raise_and_degrade_to_none(self, tmp_path):
        missing = tmp_path / "gone.NEF"
        assert read_capture_optics(missing) == CaptureOptics(None, None)
        framed = _raw(tmp_path / "a.NEF", f_number=(56, 10), focal_length=(400, 1))
        assert read_capture_optics(framed) == CaptureOptics(400.0, 5.6)


def test_a_reading_is_frozen_and_comparable():
    one = SubjectReading(SubjectCategory.ASTRO, 0.85, "inference", ("exif: 20.0 s",))
    assert one == SubjectReading(SubjectCategory.ASTRO, 0.85, "inference", ("exif: 20.0 s",))
    with pytest.raises(Exception):
        one.confidence = 0.1


def test_capture_utc_is_carried_but_never_required():
    assert (
        recognise_subject(
            exif=PhotoExif(
                iso=6400,
                exposure_s=25.0,
                capture_utc=datetime.datetime(2024, 1, 15, 3, 0, tzinfo=datetime.timezone.utc),
                solar_elevation_deg=-40.0,
            )
        ).category
        is SubjectCategory.ASTRO
    )
