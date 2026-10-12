"""The astro stack records what subject recognition said about the frames
it was handed.

``photos/subject.py`` classifies ``astro`` (among wildlife, portrait,
long_exposure, exposure_bracket and landscape) with a confidence, an honest ``unknown``, and an operator-correction sidecar. Not one
of the twenty ``astro_*.py`` engines mentioned it: the stack that exists to
process exactly that subject never asked what its frames were, and the run
record said nothing either way. A stack run over frames nothing recognises as
astro was indistinguishable, afterwards, from one over a clean Milky Way
sequence.

This RECORDS; it never gates. Whether a mismatch should stop a stack is a
decision this makes answerable, not one it makes.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.photos import astro_sidecar as sidecar_mod
from gomoveshift.pipeline.photos.astro_develop import AstroDevelopResult
from gomoveshift.pipeline.photos.astro_stack import StackResult
from gomoveshift.pipeline.photos.subject import (
    SOURCE_INFERENCE,
    SOURCE_OPERATOR,
    SubjectCategory,
    SubjectReading,
)


def _develop_result():
    return AstroDevelopResult(
        rgb_uint16=np.zeros((4, 4, 3), dtype=np.uint16),
        black_point=np.float64(0.012),
        m_values=[0.31, 0.24, 0.19],
        background_before=np.float64(0.08),
        background_after=0.12,
        background_gains=(1.0, 1.02, 0.97),
    )


def _stack_result():
    return StackResult(
        rgb=np.zeros((4, 4, 3), dtype=np.float32),
        n_frames=3,
        mean_rejected_fraction=np.float64(0.008),
    )


def _payload(**overrides):
    kwargs = dict(
        used_frames=[],
        rejected_frames=[],
        develop=_develop_result(),
        stack=_stack_result(),
        decoder="d",
        algo_version="v",
    )
    kwargs.update(overrides)
    return sidecar_mod.stack_sidecar_payload(**kwargs)


@pytest.fixture
def lights(tmp_path):
    paths = []
    for i in range(3):
        p = tmp_path / f"light{i:03d}.NEF"
        p.write_bytes(b"\x00" * 16)
        paths.append(p)
    return paths


def _patch_recognition(monkeypatch, mapping):
    monkeypatch.setattr(sidecar_mod, "recognise_photos", lambda paths: dict(mapping))


def test_a_stack_records_the_subject_its_frames_were_recognised_as(monkeypatch, lights):
    reading = SubjectReading(SubjectCategory.ASTRO, 0.95, SOURCE_INFERENCE, ("iso",))
    _patch_recognition(monkeypatch, {p: reading for p in lights})

    payload = _payload(light_paths=lights)

    assert payload["subject"]["dominant"] == SubjectCategory.ASTRO.value
    assert payload["subject"]["categories"][SubjectCategory.ASTRO.value] == 3
    assert payload["subject"]["frames_read"] == 3


def test_a_disagreeing_frame_is_named_not_merely_counted(monkeypatch, lights):
    astro = SubjectReading(SubjectCategory.ASTRO, 0.95)
    odd = SubjectReading(SubjectCategory.LANDSCAPE, 0.65)
    _patch_recognition(monkeypatch, {lights[0]: astro, lights[1]: astro, lights[2]: odd})

    payload = _payload(light_paths=lights)

    assert payload["subject"]["dominant"] == SubjectCategory.ASTRO.value
    disagreeing = payload["subject"]["disagreeing"]
    assert [d["path"] for d in disagreeing] == [str(lights[2])]
    assert disagreeing[0]["category"] == SubjectCategory.LANDSCAPE.value
    # A count alone is the defect this campaign keeps closing: name the frame.
    assert len(disagreeing) == 1


def test_unknown_is_recorded_as_unknown_never_folded_into_the_dominant(monkeypatch, lights):
    astro = SubjectReading(SubjectCategory.ASTRO, 0.95)
    from gomoveshift.pipeline.photos.subject import UNKNOWN_SUBJECT

    _patch_recognition(
        monkeypatch, {lights[0]: astro, lights[1]: UNKNOWN_SUBJECT, lights[2]: UNKNOWN_SUBJECT}
    )

    payload = _payload(light_paths=lights)

    cats = payload["subject"]["categories"]
    assert cats[SubjectCategory.UNKNOWN.value] == 2
    assert cats[SubjectCategory.ASTRO.value] == 1
    # Unknown is the majority here and is a legitimate dominant: it says the
    # frames were read and nothing was proven, which is not the same as astro.
    assert payload["subject"]["dominant"] == SubjectCategory.UNKNOWN.value


def test_no_light_paths_says_so_rather_than_reading_as_all_fine(monkeypatch):
    payload = _payload()

    subject = payload["subject"]
    assert subject["frames_read"] == 0
    assert subject["dominant"] is None
    assert subject["reason"], "an empty subject block must say WHY it is empty"
    assert subject["categories"] == {}


def test_an_operator_correction_is_recorded_as_the_operator_s(monkeypatch, lights):
    corrected = SubjectReading(SubjectCategory.WILDLIFE, 1.0, SOURCE_OPERATOR, ("operator",))
    _patch_recognition(monkeypatch, {p: corrected for p in lights})

    payload = _payload(light_paths=lights)

    assert payload["subject"]["sources"] == {SOURCE_OPERATOR: 3}


def test_a_recognition_failure_never_aborts_the_run_record(monkeypatch, lights):
    def _boom(paths):
        raise OSError("EXIF read blew up")

    monkeypatch.setattr(sidecar_mod, "recognise_photos", _boom)

    payload = _payload(light_paths=lights)

    # The sidecar is a RECORD, not a gate: recognition failing must not cost the
    # operator a written run record for a stack that already succeeded.
    assert payload["subject"]["frames_read"] == 0
    assert "EXIF read blew up" in payload["subject"]["reason"]
    assert payload["stack"]["n_frames"] == 3


def test_the_subject_block_survives_json_round_trip(monkeypatch, lights):
    reading = SubjectReading(SubjectCategory.ASTRO, 0.95)
    _patch_recognition(monkeypatch, {p: reading for p in lights})

    payload = _payload(light_paths=lights)

    assert json.loads(json.dumps(payload, sort_keys=True))["subject"]["frames_read"] == 3


def test_the_sidecar_version_moved_because_the_record_shape_changed():
    # A reader keying on sidecar_version must not be handed a new block under
    # the old number.
    assert sidecar_mod.STACK_SIDECAR_VERSION != "5"
