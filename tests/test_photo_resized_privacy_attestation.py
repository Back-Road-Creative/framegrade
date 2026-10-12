"""The social-publish still export writes a privacy attestation.

`photos/face_filter.py` SAYS what it can attest, and
`photos/ingest.py` records it per gallery photo in journey frontmatter.
Neither produces the machine-readable document `publishing/dispatch.py` reads:
its `privacy_attestations(row)` callback has exactly one real implementation,
written beside the **video** day master, so the four-class gate that refuses an
unverified video export has nothing to consult for a still.

`PhotoResizer.resize_batch` is the still analogue of that day master -- it is
where the published copy is blurred and written. So it writes the attestation,
in the SAME vocabulary, beside the batch it covers.

**Writes; never gates.** Nothing here decides whether a photo may publish;
that decision belongs to the operator. What these tests pin is that the document a future gate
would read exists, says what this run could actually prove, and is already in
the shape `dispatch.privacy_hold_reason` consumes -- an attestation in a private
dialect would be a record nobody can act on, which is the defect, restated.
"""

import json

import pytest
from PIL import Image

from gomoveshift.publishing.dispatch import (
    PRIVACY_INCONCLUSIVE,
    PRIVACY_REQUIRED_CLASSES,
    PRIVACY_VERIFIED,
    privacy_hold_reason,
)

from gomoveshift.pipeline.photos import resizer as resizer_mod
from gomoveshift.pipeline.photos.resizer import (
    PHOTO_PRIVACY_ATTESTATION_NAME,
    PhotoResizer,
)

ALL_VERIFIED = {cls: {"state": PRIVACY_VERIFIED, "reason": ""} for cls in PRIVACY_REQUIRED_CLASSES}


@pytest.fixture
def photos(tmp_path):
    src = tmp_path / "source"
    src.mkdir()
    out = []
    for i in range(2):
        img = Image.new("RGB", (1920, 1440), color=(90 + i * 40, 70, 50))
        p = src / f"photo-{i:03d}.jpg"
        img.save(p, "JPEG", quality=95)
        out.append(str(p))
    return out


@pytest.fixture
def resizer(tmp_path):
    return PhotoResizer(output_base=tmp_path / "resized")


def _stub_classes(monkeypatch, classes):
    from gomoveshift.pipeline.photos import face_filter

    monkeypatch.setattr(face_filter, "attestable_privacy_classes", lambda blur=None: dict(classes))


def _attestation(resizer, week="2026-02-23"):
    return json.loads((resizer.output_base / week / PHOTO_PRIVACY_ATTESTATION_NAME).read_text())


def test_a_resized_batch_writes_an_attestation_beside_itself(resizer, photos, monkeypatch):
    _stub_classes(monkeypatch, ALL_VERIFIED)
    resizer.resize_batch(photos, "2026-02-23")

    doc = _attestation(resizer)
    assert doc["schema"] == "gms.privacy_attestation/v1"
    assert doc["required_classes"] == list(PRIVACY_REQUIRED_CLASSES)
    assert doc["week_start"] == "2026-02-23"
    assert set(doc["classes"]) == set(PRIVACY_REQUIRED_CLASSES)


def test_the_written_document_is_what_the_publish_gate_reads(resizer, photos, monkeypatch):
    """The point of the whole module: not a plausible-looking record, one the
    existing reader can act on without translation."""
    _stub_classes(monkeypatch, ALL_VERIFIED)
    resizer.resize_batch(photos, "2026-02-23")
    assert privacy_hold_reason(_attestation(resizer)) is None


def test_an_unattested_class_is_a_hold_the_gate_can_state(resizer, photos, monkeypatch):
    classes = dict(ALL_VERIFIED)
    classes["people"] = {
        "state": PRIVACY_INCONCLUSIVE,
        "reason": "person blur is off (blur_person_enabled=false)",
    }
    _stub_classes(monkeypatch, classes)
    resizer.resize_batch(photos, "2026-02-23")

    hold = privacy_hold_reason(_attestation(resizer))
    assert hold is not None
    assert "people" in hold


def test_the_attestation_is_written_before_the_first_photo_is_exported(
    resizer, photos, monkeypatch
):
    """A batch that dies on photo one must not leave exported bytes with no
    record of what was proven about them -- the video contract's "written on
    EVERY path" rule, which exists because publication reads this file and not
    the pixels."""
    _stub_classes(monkeypatch, ALL_VERIFIED)
    seen = {}

    real_open = Image.open

    def _boom(path, *a, **kw):
        seen["attestation_existed"] = (
            resizer.output_base / "2026-02-23" / PHOTO_PRIVACY_ATTESTATION_NAME
        ).exists()
        raise OSError("disk pulled")

    monkeypatch.setattr(Image, "open", _boom)
    try:
        resizer.resize_batch(photos, "2026-02-23")
    except Exception:
        pass
    finally:
        monkeypatch.setattr(Image, "open", real_open)
    assert seen.get("attestation_existed") is True


def test_an_empty_batch_still_says_what_it_could_have_proven(resizer, monkeypatch):
    """ "Nothing to publish" and "nobody wrote it down" are different claims."""
    _stub_classes(monkeypatch, ALL_VERIFIED)
    resizer.resize_batch([], "2026-02-23")
    assert _attestation(resizer)["classes"]


def test_a_failed_attestation_write_does_not_abort_the_batch(resizer, photos, monkeypatch):
    """Fail-closed lives at the READER: a missing attestation is already a hold.
    Raising here would make the record decide the outcome, which is the one
    thing this module must not do."""
    _stub_classes(monkeypatch, ALL_VERIFIED)

    def _boom(*a, **kw):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(resizer_mod, "_write_privacy_attestation", _boom)
    assert len(resizer.resize_batch(photos, "2026-02-23")) == 2


def test_a_second_batch_in_the_same_week_rewrites_its_own_record(resizer, photos, monkeypatch):
    """Config can change between runs; a stale "verified" outliving the run
    that earned it is the cached-evidence bug."""
    _stub_classes(monkeypatch, ALL_VERIFIED)
    resizer.resize_batch(photos, "2026-02-23")
    assert privacy_hold_reason(_attestation(resizer)) is None

    degraded = dict(ALL_VERIFIED)
    degraded["plates"] = {"state": PRIVACY_INCONCLUSIVE, "reason": "plate model missing"}
    _stub_classes(monkeypatch, degraded)
    resizer.resize_batch(photos, "2026-02-23")
    assert privacy_hold_reason(_attestation(resizer)) is not None
