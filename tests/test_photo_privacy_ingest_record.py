"""Per-photo privacy-attestation record persisted at ingest, beside
``photo_blur``: which required privacy classes this run could NOT attest for
one photo. Written only for newly-ingested names, in the same registration
step as ``photo_hashes``/``photo_blur`` -- an unattested list of ``[]`` means
all four classes were attested; a filename absent from the map means this
photo predates the field (never recorded), which must never read as "fine"."""

import numpy as np

from gomoveshift.pipeline.day_master_encoder import PRIVACY_INCONCLUSIVE, PRIVACY_VERIFIED
from gomoveshift.pipeline.photos import ingest as ingest_mod
from gomoveshift.pipeline.photos.develop import DevelopedPhoto
from gomoveshift.pipeline.photos.ingest import PRIVACY_UNATTESTED_KEY, ingest

JOURNEY_MD = (
    "---\ntitle: Journey A\nphoto_count: 1\ndays:\n- day_number: 1\n  date: '2025-01-01'\n"
    "  photos:\n  - location: Place A\n    photos:\n    - photo-001.jpg\n---\nBody text\n"
)
DAY = dict(day_date="2025-01-01", location="Place A")

_ALL_VERIFIED = {
    "people": {"state": PRIVACY_VERIFIED, "reason": ""},
    "faces": {"state": PRIVACY_VERIFIED, "reason": ""},
    "plates": {"state": PRIVACY_VERIFIED, "reason": ""},
    "vehicle_occupants": {"state": PRIVACY_VERIFIED, "reason": ""},
}

_VEHICLE_INCONCLUSIVE = {
    **_ALL_VERIFIED,
    "vehicle_occupants": {
        "state": PRIVACY_INCONCLUSIVE,
        "reason": "vehicle occupant blur disabled by config",
    },
}


def _site(tmp_path, journey_md=JOURNEY_MD):
    journey = tmp_path / "content" / "journeys" / "journey-a.md"
    journey.parent.mkdir(parents=True)
    journey.write_text(journey_md)
    (tmp_path / "static" / "images" / "journeys" / "journey-a").mkdir(parents=True)
    return journey


def _card(tmp_path, names):
    card = tmp_path / "card"
    card.mkdir(parents=True, exist_ok=True)
    for name in names:
        (card / name).write_bytes(b"raw bytes")
    return card


def _developed(path):
    rgb = np.random.default_rng(7).integers(0, 256, (4, 4, 3), dtype=np.uint8)
    return DevelopedPhoto(rgb=rgb, path=path, params=None, cluster_index=0, decoder="rawpy")


def _no_blur(rgb, label, blur=None):
    return rgb, 0


def _group(doc, location="Place A"):
    """The per-location photo group dict, where ``photo_blur``/
    ``photo_privacy_unattested`` are recorded — mirrors ``_location_group``."""
    for day in doc["days"]:
        for group in day["photos"]:
            if group["location"] == location:
                return group
    raise AssertionError(f"no group for {location}")


def test_fully_attested_photo_records_empty_list(tmp_path, monkeypatch):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF"])
    monkeypatch.setattr(ingest_mod, "blur_faces_for_publish", _no_blur, raising=False)
    monkeypatch.setattr(
        ingest_mod, "attestable_privacy_classes", lambda blur=None: _ALL_VERIFIED, raising=False
    )

    ingest(card, journey, develop_fn=_developed, **DAY)

    text = journey.read_text()
    assert f"{PRIVACY_UNATTESTED_KEY}:" in text
    assert "a.jpg: []" in text


def test_config_off_class_is_recorded_as_unattested(tmp_path, monkeypatch):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF"])
    monkeypatch.setattr(ingest_mod, "blur_faces_for_publish", _no_blur, raising=False)
    monkeypatch.setattr(
        ingest_mod,
        "attestable_privacy_classes",
        lambda blur=None: _VEHICLE_INCONCLUSIVE,
        raising=False,
    )

    ingest(card, journey, develop_fn=_developed, **DAY)

    from gomoveshift.pipeline.journal_publisher import load_journey

    doc, _ = load_journey(journey)
    assert _group(doc)[PRIVACY_UNATTESTED_KEY]["a.jpg"] == ["vehicle_occupants"]


def test_never_ingested_photo_is_absent_from_the_map_not_empty(tmp_path, monkeypatch):
    """A hand-registered legacy photo (``photo-001.jpg``) never went through
    this ingest path -- its filename must be MISSING from the map once a real
    ingest writes the field, never present with ``[]``."""
    journey = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF"])
    monkeypatch.setattr(ingest_mod, "blur_faces_for_publish", _no_blur, raising=False)
    monkeypatch.setattr(
        ingest_mod, "attestable_privacy_classes", lambda blur=None: _ALL_VERIFIED, raising=False
    )

    ingest(card, journey, develop_fn=_developed, **DAY)

    from gomoveshift.pipeline.journal_publisher import load_journey

    doc, _ = load_journey(journey)
    unattested = _group(doc)[PRIVACY_UNATTESTED_KEY]
    assert unattested["a.jpg"] == []
    assert "photo-001.jpg" not in unattested
