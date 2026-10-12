"""Per-photo blur evidence persisted at ingest: the proof that a photo's
export was (or was not) blurred lasts one function call today and is then
unrecoverable. This pins the frontmatter record and its readback through the
catalog reader, across all four states -- blurred-with-regions, blurred with
none found, operator-skipped, and never-recorded (legacy)."""

import numpy as np
import pytest

from gomoveshift.catalog import read_catalog
from gomoveshift.pipeline.photos import ingest as ingest_mod
from gomoveshift.pipeline.photos.develop import DevelopedPhoto
from gomoveshift.pipeline.photos.ingest import (
    BLUR_KEY,
    BLUR_NO_REGIONS,
    BLUR_REGIONS_FOUND,
    BLUR_SKIPPED,
    ingest,
)

JOURNEY_MD = (
    "---\ntitle: Journey A\nphoto_count: 1\ndays:\n- day_number: 1\n  date: '2025-01-01'\n"
    "  photos:\n  - location: Place A\n    photos:\n    - photo-001.jpg\n---\nBody text\n"
)
DAY = dict(day_date="2025-01-01", location="Place A")


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


def test_blurred_photo_with_regions_is_recorded(tmp_path, monkeypatch):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF"])
    monkeypatch.setattr(
        ingest_mod, "blur_faces_for_publish", lambda rgb, label, blur=None: (rgb, 2), raising=False
    )

    ingest(card, journey, develop_fn=_developed, **DAY)

    text = journey.read_text()
    assert f"{BLUR_KEY}:" in text
    assert f"a.jpg: {BLUR_REGIONS_FOUND}" in text


def test_blurred_photo_with_no_regions_is_recorded(tmp_path, monkeypatch):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF"])
    monkeypatch.setattr(
        ingest_mod, "blur_faces_for_publish", lambda rgb, label, blur=None: (rgb, 0), raising=False
    )

    ingest(card, journey, develop_fn=_developed, **DAY)

    text = journey.read_text()
    assert f"a.jpg: {BLUR_NO_REGIONS}" in text


def test_operator_skip_blur_is_recorded_distinctly(tmp_path, monkeypatch):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF"])
    monkeypatch.setattr(
        ingest_mod,
        "blur_faces_for_publish",
        lambda rgb, label, blur=None: (
            pytest.fail("skip_blur must never call the detector") if blur else (rgb, 0)
        ),
        raising=False,
    )

    ingest(card, journey, develop_fn=_developed, skip_blur=True, **DAY)

    text = journey.read_text()
    assert f"a.jpg: {BLUR_SKIPPED}" in text


def test_legacy_photo_with_no_recorded_state_reads_back_as_none(tmp_path):
    site = tmp_path
    jdir = site / "content" / "journeys"
    jdir.mkdir(parents=True)
    (jdir / "journey-a.md").write_text(JOURNEY_MD)  # never ingested through this code path

    entries = read_catalog(site, media_type="photo").entries
    assert len(entries) == 1
    assert entries[0].blur_state is None


def test_catalog_reader_surfaces_all_three_recorded_states(tmp_path, monkeypatch):
    site = tmp_path
    journey = _site(site)
    card = _card(site, ["found.NEF", "none.NEF"])
    monkeypatch.setattr(
        ingest_mod,
        "blur_faces_for_publish",
        lambda rgb, label, blur=None: (rgb, 3 if "found" in label else 0),
        raising=False,
    )
    ingest(card, journey, develop_fn=_developed, **dict(day_date="2025-01-01", location="Place A"))

    entries = read_catalog(site, media_type="photo").entries
    by_name = {e.catalog_id.split(":")[-1]: e for e in entries}
    assert by_name["found.jpg"].blur_state == BLUR_REGIONS_FOUND
    assert by_name["none.jpg"].blur_state == BLUR_NO_REGIONS
    # The pre-existing hand-registered entry never went through blur — unrecorded.
    assert by_name["photo-001.jpg"].blur_state is None
