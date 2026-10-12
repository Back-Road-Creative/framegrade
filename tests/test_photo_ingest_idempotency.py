"""Content-aware ingest idempotency: a skip must mean the SAME pixels.

The idempotency key used to be the bare stem, journey-wide: any card file
whose stem was registered anywhere in the journey was skipped as "already
ingested" — so a DIFFERENT photo behind a recycled stem (file-counter reset
after 9999, a second camera body, a later day) was silently dropped behind a
wrong skip reason. These tests pin the content-aware key: hash-verified
skips, day-disambiguated names for genuinely different content, and a loud
failure when a name cannot be split.
"""

import hashlib
import logging
import textwrap

import numpy as np
import pytest

from gomoveshift.pipeline.photos import ingest as ingest_mod
from gomoveshift.pipeline.photos.develop import DevelopedPhoto
from gomoveshift.pipeline.photos.ingest import PhotoIngestError, ingest

JOURNEY_MD = textwrap.dedent("""\
    ---
    title: Journey A
    photo_count: 1
    days:
    - day_number: 1
      date: '2025-01-01'
      photos:
      - location: Place A
        photos:
        - photo-001.jpg
    - day_number: 2
      date: '2025-01-02'
      photos: []
    ---
    Body text
""")
DAY1 = dict(day_date="2025-01-01", location="Place A")
DAY2 = dict(day_date="2025-01-02", location="Place B")


@pytest.fixture(autouse=True)
def _passthrough_blur(monkeypatch):
    """These tests own the idempotency decision, not the detector."""
    monkeypatch.setattr(
        ingest_mod,
        "blur_faces_for_publish",
        lambda rgb, label, blur=None: (rgb, 0),
        raising=False,
    )


def _site(tmp_path):
    journey = tmp_path / "content" / "journeys" / "journey-a.md"
    journey.parent.mkdir(parents=True)
    journey.write_text(JOURNEY_MD)
    (tmp_path / "static" / "images" / "journeys" / "journey-a").mkdir(parents=True)
    return journey


def _card(tmp_path, files, name="card"):
    """A card whose files carry DISTINCT bytes — content identity is the
    subject under test, so every file's payload is explicit."""
    card = tmp_path / name
    card.mkdir(parents=True)
    for fname, data in files.items():
        (card / fname).write_bytes(data)
    return card


def _developed(path):
    rgb = np.random.default_rng(7).integers(0, 256, (4, 4, 3), dtype=np.uint8)
    return DevelopedPhoto(rgb=rgb, path=path, params=None, cluster_index=0)


def _never(path):
    raise AssertionError(f"develop must not be called for {path}")


class TestContentVerifiedSkip:
    def test_true_reinsert_same_bytes_still_skips_before_decode(self, tmp_path):
        journey = _site(tmp_path)
        card = _card(tmp_path, {"DSC_0001.NEF": b"pixels-day-one"})
        ingest(card, journey, develop_fn=_developed, **DAY1)
        first = journey.read_text()
        report = ingest(card, journey, develop_fn=_never, **DAY1)
        assert report.ingested == []
        assert report.skipped == [("DSC_0001.NEF", "already ingested as dsc_0001.jpg")]
        assert journey.read_text() == first

    def test_same_stem_different_content_ingests_under_day_prefixed_name(self, tmp_path):
        """A recycled stem carrying different bytes is a DIFFERENT photo —
        it must ingest under the day-disambiguated name, never be dropped
        behind an "already ingested" skip."""
        journey = _site(tmp_path)
        card1 = _card(tmp_path, {"DSC_0001.NEF": b"pixels-day-one"})
        ingest(card1, journey, develop_fn=_developed, **DAY1)
        card2 = _card(tmp_path, {"DSC_0001.NEF": b"pixels-day-two"}, name="card2")
        report = ingest(card2, journey, develop_fn=_developed, **DAY2)
        assert report.ingested == ["2025-01-02-dsc_0001.jpg"]
        assert report.skipped == []
        assert "- 2025-01-02-dsc_0001.jpg" in journey.read_text()
        gallery = tmp_path / "static" / "images" / "journeys" / "journey-a"
        assert (gallery / "2025-01-02-dsc_0001.jpg").stat().st_size > 0

    def test_reinsert_of_disambiguated_ingest_skips(self, tmp_path):
        """Both name forms are skip candidates: a re-run after a
        day-prefixed ingest is still a verified no-op."""
        journey = _site(tmp_path)
        ingest(
            _card(tmp_path, {"DSC_0001.NEF": b"pixels-day-one"}),
            journey,
            develop_fn=_developed,
            **DAY1,
        )
        card2 = _card(tmp_path, {"DSC_0001.NEF": b"pixels-day-two"}, name="card2")
        ingest(card2, journey, develop_fn=_developed, **DAY2)
        first = journey.read_text()
        report = ingest(card2, journey, develop_fn=_never, **DAY2)
        assert report.ingested == []
        assert report.skipped == [("DSC_0001.NEF", "already ingested as 2025-01-02-dsc_0001.jpg")]
        assert journey.read_text() == first

    def test_unsplittable_collision_fails_loud_naming_both_names(self, tmp_path):
        """Third distinct content behind one stem, same day: the bare name
        AND the day-prefixed name are both taken by different bytes — no
        honest name is left, so the card hard-fails naming both."""
        journey = _site(tmp_path)
        ingest(
            _card(tmp_path, {"DSC_0001.NEF": b"pixels-day-one"}),
            journey,
            develop_fn=_developed,
            **DAY1,
        )
        ingest(
            _card(tmp_path, {"DSC_0001.NEF": b"pixels-day-two"}, name="card2"),
            journey,
            develop_fn=_developed,
            **DAY2,
        )
        before = journey.read_text()
        card3 = _card(tmp_path, {"DSC_0001.NEF": b"pixels-third-body"}, name="card3")
        with pytest.raises(PhotoIngestError, match=r"dsc_0001\.jpg.*2025-01-02-dsc_0001\.jpg"):
            ingest(card3, journey, develop_fn=_developed, **DAY2)
        assert journey.read_text() == before


class TestLegacyEntriesWithoutHash:
    def test_legacy_no_hash_entry_skips_with_unverified_warning(self, tmp_path, caplog):
        """photo-001.jpg predates hash persistence: the skip is kept (never
        re-develop a registered name blind) but loudly marked unverified."""
        journey = _site(tmp_path)
        card = _card(tmp_path, {"PHOTO-001.JPG": b"possibly different pixels"})
        with caplog.at_level(logging.WARNING):
            report = ingest(card, journey, develop_fn=_never, **DAY1)
        assert report.ingested == []
        assert report.skipped == [
            ("PHOTO-001.JPG", "already ingested as photo-001.jpg (unverified)")
        ]
        assert any("no recorded content hash" in r.message for r in caplog.records)
        assert journey.read_text() == JOURNEY_MD


class TestHashPersistence:
    def test_source_hash_is_persisted_for_new_ingests(self, tmp_path):
        journey = _site(tmp_path)
        data = b"pixels-day-one"
        ingest(_card(tmp_path, {"DSC_0001.NEF": data}), journey, develop_fn=_developed, **DAY1)
        text = journey.read_text()
        assert "photo_hashes" in text
        assert hashlib.sha256(data).hexdigest() in text
