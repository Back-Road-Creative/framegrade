"""Ingest refuses a card whose newly developed frames ALL fail the cull floor."""

import numpy as np
import pytest

from gomoveshift.pipeline.photos import ingest as ingest_mod
from gomoveshift.pipeline.photos.develop import DevelopedPhoto
from gomoveshift.pipeline.photos.ingest import PhotoIngestError, ingest

JOURNEY_MD = "---\ntitle: Journey A\ndays:\n- day_number: 1\n  date: '2025-01-01'\n---\nBody text\n"
DAY = dict(day_date="2025-01-01", location="Place A")


@pytest.fixture(autouse=True)
def _passthrough_blur(monkeypatch):
    monkeypatch.setattr(
        ingest_mod, "blur_faces_for_publish", lambda rgb, label, blur=None: (rgb, 0), raising=False
    )


def _site(tmp_path):
    journey = tmp_path / "content" / "journeys" / "journey-a.md"
    journey.parent.mkdir(parents=True)
    journey.write_text(JOURNEY_MD)
    gallery = tmp_path / "static" / "images" / "journeys" / "journey-a"
    gallery.mkdir(parents=True)
    return journey, gallery


def _card(tmp_path, names):
    card = tmp_path / "card"
    card.mkdir()
    for name in names:
        (card / name).write_bytes(b"raw bytes")
    return card


def _frame(path, sharp):
    if sharp:
        rgb = np.random.default_rng(7).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    else:
        rgb = np.full((64, 64, 3), 120, dtype=np.uint8)
    return DevelopedPhoto(rgb=rgb, path=path, params=None, cluster_index=0, decoder="rawpy")


def test_ingest_refuses_all_soft_card(tmp_path):
    journey, gallery = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF", "b.NEF"])
    before = journey.read_text()

    with pytest.raises(PhotoIngestError) as err:
        ingest(card, journey, develop_fn=lambda p: _frame(p, False), **DAY)

    msg = str(err.value)
    assert f"card {card}: all 2 developed frame(s) fail the cull floor" in msg
    assert "a.NEF" in msg and "b.NEF" in msg
    assert "sharpness 0.0 < floor 30" in msg
    assert "floors:" in msg
    assert list(gallery.iterdir()) == []
    assert journey.read_text() == before


def test_ingest_keeps_card_with_one_sharp_frame(tmp_path):
    journey, gallery = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF", "b.NEF"])

    report = ingest(card, journey, develop_fn=lambda p: _frame(p, p.name == "a.NEF"), **DAY)

    assert report.cull_failed == 1
    assert sorted(report.ingested) == ["a.jpg", "b.jpg"]
    assert (gallery / "b.jpg").exists()


def test_dry_run_stays_decode_free(tmp_path):
    """A dry run decodes nothing, so it cannot score frames: it plans, never refuses."""
    journey, gallery = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF"])

    def _never(path):
        raise AssertionError("dry run must not develop")

    report = ingest(card, journey, develop_fn=_never, dry_run=True, **DAY)

    assert report.ingested == ["a.jpg"] and report.cull_failed == 0
