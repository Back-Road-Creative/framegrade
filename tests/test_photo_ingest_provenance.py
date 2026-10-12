"""Decode provenance persisted per photo + the ``--redevelop`` / ``--dry-run``
paths: without it, filename idempotency pinned a preview develop forever."""

from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.color_correction import CorrectionParams
from gomoveshift.pipeline.photos import ingest as ingest_mod
from gomoveshift.pipeline.photos.develop import (
    PREVIEW_FALLBACK_DECODER,
    DecodedPhoto,
    DevelopedPhoto,
    develop,
)
from gomoveshift.pipeline.photos.exif_gps import RESOLVED, PhotoLocation
from gomoveshift.pipeline.photos.ingest import IngestReport, ingest

JOURNEY_MD = (
    "---\ntitle: Journey A\nphoto_count: 1\ndays:\n- day_number: 1\n  date: '2025-01-01'\n"
    "  photos:\n  - location: Place A\n    photos:\n    - photo-001.jpg\n---\nBody text\n"
)
#: The registered photo's recorded provenance is the preview fallback.
JOURNEY_WITH_PROVENANCE = JOURNEY_MD.replace(
    "---\nBody text",
    f"photo_provenance:\n  photo-001.jpg: {PREVIEW_FALLBACK_DECODER}\n---\nBody text",
)
DAY = dict(day_date="2025-01-01", location="Place A")


@pytest.fixture(autouse=True)
def _passthrough_blur(monkeypatch):
    monkeypatch.setattr(
        ingest_mod, "blur_faces_for_publish", lambda rgb, label, blur=None: (rgb, 0), raising=False
    )


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


def _developed(path, decoder, fill=0):
    rgb = np.random.default_rng(fill).integers(0, 256, (4, 4, 3), dtype=np.uint8)
    return DevelopedPhoto(rgb=rgb, path=path, params=None, cluster_index=0, decoder=decoder)


def _never(path):
    raise AssertionError(f"develop must not be called for {path}")


def test_develop_propagates_the_decoder_tag():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    decoded = DecodedPhoto(rgb, Path("x.nef"), PREVIEW_FALLBACK_DECODER, 8, 8, 8, False)
    assert develop(decoded, CorrectionParams()).decoder == PREVIEW_FALLBACK_DECODER


def test_provenance_persists_and_redevelop_replaces_only_the_fallback(tmp_path):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["a.NEF", "b.NEF"])
    tags = {"a.NEF": PREVIEW_FALLBACK_DECODER, "b.NEF": "rawpy"}

    report = ingest(card, journey, develop_fn=lambda p: _developed(p, tags[p.name], 10), **DAY)

    assert report.decoder_counts == {PREVIEW_FALLBACK_DECODER: 1, "rawpy": 1}
    text = journey.read_text()
    assert "photo_provenance:" in text and f"a.jpg: {PREVIEW_FALLBACK_DECODER}" in text
    gallery = tmp_path / "static" / "images" / "journeys" / "journey-a"
    first_bytes = (gallery / "a.jpg").read_bytes()

    # Without --redevelop the recorded fallback still skips, page untouched.
    report = ingest(card, journey, develop_fn=_never, **DAY)
    assert report.skipped[0] == ("a.NEF", "already ingested as a.jpg")

    calls = []

    def _true_raw(path):
        calls.append(path.name)
        return _developed(path, "rawpy-dng", 200)

    report = ingest(card, journey, develop_fn=_true_raw, redevelop=True, **DAY)

    assert calls == ["a.NEF"]  # b.NEF's true-raw provenance still skips
    assert report.redeveloped == ["a.jpg"] and report.ingested == []
    assert (gallery / "a.jpg").read_bytes() != first_bytes
    text = journey.read_text()  # provenance updated; the name registered once
    assert "a.jpg: rawpy-dng" in text and text.count("- a.jpg") == 1

    # photo-001.jpg is hand-registered (no provenance) — redevelop must not guess.
    card2 = _card(tmp_path / "s", ["photo-001.jpg"])
    report = ingest(card2, journey, develop_fn=_never, redevelop=True, **DAY)
    assert report.ingested == [] and report.redeveloped == []


def test_dry_run_resolves_the_label_and_writes_nothing(tmp_path, monkeypatch):
    journey = _site(tmp_path, JOURNEY_WITH_PROVENANCE)
    card = _card(tmp_path, ["new.NEF", "photo-001.jpg"])
    monkeypatch.setattr(ingest_mod, "export_jpeg", lambda *a, **k: pytest.fail("dry run exported"))
    monkeypatch.setattr(ingest_mod, "atomic_write_text", lambda *a, **k: pytest.fail("wrote"))
    loc = PhotoLocation("County Clare, Ireland", RESOLVED)
    monkeypatch.setattr(ingest_mod, "derive_location_from_photos", lambda paths, **k: loc)

    report = ingest(card, journey, develop_fn=_never, day_date="2025-01-01", dry_run=True)

    assert report.dry_run is True and report.ingested == ["new.jpg"]
    assert report.location == "County Clare, Ireland"  # resolved via the real cascade
    # Legacy entry (no recorded hash) → the content-aware skip marks it
    # unverified; the dry run appends the recorded provenance on top.
    assert (
        "photo-001.jpg",
        f"already ingested as photo-001.jpg (unverified) (developed via {PREVIEW_FALLBACK_DECODER})",
    ) in report.skipped

    report = ingest(card, journey, develop_fn=_never, redevelop=True, dry_run=True, **DAY)
    assert report.redeveloped == ["photo-001.jpg"] and report.ingested == ["new.jpg"]

    assert journey.read_text() == JOURNEY_WITH_PROVENANCE
    assert list((tmp_path / "static" / "images" / "journeys" / "journey-a").iterdir()) == []


def test_cli_flags_provenance_summary_and_dry_run_output(capsys, monkeypatch):
    from gomoveshift.cli.handlers.photos import _cmd_photos_ingest
    from gomoveshift.cli.parser import _build_parser

    base = ["photos", "ingest", "card", "--journey", "j.md", "--day", "2025-01-01"]
    args = _build_parser().parse_args(base)
    assert args.redevelop is False and args.dry_run is False
    flagged = _build_parser().parse_args([*base, "--redevelop", "--dry-run"])
    assert flagged.redevelop is True and flagged.dry_run is True

    report = IngestReport(
        ingested=["a.jpg"], location="Place A", decoder_counts={PREVIEW_FALLBACK_DECODER: 1}
    )
    monkeypatch.setattr(ingest_mod, "ingest", lambda *a, **k: report)
    assert _cmd_photos_ingest(args) == 0
    out = capsys.readouterr().out
    assert PREVIEW_FALLBACK_DECODER in out and "--redevelop" in out  # note names the remedy

    report.dry_run = True
    assert _cmd_photos_ingest(args) == 0
    assert "DRY RUN" in capsys.readouterr().out
