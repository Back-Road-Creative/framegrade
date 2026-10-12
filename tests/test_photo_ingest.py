"""Camera-card photo ingest: discover -> develop -> register in frontmatter."""

import dataclasses
import fcntl
import json
import logging
import os
import textwrap
from pathlib import Path

import numpy as np
import pytest

from gomoveshift.pipeline.photos import develop as develop_mod
from gomoveshift.pipeline.photos import face_filter
from gomoveshift.pipeline.photos import ingest as ingest_mod
from gomoveshift.pipeline.photos.cull import CullFloors
from gomoveshift.pipeline.photos.develop import DevelopedPhoto, PhotoDecodeError
from gomoveshift.pipeline.photos.face_filter import FaceBlurError
from gomoveshift.pipeline.photos.exif_gps import (
    NO_EXIF_GPS,
    RESOLVED,
    PhotoLocation,
    extract_exif_capture_date,
)
from gomoveshift.pipeline.photos.ingest import PhotoIngestError, discover, ingest

JOURNEY_MD = textwrap.dedent("""\
    ---
    title: Journey A
    # keep this comment - operators annotate journeys by hand
    photo_count: 1
    days:
    - day_number: 1
      date: '2025-01-01'
      photos:
      - location: Place A
        photos:
        - photo-001.jpg
    ---
    Body text
""")
DAY = dict(day_date="2025-01-01", location="Place A")
#: No operator label — the card's own EXIF elects the location.
CARD_ONLY = dict(day_date="2025-01-01")


def _site(tmp_path):
    journey = tmp_path / "content" / "journeys" / "journey-a.md"
    journey.parent.mkdir(parents=True)
    journey.write_text(JOURNEY_MD)
    (tmp_path / "static" / "images" / "journeys" / "journey-a").mkdir(parents=True)
    return journey


def _card(tmp_path, names):
    card = tmp_path / "card"
    card.mkdir(parents=True)
    for name in names:
        (card / name).parent.mkdir(parents=True, exist_ok=True)
        (card / name).write_bytes(b"raw bytes")
    return card


def _developed(path):
    """Stand-in for the real decode/grade chain, which develop.py covers."""
    rgb = np.random.default_rng(7).integers(0, 256, (4, 4, 3), dtype=np.uint8)
    return DevelopedPhoto(rgb=rgb, path=path, params=None, cluster_index=0)


def _never(path):
    raise AssertionError(f"develop must not be called for {path}")


def _stub_derive(monkeypatch, result, calls=None):
    """Replace the EXIF ladder — these tests own the wiring, not the geocode."""

    def _derive(photo_paths, **kwargs):
        if calls is not None:
            calls.append([Path(p).name for p in photo_paths])
        return result

    monkeypatch.setattr(ingest_mod, "derive_location_from_photos", _derive)


class TestLocationResolvedOnceDuringIngest:
    def test_exif_location_is_resolved_once_and_persisted(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["a.DNG", "b.DNG"])
        calls = []
        _stub_derive(monkeypatch, PhotoLocation("County Clare, Ireland", RESOLVED), calls)

        report = ingest(card, journey, develop_fn=_developed, **CARD_ONLY)

        # ONE resolution, over the card's own sources — never re-derived later.
        assert calls == [["a.DNG", "b.DNG"]]
        assert report.location == "County Clare, Ireland"
        assert "location: County Clare, Ireland" in journey.read_text()

    def test_card_without_gps_fails_closed_and_never_invents_a_label(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        _stub_derive(monkeypatch, PhotoLocation(reason=NO_EXIF_GPS))
        with pytest.raises(PhotoIngestError, match=NO_EXIF_GPS):
            ingest(_card(tmp_path, ["a.DNG"]), journey, develop_fn=_developed, **CARD_ONLY)
        # Fail-closed leaves the journey byte-identical, as a bad decode does.
        assert journey.read_text() == JOURNEY_MD

    def test_operator_label_is_authoritative_and_skips_the_geocode(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        monkeypatch.setattr(
            ingest_mod,
            "derive_location_from_photos",
            lambda *a, **k: pytest.fail("an operator label must never be geocoded over"),
        )
        report = ingest(
            _card(tmp_path, ["a.DNG"]),
            journey,
            location="Place B",
            develop_fn=_developed,
            **CARD_ONLY,
        )
        assert report.location == "Place B"
        assert "location: Place B" in journey.read_text()


def test_discover_finds_supported_and_ignores_others(tmp_path):
    card = _card(tmp_path, ["b.DNG", "a.jpg", "notes.txt", "clip.mp4", "c.heic"])
    assert [p.name for p in discover(card)] == ["a.jpg", "b.DNG", "c.heic"]


def test_ingest_develops_each_file_into_the_gallery(tmp_path):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["DSC_0001.DNG", "DSC_0002.dng"])
    report = ingest(card, journey, develop_fn=_developed, **DAY)
    assert report.ingested == ["dsc_0001.jpg", "dsc_0002.jpg"]
    gallery = tmp_path / "static" / "images" / "journeys" / "journey-a"
    assert all((gallery / name).stat().st_size > 0 for name in report.ingested)


def test_ingest_registers_entries_and_preserves_comments(tmp_path):
    journey = _site(tmp_path)
    ingest(_card(tmp_path, ["DSC_0001.DNG"]), journey, develop_fn=_developed, **DAY)
    text = journey.read_text()
    assert "# keep this comment" in text
    assert "- dsc_0001.jpg" in text and "- photo-001.jpg" in text
    assert "photo_count: 2" in text and text.endswith("Body text\n")


def test_ingest_registers_a_new_location_group(tmp_path):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["x.jpg"])
    ingest(card, journey, day_date="2025-01-01", location="Place B", develop_fn=_developed)
    assert "location: Place B" in journey.read_text()


def test_reingest_is_a_noop_and_reports_skips(tmp_path):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["DSC_0001.DNG"])
    ingest(card, journey, develop_fn=_developed, **DAY)
    first = journey.read_text()
    # _never proves the skip lands upstream of any decode.
    report = ingest(card, journey, develop_fn=_never, **DAY)
    assert report.ingested == []
    assert report.skipped == [("DSC_0001.DNG", "already ingested as dsc_0001.jpg")]
    assert journey.read_text() == first


def test_unknown_day_fails_closed(tmp_path):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["x.jpg"])
    with pytest.raises(PhotoIngestError, match="2099-01-01"):
        ingest(card, journey, day_date="2099-01-01", location="Place A", develop_fn=_developed)


def test_raw_jpeg_pair_dedupes_to_the_raw_without_failing(tmp_path):
    """Pair shooting (DSC_0001.NEF + DSC_0001.JPG) is a standard card layout,
    not a collision: one gallery photo, developed from the RAW, JPEG skipped
    with a reason — never a hard-fail after the first export."""
    journey = _site(tmp_path)
    card = _card(tmp_path, ["DSC_0001.NEF", "DSC_0001.JPG"])
    developed = []

    def _develop(path):
        developed.append(path.name)
        return _developed(path)

    report = ingest(card, journey, develop_fn=_develop, **DAY)
    assert developed == ["DSC_0001.NEF"]
    assert report.ingested == ["dsc_0001.jpg"]
    assert report.skipped == [("DSC_0001.JPG", "RAW+JPEG pair: developed from DSC_0001.NEF")]


def test_raw_jpeg_pair_leaves_unpaired_stems_alone(tmp_path):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["DSC_0001.NEF", "DSC_0001.JPG", "DSC_0002.JPG", "DSC_0003.NEF"])
    report = ingest(card, journey, develop_fn=_developed, **DAY)
    assert report.ingested == ["dsc_0001.jpg", "dsc_0002.jpg", "dsc_0003.jpg"]
    assert [name for name, _ in report.skipped] == ["DSC_0001.JPG"]


def test_raw_jpeg_pair_reingest_is_still_a_noop(tmp_path):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["DSC_0001.NEF", "DSC_0001.JPG"])
    ingest(card, journey, develop_fn=_developed, **DAY)
    first = journey.read_text()
    report = ingest(card, journey, develop_fn=_never, **DAY)
    assert report.ingested == []
    assert sorted(name for name, _ in report.skipped) == ["DSC_0001.JPG", "DSC_0001.NEF"]
    assert journey.read_text() == first


def test_same_kind_stem_collision_still_fails_closed(tmp_path):
    """Two JPEGs (or two RAWs) sharing a stem are genuinely different content —
    only the RAW+developed-sibling pair dedupes."""
    journey = _site(tmp_path)
    jpegs = _card(tmp_path / "j", ["DCIM/100/DSC_0001.JPG", "DCIM/101/DSC_0001.JPG"])
    with pytest.raises(PhotoIngestError, match="collides"):
        ingest(jpegs, journey, develop_fn=_developed, **DAY)
    raws = _card(tmp_path / "r", ["DSC_0002.NEF", "DSC_0002.DNG"])
    with pytest.raises(PhotoIngestError, match="collides"):
        ingest(raws, journey, develop_fn=_developed, **DAY)
    assert journey.read_text() == JOURNEY_MD


def _stub_dates(monkeypatch, dates):
    """EXIF capture dates by source filename; a file absent from the map has none."""
    monkeypatch.setattr(ingest_mod, "extract_exif_capture_date", lambda p: dates.get(Path(p).name))


def _exif_jpeg(path, datetime_original=None):
    """Write a real JPEG, optionally carrying ``DateTimeOriginal`` EXIF."""
    from PIL import Image

    exif = Image.Exif()
    if datetime_original is not None:
        exif[0x8769] = {0x9003: datetime_original}
    Image.new("RGB", (8, 8)).save(path, "JPEG", exif=exif)
    return path


class TestExtractExifCaptureDate:
    def test_reads_datetime_original_as_iso_date(self, tmp_path):
        path = _exif_jpeg(tmp_path / "p.jpg", "2025:01:02 10:30:00")
        assert extract_exif_capture_date(path) == "2025-01-02"

    def test_malformed_absent_or_unreadable_returns_none(self, tmp_path):
        assert extract_exif_capture_date(_exif_jpeg(tmp_path / "bad.jpg", "not a date")) is None
        assert extract_exif_capture_date(_exif_jpeg(tmp_path / "bare.jpg")) is None
        broken = tmp_path / "broken.jpg"
        broken.write_bytes(b"not a jpeg")
        assert extract_exif_capture_date(broken) is None


class TestExifDatePartition:
    """A card is rarely wiped daily. A file from another day must never register
    under this run's --day (wrong travel date), and its GPS must never vote on
    this day's ONE location (cross-county label bleed)."""

    def test_out_of_day_files_skip_before_decode_naming_their_date(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["a.DNG", "b.DNG"])
        _stub_dates(monkeypatch, {"a.DNG": "2025-01-01", "b.DNG": "2025-01-02"})

        def _develop(path):
            assert path.name == "a.DNG", f"develop must never run for out-of-day {path.name}"
            return _developed(path)

        report = ingest(card, journey, develop_fn=_develop, **DAY)
        assert report.ingested == ["a.jpg"]
        assert ("b.DNG", "captured 2025-01-02, not 2025-01-01") in report.skipped
        assert "b.jpg" not in journey.read_text()

    def test_location_vote_runs_over_the_in_day_subset_only(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["a.DNG", "far.DNG"])
        _stub_dates(monkeypatch, {"a.DNG": "2025-01-01", "far.DNG": "2024-12-31"})
        calls = []
        _stub_derive(monkeypatch, PhotoLocation("County Clare, Ireland", RESOLVED), calls)

        report = ingest(card, journey, develop_fn=_developed, **CARD_ONLY)

        # far.DNG's fix never reaches the vote — it belongs to another day.
        assert calls == [["a.DNG"]]
        assert report.location == "County Clare, Ireland"

    def test_missing_exif_date_ingests_under_day_with_warning(self, tmp_path, monkeypatch, caplog):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["nodate.DNG"])
        _stub_dates(monkeypatch, {})
        with caplog.at_level(logging.WARNING):
            report = ingest(card, journey, develop_fn=_developed, **DAY)
        assert report.ingested == ["nodate.jpg"]
        assert any("no readable EXIF capture date" in r.message for r in caplog.records)

    def test_pair_skip_reason_composes_with_the_date_partition(self, tmp_path, monkeypatch):
        """A paired JPEG stays 'pair' skipped — dedupe runs first, so it is
        never double-reported under a date reason."""
        journey = _site(tmp_path)
        card = _card(tmp_path, ["DSC_1.NEF", "DSC_1.JPG", "old.NEF"])
        _stub_dates(
            monkeypatch,
            {"DSC_1.NEF": "2025-01-01", "DSC_1.JPG": "2024-12-30", "old.NEF": "2024-12-30"},
        )
        report = ingest(card, journey, develop_fn=_developed, **DAY)
        assert report.ingested == ["dsc_1.jpg"]
        assert sorted(report.skipped) == [
            ("DSC_1.JPG", "RAW+JPEG pair: developed from DSC_1.NEF"),
            ("old.NEF", "captured 2024-12-30, not 2025-01-01"),
        ]

    def test_reingest_after_partition_is_still_a_noop(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["a.DNG", "b.DNG"])
        _stub_dates(monkeypatch, {"a.DNG": "2025-01-01", "b.DNG": "2025-01-02"})
        ingest(card, journey, develop_fn=_developed, **DAY)
        first = journey.read_text()
        report = ingest(card, journey, develop_fn=_never, **DAY)
        assert report.ingested == []
        assert journey.read_text() == first


# ── Gallery blur — the gallery dir IS the public website surface ─────────

BLUR_PATCH = (60, 40, 50, 50)  # x, y, w, h of the synthetic "face"


@pytest.fixture(autouse=True)
def _passthrough_blur(monkeypatch):
    """Wiring tests own the ingest loop, not the detector: blur passes through
    by default. The gallery-blur tests rebind the real thing (or a spy)."""
    monkeypatch.setattr(
        ingest_mod,
        "blur_faces_for_publish",
        lambda rgb, label, blur=None: (rgb, 0),
        raising=False,
    )


def _striped_rgb():
    """Frame with a STRIPED bright patch — stripes so a real blur must change pixels."""
    img = np.full((200, 200, 3), 90, dtype=np.uint8)
    x, y, w, h = BLUR_PATCH
    img[y : y + h, x : x + w], img[y : y + h : 2, x : x + w] = 255, 205
    return img


def _striped_developed(path):
    return DevelopedPhoto(rgb=_striped_rgb(), path=path, params=None, cluster_index=0)


class _PatchDetector:
    """Fake face detector (no model dep): reports the patch when it is present."""

    def detect(self, frame, conf):
        return [BLUR_PATCH] if (frame[:, :, 0] > 200).any() else []


def _core_std(rgb) -> float:
    """Stripe contrast in the CORE of the patch — collapses towards 0 once blurred."""
    x, y, w, h = BLUR_PATCH
    return float(np.asarray(rgb)[y + h // 3 : y + 2 * h // 3, x + w // 3 : x + 2 * w // 3].std())


def _gallery_rgb(tmp_path, name):
    from PIL import Image

    path = tmp_path / "static" / "images" / "journeys" / "journey-a" / name
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


class TestGalleryBlurBeforeExport:
    """Blur lands BEFORE the gallery JPEG is written, mirroring the video
    contract: blur the master, every derived/published surface inherits it."""

    def _real_blur(self, monkeypatch):
        monkeypatch.setattr(
            ingest_mod, "blur_faces_for_publish", face_filter.blur_faces_for_publish, raising=False
        )

    def test_gallery_jpeg_is_blurred_by_default(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["face.DNG"])
        self._real_blur(monkeypatch)
        monkeypatch.setattr(face_filter, "_get_detectors", lambda: (_PatchDetector(), None))

        report = ingest(card, journey, develop_fn=_striped_developed, **DAY)

        assert report.blurred == 1 and report.blur_skipped is False
        assert _core_std(_gallery_rgb(tmp_path, "face.jpg")) < _core_std(_striped_rgb()) / 2

    def test_skip_blur_exports_unblurred_and_reports_it(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["face.DNG"])
        self._real_blur(monkeypatch)
        monkeypatch.setattr(
            face_filter,
            "_get_detectors",
            lambda: pytest.fail("--skip-blur must never consult a detector"),
        )

        report = ingest(card, journey, develop_fn=_striped_developed, skip_blur=True, **DAY)

        assert report.blur_skipped is True and report.blurred == 0
        assert _core_std(_gallery_rgb(tmp_path, "face.jpg")) > _core_std(_striped_rgb()) / 2

    def test_detector_failure_fails_closed_naming_the_file(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["face.DNG"])

        def _boom(rgb, label, blur=None):
            raise FaceBlurError(f"Privacy detection failed for {label}: session died")

        monkeypatch.setattr(ingest_mod, "blur_faces_for_publish", _boom, raising=False)

        with pytest.raises(PhotoIngestError, match="face.DNG"):
            ingest(card, journey, develop_fn=_striped_developed, **DAY)
        # Fail-closed BEFORE the write: no unblurred file may land in the
        # gallery dir the website renders, and the journey stays untouched.
        gallery = tmp_path / "static" / "images" / "journeys" / "journey-a"
        assert list(gallery.iterdir()) == []
        assert journey.read_text() == JOURNEY_MD

    def test_blur_composes_with_pair_dedupe_and_date_partition(self, tmp_path, monkeypatch):
        """A skipped file (paired JPEG, out-of-day) is never decoded OR blurred."""
        journey = _site(tmp_path)
        card = _card(tmp_path, ["DSC_1.NEF", "DSC_1.JPG", "old.NEF"])
        _stub_dates(monkeypatch, {"DSC_1.NEF": "2025-01-01", "old.NEF": "2024-12-30"})
        seen = []

        def _spy(rgb, label, blur=None):
            seen.append(label)
            return rgb, 1

        monkeypatch.setattr(ingest_mod, "blur_faces_for_publish", _spy, raising=False)

        report = ingest(card, journey, develop_fn=_developed, **DAY)

        assert seen == ["DSC_1.NEF"]
        assert report.ingested == ["dsc_1.jpg"] and report.blurred == 1


def test_cli_registers_skip_blur_on_photos_ingest():
    from gomoveshift.cli.parser import _build_parser

    base = ["photos", "ingest", "card", "--journey", "j", "--day", "2025-01-01"]
    assert _build_parser().parse_args(base).skip_blur is False
    assert _build_parser().parse_args([*base, "--skip-blur"]).skip_blur is True


def test_corrupt_source_fails_loudly_naming_the_file(tmp_path):
    journey = _site(tmp_path)
    card = _card(tmp_path, ["good.jpg", "broken.DNG"])

    def _develop(path):
        if path.name == "broken.DNG":
            raise PhotoDecodeError(f"{path}: rawpy could not decode this file")
        return _developed(path)

    with pytest.raises(PhotoIngestError, match="broken.DNG"):
        ingest(card, journey, develop_fn=_develop, **DAY)
    # Fail-closed: a partial card never half-registers into the journey.
    assert journey.read_text() == JOURNEY_MD


# ── Journey lock + partial-card residue + export error contract ──────────


def _gallery_dir(tmp_path):
    return tmp_path / "static" / "images" / "journeys" / "journey-a"


class TestJourneyFileLock:
    """Every journey writer does its own load -> mutate -> atomic-write, so two
    concurrent writers are a last-writer-wins lost update. Ingest holds a
    sidecar flock on the journey for its whole read -> register -> write
    window, so concurrent ingests exclude each other."""

    def _lock_path(self, journey):
        return journey.with_name(journey.name + ".lock")

    def test_ingest_refuses_while_the_journey_lock_is_held(self, tmp_path):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["a.jpg"])
        fd = os.open(self._lock_path(journey), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, json.dumps({"pid": os.getpid(), "started": "now"}).encode())
            # _never proves the refusal lands before any decode or export.
            with pytest.raises(PhotoIngestError, match="lock"):
                ingest(card, journey, develop_fn=_never, **DAY)
        finally:
            os.close(fd)
        assert journey.read_text() == JOURNEY_MD
        assert list(_gallery_dir(tmp_path).iterdir()) == []

    def test_lock_is_released_and_removed_after_a_run(self, tmp_path):
        journey = _site(tmp_path)
        ingest(_card(tmp_path, ["a.jpg"]), journey, develop_fn=_developed, **DAY)
        assert not self._lock_path(journey).exists()


class TestPartialCardResidueCleanup:
    """Frontmatter registers once at the end, but gallery JPEGs land per-file
    inside the develop loop — so a LATER file failing (decode, blur, export)
    used to strand the earlier files' exports in the public gallery dir,
    unregistered, where a re-run's frontmatter idempotency check never
    reclaims them. On any failure the run removes exactly what it wrote —
    never a pre-existing file — and the journey stays byte-identical."""

    def test_later_decode_failure_removes_earlier_exports(self, tmp_path):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["a.jpg", "z_broken.DNG"])
        preexisting = _gallery_dir(tmp_path) / "photo-001.jpg"
        preexisting.write_bytes(b"registered by an earlier run")

        def _develop(path):
            if path.name == "z_broken.DNG":
                raise PhotoDecodeError(f"{path}: rawpy could not decode this file")
            return _developed(path)

        with pytest.raises(PhotoIngestError, match="z_broken.DNG"):
            ingest(card, journey, develop_fn=_develop, **DAY)
        assert sorted(p.name for p in _gallery_dir(tmp_path).iterdir()) == ["photo-001.jpg"]
        assert preexisting.read_bytes() == b"registered by an earlier run"
        assert journey.read_text() == JOURNEY_MD

    def test_later_blur_failure_removes_earlier_exports(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["a.jpg", "b.jpg"])

        def _blur(rgb, label, blur=None):
            if label == "b.jpg":
                raise FaceBlurError(f"Privacy detection failed for {label}: session died")
            return rgb, 0

        monkeypatch.setattr(ingest_mod, "blur_faces_for_publish", _blur, raising=False)
        with pytest.raises(PhotoIngestError, match="b.jpg"):
            ingest(card, journey, develop_fn=_developed, **DAY)
        assert list(_gallery_dir(tmp_path).iterdir()) == []
        assert journey.read_text() == JOURNEY_MD

    def test_later_export_failure_removes_earlier_exports(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _card(tmp_path, ["a.jpg", "b.jpg"])
        real_export = ingest_mod.export_jpeg

        def _export(rgb, out_path):
            if Path(out_path).name == "b.jpg":
                raise PermissionError(13, "Read-only file system")
            return real_export(rgb, out_path)

        monkeypatch.setattr(ingest_mod, "export_jpeg", _export)
        with pytest.raises(PhotoIngestError, match="b.jpg"):
            ingest(card, journey, develop_fn=_developed, **DAY)
        assert list(_gallery_dir(tmp_path).iterdir()) == []
        assert journey.read_text() == JOURNEY_MD

    def test_unresolvable_location_removes_the_whole_cards_exports(self, tmp_path, monkeypatch):
        """Location resolves only after the whole card develops — the maximal
        residue window: every export already sits in the gallery dir."""
        journey = _site(tmp_path)
        card = _card(tmp_path, ["a.DNG"])
        _stub_derive(monkeypatch, PhotoLocation(reason=NO_EXIF_GPS))
        with pytest.raises(PhotoIngestError, match=NO_EXIF_GPS):
            ingest(card, journey, develop_fn=_developed, **CARD_ONLY)
        assert list(_gallery_dir(tmp_path).iterdir()) == []


# ── Cluster-graded production develop loop ───────────────────────────────


def _image_card(tmp_path, colors):
    """Card of real tiny JPEGs — the production path measures and grades these."""
    from PIL import Image

    card = tmp_path / "card"
    card.mkdir()
    for name, color in colors.items():
        Image.new("RGB", (32, 24), color).save(card / name)
    return card


class TestClusterGradedIngest:
    """Production ingest (no injected develop_fn) grades per lighting cluster:
    same-scene stills share ONE correction instead of drifting shot to shot.
    An injected ``develop_fn`` is the test seam and owns its own grading."""

    WARM, WARM2, COOL = (200, 150, 100), (202, 152, 102), (100, 150, 200)

    @pytest.fixture(autouse=True)
    def _flat_frames_are_the_subject(self, monkeypatch):
        """These tests grade flat solid-colour JPEGs on purpose; the cull floor
        is not what they exercise."""
        monkeypatch.setattr(ingest_mod, "CULL_FLOORS", CullFloors(0.0, 1.0, 1.0))

    def _spy_develop(self, monkeypatch):
        calls = []
        real = develop_mod.develop

        def spy(source, params=None, cluster_index=0):
            calls.append((Path(source).name, params, cluster_index))
            return real(source, params, cluster_index)

        monkeypatch.setattr(ingest_mod, "develop", spy)
        return calls

    def test_same_scene_shares_one_correction_and_odd_scene_differs(self, tmp_path, monkeypatch):
        journey = _site(tmp_path)
        card = _image_card(
            tmp_path, {"s1.jpg": self.WARM, "s2.jpg": self.WARM2, "odd.jpg": self.COOL}
        )
        calls = self._spy_develop(monkeypatch)

        report = ingest(card, journey, **DAY)

        by_name = {n: (dataclasses.astuple(p), c) for n, p, c in calls}
        assert set(by_name) == {"s1.jpg", "s2.jpg", "odd.jpg"}
        assert by_name["s1.jpg"] == by_name["s2.jpg"], "same scene must share ONE correction"
        assert by_name["odd.jpg"][0] != by_name["s1.jpg"][0]
        assert by_name["odd.jpg"][1] != by_name["s1.jpg"][1]
        assert sorted(report.ingested) == ["odd.jpg", "s1.jpg", "s2.jpg"]

    def test_skipped_files_are_never_measured(self, tmp_path, monkeypatch):
        """Dedupe/partition/idempotency decisions land BEFORE the measure pass —
        a skipped file is never decoded, not even at measure resolution."""
        journey = _site(tmp_path)
        card = _image_card(tmp_path, {"new.jpg": self.WARM, "old.jpg": self.COOL})
        _stub_dates(monkeypatch, {"new.jpg": "2025-01-01", "old.jpg": "2024-12-31"})
        measured = []
        real = getattr(develop_mod, "compute_batch_corrections", None)

        def spy(paths, *args, **kwargs):
            measured.append([Path(p).name for p in paths])
            return real(paths, *args, **kwargs)

        monkeypatch.setattr(ingest_mod, "compute_batch_corrections", spy, raising=False)

        ingest(card, journey, **DAY)
        assert measured == [["new.jpg"]]
        # A fully-skipped re-run never reaches the measure pass at all.
        ingest(card, journey, **DAY)
        assert measured == [["new.jpg"]]

    def test_grades_are_per_file_never_an_all_at_once_batch(self, tmp_path, monkeypatch):
        """The OOM contract: each file is graded and exported before the next
        one is decoded — never a held list of graded full-resolution arrays."""
        journey = _site(tmp_path)
        card = _image_card(tmp_path, {"a.jpg": self.WARM, "b.jpg": self.WARM2})
        events = []
        real_export = ingest_mod.export_jpeg

        def dspy(source, *args):
            events.append("develop")
            return develop_mod.develop(source, *args)

        def espy(rgb, out_path, *args, **kwargs):
            events.append("export")
            return real_export(rgb, out_path, *args, **kwargs)

        monkeypatch.setattr(ingest_mod, "develop", dspy)
        monkeypatch.setattr(ingest_mod, "export_jpeg", espy)

        ingest(card, journey, **DAY)
        assert events == ["develop", "export", "develop", "export"]

    def test_dry_run_skips_the_measure_pass(self, tmp_path, monkeypatch):
        """The measure pass counts as a decode, so a dry run must never run it."""
        journey = _site(tmp_path)
        card = _image_card(tmp_path, {"a.jpg": self.WARM})
        fail = lambda *a, **k: pytest.fail("dry run must never measure or develop")  # noqa: E731
        monkeypatch.setattr(ingest_mod, "compute_batch_corrections", fail, raising=False)
        monkeypatch.setattr(ingest_mod, "develop", fail)

        report = ingest(card, journey, dry_run=True, **DAY)
        assert report.ingested == ["a.jpg"] and report.dry_run is True


def test_export_oserror_surfaces_as_photo_ingest_error_naming_the_file(tmp_path, monkeypatch):
    """A missing/readonly gallery dir must not escape as a raw traceback — the
    CLI handler catches PhotoIngestError only."""
    journey = _site(tmp_path)
    card = _card(tmp_path, ["a.jpg"])

    def _export(rgb, out_path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(ingest_mod, "export_jpeg", _export)
    with pytest.raises(PhotoIngestError, match="a.jpg") as excinfo:
        ingest(card, journey, develop_fn=_developed, **DAY)
    assert isinstance(excinfo.value.__cause__, OSError)
    assert journey.read_text() == JOURNEY_MD
