"""The EXIF readers must see camera RAW, or the ingest date partition is dead.

Pillow cannot open a Nikon ``.NEF`` at all, so ``extract_exif_capture_date``
read ``None`` for every RAW on a real card: the date partition never fired, its
"no readable capture date" warning was the normal path, and a card spanning two
days registered everything under one ``--day``. Same blind spot made
``extract_exif_capture_metadata`` return an all-``None`` record, so ISO-aware
noise and golden-hour selection were dead on RAW sources.

A RAW *is* a TIFF container: its Exif sub-IFD is plain TIFF metadata sitting in
front of the mosaic. The fixtures below build one by hand — a real Exif sub-IFD
behind a compression tag Pillow refuses, exactly like a NEF — so the parser is
covered without committing a 15 MB binary.
"""

import datetime
import json
import logging
import os
import struct
from pathlib import Path

import pytest
from PIL import Image

from gomoveshift.pipeline.photos.exif_gps import (
    extract_exif_capture_date,
    extract_exif_capture_metadata,
)
from gomoveshift.pipeline.photos.ingest import ingest
from gomoveshift.pipeline.photos.scene import PhotoExif
from test_photo_ingest import DAY, _developed, _site

CARD = Path(os.environ.get("GMS_TEST_RAW_CARD_DIR") or "/nonexistent-raw-card")


def _load_nef_manifest() -> dict:
    """The optional "nef" entry of GMS_TEST_RAW_MANIFEST: {"name", "date", "iso"}."""
    path = os.environ.get("GMS_TEST_RAW_MANIFEST")
    if not path:
        return {}
    try:
        return dict(json.loads(Path(path).read_text()).get("nef", {}))
    except (OSError, ValueError, AttributeError):
        return {}


_NEF = _load_nef_manifest()


def _ifd(entries, base):
    """One little-endian TIFF IFD at file offset ``base``; a value wider than
    4 bytes spills onto a heap right after the directory. Entries are
    ``(tag, type, count, value bytes)``."""
    heap_at = base + 2 + 12 * len(entries) + 4
    body, heap = b"", b""
    for tag, typ, count, raw in sorted(entries):
        if len(raw) > 4:
            payload, heap = struct.pack("<I", heap_at + len(heap)), heap + raw
        else:
            payload = raw.ljust(4, b"\0")
        body += struct.pack("<HHI", tag, typ, count) + payload
    return struct.pack("<H", len(entries)) + body + struct.pack("<I", 0) + heap


def _raw(
    path,
    *,
    when="2026:07:22 10:16:15",
    iso=100,
    exposure=(1, 1600),
    f_number=None,
    focal_length=None,
):
    """A NEF-like file: a TIFF container whose IFD0 points at a real Exif
    sub-IFD and whose compression is Nikon's (34713), which Pillow refuses.

    ``f_number`` is an EXIF rational like the shutter — ``(56, 10)`` is f/5.6 —
    and defaults to absent, which is what a card with no aperture tag looks like.
    ``focal_length`` is the same shape in mm (``(400, 1)`` is 400 mm), likewise
    absent by default, so every existing fixture is byte-identical.
    """
    sub = []
    if when is not None:
        sub.append((0x9003, 2, len(when) + 1, when.encode() + b"\0"))
    if iso is not None:
        sub.append((0x8827, 3, 1, struct.pack("<H", iso)))
    if exposure is not None:
        sub.append((0x829A, 5, 1, struct.pack("<II", *exposure)))
    if f_number is not None:
        sub.append((0x829D, 5, 1, struct.pack("<II", *f_number)))
    if focal_length is not None:
        sub.append((0x920A, 5, 1, struct.pack("<II", *focal_length)))
    sub_at = 8 + 2 + 12 * 2 + 4
    ifd0 = _ifd(
        [
            (0x0103, 3, 1, struct.pack("<H", 34713)),
            (0x8769, 4, 1, struct.pack("<I", sub_at)),
        ],
        8,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"II\x2a\x00" + struct.pack("<I", 8) + ifd0 + _ifd(sub, sub_at))
    return path


def _camera_shaped_raw(path, *, when="2026:07:22 10:16:15", iso=100):
    """A NEF-like file laid out the way a camera writes one.

    Unlike ``_raw``, IFD0 carries the usual identification tags (make, model,
    orientation), the Exif sub-IFD carries an opaque maker note next to the
    date and ISO, and a large image payload follows the directories. The
    parser must find the sub-IFD through the pointer, not by position.
    """
    make = b"SYNTH CAMERA CO\0"
    model = b"SYNTH Z0\0"
    sub = [
        (0x9003, 2, len(when) + 1, when.encode() + b"\0"),
        (0x8827, 3, 1, struct.pack("<H", iso)),
        (0x927C, 7, 64, bytes(range(64))),
    ]

    def ifd0_entries(sub_at):
        return [
            (0x010F, 2, len(make), make),
            (0x0110, 2, len(model), model),
            (0x0112, 3, 1, struct.pack("<H", 1)),
            (0x0103, 3, 1, struct.pack("<H", 34713)),
            (0x8769, 4, 1, struct.pack("<I", sub_at)),
        ]

    sub_at = 8 + len(_ifd(ifd0_entries(0), 8))
    payload = b"\xa5" * (256 * 1024)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"II\x2a\x00"
        + struct.pack("<I", 8)
        + _ifd(ifd0_entries(sub_at), 8)
        + _ifd(sub, sub_at)
        + payload
    )
    return path


def _jpeg(path, when="2026:07:20 19:05:26"):
    exif = Image.Exif()
    exif[0x8769] = {0x9003: when, 0x8827: 1250, 0x829A: 1 / 60}
    Image.new("RGB", (8, 8)).save(path, "JPEG", exif=exif)
    return path


class TestRawIsReadable:
    def test_the_fixture_is_a_container_pillow_cannot_open(self, tmp_path):
        """Pins WHY this file exists: the Pillow path cannot reach these tags."""
        with pytest.raises(Exception):
            Image.open(_raw(tmp_path / "a.NEF")).load()

    def test_capture_date_reads_off_a_raw(self, tmp_path):
        assert extract_exif_capture_date(_raw(tmp_path / "a.NEF")) == "2026-07-22"

    def test_capture_metadata_reads_off_a_raw(self, tmp_path):
        meta = extract_exif_capture_metadata(_raw(tmp_path / "a.NEF", iso=1250))
        assert (meta.capture_hour, meta.iso) == (10, 1250)
        assert meta.exposure_s == pytest.approx(1 / 1600, rel=1e-3)

    def test_a_camera_shaped_raw_reads_its_capture_date(self, tmp_path):
        """Make/model tags, a maker note and a large payload must not hide the
        Exif sub-IFD: the date and ISO read back exactly."""
        path = _camera_shaped_raw(tmp_path / "shot.NEF")
        assert extract_exif_capture_date(path) == "2026-07-22"
        assert extract_exif_capture_metadata(path).iso == 100

    @pytest.mark.skipif(
        not sorted(CARD.glob("*.NEF")),
        reason="set GMS_TEST_RAW_CARD_DIR to a directory holding a real NEF to run",
    )
    def test_a_real_nef_reads_its_capture_date_and_iso(self):
        """A real camera file, not a hand-built container, must yield a date and ISO,
        and the exact recorded ones when the manifest's "nef" entry names them."""
        path = CARD / _NEF["name"] if "name" in _NEF else sorted(CARD.glob("*.NEF"))[0]
        date = extract_exif_capture_date(path)
        assert date is not None
        datetime.date.fromisoformat(date)
        meta = extract_exif_capture_metadata(path)
        assert isinstance(meta.iso, int) and meta.iso > 0
        if "date" in _NEF:
            assert date == _NEF["date"]
        if "iso" in _NEF:
            assert meta.iso == int(_NEF["iso"])


class TestUnchangedAndNeverRaising:
    def test_a_jpeg_parses_exactly_as_before(self, tmp_path):
        path = _jpeg(tmp_path / "p.jpg")
        assert extract_exif_capture_date(path) == "2026-07-20"
        meta = extract_exif_capture_metadata(path)
        assert (meta.capture_hour, meta.iso) == (19, 1250)

    @pytest.mark.parametrize(
        "name,payload",
        [
            ("trunc", lambda p: _raw(p).write_bytes(p.read_bytes()[:20])),
            ("notag", lambda p: _raw(p, when=None, iso=None, exposure=None)),
            ("garbage", lambda p: p.write_bytes(b"MM\x00\x2a" + b"\xff" * 60)),
        ],
    )
    def test_corrupt_or_dateless_raw_yields_none_without_raising(self, tmp_path, name, payload):
        path = tmp_path / f"{name}.NEF"
        payload(path)
        assert extract_exif_capture_date(path) is None
        assert extract_exif_capture_metadata(path) == PhotoExif()

    def test_a_malformed_raw_date_stays_per_field(self, tmp_path):
        """Never-raise is per field: a junk date must not take ISO down with it."""
        path = _raw(tmp_path / "badday.NEF", when="not a date")
        assert extract_exif_capture_date(path) is None
        meta = extract_exif_capture_metadata(path)
        assert (meta.capture_hour, meta.iso) == (None, 100)


class TestPartitionOnRawCards:
    """The point of the fix: the partition now actually fires on a RAW card."""

    def test_an_out_of_day_raw_is_skipped_naming_its_date(self, tmp_path):
        journey = _site(tmp_path)
        card = tmp_path / "card"
        _raw(card / "today.NEF", when="2025:01:01 08:00:00")
        _raw(card / "yesterday.NEF", when="2024:12:31 08:00:00")

        def _develop(path):
            assert path.name == "today.NEF", f"out-of-day {path.name} must never decode"
            return _developed(path)

        report = ingest(card, journey, develop_fn=_develop, **DAY)
        assert report.ingested == ["today.jpg"]
        assert ("yesterday.NEF", "captured 2024-12-31, not 2025-01-01") in report.skipped

    def test_a_dateless_file_still_ingests_under_day_with_the_warning(self, tmp_path, caplog):
        journey = _site(tmp_path)
        card = tmp_path / "card"
        _raw(card / "nodate.NEF", when=None)
        with caplog.at_level(logging.WARNING):
            report = ingest(card, journey, develop_fn=_developed, **DAY)
        assert report.ingested == ["nodate.jpg"]
        assert any("no readable EXIF capture date" in r.message for r in caplog.records)
