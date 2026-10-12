"""A resized export can name the source photo it was made from.

The resized batch is the last export surface that carried no identity.

``resize_batch`` already writes ``photos_privacy_attestation.json`` per batch,
so a publish gate can ask "what was proven about this week's exports".
What it could not ask is "which source photo is THIS file", and the answer
mattered for exactly the reason the video side needed it: the published copy is
not the source. It is centre-cropped, blurred, re-encoded and renamed -- given
a Drive link and a complaint, nothing on disk led back to the original.

Same contract as every other provenance record: it **records; it never gates**. A
write failure is logged and dropped, because by the time it runs the JPEGs are
already written, and a record that can abort an export is a record deciding an
outcome.
"""

import json
from pathlib import Path

import pytest
from PIL import Image

from gomoveshift.pipeline.photos import resizer as rz

WEEK = "2026-02-16"


def _photo(path: Path, size=(1200, 800), colour=(10, 120, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, "JPEG", quality=90)
    return path


@pytest.fixture
def batch(tmp_path, monkeypatch):
    """A real two-photo resize into a temp output base."""
    monkeypatch.setattr(rz, "blur_faces_for_publish", None, raising=False)
    src = tmp_path / "src"
    photos = [
        _photo(src / "alpha.jpg", colour=(10, 120, 200)),
        _photo(src / "beta.jpg", colour=(200, 40, 40)),
    ]
    r = rz.PhotoResizer(output_base=tmp_path / "out")
    written = r.resize_batch([str(p) for p in photos], WEEK)
    return tmp_path / "out" / WEEK, photos, written


def _doc(out_dir: Path) -> dict:
    return json.loads((out_dir / rz.PHOTO_PROVENANCE_NAME).read_text())


def test_the_batch_writes_a_provenance_document(batch):
    out_dir, _photos, _written = batch
    assert (out_dir / rz.PHOTO_PROVENANCE_NAME).exists()


def test_one_entry_per_exported_file(batch):
    out_dir, _photos, written = batch
    doc = _doc(out_dir)
    assert len(doc["photos"]) == len(written)


def test_each_entry_names_its_source(batch):
    """The whole point: from an exported filename, reach the original."""
    out_dir, photos, _written = batch
    by_source = {e["source"]: e for e in _doc(out_dir)["photos"]}
    for p in photos:
        assert str(p) in by_source


def test_each_entry_names_the_file_that_was_actually_written(batch):
    out_dir, _photos, written = batch
    names = {e["exported"] for e in _doc(out_dir)["photos"]}
    assert names == {p.name for p in written}
    for name in names:
        assert (out_dir / name).exists()


def test_the_source_fingerprint_changes_when_the_source_does(tmp_path, monkeypatch):
    """A cheap size:mtime identity, like the master's -- enough to tell "a
    different file is now in play", never a tamper-proof hash."""
    monkeypatch.setattr(rz, "blur_faces_for_publish", None, raising=False)
    src = _photo(tmp_path / "src" / "one.jpg")
    r = rz.PhotoResizer(output_base=tmp_path / "out")
    r.resize_batch([str(src)], WEEK)
    first = _doc(tmp_path / "out" / WEEK)["photos"][0]["source_fingerprint"]

    _photo(src, size=(900, 600), colour=(5, 5, 5))
    r.resize_batch([str(src)], WEEK)
    second = _doc(tmp_path / "out" / WEEK)["photos"][0]["source_fingerprint"]
    assert first != second


def test_it_carries_the_week_it_covers(batch):
    out_dir, _photos, _written = batch
    assert _doc(out_dir)["week_start"] == WEEK


def test_it_sits_beside_the_privacy_attestation_it_shares_a_batch_with(batch):
    """Both answer questions about the same exports; splitting them across
    directories is how one goes stale without the other noticing."""
    out_dir, _photos, _written = batch
    assert (out_dir / rz.PHOTO_PRIVACY_ATTESTATION_NAME).exists()
    assert (out_dir / rz.PHOTO_PROVENANCE_NAME).exists()


def test_a_missing_source_is_recorded_with_a_reason_not_crashed(tmp_path, monkeypatch):
    """A source deleted between the resize and the record leaves the entry, the
    fingerprint null, and the reason -- never a null nobody can explain."""
    monkeypatch.setattr(rz, "blur_faces_for_publish", None, raising=False)
    src = _photo(tmp_path / "src" / "gone.jpg")
    r = rz.PhotoResizer(output_base=tmp_path / "out")

    real = rz._source_fingerprint

    def vanish(path):
        Path(path).unlink(missing_ok=True)
        return real(path)

    monkeypatch.setattr(rz, "_source_fingerprint", vanish)
    r.resize_batch([str(src)], WEEK)

    entry = _doc(tmp_path / "out" / WEEK)["photos"][0]
    assert entry["source_fingerprint"] is None
    assert entry["null_reasons"]["source_fingerprint"]


def test_a_write_failure_does_not_fail_the_batch(tmp_path, monkeypatch, caplog):
    """The JPEGs are already on disk; a record must not undo an export."""
    monkeypatch.setattr(rz, "blur_faces_for_publish", None, raising=False)
    src = _photo(tmp_path / "src" / "one.jpg")

    def boom(*_a, **_k):
        raise OSError("read-only")

    monkeypatch.setattr(rz, "_write_provenance", boom)
    r = rz.PhotoResizer(output_base=tmp_path / "out")
    written = r.resize_batch([str(src)], WEEK)  # must not raise
    assert len(written) == 1


def test_a_batch_that_dies_partway_still_records_what_it_wrote(tmp_path, monkeypatch):
    """Same reasoning as the privacy attestation being written before the first
    decode: the bytes that reached disk are facts, and a failure later in the
    batch does not unmake them."""
    monkeypatch.setattr(rz, "blur_faces_for_publish", None, raising=False)
    src = tmp_path / "src"
    good = _photo(src / "good.jpg")
    bad = src / "bad.jpg"
    bad.write_text("not an image")

    r = rz.PhotoResizer(output_base=tmp_path / "out")
    with pytest.raises(Exception):
        r.resize_batch([str(good), str(bad)], WEEK)

    doc = _doc(tmp_path / "out" / WEEK)
    assert [e["source"] for e in doc["photos"]] == [str(good)]


def test_it_records_never_gates(tmp_path, monkeypatch):
    """No entry, however bad, may change what resize_batch returns."""
    monkeypatch.setattr(rz, "blur_faces_for_publish", None, raising=False)
    src = _photo(tmp_path / "src" / "one.jpg")
    monkeypatch.setattr(rz, "_source_fingerprint", lambda _p: None)
    r = rz.PhotoResizer(output_base=tmp_path / "out")
    assert len(r.resize_batch([str(src)], WEEK)) == 1
