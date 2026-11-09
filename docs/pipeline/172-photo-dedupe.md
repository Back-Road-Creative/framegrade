## Photo capture reader and duplicate ledger

A card offloaded twice (renamed copies, overlapping folders) must not be processed twice. Name-keyed dedupe (`ingest._dedupe_raw_pairs`) only sees the same directory. Two new library modules under `gomoveshift/pipeline/photos/` key a frame on what the camera recorded. Nothing calls them yet.

## Camera facts (`capture_meta.py`)

`read_capture_records(paths)` runs exiftool (`-json -n -fast`, paths in a temp argfile, never a shell) in batches of 200 and returns a `CaptureRecord` per path: make, model, body serial, `ShutterCount`, capture time and sub-second, focus distance (metres, `inf` becomes `math.inf`), lens, focal length, f-number, ISO, exposure. Rows are mapped back by `SourceFile`. The Nikon serial and shutter count live in the MakerNote, which Pillow cannot read.

Refusals and fallbacks, all warnings and never exceptions: exiftool missing, a batch that times out or exits non-zero (that batch falls back to the Pillow readers for ISO, exposure, focal length and f-number), a path exiftool did not return (a record with only `path`). Unusable numeric values read `None`.

## Duplicates (`dedupe.py`)

- Identity: `shot:<make>|<model>|<serial>|sc:<shutter count>`, else serial plus capture time and sub-second. Same identity collapses to one keeper: a RAW first, then the shorter path, then lexicographic. A NEF and a JPEG of one shot in two folders collapse the same way.
- A file with no identity is keyed on its sha256 (only those files are hashed); same hash is an exact copy.
- `Ledger` (sqlite, `PRAGMA user_version = 1`, WAL off) remembers keys between runs: a key already there is `seen_before` unless `reprocess=True`. `dedupe` never writes it; the caller calls `record()` after a successful run. First sighting wins. It also caches per-file scores by sha256 and name.
- The ledger's parent directory must exist, and a database with a newer `user_version` is refused. Originals are only opened read-only.

## Verification

`tests/test_photo_capture_meta.py` (fake exiftool runner; one real run on a tiny JPEG, skipped when exiftool is absent) and `tests/test_photo_dedupe.py`.
