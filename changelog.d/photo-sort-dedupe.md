### Added

- Photo duplicate detection building blocks (not yet wired to a command): an exiftool capture reader (body serial, shutter count, focus distance, lens; Pillow fallback with a warning when exiftool is missing) and a dedupe module that collapses renamed or cross-folder copies of one shot by capture identity (RAW kept over JPEG), hashes only frames without an identity, and remembers processed shots in a sqlite ledger across runs.
