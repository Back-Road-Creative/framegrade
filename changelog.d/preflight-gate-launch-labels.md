### Fixed

- The launch-time LF metadata gate now uses each LF's launch-time label, the one Phase D will read. A resumed county-fallback drive (empty drive-wide label, no labels in state yet) used to fail every retry with `LongFormMetadataGenerator.generate: location is empty`.
