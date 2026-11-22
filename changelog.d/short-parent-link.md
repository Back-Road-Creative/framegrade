### Added

- A reviewed Short's description now ends with `Full drive: https://youtu.be/<id>`, a link to its parent drive that goes through Waypost's final-text review. No link when the record has no valid parent id or the Short is scheduled before its parent.

### Changed

- Pin Waypost (`seo-manager`) to fe920eb in `pyproject.toml` and `ci/siblings.txt`. Its drive templates moved to version 5, so a final-text review stored under version 4 reads as stale. A queued Short is re-reviewed at publish, so it is unaffected.
