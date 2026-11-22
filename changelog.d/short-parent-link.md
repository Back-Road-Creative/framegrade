### Added

- A reviewed Short's description now ends with `Full drive: https://youtu.be/<id>`, a link to its parent drive that goes through Waypost's final-text review. No link when the record has no valid parent id or the Short is scheduled before its parent, or either time has no timezone.

### Changed

- Pin Waypost (`seo-manager`) to fe920eb in `pyproject.toml` and `ci/siblings.txt`. Its drive templates moved to version 5, so a final-text review stored under version 4 reads as stale. A queued Short is re-reviewed at publish, so it is unaffected. Each LF's `*_youtube.json` carries the review's text policy, so re-running Phase D on a drive whose Gate C/D was already confirmed changes the gate hash and asks for confirmation again.
