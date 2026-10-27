### Changed

- Changelog entries are now fragments in `changelog.d/<name>.md` (one or more `### <Section>` blocks), not edits to `CHANGELOG.md`, so parallel PRs no longer conflict on it. `tools/release.py` folds the fragments into the release by section (Keep a Changelog order, entries already under `[Unreleased]` first), then deletes them. A fragment with no heading, an unknown section or no entries is refused by the release tool and by `tests/test_changelog.py`.
