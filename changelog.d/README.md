# Changelog fragments

Each change adds one file here instead of editing `CHANGELOG.md`. Two PRs that
both edit `CHANGELOG.md` conflict there, and a merge queue stops at the second
one. Two PRs that each add their own file never conflict.

A fragment is `changelog.d/<short-name>.md`, holding one or more Keep a
Changelog sections (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`,
`Security`), each with its bullets:

```markdown
### Fixed

- What was wrong, and what happens now.
```

`python3 -I tools/release.py X.Y.Z` folds every fragment into the release's
section of `CHANGELOG.md`. Fragments are taken in file-name order, and entries
already under `[Unreleased]` come first. The tool then deletes the fragments.
`tests/test_changelog.py` refuses a fragment the release would refuse: a missing
heading, an unknown section, or a section with no entries. It also refuses a name
starting with `deploy`, `publish` or `upload`: the merge robot holds those paths
for a human, which would stop the release PR that deletes the fragment. This README is never
folded in.
