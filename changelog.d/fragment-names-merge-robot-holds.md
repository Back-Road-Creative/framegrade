### Changed

- The release tool and `tests/test_changelog.py` refuse a `changelog.d` fragment whose name starts with `deploy`, `publish` or `upload`. The merge robot holds those paths for a human, so such a fragment stopped release 0.3.116's PR.
