### Fixed

- The health loop (heal resume, `health publish`, `health backlog`, retry fingerprints and the install-pending check) now uses the published venv, `.venvs/current`, else `.venv`. The staging sync leaves `.venv` on the old venv while any process runs through it, so on 2026-11-27 every launch was refused as `install pending: checkout pins 0.3.177, venv has 0.3.174` although 0.3.177 was installed and published.
