## `health heal`: a resume takes the backlog row's GPX when the recipe's is gone (`gomoveshift/health/heal.py`)

After code-root binding (`156-health-check.md`), `heal.refresh_gpx` checks the recipe's
`--day-gpx` (one file) and `--bracket-gpx` (two files). A file the recipe names that no longer
exists is replaced by the drive's `drive_backlog.yaml` row's own (`day_gpx` / `bracket_gpx`,
found by `retry_policy.default_manifest_row`), when every file the row names exists; the
attempt records `gpx_refreshed` (flag -> new paths). A recipe GPX that still exists is kept:
the checkpoint was built from it. With no row, or a row whose files are missing too, the
recipe is left as it is and the drive CLI refuses it as before.

Before, gomoveshiftsite#334's rename of the Ireland GPX files left Galley Head's resume naming
`ireland-2025-09-enhanced-day14.gpx`, and the drive CLI refused it, exit 2 (2026-11-26).
Tests: `tests/test_health_heal.py`.
