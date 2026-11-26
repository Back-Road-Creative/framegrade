## `health heal`: a resume takes the backlog row's GPX when the recipe's is gone (`gomoveshift/health/heal.py`)

Before code-root binding (`156-health-check.md`), `heal.refresh_recipe` checks the recipe's
inputs: the drive's source directory, `--day-gpx` (one file), `--bracket-gpx` (two files),
`--map-inset-bundle` and `--trail-profile`. A GPX the recipe names that no longer exists is
replaced by the drive's `drive_backlog.yaml` row's own (`day_gpx` / `bracket_gpx`, found by
`retry_policy.default_manifest_row`), when every file the row names exists; the attempt records
`gpx_refreshed` (flag -> new paths) and the replaced value is written back into `resume.json`,
so the next replay names the current file. A recipe GPX that still exists is kept: the
checkpoint was built from it. A gone input with no current replacement (no row, a row whose
files are missing too, or an input the row does not own) is refused, the attempt naming the
file, and nothing is launched.

Every replay of a recipe takes this one step: heal, the confirmed-audit resume below,
`health publish` (Phase D slices `--day-gpx` too) and `drive --from-recipe`. Guard:
`tests/test_recipe_replay_inputs.py`.

Before, gomoveshiftsite#334's rename of the Ireland GPX files left Galley Head's resume naming
`ireland-2025-09-enhanced-day14.gpx`, and the drive CLI refused it, exit 2 (2026-11-26).
Tests: `tests/test_health_heal.py`.

`health backlog` resumes a drive whose OST audit was confirmed (`launch._resume_confirmed`) the
same way: the region's backlog row for that project goes to `refresh_recipe`, and the result records
`gpx_refreshed`. Before, that resume bound the recipe as written, so Kerry-to-Cork, Old Head and
Galley Head, whose recipes all name pre-rename files, would have relaunched into the same exit 2
once confirmed (2026-11-26). Tests: `tests/test_health_backlog_launch.py`
(`test_a_confirmed_resume_*`).
