## `--fresh`

`_try_resume._purge_stale_stage_artifacts` deletes `drive_plan.json` + `pipeline_state.json` from both `stages_dir` and the legacy `output_base/_stages` mirror before a fresh run (a stale plan from a different source would reassemble LFs from the wrong segments). Does NOT delete `drive_scores.npz`. Each deletion logs `--fresh: cleared stale stage artifact`.

`auto_populate_per_day_completed(..., fresh=...)` (keyword required) is a no-op on `--fresh`: it would otherwise re-add a `{"master_path": None}` recovery marker for every `day_<date>/` with finals on disk, and Phase BC's finals-only fast-path would reuse the prior run's burned LFs and shorts (shipping shorts that carry a stale, wrong place). Day finals stay on disk; a fresh run re-cuts them, and the per-LF OST fingerprint still skips re-burning an unchanged LF.
