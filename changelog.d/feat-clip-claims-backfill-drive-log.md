### Changed

- `clip-claims backfill` now also covers outputs written before `lf_provenance.json` named source clips: with no per-LF clips (or no provenance file) it claims the camera files named in the folder's `drive.log` for every uploaded long-form, tagged `basis: drive_log_project`. The reuse gate matches clip names by case-folded stem (a `GX010064` claim catches `GX010064.MP4`) and its refusal marks a whole-project claim as coarse.
