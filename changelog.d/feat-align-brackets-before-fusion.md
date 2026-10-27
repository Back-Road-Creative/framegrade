### Added

- `photos develop --hdr` now aligns each bracket frame onto the middle exposure before the merge (median-threshold bitmap translation, applied at the frame's own bit depth), so a handheld bracket no longer ghosts. A shift over 3% of the shorter side is skipped and recorded, a tripod bracket is bit-identical to before, and `--no-bracket-align` / `photos.develop_hdr_align = false` restores the unaligned merge. Rotation and deghosting are not handled. See `docs/pipeline/83-photo-pipeline.md`.
