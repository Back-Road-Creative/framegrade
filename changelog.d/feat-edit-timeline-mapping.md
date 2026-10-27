### Added

- `EditTimeline` in the same module: `output_to_source` (none in an insert hold, two weighted positions in a crossfade) and `source_to_output` (`None` for a cut second, never interpolated), built from a receipt plus optional `Crossfade` and `Insert` extras.

### Changed

- The shorts edited-to-source mapper, the chapter source-to-edited mapper and the assembler's chapter trim adjustment now share `EditTimeline`; their signatures and answers are unchanged. `MarkerMerger.map_to_edited_time` takes an optional `crossfade` (default 0, which is all the chapter generator can pass).
