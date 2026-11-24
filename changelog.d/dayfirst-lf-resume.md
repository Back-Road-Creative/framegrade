### Fixed

- A resume now re-runs a day that was recorded complete with fewer long-forms than planned, even when Phase B/C was checkpointed: the orchestrator drops the checkpoint when a planned long-form index has no non-empty final. A Galley Head style job therefore heals on retry instead of failing Phase D with `shorts_per_lf SPLIT-BRAIN` every time. A full_drive day with no supplied plan now builds a montage instead of dropping its long-form.
- A repeated identical ffmpeg stats line (frozen time and size) now counts as a stall, not progress; the montage prepend concat has its own size-based ceiling and `-nostdin`.
