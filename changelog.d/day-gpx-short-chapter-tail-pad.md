### Fixed

- The day-GPX slicer no longer refuses a drive whose recording ends in a short chapter. Its 15 s + 120 s padding was over the span gate's 120 s floor for any clip under 540 s, so a fully covered 198 s last chapter failed as "GPX likely belongs to a different drive". The tail now shrinks (down to 105 s) to fit the unchanged gate.
