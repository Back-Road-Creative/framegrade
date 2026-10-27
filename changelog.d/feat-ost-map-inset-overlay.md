### Added

- `apply_ost` takes an optional `map_inset` (`MapInsetOverlay`: an RGBA clip plus an x/y position). It is composited after the title and logo in the same encode, in explicit 10-bit formats so a 10-bit master keeps its gradients; the inset drops away when its clip ends. A missing clip is an error, never a silent encode without it. Without `map_inset` the ffmpeg command is unchanged.
