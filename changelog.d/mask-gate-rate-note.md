### Fixed

- A day the blur-mask gate cannot count because its master runs at another frame rate now says so. The `mask_frame_count_unknown` reason names the master's rate and the mask's (Galley Head: 29.970 fps against 59.939 fps), instead of only `container frames None`, which read the same as a master that could not be probed.
