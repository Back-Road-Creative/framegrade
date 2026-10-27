### Changed

- The stabilise step is judged on camera motion instead of luma jitter, which cannot see a translating image. It keeps the vidstab output only when the shake residual (the high-frequency part of the phase-correlation camera trajectory, so a deliberate pan is not shake) drops to at most 75% of the input's and black-border coverage stays at or under 5%; otherwise it falls back to the input and logs why. Deflicker keeps the luma-jitter gate. The picture-steps record gains `input_residual`, `output_residual` and `border_coverage` for the stabilise step.
