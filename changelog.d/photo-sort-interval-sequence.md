### Fixed

- `photos sort` groups an intervalometer run outside the night sky, such as a dawn sequence, as one `sequence-<n>` with its own time-lapse. Before, look-alike bursts split it into several groups and loose shots. A run needs 24 or more frames of one category, at most 60 s apart, at a steady cadence.
