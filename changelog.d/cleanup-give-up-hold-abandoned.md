### Fixed

- `gomoveshift cleanup <project> --abandoned` now reclaims a run that `health backlog` gave up. Its `no-auto-resume` hold was read as an operator's "waiting" hold, and the age gate refused it too, so the 57 GB of a given-up drive stayed on the stage volume while the next drive failed the work-space gate 27.6 GB short. An operator's own hold still refuses.
