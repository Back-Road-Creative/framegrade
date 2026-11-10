### Fixed

- `photos sort` no longer ranks lens-cap darks as night-sky shots. The night frames go through the stack's light/dark triage once; the darks move to a `lens-cap-darks/` folder, join no group, and show on the page as a count only. Darks shot within 15 minutes of a night session still calibrate its star trail and time-lapse.
