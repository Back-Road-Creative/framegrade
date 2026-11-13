### Fixed

- `--placeless-label` no longer overwrites the measured label of a long-form that has GPS; it labels only the GPS-less ones, and one warning names the long-forms where it was withheld.
- The unattended backlog launcher skips every row carrying `placeless_label` (claim-free or not) and tells the operator to give `bracket_gpx` instead; `gomoveshift health check` rule 5 reports each such row. A placeless label stays a hand-run last resort.
