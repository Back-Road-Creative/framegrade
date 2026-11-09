### Fixed

- A placeless label may say nothing the footage cannot prove, so `--placeless-label` now refuses any word outside `PLACELESS_NO_CLAIM_WORDS` (scenery, weather, time of day, road kind) at argparse, naming the claiming words, instead of burning it into the OST for a person to catch at the audit gate. A `drive_backlog.yaml` row carrying such a label was skipped silently; `gomoveshift health check` now opens a `placeless_label_unverifiable` incident (row, label, claiming words) for it, open while the yaml carries the label and closed once it does not. The other rows still launch.
