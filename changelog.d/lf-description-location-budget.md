### Fixed

- A GPS-less drive labelled by its region alone ("Ireland") no longer fails Phase D after its encode with `description redundancy gate failed … location 'Ireland' appears 4×` (ireland-2025-09-19-early-clonakilty, 12 h lost). The location budget (2×: hook + `📍 Location:`) now counts only the text above the description template's divider; below it the config footer (`▶ Full Ireland playlist`, the cadence line) and the `#Ireland` hashtag name the region by construction. A restatement above the divider still fails.
