### Fixed

- The final-text review sent a region claim's `contains` as its required places only, so a region whose places were all retained under the required floor went to Waypost empty and was refused as `region_claim_malformed` (Galley Head LF 2). A region claim now lists every route place inside it, required ones first. Gaps, coverage and outside samples are unchanged, so an unproven gap still refuses.
