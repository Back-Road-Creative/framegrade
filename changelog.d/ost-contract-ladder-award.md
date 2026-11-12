### Fixed

- A GPS-less drive whose bracket route only proves the country no longer fails
  at the long-form burn. The OST-burn contract check refused any per-LF label
  equal to the drive-wide `--location`, so a bracket label of "United States"
  on a `--region usa` drive stopped the day after its master encode. The check
  now lets through an LF whose `per_lf_tier_evidence` records the label as the
  ladder's own award (bracket route or placeless operator label), and still
  refuses every other LF.
