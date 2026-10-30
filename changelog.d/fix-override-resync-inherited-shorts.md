### Fixed

- A `--per-lf-location` correction on a Phase-D-only resume now reaches shorts whose burned label was inherited from their parent long-form. Phase D read every short's label from the day's shorts OST audit sidecar, which a resume never re-derives, so the corrected text stayed in those shorts' metadata. A short whose own clip GPS named the place is unchanged.
