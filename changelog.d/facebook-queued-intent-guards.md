### Fixed

- Facebook queued publishing now refuses unsupported drafts and incompatible
  state/time requests before acquiring an upload lease or contacting Meta.
  Future slots within the minimum scheduling lead wait until due instead of
  publishing early; at/after-slot drains keep their existing behavior.
