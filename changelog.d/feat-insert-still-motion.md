### Added

- `plan_inserts(..., still_motion="push_in")` renders still inserts with the existing eased push-in (up to the existing maximum scale) as a per-frame card sequence; the default `"none"` keeps the static card and encode unchanged.
- `InsertPlan.upscaled` and `InsertPlan.upscale_factor` record whether, and by how much, a clip insert is upscaled to the long-form frame.
