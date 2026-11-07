### Fixed

- When the route-claim check broadens a long-form's burned label (for example to the state), the LF's location label now broadens with it. Phase D used to keep the narrower county on the `📍 Location:` line, so the location invariant failed after the full encode (US-64 LF 1: burn "North Carolina", location "Transylvania County, North Carolina", Henderson County 37% of the footage).
