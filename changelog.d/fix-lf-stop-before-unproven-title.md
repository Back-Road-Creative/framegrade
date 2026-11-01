### Fixed

- A long-form whose Waypost title decision is `facts_unavailable`, or whose locked-GPS fix density is under the 60% tier-1 floor, now stops before the OST burn instead of falling back and burning an unproven place name. `GMS_ALLOW_UNPROVEN_TITLE=1` overrides.
