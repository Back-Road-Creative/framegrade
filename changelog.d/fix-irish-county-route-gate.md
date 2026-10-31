### Fixed

- A route through two counties that carry no state (such as Irish counties) now labels itself "County A & County B, <country>" from the GPS already retained, instead of being refused before encode.
- On such stateless-county routes, a GPS gap or an unresolved bin with the same county on both sides is counted inside that county. A gap or bin next to an unresolved neighbour, or between two different counties, still refuses.
