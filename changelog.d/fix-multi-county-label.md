### Fixed

- A long-form whose retained GPS runs through exactly two counties of one state is now labelled `"A & B Counties, ST"` instead of falling back to the bare state. The route-claim fallback ladder gains a `counties` level between `county` and `state`; its boundary proof tests the route against the union of the two Census county polygons, the claim reaches Waypost's region claims, and the title and place-text gates accept the pair as naming the route. Ace Basin LF 1 (2026-02-21, Beaufort and Colleton) shipped as "South Carolina in 4K" because no such level existed.
