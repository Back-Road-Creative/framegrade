### Fixed

- An Irish county's boundary proof measures the route's reach against every other county on the Tailte map instead of the county's own edge, which ends at the shore, so a coastal drive such as Galley Head is no longer refused as a "possible boundary crossing" into the sea. Counties on the Northern Ireland border keep the full-boundary test, because the map has no Northern Ireland polygon. A refused proof's reason is now kept in the route record's `boundary_notes`. Waypost (`seo-manager`) is pinned to cbb5f4d for proof version `tiger2026-tailte2026-bounded-v3`.
