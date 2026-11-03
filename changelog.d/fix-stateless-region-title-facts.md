### Fixed

- A region-only long-form title for a country with no states (an Irish county,
  "County Cork in 4K") now carries Waypost title facts. Before, only a US
  "County, ST" label could match, so every Irish region title was refused before
  the OST burn as `facts_unavailable`. The county must still pass the route gate,
  and its retained footage must resolve to exactly one known country.
