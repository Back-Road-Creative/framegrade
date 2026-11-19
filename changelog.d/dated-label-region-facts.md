### Fixed

- A cross-drive dated county label (`County Cork, Ireland (Sep 19, 2025)`) keeps its region facts at the pre-render title gate instead of refusing the encode with `facts_unavailable`; the gate removes exactly the date suffix the uniqueness backstop appends before matching the county award.
