### Fixed

- A stateless-county long-form label with its country tail ("County Cork, Ireland", the label the pipeline itself builds) now gets region facts at the pre-render title gate instead of refusing the encode with `facts_unavailable`; `region_label` appends the country for a single Irish county as it already did for a pair.
