### Added

- `gomoveshift queue edit-facebook-caption --id N [--text "..."] [--apply]` rewrites the description of a published Facebook Reel (`FacebookReelsAdapter.update_description`), reads it back, and records it in the publish-plan sidecar. The caption is built from the Short's own measured county (`sampled_regions`) and recording month; typed `--text` is refused for such a Short, and for a Short with no measured county it must claim nothing (`placeless_label_claims_nothing`). Dry run by default; never changes the queue row or deletes the post.
