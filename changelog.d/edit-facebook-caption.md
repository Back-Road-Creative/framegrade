### Added

- `gomoveshift queue edit-facebook-caption --id N --text "..." [--apply]` rewrites the description of a published Facebook Reel (`FacebookReelsAdapter.update_description`), reads it back, and records it in the publish-plan sidecar. The new text must claim nothing (`placeless_label_claims_nothing`). Dry run by default; never changes the queue row or deletes the post.
