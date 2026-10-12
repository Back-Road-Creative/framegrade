## Route Story projects (`gomoveshift/pipeline/route_story/project.py`)

`RouteStoryStore` creates private, brand-scoped drafts before preview or rendering. Each project
fingerprints its regular-file inputs, records only files under configured source roots, and stores
the complete reviewed recipe. Repeating the same request returns the same project identity; a
different brand cannot read that record. Privacy remains `review_required` until the future export
gate has checked route geometry and media derivatives. `RouteStoryStore.update` persists reviewer
decisions (e.g. a media placement correction) onto an existing project by field, so a re-render never
re-asks a question the operator already answered; it refuses an unknown project id and requires at
least one field to merge.

The private `GET /route-stories/{id}/telemetry` console endpoint loads the draft's one timed GPX
source through the same importer and summary function as a renderer would. It returns SI facts
without creating a preview or render job; malformed sources and ambiguous project input are 422.
