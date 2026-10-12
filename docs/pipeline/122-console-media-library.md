### Console Media library (`gomoveshift/console/library.py`, `GET /library`)

The reader's first production caller. `build_context` resolves `site_path` from this brand's
`GMS_SITE_PATH` config only — no path argument, so nothing operator-supplied can bind to it.
Filters map onto `read_catalog`'s own kwargs; the template never filters. `hash_recorded` renders
as "hash recorded"/"no hash recorded", never "verified". Damage is a banner (a source name + fixed
reason); `DamagedSource.path`/`.error` are logged, never rendered. The source name is the journey
stem for a journey and "the video routing file" for the routing file — calling that one the
journey `video_routing` would send an operator looking through `content/journeys` for a file that
is not there. Empty vs. damaged-only render
different bodies.

**An empty page says which kind of empty it is.** "Nothing here yet" was the whole empty
state, and it is the right sentence for a site whose journeys directory exists and holds nothing.
Over a site that has no journeys directory at all it read the same way, and that case is almost
always a `GMS_SITE_PATH` pointing at the wrong tree or a checkout that was never synced —
something to go and fix, phrased as something to wait for. `build_context` now carries `absent`
(source names only; the paths go to the log at INFO) and the template names the missing sources
and says the two things it could be. Every degraded return carries the key too, so the unconfigured
and unreadable-site branches still render.

**The page is bounded; the catalog is not.** `build_context` returns at most `library.PER_PAGE`
(100) rows, sliced *after* the filters run — slicing first would hide a filter's only match when
it lives at row 400. The count in the header describes the whole collection, not the slice: a page
that says "100 items" when there are 1507 reads as a complete catalog. `page` arrives off a query
string, so `_pager` takes anything — 0, -3, past the end, a word, `None` — and lands on the nearest
real page rather than an empty table, which an operator reads as an empty catalog. A collection
that fits on one page renders no pager at all. The pager keys are present on the degraded contexts
too (no site configured, unreadable site path), or the template would raise on a missing name
instead of showing the degraded state.

What this does **not** do is bound the read: `read_catalog` still walks every journey file on every
request. That is a rebuildable index, and the plan introduces one only when the scan itself is the
bottleneck.

**The row is recognisable, not just addressable.** Type/Journey/Date/Catalog ID/Hash were
five identifiers and no name -- an operator could not tell their own photo from the row above it.
`build_context` now also returns `display`, a `catalog_id -> {name, kind, treatment,
privacy_status}` dict; the template renders it beside the existing id/hash columns, never instead
of them, since some operators use those directly. `_display_name` reads the export filename the
catalog id already encodes for a photo, or `routing_id` for a video -- never a hash or the id
itself. `treatment` is `CatalogEntry.decoder` ("not recorded" when absent); nothing invents a
style/look field the catalog does not carry. **No thumbnail is ever rendered on this page.**
Neither `CatalogEntry` nor the journal frontmatter it is read from records a per-photo blur/privacy
signal -- `pipeline/photos/ingest.py` decides blur at export time and only an aggregate run-level
count survives, never written back per file. Showing an image the reader cannot attest to blurring
would be the unproven-blur risk the publish-side hard-fail exists to prevent, moved to the console.
`_privacy_status` returns one honest state (`_PRIVACY_STATUS_UNRECORDED`) for every entry today and
never the word "verified" -- nothing here has verified anything; it takes `entry` so a future signal
has somewhere to branch without touching call sites.

**The listed id is the link.** The list and the asset page were both built, and nothing
joined them: each `catalog_id` rendered as inert text, so the only way to reach `/library/{id}`
was to copy the id out of the table and type it into the address bar. The id stays visible as the
link text rather than being replaced by a friendlier word -- it is the string an operator quotes
into a bug report and pastes into a CLI, and a link that hides it takes that away to save nothing.
`urlencode` on the href is load-bearing, not tidiness: a `catalog_id` is built from a real
filename (`catalog/reader.py`), and a single `#` in one would truncate the href at the fragment,
producing a link that looks correct in the page source and opens a different asset -- or none.
