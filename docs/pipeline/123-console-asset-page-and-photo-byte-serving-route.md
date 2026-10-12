### Console asset page and photo byte-serving route (`GET /library/{catalog_id}`,
`GET /library/media/{catalog_id}`)

`build_asset_context` looks the id up with `read_catalog(..., catalog_id=...)` — the reader already
filters by identity, so no new lookup logic exists. No match is a 404. A video entry gets no media
link; the page says its published URL is the preview and shows only `routing_id` and the routing
table's travel date (never a publish date). History never joins to the publishing queue — that is
the publishing page, and a half-join here would be a fourth, inconsistent answer to "was this published?".

**The photo media root is derived, never invented**: `resolve_photo_path` builds it from
`resolve_site_path()`'s `site_path`, the entry's own `journey`, and
`gomoveshift.pipeline.photos.selector.IMAGES_BASE` (`static/images/journeys/<journey>`) — the same
directory `photos ingest` writes gallery JPEGs into (`pipeline/photos/ingest.py`). The filename
itself is decoded out of the catalog id, since a photo `CatalogEntry` carries no separate filename
field.

**Resolve-then-compare, never strip-then-join.** The candidate path is joined onto the media root
and both sides are `Path.resolve()`d *before* comparison (`candidate.is_relative_to(root)`); this is
the only check applied — filtering `..` or characters out of the filename first is deliberately not
done, because a filter that looks like it neutralizes traversal is worse than no filter: it invites
trusting a sanitized string instead of verifying where it actually points. Resolving before
comparing is also what catches a symlink inside the media root that targets a file outside it —
`.resolve()` follows the link before the containment check runs.

**Suffix allowlist**, checked case-insensitively against `ALLOWED_PHOTO_SUFFIXES`: anything not on it
is a 404, never a download, regardless of what `is_relative_to` says.

A photo entry whose file is missing (or whose path/suffix fails validation) renders a visible
"Photo unavailable" state on the asset page rather than a broken `<img>` — the same honesty rule as
the damage banner. Videos are not proxied: `routing_id` points into
`data/video_routing.yaml`, and the published URL already serves as the preview.

**The page serves a derivative and names it one.** `photo_hashes[filename]` is the SHA-256
of the *original* card file — `_source_hash` in `pipeline/photos/ingest.py` is explicit that it is "never derived from the
developed output" — while what `GET /library/media/{id}`
serves is the developed export under `static/images/journeys`. Showing the image
beside a bare "hash recorded" line would invite an operator to hash the bytes in front of them,
get a different number, and conclude the file was damaged. `_provenance` now returns two halves:
`original` (`content_hash`, `decoder`, `present_here`) and `derivative` (`relative_path`,
`present`, `size_bytes`). `present_here` is always `False` today, because the original stays on the
card the photos were ingested from and the site checkout never holds it; it is a field rather than
a fixed sentence so the template states a fact it was given, and so an import that *does* keep
originals has somewhere to say so. `size_bytes` is a `stat`, not a read: this module never opens
media, and a hash of the export would have nothing to compare against — the recorded one is the
original's. Nothing on the page says "verified": nothing has compared any bytes. A video row
carries `None` for both halves; its bytes live on the destination platform, and borrowing the photo
shape would name a file that does not exist.
