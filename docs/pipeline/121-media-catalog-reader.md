### Media-catalog reader (`gomoveshift/catalog/`; backend contract only)

Read-only, offline: `read_catalog(site_path, *, media_type=None, journey=None, date=None,
catalog_id=None) -> CatalogResult`. No templates/routes/writes — the library VIEW (a later
slice) calls this. Filtering is by catalog id/type/date/journey only, never an operator path.

- **Photos**: `<site_path>/content/journeys/*.md` (same glob as `journey_by_date.py`), one entry
  per filename under `days[].photos[].photos[]`. `content_hash`/`hash_recorded` read
  `photo_hashes[filename]` (`pipeline/photos/ingest.py`'s content-identity registry) — a legacy
  entry with no recorded hash is `content_hash=None, hash_recorded=False`, never fabricated or
  dropped. `hash_recorded` means only "a hash string was present in the journey's own front
  matter" — nothing in this module opens the media file or compares bytes, so it must never be
  read as "verified". `decoder` reads the journey's `photo_provenance[filename]`.
- **Videos**: `data/video_routing.yaml`, read via `journey_reconcile.py:load_routing` (not
  reimplemented) so its no-op-on-missing-file and journey+date-required rules stay in one place.
  `routing_id` is the routing key; `date` is that table's **travel** date, never a publish date.
- **Unavailable vs. empty**: `CatalogSourceUnavailableError` only when `site_path` isn't a
  directory. A missing `content/journeys` dir or `video_routing.yaml` is that source's own empty
  subset, not an error.
- **Damaged vs. empty, per source file**: `CatalogResult.damaged` lists every source file that
  raised while being read/parsed — a journey markdown (`OSError`, `ValueError` for a bad/missing
  frontmatter delimiter, or a YAML parse error) **or** `video_routing.yaml` — each as a
  `DamagedSource(path, error)`. Every other source in the same call is still returned in full: one
  bad file never shrinks the rest of the catalog, and in particular a damaged routing file no
  longer takes the photo entries down with it (that call had no error handling at all, so it
  emptied the whole library and surfaced as a 500 rather than a visible partial state). `damaged`
  names only sources the call actually read, so a `media_type` filter narrows the damage report
  with it. A source that parses cleanly and simply holds nothing is a real, empty subset and is
  never added to `damaged`.
- **Absent vs. empty, per source**: a source that is not there at all — no
  `content/journeys` directory, no `data/video_routing.yaml` — is still that source's own empty
  subset for the read (a site with no videos yet must not 500), but it is a different fact about
  the operator's configuration and `CatalogResult.absent` reports it as an `AbsentSource(path)`.
  The three states are exclusive: absent (not there), damaged (there, would not parse), empty
  (there, parses, holds nothing). The routing check is an explicit `exists()` *before* the load
  rather than an inference from its error, because a missing file and a truncated one both raise
  out of `load_routing` and telling an operator to go find a file they already have is worse than
  saying nothing. `absent`, like `damaged`, names only sources the call actually read, so a
  photo-only read says nothing about whether the routing file exists.
- **Two yaml libraries, two `YAMLError`s**: `load_journey` parses with ruamel, `load_routing` with
  PyYAML, and neither exception class subclasses the other — hence the two separate error tuples
  in `catalog/reader.py`. One merged tuple would silently stop catching half of what it names.
- **A routing document that is not a mapping is no routes, not a crash**: `load_routing` used to
  raise `AttributeError` from `.get`/`.items` on a YAML list or bare scalar, which is neither of
  its two documented outcomes (missing file → `{}`, malformed rows skipped) and crashed both of
  its callers — `reconcile_journeys`, which promises it never raises on a routing/data
  inconsistency, and the catalog reader.
