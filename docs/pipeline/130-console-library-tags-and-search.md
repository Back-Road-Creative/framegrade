## Console library tags and search (`gomoveshift/console/library.py`, `gomoveshift/catalog/tags.py`)<a id="console-library-tags-and-search"></a>

The library had structural filters only (type/journey/date/id). The library tags and search page adds two things: operator
tags on a library item, and free-text search (`q=`) that combines with the existing structural
filters and pagination.

**Tags are per-brand storage, path-separated (`gomoveshift/catalog/tags.py`).** One JSON file per
brand under `console_data_dir()/brands/<slug>/library_tags.json`, keyed by `catalog_id`, never a
shared store keyed by a brand column — the same isolation primitive `console/tenant.py`'s stamp
guard documents for the publishing queue. `add_tag`/`remove_tag` read-modify-write the whole file;
a missing, damaged, or malformed file degrades to "no tags recorded" rather than raising, since
tags are supplementary metadata, never load-bearing for the library itself. `POST
/library/{catalog_id}/tags` (form fields `tag`, `action` — `add` or `remove`) is the only mutation
path; `library.add_library_tag`/`remove_library_tag` resolve `console_data_dir()`/`current_brand()`
themselves, the same pattern `resolve_site_path` uses, so a brand's tags can never be written to
or read from another brand's file.

**Search matches title, journey, and this brand's own tags — case-insensitively.** `q` is combined
with the structural filters (`media_type`/`journey`/`date`), never a replacement for them, and runs
*before* `_pager` slices the page — the same rule the existing filters already followed, so a
search match on row 400 still counts toward the right page. There is no separate "location label"
field on `CatalogEntry`; journey is the closest thing this catalog records to one, and search
matches it directly rather than inventing a field the reader does not carry.

**Tags render escaped, never as raw HTML.** The template relies on Jinja2's default autoescaping
for `{{ tag }}` — the same mechanism every other user-supplied string on this page already goes
through; nothing here adds a `|safe` filter or its own escaping.
