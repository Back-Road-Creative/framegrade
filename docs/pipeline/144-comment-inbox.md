### Unified comment inbox -- list only (`pipeline/comments.py`)

`build_inbox_rows()` and `inbox_platform_status()` add a read-only, cross-platform view over the
JSONL ledger `pipeline/comments.py` already writes (`fetch_youtube_comments`, `import_tiktok_csv`
/`import_tiktok_txt`, `load_comments`) -- no new network call, no new host, no reply/send code.
This is deliberately the smaller of two slices; the human-reply half (never auto) is a separate
follow-up.

`build_inbox_rows(comments, *, platform=None, unanswered_only=False)` sorts every ledger row
newest-`published_at`-first regardless of which pull/import wrote it, and carries platform, video,
author, text, and `replied`. `replied` is always `None` ("unknown") -- the ledger records no reply
state yet, so `unanswered_only` is accepted for the console filter to wire up against later but is
currently a no-op (nothing is known to be replied, so nothing is excluded). Faking a `False` here
would tell an operator "unanswered" about a comment we've simply never checked.

`inbox_platform_status(comments)` reports one row per `KNOWN_PLATFORMS` entry
(`youtube`, `instagram`, `facebook`, `tiktok`): a ledger count for a platform with an existing
pull/import path, or the literal status `"not connected"` (count forced to `0`, never a real tally)
for `UNCONNECTED_PLATFORMS` (`instagram`, `facebook` -- confirmed absent by grep for
`graph.facebook`/`instagram` comment reads under `pipeline/`/`publishing/`; only outbound
first-comment *posting* text exists, not comment *reading*). A platform showing zero real comments
must never render identically to one nobody has ever tried to connect.

**Follow-up, not yet wired**: `console/app.py`'s `GET /comments` route (`comments_tab`)
still renders `load_comments()` rows directly through `comments.html`'s single-platform-agnostic
table. Swapping it to call `build_inbox_rows()`/`inbox_platform_status()` and adding the
platform/unanswered query-string filters to the template is a one-line-route follow-up, not built yet.
