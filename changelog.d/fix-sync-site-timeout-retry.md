### Fixed

- `sync-site` no longer dies on one transient network failure. The four read-only YouTube Data API listing calls behind the channel manifest (`channels.list`, `playlistItems.list`, `videos.list`) now retry up to 4 times with exponential backoff on a read timeout, connection reset, SSL error or HTTP 5xx, then raise the last error. A 4xx still fails at once. The 2026-10-31 daily run was lost to a single `TimeoutError: The read operation timed out`.
