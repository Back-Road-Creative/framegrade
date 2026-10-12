### TikTok metrics collector (`tracking/collector.py`)

`TikTokCollector` (`"tiktok"`, registered; it was once an unregistered
`NotImplementedError` stub) pulls stats from `POST video/query/?fields=...`, `video_ids`
batched up to 20/request and paged via `cursor`/`has_more`. Degrades like
`YouTubeCollector` on missing/revoked credentials or an API error. Upload handles
(`v_inbox_file~v2.<digits>`, `platform_upload_id` for `awaiting_manual_publish` rows)
aren't queryable and are filtered out before any request.

**Network allowlist**: `graph.facebook.com` is documented in the "Network allowlist &
credential stores" section above under `publishing/{facebook,instagram}_adapter.py`'s own
`ALLOWED_UPLOAD_HOSTS`. This collector follows that pattern.
