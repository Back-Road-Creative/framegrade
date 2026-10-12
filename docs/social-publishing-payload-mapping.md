# Consolidated shorts CSV → platform API payload mapping

The exact field mapping from a
row of the consolidated Metricool CSV (`build_consolidated_shorts_csv` in
`gomoveshift/pipeline/csv_generator.py`, header = `GOLDEN_COLUMNS`) to the
four first-party platform API payloads that replace Metricool scheduling.

The frozen, redacted example row used to derive this table lives at
`tests/fixtures/consolidated_shorts_csv_row.csv` (guarded against header
drift by `tests/test_shorts_csv_fixture_header.py`).

## IMPORTANT: the shared `Text` column is Twitter-truncated

`row["Text"]` is built by `truncate_for_platforms()` and is silently cut to
`TWITTER_CHAR_LIMIT` (280 chars, word/grapheme-safe, `…`-suffixed) whenever
the row's `Twitter` toggle is `TRUE` — which it is by default
(`DEFAULT_PLATFORMS`). This truncation exists ONLY because Metricool's CSV
schema has one shared caption column across all platforms.

**None of the four platform payloads below may inherit that truncation.**
Each platform API accepts (and several require) untruncated captions:
Instagram allows up to 2200 chars, Facebook Reels description has no
practical limit near this length, TikTok's `post_info.title` allows 2200
chars, and YouTube's `snippet.description` allows 5000 chars. The per-platform
integration must read the caption from `SocialPost.caption` (or an
un-truncated upstream source), not from the CSV `Text` column, or re-derive
the caption via `truncate_for_platforms(text, twitter_enabled=False)` to get
the un-truncated form back.

## Field mapping

| CSV column (`GOLDEN_COLUMNS`) | YouTube Shorts (`videos.insert`) | Instagram Reels (`/media` → `/media_publish`) | Facebook Reels (`/{page_id}/video_reels`) | TikTok Direct Post (`/v2/post/publish/video/init/`) |
|---|---|---|---|---|
| `Text` | NOT used directly — see truncation note above; use `SocialPost.caption` for `snippet.description` prefix | NOT used directly — use un-truncated caption for `caption` | NOT used directly — use un-truncated caption for `description` | NOT used directly — use un-truncated caption if the platform also gets a body text field |
| `Date` / `Time` | `status.publishAt` (RFC3339, combines Date+Time) | not used (Reels publish immediately in this flow) | `scheduled_publish_time` (Unix ts, must be >10 min and <=29 days from now) | not used — TikTok Direct Post API has no scheduling field |
| `Draft` | inverse informs `status.privacyStatus` (`private` until scheduled) | n/a | informs `video_state` (`DRAFT` vs `SCHEDULED`/`PUBLISHED`) | n/a |
| `Facebook` | n/a | n/a | gates whether this row is sent to Facebook at all | n/a |
| `Twitter` | n/a (dropped — see below) | n/a | n/a | n/a |
| `LinkedIn` | n/a (dropped) | n/a | n/a | n/a |
| `GBP` | n/a (dropped) | n/a | n/a | n/a |
| `Instagram` | n/a | gates whether this row is sent to Instagram at all | n/a | n/a |
| `Pinterest` | n/a (dropped) | n/a | n/a | n/a |
| `TikTok` | n/a | n/a | n/a | gates whether this row is sent to TikTok at all |
| `YouTube` | gates whether this row is sent to YouTube at all | n/a | n/a | n/a |
| `Threads` | n/a (dropped) | n/a | n/a | n/a |
| `Picture Url 1` | video file source (resolved to the actual media bytes, not this URL — the API needs an upload, not a link) | video source for the resumable upload that backs `/media` | video source for the `upload_phase=start`/`finish` resumable upload | video source for `source_info` (`source=FILE_UPLOAD`) |
| `Shortener` | n/a (Metricool-only link-shortening toggle, dropped) | n/a | n/a | n/a |
| `Instagram Post Type` | n/a | confirms `media_type=REELS` | n/a | n/a |
| `Facebook Post Type` | n/a | n/a | confirms this is a Reels post (vs feed video) | n/a |
| `YouTube Video Title` | `snippet.title` | n/a | n/a | n/a |
| `YouTube Video Type` | informs the upload as a Short (vertical, <=3min; enforced by video dimensions, not an explicit API field) | n/a | n/a | n/a |
| `YouTube Video Privacy` | `status.privacyStatus` baseline (`public`, overridden to `private` until `publishAt` fires) | n/a | n/a | n/a |
| `YouTube video for kids` | `status.selfDeclaredMadeForKids` (`false`) | n/a | n/a | n/a |
| `YouTube Video Category` | `snippet.categoryId` — currently forced empty in the CSV (Metricool-only quirk, see below); the API payload should populate the real `SocialPost.youtube_category` instead | n/a | n/a | n/a |
| `YouTube Video Tags` | `snippet.tags` (split on `,`) | n/a | n/a | n/a |
| `TikTok Title` | n/a | n/a | n/a | `post_info.title` |
| `TikTok disable comments` | n/a | n/a | n/a | `post_info.disable_comment` |
| `TikTok disable duet` | n/a | n/a | n/a | `post_info.disable_duet` |
| `TikTok disable stitch` | n/a | n/a | n/a | `post_info.disable_stitch` |
| `TikTok Post Privacy` | n/a | n/a | n/a | `post_info.privacy_level` |
| `TikTok Auto Add Music` | n/a | n/a | n/a | informs `post_info.brand_organic_toggle` / `is_aigc` flags |
| `Pinterest Board` | n/a (dropped — see below) | n/a | n/a | n/a |
| `First Comment Text` | posted as a separate comment via `commentThreads.insert` after upload (not part of `videos.insert`) | posted as a separate comment via `/comments` after `/media_publish` | not supported (Facebook Reels API has no first-comment endpoint at this cutover) | not supported (TikTok Direct Post API has no first-comment endpoint) |

## Additional payload fields with no CSV source

- **YouTube**: `snippet.description` — built from `SocialPost.youtube_description` plus the parent-LF link (`https://youtu.be/<parent_video_id>`), not present verbatim as a CSV column (the CSV encodes the link separately in `First Comment Text`).
- **YouTube**: `snippet.defaultLanguage`, `status.license`, `status.embeddable` — sent on every `videos.insert` from `config.drive.youtube_default_language` / `youtube_license` / `youtube_embeddable` (defaults `"en"` / `"youtube"` / `True`), not CSV-sourced. `status.containsSyntheticMedia` is sent only when `YouTubeUploader.upload_video`'s `contains_synthetic_media` kwarg (or its config fallback `youtube_contains_synthetic_media`) is explicitly `True`/`False`; `None` omits the field.
- **Instagram**: `media_type=REELS` is a payload constant, not read from a column value (the `Instagram Post Type` column just confirms the row was authored as a Reel).
- **Facebook**: `upload_phase` (`start`/`finish`) is protocol state for the resumable upload, not sourced from the CSV.
- **TikTok**: `source_info.source=FILE_UPLOAD` is a payload constant.

## Dropped at cutover

These CSV columns / toggles have no destination in any of the four
first-party payloads and are intentionally dropped when the pipeline moves
off Metricool:

- `Twitter` / X — platform not carried forward.
- `LinkedIn` — platform not carried forward.
- `GBP` (Google Business Profile) — platform not carried forward.
- `Pinterest` (toggle) and `Pinterest Board` — platform not carried forward.
- `Threads` — platform not carried forward.
- `Shortener` — Metricool-specific link-shortening toggle; no equivalent on any first-party API.
- The **empty `YouTube Video Category`** quirk: `csv_generator.py` currently
  forces this column to `""` on every row because Metricool rejects a
  populated category ID on shorts rows and silently routes the import to
  draft (see the comment in
  `build_consolidated_shorts_csv`). That workaround is Metricool-specific —
  the YouTube Data API's own `videos.insert` has no such restriction, so the
  first-party payload should use the real `SocialPost.youtube_category`
  value, not the CSV's forced-empty column.
