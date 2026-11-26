## `metrics growth-review`: the weekly suggest-only growth review

Nothing else in GMS says, in one place, how far the channel is from YouTube's
Partner tiers, which kinds of post are falling behind, or which posts never went
out. The nightly metrics job ([134-channel-feed-share-pull.md](134-channel-feed-share-pull.md))
stores the numbers and the hourly health check ([156-health-check.md](156-health-check.md))
heals failed drives, but neither answers those three questions. `gomoveshift metrics
growth-review [--offline]` does, from what is already stored. It reads only: it opens
the queue read-only, changes no setting, cadence or post, and always exits 0 (thin
data is the normal case, never a fault), like `metrics feed-share`.

**Partner progress (`tracking/partner_progress.py`).** One `PARTNER_PROGRESS` line per
tier a new applicant faces today: `fan_funding` (500 subscribers, 3 public uploads in
90 days, 3,000 public long-form watch hours in 365 days or 3 million Shorts views in
90 days) and `ad_revenue` (1,000 subscribers and 4,000 hours or 10 million Shorts
views; from `NEW_TERMS_DATE` 2027-02-01, 8,000 hours or 20 million). Either viewing
count satisfies the viewing requirement. Before that date a `PARTNER_DEADLINE` line
gives the days left. The numbers are YouTube's published ones (sources in the module
docstring), not tuned here. Subscribers come from `channels.list(mine=True)`; the two
windows from one Analytics `reports.query` each, by `creatorContentType`
(`VIDEO_ON_DEMAND` and `LIVE_STREAM` minutes are long-form watch time, `SHORTS` views
are Shorts views), through `YouTubeCollector`'s own clients; uploads from the tracking
store. Every count is approximate (the API counts private and unlisted minutes and
cannot leave out ad-campaign views) and every line says `approximate=yes`; Studio's Earn
tab is the authoritative number. A count that could not be read is `unmeasured`, and
the verdict is `met=unknown` unless a measured count already falls short. `--offline`
skips the live part and prints `PARTNER_PROGRESS skipped=offline`.

**Format scorecard (`tracking/growth_review.py`).** Every short posted in the last
`WINDOW_DAYS` (28) plus the settle lag with a settled day-7 figure
(`tracking/day7.py`) is one sample per platform, in up to three groups: its clip-length band
(`under15s`, `15to30s`, `30to60s`, `over60s`), its source (`source:day-reel` for a
project named `day-reel-…` by the day-reel route, `source:drive` otherwise) and its
tracked place. The day-reel route registers each reel it queues in the tracking store
(`publishing/day_reels.py`), the same as `enqueue-plan --apply` does for drive Shorts, so
the publish step can store its platform ids and `metrics collect` can measure it. Each group's median
is compared with its platform's median: `GROWTH_FORMAT` normally, `GROWTH_UNDERPERFORMER`
when the group has at least `MIN_GROUP_POSTS` (5) and sits below `UNDERPERFORM_RATIO`
(0.5) of the platform's median. A short with no settled figure is left out, never
counted as 0. The three numbers are a stated starting policy, not measured values.

**Post problems.** From `social_queue.sqlite3` under `seo.storage_path`, shadow rows
excluded, in the same window: `POST_FAILED` for a `failed_permanent` row, `POST_STUCK`
for a non-terminal row at least `STUCK_AFTER_H` (24) hours past its slot, and
`POST_DUPLICATE` when one short went out more than once to the same platform. This
covers Facebook and Instagram too; health rule 8 covers YouTube failures only. A
missing queue prints `POST_CHECK_SKIPPED`, an unreadable one `POST_CHECK_FAILED`.

The run ends with one `GROWTH_REVIEW window_days= groups= underperforming= failed=
stuck= duplicate=` line. It is meant to run weekly and reach the owner as a digest
section; scheduling it is the workspaces repo's job.
