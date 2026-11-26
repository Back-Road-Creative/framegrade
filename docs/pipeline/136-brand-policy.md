### Per-brand policy block (`console/brands/<slug>/brand.yaml`, `policy:`)

`console/brands/<slug>/brand.yaml` may now carry an optional `policy:` block alongside the existing
`disabled_tabs:` key. `gomoveshift.publishing.brand_policy.load_brand_policy(slug)` reads it and
returns a `BrandPolicy` dataclass; a missing file, missing block, or missing key falls back to
today's behaviour, so a brand that never opts in is unaffected. A present-but-invalid value raises
`ValueError` naming the offending `policy.<key>`.

Each key is validated on load; the table names its reader where one exists:

| Key | Validation | Reader |
|---|---|---|
| `per_location_cap` | int >= 1, or omitted = no cap | `day_pipeline.py`'s shorts-budget block, and the batch check in `cli/handlers/enqueue_plan.py` |
| `shorts_per_drive` | int >= 1, or omitted = no cap | `day_pipeline.py`'s shorts-budget block and the schedule report, via `publishing/title_kit.py` (the smaller of this and `per_location_cap`); never the enqueue batch gate, so several drives in one region still enqueue together |
| `weekly_ceiling` | int >= 1, or omitted = no cap | `cmd_enqueue_plan` in `cli/handlers/enqueue_plan.py`, rolling-ceiling gate, `min(config ceiling, weekly_ceiling)` |
| `daily_ladder` | non-empty sequence of ints >= 1, strictly increasing, stored as a tuple (e.g. `[1, 2, 3]`) | `cli/handlers/metrics.py`; shorts-per-day ramp |
| `stagger_hours` | int >= 0 | cross-platform post delay |
| `region_weight` | non-empty mapping of non-empty lowercase string -> number > 0, stored as an immutable `types.MappingProxyType` | per-region scheduling weight |
| `blackout` | sequence of ISO dates (`YYYY-MM-DD`), stored as a tuple of `datetime.date`, or omitted = no blackout | `dispatch._select_due` |
| `quiet_after_alert_hours` | int >= 0, or omitted = no quiet period | `dispatch._select_due` |
| `experiments` | list of caption experiments (below), or omitted = none; stored as a tuple of `publishing.experiments.Experiment` | `SocialGenerator`; the nightly report in `cli/handlers/metrics.py` |

An unknown key under `policy:` raises `ValueError` naming it -- a typo in a policy key must not
silently do nothing.

A brand opts in by adding a `policy:` block to its own `brand.yaml`. An illustrative example, with
fictional values:

```yaml
policy:
  per_location_cap: 4
  weekly_ceiling: 9
  daily_ladder: [1, 2, 3]
  stagger_hours: 12
  region_weight: {north: 2, south: 1}
```

`per_location_cap`, `weekly_ceiling` and `daily_ladder` (`cli/handlers/metrics.py`) have readers today;
`stagger_hours` and `region_weight` are still inert data until their readers land.
`console/settings.brand_disabled_tabs()` only ever reads the `disabled_tabs` key via `.get()`, so
the `policy:` block does not change console nav/tab behaviour for any brand.

The shipped GoMoveShift `brand.yaml` carries only an `experiments` key (below); a brand without a
`policy:` block, or a key it omits, falls back to `None`.

### `blackout`

`dispatch._select_due` resolves the current brand via `console.settings.current_brand()`
(`GMS_CONSOLE_BRAND`, default `gomoveshift`) and excludes any due row whose `scheduled_at` date
(or `now`'s date, for a row with no recorded schedule) is one of `policy.blackout`'s dates. This is
the same "no status change, no attempt bump" treatment already given to a paused platform: the row
is picked up automatically the day after the blackout window ends. `deliveries` carries no `brand`
column -- this service runs one brand instance at a time, so `_select_due` resolves the policy once
per call rather than per row. The console Publishing tab shows `held: blackout` next to a
non-terminal row whose effective date falls in the window (`console/pubqueue.py:
enrich_rows_with_sidecar`).

### `quiet_after_alert_hours`

The "alert" is a platform's TRANSITION INTO `SUPPRESSED`: the most recent `response_log` row with
`state='SUPPRESSED'` whose immediately preceding row for that platform (by `id`) has a different
state, or has no preceding row (`dispatch.platform_quiet_alert_at`). A nightly re-log while still
SUPPRESSED does not restart the clock -- only the transition row's `at` anchors the window.
`RECOVERING` and `HEALTHY` are never alerts. Only `mode='live'` rows count, matching how caps
behave in shadow.

Scope is per platform: `_select_due` holds only the non-terminal, non-shadow rows of the ONE
platform whose alert fired -- `now - alert.at < quiet_after_alert_hours` hours -- with the same
"no status change, no attempt bump, no error" treatment as `blackout` and a pause. After the window,
the existing cap ladder governs again. `quiet_after_alert_hours` omitted/`None` means no hold and
no `response_log` query at all; `0` means a zero-length window, so it never holds. The console
Publishing tab shows `held: quiet` the same way it shows `held: blackout`
(`console/pubqueue.py: enrich_rows_with_sidecar`) -- blackout wins when both apply, checked first.

### `experiments`

A caption experiment chooses WHICH hashtag treatment a short's Facebook caption gets. It never
changes how many shorts post or when -- cadence and volume stay Waypost's.

```yaml
policy:
  experiments:
    - name: fb_hashtags
      platforms: [facebook]
      start: 2026-11-02          # first day an arm is assigned
      stop_after_days: 45        # no new arm from start + 45 days on
      min_n: 20                  # settled, measured shorts each arm needs before a median is reported
      arms:                      # equal split; add `weight: <int >= 1>` to bias
        A: {hashtag_mode: none}      # composed caption, no generic hashtags
        B: {hashtag_mode: location}  # composed caption + 2 location hashtags
        C: {hashtag_mode: june}      # June-style: 3 fixed generics + place + region
```

Validation is loud (`policy.experiments[i].<key>` in the error): non-empty unique `name`,
`platforms` a non-empty subset of `[facebook]` (reach is the only outcome measured), ISO `start`,
`min_n` and `stop_after_days` ints >= 1, at least two arms, each `hashtag_mode` one of
`none`/`location`/`june`, `weight` an int >= 1, no unknown keys.

The shipped GoMoveShift template (`console/brands/gomoveshift/brand.yaml`) carries exactly the
block above. A `GMS_BRANDS_DIR/gomoveshift/` overlay replaces the whole template, so it must
carry the block itself to keep the experiment running.

- **Assignment** (`experiments.assign_arm`) is a weighted sha256 of `name:project:short_index`,
  so it is reproducible and never random at runtime. `SocialGenerator` assigns when it generates
  a post and writes the `{experiment: arm}` map to `SocialPost.experiment_arms`, then
  `PublishRecord.experiment_arms` in the publish-plan sidecar (omitted from the file when empty, so
  plans outside an experiment keep their Gate C/D bytes). The sidecar is hashed at Gate C/D, so
  the tag is written at generation, not at `enqueue-plan`. `deliveries` has no column for it and
  adding one would need a queue migration; consumers join through the sidecar.
- **Arm modes** (`SocialGenerator._platform_hashtags`): `none` = no hashtags; `location` = the
  place and region tags (first two location tags); `june` = `JUNE_STYLE_HASHTAGS`
  (`#POVDriving #ScenicDrive #RoadTrip`, the June 2026 posts' fixed tags) then those two location
  tags. Only `none` omits the region hashtag, so the Facebook `_assert_region_present` check is
  skipped for it; the caption text still names the place.
- **Enrollment** (`SocialGenerator._enrolled_arms`): a Short is tagged into the experiment only
  when its arm renders differently from arm A. An arm-B Short with no location tag to carry, or an
  interactive one (which withholds the place), would ship zero hashtags exactly like `none`; it
  gets no `experiment_arms` entry and keeps the normal hashtags; so does a hand-captioned Short
  with no recording date whose bare arm-A caption would equal its YouTube caption. The decision calls the same
  `_platform_hashtags` the caption uses, so tag and text cannot disagree.
- **Outcome keying**: `experiments.arm_observations` joins the sidecar's `experiment_arms` to
  `reel_initial_plays` (the collector's `blue_reels_play_count`); it needs no opener template, so
  a hand-captioned Short counts the same as a composed one. (`tracking/caption_evidence.py`, the
  per-opener evidence, does need `caption_templates` and stays separate.)
- **Caption edits** (`queue edit-facebook-caption`): a built caption keeps the Short's arm
  (`facebook_caption_edit.measured_caption`): arm A adds no hashtags, B the county's two tags, C
  the three fixed generics then those two; a Short outside the experiment keeps the county tags.
- **Report**: `metrics collect` prints one line per arm after the bounded-response step:
  `EXPERIMENT <name> platform=facebook arm=<id> n=<settled> median_plays=<value|unknown>`. A short
  counts toward `n` only with measured `PlatformMetrics.reel_initial_plays` and a settled day-7
  window (`tracking.day7.window_settled`; Facebook keeps no daily series, so `has_settled_series`
  does not apply). The median stays `unknown` until EVERY arm holds `min_n`; unmeasured plays are
  never 0. The outcome was Reel reach until 2026-11-14 (#385): Graph answers no per-Reel reach for
  the live Page, so every arm sat at n=0.
  `experiments.arm_results` returns the same data for a caller that wants the numbers.
- **Shorts arms and decision record** (`experiments.py`, YouTube Shorts): the factors are
  `yt_title_body` and `yt_hashtag_set` (`SHORTS_FACTORS`), tagged in the sidecar's
  `experiment_arms` like any other experiment. `assign_stratified_arm` gives a new Short the arm
  its stratum holds fewest of (ties on a sha256 of factor, project, slot, short and arm).
  The arms are tracked config in `experiments.py` (`SHORTS_ARMS`; bodies in
  `SHORTS_TITLE_BODY_ARMS`, hashtags in `SHORTS_HASHTAG_SET_ARMS`). `yt_title_body` is `place_led`
  (`{location}` or `{location}, on a {weekday}`) or `clip_led` (`{secs} seconds, {location}`),
  each one of the generator's own `STAMPED_BODY_TEMPLATES`; `yt_hashtag_set` is `pov`
  (`#POVDriving`, the legacy set) or `road` (`#RoadTrip #OpenRoad`), both from `GENERIC_HASHTAGS`.
  Phase D's social pass calls `generate_all(shorts_arms_enabled=True)` once per long-form with ONE
  shared `shorts_arm_history`: `assign_shorts_arms` gives each new Short an arm per factor,
  stratified by the BASE project alone (the generator's `<base>_lfNN` with `_lfNN` stripped, the
  project `decide_shorts_arms` also groups on, which ignores the slot). A Short's day position is
  not part of the stratum: Phase D's title ordinals are unique across the whole day, so a
  per-position stratum would hold one Short and balance nothing. The history counts only the arms
  a Short actually kept (`record_shorts_arms`, after the post is rendered). With no drops, the
  per-arm counts of one factor differ by at most one across a drive. An arm dropped at render
  time (below) does not count, so the next Short gets that arm; the count is refilled, not the
  mix, and the longer arm text is dropped more often on long places (road kept 84 of 90 vs pov
  89 of 90 in a long-POI sweep), so it slightly under-represents long descriptions. The same inputs give the same arms; `SocialGenerator` renders them and the plan
  stores them in `experiment_arms`.
  A factor the text cannot carry is not assigned (a hand title or an unverified place has no
  title-body arm; an interactive Short has no hashtag-set arm), and an unknown arm renders as
  today and is not stored. An arm is also dropped, never stored, when its text cannot be
  rendered whole: every body of the title arm is already a published title or would not fit in
  100 characters without a mid-word cut (the full template walk is used instead), or the
  description limit would cut one of the hashtag arm's tags, or cut more of the description's
  body than the legacy render does (the arm's tags are pinned, so only a description with no
  room for them loses the arm). A caller that does not
  enable arms, and every already-queued Short, keeps its text byte for byte
  (`tests/fixtures/shorts_text_origin_main.json`, generated from origin/main).
  Regeneration never assigns: `sync_bracket_short_text` passes each Short's stored arms
  (`SHORTS_FACTORS` only) to `generate_all(per_short_shorts_arms=...)`, which renders them
  verbatim, and refuses a Short whose stored arm the generator can no longer render (hand title,
  place not verified, unknown arm, no usable text) rather than write text that disagrees with
  the stored arm. `queue sync-short-text` shares the same `_generate_lf`, so it renders them too.
  **Power of the hashtag factor.** `plan_builder._title_with_hashtags` copies hashtags from the
  END of the description into the visible title, so the arm's tags (which come first) reach the
  title only when the Short has no terrain and no mid-location tag (the title takes at most
  three tags: place, region, then one more, so of the two `road` tags only `#OpenRoad` can
  arrive). Otherwise they sit in the
  description, which the Shorts player folds away; the factor then has little to act on, and its
  measured gap is correspondingly small. The outcome is
  `youtube_outcome`: the measured observation of the Short's current YouTube id nearest 72 h after
  YouTube's own publish time (the queue's `deliveries.published_at` with its source, never
  `posted_at`, which the last platform to register overwrites; within 12 h) supplies `views` and
  `avg_view_percentage`; with no YouTube time there is no outcome, and a Short counts once
  `published + 72 h <= today - 2 d`, and
  unreported retention is unknown, never 0. `metrics collect` prints, after the `EXPERIMENT`
  lines, one `ARM_DECISION factor= arm= decision= n= min_n= score= reference= gap= reason=` line
  per arm (plus `retest_days=` for a pause); an arm tagged on a Short but with no countable Short prints
  `INSUFFICIENT_EVIDENCE n=0`. Each Short's views are divided by its project's
  median (projects holding two or more arms and a non-zero median only); an arm's score is the
  median ratio and its reference the median score of the other arms holding `min_n` (the
  `caption_ranking` pool-median policy). `INSUFFICIENT_EVIDENCE` below `min_n` or with no
  reference; otherwise a gap (reference / score) of `min_gap` that also holds in each
  chronological half of the arm's Shorts is `demote`, or `pause` at `pause_gap` when the arm's
  median retention is known and no better than the others'; anything else is `hold`. The record
  only reports: it changes no text sent to YouTube. Once both factors are decided it also writes
  `<seo storage>/shorts_arm_decisions.json` (`schema`, `decided_at`, one entry per arm); a run that
  fails part-way leaves the old `decided_at`, which `health check` rule 13 reads as stale
  (`docs/seo-feedback-stale.md`). A paused arm is re-tested every
  `retest_days` (`pause_retest_due`). Thresholds: `policy.arm_thresholds`, parsed only by the report path (`load_arm_thresholds`), so a bad value
  never stops dispatch or planning (`min_n` 36 (at least 4),
  `min_gap` 3.0, `pause_gap` 6.0, `retest_days` 28; named `DEFAULT_*` constants, JP-approved
  conservative defaults; gaps must be finite), no other keys. A caption experiment may not take a
  Shorts factor name. Known limits: an arm's halves split by publish date only (same-day Shorts
  keep queue order), the per-project median pools every arm's Shorts, and retention is a
  72 h snapshot that YouTube may still revise, not settled data. Weekly volume stays Waypost's (`volume_decide`,
  `docs/social-metrics-volume.md`); the nightly wrapper reads this record from the same
  `metrics collect` output.
- **Caption opener ids** (`social_generator.caption_template_id`): every composed caption also
  records which opener template it was built from, as `SocialPost.caption_templates`
  (`{platform: id}`), then `PublishRecord.caption_templates` in the sidecar, with the same
  omitted-when-empty rule as `experiment_arms`. The id is the first 12 hex digits of the
  template text's SHA-256, so reordering a pool keeps every id and rewording a template gives it
  a new one. A hand caption has no template, so its platforms are absent. This is the join key
  Meta caption learning credits reach and plays to; recording it changes no caption.
- **Meta caption learning** (`tracking/caption_evidence.py`, Waypost
  `analytics.caption_ranking`): the nightly `metrics collect` joins each tracked Facebook and
  Instagram post's measured snapshot nearest day 7 (within 12h, the `evidence_export` rule) to
  the opener id its sidecar recorded, and writes `<seo storage>/caption_evidence.json`: per
  platform and opener, the median, the post count and the snapshots' fetch-time span. Facebook
  counts Reel initial plays (Graph answers no per-Reel reach for the live Page); Instagram counts
  reach (stored as `impressions`). It prints `CAPTION_EVIDENCE platform=<p> metric=<m>
  openers=<k> posts=<n>`; a failure is logged and never fails the collection. The generator's
  `_fresh_rotating_pick` hands its shuffled Facebook and Instagram opener order to
  `rank_captions`, which moves openers more than 10% above the measured openers' median to the
  front and those more than 10% below to the back, and changes nothing until at least 2 openers
  each have 10 settled posts. The anti-spam freshness split runs after the ranking, so a winning
  opener is still never reused inside the gate's window. Each ranking is recorded on the post as
  `SocialPost.caption_rankings` (`social_posts.json`). No evidence file, or one Waypost refuses,
  leaves the plain rotation and records nothing. YouTube and TikTok openers are not ranked.
- **YouTube title learning** (`tracking/title_evidence.py`, Waypost
  `analytics.title_evidence`): the nightly `metrics collect` maps each YouTube video id to the
  template that titled it (a Short's `title_decision` in `shorts/lf_*/social_posts.json`, joined
  to `ShortRecord.platform_ids["youtube"]`; a long-form's in `youtube_uploads.json`), skipping a
  decision with no chosen template id or `facts_unavailable`, and writes
  `<seo storage>/title_evidence.json` from the collected `youtube_reach_combined` rows (Shorts
  count browse, long-forms search). It prints `TITLE_EVIDENCE format=<f> traffic=<t>
  templates=<k> impressions=<n>` plus a `TITLE_EVIDENCE_ROWS` line of skip counters. No reach
  store (`SEO_COLLECTION_DB`, nightly reach collection not yet switched on) is a logged skip that
  creates nothing; a failure is logged and never fails the collection. Every shipped-title
  decision (Shorts, long-forms, the pre-render long-form title, superlong) passes the file via
  `title_decision.shipped_title_snapshot()`; none, or one Waypost refuses, keeps the baseline as
  `insufficient_evidence`. **A title moves only once the baseline and a non-baseline template
  each have their own 400 impressions** (and the alternative beats the baseline by more than
  10%). **Exploration** (JP 2026-11-15, Waypost `title-policy-3`): 1 long-form in 4, picked by
  the SHA-256 of its baseline title, ships its least-measured renderable alternative (reason
  `exploration`) so alternatives earn their own impressions. Only the pre-render approval
  (`approve_retained_title`) passes the key, so the explored title is geo-checked before the OST
  burn and Phase D reuses it; Shorts never explore. A single-place Short has no second
  renderable template, so its decision stays the baseline regardless. The final-text review
  (`youtube_adapter`, `final_text_record`) still decides with no snapshot.
- **Where the real policy lives**: the host has no `GMS_BRANDS_DIR/gomoveshift/` overlay, so the
  shipped template's block is the live one. Adding an overlay later means copying the block into it.
