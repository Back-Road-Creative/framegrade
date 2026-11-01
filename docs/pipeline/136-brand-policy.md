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

The shipped `brand.yaml` is an example seed with no `policy:` block; a brand without a `policy:` block has every key fall back
to `None`.

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
- **Report**: `metrics collect` prints one line per arm after the bounded-response step:
  `EXPERIMENT <name> platform=facebook arm=<id> n=<settled> median_reach=<value|unknown>`. A short
  counts toward `n` only with a measured `PlatformMetrics.reach` and a settled day-7 window
  (`tracking.day7.window_settled`; Facebook keeps no daily series, so `has_settled_series` does not
  apply). The median stays `unknown` until EVERY arm holds `min_n`; unmeasured reach is never 0.
  `experiments.arm_results` returns the same data for a caller that wants the numbers.
- **Where the real policy lives**: `console/brands/gomoveshift/brand.yaml` here is the public
  example seed, and `GMS_BRANDS_DIR/<slug>` replaces it wholesale on the host, so the live
  experiment block goes in the host's `brand.yaml`, not this repository.
