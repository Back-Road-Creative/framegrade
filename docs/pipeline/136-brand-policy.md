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

No `brand.yaml` ships in this repository; a brand without a `policy:` block has every key fall back
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
