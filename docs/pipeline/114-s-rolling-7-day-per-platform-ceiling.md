### `enqueue-plan`'s rolling 7-day per-platform ceiling (`cli/handlers/enqueue_plan.py`)

A per-drive gate cannot see a second region's plan landing in the same rolling week, so the
ceiling is enforced here, against the delivery queue -- the only place the cross-drive total is
known. On `--apply`, before any row is written, `_planned_deliveries` collects every
`(platform, project_name, short_index, scheduled_at)` this run would spool (the same
`select_capped_shorts` selection `_media_verification_errors` already scopes media checks to, so
a deferred short is never counted). `_rolling_ceiling_violations` groups those by platform; for
each platform it opens a 7-day window starting at this run's earliest `scheduled_at` for that
platform, counts existing queue rows for that platform in the window (`status != 'failed_permanent'`
-- queued, retrying, awaiting-manual-publish and already-published all still occupy the slot),
and adds this run's own in-window count -- the query carries no region/project filter, which is
what lets it catch a second region's plan landing in the same week. Any platform whose total would
exceed `config.seo.rolling_week_ceiling_per_platform` (default 21, `_DEFAULT_ROLLING_WEEK_CEILING`
when `config.seo` carries no such field) **refuses the whole plan** -- nothing is written for any
platform, even one under its own ceiling -- printing, per violating platform, the ceiling, the
existing and would-add counts, the window, and exactly which `project#short_index` deliveries were
NOT spooled. Refuse-whole was chosen over spool-to-ceiling because a partial spool across
platforms is a silent, uneven under-publish; a full refusal is loud and re-runnable once the
week's existing volume clears.

The default of 21 is the **whole-account** steady state, not a per-region number: "Publish
cadence" schedules 3 long-forms/week (region A Sun, region B Tue, region B Fri) x 7 shorts/long-form
(`DriveConfig.min_shorts_per_lf`) x 1 delivery/platform/short = 21 per platform per rolling week.
This is a ceiling against *accidental* doubling (two regions' plans landing in the same week, the
case a per-drive gate cannot see), not a volume target -- lowering it below the documented cadence
is a deliberate volume decision gated on measured evidence, never a knob to tune
casually.
