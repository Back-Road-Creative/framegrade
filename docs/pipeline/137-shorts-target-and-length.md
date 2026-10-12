### Shorts target precedence and moment-length clips

**Target precedence (`day_pipeline.py`'s shorts-budget block).** `target_count` (the number passed
to `plan_shorts` and checked by the SHORTS-CONTRACT gate / master-deletion gate) is derived, per LF,
in this order:

1. The per-LF count is Waypost's decision, read via `volume_source.shorts_per_lf()` (no config
   key sets it; `required_shorts_per_lf(config)` survives only as the slot-capacity check).
2. `state["shorts_target"]` — `--shorts-target N` — may only lower that decision
   (`volume_source.apply_flag`); a flag-set target is never capped by brand policy.
3. The value from step 1 is then capped (never raised) by the current brand's
   `load_brand_policy(current_brand()).per_location_cap`, when set.

A resumed drive re-checks each already-completed day against the same number: with
`--shorts-target` set, `min(flag, decision)` per LF
(`PhaseBCDayFirst._reconstructed_shortfall_target`); with no flag the re-check is off and
the decision store is not read.

Each rung logs at INFO (`Day %s: shorts target %s (per LF %s × %d LFs, rung=%s)`) with
`rung` one of `cli`, `decision`, `brand-policy-cap`, or `none`. The moving-seconds-budget
cap (`_cap_shorts_target_by_moving_budget`) and per-LF capacity trim
(`cap_shorts_to_per_lf_capacity`) run downstream of this precedence and are unchanged — a thin day
or a small publish window still yields fewer shorts than the target, same as before.

A per-location cap below `config.min_shorts_per_lf` is not a bug: `min_shorts_per_lf` is only ever
a soft warning (`config.py`'s `check_shorts_consistency`), never an enforced floor. The SHORTS-
CONTRACT gate and the master-deletion gate always check against the cap-derived `target_count`
computed above — never the uncapped config/derived value.

**Moment-length clips (`short_planner.py`).** `DEFAULT_MIN_DURATION_SEC`/`DEFAULT_MAX_DURATION_SEC`
widen to 15/55 seconds (was 30/45); `DEFAULT_TARGET_DURATION_SEC` (38) is unchanged and is now only
the fixed search width used to find each candidate window, and the fallback duration when a moment
has no signal to grow into. After the existing rung-based greedy pick, `_grow_to_moment_length`
extends each chosen window's end one second at a time for as long as the next second's composite
score is *strictly greater* than the window's own running mean — a flat or declining moment (the
common case: plateaus, isolated peaks, and every flat-score fixture in the existing suite) never
grows, so nothing changes for a fixed-strength moment; only a moment that is still gaining strength
right past the search width lengthens. Growth stops at `max_duration_sec`, the next chosen window's
start minus `min_gap_sec`, a stop-masked second, or the timeline end — so it can never violate the
non-overlap/min-gap/stop-mask invariants the rest of the planner already enforces. `SHORTS_MAX_DURATION`
(60, `video_types.py`, YouTube's hard ceiling) is untouched and still strictly above the new default
max (55).
