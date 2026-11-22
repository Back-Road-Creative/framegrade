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
or a small publish window still yields fewer shorts than the target, same as before. The target
is a maximum: the planner also drops windows that are not above the timeline mean and never relaxes
the gap to reach it (`59-shorts.md`, "Shorts quality bar").

A per-location cap below `config.min_shorts_per_lf` is not a bug: `min_shorts_per_lf` is only ever
a soft warning (`config.py`'s `check_shorts_consistency`), never an enforced floor. The SHORTS-
CONTRACT render-stage gate and the master-deletion gate check the count the day planned (at most
the cap-derived `target_count` computed above), never the uncapped config/derived value.

**Moment-length clips (`short_planner.py`).** `DEFAULT_MIN_DURATION_SEC`/`DEFAULT_MAX_DURATION_SEC`
widen to 15/55 seconds (was 30/45); `DEFAULT_TARGET_DURATION_SEC` (38) is unchanged and is now only
the fixed search width used to find each candidate window, and the fallback duration when a moment
has no signal to grow into. After the quality-bar greedy pick, `_grow_to_moment_length`
extends each chosen window's end one second at a time for as long as the next second's composite
score is *strictly greater* than the window's own running mean — a flat or declining moment (the
common case: plateaus, isolated peaks, and every flat-score fixture in the existing suite) never
grows, so nothing changes for a fixed-strength moment; only a moment that is still gaining strength
right past the search width lengthens. Growth stops at `max_duration_sec`, the next chosen window's
start minus `min_gap_sec`, a stop-masked second, or the timeline end — so it can never violate the
non-overlap/min-gap/stop-mask invariants the rest of the planner already enforces. `SHORTS_MAX_DURATION`
(60, `video_types.py`, YouTube's hard ceiling) is untouched and still strictly above the new default
max (55).

**Opening seconds and trimmed length (`short_planner.py`).** A window now qualifies only when both
its whole-window mean composite and the mean of its first `OPENING_SECONDS` (4) seconds are strictly
above the quality bar, so a weak opening never ships. Pick order uses `sum * (1 + opening_mean)`
(`_ranking_score`), ties to the earlier start; the hard gap, stride, stop/GPS/county masks, the bar
and rank-to-publish order are unchanged. After the grow step, `_trim_to_moment_length` pulls each
pick's end back to the last second above the bar, never shorter than `min_duration_sec`. Trimming
only moves an end earlier, so it cannot reach a masked second; growth already stops at one. Length
therefore ends anywhere in [min, max] following the moment, and `target_duration_sec` stays the
search width. Each `ShortRange` now carries `window_mean`, `opening_mean` and `quality_bar`; they
are written to the shorts audit sidecar with `moment_kind` and excluded from the audit hash.

**Landmark bonus (`short_planner.py`, `destination_resolver.landmark_seconds_mask`).** A window
inside a measured destination polygon ranks higher: `plan_shorts(landmark_mask=...)` multiplies its
pick-order score by `1 + LANDMARK_BONUS` (0.25) when ONE polygon holds at least `LANDMARK_QUORUM`
(the resolver's `MIN_PEAK_MASS_SHARE`, 0.5) of the window's seconds, so a polygon the fixes only
graze, or two parks that together reach half, earn nothing. The planner picks with and without the
bonus and keeps the boosted picks only when they hold at least as many Shorts. What actually renders
is decided after the per-LF cap and refill, so `day_pipeline._adopt_landmark_plan` runs the plain
plan through the same cap and refill too and adopts the landmark plan only if it renders at least as
many Shorts AND its total window mean (each clip's mean composite, not the ranking score), every
landmark clip on both sides weighted by 1.25, is at least the plain total: the bonus may trade at
most 25% of the score of a landmark window only the landmark plan picks, and never costs a Short. The mask is one row per polygon, aligned to the score
rows' own times, built once per day beside `_shorts_planning_mask` and reaching the per-LF refill
through the same planner kwargs. It uses the resolver's destination tags, `out geom` rings and
point-in-ring test; only CLOSED rings count (point POIs, bounds-only areas and open ways never do);
fixes failing the route-claim valid-fix rule are dropped first; a fix holds its seconds only across a
fix-to-fix gap within the Shorts GPS-gap bound (`stop_mask.GPS_MAX_FIX_GAP_S`).

The POI lookup is optional and budgeted: one Overpass endpoint, one attempt, a 15 s server and 20 s
client timeout (per socket operation, not a total deadline), no cache write for a partial
(remark-bearing) reply (a valid reply without a remark is cached), and no query at all when the
day's route spans more than 2 degrees on an axis. A GPS-less day, an over-wide day, a failing source
or no polygon logs `landmark bonus disabled` and plans exactly as before. Because the bonus depends
on whether Overpass answered, a day's picks can differ between a run that got an answer and one that
did not; a successful answer is cached, so reruns of the same day are stable. Titles and OST naming
are unchanged. `ShortRange.landmark` is true when the FINAL clip (after growth and trim) holds the quorum in one
polygon, and is written to the audit sidecar outside its hash.
