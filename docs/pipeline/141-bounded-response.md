### Automatic, bounded response to platform suppression (shadow-first)

`tracking/suppression.py` classifies each platform's day as `HEALTHY` / `SUPPRESSED` /
`RECOVERING`; nothing acted on that classification. `publishing/response_policy.py` and
`publishing/dispatch.apply_suppression_response` close that gap with a bounded, reversible,
**shadow-first** response — no code here flips it to live.

**The math (`response_policy.py`, pure — no DB, no clock).**
- `daily_cap_for(state, ladder, healthy_days)`: `SUPPRESSED` always caps at 1/day, regardless of
  ladder position — the safety floor a brand cannot configure away. `RECOVERING` climbs the
  ladder one rung per 7 consecutive healthy days, starting at rung 0 (the floor) the day
  suppression is confirmed lifted, never exceeding the brand's own ceiling (`ladder[-1]`).
  `HEALTHY` is uncapped by this policy (the ladder ceiling). `ladder=None` (a brand with no
  `policy.daily_ladder`, `publishing/brand_policy.py`) falls back to `DEFAULT_LADDER = (1, 2, 3)`.
- `should_auto_pause(states_by_day, today, days=7)`: true only for a run of `days` **consecutive**
  SUPPRESSED days ending today — a gap or a missing day resets the streak; an ungraded day is not
  evidence of suppression.
- `consecutive_healthy_days(states_by_day, today)`: counts backward from the day *before* today
  (today itself is the day being decided, not yet counted toward its own recovery), stopping at
  the first SUPPRESSED or missing day.

**The write path (`queue.py`).** Two new tables, both brand-new (`CREATE TABLE IF NOT EXISTS`,
same zero-migration story as `platform_pauses`): `platform_daily_caps` (one upserted row per
platform — the LATEST decision, `mode` is `"shadow"` or `"live"`) and `response_log`
(append-only — every decision, shadow or live, so the 14-day shadow log can be compared against
what an operator would have done). `platform_pauses` gains a nullable `reason` column
(migrated via `ALTER TABLE` like the other post-launch columns); `pause_platform` takes an
optional `reason` (existing callers unaffected — default `None`), and the auto-pause path passes
`reason="auto:suppressed"`.

**The read path (`dispatch._select_due`).** The one funnel every caller (CLI, cron, console)
already goes through for shadow-row exclusion and pause enforcement now also honours
`platform_daily_caps`: a `mode="live"` row truncates that platform's due rows to
`cap - <published today>`, keeping the lowest `(scheduled_at, id)` rows first. There is no score function yet; `id` is the tie-break, and a score can later be
fed into `scheduled_at`'s ordering or added as a third sort key without changing the cap logic. A `mode="shadow"` row changes nothing about selection — it only logs
(`"shadow: would cap %s at %d (%d due, state=%s)"`) what a live cap would have limited.

**`apply_suppression_response(conn, *, brand, states, today, ladder, mode="shadow", now=None)`**
(`dispatch.py`) is the entry point: given the already-classified `{platform: {day: status}}`,
it computes each platform's cap and auto-pause decision, writes `platform_daily_caps` +
`response_log` (tagged with `mode`), and — **only in `mode="live"`** — calls `pause_platform(...,
reason="auto:suppressed")` when `should_auto_pause` is true. In `mode="shadow"` (the default, and
the only mode it is called with) nothing is ever paused and nothing ever limits a real
delivery; the log rows exist purely so the shadow window's output can be read back and compared.

**Reversal.** Every bounded-response decision reverses through the existing `queue.resume_platform` (no new
resume path) — a cap alone (no pause) reverses simply by the next `apply_suppression_response`
call recording a higher cap once the state improves.

### Nightly wiring and manual override

`gomoveshift metrics collect <project>` now ends by calling `apply_suppression_response`
(`cli/handlers/metrics.py::_apply_bounded_response`, right after the channel-daily pull) --
every project's nightly `metrics collect` invocation reaches it once the cron is enabled, so
the shadow log fills without anyone running anything by hand.

- **YouTube and Facebook.** YouTube is classified via `tracking.suppression.classify_youtube_shorts_feed`
  over every stored `channel_daily` row for the brand. Facebook is classified via
  `classify_facebook_reel_reach` over an interim daily reel-plays series that
  `tracking.facebook_series.facebook_reel_plays_series` builds from the nightly cumulative
  observation log (`short.observation_log["facebook"]`; there is no per-day Facebook pull in
  `channel_daily`): the per-fetch delta of `reel_total_plays` between consecutive MEASURED entries
  of a post, attributed to the New York day of the later fetch and summed over shorts; a counter
  that moved backwards is skipped, and a post's first observation counts in full only when it was
  fetched within a day of posting. Days with no observation are absent, so gaps shorten the
  detector's windows. Each classified platform prints `SUPPRESSION <platform>=<status> days=<n>`
  to the cron log; Facebook prints `WARMUP` until 28 days of series exist. Facebook has no
  recovery branch: once SUPPRESSED it stays so (known gap).
- **Shadow by default, always.** `--response-mode {shadow,live}` on `metrics collect` (default
  `shadow`) is the only switch; the scheduled line that runs `metrics collect` is operator-only to edit, so
  going live for real is a human editing that line, not a code change in this repo. A missing/no
  channel-daily data prints `Bounded response: no channel-daily rows for <brand>` and does nothing
  else -- no DB is even opened.
- **Ladder** comes from `publishing.brand_policy.load_brand_policy(brand).daily_ladder`
  (`None` falls back to `response_policy.DEFAULT_LADDER` the same as everywhere else).
- **Manual override sticks.** A `platform_daily_caps` row whose `reason` starts with `manual:`
  is left untouched by `apply_suppression_response` -- it logs one `response_log` row
  (`action="skipped-manual"`) and moves on, never overwriting the cap or state an operator set.
  `queue.clear_platform_daily_cap` removes the row entirely (the override is lifted; the next
  nightly run's own decision applies again).
- **Console.** The Publishing tab (`console/pubqueue.py::platform_daily_caps` /
  `set_state_override`, `console/app.py`'s `POST /publishing/override-state`) lists every
  platform's current cap decision and lets an operator pin a state (`HEALTHY` / `RECOVERING` /
  `SUPPRESSED`, with an optional explicit cap -- defaulting to `daily_cap_for(state, None, 0)`) or
  `clear` a previous override. Every override this route writes is `mode="live"` with
  `reason="manual:<STATE>"`, matching the convention `apply_suppression_response` checks for.

**Not built here (follow-ups).**
- **Score-based ordering** — if a future score needs to reorder `_select_due`'s cap truncation beyond
  `(scheduled_at, id)`, that is a `_apply_daily_caps` change, not a new funnel.
- **The 14-day shadow window** (`mode="live"` for the first time) is the operator's call, made either by
  editing the `metrics collect` scheduled line to add `--response-mode live`, or by a manual console
  override on a specific platform -- not a config flag flipped by code in this repo.
