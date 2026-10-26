# Volume and cadence — decided by Waypost

GMS no longer recommends a volume. `gomoveshift volume-report` and `recommend_volume`
(report-only) were removed. If you schedule these nightly, run them in this order:
`gomoveshift metrics enrol-channel`, `gomoveshift metrics analytics-probe`, then
`gomoveshift metrics export-evidence` (settled day-7 facts, no thresholds, no recommendation)
piped as a file into `python -m seo_manager.analytics.volume_decide --evidence - --db <store>`,
where `<store>` is `Path(get_config().seo.storage_path) / "cadence_decisions.sqlite3"`.
`volume_decide` appends one `VOLUME_DECISION knob= region= action= value= reason= ...` line
per knob to the decision store and to the scheduler's log; the pipeline reads that store.

A nightly wrapper should log one line per failed step (for example `VOLUME_DECISION_FAILED stage=<import|store|export|decide>`) and still exit 0. A daily digest can render the section "Volume & cadence (auto-decided by
Waypost)": non-hold decisions with from->to and evidence, each knob's latest hold reason, the
"running on seed" age, and alarms for a missing decision line (none in >36h), a failed step, unreadable state and `ANALYTICS_*` conditions. Nothing in it asks for a
number; the one human act it names is the one-time `gomoveshift youtube-auth` consent when
Analytics reports `ANALYTICS_SCOPE_MISSING` or `ANALYTICS_NO_TOKEN`.

The wrapper depends on `seo_manager` being importable in the service venv;
it should check `import seo_manager` first and log a failed `import` stage if not.

## Library hygiene — `gomoveshift library-hygiene`, dry-run by default

`gomoveshift library-hygiene` reads the LIVE uploads listing (three read-only Data API
calls), keeps the templated shorts (≤180 s, title matching a `{location}` template from
`publishing.anti_spam_gate.PRE_POOL_TITLE_SHAPES` (the four shapes used before the pool),
`LEGACY_TITLE_POOL_RAW` or `social_generator.TITLE_FRAME_TEMPLATES`;
series = the captured location, an unmatched title is skipped, never guessed), runs the pure
selector below, and prints the candidates. `--apply` sets at most `--max` (50) of them private,
heaviest series first, appending one ledger line with the prior status to
`<GMS_OUTPUT_DIR>/library-hygiene/<date>-ledger.jsonl` BEFORE each call (each row also
carries the `series` it was selected under); `--undo <ledger>`
restores every row. `--ids ID…` privates an explicit list instead (the unlisted test renders).
The command replaces doing this by hand in YouTube Studio; the selector module itself
stays pure and network-free.

```
gomoveshift library-hygiene                      # list, change nothing
gomoveshift library-hygiene --apply              # first 50, ledger written
gomoveshift library-hygiene --apply --ids A B C  # exactly these
gomoveshift library-hygiene --undo <ledger>      # put them back
```

### Daily sweep (scheduled, optional)

A daily scheduled job can run `gomoveshift library-hygiene --apply` (for example at 04:10) as a
backlog drain, not a standing policy: schedule it only while candidates remain, and remove the
crontab line once a run finds 0 candidates. Every run's ledger stays under
`<GMS_OUTPUT_DIR>/library-hygiene/` for `--undo`.

### The tracking-store list (hand run, read-only)


`python3 -m gomoveshift.tracking.volume library-hygiene` lists public shorts
that are old, low-viewed, and part of a heavy series — candidates, batched ~50 at a time, heaviest series first. This module is read-only end to
end: no publishing-adapter import, no network call, nothing here can flip a short's privacy —
the CLI handler above holds the adapter and acts only on `--apply`.

A short qualifies only when it is strictly older than 30 days, strictly under 100 views, its
`privacy` is confirmed exactly `"public"` (never assumed — a short with no known privacy status
is excluded, not defaulted to public), and its series has strictly more than 5 qualifying
members. **Series** = `TrackingRecord.location` (falling back to `project`), the only
place-grouping the tracking schema carries today — there is no dedicated series field.
**Privacy** is read via `getattr(short, "privacy_status", None)` on `ShortRecord`
(`tracking/schema.py`), which has no such field yet; until the schema grows one this always
returns `None`, and `library_hygiene_report()` says so explicitly ("no short in the tracking
store carries a privacy status yet") instead of silently reporting no candidates.

```
LIBRARY_HYGIENE_CANDIDATE id=example-2030-01-01#3 series='Example County' views=42 age_days=61
```

The YouTube Analytics client (`YouTubeCollector._analytics_client`) uses the same credential precedence as the Data API client: an explicit token file if present, else the shared SOPS credentials. That token must carry the `yt-analytics.readonly` scope, or every Analytics query returns empty.

`gomoveshift metrics analytics-probe` makes one tiny live Analytics query and prints `ANALYTICS_OK` or the reason line (`ANALYTICS_SCOPE_MISSING`, `ANALYTICS_NO_TOKEN`, `ANALYTICS_UNAVAILABLE reason=...`), exiting 0 on ok and 2 otherwise; the digest will pick that status line up. `gomoveshift metrics enrol-channel` registers every channel upload (all privacies) the tracking store lacks and prints `EVIDENCE_COVERAGE shorts_total= long_total= oldest= newest=`. `gomoveshift metrics export-evidence` prints the EvidenceBundle v2 JSON to stdout: per YouTube short a row with `post_format: short` and `day7: {views}` (`source: analytics_daily`, else the measured snapshot nearest age 7 days within 12h, `source: snapshot`); per Facebook short a row from the measured Facebook observation nearest age 7 days within 12h (an observation reporting no counter is not chosen), `source: snapshot`, `post_format` `reel`/`video`/null and `day7: {views, reel_total_plays, reel_initial_plays, watch_time_seconds}` with null for anything unreported (Reel plays are never written into `views`; the consumer decides which counter is reach); plus settled long-form day-7 views and `health` counts (`n_registered` counts shorts on either platform). Shorts with no row are listed under `health.excluded` (Facebook ones carry `platform: facebook`) and counted on stderr as `EVIDENCE_EXCLUDED n=<k> reason=no_settled_series`. It carries no threshold and no recommendation. `metrics collect` fetches a settled window's daily series once and stops on a quota 403/429 with `ANALYTICS_BUDGET_EXHAUSTED remaining=<N> reason=quota`.
