### SEO corrections editor + comments inbox (`console/seo.py`, `GET /seo`, `POST /seo/corrections`, `GET /comments`)

`GET /seo` renders the place-name corrections ledger the pipeline already reads
(`pipeline/validation/place_corrections.py`'s `GMS_CORRECTIONS_PATH`): the `corrections`
(`wrong`/`correct` pairs) and `denylist` lists, straight off `console/seo.py::read()`, alongside the
`revision` (a digest of the exact bytes) they were read at. Missing or empty renders as an empty
state -- an uncurated ledger is not damage. `POST /seo/corrections`
takes either a `wrong`+`correct` pair or a `denylist_name` and appends it via a safe
load-modify-dump (`yaml.safe_load`/`yaml.safe_dump`, creating parent dirs as needed) -- never
reimplements the pipeline's loader, so the very next `load_place_corrections()` call (a fresh GET,
or the drive pipeline's own read) sees the save immediately. Neither field present is a 400.

**Every refusal hands the page back, and the 409 is the one that matters.** Both statuses now
re-render `seo.html` with the submitted text rather than a separate error page. `form.kind` says
which of the page's two forms was submitted, so only that one refills -- a denylist name in the
correction boxes would be text in a form nobody touched. On a **409** the context is deliberately
**re-read**, not reused: a revision conflict means another editor wrote while this page was open,
which is precisely when discarding the operator's typed correction is worst, and the page they get
back now shows the entry that other editor added. The **fresh** revision rides along in the hidden
field, so the next submit is an informed one rather than a retry that fails identically forever.

**The writer is strict where the loader is lenient.** `load_place_corrections()` degrades a
malformed ledger to "no corrections" so a bad byte cannot stop a drive; a writer that borrowed that
tolerance read `{}` from bytes it had failed to parse, then dumped a fresh two-key file over them,
destroying hand-curation silently. `console/seo.py` refuses instead: anything that will not parse as a mapping raises
`LedgerCorruptError` from `read()` and from every edit, bytes untouched (`GET /seo` renders the
damage with the editor withheld, `POST` answers **409**); every save re-reads and compares the
caller's `expected_revision` against the file as it is now, raising `LedgerConflictError` (**409**)
when they differ, so a second operator's edit is never overwritten (both forms carry their rendered
revision in a hidden field; a request with none skips the check); and the dump goes to an fsynced
sibling temp `os.replace` swaps in, so a failed write leaves the ledger whole, no `.tmp` behind.

**The revision check cannot close a window it is inside.** Atomic replacement makes a torn
ledger impossible and says nothing about a lost one: two editors could both read one snapshot,
both compare `expected_revision` against it and pass, and the second `os.replace` would drop the
first one's rule with no error on either page -- neither editor's `_check_*` validators can see a
rule that landed after its own read. So the whole read-check-mutate-replace in `_edit` runs under
`pipeline._file_lock.make_blocking_file_lock`, the same blocking `fcntl.flock` `.lock` sidecar
`content_ledger` and `pending_journal` already use for this bug class -- beside the ledger, on the
filesystem the atomic replace already requires, advisory and released on fd close or process death
so a crashed editor cannot wedge the page. The second editor serializes behind the first and then
reads what the first wrote, which is what makes the revision check and the validators decisive
rather than advisory. Blocking rather than refusing on contention: the waiting editor is a person
who has already filled in the form, and the wait is one small YAML dump long. Every exit path --
corrupt bytes, a stale revision, a refused rule -- leaves through the context manager, so a refusal
never holds the lock.

The same asymmetry applies to the *content* of an edit, not just the bytes around it.
`load_place_corrections()` silently drops a rule with a blank side, and `mapping[wrong] = correct`
silently keeps only the LAST rule for a given name -- so a ledger can hold rows that render on the
page exactly like working ones while doing nothing, or something other than what they say. The
writer raises `LedgerInvalidError` (**400**, the message is the point) rather than record one:
a blank `wrong`/`correct`/denylist name; a correction whose two sides are the same name
(case-insensitive, stripped, matching `_word_re`'s own matching) and so could never change any
text; a second correction for a name already corrected, naming the existing target; a repeated
denylist name, which would make `find_denylisted` report it twice in one failure. Two cross-checks
catch a ledger that contradicts itself: a correction whose *result* is denylisted would be
substituted in by `title_builder` and then hard-failed by `check_metadata_quality`, and
denylisting a name some correction already produces is the same contradiction from the other side.
Denylisting a correction's *wrong* side stays legal -- that is belt-and-braces, not a conflict.
Validation runs inside the same re-read as the revision check, so a rule can't be cleared against a
stale copy and then collide with one that landed in between. A half-filled correction now reaches
the writer rather than the route's generic "provide wrong+correct or a denylist name", because the
writer can say which side is missing and the operator plainly meant to add a correction.

Launching a `seo-manager` job (`sync`/`report`/`suggest`) from this tab is a follow-up -- it needs
a cross-service argv the console's `jobs.py` launcher doesn't build yet.

**Performance half (`console/insights.py`).** The tab has been called "SEO & Performance"
since it was built and showed no performance at all. `insights.performance_view()` reads the
top videos by views out of the event log the pipeline already writes -- the path comes from
`SEOConfig.events_jsonl`, never rebuilt here, and a test pins the two against each other so the
tab cannot end up confidently reading a different file. `seo_manager` is an optional dependency,
so the import is lazy inside the function like every other cross-service call in this service.

Each row carries the four things that make a rate readable, all of them straight from
`AnalyticsTracker` and none of them invented here: the rate, **what it measures**
(`ctr_source`, rendered in words via `SOURCE_LABELS` -- a card click rate is in-player and says
nothing about a thumbnail), **its denominator** (`ctr_basis` -- recomputed from counts we hold, or
a rate the platform reported), and **when we last heard anything** (`last_updated`, plus
`data_points`). An unmeasured rate stays `None` through the reader and renders as the words "Not
measured": `ctr_percent` is `None` on exactly those rows so there is no numeric form for a
template default to find, because a `0` is the real measurement "every viewer shown this declined
to click".

**Observation window and freshness.** Each row also carries
`first_updated` -- the tracker's own earliest-sample timestamp for that
video, on the same never-invent contract as `last_updated` -- and
`observation_window_days`, the whole days between the two, computed here
because the tracker does not compute it. Either timestamp missing renders
the window as "unknown", never a fabricated `0`, which would read as "we
watched this video for zero days" instead of "we don't know". At the page
level, `performance_view()` also returns `data_as_of` (the newest
`last_updated` among the rows actually shown) and `oldest_sample` (the
oldest `first_updated` among them), each `None` when no row carries that
timestamp; `seo.html` states both in one line above the table, and only
when at least one is present -- a fresh install with no rows makes no
freshness claim at all. No staleness threshold or verdict is introduced:
this states what was observed, not whether it is enough (that call is
the operator's, same as the evidence-age module's `STALE_AFTER_DAYS` in
the separate SEO service).

The ten rows shown are a slice, and the page says of what: `total_content_with_performance`
and `top_performers_cap` come straight from `AnalyticsTracker.get_summary`'s `performance_metrics`
(`.get(...)`, never re-derived from `len(rows)` -- that would state the cap as if it were the
population against an older `seo_manager` lacking the key, in which case the page says the
population is unknown rather than guessing). The site-wide totals in that same
`performance_metrics` dict -- pooled views/clicks/impressions and a click rate through the same
`_resolve_rate` contract as each row, with its own `ctr_basis_label` -- are shown too, visibly
separate from the per-video table, because "ten highest-view videos" and "the whole channel" are
different questions and the screen used to answer only the first while silently discarding the
second.

Three degradations, each a sentence on the page rather than an exception, and each distinct:
no log yet; `seo_manager` not installed; a log that exists but yielded nothing (the store skips
unparseable lines with only a warning, so "zero views everywhere" and "we could not read this
file" would otherwise look identical). The log's existence is checked BEFORE any
`SEOEventStore` is constructed -- its `__init__` mkdirs, and viewing a tab must not write to
disk. The read sits outside the ledger's `corrupt` branch on purpose: an unreadable corrections
ledger must not hide the numbers, and unavailable numbers must not withhold the editor.

**Trust badge.** `performance_view()` also returns `unknowns` (a list) and
`unknowns_badge` (one summary sentence) alongside the keys above -- both derived from the exact
same state the rows and totals were just built from, never static copy that could drift from the
numbers beside it. A row with no measured rate, a row whose `ctr_source` was never declared, an
unmeasured site-wide rate, and an unknown population count each add their own sentence; every
early-return path (no log, `seo_manager` missing, an unreadable log) carries the `hint` string
through as its one-item `unknowns` list instead of a separate shape. `seo.html` renders the badge
sentence above the performance table and, when `unknowns` is non-empty, a collapsible `<details>`
panel listing each one. A/B thumbnail-test refusal reasons live in the separate SEO service
(`thumbnail_ab_tracker.py`) and are not surfaced here yet -- there is no call site for that tracker
in this service.

`GET /comments` is read-only: it loads `pipeline/comments.py`'s `GMS_COMMENTS_PATH` JSONL ledger
(`load_comments()`) and renders each row's platform/video/text alongside a triage hint computed
via the existing `flag_reasons()` (comma-joined cues/place-names, or `clear`) -- no new triage
state is persisted. A missing comments store renders the same empty state as a missing ledger.

**Anomalies & suppression signals.** `tracking/anomaly.py` (per-post dip/spike) and
`tracking/suppression.py` (per-platform feed-suppression) have existed for several plans with only
a CLI/cron call site -- an operator who never tails that log has never seen either. `insights.py`
gains `anomaly_view()`, a second reader on the exact same never-raise, never-invent-a-number
contract as `performance_view()` above: it opens `TrackingStore` read-only, calls
`check_feasibility()`/`detect_post_anomaly()`/`classify_youtube_shorts_feed()` unmodified, and
degrades any read failure to the same empty shape as a genuinely empty store, logged rather than
shown. Neither kernel gained a `shadow` field -- `AnomalyFinding`/`AnomalyRefusal` only stamp it
into `log_line()` -- so the view adds `shadow: True` itself before a row ever reaches the
template; the suppression classifier returns a bare `{day: status}` map with no statistic at all,
so its translation (latest status, consecutive days) never invents an n or p-value the classifier
never produced.

`seo.html` gains an "Anomalies & suppression signals (hypotheses, not proof)" section, outside the
`corrupt` branch (same placement rule as Performance: an unreadable corrections ledger must not
hide it, and it must not withhold the editor). It is absent entirely -- not blank -- when
`anomaly_view()`'s `has_data` is `False`, i.e. no platform has a tracked short AND no
channel-daily rows exist; present the moment either source has anything, even with zero current
findings, so "nothing anomalous" and "never checked" render differently. Every row is captioned
"hypotheses, not proof" / "Unvalidated" -- this is a statistics kernel with no operating call
site, and nothing in the pipeline acts on any row shown here.

**Wiring:** `app.py::_seo_context()` adds `context.update(anomalies=insights.anomaly_view())`
beside the performance read, so both `GET /seo` and a refused corrections POST carry the section;
`tests/test_console_seo_anomalies.py::test_the_live_seo_route_shows_the_anomalies_section` drives
the live route.
