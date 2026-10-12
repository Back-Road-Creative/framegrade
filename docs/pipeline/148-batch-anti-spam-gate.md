## Batch anti-spam gate

`gomoveshift.publishing.anti_spam_gate.check_batch` is a pure function over
the `PublishRecord`s one `enqueue-plan --apply` run is about to spool. It is
the one **upstream gate** in the anti-spam design (its tier rule): batch-wide uniqueness/count is a cross-cutting property no
single sidecar's generator call can see, so it is checked once here, before
`cli.handlers.enqueue_plan.cmd_enqueue_plan` writes anything to the queue —
right after the rolling-ceiling gate and before the spool loop. A violation
refuses the **whole plan**: nothing is written, every offending record is
listed.

## Rules

`check_batch(records, *, per_location_cap)` returns a list of `Violation(index,
platform, rule, detail)`, `index` being the record's position in the batch
(after the existing `select_capped_shorts` selection — a capped-out short is
never checked, matching the other pre-spool gates):

1. **`legacy-title`** — the YouTube title, normalised (`.strip().lower()`),
   matches an entry in `LEGACY_TITLE_POOL`: a frozen, normalised copy of the
   legacy `TITLE_TEMPLATES`/`EXTENDED_TITLE_TEMPLATES`/
   `_pick_tiktok_title` pool, owned by `anti_spam_gate.LEGACY_TITLE_POOL_RAW`
   and re-exported by the test fixture `tests/fixtures/legacy_title_pool.py`
   (see Follow-up below).
2. **`shared-prefix`** — two YouTube titles share the same first three words
   after normalising to lowercase alphanumeric tokens (punctuation
   stripped). Titles under three words are never compared. Both/every
   sharing index is reported.
3. **`caption-collision`** — within one record, two platforms' comparable
   text are identical after `.strip()`: YouTube's `description`,
   Instagram's `caption`, Facebook's `description`, TikTok's `caption`.
   Empty text is never compared.
4. **`location-cap`** — more than `per_location_cap` records in the batch
   share the same location key (the first entry of `sampled_regions`;
   records with no sampled regions are excluded, matching the region
   invariants' own no-op rule for GPS-less/legacy content). Every index
   beyond the cap for that location is reported. A `None` cap (the default —
   most brands ship no `policy.per_location_cap`) disables the rule
   entirely.

## Wiring

`cmd_enqueue_plan` calls `check_batch(_planned_records(sidecars, max_shorts),
per_location_cap=policy.per_location_cap)`, reusing the same
`load_brand_policy(current_brand())` call the weekly-ceiling check (`weekly_ceiling`) already
makes — one policy load, two consumers. On any violation it prints `Error:
batch anti-spam gate refused the following short(s); nothing was written to
the queue:` plus one `index=… [platform=…] rule=…: …` line per violation and
returns `1`, the same refuse-whole shape the media-verification, privacy,
and rolling-ceiling gates already use.

## Follow-up (index-plumbing cleanup) -- done

`tests/fixtures/legacy_title_pool.py` used to carry its own copy of the same
100 template strings. It now re-exports
`gomoveshift.publishing.anti_spam_gate.LEGACY_TITLE_POOL_RAW` as a list, so
the pool has one home; `tests/test_publishing_anti_spam_gate.py::
test_fixture_pool_is_the_gate_pool_not_a_copy` pins the fixture to the gate's
tuple.
