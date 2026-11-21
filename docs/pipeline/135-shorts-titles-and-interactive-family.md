## Shorts title dedup and dropped Pace line (`gomoveshift/pipeline/social_generator.py`, `gomoveshift/pipeline/longform_metadata.py`)

Two independent fixes to the shorts/long-form social surfaces.

### Shorts title dedup against the published-titles ledger

`SocialGenerator._pick_title` now checks each candidate short title against
`longform_metadata.load_published_titles()` — the same `events.jsonl` ledger the
long-form dedup path reads (`longform_metadata.read_published_titles`), imported
lazily to avoid a circular import (the same pattern `title_builder.ensure_unique_titles`
uses for `assemble_final_title`). No new reader was written.

A collision is resolved **deterministically**: `_pick_title` walks forward through the
same project-seeded shuffled `TITLE_TEMPLATES` rotation already used for the primary
pick (`shuffled[(template_idx + offset) % len(shuffled)]` for increasing `offset`),
returning the first candidate not already in the ledger. Same inputs (project name,
series part, short index, ledger contents) always produce the same alternative — never
a fresh random draw. If every template in the pool collides (unlikely — the pool is
larger than a channel's realistic title density), it falls back to the primary pick;
the channel-level dedup gate on the long-form path is the hard backstop for a genuine
duplicate that slips through.

Also fixed: `location_short_full` kept a leading curated title word verbatim (e.g. "To
Clonakilty"), which leaked into the shorts YouTube description as "driving through To
Clonakilty" — a leading preposition read as a place noun. It now goes through
`parse_lf_label(...).place`, whose title-prefix strip (`_TITLE_PREFIX_RE`) is anchored
at the string start, so a leading "To "/"Through "/etc. is stripped while a mid-string
arc label ("Fancy Gap to Hillsville") is left untouched.

### Place-free "Where is this?" interactive family — DEFERRED

Attempted and reverted. `INTERACTIVE_CAPTION_TEMPLATES` folded into the
existing per-project `CAPTION_TEMPLATES` shuffle satisfied the "impossible by
construction" and "reuse the existing selection mechanism" requirements, but the full
targeted test sweep surfaced 35 failures across `test_social_quality.py`,
`test_shorts_yt_description_category.py`, `test_phase_d.py`, and
`test_social_generator_safety.py` — all rooted in an existing hard invariant that every
short's caption/tags/description names the location (e.g.
`test_tags_include_location`, `test_caption_contains_destination`), which this family
necessarily violates for whichever shorts land in it. `ContentValidator` also flags at
least one interactive template's own wording ("guess below") as self-contradictory
against its own "location visible" heuristic.

This is a genuine design conflict, not a mechanical bug: making the family safe would
mean either rewriting ~35 assertions across four test files to carve out an
interactive-short exception (broad, needs careful per-assertion review of what each
protects) or redesigning selection so it can't land within any of the existing quality
tests' seeded short ranges (which risks making the "reuses the existing mechanism,
no new knob" requirement hollow). Left for later work with room to resolve that
conflict deliberately. The reverted design was a deterministic template-pool rotation plus place-free caption/tags branching.

### Dropped the long-form 🚗 Pace line

`_build_gms_description` no longer splices a `🚗 {label}: N mph/km-h avg · {character}`
line into the long-form description, for either the mph or km/h variant. A bare average
speed over an unwindowed GPS trace read as a false-precision claim, and the derived
"character" (highway cruise / winding roads / town pace) duplicated the terrain
descriptor already present in the title/body. `_format_speed_line` and `_speed_character`
are unchanged and still directly unit-tested — they're just no longer called from
`_build_gms_description`.

### Hand titles

`SocialGenerator` accepts `titles_file` (a `Path` to a drive-level
`<project_dir>/social/titles.yaml`), an optional override. This is the code path hand titles take: an operator (or a future composed-title tool) writes one YAML mapping, and every short covered by it skips the template pool
entirely.

**File shape** — a mapping keyed by `short_index` (int, 1-based, the same
post-`index_offset` index `generate_all` assigns to `SocialPost.short_index`):

```yaml
1:
  title: "Guess this coastal road"
  caption: "A quiet coastal drive."   # optional
  interactive: true                   # optional, default false
```

**Precedence** — when `titles_file` covers an index, `generate_post` uses that entry's
`title`/`caption` and never calls `_pick_title`/`_build_caption` for it (the YouTube
`100`-char truncation and `ContentValidator.enforce` gate still run, same as a pool
pick). A missing `caption` becomes `""`, not a pool caption. `interactive` lands on
`SocialPost.interactive` (default `False`, excluded from `to_dict()` so every existing
serializer stays byte-identical) — not yet acted on outside this round trip; the interactive-family handling reads
it to build the place-free family.

**Optional on every run** — an index NOT covered by the file (or no file at all)
composes its own stamped title on the drive path, publish runs included; `titles.yaml`
is never required. A malformed file (not a mapping, a non-int key, a missing/non-string/
empty `title`, a non-string `caption`, or a non-bool `interactive`) is a `ValueError` at
load time naming the path and the offending key. A hand title still wins over the
composed one and is still place-checked (`_check_hand_text_place`).

**The place gate owns the slot** — the old publish-run coverage gate
(`HandTitleMissing`, `assert_hand_titles_cover`), the audit-pass `HAND-TITLES` banner and
the old DoD #20 are retired. What stops a wrong place is
`phase_d_scheduling.assert_all_places_verified`: upstream of chapters and scheduling,
a publish run refuses every short whose place its own clip's GPS did not verify
(`unverified_place_shorts` builds the offender list); the audit pass logs one
`UNVERIFIED-PLACES` warning. DoD #20 `places_verified`
(`tools/validate_pipeline_output.py`, both modes) FAILs the audit on the same set, and
PASSes "skipped" when the state has no `shorts_per_lf`. Both also name a short whose
place is verified but whose `TitleFacts` cannot be read from the day sidecar
(date/ordinal/duration), since its stamped title cannot be composed.

**The ledger rule** — a hand title still runs through the same
`longform_metadata.load_published_titles()` channel ledger `_pick_title` checks. Unlike
a pool pick (which walks forward to the next unpublished template), a hand-title
collision is a hard `ValueError` naming the index and the title — there is no pool to
fall back to, and silently substituting a different string would defeat the point of
hand-authoring it.

**Where it's wired** — `stages/social.py` (the single-video stage) and the drive path
(`phase_d.py`) both pass `titles_file=<project_dir>/social/titles.yaml`. No CLI flag.


### A short names only the place its clip was filmed

A short is ~30 s of a drive; its title/caption may name ONLY the place its own
clip GPS voted (the short's label, `self.location`).

- **Direction framing refused.** A hand title or caption (`titles.yaml`) using
  "to / into / toward / approaching / entering / heading to|for / bound for /
  on the way to / en route to" before a capitalised word raises `ValueError`
  naming the short_index and text ("Back roads into Gettysburg, PA"). The
  composed frame `{terrain_desc} that lead into {location}` is now
  `{terrain_desc} in {location}` (frame count stays 6).
- **Parent-LF places refused on a variant.** `_per_short_variant` records
  `_parent_place_tokens` — capitalised words (>=4 letters, generic road/region
  words excluded) of the parent's place strings and `resolved_location.destination`
  that the short's own label lacks. A hand title/caption containing one raises.
- **GPS-bearing shorts always use the variant.** `generate_all` builds
  `_per_short_variant` for any short with a non-empty `per_short_regions` entry,
  even when its label equals the parent's, so it never inherits the LF's
  destination POI, nearest POI, road(s), county or terrain. GPS-less shorts keep
  the parent generator.
- **A published short names only the place its own clip's GPS verified.**
  - Phase D (`assert_all_places_verified`, `phase_d_scheduling.py`) runs right
    after the hand-title gate and before chapters/scheduling. Every short in
    `shorts_per_lf` (each is posted, whether or not `short_global_index` lists it
    — a checkpoint from before the global index has no index) must have a sidecar `ost_text` label, non-empty `regions`, and
    `place_source == "clip_gps"`. A publish run raises `RuntimeError` naming
    EVERY offender at once; an audit run logs one `UNVERIFIED-PLACES` WARNING and
    continues.
  - On a publish run `_build_social_posts_per_lf(require_verified_places=True)`
    removes the LF-label stand-in: the per-short location list comes only from
    sidecar labels, and a short without a verified place raises before any
    generator is built. Audit runs keep the legacy stand-in.
  - `day_pipeline._render_day_shorts` refuses before encoding when a publish run
    (`state.youtube_upload is True`) would label a GPS-less short from anything
    but its own clip GPS. Audit runs keep the GPS-less anchor.
  - A per-short variant derives its country/state tags from the short's own
    `regions` (`_region_tags(own_regions=...)`), not the parent's country. Regions
    carry no country, so only a US state is derivable: a non-US short's own data
    cannot yet give its country and gets no country tag (never the parent's).

### Composed titles replace the fixed template pools

`TITLE_TEMPLATES` and `EXTENDED_TITLE_TEMPLATES` (86 fixed-string templates) and the
15-entry TikTok-only pool inside `_pick_tiktok_title` are gone. Absent a hand title
(a hand title still wins first), `_pick_title`/`_pick_tiktok_title` now COMPOSE a title from
grounded fact slots already on the generator: `location_short`, `terrain_desc` (both
always populated — default "roads" when terrain is unknown), and the optional
`road_name`, `county`, and a new `moment_kind` slot.

**`TITLE_FRAME_TEMPLATES`** (`social_generator.py`) holds at most 6 grammar frames — a
frame is sentence *shape* only (`"{location}, by {terrain_desc}"`,
`"{road} cuts through {location}"`, …) and never stands in as a complete title without
slot data. Every frame references >=2 distinct fact placeholders. `_compose_title`
filters to the frames whose placeholders are ALL backed by a non-empty slot for this
short (so a road/county/moment frame is skipped entirely when that fact is absent —
never rendered with a blank). **There is no decorator/cosmetic-suffix layer** — the
candidate set for a batch is EXACTLY the applicable frames, nothing pads it. A frame
reused with a different trailing emoji is a near-duplicate (shared ≥3-word prefix),
which is exactly the templated-spam signal the later anti-spam check refuses at
enqueue, so composition never produces that shape in the first place. `_pick_tiktok_title`
is folded into the same `_compose_title` (a distinct project-seed suffix so its frame
order differs from the YouTube pick, truncated to TikTok's 90-char max) — no second
pool.

**Capacity is honest, not padded** (legacy composition only -- the drive path composes
stamped titles and has no capacity ceiling, see "Stamped titles on the drive path"
below). `_compose_title` raises `TitleCapacityExceeded`
(`ValueError` subclass) when `short_index` exceeds the number of distinct applicable
frames — naming the short index, how many distinct titles are actually available, and
the fix: add a `titles.yaml` hand title for that index, or lower
`--shorts-target` (this applies to the legacy path, i.e. `stages/social.py`). The cap tracks the grounded facts on hand: location+terrain alone
supports 3 distinct titles, `+road_name` 4, `+road_name+county` 5 (matching the brand
per-location cap, `publishing/brand_policy.py`), `+moment_kind` up to the full 6-frame
set. Hand titles from `titles.yaml` bypass composition entirely and are unaffected.

**Regression guard, not a live path** — if fewer than 2 distinct fact slots are ever
available (a future field regression; `location_short`/`terrain_desc` are always
populated today so this never fires in normal operation), `_compose_title` raises
`ValueError` naming the short index instead of emitting an under-grounded title.

**Selection stays deterministic** the same way it always was: a project+series-seeded
shuffle (`_project_seed`) orders the applicable-frame candidates, `short_index` picks
one, and the published-titles ledger walk still applies — a collision walks
forward through that same shuffled candidate list (never a random re-pick), so the same
inputs always resolve to the same alternative. (The stamped drive path walks the same
ledger but rotates the BODY only; its stamp never changes.)

**`moment_kind` (new fact slot)** — `short_planner.ShortRange` gained an optional
`moment_kind: "peak" | "sustained"` field: `plan_shorts` sets it to `"sustained"` when
`_grow_to_moment_length` actually grew that window past its fixed search width (still
gaining strength at the edge), `"peak"` otherwise (a fixed-strength moment that never
grew). `SocialGenerator(moment_kinds={short_index: kind, ...})` is an optional
constructor arg composition may read as the `{moment}` slot; absent it, that slot is
simply unavailable — never a raise. Wiring `plan_shorts`' per-short `moment_kind` into
the `SocialGenerator(moment_kinds=...)` call at the pipeline-stage/drive-orchestration
level is not done here (kept out to stay small); a later change threads it
through `pipeline/stages/social.py` and the drive orchestrator.

**Legacy single-video stage** — `SocialGenerationStage` (single-video only) defaults
`num_shorts` to 15 with no `road_name`/`county` (bare capacity 3), so it fails loud
(`TitleCapacityExceeded`, named in the failed `StageResult`) past 3 shorts absent a
`titles.yaml` entry per index — intentional: this stage is where the legacy 15-shorts
spam came from. A hand title now bypasses TikTok composition too, not just YouTube.

**Retired pool, one home** — the retired 100 literal template strings
(`TITLE_TEMPLATES` + `EXTENDED_TITLE_TEMPLATES` + the old TikTok pool) live in
production as `gomoveshift/publishing/anti_spam_gate.py:LEGACY_TITLE_POOL_RAW`; the
batch anti-spam gate (doc 148) refuses any shipped title that still matches them.
`tests/fixtures/legacy_title_pool.py:LEGACY_TITLE_POOL` is a thin re-export of that
tuple as a list so tests keep their historical import path —
never a second copy of the literal;
`tests/test_publishing_anti_spam_gate.py::test_fixture_pool_is_the_gate_pool_not_a_copy`
pins the fixture to the gate's tuple.

### Fact titles on the drive path (2026-11)

A drive Short's title is the most specific place its own clip measured and nothing
else: `Tryon`, or `Polk County` when no town is accepted. The final-text review
(`seo_manager.analytics.final_text_review`) backs one cited place per title (a town
held for every retained second, else the county the clip voted) and refuses a road in
a title, so a "Town, County" title or a road title is not possible without loosening
the review, and the review is unchanged. No "Nth stretch" stamp, no "{secs} seconds",
no weekday, no filler.

Candidates are ordered town, county, then the Short's `location_short`
(`_title_place_bodies`). `_fact_title` takes the first one that fits, is not in the
published ledger, is not already used by this batch (`used_titles`, shared across the
LFs of one drive when the caller passes `generate_all(used_titles=...)`), shares no
first-3-word prefix with a batch title, and does not open like a spooled YouTube text
(the queue gate's `cross-batch-repeat`). On a collision the next measured place is
used; when none is left that one Short takes its stamped title (below) and the
generator logs `no unused measured place for the ... title`. A Short whose
`place_ok` is false keeps the place-free stamped title. `yt_title_body` is no longer
assigned (a fact title has no body to vary); `SHORTS_TITLE_BODY_ARMS` shapes only the
stamped fallback.

**One title ledger per drive.** Phase D hands every LF's `generate_all` the same
`used_titles` dict, so a measured place titles one Short of the drive, not one per LF.

**Queue and bracket sync re-derive fact titles.** `plan_bracket_sync.ordinal_basis` takes a
Short's day ordinal from the plan itself: the day sidecar beside its media
(`shorts_ost_audit_<DATE>.json`, start-time order, the same one Phase D reads), else the
old title's verified stamp (a plan written before fact titles), else its position among the
plan's Shorts of that recording date. It never parses "Nth stretch" out of a fact title.
`generated_title` is the safety check that the old title is one this plan wrote: its stamp,
or exactly one of the Short's measured places (or the LF location / bracket label); any
other title is refused and left as it was. A discarded sibling (a stored Short the sync was
not asked to rewrite) names no place, so it takes no measured place, and its stored title
is seeded into `used_titles` so a rewritten Short cannot repeat it. Running sync over a plan
the generator just wrote changes nothing.

### Stamped titles on the drive path

The stamped title now serves only the fallback above and `place_ok` false. The drive
path passes each short a `TitleFacts` and `_compose_title` returns
`"{stamp}: {body}"` there. The legacy <=6 frames and `TitleCapacityExceeded` now serve only
callers with no facts (`stages/social.py`); audit runs no longer raise it at short 4.
`titles.yaml` stays an optional override on every run, publish included.

**Format.** Stamp = `"{Mon} {D}, {k}{st|nd|rd|th} stretch"` (fixed month tuple, never
locale `%b`; `D` has no leading zero; `11th`-`13th` are `th`), e.g.
`Sep 14, 3rd stretch`. Body, place verified (`STAMPED_BODY_TEMPLATES`): `{location}` /
`{location}, on a {weekday}` / `{secs} seconds, {location}`. Body, place not verified
(`STAMPED_PLACE_FREE_TEMPLATES`, audit runs only -- a publish run refuses those
shorts): `{secs} seconds on a {weekday}`. No preposition precedes `{location}` (a town
label is the nearest settlement, not proof of containment) and there is no
`{weekday} drive` shape; a static test pins both.

**Facts (per short, read at composition time).**

| Slot | Source | Why it cannot become a place claim |
|---|---|---|
| `{location}` | the variant's `location_short`: first comma component of the short's own sidecar `ost_text`, no state | filled only when `place_ok` (regions non-empty AND `place_source == "clip_gps"`, the same rule as the verified-place gate); reads no other generator field, so no LF name, destination or neighbour enters |
| date, `{weekday}` | the short's own `day_<YYYY-MM-DD>` dir (the pipeline's day key) | calendar values, not place tokens |
| ordinal `k` | rank's position in that day's `shorts_ost_audit_<date>.json`, sorted by `(start_sec, rank)`; the sidecar spans every LF of the day | a pure count; a failed render keeps its row so later shorts are never renumbered |
| `{secs}` | `round(end_sec - start_sec)` of the same row | a number the viewer can see |

Excluded: `median_speed_mps` (pace claim), `regions`, LF-level terrain/road/county/POI,
global index, day-of-trip. Any unparsable path/date/row gives `None` facts (legacy
composition for that short).

**Place variety (`social_generator.choose_place_body`).** Near-zero-view Shorts lined up with
the same place named Short after Short. A short's place body (the title's `{location}`, the
description's place) now differs from the previous short's whenever its own clip measured
another place. The previous short is the one before it in `generate_all`, or
`previous_places` (the previous publish day's last short, its `SocialPost.place_variety`
bodies) for the first.

- Candidates, today's text first then most specific first: town, road, county. Towns and
  counties come from `publishing/clip_place_evidence.clip_towns` / `clip_places` (the places the short's
  final-text review accepts: a town held for every retained second, a county the clip's GPS
  voted), so a varied title cannot be refused by that review. A road is a **description-only**
  candidate: the review cites one place in a title and treats a road as tag evidence, so a road in
  a title is refused (checked against `build_short_final_text_review`).
- `choose_place_body(candidates, previous, *, kinds=None) -> PlaceChoice(body, reason, kind)`
  is pure and deterministic; `candidates[0]` is today's body. Reasons: `distinct` (not a
  repeat; kept), `varied` (a repeat replaced by the first candidate that is not a repeat),
  `no_alternative` (a repeat, no other measured place; today's text kept).
- Recorded per short as `SocialPost.place_variety` (`{"title": {...}, "description": {...}}`,
  in `social_posts.json`). Never invented: with no measured alternative the text is unchanged.
- Hand titles, interactive shorts and legacy (no `TitleFacts`) titles are untouched. The TikTok
  title and Meta captions still name today's place.
- Wiring: a caller composing per LF passes the previous LF's last `place_variety` bodies as
  `previous_places`; without it only the shorts inside one `generate_all` call are compared.

**Why titles are unique.** The batch gate keys on the first 3 tokens of a title
(`anti_spam_gate.title_prefix_key`: lowercased `[a-z0-9]+` words, trailing hashtags
stripped). A stamped title begins `"{Mon} {D}, {k}{sfx} stretch: "`, so its key is
exactly `(mon, str(D), f"{k}{sfx}")` (digits plus letters are one token; hashtags trail;
truncation only cuts the tail), which is injective in `(month, day, k)`. Within a drive,
the same day token means the same sidecar, so `k` is distinct (`day_title_ordinals` is a
bijection on unique ranks); different day tokens differ in `(month, day)`, unless two
dates are exactly a year apart, which the Phase D assert refuses. `k` is day-scoped, so
LF boundaries do not matter. An enqueue batch is a subset of the drive's shorts (a subset
of an injective set is injective); two drives filmed on one date are caught by the batch gate. The
body, ledger walk and seed never touch the first 3 tokens, and no legacy pool title
starts with a month token.

**Asserts (upstream of any render or enqueue).**
1. `day_pipeline._render_day_shorts` calls `assert_day_title_stamps(date, entries)`
   immediately before `shorts_gate_or_raise`, ahead of the burn: every entry gets a
   stamp (numeric `start_sec`/`rank`, no duplicate rank) and no two share a prefix.
2. `phase_d` calls `assert_title_prefixes_unique(..., where="phase D")` right after the
   posts are built and before the final validation gate and Gate C/D. Composed titles
   cannot trip it; a `titles.yaml` override can. The batch gate at enqueue stays as the backstop and
   uses the same key function (`shared_prefix_groups`).

## Per-platform captions (`gomoveshift/pipeline/social_generator.py`, `gomoveshift/publishing/plan_builder.py`)

The single shared `CAPTION_TEMPLATES` pool + `HOOKS` prefix is gone. `SocialGenerator`
composes ONE caption per platform (`SocialPost.captions: dict[str, str]`, keyed
`"youtube" | "facebook" | "instagram" | "tiktok"`).

**Facts only (2026-11).** A caption is the measured place and nothing else: one sentence
shape from the platform's own pool (`YOUTUBE_CAPTION_TEMPLATES`, `FACEBOOK_CAPTION_TEMPLATES`,
`INSTAGRAM_CAPTION_TEMPLATES`, `TIKTOK_CAPTION_TEMPLATES`), such as "Passing through Polk
County." The pools are three shapes each, disjoint across platforms (so no two platforms
collide), and every shape is a stable `caption_template_id` for the meta-learning ranker.
A terrain clause ("Coastal roads.", from `CAPTION_DETAIL_TEMPLATES`) follows only when the
terrain is classified (`coastal`, `mountain`, `winding`); `countryside` and unknown terrain
state nothing, and nothing says "Mostly roads", "Just roads", "stretch", a count of seconds
or a weekday. The drive path's per-Short clone carries no terrain (a clip's own GPS never
classified one), so its captions are the place alone. The YouTube caption's closing line is
"Filmed <Month YYYY>." (full month name), the plan builder adds the same sentence to the
other platforms from the recording date, and hashtags and experiment arms are unchanged.
An `interactive` short is its guess prompt alone (`INTERACTIVE_PROMPT_TEMPLATES`, one
distinct prompt per platform): no opener and no terrain clause, since either would hint at
the withheld place. A road is never named in a caption: the final-text review backs one
cited place per text. Variety across a batch comes from each Short naming its own measured
place (`desc_place`), not from padding the pools with filler; a batch at one place wraps
after three shapes per platform, and `cross-batch-repeat` stays the gate on a repeat.
Each pool walks a per-project shuffled order (`_rotating_pick`, seeded with `_project_seed`),
so output is deterministic per project + short. `SocialPost.caption` stays and is an alias for the Instagram text —
every legacy reader of `.caption` (`content_gates.py`, the Metricool CSV shared column,
`console/pubqueue.py`, `instagram_adapter.py`) keeps reading it unchanged.

**Mandatory fact slot.** Every composed (non-hand) caption must contain the place
(`self.social_location`) — enforced in `_build_platform_captions`, which raises
`ValueError` naming the short index and platform if a composed caption is missing it.
When the short is `interactive` (the `HandTitle.interactive` flag, read here by the interactive-family handling),
`LOCATION_WITHHELD` ("this route") fills the `{location}` slot on every platform
instead, so the place can never leak through a caption on an interactive short.

**No two platforms identical.** `_build_platform_captions` also raises `ValueError`
naming the short index and the two colliding platforms if any two platform captions for
the same short come out byte-identical. Distinctness across shorts is by construction: a batch never repeats an
opener until the pool wraps (three shapes), and the generic hashtags rotate per short
(`GENERIC_HASHTAGS`, `GENERIC_HASHTAGS_PER_POST` drawn with a `short_index` seed; the
place and region tags are never part of that draw). Across batches,
`anti_spam_gate.check_batch(..., recent_texts=...)` refuses (`cross-batch-repeat`) any
caption or YouTube title whose first 6 normalized words match text spooled in the last 60
days; `cmd_enqueue_plan` loads that text with `recent_platform_texts` from the sidecars
the queue's deliveries point at. The generator is built not to trip that rule:
`SocialGenerator(spooled_texts=None)` reads the same text read-only from the queue beside
`DEFAULT_SEO_STORAGE` (excluding its own project), and `_fresh_rotating_pick` walks the
project's shuffled opener order past every entry that would open like it, falling back to
the stale entries (which the gate then refuses) only when the fresh ones run out. The
shorts YouTube description's lead ("POV", "Dashcam POV", …, `YOUTUBE_DESCRIPTION_LEADS`)
rotates the same way, so "… driving through {place} Pt.N" no longer opens alike for every
drive at a place. No lead opens with a bare preposition ("Through …"): the plan grammar gate
(`find_grammar_issues`) refuses that shape and the whole publish plan with it.

**Shape per platform:**
- **YouTube** — description-style, 1–2 sentences naming place + terrain, a trailing
  `Filmed <month year>.` line, no hashtag wall (0–2 tags).
- **Facebook** — one line (a question or a fact about the place), ≤2 hashtags.
- **Instagram** — short line + recording-date emoji line + up to 8 hashtags (location
  hashtags per component, terrain tag, `#POVDriving`/`#RoadTrip`) — the legacy shared
  shape, held to 280 chars (only this platform's text gets that cut; YouTube's does not).
- **TikTok** — ≤150 chars, place + terrain, ≤4 hashtags.

**Place + region tags are mandatory (non-interactive, 2+ location components).** The
place tag (first component) and the region tag (last: `#Tennessee`, `#PA`,
`#CountyCork`) are on every Facebook/Instagram/TikTok caption and in the YouTube Shorts
description. They take their hashtag slots before generic tags (`#POVDriving`,
`#RoadTrip`, terrain, middle components), which fill the rest via the seeded shuffle; the
final order is shuffled per short. Over a length limit, generic tags drop first, then the
base sentence is shortened on a word boundary (never inside the place) -- a mandatory tag
is never cut. The YouTube description pins them last, region then place, so the plan
builder's title hashtags lead with place then region. `_assert_region_present` raises
`ValueError` upstream if any of those surfaces lacks the region tag. A single-component
location has no region and is not gated.

**Hand captions.** A `titles.yaml` hand caption (`HandTitle.caption`) becomes the
YouTube caption verbatim; the other three platforms compose it as their opening
sentence, still made distinct from each other by their own hashtag count/shape.

**`plan_builder.build_publish_plan`** reads each platform's own text from
`post.captions[platform]`, falling back to the legacy shared `post.caption` when the
dict lacks the key (an old `social_posts.json` sidecar written before per-platform captions, or a caller
that never populated `captions`) — same `Filmed <date>.` suffix as before either way.

The resolved interactive family and the per-short OST skip design are in
`docs/pipeline/181-shorts-interactive-family-and-ost-skip.md`.
