## Interactive family, resolved (`gomoveshift/pipeline/social_generator.py`,
`gomoveshift/pipeline/content_validator.py`)

Owner decision: an interactive short may omit the place from title, captions
and tags when the YouTube description's last line and the pinned first comment name
it. This closes the deferral above — the reverted design's approach ("carve out
an interactive exception rather than rewriting ~35 assertions") is what shipped, via
one new test file (`tests/test_social_interactive_family.py`) instead of touching the
four assertion files the deferral note names, none of which asserts anything that
contradicts the new hand-only, place-withheld path.

**Hand-only, by construction.** `SocialPost.interactive` is `True` only when
`generate_post` reads it off a `titles.yaml` `HandTitle.interactive` entry
(`interactive = hand.interactive if hand is not None else False`) — there is no
"every k-th composed short" path, so a composed (non-hand) title/caption is never
interactive and the mandatory-place-in-caption check is never in tension with the interactive-family handling.
`SocialPost.interactive` is now included in `to_dict()` (previously excluded to keep
serialization byte-identical) so downstream sidecars carry it.

**Place absent from title/captions/tags, by construction:**
- `youtube_title`/`tiktok_title` are the hand title verbatim (the existing hand-title bypass);
  a hand title naming the place raises `ValueError` at `generate_post` time, naming
  the index — the operator must write it place-free. A hand `caption` gets the same
  guard before it becomes the YouTube caption verbatim.
- `_build_platform_captions(interactive=True)` already withheld the place via
  `LOCATION_WITHHELD` for the composed branch.
- `_build_tags(interactive=True)` (new) skips the location-phrase tags, the county
  tag, and the road+city compound tag; a construction check on the finished tag
  string still raises if any place string leaked through.

**Place present in the description tail and pinned comment, by construction:**
- `_build_youtube_description(interactive=True)` withholds the place from the base
  sentence (no POI/road branch — a generic `"POV driving through this route
  Pt.N"`), drops the `#{location_tag}` hashtag, and appends a final line
  `"📍 {social_location_full}"`.
- `_build_first_comment(interactive=True)` prepends `"📍 {social_location_full}"` as
  its first line, then the existing hashtag line unchanged (the pinned comment
  already always carried `#`-form location hashtags; the place name is new).

**Burned-label guard (`InteractiveShortHasBurnedLabel`, a `ValueError` subclass).**
`generate_all` raises when an index flagged `interactive` has a non-empty
`per_short_locations` entry — that parameter is each short's burned on-screen OST
label, so a non-empty entry means the answer is already on screen and "where is
this?" would be a lie. There is no per-short OST skip today (see **Per-short OST skip** below), so
this fires for every interactive index unless the caller passes an empty/falsy label
for it — **today interactive shorts are the hand-made / label-free lane only.**

**`ContentValidator` carve-out.** `validate`/`validate_all`/`enforce` gain a
keyword-only `interactive: bool = False`. `True` skips exactly the 4
`SELF_CONTRADICTION_PATTERNS` ("guess where/below", "where is this / in the world",
"name this road…", "recognise/recognize this place…") — every blocked phrase and
every other pattern still applies. `generate_post` passes
`interactive=post.interactive` to its `validator.enforce(...)` call. Every other
caller (`content_gates.py`, `ost_validator.py`, the review-gate path) keeps the
default `False`, unchanged.

## Per-short OST skip (design; not built, blocked on a render-time global index)

**Goal.** An index flagged `interactive` in `titles.yaml` renders with *no* burned
place label, so the burned-label guard above becomes unreachable by construction
instead of refusing at phase D. `single_pass.py`'s `skip_ost` is per-*render*
(`process(..., skip_ost=)` → `_run_encode_step` → `_build_video_filter` → the
`ost_applied` flag on the result), not per-short: one drive render pass cannot burn 8
labels and skip the 9th.

**Why it is not a small change today — the index gap.** `titles.yaml` is keyed by
the *global* short index (`social_generator.generate_all` docstring, "keyed by the
GLOBAL (post-index_offset) index"), and that index is assigned only in phase D, after
render: `phase_d_scheduling.py` walks `drive_plan.regular_lfs` with
`global_index_offset += len(posts)` (posts = the shorts that actually exist on disk
per LF). Render time (`day_pipeline.py`, the `short_ost_text` derivation around the
`_short_idx_to_lf_idx` map built from `assign_shorts_to_lfs_by_source_window`) knows
only the per-LF *planned* position. A planned-count offset computed before render
drifts the moment one short fails or is culled — the skip would land on the wrong
clip, i.e. burn the answer onto the interactive short and blank a normal one. There
is no render-time global index anywhere in `day_pipeline.py` / `phase_bc_day_first.py`
(grep for `global_index`, `titles.yaml`, `interactive` there is empty).

**Prerequisite (global-index plumbing) — mechanism BUILT and wired.** Give each
planned short a stable global index *before* render and carry it through
state:
1. `social_generator.assign_global_short_indexes(shorts_per_lf, existing=None)`
   assigns a `global_index` per short, keyed by `str(short_path)`, in plan
   order (ascending LF index, then list order). Given a previously-assigned
   `{path: index}` map, an already-covered path keeps its index forever; a
   path that drops out (culled/failed) is never handed to a different short
   — a hole, never a shift. `PipelineState.short_global_index` is the
   persisted map (`pipeline_state.json`); `None` on a checkpoint from before the global index.
2. `_build_social_posts_per_lf` (`phase_d_scheduling.py`) consumes an
   optional `short_global_index` map: when supplied and every short in an LF
   is covered, it calls `generate_all(global_indexes=[...])` — an explicit
   per-short index list that WINS over the legacy `index_offset` recompute.
   Any LF with an uncovered short (legacy state, or a genuinely missing
   entry) falls back to the old `len(posts)`-derived `index_offset` for that
   LF only, with a logged warning naming the fallback, so an in-flight
   resume on an old checkpoint is never broken. The shortfall gate
   (the shorts-shortfall hard-fail) already refuses a thinner
   week, so holes are a hard fail today, not a silent renumber.
3. `social_generator.derive_interactive_indexes(hand_titles)` derives
   `interactive_indexes: set[int]` from an already-loaded
   `load_hand_titles()` result (never re-parses `titles.yaml`).
   `PipelineState.interactive_indexes` persists it (a sorted list — JSON has
   no set type).

**Wired at both live call sites (index wiring).** `PhaseBCDayFirst._run_day_first`
(`phase_bc_day_first.py`, immediately after `state.shorts_per_lf = shorts_per_lf`
is set) now assigns `state.short_global_index =
assign_global_short_indexes(shorts_per_lf, existing=state.short_global_index)`,
so a resumed run keeps its prior indexes and a fresh run starts at 1.
`PhaseD.run` (`phase_d.py`, the `_build_social_posts_per_lf` call site) passes
`short_global_index=state.short_global_index` through, and derives
`state.interactive_indexes` once from `titles.yaml` (via `load_hand_titles` +
`derive_interactive_indexes`) when the file exists and the field hasn't been
derived yet — `None` stays `None` when there's no `titles.yaml`.
`_build_social_posts_per_lf` itself is unchanged: a legacy checkpoint with
`short_global_index=None` still reproduces today's `len(posts)` recompute
exactly, per LF, with a logged warning.

**The skip itself, once 1–3 exist (small).**
- `day_pipeline.py` OST derivation: for a short whose global index is in
  `interactive_indexes`, set `short_ost_text = ""` and record provenance
  `interactive-no-ost` in the `shorts_ost_audit_<date>.json` sidecar (the audit gate
  must show the operator an intentionally blank label, not a missing one — today an
  empty burn location is a hard fail, so the skip must be explicit in the sidecar and
  the render call must pass `skip_ost=True` for that clip, not an empty `location`).
- `shorts_extractor.py` label gate (`_gate_short_ost_text` cascade, single-video path
  via `stages/shorts.py`): same rule, same sidecar provenance.
- `phase_d_scheduling.py` passes `lab or lf_location` — for an interactive index
  the parent-LF fallback must *not* fill in, so the sidecar's blank label reaches
  `generate_all` as `""` and the desync check treats `interactive-no-ost` as an
  intentional blank (it currently raises for any GPS-bearing short with no label).
- The `InteractiveShortHasBurnedLabel` guard stays as the late tripwire; with the per-short skip in
  place it can only fire on a hand-edited sidecar.
- Test: a 3-short LF with index 2 interactive renders labels for 1 and 3, `skip_ost`
  for 2, sidecar provenance `interactive-no-ost`, and phase D produces an interactive
  post for global index 2 with no burned-label exception.

Until then interactive shorts remain the hand-made / label-free lane (`per_short_locations`
entry empty for that index).
