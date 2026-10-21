# Publish-plan builder (`gomoveshift/publishing/plan_builder.py`)

`build_publish_plan()` turns the same Phase D
inputs `build_consolidated_shorts_csv()` consumes into a `PublishPlan`,
alongside the CSV. Wired into Phase D via `_emit_publish_plan_sidecar`
(`phase_d.py`), which writes `<project>_publish_plan.json` iff a
`DriveConfig.publishing_route_*` is `first_party` (every route defaults to `off`,
so the sidecar is written only when a deployment's config opts a platform in)
**and** the run is a publish pass (`publish_pass=bool(state.youtube_upload)`,
a required argument). An audit pass only warns about an unverified place or a
missing hand title, so it never writes the plan and leaves the publish pass's
plan on disk untouched. Otherwise an audit re-run over an uploaded drive, which
still holds every LF's video id, would hand `enqueue-plan --apply` shorts that
no gate ever refused.

## Publish preview (skip-upload / audit runs)

The two deferral branches above (audit pass; publish pass with no LF video id)
also call `_emit_publish_preview_sidecar`, which writes
`<project>_publish_preview.json` so the final Short title, description and
hashtags can be reviewed before any upload. It is **not a plan**: the name
matches no `*_publish_plan.json` glob or `<project>_publish_plan.json`
convention, the file carries `"kind": "publish_preview"` and
`"not_for_publish": true`, and `load_publish_plan` raises
`PublishPlanFormatError` on it. `_emit_publish_plan_sidecar` still returns
`None` on those branches and the writer returns nothing.

Both consumers refuse a preview on disk, even one copied over a plan's name
(GMS-17a): `enqueue-plan` reads every sidecar before its dry-run exit as well
as on `--apply`, so a preview is an `Error:` with exit 1 and nothing queued,
never a "ready to spool" listing. `dispatch-queue`'s `SidecarRecordLoader`
raises `PublishPlanFormatError` for a preview, whether it found the file by
name or through a queue row's `sidecar_path`. Tests:
`tests/test_publish_preview_consumers.py`.

The same loader re-checks every real plan it reads (GMS-17c): each short's
title hashtags must appear in its description, and the two `plan_gates`
checks `build_publish_plan` runs at build time (in-region counties,
no preposition splice) run again on load. A plan edited or written apart
from the builder raises `PublishPlanFormatError` naming the file, in
`enqueue-plan` (dry run and `--apply`) and in `dispatch-queue`. Detail:
`docs/pipeline/157-dod-validator-and-plan-consistency.md`.

It runs the same `build_publish_plan` gates over the same inputs. Differences
are listed per short under `placeholders`: `parent_lf_video_id`
(`PREVIEW-NO-LF-ID-<lf_index>`) and the YouTube description's parent link,
`parent_lf_publish_at` (the drive schedule's LF slot, else the epoch), and
`media` (not hashed, `PREVIEW-NOT-HASHED`). A build error (title, region or
missing-slot gate) logs a `PREVIEW NOT WRITTEN` warning and writes nothing; it
never fails the run.

## Inputs beyond the CSV path

- `media_by_short_index: Dict[int, MediaFingerprint]` — real local media
  identity, not the Drive shareable link the CSV needs for Metricool. The
  Drive upload stays the archive path; `drive_links_by_short_index` is
  **deliberately not accepted** — no first-party payload destination.
- `route_selected` (defaulted) feeds `RouteSelection`. Metricool is retired —
  there is no rollback publisher and no `route_rollback` parameter; a
  platform not in `route_selected` simply ships nothing (`off`).
- `consent_hash` / `reviewed_plan_hash` — threaded through as given, not
  computed here (no dispatcher/review-gate wiring yet).

## Payload composition (per the mapping doc)

- **Title decision (U5)**: every generated `SocialPost` carries Waypost's
  `title_decision` (format `short`), and `youtube_title` is its
  `chosen_title`. The plan builder refuses (`TitleDecisionMismatchError`) a
  post whose `youtube_title` differs from that choice, so the hashtag append
  below is the only change between Waypost's pick and the uploaded title.
  A Short whose stamped title names its place, backed by one of the clip's
  own GPS region votes, goes to Waypost as a single-place fact (U5c,
  waypost#4: template ids `place` and `highlight_place`), each with an
  evidence id per short; a place-free, hand or GPS-less Short records
  `facts_unavailable` and keeps the composed title. A pre-U5 sidecar (no
  decision) is not checked.
- **YouTube**: `category`/`tags` from `SocialPost`. `description`
  = `SocialPost.youtube_description` + the parent-LF link
  (`https://youtu.be/<parent_video_id>`). `title` = `SocialPost.youtube_title`
  + the description's own hashtags, most specific first (place, region
  (state/county), terrain or `#POVDriving`), at most three in all, stopping at the first that
  would pass the 100-char title limit. The Shorts player shows the title and
  folds the description away, so hashtags left only in the description are never seen by a Shorts viewer. Taking
  them from the description keeps one hashtag choice per short (an
  interactive short's description withholds the place tag, so its title
  does too). The anti-spam gate judges the title without these hashtags, so
  a legacy pool title cannot pass by wearing them. The generator pins the place
  and region tags as the LAST two description hashtags (region, then place), so
  the title always ends `... #Unicoi #Tennessee #POVDriving`.
- **Instagram**/**Facebook**: caption = `SocialPost.caption` with the
  `"Filmed <Month YYYY>."` sentence placed after the caption's sentence and
  **before** its closing hashtags, joined with a space (the CSV path joins it
  the same way, minus the full stop below): `"Worth the detour? Filmed March 2030.\n\n#POVDriving
  #PA"`, never the date glued onto the hashtag line. A sentence that ends on a word, a number, or a closing
  bracket or quote gets its full stop first (`"Just driving through Unicoi.
  Filmed March 2030."`, never the run-on `"...Unicoi Filmed March 2030."`). A number sign
  is not a hashtag (`"Stop #2"` stays in the sentence). **Without** the CSV's
  280-char Twitter truncation (retired
  X/Twitter column only). No parent link here.
- **TikTok**: `caption` = the SAME text Instagram and Facebook get
  (`SocialPost.caption` + the filmed-date sentence, same placement); `title` =
  `SocialPost.tiktok_title`, a short label for the console list and
  `ost_validator`. TikTok has one text field, so `caption` is what publishes
  — `post_info.title` on Direct Post, and the text a human pastes on the
  inbox path — with `title` only as the fallback for sidecars written before
  captions existed, so such a sidecar falls back to the short label instead of publishing with no text. `privacy_level` from the
  builder's `tiktok_privacy_level` argument, which Phase D fills from
  `DriveConfig.publishing_tiktok_privacy_level`. Both default to `SELF_ONLY`
  — the PRIVATE audience — so a caller that omits the argument, or an
  operator who never sets the field, cannot publish to everyone by accident.
  A value outside `models.TIKTOK_PRIVACY_LEVELS` raises before any record is
  built; TikTok would otherwise answer at `/video/init/` time, long after the
  plan is written, reviewed and approved. The live authority on what an
  account may actually request stays `creator_info["privacy_level_options"]`,
  enforced in `assert_public_routing_allowed` at submit time.

## Validation

Reuses `_assert_shorts_metadata_in_region` /
`_assert_shorts_not_scheduled_before_publish` from `csv_generator.py`. That
module's partial-parent-id-map check is inline (no importable helper); here
every record needs a real parent link regardless, so one check
(`PUBLISH-PLAN-NO-PARENT-ID`) covers both cases. Missing slot / empty-sha256
media are new, stricter checks — unbuildable, not warn-and-continue.

## Building a `MediaFingerprint` (`gomoveshift/publishing/media_fingerprint.py`)

`fingerprint_media(path) -> MediaFingerprint` is the production builder for
the `media_by_short_index` values above: `size_bytes` from `stat()`,
`sha256` streamed in 1 MiB chunks (never `read_bytes()` — LFs run GB-scale),
and `duration_seconds`/`width`/`height`/`codec` from ffprobe. `path` is
stored as `str(path)` exactly as passed in — callers that need it stable
across working directories should pass an absolute `Path`.

Reuses the existing `FFprobeClient`
(`gomoveshift/pipeline/validation/ffprobe_client.py`) rather than a new
inline `subprocess.run(["ffprobe", ...])` through `probe_video_summary`, the
engine-side probe `fingerprint_media` shares with the night-product validator
(the engine never imports `publishing`; `tests/test_media_boundary.py`) — it already has retry-on-timeout
and a configurable timeout matching `DriveConfig.ffprobe_timeout_sec`
(default 30s, overridable via the `ffprobe_timeout_sec` kwarg).

Every failure names the path and raises rather than returning a
zeroed/placeholder fingerprint: missing file (`FileNotFoundError`),
non-zero ffprobe exit, no video stream, or an unparseable duration
(`RuntimeError` for the latter three) — a silently-wrong fingerprint would
flow into the plan hash and misidentify media a post consented to.

Wired into `build_publish_plan()` via Phase D's `_emit_publish_plan_sidecar`.

## Metricool-CSV parity

`tests/test_publishing_plan_csv_parity.py` drives `build_consolidated_shorts_csv`
and `build_publish_plan` from ONE shared fixture and asserts per-short parity
on caption/title text, YouTube tags, TikTok title, scheduled slot (converted
EST-local -> UTC with `plan_builder._slot_datetime_utc`, the single source of
truth), media reference, and shared platform routes. All checks pass against
the current code — no plan_builder/models change was needed. Legitimate,
by-design divergences the test documents rather than treats as bugs:

- **`YouTube Video Category`**: the CSV forces `""` on every row (Metricool
  drafts silently on a populated numeric ID); the plan carries the real
  `SocialPost.youtube_category` for the direct YouTube API call.
- **Media reference**: the CSV's `Picture Url 1` is the Drive-hosted download
  link Metricool fetches; the plan's `MediaFingerprint.path` is the local
  rendered file the first-party upload reads. Different representations of
  the same short, verified for the same `short_index` rather than compared
  by value.
- **Platform coverage**: Pinterest/Threads/Twitter/LinkedIn/GBP are CSV-only
  columns with no `models.KNOWN_PLATFORMS` counterpart — there is no
  first-party adapter for them (`gomoveshift/publishing/` has only
  facebook/instagram/tiktok/youtube adapters), so the queue cannot yet
  replace the CSV for those platforms.
