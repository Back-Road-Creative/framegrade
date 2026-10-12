### `--preset full|social` (`presets.py`)

`FilmPreset` (`presets.py`) is a frozen dataclass of editorial defaults for delivery size and
pacing; `PRESETS = {"full": FULL, "social": SOCIAL}` are the only two shapes. `full` (the CLI
default) reproduces every pre-preset literal exactly -- `DELIVERY_WIDTH`/`DELIVERY_HEIGHT`
(1920x1080), `DEFAULT_INTRO_SECONDS`/`DEFAULT_RECAP_SECONDS` (12s/6s), the historical
`--photo-seconds` default of 2.5s, and `--map-seconds` still unset (the old per-leg reading-time
clock). `social` is a 9:16 short-form cut: 1080x1920, a 6s intro, a 4s recap, ~2s photo holds, 27s
of combined map legs, at most 8 kept stills, and a hard 90s ceiling.

A preset supplies DEFAULTS ONLY -- `apply_preset` fills `--width`/`--height`/`--photo-seconds`/
`--map-seconds` only when the operator left them unset (`None`; none of the four flags has a way
to *ask* for `None` from the CLI, so post-parse `None` unambiguously means "not passed"), and the
preset is resolved *after* the existing `--target-seconds`/`--map-seconds` mutual-exclusivity
checks so a preset fill can never look like an explicit `--map-seconds`. `capped_still_count` caps
the existing `still_selection.max_stills_for_cap` limit at the preset's own `max_photos` when it is
lower. `check_ceiling` refuses a plan whose intro + recap + map + per-still seconds (plus a placed
source video's own duration) would exceed the preset's `max_seconds` -- raised as `FilmTooLongError`
before any render starts, never after an encode.

The receipt's `preset` block (`{"name", "size": [width, height], "rendition"}`, plus
`rendition_check` for `social`) names the delivery rendition this preset targets --
`route_story_short` for `social`, which keeps audio, unlike the drive-shorts
`youtube_short` row it deliberately does not reuse. There is still no second encode:
`encode_final` delivers both presets, and `rendition_check` (`{"audio_stream", "size_ok",
"within_ceiling"}`) is a record-only, after-the-fact confirmation read from the same
`check_delivery` measurement the receipt already makes -- see `26-route-story-audio-and-final-encode.md`.

The vertical HUD layout is live: `layout.hud_layout(width, height)` is the single place that
resolves every HUD anchor from a frame's own size, and every draw function in `hud.py` reads its
position from the `HudLayout` it returns rather than computing one inline. Landscape
(`width >= height`) reproduces the historical literal positions exactly. Portrait moves the two
badges (`draw_high_point`, `_draw_media_arrival`) and the watermark to a bottom-centre stack, clear
of `layout.SHORT_BOTTOM_SAFE_PX` (an operator-adjustable safe zone reserved for a social platform's own
caption/share-rail chrome), and scales the dashboard panel and title from `width` -- the short side
in portrait -- instead of `height`, which is the short side in landscape.

Every still is blurred and EXIF-stripped into a working copy before any compositing or export
(`--skip-blur` opts out); see `125-route-story-privacy-gate.md`.

**Publishing the cut.** A rendered film has no drive run behind it, so it does not go through
`enqueue-plan`; it goes through `gomoveshift enqueue-asset <film> --receipt <film.receipt.json>
--platform ... --at ...` instead (`publishing/asset_plan.py`, `docs/social-publishing-queue.md`
"Standalone assets"). `--receipt` passes the film's own `privacy` block straight through as the
delivery's attestation; without it, every required privacy class is recorded inconclusive and the
drain holds the delivery rather than publishing it unattested.

<a id="route-story-film-from-the-console"></a>
**From the console (Waypost route-story job).** `gomoveshift/pipeline/route_story/profile.py`
is the profile -> argv mapping, so a tenant's console can launch a film
in-process as a regular job -- a second door onto `film.main()`. `console/route_story.py`'s `profile_dirs()`
reads `<console data dir>/route-story-profiles/` (and a `route_story_profiles/` directory beside the package if one is present; none ships, so the operator supplies profiles);
`resolve_profile` accepts only a bare `*.yaml` basename present in one. `POST /route-stories/film`
resolves the profile, builds argv, and `jobs.launch`s it as the allowlisted `"route-story-film"`
subcommand -- `jobs.launch` is the supervisor here, same as any other console job.
`console/route_story_progress.py`'s `parse_progress` reads film.py's own log lines (map-leg,
frame-QA fault/evidence, the final bare `print(args.output)`) for the job detail page's "Route
story" block, ending in a "Publish this cut" `enqueue-asset` command (receipt read back from the
profile) linking to `/publishing/enqueue-asset`. One brand = one tenant throughout, via
`brand_launch_env()`.
