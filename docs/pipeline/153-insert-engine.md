## Insert engine

`gomoveshift/pipeline/inserts.py` key-cuts stills and clips into a rendered long-form without re-encoding the rest of it.

## Data model

- `Insert(at_s, media, hold_s=4.0, style="full", capture=None)`: `capture` is the `(lat, lon)` the media was taken at (the county gate reads it); `style` is `full` or `letterbox`; a clip ignores `hold_s` and uses its own duration. A still is recognised by its suffix, anything else is a clip.
- `plan_inserts(lf, inserts, *, profile, rendition, gate=None, ...) -> list[InsertPlan]`: with a `gate` runs the gates below first; probes the LF, renders still cards, fits clips, computes the cuts. Raises before any encode: `InsertError`/`BlurGateError` from the gates, `ValueError` for a bad style, `UpscaleRefusedError` for a small clip under `media.upscale: off`.
- `apply_inserts(lf, plans, output, *, run=subprocess.run) -> InsertResult`: `result.spans` lists each `Span(start, end, stream_copied)`; `result.upscaled` lists the source sizes of upscaled clips. All ffmpeg calls are argv lists.

## Keycut rule

`cut_in` is the keyframe at or before `at_s`; `cut_out` is the first keyframe at or after `at_s + duration` (the file end if none). Only `[cut_in, cut_out)` is re-encoded, with the join-repair args (`montage_builder.lf_reencode_args`); `[0, cut_in)` and `[cut_out, end)` are `-c copy`, then joined with the concat demuxer. Keyframes are the probed ones (`keycut.probe_keyframes`), not arithmetic on `GOP_DURATION_SEC` (2.1 s, `montage_builder.py`): that constant is a merge threshold and scene cuts add real keyframes, so a fixed grid can name a time that is not a keyframe. No keyframe index means `InsertError`. Inserts whose bounding GOPs touch are refused.

## Audio rule

Under a still the LF's own track (already attenuated by the loudness pass) is kept; no new gain is applied. Under a clip the LF audio is muted for the span and the clip's own audio plays.

## Upscale record

A clip is fitted to the LF frame through `quality.upscale_plan(w, h, rendition, policy)`, with the rendition's size set to the LF's. When the plan is not `None` the clip is an upscale: `InsertPlan.upscaled_from = [w, h]`, `scaler_filter` begins with the `zscale` chain, and the result repeats the source size.

## GPS anchoring

`anchor_photo_insert(photo, *, captured_at_utc, track_started_at_utc, segments, lf_start_s, lf_duration_s, gps=None)` returns the `Insert` for a photo. The zoned capture instant is moved onto the track's clock (`anchors.captured_at_route_seconds`) and placed by `anchors.anchor_media_to_segments` against the day's track; `at_s = track_time_s - lf_start_s` (the LF start on that same clock). The photo's own EXIF fix is used unless it is physically impossible for the clock, in which case the clock position wins (the anchor's own rule). A photo the track cannot place (outside the recording, in a gap, GPS contradicting the clock, no capture time) or that falls outside this LF raises `InsertError`; it is never guessed. The anchored position becomes `Insert.capture`.

## Gates (`InsertGate`, `gate_inserts`)

Run inside `plan_inserts(..., gate=...)`, before any card is drawn or ffmpeg call, in this order for each insert:

1. **County.** The capture position (`Insert.capture`, else a still's EXIF fix) is reverse-geocoded to a county and must be one of the LF label's counties. The label is `RunSpec.per_lf_locations[lf_index]` (`"A County, SC"`, `"A & B County, SC"`, `"A County, SC & B County, GA"`), parsed by `label_counties`. The same county name in another state is another county. No capture position, an unresolved county, or a label that names no county all refuse (fail closed).
2. **Blur and GPS strip, stills.** `route_story.privacy.apply_privacy_gate` writes a blurred, EXIF/GPS-stripped working copy; the card is drawn from the copy and the original is never read again. A withheld photo refuses.
3. **Blur, clips.** When the run requires blur (`blur_gate.blur_required`), a clip must carry the `.blurred` marker (`blur_gate.mark_applied`, tested by `master_blur_proven`) or `BlurGateError` is raised; `--skip-blur` is the named escape. A clip's container tags (including location) never reach the splice: its encode pins `-map_metadata 0`, metadata from the LF only.

`InsertGate.from_config(config, label)` builds the gate for one LF. `plan_inserts` without a gate is unchanged; the drive wiring must always pass one.

## Drive run

`gomoveshift drive --insert-photos <dir> --insert-clip <path>@<mm:ss>` (clip flag repeatable). Both are per-invocation (`NOT_PERSISTED` in `flag_registry`), mirrored onto `config.drive.insert_photos` / `insert_clips` by `apply_insert_flags`.

- **Preflight.** `describe_drive_inserts` prints one line per insert in the plan; a missing directory, an empty one, a clip path that is not a file, or a clip that is not `PATH@MM:SS` exits 2 before any work.
- **Placement and gate, before the first encode.** `plan_drive_inserts` (called from `run_all_days`) places every insert (`plan_day_from_batch` -> `place_day_inserts`) and then runs the whole `InsertGate` on it (county, blur, GPS-strip) against the LF's county label (`per_lf_locations`, via `insert_label`; never the OST text), so a refusal costs no OST burn and no encode. A clip has no capture position of its own: it takes the day track's fix nearest its cut point (within 60 s, else none, and the county gate refuses it). A photo is matched by its camera-clock EXIF time to an episode's `creation_time`, anchored on the day's track (`anchor_photo_insert`), then moved onto the long-form whose source ranges hold that master second plus the montage lead. A photo on no day, or in no long-form, refuses (`InsertError`). Clips land in the first long-form of the drive's first day at their `@mm:ss`. Placement is approximate to about one GOP.
- **All days, rendered days gated.** `plan_drive_inserts` takes every day of the drive and a `render_dates` set. Placement runs on all of them (so a photo of a finished day still resolves and `clip_date` is the drive's first day); the gate runs only on the days being rendered, since a finished day already carries its inserts.
- **OST audit sidecar.** Each LF entry carries `inserts` (kind, media name, `at_s`, style, hold), and the audit hash covers LF entries, so the operator confirms the inserts with the burn text.
- **Splice gate.** `splice_lf_inserts` builds the `InsertGate` itself and always passes it to `plan_inserts` (fail closed; no config gives the closed gate).
- **Splice, after the burn.** `run_day_pipeline` folds `insert_records` for the LF into its burn fingerprint (`lf_NN.ost.sha256`), so changed inserts re-burn and unchanged ones leave a fresh final alone. Order per LF: `apply_ost`, then `splice_day_lf_inserts` on the burned result (in place), then the fingerprint is written; a splice that raises leaves no fingerprint. Stills were placed against the planned montage length, so `splice_day_lf_inserts` shifts them by the probed lead (`prepended` minus `concatenated` duration) minus the planned one (`shift_photo_inserts`, clamped at 0); clips keep their `@mm:ss`. The per-day OST gate hashes the same inserts as the front-load, so a confirmation made without them is stale.
- **Resume.** The flags are not restored from state. `PipelineState.insert_request` (`insert_request(drive)`: `{"photos", "clips"}` or none) records what the finished days were rendered with; `reconcile_insert_request` runs in phase BC before `run_all_days` and refuses (`InsertError`, naming the original flags) a resume whose request differs while any day is finished, since a finished day is never re-rendered and the drive would end half with, half without inserts. Re-run with the original flags, or `--fresh`. With nothing finished the request may change. The planner gets every day (`all_day_batches`, `all_gps_points_by_date`) and gates only the rendered ones. A finished day's recorded inserts are carried through the disk reconstruction so Phase D keeps them.
- **Phase D.** `Deliverables.inserts` (`{lf_index: [record]}`, persisted in `per_day_completed[date]["deliverables"]`) gives each LF a `📸` chapter marker per insert (`add_insert_markers`; none within 10 s of an existing chapter) and a `📸 N photo/clip insert(s)` description line.
