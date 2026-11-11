## Photo sort groups, stacks, HDR and panoramas

`gomoveshift photos sort` tags shots that belong to one scene, then builds the combined pictures for them. A group is a tag: every member stays in its own category at its own rank. Nothing is removed or hidden, and a failed output never touches a member photo.

## Group tags

`ranked.csv` gains `group_id` and `group_best_rank`. A shot is claimed by the first kind that fits:

- `night-<n>`: an intervalometer run chunked the way `photos astro` does it (`astro_sequence.detect_sequences` then `chunk_sequence`), among the night-sky shots.
- `bracket-<n>`: an exposure bracket (`bracket.detect_brackets`).
- `sequence-<n>`: an intervalometer run outside the night sky, such as a dawn sequence (`detect_interval_runs`). It needs at least `TIMELAPSE_MIN_FRAMES` (24) frames of one category, each at most `SEQUENCE_GAP_S` (60 s) after the one before, with at least 80% of the gaps within 10% (at least 1 s) of the median gap. A timer keeps that cadence and a hand-held session does not. Without this, look-alike bursts split JP's 100-frame dawn run into seven groups. A sequence gets a time-lapse like a long burst (step 5).
- `pano-<n>`: three or more consecutive frames at most `PANO_GAP_S` (8 s) apart with the same focal length, f-number and exposure time. There was no panorama detector before this. A false positive costs one failed stitch, logged, because `stitch_panorama` gates its own seams and never writes a partial file.
- `burst-<n>`: look-alikes inside one category (`cull.group_near_duplicates` over the pHash of the developed JPEG, read during scoring).

Lens-cap darks never join a group. Before developing, the night-sky shots go through the stack's light/dark triage once (`classify_night`, `exposure_class == "cap_on"`). The darks move to their own folder, `lens-cap-darks/`. They are still `ranked.csv` rows, but `index.html` shows only their count ("N lens-cap darks kept for calibration, not ranked") and the top-shots list skips them. If the triage fails, a warning is logged and every frame stays in night-sky. `--no-stacks` skips the triage, since it decodes no night frame. The group stage reuses this verdict, so a fresh run decodes each night frame only once.

`group_best_rank` is the rank of the group's best member (highest total). `index.html` notes "same scene as #NNN" under every other member.

## Outputs

After the gallery is complete (so the page can be opened while this runs), one group at a time:

1. HDR fusions to `hdr/<group>-<stem>.jpg` (`develop.fuse_bracket`, then the normal develop and JPEG export, graded from the middle frame).
2. Panoramas to `pano/<group>-<stem>.jpg` (`panorama.stitch_panorama`), stitched from the developed JPEGs this run wrote, never the RAW sources: the stitcher reads with `cv2.imread`, which cannot open a RAW.
3. Night stacks to `stacks/<group>.tif` (the master) and `stacks/<group>.jpg` (the preview), highest mean total first. The call sequence mirrors `photos astro`: the RAM and disk gate, `prepare_frames` with the run's `--workers`, then `stack_run`. No calibration frames. Stacking does not use the GPU.
   Each chunk stack also keeps `stacks/<group>.linear` (`stack_run(linear_artifact_path=)`). After all chunks, each night session with two or more built chunks gets one session result: the `photos astro` deep combine (`astro_combine.deep_combine` plus `publish_deep_product`, through `GroupOps.combine_session`) over the chunk `.linear` artifacts, written to `stacks/<first-id>-session.tif` and `.jpg`. Only that result is a page output (kind `stacks`, `group_id` the session's first chunk); the chunk stacks stay in `stacks/` as intermediates, and their `.linear` artifacts (~0.7 GB each at 24 MP) are removed once the session result is built; a failed combine keeps them so a resume can retry. It uses the resume marker under the key of kind `session`. If the combine raises (including a chunk stack from an older run with no `.linear`), the page shows the session's best chunk stack (highest mean total) and `skipped.csv` gets `session combine failed: <reason>`. A single-chunk session is unchanged.

4. Star trails to `trails/<id>.tif` (the master) and `trails/<id>.jpg` (the browser JPEG, `night_products.render_master_preview`), one per night session (sessions as in step 5). Rendered by `astro_trails.render_trail`: calibrated, unaligned, per-pixel maximum, so the sky's rotation draws arcs. The session's light/dark triage runs once and is shared with its time-lapse; darks calibrate only when `build_validated_master_dark` accepts them, since a maximum keeps an unsubtracted hot pixel forever. A session with fewer than `TRAILS_MIN_FRAMES` (24) usable lights, about 13 minutes at the measured 32 s interval, gets no trail and is remembered (`trails-skip:v1`). The output kind is `trails`; on the session's card it sits after the stacks and before the time-lapse.
5. Time-lapses to `timelapse/<id>.mp4` with a same-stem poster `timelapse/<id>.jpg` (the video's middle frame, at most 1080 px tall). One per night session, not per chunk: chunks join into a session while each starts within `astro_sequence.MAX_FRAME_GAP_S` of the previous chunk's last frame, so a session is one `detect_sequences` run, found again from `ranked.csv` alone. The session is named for its first chunk (`night-1`). Lens-cap darks are told from lights with the stack's own triage verdict (`exposure_class == "cap_on"`, #289) and are never rendered as lights; the `lens-cap-darks/` rows shot within `DARKS_WINDOW_S` (15 min) of the session's lights join the session's own darks; they go to `render_timelapse` as dark calibration when `build_validated_master_dark` accepts them against the lights, else they are dropped. Rendered by `astro_timelapse.render_timelapse` at the night product's fps and 2160 px height ceiling. A sequence, or a burst of `TIMELAPSE_MIN_FRAMES` (24) or more, renders its developed JPEGs, in capture order, through `timelapse.render_stills_timelapse`. A session with fewer than 24 usable lights gets no time-lapse and is remembered, so `--resume` does not decode it again. The output kind is `timelapse` and the page matches it to its card by `group_id`.

`index.html` is rewritten after each output lands: stacks at the top of night-sky, HDR and panoramas as their own sections. `--no-stacks` skips the night stacks, star trails and night time-lapses; burst time-lapses still render.

## Failures and resume

A group that fails (too few usable frames, a stitch that fails its seam gate, a RAM refusal) adds a `skipped.csv` row `group:<id>` with the reason and the run goes on.

Resume marker: an output is done when its file exists and the ledger holds `put_score(<sha256 of kind and sorted member paths>, "stack:v1", 1)`, written after the output lands (a time-lapse or star trail uses the same marker under the key of kind `timelapse` or `trails` and its session's sorted member paths). A crash between the two redoes that group. `photos sort <folder> --resume <run folder>` (`run_sort(..., resume=)`) reads `ranked.csv`, rebuilds the groups from `group_id`, and does only the unfinished ones; it does not re-develop or re-score.
