## Photo sort groups, stacks, HDR and panoramas

`gomoveshift photos sort` tags shots that belong to one scene, then builds the combined pictures for them. A group is a tag: every member stays in its own category at its own rank. Nothing is removed or hidden, and a failed output never touches a member photo.

## Group tags

`ranked.csv` gains `group_id` and `group_best_rank`. A shot is claimed by the first kind that fits:

- `night-<n>`: an intervalometer run chunked the way `photos astro` does it (`astro_sequence.detect_sequences` then `chunk_sequence`), among the night-sky shots.
- `bracket-<n>`: an exposure bracket (`bracket.detect_brackets`).
- `pano-<n>`: three or more consecutive frames at most `PANO_GAP_S` (8 s) apart with the same focal length, f-number and exposure time. There was no panorama detector before this. A false positive costs one failed stitch, logged, because `stitch_panorama` gates its own seams and never writes a partial file.
- `burst-<n>`: look-alikes inside one category (`cull.group_near_duplicates` over the pHash of the developed JPEG, read during scoring).

`group_best_rank` is the rank of the group's best member (highest total). `index.html` notes "same scene as #NNN" under every other member.

## Outputs

After the gallery is complete (so the page can be opened while this runs), one group at a time:

1. HDR fusions to `hdr/<group>-<stem>.jpg` (`develop.fuse_bracket`, then the normal develop and JPEG export, graded from the middle frame).
2. Panoramas to `pano/<group>-<stem>.jpg` (`panorama.stitch_panorama`).
3. Night stacks to `stacks/<group>.tif` (the master) and `stacks/<group>.jpg` (the preview), highest mean total first. The call sequence mirrors `photos astro`: the RAM and disk gate, `prepare_frames` with the run's `--workers`, then `stack_run`. No calibration frames. Stacking does not use the GPU.

`index.html` is rewritten after each output lands: stacks at the top of night-sky, HDR and panoramas as their own sections. `--no-stacks` skips the night stacks only.

## Failures and resume

A group that fails (too few usable frames, a stitch that fails its seam gate, a RAM refusal) adds a `skipped.csv` row `group:<id>` with the reason and the run goes on.

Resume marker: an output is done when its file exists and the ledger holds `put_score(<sha256 of kind and sorted member paths>, "stack:v1", 1)`, written after the output lands. A crash between the two redoes that group. `run_sort(..., resume=<run folder>)` reads `ranked.csv`, rebuilds the groups from `group_id`, and does only the unfinished ones; it does not re-develop or re-score. The command-line flag is not wired yet (the parser is outside this change).
