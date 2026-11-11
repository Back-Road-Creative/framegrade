## Photo sort

`gomoveshift photos sort <folder>` takes a whole folder of photos (searched recursively), drops duplicates, sorts what is left by subject and develops each photo into a fixed output home. Outputs are private working JPEGs: no face blur, never the website. Originals are only opened read-only.

Flags: `--dry-run`, `--index-only`, `--reprocess`, `--workers N` (default 2, clamped to 1..4; the develop workers are spawned, never forked, because the parent has already started rawpy's OpenMP threads and a forked worker hangs on its first decode), `--no-stacks` (skip the night-sky stacks; note 178), `--resume RUN_DIR` (finish an earlier run's group outputs; note 178). There is no `--gpu`: the sort never takes the shared GPU lock, which belongs to the GoMoveShift drives. The command runs at the lowest CPU priority (`os.nice(19)`) so GoMoveShift renders on the same box always win the CPU. There is no `--out`: the home is `photos.sort_root` (default `/mnt/d/camPics_output`). The process takes the shared media admission fence like the other heavy photo commands (busy: "Waiting for shared media admission; no images decoded.", exit 2).

## Refusals

Exit 2 before anything is decoded or written: the folder is missing; the sort root's parent is missing (drive not mounted); the root and the folder nest, either way round.

## Stages

1. Inventory: supported photos (`ingest.discover`); a RAW with a same-directory JPEG sibling collapses onto the RAW. Other files (for example `.MOV`) go to `skipped.csv`; sidecars (`.xmp`, `.json`, ...) are ignored.
2. Camera facts: `capture_meta.read_capture_records`; warnings (no exiftool, a failed batch) go to `run.json`.
3. Dedupe against `<root>/ledger.sqlite` (`dedupe.dedupe`): renamed copies, cross-folder copies and shots seen in an earlier run. `--reprocess` ignores the ledger.
4. Classify: `subject.recognise_photos` with each photo's optics (focal length, aperture, focus distance, lens).
5. Develop: per category, corrections are measured in chunks of 60 (`CORRECTION_CHUNK`), then each photo is graded in a `ProcessPoolExecutor` and exported to `<run>/<folder>/NNN-<stem>.jpg`. The grade is the batch correction; an astro reading routes to `moody` and an exposure-bracket reading to `matte_shoulder` (the styles `SUBJECT_STYLE_ROUTING` supports, at its confidence floor). Other subjects use the base grade. The full variant machinery of `photos develop` is not used: one JPEG per photo.
6. Report, then `ledger.record(run_id, ...)` for every developed shot. A run that raises records nothing; a photo that failed to develop is not recorded, so the next run retries it.

Folders: `night-sky` (astro), `landscape`, `close-up`, `wildlife`, `portrait`, `long-exposure`, `hdr-source` (exposure-bracket members), `other` (unknown). NNN is the photo's place in capture order within its category. Score and category never drop a photo; a decode or develop error becomes a `skipped.csv` row and the run continues.

## Output

`<root>/runs/<YYYY-MM-DD>_<folder name>` (`_2`, `_3`, ... if it exists) holds:

- `run.json`: source, framegrade version, args, counts per category, developed, duplicates, skipped, warnings, timings.
- `ranked.csv`: category, rank, source, output, confidence, evidence, capture time; `quality_score` and `aesthetic_score` stay empty until those stages land.
- `duplicates.csv`: duplicate, kept, tier, key, reason, seen_run.
- `skipped.csv`: path, reason.
- `index.html`: static, no script, dark self-contained page with a summary strip (shots per category, number of sequences), one section per non-empty category, lazy thumbnails linking to the full JPEG, relative links, escaped names. A group of 6 or more frames (`SEQUENCE_CARD_MIN_FRAMES`) is one card at the top of its category, with its outputs (stacks first, any `.mp4` as a video player, poster only if the `.jpg` exists) and its frames folded in a `<details>`; night groups less than 15 minutes apart share one card. Each section shows its top 6 single shots and folds the rest.

## Modes
- `--dry-run`: inventory, camera facts, dedupe, classify. Prints counts per category, duplicates, skipped and an estimate (`DEVELOP_S_PER_PHOTO_ESTIMATE` = 20 s x photos / workers). Writes nothing, creates no run folder or ledger (an existing ledger is read).
- `--index-only`: inventory, camera facts and `ledger.record` under run id `index:<folder name>:<date>`; no decode. The run folder holds only `run.json`. Use it once over folders that were already sorted by hand.
- `--resume RUN_DIR`: finish the unfinished group outputs of an earlier run; nothing is re-developed or re-scored.
- `--resort RUN_DIR`: re-run classify (`hooks.recognise`) over an existing run folder. The lens-cap dark check keeps the first run's verdict (its `lens-cap-darks/` and `night-sky/` rows) and decodes only frames that newly read as night, or every night frame if the first run's check failed. It then move each developed JPEG already on disk to its new `<folder>/<NNN>-<stem>.jpg` (rank follows the `quality_score` already in `ranked.csv`), rewrite `ranked.csv`, `run.json` (`resort_of`, `args.resort_of`) and `index.html`, then re-run the group stage. Finished group outputs are reused through the ledger `STACK_MARKER` check; only groups whose membership changed are rebuilt. A group output that failed before with the same members (ledger `stack-failed:v1`) is not retried; `skipped.csv` says `<kind> failed in an earlier run; --resume retries it`. Runs made before that marker existed have none, so their failed groups are tried once more. It never develops and never re-scores. Refuses a directory outside the sort root, one without `ranked.csv`, or a JPEG named in `ranked.csv` that is missing; it cannot be combined with `--resume`, `--index-only` or `--dry-run`. `--reprocess` is not a re-sort: it re-develops shots already in the ledger.

## Verification

`tests/test_photo_sort.py` (fake hooks plus real runs on tiny JPEGs) and `tests/test_photo_sort_cli.py` (parser, config, admission fence, CLI dry-run).
