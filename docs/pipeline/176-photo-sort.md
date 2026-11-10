## Photo sort

`gomoveshift photos sort <folder>` takes a whole folder of photos (searched recursively), drops duplicates, sorts what is left by subject and develops each photo into a fixed output home. Outputs are private working JPEGs: no face blur, never the website. Originals are only opened read-only.

Flags: `--dry-run`, `--index-only`, `--reprocess`, `--workers N` (default 2, clamped to 1..4), `--no-stacks` (skip the night-sky stacks; note 178), `--resume RUN_DIR` (finish an earlier run's group outputs; note 178). There is no `--gpu`: the sort never takes the shared GPU lock, which belongs to the GoMoveShift drives. The command runs at the lowest CPU priority (`os.nice(19)`) so GoMoveShift renders on the same box always win the CPU. There is no `--out`: the home is `photos.sort_root` (default `/mnt/d/camPics_output`). The process takes the shared media admission fence like the other heavy photo commands (busy: "Waiting for shared media admission; no images decoded.", exit 2).

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
- `index.html`: static, no script, one section per non-empty category, lazy thumbnails linking to the full JPEG, relative links, escaped names.

## Modes
- `--dry-run`: inventory, camera facts, dedupe, classify. Prints counts per category, duplicates, skipped and an estimate (`DEVELOP_S_PER_PHOTO_ESTIMATE` = 20 s x photos / workers). Writes nothing, creates no run folder or ledger (an existing ledger is read).
- `--index-only`: inventory, camera facts and `ledger.record` under run id `index:<folder name>:<date>`; no decode. The run folder holds only `run.json`. Use it once over folders that were already sorted by hand.

## Verification

`tests/test_photo_sort.py` (fake hooks plus real runs on tiny JPEGs) and `tests/test_photo_sort_cli.py` (parser, config, admission fence, CLI dry-run).
