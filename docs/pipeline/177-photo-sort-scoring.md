## Photo sort scoring

`gomoveshift photos sort` scores every developed shot and orders each category by that score. Quality decides order, never inclusion: nothing is hidden or dropped for a low score. Output files are `NNN-<stem>.jpg` with NNN the quality rank inside the category (1 = best). A shot that failed to develop leaves no gap, because the files are numbered after scoring.

## How a shot is scored

The score is read from the developed JPEG, re-read in the parent process after the workers finish. The workers stay as small as before (they return nothing), the number matches the picture you look at, and the parent holds one decoded image at a time.

- Technical: sharpness and clipping (`quality_order.technical_score`); a night sky adds star count, star sharpness and trailing, with the exposure and focal length from the capture record.
- Category signal: `quality_order.category_signals` (light and sky for landscape, stars and dark sky for night sky, subject isolation for close-up).
- Aesthetic: CLIP ViT-L/14 plus the LAION head (`pipeline/photos/aesthetic.py`), 0 to 1. Total = technical x (0.5 x category + 0.5 x aesthetic); with no aesthetic score it is technical x category.
- Order: total descending; a tie breaks on the source path, ascending. A shot whose JPEG can not be read or scored follows the scored ones in capture order, with a blank score.

## The aesthetic pass

Built once per run in the parent process (the model is 307 MB), on the CPU with 2 threads. Each value is cached in `ledger.sqlite` by the source file's sha256 under `aesthetic:v1`, so a re-run (`--reprocess`) skips the model. The dry-run estimate adds `AESTHETIC_S_PER_PHOTO_ESTIMATE` (3 s, a guess to be measured) per photo.

- No GPU: the GoMoveShift drives hold the shared GPU lock, and a waiter that times out fails, so the sort never takes it.
- Weights missing: the run goes on with no aesthetic score for any shot and one warning, in the terminal and in `run.json`, naming `gomoveshift photos fetch-models`.
- One image's aesthetic failing leaves that shot's aesthetic blank, counts it in `run.json` (`scoring.aesthetic_failed`), and keeps the shot.

## Outputs

- `ranked.csv`: `quality_score` is the total, `aesthetic_score` the aesthetic value, plus `technical`, `category_score` and every part as `technical.<name>` or `category.<name>` (a night sky has `stars` in both, so the prefix keeps them apart). Columns are the union across rows; blank where a shot has none.
- `index.html`: each section in rank order, the total (2 decimals) under each thumbnail.
- `run.json`: `scoring` counts (`aesthetic_scored`, `aesthetic_cached`, `aesthetic_failed`, `score_failed`) and the score timing.
- Terminal: the top 5 per category (rank, file, total).

## `photos fetch-models`

Downloads and hash-checks the two weight files into `$FRAMEGRADE_AESTHETIC_DIR` (else `<gms data home>/models/aesthetic`) and prints both paths; exit 0. A failed or mismatched download prints the reason to stderr and exits 1. It is the only code that touches the network, and it does not take the media admission fence.
