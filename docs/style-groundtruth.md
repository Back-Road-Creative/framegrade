# Photo style ground truth

The photo pipeline auto-picks a `natural` grade plus two alternatives per photo
(`select_styles` -> `_distinct_styled`). Every suitability score in that picker is a
hand-authored guess and **nobody has measured whether the auto-pick agrees with a human**;
there is no label set. This harness builds one. Three tools, run in order:

- `tools/style_groundtruth_render.py` — pin the corpus, render each capture's offered
  variants, compose one **blind** judging grid per capture.
- `tools/style_groundtruth_judge.py` — show a grid to an LLM judge, record its pick.
- `tools/style_groundtruth_score.py` — agreement metrics + a leave-one-out bake-off.
- `tools/style_groundtruth_signal.py` — dormant-trait signal + mono-leaning candidates.

## Corpus

A local corpus of **24 captures**, each shipping as `.JPG` + `.NEF` (four also as `.dng`),
so a file count over-reports the corpus ~2x — the manifest dedupes by capture stem.
`gms_test/` (rendered pipeline OUTPUT), `luts/` and non-photo files (`kodachrome.cube`)
are excluded. 24 captures catches **gross** disagreement between picker and judge; it is
not enough to fit a model, and no number from it is a weight.

## Blindness

Panels are captioned `A`/`B`/`C` and the panel -> style mapping is shuffled with
`random.Random(capture_id)`, so it reproduces on any host on any day and no wall clock leaks
in. The style name goes to the sidecar JSON only — never into the image, never into a panel
filename — so a judge cannot recover the machine's own ordering from what it is shown.

## Schemas (both pinned at `schema: 1`)

Manifest — `style_groundtruth_render.py manifest --corpus DIR --out FILE`. `raw`, `jpeg`
and `exif_dt` are null when absent; rendering prefers the raw.

```json
{"schema": 1, "corpus_root": "/abs/corpus", "captures": [
  {"id": "CAPTURE_0020", "raw": "/abs/CAPTURE_0020.NEF", "jpeg": "/abs/CAPTURE_0020.JPG",
   "exif_dt": "2030-01-01T19:05:26"}]}
```

Render sidecar — `... render --manifest FILE --out DIR`, written to `<out>/render.json`.
It is the answer key: the only place the label -> style mapping exists. Keep it from the judge.

```json
{"schema": 1, "captures": [
  {"id": "CAPTURE_0020", "source": "/abs/CAPTURE_0020.NEF", "grid": "/abs/CAPTURE_0020_grid.jpg",
   "panels": [{"label": "A", "style": "moody", "file": "/abs/CAPTURE_0020_A.jpg"}]}]}
```

## Composed workflow and the signal tool

manifest → render (narrow + `--styles` wide) → judge both → `score --wide-labels` → `signal
--manifest render.json --labels labels.json --json out.json [--report out.md]`. Read-only —
rankings are the shipped `_score`/tie-break with only the probed delta; the `--json` doc is
the load-bearing artifact (the report is a stub). Per dormant trait (derived from predicate
source, plus EXIF capture hour) it carries NET FLIPS — agreements gained minus broken were
the judge-preferred style given one `SCORE_LEANING` nudge on the trait's best single split —
plus that split's Cramér's V; net flips decides. The mono ladder holds `C_delete` plus every
`incumbent AND one extra conjunct` leaning: mono-wrong captures un-stuck from first-backfill
(rank 4), Δ alt-rank agreement, conjunct count. Ship rule: the best
candidate ships iff unstuck ≥ 5 with Δalt ≥ −1; else `C_delete` iff **its** unstuck ≥ 3; else none.

## What the labels actually are

Until a human calibration pass replaces them, the labels are an **LLM judge's proxy for
the photographer's taste**, not ground truth in any stronger sense. Read a disagreement rate
as "the picker and one proxy judge differ this often," and re-check the proxy against a human
on a sample before moving a weight because of it.
