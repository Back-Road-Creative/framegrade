## Review set (`gomoveshift review-set`)

> Reference, not contract. The source wins if this disagrees with it.

The review set is the small, reproducible collection of source-vs-output pairs a human
looks at to judge the grade, tonemap and edit per driving condition (daylight, overcast,
dusk/night, tunnel, creeping, parked traffic, HDR gradients, GPS gaps). The command group
only gathers evidence: it never changes a render. Each subcommand is a thin wrapper over
`pipeline/review_set_scan.py`, `review_set_stops.py` and `review_set_render.py`.

Every failure (missing file, non-empty output directory, schema error) prints one
`Error: ...` line to stderr and exits 1.

### `scan` -- candidate windows per condition

```
gomoveshift review-set scan --scores <stages_dir>/drive_scores.npz \
    --stop-mask <stages_dir>/stop_mask_2026-01-02.npz \
    --gps <stages_dir>/gps_points_2026-01-02.json --tonemap clip.tonemap.json \
    --out scan.json
```

A drive writes all three inputs to its stages directory: the score cache
(`drive_scores.npz`), the stop mask (`stop_mask_<date>.npz`) and the day's usable GPS
fixes (`gps_points_<date>.json`, a JSON list of `[t, lat, lon]` floats in the same
day-start seconds the stop mask uses). The GPS file is written beside the stop mask and
only when the day has at least one usable fix; a write failure logs a warning and never
aborts the day.

Reads the per-second score cache (`scores` array) and, optionally, the stop-mask npz
(`states`), a JSON list of GPS fixes (`[t, lat, lon]` rows or `{time, lat, lon}` objects)
and the `.tonemap.json` sidecar. Writes, per condition, the windows with start/end
seconds, the rule that fired and its inputs, to `--out` or stdout. A condition whose
optional input was not given reports `no_input`.

Conditions with no detector are reported as `unsupported`: **backlight, foliage and wet
windshield**. A human tags those; the tool never guesses them. Parked traffic is a
labelled heuristic.

The `ScanThresholds` defaults are **provisional starting values, not tuned against real
footage**. Treat the windows as candidates to look at, not as a classification. The
command exposes no tuning flags.

### `stops` -- stop and kept runs for one sequence

```
gomoveshift review-set stops --stop-mask stop_mask_2026-01-02.npz \
    --receipt keycut_receipt.json --out stops_2026-01-02.json
```

Writes the schema-versioned `review-set-stops` JSON: contiguous state runs with source
seconds and a kept/cut decision. With `--receipt`, each kept run also carries its output
time ranges; without one, output times are omitted and the document says so. `--sequence`
overrides the label taken from the file name.

### `pairs` -- scan windows mapped to output times

```
gomoveshift review-set pairs --scan scan.json --stops stops.json --out pairs.json
```

For each scan window the midpoint is looked up in the stops runs. When it falls in a
kept run that has output times (`stops` was given a `--receipt`), the pair is
`[source_t, output_t]` with `output_t = output_start + (midpoint - source_start)`.
Otherwise the window is not paired; it is listed under `unpaired` with its condition,
window and a reason: `cut`, `no_output_times`, `outside_runs` or `not_in_output`
(a kept second the receipt does not carry). Output times are never estimated.

`pairs.json` is the `review-set-pairs/1` document: `pairs`, `detail` (condition,
window, source and output time per pair) and `unpaired`. `render --pairs` accepts that
document directly, as well as a bare JSON list. `--out` defaults to stdout.

### `render` -- paired stills and excerpts

```
gomoveshift review-set render --source src.mp4 --output out.mp4 \
    --pair 12.5:9.0 --pair 40:31.5 --out-dir review/ --excerpt-seconds 3
```

Pairs come from `--pairs FILE` (a JSON list of `[source_t, output_t]`, or the `pairs` document above), repeated
`--pair S:O`, or both. For each pair it writes a still and a full-speed excerpt from each
side plus `provenance.json` (version, git sha, ffmpeg version, effective drive config,
input sizes and sha256). `--out-dir` must be new or empty. Pass `--synthetic` only for
generated test media; it is recorded in the provenance.
