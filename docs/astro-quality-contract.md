# Astro quality contract

Establish a measurement contract before tuning anything against a real render. Do not judge astro quality by eyeballing a final TIFF — measure a synthetic property that was actually injected.

## The nine acceptance gates (in brief)

| # | Gate | Summary |
|---|------|---------|
| 1 | Freeze inputs | Stage real lights/darks/sidecars read-only; never commit private RAWs. |
| 2 | Synthetic fixtures | Known gradients, broad 1-3% nebulosity, stars with known flux/RGB/PSF, cloud pedestal, hot pixels, sensor-fixed streaks, single-frame trails, warp borders, irregular horizons, n=3/5/8/20 stacks. |
| 3 | Signal-integrity | >=90% injected broad-feature contrast retained; noise tracks sqrt(n); RGB ratios pinned; no sky/ground seam. |
| 4 | Rejection | False rejection on clean n=5/n=20 Gaussian stacks under a pinned low ceiling; a planted bright artifact above a documented SNR is rejected reliably; n=3/4 fallback pinned. |
| 5 | Geometry | Transform residual median/p95, contributor count per output pixel, and output crop recorded; no exported pixel silently includes warp-fill zeros. |
| 6 | Develop | Background lands in the declared window; faint-feature separation increases without clipping star cores; zero/constant inputs stay finite; tone/color replay byte-identical. |
| 7 | Meteor | Automatic reinsertion stays off until a labeled set meets a high precision target; the base master always exports. |
| 8 | Resource | Measured child-process peak RSS, never a lower arithmetic estimate. |
| 9 | Human proof | Linear-stack/background-model/coverage/rejection/sky-mask/final panels at 100% crops plus a contact sheet, per quality-changing change. |

Gate 8 is now measured, not estimated: `run_benchmark` (`astro_benchmark.py`, no CLI wiring yet) forks (Linux-only: on Windows it raises `LinuxOnlyBenchmarkError`) `prepare_frames`+`stack_run` into a child process that self-measures its own peak RSS (per-run, not `RUSAGE_CHILDREN`'s cross-call max) and returns it over a queue. The child first resets its kernel high-water mark (`/proc/self/clear_refs`) and reads `VmHWM`, so nothing it held before its run floors the figure; if the reset or read-back is refused it falls back to the un-reset `ru_maxrss` and says so in `rss.unit_note`; that figure, plus input/output content hashes, lands in `benchmark.json` in the run's `output_dir`. Three gate-9 panels aren't produced by the benchmark yet, each for an honestly-recorded reason in that same JSON's `unavailable_panels`: `background_model` (`astro_develop` returns only before/after background scalars, not the map), `rejection` (the sigma-clip mask is discarded after its `mean_rejected_fraction`), and `sky_mask` (`stack_run`'s horizon mask is local to the call and never returned). `coverage` is no longer on that list: it is produced once real per-frame validity is threaded through `astro_stack`.
## Deep-linear component contract

`astro_stack.stack` still ships its existing blended `StackResult.rgb`, but it now retains the
pre-blend scene-linear parts needed by the forthcoming deep-combine artifact: `sky_sum` (registered
RGB numerator), `sky_count` (uint16 per-pixel contributors after warp validity and sigma rejection),
and `ground` (unwarped fixed-camera RGB median). `sky_sum / sky_count`, blended with `ground` under
the horizon mask, reconstructs the shipped RGB up to ordinary float32 roundoff. These fields are internal until results are persisted; ordinary TIFFs and sidecars are unchanged.

The short-stack degeneracy median now sorts the at-most-20 valid frame samples directly instead of
materialising NaN planes for two `nanmedian` passes. Odd/even/masked/all-missing fixtures are exactly
equal to the former median contract, and every rejection/noise/coverage gate remains unchanged. A
repeatable 10-frame 768x768 synthetic benchmark shows the sorted-sample median is faster per rejection
run than the former `nanmedian` passes. On full-resolution aligned arrays in one paired process the
candidate is also faster, with every return value exactly equal. Absolute times depend on machine
load and are not a baseline; the paired comparison is the evidence.

Gradient removal now builds its immutable full-frame polynomial design once and reuses it for all
three channels; coefficient fitting and per-channel surface math are unchanged. The design is now built once
rather than once per channel, and the float32 result remains byte-exact.

## Deep-linear artifact contract

`astro_linear.write_linear_artifact` persists those three components for a future deep
combine as one atomic directory: uncompressed, memory-mappable `sky_sum.npy` (float32 RGB),
`sky_count.npy` (uint16 through schema 2, float32 from schema 3 -- see below), and `ground.npy`
(float32 RGB), plus `manifest.json`. The manifest
pins schema, algorithm version, reference RAW path and capture identity, the ordered source-frame list, mask,
registration and photometry provenance, stack metrics, and a SHA-256 plus shape/dtype record for
every payload. Since schema 2, each `registration` entry may additionally carry a `triage` block
(`star_count`, `background_median`, `fwhm_px`, the major/minor/ellipticity shape split reused
from `astro_sidecar._star_shape_payload`), `radial_inlier_residuals_px_in_sample` (center/middle/
outer RANSAC-inlier residuals -- explicitly in-sample, not held-out geometry evidence), and
`exposure_s`/`captured_at` read from the source frame's own EXIF -- so a later deep-combine pass
can select or weight frames by measured quality instead of consuming every registered frame
uniformly. `load_linear_artifact` accepts any schema from 1 through the current value rather than
crashing on an artifact written before this addition; every schema-1 key and shape is unchanged.
The destination is only renamed into view
after the complete artifact validates; a missing, changed, or schema-incompatible payload is
refused on load. A combine may reload with `verify_hashes=False` only after it has verified every
artifact once; structural, dtype, and shape checks remain mandatory on those trusted reloads. This
avoids re-hashing multi-gigabyte payloads during each bounded-memory combine pass.
`astro_orchestrator.stack_run` can now publish that artifact directly from the components it already
computed. Its manifest records the resolved mask boundary, every attempted source path,
per-used-frame registration fit, and photometric offset/scale; `linear_only=True` returns
immediately after validated publication, before develop, meteor processing, TIFF export, or
sidecar work.

## Deep-combine input validation

`astro_combine.validate_artifact_set` validates the entire ordered input set before any deep output
may be written. Every payload is hash/shape/dtype checked through the artifact codec; schema
algorithm, sensor capture identity, image shape, and mask mode must agree; explicit masks must name
the same requested boundary; and source RAW lists may not overlap. Automatic masks may retain their
honestly different resolved rows. Developed TIFFs and one-chunk “deep” sets are refused. The default
reference is the temporal midpoint AMONG the top quality quartile, with registration residual,
contributing frame count, and input order as deterministic tie-breaks within that quartile; an
explicit reference must be a member of the validated set.

Quality here is the persisted per-chunk `sky_weight` (defaulting to 1.0 for a schema older than 3,
same as `chunk_sky_weight`'s own "nothing to measure" value); the quartile is a threshold on that
weight (its own 75th percentile), not a fixed top-N slice, so a night where every chunk ties at the
same weight still lets every chunk compete on proximity to the midpoint.

## Deep-combine geometry: one global reference, no chaining

`astro_combine.plan_direct_transforms` is the deep path's only geometry planner: every chunk fits
directly against the single global reference chosen above, and a chunk whose direct fit fails is a
hard `AstroCombineError`, never a silently-composed fallback through a neighbour. This retires the
adjacent-chain fallback for the deep path -- composing a transform through a neighbour compounds
that neighbour's own registration error, and on a long night the deepest chunk sits under many composed
links. Each plan still records `direct` or `identity` plus its (single) link, so
`deep.json`'s `chunks[].transform.method` is one of exactly those two values for the deep path.

`plan_artifact_transforms` (the direct-fit-then-adjacent-chain planner) is kept only for callers
that still need chaining; the deep path no longer calls it.

## Quality-weighted sky contribution

`astro_linear.chunk_sky_weight` derives a scalar in `(0, 1]` from a chunk's own persisted
per-frame triage (`fwhm_px`, `background_median` -- schema 2): the more its own frames'
focus and sky background level disagreed with each other, the less that whole chunk
contributes. `write_linear_artifact` multiplies it into `sky_count` before writing (schema 3;
the scalar itself is recorded as `sky_weight` in the manifest for traceability), so `sky_count`
becomes a per-pixel quality-weighted contribution sum rather than a literal frame count.
Fewer than two triage-bearing registration entries -- every schema-1/2 artifact, or a schema-2
artifact with no triage recorded -- weight 1.0: identical to today's plain frame count.
`load_linear_artifact` reads `sky_count.npy` as uint16 below schema 3 and float32 from schema 3
on, so a pre-existing artifact set never needs rewriting to combine. Every reader downstream
(`astro_combine._sky_mean`, `_warped_components`, `combine_sky_two_pass`,
`astro_noise_gate.effective_contributors`) already treated `sky_count` as a generic
non-negative numeric weight rather than a literal integer, so none of them needed a structural
change -- except the noise gate's ideal-gain estimate, which replaced a plain
`sum(counts) / frames_per_chunk` with the weighted effective sample size
`(sum r)^2 / sum r^2` (`r = count / frames_per_chunk`): the two agree exactly when every chunk
weighs a full chunk, and the weighted form no longer lets one over-weighted chunk masquerade as
several independent ones.

The production pair fitter derives each registration image strictly from `sky_sum / sky_count`,
never the blended TIFF or ground plane, and applies the shallower of the two persisted resolved sky
boundaries to star detection. Each successful direct or adjacent link retains matched-star count,
inlier fraction, match yield, and median/p95 RANSAC-inlier residual alongside the composed
transform. It also records count/median/p95 inlier residuals for the centre, middle, and outer
radial sensor thirds, so an apparently good whole-frame fit cannot hide edge degradation.

Sky accumulation is exactly two passes and holds at most one chunk payload live. Pass 1 warps the
sky numerator and float contributor weights together, robustly maps each chunk pedestal and bright
signal scale to the reference, and accumulates per-pixel chunk moments. Pass 2 reopens one chunk,
applies the same offset/scale algebra directly to its numerator, performs a leave-one-out bright
chunk test, and accumulates surviving numerator/count. The output count is the effective contributing
frame weight after interpolation and rejection, not a nominal chunk count; foreground ground planes
are never consumed by this sky path. Set validation hashes each payload once; these later bounded
passes use structurally validated trusted reloads instead of re-hashing the full artifact set.

Ground accumulation is a separate two-pass reducer over the unwarped `ground.npy` planes. It maps
each chunk’s fixed-coordinate foreground illumination to the global reference, performs the same
chunk-level bright outlier test, then weights surviving values by that artifact’s contributing
frame count. No geometry argument exists on the ground API, making a sky transform structurally
impossible; a light surviving one short ground median is rejected rather than made permanent.

`astro_combine.deep_combine` joins those pure stages into one scene-linear product and blends only
once under the global reference artifact’s resolved mask. It preserves the full sensor rectangle
and reports the nominal frame total, effective-count minimum/median, and the fraction below a
documented floor of half the nominal contributors (never below the ordinary three-sample stack
floor). It deliberately performs no develop, meteor, export, or sidecar work.

`astro_combine_output.publish_deep_product` refuses TIFF, `.coverage.npy`, or `.deep.json`
collisions and preflights their aggregate disk need before developing/exporting exactly once, with
meteors off. Its atomic coverage map retains every effective sky contributor count; its atomic
sidecar pins artifact/payload hashes and source RAWs, capture/frame identity, photometry, geometry
diagnostics, rejection/coverage, reference/mask, develop parameters, warnings, and TIFF hash.

## What the fixtures provide

`tests/astro_fixtures.py` — deterministic, pure-numpy synthetic scene-linear fixtures (`(H, W, 3)` float32, matching `astro_calibrate`/`astro_stack`'s frame contract; lower-clipped at 0 but not upper-clipped, so bright stars/pedestals may exceed 1.0 by design; no file I/O, no RAW dependency). `gaussian_stack(n, shape, sigma, seed)` gives a clean iid-Gaussian stack (n=3/4/5/8/20) for false-rejection/noise tests. `sky_frame(...)` composes a light-pollution gradient, a nebulosity patch at a caller-specified contrast (+ mask), stars with known flux/PSF/RGB, a cloud pedestal, hot pixels, a single-frame or sensor-fixed streak, warp borders, and an irregular ground silhouette (+ mask) — each component independently switchable, always recorded in a ground-truth dict. `multi_frame_sequence(...)` builds an n-frame sequence from one `sky_frame` spec: sky content shifts per frame while sensor artifacts repaint at identical coordinates every frame.

The numeric metric helpers (signal retention, noise/SNR, RGB stability, rejection rate, coverage, PSF shape, seams) land separately in `astro_quality.py` (a sibling module) — this module stops at generating ground-truthed pixels, not scoring them.

## Retention metric correction

`contrast_retention` (gate 3's `>=90%` input) originally divided by `broad_feature_contrast`
measured on the UNCORRECTED frame, whose gradient core-vs-ring sampling dominates the ratio and
flips sign with geometry alone. Measured directly against the earlier implementation (
`contrast_retention(frame, oracle_after, feature_mask, background_mask)` — its signature, no
correction) over the 48-combination sweep in `TestContrastRetentionOracleMatrix` (6 gradient
shapes x 4 geometries x 2 contrasts, an oracle-perfect subtraction): scores range **-28.32 to
+4.30** and land within 0.05 of 1.0 for only 2/48 combinations — never reliably ~1.0.

The correction fixes this by requiring a gradient-free `reference`. Production has no gradient-free frame,
so the caller supplies `known_nuisance` (the fixture's own ground-truth field); `reference -
known_nuisance` stands in as the nuisance-free denominator via `contrast_retention(...,
known_nuisance=gradient)`. That correction is the fix.

### The contamination gate is removed

Four earlier rounds also tried to make `contrast_retention` VERIFY `known_nuisance` against
`reference` and raise `ContaminatedReferenceError` on a mismatch: a null space in a planar-tilt
fit; blindness to a nuisance ON the feature; an untested annulus between `feature_mask` and a
caller-chosen `background_mask`; a noise-normalised z-score that flagged a clean reference whose
ordinary sky content (stars, ground) merely shared coarse scale with a smooth nuisance field.
A fifth round's Pearson `|r| >= 0.25` (over `reference` demeaned within `feature_mask` and its
complement) finally stopped false-rejecting a clean reference WITH stars (0/240 across four
component mixes) — but only because a bright point source inflates the correlation denominator;
without a star the false-reject rate measured ground 200/240 (83.3%), cloud 102/240 (42.5%), warp
80/240 (33.3%), ground+cloud 172/240 (71.7%). The detection floor paid for that fix one-for-one:
amplitude 0.000852 -> 0.00814 (9.6x worse) on a flat fixture, 70.9x-99.6x worse on a
structurally-rich reference, and on a realistic 60-star + ground reference the false-reject
cutover moved past the original null-space failure mark. Rejecting on the nuisance's OWN
contribution to `broad_feature_contrast` was tried and refuted too: a ramp- or gap-shaped
nuisance gives an exact 0 against concentric masks by construction, reopening the null space the
first four rounds closed.

**Conclusion: no correlation-based statistic separates "the nuisance" from "any other broad sky
structure" without either false-rejecting ordinary references or losing the sensitivity to catch
a real mismatch.** The gate (`ContaminatedReferenceError`, `_nuisance_correlation`,
`_NUISANCE_CORR_REJECT`) is deleted. **`contrast_retention` now TRUSTS `known_nuisance`** — it
does not check it against `reference` at all. A caller passing a wrong or incomplete
`known_nuisance` gets a silently wrong retention value; nothing here detects it.

**Untried follow-up (not built, nothing shipped depends on it):** a robust matched-filter
amplitude test — fit `known_nuisance`'s amplitude to `reference` by iteratively reweighted least
squares over `feature_mask ∪ gap ∪ background_mask` so stars and ground get down-weighted rather
than counted as signal, then test that fitted amplitude against the residual's robust scale
(e.g. MAD). Correlation-based tests fail because Pearson `r` (or any variance-based statistic)
can't tell "the nuisance" from "any other large smooth structure" that happens to correlate with
it; a matched-filter amplitude fit against a robust residual scale asks a different question —
"how much of THIS shape is present," not "how correlated is the whole field" — and was never
implemented or measured.

**INJECTED-SIGNAL FIXTURE METRIC ONLY** — a gradient-free reference exists only in a synthetic
fixture, never production, so `contrast_retention` has no valid production caller without
`known_nuisance`. Even with a correct `known_nuisance`, the ratio is exact (1.0) only for a
perfect correction of `after` — a partial one stays geometry-sensitive and can go negative, so a
`>=90%` gate needs a fixed geometry across runs.

**Every retention figure computed before this fix — including gate 3's `>=90%` check
"19.5% retained" headline — REQUIRES RE-MEASUREMENT** against the corrected denominator.

## Gate 3, background half — implemented, not just specified, gate conditional

`astro_develop.remove_gradient`/`develop_stack` take an optional `sky_mask`:
pixels marked False (ground/foreground/detected bright sources) contribute no samples to
the gradient fit. Rejection stays two-sided with a mask -- a one-sided rule (reject only
cells too BRIGHT) can't catch a cell reading too DARK, so an imperfect mask leaking a few
rows of dark foreground would tilt the fit instead (measured: two-sided left no residual
spread at a 4-row leak vs 1.36% for one-sided, same samples). Default `sky_mask=None`
is byte-for-byte the old whole-frame behaviour (frames too small for the grid to reach
every polynomial term can differ, but only when a mask IS passed).

Nothing here builds a real mask, so production still fits whole-frame; retention must be measured end-to-end against the corrected `known_nuisance`
metric above before the orchestrator uses a real mask. Noise/RGB-ratio/seam halves of gate 3 are untouched.

**Re-measured against the corrected metric, and re-measured AGAIN (the first re-measure
used an incomplete hand-reconstructed `known_nuisance` -- it recreated only the ramp,
not the ramp's own contribution to nebulosity, which `sky_frame` scales off the REALISED
sky including the gradient):** using the metric's documented primary contract instead --
`reference` is the fixture regenerated with `gradient_amplitude=0.0`, everything else
identical, `known_nuisance=None` -- on `astro_fixtures.sky_frame`'s ramp + oracle mask
(128x128, `gradient_amplitude=0.02`, degree-2), masked retention is ~1.005-1.009 across
0.03/0.01 nebulosity contrast: it CLEARS the 0.90 gate. The earlier ">=94%"/"19.5%
retained" figures were still broken-metric artifacts, but the immediately-prior "~0.67,
gate not cleared" verdict was itself wrong -- an incomplete reconstruction, not the fit.
Masked retention (~1.01) measurably beats unmasked (~0.80, same fixture) and flattens the
ramp's corner spread from ~0.017 to ~0.00001.

**Underdetermined fallback is now observable, not silent:** the constant
(no-op) surface fallback below `UNDERDETERMINED_MARGIN * n_terms` samples used to be
silent -- the earlier gate scored a no-op subtraction a false 1.000 since it wasn't
dividing by a nuisance-free denominator. It now `warnings.warn`s (`RuntimeWarning`).

**Masked fit now also checks CONDITIONING, not just sample count:** a mask
spanning only 1-2 grid rows can clear the sample-count guard (measured: 32 samples, above
a degree-2 fit's 18-sample floor) while the design matrix is catastrophically
ill-conditioned (measured cond ~1.5e18) -- that case now falls back too, via
`CONDITION_NUMBER_MAX = 1e8` (float64's standard ~8-good-digit cutoff).

**Non-polynomial (dome) gradient -- measured, not fixed, explicit follow-up:**
on a centred Gaussian dome (128x128, background 0.02, amplitude 0.02, centre
`(0.5, 0.5)` of the frame, `sigma = 0.5 * min(H, W) = 64px`, no mask), the degree-2
default fit leaves ~36.7% of the corner-spread in the sky (vs <5% for the ramp case);
degree-4 closes most of it (~2.9%) at the cost of more samples/terms. Retention is NOT
the failure signal here -- it read >1.0 (inflated, not degraded) on this fixture -- so
only the flatness check catches it. Follow-up: raise the default degree, or fall back on
measured residual size, before claiming gate 3 for a non-polynomial sky.

## Automatic horizon: refused below a measured star floor

`stack_run` places the sky/ground boundary from the reference frame's own star density whenever the operator passes no `--horizon FRACTION`, and that estimate had no floor on how much evidence it needed. Driven end to end, a frame yielding 10 detections put the boundary at row 57 of 200 against a ridge injected at row 160 — 63.7% of the true sky handed to the unaligned ground median, written to the sidecar as a confident automatic estimate, no refusal. Two attempts to replace the estimator were closed after adversarial verification: with few detections the density map cannot tell "the sky ends here" from "the only stars happen to be up here", and no statistic derived from that density recovers the difference.

So the automatic path is **gated, not improved**. `astro_stack.MIN_AUTO_HORIZON_STARS` (40) is an evidence floor on the reference frame's DETECTED star count, checked before the estimate and before any irreversible artifact; below it `stack_run` raises `AstroOrchestratorError` naming `--horizon FRACTION` as the remedy, and `photos astro`'s existing per-chunk handler skips that chunk and continues — no second refusal path. Measured on 3600 synthetic 200x200 fields through the shipped path (`detect_stars` → blurred impulse density → `horizon_mask`), extinction falloff 0.5, ridge at 0.65 and 0.80 of frame height, 120 seeds per cell, boundary error as % of frame height: at 40–44 detections it is median 1.5 / p90 6.5 / p99 10.0 with 0.8% of frames off by more than 10% of frame height, against 2.3% gross at 35–39, 5.5% at 30–34, 8.5% at 20–24, 16.0% at 15–19, 24.8% at 10–14 and 44.5% below 10 — 40 is the smallest bin edge at which that gross-error rate drops under 1%.

**`astro_triage`'s usable-frame floor is not a defence against this case.** `MIN_USABLE_STAR_COUNT = 5` decides whether a frame is stackable at all, so a 6-, 10- or 27-star frame clears it and always did. The two floors answer different questions and the low one is deliberately left alone: a frame under `MIN_AUTO_HORIZON_STARS` still stacks normally once an operator places the boundary with `--horizon`.

The same change closes a fail-**OPEN** defect in the estimator's own rule. With a sky shallower than 0.25 of frame height the row profile's own upper quartile is 0, the threshold degenerates to 0, and the old non-strict `profile >= threshold` held for every row of any non-negative density — so the boundary came back as the frame HEIGHT, "sky everywhere, no masking at all", never the "ground everywhere" the docstring implied, and the empty-`sky_rows` branch was unreachable. The clearance test is now strict. Swept over the full 1..199 ridge range on a 200-row fixture: non-strict was wrong for every ridge ≤ 49, strict is exact for all 199.

## Dark identity: validated at both production call sites

`astro_calibrate.validate_dark_identity` refuses a dark whose camera / raw shape / CFA pattern / ISO / exposure disagrees with the lights, and `build_master_dark` runs it whenever a caller supplies `light_identity` + `dark_identities`. Neither production caller supplied them: `prepare_frames` and `astro_trails.render_trail` both called `build_master_dark([...])` bare, so the check was skipped on every real run since it shipped. Dark current scales with exposure time and with ISO, so a 10-second dark averaged into a master for 15-second lights under-subtracts and a mismatched-ISO dark subtracts the wrong read-noise floor — no error, just a quietly wrong image. A 10-second frame among 15-second darks is the typical case, and previously had to be excluded by hand after manual inspection.

Both call sites now go through one seam, `astro_orchestrator.build_validated_master_dark(light_paths, dark_paths)`:

- **Light identity is the first light path given** (a chunk's earliest frame in capture order on the CLI path), and the refusal names it. `astro_sequence` splits a run the moment ISO changes at all, and camera/raw shape/CFA pattern are body properties fixed for one card, so four of the five compared fields cannot vary inside a chunk. The fifth, exposure, is held only within `astro_sequence.MAX_HOLD_RATIO` (1.15x, measured between neighbouring frames of one detected run) while the comparison is exact equality — judging the darks against every light in turn would refuse runs the sequencer deliberately admits.
- **An unreadable field refuses rather than passing as a match.** `validate_dark_identity` skips a field that is `None` on either side, so an unreadable exposure would otherwise sail through; the seam refuses first, with `AstroCalibrationError` naming the file and the fields, before any dark mosaic is read.
- **A refusal is fatal to the chunk, never per-dark.** `--darks` is one per-run set, so dropping the odd frame silently would hand back a master built from an unexamined subset. `photos astro`'s existing per-chunk handler prints the message and skips the chunk; a run that writes nothing still exits non-zero. The message names the offending file and the disagreeing field (`<dark>.NEF: dark exposure_s does not match light <first light> -- light=15.0 dark=10.0`), so an operator drops one frame from `--darks` and re-runs instead of guessing.

A run with no `--darks` is untouched: no identity is read and no master is built, pinned by a test on each path.

## Cloud triage judges a group against its clearest frames, not its own median

`astro_triage.triage` flagged cloud purely relative to the group it was handed: a frame was cloud only if it held under 0.4x that group's MEDIAN star count or over 1.5x its median background. That finds a cloud crossing an otherwise clear run and is structurally blind to a night that clouds over and stays — every frame degrades together, the median degrades with them, nothing is anomalous. On a capture where cloud moves in and stays, the old detector flagged only the few frames inside the chunk where the transition happened: the post-onset median star count had dropped well below the pre-onset one, but the background barely moved.

The cloud comparison now runs against `_cloud_baseline` — the median of the group's `CLOUD_BASELINE_FRAME_COUNT` (3) HIGHEST star counts and, independently, of its 3 LOWEST backgrounds; the clearest condition the group actually shows. Dew's baseline (`FWHM_BASELINE_FRAME_COUNT`, the opening 3 frames) is deliberately left alone and now has its own constant, so retuning one can't silently retune the other. Both remain report flags: nothing refuses a frame or a run.

- **Why the clearest frames, not the first 3.** A first-N baseline assumes the run opens clear. Take a 20-frame chunk that opens with 15 clouded frames and clears for the last 5: the first-N baseline flags none of them, the clearest-frames baseline flags all 15. Both need 2 clear frames; only one of them needs those frames at the start.
- **Why 3.** It is the smallest window whose median ignores one anomalous frame, and it costs 3 of a chunk's 20 (`astro_sequence.STACK_LEN_DEFAULT`). With cloud onset at frame 0 or 1 of a 20-frame chunk nothing is flagged (fewer than 2 clear frames are left), and with onset at frame 2 or later every degraded frame is flagged.
- **This widens the old rule, it never moves it.** The top-3 median is ≥ the group median for stars and ≤ it for background by construction, so every frame the median rule flagged is still flagged — a transient cloud crossing a clear run included (pinned by test).
- **Envelope — where it stops catching a slow overcast.** A linear decline across one chunk is caught only once its last frame falls under 0.4x baseline, that is, a star loss end to end of more than 60% (just over 61% over a 20-frame chunk). The old median rule needed a 75% loss, so a slower decline than that is a documented miss.
- **The background half does not catch a cloud-in-and-stay night and never did**: the background rises by only about a tenth against a 1.5x rule, so the star collapse carries it. Loosening 1.5 was not attempted: a rise that small is inside what an ordinary clear night does (moonrise, airglow), and cloud that dims stars several-fold is the far cleaner signal.
- A heavy dew progression removes stars too and can now read as cloud as well as dew. Both flags are advisory and appear side by side in the report.

## Cloud baseline carries forward across chunks

The residual named above — an entirely overcast chunk, or a run that starts cloudy and never clears, has no clear frame of its own to be judged against — is closed for the first case by threading a RUN-level baseline across chunks: `astro_triage.update_run_cloud_baseline` folds each chunk's clearest condition into a running `(star, background)` baseline (max star, min background, so a later cloudier chunk can never erode it), and `triage` takes an optional `run_baseline` that raises/lowers its per-chunk baseline the same way — omitted, behavior is unchanged. `prepare_frames` threads it through as `cloud_baseline`; the `photos astro` chunk loop carries the running value forward, updating it after every chunk. No second decode pass: chunks are still processed once, in time order.

- **The hazard: star count alone conflates defocus with cloud.** A defocused frame's noise blobs count as "stars" the same as real ones, so a baseline built from defocused frames could in principle read genuinely sharp later frames as a collapse and flag the best frames of a night.
- **Defocus is preferred, not required.** The filter originally dropped every `likely_defocus` frame outright, and **that made the feature inert on a night with uniformly soft focus.** The clear frames on such a night can be *soft-focused* and read `likely_defocus`, while the **clouded** frames, with few detections, can read `indeterminate`. Dropping defocus discarded the only clear sky in the run and let the cloud define the baseline. A chunk with no in-focus frame now **falls back to its own frames** instead of contributing nothing.
- **The fallback's risk is bounded by provenance, not by a threshold.** `update_run_cloud_baseline` returns a `RunCloudBaseline(star, background, from_focused)`, and a defocus-derived baseline is held *provisionally*: a defocused chunk can never raise a baseline in-focus frames already set, and the first genuinely in-focus chunk **replaces** a provisional one outright rather than merging with it — so an inflated blob count cannot survive past real focus evidence. `triage` reads the baseline positionally, so a bare `(star, background)` pair from any other caller still works.
- **This is a chosen trade, not a solved problem.** Nothing in a star count distinguishes "soft focus, clear sky" from "defocused blobs"; the two cases are numerically identical and the test fixture for each is the same. Over-flagging genuinely sharp frames after an all-defocused opening chunk is the accepted cost, bounded as above.
- **Envelope: exactly `classify_shape`'s `DEFOCUS_FWHM_MIN_PX` (8.0) boundary.** A frame at or under it reads `indeterminate`, not `likely_defocus`, so it counts as focused and its star count is trusted outright — genuinely mild defocus this close to in-focus is a real, documented miss. What the boundary governs is now *trust* (`from_focused`), not whether a contribution happens at all.
- **Recall only grows.** Folding in a run baseline can raise the effective star baseline and lower the effective background baseline, never the reverse, so every frame the old per-chunk-only rule flagged is still flagged.
- **Still not flagged: a run that starts cloudy and never clears anywhere.** No chunk ever contributes a genuinely clear condition, so the carried baseline reflects the same degraded sky throughout and nothing reads as anomalous against it — this is documented, unavoidable behavior, not a bug: there is still no clear reference anywhere in the run to judge against.

## Flat and bias frame calibration

`astro_calibrate.py` gains the standard flat/bias half of calibration, at the same
raw-Bayer-mosaic layer as darks: `build_master_bias(bias_mosaics)` is the per-pixel
median of zero-exposure frames; `build_master_dark(..., master_bias=...)` optionally
subtracts it before its own median (omitted, today's bias-blind behavior is
unchanged); `build_master_flat(flat_mosaics, master_bias=...)` is the median of
(flat − bias) frames normalised to its own median, so it centres on 1.0.
`apply_flat_field(mosaic, master_flat, floor=FLAT_FIELD_FLOOR)` divides a
dark/bias-subtracted light by that normalised flat, correcting vignetting/PRNU;
normalised flat pixels under `FLAT_FIELD_FLOOR` (0.1, provisional pending a real
flat card) are clipped up to it before dividing — guards a near-dead or
extreme-vignette photosite from being amplified rather than corrected — and the
clipped count comes back on the returned `FlatFieldResult`.

`decode_linear` takes an optional `master_flat` and applies it last, after dark
subtraction and the hot-pixel cosmetic pass, so flat-fielding corrects the
already-cleaned mosaic. The clipped-pixel count lands on
`DecodedAstroFrame.flat_field_clipped_pixels` (0 when no flat is supplied) as this
module's calibration receipt — the same place `decoder`/`bit_depth` already record
how a frame was decoded. A decode with neither darks nor flats is unchanged from
as before: `master_flat=None` skips the new code path entirely.

Shape/dtype mismatches between a master and its light raise `AstroCalibrationError`
naming the shapes being compared, the same idiom `subtract_dark`/`build_master_dark`
already use — this module has no file paths to name (pure numpy arrays); a caller
with paths (CLI/orchestrator) names them itself.

The calibration math is exercised directly by tests; `--flats`/`--bias` flags are not
wired into `gomoveshift photos astro`.

## Private real RAWs — local staging policy

Real night lights/darks (e.g. your own capture card) are never committed to this repo. They stay staged locally, outside version control, and are referenced from any script/notebook by **absolute path plus a content hash** (e.g. SHA256) — never by committing the RAW or a derived full-resolution render. Only small, explicitly approved derived crops/arrays (gated by acceptance-contract §1) may ever land in the repo.

## Develop-variation contact sheet

`astro_contact.py` renders ONE run's stack through several develop knobs (background WB on/off, a darker/default/lighter stretch target) as one labeled grid, from the single `AstroReport` `stack_run` already produced — never re-stacking per panel. `--from-master PATH` skips stacking entirely, recovering the pre-stretch array from an existing `.tif`+`.stack.json` pair's recorded stretch parameters via MTF's `m -> 1 - m` self-inverse (`test_astro_contact.py`). Gradient removal, star trails, meteor reinsertion and a timelapse ship no panel, each for a stated reason (module docstring).

`--contact-sheet` needs an actual stacked master, so it refuses alongside `--trails`/`--timelapse`/`--triage-only`; pair it with a normal stacking run, or use `--from-master` against a past one. The sheet lands under `photos.contact_sheet_root`, same review-drive convention as `photos develop --contact-sheet`.

## Shared time-lapse core

`astro_timelapse.render_timelapse` no longer inlines its own ffmpeg subprocess
loop: the frame-by-frame pipe write, exit-code/stub-cleanup handling and "the
encode failed" error text now live in `pipeline/timelapse.stream_frames_to_video`,
a source-agnostic streamer that `astro_timelapse.py` and the daylight sibling
below both call. Everything astro-specific — dark/hot-pixel calibration, the
middle-frame shared tone, the per-frame develop — stays in `astro_timelapse.py`
unchanged; only the "write these bytes to ffmpeg" plumbing moved.

`photos daylight-timelapse` (`pipeline/timelapse.render_stills_timelapse`) is
the daytime sibling: ordinary stills, ordered by filename (no astro-specific
calibration or shared-tone develop needed), decoded via `photos.develop.decode`,
scaled and streamed through the same encoder. `pipeline/timelapse.decimate_video`
covers the third case from the same module — dropping frames from an existing
video source via ffmpeg's `select`/`setpts` filters, for a video-source
time-lapse/hyperlapse rather than a stills sequence.

## Scoped deep output

`deep_combine` ships the **scoped** frame by default — the central two thirds of each axis,
3732x2484 on a 5600x3728 sensor, 9.3 MP of 20.9 MP. This is the largest rectangle wholly inside
`r <= (2/3)*rmax`, the region that passes the star-shape gate.

Basis: with per-artifact polynomial placement the outer radial third fails the 10% star-shape
boundary at every useful depth, while the centre and middle thirds pass with margin. Shipping the full
frame would make the default output one that fails the gate the product is measured against.

`scope="full"` returns the whole frame and is explicit opt-in. The blend runs at full frame
either way — the horizon row is defined against the full height — and the crop is applied after.
Coverage statistics are reported over the shipped pixels only.
