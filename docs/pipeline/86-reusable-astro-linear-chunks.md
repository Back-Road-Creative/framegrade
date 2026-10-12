## Reusable astro linear chunks

`gomoveshift photos astro --keep-linear` writes one `<astro_stack_...>.linear/` directory per
stack chunk beside its TIFF. `--linear-only` implies it and returns after validated artifact
publication, skipping gradient removal, stretch, meteor work, TIFF export, and stack sidecar.
Each artifact contains uncompressed, memory-mappable float32 sky numerator and ground planes,
the post-rejection uint16 contributor count, and a hash-validated provenance manifest.

At 5600x3728 the payload is about 543 MB per chunk. Before the first decode, the aggregate disk
gate reserves every planned artifact, every requested TIFF, one complete atomic-write temporary,
and 10% headroom. An existing artifact always refuses even with `--overwrite`; reusable linear
inputs must be moved or removed explicitly. Omitting both flags preserves the ordinary TIFF path.

Combine two or more ordered artifacts with
`gomoveshift photos astro-combine chunk-*.linear --out night.tif`; `--reference chunk.linear`
optionally pins a member instead of the quality-qualified midpoint. Developed TIFF inputs and all
output collisions are refused. The result is developed once with meteor reinsertion off and
publishes `night.tif`, effective counts in `night.coverage.npy`, and complete transform,
photometry, rejection, capture, source-frame, mask, scope, develop, and hash provenance in
`night.deep.json`. A material below-floor coverage fraction prints an operator warning.

Chunk-to-chunk registration is lens-aware. A chunk pair is minutes of sky apart, and
`astro_register.register`'s only estimator is a 4-DOF similarity (rotation, uniform scale,
translation) — a model with no term that can absorb Brown radial distortion, so its residual
grows monotonically with radius, RANSAC labels every corner star an outlier, and
`MIN_OUTER_THIRD_INLIERS` refuses the fit. On a typical full-frame night of a couple of hundred frames in about two dozen chunks, a similarity fit registered no chunk to the auto reference: the nearest pair,
a few minutes apart, matched hundreds of pairs and kept many centre inliers at sub-pixel median
residual, a middle band at about a pixel, and none of its matched outer-third pairs — while the chunk means themselves carry dozens of detectable stars
in that outer third, so the corner stars existed and were being matched. The gate was right; the
MODEL was wrong. `fit_artifact_pair` therefore matches stars exactly as before (coarse prior,
then `match_progressive`) and hands the pairs to `astro_geometry.solve_night` — undistort,
rotation on the sphere, re-distort, the shared lens solved jointly, focal length seeded off the
frame width by the same rig fact the per-frame stacking path seeds with — then composes the
answer into one `(map_x, map_y)` remap, the shape `astro_register.warp_to_reference` already
applies. BOTH of `solve_night`'s answers are accepted and are distinguishable in the diagnostics:
a rotation (`lens_rotation`, composed by `astro_lens.build_remap`) and the 8-DOF homography it
falls back to for a pair no Rodrigues vector describes — a tripod that shifted as well as turned
(`lens_homography`, the same composition around `astro_geometry`'s own homography apply). That
homography is fitted in UNDISTORTED space, after the Brown model is removed, and a sky rotation
IS a homography in gnomonic coordinates, so accepting it is that module's documented contract
rather than a weakening — it is emphatically NOT the homography inside `match_progressive`, which
is fitted in raw distorted pixels as a search-window model only and must never be used as a
registration transform.

Which of the two describes a frame is decided on HELD-OUT points, never on the rotation's train
residual alone. The rotation's train p95 against `ROTATION_RESIDUAL_GATE_PX` still admits a clean
rotation immediately, but missing that gate is not evidence that a homography fits better: an 8-DOF
model can always drive a train residual below a 3-DOF one, because it has the spare freedom to bend
toward centroid noise. Selecting on the train residual alone therefore handed every thin, noisy
match set to the model that overfits it. So when the rotation misses its train gate both candidates are now fitted and scored on
the held-out split neither was fitted on, and the better p95 is kept. That split is a validation
split for this decision — it arbitrates as well as reports — and the reported per-third residuals
are always the winning model's own held-out numbers, never its train numbers. A frame whose motion
genuinely is not a rotation still selects the homography, on that same evidence.

Refusing the homography outright is not enough: a chunk that needs a homography then fails in `register`'s gates. The SAME inlier floors run unchanged against whichever
model fitted, measured by sampling the very remap that will warp the pixels and binned into the
same radial thirds, so nothing about the acceptance contract is relaxed. Synthetic pairs built
with a known distortion: against a rotation pair shaped like the measured one, the similarity
keeps 70/116/0 centre/middle/outer inliers (0.55 and 1.40 px medians) where the lens model keeps
67/192/52 at 0.23/0.27/0.28 px; against a pair carrying a small non-rotational component the
similarity is refused outright (inlier fraction 0.32) where the homography branch keeps
66/201/125 at 1.22/0.43/0.67 px. Only a solve that genuinely does not converge falls back to the
old similarity, and it SAYS SO: `geometry_method` `similarity_fallback` plus the refusal text
ride on the fit into `night.deep.json`'s per-chunk diagnostics, and — since a downgrade is
invisible exactly where it matters otherwise — into the `AstroCombineError` that aborts the
combine, which names the model each failed pair was measuring and, when it had been downgraded,
why. An `ArtifactTransform` consequently carries EITHER a 2x3 `transform` or a `remap`, never
both — `warp` is whichever exists, and a remapped chunk's serialized `transform.matrix` is `null`
rather than two full-frame resampling grids rendered as JSON. `plan_direct_transforms` (the only
production planner) accepts a remap; the retired adjacent-chain planner refuses one instead of
composing it, since a remap is a resampling grid and chaining two would resample twice.

`plan_direct_transforms` attempts EVERY pair before it raises. A lens fit costs 64-98 s per pair
against the similarity's 5-9 s, so aborting on the first refusal made a 25-chunk night spend that
cost and learn about exactly one chunk — the operator then re-ran to discover the next one, and
a long night could burn several passes that way. The refusal now names every chunk that could
not register, each with its own `geometry_method`, and then the inclusive run of chunks around the
reference that DID register (`_registrable_span`) — the subset of paths a re-run can actually ship.
That span is the run CONTAINING the reference, never the longest run anywhere: a stretch behind a
gap has no transform into the reference's frame, which is the whole point of the single-reference
planner. Nothing is dropped automatically; a short span is an operator decision, and the combine
still refuses rather than silently shipping fewer chunks than it was handed.

`--scope auto|full|scoped` (default `auto`) chooses the shipped region. The sky+ground blend
always happens at the FULL, uncropped frame first — the horizon row is defined against the full
height, and measuring corner quality after a crop would hide exactly the corners the measurement
exists to see. `scoped` always crops to the central two-thirds rectangle that measurement showed to pass A3 at every useful depth (`astro_combine.SCOPED_FRAME_FRACTION`); `full` always ships the
whole sensor unconditionally. `auto` measures `astro_quality.a3_by_radial_third`'s outer radial
third against that same pinned growth boundary on the blended full-frame RGB and ships the full
sensor only when that outer band measurably passes — the largest rectangle that can be proven
clean, since every other pixel sits closer to centre than the worst corner. Anything else (outer
fails, or no per-frame reference FWHM survives in the combined artifacts' triage provenance to
grow against) falls back to the scoped rectangle. The decision — requested scope, resolved scope,
and (for `auto`) the full radial measurement — is recorded under `deep.json`'s `"scope"` key so a
shipped master's crop is always auditable after the fact.
