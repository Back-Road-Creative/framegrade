## Route Story best-of-look-alike-group photo selection (`gomoveshift/pipeline/route_story/select.py`)<a id="route-story-best-pick"></a>

Photographers shoot several frames of the same view; `select_best_photos` picks the
best one of each look-alike group so a film doesn't repeat itself. It is a pure
library — the caller decodes RAW/JPEG into an RGB preview and supplies a
`PhotoCandidate(name, image, captured_at, placement=None)`; nothing here touches
the filesystem or EXIF. `still_selection.select_kept_photos` (called from `film.main`) wires
this selector into the film's photo-source collection.

Grouping is chain-linked along the capture-time-sorted sequence: consecutive
candidates join one group only when EVERY test passes between that adjacent
pair — `max_gap_s` (120.0), `max_distance_m` (50.0, skipped when either candidate
lacks a `placement`), and a visual "same view" test. The visual test is ORB
keypoints on a ~1024px-wide grayscale copy, Lowe's-ratio matching, and a RANSAC
homography; "same view" needs `min_inliers` (25) inlier matches AND an inlier
ratio of `min_inlier_ratio` (0.6). Either frame yielding fewer than 20 ORB
keypoints (haze, open sky) falls back to an HSV histogram correlation (coarse
8×8 hue/saturation bins — fine bins make a near-uniform frame's correlation
swing wildly on a couple of pixels of noise) at `hist_corr_threshold` (0.9). A
plain difference-hash does not work here: on a real look-alike burst, frames a
human called obviously alike scored 14-31 of 64 bits apart.

Within a group, photos are scored by `_sharpness_score` first — the mean of the
top-10% tile variances of a Laplacian over a 10×10 grid on a ~1500px copy, so a
shallow-focus subject with a soft background isn't punished for the background.
Scores within `tie_fraction` (10%) of the group's best sharpness tie; a tie
breaks on `_exposure_badness` (fraction of pixels at or past a near-clip bound
on a 512px copy, smaller wins — but fractions within `exposure_tie_fraction`, 0.2% of the
frame, tie: a few blown pixels never pick the winner), then on `gomoveshift.analysis.visual_scorer.
score_color_vibrancy` (larger wins), then on name order. `keep` names always
survive selection even when that leaves two winners from one group; `enabled=
False` returns every candidate untouched. `SelectionResult.to_dict()` gives a
plain-dict receipt (every group's members/winner/losers/scores/reason plus the
flat `kept` list), input-order independent since candidates are always
processed in `(captured_at, name)` order.

Bracketed exposures (`gomoveshift/pipeline/photos/bracket.py`'s
`detect_brackets`) are a different concept — metered exposure runs off
`FNumber`/shutter EXIF — and out of scope here; this module never calls it.
