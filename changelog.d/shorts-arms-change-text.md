### Added

- The Shorts experiment arms now change the text. Phase D gives each new Short a `yt_title_body` arm (`place_led` or `clip_led`) and a `yt_hashtag_set` arm (`pov` or `road`) through `assign_stratified_arm`, `SocialGenerator` renders them, and the plan stores them in `experiment_arms`, so the `ARM_DECISION` lines get real outcomes once analytics arrive. The same inputs give the same arms. A Short with no arm, an unknown arm, or one already queued keeps its text byte for byte. One arm history spans the whole drive and strata use the base project, so each stratum splits within one. An arm is stored only when its text is whole: a title body that is published or would be cut mid-word drops the arm, and the arm's hashtags are pinned in the description.

### Fixed

- `queue sync-bracket-short-text` regenerates a Short under its stored Shorts arms instead of rewriting it as legacy text, and refuses a Short whose stored arm can no longer be rendered.
