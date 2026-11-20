### Added

- The Shorts experiment arms now change the text. Phase D gives each new Short a `yt_title_body` arm (`place_led` or `clip_led`) and a `yt_hashtag_set` arm (`pov` or `road`) through `assign_stratified_arm`, `SocialGenerator` renders them, and the plan stores them in `experiment_arms`, so the `ARM_DECISION` lines get real outcomes once analytics arrive. The same inputs give the same arms. A Short with no arm, an unknown arm, or one already queued keeps its text byte for byte.
