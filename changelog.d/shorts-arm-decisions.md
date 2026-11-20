### Added

- `metrics collect` prints an `ARM_DECISION` line per arm of the YouTube Shorts title-body (`yt_title_body`) and hashtag-set (`yt_hashtag_set`) factors: `hold`, `demote`, `pause` or `INSUFFICIENT_EVIDENCE`, from each Short's 72 h views and average view percentage on within-project medians. It acts only on a clear, repeated gap (default: 36 settled Shorts per arm, a 3x median gap; `policy.arm_thresholds` overrides) and a paused arm is re-tested every 28 days. `experiments.assign_stratified_arm` balances arms within a project and slot. The record changes no text sent to YouTube.
