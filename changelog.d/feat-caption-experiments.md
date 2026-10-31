### Added

- A brand policy can declare caption experiments (`policy.experiments`): arms with a `hashtag_mode` (`none`, `location`, `june`) are assigned by a stable hash, stop `stop_after_days` after `start`, and are tagged in the publish-plan sidecar (`experiment_arms`). `metrics collect` prints `EXPERIMENT <name> platform=facebook arm=<id> n=<settled> median_reach=<value|unknown>` per arm; the median waits until every arm has `min_n` settled, measured shorts.
