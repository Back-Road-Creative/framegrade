### Added

- `photos sort` tags scene groups in `ranked.csv` (`group_id`, `group_best_rank`): night
  sessions, exposure brackets, panorama runs (new detector: same settings, at most 8 s
  apart, three or more frames) and look-alike bursts inside a category. It then builds
  `hdr/`, `pano/` and `stacks/` outputs one at a time, best night session first, rewrites
  `index.html` as each lands, and skips night stacks with `--no-stacks`. A failed group is a
  `skipped.csv` row. `run_sort(resume=...)` finishes only the unfinished groups.
