### Changed

- The shorts planner no longer fills the count. A candidate window qualifies only if its mean per-second composite is strictly above the mean of the day's eligible (non-stopped) timeline, and the gap and stride are never relaxed to reach the target. `target_count` is now a maximum, so a day can plan fewer shorts than the target, or none. Each plan logs `SHORTS-PLAN: <n> of target <t> clear the quality bar (...)` and one line per chosen short with its mean score.
- The per-LF refill uses the day's bar, so an LF with no window above it stays under capacity.
- The SHORTS-CONTRACT planned-stage gate is removed. The planned count is the contract for the rest of the day: a planned short that fails to render still hard-fails, and a resumed day is checked against the count it planned (or a lowered `--shorts-target`). `--allow-short-count` still works.
