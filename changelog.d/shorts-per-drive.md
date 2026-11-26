### Changed

- Brand policy key `shorts_per_drive` caps the Shorts planned per drive (the planner keeps its best-ranked clips) without feeding the enqueue batch gate, so several drives in one region still enqueue together. The GoMoveShift brand now plans one Short per drive. See docs/pipeline/136-brand-policy.md.
