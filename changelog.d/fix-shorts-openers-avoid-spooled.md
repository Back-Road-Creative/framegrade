### Fixed

- A second drive at the same place no longer has its shorts refused by `enqueue-plan`'s `cross-batch-repeat` rule. The YouTube description led every short with the fixed "POV driving through {place} Pt.N", whose first six words are the same for every drive there, and the caption openers could repeat one another project had already spooled. Caption openers now skip any opener whose first six words match text spooled in the last 60 days, read-only from the queue, and the description lead rotates through 14 phrases in the same way.
