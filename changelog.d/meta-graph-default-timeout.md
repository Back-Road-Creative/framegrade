### Fixed

- Every Facebook/Instagram Graph request now carries a `(10, 120)` s timeout unless its caller sets one; a dead connection hung `dispatch-queue --apply` for 21 h.
