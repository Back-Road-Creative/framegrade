### Fixed

- A day reel whose journal day link cannot be found is no longer queued without it: it is held (no download, no render, no ledger record) until the journal day is live, and the outcome says whether the site's trip list failed to load, was malformed, has no trip on that date or has two.
- A day reel held over 24 hours for its link raises one `day_reel_held_for_link` health incident naming the request, the recording date and the cause.
