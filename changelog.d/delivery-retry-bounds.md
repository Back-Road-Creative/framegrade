### Fixed

- A delivery whose retries never counted against the 8-try limit (an unknown Facebook outcome, an Instagram container still processing, unreadable media, a platform throttle) no longer retries silently for good: after 24 h it parks at `needs_operator` with its upload ids kept and raises a `delivery_needs_operator` incident.
- An upload that starts over (an expired Instagram container, a Facebook reel whose finish never ran) now gets a fresh attempt count, and an Instagram container still processing is polled again instead of spending the row's ordinary tries (delivery #391 failed for good this way).
