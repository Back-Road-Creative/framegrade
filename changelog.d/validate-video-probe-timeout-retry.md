### Fixed

- `validate_video` retried nothing: one 30 s ffprobe timeout on a freshly written file called it corrupt, so a finished 92-minute OST burn was discarded and re-encoded from scratch. A timeout is now retried with the same backoff as the other ffprobe calls; a non-zero ffprobe exit still fails at once, and a file whose probe times out on every attempt is still invalid.
