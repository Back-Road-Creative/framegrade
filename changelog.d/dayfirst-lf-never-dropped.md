### Fixed

- A planned long-form that fails (no segments, concat returns nothing, or the montage prepend fails) now stops the day with an error naming the day, the LF and the cause, before any short is rendered. It used to log "skipping" and carry on, so a day spent about an hour on shorts and then died in Phase D with `shorts_per_lf SPLIT-BRAIN`. A resume re-runs a day that was recorded complete with fewer long-forms than planned.
- The montage prepend concat no longer has a fixed 30-minute limit: it runs under the progress-based ffmpeg runner, so a 38 GB `-c copy +faststart` concat that keeps writing is not killed, and a stalled one still is (partial output removed).
- A montage clip whose start has no usable keyframe is cut with an accurate seek instead of `-c copy`, which replayed up to one GOP and pushed a 15.0 s montage to 16.7 s. The 10% duration check is unchanged.
