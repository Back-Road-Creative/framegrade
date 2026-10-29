### Fixed

- `gomoveshift drive --dry-run` (and `--list-lfs`/`--label-dry-run`) with `--youtube-upload` no longer runs the post-Phase-D journal hook, so a plan render no longer creates a draft journey on the site and then fails with "state['youtube_uploads'] is empty".
