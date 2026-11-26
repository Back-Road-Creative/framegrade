### Fixed

- `gomoveshift day-reels authorize` signs in through the pipeline's own Google OAuth client (from its Drive token) instead of rclone's built-in client, whose shared quota returned `403 rateLimitExceeded` and stalled the hourly listing. The client is stored with the token and passed to rclone only through its environment.
- Day-reels listing and reading give up within a minute (`--low-level-retries 2`, `--contimeout 15s`) and report `Google is throttling Drive reads` instead of a bare timeout.
