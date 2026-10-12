### Auth (`console/auth.py`, `GMS_CONSOLE_TOKEN`)

The Docker container binds `0.0.0.0` (`docker/console.Dockerfile`), so a LAN peer can reach the
console unless a token is set. `GMS_CONSOLE_TOKEN` unset/empty leaves `ConsoleAuthMiddleware` a
no-op -- the console's prior unauthenticated behavior, unchanged. Set it and every route but
`GET /healthz` (the compose healthcheck) 401s without the token, supplied via
`Authorization: Bearer <token>`, a one-time `?token=` query param (which sets a `gms_console_token`
session cookie so a phone bookmark keeps working), or that cookie. `jobs_create` (`POST /jobs`) also
now runs the same `_require_tab` gate the other routes apply, keyed off `subcommand` (drive ->
drives, sync-site/publish-journal -> journeys, enqueue-plan/dispatch-queue -> publishing, photos ->
photos) -- previously it was the one job-launching route a disabled brand tab couldn't block.
