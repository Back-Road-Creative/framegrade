### Fixed

- An expired or revoked Facebook, Instagram or TikTok login no longer fails every queued post one by one: the platform pauses, its posts stay queued, and `health check` raises one `platform_auth_blocked` incident naming it.
- A TikTok chunk upload that fails no longer fails the post: when TikTok provably never got the whole file, the row goes back to queued and the next try starts a fresh upload.
