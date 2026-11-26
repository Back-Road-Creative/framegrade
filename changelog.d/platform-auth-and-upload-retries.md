### Fixed

- An expired or revoked Facebook, Instagram or TikTok login no longer fails every queued post one by one: the platform pauses, its posts stay queued, and `health check` raises one `platform_auth_blocked` incident naming it.
