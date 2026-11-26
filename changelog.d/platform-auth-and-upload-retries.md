### Fixed

- An expired or revoked Facebook, Instagram or TikTok login no longer fails every queued post one by one: the platform pauses, its posts stay queued, and `health check` raises one `platform_auth_blocked` incident naming it.
- A TikTok chunk upload that fails no longer fails the post: when TikTok provably never got the whole file, the row goes back to queued and the next try starts a fresh upload.
- A Facebook reel is no longer recorded published just because the finish call returned: it is read back first, and a reel not found, failed in processing or not yet live stays an unknown outcome for the reconcile reads.
