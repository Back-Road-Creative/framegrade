### Fixed

- CI caches the pinned ffmpeg tarball under a key built from its checksum, and retries the download. On 11-08 the release host answered 504 and every test shard went red at "Install ffmpeg". A cached tarball is still checked against the pinned checksum.
