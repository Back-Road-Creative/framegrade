### Fixed

- The Instagram rendition is capped at 1080x1920 and 30 fps, a master over that frame is transcoded whatever its bitrate, and a cached rendition over it is rebuilt. A 2160x3840 60 fps day reel kept its frame in the rendition (tagged H.264 level 4.1, over Meta's 1920 px width), and on 2026-11-26 and 2026-11-27 both of its Instagram posts failed: a container stuck `IN_PROGRESS`, then `ProcessingFailedError`.
