### Changed

- `--day-gpx` no longer needs `--camera-utc-offset-hours`: the slicer measures the offset from each recording's first-chapter ffprobe `creation_time` (filename local time minus UTC). It refuses, naming the clips and values, when a clip has no `creation_time`, recordings disagree, the value is not a whole multiple of 15 min (within 60 s) or lies outside -12..+14 h, or the slice fails the Phase-A gate. A given offset that disagrees with `creation_time` is now refused instead of warned about. `health backlog` passes the flag only when the manifest row sets it.
