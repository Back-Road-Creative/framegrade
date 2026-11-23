### Fixed

- A drive whose publish `health publish` refuses (a mask-frame or duplicate-title gate) no longer holds its region's backlog: `health backlog` logs the refusal at WARNING, lists it under `failed_publishes` and prints `failed publish <project>: <reason>`, leaves the drive unpublished, and launches the next eligible row in the same tick (Kerry blocked morning/afternoon-kerry for days). A Waypost slot wait or a replace backoff is unchanged.
