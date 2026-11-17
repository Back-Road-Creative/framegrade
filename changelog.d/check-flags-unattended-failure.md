### Added

- `health check` files a `failure_unattended` incident, pushed once, for any open failed drive that `health heal` is not retrying: no launched attempt for 26 h, or its newest attempt was refused. It resolves when a newer launch lands or the drive does. A drive under an operator hold (`no-auto-resume`) is reported as a note, not an incident. A drive and its heal resumes raise one alarm, named for the drive. A hand-typed launch that the tracked-launcher gate refused is not a failed drive: it never started one.
