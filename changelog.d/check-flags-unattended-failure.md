### Added

- `health check` files a `failure_unattended` incident, pushed once, for any open failed drive that `health heal` is not retrying: no launched attempt for 26 h, or its newest attempt was refused. It resolves when a newer launch lands or the drive does. A drive under an operator hold (`no-auto-resume`) is reported as a note, not an incident.
