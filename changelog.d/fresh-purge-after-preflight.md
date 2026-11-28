### Fixed

- `--fresh` no longer deletes the prior run's plan, checkpoint and OST confirmations before pre-flight has accepted the run. A run that pre-flight refuses (for example for staging disk space) now leaves them in place; a run that passes still clears them before any phase reads them.
