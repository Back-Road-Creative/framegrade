### Fixed

- `health backlog` publishes a drive again when its audit passed and its publish job failed. Before, the store row stayed `failed` and every tick skipped it, so the audited drive never aired.
