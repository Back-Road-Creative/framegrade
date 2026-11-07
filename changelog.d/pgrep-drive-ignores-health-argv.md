### Fixed

- `health` no longer reports "a drive is running" because of its own arguments. With no live job record, the process check matched any `gomoveshift` command line containing " drive ", so `health backlog --supersede ... --reason "queue next Ireland drive"` blocked itself. Now it matches only when `drive` is the subcommand.
