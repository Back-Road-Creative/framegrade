### Fixed

- The Windows release gate no longer refuses a tag pushed while main's CI is still running on that commit: it waits for the run, up to 30 minutes, and refuses only when CI failed, never started, or the wait ran out. v0.3.14's installer build was refused this way.
