### Fixed

- `gomoveshift cleanup` finds a drive `health backlog` staged (`<stages root>/<project>/<project>/_stages`), named or discovered. Before, only `<base>/<project>/_stages` was looked up, so cleaning a published backlog drive with `GMS_INTERMEDIATE_DIR=<stages root>` skipped it ("pipeline not complete") and freed nothing.
